import json
import os
import time
import logging
import re
from .config import (
    get_llm_response,
    get_llm_client,
    estimate_tokens,
    log_token_usage,
    LLMBackendTransientError,
    is_gemini_transient_error,
    is_gemini_provider,
    GEMINI_STREAM_RETRY_WAIT,
    GEMINI_STREAM_MAX_RETRIES,
)

logger = logging.getLogger("LLM_Backend")
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_LLM_DIR)
_ROUTER_PROMPT_PATH = os.path.join(_SRC_DIR, "llm", "router.md")
_LEGACY_SKILLS_PATH = os.path.join(_SRC_DIR, "llm", "skills.md")

VALID_BLUEPRINTS: frozenset = frozenset(
    {
        "aero_analysis.py",
        "aero_multipoint.py",
        "aero_opt.py",
        "aerostruct_tube.py",
        "aerostruct_wingbox.py",
        "struct_optimization.py",
    }
)
_ROUTER_PARAMETER_FIELDS: frozenset = frozenset(
    {
        "intent",
        "objective",
        "design_variables",
        "constraints",
        "flight_conditions",
        "geometry",
        "loads",
        "materials",
        "settings",
        "requested_outputs",
        "mapped_vars",
    }
)


def _load_system_prompt() -> str:
    for path in (_ROUTER_PROMPT_PATH, _LEGACY_SKILLS_PATH):
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            return f.read()
    return "Select an OpenAeroStruct blueprint."


class RouterContractError(ValueError):
    pass


def _parse_routing_response(response: str) -> dict:
    """
    Extract JSON from <routing> tags, with plain-JSON fallback.
    """
    try:
        # 1. Try the required <routing> tags first, then plain JSON fallback.
        match = re.search(r"<routing>(.*?)</routing>", response, re.DOTALL)

        if match:
            json_str = match.group(1).strip()
        else:
            # Fallback: find the first { and last } if tags are missing
            # This is a 'safety net' for conversational LLMs
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
            else:
                json_str = response.strip()

        # 2. Clean up any markdown code fencing that might be inside the tags
        json_str = re.sub(r"^```json\s*|^```\s*", "", json_str, flags=re.MULTILINE)
        json_str = re.sub(r"```$", "", json_str, flags=re.MULTILINE).strip()

        # 3. Parse the JSON
        data = json.loads(json_str)

        # 4. Standardize 'blueprint' vs 'blueprints'
        if "blueprint" in data and "blueprints" not in data:
            data["blueprints"] = [data["blueprint"]]

        # 5. Validate blueprint selection. The router contract is exactly one
        # blueprint; retry upstream if the model returns a list.
        raw = data.get("blueprints", [])
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list) and len(raw) > 1:
            raise RouterContractError(
                f"Router returned multiple blueprint entries: {raw}"
            )
        validated = [b for b in raw if b in VALID_BLUEPRINTS]
        if len(validated) != 1:
            raise RouterContractError(
                f"Router must return exactly one supported blueprint, received: {raw}"
            )
        data["blueprints"] = validated

        return data

    except RouterContractError:
        raise
    except Exception as e:
        logger.error(f"Routing parse error: {e}. Raw response: {response}")
        raise RouterContractError(f"Router response could not be parsed: {e}") from e


def _validate_routing_contract(data: dict, user_prompt: str) -> dict:
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    data["parameters"] = {
        key: value
        for key, value in parameters.items()
        if key in _ROUTER_PARAMETER_FIELDS
    }
    return data


def _router_repair_prompt(user_prompt: str, previous_response: str, error: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "Router contract error: "
        f"{error}\n"
        "Return exactly one best blueprint in blueprints. Do not combine "
        "blueprints. Previous response:\n"
        f"{previous_response}"
    )


def route_intent(
    user_prompt: str, model_name: str = "gemini-2.0-flash", provider: str = "Gemini API"
) -> dict:
    system_prompt = _load_system_prompt()
    logger.info(
        f"========== NEW LLM REQUEST (router) ({model_name} via {provider}) =========="
    )
    logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{user_prompt}")

    response = get_llm_response(
        user_prompt, model_name, system_prompt, provider=provider
    )
    logger.info(f"--- LLM RESPONSE ---\n{response}")

    input_text = system_prompt + "\n" + user_prompt
    output_text = response
    try:
        data = _parse_routing_response(response)
        data = _validate_routing_contract(data, user_prompt)
    except RouterContractError as exc:
        logger.warning(f"{exc}; retrying router once.")
        repair_prompt = _router_repair_prompt(user_prompt, response, str(exc))
        retry_response = get_llm_response(
            repair_prompt, model_name, system_prompt, provider=provider
        )
        logger.info(f"--- LLM RESPONSE (router retry) ---\n{retry_response}")
        data = _parse_routing_response(retry_response)
        data = _validate_routing_contract(data, user_prompt)
        input_text += "\n" + repair_prompt
        output_text += "\n" + retry_response
        data["router_retry"] = True

    data["input_tokens"] = estimate_tokens(input_text)
    data["output_tokens"] = estimate_tokens(output_text)
    return data


def route_intent_stream(
    user_prompt: str, model_name: str = "gemini-2.0-flash", provider: str = "Gemini API"
):
    system_prompt = _load_system_prompt()
    try:
        client = get_llm_client(provider, model_name)
    except Exception:
        client = None

    if client is None or not is_gemini_provider(provider):
        response = get_llm_response(
            user_prompt, model_name, system_prompt, provider=provider
        )
        yield response
        input_text = system_prompt + "\n" + user_prompt
        output_text = response
        try:
            data = _parse_routing_response(response)
            data = _validate_routing_contract(data, user_prompt)
        except RouterContractError as exc:
            logger.warning(f"{exc}; retrying router once.")
            yield f"\n\nRouter contract issue — retrying once: {exc}\n"
            repair_prompt = _router_repair_prompt(user_prompt, response, str(exc))
            retry_response = get_llm_response(
                repair_prompt, model_name, system_prompt, provider=provider
            )
            yield retry_response
            data = _parse_routing_response(retry_response)
            data = _validate_routing_contract(data, user_prompt)
            input_text += "\n" + repair_prompt
            output_text += "\n" + retry_response
            data["router_retry"] = True
        data["input_tokens"] = estimate_tokens(input_text)
        data["output_tokens"] = estimate_tokens(output_text)
        yield data
        return

    logger.info(
        f"========== NEW LLM REQUEST (stream/router) ({model_name} via {provider}) =========="
    )
    logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{user_prompt}")

    from google.genai import types as _types

    stream_config = _types.GenerateContentConfig(
        system_instruction=system_prompt, temperature=0.2, max_output_tokens=2048
    )

    input_tokens, output_tokens = 0, 0
    for gemini_attempt in range(GEMINI_STREAM_MAX_RETRIES):
        full_response, last_chunk, transient_hit = "", None, False
        try:
            for chunk in client.models.generate_content_stream(
                model=model_name, contents=user_prompt, config=stream_config
            ):
                last_chunk = chunk
                text = chunk.text or ""
                full_response += text
                yield text
        except Exception as exc:
            if is_gemini_transient_error(exc):
                if gemini_attempt < GEMINI_STREAM_MAX_RETRIES - 1:
                    transient_hit = True
                    yield "\n\n⚠️ Gemini API overloaded — retrying...\n"
                    time.sleep(GEMINI_STREAM_RETRY_WAIT)
                else:
                    raise LLMBackendTransientError(
                        f"LLM stream failed after retries: {exc}"
                    ) from exc
            elif not full_response:
                full_response = get_llm_response(
                    user_prompt, model_name, system_prompt, provider=provider
                )
                yield full_response
        if transient_hit:
            continue
        break

    logger.info(f"--- LLM RESPONSE ---\n{full_response}")

    try:
        if (
            last_chunk is not None
            and hasattr(last_chunk, "usage_metadata")
            and last_chunk.usage_metadata
        ):
            usage = last_chunk.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0)
            logger.info(
                f"Tokens (router) — input: {input_tokens}, output: {output_tokens}"
            )
    except Exception:
        pass

    input_text = system_prompt + "\n" + user_prompt
    output_text = full_response
    try:
        data = _parse_routing_response(full_response)
        data = _validate_routing_contract(data, user_prompt)
    except RouterContractError as exc:
        logger.warning(f"{exc}; retrying router once.")
        yield f"\n\nRouter contract issue — retrying once: {exc}\n"
        repair_prompt = _router_repair_prompt(user_prompt, full_response, str(exc))
        retry_response = get_llm_response(
            repair_prompt, model_name, system_prompt, provider=provider
        )
        logger.info(f"--- LLM RESPONSE (router retry) ---\n{retry_response}")
        yield retry_response
        data = _parse_routing_response(retry_response)
        data = _validate_routing_contract(data, user_prompt)
        input_text += "\n" + repair_prompt
        output_text += "\n" + retry_response
        data["router_retry"] = True

    if data.get("router_retry"):
        data["input_tokens"] = estimate_tokens(input_text)
        data["output_tokens"] = estimate_tokens(output_text)
    else:
        data["input_tokens"] = input_tokens or estimate_tokens(input_text)
        data["output_tokens"] = output_tokens or estimate_tokens(output_text)
    log_token_usage(provider, model_name, data["input_tokens"], data["output_tokens"])
    yield data
