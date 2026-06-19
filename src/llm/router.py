import json
import os
import time
import logging
import re
from .config import (
    get_llm_response,
    get_llm_client,
    log_token_usage,
    is_gemini_transient_error,
    is_gemini_provider,
    GEMINI_STREAM_RETRY_WAIT,
    GEMINI_STREAM_MAX_RETRIES,
)

logger = logging.getLogger("LLM_Backend")
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_LLM_DIR)
_SKILLS_PATH = os.path.join(_SRC_DIR, "llm", "skills.md")

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


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(str(text)))
    except ImportError:
        return len(str(text)) // 4


def _load_system_prompt() -> str:
    if os.path.exists(_SKILLS_PATH):
        with open(_SKILLS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "Select an OpenAeroStruct blueprint."


def _parse_routing_response(response: str) -> dict:
    """
    Extracts JSON from <routing> tags using regex.
    """
    try:
        # 1. Try to find content between <routing> tags
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

        # 5. Validate blueprint selection
        raw = data.get("blueprints", [])
        validated = [b for b in raw if b in VALID_BLUEPRINTS][:1]
        data["blueprints"] = validated if validated else ["aero_opt.py"]

        return data

    except Exception as e:
        logger.error(f"Routing parse error: {e}. Raw response: {response}")
        return {
            "blueprints": ["aero_opt.py"],
            "is_vague": False,
            "reason": f"Parsing failure: {str(e)}",
        }


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

    data = _parse_routing_response(response)
    data["input_tokens"] = _approx_tokens(system_prompt + "\n" + user_prompt)
    data["output_tokens"] = _approx_tokens(response)
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
        data = _parse_routing_response(response)
        data["input_tokens"] = _approx_tokens(system_prompt + "\n" + user_prompt)
        data["output_tokens"] = _approx_tokens(response)
        yield data
        return

    logger.info(
        f"========== NEW LLM REQUEST (stream/router) ({model_name} via {provider}) =========="
    )
    logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{user_prompt}")

    from google.genai import types as _types

    stream_config = _types.GenerateContentConfig(
        system_instruction=system_prompt, temperature=0.2, max_output_tokens=1024
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
            if (
                is_gemini_transient_error(exc)
                and gemini_attempt < GEMINI_STREAM_MAX_RETRIES - 1
            ):
                transient_hit = True
                yield f"\n\n⚠️ Gemini API overloaded — retrying...\n"
                time.sleep(GEMINI_STREAM_RETRY_WAIT)
            else:
                if not full_response:
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

    data = _parse_routing_response(full_response)
    data["input_tokens"] = input_tokens or _approx_tokens(
        system_prompt + "\n" + user_prompt
    )
    data["output_tokens"] = output_tokens or _approx_tokens(full_response)
    log_token_usage(provider, model_name, input_tokens, output_tokens)
    yield data
