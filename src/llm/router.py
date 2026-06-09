import json
import os
import time
import logging
from .config import (
    get_llm_response,
    get_llm_client,
    log_token_usage,
    is_gemini_transient_error,
    GEMINI_STREAM_RETRY_WAIT,
    GEMINI_STREAM_MAX_RETRIES,
)

logger = logging.getLogger("LLM_Backend")

# ---------------------------------------------------------------------------
# __file__-relative paths
# ---------------------------------------------------------------------------
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_LLM_DIR)
_SKILLS_PATH = os.path.join(_SRC_DIR, "llm", "skills.md")

# ---------------------------------------------------------------------------
# Blueprint allowlist — only these filenames are permitted as router outputs.
# ---------------------------------------------------------------------------
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


def _load_system_prompt() -> str:
    if os.path.exists(_SKILLS_PATH):
        with open(_SKILLS_PATH, "r") as f:
            return f.read()
    return "Select an OpenAeroStruct blueprint. Return JSON: {'blueprints': [], 'reason': ''}"


def _parse_routing_response(response: str) -> dict:
    """
    Parse the router's JSON response and validate blueprint names.
    Accepts 1 or 2 blueprints; silently drops any beyond the second or
    any name not in VALID_BLUEPRINTS.
    """
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        data = json.loads(cleaned.strip())

        # Accept singular 'blueprint' key as well
        if "blueprint" in data and "blueprints" not in data:
            data["blueprints"] = [data["blueprint"]]

        raw = data.get("blueprints", [])
        validated = [b for b in raw if b in VALID_BLUEPRINTS][:2]  # max 2
        data["blueprints"] = validated if validated else ["aero_opt.py"]
        return data
    except Exception as e:
        return {
            "blueprints": ["aero_opt.py"],
            "reason": f"Fallback due to parsing error: {e}",
        }


def route_intent(
    user_prompt: str,
    model_name: str = "gemini-2.0-flash",
    provider: str = "Gemini API",
) -> dict:
    """
    Non-streaming router: picks the right blueprint(s) for the user's request.
    Returns a dict with 'blueprints', 'reason', and optionally 'is_vague'/'missing_info'.
    Transient retry is handled inside get_llm_response() in config.py.
    """
    system_prompt = _load_system_prompt()
    response = get_llm_response(
        user_prompt, model_name, system_prompt, provider=provider
    )
    return _parse_routing_response(response)


def route_intent_stream(
    user_prompt: str,
    model_name: str = "gemini-2.0-flash",
    provider: str = "Gemini API",
):
    """
    Streaming router — yields text chunks as they arrive, then a final dict.

    Usage (same sentinel pattern as generate_code_stream):
        for chunk in route_intent_stream(...):
            if isinstance(chunk, dict):
                routing_data = chunk   # final result
            else:
                display(chunk)         # stream this to the UI

    Retries on transient Gemini API errors without propagating them to the caller.
    """
    system_prompt = _load_system_prompt()

    try:
        client = get_llm_client(provider, model_name)
    except Exception:
        client = None

    if client is None or provider != "Gemini API":
        response = get_llm_response(
            user_prompt, model_name, system_prompt, provider=provider
        )
        yield response
        yield _parse_routing_response(response)
        return

    from google.genai import types as _types

    stream_config = _types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
        max_output_tokens=1024,
    )

    logger.info(
        f"========== NEW LLM REQUEST (stream/router) ({model_name} via {provider}) =========="
    )
    logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{user_prompt}")

    for gemini_attempt in range(GEMINI_STREAM_MAX_RETRIES):
        full_response = ""
        last_chunk = None
        transient_hit = False

        try:
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=user_prompt,
                config=stream_config,
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
                logger.warning(
                    f"Gemini transient error in stream/router (attempt {gemini_attempt + 1}/"
                    f"{GEMINI_STREAM_MAX_RETRIES}): {exc}. "
                    f"Waiting {GEMINI_STREAM_RETRY_WAIT}s before retry."
                )
                yield f"\n\n⚠️ Gemini API overloaded — retrying in {GEMINI_STREAM_RETRY_WAIT}s...\n"
                time.sleep(GEMINI_STREAM_RETRY_WAIT)
            else:
                if not full_response:
                    full_response = get_llm_response(
                        user_prompt, model_name, system_prompt, provider=provider
                    )
                    yield full_response

        if transient_hit:
            continue  # retry the entire stream

        break  # stream completed without transient error

    logger.info(f"--- LLM RESPONSE ---\n{full_response}")

    # Extract token counts from the final chunk's usage_metadata
    input_tokens = None
    output_tokens = None
    try:
        if (
            last_chunk is not None
            and hasattr(last_chunk, "usage_metadata")
            and last_chunk.usage_metadata
        ):
            usage = last_chunk.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            logger.info(
                f"Tokens (stream/router) — input: {input_tokens}, "
                f"output: {output_tokens}"
            )
    except Exception:
        pass

    log_token_usage(provider, model_name, input_tokens, output_tokens)

    yield _parse_routing_response(full_response)
