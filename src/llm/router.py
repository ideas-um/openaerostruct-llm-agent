import json
import os
import logging
from .config import get_llm_response, log_token_usage
try:
    from .config import get_llm_client
except ImportError:
    get_llm_client = None

logger = logging.getLogger("LLM_Backend")

# ---------------------------------------------------------------------------
# __file__-relative paths
# ---------------------------------------------------------------------------
_LLM_DIR     = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.dirname(_LLM_DIR)
_SKILLS_PATH = os.path.join(_SRC_DIR, "llm", "skills.md")

# ---------------------------------------------------------------------------
# Blueprint allowlist — only these filenames are permitted as router outputs.
# ---------------------------------------------------------------------------
VALID_BLUEPRINTS: frozenset = frozenset({
    "aero_analysis.py",
    "aero_multipoint.py",
    "aero_rect.py",
    "aerostruct_tube.py",
    "aerostruct_wingbox.py",
    "struct_optimization.py",
})


def _load_system_prompt() -> str:
    if os.path.exists(_SKILLS_PATH):
        with open(_SKILLS_PATH, "r") as f:
            return f.read()
    return "Select an OpenAeroStruct blueprint. Return JSON: {'blueprints': [], 'reason': ''}"


def _parse_routing_response(response: str) -> dict:
    """Parse the router's JSON response and validate blueprint names."""
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        data = json.loads(cleaned.strip())

        if "blueprint" in data and "blueprints" not in data:
            data["blueprints"] = [data["blueprint"]]

        raw = data.get("blueprints", [])
        validated = [b for b in raw if b in VALID_BLUEPRINTS]
        data["blueprints"] = validated if validated else ["aero_rect.py"]
        return data
    except Exception as e:
        return {"blueprints": ["aero_rect.py"], "reason": f"Fallback due to parsing error: {e}"}


def route_intent(user_prompt: str, model_name: str = "gemini-2.0-flash", provider: str = "Gemini API") -> dict:
    """
    Non-streaming router: picks the right blueprint(s) for the user's request.
    Returns a dict with 'blueprints', 'reason', and optionally 'is_vague'/'missing_info'.
    """
    system_prompt = _load_system_prompt()
    response = get_llm_response(user_prompt, model_name, system_prompt, provider=provider)
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
    """
    system_prompt = _load_system_prompt()

    try:
        client = get_llm_client(provider, model_name) if get_llm_client else None
    except Exception:
        client = None

    if client is None or provider != "Gemini API":
        response = get_llm_response(user_prompt, model_name, system_prompt, provider=provider)
        yield response
        yield _parse_routing_response(response)
        return

    from google.genai import types as _types
    stream_config = _types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
        max_output_tokens=1024,
    )

    logger.info(f"========== NEW LLM REQUEST (stream/router) ({model_name} via {provider}) ==========")
    logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{user_prompt}")

    full_response = ""
    last_chunk = None
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
    except Exception:
        if not full_response:
            full_response = get_llm_response(user_prompt, model_name, system_prompt, provider=provider)
            yield full_response

    logger.info(f"--- LLM RESPONSE ---\n{full_response}")

    # Extract token counts from the final chunk's usage_metadata
    input_tokens = None
    output_tokens = None
    try:
        if last_chunk is not None and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
            usage = last_chunk.usage_metadata
            input_tokens  = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            logger.info(
                f"Tokens (stream/router) — input: {input_tokens}, "
                f"output: {output_tokens}"
            )
    except Exception:
        pass

    log_token_usage(provider, model_name, input_tokens, output_tokens)

    yield _parse_routing_response(full_response)