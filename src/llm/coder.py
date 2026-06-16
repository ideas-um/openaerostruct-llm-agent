import os
import time
import logging
import re  # 1. Added re for regex parsing

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
_BLUEPRINTS_DIR = os.path.realpath(os.path.join(_SRC_DIR, "blueprints"))


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(str(text)))
    except ImportError:
        return len(str(text)) // 4


def _build_prompt(
    user_prompt: str, blueprint_names: list[str], feedback: str, prior_code: str = ""
) -> tuple[str, str]:
    blueprints_context = ""
    for name in blueprint_names:
        p = os.path.join(_BLUEPRINTS_DIR, name)
        if os.path.exists(p):
            with open(p, "r") as f:
                blueprints_context += f"\n--- BLUEPRINT: {name} ---\n{f.read()}\n"

    with open(os.path.join(_LLM_DIR, "coder.md"), "r") as f:
        system_prompt = f.read()

    prompt = f"User Request: {user_prompt}\n\n"

    if prior_code:
        prompt += (
            "### CURRENT WORKING CODE ###\n"
            "The following code was generated in the previous turn and is working. "
            "MODIFY this code to meet the new user request instead of starting from a blueprint:\n"
            f"```python\n{prior_code}\n```\n\n"
        )

    prompt += f"### BASE BLUEPRINTS (For Reference) ###\n{blueprints_context}\n"

    if feedback and feedback != "Initial generation":
        prompt += f"\n### ERROR FEEDBACK FROM PREVIOUS ATTEMPT ###\n{feedback}\nFix the code above."

    return system_prompt, prompt


def _parse_response(response: str) -> tuple[str, str]:
    """
    Extracts reasoning and code, supporting both the new XML format
    and the legacy ##### delimiter format.
    """
    # 1. Extract Reasoning (XML and legacy fallback)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.S | re.I)
    if not reasoning_match:
        reasoning_match = re.search(r"<reasoning>(.*?)$", response, re.S | re.I)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    elif "REASONING:" in response:
        # Legacy fallback for reasoning
        raw_parts = response.split("##### REASONING ENDS #####")[0]
        reasoning = raw_parts.replace("REASONING:", "").strip()
    else:
        reasoning = "Reasoning details parsed from turn."

    # 2. Extract Code (XML and legacy fallback)
    code_match = re.search(r"<code>(.*?)</code>", response, re.S | re.I)
    if not code_match:
        code_match = re.search(r"<code>(.*?)$", response, re.S | re.I)

    if code_match:
        code = code_match.group(1).strip()
    else:
        # Legacy fallback for code
        if "##### REASONING ENDS #####" in response:
            code = response.split("##### REASONING ENDS #####")[-1].strip()
        else:
            code = response.strip()

    # Cleanup any residual markdown markers
    code = re.sub(r"^```python\s*|^```\s*", "", code, flags=re.MULTILINE)
    code = re.sub(r"```$", "", code, flags=re.MULTILINE).strip()

    return reasoning, code


def generate_code(
    user_prompt, blueprints, feedback, model_name, provider, prior_code=""
):
    sys, p = _build_prompt(user_prompt, blueprints, feedback, prior_code)
    logger.info(f"--- SYSTEM ---\n{sys}\n--- PROMPT ---\n{p}")
    resp = get_llm_response(p, model_name, sys, provider=provider)
    logger.info(f"--- RESPONSE ---\n{resp}")
    reasoning, code = _parse_response(resp)
    return code, reasoning, _approx_tokens(sys + p), _approx_tokens(resp)


def generate_code_stream(
    user_prompt, blueprints, feedback, model_name, provider, prior_code=""
):
    sys, p = _build_prompt(user_prompt, blueprints, feedback, prior_code)
    client = get_llm_client(provider, model_name)

    if not client or not is_gemini_provider(provider):
        resp = get_llm_response(p, model_name, sys, provider=provider)
        yield resp
        reasoning, code = _parse_response(resp)
        yield (code, reasoning, _approx_tokens(sys + p), _approx_tokens(resp))
        return

    logger.info(f"--- PROMPT (Stream) ---\n{p}")
    full_resp, last_chunk = "", None
    for gemini_attempt in range(GEMINI_STREAM_MAX_RETRIES):
        try:
            from google.genai import types

            cfg = types.GenerateContentConfig(system_instruction=sys, temperature=0.2)
            for chunk in client.models.generate_content_stream(
                model=model_name, contents=p, config=cfg
            ):
                last_chunk = chunk
                txt = chunk.text or ""
                full_resp += txt
                yield txt
            break
        except Exception as e:
            if gemini_attempt < GEMINI_STREAM_MAX_RETRIES - 1:
                yield f"\n\nRetrying Gemini... {e}"
                time.sleep(2)
            else:
                yield f"Final error: {e}"

    logger.info(f"--- RESPONSE ---\n{full_resp}")
    in_t, out_t = 0, 0
    try:
        if last_chunk and last_chunk.usage_metadata:
            in_t = last_chunk.usage_metadata.prompt_token_count
            out_t = last_chunk.usage_metadata.candidates_token_count
    except:
        pass

    reasoning, code = _parse_response(full_resp)
    yield (
        code,
        reasoning,
        in_t or _approx_tokens(sys + p),
        out_t or _approx_tokens(full_resp),
    )
