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

    # THE CORE FIX: Providing context of the previous turn
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
    delimiter = "##### REASONING ENDS #####"
    if delimiter in response:
        res, code = response.split(delimiter, 1)
        reasoning = res.replace("REASONING:", "").strip()
        code = code.strip()
    else:
        reasoning = "Parsing warning."
        code = response.strip()

    if "```python" in code:
        code = code.split("```python")[-1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    return reasoning, code.strip()


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

    if not client or provider != "Gemini API":
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
