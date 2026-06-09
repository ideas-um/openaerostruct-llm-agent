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
# Resolve the blueprints directory relative to this file.
# ---------------------------------------------------------------------------
_LLM_DIR        = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR        = os.path.dirname(_LLM_DIR)
_BLUEPRINTS_DIR = os.path.realpath(os.path.join(_SRC_DIR, "blueprints"))


def _build_prompt(user_prompt: str, blueprint_names: list[str], feedback: str) -> tuple[str, str]:
    """
    Assemble the system prompt and user prompt that get sent to the LLM.
    Separated out so both the streaming and non-streaming paths can share it.
    """
    blueprints_context = ""
    for name in blueprint_names:
        candidate = os.path.realpath(os.path.join(_BLUEPRINTS_DIR, name))
        if not candidate.startswith(_BLUEPRINTS_DIR + os.sep):
            blueprints_context += f"\nWarning: Blueprint '{name}' rejected (path traversal).\n"
            continue
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                blueprints_context += f"\n--- BLUEPRINT: {name} ---\n```python\n{f.read()}\n```\n"
        else:
            blueprints_context += f"\nWarning: Blueprint '{name}' not found.\n"

    system_prompt_path = os.path.join(_LLM_DIR, "coder.md")
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        system_prompt = (
            "You are an OpenAeroStruct expert developer. "
            "Synthesize Python scripts based on the provided blueprints."
        )

    prompt = (
        f"User Prompt: {user_prompt}\n\n"
        f"Base Blueprints provided for reference:\n{blueprints_context}\n"
    )
    if feedback and feedback != "Initial generation":
        prompt += f"\nPrevious execution failed. Feedback:\n{feedback}\nPlease fix the logic."

    return system_prompt, prompt


def _parse_response(response: str) -> tuple[str, str]:
    """
    Split the raw LLM response into (reasoning, code).
    """
    delimiter = "##### REASONING ENDS #####"
    if delimiter in response:
        reasoning_part, code_part = response.split(delimiter, 1)
        reasoning = reasoning_part.replace("REASONING:", "").strip()
        code = code_part.strip()
    else:
        import_index = response.find("import ")
        from_index   = response.find("from ")
        candidates   = [i for i in [import_index, from_index] if i != -1]
        start_index  = min(candidates) if candidates else -1

        if start_index != -1:
            reasoning = response[:start_index].replace("REASONING:", "").strip()
            code = response[start_index:].strip()
        else:
            reasoning = "Warning: delimiter missing — full response treated as code."
            code = response.strip()

    if "```python" in code:
        code = code.split("```python")[-1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    return reasoning, code.strip()


def generate_code(
    user_prompt: str,
    blueprint_names: list[str],
    feedback: str,
    model_name: str = "gemini-2.0-flash",
    provider: str = "Gemini API",
) -> tuple[str, str]:
    """
    Generate an OpenAeroStruct script (non-streaming).
    Returns (code, reasoning).
    Transient retry is handled inside get_llm_response() in config.py.
    """
    system_prompt, prompt = _build_prompt(user_prompt, blueprint_names, feedback)
    response = get_llm_response(prompt, model_name, system_prompt, provider=provider)
    reasoning, code = _parse_response(response)
    return code, reasoning


def generate_code_stream(
    user_prompt: str,
    blueprint_names: list[str],
    feedback: str,
    model_name: str = "gemini-2.0-flash",
    provider: str = "Gemini API",
):
    """
    Generate an OpenAeroStruct script with streaming output.

    Yields text chunks as they arrive from the model so the UI can display
    them in real time. Also returns the final (code, reasoning) tuple via a
    special sentinel yielded last:

        for chunk in generate_code_stream(...):
            if isinstance(chunk, tuple):
                code, reasoning = chunk   # final result — stop iterating
            else:
                display(chunk)            # stream this text to the UI

    Falls back to non-streaming if the provider doesn't support it or if
    get_llm_client is not available.

    Retries on transient Gemini API errors (rate limit / overload) without
    propagating the error to the caller — these are infrastructure issues,
    not code bugs, and should not consume a user retry attempt.
    """
    system_prompt, prompt = _build_prompt(user_prompt, blueprint_names, feedback)

    try:
        client = get_llm_client(provider, model_name)
    except Exception:
        client = None

    if client is None or provider != "Gemini API":  # noqa: SIM102
        response = get_llm_response(prompt, model_name, system_prompt, provider=provider)
        yield response
        reasoning, code = _parse_response(response)
        yield (code, reasoning)
        return

    # --- Gemini streaming path ---
    from google.genai import types as _types
    stream_config = _types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        temperature=0.2,
        max_output_tokens=8192,
    )

    logger.info(f"========== NEW LLM REQUEST (stream) ({model_name} via {provider}) ==========")
    logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{prompt}")

    for gemini_attempt in range(GEMINI_STREAM_MAX_RETRIES):
        full_response = ""
        last_chunk = None
        transient_hit = False

        try:
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=stream_config,
            ):
                last_chunk = chunk
                text = chunk.text or ""
                full_response += text
                yield text

        except Exception as exc:
            if is_gemini_transient_error(exc) and gemini_attempt < GEMINI_STREAM_MAX_RETRIES - 1:
                transient_hit = True
                logger.warning(
                    f"Gemini transient error during stream (attempt {gemini_attempt + 1}/"
                    f"{GEMINI_STREAM_MAX_RETRIES}): {exc}. "
                    f"Waiting {GEMINI_STREAM_RETRY_WAIT}s before retry."
                )
                
                yield f"\n\n⚠️ Gemini API overloaded — retrying in {GEMINI_STREAM_RETRY_WAIT}s (attempt {gemini_attempt + 1}/{GEMINI_STREAM_MAX_RETRIES})...\n"
                time.sleep(GEMINI_STREAM_RETRY_WAIT)
            else:
                # Non-transient error or final attempt — fall back to non-streaming
                if not full_response:
                    full_response = get_llm_response(prompt, model_name, system_prompt, provider=provider)
                    yield full_response

        if transient_hit:
            continue  # retry the entire stream

        # Stream completed without transient error — exit retry loop
        break

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
                f"Tokens (stream) — input: {input_tokens}, "
                f"output: {output_tokens}"
            )
    except Exception:
        pass

    log_token_usage(provider, model_name, input_tokens, output_tokens)

    reasoning, code = _parse_response(full_response)
    yield (code, reasoning)