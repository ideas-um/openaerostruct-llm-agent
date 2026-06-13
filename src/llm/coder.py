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
    """Fallback token estimator if API metadata is missing."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(str(text)))
    except ImportError:
        return len(str(text)) // 4


def _build_prompt(
    user_prompt: str, blueprint_names: list[str], feedback: str
) -> tuple[str, str]:
    blueprints_context = ""
    for name in blueprint_names:
        candidate = os.path.realpath(os.path.join(_BLUEPRINTS_DIR, name))
        if not candidate.startswith(_BLUEPRINTS_DIR + os.sep):
            blueprints_context += (
                f"\nWarning: Blueprint '{name}' rejected (path traversal).\n"
            )
            continue
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                blueprints_context += (
                    f"\n--- BLUEPRINT: {name} ---\n```python\n{f.read()}\n```\n"
                )
        else:
            blueprints_context += f"\nWarning: Blueprint '{name}' not found.\n"

    system_prompt_path = os.path.join(_LLM_DIR, "coder.md")
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        system_prompt = "You are an OpenAeroStruct expert developer. Synthesize Python scripts based on the provided blueprints."

    prompt = f"User Prompt: {user_prompt}\n\nBase Blueprints provided for reference:\n{blueprints_context}\n"
    if feedback and feedback != "Initial generation":
        prompt += (
            f"\nPrevious execution failed. Feedback:\n{feedback}\nPlease fix the logic."
        )

    return system_prompt, prompt


def _parse_response(response: str) -> tuple[str, str]:
    delimiter = "##### REASONING ENDS #####"
    if delimiter in response:
        reasoning_part, code_part = response.split(delimiter, 1)
        reasoning = reasoning_part.replace("REASONING:", "").strip()
        code = code_part.strip()
    else:
        import_index = response.find("import ")
        from_index = response.find("from ")
        candidates = [i for i in [import_index, from_index] if i != -1]
        start_index = min(candidates) if candidates else -1

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
) -> tuple[str, str, int, int]:
    system_prompt, prompt = _build_prompt(user_prompt, blueprint_names, feedback)
    response = get_llm_response(prompt, model_name, system_prompt, provider=provider)
    reasoning, code = _parse_response(response)

    # Fallback to estimate since get_llm_response drops raw API metadata
    in_tok = _approx_tokens(system_prompt + "\n" + prompt)
    out_tok = _approx_tokens(response)
    return code, reasoning, in_tok, out_tok


def generate_code_stream(
    user_prompt: str,
    blueprint_names: list[str],
    feedback: str,
    model_name: str = "gemini-2.0-flash",
    provider: str = "Gemini API",
):
    system_prompt, prompt = _build_prompt(user_prompt, blueprint_names, feedback)

    try:
        client = get_llm_client(provider, model_name)
    except Exception:
        client = None

    if client is None or provider != "Gemini API":
        response = get_llm_response(
            prompt, model_name, system_prompt, provider=provider
        )
        yield response
        reasoning, code = _parse_response(response)
        in_tok = _approx_tokens(system_prompt + "\n" + prompt)
        out_tok = _approx_tokens(response)
        yield (code, reasoning, in_tok, out_tok)
        return

    from google.genai import types as _types

    stream_config = _types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        temperature=0.2,
        max_output_tokens=8192,
    )

    logger.info(
        f"========== NEW LLM REQUEST (stream) ({model_name} via {provider}) =========="
    )

    input_tokens, output_tokens = 0, 0

    for gemini_attempt in range(GEMINI_STREAM_MAX_RETRIES):
        full_response = ""
        last_chunk = None
        transient_hit = False

        try:
            for chunk in client.models.generate_content_stream(
                model=model_name, contents=prompt, config=stream_config
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
                yield f"\n\n⚠️ Gemini API overloaded — retrying in {GEMINI_STREAM_RETRY_WAIT}s...\n"
                time.sleep(GEMINI_STREAM_RETRY_WAIT)
            else:
                if not full_response:
                    full_response = get_llm_response(
                        prompt, model_name, system_prompt, provider=provider
                    )
                    yield full_response

        if transient_hit:
            continue
        break

    try:
        if (
            last_chunk is not None
            and hasattr(last_chunk, "usage_metadata")
            and last_chunk.usage_metadata
        ):
            usage = last_chunk.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0)
    except Exception:
        pass

    input_tokens = input_tokens or _approx_tokens(system_prompt + "\n" + prompt)
    output_tokens = output_tokens or _approx_tokens(full_response)
    log_token_usage(provider, model_name, input_tokens, output_tokens)

    reasoning, code = _parse_response(full_response)
    yield (code, reasoning, input_tokens, output_tokens)
