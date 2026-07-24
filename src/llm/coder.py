import json
import os
import time
import logging
import re  # 1. Added re for regex parsing

from .config import (
    get_llm_response,
    get_llm_client,
    estimate_tokens,
    LLMBackendTransientError,
    is_gemini_transient_error,
    is_gemini_provider,
    GEMINI_STREAM_RETRY_WAIT,
    GEMINI_STREAM_MAX_RETRIES,
)

logger = logging.getLogger("LLM_Backend")
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_LLM_DIR)
_BLUEPRINTS_DIR = os.path.realpath(os.path.join(_SRC_DIR, "blueprints"))


def _format_routing_context(routing_context: dict | None) -> str:
    if not isinstance(routing_context, dict):
        return ""
    payload = {
        key: routing_context[key]
        for key in ("blueprints", "parameters")
        if key in routing_context
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_prompt(
    user_prompt: str,
    blueprint_names: list[str],
    feedback: str,
    prior_code: str = "",
    routing_context: dict | None = None,
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
    formatted_routing = _format_routing_context(routing_context)
    if formatted_routing:
        prompt += (
            "### ROUTER CONTEXT ###\n"
            "This is a structured extraction of the original request. Use it to "
            "locate explicitly stated objectives, design variables, constraints, "
            "conditions, and geometry. The original user request remains "
            "authoritative. The extracted lists contain user-stated fields, not an "
            "exhaustive replacement for the active blueprint formulation. Do not "
            "delete an existing blueprint constraint, objective, or fixed "
            "assumption merely because it is absent from Router Context. Do not "
            "implement any router value that is absent from or conflicts with the "
            "original request.\n"
            f"{formatted_routing}\n\n"
        )

    if prior_code:
        prompt += (
            "### CURRENT WORKING CODE ###\n"
            "The following code was generated in the previous turn and is working. "
            "MODIFY this code to meet the new user request instead of starting from a blueprint:\n"
            f"```python\n{prior_code}\n```\n\n"
        )

    prompt += (
        "### BASE BLUEPRINTS (SOURCE SCRIPTS TO EDIT) ###\n"
        "Use the selected blueprint as the executable base. Preserve its code "
        "structure, values, formulas, OAS wiring, recording, reporting, output "
        "paths, and plotting/post-processing blocks unless the user request or "
        "retry feedback requires a change. Do not copy long instructional comment "
        "blocks, DV catalogs, prompt guidance, or editable-section markers into "
        "the final script.\n"
        f"{blueprints_context}\n"
    )

    if feedback and feedback != "Initial generation":
        prompt += f"\n### ERROR FEEDBACK FROM PREVIOUS ATTEMPT ###\n{feedback}\nFix the code above."

    return system_prompt, prompt


def _strip_comment_only_lines(code: str) -> str:
    lines = []
    blank_pending = False
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if not line.strip():
            if lines and not blank_pending:
                lines.append("")
            blank_pending = True
            continue
        lines.append(line.rstrip())
        blank_pending = False
    return "\n".join(lines).strip() + "\n"


def _parse_response(response: str) -> tuple[str, str]:
    """
    Extract XML reasoning/code, with defensive fallbacks for malformed responses.
    """
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.S | re.I)
    if not reasoning_match:
        reasoning_match = re.search(
            r"<reasoning>(.*?)(?:<code>|$)", response, re.S | re.I
        )
    reasoning = (
        reasoning_match.group(1).strip()
        if reasoning_match
        else "Generated complete Python script."
    )

    code_match = re.search(r"<code>(.*?)</code>", response, re.S | re.I)
    if not code_match:
        code_match = re.search(r"<code>(.*?)$", response, re.S | re.I)

    if code_match:
        code = code_match.group(1).strip()
    elif "##### REASONING ENDS #####" in response:
        code = response.split("##### REASONING ENDS #####")[-1].strip()
    else:
        fence_match = re.search(r"```(?:python)?\s*(.*?)```", response, re.S | re.I)
        code = fence_match.group(1).strip() if fence_match else response.strip()

    code = re.sub(r"</?(?:reasoning|code)>", "", code, flags=re.I).strip()
    code = re.sub(r"^```(?:python)?\s*|^```\s*", "", code, flags=re.M).strip()

    starts = [m.start() for m in re.finditer(r"(?m)^(?:import|from)\s+", code)]
    if starts:
        code = code[min(starts) :].strip()

    code = _strip_comment_only_lines(code)

    return reasoning, code


def generate_code(
    user_prompt,
    blueprints,
    feedback,
    model_name,
    provider,
    prior_code="",
    routing_context=None,
):
    sys, p = _build_prompt(
        user_prompt,
        blueprints,
        feedback,
        prior_code,
        routing_context=routing_context,
    )
    logger.info(f"--- SYSTEM ---\n{sys}\n--- PROMPT ---\n{p}")
    resp = get_llm_response(p, model_name, sys, provider=provider)
    logger.info(f"--- RESPONSE ---\n{resp}")
    reasoning, code = _parse_response(resp)
    return code, reasoning, estimate_tokens(sys + p), estimate_tokens(resp)


def generate_code_stream(
    user_prompt,
    blueprints,
    feedback,
    model_name,
    provider,
    prior_code="",
    routing_context=None,
):
    sys, p = _build_prompt(
        user_prompt,
        blueprints,
        feedback,
        prior_code,
        routing_context=routing_context,
    )
    client = get_llm_client(provider, model_name)

    if not client or not is_gemini_provider(provider):
        resp = get_llm_response(p, model_name, sys, provider=provider)
        yield resp
        reasoning, code = _parse_response(resp)
        yield (code, reasoning, estimate_tokens(sys + p), estimate_tokens(resp))
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
            if (
                is_gemini_transient_error(e)
                and gemini_attempt < GEMINI_STREAM_MAX_RETRIES - 1
            ):
                yield f"\n\nRetrying Gemini... {e}"
                time.sleep(GEMINI_STREAM_RETRY_WAIT)
            elif is_gemini_transient_error(e):
                raise LLMBackendTransientError(
                    f"LLM stream failed after retries: {e}"
                ) from e
            else:
                yield f"Final error: {e}"

    logger.info(f"--- RESPONSE ---\n{full_resp}")
    in_t, out_t = 0, 0
    try:
        if last_chunk and last_chunk.usage_metadata:
            in_t = last_chunk.usage_metadata.prompt_token_count
            out_t = last_chunk.usage_metadata.candidates_token_count
    except Exception:
        pass

    reasoning, code = _parse_response(full_resp)
    yield (
        code,
        reasoning,
        in_t or estimate_tokens(sys + p),
        out_t or estimate_tokens(full_resp),
    )
