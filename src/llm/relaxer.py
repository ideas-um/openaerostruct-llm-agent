import ast
import os
import re
import json
import logging
from .config import get_llm_response, estimate_tokens

logger = logging.getLogger("LLM_Backend")
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))


def _optimization_formulation(generated_code: str) -> str:
    if not generated_code:
        return ""
    try:
        tree = ast.parse(generated_code)
    except SyntaxError:
        return ""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"add_design_var", "add_constraint", "add_objective"}:
            continue
        calls.append((getattr(node, "lineno", 0), ast.unparse(node)))
    return "\n".join(code for _, code in sorted(calls))


def _parse_relaxation_response(response: str) -> str:
    """
    Extract the suggestion markdown from <relaxation> JSON, with fallback.
    """
    try:
        # Required wrapper first, fallback below for malformed responses.
        match = re.search(
            r"<relaxation>(.*?)</relaxation>", response, re.DOTALL | re.IGNORECASE
        )
        if not match:
            match = re.search(
                r"<relaxation>(.*?)$", response, re.DOTALL | re.IGNORECASE
            )

        if match:
            json_str = match.group(1).strip()
        else:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
            else:
                return response.strip()

        # Clean code fencing if present.
        json_str = re.sub(r"^```json\s*|^```\s*", "", json_str, flags=re.MULTILINE)
        json_str = re.sub(r"```$", "", json_str, flags=re.MULTILINE).strip()

        data = json.loads(json_str)
        return data.get("suggestion", "No suggestions generated.")
    except Exception as e:
        logger.error(f"Relaxer parse error: {e}. Raw response: {response}")
        return f"Suggested relaxations:\n{response}"


def suggest_relaxation(
    user_prompt: str,
    error_logs: list,
    model_name: str,
    provider: str,
    *,
    blueprints: list[str] | None = None,
    optimizer_status: str = "",
    db_summary: str = "",
    result_metrics: dict | None = None,
    generated_code: str = "",
) -> tuple[str, int, int]:
    """
    Loads relaxer.md and prompts the LLM to analyze the failure path
    and suggest valid physical relaxations.
    """
    _RELAX_PATH = os.path.join(_LLM_DIR, "relaxer.md")

    if os.path.exists(_RELAX_PATH):
        with open(_RELAX_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        system_prompt = "Suggest 2-3 physical relaxations for non-convergence."

    # Keep only the last two attempts to avoid context bloat and focus on the latest error
    recent_errors = "\n\n".join(error_logs[-2:])
    formulation = _optimization_formulation(generated_code)
    metrics_json = json.dumps(
        result_metrics or {}, indent=2, sort_keys=True, default=str
    )

    formatted_user_prompt = (
        f"### USER'S DESIGN REQUEST ###\n{user_prompt}\n\n"
        f"### SELECTED BLUEPRINT ###\n{', '.join(blueprints or [])}\n\n"
        f"### ACTIVE OPTIMIZATION FORMULATION ###\n"
        f"{formulation or 'Unavailable'}\n\n"
        f"### INITIAL AND FINAL OPTIMIZATION RECORD ###\n"
        f"{db_summary or 'Unavailable'}\n\n"
        f"### STRUCTURED RESULT METRICS ###\n{metrics_json}\n\n"
        f"### OPTIMIZER TERMINATION OUTPUT ###\n"
        f"{optimizer_status or 'Unavailable'}\n\n"
        f"### EXECUTION ATTEMPTS & FAILURES ###\n{recent_errors}\n\n"
        f"Generate the relaxation response object:"
    )

    in_t = estimate_tokens(system_prompt + "\n" + formatted_user_prompt)

    try:
        # FIXED: Calling with exact positional arguments matching get_llm_response signature
        ans = get_llm_response(
            formatted_user_prompt, model_name, system_prompt, provider=provider
        )
        parsed_suggestion = _parse_relaxation_response(ans)
        return parsed_suggestion, in_t, estimate_tokens(ans)
    except Exception as e:
        logger.error(f"Failed to generate relaxation suggestion: {e}")
        fallback_msg = (
            "- **Review Active Bounds**: Inspect variables that terminate at a bound while a related constraint remains infeasible.\n"
            "- **Review the Initial Point**: Move only out-of-bounds initial values to a sensible interior point."
        )
        return fallback_msg, in_t, estimate_tokens(fallback_msg)
