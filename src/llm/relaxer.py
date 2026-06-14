import os
import re
import json
import logging
from .config import get_llm_response

logger = logging.getLogger("LLM_Backend")
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(str(text)))
    except ImportError:
        return len(str(text)) // 4


def _parse_relaxation_response(response: str) -> str:
    """
    Extracts the markdown string from the JSON inside the <relaxation> tags.
    Handles case-insensitive matching and open-ended tags in case of stream limits.
    """
    try:
        # Case-insensitive XML matching
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
            # Fallback: find the first { and last } if tags are completely missing
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
            else:
                return response.strip()

        # Clean code fencing if present inside XML tags
        json_str = re.sub(r"^```json\s*|^```\s*", "", json_str, flags=re.MULTILINE)
        json_str = re.sub(r"```$", "", json_str, flags=re.MULTILINE).strip()

        data = json.loads(json_str)
        return data.get("suggestion", "No suggestions generated.")
    except Exception as e:
        logger.error(f"Relaxer parse error: {e}. Raw response: {response}")
        return f"Suggested relaxations:\n{response}"



def suggest_relaxation(
    user_prompt: str, error_logs: list, model_name: str, provider: str
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
    
    formatted_user_prompt = (
        f"### USER'S DESIGN REQUEST ###\n{user_prompt}\n\n"
        f"### EXECUTION ATTEMPTS & FAILURES ###\n{recent_errors}\n\n"
        f"Generate the relaxation response object:"
    )

    in_t = _approx_tokens(system_prompt + "\n" + formatted_user_prompt)
    
    try:
        # FIXED: Calling with exact positional arguments matching get_llm_response signature
        ans = get_llm_response(
            formatted_user_prompt,
            model_name,
            system_prompt,
            provider=provider
        )
        parsed_suggestion = _parse_relaxation_response(ans)
        return parsed_suggestion, in_t, _approx_tokens(ans)
    except Exception as e:
        logger.error(f"Failed to generate relaxation suggestion: {e}")
        fallback_msg = (
            "- **Relax Bounds**: Expand the upper and lower limits of your design variables.\n"
            "- **Relax Safety Margin**: Reduce the structural safety factor."
        )
        return fallback_msg, in_t, _approx_tokens(fallback_msg)
