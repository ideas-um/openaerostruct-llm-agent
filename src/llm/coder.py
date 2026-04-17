import os
import json
from .config import get_llm_response

# ---------------------------------------------------------------------------
# __file__-relative blueprint directory
# ---------------------------------------------------------------------------
_LLM_DIR       = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR       = os.path.dirname(_LLM_DIR)
_BLUEPRINTS_DIR = os.path.realpath(os.path.join(_SRC_DIR, "blueprints"))


def generate_code(user_prompt: str, blueprint_names: list[str], feedback: str, model_name: str = "gemini-2.5-flash", provider: str = "Gemini API") -> tuple[str, str]:
    """
    Takes one or more base blueprints and modifies them to fulfill the user's request.
    Incorporates feedback from previous failed iterations in the retry loop.
    Returns: (generated_code, reasoning)
    """
    blueprints_context = ""
    for name in blueprint_names:
        # ── Path-traversal guard ──────────────────────────────────────────
        # Resolve to an absolute, normalised path and confirm it stays inside
        # _BLUEPRINTS_DIR so a crafted name like "../../.env" is rejected.
        # os.path.realpath() normalises paths and never produces a trailing
        # separator, so appending os.sep gives a clean prefix that prevents
        # false matches against sibling directories (e.g. /blueprints_evil).
        candidate = os.path.realpath(os.path.join(_BLUEPRINTS_DIR, name))
        blueprints_prefix = _BLUEPRINTS_DIR + os.sep
        if not candidate.startswith(blueprints_prefix):
            blueprints_context += f"\nWarning: Blueprint '{name}' rejected (path traversal).\n"
            continue

        blueprint_path = candidate
        if os.path.exists(blueprint_path):
            with open(blueprint_path, "r") as f:
                content = f.read()
                blueprints_context += f"\n--- BLUEPRINT: {name} ---\n```python\n{content}\n```\n"
        else:
            blueprints_context += f"\nWarning: Blueprint {name} not found.\n"
        
    system_prompt_path = os.path.join(_LLM_DIR, "coder.md")
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        system_prompt = "You are an OpenAeroStruct Expert developer. You synthesize Python scripts based on working Blueprints."

    
    prompt = (
        f"User Prompt: {user_prompt}\n\n"
        f"Base Blueprints provided for reference:\n{blueprints_context}\n"
    )
    
    if feedback and feedback != "Initial generation":
        prompt += f"\nPrevious execution failed. Feedback:\n{feedback}\nPlease fix the logic."
        
    response = get_llm_response(prompt, model_name, system_prompt, provider=provider)
    
    # Robust Parsing
    delimiter = "##### REASONING ENDS #####"
    if delimiter in response:
        reasoning_part, code_part = response.split(delimiter, 1)
        reasoning = reasoning_part.replace("REASONING:", "").strip()
        code = code_part.strip()
    else:
        # Fallback: Find the first 'import' or 'from' as the start of code
        import_index = response.find("import ")
        from_index = response.find("from ")
        start_index = min(i for i in [import_index, from_index] if i != -1) if (import_index != -1 or from_index != -1) else -1
        
        if start_index != -1:
            reasoning = response[:start_index].replace("REASONING:", "").strip()
            code = response[start_index:].strip()
        else:
            reasoning = "Warning: Delimiter missing, full response treated as code."
            code = response.strip()

    # Strip markdown code fencing if found
    if "```python" in code:
        code = code.split("```python")[-1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
            
    return code.strip(), reasoning
