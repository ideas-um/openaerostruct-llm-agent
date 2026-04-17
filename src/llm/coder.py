import os
import json
from .config import get_llm_response

def generate_code(user_prompt: str, blueprint_names: list[str], feedback: str, model_name: str = "gemini-2.5-flash", provider: str = "Gemini API") -> tuple[str, str]:
    """
    Takes one or more base blueprints and modifies them to fulfill the user's request.
    Incorporates feedback from previous failed iterations in the retry loop.
    Returns: (generated_code, reasoning)
    """
    blueprints_context = ""
    for name in blueprint_names:
        blueprint_path = os.path.join("src", "blueprints", name)
        if os.path.exists(blueprint_path):
            with open(blueprint_path, "r") as f:
                content = f.read()
                blueprints_context += f"\n--- BLUEPRINT: {name} ---\n```python\n{content}\n```\n"
        else:
            blueprints_context += f"\nWarning: Blueprint {name} not found.\n"

    plotting_guidance = ""
    plotting_md_path = os.path.join("src", "blueprints", "plotting.md")
    if os.path.exists(plotting_md_path):
        with open(plotting_md_path, "r") as f:
            plotting_guidance = f"\n--- PLOTTING GUIDANCE (plotting.md) ---\n{f.read()}\n"
        
    system_prompt = (
        "You are an OpenAeroStruct Expert developer. You synthesize Python scripts based on working Blueprints.\n\n"
        "CRITICAL RULES:\n"
        "1. PATHS: Always use os.path.join() for all file paths — never hardcode forward-slash strings like "
        "'src/openaerostruct_out/...'. This ensures cross-platform compatibility on both Windows and macOS/Linux.\n"
        "   Example: os.path.join('src', 'openaerostruct_out', 'aero.db')  NOT 'src/openaerostruct_out/aero.db'\n"
        "2. PLOTTING: Do NOT add plotting code inside the main analysis/optimisation script unless the user explicitly "
        "requests a plot. When plotting IS requested, follow the patterns in src/blueprints/plotting.md exactly. "
        "Always use matplotlib.use('Agg') before importing pyplot, and save to "
        "os.path.join('src', 'openaerostruct_out', 'agent_plots', 'filename.png').\n"
        "3. IMPORTS: Only import packages that are already present in the provided blueprint. Do not add new "
        "dependencies (e.g. niceplots, plotly) unless they are already imported in the blueprint.\n"
        "4. SCHEMA: Do not change the OpenMDAO problem structure — keep surface dict keys and connection patterns "
        "exactly as shown in the blueprint.\n\n"
        "REQUIRED FORMAT:\n"
        "1. Start your response with 'REASONING: ' followed by your logic.\n"
        "2. End your reasoning with the EXACT STRING: ##### REASONING ENDS #####\n"
        "3. Provide ONLY the full Python code immediately after that tag (No markdown fencing).\n\n"
        "CRITICAL: The '##### REASONING ENDS #####' tag is MANDATORY. Do not omit it."
    )
    
    prompt = (
        f"User Prompt: {user_prompt}\n\n"
        f"Base Blueprints provided for reference:\n{blueprints_context}\n"
        f"{plotting_guidance}"
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
