import json
import os
from .config import get_llm_response

# ---------------------------------------------------------------------------
# __file__-relative paths
# ---------------------------------------------------------------------------
_LLM_DIR    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.dirname(_LLM_DIR)
_SKILLS_PATH = os.path.join(_SRC_DIR, "blueprints", "skills.md")

# ---------------------------------------------------------------------------
# Blueprint allowlist
# ---------------------------------------------------------------------------
# Only names in this set are permitted as return values from the LLM router.
# Any name not present here is silently dropped before reaching the coder.
VALID_BLUEPRINTS: frozenset = frozenset({
    "aero_analysis.py",
    "aero_multipoint.py",
    "aero_rect.py",
    "aerostruct_multipoint.py",
    "aerostruct_tube.py",
    "aerostruct_wingbox.py",
    "agent_plotting.py",
    "custom_mesh.py",
    "multi_section_aero.py",
    "multi_section_aerostructural.py",
    "multiple_lifting_surfaces.py",
    "plot_wing.py",
    "stability_derivatives.py",
    "struct_optimization.py",
})


def route_intent(user_prompt: str, model_name: str = "gemini-2.5-flash", provider: str = "Gemini API") -> dict:
    """
    Identifies the appropriate blueprint(s) based on user prompt.
    All selection logic is managed in src/blueprints/skills.md.
    """
    if os.path.exists(_SKILLS_PATH):
        with open(_SKILLS_PATH, "r") as f:
            system_prompt = f.read()
    else:
        system_prompt = "Select an OpenAeroStruct blueprint. Return JSON: {'blueprints': [], 'reason': ''}"

    response = get_llm_response(user_prompt, model_name, system_prompt, provider=provider)
    
    try:
        # Clean response and parse JSON
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

        # ── Allowlist validation ───────────────────────────────────────────
        # Reject any blueprint name not in VALID_BLUEPRINTS to prevent path
        # traversal or injection of arbitrary file names by the LLM.
        raw_blueprints = data.get("blueprints", [])
        validated = [b for b in raw_blueprints if b in VALID_BLUEPRINTS]
        if not validated:
            validated = ["aero_rect.py"]
        data["blueprints"] = validated

        return data
    except Exception as e:
        return {"blueprints": ["aero_rect.py"], "reason": f"Fallback due to parsing error: {str(e)}"}
