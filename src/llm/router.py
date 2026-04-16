import json
import os
from .config import get_llm_response

def route_intent(user_prompt: str, model_name: str = "gemini-2.5-flash", provider: str = "Gemini API") -> dict:
    """
    Identifies the appropriate blueprint(s) based on user prompt.
    All selection logic is managed in src/blueprints/skills.md.
    """
    skills_path = os.path.join("src", "blueprints", "skills.md")
    
    if os.path.exists(skills_path):
        with open(skills_path, "r") as f:
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
        
        return data
    except Exception as e:
        return {"blueprints": ["aero_rect.py"], "reason": f"Fallback due to parsing error: {str(e)}"}
