import ollama

class OllamaTextModel:
     #using general quick onboard model to prevent API limit issues
    def __init__(self, model_name: str = "gemma3:12b-it-qat"):
        self.model_name = model_name

    def generate_content(self, prompt: str):
        res = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        text = res["message"]["content"]
        class R:
            def __init__(self, t): self.text = t
        return R(text)

def get_model():
    return OllamaTextModel()
