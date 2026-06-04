import os
import csv
import logging
from google import genai
from google.genai import types
import ollama
from dotenv import load_dotenv

# Load .env from the project root (parent of src directory)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")
load_dotenv(_ENV_FILE, override=True, encoding="utf-8")

# ---------------------------------------------------------------------------
# Resolve paths relative to this file.
# ---------------------------------------------------------------------------
_LLM_DIR    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.dirname(_LLM_DIR)
_LOG_FILE   = os.path.join(_SRC_DIR, "agent_backend.log")
_STATS_FILE = os.path.join(_SRC_DIR, "usage_stats.csv")

logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLM_Backend")


def _make_gemini_client() -> genai.Client:
    """Create and return a Gemini client using the API key from the environment."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
            "(or add it to a .env file)."
        )
    return genai.Client(api_key=api_key)


def get_llm_client(provider: str, model_name: str):
    """
    Return a raw SDK client for the given provider so callers can use
    streaming APIs directly. Returns None for providers that don't support it.
    """
    if "Gemini" in provider:
        return _make_gemini_client()
    return None


def log_token_usage(provider, model, input_tokens, output_tokens):
    """
    Append a row to the usage CSV so token consumption can be tracked over time.
    Uses csv.writer to handle escaping and prevent CSV injection.
    Treats None counts as 0 so a missing field from the API doesn't crash the log.
    """
    from datetime import datetime
    # Guard against the Gemini API occasionally returning None for token counts.
    input_tokens  = input_tokens  or 0
    output_tokens = output_tokens or 0
    total = input_tokens + output_tokens

    exists = os.path.isfile(_STATS_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_STATS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Timestamp", "Provider", "Model", "InputTokens", "OutputTokens", "TotalTokens"])
        writer.writerow([timestamp, provider, model, input_tokens, output_tokens, total])


def get_llm_response(prompt: str, model_name: str, system_prompt: str = None, provider: str = "Gemini API") -> str:
    """
    Send a prompt to either Gemini or Ollama and return the response text.
    Retries automatically on transient Gemini errors (503, quota exhaustion).
    """
    logger.info(f"========== NEW LLM REQUEST ({model_name} via {provider}) ==========")
    if system_prompt:
        logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{prompt}")

    if "Gemini" in provider:
        import time
        client = _make_gemini_client()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=0.2,
            max_output_tokens=8192,
        )

        max_retries = 5
        wait_times  = [5, 10, 20, 40, 60]

        for i in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "high demand" in err_str or "quota" in err_str:
                    if i < max_retries - 1:
                        wait_time = wait_times[i]
                        logger.warning(
                            f"Transient LLM error ({err_str[:50]}). "
                            f"Retrying in {wait_time}s (attempt {i+1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                logger.error(f"Permanent LLM error: {e}")
                raise Exception(f"LLM request failed: {e}")

        usage = response.usage_metadata
        logger.info(
            f"Tokens — input: {usage.prompt_token_count}, "
            f"output: {usage.candidates_token_count}, "
            f"total: {usage.total_token_count}"
        )
        log_token_usage(provider, model_name, usage.prompt_token_count, usage.candidates_token_count)
        logger.info(f"--- LLM RESPONSE ---\n{response.text}")
        return response.text

    else:
        # Ollama path
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(model=model_name, messages=messages)

            prompt_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)
            logger.info(
                f"Tokens — input: {prompt_tokens}, output: {output_tokens}, "
                f"total: {prompt_tokens + output_tokens}"
            )
            log_token_usage(provider, model_name, prompt_tokens, output_tokens)

            content = response['message']['content']
            logger.info(f"--- LLM RESPONSE ---\n{content}")
            return content

        except Exception as e:
            error_msg = (
                f"Error: could not connect to Ollama for model '{model_name}'. "
                f"Make sure Ollama is running locally. Details: {e}"
            )
            logger.error(error_msg)
            return error_msg