import os
import csv
import logging
from google import genai
from google.genai import types
import ollama
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# __file__-relative paths
# ---------------------------------------------------------------------------
_LLM_DIR  = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_LLM_DIR)
_LOG_FILE   = os.path.join(_SRC_DIR, "agent_backend.log")
_STATS_FILE = os.path.join(_SRC_DIR, "usage_stats.csv")

logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLM_Backend")

def log_token_usage(provider, model, input_tokens, output_tokens):
    """
    Log usage statistics to a permanent CSV file.
    Uses csv.writer to ensure values are properly escaped (prevents CSV injection).
    """
    from datetime import datetime
    exists = os.path.isfile(_STATS_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = input_tokens + output_tokens
    with open(_STATS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Timestamp", "Provider", "Model", "InputTokens", "OutputTokens", "TotalTokens"])
        writer.writerow([timestamp, provider, model, input_tokens, output_tokens, total])

def get_llm_response(prompt: str, model_name: str, system_prompt: str = None, provider: str = "Gemini API") -> str:
    """
    Unified function to get response from Gemini or Ollama.
    """
    logger.info(f"========== NEW LLM REQUEST ({model_name} via {provider}) ==========")
    if system_prompt:
        logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{prompt}")
    
    if "Gemini" in provider:
        import time
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No Gemini API key found. Set the GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable (or add it to a .env file)."
            )
        client = genai.Client(api_key=api_key)
        
        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=0.2,
            max_output_tokens=8192
        )
            
        max_retries = 5
        wait_times = [5, 10, 20, 40, 60]
        
        for i in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                break # Success!
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "high demand" in err_str or "quota" in err_str:
                    if i < max_retries - 1:
                        wait_time = wait_times[i]
                        logger.warning(f"LLM Temporary Error ({err_str[:50]}...). Retrying in {wait_time}s... (Attempt {i+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                logger.error(f"LLM Permanent Error: {str(e)}")
                raise Exception(f"LLM Request Failed: {str(e)}")
        
        # Log Tokens
        usage = response.usage_metadata
        token_info = f"Tokens: Input={usage.prompt_token_count}, Output={usage.candidates_token_count}, Total={usage.total_token_count}"
        logger.info(f"--- {token_info} ---")
        
        # Permanent Log
        log_token_usage(provider, model_name, usage.prompt_token_count, usage.candidates_token_count)
        
        logger.info(f"--- LLM RESPONSE ---\n{response.text}")
        return response.text
    else:
        # Assuming Ollama
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(model=model_name, messages=messages)
            
            # Log Tokens (Ollama metadata)
            prompt_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)
            token_info = f"Tokens: Input={prompt_tokens}, Output={output_tokens}, Total={prompt_tokens + output_tokens}"
            logger.info(f"--- {token_info} ---")
            
            # Permanent Log
            log_token_usage(provider, model_name, prompt_tokens, output_tokens)
            
            content = response['message']['content']
            logger.info(f"--- LLM RESPONSE ---\n{content}")
            return content
        except Exception as e:
            error_msg = f"Error: Could not connect to Ollama server for model '{model_name}'. Ensure Ollama is running locally. Original Error: {str(e)}"
            logger.error(error_msg)
            return error_msg
