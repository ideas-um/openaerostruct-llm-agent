import os
import csv
import logging
from dotenv import load_dotenv
import time as _time
import threading as _threading
import collections as _collections

# Load .env — try several candidate locations so the key is found regardless
# of which directory the script is run from:
#   1. project root  (parent of src/)
#   2. src/          (one level up from this file's llm/ package)
#   3. cwd           (wherever the caller launched from)
_LLM_DIR_ENV = os.path.dirname(os.path.abspath(__file__))  # src/llm/
_SRC_DIR_ENV = os.path.dirname(_LLM_DIR_ENV)  # src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR_ENV)  # project root
_env_loaded_from = None
for _env_candidate in [
    os.path.join(_PROJECT_ROOT, ".env"),
    os.path.join(_SRC_DIR_ENV, ".env"),
    os.path.join(os.getcwd(), ".env"),
]:
    if os.path.exists(_env_candidate):
        load_dotenv(_env_candidate, override=True, encoding="utf-8")
        _env_loaded_from = _env_candidate
        break

_api_key_found = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
print(f"[config] .env loaded from: {_env_loaded_from}")
print(f"[config] Gemini API key configured: {_api_key_found}")

# ---------------------------------------------------------------------------
# Resolve paths relative to this file.
# ---------------------------------------------------------------------------
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_LLM_DIR)
_LOG_FILE = os.path.join(_SRC_DIR, "agent_backend.log")
_STATS_FILE = os.path.join(_SRC_DIR, "usage_stats.csv")

# ---------------------------------------------------------------------------
# Gemini streaming retry config — imported by coder.py and router.py.
# get_llm_response() handles non-streaming retries internally; these
# constants are only needed for the streaming paths that call the SDK
# directly and therefore bypass get_llm_response().
# ---------------------------------------------------------------------------
GEMINI_STREAM_RETRY_WAIT = 60  # seconds to wait before retrying a stream
GEMINI_STREAM_MAX_RETRIES = 3  # maximum stream retries per call

_GEMINI_TRANSIENT_MESSAGES = (
    "resource_exhausted",
    "quota",
    "rate limit",
    "overloaded",
    "503",
    "429",
    "service unavailable",
    "too many requests",
)


def is_gemini_transient_error(exc: Exception) -> bool:
    """Return True if the exception looks like a Gemini rate-limit / overload error."""
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _GEMINI_TRANSIENT_MESSAGES)


# ---------------------------------------------------------------------------
# Proactive rate limiter — keeps Gemini calls under RPM_LIMIT per minute.
# Both get_llm_response() (non-streaming) and get_llm_client() (streaming)
# call _gemini_rate_limit() before dispatching, so the cap is enforced on
# every path automatically. No changes needed elsewhere.
# ---------------------------------------------------------------------------

RPM_LIMIT = 14  # stay one below the hard 15-RPM cap for safety
_RL_WINDOW = 60.0  # sliding window in seconds
_rl_lock = _threading.Lock()
_rl_times: _collections.deque = _collections.deque()


def _gemini_rate_limit() -> None:
    """Block until issuing a Gemini call would not exceed RPM_LIMIT/min."""
    while True:
        with _rl_lock:
            now = _time.monotonic()
            # Drop timestamps older than the window
            while _rl_times and _rl_times[0] <= now - _RL_WINDOW:
                _rl_times.popleft()
            if len(_rl_times) < RPM_LIMIT:
                _rl_times.append(now)
                return  # under the cap — proceed
            # Oldest call drops out of the window after this many seconds
            wait = _RL_WINDOW - (now - _rl_times[0]) + 0.05
        # Sleep outside the lock so other threads can check concurrently
        _time.sleep(wait)


logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LLM_Backend")


def is_gemini_provider(provider: str) -> bool:
    return "gemini" in (provider or "").lower()


def is_ollama_provider(provider: str) -> bool:
    return "ollama" in (provider or "").lower()


def _make_gemini_client():
    """Create and return a Gemini client using the API key from the environment."""
    from google import genai

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
    if is_gemini_provider(provider):
        _gemini_rate_limit()
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
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    total = input_tokens + output_tokens

    exists = os.path.isfile(_STATS_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_STATS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Provider",
                    "Model",
                    "InputTokens",
                    "OutputTokens",
                    "TotalTokens",
                ]
            )
        writer.writerow(
            [timestamp, provider, model, input_tokens, output_tokens, total]
        )


def get_llm_response(
    prompt: str,
    model_name: str,
    system_prompt: str = None,
    provider: str = "Gemini API",
) -> str:
    """
    Send a prompt to either Gemini or Ollama and return the response text.
    Retries automatically on transient Gemini errors (503, quota exhaustion).
    """
    logger.info(f"========== NEW LLM REQUEST ({model_name} via {provider}) ==========")
    if system_prompt:
        logger.info(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    logger.info(f"--- USER PROMPT ---\n{prompt}")

    if is_gemini_provider(provider):
        import time
        from google.genai import types

        _gemini_rate_limit()
        client = _make_gemini_client()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=0.2,
            max_output_tokens=8192,
        )

        max_retries = 5
        wait_times = [5, 10, 20, 40, 60]

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
                            f"Retrying in {wait_time}s (attempt {i + 1}/{max_retries})"
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
        log_token_usage(
            provider, model_name, usage.prompt_token_count, usage.candidates_token_count
        )
        logger.info(f"--- LLM RESPONSE ---\n{response.text}")
        return response.text

    elif is_ollama_provider(provider):
        # Ollama path
        try:
            import ollama

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(model=model_name, messages=messages)

            prompt_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)
            logger.info(
                f"Tokens — input: {prompt_tokens}, output: {output_tokens}, "
                f"total: {prompt_tokens + output_tokens}"
            )
            log_token_usage(provider, model_name, prompt_tokens, output_tokens)

            content = response["message"]["content"]
            logger.info(f"--- LLM RESPONSE ---\n{content}")
            return content

        except Exception as e:
            error_msg = (
                f"Error: could not connect to Ollama for model '{model_name}'. "
                f"Make sure Ollama is running locally. Details: {e}"
            )
            logger.error(error_msg)
            return error_msg
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Expected Gemini API or Ollama."
        )
