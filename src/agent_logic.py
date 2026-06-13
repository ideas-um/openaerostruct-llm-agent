"""
agent_logic.py — shared backend for app.py and benchmark.py.

Provides run_agent() which handles:
  - iterative code generation + execution
  - safety scanning
  - full error history accumulation
  - optional streaming via a callback for the Streamlit UI
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Callable, Optional

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_OUT_DIR = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR = os.path.join(_OUT_DIR, "agent_plots")
_GEN_RUN_DIR = os.path.join(_OUT_DIR, "generated_run_out")
_GEN_SCRIPT = os.path.join(_SRC_DIR, "generated_run.py")


def _approx_tokens(text: str) -> int:
    """Helper to estimate tokens without altering the hidden LLM API layer."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(str(text)))
    except ImportError:
        return len(str(text)) // 4


# ---------------------------------------------------------------------------
# Sanitize feedback
# ---------------------------------------------------------------------------
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")
_INJECTION_PATTERN = re.compile(
    r"(?i)("
    r"ignore\s+(previous|all|prior)\s+(instructions?|prompts?|context)"
    r"|act\s+as\s+(a|an)\s+"
    r"|forget\s+(everything|all)"
    r"|you\s+are\s+now\s"
    r"|disregard\s+(all|previous|prior)"
    r"|new\s+instruction"
    r"|\bsystem\s*:"
    r")"
)


def sanitize_feedback(text: str, max_chars: int = 1000) -> str:
    text = _ANSI_ESCAPE.sub("", text)
    lines = [line for line in text.splitlines() if not _INJECTION_PATTERN.search(line)]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# Safety scan
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "Destructive shell command (rm/rmdir/del)",
        re.compile(
            r"""(?x)
         os\.system\s*\(.*\brm\b
         | subprocess\.\w+\s*\(.*\brm\b
         | shutil\.rmtree
         | os\.remove\s*\(
         | os\.unlink\s*\(
         | os\.rmdir\s*\(
     """,
            re.IGNORECASE,
        ),
    ),
    (
        "Network access (socket/requests/urllib/httpx)",
        re.compile(
            r"""(?x)
         import\s+socket
         | from\s+socket\s+import
         | import\s+requests
         | from\s+requests\s+import
         | import\s+urllib
         | from\s+urllib\s+import
         | import\s+httpx
         | import\s+aiohttp
         | urllib\.request
         | requests\.(get|post|put|delete|patch|head|session)
         | socket\.connect
         | socket\.bind
     """,
            re.IGNORECASE,
        ),
    ),
    (
        "Arbitrary subprocess / shell execution",
        re.compile(
            r"""(?x)
         subprocess\.(run|call|Popen|check_output|check_call)\s*\(
         | os\.system\s*\(
         | os\.popen\s*\(
         | commands\.getoutput
     """,
            re.IGNORECASE,
        ),
    ),
    (
        "File write outside allowed output paths",
        re.compile(
            r"""(?x)
         open\s*\(\s*['"]/(?!.*openaerostruct_out)
         | open\s*\(\s*['"]\.\.
         | open\s*\(\s*['"]~
     """,
            re.IGNORECASE,
        ),
    ),
    (
        "Dynamic code execution (eval/exec/__import__)",
        re.compile(
            r"""(?x)
         \beval\s*\(
         | \bexec\s*\(
         | \b__import__\s*\(
         | \bcompile\s*\(.*exec
     """,
            re.IGNORECASE,
        ),
    ),
    (
        "Environment / credential access",
        re.compile(
            r"""(?x)
         os\.environ\s*\[
         | os\.getenv\s*\(
         | keyring\.
     """,
            re.IGNORECASE,
        ),
    ),
]


def check_script_safety(code: str) -> list[str]:
    violations = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for label, pattern in _DANGEROUS_PATTERNS:
            if pattern.search(line):
                violations.append(f"Line {lineno} [{label}]: `{stripped}`")
    return violations


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------
def cleanup_artifacts():
    for p in [_OUT_DIR, _PLOTS_DIR, _GEN_RUN_DIR]:
        if os.path.exists(p):
            for filename in os.listdir(p):
                file_path = os.path.join(p, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            os.makedirs(p, exist_ok=True)


def get_generated_plots() -> list[str]:
    plots = []
    if os.path.exists(_PLOTS_DIR):
        for f in os.listdir(_PLOTS_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                plots.append(os.path.join(_PLOTS_DIR, f))
    return sorted(plots)


def _find_db_summary() -> str:
    try:
        from tools.db_reader import summarize_optimization
    except ImportError:
        return "No optimization database found."
    possible_paths = [
        os.path.join(_GEN_RUN_DIR, "aero.db"),
        os.path.join(_GEN_RUN_DIR, "aerostruct.db"),
        os.path.join(_GEN_RUN_DIR, "struct.db"),
        os.path.join(_OUT_DIR, "aero.db"),
        os.path.join(_OUT_DIR, "aerostruct.db"),
        os.path.join(_OUT_DIR, "struct.db"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            summary = summarize_optimization(path)
            if not summary.startswith("Error"):
                return summary
    return "No optimization database found."


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    success: bool = False
    final_code: str = ""
    final_summary: str = ""
    plots: list[str] = field(default_factory=list)
    attempts: int = 0
    error_logs: list[str] = field(default_factory=list)
    converged: str = "n/a"
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------
def _get_relaxation_suggestion(
    user_prompt: str, error_logs: list, model_name: str, provider: str
) -> tuple[str, int, int]:
    from llm.config import get_llm_response

    prompt = (
        "An OpenAeroStruct optimization ran successfully (no Python errors) but the "
        "optimiser failed to converge after multiple attempts. This is a problem setup "
        "issue, not a code bug.\n\n"
        f"Original user request:\n{user_prompt}\n\n"
        "Convergence history:\n" + "\n".join(error_logs) + "\n\n"
        "Suggest 2–3 specific, concrete changes to the problem setup that would make "
        "it more likely to converge. Focus on: relaxing DV bounds, changing initial "
        "values, adding or removing design variables, loosening constraints, or "
        "adjusting flight conditions. Be specific with numbers. Keep the response "
        "under 150 words."
    )
    in_tok = _approx_tokens(prompt)
    try:
        ans = get_llm_response(prompt, model_name, provider=provider)
        return ans, in_tok, _approx_tokens(ans)
    except Exception:
        fallback = (
            "Could not generate a suggestion automatically. Consider relaxing bounds."
        )
        return fallback, in_tok, _approx_tokens(fallback)


def _build_feedback(error_history: list[str]) -> str:
    if not error_history:
        return "Initial generation"
    parts = []
    for i, err in enumerate(error_history, start=1):
        parts.append(f"--- Attempt {i} error ---\n{err}")
    return (
        "The following errors occurred on previous attempts. "
        "Study all of them to understand what has been tried and failed before writing new code.\n\n"
        + "\n\n".join(parts)
    )


def run_agent(
    user_prompt: str,
    blueprints: list[str],
    model_name: str,
    provider: str,
    max_retries: int = 3,
    stream: bool = False,
    callback: Optional[Callable[[str, dict], None]] = None,
    gen_script_path: Optional[str] = None,
    prior_error_logs: Optional[list[str]] = None,
    retry_on_no_converge: bool = False,
) -> AgentResult:

    from llm.coder import generate_code, generate_code_stream
    from tools.executor import execute_run

    script_path = gen_script_path or _GEN_SCRIPT

    def emit(event: str, data: dict):
        if callback is not None:
            callback(event, {"attempt": attempt + 1, **data})

    result = AgentResult()
    error_history: list[str] = list(prior_error_logs) if prior_error_logs else []
    attempt = 0

    while attempt < max_retries:
        emit("attempt_start", {"max_retries": max_retries})
        feedback = _build_feedback(error_history)

        code, reasoning = "", ""

        try:
            if stream and callback is not None:
                for chunk in generate_code_stream(
                    user_prompt,
                    blueprints,
                    feedback,
                    model_name=model_name,
                    provider=provider,
                ):
                    if isinstance(chunk, tuple):
                        code, reasoning, in_tok, out_tok = chunk
                        result.input_tokens += in_tok
                        result.output_tokens += out_tok
                    else:
                        emit("code_chunk", {"chunk": chunk})
            else:
                code, reasoning, in_tok, out_tok = generate_code(
                    user_prompt,
                    blueprints,
                    feedback,
                    model_name=model_name,
                    provider=provider,
                )
                result.input_tokens += in_tok
                result.output_tokens += out_tok
        except Exception as exc:
            err = f"Code generation error: {sanitize_feedback(str(exc), max_chars=500)}"
            error_history.append(err)
            result.error_logs.append(f"[attempt {attempt + 1}] {err}")
            attempt += 1
            continue

        result.final_code = code
        emit("code_ready", {"code": code, "reasoning": reasoning})

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        violations = check_script_safety(code)
        if violations:
            violation_text = "\n".join(f"- {v}" for v in violations)
            err = f"Script blocked by safety checker. Violations:\n{violation_text}"
            error_history.append(f"Code:\n```python\n{code}\n```\nError:\n{err}")
            result.error_logs.append(f"[attempt {attempt + 1}] {err}")
            emit("safety_blocked", {"violations": violations})
            attempt += 1
            continue

        exec_result = execute_run(script_path, timeout=120)

        if exec_result.exit_code == 0:
            db_summary = _find_db_summary()
            result.final_summary = db_summary
            plots = get_generated_plots()
            result.plots = plots
            emit("exec_success", {"db_summary": db_summary, "plots": plots})

            is_optimization = "run_driver()" in code
            if not is_optimization:
                result.converged = "n/a"
                result.success = True
                result.attempts = attempt + 1
                emit("done", {"success": True, "attempts": result.attempts})
                return result

            _stdout_lower = exec_result.stdout.lower()
            _failed = any(
                m in _stdout_lower
                for m in [
                    "optimization failed",
                    "exit mode 8",
                    "exit mode 9",
                    "exit mode 6",
                    "exit mode 7",
                    "maximum number of function",
                    "positive directional derivative",
                ]
            )
            _succeeded = any(
                m in _stdout_lower
                for m in [
                    "optimization terminated successfully",
                    "optimization complete",
                    "optimization successful",
                    "exit mode 0",
                ]
            )
            _no_signal = not _succeeded and not _failed
            converged_bool = (_succeeded and not _failed) or (
                _no_signal and db_summary and "No optimization" not in db_summary
            )

            if converged_bool:
                result.converged = "yes"
                result.success = True
                result.attempts = attempt + 1
                emit("done", {"success": True, "attempts": result.attempts})
                return result
            else:
                result.converged = "no"
                stdout_tail = sanitize_feedback(exec_result.stdout, max_chars=400)
                err_msg = f"Optimization did not converge.\n"
                if db_summary and "No optimization" not in db_summary:
                    err_msg += f"DB summary:\n{db_summary}\n\n"
                err_msg += f"Stdout tail:\n{stdout_tail}"

                result.error_logs.append(f"[attempt {attempt + 1}] {err_msg}")
                emit(
                    "no_converge",
                    {"db_summary": db_summary, "stdout_tail": stdout_tail},
                )

                if retry_on_no_converge:
                    error_history.append(
                        f"Code:\n```python\n{code}\n```\nError:\n{err_msg}"
                    )
                else:
                    result.attempts = attempt + 1
                    break
        else:
            err = f"Python Execution Error:\n{sanitize_feedback(exec_result.stderr, max_chars=1000)}"
            error_history.append(f"Code:\n```python\n{code}\n```\nError:\n{err}")
            result.error_logs.append(f"[attempt {attempt + 1}] {err}")
            emit("exec_error", {"stderr_tail": exec_result.stderr[-500:]})

        attempt += 1

    result.attempts = attempt

    # Generate relaxation suggestion if it failed to converge
    if result.converged == "no" and not retry_on_no_converge:
        suggestion, s_in, s_out = _get_relaxation_suggestion(
            user_prompt, result.error_logs, model_name, provider
        )
        result.input_tokens += s_in
        result.output_tokens += s_out
        emit(
            "no_converge_final",
            {
                "db_summary": result.final_summary,
                "error_logs": result.error_logs,
                "suggestion": suggestion,
            },
        )

    emit("done", {"success": False, "attempts": result.attempts})
    return result
