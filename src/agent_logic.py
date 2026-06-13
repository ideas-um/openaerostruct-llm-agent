"""
agent_logic.py — Hardened Memory & Error Feedback
"""

from __future__ import annotations
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_OUT_DIR = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR = os.path.join(_OUT_DIR, "agent_plots")
_GEN_RUN_DIR = os.path.join(_OUT_DIR, "generated_run_out")
_GEN_SCRIPT = os.path.join(_SRC_DIR, "generated_run.py")


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(str(text)))
    except ImportError:
        return len(str(text)) // 4


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")
_INJECTION_PATTERN = re.compile(
    r"(?i)(ignore\s+(previous|all|prior)\s+(instructions?|prompts?|context)|act\s+as\s+(a|an)\s+|forget\s+(everything|all)|you\s+are\s+now\s|disregard\s+(all|previous|prior)|new\s+instruction|\bsystem\s*:)"
)


def sanitize_feedback(text: str, max_chars: int = 1000) -> str:
    text = _ANSI_ESCAPE.sub("", text)
    lines = [line for line in text.splitlines() if not _INJECTION_PATTERN.search(line)]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "Destructive command",
        re.compile(
            r"""(?x)os\.system\s*\(.*\brm\b | subprocess\.\w+\s*\(.*\brm\b | shutil\.rmtree | os\.remove\s*\( | os\.unlink\s*\( | os\.rmdir\s*\(""",
            re.IGNORECASE,
        ),
    ),
    (
        "Network access",
        re.compile(
            r"""(?x)import\s+socket | from\s+socket\s+import | import\s+requests | from\s+requests\s+import | import\s+urllib | from\s+urllib\s+import | import\s+httpx | import\s+aiohttp | urllib\.request | requests\.(get|post|put|delete|patch|head|session) | socket\.connect | socket\.bind""",
            re.IGNORECASE,
        ),
    ),
    (
        "Subprocess",
        re.compile(
            r"""(?x)subprocess\.(run|call|Popen|check_output|check_call)\s*\( | os\.system\s*\( | os\.popen\s*\( | commands\.getoutput""",
            re.IGNORECASE,
        ),
    ),
    (
        "File write",
        re.compile(
            r"""(?x)open\s*\(\s*['"]/(?!.*openaerostruct_out) | open\s*\(\s*['"]\.\. | open\s*\(\s*['"]~""",
            re.IGNORECASE,
        ),
    ),
    (
        "Eval/Exec",
        re.compile(
            r"""(?x)\beval\s*\( | \bexec\s*\( | \b__import__\s*\( | \bcompile\s*\(.*exec""",
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


def cleanup_artifacts():
    for p in [_OUT_DIR, _PLOTS_DIR, _GEN_RUN_DIR]:
        if os.path.exists(p):
            for filename in os.listdir(p):
                file_path = os.path.join(p, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass
        else:
            os.makedirs(p, exist_ok=True)


def get_generated_plots() -> list[str]:
    plots = []
    if os.path.exists(_PLOTS_DIR):
        for f in os.listdir(_PLOTS_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                plots.append(os.path.join(_PLOTS_DIR, f))
    return sorted(plots)


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


def _find_db_summary() -> str:
    """Safely attempts to load the database reader module."""
    try:
        from tools.db_reader import summarize_optimization
    except ImportError:
        try:
            # Fallback path if the execution root is the project base
            from src.tools.db_reader import summarize_optimization
        except ImportError:
            return "No optimization database found."

    paths = [
        os.path.join(_GEN_RUN_DIR, "aero.db"),
        os.path.join(_OUT_DIR, "aero.db"),
        os.path.join(_OUT_DIR, "aerostruct.db"),
    ]
    for p in paths:
        if os.path.exists(p):
            res = summarize_optimization(p)
            if not res.startswith("Error"):
                return res
    return "No optimization database found."


def _get_relaxation_suggestion(
    user_prompt: str, error_logs: list, model_name: str, provider: str
) -> tuple[str, int, int]:
    from llm.config import get_llm_response

    prompt = (
        f"An OpenAeroStruct optimization ran but failed to converge...\nUser request: {user_prompt}\n"
        f"History: {error_logs}\nSuggest 2-3 concrete relaxations under 150 words."
    )
    in_t = _approx_tokens(prompt)
    try:
        ans = get_llm_response(prompt, model_name, provider=provider)
        return ans, in_t, _approx_tokens(ans)
    except Exception:
        return "Relax bounds.", in_t, 5


def _build_feedback(error_history: list[str], prior_code: str = "") -> str:
    parts = []
    if prior_code:
        parts.append(
            "### BASELINE WORKING CODE (DO NOT BREAK THIS STRUCTURE) ###\n"
            "The following code is syntactically correct and includes necessary imports. "
            "Use it as your template and only make surgical changes:\n"
            f"```python\n{prior_code}\n```"
        )

    if error_history:
        parts.append("### RECENT ERRORS AND FEEDBACK ###")
        for i, err in enumerate(error_history, start=1):
            parts.append(f"--- Attempt {i} error ---\n{err}")

    return "\n\n".join(parts) if parts else "Initial generation"


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
    prior_code: str = "",
) -> AgentResult:

    from llm.coder import generate_code, generate_code_stream
    from tools.executor import execute_run

    script_path = gen_script_path or _GEN_SCRIPT

    def emit(event: str, data: dict):
        if callback:
            callback(event, {"attempt": attempt + 1, **data})

    result = AgentResult()
    error_history = prior_error_logs or []
    attempt = 0

    while attempt < max_retries:
        emit("attempt_start", {"max_retries": max_retries})
        feedback = _build_feedback(error_history, prior_code)

        try:
            if stream and callback:
                for chunk in generate_code_stream(
                    user_prompt, blueprints, feedback, model_name, provider
                ):
                    if isinstance(chunk, tuple):
                        code, reasoning, in_tok, out_tok = chunk
                        result.input_tokens += in_tok
                        result.output_tokens += out_tok
                    else:
                        emit("code_chunk", {"chunk": chunk})
            else:
                code, reasoning, in_tok, out_tok = generate_code(
                    user_prompt, blueprints, feedback, model_name, provider
                )
                result.input_tokens += in_tok
                result.output_tokens += out_tok
        except Exception as exc:
            err = f"Generation error: {sanitize_feedback(str(exc))}"
            error_history.append(err)
            result.error_logs.append(err)
            attempt += 1
            continue

        result.final_code = code
        emit("code_ready", {"code": code, "reasoning": reasoning})
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        violations = check_script_safety(code)
        if violations:
            err = f"Safety block: {violations}"
            error_history.append(f"Code:\n{code}\nError:\n{err}")
            result.error_logs.append(err)
            emit("safety_blocked", {"violations": violations})
            attempt += 1
            continue

        exec_res = execute_run(script_path, timeout=120)
        _stderr = exec_res.stderr.lower()
        _is_solver_fail = any(
            k in _stderr
            for k in ["analysiserror", "failed to converge", "nlbgs failed"]
        )

        if exec_res.exit_code == 0:
            db_sum = _find_db_summary()
            result.final_summary = db_sum
            result.plots = get_generated_plots()
            emit("exec_success", {"db_summary": db_sum, "plots": result.plots})

            # Guard 1: Verify code is not empty or truncated
            if not code or len(code.strip()) < 100:
                err = "Python error: Generated script was empty or incomplete."
                error_history.append(err)
                result.error_logs.append(err)
                attempt += 1
                continue

            # Guard 2: Enforce optimization convergence for optimization tasks
            is_opt_blueprint = any(
                b
                in [
                    "aero_opt.py",
                    "aero_multipoint.py",
                    "struct_optimization.py",
                    "aerostruct_tube.py",
                    "aerostruct_wingbox.py",
                ]
                for b in blueprints
            )

            # If the blueprint is optimization but 'run_driver()' is omitted,
            # the result would incorrectly be set to converged = 'n/a'.
            if is_opt_blueprint and "run_driver()" not in code:
                err = "The task requires optimization, but no convergence was caught."
                error_history.append(err)
                result.error_logs.append(err)
                attempt += 1
                continue

            converged = any(
                k in exec_res.stdout.lower()
                for k in ["successfully", "complete", "exit mode 0"]
            )

            if converged or "run_driver()" not in code:
                result.success = True
                result.converged = "yes" if converged else "n/a"
                result.attempts = attempt + 1
                emit("done", {"success": True, "attempts": result.attempts})
                return result

        elif _is_solver_fail:
            result.converged = "no"
            err = f"Solver Error (Physics Failure): {sanitize_feedback(exec_res.stderr, 400)}"
            result.error_logs.append(err)
            emit("no_converge", {"db_summary": _find_db_summary(), "stdout_tail": err})

            # FIX: In Benchmark mode, feed physics crash back and keep going.
            if retry_on_no_converge:
                error_history.append(f"Code:\n{code}\nError:\n{err}")
            else:
                result.attempts = attempt + 1
                break

        else:
            err = f"Python error:\n{sanitize_feedback(exec_res.stderr)}"
            error_history.append(f"Code:\n{code}\nError:\n{err}")
            result.error_logs.append(f"[attempt {attempt + 1}] {err}")
            emit("exec_error", {"stderr_tail": exec_res.stderr[-500:]})

        attempt += 1

    # Only provide suggestion if we didn't succeed and are in non-retry (App) mode
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

    result.attempts = attempt
    emit("done", {"success": False, "attempts": attempt})
    return result
