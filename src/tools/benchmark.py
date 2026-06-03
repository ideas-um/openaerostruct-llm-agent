import os
import csv
import re
import sys
import time
import shutil
from datetime import datetime

# Add src to sys.path so we can import internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.router import route_intent
from llm.coder import generate_code
from tools.executor import execute_run

# ---------------------------------------------------------------------------
# __file__-relative paths
# ---------------------------------------------------------------------------
_TOOLS_DIR    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR      = os.path.dirname(_TOOLS_DIR)
_INPUT_FILE   = os.path.join(_TOOLS_DIR, "test_queries.csv")
_OUTPUT_DIR   = os.path.join(_SRC_DIR, "openaerostruct_out")
_BENCH_SCRIPT = os.path.join(_SRC_DIR, "benchmark_run.py")

MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Convergence detection — mirrors app.py logic
# ---------------------------------------------------------------------------
_CONVERGENCE_STDOUT = ["Optimization terminated successfully", "Optimization Complete"]

_DIVERGENCE_KEYWORDS = [
    "iteration limit", "maximum iterations", "did not converge",
    "infeasible", "exitflag: -", "exit mode 8", "exit mode 9",
]

# ---------------------------------------------------------------------------
# Sanitize stderr/stdout before feeding back to the LLM (mirrors app.py)
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
    lines = [l for l in text.splitlines() if not _INJECTION_PATTERN.search(l)]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
HEADERS = [
    "id", "category", "query",
    "expected_blueprints", "selected_blueprints", "routing_correct",
    "is_vague", "attempts", "exit_code", "converged", "elapsed_s",
    "success", "error_log",
]

def _append_result(results_file: str, row: dict, write_header: bool):
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------------------------------------------------------
# Artifact helpers — mirrors app.py cleanup between attempts
# ---------------------------------------------------------------------------
def _cleanup_run_artifacts():
    """Clear agent_plots and generated_run_out before each attempt."""
    for subdir in ["agent_plots", "generated_run_out"]:
        p = os.path.join(_OUTPUT_DIR, subdir)
        if os.path.exists(p):
            for fname in os.listdir(p):
                fpath = os.path.join(p, fname)
                try:
                    if os.path.isfile(fpath) or os.path.islink(fpath):
                        os.unlink(fpath)
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath)
                except Exception:
                    pass
        else:
            os.makedirs(p, exist_ok=True)

def _copy_artifacts(case_dir: str, attempt: int):
    """Copy plots and .db files produced by the current attempt into case_dir."""
    attempt_dir = os.path.join(case_dir, f"attempt_{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)

    plot_dir = os.path.join(_OUTPUT_DIR, "agent_plots")
    if os.path.exists(plot_dir):
        for fname in os.listdir(plot_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                shutil.copy2(os.path.join(plot_dir, fname), os.path.join(attempt_dir, fname))

    generated_out = os.path.join(_OUTPUT_DIR, "generated_run_out")
    if os.path.exists(generated_out):
        for fname in os.listdir(generated_out):
            if fname.endswith(".db"):
                shutil.copy2(os.path.join(generated_out, fname), os.path.join(attempt_dir, fname))

    return attempt_dir

# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(limit=None, model="gemma-4-26b-a4b-it", provider="Gemini API"):
    """
    Mirrors the app's retry loop: up to MAX_RETRIES attempts per query,
    feeding stderr back to the LLM on failure — exactly as app.py does.
    CSV is written after every test case so partial runs are never lost.
    Each run gets a timestamped directory so history is preserved.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_OUTPUT_DIR, "benchmark_history", f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    results_file = os.path.join(run_dir, "benchmark_results.csv")

    queries = []
    with open(_INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)

    if limit:
        queries = queries[:limit]

    print(f"--- Starting Benchmark Test Suite ({len(queries)} cases, up to {MAX_RETRIES} attempts each) ---")
    print(f"Run dir: {run_dir}\n")

    total_success = 0

    for idx, q in enumerate(queries):
        case_id = q["id"]
        print(f"\n[Test {case_id}] Category: {q['category']}")
        print(f"  Query: {q['query'][:120]}...")

        start_time = time.time()

        # ------------------------------------------------------------------
        # 1. Intent routing (single call — same as app)
        # ------------------------------------------------------------------
        selected = "ERROR"
        is_vague = False
        routing_correct = False
        blueprints = []

        try:
            routing = route_intent(q["query"], model_name=model, provider=provider)
            blueprints = routing.get("blueprints", [])
            selected = ", ".join(blueprints)
            is_vague = routing.get("is_vague", False)
            expected_set = set(b.strip() for b in q["expected_blueprints"].split(","))
            selected_set = set(blueprints)
            routing_correct = (expected_set == selected_set)
            print(f"  Routing: {selected}  (correct={routing_correct})")
        except Exception as e:
            selected = "ERROR"
            print(f"  Routing Error: {e}")

        # ------------------------------------------------------------------
        # 2. Iterative generation + execution (mirrors app.py while loop)
        # ------------------------------------------------------------------
        exit_code  = -1
        converged  = "n/a"
        error_log  = ""
        success    = False
        attempt    = 0
        feedback   = "Initial generation"
        final_code = ""

        case_dir = os.path.join(run_dir, f"case_{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        if not is_vague:
            while attempt < MAX_RETRIES:
                print(f"  Attempt {attempt + 1}/{MAX_RETRIES}...")
                _cleanup_run_artifacts()

                try:
                    code, reasoning = generate_code(
                        q["query"], blueprints, feedback,
                        model_name=model, provider=provider,
                    )
                    final_code = code

                    # Save per-attempt code and reasoning
                    attempt_dir = os.path.join(case_dir, f"attempt_{attempt + 1}")
                    os.makedirs(attempt_dir, exist_ok=True)
                    with open(os.path.join(attempt_dir, "reasoning.txt"), "w") as fh:
                        fh.write(reasoning)
                    with open(os.path.join(attempt_dir, "code.py"), "w") as fh:
                        fh.write(code)
                    # Also save flat copy at case level as code_<trial>.py
                    with open(os.path.join(case_dir, f"code_{attempt + 1}.py"), "w") as fh:
                        fh.write(code)

                    with open(_BENCH_SCRIPT, "w") as fh:
                        fh.write(code)

                    exec_res = execute_run(_BENCH_SCRIPT, timeout=120)
                    exit_code = exec_res.exit_code

                    # Save execution log for this attempt
                    with open(os.path.join(attempt_dir, "execution.log"), "w") as fh:
                        fh.write(f"--- STDOUT ---\n{exec_res.stdout}\n\n--- STDERR ---\n{exec_res.stderr}")

                    # Copy plots and DBs
                    _copy_artifacts(case_dir, attempt + 1)

                    if exit_code == 0:
                        # Mirror app.py convergence check exactly
                        is_optimization = "run_driver()" in code
                        converged_bool = any(m in exec_res.stdout for m in _CONVERGENCE_STDOUT)

                        if is_optimization:
                            if converged_bool:
                                converged = "yes"
                                success = True
                                print(f"    SUCCESS — converged on attempt {attempt + 1}")
                                break
                            else:
                                converged = "no"
                                print(f"    Ran but did not converge. Retrying with feedback...")
                                feedback = (
                                    f"Optimization failed to converge.\n"
                                    f"Stdout tail:\n{sanitize_feedback(exec_res.stdout, max_chars=400)}"
                                )
                        else:
                            # Analysis run — clean exit is enough
                            converged = "n/a"
                            success = True
                            print(f"    SUCCESS — analysis complete on attempt {attempt + 1}")
                            break
                    else:
                        error_log = exec_res.stderr[-600:]
                        print(f"    FAILED (exit {exit_code}). Retrying with error feedback...")
                        feedback = f"Python Execution Error:\n{sanitize_feedback(exec_res.stderr, max_chars=1000)}"

                except Exception as e:
                    error_log = str(e)
                    print(f"    Generation/Execution Error: {e}")
                    feedback = f"Code generation or execution error: {sanitize_feedback(str(e), max_chars=500)}"

                attempt += 1

            if not success:
                print(f"  FAILED after {MAX_RETRIES} attempts.")

        else:
            success = True
            converged = "n/a"
            attempt = 0
            print("  VAGUE (clarification requested)")

        # Save final code at case level for quick inspection
        if final_code:
            with open(os.path.join(case_dir, "final_code.py"), "w") as fh:
                fh.write(final_code)

        elapsed = round(time.time() - start_time, 2)
        if success:
            total_success += 1

        row = {
            "id":                  case_id,
            "category":            q["category"],
            "query":               q["query"],
            "expected_blueprints": q["expected_blueprints"],
            "selected_blueprints": selected,
            "routing_correct":     routing_correct,
            "is_vague":            is_vague,
            "attempts":            attempt + 1 if not is_vague else 0,
            "exit_code":           exit_code,
            "converged":           converged,
            "elapsed_s":           elapsed,
            "success":             success,
            "error_log":           error_log.replace("\n", " "),
        }

        _append_result(results_file, row, write_header=(idx == 0))
        print(f"  Elapsed: {elapsed}s  |  Running success rate: {total_success}/{idx + 1}")

    n = len(queries)
    print(f"\n--- Benchmark Complete! ---")
    print(f"Run artifacts: {run_dir}")
    print(f"Results CSV:   {results_file}")
    print(f"Overall Success Rate: {total_success}/{n} ({total_success / n * 100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",    type=int, help="Limit number of tests")
    parser.add_argument("--model",    type=str, default="gemma-4-26b-a4b-it")
    parser.add_argument("--provider", type=str, default="Gemini API")
    args = parser.parse_args()

    run_benchmark(limit=args.limit, model=args.model, provider=args.provider)