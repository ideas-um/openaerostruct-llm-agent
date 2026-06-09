import os
import csv
import re
import sys
import time
import json
import shutil
import statistics
from datetime import datetime

# Add src to sys.path so we can import internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.router import route_intent
from llm.coder import generate_code
from tools.executor import execute_run

# ---------------------------------------------------------------------------
# __file__-relative paths
#
# benchmark.py lives in  src/tools/
# _TOOLS_DIR  = src/tools/
# _SRC_DIR    = src/
# _PROJECT_DIR = project root (parent of src/)
#
# Generated scripts write artifacts to src/openaerostruct_out/  (same as app.py)
# Benchmark history/results go to  <project_root>/benchmark_run_out/
# ---------------------------------------------------------------------------
_TOOLS_DIR   = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.dirname(_TOOLS_DIR)
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

_INPUT_FILE  = os.path.join(_TOOLS_DIR, "test_queries.csv")

# Where generated scripts write their output — project_root/openaerostruct_out/
_OAS_OUT_DIR  = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR    = os.path.join(_OAS_OUT_DIR, "agent_plots")
_GEN_RUN_DIR  = os.path.join(_OAS_OUT_DIR, "generated_run_out")

# Where the benchmark runner writes results and history
_BENCH_OUT_DIR = os.path.join(_PROJECT_DIR, "benchmark_run_out")

# Temporary script file used by the benchmark (mirrors app.py's generated_run.py)
_BENCH_SCRIPT = os.path.join(_SRC_DIR, "benchmark_run.py")

# MAX_RETRIES intentionally higher than app.py (3) so the benchmark measures
# the true capability ceiling rather than the production retry limit.
MAX_RETRIES = 5
NUM_REPS    = 10  # Number of repetitions per test case

# ---------------------------------------------------------------------------
# Convergence detection — mirrors app.py logic exactly
# ---------------------------------------------------------------------------
_CONVERGENCE_STDOUT = ["Optimization terminated successfully", "Optimization Complete"]

# ---------------------------------------------------------------------------
# Sanitize stderr/stdout before feeding back to the LLM — mirrors app.py
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
# Safety scan — mirrors app.py check_script_safety() exactly
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Destructive shell command (rm/rmdir/del)",
     re.compile(r"""(?x)
         os\.system\s*\(.*\brm\b
         | subprocess\.\w+\s*\(.*\brm\b
         | shutil\.rmtree
         | os\.remove\s*\(
         | os\.unlink\s*\(
         | os\.rmdir\s*\(
     """, re.IGNORECASE)),

    ("Network access (socket/requests/urllib/httpx)",
     re.compile(r"""(?x)
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
     """, re.IGNORECASE)),

    ("Arbitrary subprocess / shell execution",
     re.compile(r"""(?x)
         subprocess\.(run|call|Popen|check_output|check_call)\s*\(
         | os\.system\s*\(
         | os\.popen\s*\(
         | commands\.getoutput
     """, re.IGNORECASE)),

    ("File write outside allowed output paths",
     re.compile(r"""(?x)
         open\s*\(\s*['"]/(?!.*openaerostruct_out)
         | open\s*\(\s*['"]\.\.
         | open\s*\(\s*['"]~
     """, re.IGNORECASE)),

    ("Dynamic code execution (eval/exec/__import__)",
     re.compile(r"""(?x)
         \beval\s*\(
         | \bexec\s*\(
         | \b__import__\s*\(
         | \bcompile\s*\(.*exec
     """, re.IGNORECASE)),

    ("Environment / credential access",
     re.compile(r"""(?x)
         os\.environ\s*\[
         | os\.getenv\s*\(
         | keyring\.
     """, re.IGNORECASE)),
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
# CSV helpers
# ---------------------------------------------------------------------------
# Per-repetition CSV (one row per rep per case)
REP_HEADERS = [
    "id", "category", "query", "rep",
    "expected_blueprints", "selected_blueprints", "routing_correct",
    "attempts", "exit_code", "converged", "elapsed_s",
    "success", "error_log",
]

# Summary CSV (one row per case, aggregated across all reps)
SUMMARY_HEADERS = [
    "id", "category", "query",
    "expected_blueprints", "selected_blueprints",
    "num_runs",
    "routing_accuracy",
    "execution_success_rate",
    "convergence_rate",
    "attempts_mean", "attempts_median", "attempts_std", "attempts_min", "attempts_max",
    "elapsed_s_mean", "elapsed_s_median", "elapsed_s_std", "elapsed_s_min", "elapsed_s_max",
    "error_categories",
    "model",
    "max_retry_count",
]

def _append_result(results_file: str, row: dict, headers: list, write_header: bool):
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------
def _cleanup_run_artifacts():
    """Clear all contents of openaerostruct_out/ before each attempt, then recreate subdirs."""
    if os.path.exists(_OAS_OUT_DIR):
        for fname in os.listdir(_OAS_OUT_DIR):
            fpath = os.path.join(_OAS_OUT_DIR, fname)
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception:
                pass
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    os.makedirs(_GEN_RUN_DIR, exist_ok=True)

def _copy_artifacts(attempt_dir: str):
    """Copy everything produced in openaerostruct_out/ into artifacts/ inside the attempt dir."""
    artifacts_dst = os.path.join(attempt_dir, "artifacts")
    os.makedirs(artifacts_dst, exist_ok=True)

    if os.path.exists(_OAS_OUT_DIR):
        for fname in os.listdir(_OAS_OUT_DIR):
            src = os.path.join(_OAS_OUT_DIR, fname)
            dst = os.path.join(artifacts_dst, fname)
            try:
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            except Exception as e:
                print(f"    Warning: could not copy {src}: {e}")

# ---------------------------------------------------------------------------
# db summary helper — mirrors app.py logic
# ---------------------------------------------------------------------------
def _find_db_summary(rep_dir: str) -> str:
    """Look for an optimization DB in the standard locations and return a summary string."""
    try:
        from tools.db_reader import summarize_optimization
    except ImportError:
        return ""
    possible_paths = [
        os.path.join(_GEN_RUN_DIR, "aero.db"),
        os.path.join(_GEN_RUN_DIR, "aerostruct.db"),
        os.path.join(_GEN_RUN_DIR, "struct.db"),
        os.path.join(_OAS_OUT_DIR, "aero.db"),
        os.path.join(_OAS_OUT_DIR, "aerostruct.db"),
        os.path.join(_OAS_OUT_DIR, "struct.db"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            summary = summarize_optimization(path)
            if not summary.startswith("Error"):
                return summary
    return ""

# ---------------------------------------------------------------------------
# Single-repetition runner (core logic mirrors app.py while loop exactly)
# ---------------------------------------------------------------------------
def _run_single_rep(q: dict, rep_dir: str, model: str, provider: str) -> dict:
    """
    Run one repetition of a single test case.
    Returns a dict with all per-rep metrics.
    rep_dir: directory to store per-attempt files for this repetition.
    """
    case_id = q["id"]

    # ------------------------------------------------------------------
    # 1. Intent routing
    # ------------------------------------------------------------------
    selected = "ERROR"
    routing_correct = False
    blueprints = []

    try:
        routing = route_intent(q["query"], model_name=model, provider=provider)
        blueprints = routing.get("blueprints", [])
        selected = ", ".join(blueprints)

        is_vague = routing.get("is_vague", False)
        if is_vague:
            print(f"    WARNING: router marked query as vague — counted as routing failure")
            blueprints = blueprints or [json.loads(q["expected_blueprints"])[0]]

        def _norm(name):
            name = name.strip()
            return name if name.endswith(".py") else name + ".py"
        expected_set = set(_norm(b) for b in json.loads(q["expected_blueprints"]))
        selected_set = set(blueprints)
        routing_correct = (expected_set == selected_set)
        print(f"    Routing: {selected}  (correct={routing_correct})")
    except Exception as e:
        selected = "ERROR"
        blueprints = [json.loads(q["expected_blueprints"])[0]]
        print(f"    Routing Error: {e}  — falling back to expected blueprint")

    # ------------------------------------------------------------------
    # 2. Iterative generation + execution — mirrors app.py while loop
    # ------------------------------------------------------------------
    exit_code   = -1
    converged   = "n/a"
    success     = False
    attempt     = 0
    feedback    = "Initial generation"
    final_code  = ""
    error_logs  = []

    os.makedirs(rep_dir, exist_ok=True)

    while attempt < MAX_RETRIES:
        print(f"    Attempt {attempt + 1}/{MAX_RETRIES}...")
        _cleanup_run_artifacts()

        try:
            code, reasoning = generate_code(
                q["query"], blueprints, feedback,
                model_name=model, provider=provider,
            )
            final_code = code

            attempt_dir = os.path.join(rep_dir, f"attempt_{attempt + 1}")
            os.makedirs(attempt_dir, exist_ok=True)
            with open(os.path.join(attempt_dir, "reasoning.txt"), "w") as fh:
                fh.write(reasoning)
            with open(os.path.join(attempt_dir, "code.py"), "w") as fh:
                fh.write(code)

            # Safety scan — mirrors app.py check_script_safety() block
            violations = check_script_safety(code)
            if violations:
                violation_text = "\n".join(f"- {v}" for v in violations)
                err_msg = f"[attempt {attempt + 1}] Safety check failed:\n{violation_text}"
                error_logs.append(err_msg)
                print(f"      BLOCKED by safety scan. Retrying with feedback...")
                feedback = (
                    f"Your previous script was blocked by the safety checker. "
                    f"Do NOT include any of the following:\n{violation_text}\n"
                    f"Rewrite the script without these patterns."
                )
                attempt += 1
                continue

            with open(_BENCH_SCRIPT, "w") as fh:
                fh.write(code)

            exec_res = execute_run(_BENCH_SCRIPT, timeout=120)
            exit_code = exec_res.exit_code

            with open(os.path.join(attempt_dir, "execution.log"), "w") as fh:
                fh.write(f"--- STDOUT ---\n{exec_res.stdout}\n\n--- STDERR ---\n{exec_res.stderr}")

            _copy_artifacts(attempt_dir)

            if exit_code == 0:
                is_optimization = "run_driver()" in code
                converged_bool  = any(m in exec_res.stdout for m in _CONVERGENCE_STDOUT)

                if is_optimization:
                    if converged_bool:
                        converged = "yes"
                        success = True
                        print(f"      SUCCESS — converged on attempt {attempt + 1}")
                        break
                    else:
                        converged = "no"
                        err_msg = f"[attempt {attempt + 1}] Optimization did not converge"
                        error_logs.append(err_msg)
                        print(f"      Ran but did not converge. Retrying with feedback...")
                        db_summary = _find_db_summary(rep_dir)
                        feedback = (
                            f"Optimization failed to converge."
                            + (f" Results so far:\n{db_summary}\n\n" if db_summary else "\n\n")
                            + f"Stdout tail:\n{sanitize_feedback(exec_res.stdout, max_chars=400)}"
                        )
                else:
                    converged = "n/a"
                    success = True
                    print(f"      SUCCESS — analysis complete on attempt {attempt + 1}")
                    break
            else:
                err_msg = f"[attempt {attempt + 1}] {exec_res.stderr[-600:]}"
                error_logs.append(err_msg)
                print(f"      FAILED (exit {exit_code}). Retrying with error feedback...")
                feedback = f"Python Execution Error:\n{sanitize_feedback(exec_res.stderr, max_chars=1000)}"

        except Exception as e:
            err_msg = f"[attempt {attempt + 1}] Generation/Execution Error: {e}"
            error_logs.append(err_msg)
            print(f"      Generation/Execution Error: {e}")
            feedback = f"Code generation or execution error: {sanitize_feedback(str(e), max_chars=500)}"

        attempt += 1

    if not success:
        print(f"    FAILED after {MAX_RETRIES} attempts.")

    if final_code:
        with open(os.path.join(rep_dir, "final_code.py"), "w") as fh:
            fh.write(final_code)

    return {
        "selected_blueprints": selected,
        "routing_correct":     routing_correct,
        "attempts":            attempt + 1,
        "exit_code":           exit_code,
        "converged":           converged,
        "success":             success,
        "error_logs":          error_logs,
    }

# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(limit=None, model="gemini-flash-lite-latest", provider="Gemini API"):
    """
    Runs each test case NUM_REPS times to assess LLM stability.

    Directory layout under benchmark_run_out/run_<ts>/:
        rep_results.csv          — one row per (case, rep)
        benchmark_results.csv    — one row per case, aggregated over all reps
        run_metadata.json        — model, provider, max_retries, num_reps, hardware info
        case_<id>/
            rep_<n>/
                attempt_<k>/
                    code.py
                    reasoning.txt
                    execution.log
                    artifacts/
                final_code.py

    CSV is written after every rep so partial runs are never lost.
    """
    os.makedirs(_OAS_OUT_DIR, exist_ok=True)
    os.makedirs(_BENCH_OUT_DIR, exist_ok=True)

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_BENCH_OUT_DIR, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    rep_results_file     = os.path.join(run_dir, "rep_results.csv")
    summary_results_file = os.path.join(run_dir, "benchmark_results.csv")
    metadata_file        = os.path.join(run_dir, "run_metadata.json")

    metadata = {
        "model":           model,
        "provider":        provider,
        "max_retry_count": MAX_RETRIES,
        "num_reps":        NUM_REPS,
        "timestamp":       run_ts,
        "hardware_note":   "API-based inference; hardware and API conditions constant across all runs",
        "temperature":     "default (not explicitly set)",
        "sampling":        "default (not explicitly set)",
    }
    with open(metadata_file, "w") as fh:
        json.dump(metadata, fh, indent=2)

    queries = []
    with open(_INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)

    if limit:
        queries = queries[:limit]

    print(f"--- Starting Benchmark Test Suite ({len(queries)} cases × {NUM_REPS} reps, up to {MAX_RETRIES} attempts each) ---")
    print(f"Model:   {model} via {provider}")
    print(f"Run dir: {run_dir}\n")

    total_success = 0
    total_runs    = 0
    rep_row_idx   = 0
    sum_row_idx   = 0

    for idx, q in enumerate(queries):
        case_id  = q["id"]
        case_dir = os.path.join(run_dir, f"case_{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n[Case {case_id}] Category: {q['category']}")
        print(f"  Query: {q['query'][:120]}...")

        rep_successes        = 0
        rep_attempts_list    = []
        rep_elapsed_list     = []
        rep_converged_count  = 0
        rep_routing_correct  = []
        all_errors           = []
        selected_blueprints_last = "ERROR"
        optimization_reps = 0

        for rep in range(1, NUM_REPS + 1):
            print(f"\n  [Rep {rep}/{NUM_REPS}]")

            rep_dir    = os.path.join(case_dir, f"rep_{rep}")
            start_time = time.time()

            rep_result = _run_single_rep(q, rep_dir, model, provider)

            elapsed = round(time.time() - start_time, 2)

            if rep_result["success"]:
                rep_successes += 1
                total_success += 1
            if rep_result["converged"] == "yes":
                rep_converged_count += 1
            if rep_result["converged"] in ("yes", "no"):
                optimization_reps += 1

            rep_attempts_list.append(rep_result["attempts"])
            rep_elapsed_list.append(elapsed)
            rep_routing_correct.append(rep_result["routing_correct"])
            selected_blueprints_last = rep_result["selected_blueprints"]
            total_runs += 1

            all_errors.extend(rep_result["error_logs"])

            rep_row = {
                "id":                  case_id,
                "category":            q["category"],
                "query":               q["query"],
                "rep":                 rep,
                "expected_blueprints": q["expected_blueprints"],
                "selected_blueprints": rep_result["selected_blueprints"],
                "routing_correct":     rep_result["routing_correct"],
                "attempts":            rep_result["attempts"],
                "exit_code":           rep_result["exit_code"],
                "converged":           rep_result["converged"],
                "elapsed_s":           elapsed,
                "success":             rep_result["success"],
                "error_log":           " ||| ".join(rep_result["error_logs"]).replace("\n", " "),
            }
            _append_result(rep_results_file, rep_row, REP_HEADERS, write_header=(rep_row_idx == 0))
            rep_row_idx += 1

            print(f"  Rep {rep} done — success={rep_result['success']}  elapsed={elapsed}s  "
                  f"running case success rate: {rep_successes}/{rep}")

        n_reps = len(rep_attempts_list)

        attempts_mean   = round(sum(rep_attempts_list) / n_reps, 3)
        attempts_median = round(statistics.median(rep_attempts_list), 3)
        attempts_std    = round(statistics.stdev(rep_attempts_list), 3) if n_reps > 1 else 0.0
        attempts_min    = min(rep_attempts_list)
        attempts_max    = max(rep_attempts_list)

        elapsed_mean    = round(sum(rep_elapsed_list) / n_reps, 3)
        elapsed_median  = round(statistics.median(rep_elapsed_list), 3)
        elapsed_std     = round(statistics.stdev(rep_elapsed_list), 3) if n_reps > 1 else 0.0
        elapsed_min     = round(min(rep_elapsed_list), 3)
        elapsed_max     = round(max(rep_elapsed_list), 3)

        convergence_rate = (
            round(rep_converged_count / optimization_reps, 3)
            if optimization_reps > 0 else "n/a"
        )

        error_category_set = set()
        for err in all_errors:
            first_line = err.strip().splitlines()[0] if err.strip() else ""
            if first_line:
                error_category_set.add(first_line[:120])
        error_categories_str = " ||| ".join(sorted(error_category_set))

        summary_row = {
            "id":                    case_id,
            "category":              q["category"],
            "query":                 q["query"],
            "expected_blueprints":   q["expected_blueprints"],
            "selected_blueprints":   selected_blueprints_last,
            "num_runs":              n_reps,
            "routing_accuracy":      round(sum(rep_routing_correct) / n_reps, 3),
            "execution_success_rate": round(rep_successes / n_reps, 3),
            "convergence_rate":      convergence_rate,
            "attempts_mean":         attempts_mean,
            "attempts_median":       attempts_median,
            "attempts_std":          attempts_std,
            "attempts_min":          attempts_min,
            "attempts_max":          attempts_max,
            "elapsed_s_mean":        elapsed_mean,
            "elapsed_s_median":      elapsed_median,
            "elapsed_s_std":         elapsed_std,
            "elapsed_s_min":         elapsed_min,
            "elapsed_s_max":         elapsed_max,
            "error_categories":      error_categories_str,
            "model":                 model,
            "max_retry_count":       MAX_RETRIES,
        }
        _append_result(summary_results_file, summary_row, SUMMARY_HEADERS, write_header=(sum_row_idx == 0))

        errors_json_path = os.path.join(case_dir, "all_errors.json")
        with open(errors_json_path, "w") as fh:
            json.dump(all_errors, fh, indent=2)

        sum_row_idx += 1

        print(f"\n  Case {case_id} complete — {rep_successes}/{NUM_REPS} reps succeeded  "
              f"| Overall running success: {total_success}/{total_runs}")

    print(f"\n--- Benchmark Complete! ---")
    print(f"Run artifacts:    {run_dir}")
    print(f"Per-rep CSV:      {rep_results_file}")
    print(f"Summary CSV:      {summary_results_file}")
    print(f"Run metadata:     {metadata_file}")
    print(f"Overall Success Rate: {total_success}/{total_runs} reps  "
          f"({total_success / total_runs * 100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",    type=int, help="Limit number of test cases")
    parser.add_argument("--model",    type=str, default="gemini-flash-lite-latest")
    parser.add_argument("--provider", type=str, default="Gemini API")
    args = parser.parse_args()

    run_benchmark(limit=args.limit, model=args.model, provider=args.provider)