import os
import csv
import sys
import time
import json
import shutil
import statistics
from datetime import datetime

from llm.router import route_intent
from agent_logic import run_agent, AgentResult

# ---------------------------------------------------------------------------
# __file__-relative paths
#
# benchmark.py lives in  src/  (same level as app.py)
# _SRC_DIR    = src/
# _PROJECT_DIR = project root (parent of src/)
# ---------------------------------------------------------------------------
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

_INPUT_FILE  = os.path.join(_SRC_DIR, "tools", "test_queries.csv")

# Where generated scripts write their output — project_root/openaerostruct_out/
_OAS_OUT_DIR  = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR    = os.path.join(_OAS_OUT_DIR, "agent_plots")
_GEN_RUN_DIR  = os.path.join(_OAS_OUT_DIR, "generated_run_out")

# Where the benchmark runner writes results and history
_BENCH_OUT_DIR = os.path.join(_PROJECT_DIR, "benchmark_run_out")

# Temporary script file (same dir as benchmark, mirrors generated_run.py)
_BENCH_SCRIPT = os.path.join(_SRC_DIR, "benchmark_run.py")

# MAX_RETRIES intentionally higher than app.py (3) so the benchmark measures
# the true capability ceiling rather than the production retry limit.
MAX_RETRIES = 5
NUM_REPS    = 1  # Number of repetitions per test case

# Safety scan, sanitize_feedback, and check_script_safety are defined once
# in agent_logic.py and used there. benchmark.py delegates to run_agent().

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
# Single-repetition runner — delegates to agent_logic.run_agent()
# ---------------------------------------------------------------------------
def _run_single_rep(q: dict, rep_dir: str, model: str, provider: str) -> dict:
    """
    Run one repetition of a single test case.
    Routing is done here (benchmark needs the routing metrics); execution is
    fully delegated to run_agent() in agent_logic so the logic is identical
    to what app.py uses.

    Returns a dict with all per-rep metrics.
    """
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
            print("    WARNING: router marked query as vague — counted as routing failure")
            blueprints = blueprints or [json.loads(q["expected_blueprints"])[0]]

        def _norm(name: str) -> str:
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
    # 2. Iterative generation + execution via shared agent_logic
    # ------------------------------------------------------------------
    os.makedirs(rep_dir, exist_ok=True)

    # Per-attempt logging callback so the benchmark still captures artifacts.
    attempt_dirs: dict[int, str] = {}

    def bench_callback(event: str, data: dict):
        attempt = data.get("attempt", 0)

        if event == "attempt_start":
            _cleanup_run_artifacts()
            attempt_dir = os.path.join(rep_dir, f"attempt_{attempt}")
            os.makedirs(attempt_dir, exist_ok=True)
            attempt_dirs[attempt] = attempt_dir
            print(f"    Attempt {attempt}/{data['max_retries']}...")

        elif event == "code_ready":
            attempt_dir = attempt_dirs.get(attempt, rep_dir)
            with open(os.path.join(attempt_dir, "reasoning.txt"), "w") as fh:
                fh.write(data.get("reasoning", ""))
            with open(os.path.join(attempt_dir, "code.py"), "w") as fh:
                fh.write(data.get("code", ""))

        elif event in ("exec_success", "exec_error", "no_converge"):
            attempt_dir = attempt_dirs.get(attempt, rep_dir)
            # Copy artifacts produced by this attempt
            _copy_artifacts(attempt_dir)
            # Write execution log if exit code info is available
            log_path = os.path.join(attempt_dir, "execution.log")
            if event == "exec_error":
                with open(log_path, "w") as fh:
                    fh.write(f"--- STDERR TAIL ---\n{data.get('stderr_tail', '')}")
            if event == "exec_success":
                with open(log_path, "w") as fh:
                    fh.write(f"--- DB SUMMARY ---\n{data.get('db_summary', '')}")

        elif event == "done":
            status = "SUCCESS" if data["success"] else f"FAILED after {data['attempts']} attempts"
            print(f"      {status}")

    result: AgentResult = run_agent(
        user_prompt=q["query"],
        blueprints=blueprints,
        model_name=model,
        provider=provider,
        max_retries=MAX_RETRIES,
        stream=False,
        callback=bench_callback,
        gen_script_path=_BENCH_SCRIPT,
    )

    if result.final_code:
        with open(os.path.join(rep_dir, "final_code.py"), "w") as fh:
            fh.write(result.final_code)

    # Map AgentResult.converged back to the exit_code convention used by the
    # benchmark CSV (exit_code = 0 on any successful run, -1 on failure).
    exit_code = 0 if result.success else -1

    return {
        "selected_blueprints": selected,
        "routing_correct":     routing_correct,
        "attempts":            result.attempts,
        "exit_code":           exit_code,
        "converged":           result.converged,
        "success":             result.success,
        "error_logs":          result.error_logs,
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