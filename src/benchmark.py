import os
import csv
import sys
import time
import json
import shutil
import logging
import statistics
from datetime import datetime

from llm.router import route_intent_stream
from agent_logic import run_agent, AgentResult

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

_INPUT_FILE = os.path.join(_SRC_DIR, "tools", "test_queries.csv")
_OAS_OUT_DIR = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR = os.path.join(_OAS_OUT_DIR, "agent_plots")
_GEN_RUN_DIR = os.path.join(_OAS_OUT_DIR, "generated_run_out")
_BENCH_OUT_DIR = os.path.join(_PROJECT_DIR, "benchmark_run_out")
_BENCH_SCRIPT = os.path.join(_SRC_DIR, "benchmark_run.py")

DEFAULT_MAX_RETRIES = 5
NUM_REPS = 10

# ---------------------------------------------------------------------------
# CSV headers
# ---------------------------------------------------------------------------
REP_HEADERS = [
    "id",
    "category",
    "query",
    "rep",
    "expected_blueprints",
    "selected_blueprints",
    "routing_correct",
    "attempts",
    "exit_code",
    "converged",
    "elapsed_s",
    "input_tokens",
    "output_tokens",
    "success",
    "error_log",
]

SUMMARY_HEADERS = [
    "id",
    "category",
    "query",
    "expected_blueprints",
    "selected_blueprints",
    "num_runs",
    "routing_accuracy",
    "execution_success_rate",
    "convergence_rate",
    "attempts_mean",
    "attempts_median",
    "attempts_std",
    "attempts_min",
    "attempts_max",
    "elapsed_s_mean",
    "elapsed_s_median",
    "elapsed_s_std",
    "elapsed_s_min",
    "elapsed_s_max",
    "input_tokens_mean",
    "output_tokens_mean",
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


def _cleanup_run_artifacts():
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
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Single Repetition Runner
# ---------------------------------------------------------------------------
def _run_single_rep(
    q: dict, rep_dir: str, model: str, provider: str, max_retries: int
) -> dict:
    os.makedirs(rep_dir, exist_ok=True)

    # Setup Dedicated File Logger for this repetition
    log_file_path = os.path.join(rep_dir, "agent_backend.log")
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    backend_logger = logging.getLogger("LLM_Backend")
    backend_logger.addHandler(file_handler)

    selected = "ERROR"
    routing_correct = False
    blueprints = []

    # 1. Intent routing
    routing_data = {}
    try:
        for chunk in route_intent_stream(
            q["query"], model_name=model, provider=provider
        ):
            if isinstance(chunk, dict):
                routing_data = chunk

        blueprints = routing_data.get("blueprints", [])
        selected = ", ".join(blueprints)

        is_vague = routing_data.get("is_vague", False)
        if is_vague:
            print(
                "    WARNING: router marked query as vague — using fallback for benchmark"
            )
            blueprints = blueprints or [json.loads(q["expected_blueprints"])[0]]

        def _norm(name: str) -> str:
            return name if name.endswith(".py") else name + ".py"

        expected_set = set(_norm(b) for b in json.loads(q["expected_blueprints"]))
        routing_correct = expected_set == set(blueprints)
        print(f"    Routing: {selected}  (correct={routing_correct})")
    except Exception as e:
        selected = "ERROR"
        blueprints = [json.loads(q["expected_blueprints"])[0]]
        print(f"    Routing Error: {e}")

    attempt_dirs: dict[int, str] = {}

    def bench_callback(event: str, data: dict):
        attempt = data.get("attempt", 0)
        if event == "attempt_start":
            _cleanup_run_artifacts()
            attempt_dir = os.path.join(rep_dir, f"attempt_{attempt}")
            os.makedirs(attempt_dir, exist_ok=True)
            attempt_dirs[attempt] = attempt_dir
        elif event == "code_ready":
            attempt_dir = attempt_dirs.get(attempt, rep_dir)
            with open(os.path.join(attempt_dir, "code.py"), "w") as fh:
                fh.write(data.get("code", ""))
        elif event in ("exec_success", "exec_error", "no_converge"):
            attempt_dir = attempt_dirs.get(attempt, rep_dir)
            _copy_artifacts(attempt_dir)

    # 2. Execution
    result: AgentResult = run_agent(
        user_prompt=q["query"],
        blueprints=blueprints,
        model_name=model,
        provider=provider,
        max_retries=max_retries,
        stream=True,
        callback=bench_callback,
        gen_script_path=_BENCH_SCRIPT,
        retry_on_no_converge=True,
    )

    if result.final_code:
        with open(os.path.join(rep_dir, "final_code.py"), "w") as fh:
            fh.write(result.final_code)

    exit_code = 0 if result.success else -1

    # Cleanup and release current repetition log handler
    backend_logger.removeHandler(file_handler)
    file_handler.close()

    return {
        "selected_blueprints": selected,
        "routing_correct": routing_correct,
        "attempts": result.attempts,
        "exit_code": exit_code,
        "converged": result.converged,
        "success": result.success,
        "error_logs": result.error_logs,
        "input_tokens": routing_data.get("input_tokens", 0) + result.input_tokens,
        "output_tokens": routing_data.get("output_tokens", 0) + result.output_tokens,
    }


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------
def run_benchmark(
    limit=None,
    model="gemini-flash-lite-latest",
    provider="Gemini API",
    max_retries=DEFAULT_MAX_RETRIES,
):
    os.makedirs(_OAS_OUT_DIR, exist_ok=True)
    os.makedirs(_BENCH_OUT_DIR, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_BENCH_OUT_DIR, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    rep_results_file = os.path.join(run_dir, "rep_results.csv")
    summary_results_file = os.path.join(run_dir, "benchmark_results.csv")
    metadata_file = os.path.join(run_dir, "run_metadata.json")

    with open(metadata_file, "w") as fh:
        json.dump(
            {
                "model": model,
                "provider": provider,
                "max_retry_count": max_retries,
                "num_reps": NUM_REPS,
                "timestamp": run_ts,
            },
            fh,
            indent=2,
        )

    queries = []
    with open(_INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    if limit:
        queries = queries[:limit]

    print(f"--- Starting Benchmark ({len(queries)} cases × {NUM_REPS} reps) ---")

    total_success, total_runs, rep_row_idx, sum_row_idx = 0, 0, 0, 0

    for idx, q in enumerate(queries):
        case_id = q["id"]
        case_dir = os.path.join(run_dir, f"case_{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n[Case {case_id}] {q['category']}: {q['query'][:80]}...")

        rep_successes, rep_converged_count, opt_reps = 0, 0, 0
        rep_attempts_list, rep_elapsed_list, rep_in_tok, rep_out_tok = [], [], [], []
        rep_routing_correct, all_errors = [], []
        selected_last = "ERROR"

        for rep in range(1, NUM_REPS + 1):
            print(f"  [Rep {rep}/{NUM_REPS}]", end=" ", flush=True)
            rep_dir = os.path.join(case_dir, f"rep_{rep}")
            start_time = time.time()
            res = _run_single_rep(q, rep_dir, model, provider, max_retries)
            elapsed = round(time.time() - start_time, 2)

            if res["success"]:
                rep_successes += 1
                total_success += 1
            if res["converged"] == "yes":
                rep_converged_count += 1
            if res["converged"] in ("yes", "no"):
                opt_reps += 1

            rep_attempts_list.append(res["attempts"])
            rep_elapsed_list.append(elapsed)
            rep_in_tok.append(res["input_tokens"])
            rep_out_tok.append(res["output_tokens"])
            rep_routing_correct.append(res["routing_correct"])
            selected_last = res["selected_blueprints"]
            total_runs += 1
            all_errors.extend(res["error_logs"])

            rep_row = {
                "id": case_id,
                "category": q["category"],
                "query": q["query"],
                "rep": rep,
                "expected_blueprints": q["expected_blueprints"],
                "selected_blueprints": res["selected_blueprints"],
                "routing_correct": res["routing_correct"],
                "attempts": res["attempts"],
                "exit_code": res["exit_code"],
                "converged": res["converged"],
                "elapsed_s": elapsed,
                "input_tokens": res["input_tokens"],
                "output_tokens": res["output_tokens"],
                "success": res["success"],
                "error_log": " ||| ".join(res["error_logs"]).replace("\n", " "),
            }
            _append_result(
                rep_results_file, rep_row, REP_HEADERS, write_header=(rep_row_idx == 0)
            )
            rep_row_idx += 1
            print(f"Done (success={res['success']}, {elapsed}s)")

        n_reps = len(rep_attempts_list)
        summary_row = {
            "id": case_id,
            "category": q["category"],
            "query": q["query"],
            "expected_blueprints": q["expected_blueprints"],
            "selected_blueprints": selected_last,
            "num_runs": n_reps,
            "routing_accuracy": round(sum(rep_routing_correct) / n_reps, 3),
            "execution_success_rate": round(rep_successes / n_reps, 3),
            "convergence_rate": round(rep_converged_count / opt_reps, 3)
            if opt_reps > 0
            else "n/a",
            "attempts_mean": round(statistics.mean(rep_attempts_list), 3),
            "attempts_median": statistics.median(rep_attempts_list),
            "attempts_std": round(statistics.stdev(rep_attempts_list), 3)
            if n_reps > 1
            else 0,
            "attempts_min": min(rep_attempts_list),
            "attempts_max": max(rep_attempts_list),
            "elapsed_s_mean": round(statistics.mean(rep_elapsed_list), 3),
            "elapsed_s_median": statistics.median(rep_elapsed_list),
            "elapsed_s_std": round(statistics.stdev(rep_elapsed_list), 3)
            if n_reps > 1
            else 0,
            "elapsed_s_min": min(rep_elapsed_list),
            "elapsed_s_max": max(rep_elapsed_list),
            "input_tokens_mean": int(statistics.mean(rep_in_tok)),
            "output_tokens_mean": int(statistics.mean(rep_out_tok)),
            "error_categories": " ||| ".join(
                sorted(set(e.splitlines()[0][:100] for e in all_errors if e))
            ),
            "model": model,
            "max_retry_count": max_retries,
        }
        _append_result(
            summary_results_file,
            summary_row,
            SUMMARY_HEADERS,
            write_header=(sum_row_idx == 0),
        )
        sum_row_idx += 1

    print(f"\n--- Benchmark Complete! ---")
    print(
        f"Overall Success Rate: {total_success}/{total_runs} ({total_success / total_runs * 100:.1f}%)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit test cases")
    parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
    parser.add_argument("--provider", type=str, default="Gemini API")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum number of coder retry attempts per benchmark case",
    )
    args = parser.parse_args()
    run_benchmark(
        limit=args.limit,
        model=args.model,
        provider=args.provider,
        max_retries=args.max_retries,
    )
