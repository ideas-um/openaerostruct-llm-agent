import os
import csv
import sys
import time
import json
import shutil
import statistics
from datetime import datetime

from llm.router import route_intent_stream
from agent_logic import run_agent, AgentResult

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

_INPUT_FILE = os.path.join(_SRC_DIR, "tools", "test_queries.csv")
_OAS_OUT_DIR = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR = os.path.join(_OAS_OUT_DIR, "agent_plots")
_GEN_RUN_DIR = os.path.join(_OAS_OUT_DIR, "generated_run_out")
_BENCH_OUT_DIR = os.path.join(_PROJECT_DIR, "benchmark_run_out")
_BENCH_SCRIPT = os.path.join(_SRC_DIR, "benchmark_run.py")

MAX_RETRIES = 5
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
            except Exception as e:
                print(f"    Warning: could not copy {src}: {e}")


# ---------------------------------------------------------------------------
# Single Repetition Runner
# ---------------------------------------------------------------------------
def _run_single_rep(q: dict, rep_dir: str, model: str, provider: str) -> dict:
    selected = "ERROR"
    routing_correct = False
    blueprints = []

    # ------------------------------------------------------------------
    # 1. Intent routing (using STREAM to extract accurate token usage)
    # ------------------------------------------------------------------
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
                "    WARNING: router marked query as vague — counted as routing failure"
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
        print(f"    Routing Error: {e}  — falling back to expected blueprint")

    os.makedirs(rep_dir, exist_ok=True)
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
            _copy_artifacts(attempt_dir)
            log_path = os.path.join(attempt_dir, "execution.log")
            if event == "exec_error":
                with open(log_path, "w") as fh:
                    fh.write(f"--- STDERR TAIL ---\n{data.get('stderr_tail', '')}")
            if event == "exec_success":
                with open(log_path, "w") as fh:
                    fh.write(f"--- DB SUMMARY ---\n{data.get('db_summary', '')}")
        elif event == "done":
            status = (
                "SUCCESS"
                if data["success"]
                else f"FAILED after {data['attempts']} attempts"
            )
            print(f"      {status}")

    # ------------------------------------------------------------------
    # 2. Execution (using stream=True to extract accurate token usage)
    # ------------------------------------------------------------------
    result: AgentResult = run_agent(
        user_prompt=q["query"],
        blueprints=blueprints,
        model_name=model,
        provider=provider,
        max_retries=MAX_RETRIES,
        stream=True,  # Critical to set True so it captures `usage_metadata` from the chunks
        callback=bench_callback,
        gen_script_path=_BENCH_SCRIPT,
        retry_on_no_converge=True,
    )

    if result.final_code:
        with open(os.path.join(rep_dir, "final_code.py"), "w") as fh:
            fh.write(result.final_code)

    exit_code = 0 if result.success else -1

    updated_error_logs = []
    for i, err in enumerate(result.error_logs):
        attempt_num = i + 1
        code_path = os.path.join(rep_dir, f"attempt_{attempt_num}", "code.py")
        if os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8") as f:
                errored_code = f.read()
            updated_error_logs.append(f"{err}\n\n--- Errored Code ---\n{errored_code}")
        else:
            updated_error_logs.append(err)

    return {
        "selected_blueprints": selected,
        "routing_correct": routing_correct,
        "attempts": result.attempts,
        "exit_code": exit_code,
        "converged": result.converged,
        "success": result.success,
        "error_logs": updated_error_logs,
        "input_tokens": routing_data.get("input_tokens", 0) + result.input_tokens,
        "output_tokens": routing_data.get("output_tokens", 0) + result.output_tokens,
    }


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------
def run_benchmark(limit=None, model="gemini-flash-lite-latest", provider="Gemini API"):
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
                "max_retry_count": MAX_RETRIES,
                "num_reps": NUM_REPS,
                "timestamp": run_ts,
                "hardware_note": "API-based inference",
                "temperature": "default",
                "sampling": "default",
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

    print(
        f"--- Starting Benchmark Test Suite ({len(queries)} cases × {NUM_REPS} reps) ---"
    )

    total_success, total_runs, rep_row_idx, sum_row_idx = 0, 0, 0, 0

    for idx, q in enumerate(queries):
        case_id = q["id"]
        case_dir = os.path.join(run_dir, f"case_{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        print(
            f"\n[Case {case_id}] Category: {q['category']}\n  Query: {q['query'][:120]}..."
        )

        rep_successes, rep_converged_count, optimization_reps = 0, 0, 0
        rep_attempts_list, rep_elapsed_list = [], []
        rep_in_tokens_list, rep_out_tokens_list = [], []
        rep_routing_correct, all_errors = [], []
        selected_blueprints_last = "ERROR"

        for rep in range(1, NUM_REPS + 1):
            print(f"\n  [Rep {rep}/{NUM_REPS}]")
            rep_dir = os.path.join(case_dir, f"rep_{rep}")
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
            rep_in_tokens_list.append(rep_result["input_tokens"])
            rep_out_tokens_list.append(rep_result["output_tokens"])
            rep_routing_correct.append(rep_result["routing_correct"])

            selected_blueprints_last = rep_result["selected_blueprints"]
            total_runs += 1
            all_errors.extend(rep_result["error_logs"])

            rep_row = {
                "id": case_id,
                "category": q["category"],
                "query": q["query"],
                "rep": rep,
                "expected_blueprints": q["expected_blueprints"],
                "selected_blueprints": rep_result["selected_blueprints"],
                "routing_correct": rep_result["routing_correct"],
                "attempts": rep_result["attempts"],
                "exit_code": rep_result["exit_code"],
                "converged": rep_result["converged"],
                "elapsed_s": elapsed,
                "input_tokens": rep_result["input_tokens"],
                "output_tokens": rep_result["output_tokens"],
                "success": rep_result["success"],
                "error_log": " ||| ".join(rep_result["error_logs"]).replace("\n", " "),
            }
            _append_result(
                rep_results_file, rep_row, REP_HEADERS, write_header=(rep_row_idx == 0)
            )
            rep_row_idx += 1

            print(
                f"  Rep {rep} done — success={rep_result['success']}  elapsed={elapsed}s  tokens=({rep_result['input_tokens']} in / {rep_result['output_tokens']} out)"
            )

        n_reps = len(rep_attempts_list)
        error_category_set = set(
            err.strip().splitlines()[0][:120] for err in all_errors if err.strip()
        )

        summary_row = {
            "id": case_id,
            "category": q["category"],
            "query": q["query"],
            "expected_blueprints": q["expected_blueprints"],
            "selected_blueprints": selected_blueprints_last,
            "num_runs": n_reps,
            "routing_accuracy": round(sum(rep_routing_correct) / n_reps, 3),
            "execution_success_rate": round(rep_successes / n_reps, 3),
            "convergence_rate": round(rep_converged_count / optimization_reps, 3)
            if optimization_reps > 0
            else "n/a",
            "attempts_mean": round(sum(rep_attempts_list) / n_reps, 3),
            "attempts_median": round(statistics.median(rep_attempts_list), 3),
            "attempts_std": round(statistics.stdev(rep_attempts_list), 3)
            if n_reps > 1
            else 0.0,
            "attempts_min": min(rep_attempts_list),
            "attempts_max": max(rep_attempts_list),
            "elapsed_s_mean": round(sum(rep_elapsed_list) / n_reps, 3),
            "elapsed_s_median": round(statistics.median(rep_elapsed_list), 3),
            "elapsed_s_std": round(statistics.stdev(rep_elapsed_list), 3)
            if n_reps > 1
            else 0.0,
            "elapsed_s_min": round(min(rep_elapsed_list), 3),
            "elapsed_s_max": round(max(rep_elapsed_list), 3),
            "input_tokens_mean": int(sum(rep_in_tokens_list) / n_reps),
            "output_tokens_mean": int(sum(rep_out_tokens_list) / n_reps),
            "error_categories": " ||| ".join(sorted(error_category_set)),
            "model": model,
            "max_retry_count": MAX_RETRIES,
        }
        _append_result(
            summary_results_file,
            summary_row,
            SUMMARY_HEADERS,
            write_header=(sum_row_idx == 0),
        )

        with open(os.path.join(case_dir, "all_errors.json"), "w") as fh:
            json.dump(all_errors, fh, indent=2)
        sum_row_idx += 1

    print(f"\n--- Benchmark Complete! ---")
    print(
        f"Overall Success Rate: {total_success}/{total_runs} reps ({total_success / total_runs * 100:.1f}%)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
    parser.add_argument("--provider", type=str, default="Gemini API")
    args = parser.parse_args()
    run_benchmark(limit=args.limit, model=args.model, provider=args.provider)
