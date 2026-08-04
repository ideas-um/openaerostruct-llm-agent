from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import statistics
import time
from datetime import datetime

import benchmark as bench
from agent_logic import AgentResult, build_approved_relaxation_prompt, run_agent
from llm.config import LLMBackendTransientError
from llm.router import route_intent


CASE3_PROMPT = (
    "Minimize fuel burn over a 1000 km mission for a rectangular wing with a "
    "12 m span and 1.2 m chord using a tube spar. Fly at Mach 0.5 and 5000 m "
    "altitude with an aircraft empty mass of 4000 kg. Use aluminum with "
    "E = 70 GPa, yield stress = 500 MPa, and density = 2700 kg/m3. Limit alpha "
    "to 0 to 1 degrees and tube thickness to 5 to 15 mm. Require no structural "
    "failure and lift equal to weight. Plot the thickness distribution."
)

CASE3 = {
    "id": "case_study_3",
    "category": "framework_case3",
    "query": CASE3_PROMPT,
    "expected_blueprints": '["aerostruct_tube.py"]',
}

CASE3_RESULT_HEADERS = [
    "rep",
    "success",
    "converged",
    "auditor_loops",
    "audit_failures",
    "convergence_approvals",
    "coder_attempts",
    "final_fuel_burn_kg",
    "final_objective_name",
    "final_objective_value",
    "final_alpha_deg",
    "final_structural_mass_kg",
    "elapsed_s",
    "input_tokens",
    "output_tokens",
]


def _as_scalar(value):
    if isinstance(value, list):
        return _as_scalar(value[0]) if value else ""
    if isinstance(value, dict) and "final" in value:
        return _as_scalar(value["final"])
    if isinstance(value, (int, float)):
        return value
    return ""


def _case3_result_metrics(metrics: dict) -> dict:
    values = metrics.get("stdout", {}).get("values", {})
    objectives = metrics.get("db", {}).get("objectives", {})
    design_vars = metrics.get("db", {}).get("design_vars", {})

    objective_name = ""
    objective_value = ""
    if objectives:
        objective_name, objective_data = next(iter(objectives.items()))
        objective_value = _as_scalar(objective_data)

    fuel_burn = values.get("Final_fuel_burn", "")
    if fuel_burn == "" and len(objectives) == 1:
        fuel_burn = objective_value

    alpha = values.get("Final_alpha", "")
    if alpha == "":
        for name, data in design_vars.items():
            if name.endswith("alpha") or name == "alpha":
                alpha = _as_scalar(data)
                break

    return {
        "final_fuel_burn_kg": fuel_burn,
        "final_objective_name": objective_name,
        "final_objective_value": objective_value,
        "final_alpha_deg": alpha,
        "final_structural_mass_kg": values.get("Final_structural_mass", ""),
    }


def _run_single_rep_with_approval(
    q: dict,
    rep_dir: str,
    model: str,
    provider: str,
    max_retries: int,
    max_convergence_tries: int,
) -> dict:
    os.makedirs(rep_dir, exist_ok=True)

    log_file_path = os.path.join(rep_dir, "agent_backend.log")
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    backend_logger = logging.getLogger("LLM_Backend")
    backend_logger.addHandler(file_handler)

    selected = "ERROR"
    routing_correct = False
    blueprints = []
    routing_data = {}
    attempt_records: list[dict] = []
    total_attempts = 0
    total_input_tokens = 0
    total_output_tokens = 0
    all_errors: list[str] = []
    approvals: list[str] = []
    audit_count = 0
    audit_failures = 0

    def _rel(path: str) -> str:
        return os.path.relpath(path, os.path.dirname(rep_dir))

    try:
        routing_data = route_intent(q["query"], model_name=model, provider=provider)
        total_input_tokens += routing_data.get("input_tokens", 0)
        total_output_tokens += routing_data.get("output_tokens", 0)
        blueprints = routing_data.get("blueprints", [])
        selected = ", ".join(blueprints)

        expected = set(json.loads(q["expected_blueprints"]))
        routing_correct = expected == set(blueprints)
        if not blueprints:
            blueprints = [json.loads(q["expected_blueprints"])[0]]
        print(f"    Routing: {selected}  (correct={routing_correct})")

        prompt = q["query"]
        prior_errors: list[str] = []
        prior_code = ""
        final_result: AgentResult | None = None

        while True:
            approval_count = len(approvals)
            stage = "pre_approval" if approval_count == 0 else f"approval_{approval_count}"
            attempt_dirs: dict[int, str] = {}
            convergence_suggestion = ""

            def bench_callback(event: str, data: dict):
                nonlocal audit_count, audit_failures, convergence_suggestion

                local_attempt = data.get("attempt", 0)
                attempt = total_attempts + local_attempt
                if event == "attempt_start":
                    bench._cleanup_run_artifacts()
                    attempt_dir = os.path.join(rep_dir, f"{stage}_attempt_{local_attempt}")
                    os.makedirs(attempt_dir, exist_ok=True)
                    attempt_dirs[local_attempt] = attempt_dir
                    attempt_records.append(
                        {
                            "id": q["id"],
                            "category": q["category"],
                            "rep": os.path.basename(rep_dir).replace("rep_", ""),
                            "attempt": attempt,
                            "event": event,
                            "passed_audit": "",
                            "status": "started",
                            "summary": stage,
                            "details_path": _rel(attempt_dir),
                        }
                    )
                elif event == "code_ready":
                    attempt_dir = attempt_dirs.get(local_attempt, rep_dir)
                    code = data.get("code", "")
                    reasoning = data.get("reasoning") or ""
                    code_path = os.path.join(attempt_dir, "code.py")
                    with open(code_path, "w", encoding="utf-8") as fh:
                        fh.write(code)
                    with open(
                        os.path.join(attempt_dir, "coder_reasoning.md"),
                        "w",
                        encoding="utf-8",
                    ) as fh:
                        fh.write(reasoning)
                    attempt_records.append(
                        {
                            "id": q["id"],
                            "category": q["category"],
                            "rep": os.path.basename(rep_dir).replace("rep_", ""),
                            "attempt": attempt,
                            "event": event,
                            "passed_audit": "",
                            "status": "code_generated",
                            "summary": reasoning[:240],
                            "details_path": _rel(code_path),
                        }
                    )
                elif event == "blueprint_audit":
                    attempt_dir = attempt_dirs.get(local_attempt, rep_dir)
                    report = data.get("report", {})
                    diff_text = data.get("diff", "") or report.get("diff", "")
                    report = {k: v for k, v in report.items() if k != "diff"}
                    report_path = os.path.join(attempt_dir, "blueprint_audit.json")
                    diff_path = os.path.join(attempt_dir, "blueprint_audit.diff")
                    bench._write_json(report_path, report)
                    if diff_text:
                        with open(diff_path, "w", encoding="utf-8") as fh:
                            fh.write(diff_text)
                    audit_count += 1
                    if not report.get("passed", True):
                        audit_failures += 1
                    violations = report.get("violations") or []
                    summary = "passed"
                    if violations:
                        first = violations[0]
                        summary = (
                            f"{first.get('changed_item', 'violation')}: "
                            f"{first.get('reason', '')}"
                        )
                    elif report.get("warning"):
                        summary = report["warning"]
                    attempt_records.append(
                        {
                            "id": q["id"],
                            "category": q["category"],
                            "rep": os.path.basename(rep_dir).replace("rep_", ""),
                            "attempt": attempt,
                            "event": event,
                            "passed_audit": report.get("passed", True),
                            "status": "audit_passed"
                            if report.get("passed", True)
                            else "audit_failed",
                            "summary": summary[:240],
                            "details_path": _rel(report_path),
                        }
                    )
                elif event in ("exec_success", "exec_error", "no_converge"):
                    attempt_dir = attempt_dirs.get(local_attempt, rep_dir)
                    bench._copy_artifacts(attempt_dir)
                    if event == "exec_success":
                        summary = "execution completed"
                    elif event == "exec_error":
                        summary = bench._summarize_error(
                            "Python error: " + data.get("stderr_tail", "")
                        )
                    else:
                        summary = bench._summarize_error(
                            data.get("stdout_tail", "optimizer did not converge")
                        )
                    attempt_records.append(
                        {
                            "id": q["id"],
                            "category": q["category"],
                            "rep": os.path.basename(rep_dir).replace("rep_", ""),
                            "attempt": attempt,
                            "event": event,
                            "passed_audit": "",
                            "status": event,
                            "summary": summary[:240],
                            "details_path": _rel(attempt_dir),
                        }
                    )
                elif event == "no_converge_final":
                    convergence_suggestion = data.get("suggestion", "")
                    path = os.path.join(rep_dir, f"{stage}_convergence_agent.json")
                    bench._write_json(path, data)
                    with open(
                        os.path.join(rep_dir, f"{stage}_convergence_suggestion.md"),
                        "w",
                        encoding="utf-8",
                    ) as fh:
                        fh.write(convergence_suggestion)
                    attempt_records.append(
                        {
                            "id": q["id"],
                            "category": q["category"],
                            "rep": os.path.basename(rep_dir).replace("rep_", ""),
                            "attempt": total_attempts + local_attempt,
                            "event": event,
                            "passed_audit": "",
                            "status": "suggested_relaxation",
                            "summary": convergence_suggestion[:240],
                            "details_path": _rel(path),
                        }
                    )

            final_result = run_agent(
                user_prompt=prompt,
                blueprints=blueprints,
                model_name=model,
                provider=provider,
                max_retries=max_retries,
                stream=False,
                callback=bench_callback,
                gen_script_path=bench._BENCH_SCRIPT,
                retry_on_no_converge=False,
                prior_error_logs=prior_errors,
                prior_code=prior_code,
                routing_data=routing_data,
            )
            total_attempts += final_result.attempts
            total_input_tokens += final_result.input_tokens
            total_output_tokens += final_result.output_tokens
            all_errors.extend(final_result.error_logs)

            with open(
                os.path.join(rep_dir, f"{stage}_final_code.py"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write(final_result.final_code or "")
            bench._write_json(
                os.path.join(rep_dir, f"{stage}_result.json"),
                {
                    "success": final_result.success,
                    "converged": final_result.converged,
                    "attempts": final_result.attempts,
                    "error_logs": final_result.error_logs,
                    "input_tokens": final_result.input_tokens,
                    "output_tokens": final_result.output_tokens,
                    "result_metrics": final_result.result_metrics,
                },
            )

            if final_result.success or final_result.converged != "no":
                break
            if len(approvals) >= max_convergence_tries or not convergence_suggestion.strip():
                break

            approvals.append(
                "Approved convergence repair "
                f"{len(approvals) + 1}:\n{convergence_suggestion.strip()}"
            )
            prompt = build_approved_relaxation_prompt(q["query"], "\n\n".join(approvals))
            with open(
                os.path.join(rep_dir, f"approved_relaxation_prompt_{len(approvals)}.md"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write(prompt)
            prior_errors = final_result.error_logs
            prior_code = final_result.final_code or ""

    except LLMBackendTransientError:
        raise
    finally:
        backend_logger.removeHandler(file_handler)
        file_handler.close()

    assert final_result is not None

    if final_result.final_code:
        with open(os.path.join(rep_dir, "final_code.py"), "w", encoding="utf-8") as fh:
            fh.write(final_result.final_code)
        bench._write_code_diffs(final_result.final_code, blueprints, rep_dir, "final_vs_blueprint")

    if final_result.result_metrics:
        bench._write_json(
            os.path.join(rep_dir, "final_result_metrics.json"),
            final_result.result_metrics,
        )

    bench._write_json(
        os.path.join(rep_dir, "final_result.json"),
        {
            "selected_blueprints": blueprints,
            "routing_correct": routing_correct,
            "success": final_result.success,
            "converged": final_result.converged,
            "convergence_approvals": len(approvals),
            "auditor_loops": audit_count,
            "audit_failures": audit_failures,
            "attempts": total_attempts,
            "error_logs": all_errors,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "result_metrics": final_result.result_metrics,
        },
    )
    bench._validate_rep_archive(rep_dir, attempt_records)

    return {
        "selected_blueprints": selected,
        "routing_correct": routing_correct,
        "attempts": total_attempts,
        "exit_code": 0 if final_result.success else -1,
        "converged": final_result.converged,
        "success": final_result.success,
        "error_logs": all_errors,
        "result_metrics": final_result.result_metrics,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "attempt_records": attempt_records,
        "convergence_approvals": len(approvals),
        "auditor_loops": audit_count,
        "audit_failures": audit_failures,
    }


def run_case3_benchmark(
    model: str,
    provider: str,
    max_retries: int,
    num_reps: int,
    max_convergence_tries: int,
) -> str:
    os.makedirs(bench._OAS_OUT_DIR, exist_ok=True)
    os.makedirs(bench._BENCH_OUT_DIR, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        bench._BENCH_OUT_DIR, f"run_{run_ts}_framework_case3_{bench._safe_name(model)}"
    )
    os.makedirs(run_dir, exist_ok=True)
    bench._write_json(
        os.path.join(run_dir, "run_metadata.json"),
        {
            "model": model,
            "provider": provider,
            "max_retry_count": max_retries,
            "num_reps": num_reps,
            "case_ids": [CASE3["id"]],
            "timestamp": run_ts,
            "case_prompt": CASE3_PROMPT,
            "max_convergence_tries": max_convergence_tries,
            "policy": (
                "Approve each Convergence Agent recommendation verbatim, stop "
                "on success or after the configured maximum."
            ),
        },
    )

    rep_results_file = os.path.join(run_dir, "rep_results.csv")
    attempt_results_file = os.path.join(run_dir, "attempt_results.csv")
    summary_results_file = os.path.join(run_dir, "benchmark_results.csv")
    case3_results_file = os.path.join(run_dir, "case3_results.csv")
    case_dir = os.path.join(run_dir, f"case_{CASE3['id']}")
    os.makedirs(case_dir, exist_ok=True)

    rep_rows = []
    attempt_row_idx = 0
    print(f"--- Starting Framework Case 3 Benchmark (1 case x {num_reps} reps) ---")
    print(f"\n[Case {CASE3['id']}] {CASE3['category']}: {CASE3['query'][:80]}...")

    for rep in range(1, num_reps + 1):
        print(f"  [Rep {rep}/{num_reps}]", end=" ", flush=True)
        rep_dir = os.path.join(case_dir, f"rep_{rep}")
        start_time = time.time()
        res = _run_single_rep_with_approval(
            CASE3, rep_dir, model, provider, max_retries, max_convergence_tries
        )
        elapsed = round(time.time() - start_time, 2)
        hash_metrics_json = json.dumps(
            bench._metrics_for_hash(res["result_metrics"]),
            sort_keys=True,
            separators=(",", ":"),
        )
        csv_metrics_json = json.dumps(
            bench._compact_metrics_for_csv(res["result_metrics"]),
            sort_keys=True,
            separators=(",", ":"),
        )
        metrics_hash = (
            hashlib.sha1(hash_metrics_json.encode("utf-8")).hexdigest()[:12]
            if hash_metrics_json != "{}"
            else ""
        )
        rep_row = {
            "id": CASE3["id"],
            "category": CASE3["category"],
            "query": CASE3["query"],
            "rep": rep,
            "expected_blueprints": CASE3["expected_blueprints"],
            "selected_blueprints": res["selected_blueprints"],
            "routing_correct": res["routing_correct"],
            "attempts": res["attempts"],
            "exit_code": res["exit_code"],
            "converged": res["converged"],
            "elapsed_s": elapsed,
            "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "success": res["success"],
            "result_metrics": csv_metrics_json,
            "result_metrics_hash": metrics_hash,
            "error_log": " ||| ".join(res["error_logs"]).replace("\n", " "),
        }
        case3_metric_row = {
            "rep": rep,
            "success": res["success"],
            "converged": res["converged"],
            "auditor_loops": res["auditor_loops"],
            "audit_failures": res["audit_failures"],
            "convergence_approvals": res["convergence_approvals"],
            "coder_attempts": res["attempts"],
            **_case3_result_metrics(res["result_metrics"]),
            "elapsed_s": elapsed,
            "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
        }
        bench._append_result(
            rep_results_file,
            rep_row,
            bench.REP_HEADERS,
            write_header=(rep == 1),
        )
        for attempt_record in res.get("attempt_records", []):
            bench._append_result(
                attempt_results_file,
                attempt_record,
                bench.ATTEMPT_HEADERS,
                write_header=(attempt_row_idx == 0),
            )
            attempt_row_idx += 1
        bench._append_result(
            case3_results_file,
            case3_metric_row,
            CASE3_RESULT_HEADERS,
            write_header=(rep == 1),
        )
        rep_rows.append(
            {
                **rep_row,
                **case3_metric_row,
            }
        )
        print(
            f"Done (success={res['success']}, audits={res['auditor_loops']}, "
            f"fuelburn={case3_metric_row['final_fuel_burn_kg']}, "
            f"approvals={res['convergence_approvals']}, {elapsed}s)"
        )

    n_reps = len(rep_rows)
    opt_reps = sum(row["converged"] in ("yes", "no") for row in rep_rows)
    metric_hashes = [row["result_metrics_hash"] for row in rep_rows]
    errors = [
        error
        for row in rep_rows
        for error in row["error_log"].split(" ||| ")
        if error
    ]
    fuelburn_values = [
        float(row["final_fuel_burn_kg"])
        for row in rep_rows
        if row.get("final_fuel_burn_kg") not in ("", None)
    ]
    summary_row = {
        "id": CASE3["id"],
        "category": CASE3["category"],
        "query": CASE3["query"],
        "expected_blueprints": CASE3["expected_blueprints"],
        "selected_blueprints": rep_rows[-1]["selected_blueprints"],
        "num_runs": n_reps,
        "routing_accuracy": round(
            sum(bench._csv_bool(row["routing_correct"]) for row in rep_rows) / n_reps, 3
        ),
        "execution_success_rate": round(
            sum(bench._csv_bool(row["success"]) for row in rep_rows) / n_reps, 3
        ),
        "convergence_rate": round(
            sum(row["converged"] == "yes" for row in rep_rows) / opt_reps, 3
        )
        if opt_reps > 0
        else "n/a",
        "attempts_mean": round(statistics.mean(int(row["attempts"]) for row in rep_rows), 3),
        "attempts_median": statistics.median(int(row["attempts"]) for row in rep_rows),
        "attempts_std": round(
            statistics.stdev(int(row["attempts"]) for row in rep_rows), 3
        )
        if n_reps > 1
        else 0,
        "attempts_min": min(int(row["attempts"]) for row in rep_rows),
        "attempts_max": max(int(row["attempts"]) for row in rep_rows),
        "elapsed_s_mean": round(statistics.mean(float(row["elapsed_s"]) for row in rep_rows), 3),
        "elapsed_s_median": statistics.median(float(row["elapsed_s"]) for row in rep_rows),
        "elapsed_s_std": round(
            statistics.stdev(float(row["elapsed_s"]) for row in rep_rows), 3
        )
        if n_reps > 1
        else 0,
        "elapsed_s_min": min(float(row["elapsed_s"]) for row in rep_rows),
        "elapsed_s_max": max(float(row["elapsed_s"]) for row in rep_rows),
        "input_tokens_mean": int(statistics.mean(int(row["input_tokens"]) for row in rep_rows)),
        "output_tokens_mean": int(statistics.mean(int(row["output_tokens"]) for row in rep_rows)),
        "unique_result_metrics": len(set(h for h in metric_hashes if h)),
        "error_categories": " ||| ".join(
            sorted(set(bench._summarize_error(e) for e in errors if e))
        ),
        "model": model,
        "max_retry_count": max_retries,
    }
    bench._append_result(
        summary_results_file,
        summary_row,
        bench.SUMMARY_HEADERS,
        write_header=True,
    )
    bench._write_json(
        os.path.join(run_dir, "summary.json"),
        {
            "num_runs": n_reps,
            "success_count": sum(bench._csv_bool(row["success"]) for row in rep_rows),
            "converged_count": sum(row["converged"] == "yes" for row in rep_rows),
            "runs_using_convergence_agent": sum(
                int(row["convergence_approvals"]) > 0 for row in rep_rows
            ),
            "total_convergence_approvals": sum(
                int(row["convergence_approvals"]) for row in rep_rows
            ),
            "auditor_loops_total": sum(int(row["auditor_loops"]) for row in rep_rows),
            "audit_failures_total": sum(int(row["audit_failures"]) for row in rep_rows),
            "final_fuel_burn_kg_mean": round(statistics.mean(fuelburn_values), 6)
            if fuelburn_values
            else "",
            "final_fuel_burn_kg_min": min(fuelburn_values) if fuelburn_values else "",
            "final_fuel_burn_kg_max": max(fuelburn_values) if fuelburn_values else "",
        },
    )
    print("\n--- Benchmark Complete! ---")
    print(f"Saved to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-reps", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=bench.DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-convergence-tries", type=int, default=3)
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--provider", type=str, default="Gemini API")
    args = parser.parse_args()
    run_case3_benchmark(
        model=args.model,
        provider=args.provider,
        max_retries=args.max_retries,
        num_reps=args.num_reps,
        max_convergence_tries=args.max_convergence_tries,
    )
