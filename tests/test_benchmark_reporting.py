import csv
import json

import benchmark


def test_rep_archive_requires_backend_code_and_audit_records(tmp_path):
    case_dir = tmp_path / "case_1"
    rep_dir = case_dir / "rep_1"
    attempt_dir = rep_dir / "attempt_1"
    attempt_dir.mkdir(parents=True)
    (rep_dir / "agent_backend.log").write_text("request and response")
    (attempt_dir / "code.py").write_text("print('ok')")
    (attempt_dir / "blueprint_audit.json").write_text('{"passed": true}')

    records = [
        {"event": "code_ready", "details_path": "rep_1/attempt_1/code.py"},
        {
            "event": "blueprint_audit",
            "details_path": "rep_1/attempt_1/blueprint_audit.json",
        },
    ]

    benchmark._validate_rep_archive(str(rep_dir), records)


def test_rep_archive_rejects_missing_backend_log(tmp_path):
    rep_dir = tmp_path / "case_1" / "rep_1"
    rep_dir.mkdir(parents=True)

    try:
        benchmark._validate_rep_archive(str(rep_dir), [])
    except RuntimeError as exc:
        assert "agent_backend.log" in str(exc)
    else:
        raise AssertionError("missing backend log should fail archive validation")


def test_default_benchmark_reps_is_full_run_count():
    assert benchmark.NUM_REPS == 10


def test_blueprint_error_summary_keeps_actionable_item_and_reason():
    error = """
    [attempt 1] Blueprint consistency error:
    Blueprint consistency error:
    Make only these auditor-requested repairs.
    1. dict:mesh_dict.num_y
    Reason: The user did not request changing mesh resolution key num_y.
    Replace generated snippet:
    mesh_dict['num_y'] = 11
    """

    assert benchmark._summarize_error(error) == (
        "Blueprint consistency error: dict:mesh_dict.num_y - "
        "The user did not request changing mesh resolution key num_y."
    )


def test_docker_missing_script_error_is_collapsed_to_clear_category():
    error = (
        "[attempt 2] Python error: Using Docker sandbox image "
        "'openaerostruct-sandbox:latest' [docker sandbox] python: can't open file "
        "'/workspace/src/benchmark_run.py': [Errno 2] No such file or directory"
    )

    assert benchmark._summarize_error(error) == (
        "Python error: generated script path missing in execution sandbox"
    )


def test_attempt_results_headers_cover_per_iteration_logging():
    assert benchmark.ATTEMPT_HEADERS == [
        "id",
        "category",
        "rep",
        "attempt",
        "event",
        "passed_audit",
        "status",
        "summary",
        "details_path",
    ]


def test_resume_restarts_only_incomplete_repetition(tmp_path, monkeypatch):
    bench_root = tmp_path / "benchmark_run_out"
    run_dir = bench_root / "run_test_model"
    partial_rep = run_dir / "case_1" / "rep_3"
    partial_rep.mkdir(parents=True)
    (partial_rep / "partial.log").write_text("interrupted")

    metadata = {
        "model": "stored-model",
        "provider": "stored-provider",
        "max_retry_count": 5,
        "num_reps": 3,
        "case_ids": None,
        "limit": None,
        "timestamp": "test",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata))

    queries_file = tmp_path / "queries.csv"
    queries_file.write_text(
        "id,category,query,expected_blueprints\n"
        '1,test_case,"Run the test.","[""test.py""]"\n'
    )

    rep_results = run_dir / "rep_results.csv"
    for rep in (1, 2):
        row = {header: "" for header in benchmark.REP_HEADERS}
        row.update(
            {
                "id": "1",
                "category": "test_case",
                "query": "Run the test.",
                "rep": rep,
                "selected_blueprints": "test.py",
                "routing_correct": True,
                "attempts": 1,
                "exit_code": 0,
                "converged": "yes",
                "elapsed_s": 1.0,
                "input_tokens": 10,
                "output_tokens": 5,
                "success": True,
                "result_metrics": "{}",
                "result_metrics_hash": "",
                "error_log": "",
            }
        )
        benchmark._append_result(
            str(rep_results),
            row,
            benchmark.REP_HEADERS,
            write_header=(rep == 1),
        )

    calls = []

    def fake_run(q, rep_dir, model, provider, max_retries):
        calls.append((rep_dir, model, provider, max_retries))
        assert not (partial_rep / "partial.log").exists()
        return {
            "selected_blueprints": "test.py",
            "routing_correct": True,
            "attempts": 1,
            "exit_code": 0,
            "converged": "yes",
            "success": True,
            "error_logs": [],
            "result_metrics": {},
            "input_tokens": 12,
            "output_tokens": 6,
            "attempt_records": [],
        }

    monkeypatch.setattr(benchmark, "_BENCH_OUT_DIR", str(bench_root))
    monkeypatch.setattr(benchmark, "_OAS_OUT_DIR", str(tmp_path / "oas_out"))
    monkeypatch.setattr(benchmark, "_INPUT_FILE", str(queries_file))
    monkeypatch.setattr(benchmark, "_run_single_rep", fake_run)

    benchmark.run_benchmark(resume_run=run_dir.name)

    assert calls == [
        (
            str(partial_rep),
            "stored-model",
            "stored-provider",
            5,
        )
    ]
    with rep_results.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["rep"] for row in rows] == ["1", "2", "3"]

    with (run_dir / "benchmark_results.csv").open(newline="") as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == 1
    assert summary_rows[0]["num_runs"] == "3"
