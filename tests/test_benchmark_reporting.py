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
