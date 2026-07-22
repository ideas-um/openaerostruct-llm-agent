import benchmark


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
