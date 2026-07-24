import pytest

from agent_logic import APPROVED_RELAXATION_HEADER, build_approved_relaxation_prompt
from llm.auditor import (
    _make_diff,
    _ordered_semantic_changes,
    _parse_audit_response,
    _semantic_changes,
    audit_blueprint_consistency,
)
from llm.coder import _build_prompt, _parse_response
from llm.config import is_gemini_transient_error
from llm.relaxer import suggest_relaxation


def test_generated_code_parser_strips_full_line_comments_only():
    response = """
    <reasoning>
    - Blueprint: aero_analysis.py
    - Retry fix: none.
    </reasoning>
    <code>
    import os
    # handbook comment
    x = 1  # useful inline comment

    # === AGENT EDITABLE SECTION START ===
    y = 2
    </code>
    """

    reasoning, code = _parse_response(response)

    assert "Blueprint: aero_analysis.py" in reasoning
    assert "# handbook comment" not in code
    assert "# === AGENT EDITABLE SECTION START ===" not in code
    assert "x = 1  # useful inline comment" in code
    assert "y = 2" in code


def test_coder_prompt_receives_filtered_router_context():
    routing_context = {
        "blueprints": ["aero_opt.py"],
        "is_vague": False,
        "reason": "Single-point optimization.",
        "parameters": {
            "design_variables": [{"name": "twist_cp"}],
            "objective": "minimize drag",
        },
        "input_tokens": 100,
    }

    system_prompt, prompt = _build_prompt(
        "Minimize drag using twist.",
        ["aero_opt.py"],
        "Initial generation",
        routing_context=routing_context,
    )

    assert "Omission generally means preserve" in system_prompt
    assert "lower <= initial <= upper" in system_prompt
    assert "`taper`, `sweep`, and `dihedral` are Surface Dict" in system_prompt
    assert "Optimize only the DVs the user" in system_prompt
    assert "initializer is an optimizer starting point" in system_prompt
    assert "### ROUTER CONTEXT ###" in prompt
    assert '"name": "twist_cp"' in prompt
    assert "original user request remains authoritative" in prompt
    assert "Router omission does not erase an explicit user instruction" in prompt
    assert "design variables as the active optimization freedoms" in prompt
    assert "geometry, loads, settings, units" in prompt
    assert "AGENT EDITABLE SECTION" not in prompt
    assert "input_tokens" not in prompt


def test_relaxer_receives_structured_optimization_evidence(monkeypatch):
    captured = {}

    def fake_response(prompt, model_name, system_prompt, provider):
        captured["prompt"] = prompt
        return '<relaxation>{"suggestion":"Increase alpha upper bound."}</relaxation>'

    monkeypatch.setattr("llm.relaxer.get_llm_response", fake_response)

    suggestion, _, _ = suggest_relaxation(
        "Minimize fuel burn.",
        ["Optimizer failed to converge."],
        "test-model",
        "test-provider",
        blueprints=["aerostruct_tube.py"],
        optimizer_status="Exit mode 8",
        db_summary="alpha final: 1.0",
        result_metrics={"db": {"constraints": {"L_equals_W": {"final": 0.2}}}},
        generated_code=(
            "prob.model.add_design_var('alpha', lower=0.0, upper=1.0)\n"
            "prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.0)\n"
            "prob.model.add_objective('AS_point_0.fuelburn')\n"
        ),
    )

    assert suggestion == "Increase alpha upper bound."
    assert "### ACTIVE OPTIMIZATION FORMULATION ###" in captured["prompt"]
    assert "add_design_var('alpha', lower=0.0, upper=1.0)" in captured["prompt"]
    assert "### STRUCTURED RESULT METRICS ###" in captured["prompt"]
    assert "L_equals_W" in captured["prompt"]
    assert "Exit mode 8" in captured["prompt"]


def test_auditor_diff_ignores_comment_only_lines_but_keeps_executable_changes():
    blueprint = "x = 1\n# long guidance\ny = 2\n"
    generated = "x = 1\ny = 3\n"

    diff = _make_diff("demo.py", blueprint, generated)

    assert "long guidance" not in diff
    assert "-y = 2" in diff
    assert "+y = 3" in diff


def test_approved_relaxation_prompt_marks_retry_scope():
    prompt = build_approved_relaxation_prompt(
        "Minimize drag with CL = 0.5.",
        "Apply these relaxations and retry:\nIncrease alpha upper bound to 10 deg.",
    )

    assert APPROVED_RELAXATION_HEADER in prompt
    assert "explicitly approved by the user" in prompt
    assert "Preserve all unrelated blueprint assumptions." in prompt
    assert "Increase alpha upper bound to 10 deg." in prompt


def test_semantic_changes_watch_surface_update_solver_keys():
    blueprint = "surf_dict = {}\nsurf_dict.update({'with_wave': False})\n"
    generated = "surf_dict = {}\nsurf_dict.update({'with_wave': True})\n"

    changes = _semantic_changes(blueprint, generated)

    assert any(change["item"] == "dict:surf_dict.with_wave" for change in changes)


def test_semantic_changes_include_vertical_load_assignment():
    blueprint = """
loads = np.zeros((ny, 6))
loads[:, 2] = 2e5 / ny
"""
    generated = """
loads = np.zeros((ny, 6))
loads[:, 2] = (4e4 / 2.0) / ny
"""

    changes = _semantic_changes(blueprint, generated)

    assert any(change["item"] == "assign:loads[:, 2]" for change in changes)


def test_semantic_changes_include_individual_mesh_resolution_keys():
    blueprint = """
mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "CRM"}
"""
    generated = """
mesh_dict = {"num_y": 15, "num_x": 3, "wing_type": "rect"}
"""

    changes = _semantic_changes(blueprint, generated)
    items = {change["item"] for change in changes}

    assert "dict:mesh_dict.num_y" in items
    assert "dict:mesh_dict.num_x" in items
    assert "assign:mesh_dict" not in items
    mesh_changes = [
        change for change in changes if change["item"].startswith("dict:mesh_dict.num_")
    ]
    assert all("Protected mesh discretization" in change["audit_rule"] for change in mesh_changes)


def test_semantic_changes_split_bounds_from_existing_scaling():
    blueprint = """
prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, ref=0.1)
prob.model.add_objective("wing.structural_mass", scaler=1e-5)
"""
    generated = """
prob.model.add_design_var("wing.thickness_cp", lower=0.005, upper=0.1, ref=0.01)
prob.model.add_objective("wing.structural_mass", scaler=1e-2)
"""

    changes = _semantic_changes(blueprint, generated)
    by_item = {change["item"]: change for change in changes}

    assert "call:add_design_var:wing.thickness_cp.arg:lower" in by_item
    assert "call:add_design_var:wing.thickness_cp.arg:upper" in by_item
    assert "call:add_design_var:wing.thickness_cp.arg:ref" in by_item
    assert "call:add_objective:wing.structural_mass.arg:scaler" in by_item
    assert (
        "Protected existing numerical scaling"
        in by_item["call:add_design_var:wing.thickness_cp.arg:ref"]["audit_rule"]
    )
    assert (
        "Protected existing numerical scaling"
        in by_item["call:add_objective:wing.structural_mass.arg:scaler"]["audit_rule"]
    )


def test_semantic_changes_keep_array_elements_when_call_arguments_are_split():
    blueprint = """
indep_var_comp.add_output("load_factor", val=np.array([1.0, 2.5]))
"""
    generated = """
indep_var_comp.add_output("load_factor", val=np.array([1.0, 2.0]))
"""

    changes = _semantic_changes(blueprint, generated)

    assert changes == [
        {
            "item": "call:add_output:load_factor.arg:val",
            "status": "changed",
            "blueprint": "np.array([1.0, 2.5])",
            "generated": "np.array([1.0, 2.0])",
            "element_changes": [
                {
                    "index": 1,
                    "status": "changed",
                    "blueprint": "2.5",
                    "generated": "2.0",
                }
            ],
        }
    ]


def test_audit_orders_removed_constraint_before_other_changes():
    changes = [
        {
            "item": "call:add_output:rho",
            "status": "changed",
            "blueprint": "rho=1.0",
            "generated": "rho=0.8",
        },
        {
            "item": "dict:mesh_dict.num_y",
            "status": "changed",
            "blueprint": "num_y=7",
            "generated": "num_y=15",
        },
        {
            "item": "call:add_constraint:AS_point_1.L_equals_W",
            "status": "removed",
            "blueprint": "add_constraint('AS_point_1.L_equals_W')",
            "generated": "",
        },
    ]

    ordered = _ordered_semantic_changes(changes)

    assert ordered[0]["item"] == "call:add_constraint:AS_point_1.L_equals_W"
    assert ordered[1]["item"] == "dict:mesh_dict.num_y"


@pytest.mark.parametrize("passed", [True, False])
def test_audit_parser_preserves_llm_top_level_decision(passed):
    report = _parse_audit_response(
        f'<audit>{{"passed": {str(passed).lower()}, "violations": []}}</audit>'
    )

    assert report["passed"] is passed


def test_auditor_does_not_override_llm_pass_decision(monkeypatch):
    calls = []
    system_prompts = []

    def fake_response(prompt, model_name, system_prompt, provider):
        calls.append(prompt)
        system_prompts.append(system_prompt)
        return (
            '<audit>{"passed":true,"reviewed_changes":[],'
            '"violations":[],"warnings":[]}</audit>'
        )

    monkeypatch.setattr("llm.auditor.get_llm_response", fake_response)

    report, _, _ = audit_blueprint_consistency(
        user_prompt="Apply a total upward load of 4e4 N.",
        blueprints=["struct_optimization.py"],
        generated_code="loads = np.ones((ny, 6)) * (4e4 / ny)\n",
        model_name="test-model",
        provider="test-provider",
        routing_context={
            "blueprints": ["struct_optimization.py"],
            "parameters": {"geometry": {"total_load": 4e4}},
        },
    )

    assert report["passed"] is True
    assert len(calls) == 1
    assert "Omission generally means preserve" in system_prompts[0]
    assert '"authorization"' in system_prompts[0]
    assert "Never classify" in system_prompts[0]
    assert "including equality at either" in system_prompts[0]
    assert "### ROUTER CONTEXT ###" in calls[0]
    assert '"total_load": 40000.0' in calls[0]
    assert "### REQUIRED CHANGE-BY-CHANGE REVIEW" in calls[0]
    assert "### SUPPORTING EXECUTABLE DIFF ###" in calls[0]
    assert "loads = np.ones" in calls[0]


def test_server_disconnect_is_transient_backend_error():
    assert is_gemini_transient_error(
        RuntimeError("Server disconnected without sending a response.")
    )
