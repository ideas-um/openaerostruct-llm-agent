import pytest

from agent_logic import APPROVED_RELAXATION_HEADER, build_approved_relaxation_prompt
from llm.auditor import (
    _contract_violations,
    _make_diff,
    _semantic_changes,
    _var_requested,
)
from llm.coder import _build_prompt, _parse_response
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


@pytest.mark.parametrize(
    ("natural_name", "canonical_name"),
    [
        ("twist control points", "twist_cp"),
        ("chord distribution", "chord_cp"),
        ("x-shear control points", "xshear_cp"),
        ("z-shear control points", "zshear_cp"),
        ("tube wall thickness", "thickness_cp"),
        ("tube outer radius", "radius_cp"),
        ("spar wall thickness", "spar_thickness_cp"),
        ("skin panel thickness", "skin_thickness_cp"),
        ("thickness-to-chord control points", "t_over_c_cp"),
        ("t_over_c_cp", "t_over_c_cp"),
    ],
)
def test_auditor_maps_physical_control_point_aliases(
    natural_name, canonical_name
):
    assert _var_requested(f"Vary the {natural_name}.", canonical_name)


def test_contract_allows_initialization_moved_inside_requested_tube_bounds():
    prompt = "Limit tube thickness to 0.005 to 0.015 m."
    changes = [
        {
            "item": "dict:surface.thickness_cp",
            "status": "changed",
            "blueprint": "surface['thickness_cp'] = np.array([0.1, 0.2, 0.3])",
            "generated": "surface['thickness_cp'] = np.array([0.01, 0.01, 0.01])",
        },
        {
            "item": "call:add_design_var:wing.thickness_cp",
            "status": "changed",
            "blueprint": (
                "prob.model.add_design_var('wing.thickness_cp', lower=0.01, "
                "upper=0.5, scaler=100.0)"
            ),
            "generated": (
                "prob.model.add_design_var('wing.thickness_cp', lower=0.005, "
                "upper=0.015, scaler=100.0)"
            ),
        },
    ]

    assert _contract_violations(prompt, changes) == []


def test_contract_blocks_initialization_on_requested_bound():
    prompt = "Limit tube thickness to 0.005 to 0.015 m."
    changes = [
        {
            "item": "dict:surface.thickness_cp",
            "status": "changed",
            "blueprint": "surface['thickness_cp'] = np.array([0.1, 0.2, 0.3])",
            "generated": "surface['thickness_cp'] = np.array([0.005, 0.01, 0.015])",
        },
        {
            "item": "call:add_design_var:wing.thickness_cp",
            "status": "changed",
            "blueprint": (
                "prob.model.add_design_var('wing.thickness_cp', lower=0.01, "
                "upper=0.5, scaler=100.0)"
            ),
            "generated": (
                "prob.model.add_design_var('wing.thickness_cp', lower=0.005, "
                "upper=0.015, scaler=100.0)"
            ),
        },
    ]

    violations = _contract_violations(prompt, changes)

    assert len(violations) == 1
    assert violations[0]["changed_item"] == "dict:surface.thickness_cp"


def test_contract_allows_explicit_natural_language_initial_value():
    changes = [
        {
            "item": "dict:surface.thickness_cp",
            "status": "changed",
            "blueprint": "surface['thickness_cp'] = np.array([0.1, 0.2, 0.3])",
            "generated": "surface['thickness_cp'] = np.array([0.01, 0.01, 0.01])",
        }
    ]

    assert (
        _contract_violations(
            "Initialize the tube thickness to 0.01 m.", changes
        )
        == []
    )


def test_contract_blocks_unneeded_initial_change_when_blueprint_is_feasible():
    prompt = "Limit skin thickness to 0.003 to 0.02 m."
    changes = [
        {
            "item": "dict:surface.skin_thickness_cp",
            "status": "changed",
            "blueprint": (
                "surface['skin_thickness_cp'] = "
                "np.array([0.003, 0.006, 0.01, 0.012])"
            ),
            "generated": (
                "surface['skin_thickness_cp'] = "
                "np.array([0.005, 0.005, 0.005, 0.005])"
            ),
        },
        {
            "item": "call:add_design_var:wing.skin_thickness_cp",
            "status": "changed",
            "blueprint": (
                "prob.model.add_design_var('wing.skin_thickness_cp', lower=0.003, "
                "upper=0.1, scaler=100.0)"
            ),
            "generated": (
                "prob.model.add_design_var('wing.skin_thickness_cp', lower=0.003, "
                "upper=0.02, scaler=100.0)"
            ),
        },
    ]

    violations = _contract_violations(prompt, changes)

    assert len(violations) == 1
    assert violations[0]["changed_item"] == "dict:surface.skin_thickness_cp"


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

    _, prompt = _build_prompt(
        "Minimize drag using twist.",
        ["aero_opt.py"],
        "Initial generation",
        routing_context=routing_context,
    )

    assert "### ROUTER CONTEXT ###" in prompt
    assert '"name": "twist_cp"' in prompt
    assert "original user request remains authoritative" in prompt
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


def test_contract_allows_requested_cl_and_twist_cp_replacements():
    prompt = (
        "Minimize drag. DVs: alpha, twist_cp (3 CPs, -6 to 6 deg), "
        "chord_cp (3 CPs). Constraint: CL = 0.45."
    )
    changes = [
        {
            "item": "call:add_constraint:prob.model.add_constraint(f'{point_name}.wing_perf.CL',equals=0.5)",
            "status": "removed",
            "blueprint": "prob.model.add_constraint(f'{point_name}.wing_perf.CL', equals=0.5)",
            "generated": "",
        },
        {
            "item": "call:add_constraint:prob.model.add_constraint(f'{point_name}.wing_perf.CL',equals=0.45)",
            "status": "added",
            "blueprint": "",
            "generated": "prob.model.add_constraint(f'{point_name}.wing_perf.CL', equals=0.45)",
        },
        {
            "item": "dict:surface.twist_cp",
            "status": "changed",
            "blueprint": "surface['twist_cp'] = _crm_twist_cp if _crm_twist_cp is not None else np.zeros(5)",
            "generated": "surface['twist_cp'] = _crm_twist_cp if _crm_twist_cp is not None else np.zeros(3)",
        },
    ]

    assert _contract_violations(prompt, changes) == []


def test_contract_allows_neutral_twist_initialization_for_rectangular_wing():
    prompt = (
        "Optimize a rectangular wing. Vary three twist control points from "
        "-10 to 15 deg."
    )
    changes = [
        {
            "item": "dict:surface.twist_cp",
            "status": "changed",
            "blueprint": (
                "surface['twist_cp'] = _crm_twist_cp "
                "if _crm_twist_cp is not None else np.zeros(5)"
            ),
            "generated": "surface['twist_cp'] = np.zeros(3)",
        }
    ]

    assert _contract_violations(prompt, changes) == []


def test_contract_allows_requested_weighted_cd_objective_replacement():
    prompt = (
        "Use three points with CL targets 0.40, 0.50, 0.55 and minimize a "
        "weighted sum of CD."
    )
    changes = [
        {
            "item": "call:add_constraint:prob.model.add_constraint(f'aero_point_{i}.wing_perf.CL',equals=[0.45,0.5][i])",
            "status": "removed",
            "blueprint": "prob.model.add_constraint(f'aero_point_{i}.wing_perf.CL', equals=[0.45, 0.5][i])",
            "generated": "",
        },
        {
            "item": "call:add_constraint:prob.model.add_constraint(f'aero_point_{i}.wing_perf.CL',equals=cl_targets[i])",
            "status": "added",
            "blueprint": "",
            "generated": "prob.model.add_constraint(f'aero_point_{i}.wing_perf.CL', equals=cl_targets[i])",
        },
        {
            "item": "call:add_objective:CD",
            "status": "removed",
            "blueprint": "prob.model.add_objective('CD', scaler=10000.0)",
            "generated": "",
        },
        {
            "item": "call:add_objective:weighted_CD",
            "status": "added",
            "blueprint": "",
            "generated": "prob.model.add_objective('weighted_CD', scaler=10000.0)",
        },
    ]

    assert _contract_violations(prompt, changes) == []


def test_contract_still_blocks_unreplaced_constraint_removal():
    changes = [
        {
            "item": "call:add_constraint:AS_point_1.L_equals_W",
            "status": "removed",
            "blueprint": "prob.model.add_constraint('AS_point_1.L_equals_W', equals=0.0)",
            "generated": "",
        }
    ]

    violations = _contract_violations("Minimize fuel burn.", changes)

    assert len(violations) == 1
    assert violations[0]["changed_item"] == "call:add_constraint:AS_point_1.L_equals_W"


def test_contract_allows_optional_twist_removal_when_not_requested():
    changes = [
        {
            "item": "call:add_design_var:wing.twist_cp",
            "status": "removed",
            "blueprint": "prob.model.add_design_var('wing.twist_cp', lower=-10.0, upper=15.0)",
            "generated": "",
        },
        {
            "item": "dict:surface.twist_cp",
            "status": "removed",
            "blueprint": "surface['twist_cp'] = np.zeros(5)",
            "generated": "",
        },
    ]

    assert _contract_violations("Analyze a rectangular wing at Mach 0.45.", changes) == []


def test_contract_blocks_inferred_wave_drag_change():
    changes = [
        {
            "item": "dict:surf_dict.with_wave",
            "status": "changed",
            "blueprint": "surf_dict['with_wave'] = False",
            "generated": "surf_dict['with_wave'] = True",
        }
    ]

    violations = _contract_violations(
        "Three-point drag minimization at Mach 0.78.", changes
    )

    assert len(violations) == 1
    assert violations[0]["changed_item"] == "dict:surf_dict.with_wave"


def test_contract_allows_explicit_wave_drag_request():
    changes = [
        {
            "item": "dict:surf_dict.with_wave",
            "status": "changed",
            "blueprint": "surf_dict['with_wave'] = False",
            "generated": "surf_dict['with_wave'] = True",
        }
    ]

    assert _contract_violations("Turn wave drag on.", changes) == []


def test_approved_relaxation_prompt_marks_retry_scope():
    prompt = build_approved_relaxation_prompt(
        "Minimize drag with CL = 0.5.",
        "Apply these relaxations and retry:\nIncrease alpha upper bound to 10 deg.",
    )

    assert APPROVED_RELAXATION_HEADER in prompt
    assert "explicitly approved by the user" in prompt
    assert "Preserve all unrelated blueprint assumptions." in prompt
    assert "Increase alpha upper bound to 10 deg." in prompt


def test_approved_relaxation_still_blocks_unrelated_mesh_drift():
    prompt = build_approved_relaxation_prompt(
        "Minimize drag with CL = 0.5.",
        "Apply these relaxations and retry:\nIncrease alpha upper bound to 10 deg.",
    )
    changes = [
        {
            "item": "dict:mesh_dict.num_y",
            "status": "changed",
            "blueprint": "mesh_dict['num_y'] = 7",
            "generated": "mesh_dict['num_y'] = 15",
        }
    ]

    violations = _contract_violations(prompt, changes)

    assert len(violations) == 1
    assert violations[0]["changed_item"] == "dict:mesh_dict.num_y"


def test_contract_blocks_upward_load_filling_all_six_components():
    changes = [
        {
            "item": "call:add_output:loads",
            "status": "changed",
            "blueprint": "indep_var_comp.add_output('loads', val=loads, units='N')",
            "generated": "indep_var_comp.add_output('loads', val=np.ones((ny, 6)) * (4e4 / ny), units='N')",
        }
    ]

    violations = _contract_violations("Apply a total upward load of 4e4 N.", changes)

    assert len(violations) == 1
    assert violations[0]["changed_item"] == "call:add_output:loads"


def test_contract_blocks_upward_load_variable_filling_all_six_components():
    changes = [
        {
            "item": "assign:loads_array",
            "status": "changed",
            "blueprint": "loads_array = np.zeros((ny, 6))",
            "generated": "loads_array = np.ones((ny, 6)) * (4e4 / ny)",
        },
        {
            "item": "call:add_output:loads",
            "status": "changed",
            "blueprint": "indep_var_comp.add_output('loads', val=loads, units='N')",
            "generated": "indep_var_comp.add_output('loads', val=loads_array, units='N')",
        },
    ]

    violations = _contract_violations("Apply a total upward load of 4e4 N.", changes)

    assert len(violations) == 1
    assert violations[0]["changed_item"] == "assign:loads_array"


def test_semantic_changes_watch_surface_update_solver_keys():
    blueprint = "surf_dict = {}\nsurf_dict.update({'with_wave': False})\n"
    generated = "surf_dict = {}\nsurf_dict.update({'with_wave': True})\n"

    changes = _semantic_changes(blueprint, generated)

    assert any(change["item"] == "dict:surf_dict.with_wave" for change in changes)
