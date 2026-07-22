from llm.auditor import _contract_violations, _make_diff, _semantic_changes
from llm.coder import _parse_response


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
