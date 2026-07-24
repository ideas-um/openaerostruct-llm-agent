import ast
import difflib
import json
import logging
import os
import re

from .config import LLMBackendTransientError, estimate_tokens, get_llm_response

logger = logging.getLogger("LLM_Backend")
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_LLM_DIR)
_BLUEPRINTS_DIR = os.path.realpath(os.path.join(_SRC_DIR, "blueprints"))


def _compact_code(text: str) -> str:
    text = re.sub(r"#.*", "", str(text))
    text = re.sub(r"\s+", "", text)
    return text.strip().rstrip(",")


def _number_tokens(text: str) -> list[str]:
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", text, re.I)
    return [str(float(n)) for n in nums]


def _semantically_same(left: str, right: str) -> bool:
    if _compact_code(left) == _compact_code(right):
        return True
    left_nums = _number_tokens(left)
    right_nums = _number_tokens(right)
    return bool(left_nums) and left_nums == right_nums


_HIGH_RISK_RE = re.compile(
    r"""
    \b(
        speed_of_sound|Mach_number|rho|load_factor|altitude|
        _Mach_numbers|_rho_vals|_altitudes|_a_vals|_v_vals|_mu_vals|
        num_y|num_x|generate_mesh|
        loads|forces|
        span|root_chord|chord|tip_chord|taper|sweep|dihedral|
        t_over_c_cp|thickness_cp|radius_cp|spar_thickness_cp|skin_thickness_cp|
        fuel_mass|fuelburn|W0|CT|R|grav_constant|
        E|G|yield|safety_factor|mrho|fem_origin|
        add_design_var|add_constraint|add_objective|
        SqliteRecorder|add_recorder|recording_options|work_dir|
        to_csv|analysis_csv
    )\b
    |(?<!\w)v\s*=
    |(?<!\w)re\s*=
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _changed_executable_lines(diff_text: str) -> list[str]:
    lines = []
    for line in diff_text.splitlines():
        if not line.startswith(("+", "-")):
            continue
        if line.startswith(("+++", "---")):
            continue
        code = line[1:].strip()
        if not code or code.startswith("#"):
            continue
        lines.append(line)
    return lines


def _high_risk_diff_lines(diff_text: str) -> list[str]:
    return [
        line
        for line in _changed_executable_lines(diff_text)
        if _HIGH_RISK_RE.search(line[1:])
    ]


def _format_high_risk_summary(diff_text: str, limit: int = 80) -> str:
    lines = _high_risk_diff_lines(diff_text)
    if not lines:
        return "No high-risk executable diff lines detected."
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(f"... {len(lines) - limit} more high-risk lines omitted")
    return "\n".join(shown)


_WATCHED_CALLS = {
    "add_subsystem",
    "add_constraint",
    "add_design_var",
    "add_objective",
    "add_output",
    "connect",
    "set_input_defaults",
}

_WATCHED_ASSIGNMENTS = {
    "mesh_dict",
    "surf_dict",
    "surface",
    "loads",
    "loads_array",
    "forces",
    "forces_val",
    "forces_array",
    "_altitudes",
    "_Mach_numbers",
    "_rho_vals",
    "_a_vals",
    "_v_vals",
    "_mu_vals",
    "Mach_number",
    "rho",
    "speed_of_sound",
    "v",
    "re",
    "altitude",
    "CT",
    "R",
    "W0",
    "fuel_mass",
    "load_factor",
}

_WATCHED_DICT_KEYS = {
    "num_y",
    "num_x",
    "wing_type",
    "span",
    "root_chord",
    "chord",
    "tip_chord",
    "taper",
    "sweep",
    "dihedral",
    "span_cos_spacing",
    "chord_cos_spacing",
    "num_twist_cp",
    "twist_cp",
    "t_over_c_cp",
    "thickness_cp",
    "radius_cp",
    "spar_thickness_cp",
    "skin_thickness_cp",
    "CL0",
    "CD0",
    "with_viscous",
    "with_wave",
    "S_ref_type",
    "E",
    "G",
    "yield",
    "safety_factor",
    "mrho",
    "fem_origin",
    "Wf_reserve",
    "distributed_fuel_weight",
    "struct_weight_relief",
}


def _ast_code(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


def _numeric_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return str(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _numeric_literal(node.operand)
        return f"-{value}" if value is not None else None
    return None


def _call_attr_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        base = _call_attr_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _literal_array_values(node: ast.AST) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for elt in node.elts:
            value = _numeric_literal(elt)
            if value is None:
                return None
            values.append(value)
        return values
    if isinstance(node, ast.Call) and _call_attr_name(node.func).endswith("array"):
        if node.args:
            return _literal_array_values(node.args[0])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_call = node.left if isinstance(node.left, ast.Call) else None
        scalar = _numeric_literal(node.right)
        if left_call is None or scalar is None:
            left_call = node.right if isinstance(node.right, ast.Call) else None
            scalar = _numeric_literal(node.left)
        if left_call is None or scalar is None:
            return None
        if not _call_attr_name(left_call.func).endswith("ones") or not left_call.args:
            return None
        shape = left_call.args[0]
        if isinstance(shape, ast.Tuple) and shape.elts:
            shape = shape.elts[0]
        if isinstance(shape, ast.Constant) and isinstance(shape.value, int):
            return [scalar] * shape.value
    return None


def _statement_array_values(statement: str) -> list[str] | None:
    try:
        tree = ast.parse(statement)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    node = tree.body[0]
    if isinstance(node, ast.Assign):
        return _literal_array_values(node.value)
    if isinstance(node, ast.AnnAssign):
        return _literal_array_values(node.value) if node.value else None
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        for keyword in node.value.keywords:
            if keyword.arg == "val":
                values = _literal_array_values(keyword.value)
                if values is not None:
                    return values
    return None


def _array_element_changes(old: str, new: str) -> list[dict[str, str]] | None:
    old_values = _statement_array_values(old)
    new_values = _statement_array_values(new)
    if old_values is None or new_values is None:
        return None
    changes = []
    for idx in range(max(len(old_values), len(new_values))):
        old_value = old_values[idx] if idx < len(old_values) else ""
        new_value = new_values[idx] if idx < len(new_values) else ""
        if old_value == new_value:
            continue
        status = "changed" if old_value and new_value else "added" if new_value else "removed"
        changes.append(
            {
                "index": idx,
                "status": status,
                "blueprint": old_value,
                "generated": new_value,
            }
        )
    return changes or None


def _literal_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _target_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        return _target_name(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return ",".join(_target_name(elt) for elt in node.elts)
    return ""


def _dict_key_name(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return str(node.value)
    return _ast_code(node)


def _subscript_dict_item(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Subscript):
        return None
    target = _target_name(node.value)
    if target not in {"mesh_dict", "surf_dict", "surface"}:
        return None
    key = _dict_key_name(node.slice)
    if key not in _WATCHED_DICT_KEYS:
        return None
    return target, key


def _subscript_assignment_item(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if _target_name(node.value) not in {
        "loads",
        "loads_array",
        "forces",
        "forces_val",
        "forces_array",
    }:
        return None
    return _ast_code(node)


def _call_signature(call: ast.Call) -> tuple[str, str] | None:
    func_name = _call_attr_name(call.func)
    short_name = func_name.split(".")[-1]
    if short_name not in _WATCHED_CALLS:
        return None
    first_arg = _literal_string(call.args[0]) if call.args else ""
    if short_name == "connect":
        source = _ast_code(call.args[0]) if call.args else ""
        target = _ast_code(call.args[1]) if len(call.args) > 1 else ""
        first_arg = f"{source}->{target}"
    elif not first_arg:
        first_arg = _compact_code(_ast_code(call))
    return short_name, first_arg


def _collect_semantic_units(code: str) -> dict[str, str]:
    tree = ast.parse(code)
    units: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_names = [
                _target_name(target)
                for target in node.targets
                if not isinstance(target, ast.Subscript)
            ]
            for target in target_names:
                if target in _WATCHED_ASSIGNMENTS:
                    units[f"assign:{target}"] = _ast_code(node)
            for target_node in node.targets:
                item = _subscript_dict_item(target_node)
                if item:
                    target, key = item
                    units[f"dict:{target}.{key}"] = (
                        f"{target}[{key!r}] = {_ast_code(node.value)}"
                    )
                assignment_item = _subscript_assignment_item(target_node)
                if assignment_item:
                    units[f"assign:{assignment_item}"] = _ast_code(node)
            if isinstance(node.value, ast.Dict):
                for target in target_names:
                    if target not in {"mesh_dict", "surf_dict", "surface"}:
                        continue
                    for key_node, val_node in zip(node.value.keys, node.value.values):
                        if key_node is None:
                            continue
                        key = _dict_key_name(key_node)
                        if key in _WATCHED_DICT_KEYS:
                            units[f"dict:{target}.{key}"] = (
                                f"{target}[{key!r}] = {_ast_code(val_node)}"
                            )

        elif isinstance(node, ast.AnnAssign):
            target = _target_name(node.target)
            if target in _WATCHED_ASSIGNMENTS:
                units[f"assign:{target}"] = _ast_code(node)

        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and _target_name(func.value) in {"mesh_dict", "surf_dict", "surface"}
                and call.args
                and isinstance(call.args[0], ast.Dict)
            ):
                target = _target_name(func.value)
                for key_node, val_node in zip(call.args[0].keys, call.args[0].values):
                    if key_node is None:
                        continue
                    key = _dict_key_name(key_node)
                    if key in _WATCHED_DICT_KEYS:
                        units[f"dict:{target}.{key}"] = (
                            f"{target}[{key!r}] = {_ast_code(val_node)}"
                        )

            sig = _call_signature(call)
            if sig:
                units[f"call:{sig[0]}:{sig[1]}"] = _ast_code(node)

    return units


def _semantic_changes(blueprint_code: str, generated_code: str) -> list[dict[str, str]]:
    try:
        old_units = _collect_semantic_units(blueprint_code)
        new_units = _collect_semantic_units(generated_code)
    except SyntaxError:
        return []

    changes = []
    for key in sorted(set(old_units) | set(new_units)):
        old = old_units.get(key, "")
        new = new_units.get(key, "")
        if old and new and _semantically_same(old, new):
            continue
        status = "changed" if old and new else "added" if new else "removed"
        change = {
            "item": key,
            "status": status,
            "blueprint": old,
            "generated": new,
        }
        element_changes = _array_element_changes(old, new)
        if element_changes:
            change["element_changes"] = element_changes
        changes.append(change)
    return changes


def _format_semantic_changes(changes: list[dict[str, str]], limit: int = 80) -> str:
    if not changes:
        return "No semantic high-risk statement changes detected."
    shown = changes[:limit]
    if len(changes) > limit:
        shown = shown + [
            {
                "item": "omitted",
                "status": "omitted",
                "blueprint": "",
                "generated": f"{len(changes) - limit} more semantic changes omitted",
            }
        ]
    return json.dumps(shown, indent=2)


def _format_repair_feedback(report: dict) -> str:
    lines = [
        "Blueprint consistency error:",
        "Make only these auditor-requested repairs. Do not reinterpret the original task.",
    ]
    for idx, violation in enumerate(report.get("violations", []), start=1):
        item = violation.get("changed_item", f"violation {idx}")
        old = violation.get("blueprint_value", "")
        new = violation.get("generated_value", "")
        reason = violation.get("reason", "")
        repair = violation.get("repair_instruction", "")
        lines.append(f"{idx}. {item}")
        if reason:
            lines.append(f"Reason: {reason}")
        if new:
            lines.append("Replace generated snippet:")
            lines.append(str(new))
        if old:
            lines.append("With blueprint snippet:")
            lines.append(str(old))
        if repair:
            lines.append(f"Instruction: {repair}")
    return "\n".join(lines)


def _parse_audit_response(response: str) -> dict:
    match = re.search(r"<audit>(.*?)</audit>", response, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in auditor response.")
        json_str = response[start:end]

    json_str = re.sub(r"^```json\s*|^```\s*", "", json_str, flags=re.MULTILINE)
    json_str = re.sub(r"```$", "", json_str, flags=re.MULTILINE).strip()
    data = json.loads(json_str)
    if "passed" not in data:
        raise ValueError("Auditor response is missing the required passed field.")
    passed = data["passed"]
    if isinstance(passed, str):
        passed = passed.strip().lower() == "true"
    data["passed"] = bool(passed)
    data["violations"] = data.get("violations") or []
    data["reviewed_changes"] = data.get("reviewed_changes") or []
    data["warnings"] = data.get("warnings") or []
    data["feedback_for_coder"] = data.get("feedback_for_coder", "")
    return data


def _malformed_audit_retry_message(user_message: str, error: Exception) -> str:
    return (
        f"{user_message}\n\n"
        "### PREVIOUS AUDIT RESPONSE WAS MALFORMED ###\n"
        f"The previous auditor response could not be parsed as the required JSON object: {error}\n\n"
        "Return exactly one <audit> XML section containing valid JSON. Do not include "
        "Markdown fences, comments, trailing commas, or prose outside the XML tags."
    )


def _audit_infrastructure_report(reason: str) -> dict:
    return {
        "passed": False,
        "audit_infrastructure_error": True,
        "invalid_audit_reasons": [reason],
        "violations": [],
        "reviewed_changes": [],
        "warnings": [f"Blueprint auditor infrastructure failure: {reason}"],
        "feedback_for_coder": "",
    }


def _attach_audit_artifacts(
    report: dict, full_diff: str, semantic_changes: list[dict[str, str]]
) -> dict:
    report["diff"] = full_diff
    report["semantic_changes"] = semantic_changes
    report["high_risk_diff_lines"] = _high_risk_diff_lines(full_diff)
    return report


def _load_blueprint(blueprint: str) -> tuple[str, str]:
    path = os.path.join(_BLUEPRINTS_DIR, blueprint)
    if not os.path.exists(path):
        return "", ""
    with open(path, "r", encoding="utf-8") as fh:
        return path, fh.read()


def _without_comment_only_lines(code: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in code.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _make_diff(blueprint: str, blueprint_code: str, generated_code: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            _without_comment_only_lines(blueprint_code).splitlines(),
            _without_comment_only_lines(generated_code).splitlines(),
            fromfile=f"blueprints/{blueprint}",
            tofile="generated/code.py",
            lineterm="",
        )
    )


def audit_blueprint_consistency(
    user_prompt: str,
    blueprints: list[str],
    generated_code: str,
    model_name: str,
    provider: str,
) -> tuple[dict, int, int]:
    """Give the LLM Auditor the evidence and return its decision unchanged."""
    blueprint = blueprints[0] if blueprints else ""
    blueprint_path, blueprint_code = _load_blueprint(blueprint)
    if not blueprint_code:
        report = _audit_infrastructure_report(f"Blueprint not found: {blueprint}")
        return report, 0, 0

    full_diff = _make_diff(blueprint, blueprint_code, generated_code)
    semantic_changes = _semantic_changes(blueprint_code, generated_code)

    prompt_path = os.path.join(_LLM_DIR, "auditor.md")
    with open(prompt_path, "r", encoding="utf-8") as fh:
        system_prompt = fh.read()

    user_message = (
        f"### USER REQUEST ###\n{user_prompt}\n\n"
        f"### SELECTED BLUEPRINT ###\n{blueprint}\n{blueprint_path}\n\n"
        f"### HIGH-RISK SEMANTIC STATEMENT CHANGES TO REVIEW ###\n"
        f"```json\n{_format_semantic_changes(semantic_changes)}\n```\n\n"
        f"### RAW HIGH-RISK DIFF LINES FOR DEBUG ONLY ###\n"
        f"```diff\n{_format_high_risk_summary(full_diff)}\n```\n\n"
        f"### UNIFIED DIFF ###\n```diff\n{full_diff}\n```\n\n"
        "Return the audit object."
    )
    in_t = estimate_tokens(system_prompt + "\n" + user_message)
    out_t = 0

    try:
        response = get_llm_response(
            user_message, model_name, system_prompt, provider=provider
        )
        out_t += estimate_tokens(response)
        try:
            report = _parse_audit_response(response)
        except Exception as parse_exc:
            logger.warning("Blueprint auditor returned malformed output; retrying once.")
            retry_message = _malformed_audit_retry_message(user_message, parse_exc)
            retry_response = get_llm_response(
                retry_message, model_name, system_prompt, provider=provider
            )
            in_t += estimate_tokens(system_prompt + "\n" + retry_message)
            out_t += estimate_tokens(retry_response)
            try:
                report = _parse_audit_response(retry_response)
                report["auditor_retry"] = True
            except Exception as retry_exc:
                report = _audit_infrastructure_report(
                    f"Auditor response malformed after retry: {retry_exc}"
                )
                report["auditor_retry"] = True

        if not report.get("passed", True) and not report.get(
            "audit_infrastructure_error"
        ):
            report["feedback_for_coder"] = _format_repair_feedback(report)
        report = _attach_audit_artifacts(report, full_diff, semantic_changes)
        return report, in_t, out_t
    except LLMBackendTransientError:
        raise
    except Exception as exc:
        logger.error(f"Blueprint auditor infrastructure failure: {exc}")
        report = _audit_infrastructure_report(str(exc))
        report = _attach_audit_artifacts(report, full_diff, semantic_changes)
        return report, in_t, out_t
