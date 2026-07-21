import difflib
import ast
import json
import logging
import os
import re

from .config import get_llm_response, LLMBackendTransientError

logger = logging.getLogger("LLM_Backend")
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_LLM_DIR)
_BLUEPRINTS_DIR = os.path.realpath(os.path.join(_SRC_DIR, "blueprints"))


def _diff_lines(diff_text: str, prefix: str) -> list[str]:
    lines = []
    for line in diff_text.splitlines():
        if line.startswith(prefix) and not line.startswith(prefix * 3):
            lines.append(line[1:])
    return lines


def _compact_code(text: str) -> str:
    text = re.sub(r"#.*", "", str(text))
    text = re.sub(r"\s+", "", text)
    return text.strip().rstrip(",")


def _number_tokens(text: str) -> list[str]:
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", text, re.I)
    return [str(float(n)) for n in nums]


def _is_absent_value(text: str) -> bool:
    return bool(re.search(r"\b(absent|inactive|not present|removed|missing)\b", text, re.I))


def _snippet_present(snippet: str, diff_text: str, prefix: str) -> bool:
    if _is_absent_value(snippet):
        return True
    snippet = _compact_code(snippet)
    if not snippet:
        return False
    side = _compact_code("\n".join(_diff_lines(diff_text, prefix)))
    return snippet in side


def _semantically_same(left: str, right: str) -> bool:
    if _compact_code(left) == _compact_code(right):
        return True
    left_nums = _number_tokens(left)
    right_nums = _number_tokens(right)
    return bool(left_nums) and left_nums == right_nums


_CONTRADICTION_RE = re.compile(
    r"\b(consistent with the request|acceptable|no actual violation|current "
    r"implementation is correct|finds no actual violation)\b",
    re.IGNORECASE,
)

_HIGH_RISK_RE = re.compile(
    r"""
    \b(
        speed_of_sound|Mach_number|rho|load_factor|altitude|
        _Mach_numbers|_rho_vals|_altitudes|_a_vals|_v_vals|_mu_vals|
        num_y|num_x|generate_mesh|
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
    suffix = []
    if len(lines) > limit:
        suffix.append(f"... {len(lines) - limit} more high-risk lines omitted")
    return "\n".join(shown + suffix)


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

_POINTWISE_REVIEW_ITEMS = {
    "assign:_Mach_numbers",
    "assign:_rho_vals",
    "assign:_altitudes",
    "assign:_v_vals",
    "assign:_a_vals",
    "assign:_mu_vals",
    "assign:load_factor",
    "call:add_output:Mach_number",
    "call:add_output:rho",
    "call:add_output:load_factor",
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


def _normalized_statement(node: ast.AST) -> str:
    return _compact_code(_ast_code(node))


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


def _call_attr_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        base = _call_attr_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _dict_key_name(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return str(node.value)
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
            target_names = [_target_name(target) for target in node.targets]
            for target in target_names:
                if target in _WATCHED_ASSIGNMENTS:
                    units[f"assign:{target}"] = _ast_code(node)
            if isinstance(node.value, ast.Dict):
                for target in target_names:
                    if target not in {"mesh_dict", "surf_dict", "surface"}:
                        continue
                    for key_node, val_node in zip(node.value.keys, node.value.values):
                        if key_node is None:
                            continue
                        key = _dict_key_name(key_node)
                        if key in _WATCHED_DICT_KEYS:
                            units[f"dict:{target}.{key}"] = f"{target}[{key!r}] = {_ast_code(val_node)}"

        elif isinstance(node, ast.AnnAssign):
            target = _target_name(node.target)
            if target in _WATCHED_ASSIGNMENTS:
                units[f"assign:{target}"] = _ast_code(node)

        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            sig = _call_signature(node.value)
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


def _flatten_strings(value) -> list[str]:
    if isinstance(value, dict):
        parts = []
        for item in value.values():
            parts.extend(_flatten_strings(item))
        return parts
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.extend(_flatten_strings(item))
        return parts
    return [str(value)]


def _review_material(report: dict) -> str:
    material = {
        "reviewed_changes": report.get("reviewed_changes") or [],
        "violations": report.get("violations") or [],
    }
    return _compact_code("\n".join(_flatten_strings(material)))


def _review_names(report: dict) -> str:
    return "\n".join(_flatten_strings(report.get("reviewed_changes") or [])).lower()


def _review_text(report: dict) -> str:
    material = {
        "reviewed_changes": report.get("reviewed_changes") or [],
        "violations": report.get("violations") or [],
    }
    return "\n".join(_flatten_strings(material)).lower()


def _identifier_markers(text: str) -> list[str]:
    markers = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        token = token.lower()
        if token in {
            "prob",
            "model",
            "add_design_var",
            "add_constraint",
            "add_objective",
            "lower",
            "upper",
            "equals",
            "scaler",
            "ref",
            "np",
            "array",
            "ones",
            "zeros",
        }:
            continue
        markers.append(token)
    return markers


def _line_markers(line: str) -> list[str]:
    code = line[1:].strip()
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', code)
    names = []
    for left, right in quoted:
        value = left or right
        if not value:
            continue
        names.append(value)
        names.extend(_identifier_markers(value))
    assign = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", code)
    if assign:
        names.append(assign.group(1))
    names.extend(_identifier_markers(code))
    return [name.lower() for name in names if len(name.strip()) > 1]


def _line_is_reviewed(line: str, material: str, names: str) -> bool:
    snippet = _compact_code(line[1:])
    if snippet and snippet in material:
        return True
    markers = _line_markers(line)
    return bool(markers) and any(marker in names for marker in markers)


def _semantic_item_markers(change: dict[str, str]) -> list[str]:
    markers = [change.get("item", "").lower()]
    markers.extend(_identifier_markers(change.get("item", "")))
    for side in ("blueprint", "generated"):
        text = change.get(side, "")
        markers.extend(_identifier_markers(text))
        for left, right in re.findall(r'"([^"]+)"|\'([^\']+)\'', text):
            value = left or right
            if value:
                markers.append(value.lower())
                markers.extend(_identifier_markers(value))
    return [marker for marker in markers if marker and len(marker) > 1]


def _requires_exact_semantic_review(change: dict[str, str]) -> bool:
    item = change.get("item", "")
    if not item.startswith("dict:"):
        return False
    key = item.rsplit(".", 1)[-1]
    return key in _WATCHED_DICT_KEYS


def _requires_pointwise_review(change: dict[str, str]) -> bool:
    return bool(change.get("element_changes")) and change.get("item") in _POINTWISE_REVIEW_ITEMS


def _pointwise_review_is_complete(change: dict[str, str], text: str) -> bool:
    for element in change.get("element_changes", []):
        idx = str(element.get("index"))
        markers = (
            f"index {idx}",
            f"point {idx}",
            f"element {idx}",
            f"[{idx}]",
        )
        if not any(marker in text for marker in markers):
            return False
    return True


def _semantic_item_is_named(change: dict[str, str], names: str) -> bool:
    item = change.get("item", "").lower()
    return bool(item and item in names)


def _semantic_change_is_reviewed(change: dict[str, str], material: str, names: str) -> bool:
    old = _compact_code(change.get("blueprint", ""))
    new = _compact_code(change.get("generated", ""))
    if _semantic_item_is_named(change, names):
        return True
    if old and old in material:
        return True
    if new and new in material:
        return True
    if _requires_exact_semantic_review(change):
        return False
    markers = _semantic_item_markers(change)
    return bool(markers) and any(marker in names for marker in markers)


def _snippet_present_in_semantic(snippet: str, changes: list[dict[str, str]], side: str) -> bool:
    if _is_absent_value(snippet):
        return True
    snippet = _compact_code(snippet)
    if not snippet:
        return False
    return any(snippet in _compact_code(change.get(side, "")) for change in changes)


def _unreviewed_semantic_changes(report: dict, changes: list[dict[str, str]]) -> list[dict[str, str]]:
    material = _review_material(report)
    names = _review_names(report)
    text = _review_text(report)
    missing = []
    for change in changes:
        if not _semantic_change_is_reviewed(change, material, names):
            missing.append(change)
            continue
        if _requires_pointwise_review(change) and not _pointwise_review_is_complete(change, text):
            missing.append(change)
    return missing


def _validate_audit_report(report: dict, diff_text: str, semantic_changes: list[dict[str, str]] | None = None) -> dict:
    """Reject impossible auditor failures without judging engineering intent."""
    semantic_changes = semantic_changes or []
    if report.get("passed", True):
        missing = _unreviewed_semantic_changes(report, semantic_changes)
        if not missing:
            return report
        fixed = dict(report)
        fixed["passed"] = False
        fixed["violations"] = [
            {
                "severity": "blocking",
                "changed_item": "unreviewed high-risk semantic changes",
                "blueprint_value": "",
                "generated_value": _format_semantic_changes(missing, limit=20),
                "reason": (
                    "The auditor returned passed=true but did not explicitly review "
                    "these high-risk semantic statement changes."
                ),
                "repair_instruction": (
                    "Return a new attempt that either restores these lines or includes "
                    "an explicit, statement-specific justification tied to the user request "
                    "or required wiring."
                ),
            }
        ]
        fixed["invalid_audit"] = True
        fixed["invalid_audit_reasons"] = [
            "passed=true without reviewing every high-risk semantic statement change"
        ]
        return fixed

    missing = _unreviewed_semantic_changes(report, semantic_changes)
    if missing:
        report = dict(report)
        report["unreviewed_semantic_changes"] = missing[:20]

    problems = []
    text = " ".join(
        [
            str(report.get("feedback_for_coder", "")),
            " ".join(str(v.get("reason", "")) for v in report.get("violations", [])),
            " ".join(
                str(v.get("repair_instruction", ""))
                for v in report.get("violations", [])
            ),
        ]
    )
    if _CONTRADICTION_RE.search(text):
        problems.append("auditor contradicted passed=false")

    violations = report.get("violations") or []
    if not violations:
        problems.append("passed=false without violations")

    for idx, violation in enumerate(violations, start=1):
        old = str(violation.get("blueprint_value", ""))
        new = str(violation.get("generated_value", ""))
        if not (
            _snippet_present(old, diff_text, "-")
            or _snippet_present_in_semantic(old, semantic_changes, "blueprint")
        ):
            problems.append(f"violation {idx} blueprint_value is not in diff or semantic changes")
        if not (
            _snippet_present(new, diff_text, "+")
            or _snippet_present_in_semantic(new, semantic_changes, "generated")
        ):
            problems.append(f"violation {idx} generated_value is not in diff or semantic changes")
        if _semantically_same(old, new):
            problems.append(f"violation {idx} old/new snippets are equivalent")

    if not problems:
        return report

    fixed = dict(report)
    fixed["passed"] = True
    fixed["violations"] = []
    fixed["feedback_for_coder"] = ""
    fixed["invalid_audit"] = True
    fixed["invalid_audit_reasons"] = problems
    return fixed


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


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(str(text)))
    except ImportError:
        return len(str(text)) // 4


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
    passed = data.get("passed", False)
    if isinstance(passed, str):
        passed = passed.strip().lower() == "true"
    data["passed"] = bool(passed)
    data["violations"] = data.get("violations") or []
    data["reviewed_changes"] = data.get("reviewed_changes") or []
    data["warnings"] = data.get("warnings") or []
    data["feedback_for_coder"] = data.get("feedback_for_coder", "")
    return data


def _load_blueprint(blueprint: str) -> tuple[str, str]:
    path = os.path.join(_BLUEPRINTS_DIR, blueprint)
    if not os.path.exists(path):
        return "", ""
    with open(path, "r", encoding="utf-8") as fh:
        return path, fh.read()


def _make_diff(blueprint: str, blueprint_code: str, generated_code: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            blueprint_code.splitlines(),
            generated_code.splitlines(),
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
    """LLM-assisted guard that turns unrequested blueprint drift into retry feedback."""
    blueprint = blueprints[0] if blueprints else ""
    blueprint_path, blueprint_code = _load_blueprint(blueprint)
    if not blueprint_code:
        return {
            "passed": True,
            "violations": [],
            "feedback_for_coder": "",
            "warning": f"Blueprint not found: {blueprint}",
        }, 0, 0

    full_diff = _make_diff(blueprint, blueprint_code, generated_code)
    if not full_diff.strip():
        return {"passed": True, "violations": [], "feedback_for_coder": ""}, 0, 0
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
    in_t = _approx_tokens(system_prompt + "\n" + user_message)

    try:
        response = get_llm_response(
            user_message, model_name, system_prompt, provider=provider
        )
        report = _parse_audit_response(response)
        report = _validate_audit_report(report, full_diff, semantic_changes)
        if not report.get("passed", True):
            report["feedback_for_coder"] = _format_repair_feedback(report)
        report["diff"] = full_diff
        report["semantic_changes"] = semantic_changes
        report["high_risk_diff_lines"] = _high_risk_diff_lines(full_diff)
        return report, in_t, _approx_tokens(response)
    except LLMBackendTransientError:
        raise
    except Exception as exc:
        logger.error(f"Blueprint auditor failed open: {exc}")
        return {
            "passed": True,
            "violations": [],
            "feedback_for_coder": "",
            "warning": f"Blueprint auditor failed open: {exc}",
            "diff": full_diff,
        }, in_t, 0
