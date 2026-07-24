import difflib
import ast
import json
import logging
import os
import re

from .config import get_llm_response, LLMBackendTransientError, estimate_tokens

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

_PRESERVE_WORD_RE = re.compile(
    r"\b(preserve|preserved|keep|kept|unchanged|leave\s+unchanged|blueprint\s+value)\b",
    re.IGNORECASE,
)
_REMOVAL_WORD_RE = re.compile(r"\b(remove|delete|drop|omit|disable)\b", re.IGNORECASE)
_INITIAL_WORD_RE = re.compile(
    r"\b(initial|init|initially|initiali[sz]e|initiali[sz]ed|"
    r"initiali[sz]ation|guess)\b",
    re.IGNORECASE,
)
_SCALING_WORD_RE = re.compile(r"\b(ref|scaler|scaling|scale|conditioning)\b", re.IGNORECASE)
_MESH_RESOLUTION_RE = re.compile(
    r"\b(num_y|num_x|mesh\s+resolution|panel\s+count|panels?|discretization|discretisation)\b",
    re.IGNORECASE,
)
_T_OVER_C_ALIAS_RE = re.compile(
    r"\b(t_over_c_cp|t/c|t\s*over\s*c|thickness[-\s]*to[-\s]*chord|"
    r"thickness\s+ratio)\b",
    re.IGNORECASE,
)
_TWIST_REQUEST_RE = re.compile(r"\b(twist|twist_cp|washout|washin)\b", re.IGNORECASE)
_VAR_ALIAS_PATTERNS = {
    "twist_cp": re.compile(
        r"\b(twist(?:\s+(?:control\s+points?|CPs?))?|twist_cp|washout|washin)\b",
        re.IGNORECASE,
    ),
    "chord_cp": re.compile(
        r"\b(chord_cp|chord\s+(?:distribution|control\s+points?|CPs?))\b",
        re.IGNORECASE,
    ),
    "xshear_cp": re.compile(
        r"\b(xshear_cp|x[-\s]*shear(?:\s+(?:control\s+points?|CPs?))?|"
        r"spanwise\s+x[-\s]*(?:offset|shear))\b",
        re.IGNORECASE,
    ),
    "zshear_cp": re.compile(
        r"\b(zshear_cp|z[-\s]*shear(?:\s+(?:control\s+points?|CPs?))?|"
        r"spanwise\s+z[-\s]*(?:offset|shear))\b",
        re.IGNORECASE,
    ),
    "thickness_cp": re.compile(
        r"\b(thickness_cp|tube(?:\s+wall)?\s+thickness"
        r"(?:\s+(?:distribution|control\s+points?|CPs?))?)\b",
        re.IGNORECASE,
    ),
    "radius_cp": re.compile(
        r"\b(radius_cp|tube(?:\s+outer)?\s+radius"
        r"(?:\s+(?:distribution|control\s+points?|CPs?))?)\b",
        re.IGNORECASE,
    ),
    "spar_thickness_cp": re.compile(
        r"\b(spar_thickness_cp|spar(?:\s+wall)?\s+thickness"
        r"(?:\s+(?:distribution|control\s+points?|CPs?))?)\b",
        re.IGNORECASE,
    ),
    "skin_thickness_cp": re.compile(
        r"\b(skin_thickness_cp|skin(?:\s+panel)?\s+thickness"
        r"(?:\s+(?:distribution|control\s+points?|CPs?))?)\b",
        re.IGNORECASE,
    ),
    "t_over_c_cp": _T_OVER_C_ALIAS_RE,
}
_POINT1_WORD_RE = re.compile(
    r"\b(maneuver|manoeuvre|secondary|second\s+point|point\s*1|index\s*1)\b",
    re.IGNORECASE,
)
_POINT1_FLIGHT_ITEMS = {
    "assign:_Mach_numbers",
    "assign:_rho_vals",
    "assign:_altitudes",
    "assign:_v_vals",
    "assign:_a_vals",
    "assign:_mu_vals",
    "call:add_output:Mach_number",
    "call:add_output:rho",
}
_INITIAL_DICT_KEYS = {
    "twist_cp",
    "thickness_cp",
    "t_over_c_cp",
    "spar_thickness_cp",
    "skin_thickness_cp",
    "radius_cp",
}
_AERO_SOLVER_KEYS = {
    "CL0",
    "CD0",
    "S_ref_type",
    "with_viscous",
    "with_wave",
    "k_lam",
    "c_max_t",
}
_UPWARD_LOAD_RE = re.compile(
    r"\b(upward|vertical|lift\s+load|z[-\s]*force|force\s+column\s*2|loads?\s*\[:,\s*2\])\b",
    re.IGNORECASE,
)


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
                        units[f"dict:{target}.{key}"] = f"{target}[{key!r}] = {_ast_code(val_node)}"

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


def _prompt_preserves_point1(user_prompt: str) -> bool:
    return bool(
        _PRESERVE_WORD_RE.search(user_prompt)
        and _POINT1_WORD_RE.search(user_prompt)
    )


def _prompt_requests_removal(user_prompt: str) -> bool:
    return bool(_REMOVAL_WORD_RE.search(user_prompt))


def _var_initial_requested(user_prompt: str, var_name: str) -> bool:
    pattern = _VAR_ALIAS_PATTERNS.get(
        var_name, re.compile(rf"\b{re.escape(var_name)}\b", re.IGNORECASE)
    )
    for match in pattern.finditer(user_prompt):
        before = user_prompt[max(0, match.start() - 60) : match.start()]
        after = user_prompt[match.end() : match.end() + 140]
        after = re.split(
            r"(?=,\s*[A-Za-z_][A-Za-z0-9_]*\s*\()|[.;\n]",
            after,
            maxsplit=1,
        )[0]
        before = re.split(r"[,.;\n]", before)[-1]
        chunk = f"{before}{var_name}{after}"
        if _INITIAL_WORD_RE.search(chunk):
            return True
    return False


def _var_requested(user_prompt: str, var_name: str) -> bool:
    pattern = _VAR_ALIAS_PATTERNS.get(
        var_name, re.compile(rf"\b{re.escape(var_name)}\b", re.IGNORECASE)
    )
    return bool(pattern.search(user_prompt))


def _requested_control_point_count(user_prompt: str, var_name: str) -> int | None:
    pattern = _VAR_ALIAS_PATTERNS.get(
        var_name, re.compile(rf"\b{re.escape(var_name)}\b", re.IGNORECASE)
    )
    for match in pattern.finditer(user_prompt):
        chunk = user_prompt[
            max(0, match.start() - 80) : min(len(user_prompt), match.end() + 120)
        ]
        count_match = re.search(
            r"\b(\d+)\s*(?:control\s+points?|CPs?)\b", chunk, re.IGNORECASE
        )
        if count_match:
            return int(count_match.group(1))
    return None


def _control_point_count_change_only(
    user_prompt: str, change: dict[str, str], var_name: str
) -> bool:
    requested_count = _requested_control_point_count(user_prompt, var_name)
    if requested_count is None:
        return False
    old = _compact_code(change.get("blueprint", ""))
    new = _compact_code(change.get("generated", ""))
    if not old or not new:
        return False

    def normalize_count(text: str) -> str:
        return re.sub(
            r"((?:np\.)?(?:zeros|ones))\(\(?\d+\)?\)",
            r"\1(<count>)",
            text,
        )

    generated_count = re.search(
        r"(?:np\.)?(?:zeros|ones)\(\(?" + str(requested_count) + r"\)?\)", new
    )
    return bool(generated_count and normalize_count(old) == normalize_count(new))


def _changed_index(change: dict[str, str], index: int) -> dict[str, str] | None:
    for element in change.get("element_changes", []) or []:
        if int(element.get("index", -1)) == index and element.get("status") == "changed":
            return element
    return None


def _dict_key_from_item(item: str) -> str:
    return item.rsplit(".", 1)[-1] if "." in item else ""


def _call_keyword_value(statement: str, keyword_name: str) -> str | None:
    try:
        tree = ast.parse(statement)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    node = tree.body[0]
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    for keyword in node.value.keywords:
        if keyword.arg == keyword_name:
            return _normalized_statement(keyword.value)
    return None


def _design_var_bounds(
    semantic_changes: list[dict[str, str]], var_name: str
) -> tuple[float, float] | None:
    for change in semantic_changes:
        item = change.get("item", "")
        if not item.startswith("call:add_design_var:"):
            continue
        design_var_path = item.split("call:add_design_var:", 1)[1]
        if design_var_path.rsplit(".", 1)[-1] != var_name:
            continue
        statement = change.get("generated", "")
        lower = _call_keyword_value(statement, "lower")
        upper = _call_keyword_value(statement, "upper")
        try:
            if lower is not None and upper is not None:
                return float(lower), float(upper)
        except ValueError:
            return None
    return None


def _initial_change_moves_inside_requested_bounds(
    user_prompt: str,
    change: dict[str, str],
    semantic_changes: list[dict[str, str]],
    var_name: str,
) -> bool:
    if not _var_requested(user_prompt, var_name):
        return False
    bounds = _design_var_bounds(semantic_changes, var_name)
    old_values = _statement_array_values(change.get("blueprint", ""))
    new_values = _statement_array_values(change.get("generated", ""))
    if bounds is None or old_values is None or new_values is None or not new_values:
        return False
    lower, upper = bounds
    old = [float(value) for value in old_values]
    new = [float(value) for value in new_values]
    blueprint_is_outside = any(value < lower or value > upper for value in old)
    generated_is_strictly_inside = all(lower < value < upper for value in new)
    return blueprint_is_outside and generated_is_strictly_inside


def _call_scaling_changed(change: dict[str, str]) -> bool:
    old = change.get("blueprint", "")
    new = change.get("generated", "")
    for keyword in ("ref", "scaler"):
        if _call_keyword_value(old, keyword) != _call_keyword_value(new, keyword):
            return True
    return False


def _call_val_changed(change: dict[str, str]) -> bool:
    return _call_keyword_value(change.get("blueprint", ""), "val") != _call_keyword_value(
        change.get("generated", ""), "val"
    )


def _prompt_requests_scaling(user_prompt: str) -> bool:
    return bool(_SCALING_WORD_RE.search(user_prompt))


def _prompt_requests_mesh_resolution(user_prompt: str) -> bool:
    return bool(_MESH_RESOLUTION_RE.search(user_prompt))


def _prompt_requests_twist(user_prompt: str) -> bool:
    return bool(_TWIST_REQUEST_RE.search(user_prompt))


def _prompt_requests_aero_solver_key(user_prompt: str, key: str) -> bool:
    patterns = {
        "with_viscous": r"\b(with_viscous|viscous(?:\s+drag)?(?:\s+(?:on|off|true|false))?)\b",
        "with_wave": r"\b(with_wave|wave(?:\s+drag)?(?:\s+(?:on|off|true|false))?)\b",
        "CL0": r"\b(CL0|zero[-\s]*lift\s+lift)\b",
        "CD0": r"\b(CD0|profile\s+drag|zero[-\s]*lift\s+drag)\b",
        "S_ref_type": r"\b(S_ref_type|reference\s+area|wetted|projected)\b",
        "k_lam": r"\b(k_lam|laminar)\b",
        "c_max_t": r"\b(c_max_t|max(?:imum)?\s+thickness\s+location)\b",
    }
    pattern = patterns.get(key)
    return bool(pattern and re.search(pattern, user_prompt, re.IGNORECASE))


def _prompt_requests_upward_load(user_prompt: str) -> bool:
    return bool(_UPWARD_LOAD_RE.search(user_prompt))


def _load_change_fills_all_components(change: dict[str, str]) -> bool:
    generated = _compact_code(change.get("generated", ""))
    if not generated:
        return False
    if "[:,2]" in generated or "[:, 2]" in change.get("generated", ""):
        return False
    return bool(
        re.search(r"ones\(\(?[^)]*,6\)?\)", generated)
        or re.search(r"ones_like\(", generated)
    )


def _call_change_text(change: dict[str, str]) -> str:
    return " ".join(
        str(change.get(field, "")) for field in ("item", "blueprint", "generated")
    )


def _call_marker_set(text: str) -> set[str]:
    markers = set()
    checks = {
        "cl": r"(?<![A-Za-z0-9_])CL(?![A-Za-z0-9_])|wing_perf\.CL|\.CL\b",
        "cd": r"(?<![A-Za-z0-9_])CD(?![A-Za-z0-9_])|wing_perf\.CD|\.CD\b|weighted_CD",
        "weighted_cd": r"weighted_CD|weighted_cd",
        "l_equals_w": r"L_equals_W",
        "failure": r"\bfailure\b",
        "fuel_vol_delta": r"fuel_vol_delta",
        "fuel_diff": r"fuel_diff",
        "fuelburn": r"fuelburn",
        "structural_mass": r"structural_mass",
    }
    for marker, pattern in checks.items():
        if re.search(pattern, text, re.IGNORECASE):
            markers.add(marker)
    return markers


def _call_replacement_exists(
    change: dict[str, str], semantic_changes: list[dict[str, str]], user_prompt: str
) -> bool:
    item = change.get("item", "")
    if item.startswith("call:add_constraint:"):
        old_markers = _call_marker_set(_call_change_text(change))
        if not old_markers:
            return False
        return any(
            other.get("status") == "added"
            and other.get("item", "").startswith("call:add_constraint:")
            and bool(old_markers & _call_marker_set(_call_change_text(other)))
            for other in semantic_changes
        )

    if item.startswith("call:add_objective:"):
        objective_requested = bool(
            re.search(
                r"\b(objective|minimi[sz]e|maximi[sz]e|weighted|drag|CD|fuel\s*burn|fuelburn|mass)\b",
                user_prompt,
                re.IGNORECASE,
            )
        )
        if not objective_requested:
            return False
        old_markers = _call_marker_set(_call_change_text(change))
        return any(
            other.get("status") == "added"
            and other.get("item", "").startswith("call:add_objective:")
            and (
                not old_markers
                or bool(old_markers & _call_marker_set(_call_change_text(other)))
                or ("cd" in old_markers and "weighted_cd" in _call_marker_set(_call_change_text(other)))
            )
            for other in semantic_changes
        )

    return False


def _contract_violations(
    user_prompt: str, semantic_changes: list[dict[str, str]]
) -> list[dict[str, str]]:
    violations = []
    preserve_point1 = _prompt_preserves_point1(user_prompt)
    removal_requested = _prompt_requests_removal(user_prompt)
    scaling_requested = _prompt_requests_scaling(user_prompt)
    mesh_resolution_requested = _prompt_requests_mesh_resolution(user_prompt)
    twist_requested = _prompt_requests_twist(user_prompt)
    upward_load_requested = _prompt_requests_upward_load(user_prompt)

    for change in semantic_changes:
        item = change.get("item", "")

        if preserve_point1 and item in _POINT1_FLIGHT_ITEMS:
            element = _changed_index(change, 1)
            if element:
                violations.append(
                    {
                        "severity": "blocking",
                        "changed_item": f"{item} index 1",
                        "blueprint_value": element.get("blueprint", ""),
                        "generated_value": element.get("generated", ""),
                        "reason": (
                            "The user explicitly asked to preserve the maneuver/secondary "
                            "point, but index 1 changed."
                        ),
                        "repair_instruction": (
                            "Restore index 1 to the blueprint value. Only update the "
                            "explicitly requested point/index."
                        ),
                    }
                )

        if (
            change.get("status") == "changed"
            and item.startswith(
                (
                    "call:add_design_var:",
                    "call:add_objective:",
                    "call:add_constraint:",
                )
            )
            and _call_scaling_changed(change)
            and not scaling_requested
        ):
            violations.append(
                {
                    "severity": "blocking",
                    "changed_item": item,
                    "blueprint_value": change.get("blueprint", ""),
                    "generated_value": change.get("generated", ""),
                    "reason": "Existing optimizer scaling changed without an explicit scaling request.",
                    "repair_instruction": (
                        "Restore the blueprint ref/scaler value. Only change existing "
                        "scaling for an explicit scaling request or a scaling-specific runtime repair."
                    ),
                }
            )

        if (
            change.get("status") == "removed"
            and item.startswith(("call:add_constraint:", "call:add_objective:"))
            and not removal_requested
            and not _call_replacement_exists(change, semantic_changes, user_prompt)
        ):
            violations.append(
                {
                    "severity": "blocking",
                    "changed_item": item,
                    "blueprint_value": change.get("blueprint", ""),
                    "generated_value": "removed",
                    "reason": "The blueprint setup was removed without an explicit user removal request.",
                    "repair_instruction": "Restore the blueprint line unless the user explicitly asked to remove it.",
                }
            )

        if item.startswith(("dict:surf_dict.", "dict:surface.")):
            key = _dict_key_from_item(item)
            if key in _AERO_SOLVER_KEYS and not _prompt_requests_aero_solver_key(
                user_prompt, key
            ):
                violations.append(
                    {
                        "severity": "blocking",
                        "changed_item": item,
                        "blueprint_value": change.get("blueprint", ""),
                        "generated_value": change.get("generated", ""),
                        "reason": (
                            f"The user did not request changing aerodynamic solver assumption {key}."
                        ),
                        "repair_instruction": (
                            f"Restore the blueprint {key} value. Do not infer {key} "
                            "from Mach, altitude, or neighboring aero settings."
                        ),
                    }
                )
            elif (
                key == "twist_cp"
                and change.get("status") in {"added", "changed"}
                and not twist_requested
            ):
                violations.append(
                    {
                        "severity": "blocking",
                        "changed_item": item,
                        "blueprint_value": change.get("blueprint", ""),
                        "generated_value": change.get("generated", ""),
                        "reason": "The user did not request twist, so twist_cp should not be introduced or changed.",
                        "repair_instruction": (
                            "Remove twist_cp for rectangular analysis wings unless the "
                            "user explicitly requests twist. Preserve CRM/uCRM twist only "
                            "when it comes from the blueprint mesh generator."
                        ),
                    }
                )
            elif (
                key in _INITIAL_DICT_KEYS
                and change.get("status") != "removed"
                and not _var_initial_requested(user_prompt, key)
                and not _initial_change_moves_inside_requested_bounds(
                    user_prompt, change, semantic_changes, key
                )
                and not _control_point_count_change_only(user_prompt, change, key)
            ):
                violations.append(
                    {
                        "severity": "blocking",
                        "changed_item": item,
                        "blueprint_value": change.get("blueprint", ""),
                        "generated_value": change.get("generated", ""),
                        "reason": (
                            f"The initial {key} values changed without an explicit "
                            "initial-value request and the change was not required to "
                            "place the starting point inside newly requested bounds."
                        ),
                        "repair_instruction": (
                            f"Restore the blueprint {key} initial values when they remain "
                            "inside the requested bounds. If requested bounds exclude the "
                            "blueprint values, use sensible interior initial values."
                        ),
                    }
                )

        if item.startswith("dict:mesh_dict."):
            key = _dict_key_from_item(item)
            if (
                key in {"num_y", "num_x"}
                and change.get("status") == "changed"
                and not mesh_resolution_requested
            ):
                violations.append(
                    {
                        "severity": "blocking",
                        "changed_item": item,
                        "blueprint_value": change.get("blueprint", ""),
                        "generated_value": change.get("generated", ""),
                        "reason": (
                            f"The user did not request changing mesh resolution key {key}."
                        ),
                        "repair_instruction": (
                            f"Restore the blueprint {key} value unless the user explicitly "
                            "asks for mesh resolution, panel count, or discretization changes."
                        ),
                    }
                )

        if (
            upward_load_requested
            and (
                item.startswith("call:add_output:loads")
                or item
                in {
                    "assign:loads",
                    "assign:loads_array",
                    "assign:forces",
                    "assign:forces_val",
                    "assign:forces_array",
                }
            )
            and _load_change_fills_all_components(change)
        ):
            violations.append(
                {
                    "severity": "blocking",
                    "changed_item": item,
                    "blueprint_value": change.get("blueprint", ""),
                    "generated_value": change.get("generated", ""),
                    "reason": (
                        "The user requested an upward/vertical load, but the generated "
                        "loads expression fills all six force/moment components."
                    ),
                    "repair_instruction": (
                        "Use a zero loads array and assign only the vertical force "
                        "component, e.g. loads[:, 2] = load_per_node."
                    ),
                }
            )

    return violations


def _invalid_contract_report(
    report: dict, violations: list[dict[str, str]], reason: str
) -> dict:
    fixed = dict(report)
    fixed["passed"] = False
    fixed["violations"] = violations
    fixed["wrapper_contract_violation"] = True
    fixed["invalid_audit_reasons"] = [reason]
    return fixed


def _per_change_pass_violations(report: dict) -> list[dict[str, str]]:
    violations = []
    for idx, reviewed in enumerate(report.get("reviewed_changes") or [], start=1):
        if "passed" not in reviewed:
            reviewed["passed"] = True
            report.setdefault("warnings", []).append(
                "Auditor omitted per-change passed on reviewed_changes; "
                f"wrapper inferred passed=true for reviewed_change {idx}."
            )
        if reviewed.get("passed") is False:
            violations.append(
                {
                    "severity": "blocking",
                    "changed_item": reviewed.get("changed_item", f"reviewed_change {idx}"),
                    "blueprint_value": reviewed.get("blueprint_value", ""),
                    "generated_value": reviewed.get("generated_value", ""),
                    "reason": reviewed.get("reason", "This per-change audit check failed."),
                    "repair_instruction": reviewed.get(
                        "repair_instruction",
                        "Repair this changed item or restore the blueprint value.",
                    ),
                }
            )

    for idx, violation in enumerate(report.get("violations") or [], start=1):
        if "passed" not in violation:
            violation["passed"] = False
        elif violation.get("passed") is not False:
            violations.append(
                {
                    "severity": "blocking",
                    "changed_item": violation.get("changed_item", f"violation {idx}"),
                    "blueprint_value": violation.get("blueprint_value", ""),
                    "generated_value": violation.get("generated_value", ""),
                    "reason": "A blocking violation cannot have passed=true.",
                    "repair_instruction": "Return passed=false for blocking violations.",
                }
            )
    return violations


def _invalid_audit_failed_report(report: dict) -> dict:
    fixed = dict(report)
    fixed["passed"] = False
    fixed["audit_infrastructure_error"] = True
    fixed["violations"] = [
        {
            "passed": False,
            "severity": "blocking",
            "changed_item": "blueprint auditor",
            "blueprint_value": "valid blueprint consistency audit",
            "generated_value": "invalid or contradictory audit response",
            "reason": "The auditor returned an invalid audit after retry, so the code was not safely reviewed.",
            "repair_instruction": "Do not ask the coder to repair this. Retry or inspect the auditor prompt/schema.",
        }
    ]
    fixed["feedback_for_coder"] = ""
    return fixed


def _validate_audit_report(
    report: dict,
    diff_text: str,
    semantic_changes: list[dict[str, str]] | None = None,
    user_prompt: str = "",
) -> dict:
    """Reject impossible auditor failures without judging engineering intent."""
    semantic_changes = semantic_changes or []
    per_change_violations = _per_change_pass_violations(report)
    if per_change_violations:
        return _invalid_contract_report(
            report,
            per_change_violations,
            "audit omitted or contradicted required per-change passed=true/false decisions",
        )

    if report.get("passed", True):
        contract_violations = _contract_violations(user_prompt, semantic_changes)
        if contract_violations:
            return _invalid_contract_report(
                report,
                contract_violations,
                "passed=true contradicted explicit preserve/removal/initial-value/scaling contract",
            )
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
        fixed["unreviewed_semantic_changes"] = missing[:20]
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
    for section in ("violations", "reviewed_changes"):
        for item in data[section]:
            if isinstance(item.get("passed"), str):
                item["passed"] = item["passed"].strip().lower() == "true"
    data["warnings"] = data.get("warnings") or []
    data["feedback_for_coder"] = data.get("feedback_for_coder", "")
    return data


def _audit_retry_message(user_message: str, report: dict) -> str:
    return (
        f"{user_message}\n\n"
        "### PREVIOUS AUDIT WAS INVALID ###\n"
        "Re-audit the same generated code. Do not request code changes just because "
        "your previous audit omitted review details. Return a per-change "
        "passed=true/false decision for every reviewed change. The top-level "
        "passed value must be true only if every per-change decision passed. "
        "For pointwise arrays, name each changed index/point. For protected "
        "dict keys, name each exact dict item.\n\n"
        f"Invalid audit reasons:\n{json.dumps(report.get('invalid_audit_reasons', []), indent=2)}\n\n"
        f"Unreviewed semantic changes, if any:\n"
        f"{json.dumps(report.get('unreviewed_semantic_changes', []), indent=2)}\n\n"
        "Return the corrected audit object."
    )


def _malformed_audit_retry_message(user_message: str, error: Exception) -> str:
    return (
        f"{user_message}\n\n"
        "### PREVIOUS AUDIT RESPONSE WAS MALFORMED ###\n"
        f"The previous auditor response could not be parsed as the required JSON object: {error}\n\n"
        "Return exactly one <audit> XML section containing valid JSON. Do not include "
        "Markdown fences, comments, trailing commas, or prose outside the XML tags."
    )


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
    in_t = estimate_tokens(system_prompt + "\n" + user_message)

    try:
        response = get_llm_response(
            user_message, model_name, system_prompt, provider=provider
        )
        out_t = estimate_tokens(response)
        retry_used = False

        try:
            report = _parse_audit_response(response)
        except Exception as parse_exc:
            logger.warning("Blueprint auditor returned malformed output; retrying auditor once.")
            retry_used = True
            retry_message = _malformed_audit_retry_message(user_message, parse_exc)
            retry_response = get_llm_response(
                retry_message, model_name, system_prompt, provider=provider
            )
            in_t += estimate_tokens(system_prompt + "\n" + retry_message)
            out_t += estimate_tokens(retry_response)
            try:
                report = _parse_audit_response(retry_response)
                report["auditor_retry"] = True
            except Exception as retry_parse_exc:
                report = _invalid_audit_failed_report(
                    {
                        "invalid_audit_reasons": [
                            f"auditor response malformed after retry: {retry_parse_exc}"
                        ],
                        "auditor_retry": True,
                    }
                )

        report = _validate_audit_report(
            report, full_diff, semantic_changes, user_prompt=user_prompt
        )
        if report.get("invalid_audit"):
            if retry_used:
                report = _invalid_audit_failed_report(report)
                report["auditor_retry"] = True
            else:
                retry_used = True
                logger.warning("Blueprint auditor returned invalid audit; retrying auditor once.")
                retry_message = _audit_retry_message(user_message, report)
                retry_response = get_llm_response(
                    retry_message, model_name, system_prompt, provider=provider
                )
                in_t += estimate_tokens(system_prompt + "\n" + retry_message)
                out_t += estimate_tokens(retry_response)
                try:
                    retry_report = _parse_audit_response(retry_response)
                    report = _validate_audit_report(
                        retry_report, full_diff, semantic_changes, user_prompt=user_prompt
                    )
                    report["auditor_retry"] = True
                except Exception as retry_exc:
                    report = _invalid_audit_failed_report(
                        {
                            "invalid_audit_reasons": [
                                f"auditor retry could not be parsed: {retry_exc}"
                            ],
                            "auditor_retry": True,
                        }
                    )
                if report.get("invalid_audit"):
                    report = _invalid_audit_failed_report(report)
                    report["auditor_retry"] = True
        if not report.get("passed", True):
            report["feedback_for_coder"] = _format_repair_feedback(report)
        report = _attach_audit_artifacts(report, full_diff, semantic_changes)
        return report, in_t, out_t
    except LLMBackendTransientError:
        raise
    except Exception as exc:
        logger.error(f"Blueprint auditor failed closed: {exc}")
        report = {
            "passed": False,
            "audit_infrastructure_error": True,
            "invalid_audit_reasons": [str(exc)],
            "violations": [
                {
                    "passed": False,
                    "severity": "blocking",
                    "changed_item": "blueprint audit",
                    "blueprint_value": "valid blueprint consistency audit",
                    "generated_value": f"audit failure: {exc}",
                    "reason": "The generated code could not be safely audited.",
                    "repair_instruction": "Retry generation and audit before execution.",
                }
            ],
            "feedback_for_coder": "",
            "warning": f"Blueprint auditor failed closed: {exc}",
            "diff": full_diff,
        }
        return {
            **report,
            "diff": full_diff,
        }, in_t, 0
