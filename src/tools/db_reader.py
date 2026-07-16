import json
import sqlite3
import sys
import zlib

try:
    import openmdao.api as om
except ImportError:
    om = None


def _decompress_json(blob):
    """Decode OpenMDAO's compressed JSON metadata blobs."""
    if blob is None:
        return {}
    try:
        return json.loads(zlib.decompress(blob))
    except Exception:
        return {}


def _clean_val(val, max_len=60) -> str:
    """
    Collapses multi-line NumPy arrays into a single clean line
    and truncates long outputs to prevent Markdown table breaks.
    """
    if val is None:
        return "n/a"
    # Convert to string, replace newlines with spaces, and collapse multiple spaces
    s = str(val).replace("\n", " ")
    s = " ".join(s.split())

    # Truncate if too long
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _json_val(val):
    """Convert NumPy/OpenMDAO values into compact JSON-serializable values."""
    if val is None:
        return None
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, tuple):
        val = list(val)
    if isinstance(val, list):
        return [_json_val(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _json_val(v) for k, v in val.items()}
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


def _store_metric(metrics, group, name, init_val, final_val):
    metrics[group][name] = {
        "initial": _json_val(init_val),
        "final": _json_val(final_val),
    }


def _render_summary(objectives, constraints, desvars) -> str:
    """Build the markdown summary tables."""
    markdown_output = "## Optimization Summary\n\n"

    markdown_output += "### Objectives\n"
    markdown_output += "| Objective | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for name, init_val, final_val in objectives:
        markdown_output += (
            f"| `{name}` | {_clean_val(init_val)} | {_clean_val(final_val)} |\n"
        )

    markdown_output += "\n### Constraints\n"
    markdown_output += "| Constraint | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for name, init_val, final_val in constraints:
        markdown_output += (
            f"| `{name}` | {_clean_val(init_val)} | {_clean_val(final_val)} |\n"
        )

    markdown_output += "\n### Design Variables (Truncated)\n"
    markdown_output += "| Design Variable | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for name, init_val, final_val in desvars:
        markdown_output += (
            f"| `{name}` | {_clean_val(init_val)} | {_clean_val(final_val)} |\n"
        )

    return markdown_output


def _summarize_with_openmdao(db_path):
    """Use the OpenMDAO CaseReader when the local install is healthy."""
    cr = om.CaseReader(db_path)
    driver_cases = list(cr.get_cases("driver"))
    if len(driver_cases) == 0:
        return None

    initial_case = driver_cases[0]
    final_case = driver_cases[-1]

    objectives = []
    for name in initial_case.get_objectives().keys():
        objectives.append((name, initial_case.get_val(name), final_case.get_val(name)))

    constraints = []
    for name in initial_case.get_constraints().keys():
        constraints.append((name, initial_case.get_val(name), final_case.get_val(name)))

    desvars = []
    for name in initial_case.get_design_vars().keys():
        desvars.append((name, initial_case.get_val(name), final_case.get_val(name)))

    return _render_summary(objectives, constraints, desvars)


def _metrics_with_openmdao(db_path):
    """Use the OpenMDAO CaseReader to extract structured benchmark metrics."""
    cr = om.CaseReader(db_path)
    driver_cases = list(cr.get_cases("driver"))
    if len(driver_cases) == 0:
        return None

    initial_case = driver_cases[0]
    final_case = driver_cases[-1]
    metrics = {"objectives": {}, "constraints": {}, "design_vars": {}}

    for name in initial_case.get_objectives().keys():
        _store_metric(
            metrics,
            "objectives",
            name,
            initial_case.get_val(name),
            final_case.get_val(name),
        )

    for name in initial_case.get_constraints().keys():
        _store_metric(
            metrics,
            "constraints",
            name,
            initial_case.get_val(name),
            final_case.get_val(name),
        )

    for name in initial_case.get_design_vars().keys():
        _store_metric(
            metrics,
            "design_vars",
            name,
            initial_case.get_val(name),
            final_case.get_val(name),
        )

    return metrics


def _resolve_case_value(case_inputs, case_outputs, prom_name, setting, prom2abs):
    """Resolve a promoted variable name against the raw recorded case payload."""
    candidates = []

    if prom_name in case_outputs or prom_name in case_inputs:
        candidates.append(prom_name)

    source = setting.get("source")
    if source:
        candidates.append(source)

    candidates.extend(prom2abs.get("output", {}).get(prom_name, []) or [])
    candidates.extend(prom2abs.get("input", {}).get(prom_name, []) or [])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in case_outputs:
            return case_outputs[candidate]
        if candidate in case_inputs:
            return case_inputs[candidate]

    return None


def _summarize_with_sqlite(db_path):
    """Fallback reader that parses the OpenMDAO SQLite database directly."""
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        metadata = cur.execute("SELECT prom2abs, var_settings FROM metadata").fetchone()
        if metadata is None:
            return None

        prom2abs = _decompress_json(metadata[0])
        var_settings = _decompress_json(metadata[1])

        rows = cur.execute(
            """
            SELECT inputs, outputs
            FROM driver_iterations
            WHERE success = 1
            ORDER BY id ASC
            """
        ).fetchall()

    if not rows:
        return None

    first_inputs = json.loads(rows[0][0] or "{}")
    first_outputs = json.loads(rows[0][1] or "{}")
    last_inputs = json.loads(rows[-1][0] or "{}")
    last_outputs = json.loads(rows[-1][1] or "{}")

    objectives = []
    constraints = []
    desvars = []

    for name, setting in var_settings.items():
        if name == "execution_order" or not isinstance(setting, dict):
            continue
        var_type = setting.get("type")

        init_val = _resolve_case_value(
            first_inputs, first_outputs, name, setting, prom2abs
        )
        final_val = _resolve_case_value(
            last_inputs, last_outputs, name, setting, prom2abs
        )
        if init_val is None and final_val is None:
            continue

        item = (name, init_val, final_val)
        if var_type == "obj":
            objectives.append(item)
        elif var_type == "con":
            constraints.append(item)
        else:
            desvars.append(item)

    if not (objectives or constraints or desvars):
        return None

    return _render_summary(objectives, constraints, desvars)


def _metrics_with_sqlite(db_path):
    """Fallback structured metrics reader for OpenMDAO SQLite databases."""
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        metadata = cur.execute("SELECT prom2abs, var_settings FROM metadata").fetchone()
        if metadata is None:
            return None

        prom2abs = _decompress_json(metadata[0])
        var_settings = _decompress_json(metadata[1])

        rows = cur.execute(
            """
            SELECT inputs, outputs
            FROM driver_iterations
            WHERE success = 1
            ORDER BY id ASC
            """
        ).fetchall()

    if not rows:
        return None

    first_inputs = json.loads(rows[0][0] or "{}")
    first_outputs = json.loads(rows[0][1] or "{}")
    last_inputs = json.loads(rows[-1][0] or "{}")
    last_outputs = json.loads(rows[-1][1] or "{}")
    metrics = {"objectives": {}, "constraints": {}, "design_vars": {}}

    for name, setting in var_settings.items():
        if name == "execution_order" or not isinstance(setting, dict):
            continue
        var_type = setting.get("type")

        init_val = _resolve_case_value(
            first_inputs, first_outputs, name, setting, prom2abs
        )
        final_val = _resolve_case_value(
            last_inputs, last_outputs, name, setting, prom2abs
        )
        if init_val is None and final_val is None:
            continue

        if var_type == "obj":
            group = "objectives"
        elif var_type == "con":
            group = "constraints"
        else:
            group = "design_vars"
        _store_metric(metrics, group, name, init_val, final_val)

    if not any(metrics[group] for group in metrics):
        return None
    return metrics


def extract_optimization_metrics(db_path="aero.db"):
    """
    Return JSON-serializable initial/final objective, constraint, and DV values.
    """
    try:
        if om is not None:
            metrics = _metrics_with_openmdao(db_path)
            if metrics:
                return metrics
    except Exception:
        pass

    try:
        return _metrics_with_sqlite(db_path) or {}
    except Exception:
        return {}


def summarize_optimization(db_path="aero.db"):
    """
    Parses OpenMDAO case database to extract initial and final states
    of design variables, constraints, and objectives.
    Returns a clean Markdown table summarizing these.
    """
    try:
        if om is not None:
            summary = _summarize_with_openmdao(db_path)
            if summary:
                return summary
    except Exception as e:
        openmdao_error = str(e)
    else:
        openmdao_error = None

    try:
        summary = _summarize_with_sqlite(db_path)
        if summary:
            return summary
    except Exception as e:
        sqlite_error = str(e)
    else:
        sqlite_error = None

    errors = []
    if openmdao_error:
        errors.append(f"OpenMDAO reader failed: {openmdao_error}")
    if sqlite_error:
        errors.append(f"SQLite fallback failed: {sqlite_error}")

    if errors:
        return f"Error reading database {db_path}: " + " | ".join(errors)
    return "No driver cases found in the database. Did the optimization run?"


if __name__ == "__main__":
    db_path = "aero.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    print(summarize_optimization(db_path))
