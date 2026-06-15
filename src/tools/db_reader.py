import sys

try:
    import openmdao.api as om
except ImportError:
    om = None


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
        s = s[:max_len - 3] + "..."
    return s


def summarize_optimization(db_path="aero.db"):
    """
    Parses OpenMDAO case database to extract initial and final states
    of design variables, constraints, and objectives.
    Returns a clean Markdown table summarizing these.
    """
    if om is None:
        return "Error: openmdao package is not installed."

    try:
        cr = om.CaseReader(db_path)
    except Exception as e:
        return f"Error reading database {db_path}: {e}"

    driver_cases = cr.get_cases("driver")
    if len(driver_cases) == 0:
        return "No driver cases found in the database. Did the optimization run?"

    initial_case = driver_cases[0]
    final_case = driver_cases[-1]

    # Extract keys
    desvars = initial_case.get_design_vars()
    objectives = initial_case.get_objectives()
    constraints = initial_case.get_constraints()

    markdown_output = "## Optimization Summary\n\n"

    # Objectives
    markdown_output += "### Objectives\n"
    markdown_output += "| Objective | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for k in objectives.keys():
        init_val = _clean_val(initial_case.get_val(k))
        final_val = _clean_val(final_case.get_val(k))
        markdown_output += f"| `{k}` | {init_val} | {final_val} |\n"

    # Constraints
    markdown_output += "\n### Constraints\n"
    markdown_output += "| Constraint | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for k in constraints.keys():
        init_val = _clean_val(initial_case.get_val(k))
        final_val = _clean_val(final_case.get_val(k))
        markdown_output += f"| `{k}` | {init_val} | {final_val} |\n"

    # Design Variables
    markdown_output += "\n### Design Variables (Truncated)\n"
    markdown_output += "| Design Variable | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for k in desvars.keys():
        init_val = _clean_val(initial_case.get_val(k))
        final_val = _clean_val(final_case.get_val(k))
        markdown_output += f"| `{k}` | {init_val} | {final_val} |\n"

    return markdown_output


if __name__ == "__main__":
    db_path = "aero.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    print(summarize_optimization(db_path))