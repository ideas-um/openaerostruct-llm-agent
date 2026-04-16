import sys

try:
    import openmdao.api as om
except ImportError:
    om = None

def summarize_optimization(db_path='aero.db'):
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
        
    driver_cases = cr.get_cases('driver')
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
        init_val = initial_case.get_val(k)
        final_val = final_case.get_val(k)
        markdown_output += f"| `{k}` | {init_val} | {final_val} |\n"
        
    # Constraints
    markdown_output += "\n### Constraints\n"
    markdown_output += "| Constraint | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for k in constraints.keys():
        init_val = initial_case.get_val(k)
        final_val = final_case.get_val(k)
        markdown_output += f"| `{k}` | {init_val} | {final_val} |\n"
        
    # Design Variables
    markdown_output += "\n### Design Variables (Truncated)\n"
    markdown_output += "| Design Variable | Initial Value | Final Value |\n"
    markdown_output += "|---|---|---|\n"
    for k in desvars.keys():
        init_val = initial_case.get_val(k)
        final_val = final_case.get_val(k)
        
        # Convert to string and truncate for arrays
        inv = str(init_val).replace('\n', ' ')
        fnv = str(final_val).replace('\n', ' ')
        if len(inv) > 60: inv = inv[:57] + '...'
        if len(fnv) > 60: fnv = fnv[:57] + '...'
        
        markdown_output += f"| `{k}` | {inv} | {fnv} |\n"
        
    return markdown_output

if __name__ == '__main__':
    db_path = 'aero.db'
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print(summarize_optimization(db_path))
