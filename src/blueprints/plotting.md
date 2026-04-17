# Plotting Guidance for the Agent

## Overview
All plotting should be directed to the unified output directory. Use `os.path.join` for all paths to ensure cross-platform compatibility on both Windows and macOS/Linux:
```python
import os
output_dir = os.path.join("src", "openaerostruct_out", "agent_plots")
os.makedirs(output_dir, exist_ok=True)
```
Database recorders should save to `os.path.join("src", "openaerostruct_out", <db_name>)`.

## Key Plotting Scripts
1.  **`src/blueprints/agent_plotting.py`**: The primary automated plotting tool. Use this after running an optimization.
    - Usage: `python src/blueprints/agent_plotting.py src/openaerostruct_out/aero.db src/openaerostruct_out/agent_plots`
    - Generates: `wing_3d.png` (3D wireframe of the optimised geometry) and `lift_dist.png` (spanwise lift distribution).

2.  **`src/blueprints/plot_wing.py`**: A detailed GUI-based plotter that can also save static images.
    - Usage: `python src/blueprints/plot_wing.py src/openaerostruct_out/aero.db`
    - Output: Saves `optimized_wing.png` to `src/openaerostruct_out/agent_plots/`.

## Available Plot Types
The following plot types are supported by writing custom plotting code at the end of a generated script (see patterns below).

### 1. Aerodynamic Polars (from `aero_analysis.py` sweep data)
After running an alpha/Mach sweep and saving results to a pandas DataFrame `df`, generate:
- **L/D vs Alpha** (`LD_vs_Alpha.png`)
- **Drag Polar – CL vs CD** (`Drag_Polars.png`)
- **CL vs Alpha** (`CL_vs_Alpha.png`)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

df["Mach"] = df["Mach"].astype(str)
for mach, group in df.groupby("Mach"):
    plt.plot(group["Alpha"], group["L/D"], marker="o", label=f"M={mach}")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("L/D")
plt.title("Lift-to-Drag Ratio vs Angle of Attack")
plt.legend()
plt.savefig(os.path.join(output_dir, "LD_vs_Alpha.png"), bbox_inches="tight")
plt.close()
```

### 2. Spanwise Lift Distribution
After running the model at a trim condition, extract and plot sectional lift:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

mesh_out = prob.get_val("flight_condition_0.wing.def_mesh", units="m")
y_vertices = mesh_out[0, :, 1]
y_center = 0.5 * (y_vertices[:-1] + y_vertices[1:])
Cl = np.array(prob.get_val("flight_condition_0.wing_perf.Cl"))

# Mirror for full span (symmetry=True)
y_full = np.concatenate((-y_center[::-1], y_center))
Cl_full = np.concatenate((Cl[::-1], Cl))

# Elliptic reference
CL_total = prob.get_val("flight_condition_0.wing_perf.CL")[0]
Sref = prob.get_val("flight_condition_0.wing_perf.S_ref", units="m**2")[0]
semi_span = mesh_out[0, -1, 1] - mesh_out[0, 0, 1]
cl_max = 2 * CL_total * Sref / (np.pi * semi_span)
y_ell = np.linspace(-semi_span, semi_span, 200)
Cl_ell = cl_max * np.sqrt(np.maximum(0, 1 - (y_ell / semi_span) ** 2))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(y_full, Cl_full, label="VLM Cl", marker="o", markersize=3)
ax.plot(y_ell, Cl_ell, "r--", label="Elliptic reference")
ax.set_xlabel("Span (m)")
ax.set_ylabel("Sectional Lift Coefficient (Cl)")
ax.set_title("Spanwise Lift Distribution")
ax.legend()
ax.grid(True)
plt.savefig(os.path.join(output_dir, "Sectional_Lift_Distribution.png"), bbox_inches="tight")
plt.close()
```

### 3. Multi-Section Planform View
After a `multi_section_aero.py` run, plot the unified mesh planform:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

meshUni = prob.get_val(name + "." + unification_name + "." + name + "_uni_mesh")
mesh_x = meshUni[:, :, 0]
mesh_y = meshUni[:, :, 1]

fig, ax = plt.subplots(figsize=(8, 4))
for i in range(mesh_x.shape[0]):
    ax.plot(mesh_y[i, :], mesh_x[i, :], "k", lw=1)
    ax.plot(-mesh_y[i, :], mesh_x[i, :], "k", lw=1)
for j in range(mesh_x.shape[1]):
    ax.plot(mesh_y[:, j], mesh_x[:, j], "k", lw=1)
    ax.plot(-mesh_y[:, j], mesh_x[:, j], "k", lw=1)
ax.set_aspect("equal")
ax.set_xlabel("Span (m)")
ax.set_ylabel("Chord (m)")
ax.set_title("Optimized Multi-Section Planform")
plt.savefig(os.path.join(output_dir, "multi_section_planform.png"), bbox_inches="tight")
plt.close()
```

### 4. 3D Wing Mesh
Plot a 3D wireframe of the optimised geometry from a recorded database:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from openmdao.recorders.sqlite_reader import SqliteCaseReader

cr = SqliteCaseReader(os.path.join("src", "openaerostruct_out", "aero.db"), pre_load=True)
last_case = next(reversed(cr.get_cases("driver")))
mesh = last_case.outputs["wing.mesh"]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2], color="black", alpha=0.7)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Optimized Wing Geometry")
plt.savefig(os.path.join(output_dir, "wing_3d.png"), bbox_inches="tight")
plt.close()
```

## Custom Plotting (Flexible Analysis)
If the user requests specific visualizations (e.g., "plot drag vs thickness"), **PROACTIVELY** add custom plotting code to the end of your scripts.

### Standard Plotting Pattern
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np

output_dir = os.path.join("src", "openaerostruct_out", "agent_plots")
os.makedirs(output_dir, exist_ok=True)

# Extract data (example)
x_data = np.array(prob.get_val("wing.twist_cp"))
y_data = np.array(prob.get_val("flight_condition_0.wing_perf.CD"))

fig, ax = plt.subplots()
ax.plot(x_data, y_data, marker="o")
ax.set_xlabel("Twist (deg)")
ax.set_ylabel("CD")
ax.set_title("Custom Visualization")
plt.savefig(os.path.join(output_dir, "custom_plot.png"), bbox_inches="tight")
plt.close()
```

## Strict Plotting Rules
- **Matplotlib**: Use `marker='o'`, NOT `markers='o'`.
- **Arrays**: Always convert OpenMDAO outputs to NumPy arrays before plotting: `np.array(prob.get_val(...))`.
- **Headless**: Always use `matplotlib.use('Agg')` before importing `pyplot`, or call `plt.switch_backend('Agg')`.
- **Cleanliness**: Use `plt.close()` after saving to free memory.
- **Paths**: Always use `os.path.join(...)` — never hardcode forward-slash paths like `"src/openaerostruct_out/..."`.

## Common Pitfalls & Bug Fixes
- **Error**: `TypeError: 'markers' is an invalid keyword argument`
  - **Fix**: Change `markers=` to `marker=`.
- **Error**: `ValueError: x and y must have same first dimension`
  - **Fix**: Check `symmetry`. If `symmetry=True`, mirror your data to match the full-span mesh (see Spanwise Lift Distribution pattern above).
- **Error**: Plotting takes too long or crashes.
  - **Fix**: Wrap plotting in a `try...except` block to ensure the analysis completes even if visualization fails.

### Robust Plotting Pattern
```python
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # ... plotting logic ...
    plt.savefig(os.path.join(output_dir, "plot.png"), bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Warning: Plotting failed with error: {e}")
```

## Best Practices
- **Always Display Plots**: after a successful run, search `src/openaerostruct_out/agent_plots/` for new images and display them in the chat using Markdown syntax: `![plot description](path/to/plot.png)`.
- **Headless Mode**: Ensure all plotting scripts use `matplotlib.use('Agg')` before `import matplotlib.pyplot`.
- **Verification**: If a plot fails to generate, check if the `.db` file exists in `src/openaerostruct_out/`.
- **No inline plots in blueprints**: Blueprint scripts do not contain plotting code. Add plotting at the end of the generated script, after the main analysis/optimisation run.
