# Plotting Guidance for the Agent

## Overview
All plotting should be directed to the unified output directory: `src/openaerostruct_out/agent_plots/`.
Database recorders should save to `src/openaerostruct_out/`.

## Key Plotting Scripts
1.  **`src/blueprints/agent_plotting.py`**: The primary automated plotting tool. Use this after running an optimization.
    - Usage: `python src/blueprints/agent_plotting.py src/openaerostruct_out/aero.db src/openaerostruct_out/agent_plots`
    - Generates: `wing_3d.png` and `lift_dist.png`.

2.  **`src/blueprints/plot_wing.py`**: A detailed GUI-based plotter that can also save static images.
    - Usage: `python src/blueprints/plot_wing.py src/openaerostruct_out/aero.db`
    - Output: Saves `optimized_wing.png` to `src/openaerostruct_out/agent_plots/`.

## Custom Plotting (Flexible Analysis)
If the user requests specific visualizations (e.g., "plot drag vs thickness"), **PROACTIVELY** add custom plotting code to the end of your scripts.

### Standard Plotting Pattern
```python
import matplotlib.pyplot as plt
import os

# 1. Set headless mode
plt.style.use('ggplot') # or niceplots if available
plt.switch_backend('Agg') 

# 2. Extract data (example)
x_data = prob.get_val('wing.twist_cp')
y_data = prob.get_val('flight_condition_0.wing_perf.CD')

# 3. Create plot
fig, ax = plt.subplots()
ax.plot(x_data, y_data, marker='o')
ax.set_xlabel('Twist (deg)')
ax.set_ylabel('CD')
ax.set_title('Custom Visualization')

# 4. Save to unified directory
output_dir = "src/openaerostruct_out/agent_plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "custom_plot.png"))
plt.close()
```

## Strict Plotting Rules
- **Matplotlib**: Use `marker='o'`, NOT `markers='o'`.
- **Arrays**: Always convert OpenMDAO outputs to NumPy arrays before plotting: `np.array(prob.get_val(...))`.
- **Headless**: Always use `plt.switch_backend('Agg')`.
- **Cleanliness**: Use `plt.close()` after saving to free memory.

## Common Pitfalls & Bug Fixes
- **Error**: `TypeError: 'markers' is an invalid keyword argument`
  - **Fix**: Change `markers=` to `marker=`.
- **Error**: `ValueError: x and y must have same first dimension`
  - **Fix**: Check `symmetry`. If `symmetry=True`, you may need to mirror your data to match the Full Wing mesh.
- **Error**: Plotting takes too long or crashes.
  - **Fix**: Wrap plotting in a `try...except` block to ensure the analysis completes even if visualization fails.

### Robust Plotting Pattern
```python
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    # ... plotting logic ...
    plt.savefig(os.path.join(output_dir, "plot.png"))
    plt.close()
except Exception as e:
    print(f"Warning: Plotting failed with error: {e}")
```

## Best Practices
- **Always Display Plots**: after a successful run, search `src/openaerostruct_out/agent_plots/` for new images and display them in the chat using Markdown syntax: `![plot description](path/to/plot.png)`.
- **Headless Mode**: Ensure all plotting scripts use `plt.switch_backend('Agg')` or `matplotlib.use('Agg')`.
- **Verification**: If a plot fails to generate, check if the `.db` file exists in `src/openaerostruct_out/`.
