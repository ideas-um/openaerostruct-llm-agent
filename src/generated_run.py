import numpy as np
import openmdao.api as om
import os
import matplotlib
matplotlib.use('Agg')

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.integration.multipoint_comps import MultiCD

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# High Aspect Ratio UAV: Large span, small chord
mesh_dict = {
    "num_y": 7,
    "num_x": 5,
    "wing_type": "CRM",
    "symmetry": True,
    "span": 15.0,
    "root_chord": 0.8,
    "num_twist_cp": 5,
    "span_cos_spacing": 1.0,
}

mesh, twist_cp = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "fem_model_type": "tube",
    "mesh": mesh,
    "twist_cp": twist_cp,
    "CL0": 0.0,
    "CD0": 0.01,
    "k_lam": 0.03,
    "t_over_c_cp": np.array([0.12]), # Initial guess
    "c_max_t": 0.3,
    "with_viscous": True,
    "with_wave": False,
}

surfaces = [surf_dict]
n_points = 2

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

# To avoid the shape mismatch error, flight condition parameters (v, Mach, re, rho)
# must be scalars when connected to AeroPoint in this architecture.
# We use the Loiter condition as the primary flight environment.
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=51.45, units="m/s")
indep_var_comp.add_output("alpha", val=np.array([5.0, 2.0]), units="deg")
indep_var_comp.add_output("Mach_number", val=0.15)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=1.006, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

for surface in surfaces:
    name = surface["name"]
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(name + "_geom", geom_group)

for i in range(n_points):
    aero_group = AeroPoint(surfaces=surfaces)
    point_name = "aero_point_{}".format(i)
    prob.model.add_subsystem(point_name, aero_group)

    prob.model.connect("v", point_name + ".v")
    prob.model.connect("alpha", point_name + ".alpha", src_indices=[i])
    prob.model.connect("Mach_number", point_name + ".Mach_number")
    prob.model.connect("re", point_name + ".re")
    prob.model.connect("rho", point_name + ".rho")
    prob.model.connect("cg", point_name + ".cg")

    for surface in surfaces:
        name = surface["name"]
        prob.model.connect(point_name + ".CD", "multi_CD." + str(i) + "_CD")
        prob.model.connect(name + "_geom.mesh", point_name + "." + name + ".def_mesh")
        prob.model.connect(name + "_geom.mesh", point_name + ".aero_states." + name + "_def_mesh")
        prob.model.connect(name + "_geom.t_over_c", point_name + "." + name + "_perf." + "t_over_c")

prob.model.add_subsystem("multi_CD", MultiCD(n_points=n_points), promotes_outputs=["CD"])

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-6

output_dir = os.path.join("src", "openaerostruct_out", "generated_run_out")
os.makedirs(output_dir, exist_ok=True)
recorder = om.SqliteRecorder(os.path.join(output_dir, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = output_dir

# Design Variables
# alpha is size (2,)
prob.model.add_design_var("alpha", lower=-10, upper=15)
# twist_cp is size (num_twist_cp,)
prob.model.add_design_var("wing_geom.twist_cp", lower=-5, upper=5)
# t_over_c_cp is size (num_twist_cp,)
prob.model.add_design_var("wing_geom.t_over_c_cp", lower=0.08, upper=0.15)

# Constraints: Target CL for Loiter (0) and Dash (1)
prob.model.add_constraint("aero_point_0.wing_perf.CL", equals=0.8)
prob.model.add_constraint("aero_point_1.wing_perf.CL", equals=0.4)

# Objective: Minimize average CD
prob.model.add_objective("CD", scaler=1e3)

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Optimization Results ---")
print(f"Final CD (Avg): {prob.get_val('CD')[0]:.6f}")
print(f"CL Point 0: {prob.get_val('aero_point_0.wing_perf.CL')[0]:.4f}")
print(f"CL Point 1: {prob.get_val('aero_point_1.wing_perf.CL')[0]:.4f}")
print(f"Twist CP: {prob.get_val('wing_geom.twist_cp')}")
print(f"Thickness CP: {prob.get_val('wing_geom.t_over_c_cp')}")

# --- AD-HOC VISUALIZATION ---
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Extract data
    twist = np.array(prob.get_val('wing_geom.twist_cp'))
    thickness = np.array(prob.get_val('wing_geom.t_over_c_cp'))
    
    # Create control point index for plotting
    cp_idx = np.arange(len(twist))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Twist
    color = 'tab:blue'
    ax1.set_xlabel('Control Point Index')
    ax1.set_ylabel('Twist (deg)', color=color)
    ax1.plot(cp_idx, twist, marker='o', color=color, label='Twist')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create second axis for thickness
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('t/c', color=color)
    ax2.plot(cp_idx, thickness, marker='s', color=color, label='Thickness')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Optimized Wing Twist and Thickness Distribution")
    fig.tight_layout()

    # Save using standardized path
    plot_dir = os.path.join("src", "openaerostruct_out", "agent_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "wing_design_dist.png"))
    plt.close()
    print(f"Plot saved to {plot_dir}/wing_design_dist.png")

except Exception as e:
    print(f"Visualization Warning: {e}")