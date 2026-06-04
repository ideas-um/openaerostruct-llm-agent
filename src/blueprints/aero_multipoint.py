import numpy as np
import openmdao.api as om
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.integration.multipoint_comps import MultiCD

# ---------------------------------------------------------------------------
# Absolute output paths — derived from __file__ so they resolve correctly
# regardless of the CWD when this script is executed as a subprocess.
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR     = os.path.join(_SRC_DIR, "openaerostruct_out")
_PLOTS_DIR   = os.path.join(_OUT_DIR, "agent_plots")
_RUN_OUT_DIR = os.path.join(_OUT_DIR, "generated_run_out")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_RUN_OUT_DIR, exist_ok=True)

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# CRM mesh is built-in — span and root_chord are not required for "CRM".
# num_twist_cp controls how many twist control points are initialized from the CRM geometry.
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 5,
    "num_x": 3,
    "wing_type": "CRM",
    "symmetry": True,
    "num_twist_cp": 5,
    "span_cos_spacing": 0.0,
}
# === AGENT EDITABLE SECTION END ===

mesh, twist_cp = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# CRITICAL: Any geometry parameter used as a design variable MUST be declared here first.
# twist_cp is already included from generate_mesh above — do not remove it.
# === AGENT EDITABLE SECTION START ===
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh,
    "twist_cp": twist_cp,           # Required — initialized from CRM geometry
    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "t_over_c_cp": np.array([0.15]),
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,
}
# === AGENT EDITABLE SECTION END ===

surfaces = [surf_dict]
n_points = 2

# =============================================================================
# 3. PROBLEM SETUP (MULTIPOINT)
# =============================================================================
# IMPORTANT: The geometry subsystem is named "wing_geom" (not "wing").
# All geometry DV paths must use "wing_geom.<var>" e.g. "wing_geom.twist_cp".
# alpha is a vector of length n_points — one value per flight condition.
# === AGENT EDITABLE SECTION START ===
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output("alpha", val=np.ones(n_points) * 6.64, units="deg")  # shape=(n_points,)
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
# === AGENT EDITABLE SECTION END ===

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

for surface in surfaces:
    name = surface["name"]
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(name + "_geom", geom_group)   # subsystem = "wing_geom"

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
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = _RUN_OUT_DIR

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# Available DV paths:
#   "alpha"                  — vector of AoA per point, shape=(n_points,); bounds apply to all
#   "wing_geom.twist_cp"     — spanwise twist B-spline CPs (twist_cp must be in surf_dict)
#
# NOTE: geometry subsystem is "wing_geom", NOT "wing". Always use "wing_geom.<var>".

prob.model.add_design_var("alpha", lower=-15, upper=15)
prob.model.add_design_var("wing_geom.twist_cp", lower=-5, upper=8)

# --- Constraints ---
# Per-point CL constraints — each point has its own AeroPoint subsystem.
# Available constraint paths:
#   "aero_point_0.wing_perf.CL"   — CL at flight condition 0
#   "aero_point_1.wing_perf.CL"   — CL at flight condition 1
#   "aero_point_0.wing_perf.S_ref" — reference area (not "wing.S_ref" — that path is wrong)

prob.model.add_constraint("aero_point_0.wing_perf.CL", equals=0.45)
prob.model.add_constraint("aero_point_1.wing_perf.CL", equals=0.5)

# --- Objective ---
# "CD" is the sum of drag across all points (output of MultiCD component).
prob.model.add_objective("CD", scaler=1e4)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Multipoint Optimization Results ---")
print(f"Final CD (Sum): {prob.get_val('CD')[0]:.6f}")
print(f"Final alpha:    {prob.get_val('alpha')}")
print(f"Final twist_cp: {prob.get_val('wing_geom.twist_cp')}")

# =============================================================================
# 6. PLOTTING
# =============================================================================
try:
    alpha_vals = prob.get_val('alpha')
    CL_vals = [prob.get_val(f'aero_point_{i}.wing_perf.CL')[0] for i in range(n_points)]
    CD_vals = [prob.get_val(f'aero_point_{i}.wing_perf.CD')[0] for i in range(n_points)]
    twist_cp_vals = prob.get_val('wing_geom.twist_cp')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    point_labels = [f"Point {i}" for i in range(n_points)]
    axes[0].bar(point_labels, CL_vals, color="steelblue", label="CL", alpha=0.7)
    ax2 = axes[0].twinx()
    ax2.bar(point_labels, CD_vals, color="tomato", label="CD", alpha=0.5, width=0.4)
    axes[0].set_ylabel("CL", color="steelblue")
    ax2.set_ylabel("CD", color="tomato")
    axes[0].set_title("CL and CD per Flight Condition")

    cp_indices = np.arange(len(twist_cp_vals))
    axes[1].plot(cp_indices, twist_cp_vals, "o-", color="green")
    axes[1].set_xlabel("Control Point Index")
    axes[1].set_ylabel("Twist (deg)")
    axes[1].set_title("Optimized Twist Distribution")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(_PLOTS_DIR, "aero_multipoint_results.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {_PLOTS_DIR}")
except Exception as e:
    print(f"Plotting warning: {e}")
