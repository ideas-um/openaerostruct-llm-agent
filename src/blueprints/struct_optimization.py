import numpy as np
import os
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.structures.struct_groups import SpatialBeamAlone

# =============================================================================
# IMPORTANT: STRUCTURAL ONLY — use SpatialBeamAlone, NOT AerostructPoint.
# AerostructPoint is for coupled aero-structural problems and will crash here
# with "promotes_inputs failed to find any matches for 'forces'".
# Loads are applied manually via IndepVarComp inside the SpatialBeamAlone group.
# =============================================================================

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
# 1. MESH GENERATION (STRUCTURAL ONLY)
# =============================================================================
# Only num_y is needed — no aerodynamic mesh resolution required.
# num_twist_cp is unused structurally but required by generate_mesh for CRM.
# generate_mesh returns TWO values for CRM — always unpack as (mesh, twist_cp).
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 7,
    "wing_type": "CRM",
    "symmetry": True,
    "num_twist_cp": 5,
}
# === AGENT EDITABLE SECTION END ===

mesh, twist_cp = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# CRITICAL: Structural DVs (thickness_cp, radius_cp) MUST be declared here first.
# Material properties set here — modify E, G, yield, safety_factor as needed.
#
# FULL STRUCTURAL DV CATALOG for tube FEM (from God Document):
# -----------------------------------------------
#   KEY               TYPE    DESCRIPTION
#   thickness_cp      array   Tube wall thickness B-spline CPs [m]. Shape=(n_cp,).
#                             Controls the tube wall thickness spanwise.
#                             Directly affects structural mass and stress.
#   radius_cp         array   Tube outer radius B-spline CPs [m]. Shape=(n_cp,).
#                             Controls the tube cross-section radius spanwise.
#                             Use EITHER thickness_cp OR radius_cp (or both).
#                             Requires "radius_cp" key here AND add_design_var() below.
#
# NOTE: There are NO geometry DVs (twist_cp, chord_cp, etc.) in structural-only
# optimization — the mesh is fixed and only structural sizing is optimized.
# t_over_c_cp is included below for intersection checking, not as a DV.
# === AGENT EDITABLE SECTION START ===
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "fem_model_type": "tube",
    "mesh": mesh,

    # Structural DVs — declare here to use in add_design_var()
    "thickness_cp": np.ones((3)) * 0.1,     # Tube wall thickness B-spline CPs [m]
    #"radius_cp": np.ones((3)) * 0.05,      # Tube radius B-spline CPs [m] — optional DV

    # Airfoil thickness ratio — used for intersection check constraint
    "t_over_c_cp": np.array([0.15]),

    # Material properties — modify to match the user's material
    "E": 70.0e9,            # Young's modulus [Pa]
    "G": 30.0e9,            # Shear modulus [Pa]
    "yield": 500.0e6,       # Yield stress [Pa]
    "safety_factor": 2.5,   # Structural safety factor
    "mrho": 3.0e3,          # Material density [kg/m^3]
    "fem_origin": 0.35,     # Normalized chordwise FEM beam position

    "wing_weight_ratio": 2.0,
    "struct_weight_relief": False,
    "distributed_fuel_weight": False,
    "exact_failure_constraint": False,
}
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
# NOTE: mesh is a numpy array of shape (nx, ny, 3).
# Get the number of spanwise nodes as: ny = surf_dict["mesh"].shape[1]
# Do NOT use mesh.nodes — mesh has no such attribute.
prob = om.Problem()

# === AGENT EDITABLE SECTION START ===
# Applied loads — modify load magnitude to match the user's scenario.
# loads shape must be (ny, 6): [Fx, Fy, Fz, Mx, My, Mz] per node.
# Typically only Fz (index 2) is non-zero for a vertical distributed load.
ny = surf_dict["mesh"].shape[1]
load_magnitude = 2e5       # Load per node [N] — modify this value

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("loads", val=np.ones((ny, 6)) * load_magnitude, units="N")
indep_var_comp.add_output("load_factor", val=1.0)
# === AGENT EDITABLE SECTION END ===

struct_group = SpatialBeamAlone(surface=surf_dict)
struct_group.add_subsystem("indep_vars", indep_var_comp, promotes=["*"])
prob.model.add_subsystem(surf_dict["name"], struct_group)

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["disp"] = True
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = _RUN_OUT_DIR

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# FULL DV PATH REFERENCE for this blueprint (structural only, subsystem = "wing"):
#
#   PATH                  SURFACE DICT KEY    DESCRIPTION
#   "wing.thickness_cp"   "thickness_cp"      Tube wall thickness CPs [m]
#                                             lower bound typically 0.001–0.01 m
#                                             upper bound typically 0.1–0.5 m
#   "wing.radius_cp"      "radius_cp"         Tube outer radius CPs [m]
#                                             Uncomment "radius_cp" in surf_dict first.
#                                             lower bound typically 0.01 m
#                                             upper bound typically 0.5 m
#
# NOTE: There are no geometry DVs in structural-only optimization.
# The mesh is fixed — only cross-section sizing is optimized.

prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, ref=1e-1)
# prob.model.add_design_var("wing.radius_cp", lower=0.01, upper=0.5, ref=1e-1)

# --- Constraints ---
# FULL CONSTRAINT PATH REFERENCE:
#   "wing.failure"              — KS-aggregated structural failure index, upper=0
#                                 Failure index > 0 means material has yielded.
#   "wing.thickness_intersects" — prevents tube wall from exceeding airfoil thickness, upper=0
#                                 Always include this when using thickness_cp.
#   "wing.structural_mass"      — total structural mass [kg] (can be used as constraint or objective)

prob.model.add_constraint("wing.failure", upper=0.0)
prob.model.add_constraint("wing.thickness_intersects", upper=0.0)

# --- Objective ---
# FULL OBJECTIVE PATH REFERENCE:
#   "wing.structural_mass"  — minimize structural mass [kg]
#                             scaler=1e-5 normalizes to order-of-magnitude ~1
prob.model.add_objective("wing.structural_mass", scaler=1e-5)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Structural Optimization Results ---")
print(f"Final structural mass: {prob.get_val('wing.structural_mass')[0]:.4f} kg")
print(f"Final thickness_cp:    {prob.get_val('wing.thickness_cp')}")

# =============================================================================
# 6. PLOTTING
# =============================================================================
try:
    struct_mass = prob.get_val('wing.structural_mass')[0]
    thickness_cp_vals = prob.get_val('wing.thickness_cp')
    failure_val = prob.get_val('wing.failure')[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Thickness distribution
    cp_idx = np.arange(len(thickness_cp_vals))
    axes[0].plot(cp_idx, thickness_cp_vals * 1e3, "s-", color="steelblue")
    axes[0].set_xlabel("Control Point")
    axes[0].set_ylabel("Thickness (mm)")
    axes[0].set_title(f"Optimized Wall Thickness\n(Struct. mass: {struct_mass:.1f} kg)")
    axes[0].grid(True)

    # Summary bar
    axes[1].bar(["Structural\nMass (kg)", "Failure\nIndex"],
                [struct_mass, failure_val],
                color=["steelblue", "tomato" if failure_val > 0 else "green"])
    axes[1].set_title("Structural Optimization Summary")
    axes[1].set_ylabel("Value")

    fig.tight_layout()
    fig.savefig(os.path.join(_PLOTS_DIR, "struct_optimization_results.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {_PLOTS_DIR}")
except Exception as e:
    print(f"Plotting warning: {e}")