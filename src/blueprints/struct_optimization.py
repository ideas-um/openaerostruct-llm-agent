import numpy as np
import os
import openmdao.api as om
import matplotlib
import re as _re

matplotlib.use("Agg")
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
# Absolute output paths for Benchmark execution
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR = os.path.join(_SRC_DIR, "openaerostruct_out")
_PLOTS_DIR = os.path.join(_OUT_DIR, "agent_plots")
_RUN_OUT_DIR = os.path.join(_OUT_DIR, "generated_run_out")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_RUN_OUT_DIR, exist_ok=True)


def _plot_path(name: str) -> str:
    """Sanitize name and return full path inside _PLOTS_DIR."""
    safe = _re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    return os.path.join(_PLOTS_DIR, safe + ".png")


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)

# =============================================================================
# 1. MESH GENERATION (STRUCTURAL ONLY)
# =============================================================================
# === AGENT EDITABLE SECTION START ===
# Create a dictionary to store options about the mesh
mesh_dict = {
    "num_y": 7,
    "num_x": 2,
    "wing_type": "rect",
    "span": 8.0,
    "root_chord": 1.5,
    "symmetry": True,
}
# === AGENT EDITABLE SECTION END ===

_r = generate_mesh(mesh_dict)
mesh = _r[0] if isinstance(_r, tuple) else _r

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# === AGENT EDITABLE SECTION START ===
surf_dict = {
    # Wing definition
    "name": "wing",  # name of the surface
    "symmetry": True,  # if true, model one half of wing reflected across y = 0
    "fem_model_type": "tube",
    "mesh": mesh,
    # Material properties
    "E": 70.0e9,  # [Pa] Young's modulus of the spar
    "G": 30.0e9,  # [Pa] shear modulus of the spar
    "yield": 500.0e6,  # [Pa] yield stress
    "safety_factor": 1.5,  # yield stress divided by safety factor for limiting case
    "mrho": 2700.0,  # [kg/m^3] material density
    # Structural design parameters
    "fem_origin": 0.35,  # normalized chordwise location of the spar
    "t_over_c_cp": np.array([0.15]),  # maximum airfoil thickness ratio
    "thickness_cp": np.ones((3)) * 0.1,
    "radius_cp": np.ones((3)) * 0.2,  # Tube radius [m]
    # Weight and coupling flags
    "wing_weight_ratio": 2.0,
    "struct_weight_relief": False,
    "distributed_fuel_weight": False,
    "exact_failure_constraint": False,
}
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

# === AGENT EDITABLE SECTION START ===
ny = surf_dict["mesh"].shape[1]

# Define independent variables for loads
indep_var_comp = om.IndepVarComp()
# loads shape must be (ny, 6): [Fx, Fy, Fz, Mx, My, Mz]
# Here we distribute a total load of 4e4 N vertically (Fz is index 2)
total_load = 4e4
load_val = np.zeros((ny, 6))
load_val[:, 2] = total_load / ny

indep_var_comp.add_output("loads", val=load_val, units="N")
indep_var_comp.add_output("load_factor", val=1.0)
# === AGENT EDITABLE SECTION END ===

struct_group = SpatialBeamAlone(surface=surf_dict)

# Add indep_vars to the structural group (TUTORIAL STYLE)
struct_group.add_subsystem("indep_vars", indep_var_comp, promotes=["*"])

prob.model.add_subsystem(surf_dict["name"], struct_group)

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["disp"] = True
prob.driver.options["tol"] = 1e-8

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# Use ref to normalize variables to order ~1.0 for better convergence.
prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, ref=1e-1)

# --- Constraints ---
prob.model.add_constraint("wing.failure", upper=0.0)
prob.model.add_constraint("wing.thickness_intersects", upper=0.0)

# --- Objective ---
# Use scaler to bring mass [kg] to order ~1.0 (e.g., 1e-2 or 1e-3).
prob.model.add_objective("wing.structural_mass", scaler=1e-2)
# === AGENT EDITABLE SECTION END ===

prob.setup()
prob.run_driver()

# =============================================================================
# 6. PLOTTING (Analytical Engineering Style)
# =============================================================================
print("\n--- Structural Optimization Results ---")
print(f"Final structural mass: {prob.get_val('wing.structural_mass')[0]:.4f} kg")

# Wing Planform Plot
_mesh_out = prob.get_val("wing.mesh", units="m")
fig_wing, ax_wing = plt.subplots(figsize=(8, 4))
for i in range(_mesh_out.shape[0]):
    ax_wing.plot(_mesh_out[i, :, 1], _mesh_out[i, :, 0], color="black", lw=1.5)
for j in range(_mesh_out.shape[1]):
    ax_wing.plot(_mesh_out[:, j, 1], _mesh_out[:, j, 0], color="black", lw=1.5)
ax_wing.set_xlabel("Spanwise y [m]")
ax_wing.set_ylabel("Chordwise x [m]")
ax_wing.set_title("Optimized Wing Planform")
ax_wing.set_aspect("equal")
fig_wing.tight_layout()
fig_wing.savefig(_plot_path("wing_planform"), bbox_inches="tight", dpi=150)
plt.close(fig_wing)

# Sizing Distribution Plot
thickness_vals = prob.get_val("wing.thickness_cp")
fig_thick, ax_thick = plt.subplots(figsize=(8, 4))
ax_thick.plot(thickness_vals * 1000, "o-", color="black", lw=1.5)
ax_thick.set_xlabel("Control Point Index")
ax_thick.set_ylabel("Thickness [mm]")
ax_thick.set_title("Structural Sizing Distribution")
ax_thick.grid(True, linestyle="--", alpha=0.7)
fig_thick.tight_layout()
fig_thick.savefig(_plot_path("thickness_distribution"), bbox_inches="tight", dpi=150)
plt.close(fig_thick)
