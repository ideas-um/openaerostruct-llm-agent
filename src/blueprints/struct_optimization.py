import numpy as np
import os
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.structures.struct_groups import SpatialBeamAlone

# =============================================================================
# IMPORTANT: STRUCTURAL ONLY — use SpatialBeamAlone, NOT AerostructPoint.
# AerostructPoint is for coupled aero-structural problems and will crash here
# with "promotes_inputs failed to find any matches for 'forces'".
# Loads are applied manually via IndepVarComp inside the SpatialBeamAlone group.
# =============================================================================

# =============================================================================
# 1. MESH GENERATION (STRUCTURAL ONLY)
# =============================================================================
# Only num_y is needed — no aerodynamic mesh resolution required.
# num_twist_cp is unused structurally but required by generate_mesh for CRM.
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
# CRITICAL: thickness_cp must be declared here to use it as a design variable.
# Material properties set here — modify E, G, yield, safety_factor to match the user's scenario.
# === AGENT EDITABLE SECTION START ===
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "fem_model_type": "tube",
    "mesh": mesh,

    # Structural design variable — must be declared here
    "thickness_cp": np.ones((3)) * 0.1,     # Tube wall thickness B-spline CPs [m]

    # Airfoil thickness (used for intersection check)
    "t_over_c_cp": np.array([0.15]),

    # Material properties
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

output_dir = os.path.join("src", "openaerostruct_out", "generated_run_out")
os.makedirs(output_dir, exist_ok=True)
recorder = om.SqliteRecorder(os.path.join(output_dir, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = output_dir

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# Available DV paths:
#   "wing.thickness_cp"   — tube wall thickness B-spline CPs [m]  (declared in surf_dict)
#   "wing.radius_cp"      — tube radius CPs [m]                   (requires radius_cp in surf_dict)

prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, ref=1e-1)

# --- Constraints ---
# Available constraint paths:
#   "wing.failure"              — structural failure index (KS aggregated), upper=0
#   "wing.thickness_intersects" — prevents tube wall from exceeding airfoil thickness, upper=0
#   "wing.structural_mass"      — total structural mass [kg]

prob.model.add_constraint("wing.failure", upper=0.0)
prob.model.add_constraint("wing.thickness_intersects", upper=0.0)

# --- Objective ---
# Common objectives:
#   "wing.structural_mass"  — minimize structural mass [kg]
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