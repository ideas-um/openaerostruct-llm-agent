import numpy as np
import os
import openmdao.api as om
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.structures.struct_groups import SpatialBeamAlone

# =============================================================================
# 1. MESH GENERATION (STRUCTURAL ONLY)
# =============================================================================
# The agent should modify these parameters to change the wing's baseline shape.
mesh_dict = {"num_y": 7, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}
mesh, twist_cp = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# Structural only analysis.
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "fem_model_type": "tube",
    "mesh": mesh,
    "E": 70.0e9,
    "G": 30.0e9,
    "yield": 500.0e6,
    "safety_factor": 2.5,
    "mrho": 3.0e3,
    "fem_origin": 0.35,
    "t_over_c_cp": np.array([0.15]),
    "thickness_cp": np.ones((3)) * 0.1,
    "wing_weight_ratio": 2.0,
    "struct_weight_relief": False,
    "distributed_fuel_weight": False,
    "exact_failure_constraint": False,
}

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

ny = surf_dict["mesh"].shape[1]
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("loads", val=np.ones((ny, 6)) * 2e5, units="N")
indep_var_comp.add_output("load_factor", val=1.0)

struct_group = SpatialBeamAlone(surface=surf_dict)
struct_group.add_subsystem("indep_vars", indep_var_comp, promotes=["*"])
prob.model.add_subsystem(surf_dict["name"], struct_group)

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["disp"] = True
prob.driver.options["tol"] = 1e-9

os.makedirs(os.path.join("src", "openaerostruct_out"), exist_ok=True)
recorder = om.SqliteRecorder(os.path.join("src", "openaerostruct_out", "struct.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

# Design Variables
prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, ref=1e-1)

# Constraints & Objective
prob.model.add_constraint("wing.failure", upper=0.0)
prob.model.add_constraint("wing.thickness_intersects", upper=0.0)
prob.model.add_objective("wing.structural_mass", scaler=1e-5)

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Structural Optimization Results ---")
print(f"Final structural mass: {prob.get_val('wing.structural_mass')[0]:.4f} kg")