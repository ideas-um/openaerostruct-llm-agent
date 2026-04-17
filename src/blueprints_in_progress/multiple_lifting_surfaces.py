import numpy as np
import os
import openmdao.api as om
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# =============================================================================
# 1. MESH GENERATION (WING & TAIL)
# =============================================================================
# Wing mesh
mesh_dict_wing = {"num_y": 7, "num_x": 2, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}
mesh_wing, twist_cp_wing = generate_mesh(mesh_dict_wing)

# Tail mesh (offset by 50m in x)
mesh_dict_tail = {"num_y": 7, "num_x": 2, "wing_type": "rect", "symmetry": True, "offset": np.array([50.0, 0.0, 0.0])}
mesh_tail = generate_mesh(mesh_dict_tail)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
surf_dict_wing = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh_wing,
    "twist_cp": twist_cp_wing,
    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "t_over_c_cp": np.array([0.15]),
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,
}

surf_dict_tail = {
    "name": "tail",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh_tail,
    "twist_cp": np.zeros(5),
    "CL0": 0.0,
    "CD0": 0.0,
    "k_lam": 0.05,
    "t_over_c_cp": np.array([0.15]),
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,
}

surfaces = [surf_dict_wing, surf_dict_tail]

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

for surface in surfaces:
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(surface["name"], geom_group)

aero_group = AeroPoint(surfaces=surfaces)
point_name = "flight_condition_0"
prob.model.add_subsystem(point_name, aero_group, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"])

for surface in surfaces:
    name = surface["name"]
    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
    prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")
    prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf.t_over_c")

# =============================================================================
# 4. EXECUTION/OPTIMIZATION SETTINGS
# =============================================================================
# For analysis only. Add a driver if optimization is needed.
prob.driver = om.ScipyOptimizeDriver() # Placeholder if needed

os.makedirs("src/openaerostruct_out/generated_run_out", exist_ok=True)
recorder = om.SqliteRecorder("src/openaerostruct_out/generated_run_out/aero.db")
prob.driver.add_recorder(recorder)

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_model()

print("\n--- Multiple Lifting Surfaces Analysis ---")
print(f"Total CL: {prob.get_val(point_name + '.CL')[0]:.4f}")
print(f"Total CD: {prob.get_val(point_name + '.CD')[0]:.6f}")