import numpy as np
import openmdao.api as om
import os

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.integration.multipoint_comps import MultiCD

mesh_dict = {
    "num_y": 5,
    "num_x": 3,
    "wing_type": "CRM",
    "symmetry": True,
    "num_twist_cp": 5,
    "span_cos_spacing": 0.0,
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
    "CD0": 0.015,
    "k_lam": 0.05,
    "t_over_c_cp": np.array([0.15]),
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,
}

surfaces = [surf_dict]
n_points = 2

# =============================================================================
# 3. PROBLEM SETUP (MULTIPOINT)
# =============================================================================
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
# Initial guesses for alpha
indep_var_comp.add_output("v", val=240.0, units="m/s")
indep_var_comp.add_output("alpha", val=np.array([4.0, 6.0]), units="deg")
indep_var_comp.add_output("Mach_number", val=0.8)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.4, units="kg/m**3")
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

os.makedirs("src/openaerostruct_out", exist_ok=True)
recorder = om.SqliteRecorder("src/openaerostruct_out/aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

# Design Variables
# We optimize alpha at each point to satisfy the CL constraint and twist to minimize drag
prob.model.add_design_var("alpha", lower=-15, upper=15)
prob.model.add_design_var("wing_geom.twist_cp", lower=-5, upper=8)

# Constraints: CL=0.5 at point 0, CL=0.6 at point 1
prob.model.add_constraint("aero_point_0.wing_perf.CL", equals=0.5)
prob.model.add_constraint("aero_point_1.wing_perf.CL", equals=0.6)

# Objective: Minimize total drag (sum of CD at both points)
prob.model.add_objective("CD", scaler=1e4)

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Multipoint Optimization Results ---")
print(f"Final CD (Sum): {prob.get_val('CD')[0]:.6f}")
print(f"Point 0 - CL: {prob.get_val('aero_point_0.wing_perf.CL')[0]:.4f}, Alpha: {prob.get_val('alpha')[0]:.4f}")
print(f"Point 1 - CL: {prob.get_val('aero_point_1.wing_perf.CL')[0]:.4f}, Alpha: {prob.get_val('alpha')[1]:.4f}")
print(f"Final Twist: {prob.get_val('wing_geom.twist_cp')}")