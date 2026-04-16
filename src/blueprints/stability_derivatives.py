import numpy as np
import os
import openmdao.api as om
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# The agent should modify these parameters to change the wing's baseline shape.
mesh_dict = {
    "num_y": 15,
    "num_x": 3,
    "wing_type": "rect",
    "root_chord": 1.0,  
    "span": 10.0,  
    "symmetry": True,
    "num_twist_cp": 2,
    "span_cos_spacing": 0.0,
}
mesh = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh,
    "twist_cp": np.zeros(mesh_dict["num_twist_cp"]),
    "sweep": 0.0,
    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "t_over_c_cp": np.array([0.15]),
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,
}
surfaces = [surf_dict]

# =============================================================================
# 3. PROBLEM SETUP (STABILITY DERIVATIVES)
# =============================================================================
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.array([0.5, 0.0, 0.0]), units="m")
prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

alpha_FD_stepsize = 1e-4
alpha_perturb_comp = om.ExecComp("alpha_plus_delta = alpha + delta_alpha", units="deg", delta_alpha={"val": alpha_FD_stepsize, "constant": True})
prob.model.add_subsystem("alpha_for_FD", alpha_perturb_comp, promotes=["*"])

for surface in surfaces:
    name = surface["name"]
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(name + "_geom", geom_group)

point_names = ["aero_point", "aero_point_FD"]
for i in range(2):
    aero_group = AeroPoint(surfaces=surfaces)
    point_name = point_names[i]
    prob.model.add_subsystem(point_name, aero_group)
    prob.model.connect("v", point_name + ".v")
    prob.model.connect("Mach_number", point_name + ".Mach_number")
    prob.model.connect("re", point_name + ".re")
    prob.model.connect("rho", point_name + ".rho")
    prob.model.connect("cg", point_name + ".cg")
    alpha_name = "alpha" if i == 0 else "alpha_plus_delta"
    prob.model.connect(alpha_name, point_name + ".alpha")

    for surface in surfaces:
        name = surface["name"]
        prob.model.connect(name + "_geom.mesh", point_name + "." + name + ".def_mesh")
        prob.model.connect(name + "_geom.mesh", point_name + ".aero_states." + name + "_def_mesh")
        prob.model.connect(name + "_geom.t_over_c", point_name + "." + name + "_perf.t_over_c")

stability_derivatives_comp = om.ExecComp(["CL_alpha = (CL_FD - CL) / delta_alpha", "CM_alpha = (CM_FD - CM) / delta_alpha"], delta_alpha={"val": alpha_FD_stepsize, "constant": True}, CL_alpha={"val": 0.0, "units": "1/deg"}, CM_alpha={"val": np.zeros(3), "units": "1/deg"})
prob.model.add_subsystem("stability_derivs", stability_derivatives_comp, promotes_outputs=["*"])
prob.model.connect("aero_point.CL", "stability_derivs.CL")
prob.model.connect("aero_point.CM", "stability_derivs.CM")
prob.model.connect("aero_point_FD.CL", "stability_derivs.CL_FD")
prob.model.connect("aero_point_FD.CM", "stability_derivs.CM_FD")

static_margin_comp = om.ExecComp("static_margin = -CM_alpha / CL_alpha", CM_alpha={"val": 0.0, "units": "1/deg"}, CL_alpha={"val": 0.0, "units": "1/deg"})
prob.model.add_subsystem("static_margin", static_margin_comp, promotes_outputs=["*"])
prob.model.connect("CL_alpha", "static_margin.CL_alpha")
prob.model.connect("CM_alpha", "static_margin.CM_alpha", src_indices=1) 

# =============================================================================
# 4. EXECUTION SETTINGS
# =============================================================================
os.makedirs("src/openaerostruct_out", exist_ok=True)
recorder = om.SqliteRecorder("src/openaerostruct_out/aero.db")
prob.driver.add_recorder(recorder)

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.set_val("wing_geom.sweep", 10.0, units="deg")
prob.run_model()

print("\n--- Stability Derivatives Results ---")
print(f"Sweep angle: {prob.get_val('wing_geom.sweep', units='deg')[0]:.2f} deg")
print(f"CL_alpha: {prob.get_val('CL_alpha', units='1/deg')[0]:.6f} 1/deg")
print(f"CM_alpha: {prob.get_val('CM_alpha', units='1/deg')[1]:.6f} 1/deg")
print(f"Static Margin: {prob.get_val('static_margin')[0]:.4f}")