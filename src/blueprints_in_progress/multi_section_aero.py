import numpy as np
import os
import openmdao.api as om
from openaerostruct.geometry.geometry_group import MultiSecGeometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.utils import build_section_dicts, unify_mesh, build_multi_spline, connect_multi_spline
import matplotlib.pyplot as plt

# =============================================================================
# 1. MULTI-SECTION GEOMETRY DEFINITION
# =============================================================================
# The agent should define each section's name, span, taper, and sweep.
sec_chord_cp = [np.ones(2), np.ones(2)]

surface = {
    "name": "wing",
    "is_multi_section": True,
    "num_sections": 2,
    "sec_name": ["sec0", "sec1"],
    "symmetry": True,
    "S_ref_type": "wetted",
    "taper": [1.0, 1.0],
    "span": [5.0, 5.0],
    "sweep": [0.0, 0.0],
    "chord_cp": sec_chord_cp,
    "twist_cp": [np.zeros(2), np.zeros(2)],
    "root_chord": 2.0,
    "meshes": "gen-meshes",
    "nx": 3,
    "ny": [11, 11],
    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,
}

# =============================================================================
# 2. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=200.0, units="m/s")
indep_var_comp.add_output("alpha", val=3.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.7)
indep_var_comp.add_output("re", val=1e7, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

section_surfaces = build_section_dicts(surface)
uniMesh = unify_mesh(section_surfaces)
surface["mesh"] = uniMesh

chord_comp = build_multi_spline("chord_cp", surface["num_sections"], sec_chord_cp)
prob.model.add_subsystem("chord_bspline", chord_comp)

connect_multi_spline(prob, section_surfaces, sec_chord_cp, "chord_cp", "chord_bspline", surface["name"])

multi_geom_group = MultiSecGeometry(surface=surface)
prob.model.add_subsystem(surface["name"], multi_geom_group)

aero_group = AeroPoint(surfaces=[surface])
point_name = "flight_condition_0"
prob.model.add_subsystem(point_name, aero_group, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"])

name = surface["name"]
unification_name = "{}_unification".format(surface["name"])
prob.model.connect(name + "." + unification_name + "." + name + "_uni_mesh", point_name + ".wing.def_mesh")
prob.model.connect(name + "." + unification_name + "." + name + "_uni_mesh", point_name + ".aero_states.wing_def_mesh")

# =============================================================================
# 3. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-7

os.makedirs("src/openaerostruct_out/generated_run_out", exist_ok=True)
recorder = om.SqliteRecorder("src/openaerostruct_out/generated_run_out/aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

# Design Variables
prob.model.add_design_var("chord_bspline.chord_cp_spline", lower=0.1, upper=10.0)
prob.model.add_design_var("alpha", lower=-5.0, upper=10.0, units="deg")

# Constraints & Objective
prob.model.add_constraint(point_name + ".wing_perf.CL", equals=0.5)
prob.model.add_objective(point_name + ".wing_perf.CD", scaler=1e4)

# =============================================================================
# 4. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Multi-Section Optimization Results ---")
print(f"Final CL: {prob.get_val(point_name + '.wing_perf.CL')[0]:.4f}")
print(f"Final CD: {prob.get_val(point_name + '.wing_perf.CD')[0]:.6f}")

# Plotting optimized planform
meshUni = prob.get_val(name + "." + unification_name + "." + name + "_uni_mesh")
output_dir = "src/openaerostruct_out/agent_plots"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8, 4))
mesh_x = meshUni[:, :, 0]
mesh_y = meshUni[:, :, 1]
for i in range(mesh_x.shape[0]):
    plt.plot(mesh_y[i, :], mesh_x[i, :], 'k', lw=1)
    plt.plot(-mesh_y[i, :], mesh_x[i, :], 'k', lw=1)
for j in range(mesh_x.shape[1]):
    plt.plot(mesh_y[:, j], mesh_x[:, j], 'k', lw=1)
    plt.plot(-mesh_y[:, j], mesh_x[:, j], 'k', lw=1)
plt.axis("equal")
plt.title("Optimized Multi-Section Planform")
plt.savefig(os.path.join(output_dir, "multi_section_planform.png"))
plt.close()
