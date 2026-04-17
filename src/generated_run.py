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
# We use 3 sections to allow for a complex B747-like planform (sweep and taper)
num_sections = 3
sec_chord_cp = [np.ones(num_sections), np.ones(num_sections), np.ones(num_sections)]
sec_twist_cp = [np.zeros(num_sections), np.zeros(num_sections), np.zeros(num_sections)]
sec_sweep_cp = [np.full(num_sections, 25.0), np.full(num_sections, 25.0), np.full(num_sections, 25.0)]

surface = {
    "name": "wing",
    "is_multi_section": True,
    "num_sections": num_sections,
    "sec_name": ["sec0", "sec1", "sec2"],
    "symmetry": True,
    "S_ref_type": "wetted",
    "taper": [1.0, 1.0, 1.0],
    "span": [30.0, 15.0, 15.0], # Total span 60m
    "sweep": [25.0, 25.0, 25.0],
    "chord_cp": sec_chord_cp,
    "twist_cp": sec_twist_cp,
    "sweep_cp": sec_sweep_cp,
    "root_chord": 5.0,
    "meshes": "gen-meshes",
    "nx": 3,
    "ny": [11, 11, 11],
    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "c_max_t": 0.12, # Thinner profile for Mach 0.8 cruise
    "with_viscous": True,
    "with_wave": True, # Wave drag is critical at Mach 0.8
}

# =============================================================================
# 2. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=250.0, units="m/s")
indep_var_comp.add_output("alpha", val=2.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.8)
indep_var_comp.add_output("re", val=2e7, units="1/m")
indep_var_comp.add_output("rho", val=0.4, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

section_surfaces = build_section_dicts(surface)
uniMesh = unify_mesh(section_surfaces)
surface["mesh"] = uniMesh

# Create Splines for Chord, Twist, and Sweep
chord_comp = build_multi_spline("chord_cp", num_sections, sec_chord_cp)
twist_comp = build_multi_spline("twist_cp", num_sections, sec_twist_cp)
sweep_comp = build_multi_spline("sweep_cp", num_sections, sec_sweep_cp)

prob.model.add_subsystem("chord_bspline", chord_comp)
prob.model.add_subsystem("twist_bspline", twist_comp)
prob.model.add_subsystem("sweep_bspline", sweep_comp)

# Connect Splines to the Geometry
connect_multi_spline(prob, section_surfaces, sec_chord_cp, "chord_cp", "chord_bspline", surface["name"])
connect_multi_spline(prob, section_surfaces, sec_twist_cp, "twist_cp", "twist_bspline", surface["name"])
connect_multi_spline(prob, section_surfaces, sec_sweep_cp, "sweep_cp", "sweep_bspline", surface["name"])

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
prob.driver.options["tol"] = 1e-6

os.makedirs("src/openaerostruct_out/generated_run_out", exist_ok=True)
recorder = om.SqliteRecorder("src/openaerostruct_out/generated_run_out/aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

# Design Variables
# Chord spline (3 control points)
prob.model.add_design_var("chord_bspline.chord_cp_spline", lower=1.0, upper=8.0)
# Twist spline (3 control points)
prob.model.add_design_var("twist_bspline.twist_cp_spline", lower=-5.0, upper=5.0)
# Sweep spline (3 control points)
prob.model.add_design_var("sweep_bspline.sweep_cp_spline", lower=20.0, upper=45.0)
# Angle of Attack
prob.model.add_design_var("alpha", lower=-2.0, upper=8.0, units="deg")

# Constraints & Objective
# Target CL = 0.5
prob.model.add_constraint(point_name + ".wing_perf.CL", equals=0.5)
# Maximize L/D -> Minimize CD (since CL is fixed)
prob.model.add_objective(point_name + ".wing_perf.CD", scaler=1e4)

# =============================================================================
# 4. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Multi-Section Optimization Results ---")
cl_val = prob.get_val(point_name + '.wing_perf.CL')[0]
cd_val = prob.get_val(point_name + '.wing_perf.CD')[0]
print(f"Final CL: {cl_val:.4f}")
print(f"Final CD: {cd_val:.6f}")
print(f"Final L/D: {cl_val/cd_val:.4f}")
print(f"Final Alpha: {prob.get_val('alpha')[0]:.4f} deg")

# Plotting optimized planform
meshUni = prob.get_val(name + "." + unification_name + "." + name + "_uni_mesh")
output_dir = "src/openaerostruct_out/agent_plots"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 6))
mesh_x = meshUni[:, :, 0]
mesh_y = meshUni[:, :, 1]
for i in range(mesh_x.shape[0]):
    plt.plot(mesh_y[i, :], mesh_x[i, :], 'k', lw=1)
    plt.plot(-mesh_y[i, :], mesh_x[i, :], 'k', lw=1)
for j in range(mesh_x.shape[1]):
    plt.plot(mesh_y[:, j], mesh_x[:, j], 'k', lw=1)
    plt.plot(-mesh_y[:, j], mesh_x[:, j], 'k', lw=1)
plt.axis("equal")
plt.title("Optimized Multi-Section Planform (B747-like)")
plt.xlabel("Spanwise (m)")
plt.ylabel("Chordwise (m)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, "optimized_planform.png"))
plt.show()