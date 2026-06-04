import numpy as np
import openmdao.api as om
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR     = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR   = os.path.join(_OUT_DIR, "agent_plots")
_RUN_OUT_DIR = os.path.join(_OUT_DIR, "generated_run_out")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_RUN_OUT_DIR, exist_ok=True)

mesh_dict = {
    "num_y": 19,
    "num_x": 3,
    "wing_type": "rect",
    "symmetry": True,
    "span": 10.0,
    "root_chord": 2.0,
    "span_cos_spacing": 0.0,
    "chord_cos_spacing": 0.0,
}

mesh = generate_mesh(mesh_dict)

surface = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh,

    "CL0": 0.0,
    "CD0": 0.0,
    "with_viscous": True,
    "with_wave": True,
    "k_lam": 0.05,
    "c_max_t": 0.303,
    "t_over_c_cp": np.array([0.12]),

    "twist_cp": np.zeros(3),
    "sweep": 0.0,
}

prob = om.Problem()

Mach_number = 0.85
rho = 1.225
v = Mach_number * 340.0
re = rho * v / 1.81e-5

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=v, units="m/s")
indep_var_comp.add_output("alpha", val=3.0, units="deg")
indep_var_comp.add_output("Mach_number", val=Mach_number)
indep_var_comp.add_output("re", val=re, units="1/m")
indep_var_comp.add_output("rho", val=rho, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

name = surface["name"]
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(name, geom_group)

aero_group = AeroPoint(surfaces=[surface], rotational=True)
point_name = "flight_condition_0"
prob.model.add_subsystem(
    point_name,
    aero_group,
    promotes_inputs=["v", "alpha", "beta", "omega", "Mach_number", "re", "rho", "cg"],
)

prob.model.connect(f"{name}.mesh", f"{point_name}.{name}.def_mesh")
prob.model.connect(f"{name}.mesh", f"{point_name}.aero_states.{name}_def_mesh")
prob.model.connect(f"{name}.t_over_c", f"{point_name}.{name}_perf.t_over_c")

prob.driver = om.ScipyOptimizeDriver()

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = _RUN_OUT_DIR

prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=10.0)
prob.model.add_design_var("wing.sweep", lower=0.0, upper=45.0)

prob.model.add_constraint(f"{point_name}.wing_perf.CL", equals=0.5)

prob.model.add_objective(f"{point_name}.wing_perf.CD", ref=0.01)

prob.setup()
prob.run_driver()

print("\n--- Optimization Results ---")
print(f"Final Alpha: {prob.get_val('alpha', units='deg')[0]:.4f} deg")
print(f"Final CL:    {prob.get_val(f'{point_name}.wing_perf.CL')[0]:.4f}")
print(f"Final CD:    {prob.get_val(f'{point_name}.wing_perf.CD')[0]:.6f}")

try:
    alpha_val = prob.get_val('alpha', units='deg')[0]
    CL_val = prob.get_val(f'{point_name}.wing_perf.CL')[0]
    CD_val = prob.get_val(f'{point_name}.wing_perf.CD')[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["CL", "CD"], [CL_val, CD_val], color=["steelblue", "tomato"])
    axes[0].set_title("Aerodynamic Coefficients")
    axes[0].set_ylabel("Value")

    mesh_x = prob.get_val("wing.mesh")[:, :, 0]
    mesh_y = prob.get_val("wing.mesh")[:, :, 1]
    for i in range(mesh_x.shape[0]):
        axes[1].plot(mesh_y[i, :], mesh_x[i, :], color="C0", lw=1)
    for j in range(mesh_x.shape[1]):
        axes[1].plot(mesh_y[:, j], mesh_x[:, j], color="C0", lw=1)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Span (m)")
    axes[1].set_ylabel("Chord (m)")
    axes[1].set_title(f"Wing Mesh  (α={alpha_val:.2f}°)")

    fig.tight_layout()
    fig.savefig(os.path.join(_PLOTS_DIR, "aero_rect_results.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {_PLOTS_DIR}")
except Exception as e:
    print(f"Plotting warning: {e}")