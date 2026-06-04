import numpy as np
import os
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR     = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR   = os.path.join(_OUT_DIR, "agent_plots")
_RUN_OUT_DIR = os.path.join(_OUT_DIR, "generated_run_out")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_RUN_OUT_DIR, exist_ok=True)

mesh_dict = {
    "num_y": 5,
    "num_x": 2,
    "wing_type": "CRM",
    "symmetry": True,
    "num_twist_cp": 5,
}

mesh, twist_cp = generate_mesh(mesh_dict)

# === AGENT EDITABLE SECTION START ===
surface = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "fem_model_type": "tube",

    "thickness_cp": np.array([0.05, 0.05, 0.05]),
    "radius_cp": np.array([0.1, 0.1, 0.1]),
    "twist_cp": twist_cp,

    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "t_over_c_cp": np.array([0.15]),
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,

    "E": 70.0e9,
    "G": 30.0e9,
    "yield": 500.0e6,
    "safety_factor": 2.5,
    "mrho": 3000.0,
    "fem_origin": 0.35,
    "wing_weight_ratio": 2.0,
    "struct_weight_relief": False,
    "distributed_fuel_weight": False,
    "exact_failure_constraint": False,
}
# === AGENT EDITABLE SECTION END ===

prob = om.Problem()

# === AGENT EDITABLE SECTION START ===
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s")
indep_var_comp.add_output("R", val=11.165e6, units="m")
indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")
indep_var_comp.add_output("speed_of_sound", val=295.4, units="m/s")
indep_var_comp.add_output("load_factor", val=1.0)
indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")
# === AGENT EDITABLE SECTION END ===

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

name = "wing"
aerostruct_group = AerostructGeometry(surface=surface)
prob.model.add_subsystem(name, aerostruct_group)

point_name = "AS_point_0"
AS_point = AerostructPoint(surfaces=[surface])
prob.model.add_subsystem(point_name, AS_point,
    promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "CT", "R", "W0",
                     "speed_of_sound", "empty_cg", "load_factor"])

com_name = point_name + "." + name + "_perf"
prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")
prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
prob.model.connect(name + ".radius", com_name + ".radius")
prob.model.connect(name + ".thickness", com_name + ".thickness")
prob.model.connect(name + ".nodes", com_name + ".nodes")
prob.model.connect(name + ".cg_location", point_name + ".total_perf." + name + "_cg_location")
prob.model.connect(name + ".structural_mass", point_name + ".total_perf." + name + "_structural_mass")
prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = _RUN_OUT_DIR

# === AGENT EDITABLE SECTION START ===
prob.model.add_design_var("wing.thickness_cp", lower=0.005, upper=0.3)
prob.model.add_design_var("wing.radius_cp", lower=0.01, upper=0.5)
prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)

prob.model.add_constraint("AS_point_0.wing_perf.failure", upper=0.0)
prob.model.add_constraint("AS_point_0.L_equals_W", equals=0.0)

prob.model.add_objective("wing.structural_mass")
# === AGENT EDITABLE SECTION END ===

prob.setup()
prob.run_driver()

print("\n--- Aerostructural Tube Results ---")
print(f"Final structural mass:{prob.get_val('wing.structural_mass')[0]:.4f} [kg]")

try:
    struct_mass = prob.get_val('wing.structural_mass')[0]
    thickness_cp_vals = prob.get_val('wing.thickness_cp')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Struct. Mass"], [struct_mass], color="tomato")
    axes[1].plot(np.arange(len(thickness_cp_vals)), thickness_cp_vals, "s-", color="purple")
    axes[1].set_title("Optimized Wall Thickness")
    fig.savefig(os.path.join(_PLOTS_DIR, "aerostruct_tube_results.png"), bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"Plotting warning: {e}")