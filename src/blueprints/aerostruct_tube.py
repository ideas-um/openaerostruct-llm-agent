import numpy as np
import os
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant

# ---------------------------------------------------------------------------
# Absolute output paths — derived from __file__ so they resolve correctly
# regardless of the CWD when this script is executed as a subprocess.
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR     = os.path.join(_SRC_DIR, "openaerostruct_out")
_PLOTS_DIR   = os.path.join(_OUT_DIR, "agent_plots")
_RUN_OUT_DIR = os.path.join(_OUT_DIR, "generated_run_out")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_RUN_OUT_DIR, exist_ok=True)

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# CRM mesh is built-in. num_twist_cp sets the number of twist control points.
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 5,
    "num_x": 2,
    "wing_type": "CRM",
    "symmetry": True,
    "num_twist_cp": 5,
}
# === AGENT EDITABLE SECTION END ===

mesh, twist_cp = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# CRITICAL: Any geometry or structural parameter used as a design variable must
# be declared here first. twist_cp and thickness_cp are pre-included.
# Do NOT remove twist_cp — it is initialized from the CRM geometry above.
# === AGENT EDITABLE SECTION START ===
surface = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "fem_model_type": "tube",

    # Structural design variables — must be declared here to use in add_design_var()
    "thickness_cp": np.array([0.01, 0.02, 0.03]),   # Tube wall thickness B-spline CPs [m]
    "twist_cp": twist_cp,                            # Spanwise twist CPs [deg]

    # Aerodynamic properties
    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "t_over_c_cp": np.array([0.15]),
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,

    # Structural material properties
    "E": 70.0e9,            # Young's modulus [Pa]
    "G": 30.0e9,            # Shear modulus [Pa]
    "yield": 500.0e6,       # Yield stress [Pa]
    "safety_factor": 2.5,   # Structural safety factor
    "mrho": 3.0e3,          # Material density [kg/m^3]
    "fem_origin": 0.35,     # Normalized chordwise position of FEM beam
    "wing_weight_ratio": 2.0,
    "struct_weight_relief": False,
    "distributed_fuel_weight": False,
    "exact_failure_constraint": False,
}
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 3. PROBLEM SETUP (AEROSTRUCTURAL TUBE)
# =============================================================================
# Geometry subsystem is named "wing". DV paths are "wing.<var>" e.g. "wing.twist_cp".
prob = om.Problem()

# === AGENT EDITABLE SECTION START ===
# Mission and flight condition parameters — modify to match the user's scenario.
# Re is computed internally by OAS from v, rho, and geometry. Use re=1e6 as default.
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")        # Cruise speed [m/s]
indep_var_comp.add_output("alpha", val=5.0, units="deg")        # Initial AoA [deg]
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")     # Cruise altitude density
indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s")  # Thrust-specific fuel consumption
indep_var_comp.add_output("R", val=11.165e6, units="m")         # Range [m]
indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")      # Aircraft weight excl. wing+fuel [kg]
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

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = _RUN_OUT_DIR

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# Available DV paths:
#   "alpha"                  — angle of attack [deg] (must be DV if L=W is a constraint)
#   "wing.twist_cp"          — spanwise twist B-spline CPs (requires twist_cp in surface dict)
#   "wing.thickness_cp"      — tube wall thickness CPs [m] (requires thickness_cp in surface dict)
#   "wing.radius_cp"         — tube radius CPs [m] (requires radius_cp in surface dict)

prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, scaler=1e2)

# --- Constraints ---
# Available constraint paths:
#   "AS_point_0.wing_perf.failure"  — structural failure index (Kreisselmeier-Steinhauser), upper=0
#   "AS_point_0.L_equals_W"         — lift-equals-weight trim constraint, equals=0
#   "AS_point_0.wing_perf.CL"       — lift coefficient
#   "AS_point_0.fuelburn"           — fuel burn [kg]

prob.model.add_constraint("AS_point_0.wing_perf.failure", upper=0.0)
prob.model.add_constraint("AS_point_0.L_equals_W", equals=0.0)

# --- Objective ---
# Common objectives:
#   "AS_point_0.fuelburn"           — fuel burn [kg]  (most common for aerostructural)
#   "AS_point_0.wing_perf.CD"       — drag coefficient

prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Aerostructural Tube Results ---")
print(f"Final fuel burn:      {prob.get_val('AS_point_0.fuelburn')[0]:.4f} [kg]")
print(f"Final alpha:          {prob.get_val('alpha')[0]:.4f} [deg]")
print(f"Final structural mass:{prob.get_val('wing.structural_mass')[0]:.4f} [kg]")

# =============================================================================
# 6. PLOTTING
# =============================================================================
try:
    fuelburn = prob.get_val('AS_point_0.fuelburn')[0]
    struct_mass = prob.get_val('wing.structural_mass')[0]
    alpha_val = prob.get_val('alpha')[0]
    twist_cp_vals = prob.get_val('wing.twist_cp')
    thickness_cp_vals = prob.get_val('wing.thickness_cp')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Bar chart: key results
    labels = ["Fuel Burn (kg)", "Struct. Mass (kg)"]
    values = [fuelburn, struct_mass]
    axes[0].bar(labels, values, color=["steelblue", "tomato"])
    axes[0].set_title(f"Key Results  (α={alpha_val:.2f}°)")
    axes[0].set_ylabel("Value")

    # Twist distribution
    cp_idx = np.arange(len(twist_cp_vals))
    axes[1].plot(cp_idx, twist_cp_vals, "o-", color="green")
    axes[1].set_xlabel("Control Point")
    axes[1].set_ylabel("Twist (deg)")
    axes[1].set_title("Optimized Twist")
    axes[1].grid(True)

    # Thickness distribution
    axes[2].plot(np.arange(len(thickness_cp_vals)), thickness_cp_vals * 1e3, "s-", color="purple")
    axes[2].set_xlabel("Control Point")
    axes[2].set_ylabel("Thickness (mm)")
    axes[2].set_title("Optimized Wall Thickness")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(_PLOTS_DIR, "aerostruct_tube_results.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {_PLOTS_DIR}")
except Exception as e:
    print(f"Plotting warning: {e}")
