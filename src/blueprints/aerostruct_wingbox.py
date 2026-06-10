import numpy as np
import os
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta

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
# AIRFOIL COORDINATES (fixed — defines the wingbox cross-section shape)
# =============================================================================
upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")
lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")
upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype="complex128")
lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444], dtype="complex128")

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# uCRM_based wing type is required for wingbox. num_twist_cp must match
# the length of twist_cp in the surface dict below.
# CRITICAL: generate_mesh returns TWO values — always unpack as (mesh, twist_cp).
# Writing `mesh = generate_mesh(...)` (no unpack) makes mesh a tuple and crashes
# later with: TypeError: tuple indices must be integers or slices, not tuple
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 15,
    "num_x": 3,
    "wing_type": "uCRM_based",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 0,
    "num_twist_cp": 4,
}
# === AGENT EDITABLE SECTION END ===

mesh, twist_cp = generate_mesh(mesh_dict)  # MUST unpack both — mesh is ndarray, twist_cp is ndarray

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# The agent-editable section below contains ONLY the parameters the agent should
# change: structural DVs (twist_cp, spar/skin thickness, t_over_c), material
# properties, and Wf_reserve.
#
# Everything else (airfoil coords, n_point_masses, distributed_fuel_weight, etc.)
# is set in the fixed block immediately after and must not be touched.
#
# FULL DV CATALOG for aerostructural wingbox FEM (from God Document):
# -----------------------------------------------
# GEOMETRY DVs (all require matching key here AND add_design_var() below):
#   KEY           TYPE    DESCRIPTION
#   twist_cp      array   Spanwise twist CPs [deg]. Required — from uCRM geometry.
#   chord_cp      array   Chord scaling CPs. Shape=(n_cp,).
#   xshear_cp     array   x-shear CPs [m] — generalized sweep. Shape=(n_cp,).
#   zshear_cp     array   z-shear CPs [m] — generalized dihedral. Shape=(n_cp,).
#
# STRUCTURAL DVs — wingbox FEM only (require matching key here AND add_design_var()):
#   KEY                   TYPE    DESCRIPTION
#   spar_thickness_cp     array   Spar wall thickness CPs [m]. Shape=(n_cp,).
#   skin_thickness_cp     array   Skin thickness CPs [m]. Shape=(n_cp,).
#   t_over_c_cp           array   Thickness-to-chord ratio CPs. Shape=(n_cp,).
#                                 PATH: "wing.geometry.t_over_c_cp" — NOT "wing.t_over_c_cp"
#                                 The ".geometry." level is required for wingbox.
#
# FLIGHT DVs (no surface dict entry needed):
#   "alpha"           — cruise angle of attack
#   "alpha_maneuver"  — maneuver point angle of attack (second flight condition)
#   "fuel_mass"       — fuel mass [kg] (used to close the fuel loop constraint)
# === AGENT EDITABLE SECTION START ===
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "projected",
    "mesh": mesh,
    "fem_model_type": "wingbox",

    # Structural DVs — number of CPs must match add_design_var() usage
    "twist_cp": np.array([4.0, 5.0, 8.0, 9.0]),                    # Spanwise twist [deg]
    "spar_thickness_cp": np.array([0.004, 0.005, 0.008, 0.01]),     # Spar wall thickness [m]
    "skin_thickness_cp": np.array([0.005, 0.01, 0.015, 0.025]),     # Skin thickness [m]
    "t_over_c_cp": np.array([0.08, 0.08, 0.10, 0.08]),              # Thickness-to-chord ratio

    # Optional geometry DVs — uncomment to activate
    # After uncommenting here, also add the matching add_design_var() call in Section 4.
    #"chord_cp": np.ones(4),                                         # Chord scaling CPs
    #"xshear_cp": np.zeros(4),                                       # x-shear CPs [m]
    #"zshear_cp": np.zeros(4),                                       # z-shear CPs [m]

    # Structural material — modify to match the user's material specification
    "E": 73.1e9,
    "G": (73.1e9 / 2 / 1.33),
    "yield": 420.0e6,
    "safety_factor": 1.5,
    "mrho": 2.78e3,

    "Wf_reserve": 15000.0,          # Reserve fuel [kg]
}
# === AGENT EDITABLE SECTION END ===

# ---- Fixed infrastructure — DO NOT MODIFY BELOW THIS LINE ----
# These keys are required by OAS internals and must not be removed or changed.
surf_dict.update({
    # Airfoil cross-section coordinates — required by wingbox FEM, fixed for uCRM.
    # Removing or replacing these causes KeyError: 'data_y_upper' at prob.setup().
    "data_x_upper": upper_x,
    "data_x_lower": lower_x,
    "data_y_upper": upper_y,
    "data_y_lower": lower_y,
    "original_wingbox_airfoil_t_over_c": 0.12,
    # Aerodynamic solver parameters — do not remove
    "CL0": 0.0,
    "CD0": 0.0078,
    "with_viscous": True,
    "with_wave": True,
    "k_lam": 0.05,
    "c_max_t": 0.38,
    # Structural bookkeeping — do not remove
    "strength_factor_for_upper_skin": 1.0,
    "wing_weight_ratio": 1.25,
    "exact_failure_constraint": False,
    "struct_weight_relief": True,
    # distributed_fuel_weight MUST stay True — the fuel loop connections below depend on it.
    # Setting False silently drops fuel weight relief and breaks the fuel_diff constraint.
    "distributed_fuel_weight": True,
    # n_point_masses MUST stay 1. OAS always instantiates ComputePointMassLoads and
    # n_point_masses=0 creates a size-0 nodal_weightings array that crashes prob.setup()
    # with: ValueError: 'nodal_weightings' is an array of size 0.
    # The dummy point mass below (10 kg, far from wing) satisfies the requirement with
    # negligible effect on results.
    "n_point_masses": 1,
    "fuel_density": 803.0,
})

surfaces = [surf_dict]

# =============================================================================
# 3. PROBLEM SETUP (AEROSTRUCTURAL WINGBOX — TWO POINT)
# =============================================================================
# Point 0 = cruise (Mach 0.85, rho 0.348), connected to "alpha"
# Point 1 = maneuver (load_factor 2.5), connected to "alpha_maneuver"
prob = om.Problem()

# === AGENT EDITABLE SECTION START ===
# Mission and flight condition parameters.
# Point 0 = cruise, Point 1 = maneuver (sea level by convention).
# Set altitudes and Mach numbers — v and speed_of_sound are derived from ISA.
_altitudes    = np.array([11000.0, 0.0 ])   # Altitude per point [m]
_Mach_numbers = np.array([0.85,    0.64])   # Mach per point
_rho_vals     = np.array([0.348,   1.225])  # Air density per point [kg/m^3]

def _isa_a(h):
    T = max(288.15 - 0.0065 * h, 216.65)   # ISA temperature [K], clamped at tropopause
    return np.sqrt(1.4 * 287.058 * T)       # Speed of sound [m/s]

_a_vals = np.array([_isa_a(h) for h in _altitudes])   # Speed of sound per point [m/s]
_v_vals = _Mach_numbers * _a_vals                       # Velocity per point [m/s]

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("Mach_number", val=_Mach_numbers)
indep_var_comp.add_output("v", val=_v_vals, units="m/s")                           # Derived from ISA
indep_var_comp.add_output("re", val=_rho_vals * _v_vals / 1.81e-5, units="1/m")   # Derived from ISA
indep_var_comp.add_output("rho", val=_rho_vals, units="kg/m**3")
indep_var_comp.add_output("speed_of_sound", val=_a_vals, units="m/s")              # Derived from ISA
indep_var_comp.add_output("CT", val=0.53 / 3600, units="1/s")       # Thrust-specific fuel consumption
indep_var_comp.add_output("R", val=14.307e6, units="m")              # Range [m]
indep_var_comp.add_output("W0_without_point_masses", val=128000 + surf_dict["Wf_reserve"], units="kg")
indep_var_comp.add_output("load_factor", val=np.array([1.0, 2.5]))   # [cruise, maneuver]
indep_var_comp.add_output("alpha", val=0.0, units="deg")             # Cruise AoA
indep_var_comp.add_output("alpha_maneuver", val=0.0, units="deg")    # Maneuver AoA
indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")
indep_var_comp.add_output("fuel_mass", val=10000.0, units="kg")
# === AGENT EDITABLE SECTION END ===

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

# KEEP these point-mass outputs and connections unchanged — required by n_point_masses=1
point_masses = np.array([[10.0e3]])
point_mass_locations = np.array([[25, -10.0, 0.0]])
indep_var_comp.add_output("point_masses", val=point_masses, units="kg")
indep_var_comp.add_output("point_mass_locations", val=point_mass_locations, units="m")
prob.model.add_subsystem("W0_comp",
    om.ExecComp("W0 = W0_without_point_masses + 2 * sum(point_masses)", units="kg"),
    promotes=["*"])

for surface in surfaces:
    name = surface["name"]
    aerostruct_group = AerostructGeometry(surface=surface)
    prob.model.add_subsystem(name, aerostruct_group)

for i in range(2):
    point_name = "AS_point_{}".format(i)
    AS_point = AerostructPoint(surfaces=surfaces, internally_connect_fuelburn=False)
    prob.model.add_subsystem(point_name, AS_point)

    prob.model.connect("v", point_name + ".v", src_indices=[i])
    prob.model.connect("Mach_number", point_name + ".Mach_number", src_indices=[i])
    prob.model.connect("re", point_name + ".re", src_indices=[i])
    prob.model.connect("rho", point_name + ".rho", src_indices=[i])
    prob.model.connect("CT", point_name + ".CT")
    prob.model.connect("R", point_name + ".R")
    prob.model.connect("W0", point_name + ".W0")
    prob.model.connect("speed_of_sound", point_name + ".speed_of_sound", src_indices=[i])
    prob.model.connect("empty_cg", point_name + ".empty_cg")
    prob.model.connect("load_factor", point_name + ".load_factor", src_indices=[i])
    prob.model.connect("fuel_mass", point_name + ".total_perf.L_equals_W.fuelburn")
    prob.model.connect("fuel_mass", point_name + ".total_perf.CG.fuelburn")

    for surface in surfaces:
        name = surface["name"]
        if surf_dict["distributed_fuel_weight"]:
            prob.model.connect("load_factor", point_name + ".coupled.load_factor", src_indices=[i])

        com_name = point_name + "." + name + "_perf."
        prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
        prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")
        prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
        if surf_dict["struct_weight_relief"]:
            prob.model.connect(name + ".element_mass", point_name + ".coupled." + name + ".element_mass")

        prob.model.connect(name + ".nodes", com_name + "nodes")
        prob.model.connect(name + ".cg_location", point_name + ".total_perf." + name + "_cg_location")
        prob.model.connect(name + ".structural_mass", point_name + ".total_perf." + name + "_structural_mass")
        prob.model.connect(name + ".Qz", com_name + "Qz")
        prob.model.connect(name + ".J", com_name + "J")
        prob.model.connect(name + ".A_enc", com_name + "A_enc")
        prob.model.connect(name + ".htop", com_name + "htop")
        prob.model.connect(name + ".hbottom", com_name + "hbottom")
        prob.model.connect(name + ".hfront", com_name + "hfront")
        prob.model.connect(name + ".hrear", com_name + "hrear")
        prob.model.connect(name + ".spar_thickness", com_name + "spar_thickness")
        prob.model.connect(name + ".t_over_c", com_name + "t_over_c")

        coupled_name = point_name + ".coupled." + name
        prob.model.connect("point_masses", coupled_name + ".point_masses")
        prob.model.connect("point_mass_locations", coupled_name + ".point_mass_locations")

prob.model.connect("alpha", "AS_point_0.alpha")
prob.model.connect("alpha_maneuver", "AS_point_1.alpha")

prob.model.add_subsystem("fuel_vol_delta", WingboxFuelVolDelta(surface=surf_dict))
prob.model.connect("wing.struct_setup.fuel_vols", "fuel_vol_delta.fuel_vols")
prob.model.connect("AS_point_0.fuelburn", "fuel_vol_delta.fuelburn")

if surf_dict["distributed_fuel_weight"]:
    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_0.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_0.coupled.wing.struct_states.fuel_mass")
    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_1.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_1.coupled.wing.struct_states.fuel_mass")

comp = om.ExecComp("fuel_diff = (fuel_mass - fuelburn) / fuelburn", units="kg")
prob.model.add_subsystem("fuel_diff", comp, promotes_inputs=["fuel_mass"], promotes_outputs=["fuel_diff"])
prob.model.connect("AS_point_0.fuelburn", "fuel_diff.fuelburn")

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-2

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = _RUN_OUT_DIR

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# FULL DV PATH REFERENCE for this blueprint (subsystem = "wing"):
#
#   PATH                          SURFACE DICT KEY      DESCRIPTION
#   "alpha"                       (none needed)         Cruise AoA [deg]
#                                                       Add if you want cruise trim as a DV
#   "alpha_maneuver"              (none needed)         Maneuver AoA [deg]
#   "fuel_mass"                   (none needed)         Fuel mass [kg] — closes fuel loop
#   "wing.twist_cp"               "twist_cp"            Spanwise twist CPs [deg]
#   "wing.spar_thickness_cp"      "spar_thickness_cp"   Spar wall thickness CPs [m]
#   "wing.skin_thickness_cp"      "skin_thickness_cp"   Skin thickness CPs [m]
#   "wing.geometry.t_over_c_cp"   "t_over_c_cp"         Thickness-to-chord ratio CPs
#                                                       MUST use ".geometry." in the path
#                                                       "wing.t_over_c_cp" is WRONG here
#   "wing.chord_cp"               "chord_cp"            Chord scaling CPs
#                                                       Uncomment "chord_cp" in surf_dict first
#   "wing.xshear_cp"              "xshear_cp"           x-shear CPs [m] (generalized sweep)
#                                                       Uncomment "xshear_cp" in surf_dict first
#   "wing.zshear_cp"              "zshear_cp"           z-shear CPs [m] (generalized dihedral)
#                                                       Uncomment "zshear_cp" in surf_dict first

prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)
prob.model.add_design_var("wing.twist_cp", lower=-15.0, upper=15.0, scaler=0.1)
prob.model.add_design_var("wing.spar_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)
prob.model.add_design_var("wing.skin_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)
prob.model.add_design_var("wing.geometry.t_over_c_cp", lower=0.07, upper=0.2, scaler=10.0)  # .geometry. required
prob.model.add_design_var("alpha_maneuver", lower=-15.0, upper=15.0)
prob.model.add_design_var("fuel_mass", lower=0.0, upper=2e5, scaler=1e-5)
# prob.model.add_design_var("alpha", lower=-15.0, upper=15.0)          # Cruise AoA — add if needed
# prob.model.add_design_var("wing.chord_cp", lower=0.5, upper=3.0)
# prob.model.add_design_var("wing.xshear_cp", lower=-5.0, upper=5.0)
# prob.model.add_design_var("wing.zshear_cp", lower=-2.0, upper=2.0)

# --- Constraints ---
# FULL CONSTRAINT PATH REFERENCE:
#   "AS_point_0.CL"                     — cruise lift coefficient (equals target CL)
#   "AS_point_0.L_equals_W"             — lift-equals-weight at cruise, equals=0
#   "AS_point_1.L_equals_W"             — lift-equals-weight at maneuver, equals=0
#   "AS_point_1.wing_perf.failure"      — structural failure at maneuver point, upper=0
#   "fuel_vol_delta.fuel_vol_delta"     — fuel volume constraint: box must hold fuel, lower=0
#   "fuel_diff"                         — fuel mass consistency: fuel_mass == fuelburn, equals=0

prob.model.add_constraint("AS_point_0.CL", equals=0.5)
prob.model.add_constraint("AS_point_1.L_equals_W", equals=0.0)
prob.model.add_constraint("AS_point_1.wing_perf.failure", upper=0.0)
prob.model.add_constraint("fuel_vol_delta.fuel_vol_delta", lower=0.0)
prob.model.add_constraint("fuel_diff", equals=0.0)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.model.AS_point_0.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=30, use_aitken=True)
prob.model.AS_point_1.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=30, use_aitken=True)
prob.run_driver()

print("\n--- Aerostructural Wingbox Results ---")
print(f"Final fuel burn:      {prob.get_val('AS_point_0.fuelburn')[0]:.4f} [kg]")
print(f"Final structural mass:{prob.get_val('wing.structural_mass')[0] / surf_dict['wing_weight_ratio']:.4f} [kg]")
print(f"Final twist_cp:       {prob.get_val('wing.twist_cp')}")

# =============================================================================
# 6. PLOTTING
# =============================================================================
try:
    fuelburn = prob.get_val('AS_point_0.fuelburn')[0]
    struct_mass = prob.get_val('wing.structural_mass')[0] / surf_dict['wing_weight_ratio']
    twist_cp_vals = prob.get_val('wing.twist_cp')
    spar_t = prob.get_val('wing.spar_thickness_cp') * 1e3   # convert to mm
    skin_t = prob.get_val('wing.skin_thickness_cp') * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(["Fuel Burn (kg)", "Struct. Mass (kg)"], [fuelburn, struct_mass],
                color=["steelblue", "tomato"])
    axes[0].set_title("Key Wingbox Results")
    axes[0].set_ylabel("Value")

    cp_idx = np.arange(len(twist_cp_vals))
    axes[1].plot(cp_idx, twist_cp_vals, "o-", color="green")
    axes[1].set_xlabel("Control Point")
    axes[1].set_ylabel("Twist (deg)")
    axes[1].set_title("Optimized Twist")
    axes[1].grid(True)

    axes[2].plot(np.arange(len(spar_t)), spar_t, "s-", color="purple", label="Spar")
    axes[2].plot(np.arange(len(skin_t)), skin_t, "^-", color="orange", label="Skin")
    axes[2].set_xlabel("Control Point")
    axes[2].set_ylabel("Thickness (mm)")
    axes[2].set_title("Optimized Thickness")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(_PLOTS_DIR, "aerostruct_wingbox_results.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {_PLOTS_DIR}")
except Exception as e:
    print(f"Plotting warning: {e}")