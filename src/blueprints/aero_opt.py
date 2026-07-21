import numpy as np
import openmdao.api as om
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# ---------------------------------------------------------------------------
# Absolute output paths — derived from __file__ so they resolve correctly
# regardless of the CWD when this script is executed as a subprocess.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR = os.path.join(_SRC_DIR, "openaerostruct_out")
_PLOTS_DIR = os.path.join(_OUT_DIR, "agent_plots")
_RUN_OUT_DIR = os.path.join(_OUT_DIR, "generated_run_out")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_RUN_OUT_DIR, exist_ok=True)

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)


def _isa_temperature(altitude_m):
    return max(288.15 - 0.0065 * altitude_m, 216.65)


def _isa_speed_of_sound(altitude_m):
    return np.sqrt(1.4 * 287.058 * _isa_temperature(altitude_m))


def _sutherland_mu(altitude_m):
    T = _isa_temperature(altitude_m)
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# wing_type controls which geometry is used:
#   "rect"      — rectangular planform; requires "span" and "root_chord"
#   "CRM"       — NASA Common Research Model built-in geometry; span/root_chord ignored;
#                 generate_mesh returns (mesh, twist_cp) — both are unpacked below
#
# num_y must be an odd number.
# num_twist_cp is used when we want to initialize twist as a design variable
# num_y/num_x are mesh resolution assumptions. Preserve them unless the user
# explicitly asks for mesh resolution, panel count, or discretization changes.
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 7,  # Number of spanwise panels (must be odd)
    "num_x": 2,  # Number of chordwise panels
    "wing_type": "CRM",  # "rect" or "CRM"
    "symmetry": True,
    "num_twist_cp": 5,
}
# === AGENT EDITABLE SECTION END ===

# generate_mesh returns a tuple (mesh, twist_cp) for CRM, or just mesh for rect.
# We handle both cases here so no code change is needed when switching wing_type.
_mesh_result = generate_mesh(mesh_dict)
if isinstance(_mesh_result, tuple):
    mesh, _crm_twist_cp = _mesh_result  # CRM: unpack both
else:
    mesh = _mesh_result  # rect: single array
    _crm_twist_cp = None

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# CRITICAL: Any geometry parameter you want to use as a design variable MUST be
# declared here in the surface dict BEFORE it can be added via add_design_var().
# If a DV is added in Section 4 without a matching key here, OpenMDAO will crash:
#   "Output not found for design variable 'wing.<var>'"
#
# For CRM wings: twist_cp should be initialized from the CRM geometry (_crm_twist_cp).
# For rect wings: twist_cp should be set manually, e.g. np.zeros(3).
#
# FULL GEOMETRY DV CATALOG:
# -----------------------------------------------
#   KEY           TYPE    DESCRIPTION
#   twist_cp      array   Spanwise twist B-spline CPs [deg]. Shape=(n_cp,).
#                         For CRM: use _crm_twist_cp (initialized from geometry).
#                         For rect: use np.zeros(N) for N control points.
#   chord_cp      array   Chord scaling B-spline CPs. Shape=(n_cp,).
#                         Scales the chord distribution spanwise.
#   xshear_cp     array   x-shear B-spline CPs [m]. Shape=(n_cp,).
#                         Generalized sweep — shifts leading/trailing edge x-coords.
#   zshear_cp     array   z-shear B-spline CPs [m]. Shape=(n_cp,).
#                         Generalized dihedral — shifts mesh z-coords spanwise.
#   taper         scalar  Taper ratio (tip_chord / root_chord). 1.0 = rectangular.
#                         NOTE: taper is defined per section for multi-section wings.
#   sweep         scalar  Leading-edge sweep angle [deg]. 0.0 = unswept.
#   dihedral      scalar  Dihedral angle [deg]. 0.0 = flat wing.
#
# NOTE: t_over_c_cp, CL0, CD0, k_lam, c_max_t, with_viscous, with_wave are
# aerodynamic solver parameters — do not add them as DVs.
# === AGENT EDITABLE SECTION START ===
surface = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "fem_model_type": "tube",
    "mesh": mesh,
    "twist_cp": _crm_twist_cp if _crm_twist_cp is not None else np.zeros(5),
    # --- Aerodynamic solver parameters — always keep these ---
    "CL0": 0.0,  # Lift coefficient at zero AoA
    "CD0": 0.015,  # Profile drag (zero-lift drag)
    "with_viscous": True,  # Include viscous drag
    "with_wave": False,  # Include wave drag (transonic/supersonic only)
    "k_lam": 0.05,  # Fraction of laminar flow (0.05 = 5%)
    "c_max_t": 0.303,  # Chordwise location of max thickness (NACA 4-digit: 0.303)
    "t_over_c_cp": np.array([0.15]),  # Thickness-to-chord ratio — affects viscous drag
    # --- Geometry DVs — uncomment each key you want to optimize ---
    # For CRM: use _crm_twist_cp for twist_cp initialization.
    # For rect: use np.zeros(N) or np.array([...]) manually.
    # After uncommenting here, also add the matching add_design_var() call in Section 4.
    # "chord_cp": np.ones(3),          # Chord scaling CPs (1.0 = no scaling)
    # "xshear_cp": np.zeros(3),        # x-shear CPs [m] — generalized sweep
    # "zshear_cp": np.zeros(3),        # z-shear CPs [m] — generalized dihedral
    # "taper": 1.0,                    # Taper ratio
    # "sweep": 0.0,                    # Sweep angle [deg]
    # "dihedral": 0.0,                 # Dihedral angle [deg]
}
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

# Flight condition. Derive speed and Reynolds number from Mach, altitude, and rho.
# === AGENT EDITABLE SECTION START ===
altitude = 11000.0  # Altitude [m]
Mach_number = 0.84
rho = 0.38
speed_of_sound = _isa_speed_of_sound(altitude)
v = Mach_number * speed_of_sound
re = rho * v / _sutherland_mu(altitude)
# === AGENT EDITABLE SECTION END ===

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=v, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("Mach_number", val=Mach_number)
indep_var_comp.add_output("speed_of_sound", val=speed_of_sound, units="m/s")
indep_var_comp.add_output("re", val=re, units="1/m")
indep_var_comp.add_output("rho", val=rho, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

# Geometry group — uses surface dict as defined above.
# Subsystem name is "wing" — ALL DV paths must be "wing.<var_name>".
# Example: "wing.twist_cp", "wing.taper", "wing.xshear_cp", etc.
name = surface["name"]
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(name, geom_group)

aero_group = AeroPoint(surfaces=[surface])
point_name = "flight_condition_0"
prob.model.add_subsystem(
    point_name,
    aero_group,
    promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"],
)

prob.model.connect(f"{name}.mesh", f"{point_name}.{name}.def_mesh")
prob.model.connect(f"{name}.mesh", f"{point_name}.aero_states.{name}_def_mesh")
prob.model.connect(f"{name}.t_over_c", f"{point_name}.{name}_perf.t_over_c")

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder(os.path.join(_RUN_OUT_DIR, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options["work_dir"] = _RUN_OUT_DIR

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# RULE: Any geometry DV listed here MUST also have a matching key declared in the
# surface dict (Section 2). Failing to do so will crash OpenMDAO at setup.
# EXCEPTION: "alpha" is always available — it requires no surface dict entry.
#
# FULL DV PATH REFERENCE for this blueprint (geometry subsystem = "wing"):
#
#   PATH                  SURFACE DICT KEY    DESCRIPTION
#   "alpha"               (none needed)       Angle of attack [deg]
#   "wing.twist_cp"       "twist_cp"          Spanwise twist B-spline CPs [deg]
#                                             For CRM: initialize with _crm_twist_cp
#                                             For rect: initialize with np.zeros(N)
#   "wing.chord_cp"       "chord_cp"          Chord scaling B-spline CPs
#   "wing.xshear_cp"      "xshear_cp"         x-shear CPs [m] (generalized sweep)
#   "wing.zshear_cp"      "zshear_cp"         z-shear CPs [m] (generalized dihedral)
#   "wing.taper"          "taper"             Taper ratio (scalar)
#   "wing.sweep"          "sweep"             Sweep angle [deg] (scalar)
#   "wing.dihedral"       "dihedral"          Dihedral angle [deg] (scalar)
#
# Span and area are NOT design variables — use constraints instead:
#   Span constraint:  "wing.mesh.stretch.span"               (upper/lower bounds)
#   Area constraint:  "flight_condition_0.wing_perf.S_ref"   (equals/lower)

# prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
# prob.model.add_design_var("wing.chord_cp", lower=0.5, upper=3.0)
# prob.model.add_design_var("wing.xshear_cp", lower=-5.0, upper=5.0)
# prob.model.add_design_var("wing.zshear_cp", lower=-2.0, upper=2.0)
# prob.model.add_design_var("wing.taper", lower=0.1, upper=1.5)
# prob.model.add_design_var("wing.sweep", lower=-30.0, upper=30.0)
# prob.model.add_design_var("wing.dihedral", lower=-10.0, upper=10.0)

# --- Constraints ---
# FULL CONSTRAINT PATH REFERENCE:
#   "flight_condition_0.wing_perf.CL"       — lift coefficient (equals or bounds)
#   "flight_condition_0.wing_perf.CD"       — drag coefficient
#   "flight_condition_0.wing_perf.S_ref"    — reference area [m^2]
#   "wing.mesh.stretch.span"                — wing span [m]

prob.model.add_constraint(f"{point_name}.wing_perf.CL", equals=0.5)
# prob.model.add_constraint(f"{point_name}.wing_perf.S_ref", lower=30.0)   # area constraint
# prob.model.add_constraint("wing.mesh.stretch.span", upper=12.0)          # span constraint

# --- Objective ---
# FULL OBJECTIVE PATH REFERENCE:
#   "flight_condition_0.wing_perf.CD"   — drag coefficient (minimize drag)
#   "flight_condition_0.wing_perf.CL"   — lift coefficient (maximize → scaler=-1)
prob.model.add_objective(f"{point_name}.wing_perf.CD", scaler=1e4)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Optimization Results ---")
print(f"Final Alpha: {prob.get_val('alpha', units='deg')[0]:.4f} deg")
print(f"Final CL:    {prob.get_val(f'{point_name}.wing_perf.CL')[0]:.4f}")
print(f"Final CD:    {prob.get_val(f'{point_name}.wing_perf.CD')[0]:.6f}")
print("\n--- Aerodynamic Bookkeeping ---")
print("OAS reports CL = CL1 + CL0 and CD = CDi + CDv + CDw + CD0.")
print(f"CL0={surface['CL0']:.6f}, CD0={surface['CD0']:.6f}")
print(
    f"with_viscous={surface['with_viscous']}, with_wave={surface['with_wave']}, "
    f"S_ref_type='{surface['S_ref_type']}'"
)

# =============================================================================
# 6. PLOTTING
# =============================================================================
try:
    if "twist_cp" in surface:
        twist_cp_vals = prob.get_val("wing.twist_cp")
        fig_twist, ax_twist = plt.subplots(figsize=(8, 4))
        ax_twist.plot(np.arange(len(twist_cp_vals)), twist_cp_vals, "o-", color="black")
        ax_twist.set_xlabel("Control Point")
        ax_twist.set_ylabel("Twist [deg]")
        ax_twist.set_title("Optimized Twist Distribution")
        ax_twist.grid(True)
        fig_twist.tight_layout()
        fig_twist.savefig(
            os.path.join(_PLOTS_DIR, "aero_opt_twist_distribution.png"),
            bbox_inches="tight",
        )
        plt.close(fig_twist)

    if "chord_cp" in surface:
        chord_cp_vals = prob.get_val("wing.chord_cp")
        fig_chord, ax_chord = plt.subplots(figsize=(8, 4))
        ax_chord.plot(np.arange(len(chord_cp_vals)), chord_cp_vals, "o-", color="black")
        ax_chord.set_xlabel("Control Point")
        ax_chord.set_ylabel("Chord Scale")
        ax_chord.set_title("Optimized Chord Control Points")
        ax_chord.grid(True)
        fig_chord.tight_layout()
        fig_chord.savefig(
            os.path.join(_PLOTS_DIR, "aero_opt_chord_distribution.png"),
            bbox_inches="tight",
        )
        plt.close(fig_chord)

    _mesh_out = prob.get_val(f"{point_name}.wing.def_mesh", units="m")
    mesh_x = _mesh_out[:, :, 0]
    mesh_y = _mesh_out[:, :, 1]
    fig_wing, ax_wing = plt.subplots(figsize=(8, 4))
    for i in range(mesh_x.shape[0]):
        ax_wing.plot(mesh_y[i, :], mesh_x[i, :], color="C0", lw=1)
    for j in range(mesh_x.shape[1]):
        ax_wing.plot(mesh_y[:, j], mesh_x[:, j], color="C0", lw=1)
    ax_wing.set_aspect("equal")
    ax_wing.set_xlabel("Span (m)")
    ax_wing.set_ylabel("Chord (m)")
    ax_wing.set_title("Optimized Wing Mesh")
    fig_wing.tight_layout()
    fig_wing.savefig(
        os.path.join(_PLOTS_DIR, "aero_opt_wing_planform.png"), bbox_inches="tight"
    )
    plt.close(fig_wing)
    print(f"Plot saved to {_PLOTS_DIR}")
except Exception as e:
    print(f"Plotting warning: {e}")
