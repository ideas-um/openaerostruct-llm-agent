import numpy as np
import openmdao.api as om
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.integration.multipoint_comps import MultiCD

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


def _plot_path(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return os.path.join(_PLOTS_DIR, safe + ".png")


# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# CRM mesh is built-in — span and root_chord are not required for "CRM".
# num_twist_cp controls how many twist control points are initialized from the CRM geometry.
# generate_mesh returns TWO values for CRM — always unpack as (mesh, twist_cp).
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 5,
    "num_x": 3,
    "wing_type": "CRM",
    "symmetry": True,
    "num_twist_cp": 5,
    "span_cos_spacing": 0.0,
}
# === AGENT EDITABLE SECTION END ===

mesh, twist_cp = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# CRITICAL: Any geometry parameter you want to use as a design variable MUST be
# declared here in the surface dict BEFORE it can be added via add_design_var().
# twist_cp is pre-included (initialized from CRM geometry) — do not remove it.
#
# FULL GEOMETRY DV CATALOG:
# -----------------------------------------------
#   KEY           TYPE    DESCRIPTION
#   twist_cp      array   Spanwise twist B-spline CPs [deg]. Shape=(n_cp,).
#                         Required — initialized from CRM geometry above.
#   chord_cp      array   Chord scaling B-spline CPs. Shape=(n_cp,).
#                         Scales the chord distribution spanwise.
#   xshear_cp     array   x-shear B-spline CPs [m]. Shape=(n_cp,).
#                         Generalized sweep — shifts leading/trailing edge x-coords.
#   zshear_cp     array   z-shear B-spline CPs [m]. Shape=(n_cp,).
#                         Generalized dihedral — shifts mesh z-coords spanwise.
#   taper         scalar  Taper ratio (tip_chord / root_chord). 1.0 = rectangular.
#   sweep         scalar  Leading-edge sweep angle [deg]. 0.0 = unswept.
#   dihedral      scalar  Dihedral angle [deg]. 0.0 = flat wing.
#
# NOTE: CL0, CD0, k_lam, t_over_c_cp, c_max_t, with_viscous, with_wave are
# aerodynamic solver parameters — do not remove them and do not add them as DVs.
# === AGENT EDITABLE SECTION START ===
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh,
    # Required — initialized from CRM geometry; do not remove
    "twist_cp": twist_cp,
    # --- Aerodynamic solver parameters — always keep these ---
    "CL0": 0.0,  # Lift coefficient at zero AoA
    "CD0": 0.015,  # Profile drag (zero-lift drag)
    "k_lam": 0.05,  # Fraction of laminar flow
    "t_over_c_cp": np.array([0.15]),  # Thickness-to-chord ratio — affects viscous drag
    "c_max_t": 0.303,  # Chordwise location of max thickness
    "with_viscous": True,  # Include viscous drag
    "with_wave": False,  # Include wave drag (transonic/supersonic only)
    # --- Optional geometry DVs — uncomment to activate ---
    # After uncommenting here, also add the matching add_design_var() call in Section 4.
    # Use path "wing_geom.<var>" (NOT "wing.<var>") — see critical note below.
    # "chord_cp": np.ones(5),          # Chord scaling CPs (1.0 = no scaling)
    # "xshear_cp": np.zeros(5),        # x-shear CPs [m] — generalized sweep
    # "zshear_cp": np.zeros(5),        # z-shear CPs [m] — generalized dihedral
    # "taper": 1.0,                    # Taper ratio
    # "sweep": 0.0,                    # Sweep angle [deg]
    # "dihedral": 0.0,                 # Dihedral angle [deg]
}
# === AGENT EDITABLE SECTION END ===

surfaces = [surf_dict]
n_points = 2

# =============================================================================
# 3. PROBLEM SETUP (MULTIPOINT)
# =============================================================================
# CRITICAL NAMING RULE:
#   The geometry subsystem is added as "wing_geom" (surface["name"] + "_geom"),
#   NOT as "wing". This means ALL geometry DV paths must use "wing_geom.<var>",
#   e.g. "wing_geom.twist_cp", "wing_geom.taper", "wing_geom.xshear_cp".
#   Using "wing.<var>" will fail — that path does not exist in this blueprint.
#
# alpha is a vector of length n_points — one AoA value per flight condition.
# === AGENT EDITABLE SECTION START ===
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output(
    "alpha", val=np.ones(n_points) * 6.64, units="deg"
)  # shape=(n_points,)
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
# === AGENT EDITABLE SECTION END ===

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

for surface in surfaces:
    name = surface["name"]
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(name + "_geom", geom_group)  # subsystem = "wing_geom"

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
        prob.model.connect(
            name + "_geom.mesh", point_name + ".aero_states." + name + "_def_mesh"
        )
        prob.model.connect(
            name + "_geom.t_over_c", point_name + "." + name + "_perf." + "t_over_c"
        )

prob.model.add_subsystem(
    "multi_CD", MultiCD(n_points=n_points), promotes_outputs=["CD"]
)

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
# CRITICAL: Geometry subsystem is "wing_geom", NOT "wing".
# All geometry DV paths MUST use the prefix "wing_geom.<var>".
#
# FULL DV PATH REFERENCE for this blueprint:
#
#   PATH                      SURFACE DICT KEY    DESCRIPTION
#   "alpha"                   (none needed)       AoA vector [deg], shape=(n_points,)
#   "wing_geom.twist_cp"      "twist_cp"          Spanwise twist CPs [deg]
#   "wing_geom.chord_cp"      "chord_cp"          Chord scaling CPs
#   "wing_geom.xshear_cp"     "xshear_cp"         x-shear CPs [m] (generalized sweep)
#   "wing_geom.zshear_cp"     "zshear_cp"         z-shear CPs [m] (generalized dihedral)
#   "wing_geom.taper"         "taper"             Taper ratio (scalar)
#   "wing_geom.sweep"         "sweep"             Sweep angle [deg] (scalar)
#   "wing_geom.dihedral"      "dihedral"          Dihedral angle [deg] (scalar)

prob.model.add_design_var("alpha", lower=-15, upper=15)
prob.model.add_design_var("wing_geom.twist_cp", lower=-5, upper=8)
# prob.model.add_design_var("wing_geom.chord_cp", lower=0.5, upper=3.0)
# prob.model.add_design_var("wing_geom.xshear_cp", lower=-5.0, upper=5.0)
# prob.model.add_design_var("wing_geom.zshear_cp", lower=-2.0, upper=2.0)
# prob.model.add_design_var("wing_geom.taper", lower=0.1, upper=1.5)
# prob.model.add_design_var("wing_geom.sweep", lower=-30.0, upper=30.0)
# prob.model.add_design_var("wing_geom.dihedral", lower=-10.0, upper=10.0)

# --- Constraints ---
# Per-point CL constraints — each AeroPoint has its own subsystem.
# FULL CONSTRAINT PATH REFERENCE:
#   "aero_point_0.wing_perf.CL"       — CL at flight condition 0
#   "aero_point_1.wing_perf.CL"       — CL at flight condition 1
#   "aero_point_0.wing_perf.S_ref"    — reference area at point 0 (NOT "wing.S_ref")
#   "aero_point_0.wing_perf.CD"       — drag at point 0
#   "CD"                              — total weighted drag (output of MultiCD)

for i in range(n_points):
    prob.model.add_constraint(f"aero_point_{i}.wing_perf.CL", equals=[0.45, 0.5][i])

# --- Objective ---
# "CD" is the sum of drag across all flight points (output of MultiCD component).
# FULL OBJECTIVE PATH REFERENCE:
#   "CD"                              — weighted sum of drag (MultiCD output)
#   "aero_point_0.wing_perf.CD"       — drag at a single flight point
prob.model.add_objective("CD", scaler=1e4)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Multipoint Optimization Results ---")
print(f"Final CD (Sum): {prob.get_val('CD')[0]:.6f}")
print(f"Final alpha:    {prob.get_val('alpha')}")
print(f"Final twist_cp: {prob.get_val('wing_geom.twist_cp')}")
for i in range(n_points):
    print(
        f"Point {i}: "
        f"CL={prob.get_val(f'aero_point_{i}.wing_perf.CL')[0]:.4f}, "
        f"CD={prob.get_val(f'aero_point_{i}.wing_perf.CD')[0]:.6f}"
    )
print("\n--- Aerodynamic Bookkeeping ---")
print("OAS reports CL = CL1 + CL0 and CD = CDi + CDv + CDw + CD0 at each point.")
print(f"CL0={surf_dict['CL0']:.6f}, CD0={surf_dict['CD0']:.6f}")
print(
    f"with_viscous={surf_dict['with_viscous']}, with_wave={surf_dict['with_wave']}, "
    f"S_ref_type='{surf_dict['S_ref_type']}'"
)

# =============================================================================
# 6. PLOTTING
# =============================================================================
try:
    alpha_vals = prob.get_val("alpha")
    CL_vals = [prob.get_val(f"aero_point_{i}.wing_perf.CL")[0] for i in range(n_points)]
    CD_vals = [prob.get_val(f"aero_point_{i}.wing_perf.CD")[0] for i in range(n_points)]
    CDi_vals = [
        prob.get_val(f"aero_point_{i}.wing_perf.CDi")[0] for i in range(n_points)
    ]
    CDv_vals = [
        prob.get_val(f"aero_point_{i}.wing_perf.CDv")[0] for i in range(n_points)
    ]
    CDw_vals = [
        prob.get_val(f"aero_point_{i}.wing_perf.CDw")[0] for i in range(n_points)
    ]
    twist_cp_vals = prob.get_val("wing_geom.twist_cp")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    point_labels = [f"Point {i}" for i in range(n_points)]
    axes[0].bar(point_labels, CL_vals, color="steelblue", label="CL", alpha=0.7)
    ax2 = axes[0].twinx()
    ax2.bar(point_labels, CD_vals, color="tomato", label="CD", alpha=0.5, width=0.4)
    axes[0].set_ylabel("CL", color="steelblue")
    ax2.set_ylabel("CD", color="tomato")
    axes[0].set_title("CL and CD per Flight Condition")

    cp_indices = np.arange(len(twist_cp_vals))
    axes[1].plot(cp_indices, twist_cp_vals, "o-", color="green")
    axes[1].set_xlabel("Control Point Index")
    axes[1].set_ylabel("Twist (deg)")
    axes[1].set_title("Optimized Twist Distribution")
    axes[1].grid(True)

    _mesh_out = prob.get_val("wing_geom.mesh", units="m")
    for i in range(_mesh_out.shape[0]):
        axes[2].plot(_mesh_out[i, :, 1], _mesh_out[i, :, 0], color="black", lw=1)
    for j in range(_mesh_out.shape[1]):
        axes[2].plot(_mesh_out[:, j, 1], _mesh_out[:, j, 0], color="black", lw=1)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("Spanwise y [m]")
    axes[2].set_ylabel("Chordwise x [m]")
    axes[2].set_title("Optimized Wing Planform")

    fig.tight_layout()
    fig.savefig(_plot_path("aero_multipoint_results"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    x = np.arange(n_points)
    width = 0.2
    fig_drag, ax_drag = plt.subplots(figsize=(8, 4))
    ax_drag.bar(x - 1.5 * width, CDi_vals, width, label="CDi", color="steelblue")
    ax_drag.bar(x - 0.5 * width, CDv_vals, width, label="CDv", color="seagreen")
    ax_drag.bar(x + 0.5 * width, CDw_vals, width, label="CDw", color="goldenrod")
    ax_drag.bar(
        x + 1.5 * width,
        np.ones(n_points) * surf_dict["CD0"],
        width,
        label="CD0",
        color="tomato",
    )
    ax_drag.set_xticks(x)
    ax_drag.set_xticklabels(point_labels)
    ax_drag.set_ylabel("Drag Coefficient")
    ax_drag.set_title(
        f"Drag Breakdown by Flight Condition ({surf_dict['S_ref_type']} S_ref)"
    )
    ax_drag.legend(frameon=False)
    fig_drag.tight_layout()
    fig_drag.savefig(
        _plot_path("aero_multipoint_drag_breakdown"), bbox_inches="tight", dpi=150
    )
    plt.close(fig_drag)
    print(f"Plot saved to {_PLOTS_DIR}")
except Exception as e:
    print(f"Plotting warning: {e}")
