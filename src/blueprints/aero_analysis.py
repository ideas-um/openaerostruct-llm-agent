import warnings
import os
import numpy as np
import pandas as pd
import openmdao.api as om
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# import OpenAeroStruct modules
from openaerostruct.geometry.utils import generate_mesh  # noqa: E402
from openaerostruct.geometry.geometry_group import Geometry  # noqa: E402
from openaerostruct.aerodynamics.aero_groups import AeroPoint  # noqa: E402

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

# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================


def plot_mesh(mesh, filename=None):
    """Function to plot the VLM mesh"""
    if filename is None:
        filename = os.path.join(_PLOTS_DIR, "mesh_analyzed.png")
    mesh_x = mesh[:, :, 0]
    mesh_y = mesh[:, :, 1]
    plt.figure(figsize=(6, 3))
    for i in range(mesh_x.shape[0]):
        plt.plot(mesh_y[i, :], mesh_x[i, :], color="C0", lw=1)
    for j in range(mesh_x.shape[1]):
        plt.plot(mesh_y[:, j], mesh_x[:, j], color="C0", lw=1)
    plt.axis("equal")
    plt.xlabel("Span (m)")
    plt.ylabel("Chord (m)")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # =============================================================================
    # 2. MESH GENERATION
    # =============================================================================
    # Modify these parameters to set the wing's baseline geometry.
    # For "rect" wings: span and root_chord are required.
    # For "CRM" wings: span and root_chord are not used — CRM geometry is built-in.
    # num_y must be an odd number.
    # === AGENT EDITABLE SECTION START ===
    mesh_dict = {
        "num_y": 19,  # Number of spanwise panels (must be odd)
        "num_x": 3,  # Number of chordwise panels
        "wing_type": "rect",  # "rect" or "CRM"
        "symmetry": True,
        "span": 25.52,  # Full wingspan [m]
        "root_chord": 2.83,  # Root chord [m] — only used for "rect" wing_type
        "span_cos_spacing": 0.0,
        "chord_cos_spacing": 0.0,
    }
    # === AGENT EDITABLE SECTION END ===

    mesh = generate_mesh(mesh_dict)

    # =============================================================================
    # 3. SURFACE DEFINITION
    # =============================================================================
    # This script performs ANALYSIS (run_model).
    # Geometry parameters declared here define the fixed wing shape for the sweep.
    # To test a different geometry, activate the desired keys below and set values.
    #
    # FULL GEOMETRY DV CATALOG:
    # -----------------------------------------------
    # Any key declared here becomes part of the wing geometry passed to the Geometry
    # group. For analysis sweeps these are fixed values, not optimized.
    #
    #   KEY               TYPE        DESCRIPTION
    #   twist_cp          array       Spanwise twist B-spline CPs [deg]. Shape = (n_cp,).
    #                                 Controls washout/washin along the span.
    #   chord_cp          array       Chord scaling B-spline CPs. Shape = (n_cp,).
    #                                 Scales the chord distribution spanwise.
    #   xshear_cp         array       Spanwise x-shear (sweep) B-spline CPs [m]. Shape = (n_cp,).
    #                                 Generalized sweep — shifts LE/TE x-coords spanwise.
    #   zshear_cp         array       Spanwise z-shear (dihedral) B-spline CPs [m]. Shape = (n_cp,).
    #                                 Generalized dihedral — shifts mesh z-coords spanwise.
    #   taper             scalar      Taper ratio (tip_chord / root_chord). 1.0 = rectangular.
    #   sweep             scalar      Leading-edge sweep angle [deg]. 0.0 = unswept.
    #   dihedral          scalar      Dihedral angle [deg]. 0.0 = flat wing.
    #   t_over_c_cp       array       Thickness-to-chord ratio B-spline CPs. Shape = (n_cp,).
    #                                 Used for viscous/wave drag calculation — do not remove.
    #
    # NOTE: CL0, CD0, k_lam, c_max_t, with_viscous, with_wave are aerodynamic solver
    # parameters, not geometry DVs. They affect drag bookkeeping and should not be removed.
    # === AGENT EDITABLE SECTION START ===
    surface = {
        "name": "wing",
        "symmetry": True,
        "S_ref_type": "wetted",
        # --- Active geometry (modify values to test different shapes) ---
        "twist_cp": np.array([0.0, 0.0]),  # Spanwise twist [deg], 2 control points
        "t_over_c_cp": np.array([0.12]),  # Thickness-to-chord ratio
        "taper": 1.0,  # Taper ratio (1.0 = rectangular)
        "sweep": 0.0,  # Sweep angle [deg]
        "dihedral": 0.0,  # Dihedral angle [deg]
        # --- Optional geometry modifiers — uncomment to activate ---
        # "chord_cp": np.ones(2),                 # Chord B-spline CPs (1.0 = no scaling)
        # "xshear_cp": np.zeros(2),               # x-shear CPs [m] — generalized sweep
        # "zshear_cp": np.zeros(2),               # z-shear CPs [m] — generalized dihedral
        "mesh": mesh,  # Mesh generated above — do not remove
        # --- Aerodynamic solver parameters — do not remove ---
        "CL0": 0.0,  # Lift coefficient at zero AoA
        "CD0": 0.005,  # Profile drag coefficient (zero-lift drag)
        "k_lam": 0.05,  # Fraction of laminar flow (0.05 = 5%)
        "c_max_t": 0.303,  # Chordwise location of max thickness (NACA 4-digit: 0.303)
        "with_viscous": True,  # Include viscous drag in the analysis
        "with_wave": False,  # Include wave drag (transonic/supersonic only)
    }
    # === AGENT EDITABLE SECTION END ===

    # =============================================================================
    # 4. PROBLEM SETUP
    # =============================================================================
    prob = om.Problem()

    indep_var_comp = om.IndepVarComp()
    # These are initial/placeholder values — they are overridden in the sweep loop below.
    # DO NOT compute re using surface["root_chord"] — that key does not exist in surface.
    # Compute re as: rho * v / 1.81e-5  (units: 1/m, i.e. per unit length)
    # === AGENT EDITABLE SECTION START ===
    indep_var_comp.add_output("v", val=100.0, units="m/s")  # Overridden in sweep
    indep_var_comp.add_output("alpha", val=0.0, units="deg")  # Overridden in sweep
    indep_var_comp.add_output("Mach_number", val=0.3)  # Overridden in sweep
    indep_var_comp.add_output("re", val=1e6, units="1/m")  # Overridden in sweep
    indep_var_comp.add_output(
        "rho", val=1.225, units="kg/m**3"
    )  # Set to match flight condition
    indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
    # === AGENT EDITABLE SECTION END ===

    prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

    name = surface["name"]
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(name, geom_group)

    aero_group = AeroPoint(surfaces=[surface], rotational=True)
    point_name = "flight_condition_0"
    prob.model.add_subsystem(
        point_name,
        aero_group,
        promotes_inputs=[
            "v",
            "alpha",
            "beta",
            "omega",
            "Mach_number",
            "re",
            "rho",
            "cg",
        ],
    )

    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
    prob.model.connect(
        name + ".mesh", point_name + ".aero_states." + name + "_def_mesh"
    )
    prob.model.connect(
        name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c"
    )

    prob.setup()

    # =============================================================================
    # 5. ANALYSIS SWEEP
    # =============================================================================
    # Set the Mach numbers and alpha range to sweep over.
    # Re is computed from rho and v — do NOT use surface["root_chord"], it doesn't exist.
    # Formula: re_val = rho * v_val / 1.81e-5  [units: 1/m]
    # === AGENT EDITABLE SECTION START ===
    rho_val = 1.225  # Air density [kg/m^3] — change for cruise altitude
    mach_range = np.arange(0.1, 0.8, 0.1)  # Mach numbers to sweep
    alpha_range = np.arange(-10, 16, 1)  # Angle of attack range [deg]
    # === AGENT EDITABLE SECTION END ===

    results = []
    print("Running Aerodynamic Analysis Sweep...")
    for M in mach_range:
        for a in alpha_range:
            v_val = M * 340.0
            re_val = rho_val * v_val / 1.81e-5  # Re per unit length [1/m]
            prob.set_val("Mach_number", M)
            prob.set_val("v", v_val, units="m/s")
            prob.set_val("re", re_val, units="1/m")
            prob.set_val("rho", rho_val, units="kg/m**3")
            prob.set_val("alpha", a, units="deg")
            prob.run_model()

            CL = prob.get_val("flight_condition_0.wing_perf.CL")[0]
            CD = prob.get_val("flight_condition_0.wing_perf.CD")[0]
            CL1 = prob.get_val("flight_condition_0.wing_perf.CL1")[0]
            CDi = prob.get_val("flight_condition_0.wing_perf.CDi")[0]
            CDv = prob.get_val("flight_condition_0.wing_perf.CDv")[0]
            CDw = prob.get_val("flight_condition_0.wing_perf.CDw")[0]
            Sref = prob.get_val("flight_condition_0.wing_perf.S_ref", units="m**2")[0]

            results.append(
                {
                    "Mach": round(M, 1),
                    "Alpha": a,
                    "CL": CL,
                    "CL1": CL1,
                    "CD": CD,
                    "CDi": CDi,
                    "CDv": CDv,
                    "CDw": CDw,
                    "CD0": surface["CD0"],
                    "S_ref": Sref,
                    "L/D": CL / CD if CD != 0 else 0,
                }
            )

    df = pd.DataFrame(results)
    csv_path = os.path.join(_OUT_DIR, "OptimizedWing_Polars.csv")
    df.to_csv(csv_path, index=False)
    print(f"Analysis complete. Saved data to {csv_path}")
    print("\n--- Aerodynamic Bookkeeping ---")
    print("OAS reports CL = CL1 + CL0 and CD = CDi + CDv + CDw + CD0.")
    print(f"CL0={surface['CL0']:.6f}, CD0={surface['CD0']:.6f}")
    print(
        f"with_viscous={surface['with_viscous']}, with_wave={surface['with_wave']}, "
        f"S_ref_type='{surface['S_ref_type']}'"
    )
    print(
        "Viscous drag uses k_lam, t_over_c, c_max_t, Reynolds number, Mach, "
        "and the same S_ref convention."
    )

    # =============================================================================
    # 6. PLOTTING — plots must go to _PLOTS_DIR for the app to display them
    # =============================================================================
    try:
        mach_values = sorted(df["Mach"].unique())

        fig_ld, ax_ld = plt.subplots(figsize=(8, 5))
        for mach in mach_values:
            data = df[df["Mach"] == mach]
            ax_ld.plot(data["Alpha"], data["L/D"], marker="o", label=f"Mach {mach}")
        ax_ld.set_xlabel("Angle of Attack [deg]")
        ax_ld.set_ylabel("Lift / Drag")
        ax_ld.set_title("Lift-to-Drag Ratio vs Angle of Attack")
        ax_ld.legend(frameon=False)
        ax_ld.grid(True)
        fig_ld.tight_layout()
        fig_ld.savefig(os.path.join(_PLOTS_DIR, "LD_vs_Alpha.png"), bbox_inches="tight")
        plt.close(fig_ld)

        fig_polar, ax_polar = plt.subplots(figsize=(8, 5))
        for mach in mach_values:
            data = df[df["Mach"] == mach]
            ax_polar.plot(data["CD"], data["CL"], marker="o", label=f"Mach {mach}")
        ax_polar.set_xlabel("Total Drag Coefficient CD")
        ax_polar.set_ylabel("Total Lift Coefficient CL")
        ax_polar.set_title("Drag Polar")
        ax_polar.legend(frameon=False)
        ax_polar.grid(True)
        fig_polar.tight_layout()
        fig_polar.savefig(
            os.path.join(_PLOTS_DIR, "Drag_Polars.png"), bbox_inches="tight"
        )
        plt.close(fig_polar)

        fig_cl_alpha, ax_cl_alpha = plt.subplots(figsize=(8, 5))
        for mach in mach_values:
            data = df[df["Mach"] == mach]
            ax_cl_alpha.plot(
                data["Alpha"], data["CL"], marker="o", label=f"Mach {mach}"
            )
        ax_cl_alpha.set_xlabel("Angle of Attack [deg]")
        ax_cl_alpha.set_ylabel("Lift Coefficient CL")
        ax_cl_alpha.set_title("Lift Coefficient vs Angle of Attack")
        ax_cl_alpha.legend(frameon=False)
        ax_cl_alpha.grid(True)
        fig_cl_alpha.tight_layout()
        fig_cl_alpha.savefig(
            os.path.join(_PLOTS_DIR, "CL_vs_Alpha.png"), bbox_inches="tight"
        )
        plt.close(fig_cl_alpha)
    except Exception as e:
        print(f"Plotting warning: {e}")

    # =============================================================================
    # 7. SPANWISE LIFT DISTRIBUTION
    # =============================================================================
    # Run model at a single trim condition to extract spanwise Cl distribution.
    # === AGENT EDITABLE SECTION START ===
    trim_mach = 0.3
    trim_alpha = 4.0
    trim_rho = 1.225
    # === AGENT EDITABLE SECTION END ===

    v_trim = trim_mach * 340.0
    prob.set_val("Mach_number", trim_mach)
    prob.set_val("v", v_trim, units="m/s")
    prob.set_val("re", trim_rho * v_trim / 1.81e-5, units="1/m")
    prob.set_val("rho", trim_rho, units="kg/m**3")
    prob.set_val("alpha", trim_alpha, units="deg")
    prob.run_model()

    try:
        mesh_out = prob.get_val("flight_condition_0.wing.def_mesh", units="m")
        y_vertices = mesh_out[0, :, 1]
        y_center = 0.5 * (y_vertices[:-1] + y_vertices[1:])
        Cl = prob.get_val("flight_condition_0.wing_perf.Cl")
        chord_edge = prob.get_val("flight_condition_0.wing.chords", units="m")
        chord_center = 0.5 * (chord_edge[:-1] + chord_edge[1:])

        y_full = np.concatenate((y_center, -y_center[::-1]))
        lift_shape = np.concatenate((Cl * chord_center, (Cl * chord_center)[::-1]))
        order = np.argsort(y_full)
        y_full = y_full[order]
        lift_shape = lift_shape[order]

        semi_span = np.max(np.abs(y_full))

        y_ellipse = np.linspace(-semi_span, semi_span, 200)
        ellipse = np.sqrt(np.maximum(0.0, 1.0 - (y_ellipse / semi_span) ** 2))
        lift_area = np.trapezoid(lift_shape, y_full)
        ellipse_area = np.trapezoid(ellipse, y_ellipse)
        lift_shape_normalized = lift_shape / lift_area
        ellipse_normalized = ellipse / ellipse_area

        fig_lift, ax_lift = plt.subplots(figsize=(8, 5))
        ax_lift.plot(
            y_full,
            lift_shape_normalized,
            color="black",
            linewidth=1.5,
            label="OAS Cl * chord",
        )
        ax_lift.plot(
            y_ellipse,
            ellipse_normalized,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Elliptical",
        )
        ax_lift.set_xlabel("Span [m]")
        ax_lift.set_ylabel("Area-normalized Cl * chord [1/m]")
        ax_lift.set_title(
            f"Normalized Spanwise Lift Shape (Mach {trim_mach}, alpha {trim_alpha} deg)"
        )
        ax_lift.legend(frameon=False)
        ax_lift.grid(True)
        fig_lift.tight_layout()
        fig_lift.savefig(
            os.path.join(_PLOTS_DIR, "Sectional_Lift_Distribution_Trim.png"),
            bbox_inches="tight",
        )
        plt.close(fig_lift)

        print(f"Done! Plots saved to {_PLOTS_DIR}")
    except Exception as e:
        print(f"Spanwise distribution plotting warning: {e}")
