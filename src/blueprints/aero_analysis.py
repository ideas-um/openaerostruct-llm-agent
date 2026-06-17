import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import openmdao.api as om
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# import OpenAeroStruct modules
from openaerostruct.geometry.utils import generate_mesh
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

            results.append(
                {
                    "Mach": round(M, 1),
                    "Alpha": a,
                    "CL": CL,
                    "CD": CD,
                    "L/D": CL / CD if CD != 0 else 0,
                }
            )

    df = pd.DataFrame(results)
    csv_path = os.path.join(_OUT_DIR, "OptimizedWing_Polars.csv")
    df.to_csv(csv_path, index=False)
    print(f"Analysis complete. Saved data to {csv_path}")

    # =============================================================================
    # 6. PLOTTING — plots must go to _PLOTS_DIR for the app to display them
    # =============================================================================
    try:
        df["Mach"] = df["Mach"].astype(str)

        fig_ld = px.line(
            df,
            x="Alpha",
            y="L/D",
            color="Mach",
            markers=True,
            title="Lift-to-Drag Ratio (L/D) vs Angle of Attack",
            labels={"Alpha": "Angle of Attack (deg)", "L/D": "Lift / Drag"},
        )
        fig_ld.update_layout(template="plotly_white", height=600, width=900)
        fig_ld.write_image(os.path.join(_PLOTS_DIR, "LD_vs_Alpha.png"))

        fig_polar = px.line(
            df,
            x="CD",
            y="CL",
            color="Mach",
            markers=True,
            title="Drag Polars (CL vs CD)",
            labels={"CD": "Drag Coefficient (CD)", "CL": "Lift Coefficient (CL)"},
        )
        fig_polar.update_layout(template="plotly_white", height=600, width=900)
        fig_polar.write_image(os.path.join(_PLOTS_DIR, "Drag_Polars.png"))

        fig_cl_alpha = px.line(
            df,
            x="Alpha",
            y="CL",
            color="Mach",
            markers=True,
            title="Lift Coefficient (CL) vs Angle of Attack",
            labels={"Alpha": "Angle of Attack (deg)", "CL": "Lift Coefficient (CL)"},
        )
        fig_cl_alpha.update_layout(template="plotly_white", height=600, width=900)
        fig_cl_alpha.write_image(os.path.join(_PLOTS_DIR, "CL_vs_Alpha.png"))
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

        x_center = np.concatenate((y_center, -y_center[::-1]))
        chord_center = np.concatenate((chord_center, chord_center[::-1]))
        Cl_full = np.concatenate((Cl, Cl[::-1]))

        CL_total = prob.get_val("flight_condition_0.wing_perf.CL")[0]
        Sref = prob.get_val("flight_condition_0.wing_perf.S_ref", units="m**2")[0]
        semi_span = mesh_out[0, -1, 1] - mesh_out[0, 0, 1]

        cl_chord_max = 2 * CL_total * Sref / (np.pi * semi_span)
        y_ellipse = np.linspace(-semi_span, semi_span, 100)
        Cl_chord_ellipse = cl_chord_max * np.sqrt(1 - (y_ellipse / semi_span) ** 2)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_ellipse,
                y=Cl_chord_ellipse,
                mode="lines",
                name="Elliptical",
                line=dict(color="red", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_center,
                y=Cl_full,
                mode="lines",
                name="Wing Cl",
                line=dict(color="green", width=2),
            )
        )
        fig.update_layout(
            title=f"Spanwise Lift Distribution (Mach {trim_mach}, Alpha {trim_alpha}°)",
            xaxis_title="Span (m)",
            yaxis_title="Sectional Lift Coefficient (Cl)",
            template="plotly_white",
            height=600,
            width=900,
        )
        fig.write_image(
            os.path.join(_PLOTS_DIR, "Sectional_Lift_Distribution_Trim.png")
        )
        print(f"Done! Plots saved to {_PLOTS_DIR}")
    except Exception as e:
        print(f"Spanwise distribution plotting warning: {e}")
