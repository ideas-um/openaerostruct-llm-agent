import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# import OpenAeroStruct modules
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# Ensure unified directory for saving results exists
output_dir = "src/openaerostruct_out/"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================

def plot_mesh(mesh, filename=os.path.join(output_dir, "mesh_analyzed.png")):
    """Function to plot the VLM mesh"""
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
        "num_y": 19,            # Number of spanwise panels (must be odd)
        "num_x": 3,             # Number of chordwise panels
        "wing_type": "rect",    # "rect" or "CRM"
        "symmetry": True,
        "span": 25.52,          # Full wingspan [m]
        "root_chord": 2.83,     # Root chord [m] — only used for "rect" wing_type
        "span_cos_spacing": 0.0,
        "chord_cos_spacing": 0.0,
    }
    # === AGENT EDITABLE SECTION END ===

    mesh = generate_mesh(mesh_dict)

    # =============================================================================
    # 3. SURFACE DEFINITION
    # =============================================================================
    # Modify aerodynamic and geometric properties here.
    # IMPORTANT: twist_cp, chord_cp, taper, sweep, dihedral are geometry modifiers.
    # Include them here if you want them reflected in the analysis.
    # === AGENT EDITABLE SECTION START ===
    surface = {
        "name": "wing",
        "symmetry": True,
        "S_ref_type": "wetted",
        "twist_cp": np.array([0.0, 0.0]),       # Spanwise twist [deg], 2 control points
        "t_over_c_cp": np.array([0.12]),         # Thickness-to-chord ratio
        "taper": 1.0,                            # Taper ratio (1.0 = rectangular)
        "sweep": 0.0,                            # Sweep angle [deg]
        "dihedral": 0.0,                         # Dihedral angle [deg]
        "mesh": mesh,
        "CL0": 0.0,
        "CD0": 0.005,
        "k_lam": 0.05,
        "c_max_t": 0.303,
        "with_viscous": True,
        "with_wave": False,
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
    indep_var_comp.add_output("v", val=100.0, units="m/s")       # Overridden in sweep
    indep_var_comp.add_output("alpha", val=0.0, units="deg")     # Overridden in sweep
    indep_var_comp.add_output("Mach_number", val=0.3)            # Overridden in sweep
    indep_var_comp.add_output("re", val=1e6, units="1/m")        # Overridden in sweep
    indep_var_comp.add_output("rho", val=1.225, units="kg/m**3") # Set to match flight condition
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
        promotes_inputs=["v", "alpha", "beta", "omega", "Mach_number", "re", "rho", "cg"],
    )

    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
    prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")
    prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

    prob.setup()

    # =============================================================================
    # 5. ANALYSIS SWEEP
    # =============================================================================
    # Set the Mach numbers and alpha range to sweep over.
    # Re is computed from rho and v — do NOT use surface["root_chord"], it doesn't exist.
    # Formula: re_val = rho * v_val / 1.81e-5  [units: 1/m]
    # === AGENT EDITABLE SECTION START ===
    rho_val = 1.225                         # Air density [kg/m^3] — change for cruise altitude
    mach_range = np.arange(0.1, 0.8, 0.1)  # Mach numbers to sweep
    alpha_range = np.arange(-10, 16, 1)    # Angle of attack range [deg]
    # === AGENT EDITABLE SECTION END ===

    results = []
    print("Running Aerodynamic Analysis Sweep...")
    for M in mach_range:
        for a in alpha_range:
            v_val = M * 340.0
            re_val = rho_val * v_val / 1.81e-5   # Re per unit length [1/m]
            prob.set_val("Mach_number", M)
            prob.set_val("v", v_val, units="m/s")
            prob.set_val("re", re_val, units="1/m")
            prob.set_val("rho", rho_val, units="kg/m**3")
            prob.set_val("alpha", a, units="deg")
            prob.run_model()

            CL = prob.get_val("flight_condition_0.wing_perf.CL")[0]
            CD = prob.get_val("flight_condition_0.wing_perf.CD")[0]

            results.append({
                "Mach": round(M, 1),
                "Alpha": a,
                "CL": CL,
                "CD": CD,
                "L/D": CL / CD if CD != 0 else 0,
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "OptimizedWing_Polars.csv")
    df.to_csv(csv_path, index=False)
    print(f"Analysis complete. Saved data to {csv_path}")

    # =============================================================================
    # 6. PLOTTING
    # =============================================================================
    # === AGENT EDITABLE SECTION START ===
    try:
        df["Mach"] = df["Mach"].astype(str)

        fig_ld = px.line(df, x="Alpha", y="L/D", color="Mach", markers=True,
                         title="Lift-to-Drag Ratio (L/D) vs Angle of Attack",
                         labels={"Alpha": "Angle of Attack (deg)", "L/D": "Lift / Drag"})
        fig_ld.update_layout(template="plotly_white", height=600, width=900)
        fig_ld.write_image(os.path.join(output_dir, "LD_vs_Alpha.png"))

        fig_polar = px.line(df, x="CD", y="CL", color="Mach", markers=True,
                            title="Drag Polars (CL vs CD)",
                            labels={"CD": "Drag Coefficient (CD)", "CL": "Lift Coefficient (CL)"})
        fig_polar.update_layout(template="plotly_white", height=600, width=900)
        fig_polar.write_image(os.path.join(output_dir, "Drag_Polars.png"))

        fig_cl_alpha = px.line(df, x="Alpha", y="CL", color="Mach", markers=True,
                                title="Lift Coefficient (CL) vs Angle of Attack",
                                labels={"Alpha": "Angle of Attack (deg)", "CL": "Lift Coefficient (CL)"})
        fig_cl_alpha.update_layout(template="plotly_white", height=600, width=900)
        fig_cl_alpha.write_image(os.path.join(output_dir, "CL_vs_Alpha.png"))
    except Exception as e:
        print(f"Plotting warning: {e}")
    # === AGENT EDITABLE SECTION END ===

    # =============================================================================
    # 7. SPANWISE LIFT DISTRIBUTION
    # =============================================================================