import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import openmdao.api as om

# import OpenAeroStruct modules
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

import niceplots

plt.style.use(niceplots.get_style("james-light"))

# Ensure unified directory for saving results exists
output_dir = "src/openaerostruct_out/agent_plots"
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
    ax1 = plt.gca()
    niceplots.adjust_spines(ax1)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # =============================================================================
    # 2. MESH GENERATION
    # =============================================================================
    # The agent should modify these parameters to change the wing's baseline shape.
    mesh_dict = {
        "num_y": 19,            # Number of spanwise panels, must be an odd number
        "num_x": 3,             # Number of chordwise panels
        "wing_type": "rect",    # "rect" or "CRM"
        "symmetry": True,
        "span": 25.52, 
        "root_chord": 2.83, 
        "span_cos_spacing": 0.0,
        "chord_cos_spacing": 0.0,
    }

    mesh = generate_mesh(mesh_dict)

    # =============================================================================
    # 3. SURFACE DEFINITION (Using Existing Parameters)
    # =============================================================================
    # This dictionary defines the wing properties. Agent can adjust for "what-if" analysis.
    surface = {
        "name": "wing",
        "symmetry": True,
        "S_ref_type": "wetted",
        "twist_cp": np.array([0.64486856, 0.80830532]),  
        "chord_cp": np.array([0.83383695, 1.08508756, 1.26577193]), 
        "t_over_c_cp": np.array([0.02853182, 0.01715538, 0.01]), 
        "taper": 0.876,             
        "sweep": 27.98,             
        "dihedral": 0.000,          
        "mesh": mesh,
        "CL0": 0.1651338,
        "CD0": 0.00935,
        "k_lam": 0.05,
        "c_max_t": 0.303,
        "with_viscous": True,
        "with_wave": True,          
    }

    # =============================================================================
    # 4. PROBLEM SETUP
    # =============================================================================
    prob = om.Problem()

    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("v", val=100.0, units="m/s")
    indep_var_comp.add_output("alpha", val=0.0, units="deg")
    indep_var_comp.add_output("Mach_number", val=0.1)
    indep_var_comp.add_output("re", val=1e6, units="1/m")
    indep_var_comp.add_output("rho", val=1.225, units="kg/m**3")
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

    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
    prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")
    prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

    prob.setup()

    # =============================================================================
    # 5. ANALYSIS SWEEP
    # =============================================================================
    mach_range = np.arange(0.1, 0.8, 0.1) 
    alpha_range = np.arange(-10, 16, 1)    

    results = []
    print("Running Aerodynamic Analysis Sweep...")
    for M in mach_range:
        for a in alpha_range:
            v_val = M * 340.0
            re_val = 1.225 * v_val / 1.81e-5
            prob.set_val("Mach_number", M)
            prob.set_val("v", v_val, units="m/s")
            prob.set_val("re", re_val, units="1/m")
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
    # 6. PLOTTING POLARS
    # =============================================================================
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

    # =============================================================================
    # 7. SPANWISE LIFT DISTRIBUTION
    # =============================================================================
    print("Generating Spanwise Lift Distribution...")
    trim_mach = 0.8
    trim_alpha = 3.4530
    v_trim = trim_mach * 340.0
    
    prob.set_val("Mach_number", trim_mach)
    prob.set_val("v", v_trim, units="m/s")
    prob.set_val("re", 1.225 * v_trim / 1.81e-5, units="1/m")
    prob.set_val("alpha", trim_alpha, units="deg")
    prob.run_model()

    mesh_out = prob.get_val("flight_condition_0.wing.def_mesh", units="m")
    y_vertices = mesh_out[0, :, 1]  
    y_center = 0.5 * (y_vertices[:-1] + y_vertices[1:])  
    Cl = prob.get_val("flight_condition_0.wing_perf.Cl")  
    chord_edge = prob.get_val("flight_condition_0.wing.chords", units="m")  
    chord_center = 0.5 * (chord_edge[:-1] + chord_edge[1:])  

    x_center = np.concatenate((y_center, -y_center[::-1]))
    chord_center = np.concatenate((chord_center, chord_center[::-1]))
    Cl = np.concatenate((Cl, Cl[::-1]))

    CL_total = prob.get_val("flight_condition_0.wing_perf.CL")[0]  
    Sref = prob.get_val("flight_condition_0.wing_perf.S_ref", units="m**2")[0]  
    semi_span = mesh_out[0, -1, 1] - mesh_out[0, 0, 1] 

    cl_chord_max = 2 * CL_total * Sref / (np.pi * semi_span)
    y_ellipse = np.linspace(-semi_span, semi_span, 100)
    Cl_chord_ellipse = cl_chord_max * np.sqrt(1 - (y_ellipse / semi_span) ** 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_ellipse, y=Cl_chord_ellipse * 0.5 * 1.225 * v_trim**2, 
                             mode='lines', name='Elliptical Lift Distribution',
                             line=dict(color='red', width=2, dash='dash'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=x_center, y=Cl, 
                             mode='lines', name='Lift Coefficient (Cl)',
                             line=dict(color='green', width=2), yaxis='y1'))

    fig.update_layout(
        title=f"Spanwise Lift Distribution (Mach {trim_mach}, Alpha {trim_alpha}°)",
        xaxis_title='Span (m)',
        yaxis_title='Sectional Lift Coefficient (Cl)',
        yaxis2=dict(title='Sectional Lift (N/m)', overlaying='y', side='right'),
        template="plotly_white",
        height=600,
        width=900,
        legend=dict(x=1.1, y=1)
    )

    fig.write_image(os.path.join(output_dir, "Sectional_Lift_Distribution_Trim.png"))
    print(f"Done! Check the {output_dir} folder for your plots.")
