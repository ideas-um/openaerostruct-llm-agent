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
    # Using parameters derived from the user's chord distribution
    # Root chord is the first value in the chord list
    root_chord_val = 0.83383695
    
    mesh_dict = {
        "num_y": 19,            # Number of spanwise panels
        "num_x": 5,             # Number of chordwise panels
        "wing_type": "rect",    
        "symmetry": True,
        "span": 15.0,           # Assumed span for analysis
        "root_chord": root_chord_val, 
        "span_cos_spacing": 0.0,
        "chord_cos_spacing": 0.0,
    }

    mesh = generate_mesh(mesh_dict)

    # =============================================================================
    # 3. SURFACE DEFINITION
    # =============================================================================
    # Using the CP values provided by the user
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
    # 5. ANALYSIS SWEEP (Angle of Attack)
    # =============================================================================
    # We want to see performance under different AoA
    mach_fixed = 0.3
    alpha_range = np.linspace(-5, 15, 21)    

    results = []
    print(f"Running Aerodynamic Analysis Sweep at Mach {mach_fixed}...")
    
    for a in alpha_range:
        v_val = mach_fixed * 340.0
        re_val = 1.225 * v_val / 1.81e-5
        
        prob.set_val("Mach_number", mach_fixed)
        prob.set_val("v", v_val, units="m/s")
        prob.set_val("re", re_val, units="1/m")
        prob.set_val("alpha", a, units="deg")
        
        prob.run_model()

        CL = prob.get_val("flight_condition_0.wing_perf.CL")[0]
        CD = prob.get_val("flight_condition_0.wing_perf.CD")[0]

        results.append({
            "Mach": mach_fixed,
            "Alpha": a,
            "CL": CL,
            "CD": CD,
            "L/D": CL / CD if CD != 0 else 0,
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "Wing_AoA_Performance.csv")
    df.to_csv(csv_path, index=False)
    print(f"Analysis complete. Saved data to {csv_path}")

    # =============================================================================
    # 6. PLOTTING PERFORMANCE
    # =============================================================================
    # 1. CL vs Alpha
    fig_cl_alpha = px.line(df, x="Alpha", y="CL", markers=True,
                            title="Lift Coefficient (CL) vs Angle of Attack",
                            labels={"Alpha": "Angle of Attack (deg)", "CL": "Lift Coefficient (CL)"})
    fig_cl_alpha.update_layout(template="plotly_white", height=600, width=900)
    fig_cl_alpha.write_image(os.path.join(output_dir, "CL_vs_Alpha.png"))

    # 2. CD vs Alpha
    fig_cd_alpha = px.line(df, x="Alpha", y="CD", markers=True,
                            title="Drag Coefficient (CD) vs Angle of Attack",
                            labels={"Alpha": "Angle of Attack (deg)", "CD": "Drag Coefficient (CD)"})
    fig_cd_alpha.update_layout(template="plotly_white", height=600, width=900)
    fig_cd_alpha.write_image(os.path.join(output_dir, "CD_vs_Alpha.png"))

    # 3. L/D vs Alpha
    fig_ld_alpha = px.line(df, x="Alpha", y="L/D", markers=True,
                            title="Lift-to-Drag Ratio (L/D) vs Angle of Attack",
                            labels={"Alpha": "Angle of Attack (deg)", "L/D": "Lift / Drag"})
    fig_ld_alpha.update_layout(template="plotly_white", height=600, width=900)
    fig_ld_alpha.write_image(os.path.join(output_dir, "LD_vs_Alpha.png"))

    # 4. Drag Polar (CL vs CD)
    fig_polar = px.line(df, x="CD", y="CL", markers=True,
                        title="Drag Polars (CL vs CD)",
                        labels={"CD": "Drag Coefficient (CD)", "CL": "Lift Coefficient (CL)"})
    fig_polar.update_layout(template="plotly_white", height=600, width=900)
    fig_polar.write_image(os.path.join(output_dir, "Drag_Polar.png"))

    print(f"Done! Check the {output_dir} folder for your plots.")