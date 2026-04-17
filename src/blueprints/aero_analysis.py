import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import openmdao.api as om

# import OpenAeroStruct modules
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

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

    # Ensure output directory exists
    output_dir = os.path.join("src", "openaerostruct_out", "agent_plots")
    os.makedirs(output_dir, exist_ok=True)

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
    # 6. TRIM CONDITION ANALYSIS
    # =============================================================================
    print("Computing Spanwise Lift Distribution at trim...")
    trim_mach = 0.8
    trim_alpha = 3.4530
    v_trim = trim_mach * 340.0

    prob.set_val("Mach_number", trim_mach)
    prob.set_val("v", v_trim, units="m/s")
    prob.set_val("re", 1.225 * v_trim / 1.81e-5, units="1/m")
    prob.set_val("alpha", trim_alpha, units="deg")
    prob.run_model()

    CL_trim = prob.get_val("flight_condition_0.wing_perf.CL")[0]
    CD_trim = prob.get_val("flight_condition_0.wing_perf.CD")[0]
    print(f"Trim CL: {CL_trim:.4f}, Trim CD: {CD_trim:.6f}, L/D: {CL_trim / CD_trim:.2f}")
    print(f"Done! Data saved to {output_dir}")
