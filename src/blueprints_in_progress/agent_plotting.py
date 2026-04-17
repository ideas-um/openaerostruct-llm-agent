import sys
import os
import numpy as np
from openmdao.recorders.sqlite_reader import SqliteCaseReader
import matplotlib.pyplot as plt
from matplotlib import cm

# Set backend to Agg for headless plotting
import matplotlib
matplotlib.use("Agg")

def generate_plots(db_path="src/openaerostruct_out/aero.db", output_dir="src/openaerostruct_out/agent_plots"):
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        cr = SqliteCaseReader(db_path, pre_load=True)
        cases = cr.get_cases("driver")
        if not cases:
            print("No cases found in database.")
            return

        # Get the last successful case
        last_case = next(reversed(cases))
        
        # Determine surface names
        names = []
        sys_options = cr.list_model_options(out_stream=None)
        for key in sys_options.keys():
            try:
                surfaces = sys_options[key]["surfaces"]
                for surface in surfaces:
                    names.append(surface["name"])
                break
            except KeyError:
                pass
        
        if not names:
            print("No surface names found.")
            return

        # Simple 3D Mesh Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for name in names:
            mesh = last_case.outputs[name + ".mesh"]
            x = mesh[:, :, 0]
            y = mesh[:, :, 1]
            z = mesh[:, :, 2]
            ax.plot_wireframe(x, y, z, color='black', alpha=0.7)
            
        ax.set_title("Optimized Wing Geometry")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        
        plot_path = os.path.join(output_dir, "wing_3d.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved 3D plot to {plot_path}")

        # Lift Distribution Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name in names:
            try:
                # Find the flight condition point name
                pt_name = ""
                for key in last_case.outputs:
                    if "CL" in key and name in key:
                        pt_name = key.split(".")[0]
                        break
                
                if pt_name:
                    sec_forces = last_case.outputs[pt_name + ".aero_states." + name + "_sec_forces"]
                    widths = last_case.outputs[pt_name + "." + name + ".widths"]
                    v = last_case.outputs["v"]
                    rho = last_case.outputs["rho"]
                    alpha = last_case.outputs["alpha"] * np.pi / 180.0
                    
                    forces = np.sum(sec_forces, axis=0)
                    lift = (-forces[:, 0] * np.sin(alpha) + forces[:, 2] * np.cos(alpha)) / widths / 0.5 / rho / v**2
                    
                    mesh = last_case.outputs[name + ".mesh"]
                    span = mesh[0, :, 1]
                    center_span = (span[:-1] + span[1:]) / 2.0
                    
                    ax.plot(center_span, lift, label=f"{name} Lift")
            except Exception as e:
                print(f"Could not generate lift plot for {name}: {e}")

        ax.set_title("Spanwise Lift Distribution")
        ax.set_xlabel("Span (m)")
        ax.set_ylabel("Sectional Lift Coefficient (Cl)")
        ax.legend()
        ax.grid(True)
        
        lift_plot_path = os.path.join(output_dir, "lift_dist.png")
        plt.savefig(lift_plot_path)
        plt.close()
        print(f"Saved lift distribution to {lift_plot_path}")

    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "src/openaerostruct_out/generated_run_out/aero.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "src/openaerostruct_out/agent_plots"
    generate_plots(db, out)
