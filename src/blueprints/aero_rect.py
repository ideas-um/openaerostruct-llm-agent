import numpy as np
import openmdao.api as om
import os
import matplotlib
matplotlib.use('Agg')
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# The agent should modify these parameters to change the wing's baseline shape.
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 19,            # Number of spanwise panels
    "num_x": 3,             # Number of chordwise panels
    "wing_type": "rect",    # "rect" for rectangular
    "symmetry": True,       # Use symmetry to reduce computation time
    "span": 10.0,           # Full span in meters
    "root_chord": 2.0,      # Root chord in meters
    "span_cos_spacing": 0.0,
    "chord_cos_spacing": 0.0,
}
# === AGENT EDITABLE SECTION END ===

# Generate the actual coordinate array (mesh)
mesh = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# This dictionary defines the aerodynamic and geometric properties of the wing.
# === AGENT EDITABLE SECTION START ===
surface = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh,
    
    # Aerodynamic coefficients at alpha=0 (fixed offsets)
    "CL0": 0.0,
    "CD0": 0.0,

    # Airfoil/Viscous properties
    "with_viscous": True,
    "with_wave": False,
    "k_lam": 0.05,          # Percentage of laminar flow
    "c_max_t": 0.303,       # Thickness location
    "t_over_c_cp": np.array([0.12]), # Thickness-to-chord ratio

    # Geometry Control (These need to be uncommented to allow optimization)
    #"twist_cp": np.zeros(3),     # B-spline control points for twist
    #"taper": 1.0,                # Initial taper ratio
    #"sweep": 0.0,                # Initial sweep angle in degrees
    #"dihedral": 0.0,             # Initial dihedral angle
    #"chord_cp": np.ones(3),      # B-spline control points for chord scaling
}
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

# Define flight conditions
Mach_number = 0.5 # You can change this for the Mach Number
rho = 1.225
v = Mach_number * 340  # Freestream speed, m/s
Re_c = rho * v / 1.81e-5  # Reynolds number / characteristic length, 1/m

# Independent variables for flight conditions
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=v, units="m/s")
indep_var_comp.add_output("alpha", val=3.0, units="deg")
indep_var_comp.add_output("Mach_number", val=Mach_number)
indep_var_comp.add_output("re", val=1e6, units="1/m")
indep_var_comp.add_output("rho", val=1.225, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

# Setup OpenAeroStruct model
name = surface["name"]

# Add geometry group to the problem and add wing suface as a sub group.
# These groups are responsible for manipulating the geometry of the mesh, in this case spanwise twist.
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(name, geom_group)

# Create the aero point group for this flight condition and add it to the model
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
# Connect Geometry Mesh to Aero solver
prob.model.connect(f"{surface['name']}.mesh", f"{point_name}.{surface['name']}.def_mesh")
prob.model.connect(f"{surface['name']}.mesh", f"{point_name}.aero_states.{surface['name']}_def_mesh")
prob.model.connect(f"{surface['name']}.t_over_c", f"{point_name}.{surface['name']}_perf.t_over_c")

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
# Setup the driver (SLSQP is the standard gradient-based optimizer)
prob.driver = om.ScipyOptimizeDriver()

# record optimization history
output_dir = os.path.join("src", "openaerostruct_out", "generated_run_out")
os.makedirs(output_dir, exist_ok=True)
recorder = om.SqliteRecorder(os.path.join(output_dir, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = output_dir

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
#You can change the upper and lower bounds.
#You are also allowed to add the design variables, constraints, and objectives here like chord_cp, twist_cp, taper, sweep, dihedral etc.
#The way to add them is wing."var_name" (for example, wing.taper) and the lower and upper bounds are in the form of lower=0.0, upper=1.0.
#These are the var names that you can use taper, sweep, chord_cp, twist_cp, dihedral.
#remember to add alpha as a design variable if CL is a constraint for trimming.

#prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
#prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=10.0)
#prob.model.add_design_var("wing.taper", lower=0.1, upper=1.5)
#prob.model.add_design_var('wing.sweep', lower=-30.0, upper=30.0)

# --- Constraints ---
# Example: Maintain a specific Lift Coefficient
prob.model.add_constraint(f"{point_name}.wing_perf.CL", equals=0.5)

# --- Objective ---
# Example: Minimize Drag Coefficient
prob.model.add_objective(f"{point_name}.wing_perf.CD", ref=0.01)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

# Print results for the agent to read
print("\n--- Optimization Results ---")
print(f"Final Alpha: {prob.get_val('alpha', units='deg')[0]:.4f} deg")
print(f"Final Taper: {prob.get_val('wing.taper')[0]:.4f}")
print(f"Final CL:    {prob.get_val(f'{point_name}.wing_perf.CL')[0]:.4f}")
print(f"Final CD:    {prob.get_val(f'{point_name}.wing_perf.CD')[0]:.6f}")