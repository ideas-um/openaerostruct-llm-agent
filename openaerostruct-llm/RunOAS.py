###### Import functions are here, DO NOT EDIT ##########
import warnings
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import openmdao.api as om
# import OpenAeroStruct modules
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
import niceplots  # Optional but recommended
import time

""" CORE FUNCTIONS ARE DEFINED HERE, DO NOT EDIT """
def plot_mesh(mesh):
    """ this function plots to plot the mesh """
    mesh_x = mesh[:, :, 0]
    mesh_y = mesh[:, :, 1]
    plt.figure(figsize=(6, 3))
    color = 'k'
    for i in range(mesh_x.shape[0]):
        plt.plot(mesh_y[i, :], mesh_x[i, :], color, lw=1)
        # plt.plot(-mesh_y[i, :], mesh_x[i, :], color, lw=1)   # plots the other side of symmetric wing
    for j in range(mesh_x.shape[1]):
        plt.plot(mesh_y[:, j], mesh_x[:, j], color, lw=1)
        # plt.plot(-mesh_y[:, j], mesh_x[:, j], color, lw=1)   # plots the other side of symmetric wing
    plt.axis('equal')
    plt.xlabel('span(m)')
    plt.ylabel('chord(m)')
    plt.savefig('./figures/mesh.pdf', bbox_inches='tight')

"""Part 1: PUT THE BASELINE MESH OF THE WING HERE"""
mesh_dict = {
    "num_y": 19, # number of panels in the y direction
    "num_x": 3, # number of panels in the x direction
    "wing_type": "rect", # This can either be "rect" or "crm" only
    "symmetry": True, # True if the wing is symmetric
    "span": 10.0, # Full span of the wing in meters
    "root_chord": 10.0, # Root chord of the wing in meters
    "span_cos_spacing": 0.0,
    "chord_cos_spacing": 0.0,
}

# Generate VLM mesh for half-wing
mesh = generate_mesh(mesh_dict)

# Plot mesh
plot_mesh(mesh)



"""Part 2:  DO THE GEOMETRY SETUP HERE"""
surface = {
    # Wing definition
    "name": "wing",  # Name of the surface
    "symmetry": True,  # Model one half of wing reflected across y = 0
    "S_ref_type": "wetted",  # How wing area is computed ('wetted' or 'projected')
    "mesh": mesh,

    # Aerodynamic performance at angle of attack = 0
    # These CL0 and CD0 values are added to the CL and CD from aerodynamic analysis.
    # They do not vary with alpha.
    "CL0": 0.0,
    "CD0": 0.0,

    # Airfoil properties for viscous drag calculation
    "k_lam": 0.05,  # Percentage of chord with laminar flow
    "c_max_t": 0.303,  # Chordwise location of maximum thickness (NACA0015)
    "t_over_c_cp": np.array([0.12]),  # Thickness-to-chord ratio

    # Type of analysis
    "with_viscous": True,  # If true, compute viscous drag
    "with_wave": False,

    # Options for changing the wing geometry
    "chord_cp": np.ones(3),
    "taper": 0.4,
    "sweep": 28.0,
    "dihedral": 3.0,
    "twist_cp": np.zeros(2),
}



"""Part 3: PUT THE OPTIMIZER HERE """
prob = om.Problem()

# Define flight conditions
Mach_number = 0.5
rho = 1.225
v = Mach_number * 340  # Freestream speed, m/s
Re_c = rho * v / 1.81e-5  # Reynolds number / characteristic length, 1/m

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=v, units="m/s")
indep_var_comp.add_output(
    "alpha", val=0.0, units="deg"
)
indep_var_comp.add_output("Mach_number", val=Mach_number)
indep_var_comp.add_output("re", val=Re_c, units="1/m")
indep_var_comp.add_output("rho", val=rho, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

# Setup OpenAeroStruct model
name = surface["name"]

# Add geometry group to the problem and add wing surface as a sub group.
# These groups manipulate the geometry of the mesh, e.g., spanwise twist.
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

# Connect the mesh from the geometry component to the analysis point
prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

# Perform the connections with the modified names within the 'aero_states' group.
prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

# Connect the parameters within the model for each aero point
prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

prob.model.add_design_var('wing.taper', lower=0.2, upper=1.0)  # Varies the taper ratio
prob.model.add_design_var('wing.twist_cp', lower=-10.0, upper=10.0) # Varies twist control points
prob.model.add_design_var('wing.sweep', lower=5.0, upper=45.0, units='deg') # Varies sweep
prob.model.add_design_var('alpha', units='deg', lower=-5.0, upper=10.0)   # Varies angle of attack
prob.model.add_constraint('flight_condition_0.wing_perf.CL', equals=2.0)   # Impose CL = 2.0
prob.model.add_objective('flight_condition_0.wing_perf.CD', ref=0.01)   # Minimize CD.

# Use Scipy's SLSQP optimization
prob.driver = om.ScipyOptimizeDriver()

# Record optimization history
recorder = om.SqliteRecorder("aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = './openaerostruct_out'

prob.setup()
prob.run_driver()

# Print results
print("Angle of attack =", prob.get_val("alpha", units="deg")[0], "deg")
print("CL = ", prob.get_val("flight_condition_0.wing_perf.CL")[0])
print("CD = ", prob.get_val("flight_condition_0.wing_perf.CD")[0])

