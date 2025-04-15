###### Import functions are here, DO NOT EDIT ##########
import warnings
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import openmdao.api as om
# import OpenAeroStruct modules
from openaerostruct.geometry.utils import generate_mesh  # helper functions to generate mesh
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
    plt.savefig('/Users/conan/Desktop/LLM_Aerospace_Research/LLM_OpenAeroStruct/Figures/mesh.pdf', bbox_inches='tight')

"""Part 1: PUT THE BASELINE MESH OF THE WING HERE"""
# This is a sample code to generate a mesh for a rectangular wing
mesh_dict = {
    "num_y": 19, #number of panels in the y direction, 19 is a good starting number
    "num_x": 3, #number of panels in the x direction, 3 is a good starting number
    "wing_type": "rect", #This can either be "rect" or "crm" only
    "symmetry": True, # True if the wing is symmetric, False if it is not, wings are typically symmetric
    "span": 60.0, #This is the full span of the wing in meters
    "root_chord": 6.666666666666667, #This is the root chord of the wing in meters
    "span_cos_spacing": 0.0, #This is usually not edited
    "chord_cos_spacing": 0.0, #This is usually not edited
}

# Generate VLM mesh for half-wing
mesh = generate_mesh(mesh_dict)   # this creates a rectangular wing mesh, do not edit

# plot mesh
plot_mesh(mesh) # this plots the rectangular wing mesh, do not edit

"""Part 2:  DO THE GEOMETRY SETUP HERE"""
surface = {
    # Wing definition, KEEP THE SAME UNLESS ASKED TO CHANGE
    "name": "wing",  # name of the surface, keep as wing
    "symmetry": True,  # if true, model one half of wing reflected across the plane y = 0
    "S_ref_type": "wetted",  # how we compute the wing area, can be 'wetted' or 'projected'
    "mesh": mesh,

    # Aerodynamic performance of the lifting surface at an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD obtained from aerodynamic analysis of the surface to get the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha. DO NOT EDIT THEM UNLESS ASKED TO.
    "CL0": 0.0,  # CL of the surface at alpha=0
    "CD0": 0.0,  # CD of the surface at alpha=0

    # Airfoil properties for viscous drag calculation, DO NOT CHANGE UNLESS ASKED TO
    "k_lam": 0.05,  # percentage of chord with laminar flow, used for viscous drag
    "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
    "t_over_c_cp": np.array([0.12]),  # thickness-to-chord ratio

    # DO NOT CHANGE UNLESS ASKED TO, type of analysis, wave for high mach number, viscous to model viscous drag
    "with_viscous": True,  # if true, compute viscous drag,
    "with_wave": False,

    # Useful options for changing the wing geometry, CHANGE THESE
    #"chord_cp": np.ones(3),  # if chord cp is allowed to be optimized, uncomment this line and change the value for how many points for the bspline to change the chord, default is 3
    "taper" : 0.4, # if the wing can be tapered, uncomment this line and change the initial value for how much taper, default is 0.4
    "sweep" : 28.0, # if the wing can be swept, uncomment this line and change the initial value for how much sweep, default is 28.0
    #"twist_cp" : np.zeros(2),  # if the wing can be twisted, uncomment this line and change the value for how many points for the bspline to change the twist, default is 4
}  # end of surface dictionary

"""Part 3: PUT THE OPTIMIZER HERE """
# Instantiate the problem and the model group
prob = om.Problem()

# Define flight conditions
Mach_number = 0.5 # You can change this if the user specifies a different Mach number
rho = 1.225
v = Mach_number * 340  # freestream speed, m/s
Re_c = rho * v / 1.81e-5  # Reynolds number / characteristic length, 1/m

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=v, units="m/s")  # Freestream Velocity
indep_var_comp.add_output(
    "alpha", val=0.0, units="deg"
) 
indep_var_comp.add_output("Mach_number", val=Mach_number)  # Freestream Mach number
indep_var_comp.add_output("re", val=Re_c, units="1/m")  # Freestream Reynolds number times chord length
indep_var_comp.add_output("rho", val=rho, units="kg/m**3")  # Freestream air density
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")  # Aircraft center of gravity
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

# Connect the mesh from the geometry component to the analysis point
prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

# Perform the connections with the modified names within the 'aero_states' group.
prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

# Connect the parameters within the model for each aero point
prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

########## THIS IS THE PART TO EDIT ##########
#If the variables are not specified, you can comment them out, you can also change the upper and lower bounds.
#You are also allowed to add the design varaibles, constraints, and objectives here like chord_cp, twist_cp, taper, sweep etc.
#The way to add them is wing."var_name" and the lower and upper bounds are in the form of lower=0.0, upper=1.0
#these are the var names that you can use area = S_ref, taper = taper, sweep = sweep, chord_cp = chord_cp, twist_cp = twist_cp

prob.model.add_design_var('alpha', units='deg', lower=0., upper=10.)  # varies
prob.model.add_design_var('wing.taper', lower=0.2, upper=0.8)  # taper
prob.model.add_design_var('wing.sweep', lower=10., upper=30., units='deg')  # sweep

prob.model.add_constraint('flight_condition_0.wing_perf.CL', equals=0.5)  # impose CL = x
#prob.model.add_constraint('wing.S_ref', equals=400.)  # area constraint
#prob.model.add_constraint('wing.b', equals=60.)  # span constraint

prob.model.add_objective('flight_condition_0.wing_perf.CD', ref=0.01)  # dummy objective to minimize CD.
############# THIS END OF THE PART TO EDIT ##########

# use Scipy's SLSQP optimization
prob.driver = om.ScipyOptimizeDriver()

# record optimization history
recorder = om.SqliteRecorder("aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

prob.setup()
prob.run_driver() 

#print results
print("\nAngle of attack =", prob.get_val("alpha", units="deg")[0], "deg")
print("CL = ", prob.get_val("flight_condition_0.wing_perf.CL")[0])
print("CD = ", prob.get_val("flight_condition_0.wing_perf.CD")[0])

"""Part 4: PLOT THE RESULTS AND ANALYSIS HERE"""