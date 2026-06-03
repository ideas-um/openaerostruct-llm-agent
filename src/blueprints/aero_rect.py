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
# Modify span, root_chord, and mesh resolution here.
# num_y must be an odd number.
# === AGENT EDITABLE SECTION START ===
mesh_dict = {
    "num_y": 19,            # Number of spanwise panels (must be odd)
    "num_x": 3,             # Number of chordwise panels
    "wing_type": "rect",    # Always "rect" for this blueprint
    "symmetry": True,
    "span": 10.0,           # Full wingspan [m]
    "root_chord": 2.0,      # Root chord [m]
    "span_cos_spacing": 0.0,
    "chord_cos_spacing": 0.0,
}
# === AGENT EDITABLE SECTION END ===

mesh = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# CRITICAL: Any geometry parameter you want to use as a design variable (twist_cp,
# chord_cp, taper, sweep, dihedral) MUST be declared here in the surface dict first.
# If you add a design variable in Section 4 without declaring it here, the
# Geometry group will not create that output and OpenMDAO will crash with:
#   "Output not found for design variable 'wing.twist_cp'"
#
# Uncomment the keys you need and add them as design variables in Section 4.
# === AGENT EDITABLE SECTION START ===
surface = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh,

    # Aerodynamic properties
    "CL0": 0.0,
    "CD0": 0.0,
    "with_viscous": True,
    "with_wave": False,
    "k_lam": 0.05,
    "c_max_t": 0.303,
    "t_over_c_cp": np.array([0.12]),

    # Geometry design variables — uncomment to enable, then add to add_design_var() below.
    # Each one you uncomment here MUST also appear in the design variables section.
    #"twist_cp": np.zeros(3),      # Spanwise twist [deg], shape must match add_design_var bounds
    #"chord_cp": np.ones(3),       # Chord scaling B-spline control points
    #"taper": 1.0,                 # Taper ratio
    #"sweep": 0.0,                 # Sweep angle [deg]
    #"dihedral": 0.0,              # Dihedral angle [deg]
}
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 3. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

# Flight condition — modify Mach and rho to match the desired operating point.
# Re is computed as rho * v / 1.81e-5 [1/m] — do NOT use surface["root_chord"].
# === AGENT EDITABLE SECTION START ===
Mach_number = 0.5
rho = 1.225                          # Air density [kg/m^3]
v = Mach_number * 340.0              # Freestream speed [m/s]
re = rho * v / 1.81e-5               # Reynolds number per unit length [1/m]
# === AGENT EDITABLE SECTION END ===

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=v, units="m/s")
indep_var_comp.add_output("alpha", val=3.0, units="deg")
indep_var_comp.add_output("Mach_number", val=Mach_number)
indep_var_comp.add_output("re", val=re, units="1/m")
indep_var_comp.add_output("rho", val=rho, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])

# Geometry group — uses surface dict as defined above.
# Subsystem name is "wing" — all DV paths are "wing.<var_name>" (e.g. "wing.twist_cp").
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

prob.model.connect(f"{name}.mesh", f"{point_name}.{name}.def_mesh")
prob.model.connect(f"{name}.mesh", f"{point_name}.aero_states.{name}_def_mesh")
prob.model.connect(f"{name}.t_over_c", f"{point_name}.{name}_perf.t_over_c")

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()

output_dir = os.path.join("src", "openaerostruct_out", "generated_run_out")
os.makedirs(output_dir, exist_ok=True)
recorder = om.SqliteRecorder(os.path.join(output_dir, "aero.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]
prob.options['work_dir'] = output_dir

# === AGENT EDITABLE SECTION START ===
# --- Design Variables ---
# RULE: Any geometry DV listed here must also be declared in the surface dict above.
# alpha is always available (no surface dict entry needed).
# All other wing geometry DVs need a corresponding key in surface{} first.
#
# Available DV paths:
#   "alpha"             — angle of attack (always available)
#   "wing.twist_cp"     — requires "twist_cp" in surface dict
#   "wing.chord_cp"     — requires "chord_cp" in surface dict
#   "wing.taper"        — requires "taper" in surface dict
#   "wing.sweep"        — requires "sweep" in surface dict
#   "wing.dihedral"     — requires "dihedral" in surface dict
#
# NOTE: For span constraints, use "wing.mesh.stretch.span" (not "wing.span").
# NOTE: For area constraints, use "flight_condition_0.wing_perf.S_ref" (not "wing.S_ref").

prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
# prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=10.0)
# prob.model.add_design_var("wing.taper", lower=0.1, upper=1.5)
# prob.model.add_design_var("wing.sweep", lower=-30.0, upper=30.0)

# --- Constraints ---
# CL constraint requires alpha as a design variable (see above).
prob.model.add_constraint(f"{point_name}.wing_perf.CL", equals=0.5)
# prob.model.add_constraint(f"{point_name}.wing_perf.S_ref", lower=30.0)   # area constraint
# prob.model.add_constraint("wing.mesh.stretch.span", upper=12.0)          # span constraint

# --- Objective ---
prob.model.add_objective(f"{point_name}.wing_perf.CD", ref=0.01)
# === AGENT EDITABLE SECTION END ===

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Optimization Results ---")
print(f"Final Alpha: {prob.get_val('alpha', units='deg')[0]:.4f} deg")
print(f"Final CL:    {prob.get_val(f'{point_name}.wing_perf.CL')[0]:.4f}")
print(f"Final CD:    {prob.get_val(f'{point_name}.wing_perf.CD')[0]:.6f}")