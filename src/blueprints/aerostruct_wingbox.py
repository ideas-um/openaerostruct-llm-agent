import numpy as np
import os
import openmdao.api as om
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta

# =============================================================================
# 1. MESH GENERATION
# =============================================================================
# The agent should modify these parameters to change the wing's baseline shape.

# coordinates
upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")
lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")
upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype="complex128")
lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444], dtype="complex128")

mesh_dict = {
    "num_y": 15,
    "num_x": 3,
    "wing_type": "uCRM_based",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 0,
    "num_twist_cp": 4,
}

mesh, twist_cp = generate_mesh(mesh_dict)

# =============================================================================
# 2. SURFACE DEFINITION
# =============================================================================
# Using "wingbox" for high-fidelity structural representation.
surf_dict = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "projected",
    "mesh": mesh,
    "fem_model_type": "wingbox",
    "data_x_upper": upper_x,
    "data_x_lower": lower_x,
    "data_y_upper": upper_y,
    "data_y_lower": lower_y,
    "twist_cp": np.array([4.0, 5.0, 8.0, 9.0]),
    "spar_thickness_cp": np.array([0.004, 0.005, 0.008, 0.01]),
    "skin_thickness_cp": np.array([0.005, 0.01, 0.015, 0.025]),
    "t_over_c_cp": np.array([0.08, 0.08, 0.10, 0.08]),
    "original_wingbox_airfoil_t_over_c": 0.12,
    "CL0": 0.0,
    "CD0": 0.0078,
    "with_viscous": True,
    "with_wave": True,
    "k_lam": 0.05,
    "c_max_t": 0.38,
    "E": 73.1e9,
    "G": (73.1e9 / 2 / 1.33),
    "yield": 420.0e6,
    "safety_factor": 1.5,
    "mrho": 2.78e3,
    "strength_factor_for_upper_skin": 1.0,
    "wing_weight_ratio": 1.25,
    "exact_failure_constraint": False,
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    "n_point_masses": 1,
    "fuel_density": 803.0,
    "Wf_reserve": 15000.0,
}

surfaces = [surf_dict]

# =============================================================================
# 3. PROBLEM SETUP (AEROSTRUCTURAL WINGBOX)
# =============================================================================
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("Mach_number", val=np.array([0.85, 0.64]))
indep_var_comp.add_output("v", val=np.array([0.85 * 295.07, 0.64 * 340.294]), units="m/s")
indep_var_comp.add_output("re", val=np.array([0.348 * 295.07 * 0.85 * 1.0 / (1.43 * 1e-5), 1.225 * 340.294 * 0.64 * 1.0 / (1.81206 * 1e-5)]), units="1/m")
indep_var_comp.add_output("rho", val=np.array([0.348, 1.225]), units="kg/m**3")
indep_var_comp.add_output("speed_of_sound", val=np.array([295.07, 340.294]), units="m/s")
indep_var_comp.add_output("CT", val=0.53 / 3600, units="1/s")
indep_var_comp.add_output("R", val=14.307e6, units="m")
indep_var_comp.add_output("W0_without_point_masses", val=128000 + surf_dict["Wf_reserve"], units="kg")
indep_var_comp.add_output("load_factor", val=np.array([1.0, 2.5]))
indep_var_comp.add_output("alpha", val=0.0, units="deg")
indep_var_comp.add_output("alpha_maneuver", val=0.0, units="deg")
indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")
indep_var_comp.add_output("fuel_mass", val=10000.0, units="kg")

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

point_masses = np.array([[10.0e3]])
point_mass_locations = np.array([[25, -10.0, 0.0]])
indep_var_comp.add_output("point_masses", val=point_masses, units="kg")
indep_var_comp.add_output("point_mass_locations", val=point_mass_locations, units="m")
prob.model.add_subsystem("W0_comp", om.ExecComp("W0 = W0_without_point_masses + 2 * sum(point_masses)", units="kg"), promotes=["*"])

for surface in surfaces:
    name = surface["name"]
    aerostruct_group = AerostructGeometry(surface=surface)
    prob.model.add_subsystem(name, aerostruct_group)

for i in range(2):
    point_name = "AS_point_{}".format(i)
    AS_point = AerostructPoint(surfaces=surfaces, internally_connect_fuelburn=False)
    prob.model.add_subsystem(point_name, AS_point)

    prob.model.connect("v", point_name + ".v", src_indices=[i])
    prob.model.connect("Mach_number", point_name + ".Mach_number", src_indices=[i])
    prob.model.connect("re", point_name + ".re", src_indices=[i])
    prob.model.connect("rho", point_name + ".rho", src_indices=[i])
    prob.model.connect("CT", point_name + ".CT")
    prob.model.connect("R", point_name + ".R")
    prob.model.connect("W0", point_name + ".W0")
    prob.model.connect("speed_of_sound", point_name + ".speed_of_sound", src_indices=[i])
    prob.model.connect("empty_cg", point_name + ".empty_cg")
    prob.model.connect("load_factor", point_name + ".load_factor", src_indices=[i])
    prob.model.connect("fuel_mass", point_name + ".total_perf.L_equals_W.fuelburn")
    prob.model.connect("fuel_mass", point_name + ".total_perf.CG.fuelburn")

    for surface in surfaces:
        name = surface["name"]
        if surf_dict["distributed_fuel_weight"]:
            prob.model.connect("load_factor", point_name + ".coupled.load_factor", src_indices=[i])

        com_name = point_name + "." + name + "_perf."
        prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
        prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")
        prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
        if surf_dict["struct_weight_relief"]:
            prob.model.connect(name + ".element_mass", point_name + ".coupled." + name + ".element_mass")

        prob.model.connect(name + ".nodes", com_name + "nodes")
        prob.model.connect(name + ".cg_location", point_name + ".total_perf." + name + "_cg_location")
        prob.model.connect(name + ".structural_mass", point_name + ".total_perf." + name + "_structural_mass")
        prob.model.connect(name + ".Qz", com_name + "Qz")
        prob.model.connect(name + ".J", com_name + "J")
        prob.model.connect(name + ".A_enc", com_name + "A_enc")
        prob.model.connect(name + ".htop", com_name + "htop")
        prob.model.connect(name + ".hbottom", com_name + "hbottom")
        prob.model.connect(name + ".hfront", com_name + "hfront")
        prob.model.connect(name + ".hrear", com_name + "hrear")
        prob.model.connect(name + ".spar_thickness", com_name + "spar_thickness")
        prob.model.connect(name + ".t_over_c", com_name + "t_over_c")

        coupled_name = point_name + ".coupled." + name
        prob.model.connect("point_masses", coupled_name + ".point_masses")
        prob.model.connect("point_mass_locations", coupled_name + ".point_mass_locations")

prob.model.connect("alpha", "AS_point_0.alpha")
prob.model.connect("alpha_maneuver", "AS_point_1.alpha")

prob.model.add_subsystem("fuel_vol_delta", WingboxFuelVolDelta(surface=surface))
prob.model.connect("wing.struct_setup.fuel_vols", "fuel_vol_delta.fuel_vols")
prob.model.connect("AS_point_0.fuelburn", "fuel_vol_delta.fuelburn")

if surf_dict["distributed_fuel_weight"]:
    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_0.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_0.coupled.wing.struct_states.fuel_mass")
    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_1.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_1.coupled.wing.struct_states.fuel_mass")

comp = om.ExecComp("fuel_diff = (fuel_mass - fuelburn) / fuelburn", units="kg")
prob.model.add_subsystem("fuel_diff", comp, promotes_inputs=["fuel_mass"], promotes_outputs=["fuel_diff"])
prob.model.connect("AS_point_0.fuelburn", "fuel_diff.fuelburn")

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-2

os.makedirs(os.path.join("src", "openaerostruct_out"), exist_ok=True)
recorder = om.SqliteRecorder(os.path.join("src", "openaerostruct_out", "aerostruct.db"))
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)
prob.model.add_design_var("wing.twist_cp", lower=-15.0, upper=15.0, scaler=0.1)
prob.model.add_design_var("wing.spar_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)
prob.model.add_design_var("wing.skin_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)
prob.model.add_design_var("wing.geometry.t_over_c_cp", lower=0.07, upper=0.2, scaler=10.0)
prob.model.add_design_var("alpha_maneuver", lower=-15.0, upper=15)
prob.model.add_design_var("fuel_mass", lower=0.0, upper=2e5, scaler=1e-5)

prob.model.add_constraint("AS_point_0.CL", equals=0.5)
prob.model.add_constraint("AS_point_1.L_equals_W", equals=0.0)
prob.model.add_constraint("AS_point_1.wing_perf.failure", upper=0.0)
prob.model.add_constraint("fuel_vol_delta.fuel_vol_delta", lower=0.0)
prob.model.add_constraint("fuel_diff", equals=0.0)

# =============================================================================
# 5. EXECUTION
# =============================================================================
prob.setup()
prob.model.AS_point_0.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=30, use_aitken=True)
prob.model.AS_point_1.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=30, use_aitken=True)
prob.run_driver()

print("\n--- Aerostructural Wingbox Results ---")
print(f"Final fuel burn: {prob.get_val('AS_point_0.fuelburn')[0]:.4f} [kg]")
print(f"Final structural mass: {prob.get_val('wing.structural_mass')[0] / surf_dict['wing_weight_ratio']:.4f} [kg]")
