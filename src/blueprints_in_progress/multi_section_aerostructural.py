import numpy as np
import os
import openmdao.api as om
from openaerostruct.integration.aerostruct_groups import MultiSecAerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant
from openaerostruct.geometry.utils import build_section_dicts, unify_mesh, build_multi_spline, connect_multi_spline

# =============================================================================
# 1. MULTI-SECTION GEOMETRY DEFINITION
# =============================================================================
sec_twist_cp = [np.zeros(2), np.zeros(2)]

surface = {
    "name": "surface",
    "is_multi_section": True,
    "num_sections": 2,
    "sec_name": ["sec0", "sec1"],
    "symmetry": True,
    "S_ref_type": "wetted",
    "root_section": 1,
    "taper": [1.0, 1.0],
    "span": [10.0, 10.0],
    "sweep": [0.0, 0.0],
    "twist_cp": sec_twist_cp,
    "t_over_c_cp": [np.array([0.15]), np.array([0.15])],
    "root_chord": 5.0,
    "meshes": "gen-meshes",
    "nx": 3,
    "ny": [5, 5],
    "CL0": 0.0,
    "CD0": 0.015,
    "k_lam": 0.05,
    "c_max_t": 0.303,
    "with_viscous": True,
    "with_wave": False,
    "fem_model_type": "tube",
    "thickness_cp": 0.1 * np.ones((2)),
    "E": 70.0e9,
    "G": 30.0e9,
    "yield": 500.0e6 / 2.5,
    "mrho": 3.0e3,
    "fem_origin": 0.35,
    "wing_weight_ratio": 2.0,
    "struct_weight_relief": False,
    "distributed_fuel_weight": False,
    "exact_failure_constraint": False,
}

# =============================================================================
# 2. PROBLEM SETUP
# =============================================================================
prob = om.Problem()

indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output("alpha", val=9.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s")
indep_var_comp.add_output("R", val=11.165e6, units="m")
indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")
indep_var_comp.add_output("speed_of_sound", val=295.4, units="m/s")
indep_var_comp.add_output("load_factor", val=1.0)
indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

section_surfaces = build_section_dicts(surface)
uniMesh = unify_mesh(section_surfaces)
surface["mesh"] = uniMesh

twist_comp = build_multi_spline("twist_cp", len(section_surfaces), sec_twist_cp)
prob.model.add_subsystem("twist_bspline", twist_comp)
connect_multi_spline(prob, section_surfaces, sec_twist_cp, "twist_cp", "twist_bspline", surface["name"])

multi_geom_group = MultiSecAerostructGeometry(surface=surface)
prob.model.add_subsystem(surface["name"], multi_geom_group)

name = surface["name"]
point_name = "AS_point_0"
AS_point = AerostructPoint(surfaces=[surface])
prob.model.add_subsystem(point_name, AS_point, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "CT", "R", "W0", "speed_of_sound", "empty_cg", "load_factor"])

com_name = point_name + "." + name + "_perf"
prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")
prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
prob.model.connect(name + ".radius", com_name + ".radius")
prob.model.connect(name + ".thickness", com_name + ".thickness")
prob.model.connect(name + ".nodes", com_name + ".nodes")
prob.model.connect(name + ".cg_location", point_name + ".total_perf." + name + "_cg_location")
prob.model.connect(name + ".structural_mass", point_name + ".total_perf." + name + "_structural_mass")
prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")

# =============================================================================
# 3. OPTIMIZATION SETTINGS
# =============================================================================
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-6

os.makedirs("src/openaerostruct_out/generated_run_out", exist_ok=True)
recorder = om.SqliteRecorder("src/openaerostruct_out/generated_run_out/aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]

prob.model.add_design_var("twist_bspline.twist_cp_spline", lower=-10.0, upper=15.0)
prob.model.add_design_var("surface.thickness_cp", lower=0.01, upper=0.5, scaler=1e2)
prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
prob.model.add_constraint("AS_point_0.surface_perf.failure", upper=0.0)
prob.model.add_constraint("AS_point_0.surface_perf.thickness_intersects", upper=0.0)
prob.model.add_constraint("AS_point_0.L_equals_W", equals=0.0)
prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)

# =============================================================================
# 4. EXECUTION
# =============================================================================
prob.setup()
prob.run_driver()

print("\n--- Multi-Section Aerostructural Results ---")
print(f"Final fuel burn: {prob.get_val('AS_point_0.fuelburn')[0]:.4f} [kg]")
