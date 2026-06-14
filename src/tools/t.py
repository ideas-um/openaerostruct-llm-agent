prob = 2

# === AGENT EDITABLE SECTION START ===
prob.model.add_design_var("alpha", lower=0.0, upper=10.0)
prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
prob.model.add_design_var("wing.thickness_cp", lower=0.005, upper=0.04, ref=0.01)

prob.model.add_constraint("AS_point_0.wing_perf.failure", upper=0.0)
prob.model.add_constraint("AS_point_0.L_equals_W", equals=0.0)

prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-2)
# === AGENT EDITABLE SECTION END ===