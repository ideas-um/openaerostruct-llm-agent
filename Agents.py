# In Agents.py
from geminiconfig import get_model
import json
import re

def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                print(f'Error in processing json {json_string}')
                continue  # Try next match

    return None  # No valid JSON found


class GeneralAgent:
    def __init__(self, name, role, prompt, schema=None):
        self.name = name
        self.role = role
        self.prompt = prompt
        self.schema = schema
        self.model = get_model()

    def role_description(self):
        return f"You are {self.name}, specializing in {self.role}."

    def execute_task(self, task_description):
        full_prompt = f"{self.prompt}\nYou should respond with schema: {json.dumps(self.schema)}\nTask: {task_description}"
        response = self.model.generate_content(full_prompt)

        output = extract_json_between_markers(response.text)

        return output


class ReformulatorAgent(GeneralAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "objective_function": {"type": "string", "description": ""},
                    "trim_condition": {"type": "string", "description": ""},
                    "geometric_constraint": {"type": "string", "description": ""},
                    "design_variables": {"type": "string", "description": ""},
                    "baseline_wing_mesh": {"type": "string", "description": ""},
                    "optimization_algorithm": {"type": "string", "description": ""},
                    "plotting_requirements": {"type": "string", "description": ""},
                    "errors": {"type": "string", "description": ""}
                },
            },

            name="Prompt Reformulator",
            role="Changing the user prompt to a format that OpenAeroStruct can understand",
            prompt="""Your goal is to rephrase the user's input into a format that OpenAeroStruct can understand. And output using the schema provided which is an object.
            
            This is your task:
            1. Read the user's input.
            2. Identify the key information and requirements.
            3. Use the provided schema to structure your response.
            4. Generate a response that clearly outlines the key requirements and constraints for the OpenAeroStruct optimization process.
            5. Ensure that the response is well-formatted and easy to understand.
            6. Use the format below to structure your response.
    
            These are suggested information you need to include in your response:
            Objective Function:     : Maximize / Minimize objectives.
            Trim Condition          : If any, for example, minimize drag at CL = 0.5 or alpha = 3.0 deg.
            Geometric Constraint    : Wing area (S), root chord, tip chord, aspect ratio etc.
            Design Variables:       : Sweep, Twist, Taper, Chord, etc.,
            Baseline Wing Mesh      : Unless Specified, use rect, if common research model, use CRM.
            Optimization Algorithm  : Unless Specified use SLSQP. 
            Plotting Requirements   : If any, else none.
            Errors                  : Any errors in the input. E.g. Missing key information like objective, or if number of constraints > number of design variables etc.

            """
        )

class GraphReaderAgent(GeneralAgent):
    def __init__(self):
        super().__init__(
            name="GraphAnalyst",
            role="Graph data analysis",
            prompt="You are an expert in graph theory. Your goal is to analyze and interpret network structures."
        )

class BaseCoderAgent:
    def __init__(self, name, role, prompt, schema = None):
        self.name = name
        self.role = role
        self.prompt = prompt
        self.model = get_model()
        self.schema = schema

    def role_description(self):
        return f"You are {self.name}, a coder specializing in {self.role}."

    def execute_task(self, task_description):
        full_prompt = f"{self.prompt}\nTask: {task_description}\nYou should respond with schema: {json.dumps(self.schema)}\nTask: {task_description}"
        response = self.model.generate_content(full_prompt)
        output = extract_json_between_markers(response.text)
        return output


class BaseMeshAgent(BaseCoderAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "calculations_and_explain": {"type": "string", "description": "The steps you took to get the code"},
                    "python_code": {"type": "string", "description": "Only the code to generate the mesh"},
                },
            },
            name="Mesh Coder",
            role="Mesh generation",
            prompt="""Your goal is to follow some instructions that I provide to you and write OpenAeroStruct mesh code, please follow the instructions and the code samples clearly. And output using the schema provided which is text.
            
            This is your task:
            1. Read these inputs. You will be given the wing type, and the geometrical configuration. Your job is to change the span, root chord, and type of the wing if the user requirements invoke change and output the code.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.

            You might also need to do basic math calculations to get the values for the mesh generation.
            For a rectangular wing, if the span and area is given, you can calculate the root chord using the formula:
            root_chord = area / span

            This is an explained code sample for you to follow. Do not add new fields, directly edit the code and explain your changes in the calculations and explain field.
            ```python
            # This is a sample code to generate a mesh for a rectangular wing
            mesh_dict = {
                "num_y": 19, #number of panels in the y direction, 19 is a good starting number
                "num_x": 3, #number of panels in the x direction, 3 is a good starting number
                "wing_type": "rect", #This can either be "rect" or "crm" only
                "symmetry": True, # True if the wing is symmetric, False if it is not, wings are typically symmetric
                "span": 2.0, #This is the full span of the wing in meters
                "root_chord": 1.0, #This is the root chord of the wing in meters
                "span_cos_spacing": 0.0, #This is usually not edited
                "chord_cos_spacing": 0.0, #This is usually not edited
            }

            # Generate VLM mesh for half-wing
            mesh = generate_mesh(mesh_dict)   # this creates a rectangular wing mesh, do not edit

            # plot mesh
            plot_mesh(mesh) # this plots the rectangular wing mesh, do not edit
            ,,,
            """
        )

class GeometryAgent(BaseCoderAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "calculations_and_explain": {"type": "string", "description": "The steps you took to get the code"},
                    "python_code": {"type": "string", "description": "Only the code to of the geometry"},
                },
            },
            name="Geometry Agent",
            role="Setup geometries allowed to be optimized",
            prompt="""Your goal is to follow some instructions that I provide to you and write OpenAeroStruct mesh code, please follow the instructions and the code samples clearly. And output using the schema provided which is text.
            
            This is your task:
            1. Read the inputs. You will be given the variables that cna be optimized. Your job is to change the geometry preparation code and uncomment variables that could be optimized.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.

            This is an explained code sample for you to follow. Do not add new fields, directly edit the code and explain your changes in the calculations and explain field.
            ```python
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
                #"taper" : 0.4, # if the wing can be tapered, uncomment this line and change the initial value for how much taper, default is 0.4
                #"sweep" : 28.0, # if the wing can be swept, uncomment this line and change the initial value for how much sweep, default is 28.0
                #"twist_cp" : np.zeros(2),  # if the wing can be twisted, uncomment this line and change the value for how many points for the bspline to change the twist, default is 4
            }  # end of surface dictionary
            ,,,
            """
        )

class OptimizerAgent(BaseCoderAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "calculations_and_explain": {"type": "string", "description": "The steps you took to get the code"},
                    "python_code": {"type": "string", "description": "Only the code to of the optimization script"},
                },
            },
            name="Optimization Agent",
            role="Setup optimization script to run OpenAeroStruct",
            prompt="""Your goal is to follow some instructions that I provide to you and write OpenAeroStruct optimization code, please follow the instructions and the code samples clearly. And output using the schema provided which is text.
            
            This is your task:
            1. Read the inputs. You will be given the variables that can be optimized. Your job is to change the optimization code to include the variables that can be optimized.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.

            This is an explained code sample for you to follow. Directly edit the code and explain your changes in the calculations and explain field. I will tell you where you can edit.
            ```python
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

            prob.model.add_design_var('alpha', units='deg', lower=0., upper=10.)   # varies
            prob.model.add_constraint('flight_condition_0.wing_perf.CL', equals=0.5)   # impose CL = x
            prob.model.add_objective('flight_condition_0.wing_perf.CD', ref=0.01)   # dummy objective to minimize CD.
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
            ,,,
            """
        )

class PlottingAgent(BaseCoderAgent):
    def __init__(self):
        super().__init__(
            name="VisualizationCoder",
            role="Data visualization",
            prompt="You are an expert in scientific visualization. Your goal is to create clear, informative plots."
        )