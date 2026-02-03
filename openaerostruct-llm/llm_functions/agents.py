import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))  # Add current folder to path

import json
import re
import time

# Updated imports to match your new gemini_config and the new SDK
from gemini_config import get_client
from ollama_config import get_model as get_ollama_model
from google.genai import types

gemini_model = "gemma-3-27b-it"
advanced_gemini_model = "gemma-3-27b-it" #"gemini-2.5-flash" #temporarily changed back to 27b due to limit issues

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
                continue  # Try next match

    return None  # No valid JSON found


class GeneralAgent:
    def __init__(self, name, role, prompt, schema=None, PDFs=None, backend="gemini"):
        self.name = name
        self.role = role
        self.prompt = prompt
        self.schema = schema
        self.PDFs = PDFs
        self.backend = backend
        
        # New SDK initialization
        if backend == "gemini":
            self.client = get_client()
            self.model_id = gemini_model
        else:
            self.model = get_ollama_model()

    def role_description(self):
        return f"You are {self.name}, specializing in {self.role}."

    def execute_task(self, task_description):
        full_prompt = (
            f"{self.prompt}\nYou should respond with schema: {json.dumps(self.schema)}\n"
            f"Task: {task_description}. Do NOT include any 'type', 'properties', "
            f"or structure definitions. Only return the dictionary."
        )

        if self.backend == "gemini":
            # New SDK format for content (text + media parts)
            contents = [full_prompt]
            if self.PDFs:
                for pdf_path in self.PDFs:
                    try:
                        with open(pdf_path, "rb") as f:
                            # Create a Part object for the PDF
                            pdf_part = types.Part.from_bytes(
                                data=f.read(),
                                mime_type="application/pdf"
                            )
                            contents.append(pdf_part)
                    except Exception as e:
                        print(f"Error loading PDF {pdf_path}: {e}")
            
            # Use client.models.generate_content for the new SDK
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents
            )
        else:
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
            prompt="""Your goal is to rephrase the user's input into a format that OpenAeroStruct can understand and output using the provided schema, which is an object.

            Your tasks are:
            1. Read the user's input.
            2. Identify the key information and requirements.
            3. Use the provided schema to structure your response.
            4. Generate a response that clearly outlines the key requirements and constraints for the OpenAeroStruct optimization process.
            5. Ensure that the response is well-formatted and easy to understand.
            6. Use the format below to structure your response.

            Include the following information in your response as applicable:
            Objective Function:     Maximize or minimize objectives.
            Trim Condition:         If applicable; for example, minimize drag at $C_L = 0.5$ or $\alpha = 3.0$ deg.
            Geometric Constraints:  Wing area (S), root chord, tip chord, aspect ratio, etc.
            Design Variables:       Sweep, twist, taper, chord, etc.
            Baseline Wing Mesh:     Unless specified, use a rectangular mesh. If using a common research model, use CRM.
            Optimization Algorithm: Unless specified, use SLSQP.
            Plotting Requirements:  If any, otherwise state none.
            Errors:             Note any errors in the input, such as missing key information (e.g., no objective specified), or if the number of constraints exceeds the number of design variables.

            """,
            backend="gemini",
        )

class ResultsReaderAgent(GeneralAgent):

    def __init__(self):
        # Get absolute path to 'figures' folder regardless of OS or execution dir
        # agents.py is in 'llm_functions', so we go up 2 levels
        base_dir = Path(__file__).resolve().parent.parent
        figures_dir = base_dir / "figures"
        
        pdf_list = [
            str(figures_dir / "Opt_History.pdf"),
            str(figures_dir / "Optimized_Wing.pdf")
        ]

        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "Analysis": {"type": "string", "description": ""},
                    "Recommendations": {"type": "string", "description": ""},
                    "Optimization Performance": {"type": "string", "description": ""},
                    "Unrelated Observations": {"type": "string", "description": ""},
                },
            },

            PDFs=pdf_list,

            name="Results Reader and Recommender",
            role="Read the visual results and report on the key characteristics shown by them",
            prompt="""Your goal is to review the results of the OpenAeroStruct optimization and provide recommendations. Output should use the provided schema, which is an object. The initial problem statement and two PDF reports of the results will be provided.

            Your tasks are:
            1. Read the PDFs of the OpenAeroStruct optimization results.
            2. Identify the key information and results.
            3. Use the provided schema to structure your response.

            Structure your response in four parts:
            1) Analysis: Assess whether the optimization was successful, and explain what the results mean.
            2) Recommendations: Suggest next steps, such as further optimizations, adjustments to design variables, or other considerations. Evaluate whether the results are reasonable; if not, explain why.
            3) Optimization Performance: Discuss the computational performance, including the number of iterations, computation time, and, if unconverged, suggest an alternative optimization algorithm.
            4) Unrelated Observations: Note any other observations not directly related to the optimization, such as manufacturability or non-design variables.

            Include the following details:
            1) Does the optimization achieve the objectives? If not, why? State numerical values of the objective.
            2) List key design variables and their values: Do they reach their limits? Are they physically reasonable?
            3) Analyze graphical plots: Do they make sense? Are there anomalies? For drag minimization, is lift distribution elliptical? How close is it to ideal?
            4) Report computational performance.

            It is important to note that you are using OpenAeroStruct, which is a Vortex Lattice Method (VLM) based aerodynamic solver, you should analyze the results with that in mind.
            """,
            backend="gemini",
        )

class ReportWriter(GeneralAgent):
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        figures_dir = base_dir / "figures"
        
        pdf_list = [
            str(figures_dir / "Opt_History.pdf"),
            str(figures_dir / "Optimized_Wing.pdf")
        ]

        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "ReportText": {"type": "string", "description": ""},
                },
            },

            name="Report Writer",
            role="Using Latex write a report on the optimization results",
            PDFs=pdf_list,
            prompt=f"""
            Your goal is to rewrite the LLM output into a report format, using the schema provided (which is an object). You will be given the textual analysis from another LLM.

            Your tasks are:
            1. Read the analysis and recommendations.
            2. Identify the key information and requirements.
            3. Use the provided schema to structure your response.
            4. Generate a response that answers the user's question in paragraph form, formatted in properly written LaTeX.

            Use all the information given to write a detailed analysis of the results and recommendations.

            Please include this figure in the report:
            The file path is "../figures/Optimized_Wing.pdf", which contains the optimized wing visualization. Reference this figure in your analysis, as it will also be provided.

            Today's date is {time.strftime("%Y-%m-%d")}. Please include the date in the report.
            """,
            backend="gemini",
        )

class BaseCoderAgent:
    def __init__(self, name, role, prompt, schema = None):
        self.name = name
        self.role = role
        self.prompt = prompt
        self.schema = schema
        self.client = get_client()
        self.model_id = gemini_model

    def role_description(self):
        return f"You are {self.name}, a coder specializing in {self.role}."

    def execute_task(self, task_description):
        full_prompt = f"{self.prompt}\nTask: {task_description}\nYou should respond with schema: {json.dumps(self.schema)}\nTask: {task_description}, for coding, do not import any packages yourself"
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=full_prompt
        )
        output = extract_json_between_markers(response.text)
        return output
    
class AdvancedCoderAgent:
    def __init__(self, name, role, prompt, schema = None):
        self.name = name
        self.role = role
        self.prompt = prompt
        self.schema = schema
        self.client = get_client()
        self.model_id = advanced_gemini_model

    def role_description(self):
        return f"You are {self.name}, a coder specializing in {self.role}."

    def execute_task(self, task_description):
        full_prompt = f"{self.prompt}\nTask: {task_description}\nYou should respond with schema: {json.dumps(self.schema)}\nTask: {task_description}, for coding, do not import any packages yourself"
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=full_prompt
        )
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
            prompt="""Your goal is to implement the instructions provided and write OpenAeroStruct mesh code. Follow the instructions and the code samples carefully, and output using the provided schema as text.

            Your tasks are:
            1. Read the inputs. You will be given the wing type and geometrical configuration. If the requirements call for a change, update the span, root chord, and wing type, then output the code.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.

            You may need to perform basic math calculations for mesh generation. For a rectangular wing, if span and area are given, calculate the root chord as follows: root_chord = area / span.

            Follow the explained code sample provided. Do not add new fields—directly edit the code and explain any changes in the calculations and in the explanation field.

            This is an explained code sample for you to follow. Do not add new fields, directly edit the code and explain your changes in the calculations and explain field.
            ```python
            # This is a sample code to generate a mesh for a rectangular wing
            mesh_dict = {
                "num_y": 19, #number of panels in the y direction, 19 is a good starting number
                "num_x": 3, #number of panels in the x direction, 3 is a number
                "wing_type": "rect", #This can either be "rect" or "crm" only
                "symmetry": True, # True if the wing is symmetric, False if it is not, wings are typically symmetric
                "span": 2.0, #This is the full span of the wing in meters
                "root_chord": 1.0, #This is the root chord of the wing in meters
                "span_cos_spacing": 0.0, #This is usually not edited
                "chord_cos_spacing": 0.0, #This is usually not edited
            }

            # Generate VLM mesh for half-wing
            mesh = generate_mesh(mesh_dict)   # this creates a rectangular wing mesh, DO NOT EDIT THIS LINE

            # plot mesh
            plot_mesh(mesh)  #this plots the rectangular wing mesh, DO NOT EDIT THIS LINE
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
            prompt="""Your goal is to follow the instructions provided and write OpenAeroStruct geometry code. Carefully follow the instructions and code samples, and output using the provided schema as text.

            Your tasks are:
            1. Read the inputs. You will be given the variables that can be optimized. Update the geometry preparation code and uncomment variables that can be optimized.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.

            Follow the explained code sample provided. Do not add new fields—directly edit the code and explain your changes in the calculations and explanation field.

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
                #"dihedral": 3.0, # if the wing has dihedral, uncomment this line and change the initial value for how much dihedral, default is 3.0
                #"twist_cp" : np.zeros(2),  # if the wing can be twisted, uncomment this line and change the value for how many points for the bspline to change the twist, default is 4
            }  # end of surface dictionary
            ,,,
            """
        )

class OptimizerAgent(AdvancedCoderAgent):
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
            prompt="""
            Your goal is to follow the instructions provided and write OpenAeroStruct optimization code. Carefully follow the instructions and code samples, and output using the provided schema as text. Do not add package imports and do not do the parts that are not assigned to you.

            Your tasks are:
            1. Read the inputs. You will be given the variables that can be optimized. Update the optimization code to include the variables for optimization.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.
            5. DO NOT ADD ANYTHING THAT IS NOT ASKED FOR AND NOT IN THE SAMPLE CODE.

            Do not add these lines as they are not avaliable:
            prob.model.add_constraint('wing.S_ref', equals=400.0)
            prob.model.add_constraint('wing.b_half_w', equals=30.0) # Half span

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
            #You are also allowed to add the design varaibles, constraints, and objectives here like chord_cp, twist_cp, taper, sweep, dihedral etc.
            #The way to add them is wing."var_name" (for example, wing.taper) and the lower and upper bounds are in the form of lower=0.0, upper=1.0
            #these are the var names that you can use taper = taper, sweep = sweep, chord_cp = chord_cp, twist_cp = twist_cp, dihedral = dihedral
            #remember to add alpha as a design variable if CL is a constraint. 
            #DO NOT ADD THE AREA AND SPAN CONSTRAINTS HERE AS THEY DO NOT WORK YET.

            prob.model.add_design_var('wing.taper', lower=0.2, upper=1.0)  # Varies the taper ratio
            prob.model.add_design_var('alpha', units='deg', lower=0., upper=10.)   # varies
            prob.model.add_constraint('flight_condition_0.wing_perf.CL', equals=x)   # impose CL = x, where x is a number
            prob.model.add_objective('flight_condition_0.wing_perf.CD', ref=0.01)   # dummy objective to minimize CD.
            ############# THIS END OF THE PART TO EDIT ##########

            # use Scipy's SLSQP optimization
            prob.driver = om.ScipyOptimizeDriver()

            # record optimization history
            recorder = om.SqliteRecorder("aero.db")
            prob.driver.add_recorder(recorder)
            prob.driver.recording_options["includes"] = ["*"]
            prob.options['work_dir'] = './openaerostruct_out'

            prob.setup()
            prob.run_driver() 

            #print results
            print("\nAngle of attack =", prob.get_val("alpha", units="deg")[0], "deg")
            print("CL = ", prob.get_val("flight_condition_0.wing_perf.CL")[0])
            print("CD = ", prob.get_val("flight_condition_0.wing_perf.CD")[0])
            ,,,
            """
        )


class FinalCoderAgent(AdvancedCoderAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "mesh_code": {"type": "string", "description": "The tidied mesh code"},
                    "geometry_code": {"type": "string", "description": "The tidied geometry code"},
                    "optimizer_code": {"type": "string", "description": "The tidied optimizer code"},
                    "issues_found": {"type": "string", "description": "Any issues or corrections made"},
                },
            },
            name="Final Coder and Checker",
            role="Review and tidy up combined code sections to ensure no errors",
            prompt="""
            Your goal is to review the combined code sections from multiple agents, tidy them up, and ensure there are no errors.

            Your tasks are:
            1. Review all three code sections (mesh, geometry, optimizer)
            2. Check for:
               - Proper indentation and formatting
               - Consistent variable naming across sections
               - No duplicate or conflicting code
               - Syntax errors
               - Logic errors
               - Missing imports (note: imports are handled elsewhere, don't add them)
            3. Fix any issues found
            4. Return tidied versions of all three code sections
            5. Document any issues found and corrections made

            Important notes:
            - Do NOT add import statements
            - Do NOT modify the core logic unless there's an error
            - Ensure proper Python indentation (4 spaces)
            - Make sure variable names are consistent (e.g., 'mesh', 'surface', 'prob')
            - Remove any unnecessary whitespace or comments that don't add value
            
            Output the tidied code sections in the schema provided.
            """
        )