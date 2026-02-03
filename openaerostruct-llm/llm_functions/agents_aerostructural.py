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


class ReformulatorAgentAerostructural(GeneralAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "objective_function": {"type": "string", "description": ""},
                    "trim_condition": {"type": "string", "description": ""},
                    "geometric_constraint": {"type": "string", "description": ""},
                    "design_variables": {"type": "string", "description": ""},
                    "structural_constraints": {"type": "string", "description": ""},
                    "baseline_wing_mesh": {"type": "string", "description": ""},
                    "optimization_algorithm": {"type": "string", "description": ""},
                    "plotting_requirements": {"type": "string", "description": ""},
                    "errors": {"type": "string", "description": ""}
                },
            },

            name="Aerostructural Prompt Reformulator",
            role="Changing the user prompt to a format that OpenAeroStruct aerostructural analysis can understand",
            prompt="""Your goal is to rephrase the user's input into a format that OpenAeroStruct AEROSTRUCTURAL optimization can understand and output using the provided schema, which is an object.

            Your tasks are:
            1. Read the user's input.
            2. Identify the key information and requirements for AEROSTRUCTURAL optimization (coupling aerodynamics and structures).
            3. Use the provided schema to structure your response.
            4. Generate a response that clearly outlines the key requirements and constraints for the OpenAeroStruct AEROSTRUCTURAL optimization process.
            5. Ensure that the response is well-formatted and easy to understand.
            6. Use the format below to structure your response.

            Include the following information in your response as applicable:
            Objective Function:         Typically minimize fuel burn, weight, or maximize range for aerostructural problems.
            Trim Condition:             If applicable; for example, lift equals weight constraint.
            Geometric Constraints:      Wing area (S), span, aspect ratio, etc.
            Design Variables:           For aerostructural: twist, thickness distribution, alpha, etc.
            Structural Constraints:     Failure constraints, thickness intersection constraints, stress limits.
            Baseline Wing Mesh:         Unless specified, use CRM mesh for aerostructural problems.
            Optimization Algorithm:     Unless specified, use SLSQP.
            Plotting Requirements:      If any, otherwise state none.
            Errors:                     Note any errors in the input, such as missing key information.

            """,
            backend="gemini",
        )

class ResultsReaderAgentAerostructural(GeneralAgent):

    def __init__(self):
        # Get absolute path to 'figures' folder
        # Goes up 2 levels from this script: llm_functions/ -> Root/ -> figures/
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

            name="Aerostructural Results Reader and Recommender",
            role="Read the visual results and report on the key characteristics of aerostructural optimization",
            prompt="""Your goal is to review the results of the OpenAeroStruct AEROSTRUCTURAL optimization and provide recommendations. Output should use the provided schema, which is an object. The initial problem statement and two PDF reports of the results will be provided.

            Your tasks are:
            1. Read the PDFs of the OpenAeroStruct aerostructural optimization results.
            2. Identify the key information and results related to both aerodynamics AND structures.
            3. Use the provided schema to structure your response.

            Structure your response in four parts:
            1) Analysis: Assess whether the aerostructural optimization was successful. Explain what the results mean in terms of aerodynamic performance, structural integrity, and their coupling.
            2) Recommendations: Suggest next steps, such as further optimizations, adjustments to design variables, or other considerations for aerostructural design.
            3) Optimization Performance: Discuss the computational performance, including the number of iterations, computation time, and convergence behavior.
            4) Unrelated Observations: Note any other observations not directly related to the optimization.

            Include the following details:
            1) Does the optimization achieve the objectives (e.g., minimize fuel burn)? State numerical values.
            2) List key design variables (twist, thickness, alpha) and their values: Do they reach their limits?
            3) Are structural constraints satisfied (failure, thickness intersections)?
            4) Analyze the trade-offs between aerodynamic efficiency and structural weight.
            5) Report computational performance.

            Note: This is an aerostructural optimization coupling aerodynamics (VLM) and structures (beam FEM).
            """,
            backend="gemini",
        )

class ReportWriterAerostructural(GeneralAgent):
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

            name="Aerostructural Report Writer",
            role="Using Latex write a report on the aerostructural optimization results",
            PDFs=pdf_list, # <--- Updated variable here
            prompt=f"""
            Your goal is to rewrite the LLM output into a report format for AEROSTRUCTURAL optimization, using the schema provided (which is an object). You will be given the textual analysis from another LLM.

            Your tasks are:
            1. Read the analysis and recommendations.
            2. Identify the key information and requirements for aerostructural optimization.
            3. Use the provided schema to structure your response.
            4. Generate a response that answers the user's question in paragraph form, formatted in properly written LaTeX.

            Use all the information given to write a detailed analysis of the aerostructural results and recommendations.
            Emphasize the coupling between aerodynamics and structures, and how design decisions affect both domains.

            Please include this figure in the report:
            The file path is "../figures/Optimized_Wing.pdf", which contains the optimized wing visualization. Reference this figure in your analysis.

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
        full_prompt = f"{self.prompt}\nYou should respond with schema: {json.dumps(self.schema)}\nTask: {task_description}, for coding, do not import any packages yourself"
        
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
        full_prompt = f"{self.prompt}\nYou should respond with schema: {json.dumps(self.schema)}\nTask: {task_description}, for coding, do not import any packages yourself"
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=full_prompt
        )
        output = extract_json_between_markers(response.text)
        return output


class BaseMeshAgentAerostructural(BaseCoderAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "calculations_and_explain": {"type": "string", "description": "The steps you took to get the code"},
                    "python_code": {"type": "string", "description": "Only the code to generate the mesh"},
                },
            },
            name="Aerostructural Mesh Coder",
            role="Mesh generation for aerostructural analysis",
            prompt="""Your goal is to implement the instructions provided and write OpenAeroStruct mesh code FOR AEROSTRUCTURAL ANALYSIS. Follow the instructions and the code samples carefully, and output using the provided schema as text.

            Your tasks are:
            1. Read the inputs. You will be given the wing type and geometrical configuration.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.

            IMPORTANT: For aerostructural analysis, you must also specify num_twist_cp in the mesh_dict for twist control points.

            This is an explained code sample for you to follow for AEROSTRUCTURAL mesh generation:
            ```python
            # Aerostructural mesh generation
            mesh_dict = {
                "num_y": 5,  # number of panels in the y direction for aerostructural (fewer than pure aero)
                "num_x": 2,  # number of panels in the x direction
                "wing_type": "CRM",  # "rect" or "CRM" - CRM is common for aerostructural
                "symmetry": True,  # True if the wing is symmetric
                "num_twist_cp": 5,  # IMPORTANT: number of twist control points for aerostructural
            }

            # Generate mesh and twist control points
            mesh, twist_cp = generate_mesh(mesh_dict)  # Note: returns mesh AND twist_cp

            # plot mesh  
            plot_mesh(mesh)
            ```
            
            Key differences from pure aerodynamic mesh:
            - Must include num_twist_cp in mesh_dict
            - generate_mesh returns BOTH mesh and twist_cp (two return values)
            - Typically use fewer panels (num_y around 5) for computational efficiency
            """
        )

class GeometryAgentAerostructural(BaseCoderAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "calculations_and_explain": {"type": "string", "description": "The steps you took to get the code"},
                    "python_code": {"type": "string", "description": "Only the code of the geometry"},
                },
            },
            name="Aerostructural Geometry Agent",
            role="Setup geometries for aerostructural optimization",
            prompt="""Your goal is to follow the instructions provided and write OpenAeroStruct geometry code FOR AEROSTRUCTURAL ANALYSIS. Carefully follow the instructions and code samples, and output using the provided schema as text.

            Your tasks are:
            1. Read the inputs. You will be given the variables that can be optimized for aerostructural analysis.
            2. Identify the requirements.
            3. Use the provided code sample to structure your code.
            4. Ensure that the response is well-formatted and easy to understand.

            IMPORTANT: Aerostructural analysis requires ADDITIONAL structural parameters compared to pure aerodynamic analysis.

            ```python
            surface = {
                # Wing definition
                "name": "wing",  # name of the surface, keep as wing
                "symmetry": True,  # if true, model one half of wing
                "S_ref_type": "wetted",  # how we compute the wing area
                "mesh": mesh,
                
                # STRUCTURAL MODEL TYPE - REQUIRED FOR AEROSTRUCTURAL
                "fem_model_type": "tube",  # structural model type: "tube" for tube spar
                
                # Aerodynamic properties (same as pure aero)
                "CL0": 0.0,  # CL of the surface at alpha=0
                "CD0": 0.015,  # CD of the surface at alpha=0
                
                # Airfoil properties for viscous drag calculation
                "k_lam": 0.05,  # percentage of chord with laminar flow
                "c_max_t": 0.303,  # chordwise location of maximum thickness
                "t_over_c_cp": np.array([0.15]),  # thickness-to-chord ratio
                
                "with_viscous": True,  # if true, compute viscous drag
                "with_wave": False,  # if true, compute wave drag
                
                # STRUCTURAL PROPERTIES - REQUIRED FOR AEROSTRUCTURAL
                "E": 70.0e9,  # [Pa] Young's modulus (aluminum 7075)
                "G": 30.0e9,  # [Pa] shear modulus
                "yield": 500.0e6,  # [Pa] yield stress
                "safety_factor": 2.5,  # safety factor for stress
                "mrho": 3.0e3,  # [kg/m^3] material density
                "fem_origin": 0.35,  # normalized chordwise location of the spar
                "wing_weight_ratio": 2.0,  # ratio of non-structural to structural weight
                "struct_weight_relief": False,  # whether to include structural weight in loads
                "distributed_fuel_weight": False,  # whether fuel weight is distributed
                
                # STRUCTURAL DESIGN VARIABLES - UNCOMMENT AS NEEDED
                "thickness_cp": np.array([0.1, 0.2, 0.3]),  # thickness control points (design variable)
                "twist_cp": twist_cp,  # twist control points from mesh generation (design variable)
                
                # Constraints
                "exact_failure_constraint": False,  # if false, use KS function for failure
            }
            ```
            
            Key differences from pure aerodynamic geometry:
            - Must include fem_model_type (typically "tube")
            - Must include structural material properties (E, G, yield, mrho, etc.)
            - thickness_cp is a key design variable for structural optimization
            - twist_cp comes from the mesh generation step
            - Additional structural constraints available (failure, thickness_intersects)
            """
        )

class OptimizerAgentAerostructural(AdvancedCoderAgent):
    def __init__(self):
        super().__init__(
            schema = {
                "type": "object",
                "properties": {
                    "calculations_and_explain": {"type": "string", "description": "The steps you took to get the code"},
                    "python_code": {"type": "string", "description": "Only the code of the optimization script"},
                },
            },
            name="Aerostructural Optimization Agent",
            role="Setup aerostructural optimization script to run OpenAeroStruct",
            prompt="""
            Your goal is to follow the instructions provided and write OpenAeroStruct AEROSTRUCTURAL optimization code. This is different from pure aerodynamic optimization as it couples aerodynamics and structures. Carefully follow the instructions and code samples, and output using the provided schema as text.

            Your tasks are:
            1. Read the inputs for AEROSTRUCTURAL optimization requirements.
            2. Use the provided code sample to structure your code.
            3. Setup proper design variables, constraints, and objectives for aerostructural problems.
            4. DO NOT ADD ANYTHING THAT IS NOT ASKED FOR.

            This is an explained code sample for AEROSTRUCTURAL optimization:
            ```python
            # Create the problem
            prob = om.Problem()

            # Add problem information as independent variables
            indep_var_comp = om.IndepVarComp()
            indep_var_comp.add_output("v", val=248.136, units="m/s")  # velocity
            indep_var_comp.add_output("alpha", val=5.0, units="deg")  # angle of attack
            indep_var_comp.add_output("Mach_number", val=0.84)  # Mach number
            indep_var_comp.add_output("re", val=1.0e6, units="1/m")  # Reynolds number
            indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")  # air density
            indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s")  # thrust specific fuel consumption
            indep_var_comp.add_output("R", val=11.165e6, units="m")  # range
            indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")  # initial weight
            indep_var_comp.add_output("speed_of_sound", val=295.4, units="m/s")
            indep_var_comp.add_output("load_factor", val=1.0)  # load factor (1.0 for cruise, >1 for maneuver)
            indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")  # center of gravity

            prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

            # IMPORTANT: Use AerostructGeometry instead of Geometry
            from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
            
            aerostruct_group = AerostructGeometry(surface=surface)
            name = "wing"
            prob.model.add_subsystem(name, aerostruct_group)

            point_name = "AS_point_0"  # Note: AS_point for aerostructural

            # IMPORTANT: Use AerostructPoint instead of AeroPoint
            AS_point = AerostructPoint(surfaces=[surface])

            prob.model.add_subsystem(
                point_name,
                AS_point,
                promotes_inputs=[
                    "v", "alpha", "Mach_number", "re", "rho",
                    "CT", "R", "W0", "speed_of_sound", "empty_cg", "load_factor",
                ],
            )

            # Connect geometry to aerostructural point
            com_name = point_name + "." + name + "_perf"
            prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
            prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")
            prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
            
            # Connect performance calculation variables
            prob.model.connect(name + ".radius", com_name + ".radius")
            prob.model.connect(name + ".thickness", com_name + ".thickness")
            prob.model.connect(name + ".nodes", com_name + ".nodes")
            prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
            prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
            prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")

            # Setup optimizer
            prob.driver = om.ScipyOptimizeDriver()
            prob.driver.options["tol"] = 1e-9

            ########## EDIT THIS SECTION ##########
            # Common aerostructural design variables:
            # - wing.twist_cp: twist distribution (aerodynamic and structural)
            # - wing.thickness_cp: thickness distribution (structural)
            # - alpha: angle of attack (aerodynamic)
            
            # Common aerostructural constraints:
            # - AS_point_0.wing_perf.failure: structural failure constraint (upper=0.0)
            # - AS_point_0.wing_perf.thickness_intersects: thickness intersection constraint (upper=0.0)
            # - AS_point_0.L_equals_W: lift equals weight constraint (equals=0.0)
            
            # Common aerostructural objectives:
            # - AS_point_0.fuelburn: minimize fuel burn
            # - AS_point_0.total_perf.structural_mass: minimize structural weight
            
            prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
            prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, scaler=1e2)
            prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
            
            prob.model.add_constraint("AS_point_0.wing_perf.failure", upper=0.0)
            prob.model.add_constraint("AS_point_0.wing_perf.thickness_intersects", upper=0.0)
            prob.model.add_constraint("AS_point_0.L_equals_W", equals=0.0)
            
            prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)
            ########## END EDIT SECTION ##########

            # Setup and run
            prob.setup(check=True)
            prob.run_driver()
            ```
            
            Key differences from pure aerodynamic optimization:
            - Import AerostructGeometry and AerostructPoint from openaerostruct.integration.aerostruct_groups
            - Add structural variables (CT, R, W0, load_factor, etc.)
            - Use AerostructGeometry instead of Geometry
            - Use AerostructPoint instead of AeroPoint
            - Additional connections for structural properties (local_stiff_transformed, nodes, etc.)
            - Structural design variables (thickness_cp) and constraints (failure, thickness_intersects)
            - Typical objective is fuelburn instead of CD
            """
        )


class FinalCoderAgentAerostructural(AdvancedCoderAgent):
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
            name="Aerostructural Final Coder and Checker",
            role="Review and tidy up combined aerostructural code sections to ensure no errors",
            prompt="""
            Your goal is to review the combined code sections from multiple agents for AEROSTRUCTURAL optimization, tidy them up, and ensure there are no errors.

            Your tasks are:
            1. Review all three code sections (mesh, geometry, optimizer) for aerostructural analysis
            2. Check for:
               - Proper indentation and formatting
               - Consistent variable naming (especially 'mesh' and 'twist_cp' from mesh generation)
               - Correct use of AerostructGeometry and AerostructPoint (not Geometry and AeroPoint)
               - Proper structural properties and constraints
               - Syntax errors
               - Logic errors
            3. Fix any issues found
            4. Return tidied versions of all three code sections
            5. Document any issues found and corrections made

            Important notes specific to AEROSTRUCTURAL:
            - Mesh generation should return TWO values: mesh, twist_cp = generate_mesh(mesh_dict)
            - Geometry must use twist_cp from mesh generation
            - Geometry must include structural properties (E, G, yield, etc.)
            - Optimizer must import from openaerostruct.integration.aerostruct_groups
            - Optimizer must use AerostructGeometry and AerostructPoint
            - Common mistake: using Geometry/AeroPoint instead of AerostructGeometry/AerostructPoint
            
            Output the tidied code sections in the schema provided.
            """
        )
