#!/usr/bin/env python3
"""
OpenAeroStruct LLM Agent - Main Execution Script
Converted from OpenAeroStruct.ipynb for standalone execution
"""

# Import the generative ai library
import google.genai as genai
import ollama

# Import all the modules necessary to run OpenAeroStruct and paths
import re
import time
import os
import subprocess
import warnings
import numpy as np
import pandas as pd 
import openmdao.api as om
import json
import shutil
import platform
from datetime import datetime
import sys
import textwrap

# import OpenAeroStruct modules
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# Import the plotting libraries
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
import niceplots  # Optional but recommended

# Import the LaTeX libraries used for report format conversions
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import StringIO

# Ignore warnings and use the nice plots style
warnings.filterwarnings("ignore")

plt.style.use(
    niceplots.get_style("james-dark")
)  # Options: "doumont-light", "doumont-dark", "james-light", "james-dark"

# Set up paths - ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add the path to import llm_functions
sys.path.append(script_dir)

from llm_functions.agents import (
    ReformulatorAgent, 
    BaseMeshAgent, 
    GeometryAgent, 
    OptimizerAgent, 
    ResultsReaderAgent, 
    ReportWriter,
    FinalCoderAgent
)

from llm_functions.agents_aerostructural import (
    ReformulatorAgentAerostructural,
    BaseMeshAgentAerostructural,
    GeometryAgentAerostructural,
    OptimizerAgentAerostructural,
    ResultsReaderAgentAerostructural,
    ReportWriterAerostructural,
    FinalCoderAgentAerostructural
)


def safe_check_output(output, check_keys=None, agent_name="Agent"):
    """
    Validates agent output to ensure it meets minimum requirements.
    
    Args:
        output: The output from an agent
        check_keys: List of keys that must be present in the output dict (if output is dict)
        agent_name: Name of the agent for error messages
    
    Returns:
        bool: True if output is valid, False otherwise
    """
    if output is None:
        print(f"✗ {agent_name} returned None")
        return False
    
    if not isinstance(output, dict):
        print(f"✗ {agent_name} output is not a dictionary")
        return False
    
    if check_keys:
        missing_keys = [key for key in check_keys if key not in output]
        if missing_keys:
            print(f"✗ {agent_name} output missing keys: {missing_keys}")
            return False
        
        # Check if required keys have non-None values
        for key in check_keys:
            if output[key] is None:
                print(f"✗ {agent_name} output key '{key}' is None")
                return False
    
    return True


def detect_optimization_mode(user_request):
    """
    Detects whether the user request is for aerodynamic or aerostructural optimization.
    
    Args:
        user_request: The user's input request string
    
    Returns:
        str: "aerostructural" or "aerodynamic"
    """
    aerostructural_keywords = [
        "aerostructural", "aero-structural", "aerostruct", "structural",
        "fuel burn", "fuelburn", "weight", "stress", "failure", 
        "structural mass", "structural weight"
    ]
    
    user_request_lower = user_request.lower()
    
    for keyword in aerostructural_keywords:
        if keyword in user_request_lower:
            return "aerostructural"
    
    return "aerodynamic"


def main():
    """Main execution function for OpenAeroStruct LLM Agent"""
    
    # Get user request from terminal input
    print("\n" + "="*80)
    print("OpenAeroStruct LLM Agent")
    print("="*80)
    print("\nPlease enter your wing design request.")
    print("Example (Aerodynamic): 'Minimize drag for a wing with S = 100 m2 and b = 10 m.'")
    print("                       'Optimize for Taper, Twist, and Sweep. CL = 2.0'")
    print("Example (Aerostructural): 'Minimize fuel burn for aerostructural optimization.'")
    print("                          'Wing with structural constraints. Optimize twist and thickness.'\n")
    User_Request = input("Your request: ").strip()
    
    if not User_Request:
        print("Error: Empty request provided. Exiting.")
        sys.exit(1)
    
    # Detect optimization mode
    optimization_mode = detect_optimization_mode(User_Request)
    print(f"\n✓ Detected optimization mode: {optimization_mode.upper()}")
    
    # Select appropriate agents based on mode
    if optimization_mode == "aerostructural":
        ReformulatorAgentClass = ReformulatorAgentAerostructural
        BaseMeshAgentClass = BaseMeshAgentAerostructural
        GeometryAgentClass = GeometryAgentAerostructural
        OptimizerAgentClass = OptimizerAgentAerostructural
        ResultsReaderAgentClass = ResultsReaderAgentAerostructural
        ReportWriterClass = ReportWriterAerostructural
        FinalCoderAgentClass = FinalCoderAgentAerostructural
        template_file = "RunOAS_template_aerostructural.py"
    else:
        ReformulatorAgentClass = ReformulatorAgent
        BaseMeshAgentClass = BaseMeshAgent
        GeometryAgentClass = GeometryAgent
        OptimizerAgentClass = OptimizerAgent
        ResultsReaderAgentClass = ResultsReaderAgent
        ReportWriterClass = ReportWriter
        FinalCoderAgentClass = FinalCoderAgent
        template_file = "RunOAS_template.py"

    print("="*80)
    print("OpenAeroStruct LLM Agent - Starting Execution")
    print("="*80)
    print(f"User Request: {User_Request}")
    print(f"Mode: {optimization_mode.upper()}")
    print("="*80)
    
    # Step 1: Reformulator Agent
    print("\n[1/6] Running Reformulator Agent...")
    error_gen_flag = False
    while not error_gen_flag:
        reformulator = ReformulatorAgentClass()
        reformulator_output = reformulator.execute_task(User_Request)

        # SAFE CHECKS
        if safe_check_output(reformulator_output, 
                           check_keys=["objective_function", "design_variables", "trim_condition"],
                           agent_name="Reformulator"):
            error_gen_flag = True
            print("✓ Reformulator output generated successfully")
        else:
            print(f"Output: {reformulator_output}")

    print("Reformulator Output:")
    print(reformulator_output)

    # Step 2: Base Mesh Agent
    print("\n[2/6] Running Base Mesh Agent...")
    error_gen_flag = False
    while not error_gen_flag:
        MeshPrompt = f"""For this wing design, the geometric parameters are as follows: {reformulator_output["geometric_constraint"]}, and the type of the wing mesh should be: {reformulator_output["baseline_wing_mesh"]}"""
        mesher = BaseMeshAgentClass()
        mesher_output = mesher.execute_task(MeshPrompt)

        # SAFE CHECKS
        if safe_check_output(mesher_output, 
                           check_keys=["python_code"],
                           agent_name="Mesh"):
            error_gen_flag = True
            print("✓ Mesh code generated successfully")
        else:
            print(f"Output: {mesher_output}")

    print("Mesh Output:")
    print(mesher_output["python_code"])

    # Step 3: Geometry Agent
    print("\n[3/6] Running Geometry Agent...")
    error_gen_flag = False
    while not error_gen_flag:
        GeometryPrompt = f"""For this wing design, we are allowed to change the following parameters: {reformulator_output["design_variables"]}"""
        geometry_setup = GeometryAgentClass()
        geometry_output = geometry_setup.execute_task(GeometryPrompt)

        # SAFE CHECKS
        if safe_check_output(geometry_output, 
                           check_keys=["python_code"],
                           agent_name="Geometry"):
            error_gen_flag = True
            print("✓ Geometry code generated successfully")
        else:
            print(f"Output: {geometry_output}")

    print("Geometry Output:")
    print(geometry_output["python_code"])

    # Step 4: Optimizer Agent
    print("\n[4/6] Running Optimizer Agent...")
    error_gen_flag = False
    while not error_gen_flag:
        OptimizerPrompt = f"""For this wing design, the optimization parameters are as follows: {reformulator_output["design_variables"]}, the objective function is {reformulator_output["objective_function"]}, the geometric constraints are {reformulator_output["geometric_constraint"]}, the flight condition is {reformulator_output["trim_condition"]}, and the optimization algorithm is {reformulator_output["optimization_algorithm"]}"""
        optimizer_setup = OptimizerAgentClass()
        optimizer_output = optimizer_setup.execute_task(OptimizerPrompt)

        # SAFE CHECKS
        if safe_check_output(optimizer_output, 
                           check_keys=["python_code"],
                           agent_name="Optimizer"):
            error_gen_flag = True
            print("✓ Optimizer code generated successfully")
        else:
            print(f"Output: {optimizer_output}")

    print("Optimizer Output:")
    print(optimizer_output["python_code"])
    
    # Step 4.5: Final Coder Agent - Tidy up and check combined scripts
    print("\n[4.5/7] Running Final Coder Agent to tidy up scripts...")
    error_gen_flag = False
    while not error_gen_flag:
        FinalCoderPrompt = f"""Review and tidy up the following generated code sections:

Mesh Code:
{mesher_output["python_code"]}

Geometry Code:
{geometry_output["python_code"]}

Optimizer Code:
{optimizer_output["python_code"]}

Task: Check these code sections for any errors, inconsistencies, or formatting issues. Ensure proper indentation, remove any duplicate code, and verify that variable names are consistent across sections. Return the tidied versions of each section."""
        
        final_coder = FinalCoderAgentClass()
        final_coder_output = final_coder.execute_task(FinalCoderPrompt)

        # SAFE CHECKS
        if safe_check_output(final_coder_output, 
                           check_keys=["mesh_code", "geometry_code", "optimizer_code"],
                           agent_name="FinalCoder"):
            error_gen_flag = True
            print("✓ Final code review completed successfully")
            # Update outputs with tidied versions
            mesher_output["python_code"] = final_coder_output["mesh_code"]
            geometry_output["python_code"] = final_coder_output["geometry_code"]
            optimizer_output["python_code"] = final_coder_output["optimizer_code"]
        else:
            print(f"Output: {final_coder_output}")

    # Step 5: Write and Execute RunOAS.py
    print("\n[5/7] Creating and checking RunOAS.py...")
    # template_file is already set based on optimization mode

    def dedent_code(code_str):
        """Fully dedent code using Python standard library."""
        return textwrap.dedent(code_str).strip()
    
    # Check if template file exists
    if not os.path.exists(template_file):
        print(f"✗ ERROR: Template file '{template_file}' not found!")
        sys.exit(1)

    with open(template_file, "r", encoding="utf-8") as file:
        template_code = file.read()
        
        # Replace all placeholders with dedented generated code
        template_code = template_code.replace(
            '"""Part 1: PUT THE BASELINE MESH OF THE WING HERE"""',
            f'"""Part 1: PUT THE BASELINE MESH OF THE WING HERE"""\n{textwrap.dedent(mesher_output["python_code"]).strip()}'
        )
        template_code = template_code.replace(
            '"""Part 2:  DO THE GEOMETRY SETUP HERE"""',
            f'"""Part 2:  DO THE GEOMETRY SETUP HERE"""\n{textwrap.dedent(geometry_output["python_code"]).strip()}'
        )
        template_code = template_code.replace(
            '"""Part 3: PUT THE OPTIMIZER HERE """',
            f'"""Part 3: PUT THE OPTIMIZER HERE """\n{textwrap.dedent(optimizer_output["python_code"]).strip()}'
        )

    # UTF8 encoding to ensure compatibility
    run_oas_file = "RunOAS.py"
    with open(run_oas_file, "w", encoding="utf-8") as file:
        file.write(template_code)

    print("✓ Created RunOAS.py file with all generated code sections")

    # Step 5.5: Check code executability using Python's compile
    print("\n[5.5/7] Checking code executability...")
    try:
        with open(run_oas_file, "r", encoding="utf-8") as file:
            code_content = file.read()
        
        # Try to compile the code to check for syntax errors
        compile(code_content, run_oas_file, 'exec')
        print("✓ Code syntax check passed - no syntax errors found")
    except SyntaxError as e:
        print(f"\n✗ SYNTAX ERROR DETECTED in generated code:")
        print(f"   Line {e.lineno}: {e.msg}")
        print(f"   {e.text}")
        print(f"\n✗ STOPPING EXECUTION - Please fix the errors above")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR during code validation: {e}")
        print(f"\n✗ STOPPING EXECUTION - Please review the error above")
        sys.exit(1)

    # Now run the script using a subprocess, also capture all the outputs, and save it as a text file.
    print("\n[6/7] Running OpenAeroStruct optimization...")
    output_file = "openaerostruct_out/output.txt"
    os.makedirs("openaerostruct_out", exist_ok=True)
    
    with open(output_file, "w") as file:
        # Run the script and capture the output
        process = subprocess.Popen([sys.executable, run_oas_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Decode the output and write it to the file
        file.write(stdout.decode())
        file.write(stderr.decode())

    print(f"✓ Optimization complete. Output saved to {output_file}")

    # Generate optimization history PDF
    print("\n✓ Generating optimization history PDF...")
    HTML_Report = "./openaerostruct_out/RunOAS_out/reports/opt_report.html"
    pdf_output_path = "./figures/Opt_History.pdf"

    if not os.path.exists(HTML_Report):
        print(f"⚠ HTML missing: {HTML_Report}")
    else:
        os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)
        
        texbin = "/Library/TeX/texbin" if platform.system() == "Darwin" else ""
        env = os.environ.copy()
        if texbin:
            env['PATH'] = f"{texbin}:{env.get('PATH', '')}"
        
        result = subprocess.run([
            "pandoc", HTML_Report, "-o", pdf_output_path,
            "--pdf-engine=xelatex",
            "-V", "geometry:landscape,a4paper",
            "-V", "geometry:margin=5mm",
            "-V", "geometry:includeheadfoot",
            "--variable", "fontsize=5pt"
        ], env=env, capture_output=True, text=True, timeout=180, shell=False)
        
        if result.returncode == 0:
            print(f"✓ PDF generated: {pdf_output_path}")
        else:
            print(f"⚠ PDF generation failed with return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")

    # Plot wing
    print("\n✓ Plotting wing...")
    def run_plot_wing(file_path):
        """
        Execute the plot_wing command on a specified aero.db file
        
        Args:
            file_path (str): Path to the aero.db file
        """
        if not os.path.exists(file_path):
            print(f"⚠ Error: File '{file_path}' does not exist.")
            return
            
        try:
            # Run plot_wing command and capture output
            result = subprocess.run(["plot_wing", file_path], 
                                   capture_output=True, 
                                   text=True, 
                                   check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"⚠ Command failed with error code {e.returncode}")
            print(f"Error message: {e.stderr}")
        except FileNotFoundError:
            print("⚠ Error: 'plot_wing' command not found. Please ensure it's installed and in your PATH.")

    file_path = "./openaerostruct_out/RunOAS_out/aero.db"
    run_plot_wing(file_path)

    # Step 7: Results Reader Agent
    print("\n[7/7] Running Results Reader Agent...")
    error_gen_flag = False
    while not error_gen_flag:
        ResultsPrompt = f"""The initial problem by the user is: {User_Request}, the reformulated problem is: {reformulator_output}, and the optimization results are as follows: {optimizer_output}"""
        results_setup = ResultsReaderAgentClass()
        results_output = results_setup.execute_task(ResultsPrompt)

        # SAFE CHECKS
        if safe_check_output(results_output, 
                           check_keys=["Analysis"],
                           agent_name="Results"):
            error_gen_flag = True
            print("✓ Results analysis generated successfully")
        else:
            print(f"Output: {results_output}")

    print("Results Analysis:")
    print(results_output)

    # Report Writing
    print("\n✓ Generating final report...")
    error_gen_flag = False
    while not error_gen_flag:
        ReportPrompt  = f"""The initial problem by the user is: {User_Request}, the reformulated problem is: {reformulator_output}, the analysis by the LLM is {results_output}"""
        report_setup = ReportWriterClass()
        report_output = report_setup.execute_task(ReportPrompt)

        # SAFE CHECKS
        if safe_check_output(report_output, 
                           check_keys=["ReportText"],
                           agent_name="Report"):
            error_gen_flag = True
            print("✓ Report generated successfully")
        else:
            print(f"Output: {report_output}")

    # Generate filename with current date format YYMMDDHHMMSS (e.g., 260101211530_Report.tex)
    # Includes seconds to avoid collisions if script is run multiple times
    now = datetime.now()
    date_str = now.strftime("%y%m%d%H%M%S")
    report_file = f"./report_outputs/{date_str}_Report.tex"

    # Create outputs directory if it doesn't exist
    os.makedirs("./report_outputs", exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_output["ReportText"])

    print(f"✓ Report saved as: {report_file}")
    
    print("\n" + "="*80)
    print("OpenAeroStruct LLM Agent - Execution Complete!")
    print("="*80)
    print(f"✓ Final report: {report_file}")
    print(f"✓ Optimization history: {pdf_output_path}")
    print(f"✓ Output log: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
