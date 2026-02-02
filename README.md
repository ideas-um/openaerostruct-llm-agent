# OpenAeroStruct LLM Agent
**Copyright 2025-2026, The Regents of the University of Michigan, IDEAS Lab, MDO Lab**  
[https://ideas.engin.umich.edu](https://ideas.engin.umich.edu)

A multi-agent tool designed to write, execute, and analyze OpenAeroStruct optimization code from high-level natural language inputs.

## Contributors
- **Conan Lee**: Lead developer and primary author (HKUST) 
- **Gokcin Cinar**: Research supervision and concept development (U-M)
- **Joaquim R.R.A. Martins**: Research supervision and concept development (U-M)

## Introduction
The OpenAeroStruct LLM Agent leverages a coordinated team of Large Language Models (LLMs) to automate the entire aircraft wing design and optimization workflow. By providing a simple text prompt (e.g., *"Minimize drag for a wing with an area of 100m²"*), the agent pipeline handles mesh generation, geometry definition, optimization setup in OpenMDAO, results interpretation, and report generation for further iteration.

## Multi-Agent Architecture
The system utilizes a specialized 6-agent pipeline:
1.  **ReformulatorAgent**: Translates raw user requests into structured optimization specifications (objectives, constraints, and variables).
2.  **BaseMeshAgent**: Writes the baseline wing mesh script required for initialization.
3.  **GeometryAgent**: Writes the `surface` dictionary script, managing optimization parameters like taper, sweep, and twist.
4.  **OptimizerAgent**: Writes the OpenMDAO optimization scripts, including design variables (`alpha`, `twist_cp`, etc.) and constraints (`CL`).
5.  **ResultsReaderAgent**: Parses both graphical and numiercal outputs, identifies optimization failures (e.g., infeasible design spaces), and provides engineering recommendations.
6.  **ReportWriter**: Compiles the entire process, including analysis and plot references, into a professionally formatted LaTeX report.

## Features
- Natural language processing for wing design specifications
- Automated mesh generation and refinement
- Wing optimization based on specified objectives (drag, lift, etc.)
- Automated visualization of results
- Detailed report output

## Installation
### 1. Environment Setup
We recommend using `uv` for fast dependency management:

```bash
uv sync
```

### 2. API Configuration
Create a .env file in the root directory to store your LLM credentials:

```bash
GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_KEY"
```

### 3. System Requirements

To generate the final reports and plots, the following are required:

Pandoc: Required for document conversion https://pandoc.org/installing.html

XeLaTeX: Required for PDF generation. Install via MiKTeX (Windows) or BasicTeX (macOS).

Python Version: For Mac users, Python 3.13.1+ is recommended to avoid Matplotlib/Tkinter GUI issues.

## Usage

### Python Script (Recommended)

Navigate to the openaerostruct-llm directory and run the script:

```bash
cd openaerostruct-llm
python3 run_openaerostruct.py
```

The script will prompt you to enter your wing design request interactively:

```
Aerodynamic Optimization: Minimize drag for a wing with S = 100 m2 and b = 10 m. Optimize for Taper, Twist, and Sweep. CL = 2.0

Aerostructural optimization: Minimize fuel burn for aerostructural optimization. Wing with structural constraints Optimize twist. b = 10m, S = 100m2.
```

The system will execute the complete pipeline:
- Generate RunOAS.py
- Validate the generated code for syntax errors
- Execute the optimization
- Create a report in openaerostruct-llm/report_outputs (e.g., 26010121_Report.tex)

**Note:** The Jupyter notebook (OpenAeroStruct.ipynb) is no longer the primary interface. Please use the Python script for all interactions.

## Examples

Check the `Example_Outputs` folder for example outputs used in the published paper.
