**Copyright 2025-2026, The Regents of the University of Michigan, IDEAS Lab, MDO Lab**  
[https://ideas.engin.umich.edu](https://ideas.engin.umich.edu)

A multi-agent tool designed to code, optimize, and analyze wing designs from high-level natural language inputs using OpenAeroStruct.

## Contributors
- **Conan Lee**: Lead developer and primary author (HKUST)
- **Gokcin Cinar**: Research supervision and concept development (U-M)
- **Joaquim R.R.A. Martins**: Research supervision and concept development (U-M)

## Introduction
OpenAeroStruct LLM Agent is a blueprint-driven workflow for aircraft wing analysis and optimization. A user provides a natural-language request such as *"Minimize drag on a CRM wing by changing the twist at Mach 0.78 with CL fixed at 0.45"* and the agent handles intent routing, code generation, execution, plotting, and error catching.

Compared with the previous version, this version is built around stricter routing, safer code execution, refined prompts, and more reliable retry behavior for failed or non-converged runs.

## Architecture & Workflow

The system uses a specialized agent pipeline and a blueprint-based approach:
1. **Intent Routing**: The router selects the best matching OpenAeroStruct blueprint and checks whether the request is specific enough to run.
2. **Clarification Handling**: If key information is missing, the app asks for the exact parameters still needed before code generation begins.
3. **Code Synthesis**: The coder makes targeted edits to a validated blueprint in `src/blueprints/` rather than generating a script from scratch.
4. **Safety Validation**: Generated code is screened for unsafe imports, subprocess usage, destructive file operations, and other blocked patterns before execution.
5. **Execution & Retry**: The script is run, checked for Python errors and optimization convergence, and retried with feedback when appropriate.
6. **Result Summarization**: If an OpenMDAO database is produced, the agent extracts design variables, objectives, and constraints into a readable summary.
7. **Plotting & Output**: Generated plots and run artifacts are saved to a unified output directory and shown in the Streamlit interface for users.

## Supported Workflows
- **Aerodynamic analysis** for fixed wings at one or more flight conditions.
- **Aerodynamic optimization** for drag, L/D, and related objectives.
- **Structural optimization** under applied loads only.
- **Aerostructural tube-spar optimization** for coupled aero-structural problems.
- **Aerostructural wingbox optimization** a higher fidelity aerostructural model with skin, spar, thickness-to-chord, and fuel-volume style constraints.
- **Multipoint optimizations** across two or more operating points.

## Features
- **Natural Language Intent Mapping**: Translates high-level requests into the appropriate OpenAeroStruct workflow.
- **Blueprint-Based Code Generation**: Uses curated templates instead of free-form script generation.
- **Clarification-Aware Routing**: Detects vague prompts and requests the missing design variables, constraints, or flight conditions.
- **Safety Guards**: Blocks unsafe generated code before execution.
- **Automatic Retry Feedback**: Feeds execution errors back into the code generator to improve subsequent attempts.
- **Unified Plotting and Summaries**: Automatically saves plots and extracts optimization summaries from OpenMDAO databases.

## Installation
### 1. Environment Setup
We recommend using `uv` for dependency management:

```bash
uv sync --python-preference only-managed
```

### 2. API Configuration
Create a `.env` file in the project root.

For Gemini:

```bash
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_KEY"
```

The code also accepts:

```bash
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_KEY"
```

### 3. System Requirements
- **Python 3.12**: The project currently pins `requires-python = "==3.12.*"`.
- **OpenAeroStruct / OpenMDAO**: Installed through project dependencies.
- **Matplotlib**: Used for plot generation and saved figures.

## Usage

### Running the Agent

```bash
uv run streamlit run src/app.py
```

### Example Prompts
- *"Analyze a tapered wing at Mach 0.55 and altitude 6000 m. Plot CL vs alpha and drag polar."*
- *"Minimize drag on a CRM wing at Mach 0.78, altitude 11000 m. DVs: alpha, twist_cp, chord_cp. Constraint: CL = 0.45."*
- *"Minimize fuel burn for a rectangular wing with a tube spar. Cruise at Mach 0.45, range 2000 km, constrain failure <= 0 and L = W."*

## Benchmarking
To evaluate routing and execution performance across the built-in test set:

```bash
uv run python src/benchmark.py
```

By default, the benchmark runner:
1. Loads test queries from `src/tools/test_queries.csv`.
2. Routes each query to a blueprint.
3. Runs multiple repetitions per case.
4. Stores generated code, logs, and copied artifacts for each attempt.
5. Writes aggregate benchmark summaries to a timestamped run directory.

Benchmark outputs are saved under:
- `benchmark_run_out/run_YYYYMMDD_HHMMSS/rep_results.csv`
- `benchmark_run_out/run_YYYYMMDD_HHMMSS/benchmark_results.csv`
- `benchmark_run_out/run_YYYYMMDD_HHMMSS/run_metadata.json`

## Project Structure
- `src/app.py`: Streamlit user interface.
- `src/agent_logic.py`: Main orchestration, retries, safety checks, and result handling.
- `src/llm/router.py`: Intent routing and vagueness detection.
- `src/llm/coder.py`: Blueprint adaptation and code generation.
- `src/llm/relaxer.py`: Suggests relaxation strategies for non-converged optimization runs.
- `src/tools/executor.py`: Static validation and execution of generated scripts.
- `src/tools/db_reader.py`: Reads OpenMDAO databases and produces optimization summaries.
- `src/blueprints/`: Curated OpenAeroStruct templates used by the agent.

## Examples
The related preprint can be accessed at [https://www.gokcincinar.com/publication/pp-2025-agenticframework/pp-2025-AgenticFramework.pdf](https://www.gokcincinar.com/publication/pp-2025-agenticframework/pp-2025-AgenticFramework.pdf).
