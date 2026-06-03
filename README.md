**Copyright 2025-2026, The Regents of the University of Michigan, IDEAS Lab, MDO Lab**  
[https://ideas.engin.umich.edu](https://ideas.engin.umich.edu)

A multi-agent tool designed to code, optimize, and analyze wing designs from high-level natural language inputs using OpenAeroStruct.

## Contributors
- **Conan Lee**: Lead developer and primary author (HKUST) 
- **Gokcin Cinar**: Research supervision and concept development (U-M)
- **Joaquim R.R.A. Martins**: Research supervision and concept development (U-M)

## Introduction
The OpenAeroStruct LLM Agent leverages a coordinated team of Large Language Models (LLMs) to automate the entire aircraft wing design and optimization workflow. By providing a simple text prompt (e.g., *"Minimize drag for a wing with an area of 100m²"*), the agent pipeline handles mesh generation, geometry definition, optimization setup in OpenMDAO, results interpretation, and visualization.

## Architecture & Workflow

The system utilizes a specialized agent pipeline and a blueprint-based approach:
1.  **Blueprint Selection**: The agent identifies the most relevant baseline script (e.g., `aero_crm.py`, `aerostruct_wingbox.py`) based on the logic defined in `src/blueprints/skills.md`.
2.  **Code Synthesis**: Validated blueprints are modified to match the user's specific geometric and optimization requirements.
3.  **Execution**: The generated script is executed, saving logs and data.
4.  **Plotting & Analysis**: Results are saved to a unified output directory. The agent can automatically generate and display plots like lift distributions and 3D wing geometries.

### Unified Output Structure
All agent-generated data is now consolidated under:
- `openaerostruct_out/`: Contains SQLite databases (`aero.db`, `aerostruct.db`) and optimization logs.
- `openaerostruct_out/agent_plots/`: All generated figures, polars, and lift distributions.

## Features
- **Natural Language Intent**: Maps high-level goals ("reduce weight", "analyze stability") to complex optimization scripts.
- **LLM-Friendly Blueprints**: All templates in `src/blueprints/` include detailed comments to guide the agent in parameter adjustment.
- **Unified Plotting**: Automatic visualization of wing geometry and aerodynamic polars.
- **Support for Multi-Section Wings**: Specialized blueprints for complex planforms with multiple chord/twist segments.
- **Stability Analysis**: Direct support for calculating CL_alpha, CM_alpha, and Static Margin.

## Installation
### 1. Environment Setup
We recommend using `uv` for fast dependency management:

```bash
uv sync --python-preference only-managed
```

### 2. API Configuration
Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_KEY"
```

### 3. System Requirements
- **Python 3.10+**: Python 3.13+ recommended for Mac users.
- **OpenAeroStruct**: Installed via dependencies.
- **Matplotlib/Tkinter**: Required for GUI plotting (though the agent defaults to headless `.png` generation).

## Usage

### Running the Agent

```bash
uv run streamlit run src/app.py
```

### Example Prompts
- *"Optimize a rectangular wing to minimize drag at Mach 0.85, keeping CL constant at 0.5."*
- *"Perform an aerostructural optimization for a rectangular wing with a tubular spar. Minimize fuel burn for a 2000km range."*

## Output
After a run, you can find your results in:
- `openaerostruct_out/aero.db`: Optimization data.
- `openaerostruct_out/agent_plots/`: Generated visualizations (e.g., `wing_3d.png`, `lift_dist.png`).

The agent will also display relevant plots directly in the conversation interface.

## Development & Testing

### Benchmarking Suite
To ensure the agent is performing correctly across all blueprints, you can run the automated benchmark test suite:

```bash
uv run python src/tools/benchmark.py
```

This will:
1. Load test queries from `src/blueprints/test_queries.csv`.
2. Test intent routing, code generation, and execution for each case.
3. Save detailed results and error logs to `openaerostruct_out/benchmark_results.csv`.


## Examples 
Check the `Example_Outputs` folder for example outputs used in the preprint paper. The preprint can be accessed at https://www.gokcincinar.com/publication/pp-2025-agenticframework/pp-2025-AgenticFramework.pdf
