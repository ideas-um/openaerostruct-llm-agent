# OpenAeroStruct LLM Agent

A multi-agent framework for running OpenAeroStruct wing analysis and optimization from natural-language prompts.

[Open architecture diagram (PDF)](./.github/workflows/Architecture.pdf)

## Contributors

- **Conan Lee**: Lead developer and primary author (HKUST)
- **Gokcin Cinar**: Research supervision and concept development (U-M)
- **Joaquim R.R.A. Martins**: Research supervision and concept development (U-M)

## License

Copyright 2025-2026, The Regents of the University of Michigan, IDEAS Lab, MDO Lab  
[University of Michigan IDEAS Lab](https://ideas.engin.umich.edu)  
[LICENSE](./LICENSE)

## What It Does

The agent takes a natural-language aircraft design request, routes it to the most appropriate OpenAeroStruct blueprint, edits a validated bluepruint in `src/blueprints/`, runs the generated script, retries when execution fails, and saves both plots and optimization summaries for review. In other words, the user can describe the analysis or optimization problem in plain language, while the system handles workflow selection, code adaptation, execution, and result packaging in a more controlled way than writing a brand-new script from scratch each time.

Supported workflows:
- Aerodynamic analysis
- Aerodynamic optimization
- Structural optimization
- Aerostructural tube-spar optimization
- Aerostructural wingbox optimization
- Multipoint optimization

## Quick Start

### Requirements

- Python `3.12`
- Either `uv` or Conda
- One LLM provider:
  Gemini with `GEMINI_API_KEY` or `GOOGLE_API_KEY`, or Ollama running locally

### Install

Using `uv`:

```bash
uv sync --python-preference only-managed
cp .env.example .env
```

Using Conda:

```bash
conda create -n openaerostruct-agent python=3.12
conda activate openaerostruct-agent
pip install -e .
cp .env.example .env
```

### Choose a Provider and Model

The project reads provider credentials from a local `.env` file in the repository root. A starter template is provided in [.env.example](./.env.example). If you have not already created the file during installation, run:

```bash
cp .env.example .env
```

This creates a local copy that you can edit without changing the tracked example file.

Gemini:

```bash
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_KEY"
```

If you want to use Gemini, you first need to create a Gemini API key through Google AI Studio. Google’s official documentation for key creation and management is here: [Using Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key). The direct page for creating or viewing a key is here: [Google AI Studio API keys](https://aistudio.google.com/apikey). After creating a key, paste it into your local `.env` file as `GEMINI_API_KEY="..."`.

Ollama:
- Install Ollama from [ollama.com](https://ollama.com/).
- Start the Ollama app or daemon locally so the Python client can connect to your local Ollama server.
- Install any local model you want to use. The app queries Ollama for installed models, so anything available in your local Ollama instance will appear automatically in the UI model dropdown.

Installation Example:

```bash
ollama run gemini-2.0-flash
```

### Docker Sandbox (Optional)

If you want generated OpenAeroStruct scripts to run in Docker instead of the local Python subprocess, build the sandbox image once and then set the execution backend before launching the app or benchmark. This changes where the generated OpenAeroStruct scripts run, but it does not change which LLM provider or model you use.

```bash
docker build -f docker/sandbox.Dockerfile -t openaerostruct-sandbox:latest .
```

Then set:

```bash
export OAS_EXECUTION_BACKEND=docker
```

Backend options:
- `OAS_EXECUTION_BACKEND=host`
- `OAS_EXECUTION_BACKEND=docker`
- `OAS_EXECUTION_BACKEND=auto`


## Run the App

Using `uv`:

```bash
uv run streamlit run src/app.py
```

Using Conda:

```bash
conda activate openaerostruct-agent
streamlit run src/app.py
```

When the app starts, provider and model selection happens in the Streamlit sidebar rather than in `.env`:
- `Provider`: `Gemini API` or `Ollama`
- `Model`: selected from the sidebar after choosing the provider

For Gemini, the sidebar offers the Gemini models currently wired into the UI. For Ollama, the app reads your installed local models and shows them in the dropdown, which means you can switch models without editing project code as long as the model is already installed locally.

## Example Prompts

- `Analyze a tapered wing at Mach 0.55 and altitude 6000 m. Plot CL vs alpha and drag polar.`
- `Minimize drag on a CRM wing at Mach 0.78, altitude 11000 m. DVs: alpha, twist_cp, chord_cp. Constraint: CL = 0.45.`
- `Minimize fuel burn for a rectangular wing with a tube spar. Cruise at Mach 0.45, range 2000 km, constrain failure <= 0 and L = W.`

## Benchmarking

The benchmark does not use the Streamlit interface. Instead, you choose the provider and model directly from the command line, which makes benchmark runs easier to reproduce and compare across different settings.

Gemini example:

```bash
uv run python src/benchmark.py --max-retries 5 --provider "Gemini API" --model "gemini-flash-lite-latest"
```

Ollama example:

```bash
uv run python src/benchmark.py --max-retries 5 --provider "Ollama" --model "gemini-2.0-flash"
```

In these commands, `--provider` selects the LLM backend, `--model` selects the exact model name passed to that backend, and `--max-retries 5` sets the maximum number of coder retry attempts used when the benchmark tries to recover from execution errors or failed runs.

Outputs are written under `benchmark_run_out/`.

## Project Structure

- `src/app.py`: Streamlit UI
- `src/agent_logic.py`: orchestration, retries, safety checks, result handling
- `src/llm/router.py`: intent routing and vagueness detection
- `src/llm/coder.py`: blueprint adaptation and code generation
- `src/llm/relaxer.py`: non-convergence relaxation suggestions
- `src/tools/executor.py`: validation and execution
- `src/tools/db_reader.py`: OpenMDAO database summaries
- `src/blueprints/`: curated OpenAeroStruct templates

## Reference

Related preprint: [Agentic Framework PDF](https://www.gokcincinar.com/publication/pp-2025-agenticframework/pp-2025-AgenticFramework.pdf)
