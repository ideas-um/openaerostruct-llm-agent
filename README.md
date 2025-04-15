# LLM_OpenAeroStruct

#### Runs OpenAeroStruct Automatically based on text input, does all the plotting, optimizing, and meshing with LLMs.

## Introduction
LLM_OpenAeroStruct is a tool that leverages Large Language Models (LLMs) to automate aircraft wing design and analysis using the OpenAeroStruct framework. It allows users to input design specifications in natural language, and the system automatically handles meshing, analysis, optimization, visualization, and reporting.

## Features
- Natural language processing for wing design specifications
- Automated mesh generation and refinement
- Wing optimization based on specified objectives (drag, weight, etc.)
- Automated visualization of results
- Detailed report output

## Installation
```bash
# Install dependencies, use uv to build the venv
uv init
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml #Install all required dependencies
```
The user will also need to setup a .env file that contains the line
GEMINI_API_KEY = YOUR_API_KEY_HERE

## Requirements
- Will be added in the future

## Usage
- Go into OpenAeroStruct.ipynb, and change the user request.
- Paths will need to be changed by the user for now.

## Examples
Check the `Example_Outputs` folder for some example outputs.

## License
- Will be added in the future