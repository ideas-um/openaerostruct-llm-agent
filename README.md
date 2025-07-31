# OpenAeroStruct LLM Agent
Copyright 2025, The Regents of the  University of Michigan, IDEAS Lab
https://ideas.engin.umich.edu

#### Runs OpenAeroStruct Automatically based on text input, does all the plotting, optimizing, and meshing with LLMs.

## Contributors
- Conan Lee: Lead developer and primary author (HKUST) 
- Gokcin Cinar: Research supervision and concept development (U-M)

## Introduction
OpenAeroStruct LLM Agent is a tool that leverages Large Language Models (LLMs) to automate aircraft wing design and analysis using the OpenAeroStruct framework. It allows users to input design specifications in natural language, and the system automatically handles meshing, analysis, optimization, visualization, and reporting It currently supports main aerodynamic objectives of lift and drag, and geometric design variables of taper, sweep, dihedral, chord, and twist.

## Features
- Natural language processing for wing design specifications
- Automated mesh generation and refinement
- Wing optimization based on specified objectives (drag, lift, etc.)
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
TOGETHER_AI_API_KEY = YOUR_API_KEY_HERE


Please also add the TeX Live and LaTeX Workshop extension on vscode so the latex report can be generated and updated.

For the report generation and saving the figure automatically, go into the OpenAeroStruct package and change the plot_wing code by replacing one function. The directory would be .venv/lib/openaerostruct/utils/plot_wing.py

```bash
def disp_plot(args=sys.argv):
    disp = Display(args)
    disp.draw_GUI()
    plt.tight_layout()
    disp.root.protocol("WM_DELETE_WINDOW", disp.quit)
    
    # Schedule image saving and program exit after a short delay
    # to ensure GUI is fully rendered before capture
    def save_and_quit():
        disp.save_image()
        disp.quit()
    
    # Wait 1000ms to ensure the GUI is fully rendered before saving
    disp.root.after(1000, save_and_quit)
    
    Tk.mainloop()
```

and the save_image function

```bash
def save_image(self):
        fname = "Figures/Optimized_Wing.pdf"
        plt.savefig(fname)
```


## Requirements
- Install Pandoc here: https://pandoc.org/installing.html
- If using an Apple device (Mac / Macbook), there might be issues with the Tkinter package for Matplotlib GUI for generating wing plots, to fix this issue, we suggest using the newest distribution of python which is 13.3.3
- We also suggest installing the Vscode extension LaTex Workshop such that the report can be generated into PDF format for viewing.
- We currently do not provide the document used for RAG for enhanced variable understanding for the LLM, it is suggested that the user uploads a document for wing design or equivalents and run the RAG_Embeddings.py file in order to create your own simple persisted embeddings.

## Usage
- Go into OpenAeroStruct.ipynb, and change the user request.

## Examples
Check the `Example_Outputs` folder for example outputs used in the published paper.
