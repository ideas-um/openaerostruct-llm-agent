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

Please also add the TeX Live and LaTeX Workshop extension on vscode so the latex report can be generated and updated.

For the report generation and saving the figure automatically, go into the OpenAeroStruct package and change the plot_wing code by replacing one function.
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

## Requirements
- Install Pandoc here: https://pandoc.org/installing.html
- If using a Mac, there might be issues with the Tkinter package for Matplotlib GUI for generating wing plots, to fix this issue, I used the newest distribution of python which iis 13.3.3

## Usage
- Go into OpenAeroStruct.ipynb, and change the user request.

## Examples
Check the `Example_Outputs` folder for some example outputs.

## License
- Will be added in the future