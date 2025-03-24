# HDX-sFDR

Tools for analyzing hydrogen-deuterium exchange (HDX) data with protein structures using structure False Discovery Rate (sFDR) methodology.

# About
This repository contains a collection of Python modules for the analysis of HDX-MS data, integration with protein structures, and statistical analysis using structured False Discovery Rate approaches.

# Installation
Install directly from GitHub with all dependencies:

```bash
pip install git+https://github.com/ococrook/hdx-sFDR.git
```

Or clone the repository and install:

```bash
git clone https://github.com/ococrook/hdx-sFDR.git
cd hdx-sFDR
pip install -r requirements.txt
pip install -e .
```

# Usage in Google Colab
Add the following code to your Colab notebook to install and import the modules:


```python
# Install package with all dependencies

!pip install git+https://github.com/ococrook/hdx-sFDR.git

# Import modules
import hdx_utils
import hdx_plot
import hdx_structure_utils
import statistical_inference
import evalutions

# For refreshing modules after changes
import importlib
def reload_modules():
    importlib.reload(hdx_utils)
    importlib.reload(hdx_plot)
    importlib.reload(hdx_structure_utils)
    importlib.reload(statistical_inference)
    importlib.reload(evalutions)
    print("All modules reloaded successfully!")
```

# Required Data
This package requires two types of input files:

- A CSV file containing HDX-MS data containing named columns: 
 - "Start" - indicating beginning of protein
 - "End" - inidicating end of protein
 - "Exposure" - timepoints used (can be 1 timepoint)
 - "pvalue" - pvalues to be corrected/adjusted
- A protein structure file (CIF format)

See the example notebooks for detailed usage instructions.

# Modules

- hdx_utils: Core utilities for HDX data processing
- hdx_plot: Visualization tools for HDX data
- hdx_structure_utils: Functions for integrating HDX data with protein structures
- statistical_inference: Statistical analysis methods including sFDR
- evalutions: Evaluation and validation utilities

# Requirements
See requirements.txt for a complete list of dependencies.

# Citation
If you use this software in your research, please cite:
[Publication information to be added]

# License
MIT 

# Contact
You can find my emails in the contact form