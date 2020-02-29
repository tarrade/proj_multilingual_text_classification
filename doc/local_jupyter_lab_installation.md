## Use JupyterLab   
You need to copy base.yml from https://github.com/tarrade/proj_custom_ai_platform_notebook/blob/master/env/base.yml
in the ```env/``` folder.
- create the python conda env to run Jupyter Lab (or Jupyter Notebook)   
  ```conda env create -f env/base.yml```    
  (you should have nodejs installed with conda)  

- activate the env  
  ```conda activate jupyter-notebook```  
  
- install jupyter extension by copying the following lines  
check the latest instructions about jupyter lab extension from https://github.com/tarrade/proj_custom_ai_platform_notebook/blob/master/docker/derived-pytorch-cpu/Dockerfile
 
```
jupyter labextension list
jupyter lab clean
jupyter labextension update --all
jupyter lab build

## Important extension
# The JupyterLab cell tags extension enables users to easily add, view, and manipulate descriptive tags for notebook cells. Will be merged in core.
jupyter labextension install @jupyterlab/celltags --no-build --debug
# An extension for JupyterLab which allows for live-editing of LaTeX documents.
jupyter labextension install @jupyterlab/latex --no-build  --debug
# A Table of Contents extension for JupyterLab
jupyter labextension install @jupyterlab/toc --no-build --debug
# Display CPU usage in status bar
jupyter labextension install jupyterlab-cpustatus --no-build  --debug
# Create Python Files from JupyterLab
jupyter labextension install jupyterlab-python-file --no-build  --debug
# Jupyterlab extension that shows currently used variables and their values
jupyter labextension install @lckr/jupyterlab_variableinspector --no-build  --debug
# Make headings collapsible like the old Jupyter notebook extension and like Mathematica notebooks
jupyter labextension install @aquirdturtle/collapsible_headings --no-build  --debug
# This is a small Jupyterlab plugin to support using various code formatter on the server side and format code cells/files in Jupyterlab
jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build  --debug
# Provides Conda environment and package access extension from within Jupyter Notebook and JupyterLab
jupyter labextension install jupyterlab_toastify jupyterlab_conda --no-build --debug
# Jupyterlab extension to lint python notebooks and python files in the text editor. Uses flake8 python library for linting
jupyter labextension install jupyterlab-flake8 --no-build --debug
# Jupyter widgets extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build --debug

## Old extensions/not so important
# A JupyterLab extension for accessing GitHub repositories
#jupyter labextension install @jupyterlab/github --debug
# This extension adds a few Jupytext commands to the command palette. Use these to select the desired ipynb/text pairing for your notebook
#jupyter labextension install jupyterlab-jupytext --debug
# JupyterLab extension mimerenderer to render HTML files in IFrame Tab
#jupyter labextension install @mflevine/jupyterlab_html
# A spell checker extension for markdown cells in jupyterlab notebooks
#jupyter labextension install @ijmbarr/jupyterlab_spellchecker
# A JupyterLab extension for standalone integration of drawio / mxgraph into jupyterlab
#jupyter labextension install jupyterlab-drawio --debug
# Plotly support with Jupyter Lab (3 extensions)
# Jupyter widgets extension
#jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build --debug
# FigureWidget support
#jupyter labextension install plotlywidget@1.5.1 --no-build --debug
# and jupyterlab renderer support
#jupyter labextension install jupyterlab-plotly@1.5.1 --no-build --debug
# A Jupyter extension for rendering Bokeh content
#jupyter labextension install @bokeh/jupyter_bokeh --debug

# Building
jpyter lab build --debug

# Checking full list
jupyter labextension list

jupyter serverextension enable --py jupyterlab_code_formatter
```

- trick to fix an issue with Black and temp folders that are created too late: 19.10b0
```
python -c "import logging; logging.basicConfig(level=logging.INFO); import black"
touch /black.py
sed -i "import pandas as pd\nprint(pd.__version__)"  /black.py
black /black.py
python -c "import logging; logging.basicConfig(level=logging.INFO); import black"
```

- start Jupyter Lab from the terminal for which 'jupyter-notebook' env is activated  
  ```jupyter lab```  

## Note about conda env in Jupyter notebook  
   To be able to see conda env in Jupyter notebook, you need:  
   - the following package in you base env (already installed in 'jupyter-notebook'):  
   ```conda install nb_conda```  

   - the following package in each env (this is the responsibility of the creator of the env to be sure it is in the env)  
   ```conda install ipykernel```  