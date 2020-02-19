# Instruction to setup the project

## Create the python conda env  
This will provide you a unique list of python packages needed to run the code
- create a python env based on a list of packages from environment.yml  
  ```conda env create -f env/environment.yml```  
  
 - activate the env  
  ```conda activate env_nlp_text_class```  
  
 - in case of issue clean all the cache in conda  
   ```conda clean -a -y```  

## Update or delete the python conda env  
- update a python env based on a list of packages from environment.yml  
  ```conda env update -f env/environment.yml```  
  
- delete the env to recreate it when too many changes are done  
  ```conda env remove -n env_nlp_text_class```  

## Access of conda env in Jupyter notebook  
   To be able to see conda env in Jupyter notebook, you need:  
   - the following package in you base env:  
   ```conda install nb_conda```  

   - the following package in each env (this is the responsibility of the creator of the env to be sure it is in the env)  
   ```conda install ipykernel```  

## Use JupyterLab   
- create the python conda env to run Jupyter Lab (or Jupyter Notebook)   
  ```conda env create -f env/jupyter-notebook.yml```    
  (you should have nodejs installed with conda)  

- activate the env  
  ```conda activate jupyter-notebook```  
  
- install jupyter extension  
```jupyter labextension list
       
    npm set strict-ssl false   
       
    # To see issues:   
    jupyter labextension install @mflevine/jupyterlab_html --debug  
      
    jupyter labextension install @jupyterlab/github   
    jupyter labextension install @jupyterlab/latex  
    jupyter labextension install @mflevine/jupyterlab_html   
    jupyter labextension install jupyterlab-drawio   
    jupyter labextension install @jupyterlab/plotly-extension   
    jupyter labextension install jupyterlab_bokeh   
    jupyter labextension install @jupyterlab/toc   
    jupyter labextension install @aquirdturtle/collapsible_headings   
    jupyter labextension install jupyterlab-jupytext   
    jupyter labextension install jupyterlab-cpustatus   
    jupyter labextension install jupyterlab-python-file   
    jupyter labextension install jupyterlab_toastify jupyterlab_conda   
    jupyter labextension install @ijmbarr/jupyterlab_spellchecker   
    jupyter labextension install @lckr/jupyterlab_variableinspector   
    jupyter labextension install nbdime-jupyterlab
    jupyter labextension install @ryantam626/jupyterlab_code_formatter
    -- not working start --
    jupyter labextension install @jupyterlab/jupyterlab-monaco
    jupyter labextension install jupyterlab-flake8
    -- not working end --
    jupyter serverextension enable --py jupyterlab_code_formatter
    jupyter contrib nbextension install   
    jupyter nbextensions_configurator enable
   
    jupyter lab build

    # you need to create this folder structure manually  
    C:\Users\Cxxxxxx\AppData\Local\black\black\Cache\19.3b0
```

- start Jupyter Lab where is the env 'jupyter-notebook' activated  
  ```jupyter lab```  
