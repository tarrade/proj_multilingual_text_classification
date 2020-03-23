## Create Jupyter Lab notebook locally
You need to copy base.yml from https://github.com/tarrade/proj_custom_ai_platform_notebook/blob/master/env/base.yml
in the ```env/``` folder.
- clone the git repository https://github.com/tarrade/proj_multilingual_text_classification.git
- change directory to 'proj_multilingual_text_classification'
- create the python conda env to run Jupyter Lab (or Jupyter Notebook) from the command line/terminal:   
  ```conda env create -f env/base.yml```    
  (you should have nodejs installed with conda)  

- activate the env:  
  ```conda activate jupyter-notebook```  
 
- install jupyter lab extensions    
  execute the shell script copy and paste each line
  ```. script/jupyter_lab/install_jupyterlab_extension.sh ```

- implement this trick to fix an issue with Black and temp folders that are created too late: 19.10b0

```
cd test 
black black.py
python -c "import logging; logging.basicConfig(level=logging.INFO); import black"
cd ..
```

This executes the code
```
"import pandas as pd\nprint(pd.__version__)"
```
in the black.py file.

- start Jupyter Lab from the terminal for which 'jupyter-notebook' env is activated  
  ```jupyter lab```  
  
  Note: If you get any errors regarding the dynamic link library, make sure that conda is updated to the newest version and try again.

## Note about conda env in Jupyter notebook  
   To be able to see conda env in Jupyter notebook, you need:  
   - the following package in you base env (already installed in 'jupyter-notebook'):  
   ```conda install nb_conda```  

   - the following package in each env (this is the responsibility of the creator of the env to be sure it is in the env)  
   ```conda install ipykernel```  