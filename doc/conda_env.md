# How to install and manage python packages needed to run the code

## Create the python conda env  
This will provide you a unique list of python packages needed to run the code
- create a python env based on a list of packages from environment.yml  
  ```conda env create -f env/environment.yaml```  
  
  the name of the env NAME_ENV is define in the first line of ```env/environment.yml``` 
  and in this is is ```env_multilingual_class```
  
 - activate the env  
  ```conda activate NAME_ENV```  
  
 - in case of issue clean all the cache in conda  
   ```conda clean -a -y```  

## Update the python conda env  
- update a python env based on a list of packages from environment.yml  
  ```conda env update -f env/environment.yaml```  
 
## Delete the python conda env    
- delete the env to recreate it when too many changes are done  
  ```conda env remove -n NAME_ENV```  
  
- clean all the cache in conda  
   ```conda clean -a -y```