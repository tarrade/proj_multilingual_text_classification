# Create a Jupyter Lab notebook on GCP

## option 1: How to created AI Platform notebook on GCP using the UI
The easiest option to go on the GCP console and go on AI Platform. Here in the ```Notebook instances``` section click on ```New instances```
and select ```PyTorch X.X``` and ```Without GCPU``` so  you have conda installed. Soon all images should be installed with Anaconda.

## option 2: How to created self custom AI Platform notebook on GCP 
The second option with require more installation is explain in this separate Github repository: https://github.com/tarrade/proj_custom_ai_platform_notebook.
Here you have you will create a custom image of the AI Platform notebook and deploy it on GCP using some automatic pipeline.

## Jupyter Lab
When you have your Jupyter Lab instance running locally you will need to install conda environment to have all python 
packages needed to run the code as describe in this [link](doc/conda_env.md). Then you will be ready to train your model.