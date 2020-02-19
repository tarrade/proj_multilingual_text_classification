# Specify the base image and tag
#FROM gcr.io/deeplearning-platform-release/pytorch-cpu:m34
FROM gcr.io/deeplearning-platform-release/pytorch-cpu:m40

# Checking the type of machine
RUN cat /etc/os-release
RUN lsb_release -a
RUN uname -r

# Install htop
RUN apt-get update
RUN apt-get install htop

# Checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# active in the shell
SHELL ["/bin/bash", "-c"]
# active in the terminla later
ENV SHELL /bin/bash
RUN echo $0

## Creating conda env
# Adding the environment files in the docker image
ADD base.yml base.yml

# Update conda manager
RUN conda update -n base conda -y

# Define a folder to store env
RUN mkdir -p /root/.conda-env/
RUN mkdir -p /home/.conda-env/
RUN mkdir -p /home/.conda-pkg/
RUN mkdir -p /home/.jupyter_autosave/

# Create config files
RUN conda config --add channels conda-forge
RUN conda config --add envs_dirs /root/.conda-env/

# Update conda envs
RUN conda env update --file base.yml

# Create env if needed
#RUN conda env create -f environment.yml

# Checking configuration and new created envs
RUN conda config --show
RUN conda config --show-sources
RUN conda config --show envs_dirs

RUN conda info
RUN conda info -e
RUN conda env list
RUN conda list

RUN ls -la /root/.conda-env/

# checking display conda env in notebook
RUN conda list nb_conda_kernels
RUN conda list notebook
RUN conda list ipykernel

RUN jupyter kernelspec list
RUN jupyter --paths

## Clean all downloaded packages
RUN conda clean -a -y

## Check for update
RUN jupyter labextension update --all
### Check which extension exist
RUN jupyter labextension list

## Update to fix issue with extension of jupyter lab
RUN jupyter labextension install @jupyterlab/github
RUN jupyter labextension install @jupyterlab/latex
RUN jupyter labextension install @mflevine/jupyterlab_html
#RUN jupyter labextension install jupyterlab-drawio !! issue !!
#RUN jupyter labextension install @jupyterlab/plotly-extension !! issue !!
#RUN jupyter labextension install jupyterlab_bokeh !! issue !!
RUN jupyter labextension install @jupyterlab/toc
RUN jupyter labextension install @aquirdturtle/collapsible_headings
RUN jupyter labextension install jupyterlab-jupytext
RUN jupyter labextension install jupyterlab-cpustatus
RUN jupyter labextension install jupyterlab-python-file
#RUN jupyter labextension install jupyterlab_toastify jupyterlab_conda !! issue !!
RUN jupyter labextension install @ijmbarr/jupyterlab_spellchecker
RUN jupyter labextension install @lckr/jupyterlab_variableinspector
RUN jupyter labextension install nbdime-jupyterlab
RUN jupyter labextension update @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension update @jupyterlab/celltags
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter
#RUN jupyter labextension install @jupyterlab/jupyterlab-monaco  !! issue !!
#RUN jupyter labextension install jupyterlab-flake8 !! issue !!
RUN jupyter serverextension enable --py jupyterlab_code_formatter

RUN jupyter lab build

RUN jupyter labextension list

# Checking more info
RUN id
RUN whoami
RUN cat /etc/passwd
RUN cat /etc/group

# Running some checks
RUN echo $HOME

# set the right permission
RUN chmod +x /entrypoint.sh

# Activate conda
RUN conda init bash && . /root/.bashrc && conda info -e # && conda activate base

## !!!!
## !!!! NO pip, conda, git command below because of proxy config !!!!
## !!!!

## Configuration related to the entrerpise proxy server
# Setup the proxy server for conda
ADD .condarc /.condarc
ENV CONDARC /.condarc

# drop local conda-env canal
RUN conda config --remove envs_dirs /root/.conda-env/

# Setup the proxy server for pip
ADD pip.conf /pip.conf
ENV PIP_CONFIG_FILE /pip.conf

# Setup the proxy server for git
ADD .gitconfig /root/.gitconfig
# Check that the config file was created properly
RUN git config --list

# Adding the config file to read entrerise Github
#RUN mkdir -p /root/.ssh/
#ADD known_hosts /root/.ssh/known_hosts
#ADD id_rsa /root/.ssh/id_rsa
#ADD config /root/.ssh/config

# Copy env if they exist from /root to /home
RUN sed -i '1 a cp -r /root/.conda-env/env* /home/.conda-env/.' /entrypoint.sh
RUN cat  /entrypoint.sh

# Running some checks about kernels
RUN sed -i '1 a if [ -e /home/config.sh ]\nthen\n    source /home/config.sh\nfi\njupyter serverextension list\nconda info\nconda list nb_conda_kernels\nconda env list\nprintenv' /run_jupyter.sh
RUN cat  /run_jupyter.sh

# Adding option in jupyter notebook: display conda env, display hidden files
RUN sed -i -e "\$ac.NotebookApp.kernel_spec_manager_class = 'nb_conda_kernels.manager.CondaKernelSpecManager'" /root/.jupyter/jupyter_notebook_config.py
RUN sed -i -e "\$ac.ContentsManager.allow_hidden = True" /root/.jupyter/jupyter_notebook_config.py
RUN sed -i -e "\$ac.NotebookApp.notebook_dir = '/home'" /root/.jupyter/jupyter_notebook_config.py
RUN sed -i -e "\$ac.FileCheckpoints.checkpoint_dir = '/home/.jupyter_autosave'" /root/.jupyter/jupyter_notebook_config.py
RUN cat /root/.jupyter/jupyter_notebook_config.py

# Add execution of custom scripts it is exists
RUN sed -i -e "\$aif [ -e /home/config.sh ]\nthen\n    source /home/config.sh\nfi\n" /etc/skel/.profile
RUN cat /etc/skel/.profile

# Add execution of custom scripts it is exists
RUN sed -i -e "\$aif [ -e /home/config.sh ]\nthen\n    source /home/config.sh\nfi\n" /root/.bashrc
RUN cat /root/.bashrc

# Define entry points
ENTRYPOINT ["/entrypoint.sh", "/run_jupyter.sh"]

# Don't use it
#WORKDIR /home/
