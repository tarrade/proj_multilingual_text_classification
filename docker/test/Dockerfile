FROM gcr.io/deeplearning-platform-release/base-cu101
ARG PYTHONMODULE=model.tf_bert_classification.task
#model.test_log.task
#model.test.task
#model.tf_bert_classification.task

# checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# checking the python version
RUN python --version
# check env variables
RUN printenv
# active in the shell
SHELL ["/bin/bash", "-c"]

# copy the entrypoint script
ADD entrypoint.sh /entrypoint.sh
# set the right permission
RUN chmod +x /entrypoint.sh

# adding the environment files in the docker image
ADD environment.yaml environment.yaml
# update conda manager
RUN conda update -n base conda -y
# create the conda env
RUN conda env create -f environment.yaml

# copy the python code structure
COPY src /src
# add the code in the python path
ENV PYTHONPATH ${PYTHONPATH}:/src

# activate conda
RUN conda init bash
RUN conda info -e
SHELL ["conda", "run", "-n", "env_multilingual_class", "/bin/bash", "-c"]

# check the conda env
RUN conda info -e
RUN conda list -n env_multilingual_class

## clean all downloaded packages
#RUN conda clean -a -y

# option
#ENTRYPOINT ["conda", "run", "-n", "env_multilingual_class", "python", "-m", "model.tf_bert_classification.task"]

# activate conda env
#ENV PATH /opt/conda/envs/env_multilingual_class/bin:$PATH
#ENTRYPOINT ["python", "-m", "model.tf_bert_classification.task"]

ENV PYTHONMODULE ${PYTHONMODULE}
RUN echo $PYTHONMODULE

# activate conda env
ENV PATH /opt/conda/envs/env_multilingual_class/bin:$PATH
#ENTRYPOINT ["python", "-m", "$PYTHONMODULE"]
#ENTRYPOINT "python" "-m" "$PYTHONMODULE" # not working
#ENTRYPOINT ["python", "-m", "model.tf_bert_classification.task"] # working
#ENTRYPOINT ["conda", "run", "-n", "env_multilingual_class", "python", "-m", "model.test.task"]
#ENTRYPOINT ["conda", "run", "-n", "env_multilingual_class", "python", "-m", "model.tf_bert_classification.task"]
#ENTRYPOINT ["/entrypoint.sh", "python", "-m", "model.tf_bert_classification.task"]
#ENTRYPOINT ["/entrypoint.sh", "python", "-m", "model.test.task"]
#ENTRYPOINT ["/entrypoint.sh"]

# Create a script to pass command line args to python
RUN echo "python -m $PYTHONMODULE \$@" > /run_module.sh

ENTRYPOINT ["/bin/bash", "/run_module.sh"]