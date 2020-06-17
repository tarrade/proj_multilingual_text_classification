# Specify the base image and tag
ARG IMAGE=base-cu101:m49

FROM gcr.io/deeplearning-platform-release/$IMAGE

# trick to have the tag after the from
ARG IMAGE
RUN echo m${IMAGE}

# checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# checking the python version
RUN python --version
# check env variables
RUN printenv
# active in the shell
SHELL ["/bin/bash", "-c"]

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
RUN conda clean -a -y

# activate conda env
ENV PATH /opt/conda/envs/env_multilingual_class/bin:$PATH
#ENTRYPOINT ["python", "-m", "model.tf_bert_classification.task"]