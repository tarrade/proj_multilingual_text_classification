# Specify the base image and tag
ARG IMAGE=base-cu101:m49

FROM gcr.io/deeplearning-platform-release/$IMAGE

# trick to have the tag after the from
ARG IMAGE
RUN echo ${IMAGE}

# checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# checking the python version
RUN python --version
# check env variables
RUN printenv
# active in the shell
SHELL ["/bin/bash", "-c"]

# Install some linux utils
RUN curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install htop
RUN apt-get -y install net-tools
RUN apt-get -y install iputils-ping
RUN apt-get -y install nano
RUN apt-get -y install iptables
RUN add-apt-repository ppa:git-core/ppa
RUN apt-get -y install git

# adding the environment files in the docker image
ADD environment.yaml environment.yaml
# update conda manager
RUN conda update -n base conda -y
# create the conda env
RUN conda env create -f environment.yaml

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
