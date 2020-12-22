# Specify the base image and tag
ARG IMAGE=base_container_cpu:latest
ARG PROJECT=None
ARG PYTHONMODULE=model.tf_bert_classification.task

#FROM gcr.io/$PROJECT/$IMAGE
FROM europe-west4-docker.pkg.dev/$PROJECT/$IMAGE

# trick to have the tag after the from
ARG IMAGE
RUN echo ${IMAGE}
ARG PROJECT
RUN echo ${PROJECT}
ARG PYTHONMODULE
RUN echo ${PYTHONMODULE}

# checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# checking the python version
RUN python --version
# check env variables
RUN printenv
# active in the shell
SHELL ["/bin/bash", "-c"]

# Install some linux utils
RUN apt-get update
RUN apt-get install htop
RUN apt-get -y install net-tools
RUN apt-get -y install iputils-ping
RUN apt-get -y install iptables
RUN add-apt-repository ppa:git-core/ppa
RUN apt-get -y install git

# activate conda
RUN conda init bash
RUN conda info -e
SHELL ["conda", "run", "-n", "env_multilingual_class", "/bin/bash", "-c"]

# check the conda env
RUN conda info -e

# copy the python code structure
COPY src /src
# add the code in the python path
ENV PYTHONPATH ${PYTHONPATH}:/src

RUN conda info -e
RUN conda list

ENV PYTHONMODULE ${PYTHONMODULE}
RUN echo $PYTHONMODULE

# activate conda env
ENV PATH /opt/conda/envs/env_multilingual_class/bin:$PATH
#ENTRYPOINT ["python", "-m", "$ENTRYPOINT"] # not working
#ENTRYPOINT ["python", "-m", "model.tf_bert_classification.task"] # hardcoded
#ENTRYPOINT "python" "-m" "$PYTHONMODULE" # not working

# Create a script to pass command line args to python
RUN echo "python -m $PYTHONMODULE \$@" > /run_module.sh
ENTRYPOINT ["/bin/bash", "/run_module.sh"]
