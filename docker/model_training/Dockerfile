# Specify the base image and tag
ARG IMAGE=base_container_cpu:latest
ARG PROJECT=None
ARG ENTRYPOINT=model.tf_bert_classification.task

FROM gcr.io/$PROJECT/$IMAGE

# trick to have the tag after the from
ARG IMAGE
RUN echo ${IMAGE}
ARG PROJECT
RUN echo ${PROJECT}
ARG ENTRYPOINT
RUN echo ${ENTRYPOINT}

# checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# checking the python version
RUN python --version
# check env variables
RUN printenv
# active in the shell
SHELL ["/bin/bash", "-c"]

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

# activate conda env
ENV PATH /opt/conda/envs/env_multilingual_class/bin:$PATH
ENTRYPOINT ["python", "-m", $ENTRYPOINT]