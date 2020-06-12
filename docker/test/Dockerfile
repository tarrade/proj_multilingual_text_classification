FROM gcr.io/deeplearning-platform-release/base-cu101
RUN python --version
RUN pip freeze
RUN conda info -e

RUN pip install scikit-learn==0.22.1 grpcio==1.24.3 transformers==2.9.0 tensorflow==2.2.0 tensorboard==2.2.1 cloud-tpu-client==0.8 tensorboard_plugin_profile==2.2.0 absl-py==0.9.0 pip==20.1 google-cloud-bigquery==1.24.0 google-cloud-storage==1.26.0 google-cloud-logging==1.15.0 google-cloud-translate==2.0.1 google-resumable-media==0.5.0 cloudml-hypertune==0.1.0.dev6

COPY src /src
ENV PYTHONPATH ${PYTHONPATH}:/src

RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

#CMD ["python", "-m", "model.test.task"]
ENTRYPOINT ["python", "-m", "model.test.task"]

#FROM google/cloud-sdk:latest
#RUN apt-get update -y && apt-get install --no-install-recommends -y -q ca-certificates python3-dev python3-setuptools python3-pip
#RUN python3 -m pip install tensorflow==1.10 jupyter papermill
#COPY run_notebook.sh ./
#ENTRYPOINT ["bash", "./run_notebook.sh"]

#FROM gcr.io/deeplearning-platform-release/tf2-cpu
#COPY babyweight_tf2 /babyweight_tf2
#RUN apt update && \
#    apt install --yes python3-pip && \
#    pip3 install --upgrade --quiet tf-nightly-2.0-preview
#
#ENV PYTHONPATH ${PYTHONPATH}:/babyweight_tf2
#CMD ["python3", "-m", "trainer.task"]