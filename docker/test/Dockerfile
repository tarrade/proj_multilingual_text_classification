FROM google/cloud-sdk:latest
RUN apt-get update -y && apt-get install --no-install-recommends -y -q ca-certificates python3-dev python3-setuptools python3-pip
RUN python3 -m pip install tensorflow==1.10 jupyter papermill
COPY run_notebook.sh ./
ENTRYPOINT ["bash", "./run_notebook.sh"]

FROM gcr.io/deeplearning-platform-release/tf2-cpu
COPY babyweight_tf2 /babyweight_tf2
RUN apt update && \
    apt install --yes python3-pip && \
    pip3 install --upgrade --quiet tf-nightly-2.0-preview

ENV PYTHONPATH ${PYTHONPATH}:/babyweight_tf2
CMD ["python3", "-m", "trainer.task"]