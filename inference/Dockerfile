# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
# FROM pytorch/pytorch
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=algorithm:algorithm min_requirements.txt /opt/algorithm/requirements.txt
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm artifact/ /opt/algorithm/artifact/
COPY --chown=algorithm:algorithm algorithm /opt/algorithm/

ENV DATA_PATH="/input/images/ct/"
ENV LOGS_PATH="/output/"

ENTRYPOINT python -m submission $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=STOICAlgorithm



