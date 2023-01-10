# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
# FROM pytorch/pytorch
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN groupadd -r train && useradd -m --no-log-init -r -g train train

RUN mkdir -p /opt/train /input /output \
    && chown train:train /opt/train /input /output

USER train

WORKDIR /opt/train

ENV PATH="/home/train/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=train:train min_requirements.txt /opt/train/requirements.txt
RUN python -m pip install --user -r requirements.txt

# copy weights from pretraining
COPY --chown=train:train pretrained_weights/ /opt/train/cache/saved_models/
RUN mkdir -p /opt/train/cache/saved_models/multitask/convnext/ten_net0 && \
    cat /opt/train/cache/saved_models/w* > /opt/train/cache/saved_models/multitask/convnext/ten_net0/pretrained_model_TciaMosmed.pth && \
    rm /opt/train/cache/saved_models/w* && \
    echo "22b1d7e832cb19107bef566e1705dcbde48f10a02048d4e1fafe243b3db891e0  /opt/train/cache/saved_models/multitask/convnext/ten_net0/pretrained_model_TciaMosmed.pth" | sha256sum -c

COPY --chown=train:train algorithm/ /opt/train/

ENV DATA_PATH="/input/"
ENV LOGS_PATH="/scratch/logs"
ENV CACHE_PATH="/scratch/auto_cache"
ENV PRETRAINED_PATH="/opt/train/cache/saved_models"

ENTRYPOINT bash train2gpu.bash


