FROM ubuntu:22.04
LABEL version="0.1"
LABEL maintainer="docker@infinstor.com"
LABEL description="Inference container image"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt install -y bash \
                   build-essential \
                   software-properties-common \
                   git \
                   curl \
                   ca-certificates \
                   docker.io \
                   git \
                   lxc \
                   iptables \
                   unzip \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

RUN curl --output cuda-ubuntu2204.pin https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN curl --output cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda

RUN pip install -U pyopenssl cryptography
RUN pip install torchvision --pre -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
RUN pip install torch
RUN pip install pandas transformers boto3 sentence-transformers llama_index
RUN pip install aiohttp
RUN pip install pillow
RUN pip install deepspeed
RUN pip install sentencepiece
RUN pip install google
RUN pip install protobuf
RUN pip install gunicorn
RUN pip install mlflow
RUN pip install infinstor_mlflow_plugin
