# Build + test environment for hosts without a CUDA toolkit: nvcc 13.1, CMake, Ninja, Python with the
# test requirements (torch cu130). The source tree is mounted, not copied, so builds land in ./build.
#
#   docker build -t qengine-dev -f docker/dev.Dockerfile .
#   docker run --rm --gpus all --ipc=host -u $(id -u):$(id -g) -e HOME=/tmp \
#     -v $PWD:$PWD -v $HOME/models:$HOME/models -w $PWD qengine-dev \
#     sh -c 'cmake -S . -B build -G Ninja && cmake --build build && python tests/test_w8.py'

ARG CUDA_VERSION=13.1.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ git \
      python3 python3-venv python3-dev \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN python3 -m venv /opt/venv
COPY requirements.txt requirements-dev.txt /tmp/
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
      $(grep -hE '^(torch|torchvision)==' /tmp/requirements.txt /tmp/requirements-dev.txt | sed 's/ *#.*//') \
    && pip install -r /tmp/requirements-dev.txt
