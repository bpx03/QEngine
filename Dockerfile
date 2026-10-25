# syntax=docker/dockerfile:1
# QEngine: Qwen3.8-27B INT4 inference server for 3x RTX 3090 (sm_86, tensor parallel 3).
# The model is NOT baked in: mount it at /app/models/Qwen3.8-27B-INT4 (see docker-compose.yml).
# Qwen3.8-Flash-Next (preview): mount it too and set QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4.
#
#   docker build -t qengine .                       # with vision (torch, ~+4 GB)
#   docker build -t qengine --build-arg VISION=0 .  # text only, no torch

ARG CUDA_VERSION=13.1.1
ARG UBUNTU=ubuntu24.04

# ---------------------------------------------------------------- build: libqengine.so
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
# same flags as CMakeLists (sm_86, Release); g++-15 is not in Ubuntu 24.04, so the distro
# GCC 13 (supported by nvcc 13.1) is used
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build

# ---------------------------------------------------------------- runtime
# base = CUDA runtime (libcudart) only; libcuda comes from the host driver (nvidia-container-toolkit)
FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU} AS runtime
ARG VISION=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN python3 -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
# server deps first (small), then the vision-only packages in their own layer
RUN grep -vE '^(torch|pillow|safetensors)==' /tmp/requirements.txt > /tmp/req-core.txt \
    && pip install -r /tmp/req-core.txt
RUN if [ "$VISION" = "1" ]; then \
      pip install --index-url "$TORCH_INDEX" $(grep -E '^torch==' /tmp/requirements.txt | sed 's/ *#.*//') \
      && pip install $(grep -E '^(pillow|safetensors)==' /tmp/requirements.txt | sed 's/ *#.*//'); \
    fi

WORKDIR /app
COPY server ./server
COPY --from=build /src/build/libqengine.so ./build/libqengine.so
# Qwen3.8-Flash-Next engine (preview): picked by the server when QE_MODEL points to a Flash-Next directory
COPY --from=build /src/build/libqflash.so ./build/libqflash.so
COPY LICENSE ./

# Entry point: server/run.sh with the container defaults (listen on 0.0.0.0).
# QE_GPUS = GPU ids as seen in the container (with --gpus all: the host ids), default 0,1,2;
# run.sh sets CUDA_VISIBLE_DEVICES from it and passes --devices 0..n-1 (n != 3: warning, less tested).
COPY <<'EOF' /usr/local/bin/qengine-entrypoint
#!/usr/bin/env bash
set -euo pipefail
cd /app
export QE_DEVICES="${QE_GPUS:-0,1,2}"
MODEL="${QE_MODEL:-/app/models/Qwen3.8-27B-INT4}"
if [ ! -f "$MODEL/config.json" ]; then
  echo "model not found at $MODEL: mount it (-v /path/to/models:/app/models:ro)" >&2
  exit 1
fi
# QE_EXTRA_ARGS (env) and the container command line are both passed to server/app.py
exec server/run.sh "${QE_PORT:-8080}" --host 0.0.0.0 \
  --metrics-host 0.0.0.0 --metrics-port "${QE_METRICS_PORT:-9393}" \
  --model "$MODEL" ${QE_EXTRA_ARGS:-} "$@"
EOF
RUN chmod +x /usr/local/bin/qengine-entrypoint

# gemm tune table, KV split and stats live in ~/.cache/qengine (keep it in a volume: the
# first start without it tunes the GEMMs, which takes a while)
ENV HOME=/root
VOLUME ["/root/.cache/qengine"]
EXPOSE 8080 9393
ENTRYPOINT ["/usr/local/bin/qengine-entrypoint"]

# OCI metadata: docker build --build-arg REVISION=$(git rev-parse HEAD) --build-arg VERSION=1.1 .
ARG REVISION=unknown
ARG VERSION=dev
LABEL org.opencontainers.image.source="https://github.com/bpx03/QEngine" \
      org.opencontainers.image.description="QEngine: Qwen3.8-27B INT4 (and Qwen3.8-Flash-Next preview) inference server for RTX 3090 (sm_86), OpenAI/Anthropic-compatible API; model not included" \
      org.opencontainers.image.licenses="PolyForm-Noncommercial-1.0.0" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.version="${VERSION}"
