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
# included activation-steering vectors (refusal_projection, Flash-Next only); loaded only with QE_STEERING != 0 (vectors/README.md)
COPY vectors ./vectors

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
# Activation steering (off by default). QE_STEERING=0: no vectors, the server runs as without them. QE_STEERING=1:
# the included refusal_projection (Flash-Next only) is the default of every request; QE_STEERING=<name>: that vector
# is the default. With any value but 0, the included vectors (/app/vectors, Flash-Next only) and the user's (*.gguf +
# sidecar .json mounted at /vectors; a mounted vector wins over an included one of the same name) are all loaded, so a
# request can pick one ("steering": "<name>") or turn steering off ("steering": false). QE_STEER / QE_STEER_DIR set by
# the user win over all of this.
if [ -n "${QE_STEERING:-}" ] && [ "${QE_STEERING}" != 0 ]; then
  if [ -n "${QE_STEER:-}${QE_STEER_DIR:-}" ]; then
    echo "QE_STEERING=${QE_STEERING}: QE_STEER / QE_STEER_DIR already set, using those" >&2
  else
    sel=$(python3 - "$MODEL" "$QE_STEERING" <<'PY'
import json, os, sys
model, want = sys.argv[1], sys.argv[2]
def log(m): print("QE_STEERING=%s: %s" % (want, m), file=sys.stderr)
cfg = json.load(open(os.path.join(model, "config.json")))
t = cfg.get("text_config") or cfg
flash = (cfg.get("model_type") or t.get("model_type") or "").startswith("qwen4_exp")
def scan(d):  # name -> (gguf, sidecar); the name as the engine reads it: sidecar "name", else the file stem
    out = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".gguf"):
            continue
        g = os.path.join(d, f)
        side = g[:-5] + ".json" if os.path.isfile(g[:-5] + ".json") else g + ".json"
        name = f[:-5]
        try:
            n = json.load(open(side)).get("name")
            if isinstance(n, str) and n.strip() and not n.strip().startswith("<"):
                name = n
        except Exception:
            pass  # missing / bad sidecar: the engine refuses it with the reason
        if name in out or "/" in name or name.startswith("."):
            log("vector name '%s' in %s is repeated or not a file name" % (name, d)); sys.exit(1)
        out[name] = (g, side)
    return out
builtin = scan("/app/vectors") if flash else {}
if not flash:
    log("the included vectors are for Qwen3.8-Flash-Next and %s is not Flash-Next: not loaded" % model)
mounted = scan("/vectors")
for n in sorted(set(builtin) & set(mounted)):
    log("mounted vector %s replaces the included one" % n)
vecs = {**builtin, **mounted}
default = "refusal_projection" if want == "1" else want
if default not in vecs:
    if want == "1" and not flash:
        log("refusal_projection is for Flash-Next only: no default vector, requests are not steered")
        default = ""
    else:
        log("vector '%s' is not loaded; available: %s" % (default, ", ".join(sorted(vecs)) or "(none)")); sys.exit(1)
if not vecs:
    sys.exit(0)
out = "/tmp/qe-steer"
os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    os.unlink(os.path.join(out, f))
for n, (g, side) in vecs.items():
    os.symlink(g, os.path.join(out, n + ".gguf"))
    if os.path.exists(side):
        os.symlink(side, os.path.join(out, n + ".json"))
log("vectors %s, default %s" % (", ".join(sorted(vecs)), default or "(none)"))
print(out + "\t" + default)
PY
) || exit 1
    if [ -n "$sel" ]; then
      export QE_STEER_DIR="${sel%%$'\t'*}"
      [ -n "${sel#*$'\t'}" ] && export QE_STEER_DEFAULT="${sel#*$'\t'}"
    fi
  fi
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
