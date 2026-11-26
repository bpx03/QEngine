# QEngine

![QEngine: fast local AI on RTX 3000, 4000 and 5000 series cards](docs/img/qengine.jpg)

<p align="center">
  <a href="https://buymeacoffee.com/bpx03"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="60"></a>
  <br>
  <sub>Built and tested on one home machine. If QEngine helps you, a coffee keeps it going.</sub>
</p>

QEngine is an open-source (MIT) inference engine for six models on NVIDIA RTX 3000, 4000 and 5000 series cards, with
nothing extra to install or set for any of them (tuned on the RTX 3090). One server serves any of them through an
**OpenAI-compatible** and an **Anthropic-compatible** API (it works with agent clients such as Claude Code). The
C++/CUDA engines are built as shared libraries; the Python server loads them and handles the API, tool calls,
structured output, scheduling and caching.

## Models

| model | cards | weights | single-stream decode, narrative / code (tok/s) | prefill, 10K / 90K-token prompt (tok/s) | quality against a reference |
|---|---|---|---:|---:|---|
| [Qwen3.8-Flash-Next](docs/models/qwen3.8-flash-next.md) | 2 or more (tuned for 3 and 2) | [bpx03/Qwen3.8-Flash-Next-QEngine](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine) | 310 / 452 | 8,254 / 8,390 | next-token KL 0.114 against BF16, top-1 agreement 89.6 % |
| [Qwen3.8-27B](docs/models/qwen3.8-27b.md) | 3 (tuned), 2, 1 as an experiment | [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) plus its MTP head; INT8 opt-in: [bpx03/Qwen3.8-27B-INT8-QEngine](https://huggingface.co/bpx03/Qwen3.8-27B-INT8-QEngine) | 275 / 524 | 6,188 / 4,994 | next-token KL 0.0154 against BF16, top-1 agreement 95.3 % |
| [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md) (preview) | 3 × 24 GB plus about 170 GB of host RAM; RTX 3000 / 4000 series only | [peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF) (third-party Maya-S GGUF files, as they are) | 88 / 120 | 4,217 / 4,895 | negative log-likelihood 1.635 on my test texts (no full-precision reference fits on this machine) |
| [GLM-5.3-Flash](docs/models/glm-5.3-flash.md) (preview) | 3 × 24 GB plus about 192 GB of host RAM | [bpx03/GLM-5.3-Flash-INT4-QEngine](https://huggingface.co/bpx03/GLM-5.3-Flash-INT4-QEngine) (my INT4 quant) | 110 / 92 | 2,920 / 3,572 | next-token KL 0.59 against the official FP8 model (three prompts: 0.45-0.77) |
| [gpt-oss-120B](docs/models/gpt-oss-120b.md) (preview) | 3 × 24 GB, all in VRAM; compute capability 8.6 (8.9 runs, untested) | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) (OpenAI's checkpoint as it is) | 301 / 301 | 13,280 / 5,055 | next-token KL 0.118 against an fp32 reference of the model, top-1 agreement 86.9 % |
| [Inkling-2-Mini](docs/models/inkling-2-mini.md) (preview) | 1 × 24 GB, or 2 with GPU P2P; compute capability 8.6 only | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) plus two small files: [bpx03/Inkling-2-Mini-QEngine](https://huggingface.co/bpx03/Inkling-2-Mini-QEngine) | 2 cards: 230 / 330; 1 card: 162 / 223 | 2 cards: 9,430 / 8,860; 1 card: 6,186 / 4,945 | next-token KL 0.00539 against an fp32 reference of the model, top-1 agreement 97.6 % (the official bf16 forward: 0.00592) |

Measured on my machine (3× RTX 3090, 24 GB, PCIe 4.0 x16, GPU P2P on), the QEngine 1.5.0 release-candidate builds
with their default settings and the club-3090 benchmark tools (November 19, 2026); Inkling-2-Mini with the 1.6.0
release build (November 26, 2026; its KL: a development build, November 25, 2026).
How they were measured: [Benchmarks](docs/benchmarks.md#headline-numbers); task results of every model (8-pack):
[Quality](docs/benchmarks.md#quality). Each model page has what it needs, its model files, what works, its limits,
its settings and more numbers.

The current release is 1.6.2 (`latest`). What changed in it and in every earlier release:
[releases page](https://github.com/bpx03/QEngine/releases).

## Documentation

<a id="contents"></a>Everything beyond the quick start is in these pages:

- Models: [Qwen3.8-Flash-Next](docs/models/qwen3.8-flash-next.md), [Qwen3.8-27B](docs/models/qwen3.8-27b.md), [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md), [GLM-5.3-Flash](docs/models/glm-5.3-flash.md), [gpt-oss-120B](docs/models/gpt-oss-120b.md), [Inkling-2-Mini](docs/models/inkling-2-mini.md)
- [Hardware and interconnect](docs/hardware.md): GPUs and smaller cards, GPU P2P, parallel modes, the link probe, host
  memory, operating system
- [API and serving behaviour](docs/api.md): endpoints, tool calls, sampling, thinking budget, empty replies, images,
  steering, scheduling, agents, failure handling, logs
- [Benchmarks](docs/benchmarks.md): method, quality (KL, 8-pack), measuring your own setup
- [Advanced settings](docs/advanced.md): common settings, tuning, failure handling, disk cache, the
  Flash-Next model build, activation steering, measurements, the troubleshooting table, repository layout and tests
- [Changelog](#changelog)

## Quick start

You need Linux, RTX 3000, 4000 or 5000 series cards (how many depends on the model: the cards column of the
[model table](#models); see [Hardware](#hardware-and-interconnect)), an NVIDIA driver with CUDA 13.1
support, and enough host RAM (Flash-Next: about 140 GB for the full defaults; with less, the start-up sizes its host
memory to the machine, and it starts with about 28 GB free; see [Host memory](docs/hardware.md#host-memory-flash-next)).

### 1. Get a model

The `hf` command comes with the Hugging Face hub client: `pip install -U "huggingface_hub[cli]"`.

```bash
# Qwen3.8-Flash-Next: my checkpoint is the supported way to run it (~129 GB, ready to run: converted weights,
# INT8 n-gram table and three small files)
hf download bpx03/Qwen3.8-Flash-Next-QEngine --local-dir models/Qwen3.8-Flash-Next-QE-INT4
# (the previous, 1.3.4 weights: add --revision v1.3.4; see docs/models/qwen3.8-flash-next.md)

# Qwen3.8-27B (incl. its MTP head); two small tuned files are fetched automatically at the first start
hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir models/Qwen3.8-27B-INT4
```

The other four models: see "Model files" on their pages ([DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md#how-it-runs),
[GLM-5.3-Flash](docs/models/glm-5.3-flash.md#how-it-runs), [gpt-oss-120B](docs/models/gpt-oss-120b.md#how-it-runs),
[Inkling-2-Mini](docs/models/inkling-2-mini.md#how-it-runs)).

### 2a. Run with Docker

Requirements on the host: the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/). One image serves all
models; it picks the engine from the model's `config.json` (DeepSeek-V4.1-Flash: from the GGUF files).

```bash
docker run -d --name qengine --restart unless-stopped --gpus all --ipc=host --ulimit memlock=-1 \
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:9393:9393 \
  -v /path/to/models:/app/models:ro -v qengine-cache:/root/.cache/qengine \
  -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 \
  ghcr.io/bpx03/qengine:latest
```

- Without `QE_MODEL` the container serves `/app/models/Qwen3.8-27B-INT4`.
- The API listens on 127.0.0.1 only, as the source start does. To reach it from other machines, publish it as
  `-p 8080:8080` (all interfaces) and put a firewall or an authenticating proxy in front: the server has no API keys.
- `--restart unless-stopped` matters: after a stalled GPU the server exits with code 75 and relies on a restart (see
  [Failure handling](docs/api.md#serving-behaviour)).
- Images: `latest` / `1.6.2` (text only) and `vision` / `1.6.2-vision` (with the image encoder). Earlier: `1.6.1`, `1.6.1-vision`, `1.6.0`, `1.6.0-vision`, `1.5.1`, `1.5.1-vision`, `1.5.0`, `1.5.0-vision`, `1.3.6`, `1.3.6-vision`, `1.3.5.9`, `1.3.5.9-vision`, `1.3.5.8`, `1.3.5.8-vision`, `1.3.5.7`, `1.3.5.7-vision`, `1.3.5.6`, `1.3.5.6-vision`, `1.3.5` and `1.3.5-vision`; experimental `1.3.5.5`, `1.3.5.5-vision`, `1.3.5.4`, `1.3.5.4-vision`, `1.3.5.3`, `1.3.5.3-vision`, `1.3.5.2`, `1.3.5.2-vision`, `1.3.5.1`, `1.3.5.1-vision`.
  Earlier releases stay available under their version tags.
- `QE_GPUS=0,2` picks the GPUs (host ids with `--gpus all`; default: every GPU in the container, Inkling-2-Mini one;
  `QE_DEVICES`, if set, wins over `QE_GPUS`). Settings are environment variables
  (`-e FLASH_PAR=pp`); anything after the image name, or `QE_EXTRA_ARGS`, goes to the server (`--think 0`,
  `--vision 1`).
- Keep the `qengine-cache` volume: it holds the GEMM tuning table, the downloaded 27B files and the stats.
- `--ipc=host` and `--ulimit memlock=-1` are needed for the pinned host memory. On SELinux hosts (Fedora, RHEL) add
  `--security-opt label=disable`.
- Port 8080: the API. Port 9393: monitoring dashboard, `/api/stats`, Prometheus `/metrics`.
- Compose: `QE_MODELS_DIR=/path/to/models docker compose up -d` (see `docker-compose.yml`). It builds the image
  from this checkout (it does not pull `ghcr.io/bpx03/qengine`) and serves the 27B; for Flash-Next add
  `QE_MODEL: /app/models/Qwen3.8-Flash-Next-QE-INT4` under `environment:`. It passes `QE_GPUS` and `QE_DEVICES`
  from your shell (`QE_DEVICES=0,2 docker compose up -d`); with neither set it uses every GPU, as `docker run` does.
  Unlike the `docker run` line above, it publishes the API on **all interfaces** (`QE_HOST_PORT`, default 8080) and the monitoring on 127.0.0.1
  (`QE_METRICS_HOST_PORT`, 9393). The server has no API keys, so anyone who can reach that port can use it: set
  `QE_HOST_PORT=127.0.0.1:8080` to keep it on this machine, or put it behind a firewall or an authenticating proxy.
  `QE_VISION_BUILD=0` builds the text-only image, `QE_REVISION` sets the build hash.
- DeepSeek-V4.1-Flash, GLM-5.3-Flash, gpt-oss-120B and Inkling-2-Mini (previews): the mount and the settings are under
  "Docker" on each model page. The DSV4, GLM, gpt-oss and Inkling engines add cuBLAS to the image.
- Build it yourself: `docker build -t qengine .` (with vision), `--build-arg VISION=0` (text only),
  `--build-arg CUDA_ARCHS=86` (RTX 3090 code only, faster build), `--build-arg REVISION=$(git rev-parse HEAD)` (the
  build hash the monitoring headers show; without it they say `build unknown`).

### 2b. Or build from source

Requirements: CUDA Toolkit **13.1**, CMake ≥ 3.24 and Ninja (or Make), a host compiler nvcc 13.1 accepts (it
rejects GCC 16; `CMakeLists.txt` uses `/usr/bin/g++-15` when present, otherwise pass e.g.
`-DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14`), Python 3.12.

```bash
cmake -B build -G Ninja
cmake --build build                          # -> build/libqengine.so (27B), build/libqflash.so + build/flash_run (Flash-Next)
python3 -m pip install -r requirements.txt   # torch: install the CUDA build matching your driver (see the file)
```

A source build contains code for the GPUs of the machine it is configured on (their compute capabilities from
`nvidia-smi`; the cmake output names them), always plus RTX 3090 code (sm_86). To build for other cards, name
them: `cmake -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES="86;120"` (or `"86;89;120"`, or just `120`).

```bash
server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4   # Flash-Next on every GPU present, API on 127.0.0.1:8080
server/run.sh 8080                                             # Qwen3.8-27B (the default model directory)
python3 server/app.py --help                                   # all command-line options
```

### 3. Recipes by card count

| setup | command (source; Docker: the same variables with `-e`, GPUs with `QE_GPUS`) | notes |
|---|---|---|
| 3 cards (tuned) | `server/run.sh 8080 --model <dir>` | the default: every GPU present (`nvidia-smi -L`); on a machine with more cards, `QE_DEVICES=0,1,2` picks three |
| 2 cards | `QE_DEVICES=0,2 server/run.sh 8080 --model <dir>` | Flash-Next keeps about half of its experts in host RAM; the 27B serves 16 slots, as on 3 cards (`QE_MAX_SEQS=32` for 32) |
| 1 card (27B only, experimental) | `QE_DEVICES=0 QE_MAX_SEQS=1 server/run.sh 8080` | one conversation, 206,848 tokens of context; `QE_MAX_SEQS` takes 1 to 32 (default 16, lowered at start to what fits), 1 to 4 keeps long context; MTP drafts unless you lower `--max-ctx` to what the start-up line says (`--max-ctx 165K` on my 24 GB card) or set `QE_DRAFTER=dflash2` ([DFlash2](docs/models/qwen3.8-27b.md#dflash2-block-drafter-27b-on-when-it-fits)) |
| a card also drives a desktop | Flash-Next `FLASH_EC_MARGIN_MB=256,2048,256`; 27B `QE_RESERVE_GB=1.5,3,1.5` | more free VRAM on that card (values in GPU order) |
| 4 or more cards (Flash-Next) | `QE_DEVICES=0,1,2,3 server/run.sh 8080 --model <dir>` | exactly 4 equal cards with verified P2P: `tp`, otherwise a pipeline (`pp`); see [4 or more cards](docs/hardware.md#4-or-more-cards) |
| long documents, short answers (Flash-Next) | `FLASH_PAR=pp` | pipeline mode: faster long prompts, slower single-stream decode |
| many long conversations (Flash-Next) | `FLASH_KV=ram FLASH_KV_RAM_TOKENS=4M` | KV cache in host RAM, ~74K tokens per GB |
| more KV per GB (Flash-Next) | `FLASH_KV_BITS=4` | 4-bit KV cache: 7,488 instead of 13,520 bytes per token (~134K tokens per GB with `FLASH_KV=ram`); see [Host memory](docs/hardware.md#host-memory-flash-next) |
| small cards, more context (27B) | `QE_KV_BITS=4` | 4-bit KV cache: about 1.5x the context on 8-12 GB cards; see [Smaller cards](docs/hardware.md#gpus) |

The first start on a new card count tunes the GEMMs and caches the table in `~/.cache/qengine`.

### 4. First request

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-qengine",
  "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
  "max_tokens": 1024
}'
```

The served names are `qwen3.8-flash-next-qengine` and `qwen3.8-27b-qengine` (the previews: `deepseek-v4.1-flash-qengine`, `glm-5.3-flash-qengine`, `gpt-oss-120b-qengine`, `inkling-2-mini-qengine`; `--served-name` changes it).
For **Claude Code**, set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` and `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.

### 5. Behind llama-swap (or another model switcher)

I run QEngine behind [llama-swap](https://github.com/mostlygeek/llama-swap) myself. A source build:

```yaml
healthCheckTimeout: 900          # the first start of a model takes longer (see below)
models:
  qwen3.8-flash-next-qengine:
    cmd: bash -c 'cd /path/to/QEngine && exec server/run.sh ${PORT} --model /path/to/models/Qwen3.8-Flash-Next-QE-INT4'
    checkEndpoint: /health
  qwen3.8-27b-qengine:
    cmd: bash -c 'cd /path/to/QEngine && exec server/run.sh ${PORT} --model /path/to/models/Qwen3.8-27B-INT4'
    checkEndpoint: /health
```

The Docker form (the switcher stops the container by name):

```yaml
  qwen3.8-flash-next-qengine:
    cmd: >
      docker run --rm --name qengine-fn --gpus all --ipc=host --ulimit memlock=-1
      -p 127.0.0.1:${PORT}:8080 -v /path/to/models:/app/models:ro -v qengine-cache:/root/.cache/qengine
      -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 ghcr.io/bpx03/qengine:latest
    cmdStop: docker stop -t 60 qengine-fn
    checkEndpoint: /health
```

- `exec` matters: it makes the server the process the switcher stops, so a stop ends the engine and frees the cards.
- `GET /health` answers only once the model is loaded and the API is up, so it is the readiness check.
- A start takes longer the first time on a machine: the GEMMs are tuned for the card count, the 27B downloads its
  tuned MTP head and draft ranking and the DFlash2 drafter (about 1.28 GB), and with `--lsa` the index map. All of it
  is cached in `~/.cache/qengine` (Docker: keep the `qengine-cache` volume), so later starts skip it.
- The 27B and Flash-Next lock the cards they use. A second one of them that starts while the first is still exiting
  waits up to `QE_GPU_LOCK_WAIT_S` (60 s) and then stops with the first one's pid and command line. GLM-5.3-Flash,
  DeepSeek-V4.1-Flash, gpt-oss-120b and Inkling-2-Mini take no lock: nothing stops one of them on cards already in
  use. So the switcher must stop one QEngine model before it starts another on the same GPUs (llama-swap does that by default: one model at a time).
  Two models side by side need separate cards (`QE_DEVICES`, Docker `QE_GPUS`).

## Hardware and interconnect

QEngine is tuned on 3× RTX 3090 with GPU P2P and plenty of host RAM, but it adapts by itself to machines that differ
from that:

- **no P2P** (stock driver, x4 / x8 slots, dual-socket or NUMA): both engines exchange through pinned host memory by
  DMA, close to the P2P speed ([details](docs/hardware.md#with-and-without-gpu-p2p));
- **long prompts on slow links**: `FLASH_PAR=pp`, a layer pipeline, picked automatically where it wins
  ([Parallel mode](docs/hardware.md#parallel-mode-flash-next-flash_par));
- **small host RAM** (e.g. 64-96 GB): the host-memory plan adapts, with no swapping at the defaults
  ([Host memory](docs/hardware.md#host-memory-flash-next));
- **mixed or uneven cards**: a per-card calibration and a layer split by speed;
- **4 or more cards**: see [4 or more cards](docs/hardware.md#4-or-more-cards); `FLASH_PAR=pptp` is experimental;
- **small cards** (8-16 GB): see [Smaller cards](docs/hardware.md#gpus);
- **RTX 3000, 4000 and 5000 series**: built in, nothing to set ([what has run where](docs/hardware.md#gpus)).

If a setup like this is still slow or misbehaves, please open an issue with the start-up log, started with
`FLASH_BOX_REPORT=1` (it adds `[box]` lines on each card and link; on 4 or more cards they are on by default): those,
the `[qflash] parallel mode` line and the `host RAM plan` line are what I need to look into it.

Details, on [docs/hardware.md](docs/hardware.md):

- <a id="gpus"></a>[GPUs](docs/hardware.md#gpus): what has run where, cards per model, smaller cards, the 27B 4-bit KV cache, GPU selection
- <a id="with-and-without-gpu-p2p"></a>[With and without GPU P2P](docs/hardware.md#with-and-without-gpu-p2p)
- <a id="parallel-mode-flash-next-flash_par"></a>[Parallel mode (Flash-Next, `FLASH_PAR`)](docs/hardware.md#parallel-mode-flash-next-flash_par), <a id="4-or-more-cards"></a>[4 or more cards](docs/hardware.md#4-or-more-cards)
- <a id="link-probe-and-flash_comm--qe_comm"></a>[Link probe and `FLASH_COMM` / `QE_COMM`](docs/hardware.md#link-probe-and-flash_comm--qe_comm), NUMA
- <a id="host-memory-flash-next"></a>[Host memory (Flash-Next)](docs/hardware.md#host-memory-flash-next): the host-RAM plan, the Flash-Next 4-bit KV cache, overrides
- <a id="operating-system"></a>[Operating system](docs/hardware.md#operating-system)

## Feature matrix

The six engines don't all do the same things. This table is what each one supports today (✅ yes, 🟡 partly, ❌ no).

| feature | Qwen3.8-27B | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash | GLM-5.3-Flash | gpt-oss-120B | Inkling-2-Mini |
|---|---|---|---|---|---|---|
| Tool calls, streaming, OpenAI + Anthropic APIs | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Structured output (`response_format`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ HTTP 400 |
| `tool_choice` required / named / none | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 none only (the reply ends at its first call) |
| `logprobs` | ✅ | ✅ | ❌ HTTP 400 | ✅ | ❌ HTTP 400 | ❌ HTTP 400 |
| Presence / frequency penalty | ✅ | ✅ | ❌ ignored | ❌ ignored | ✅ | ✅ |
| Thinking budget | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ ignored |
| Thinking off | ✅ | ✅ | ✅ | ✅ | 🟡 always reasons (effort low) | 🟡 effort `none`: a short thought remains |
| Images (`--vision 1`) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Activation steering | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Prefix cache, shared system prompt | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Idle conversations in host RAM | ✅ | ✅ | ✅ | ✅ (8 at once, one decodes at a time) | ✅ | ✅ |
| Disk cache (`--disk-cache`) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Write after every reply (`--disk-eager`, opt-in) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| KV cache in host RAM (`FLASH_KV=ram`) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Smaller K/V formats (opt-in) | ✅ 4-bit | ✅ 4-bit | ❌ | ❌ | ✅ FP8, 12-bit K | ❌ |
| Dense weight form choice (`--dense-weights`) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ `int8` (GPTQ) / `g15` (original weights, lossless) |
| Max context (default) | 256K, 1M with YaRN | 256K, 1M with YaRN + KV in RAM | 256K | 256K | 128K | 1M, served up to the KV pool (~565K on 2 cards with INT8 dense weights, ~544K lossless; ~126K on one card, ~117K lossless) |
| Parked replies beyond the seats (`--park`) | ✅ | ✅ (3+ cards) | ❌ | ❌ | ❌ | ✅ opt-in (`--park 1`) |
| Drafts | MTP, DFlash2, lookup | MTP, lookup | DSpark | MTP | prompt lookup | DSpark |
| Sparse attention for long prompts | ✅ LSA (opt-in) | built in | built in | built in | ❌ | ❌ (sliding window on most layers) |
| Cards | 3 (tuned), 2, 1 (experimental); 4 accepted, untested | 2-4 TP, more in PP | 3 | 3 (2 with fewer expert slots) | 3 (2-4 accepted, untested) | 1 by default, 2-3 TP with `QE_DEVICES` (needs P2P) |
| Without GPU P2P | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 one card only |
| Docker image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

`n > 1`, `logit_bias` and `parallel_tool_calls: false` are ignored on every model.

Inkling-2-Mini with thinking off, its dense weight forms and its slots: [model page](docs/models/inkling-2-mini.md#thinking-off-dense-weights-and-slots).

At start the server logs one `[settings]` line for every flag or environment variable that is set but does nothing for
the loaded model, e.g. `FLASH_KV=ram is set but has no effect for gpt-oss-120B (KV stays in VRAM ...)`. A request
parameter the model doesn't support gets one `[api]` line the first time it shows up; the request is then served as
the table says (ignored, or HTTP 400).

## Troubleshooting and FAQ

**An agent stops early, or a benchmark scores tasks as zero.**

- Look for `WARNING empty reply` in the log. A reasoning-only turn ends many agent loops. Non-streamed requests are
  resampled (`--empty-retry`); streamed ones cannot be. If the line says `finish length` and `</think> no`, the reply
  ran out of `max_tokens` while thinking: set a thinking budget (`--thinking-budget`, or per request) or raise
  `max_tokens`.
- Check the context. Agent prompts grow large (AutomationBench sales reaches 94K tokens); a harness ends a task when a
  prompt no longer fits. In a replay of AutomationBench sales (November 1, 2026), a 32K-token context cut 48 of 100
  tasks and left 40 at zero, against 6 with the default 262K. Check `max_model_len` in `/v1/models`: `--max-ctx`,
  a small `QE_MAX_SEQS` or a single card lower it, and so can a proxy in between.
- `history: ... reasoning only; left out of the prompt` is informational, not an error.

**No P2P (stock driver, dual-socket server, cloud box).** Nothing to set: both engines detect it and use host-staged
exchanges (see [With and without GPU P2P](docs/hardware.md#with-and-without-gpu-p2p)). For long-prompt work on Flash-Next try
`FLASH_PAR=pp`, which needs no fast link.

**Long prompts fail with `non-finite values`, or the log shows `GUARD` lines.** Set `FLASH_COMM=ce` (27B:
`QE_COMM=ce`) and report the `[comm]` start-up lines.

**Slow decode, or poor scaling with concurrent requests, on a board with a high P2P co-run tax.** See the end of
[Link probe](docs/hardware.md#link-probe-and-flash_comm--qe_comm).

**`not enough GPU memory for a KV cache` (27B).** Too many slots for the free VRAM, e.g. on one card: lower
`QE_MAX_SEQS`, leave `--vision` off.

**A desktop on one of the cards stutters.** Flash-Next `FLASH_EC_MARGIN_MB=256,2048,256`, 27B
`QE_RESERVE_GB=1.5,3,1.5` (the larger value at that card's position).

**The server says another engine holds the GPU.** Each card serves one engine process. Stop the other one, or wait if
it is still exiting after a crash; check `dmesg` for Xid lines.

**The server restarted with exit code 75.** The engine stalled (`collective timeout`) or the watchdog saw no progress
(`[hang]`). The open requests got an error first. Please report the log, which holds the diagnostics.

**Does it run on Windows / WSL2?** Untested. WSL2 has no GPU P2P and limits pinned host memory.

**Is greedy output reproducible?** Per engine, and the settings for it: [docs/advanced.md](docs/advanced.md#is-greedy-output-reproducible).

More symptoms and log lines: [docs/advanced.md](docs/advanced.md#troubleshooting).

## Model details

Each model has its own page now; these headings keep older links working.

### Qwen3.8-Flash-Next

[docs/models/qwen3.8-flash-next.md](docs/models/qwen3.8-flash-next.md): how it runs, model files, precision, context up to 1M, KV cache in host RAM, settings, benchmarks.

### Qwen3.8-27B

[docs/models/qwen3.8-27b.md](docs/models/qwen3.8-27b.md): verify head, release files, <a id="dflash2-block-drafter-27b-on-when-it-fits"></a>[DFlash2 block drafter](docs/models/qwen3.8-27b.md#dflash2-block-drafter-27b-on-when-it-fits), <a id="int8-weights-27b-opt-in-experimental"></a>[INT8 weights](docs/models/qwen3.8-27b.md#int8-weights-27b-opt-in-experimental), <a id="long-documents-27b-opt-in"></a>[long documents](docs/models/qwen3.8-27b.md#long-documents-27b-opt-in), settings, benchmarks. <a id="both-models-context-and-the-kv-archive"></a>Context and the KV archive of both Qwen models: [docs/api.md](docs/api.md#both-models-context-and-the-kv-archive).

### DeepSeek-V4.1-Flash (preview)

[docs/models/deepseek-v4.1-flash.md](docs/models/deepseek-v4.1-flash.md)

### GLM-5.3-Flash (preview)

[docs/models/glm-5.3-flash.md](docs/models/glm-5.3-flash.md)

### gpt-oss-120B (preview)

[docs/models/gpt-oss-120b.md](docs/models/gpt-oss-120b.md)

### Inkling-2-Mini (preview)

[docs/models/inkling-2-mini.md](docs/models/inkling-2-mini.md)

## API

[docs/api.md](docs/api.md#api): endpoints, tool calls, `tool_choice`, structured output, sampling, `logprobs`, thinking
and reasoning effort, <a id="thinking-budget-and-the-answer-tail"></a>[thinking budget and the answer tail](docs/api.md#thinking-budget-and-the-answer-tail), <a id="empty-replies-are-sampled-again"></a>[empty replies](docs/api.md#empty-replies-are-sampled-again), <a id="images"></a>[images](docs/api.md#images), <a id="activation-steering"></a>[activation steering](docs/api.md#activation-steering).

## Serving behaviour

[docs/api.md](docs/api.md#serving-behaviour): scheduling, agents, the KV pool under pressure, speculative decoding, loops,
turn end, template markers, failure handling, logs, monitoring.

## Settings

<a id="both-models"></a>Common settings: [docs/advanced.md](docs/advanced.md#common-settings). Per model: <a id="qwen38-flash-next-1"></a>[Qwen3.8-Flash-Next](docs/models/qwen3.8-flash-next.md#settings), <a id="qwen38-27b-1"></a>[Qwen3.8-27B](docs/models/qwen3.8-27b.md#settings), <a id="deepseek-v41-flash-preview-1"></a>[DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md#settings), <a id="glm-53-flash-preview-1"></a>[GLM-5.3-Flash](docs/models/glm-5.3-flash.md#settings), <a id="gpt-oss-120b-preview-1"></a>[gpt-oss-120B](docs/models/gpt-oss-120b.md#settings), <a id="inkling-2-mini-preview-1"></a>[Inkling-2-Mini](docs/models/inkling-2-mini.md#settings).

## Benchmarks

[docs/benchmarks.md](docs/benchmarks.md): the method. Per model: <a id="qwen38-flash-next-2"></a>[Qwen3.8-Flash-Next](docs/models/qwen3.8-flash-next.md#benchmarks), <a id="qwen38-27b-2"></a>[Qwen3.8-27B](docs/models/qwen3.8-27b.md#benchmarks), <a id="deepseek-v41-flash-preview-2"></a>[DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md#benchmarks), <a id="glm-53-flash-preview-2"></a>[GLM-5.3-Flash](docs/models/glm-5.3-flash.md#benchmarks), <a id="gpt-oss-120b-preview-2"></a>[gpt-oss-120B](docs/models/gpt-oss-120b.md#benchmarks), <a id="inkling-2-mini-preview-2"></a>[Inkling-2-Mini](docs/models/inkling-2-mini.md#benchmarks).

### Quality

[docs/benchmarks.md](docs/benchmarks.md#quality): Flash-Next KL against BF16 and how to measure it, the club-3090 8-pack
of every model, AutomationBench. <a id="measure-your-own-setup"></a>[Measure your own setup](docs/benchmarks.md#measure-your-own-setup).

## Repository layout and tests

[docs/advanced.md](docs/advanced.md#repository-layout-and-tests)

## Changelog

Every release has notes on the [releases page](https://github.com/bpx03/QEngine/releases), with the fixes, new
settings and how to upgrade.

## Support

<p align="center"><a href="https://buymeacoffee.com/bpx03"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="50"></a></p>

QEngine is built and tested on one machine at home, and every release goes through long benchmark, quality and
stress runs before it ships. If it saves you money or lets you run something you couldn't before, you can
[buy me a coffee](https://buymeacoffee.com/bpx03). It pays for power, disks and memory for new models, and the time
it takes to fix the bugs you report. Bug reports and benchmark results from your own hardware help just as much.

## License

QEngine is licensed under the MIT License, see [LICENSE](LICENSE). Versions up to 1.3.5.7 were released under the
PolyForm Noncommercial License 1.0.0.

The model weights are not part of this repository; they are distributed by their authors under their own licenses.
`server/templates/qwen38_syssafe.jinja` is a modified version of the model's chat template. The included steering
vector falls under the model's license (see [docs/advanced.md](docs/advanced.md#included-vector-flash-next)).

## Disclaimer

QEngine is provided "as is", without warranty of any kind, express or implied, including but not limited to
the warranties of merchantability, fitness for a particular purpose and non-infringement. As far as the law
allows, the author is not liable for any claim, damages or other liability arising from the use of the
software, including hardware damage, data loss, incorrect model output or downtime. It is experimental
software, tested on a single machine (3× RTX 3090); use it at your own risk. Benchmark results describe that
machine and may not reproduce on other hardware, drivers or software versions. See [LICENSE](LICENSE) for the
binding terms.
