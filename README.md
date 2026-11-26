# QEngine

![QEngine: fast local AI on RTX 3000, 4000 and 5000 series cards](docs/img/qengine.jpg)

<p align="center">
  <a href="https://buymeacoffee.com/bpx03"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="60"></a>
  <br>
  <sub>Built and tested on one home machine. If QEngine helps you, a coffee keeps it going.</sub>
</p>

QEngine is an open-source (MIT) inference engine for six models on NVIDIA RTX 3000, 4000 and 5000 series cards, with
nothing extra to install or set for any of them (tuned on the RTX 3090):

- **Qwen3.8-Flash-Next**, run from my checkpoint [bpx03/Qwen3.8-Flash-Next-QEngine](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine)
  (routed experts INT4 from Intel's AutoRound quantization, repacked bit-exact; the parts every token passes through in
  INT8 or calibrated INT4): a sparse mixture-of-experts model with 512 routed experts, 262K
  context (up to 1M opt-in) and image input. Runs on 2 or more cards (tuned for 3 and 2).
- **Qwen3.8-27B** (INT4, the RedHatAI W4A16 checkpoint [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
  plus its MTP head; INT8 opt-in: [bpx03/Qwen3.8-27B-INT8-QEngine](https://huggingface.co/bpx03/Qwen3.8-27B-INT8-QEngine)): a dense model with 262K context (up to 1M
  opt-in, with optional sparse attention for long prompts) and image input. Tuned for 3 cards, also runs on 2, and on 1 as an experiment.
- **DeepSeek-V4.1-Flash** (preview), run from the third-party Maya-S GGUF files as they are
  ([peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)): a large sparse
  mixture-of-experts model (384 routed experts per layer, Engram n-gram memory, compressed sparse attention) on three
  24 GB cards plus about 170 GB of host RAM, with a DSpark draft model, 262K context. RTX 3000 / 4000 series only.
- **GLM-5.3-Flash** (preview), run from my INT4 quant [bpx03/GLM-5.3-Flash-INT4-QEngine](https://huggingface.co/bpx03/GLM-5.3-Flash-INT4-QEngine)
  (routed experts and KDA layers): a sparse mixture-of-experts
  model (288 routed experts per layer, 8 active) on three 24 GB cards plus about 192 GB of host RAM, with the experts in
  host RAM and a CPU expert lane next to the cards, MTP drafts, 262K context.
- **gpt-oss-120B** (preview), run from OpenAI's official checkpoint as it is,
  [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) (MXFP4 experts): a sparse
  mixture-of-experts model (128 experts per layer, 4 active) on three 24 GB cards, all in VRAM, 128K context, up to 8
  conversations decoded together. RTX 3090-class cards (compute capability 8.6) only.
- **Inkling-2-Mini** (preview), run from Thinking Machines' official checkpoint
  [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (NVFP4 / MXFP8 experts as
  shipped) plus two small files of mine ([bpx03/Inkling-2-Mini-QEngine](https://huggingface.co/bpx03/Inkling-2-Mini-QEngine):
  INT8 dense attention weights and a trained DSpark drafter): a looped sparse mixture-of-experts model (26.4B
  parameters, 3.4B active) on one 24 GB card (about 126K tokens of context) or two with GPU P2P (about 565K), with
  exact DSpark drafts and up to 8 conversations decoded together. RTX 3090-class cards (compute capability 8.6) only.

One server serves any of them through an **OpenAI-compatible** and an **Anthropic-compatible** API (it works with
agent clients such as Claude Code). The C++/CUDA engines are built as shared libraries; the Python server loads them
and handles the API, tool calls, structured output, scheduling and caching.

**Measured on my machine** (3× RTX 3090, 24 GB, PCIe 4.0 x16, GPU P2P on), QEngine 1.5.0 with its default settings
and the club-3090 benchmark tools (November 19, 2026):

| output tok/s unless noted | Qwen3.8-27B | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash | GLM-5.3-Flash | gpt-oss-120B | Inkling-2-Mini, 2 cards | Inkling-2-Mini, 1 card |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-stream decode, narrative / code | 275 / 524 | 310 / 452 | 88 / 120 | 110 / 92 | 301 / 301 | 230 / 330 | 162 / 223 |
| prefill, 10K-token prompt: tok/s, time to first token | 6,188, 1.60 s | 8,254, 1.23 s | 4,217, 2.35 s | 2,920, 3.42 s | 13,280, 0.74 s | 9,430, 1.07 s | 6,186, 1.57 s |
| prefill, 90K-token prompt: tok/s, time to first token | 4,994, 18.0 s | 8,390, 10.5 s | 4,895, 17.8 s | 3,572, 25.4 s | 5,055, 17.6 s | 8,860, 10.2 s | 4,945, 17.6 s |
| total output, 1 / 4 / 8 / 16 streams at once | 312 / 575 / 848 / 999 | 300 / 642 / 880 / 1,061 | 70 / 73 / 78 / 78 | 72 / 55 / 50 / 44 | 306 / 594 / 731 / 835 | 213 / 362 / 431 / – | 147 / – / – / – |

| model | weights | quality against a reference |
|---|---|---|
| Qwen3.8-27B | INT4 W4A16, group 128 (RedHatAI) plus its MTP head | next-token KL 0.0154 against BF16, top-1 agreement 95.3 % |
| Qwen3.8-Flash-Next | routed experts INT4 (Intel's AutoRound, repacked bit-exact); shared expert and dense parts INT8 or calibrated INT4; DeltaNet projections INT8 | next-token KL 0.114 against BF16, top-1 agreement 89.6 % |
| DeepSeek-V4.1-Flash | the Maya-S GGUF files as they are: experts IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S, dense Q8_0, Engram Q6_K | negative log-likelihood 1.635 on my test texts (no full-precision reference fits on this machine) |
| GLM-5.3-Flash | my INT4 quant: routed experts and KDA layers INT4 ([details](#glm-53-flash-preview)) | next-token KL 0.59 against the official FP8 model (three prompts: 0.45-0.77) |
| gpt-oss-120B | OpenAI's checkpoint as it is: MXFP4 experts, bf16 dense weights | next-token KL 0.118 against an fp32 reference of the model, top-1 agreement 86.9 % |
| Inkling-2-Mini | the official checkpoint's NVFP4 / MXFP8 experts as they are; dense attention weights GPTQ INT8, one scale per 128 inputs (`--dense-weights g15`: the original bf16 weights, lossless) | next-token KL 0.00539 against an fp32 reference of the model, top-1 agreement 97.6 % (the official bf16 forward: 0.00592) |

(Single-stream decode: club-3090 `bench.sh` with each model's own sampling; streams: its `concurrency-probe.sh`,
greedy, 256-token prompts and replies, total output of all streams; each number is the mean of 2 to 6 runs. The two
Qwen servers ran with thinking on at effort low and vision on. For 16 streams the servers that hold 8 conversations
by default were started with 16: `GLM_CONVS=16`, DeepSeek-V4.1-Flash `--max-active 16`, `GPTOSS_SLOTS=16`.
DeepSeek-V4.1-Flash and GLM-5.3-Flash decode one conversation at a time and give the streams turns, so their total
output stays at about one stream's. gpt-oss-120B at 90K: prompts of 86,786 and 89,998 tokens. Quality: the same
builds; the [8-pack](#quality) results are below. Inkling-2-Mini: the 1.6.0 release build, November 26, 2026, its
server defaults (8 slots on 2 cards, GPUs 0 and 2 with P2P; 2 slots on one card, GPU 0), so the stream counts above its
slots are not measured ("–"); greedy with the mix of nine prompt types it decodes about 320 tok/s single-stream on 2
cards and 214 on one card ([Inkling-2-Mini](#inkling-2-mini-preview)).)

**Flash-Next quality**, in more detail: next-token KL divergence against the official BF16 model on the public set
[jyohukuchan/qwen38-flash-next-bf16-logits](https://huggingface.co/datasets/jyohukuchan/qwen38-flash-next-bf16-logits),
scored with its own `compare.py`: mean KL **0.114**, top-1 agreement **89.6 %**, WikiText-2 KL **0.125** (0.126 when
the context went through batched prefill). 1.5.0 build, November 19, 2026, the same as 1.3.6 to every digit; details
and the command under [Quality](#quality).

**What is new in 1.6.1** (stable, `latest`; it also holds everything from 1.6.0 below): one Qwen3.8-27B fix. With
the bf16 KV cache (`QE_KV_FP8=0`), conversations decoded together no longer come out garbled: the per-conversation
attention path now writes the activations the next projection reads and uses each conversation's own position shift
([#49](https://github.com/bpx03/QEngine/issues/49)). The default FP8 cache was not affected and gives the same output
as before; the other models are unchanged. The release checks and the numbers on this page are from 1.6.0
([release notes](https://github.com/bpx03/QEngine/releases/tag/v1.6.1)).

**What is new in 1.6.0**:

- **A sixth model, as a preview: [Inkling-2-Mini](#inkling-2-mini-preview)**, Thinking Machines' looped mixture-of-experts
  model (26.4B parameters, 3.4B active), run from the official checkpoint plus two small files of mine (INT8 dense
  attention weights and a fine-tuned DSpark drafter), on one 24 GB card or on two with GPU P2P: about 320 tok/s
  single-stream on 2 cards and 214 on one card (greedy, mix of nine prompt types), 565,216 tokens of context per
  conversation on 2 cards and 125,920 on one card, up to 8 conversations decoded together, exact drafts also for
  sampled replies (`top_p` / `top_k` / `min_p`), prompt cache, idle conversations in host RAM, parked replies, Docker.
  8-pack 122 / 124 of 150 (thinking off / on).
- **Restore points at the last message starts** ([#48](https://github.com/bpx03/QEngine/issues/48)): Flash-Next keeps the
  state at the start of a prompt's last two messages, so an agent client that drops a short reminder message or the
  reasoning of its last reply resumes there instead of recomputing from the system prompt (125K-token conversation:
  15.5-16.3 s -> 0.4-0.6 s per turn); `FLASH_CKPT_TAIL=0` turns it off.
- **27B disk cache** ([#47](https://github.com/bpx03/QEngine/issues/47)): a conversation attached to a shared system prompt
  is now saved with that system prompt in its own file, so it comes back after a restart.
- **Opt-in `--disk-eager`** (27B and Flash-Next, [#33](https://github.com/bpx03/QEngine/issues/33)): with `--disk-cache`,
  each conversation is also written right after every finished reply, so a crash or power loss loses far less.
- **GLM-5.3-Flash on 3 cards without GPU P2P** (stock driver) no longer stops at load with a cuBLAS error
  ([#44](https://github.com/bpx03/QEngine/issues/44)): the prefill indexer runs on every card when the cards have no
  peer access; a collective stall prints the watchdog report and honours `QE_COLL_TIMEOUT_MS`.
- **gpt-oss-120B**: after every finished reply the idle conversation is copied to host RAM in the background
  (`GPTOSS_BG_REARM=0` = off), so the next long prompt does not wait for that copy; a request never drops the stored
  conversation it is about to restore.
- The 27B, Flash-Next and DeepSeek-V4.1-Flash engines are unchanged apart from the items above (27B greedy output and
  Flash-Next deterministic output identical to 1.5.x, decode within 1 %); the speed table above is from 1.5.0 for
  those models.

1.6.0 went through the release checks on my machine: the CPU suites, identity and speed points of every model,
Inkling-2-Mini's 8-pack (once) and its stress runs up to the full context on 2 cards and on one card with the other
slots busy, VRAM and host-RAM hogs taken before the start and mid-run (Inkling-2-Mini, gpt-oss-120B), and a Docker
start of all six models from the release image.

Older releases and what changed in each: [releases page](https://github.com/bpx03/QEngine/releases).

## Contents

- [Quick start](#quick-start)
- [Hardware and interconnect](#hardware-and-interconnect)
- [Models](#models)
- [Feature matrix](#feature-matrix)
- [API](#api)
- [Serving behaviour](#serving-behaviour)
- [Settings](#settings)
- [Benchmarks](#benchmarks)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
- [Repository layout and tests](#repository-layout-and-tests)
- [Changelog](#changelog)
- [Support](#support)
- [License](#license), [Disclaimer](#disclaimer)

More settings, measurements and the troubleshooting table: [docs/advanced.md](docs/advanced.md).

## Quick start

You need Linux, two or three RTX 3000, 4000 or 5000 series cards (see [Hardware](#hardware-and-interconnect)), an
NVIDIA driver with CUDA 13.1
support, and enough host RAM (Flash-Next: about 140 GB for the full defaults; with less, the start-up sizes its host
memory to the machine, from 64 GB up; see [Host memory](#host-memory-flash-next)).

### 1. Get a model

The `hf` command comes with the Hugging Face hub client: `pip install -U "huggingface_hub[cli]"`.

```bash
# Qwen3.8-Flash-Next: my checkpoint is the supported way to run it (~129 GB, ready to run: converted weights,
# INT8 n-gram table and three small files)
hf download bpx03/Qwen3.8-Flash-Next-QEngine --local-dir models/Qwen3.8-Flash-Next-QE-INT4
# (the previous, 1.3.4 weights: add --revision v1.3.4; see "Model files" under Models)

# Qwen3.8-27B (incl. its MTP head); two small tuned files are fetched automatically at the first start
hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir models/Qwen3.8-27B-INT4
```

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
  [Failure handling](#serving-behaviour)).
- Images: `latest` / `1.6.1` (text only) and `vision` / `1.6.1-vision` (with the image encoder). Earlier: `1.6.0`, `1.6.0-vision`, `1.5.1`, `1.5.1-vision`, `1.5.0`, `1.5.0-vision`, `1.3.6`, `1.3.6-vision`, `1.3.5.9`, `1.3.5.9-vision`, `1.3.5.8`, `1.3.5.8-vision`, `1.3.5.7`, `1.3.5.7-vision`, `1.3.5.6`, `1.3.5.6-vision`, `1.3.5` and `1.3.5-vision`; experimental `1.3.5.5`, `1.3.5.5-vision`, `1.3.5.4`, `1.3.5.4-vision`, `1.3.5.3`, `1.3.5.3-vision`, `1.3.5.2`, `1.3.5.2-vision`, `1.3.5.1`, `1.3.5.1-vision`.
  Earlier releases stay available under their version tags.
- `QE_GPUS=0,2` picks the GPUs (host ids with `--gpus all`; default `0,1,2`; `QE_DEVICES`, if set, wins over `QE_GPUS`). Settings are environment variables
  (`-e FLASH_PAR=pp`); anything after the image name, or `QE_EXTRA_ARGS`, goes to the server (`--think 0`,
  `--vision 1`).
- Keep the `qengine-cache` volume: it holds the GEMM tuning table, the downloaded 27B files and the stats.
- `--ipc=host` and `--ulimit memlock=-1` are needed for the pinned host memory. On SELinux hosts (Fedora, RHEL) add
  `--security-opt label=disable`.
- Port 8080: the API. Port 9393: monitoring dashboard, `/api/stats`, Prometheus `/metrics`.
- Compose: `QE_MODELS_DIR=/path/to/models docker compose up -d` (see `docker-compose.yml`). It builds the image
  from this checkout (it does not pull `ghcr.io/bpx03/qengine`) and serves the 27B; for Flash-Next add
  `QE_MODEL: /app/models/Qwen3.8-Flash-Next-QE-INT4` under `environment:`. Unlike the `docker run` line above, it
  publishes the API on **all interfaces** (`QE_HOST_PORT`, default 8080) and the monitoring on 127.0.0.1
  (`QE_METRICS_HOST_PORT`, 9393). The server has no API keys, so anyone who can reach that port can use it: set
  `QE_HOST_PORT=127.0.0.1:8080` to keep it on this machine, or put it behind a firewall or an authenticating proxy.
  `QE_VISION_BUILD=0` builds the text-only image, `QE_REVISION` sets the build hash.
- DeepSeek-V4.1-Flash (preview): mount the folder that holds both `Maya-S/` and `dspark/` and point `QE_MODEL` at
  `Maya-S`, e.g. `-v /path/to/DeepSeek-V4.1-Flash-1017-GGUF:/app/models/dsv4:ro -e QE_MODEL=/app/models/dsv4/Maya-S`.
  Symbolic links inside the mount must resolve inside the container (mount their targets at the same paths). The
  image carries the DSV4 engine for sm_86 / sm_89 only. Engine settings go in with `-e`
  (`-e DSV4_EGMIRROR=...`), server flags in `QE_EXTRA_ARGS` (`--route-margin 0.05`).
  One image serves all models; the DSV4, GLM, gpt-oss and Inkling engines add cuBLAS to it.
- GLM-5.3-Flash (preview): mount the `bpx03/GLM-5.3-Flash-INT4-QEngine` directory, e.g.
  `-v /path/to/GLM-5.3-Flash-INT4-QEngine:/app/models/glm:ro -e QE_MODEL=/app/models/glm`. The experts are pinned in
  host RAM (about 157 GB), so `--ipc=host --ulimit memlock=-1` matter here. Engine settings go in with `-e`
  (`-e GLM_CPU_ISA=avx2`), server flags in `QE_EXTRA_ARGS` (`--split 704,704,640`).
- gpt-oss-120B (preview): mount the `openai/gpt-oss-120b` directory, e.g.
  `-v /path/to/gpt-oss-120b:/app/models/gpt-oss-120b:ro -e QE_MODEL=/app/models/gpt-oss-120b`. Engine settings go in
  with `-e` (`-e GPTOSS_SLOTS=4`). The image carries the gpt-oss engine for sm_86 only.
- Inkling-2-Mini (preview): mount the model directory with `dense8.safetensors` and `mtp_qe.safetensors` in it, e.g.
  `-v /path/to/Inkling-2-Mini:/app/models/inkling:ro -e QE_MODEL=/app/models/inkling`. It runs on one card (the first
  of `--gpus`) unless `QE_GPUS` / `QE_DEVICES` name the cards (`-e QE_GPUS=0,1`: two cards, needs GPU P2P). The image
  carries the Inkling engine for sm_86 only.
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
server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4   # Flash-Next on GPUs 0,1,2, API on 127.0.0.1:8080
server/run.sh 8080                                             # Qwen3.8-27B (the default model directory)
python3 server/app.py --help                                   # all command-line options
```

### 3. Recipes by card count

| setup | command (source; Docker: the same variables with `-e`, GPUs with `QE_GPUS`) | notes |
|---|---|---|
| 3 cards (tuned) | `server/run.sh 8080 --model <dir>` | the default: GPUs 0,1,2 |
| 2 cards | `QE_DEVICES=0,2 server/run.sh 8080 --model <dir>` | Flash-Next keeps about half of its experts in host RAM; the 27B serves 16 slots, as on 3 cards (`QE_MAX_SEQS=32` for 32) |
| 1 card (27B only, experimental) | `QE_DEVICES=0 QE_MAX_SEQS=1 server/run.sh 8080` | one conversation, 206,848 tokens of context; `QE_MAX_SEQS` up to 4; MTP drafts unless you lower `--max-ctx` to what the start-up line says (`--max-ctx 165K` on my 24 GB card) or set `QE_DRAFTER=dflash2` ([DFlash2](#dflash2-block-drafter-27b-on-when-it-fits)) |
| a card also drives a desktop | Flash-Next `FLASH_EC_MARGIN_MB=256,2048,256`; 27B `QE_RESERVE_GB=1.5,3,1.5` | more free VRAM on that card (values in GPU order) |
| 4 or more cards (Flash-Next) | `QE_DEVICES=0,1,2,3 server/run.sh 8080 --model <dir>` | exactly 4 equal cards with verified P2P: `tp`, otherwise a pipeline (`pp`); see [4 or more cards](#4-or-more-cards) |
| long documents, short answers (Flash-Next) | `FLASH_PAR=pp` | pipeline mode: faster long prompts, slower single-stream decode |
| many long conversations (Flash-Next) | `FLASH_KV=ram FLASH_KV_RAM_TOKENS=4M` | KV cache in host RAM, ~74K tokens per GB |
| more KV per GB (Flash-Next) | `FLASH_KV_BITS=4` | 4-bit KV cache: 7,488 instead of 13,520 bytes per token (~134K tokens per GB with `FLASH_KV=ram`); see [Host memory](#host-memory-flash-next) |
| small cards, more context (27B) | `QE_KV_BITS=4` | 4-bit KV cache: about 1.5x the context on 8-12 GB cards; see [Smaller cards](#gpus) |

The first start on a new card count tunes the GEMMs and caches the table in `~/.cache/qengine`.

### 4. First request

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-qengine",
  "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
  "max_tokens": 1024
}'
```

The served names are `qwen3.8-flash-next-qengine` and `qwen3.8-27b-qengine` (Inkling-2-Mini: `inkling-2-mini-qengine`; `--served-name` changes it).
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
- Only one engine can hold a card. A second one that starts while the first is still exiting waits up to
  `QE_GPU_LOCK_WAIT_S` (60 s) and then stops with the first one's pid and command line. So the switcher must stop one
  QEngine model before it starts another on the same GPUs (llama-swap does that by default: one model at a time).
  Two models side by side need separate cards (`QE_DEVICES`, Docker `QE_GPUS`).

## Hardware and interconnect

QEngine is tuned on 3× RTX 3090 with GPU P2P and plenty of host RAM, but it adapts by itself to machines that differ
from that:

- **no P2P** (stock driver, x4 / x8 slots, dual-socket or NUMA): both engines exchange through pinned host memory by
  DMA, close to the P2P speed ([details](#with-and-without-gpu-p2p));
- **long prompts on slow links**: `FLASH_PAR=pp`, a layer pipeline, picked automatically where it wins
  ([Parallel mode](#parallel-mode-flash-next-flash_par));
- **small host RAM** (e.g. 64-96 GB): the host-memory plan adapts, with no swapping at the defaults
  ([Host memory](#host-memory-flash-next));
- **mixed or uneven cards**: a per-card calibration and a layer split by speed;
- **4 or more cards**: see [4 or more cards](#4-or-more-cards); `FLASH_PAR=pptp` is experimental;
- **small cards** (8-16 GB): see [Smaller cards](#gpus);
- **RTX 3000, 4000 and 5000 series**: built in, nothing to set ([what has run where](#gpus)).

If a setup like this is still slow or misbehaves, please open an issue with the start-up log, started with
`FLASH_BOX_REPORT=1` (it adds `[box]` lines on each card and link; on 4 or more cards they are on by default): those,
the `[qflash] parallel mode` line and the `host RAM plan` line are what I need to look into it.

### GPUs

QEngine runs on the RTX 3000, 4000 and 5000 series with no extra steps. The Docker images carry the code for all three,
and a source build targets the GPUs it finds (`-DCMAKE_CUDA_ARCHITECTURES` sets others).

| series | code | what has run on it |
|---|---|---|
| RTX 3000 | sm_86 | 3090: all tuning (82 SMs, tile shapes, split counts, memory budgets) targets it, and everything published here was measured on it. Community: 4× modded 3080 20 GB ([#9](https://github.com/bpx03/QEngine/issues/9)) |
| RTX 4000 | sm_89 | built, but no run on record yet; reports are welcome |
| RTX 5000 | sm_120 | community: 2× 5090 ([club-3090 #1826](https://github.com/noonghunna/club-3090/issues/1826)), 4× 5080 16 GB ([#13](https://github.com/bpx03/QEngine/issues/13)) |

The 4000 and 5000 series use the settings tuned for the 3090, so they may have speed left unused. What limits a card
is its memory; see below.

How many cards each model runs on:

| model | cards | how |
|---|---|---|
| Qwen3.8-Flash-Next | 3 (tuned), 2 | tensor parallel or pipeline, picked at start (`FLASH_PAR=auto`) |
| Qwen3.8-Flash-Next | 4 | tensor parallel when peer access between all 4 equal cards passes the start-up check, else pipeline (`FLASH_PAR=auto`); community runs only, see [4 or more cards](#4-or-more-cards) |
| Qwen3.8-Flash-Next | 5 or more | pipeline (`FLASH_PAR=pp`); tested only as 4 pipeline stages on my 3 cards |
| Qwen3.8-Flash-Next | 1 | not supported |
| Qwen3.8-27B | 3 (tuned), 2 | tensor parallel |
| Qwen3.8-27B | 1 | experimental, `QE_MAX_SEQS` 1 to 4 |

**Smaller cards** (Flash-Next; measured on my 3090s with all but 12 GB or 8 GB of each card taken by another process
before the start, November 3, 2026; defaults otherwise, 16 conversation slots; c=1 / c=4: greedy decode of 400-token
replies, thinking off; prefill of an 11.5K-token prompt; a 53K-token needle; the 27B follows below):

| model | cards x free VRAM | context | c=1 / c=4 tok/s | prefill tok/s | needle |
|---|---|---:|---:|---:|---|
| Flash-Next | 3 x 12 GB | 262,144 | 66.5 / 145.1 | 4,612 | pass |
| Flash-Next | 3 x 8 GB | 262,144 | 47.2 / 91.1 | 3,819 | pass |
| Flash-Next | 2 x 12 GB | 262,144 | 40.7 / 78.7 | 3,546 | pass |
| Flash-Next | 2 x 8 GB | 65,536 (fitted at start) | 34.7 / 55.1 | 2,263 | pass |

- Flash-Next keeps its dense weights, slot state and KV cache on the cards and puts the experts that do not fit in
  host RAM (on 8 GB cards nearly all of them, so decode is host-link bound). The minimum per card is about 7.5 GB on
  3 cards and about 8 GB on 2 cards at a 65K context (about 10 GB at 262K); when the requested context does not fit,
  it starts with the largest that does and says so (`--max-ctx` sets it, `FLASH_MAX_SEQS` fewer slots leave room for
  more). Below the minimum it stops with a message naming what to change.
- 27B needs its weights on the cards: about 5 GB per card on 3 cards and 7.2 GB on 2, plus buffers, the conversation
  slots' state (110 MB per slot and card on 3 cards, about 200 MB on 2) and the KV cache. On a small card it starts
  with fewer slots and a smaller VRAM reserve (logged; `QE_MAX_SEQS`, `QE_RESERVE_GB` set them) so the KV cache holds
  at least a 32K context: 8 GB cards work on 3 cards, 12 GB cards on 2. Below that it stops with a message (exit
  code 3).
- 27B on one card needs about 15.8 GB for the weights alone, so a single 16 GB card does not start; 2 x 16 GB does.

**27B on smaller cards, and with a 4-bit KV cache (opt-in, `QE_KV_BITS=4`)** for more context. Each key and value
row is rotated (a fixed random Hadamard transform, which spreads the outlier channels) and stored in 4 bits with one
fp16 scale per 32 values: 144 bytes per row instead of 256 with FP8, so the pool holds 1.6-1.7x the tokens (the MTP
layer stays FP8). Measured like the table above (November 14, 2026, same day for both caches, before the INT4
verify head of 1.3.6, which leaves each card more room for the KV pool; the 3 x 8 GB FP8 row is from November 3;
needle at about 60 % of the served context; slots as fitted for FP8):

| cards x free VRAM | KV cache | context | c=1 / c=4 tok/s | prefill tok/s | needle |
|---|---|---:|---:|---:|---|
| 3 x 12 GB | FP8 | 255,649 | 220.2 / 630.1 | 6,372 | pass |
| 3 x 12 GB | 4-bit | 262,144 | 189.8 / 653.4: the context now fits DFlash2, which is slower than MTP on this story prompt (`QE_DRAFTER=mtp`: 212.9 / 664.6) | 6,231 | pass |
| 3 x 8 GB | FP8 | 44,675 (9 slots) | 220.6 / 619.3 | 6,288 | pass |
| 3 x 8 GB | 4-bit | 67,584 (9 slots) | 215.2 / 630.2 | 6,364 | pass |
| 2 x 12 GB | FP8 | 30,720 (14 slots) | 171.5 / 338.5 | 4,568 | pass |
| 2 x 12 GB | 4-bit | 46,080 (14 slots) | 190.7 / 513.3 | 4,537 | pass |
| 1 x 24 GB, `QE_MAX_SEQS=1` | 4-bit | 240,640 with DFlash2 (262,144 with `QE_DRAFTER=mtp`; FP8 on 1.3.5.8: about 191K) | 102.9 / - | 2,550 | pass |
| 2 x 8 GB | either | does not start (the weights do not fit) | - | - | - |

- Quality: next-token KL against a bf16 KV cache over 100K / 32K / 64K-token documents is 4.4e-3 with 4 bits and
  1.5e-3 with FP8 (for scale: the INT4 weights against BF16 are 1.45e-2); top-1 agreement stays at FP8's level, a
  passphrase needle at 130K (3 depths) and 239K passes, MTP / DFlash2 acceptance within 2 %. It is opt-in because it
  is not lossless against FP8; on cards with room the FP8 default is the better choice.
- Speed: decode within 2 % of FP8 at 32K and faster at long context (128K: -9 % time per step); prefill 0.4 % slower
  at 128K and 3 % at 256K (the prompt's keys are converted for the prefill kernels, about 0.5 KB per key and KV head
  of the served context). On 3 x 12 GB the context now fits the DFlash2 drafter, which `QE_DRAFTER=auto` then loads,
  as on 24 GB cards: warm c=1, it is 10 % faster than MTP alone on code and 7 % slower on prose, the same on both
  memory sizes (the c=1 numbers in the table are a first, cold request).
- Also checked with 4 bits: conversations restored from the RAM archive continue bit-identically, the disk cache
  resumes after a restart, the KV pool shrinks and grows back when another process takes and frees 16 GB of a card,
  and `--max-ctx 512K` (YaRN) serves 524,288 tokens on 3 cards with a 331K-token needle found.
- Not with `QE_LSA` (sparse attention reads FP8 keys; it is turned off with a start-up line) and not with
  `QE_KV_FP8=0`. Conversations archived or saved to disk with one cache format are not reused by the other.

GPU selection: `QE_DEVICES=0,2` (physical ids; Docker `QE_GPUS`), or `--devices` (indices within
`CUDA_VISIBLE_DEVICES`). By default every visible GPU is used. Each engine locks the GPUs it uses: a second engine on
a card in use stops at start with the first one's pid and command line (`QE_ALLOW_SHARED_GPU=1` skips the check).

### With and without GPU P2P

GeForce cards have no GPU-to-GPU P2P with the stock NVIDIA driver. A community patch of the open kernel modules
([aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules), a fork of tinygrad's patch
that tracks current driver versions, with Resizable BAR on)
enables it. QEngine detects which one you have and needs no setting.

Since 1.3.4 both engines are fast without P2P: their tensor-parallel exchanges copy through pinned host memory by
DMA (the copy engines), and the GPU cores only wait on flags in their own memory. Measured on 3× RTX 3090 (PCIe 4.0
x16), with P2P forced off (`QE_P2P=0`) against P2P on (November 1, 2026):

| measured (s, lower is better) | without P2P | with P2P | 1.3.3 without P2P |
|---|---:|---:|---:|
| Flash-Next prefill, 10K / 32K / 90K-token prompt (`flash_run`) | 1.83 / 6.06 / 17.03 | 1.35 / 4.54 / 13.06 | 3.75 / 12.55 / 34.95 |
| Flash-Next decode step, one conversation | 9.9 ms | 9.5 ms | 12.7 ms |
| 27B prefill, 16K-token prompt | 2.68 | 2.59 | 3.94 |

The 27B's decode speed does not depend on P2P. Without P2P the 27B uses a smaller KV pool (its position-split KV
layout needs P2P). Inside Docker, P2P works exactly when it works on the host (check the `P2P:` / `[comm]` lines of the
start-up log).

### Parallel mode (Flash-Next, `FLASH_PAR`)

Flash-Next can share the cards in several ways; the 27B always runs tensor parallel. `FLASH_PAR` selects the mode
(source: `FLASH_PAR=pp server/run.sh ...`; Docker: `-e FLASH_PAR=pp`).

| mode | `FLASH_PAR` | what it does | recommended for | picked by `auto`? | how to force / options |
|---|---|---|---|---|---|
| automatic (default) | `auto` | picks `tp` or `pp` at start from a link probe, a short per-card speed calibration and a model of a typical agent request | almost every machine | - | `FLASH_PAR_AUTO_P=32768` weighs 32K-token prompts instead of 8K (favours `pp`) |
| tensor parallel | `tp` | every layer split over the cards; all cards work on every token, exchanges in every layer (P2P, or host-staged without it) | 2-3 equal cards: fastest decode and short prompts, with or without P2P | yes: 2-3 equal cards; exactly 4 equal cards with verified peer access between all of them, if it fits | `FLASH_PAR=tp`; `FLASH_COMM` sets the exchange path |
| pipeline | `pp` | each stage holds a contiguous range of whole layers; prompts flow through in micro-batches, several conversations decode as groups moving stage to stage; only a 40 KB row per token crosses between stages, so it needs no P2P | long uncached prompts with short replies (RAG, long documents); uneven cards; 4 or more cards | yes: 5+ cards; 4 cards when peer access is missing or unverified, the cards are uneven, or tp does not fit; and uneven cards (or 2-3 cards without P2P) when it wins the mix | `FLASH_PAR=pp`; `FLASH_PP_LAYERS=17,17,14` sets layers per stage; a card may hold several stages (`--devices 0,1,2,0`; a test layout, it adds no compute) |
| pipeline over TP groups (experimental, untested on 4 cards) | `pptp` | a pipeline whose stages are tensor-parallel groups of 2-3 cards; per-layer exchanges stay inside a group | 4+ cards where P2P exists only inside card pairs (two PCIe switches, two CPU sockets) | no, opt-in only | `FLASH_PAR=pptp FLASH_PP_MAP=0,1\|2,3` (`,` inside a stage, `\|` between stages); without a map `FLASH_PP_TP=2` pairs consecutive cards |

`auto` models an 8K-token uncached prompt and a 512-token reply, averaged over 1, 4 and 8 conversations at once
(capped at `FLASH_MAX_SEQS`); `pp` must beat `tp` by 10 % over that mix to be picked. With P2P on equal cards it is
always `tp`. Settings of the pick and the pipeline: [docs/advanced.md](docs/advanced.md#parallel-modes-flash_par).

**Recipes**

| your machine and workload | setting |
|---|---|
| 2-3 equal cards, with or without P2P, agents / chat | leave `auto` (= `tp`) |
| mostly long uncached prompts (RAG, long documents, short answers) | `FLASH_PAR=pp`, or `FLASH_PAR_AUTO_P=32768` to let `auto` decide on that basis |
| uneven or mixed cards | leave `auto`: it calibrates each card and picks `pp` with a split by speed when that wins |
| exactly 4 cards | leave `auto`: `tp` with verified peer access between all of them, otherwise `pp`; mostly long uncached prompts: `FLASH_PAR=pp` ([why](#4-or-more-cards)) |
| 5 or more cards | leave `auto` (= `pp`); if your cards pair up on P2P, try `FLASH_PAR=pptp FLASH_PP_MAP=0,1\|2,3` |
| steering vectors or `FLASH_KV=ram` needed | `FLASH_PAR=tp` (2-4 cards) |

```bash
FLASH_PAR=pp server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4
docker run ... -e FLASH_PAR=pp -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 ghcr.io/bpx03/qengine:latest
docker run ... -e FLASH_PAR=pptp -e FLASH_PP_MAP='0,1|2,3' -e QE_GPUS=0,1,2,3 ... ghcr.io/bpx03/qengine:latest
```

**What was picked.** The start-up log has one line with the mode and the reason, e.g.
`[qflash] parallel mode: tp (auto: no peer access, but TP not beaten by the margin over the mix); probe: ...`, and a
`[qflash] cards:` line with each card's measured speed and free memory (cards of the same model count as uneven only
when a longer re-calibration confirms the gap). `/api/stats` on the monitoring port shows the same under `parallel`.

**The trade-off.** `pp` prefills 32K-90K-token prompts up to 1.8× faster than `tp` (3 cards with P2P: a 90K prompt
in 9.9 s against 13.6 s; on 2 cards with P2P the two are about even), decodes 4-8 conversations at 0.83-1.03× of
`tp`, but one conversation at only 0.54-0.65×; 10K prompts with P2P are faster in `tp`. Measured on the server, same build, cold prompts end to end and decode of 1 / 4 / 8
conversations (November 1, 2026):

| cards, link | mode | 10K prompt | 32K prompt | 90K prompt | decode tok/s, 1 / 4 / 8 at once |
|---|---|---:|---:|---:|---:|
| 3, P2P | `tp` | 1.52 s | 5.20 s | 13.55 s | 244 / 541 / 773 |
| 3, P2P | `pp` | 1.72 s | 3.89 s | 9.88 s | 132 / 447 / 696 |
| 3, no P2P | `tp` | 2.05 s | 7.09 s | 18.03 s | 232 / 487 / 708 |
| 3, no P2P | `pp` | 1.77 s | 3.95 s | 10.21 s | 136 / 438 / 703 |
| 2, P2P | `tp` | 2.54 s | 5.67 s | 15.96 s | 91 / 200 / 246 |
| 2, P2P | `pp` | 3.08 s | 5.59 s | 14.75 s | 56 / 203 / 245 |
| 2, no P2P | `tp` | 3.78 s | 8.31 s | 23.45 s | 88 / 195 / 236 |
| 2, no P2P | `pp` | 2.97 s | 5.63 s | 14.85 s | 57 / 201 / 241 |

Each row is one fresh server; the decode runs come first with a cold expert cache, which costs 2-card single-stream
decode most. The prompts and method differ from the headline table, so compare rows within this table only.

**Uneven cards.** `pp` splits the layers by each card's measured speed and free memory. With one 3090 slowed to a
third of its compute (a test setting), the split moved 4 layers off it: prefill about 30 % and decode 13 % faster than
an equal split.

**What `pp` and `pptp` do not serve**: image input in `pptp` (refused with a clear message; `pp` serves images since 1.3.5.1, experimental), steering vectors (not loaded),
`FLASH_KV=ram` (on 2 or 3 cards the server runs `tp` instead and logs it; on 4 or more cards the start is refused with
a message), activation capture; lookup drafts are off. Structured output and the thinking budget work as in `tp`
(a constrained step runs through the stages as a batch of one).

#### 4 or more cards

I have no 4-card machine: the 4-way split was
tested as 4 pipeline stages on my 3 cards (two stages on one card), where it gave the same output bit for bit as 3
stages. `pptp` was tested only in a 3-card layout (`0,1|2`), where its next-token probabilities match `tp` within the
run-to-run noise. A community run on four RTX 5080 cards (`tp`, 1.3.5.7 and 1.3.5.9, [#24](https://github.com/bpx03/QEngine/issues/24))
confirmed Flash-Next's long-context recall there: needles found up to 240,664 tokens at 262K, and at 1M up to 843,147
on 1.3.5.7 and 813,180 on 1.3.5.9 (one ladder rung lower), the same level as on 3 cards. More reports from 4-card machines are welcome.

**Since 1.3.5.4 (experimental):** on exactly 4 equal cards, `auto` picks `tp` when peer access between all of them passes the data check
at start and `tp` fits the free VRAM; otherwise it picks `pp`. With 5 or more cards it is always `pp`. If `tp` can't start
(out of device or host memory during the load, or a CUDA error or a stall in the check), the server restarts itself in
`pp` and says so in the log, at the cost of the partial load. On a card whose free VRAM is badly fragmented, a `tp` that did
load can still stop later on a request: use `FLASH_PAR=pp` there. This rule rests on one user's measurements on one rig
(4× RTX 5080 16 GB behind one PCIe switch, 1.3.5): one 1,000-token stream at 310.3 tok/s in `tp` against 145.4 in `pp`. On
1.3.4.3, prefill at 32K was higher in `pp` (7,367 against 5,466 tok/s), so for mostly long uncached prompts set
`FLASH_PAR=pp`. The rest was tested on 3 cards only. On 1.3.5 to 1.3.5.3, `auto` always picks `pp` on 4 cards and
prints `peer access no` even if your cards have P2P, so set `FLASH_PAR=tp` yourself.

### Link probe and `FLASH_COMM` / `QE_COMM`

At start both engines measure every GPU pair for a few milliseconds (P2P loads and stores, copy-engine copies, copies
staged through host memory), how much each transfer slows a kernel running next to it (the co-run tax), and whether
data sent over P2P really arrives intact:

```
[comm] topology: numa -1,-1,-1 (1 node), PCIe x16,x16,x16, P2P rank 0-1:0 0-2:0 1-2:0
[comm] P2P data check: SM loads, SM stores and copy engine ok on every pair (...)
[comm] prefill transfers: P2P SM loads / stores, ... (FLASH_COMM=auto, auto: ...: P2P kept: tax over the cap, under the 1 NUMA node cap)
[comm] decode collectives (QE_DECODE_COMM=auto, real decode steps, median of 4 interleaved rounds of 12): single sequence: P2P SM stores (P2P ..., host memory ... ms/step, host faster in 0/4); batched (4 sequences): ...; calibration ... ms
```

With `auto` (the default) Flash-Next keeps P2P on fast links, moves the prefill transfers to the copy engine on slow,
narrow (e.g. an x4 slot) or cross-socket links, and never uses a transport that failed the data check. The line ends
with the reason. `FLASH_COMM` (Flash-Next) and `QE_COMM` (27B) force a path: `p2p`, `ce` (copy engine), `host`.
The rule and its thresholds: [docs/advanced.md](docs/advanced.md#how-flash_commauto--qe_commauto-choose-a-path).

Decode sends only small messages, but several per layer and step (all-reduces, the attention exchanges): either as
SM stores into the other cards' memory (P2P) or through the host path, with the same results. The host path is the
one the engines use without peer access (`QE_P2P=0`): each card writes its words once into pinned host memory and the
others read them; peer access stays on for prefill. Which one is faster depends on the board: peer stores can stall the
other kernels running on a card, and only the real model shows by how much. With `auto` both engines time a few real
decode steps on each path at the end of the start-up (one sequence, and four batched sequences), keep P2P or the host
path for each, and print both timings on the `[comm] decode collectives` line. `QE_DECODE_COMM=p2p` or `host` forces
the decode path (both engines; prefill keeps its own path). Before 1.3.5.6 the host candidate was a slower path
(about 45 % more time per step on 3 cards), so boards where P2P stores are expensive decoded slower than on 1.3.4.3. The measurements are cached in `~/.cache/qengine` (the Docker volume) and reused while the cards, links, driver and
build stay the same; `QE_COMM_CACHE=refresh` measures again. A cached decode line says `from the start-up cache, not
measured this boot` and carries the date of the measurement, so compare its ms/step across boots with care; a round
that was not measured (with fewer than 4 slots there is no batched round) prints `n/a`.

Set `FLASH_COMM=ce` (27B: `QE_COMM=ce`) yourself when the `[comm]` line says P2P but:

- long prompts fail with `non-finite values` (HTTP 500) or the log shows `GUARD` lines;
- a card sits behind something the link width does not show (a PCIe switch, a riser or extension cable, a chipset
  slot, a bifurcated slot);
- the cards differ (e.g. a 3090 next to a 3090 Ti) or sit in slots of different PCIe generations, and the line shows a
  P2P co-run tax of about 1.9 or more;
- prefill is clearly slower than the numbers here.

The copy engine gives the same output; on fast x16 links P2P is a few percent faster in prefill.

**NUMA.** On a machine with several NUMA nodes, Flash-Next places pinned host memory by GPU (each card's host-backed
experts and staging buffers on its node, the n-gram table interleaved over the GPUs' nodes). When the system does not
report a card's node (`GPU NUMA ID: N/A`), those buffers are interleaved over all nodes; `QE_NUMA` sets the nodes by hand.

### Host memory (Flash-Next)

Both engines plan their host-RAM parts from the machine's size, not from what happens to be free at start. If other
programs hold RAM at start, the plan warns and stays: the archive keeps below `MemAvailable` minus the margin while
memory is short and grows when it comes back. Only the Flash-Next n-gram table is decided at start: when it does not fit
next to the experts as it is read, it comes from the page cache until the next restart (decode about 30 % slower at
c=1). If the free RAM is short at start, the server first waits up to 20 s for it (a previous server still exiting;
`QE_RAM_WAIT_S`). If the model can load it runs, with warnings; the load stops with a clear message only when the
experts really do not fit. The start-up line `host RAM plan: ...` says what was reduced. While serving, when another
program takes RAM, the archive gives memory back (log lines `archive: host memory below the margin ...`) and the server
keeps running.

Measured minimum free host RAM at start (3 cards, November 3, 2026): Flash-Next about 28 GB (the host-backed experts
must be pinned: about 14 GB on 3 cards, about 31 GB on 2, plus a margin), full speed (the n-gram table pinned) from
about 85 GB; 27B started with 24 GB free (the lowest I tried). Below its minimum Flash-Next stops at start with a
message naming the shortfall, before it pins anything.

| part | host RAM |
|---|---|
| host-resident experts (what VRAM cannot hold) | about 12 GB (3 cards), about 31 GB (2 cards) |
| n-gram embedding table (INT8, pinned) | 51.8 GB |
| conversation archive (`FLASH_KV_ARCHIVE_GB`) | 48 GB, filled as conversations leave VRAM |
| pinned system prompts (`FLASH_PREFIX_GB`) | 8 GB, as agent kinds appear |
| server process (Python, vision encoder, CUDA) | about 5 GB |
| `FLASH_KV=ram` (opt-in) | 13.5 GB per 1M tokens of KV, pinned at start; smaller when the host is short (see below) |

**The defaults fit the machine (since 1.3.4).** At start-up the engine plans its host memory from `MemTotal` and
`MemAvailable` (and a cgroup / `docker --memory` limit, if one is set). It first sets aside a reserve for the system
and the desktop: max(8 GB, 8 % of RAM). The rest goes, in this order, to:

1. the host-resident experts, which are not optional;
2. the n-gram table. It is pinned when it fits next to an 8 GB minimum archive. Otherwise it is read through the page
   cache (as with `FLASH_NGRAM_PIN=0`, below), and part of the remaining RAM stays free for the page cache to hold
   its rows;
3. the archive and the pinned system prompts, which get what is left, up to 48 + 8 GB.

With `FLASH_KV=ram` the KV pool in host RAM (`FLASH_KV_RAM_TOKENS`, 13,520 bytes per token, 13.5 GB at the default 1M;
7,488 bytes and 7.5 GB with `FLASH_KV_BITS=4`) is in the plan as well. It is pinned at start, like the experts and the n-gram table, so these three together must fit
both the RAM the machine has and what is free at start (`MemAvailable` minus the margin). When they do not, the plan
gives way in this order: the archive and the pinned prompts first; then the KV pool, down to the seats times the
context (`FLASH_MAX_SEQS` x `--max-ctx`, so every seat still holds a full window); then down to one window; then the
n-gram table moves to the page cache (it costs decode speed for every request, so it goes last). The experts are never
reduced; if even the smallest plan does not fit, the log warns and the server starts anyway. The start-up line shows
the pool and what is pinned at start next to what was free.

**Flash-Next with a 4-bit KV cache (opt-in, `FLASH_KV_BITS=4`).** Each key and value row of the attention layers (the
MTP layer too) is rotated (a fixed random Hadamard transform, which spreads the outlier channels) and stored in 4 bits
with one fp16 scale per 32 values: 576 bytes per layer and token instead of 1,040 with FP8, 7,488 instead of 13,520
bytes per token in all. The VRAM pool keeps its 262,144 tokens and gives the freed memory to the expert cache; with
`FLASH_KV=ram` the same host RAM holds 1.8x the tokens (1.05M tokens: 7.9 GB pinned instead of 14.2). Measured on 3 cards
(November 14, 2026, same day for both caches): KL against the FP8 cache 4.95e-2 on 32K-100K documents (two FP8 runs
that differ only in summation order: 2.52e-2), top-1 agreement 92.2 %; needle at 130K and at 1M (`FLASH_KV=ram`,
YaRN) pass; prefill of a 90K prompt 1-2 % faster, decode steps unchanged, MTP drafts accepted 3 % less (3.00 vs 3.10
tokens per step at 90K). Other values of `FLASH_KV_BITS` and settings that select an attention kernel without a
4-bit reader (`FLASH_ATTN_TC`, `FLASH_ATTN_DEC=0`) keep the FP8 cache and log it. On 2 cards, prompts of about
115K-188K prefill in 4096-row batches with the 4-bit cache (the smaller K/V arrays have less room for the big
batches). Known limit, from one community report on 4 cards
([#25](https://github.com/bpx03/QEngine/issues/25)): on a 1M-token recall ladder the 4-bit cache filled 83 % of the
window (every miss a refusal) against 91 % with FP8, so keep FP8 when maximum reach matters.

Nothing is planned beyond the RAM the machine has, so the defaults do not push it into swap. With about 140 GB or
more (3 cards) or about 160 GB or more (2 cards), the plan is exactly the defaults above. The log has one line with
the plan, and `/api/stats` reports it as `host_memory`:

```
[qflash] host RAM plan: MemTotal 98.8 GB, MemAvailable 92.8 GB, reserve 8.0 GB (OS + desktop), server base 6.0 GB,
engine buffers 1.3 GB; experts in host memory 31.2 GB (estimate 31.3 GB), n-gram table 51.8 GB from the page cache
(12.0 GB kept for it); KV archive 34.5 GB + pinned prefixes 5.7 GB; reduced: n-gram table not pinned: ...
```

Planned sizes by machine (start-up plan, desktop using about 6 GB):

| RAM | cards | n-gram table | archive + pinned prompts |
|---|---|---|---|
| 64 GB | 2 / 3 | page cache | 9.6 + 1.5 / 25.2 + 4.1 GB |
| 92 GB | 2 | page cache | 34.5 + 5.7 GB |
| 92 GB | 3 | pinned | 16.8 + 2.8 GB |
| 128 GB | 2 / 3 | pinned | 30.9 + 5.1 / 47.4 + 7.9 GB |
| 160 GB and more | 2 / 3 | pinned | 48 + 8 GB (the defaults) |

**Overrides.** A value you set wins: `FLASH_NGRAM_PIN=1` pins the table, `=0` keeps it in the page cache;
`FLASH_KV_ARCHIVE_GB` and `FLASH_PREFIX_GB` set the archive and the pinned prompts (`0` turns them off). If a value you
set does not fit, the log says so and the value is kept. On Flash-Next `FLASH_KV_ARCHIVE_GB=0` also turns off the pinned
prompts and every reuse that goes through the archive: no checkpoints, no copy of a cached prefix from another slot, and
a conversation that leaves its slot is prefilled again on its next turn. This holds with `FLASH_KV=ram` too: the KV pool
in host RAM holds only the conversations in the slots, it does not replace the archive. `QE_HOST_BASE_GB` (6) changes the share counted for the
server process.

**Keeping RAM free for other programs: two settings, used at different times.**

| setting | when it acts | what it does | raise it when |
|---|---|---|---|
| `QE_HOST_RESERVE_GB` | once, at start | RAM the start-up plan leaves out when it sizes the n-gram table, the archive and the pinned prompts | other programs will need more RAM later, or the plan comes out too large |
| `QE_MEM_MARGIN_GB` | all the time, while serving | free RAM (`MemAvailable`) the archive keeps: below it, the archive stops growing and gives memory back | programs that start later get squeezed, or swap appears |

On a small machine, lower them to leave more for the archive: with both at their 8 GB default, a 16 GB machine has
almost no archive, so a conversation that leaves VRAM is recomputed when it returns (a planned archive under 2 GB gets a
warning line at start; the start goes on). The lower you go, the closer the
kernel's OOM killer gets, and it would end the server.

- On **2 cards** about half of the expert groups stay in pinned host memory (about 15 GB per card) and are read over
  PCIe or copied into the expert cache when hot.
- While it fills, the archive keeps at least max(8 GB, 5 % of RAM) free (`QE_MEM_MARGIN_GB`), and gives memory back
  when other programs need it. The engine also watches the kernel's memory-pressure and swap-in counters while it runs:
  when other programs squeeze the host, it stops growing, shrinks the archive, drops pinned prompts and, last, moves the
  n-gram table to the page cache, one logged stage at a time, and keeps serving (`QE_MEMP=0` turns this off;
  [details](docs/advanced.md#host-ram-safety)).
- **n-gram table in the page cache** (`FLASH_NGRAM_PIN=0`, or chosen at start-up): the table stays in the file mapping
  and a host thread copies the rows each step needs. With the table fully in the page cache, agent throughput at 8
  conversations was 793 tok/s (pinned: 805); with the whole server limited to 12 GB of RAM, 493 tok/s (3 cards,
  October 25, 2026). After the load the engine drops the page cache of the weight files (it never reads them again),
  so they don't push the table out of the page cache (`FLASH_WEIGHTS_COLD=0` keeps it). Details:
  [docs/advanced.md](docs/advanced.md#disk-mode-for-the-n-gram-table).
- The 27B needs much less: its archive defaults to 32 GB (`QE_KV_ARCHIVE_GB`) plus 8 GB of pinned prompts
  (`QE_PREFIX_GB`), sized the same way. Its defaults fit from 64 GB of RAM.

### Operating system

Linux only (developed on Fedora 44). **Windows, including Docker Desktop / WSL2, is untested**: WSL2 has no GPU P2P
and limits pinned host memory, which the archive and Flash-Next's host-resident parts rely on.

## Models

### Qwen3.8-Flash-Next

48 layers (36 Gated DeltaNet + 12 sparse-attention layers), 512 routed experts (top-10) plus a shared expert,
4-stream hyper-connections and a per-layer n-gram embedding table. What the engine does:

- experts split over the cards in 128-wide groups (3 cards: 176/160/176, so the middle card can drive a desktop;
  `--split` sets one weight per card); INT4 expert GEMV / GEMM, dense parts INT4 or INT8 (stored that way in the
  checkpoint, or converted to INT8 at load where the file carries BF16; see the precision table below);
- an expert cache: the least-used expert groups live in pinned host memory and are copied into free VRAM while hot;
- attention KV in FP8, split by position over the cards, with the indexer's top-512 block selection beyond 2K
  tokens; 262,144 tokens of context per conversation;
- MTP drafts plus lookup drafts (one conversation decoding alone: 3 to 5 drafts per step, picked from the draft head's
  confidence; batched steps: up to 3), steps launched ahead, CUDA graphs;
- up to 16 conversations at once with batched decoding; prompts that arrive together share prefill batches (the
  expert weights are read once for all of them);
- prefix reuse, checkpoints and a host-RAM archive of conversations; image input (`--vision 1`); structured output.

**Model files.** The [Hugging Face download](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine) is ready to
run. Since 1.3.4.1 its default revision stores the DeltaNet projections (in_proj_qkv / z / out_proj) as INT8 (group
128) and has the shared expert and the n-gram key/value projections calibrated with GPTQ (INT4); the routed experts
are unchanged. That brings the model about twice as close to BF16 (see
[Quality](#quality)) and costs decode speed: about 5 % with 8 conversations, about 4 % single-stream on the club-3090 bench and up to
9 % at one request in the concurrency test (the DeltaNet projections are read on every step, 0.35 GB more per card);
prefill speed is unchanged. The download is about 1 GB larger. The engine reads the format from the files, so the previous weights
still work as they are (QEngine 1.3.4 and older stop at load on the new files with `tensor not found:
...linear_attn.in_proj_qkv.weight_packed`: update QEngine, or download `--revision v1.3.4`):

```bash
hf download bpx03/Qwen3.8-Flash-Next-QEngine --revision v1.3.4 --local-dir models/Qwen3.8-Flash-Next-QE-INT4-v1.3.4
server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4-v1.3.4       # Docker: -e QE_MODEL=/app/models/...-v1.3.4
```

**What runs in which precision** (1.3.4.1 default weights; the files store exactly these formats except where noted):

| part | precision | note |
|---|---|---|
| routed experts (512 per layer) | INT4, group 128 | Intel's AutoRound quantization, repacked bit-exact |
| DeltaNet projections (in_proj_qkv / z / out_proj) | INT8, group 128 | 1.3.4 weights: INT4 |
| shared expert, n-gram key / value projections | INT4, group 128 | calibrated with GPTQ (1.3.4 weights: round-to-nearest) |
| attention q / k / v / o, lm_head, router, hyper-connection mix matrices | INT8, group 128 | BF16 in the files, converted at load |
| embeddings, n-gram table | BF16 / INT8 | the n-gram table ships as INT8 |
| norms, gates, DeltaNet a / b and conv, indexer | BF16 | |
| activations, accumulation | BF16 / FP16, FP32 accumulation | prefill GEMMs use INT8 activations |
| attention KV cache | FP8 (E4M3) | 4-bit opt-in: `FLASH_KV_BITS=4` |
| MTP draft head | INT8, group 128 (experts and shared expert INT4) | attention and fc BF16 in the files, converted at load |

**Context up to 1M tokens (opt-in)**: `FLASH_KV=ram server/run.sh 8080 --model <dir> --max-ctx 1M` turns on the
model card's YaRN RoPE scaling (Qwen3.8-27B: `--max-ctx 1M` on 3 cards, `--max-ctx 512K` on 2, no `FLASH_KV`). It applies to every request of that server, so use it only on a server meant for long
documents; see [docs/advanced.md](docs/advanced.md#context-up-to-1m-tokens-yarn-opt-in).

**KV cache in host RAM (opt-in)**: by default the attention KV of all conversations shares one VRAM pool of 262,144
tokens. With `FLASH_KV=ram` it lives in pinned host RAM (`FLASH_KV_RAM_TOKENS`, default `1M`, 13,520 bytes per token
for the whole machine) and VRAM keeps the indexer keys and a cache of recent blocks; output is bit-identical. Many
long conversations then stay resident: with 4 conversations of ~64K tokens sending follow-ups at once, output went
from 455 to 580 tok/s on 3 cards and from 233 to 332 tok/s on 2 cards (October 30, 2026). It is not the default
because short-request throughput is about 12 % lower (the indexer keys and block cache take VRAM from the experts).
Details: [docs/advanced.md](docs/advanced.md#kv-cache-in-host-ram-flash-next).

### Qwen3.8-27B

- INT4 weights repacked at load (a few seconds, no conversion) into tensor-core layouts.
- Verify head: an INT4 copy of `lm_head` screens the vocabulary, and each row's top 16 candidates (top 64 for sampled
  rows and logprobs) are rescored exactly from the BF16 rows in pinned host RAM, so greedy output takes the BF16
  argmax. It needs 2.5 GB of pinned host RAM on one card (0.85 GB per card on 3) and 0.64 GB less VRAM per card than
  an INT8 head. Without that host RAM, or with `QE_HEAD4R=0`, every card uses the INT8 head.
- Tensor parallel over PCIe; the four full-attention KV heads are split by position (3 cards with P2P), so no KV head
  is duplicated. All-reduces run inside the kernels and inside CUDA graphs.
- Speculative decoding, verified over the full vocabulary, so outputs are unchanged: the [DFlash2 block
  drafter](#dflash2-block-drafter-27b-on-when-it-fits) (7 drafts per step) for a conversation that decodes alone,
  when its memory fits; MTP with 1 to 7 drafts per step, chosen from the measured acceptance, for everything else.
  Lookup drafts on the GPU.
- Up to 32 conversations (`QE_MAX_SEQS`, default 16; more open requests are parked, see
  [Serving behaviour](#serving-behaviour)); KV cache in FP8 in one pool per card (1,232,896 tokens on 3 cards with P2P
  and 16 slots, 1,075,200 with 32; `QE_KV_BITS=4` stores it in 4 bits for small cards, see [GPUs](#gpus)); 262,144
  tokens per conversation.
- Prompt-cache states (snapshots of the recurrent state, about 150 MB of pinned host RAM each, split over the cards):
  up to 8 per sequence slot while they take at most 8 % of the host's RAM (or of a container's memory limit), at least
  4 per slot and at least 32 (`QE_SNAPSHOTS`); with 16 slots on a 256 GB host that is 128 states, about 19 GB. The
  states within the 8 % are pinned at start while the host has room, before the first request (`QE_ARCHIVE_WAIT_S`);
  the rest only as the store fills. With 32 for every slot count, 8 agent
  conversations on 32 slots dropped each other's states and re-prefilled from the system prompt; with 4 per slot, 16
  conversations on 16 slots still did. The state at the end of each
  prompt is copied to the host beside the first decode step instead of before it (about 6 ms less to the first token
  and per reply on one card, 2 ms on three; 16 MB of VRAM per card for the staged convolution rings;
  `QE_SNAP_ASYNC=0` copies it before the first step, as before).

**Release files.** Two files are attached to every [release](https://github.com/bpx03/QEngine/releases/latest):
`model_mtp_ft.safetensors` (849 MB, an MTP head fine-tuned on the engine's own answers) and `token_rank.bin` (the
draft-vocabulary ranking). The server downloads them on the first start into `~/.cache/qengine/assets`, checks them
against the release's `SHA256SUMS` and reuses them offline (the model directory is never written). Without them the
engine runs, just slower. Offline machines: put them into the model directory by hand:

```bash
cd models/Qwen3.8-27B-INT4
for f in model_mtp_ft.safetensors token_rank.bin SHA256SUMS; do
  curl -LO https://github.com/bpx03/QEngine/releases/latest/download/$f
done
sha256sum -c SHA256SUMS
```

`QE_FETCH_ASSETS=0` turns the download off; the start waits for it at most `QE_FETCH_WAIT_S` (120) seconds.

#### DFlash2 block drafter (27B, on when it fits)

Since 1.3.5.8 the 27B can draft with **DFlash2**, a small block drafter (the W4A16 build
[syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), Apache-2.0). It reads the
hidden states of five layers of the model and proposes 7 tokens per step; the model verifies them like MTP drafts, so
greedy output is identical to MTP, and sampled output follows the same distribution (only the random draws differ).

**Where it helps.** It is used for a conversation that decodes alone: one chat, one agent, one long answer. Measured
on this hardware (1.3.5.8; server, one conversation, against the MTP default of the same build):

| setup | short code | short prose | short math | 131K context |
|---|---:|---:|---:|---:|
| 1 card (`QE_DRAFTER=dflash2`) | +38 % | +17 % | +22 % | +10.5 % |
| 3 cards, P2P | +34 % (321 → 432 tok/s) | +6 % (230 → 244) | +15 % (297 → 341) | – |

The gain depends on the text: code and other predictable output gain most; long reasoning that MTP and lookup
drafts already predict well gains little (a 32K prompt with a long thinking part: +2 %).

Second choices on one card: when the drafter is unsure about its first three drafts (greedy replies), the step
verifies four drafts plus the drafter's second choice for each of the first three positions in the same 8 rows; a
right second choice keeps one more token. Output is identical. Measured on one card (engine, same build on / off):
+3.5 % on a 30-prompt mix, +4.6..4.9 % on the three short benchmark prompts, +2.7 % / +1.6 % at 32K / 96K, about level
on file edits. `QE_DF_SIB=0` turns it off.

Two conversations at once on one card: when exactly two conversations decode in the same step, each drafts with its
own DFlash2 context (8 rows each); output stays identical to each conversation decoding alone. Measured on one card
(server, two 512-token replies at once): +2.8 % for the pair as a whole; +8 % per step while both run (engine benchmark). With three or more
conversations in a step, and on 2 or 3 cards, batched steps draft with MTP as before (three at once on one card was
level to slightly slower, four 5 % slower). `QE_DF_BATCH=0` turns it off; `QE_DF_BATCH=n` allows it up to n
conversations per step. Grammar-constrained replies (`response_format`, a forced tool call until its header, `logprobs`) always keep MTP; a `tool_choice: none` reply drafts as usual. Agent turns that edit a file already in the context come out about 3 % slower,
because the model copies text the drafter no longer sees (it keeps the last ~2K positions); turns that write new text
are faster.

**What it costs.** VRAM on every card, plus 0.25 GB of pinned host RAM:

| cards | VRAM per card | KV pool (all conversations together) | context per conversation |
|---|---:|---|---|
| 1 | 1.20 GB (three context rings for the two-conversation steps; `QE_DF_BATCH=0`: 1.16 GB) | about −38K tokens | `QE_MAX_SEQS=1`: **206,848 → about 165,000** (default `QE_RESERVE_GB=1.5`) |
| 2 | 0.82 GB | 675,840 → 623,616 tokens | 262,144, unchanged |
| 3 | 0.70 GB | 1,298,432 → 1,232,896 tokens | 262,144, unchanged |

On 2 and 3 cards the default context still fits. **On one card it does not**, which is why a single card stays on MTP
by default.

**When it is on (`QE_DRAFTER`):**

- `auto` (default): at start the server loads DFlash2 only when the requested context (`--max-ctx`) and slots still
  fit with it, and prints one line saying which drafter it loaded and why (memory needed against memory free). The
  context is never lowered to make room for it. On 2 and 3 cards it is on with the defaults. **On one card the
  default `--max-ctx 262144` does not fit, so a single card stays on MTP**, and the start-up line tells you what this
  machine needs, e.g. `For DFlash2: --max-ctx <= 165K or QE_DRAFTER=dflash2` (the number is computed from your free
  VRAM). Either ask for that context or force it:
  ```bash
  # one card: room for DFlash2 by asking for less context (use the number from your start-up line)
  QE_DEVICES=0 QE_MAX_SEQS=1 server/run.sh 8080 --model <dir> --max-ctx 150K
  # or force it: the context becomes whatever still fits (logged)
  QE_DEVICES=0 QE_MAX_SEQS=1 QE_DRAFTER=dflash2 server/run.sh 8080 --model <dir>
  ```
  With the default 16 slots one card has about 43K tokens per conversation and stays on MTP; DFlash2 on one card is
  meant for `QE_MAX_SEQS` 1 to 4.
- `dflash2`: always load it; the served context shrinks to what fits, with a start-up line.
- `mtp`: never load it (and nothing is downloaded); the previous behaviour, full context.

**Files.** On the first start the server downloads the drafter (1.28 GB, a pinned revision checked against its
sha256) into `~/.cache/qengine/dflash2` (Docker: the `qengine-cache` volume) and reuses it offline. Without network
access and without the file the server starts on MTP and says so in one line; it never stops for it.
`QE_FETCH_ASSETS=0` turns the download off; `QE_DFLASH_URL` points it at a mirror. A local copy: `QE_DFLASH=<dir or
file>`, or a `dflash2/` directory inside the model directory.

#### INT8 weights (27B, opt-in, experimental)

INT4 stays the default. Pointing the server at an 8-bit checkpoint runs the INT8 layer weights: closer to the BF16 model
and slower, because a decode step reads twice the weight bytes. Three cards are recommended; one card can't hold them.
The weights are at [bpx03/Qwen3.8-27B-INT8-QEngine](https://huggingface.co/bpx03/Qwen3.8-27B-INT8-QEngine).

```bash
hf download bpx03/Qwen3.8-27B-INT8-QEngine --local-dir models/Qwen3.8-27B-INT8
server/run.sh 8080 --model models/Qwen3.8-27B-INT8     # Docker: -e QE_MODEL=/app/models/Qwen3.8-27B-INT8
```

To build one from the official BF16 weights (no GPU, no calibration data; Python packages in `requirements-ref.txt`):

```bash
hf download Qwen/Qwen3.8-27B --local-dir models/Qwen3.8-27B-BF16
python3 tools/quant_int8.py --src models/Qwen3.8-27B-BF16 --out models/Qwen3.8-27B-INT8 \
    --kv-from models/Qwen3.8-27B-INT4     # copies the FP8 KV scales, which keep the full KV pool
```

INT8 KL and top-1 are from 1.3.5.2. Everything else, the INT4 KL and top-1 included, is from November 1, on older code, and not
rerun since; 3 cards:

| metric | INT8 | INT4 | INT8 / INT4 |
|---|---:|---:|---:|
| KL vs BF16, mean (320 positions) | 6.66e-4 | 1.47e-2 | 22× closer |
| top-1 match with BF16 | 99.7 % | 95.3 % | |
| decode narrative / code, tok/s | 184.6 / 276.7 | 266.6 / 387.7 | 0.69 / 0.71 |
| prefill 10K / 90K, tok/s | 4,816 / 4,098 | 6,077 / 4,866 | 0.79 / 0.84 |
| agent c=8, output tok/s | 810 | 1,215 | 0.67 |
| KV pool, 32 slots, 262K context, tokens | 566,272 | 932,864 | 0.61 |

The 8-pack has not run on INT8, so whether the lower KL gives better answers in real tasks is not known yet.

**Your own checkpoint** goes in through `--model`. The engine reads one `model.safetensors` (not shards),
compressed-tensors pack-quantized, symmetric, group 128: `weight_packed` int32 `[N][K/4]` with bytes LSB-first and the
stored value `w + 128`, `weight_scale` bf16 `[N][K/128]`, tensor names as in the 27B INT4 checkpoint. A format it would
misread stops the start with the reason. Without FP8 KV scales it runs a bf16 KV cache with half the pool and no position
split (`QE_KV_SCALES=<a 27B INT4 model.safetensors>` borrows the scales; this fallback has not run on cards). INT8
checkpoints use their own `model_mtp.safetensors`, not the fine-tuned head (`QE_INT8_MTP_FT=1` uses it). Not tried:
`lued/Qwen3.8-27B-INT8-W8A16-MTP`; it is sharded, so the start refuses it as it is.

#### Long documents (27B, opt-in)

Above 262K tokens the 27B needs YaRN (`--max-ctx 512K` or `1M`), and for prompts that long most of the time goes to
attention; `--lsa` makes that attention sparse (the index map is downloaded once and cached, see the docs). Both are off by default. YaRN is static: it applies to every
request of that server, short ones too, so run a separate server for long documents. What `--lsa` changes
(3× RTX 3090, measured November 3, 2026; details in [docs/advanced.md](docs/advanced.md#sparse-attention-for-long-prompts-opt-in)):

| you gain | you give up |
|---|---|
| 512K prompt about 2× faster (225 → 103–114 s), 1M about 3× (736 → 239 s) | exactness: each token reads 2,048–4,096 of its keys, which carry about 60–80 % of the full attention weight in my measurement |
| decode at 1M about 2× (106 → 227 tok/s), at 512K 1.5× | tested only on 2 retrieval tasks at 128K and 512K plus one 1M needle (same answers as full attention): not RULER, not reasoning quality over long documents |
| nothing changes below a prompt position of ~64K or a context of ~192K (full attention runs there) | only one conversation at a time benefits (the last long prompt's); others, and requests on a pinned shared system prompt, use full attention |
| | about 0.5 GB per card less KV pool; needs P2P and the FP8 KV cache; at most 1.87M tokens |

### Both models: context and the KV archive

- `--max-ctx` (default 262,144) is the context of one conversation; the KV pool of all conversations together is
  sized from free VRAM. The served context is never larger than the pool: when the pool is smaller than `--max-ctx`,
  the server still starts, serves at most the pool's length (`max_model_len` in `/v1/models`, the length check) and
  says so in one start-up line (27B on 2 cards with `--max-ctx 1M`): `context: requested 1048576 tokens, KV pool 659456 tokens, served 659456 tokens per conversation`.
- Conversations that leave VRAM go to a **host-RAM archive** (`FLASH_KV_ARCHIVE_GB` 48, `QE_KV_ARCHIVE_GB` 32; smaller
  on machines with less RAM, see [Host memory](#host-memory-flash-next)): returning to a 50K-token conversation is a
  copy, not a recompute.
- **Restore points at message starts** (Flash-Next): the model carries a recurrent
  state that cannot be cut back to an arbitrary token, so a conversation resumes only where a state was saved. Each
  prompt now also keeps the state at the start of its last two messages. An agent client that appends a short
  message to every request and drops it on the next one (a runtime-context or reminder message), or that drops the
  reasoning of its last reply, resumes at that message instead of the end of the system prompt: on a 125K-token
  Flash-Next conversation 15.5-16.3 s -> 0.4-0.6 s per turn (from the second such turn; the first pays the one-time
  archive copy). Inkling-2-Mini gets the same in a later release. No state is planned when the last
  message starts within 512 tokens of the point the prompt resumed from, so an append-only agent loop is unchanged;
  each state takes about 0.12 GB of host RAM. A changed tool list still recomputes the whole prompt (the chat template
  puts the tools at its start). Flash-Next on 2 cards (more than 10 GB of host-backed
  experts per card, so every prefill batch copies them again) keeps these states off unless `FLASH_CKPT_TAIL_SNAP=1`
  (a tail turn of a 67K conversation 10.3 s -> 0.95 s, a cold multi-message prompt about 0.6 s slower). `FLASH_CKPT_TAIL=0`: off.
- **Pinned system prompts**: the KV and state at the end of each agent kind's system prompt and tools are kept apart
  from the archive (`FLASH_PREFIX_GB` / `QE_PREFIX_GB`, 8 GB), so a new agent of that kind starts after its system
  prompt even when the archive is full.
- There is no disk cache by default. With `--disk-cache DIR` (both models, 200 GB at most, `--disk-cache-gb`) conversations
  are also written to files and survive a restart: a conversation that returns is read back instead of recomputed. Flash-Next
  has it since 1.3.5.3, experimental; details in
  [docs/advanced.md](docs/advanced.md#disk-cache). A failing disk costs only the cache; the server keeps serving. Files are
  written at `--persist-idle` (120 s idle) and at shutdown, so this covers a graceful restart; the opt-in `--disk-eager` also
  writes after every reply (survives a `kill -9`, at the cost of rewriting the file every turn). On the 27B a conversation
  that shares a pinned system prompt is saved together with it, so each file restores on its own.

### DeepSeek-V4.1-Flash (preview)

DeepSeek-V4.1-Flash-1017: 40 layers, 384 routed experts per layer plus a shared expert, Engram n-gram memory tables and
compressed sparse attention. QEngine runs it on three 24 GB cards plus host RAM, with the DSpark draft model for
speculative decoding. It is a preview in 1.6: it runs in the same server and API as the other models, with the
limits listed below.

**Model files.** I do not ship weights for this model. QEngine runs from the **Maya-S GGUF files** on Hugging Face
([peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF),
revision `c33d87b8`), read as they are: the routed experts (IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S), the dense weights
(Q8_0), the Engram tables (Q6_K), the tokenizer and the chat format from the GGUF metadata, and the DSpark drafter from
the `dspark/` folder next to `Maya-S/`. Nothing is converted on disk and the model directory is never written. About
348 GB with the drafter:

    hf download peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF --revision c33d87b8 \
      --include "Maya-S/*" "dspark/*" --local-dir models/DeepSeek-V4.1-Flash-1017-GGUF
    server/run.sh 8080 --model models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-S

The server recognises the model from the GGUF metadata (architecture `deepseek41`) and loads the DSV4 engine
(`build_dsv4/libqdsv4.so`, built by `src/dsv4/build.sh`; the Docker image has it). Other revisions are not tested: the
start checks the model's dimensions and stops with the reason if they differ. The first start writes a few derived
files (tokenizer, the drafter's experts re-packed for the GPU, Engram hash constants) into
`~/.cache/qengine/dsv4/<model id>` (`--cache-dir` moves it; Docker: the `qengine-cache` volume), so the first start takes longer. After that the server
is ready about 41 s after start with the files in the page cache.

**What it needs**

- **Exactly three NVIDIA cards with 24 GB each**, compute capability 8.6 or 8.9 (RTX 3090 / 3090 Ti, RTX 4090; tuned
  on the 3090). The DSV4 engine is built for sm_86 only, which runs on RTX 4000 (sm_89) cards as is; RTX 5000 cards
  (sm_120) are not supported in 1.6. The engine uses the first three visible cards (`QE_DEVICES`); a fourth is
  ignored, two are not enough. The expert pool on each card is sized from its free VRAM, so a card that drives a
  desktop works, with fewer experts in VRAM.
- **GPU P2P between every pair of cards is recommended, not required.** With it, the context reservation is lazy:
  `--max-ctx` defaults to 262,144 and short chats run as fast as at 32K, because the long-context memory is taken from
  the expert pool only while a long prompt needs it and given back after. Without it (a stock driver, or
  `DSV4_NOCTXLAZY=1`) the reservation is static: `--max-ctx` still defaults to 262,144, but its memory is held for the
  whole run, so fewer experts fit in VRAM (on my cards without P2P, decode 2-3 % slower at short context than with
  `--max-ctx 32768`). When that context does not
  leave 8 GiB free on the fullest card, the start halves it (down to 32,768) with a WARNING line. Without P2P the cards
  also exchange through pinned host memory. A start-up line says which reservation runs. An explicit `--max-ctx`
  always wins.
- **Host RAM: about 170 GB free** for the full setup (on my machine `MemAvailable` drops by about 165 GB while it
  serves a 233K-token conversation). With less, the engine keeps fewer host copies of the experts and reads the rarest
  ones from the GGUF files; I tested that under a RAM hog taken while it serves, not on a machine with less RAM. It never allocates past `MemAvailable` minus
  a safety margin: it serves slower instead of running into the out-of-memory killer, and gives memory back while
  serving when the host runs short.
- **An NVMe drive for the model files.** The two Engram shards (5 and 6, about 161 GB) stay on disk and are read row by
  row while a prompt is processed. A hard disk or a network share is far too slow for this.
- **A CPU with AVX-512 is recommended; AVX2 is the minimum** (x86-64-v3: AVX2, FMA, F16C, BMI2). The experts that are
  not in VRAM run on the CPU. The engine picks its CPU kernels at start: the AVX-512 kernels need the Zen 4 set
  (AVX-512 with VNNI, VBMI, BITALG, BF16: AMD Zen 4 / Zen 5, Intel Sapphire Rapids and later); other CPUs get the AVX2
  kernels, with the same output but slower decode. The start-up line
  `CPU lane: N threads, AVX-512 kernels` shows which. The CPU part uses 24 threads by default
  (`--engine-args "--cpu-threads N"`).
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`. Responses carry `timings` (prompt and
  decode tokens and seconds).
- **Thinking** on by default (`--think`, a request's `enable_thinking`). `reasoning_effort`: `low`, `medium` / `high`,
  `max` / `xhigh`, or a number 1-100 (default 75); `none` turns thinking off. The reasoning comes back in
  `reasoning_content` (OpenAI) or thinking blocks (Anthropic). The thinking budget (`--thinking-budget`, a request's
  `thinking.budget_tokens` / `thinking_budget`) closes the reasoning at its count.
- **Tool calls** in the OpenAI and Anthropic formats (the model's own DSML call blocks are parsed into tool calls),
  `tool_choice` (`auto`, `none`, `required`, a named tool) and **structured output** (`response_format` `json_object`
  / `json_schema`).
- **Several requests at once** (agents): up to `--max-active 8` are in service, more wait in the queue. They take turns
  on the GPU: a long prompt in chunks of 10,240 tokens, an answer in slices of 16 decode passes, so a short request does
  not wait behind a long prompt or a long answer. The engine decodes one conversation at a time, so the total speed
  stays that of one request; a switch between conversations is exact (each one's text is the same as served alone with
  the same expert state).
- **Shared system prompts** (agent teams): when a second conversation starts with the same system message (tools
  included, at least 1,024 tokens) as an earlier one, that part is kept once, and every later conversation with it
  starts from the kept copy and computes only its own part. `--no-prefix-share` turns it off.
- **Multi-turn reuse and a host-RAM archive**: a follow-up turn reuses the cached prompt, also with thinking on;
  regenerating a reply reuses it too. Idle conversations move to host RAM (`--archive-gib`, default 32 GiB, least
  recently used out first) and come back with a copy instead of a recompute. `cache_prompt: false` in a request forces
  a full prefill. `usage.prompt_tokens_details.cached_tokens` shows what was reused.
- **Stop strings** (`stop`), matched in the answer only, never inside the reasoning.
- **Sampling**: `temperature`, `top_p`, `top_k`, `min_p`, `seed`; defaults from the GGUF metadata (temperature 1.0,
  top_p 1.0) unless set with `--temperature` / `--top-p`. Speculative decoding with DSpark keeps the sampled
  distribution exact.
- **Monitoring**: `qetop` and the dashboard on the metrics port (`/api/stats` has a DSV4 panel: held conversation,
  archive, K/V pool, prefix entries, passes). A conversation that has to be computed again because it left the archive
  shows up as a refill, counted with the prefill.

**Known limits (preview)**

- Text only: no images. No logprobs, no presence / frequency penalty (the DSV4 sampler has none), no activation
  steering (a request that asks for it gets "not supported").
- **Constrained replies**: structured output and `tool_choice` `required` / a named tool decode greedy replies
  (temperature 0) on DSpark passes, each pass checked against the grammar (the same text as one masked step per
  token, about 1.6x faster); sampled constrained replies decode one token per step without drafts, so they are slower.
  `QE_DSV4_GRAM_PASS=0` turns the passes off.
- **One conversation decodes at a time.** Several requests share the GPU in turns, so the total output speed with 8
  agents is about that of one (on my cards 66 tok/s for 4 agents with unrelated prompts, 98 tok/s for 8 agents that
  share a system prompt), and each agent gets its share.
- **Greedy text depends on more than the prompt.** Which experts sit in VRAM changes as the model runs, and the GPU and
  CPU paths round differently, so the same greedy request can give a different text after unrelated requests, with
  another amount of free VRAM (a desktop on a card), with another `--cpu-threads`, with another `--max-ctx` and when a
  prompt is fed in chunks (beside other requests, or after a shared system prompt). The last three come from the
  prefill's CPU expert split: `DSV4_NOCPUPF=1` removes them at a prefill cost. `--tier-reset` makes repeats of
  a request give the same text (see Settings).
- Editing an earlier message re-processes the prompt from that point (reuse works when the new prompt extends an old
  one or a kept system prompt).
- Quality is that of the Maya-S quantization (2- to 3-bit experts, Q8_0 dense weights, Q6_K Engram), not of the
  original checkpoint.
- Decode speed at long context (256-token replies, greedy, on my cards): 103 tok/s with 117K tokens of context and 91
  tok/s with 233K, against 76-106 tok/s for three prompts at short context.

### GLM-5.3-Flash (preview)

Z.ai's GLM-5.3-Flash: 45 layers, 288 routed experts per layer (8 active per token) plus a shared expert, attention
layers with a sparse indexer for long prompts next to KDA linear-attention layers, and one MTP layer. QEngine runs it
on three 24 GB cards plus host RAM, with its own engine (`src/glm`): the experts used most often sit in VRAM, the full
set in pinned host RAM, and the rest of a step's experts run on the CPU next to the cards. It is a preview in 1.6: it
runs in the same server and API as the other models, with the limits listed below.

**Model files.** My INT4 quant of GLM-5.3-Flash,
[bpx03/GLM-5.3-Flash-INT4-QEngine](https://huggingface.co/bpx03/GLM-5.3-Flash-INT4-QEngine)
(routed experts and KDA layers in INT4; lossy, see the limits). **The GLM engine runs only from this checkpoint**: it
holds my GPTQ INT4 experts and KDA layers in QEngine's own layout, so other GLM-5.3-Flash quants, GGUF files and the
original weights do not load. About 186 GB:

    hf download bpx03/GLM-5.3-Flash-INT4-QEngine --local-dir models/GLM-5.3-Flash-INT4-QEngine
    server/run.sh 8080 --model models/GLM-5.3-Flash-INT4-QEngine

The server recognises the model from `config.json` (`model_type` `glm5_next`) and loads the GLM engine
(`build/libqglm.so`, built by the normal CMake build when `src/glm` is in the tree; the Docker image has it). The MTP
draft vocabulary and the expert prior (`tools/glm/draft_vocab_49k.i32`, `tools/glm/expert_prior_gen.f32`) are read
from next to the library (`<root>/build/libqglm.so` -> `<root>/tools/glm/`); without them the engine still runs, with
the full drafter head and an empty expert cache at start (a start-up line says so). The chat template is the model's
own `chat_template.jinja`. Loading takes about 60 s with the files in the page cache.

**What it needs**

- **Three NVIDIA cards with 24 GB each** (tuned and tested on RTX 3090s). Two cards also work, with fewer expert slots
  per card (slower decode). With four or more cards the engine uses the first three.
- **GPU P2P is optional.** Without it (the stock GeForce driver) the all-reduces go through host memory and the
  prefill indexer runs on every card instead of split across them (a 32K prompt prefills about 10 % slower than with
  P2P; 2K-8K prompts and decode are about level). A start-up line says which path runs; `GLM_P2P=0` forces the host
  path.
- **About 192 GB of host RAM**: the experts (about 157 GB) stay pinned in host RAM, plus the conversation archive
  (below, up to 32 GB by default). Less RAM means a smaller archive; the experts themselves have to fit.
- **An x86-64 CPU with AVX2.** The CPU expert lane is built twice and picks its kernels at start: AVX-512 when the
  CPU has the full Zen 4 set (with VNNI and BF16: Zen 4, Sapphire Rapids and later), else AVX2 (`GLM_CPU_ISA`
  forces one). The lane's thread placement is derived from the CPU topology at start (one log line).
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`.
- **Tool calls** in the GLM format (`<tool_call>` with `<arg_key>` / `<arg_value>`), returned in the OpenAI and
  Anthropic formats; `tool_choice` `auto`, `none`, `required` and a named tool; **structured output**
  (`response_format` `json_schema` / `json_object`).
- **Reasoning** on or off (`enable_thinking`, Anthropic thinking), `reasoning_effort` as the model's template takes it
  (`low`, `high`, `max`); the thinking budget (`--thinking-budget`, a request's budget) closes the reasoning at its
  count.
- **logprobs** / `top_logprobs`, **stop strings** (`stop`) with an exact rollback of the tokens past the match.
- **Prompt cache**: a follow-up turn reuses the prefix it shares with earlier turns; idle conversations go to a
  conversation archive in pinned host RAM and come back with a copy instead of a new prefill.
- **MTP speculative decoding** with the model's own MTP layer.
- **Several conversations at once** (8 by default, `GLM_CONVS`): replies take turns on the GPUs. A switch parks the
  running reply between two decode steps (its state, and its expert cache contents, saved to host RAM) and resumes the
  other one exactly where it stopped, so each reply is bit-identical to the same reply served alone. A new prompt starts
  once the running reply has had its turn (`GLM_PREEMPT_MS`, by default the slice length), so short replies end
  without a switch; long replies are sliced (`GLM_SLICE_MS` 2000, at least 8x the measured cost of a switch: a park,
  ~0.06 s, plus the later resume, ~0.3-0.6 s, so about 4-5 s per turn). With 8 agents (agent replay: 2-16K-token
  prompts, 64-token replies) the first token comes in 13.2 s p50 instead of 28.3 s one at a time, and the total output
  is 22.4 instead of 12.6 tok/s; 4 clients with 256-token replies get the same total output as one at a time (70.5 vs
  70.7 tok/s); one conversation alone runs as before.
- **Long prompts return between layers**: the prefill runs in host slices of about one second, so the server keeps
  reporting progress; a cancel (or a client that disconnects) stops the prompt at the next layer, within about half a
  second, and the cache goes back exactly to the start of the interrupted prefill window.

**Known limits (preview)**

- **One reply decodes at a time**: several conversations share the GPUs in turns (see above), not in one batch, so the
  total output speed does not grow with the number of conversations; a switch between two long replies costs ~0.3 s
  (the expert cache contents of the reply that resumes are copied back).
- Parked replies and the snapshots of up to 3 x `GLM_CONVS` conversations use pinned host RAM (about 150 MB each, plus
  the archived attention rows); under host-RAM pressure a parked reply whose rows were given back is prefilled again
  (logged; that reply is then no longer bit-identical to a run alone).
- Greedy output is deterministic for the same request on the same machine and settings, but it depends on the number
  of expert slots (free VRAM, the context setting, the number of cards) and on how the prompt was split across
  prefill windows: a prompt resumed from the cache can differ in the last bits from one prefilled at once.
- Constrained replies (structured output, forced tool calls) and logprobs are slower: each step reads the full logits
  on the host.
- The default checkpoint is a lossy INT4 quant (routed experts and KDA layers).
- Text only (no image input), no presence / frequency penalties (accepted and ignored), no activation steering, no disk
  cache, no smaller K/V formats.

### gpt-oss-120B (preview)

OpenAI's gpt-oss-120B: 36 layers, 128 experts per layer (4 active per token), MXFP4 expert weights, grouped attention
with every second layer on a 128-token window, and the harmony chat format. QEngine runs it on three 24 GB cards, all of
it in VRAM, with its own engine (`src/gptoss`). It is a preview in 1.6: it runs in the same server and API as the other
models, with the limits listed below.

**Model files.** The official checkpoint as OpenAI publishes it, read as it is: the MXFP4 experts and the bf16 dense
weights, nothing converted, nothing of mine to download. I tested revision `b5c939de`; about 65 GB without the
`original/` and `metal/` folders:

    hf download openai/gpt-oss-120b --revision b5c939de --exclude "original/*" "metal/*" \
      --local-dir models/gpt-oss-120b
    server/run.sh 8080 --model models/gpt-oss-120b

The server recognises the model from `config.json` (`model_type` `gpt_oss`) and loads the gpt-oss engine
(`build_gptoss/libqgptoss.so`, built by `src/gptoss/build.sh`; the Docker image has it). Without that library the
start stops and says how to build it. The chat template is the model's own `chat_template.jinja`. The server is ready
about 11 s after start with the files in the page cache.

**What it needs**

- **Three NVIDIA cards with 24 GB each, compute capability 8.6** (RTX 3090 / 3090 Ti). The engine is built for sm_86
  only. RTX 4000 cards (8.9) run the same code untested, with the output head computed in full: its fast two-stage form
  has a rounding margin I measured on 3090s only, so a start-up line says the full head is used (slower decode: about 6 % for one conversation on my 3090s).
  Other cards (A100 8.0, RTX 5000 12.0, ...) are refused at start in 1.6. Two or four cards are accepted but not tested (two
  24 GB cards cannot hold the 65 GB of weights: the start stops with what each card would need); one card is refused.
- **GPU P2P between the cards is not needed.** With it the cards write into each other's memory. Without it (the
  stock GeForce driver) they exchange through pinned host memory, with the same output bit for bit: on my cards the
  host path reaches about 94 % of the P2P speed (2026-11-19, engine tool: decode 314 against 332 tok/s for one
  conversation, 951 against 1,009 tok/s for eight, a 10K-token prompt at 13.1K against 13.8K tok/s). A start-up line
  says which path runs; `GPTOSS_COMM=host` forces the host path. The host path takes the same VRAM as the P2P path and
  about 0.5 GB of pinned host RAM; `GPTOSS_PF_HOSTLANES=0` with `GPTOSS_LM=full` is the older, slower host path with
  about 0.5 GB more K/V room per card. When the host cannot pin that memory at start, the engine falls back by
  itself (the one-lane host form, a smaller prefill chunk, pageable row copies; one WARNING line each, slower prefill)
  and keeps serving.
- **Host RAM** only for the conversation archive (below); the model itself stays in VRAM.
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`.
- **Reasoning** always runs (the model has no off switch). `reasoning_effort` `low` / `medium` (default) / `high`;
  other APIs' names are mapped: `max` and `xhigh` to `high`, `minimal` and `none` to `low`, thinking off
  (`enable_thinking: false`) to `low`. The reasoning comes back in `reasoning_content` (OpenAI) or thinking blocks
  (Anthropic); the thinking budget (`--thinking-budget`, a request's budget) closes it at its count.
- **Tool calls** (the harmony `functions` namespace) in the OpenAI and Anthropic formats, `tool_choice` (`auto`,
  `none`, `required`, a named tool) and **structured output** (`response_format` `json_object` / `json_schema`). A
  history with several tool calls in one assistant turn (from another model) is rendered as one call and its result
  after the other, as harmony expects; several system / developer messages at the start are joined into one.
- **Several conversations at once**: 8 slots when the context per conversation is at least 32,768 tokens (the default
  131,072), else 4 (`GPTOSS_SLOTS`); they share one K/V pool (3x24 GB: 144,611 rows in bf16). Greedy and plain-temperature replies decode together in one step, and a reply's tokens are
  the same as when it runs alone. A reply that cannot grow because the pool is full ends with `finish_reason`
  `length`; a new prompt waits until the running replies have room to grow.
- **Prompt cache**: a follow-up turn reuses its conversation's K/V; a shared system prompt (tools included) is kept
  once for all conversations that start with it; idle conversations move to host RAM (`QE_KV_ARCHIVE_GB`, default
  128 GiB, never closer than `QE_KV_ARCHIVE_MARGIN_GB` = 8 GiB to `MemAvailable`; least recently used out first) and
  come back with a copy. `cache_prompt: false` forces a full prefill.
- **Lookup drafts** for greedy replies (`GPTOSS_LOOKUP`): tokens that repeat earlier text (edits, code) are checked
  in one multi-row step; the text is exactly the plain greedy text.
- **Sampling**: `temperature` (default 1.0, OpenAI's recommendation), `top_p` (default 1.0), `top_k`, `min_p`, `seed`.
  Plain temperature sampling decodes at greedy speed; requests with `top_p` < 1, `top_k` or `min_p` run the full
  output head one row per step (slower: about 6 % for one conversation on my cards, more with several at once).
- **Presence / frequency penalties** (`presence_penalty`, `frequency_penalty`, OpenAI's definition: a token's logit
  drops by `frequency_penalty` times its count in the reply so far, plus `presence_penalty` once it has appeared;
  before the temperature), per conversation, on the GPU and at the plain decode speed, also for several conversations
  at once. A greedy request with a penalty takes the highest penalised logit. A reply with a penalty runs without
  lookup drafts.
- **Stop strings** (`stop`), matched in the final answer only, never in the reasoning.
- **A device fault ends the process** (exit code 75) after the open requests get an error, so llama-swap or Docker
  restarts it instead of a server that answers `/health` and fails every request.

**Known limits (preview)**

- Text only: an image part is answered with `[image omitted: this server runs without vision]` in its place. No
  logprobs (HTTP 400). No activation
  steering, no disk cache (`--disk-cache` is ignored with a start-up line).
- One tool call per reply (the model's format); the built-in `browser` and `python` tools are not run by the server.
- Constrained replies (structured output, `tool_choice` `required` / named / `none`) decode one row per step.
- The DFlash drafter is not used by the server in 1.6 (`GPTOSS_DFLASH` is ignored with a start-up line).
- A greedy or seeded reply in a reused conversation can differ from the same prompt sent fresh where two tokens are
  nearly tied: rows computed by decode and by prefill differ in rounding. The same holds for a conversation that
  shares its beginning with another one the server still holds (the same system prompt, or just the harmony header):
  the shared rows are reused, so the rest of the prompt is computed in a call of another size, which rounds
  differently. The text is equally valid either way; requests that start from the same server state give the same
  reply.
- Context per conversation: 131,072 tokens (the model's own limit, `--max-ctx`); the pool is shared by all slots.

### Inkling-2-Mini (preview)

Thinking Machines' Inkling 2 Mini: 24 layers of which layers 6-17 run three times per token (48 layer applications), a
sparse mixture of experts (26.4B parameters, 3.4B active per token) with NVFP4 experts in layers 0-19 and MXFP8 in
20-23, sliding-window attention (1,024 tokens) on 40 of the 48 applications and full attention on 8, and its own chat
format. QEngine runs the text part with its own engine (`src/inkling`): on one 24 GB card, or split over two cards
with tensor parallelism. It is a preview in 1.6: same server and API as the other models, with the limits below.

**Model files.** The official checkpoint, read as it is (I tested revision `b85c3cfa`, about 19 GB), plus two files of
mine next to it in the same directory (about 1.3 GB):

    hf download thinkingmachines/Inkling-2-Mini --revision b85c3cfa --local-dir models/Inkling-2-Mini
    hf download bpx03/Inkling-2-Mini-QEngine dense8.safetensors mtp_qe.safetensors --local-dir models/Inkling-2-Mini
    server/run.sh 8080 --model models/Inkling-2-Mini                    # one card (GPU 0)
    QE_DEVICES=0,1 server/run.sh 8080 --model models/Inkling-2-Mini     # two cards, needs GPU P2P

- `dense8.safetensors` (0.65 GB): the dense attention weights in INT8 with one scale per 128 inputs, rounded with GPTQ
  (error feedback against activations of public text), so decode reads fewer bytes. Without it the engine uses the
  original bf16 weights (`g15`, see the table below) and says so at start.
- `mtp_qe.safetensors` (0.64 GB): the model's DSpark drafter, fine-tuned by me on the model's own outputs (same shapes,
  less over-confident, with its own stop policy). Without it the engine uses the official `mtp.safetensors` of the
  checkpoint (`INK_DRAFTER_FILE=mtp.safetensors` picks the official one when both are there). Drafts never change the
  text: every drafted token is checked against the model's own choice.

The server recognises the model from `config.json` (`model_type` `inkling2_loop`) and loads the Inkling engine
(`build_srv/libqinkling.so`, built by `OUT=build_srv src/inkling/build.sh`; the Docker image has it). Without that
library the start stops and says how to build it. The chat template is the model's own `chat_template.jinja`. The
server is ready about 12 s after start with the files in the page cache.

**What it needs**

- **One or two NVIDIA cards with 24 GB, compute capability 8.6** (RTX 3090 / 3090 Ti; the engine is built for sm_86
  only). One card by default; `QE_DEVICES=a,b` (or `--devices`) splits it over two cards, which needs GPU P2P between
  them: without peer access the start stops with that reason. Three cards work (an uneven head split) but are not part
  of the 1.6 checks.
- **Host RAM** only for the conversation archive (below); the model itself stays in VRAM.
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`.
- **Thinking**: the template's `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or a
  number from 0 to 0.99); thinking off (`enable_thinking: false`) is effort 0; without one the model's default 0.9.
  The reasoning comes back in `reasoning_content` (OpenAI) or thinking blocks (Anthropic).
- **Tool calls** in the OpenAI and Anthropic formats (the model's `content_invoke_tool_json` messages).
- **DSpark drafts** for greedy and plain-temperature replies: the drafter proposes up to 7 tokens, one multi-row step
  checks them, and the text is exactly the text without drafts.
- **Several conversations at once**: 8 slots on 2 cards, 2 on one card (`INK_SLOTS`, 1-16; when they don't fit the
  server serves fewer and says so). They share one K/V pool, and one conversation may use all of it: per conversation
  about 565K tokens on 2 cards and 126K on one card with the default weights. Replies decode together in one step,
  and a reply's tokens are the same as when it runs alone.
- **Prompt cache**: a follow-up turn reuses its conversation's K/V; idle conversations move to host RAM
  (`QE_KV_ARCHIVE_GB`, default 128 GiB, never closer than `QE_KV_ARCHIVE_MARGIN_GB` = 8 GiB to `MemAvailable`) and
  come back with a copy, also in the background after every finished reply (`INK_BG_REARM`); replies beyond the seats
  are parked in host memory (`--park`).
- **Sampling**: `temperature` (default 1.0, from the model's `generation_config.json`), `top_p`, `top_k`, `min_p`,
  `seed`, presence / frequency penalties. Temperature with `top_k` up to 64, `top_p` and `min_p` is sampled on the
  GPU with drafts on, at about the greedy speed: the engine finds the exact top logits, keeps the same set as the
  usual rule (top_k, then min_p, then the smallest top_p prefix) and draws from it with seeded noise, so drafts never
  change the text and a seed gives the same reply alone or in a batch. Without `top_k` the kept set has to end
  inside the 64 most likely tokens; a token where it doesn't (`top_p` near 1 on very flat text) is picked on the CPU
  with the same rule and seed, and a conversation where that happens often runs without drafts. `top_k` above 64
  and presence / frequency penalties are picked on the CPU without drafts. `INK_SAMP_DEV=0` puts all filters on the
  CPU.

**Known limits (preview)**

- Text only: an image part is answered with `[image omitted: this server runs without vision]` in its place.
- No structured output and no logprobs (HTTP 400); `tool_choice` `required` / named are not supported (`none` works:
  the reply ends at its first call); the thinking budget is ignored.
- Without GPU P2P only one card.

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
| Parked replies beyond the seats (`--park`) | ✅ | ✅ (3+ cards) | ❌ | ❌ | ❌ | ✅ |
| Drafts | MTP, DFlash2, lookup | MTP, n-gram, lookup | DSpark | MTP | prompt lookup | DSpark |
| Sparse attention for long prompts | ✅ LSA (opt-in) | built in | built in | built in | ❌ | ❌ (sliding window on most layers) |
| Cards | 1-4 | 2-4 TP, more in PP | 3 | 3 (2 with fewer expert slots) | 3 (2-4 accepted, untested) | 1 by default, 2-3 TP with `QE_DEVICES` (needs P2P) |
| Without GPU P2P | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 one card only |
| Docker image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

`n > 1`, `logit_bias` and `parallel_tool_calls: false` are ignored on every model.

Inkling-2-Mini with thinking off runs at reasoning effort `none`, and the model still writes a short thought (one or two
sentences) before most replies; the server returns it as reasoning (`reasoning_content` / a thinking block), never
mixed into the answer.

Inkling-2-Mini lets you pick the form of its dense attention weights (`--dense-weights`, or `INK_DENSE8=1` / `0`;
the flag wins) on any card count:

| form | what it is | decode step, 2 cards / one card | quality (set-2 KL vs the FP32 reference, decode form; BF16 bar 0.00592) | context per conversation, 2 cards / one card |
|---|---|---|---|---|
| `int8` (default) | GPTQ INT8, one scale per 128 inputs, from `dense8.safetensors` next to the checkpoint: faster, small quality loss | ~4.9 / ~7.1 ms | 0.00539 (one card 0.00538) | ~565K / ~126K |
| `g15` | original weights (lossless, bit-exact to the checkpoint, stored in a 12-bit form) | ~5.5 / ~8.0 ms | 0.00495 | ~544K / ~117K |

`int8` is the default on any card count. Prefill runs at about the same speed in both forms (`int8` within 1 %).
Without `dense8.safetensors` the default is `g15`, and the server logs one line about it. If you ask for `int8` and `dense8.safetensors` is missing, it falls back to RTN INT8 at load (lower quality,
KL 0.00589), and the server says so at start. A start-up line names the active form, the served context and the
switch, e.g. `inkling: dense attention weights int8 (...) on 2 cards, context per conversation 565216 tokens; switch:
--dense-weights g15 (INK_DENSE8=0) = original weights, lossless`.

Inkling-2-Mini serves 8 conversation slots on 2+ cards and 2 on one card. On one card each slot's fixed state
(~0.36 GB) comes out of the K/V pool, so 2 slots leave ~126K tokens per conversation (8 slots: ~62K; with the
lossless `g15` weights ~117K, 8 slots ~53K, 1 slot ~137K). `INK_SLOTS=N` (1-16) sets the count. When the requested slots don't fit, the server serves fewer
and says so in a start-up line (`requested 16 slots, serving 8`).

After every finished reply Inkling-2-Mini copies the idle conversation to host RAM in the background (`INK_BG_REARM=0`
= off-switch). A 109K-token conversation then comes back in about 0.3 s instead of a 12.7 s prefill, and a new long
prompt that needs its room does not wait for the copy.

At start the server logs one `[settings]` line for every flag or environment variable that is set but does nothing for
the loaded model, e.g. `FLASH_KV=ram is set but has no effect for gpt-oss-120B (KV stays in VRAM ...)`. A request
parameter the model doesn't support gets one `[api]` line the first time it shows up; the request is then served as
the table says (ignored, or HTTP 400).

## API

- **OpenAI**: `POST /v1/chat/completions` (streaming or not), `GET /v1/models`.
- **Anthropic**: `POST /v1/messages`, `POST /v1/messages/count_tokens`.
- `GET /health`, `GET /props` (llama.cpp style), `GET /v1/steering`.

`/v1/models` and `/props` report what a client needs to size its requests: `max_model_len` (prompt + reply tokens of
one conversation), `max_num_seqs` (conversations served at once), `max_prompt_tokens`, `max_output_tokens` (`null`:
no cap beyond the context), `kv_pool_tokens`, `kv_cache` (`vram` / `ram`) and `input_modalities`; `/props` adds
`total_slots` and `default_generation_settings.n_ctx`.

**Too-long prompts** get HTTP 400 before any streaming starts. The Anthropic path says
`prompt is too long: N tokens > M maximum` (Claude Code compacts the conversation on it). The OpenAI path uses
OpenAI's wording, `This model's maximum context length is M tokens. However, your messages resulted in N tokens ...`
with code `context_length_exceeded`, which harnesses such as verifiers / AutomationBench and LiteLLM recognise as a
context overflow.

**Tool calls**: OpenAI `tools` and Anthropic `tool_use` / `tool_result`. Tools are rendered sorted by name with
sorted JSON keys, so a client that reorders the same tools keeps the prompt cache (`--sort-tools 0`: the client's
order). The chat template is `server/templates/qwen38_syssafe.jinja` (the model's template, accepting several leading
system messages). `--template` replaces it. A reply starts as reasoning only if the template's generation prompt opens
`<think>`: with a custom template that does not, a model reasoning without writing the tag shows that text in `content`,
followed by a literal `</think>`. Since 1.3.6 a `<tool_call>` that does not parse as a call (prose inside the tag,
broken markup), or a call that `tool_choice` filters out, reaches the client as plain text in `content`, streamed or
not, on both APIs, instead of an empty reply.

**`tool_choice`** on both APIs: `auto` (default), `none` (Anthropic `{"type": "none"}`: the tools stay in the prompt, the
tool-call token is masked, so the model answers in text), `required` (Anthropic `any`) and a named function (OpenAI
`{"type": "function", "function": {"name": ...}}`, Anthropic `{"type": "tool", "name": ...}`): the answer starts with
that call after any thinking. Enforced in the engine: `none` bans the tool-call token in the normal decode path, so
such a reply keeps its batching and speculative drafts and runs as fast as `auto` (with `logprobs` it is masked on the
constrained path instead). A forced call runs through its thinking and the call header on the same masks as structured
output (on Flash-Next such a reply steps outside the batch until then). Calls to other tools than a named one are
dropped. Client stop sequences do not cut a forced reply before its call is complete; they
apply after it. Under `none` a call the model writes out as plain text is text and arrives in `content`. An
unknown name, or a forcing choice without tools or together with `response_format`, is a 400.

**Structured output**: `response_format` with `json_object` or `json_schema`, enforced in the engine (llguidance
grammars) together with speculative decoding. Stop sequences are supported.

**Sampling**: values a request doesn't send come from the Qwen3.8 model cards, per reply mode. Thinking: temperature 1.0,
top_p 0.95, top_k 20, min_p 0, presence penalty 0. Non-thinking: temperature 0.7, top_p 0.80, top_k 20, min_p 0, presence
penalty 1.5. `--temperature`, `--top-p`, `--top-k`, `--min-p` and `--presence-penalty` set one value for both modes, and a
value in the request wins. Greedy requests (temperature 0) get no penalty. The cards warn that a higher presence penalty may
occasionally mix languages and slightly lower quality: `--presence-penalty 0`, or `presence_penalty: 0` in a request, turns
it off. The speed cost of presence 1.5 was measured on Flash-Next (3 cards, 8 agents at once, November 15, 2026):
with a step's verify rows sampled one after another it took 17-20 % of the throughput; since 1.3.6 they are sampled
in parallel (`QE_SAMPLE_PAR`), which gave +16 % on that test (697 → 808 tok/s). Its effect on answer quality and
tool calls has not been measured here.
`presence_penalty` and `frequency_penalty` are applied on the GPU.

**`logprobs` / `top_logprobs`** on `/v1/chat/completions` (`top_logprobs` 0-20, streamed too; since 1.3.5.4, experimental). The numbers
are the target model's own next-token distribution: raw logits through log-softmax over the full vocabulary, before
temperature, top-p, top-k, min-p and penalties, never a draft head's. Steering, when used, is included. `content` covers
only the tokens of `content`, not reasoning or tool-call tokens; with `logprobs` the non-streamed `content` is not stripped,
so the entries join to it. Such a request decodes one token per step without draft speedup, about 3x slower, and while it
runs a normal reply beside it drops to about 0.4-0.45 of its speed alone. In `pp` the numbers carry the mode's own
rounding, so use `tp` for KL-grade measurements. On the 27B the top 64 values of each row are exact BF16 logits; the
normaliser also sums the INT4 screen's values for the rest of the vocabulary (log-prob error about 0.01 at most).
Prompt logprobs and `/v1/completions` are not available.

**Thinking and reasoning effort**: thinking is on by default (`--think 0` turns it off by default;
`chat_template_kwargs.enable_thinking` per request; a top-level `enable_thinking` is accepted too when
`chat_template_kwargs` does not set it, and a `null` counts as not sent). `reasoning_effort` (`xhigh` = the template default, `high` is the
same, `medium`, `low`) per request or as a server default with `--reasoning-effort` / `QE_REASONING_EFFORT`; a
request's value wins. `max` is taken as `xhigh` and `minimal` as `low` (the names other APIs use); other unknown
values render as `medium`. On `/v1/messages` the Anthropic `output_config.effort` is used when the request sends no
`reasoning_effort` (so `max` gets the most reasoning). The OpenAI Responses / OpenRouter form `reasoning: {"effort": ...}`
is read too, on both paths, after `chat_template_kwargs.reasoning_effort` and the top-level `reasoning_effort`.
`none` (any case, in any of these fields) turns thinking off,
the same as `enable_thinking: false`, also when the same request sets `enable_thinking: true` (an "off" always wins).
`QE_REASONING_EFFORT=none` makes thinking off the default; a request that names another effort, sets
`enable_thinking: true` or the Anthropic `thinking.type` `enabled` / `adaptive` still thinks.
The OpenRouter switches are read last: `reasoning: {"enabled": false}` is thinking off when the request sends no other
thinking setting or effort; `"enabled": true` turns thinking on over the server default (`--think 0`,
`--reasoning-effort none`) at the effort the request names, unless the same request turns thinking off.
`reasoning: {"exclude": true}` on `/v1/chat/completions`: the model reasons as usual (the reasoning tokens are in
`usage`), but the reply carries no `reasoning_content`, streamed or not.

### Thinking budget and the answer tail

Effort only asks the model to think less. A model can still reason until `max_tokens` runs out and return a turn
with reasoning only (no answer, no tool call), which an agent takes as the end of its task and a benchmark scores as
zero. A budget stops that:

- per request: Anthropic `thinking: {"type": "enabled", "budget_tokens": N}`; OpenAI path `thinking_budget` (top
  level or in `chat_template_kwargs`; `thinking_token_budget`, `max_thinking_tokens`,
  `custom_params.thinking_budget` and `reasoning.max_tokens` are accepted too);
- server default for requests that send none: `--thinking-budget N` / `QE_THINKING_BUDGET=N` (0 = off), one budget
  for every request. A request's value wins; a negative one turns the budget off for that request; 0 turns thinking
  off.
- or a default per reasoning effort: `QE_THINKING_BUDGET=low:4096,medium:16384,xhigh:32768` (the same value works
  for `--thinking-budget`), optionally with `default:N` (or `*:N`) for efforts the map does not list. The effort is
  the one the prompt is rendered with (the request's, else `--reasoning-effort`, else the template's `xhigh`), named
  the way the template renders it: `high` and `max` use the `xhigh` entry, `minimal` the `low` entry, other unknown
  values the `medium` entry. An entry of 0 or below means no budget for that effort (as a plain 0 or negative
  number). Order: the request's budget, then its effort's entry, then `default:`, then none. With a custom
  `--template` whose efforts the server does not recognise, a warning is printed at start-up, efforts are matched as
  sent, and requests without one use the `xhigh` entry.
- since 1.3.4.3 a malformed value (e.g. `8k`, `1.5`, `low:4k`, an unknown effort) stops the server at start-up with an
  error naming the entry; 1.3.4.2 and earlier warned and ran without a budget.
- the start-up line prints the map, and `/api/stats` (`thinking.budget_from`) and the per-request log line
  (`think 66/4096 (effort:low)`) show where each budget came from: `request`, `effort:<level>`, `default`, `off` (the
  request turned the budget off) or `none`.

At the budget the server writes a one-sentence wrap-up and `</think>` (27 tokens; `QE_THINK_WRAP` changes the
sentence), and the model goes on with its answer or tool call. The reasoning holds exactly the budget plus the
wrap-up: near the budget (and near the answer tail) the server caps how many drafts a step may verify, so the close
lands at the same token whatever the drafts (before 1.3.6 it could come up to one decode step late, and the
text then depended on the drafter).

`max_tokens` counts thinking and answer together. **Answer tail**: when `max_tokens` would run out inside `<think>`,
the same close comes early enough to leave the answer the last eighth of `max_tokens` (64 to 2048 tokens;
`QE_THINK_TAIL`, a token count, `0` = off), with or without a budget (not with `ignore_eos`). `finish_reason` is `length` only when the answer
itself is cut off; `usage.completion_tokens_details.reasoning_tokens` reports the thinking tokens.
`usage.completion_tokens` (Anthropic `output_tokens`) counts the end-of-turn token the model generated, as vLLM
does: a reply of 7 text tokens that ends on its own reports 8. A reply cut by `max_tokens` or a stop string has no
end token, and neither has structured output that ends when its grammar is complete (the server ends that turn
itself). Before 1.3.6 the end token was never counted.

**Cost of the forced tokens.** They are verified like drafts, up to 4 per step on Flash-Next and up to 8 on the 27B,
together with MTP and lookup drafts. On the 27B a constrained request stays in the batched step with the others. On
Flash-Next a constrained step (a forced close, and structured output in general) runs as a step of its own, so with
several conversations decoding at once it adds steps the others wait for. A request without a budget that never
reaches the tail decodes exactly as before.

### Empty replies are sampled again

Now and then a thinking model ends its turn with reasoning only (end-of-turn sampled inside or right after
`<think>`) or with a tool call that does not parse. Agent harnesses (AutomationBench / verifiers and many OpenAI-API
agent loops) end the task on a turn without text and without a tool call. Such a non-streamed, sampled reply
(finish `stop`) is sampled again up to 2 times (`--empty-retry N` / `QE_EMPTY_RETRY`, `0` = off). The prompt is
cached, so a retry costs only its decode. Never retried: streams (the bytes are already sent), greedy (`temperature`
0) and seeded requests, structured output, `ignore_eos`, requests with stop sequences. `usage.completion_tokens`
counts every try; `finish_reason` is the last try's. A tool call that does not parse is retried the same way; when
the last try still does not parse, its text is returned as `content` (streams: at once), so the client never gets an
empty turn from it.

### Images

`--vision 1` (both models; Docker: the `vision` image) accepts images in OpenAI (`image_url`, data URLs or http(s))
and Anthropic (`image` blocks) messages. The vision encoder runs in PyTorch on the last GPU (Flash-Next: ~0.6 GB of
VRAM there, weights streamed from host RAM; 27B: ~1.3 GB). Image tokens take part in prefix caching, keyed on the image
content. A 1600×900 screenshot is encoded in about 0.14 s on Flash-Next. Unreadable images and video / audio parts get
HTTP 400. No vision quality benchmark yet. Details: [docs/advanced.md](docs/advanced.md#image-input).

### Activation steering

Both engines can steer the model with control vectors in the llama.cpp control-vector GGUF format (plus a JSON
sidecar), in `add` or `project` mode, chosen per request with a `steering` field. QEngine ships one vector for
Flash-Next, **off by default**: `refusal_projection`, which projects a refusal direction out of the residual stream so
the model declines far fewer requests. Removing refusals removes a safety behaviour of the model; you are responsible
for what it outputs. Turn it on in Docker with `-e QE_STEERING=1`, from source with
`QE_STEER=vectors/refusal_projection.gguf QE_STEER_DEFAULT=refusal_projection`. Vector format, loading your own
vectors, per-request selection and credits: [docs/advanced.md](docs/advanced.md#activation-steering).

## Serving behaviour

**Scheduling.** Shortest-prefill-first queue with aging; long prompts are prefilled in chunks between decode steps
of the running conversations; short requests can cut in between the chunks of a long prefill (Flash-Next: requests
up to 8K tokens, e.g. tool results, join the next prefill batch at once while the long prompt keeps half of it, on
servers whose prefill is bound by host-backed expert copies; `--long-share`). 27B: the time split follows the queue:
while the prompt to prefill next is short (up to `--prefill-short`, 4,096 tokens), prefill passes run back to back
with one decode step between them; a longer one leaves the running conversations a share by head count (running /
(running + waiting), between `--prefill-share-min` 0.10 and `--prefill-share` 0.5), and over any 2 seconds they keep
at least 10 % of the time. With many agents in flight the first token comes several times sooner (numbers under
[Benchmarks](#benchmarks)); `--prefill-share-min 0.5` restores the fixed half split. 27B with all decode seats
taken and more requests open than seats: two seats (Flash-Next on 3 cards: four, whose prompts share prefill passes)
stay for prompts; a reply prefilled there sends its first token and is parked in host memory until a seat frees
(oldest first; `--park 0` off). A request that does not
fit the KV pool lets smaller ones start first, until it has waited `--backfill-wait` seconds (30).

**Agents.** Requests are grouped by the client's session id (Claude Code `metadata.user_id`) and by message history
(a client without a session id: by system prompt);
the **main agent thread** (the conversation that receives the others' results) is recognised for the logs and
`/api/stats` only: the prefill queue, the shared prefill batches and memory treat every conversation alike (least
left to compute, aged by waiting time). When the pool is full, a request is admitted by pausing the most
recently admitted decoding conversations (the pool's own order) once it has waited `--pause-wait` seconds (3), and not
while another one is paused; they resume where they stopped. `--main-agent-re`
forces the main-thread label by a regex on the system prompt or tool names.

**KV pool under pressure** (both models). Idle conversations go to the host-RAM archive first. The archive frees memory
in one order: least recently used whole conversation first, every conversation alike (the main agent thread too), each
loss with a log line. The same order frees VRAM slots on both models; a generating conversation never leaves VRAM, and an
idle one never makes a request wait. When the running conversations still do not fit, the most recently admitted one
is paused: its KV goes to the archive and its stream pauses without an end token; it resumes exactly where it stopped
once the others keep their headroom. A reply ends early only when a single conversation is larger than the pool.
On Flash-Next every slot release, archive drop and restore from the archive prints one line (conversation id, tokens,
seconds idle, where it went and why), and `/api/stats` (`engine.conversations`, also in `qetop`) lists the known
conversations least recently used first, in a slot or in the archive, so the next one to leave is the first.

**Speculative decoding.** MTP drafts plus lookup drafts (when recent tokens repeat an earlier part of the
conversation, as in file edits); on the 27B the [DFlash2 block drafter](#dflash2-block-drafter-27b-on-when-it-fits)
for a conversation that decodes alone. Greedy and sampled outputs keep the model's distribution.

**Loops.** A reply whose last 600 tokens are one short fragment repeated is ended (`--loop-guard-tokens`). When an
agent keeps repeating earlier turns, whether one call or a short cycle of a few calls (same tool, exactly the same
arguments apart from the `description` of a shell call), each getting exactly the same result as before, for 3 turns
(`--loop-guard-turns`; calls that differ only in an id, such as a batch of deletes, and polls whose output changes are
not a loop), the next turn is generated with reasoning on even when thinking is off (at most 2048 reasoning tokens,
`--loop-guard-think`; the cached prompt is kept), with a short note on the tool result, a presence penalty on the
repeated reply and no lookup drafts. The reasoning comes on top of the request's `max_tokens`, and stop sequences apply
only to the answer. A guarded turn thinks even when the client disabled thinking (`enable_thinking: false`, `thinking: {"type": "disabled"}`,
`thinking_budget: 0`); `--loop-guard-think 0` turns this off. A long stretch of unchanged results can be legitimate waiting, so the guard backs off: it applies at 3,
6 and 12 turns, then every 24 turns (also for a client that sends only its last turns). Right after a user message it only turns reasoning on (harness text such as
`<system-reminder>` or `<task-notification>` blocks is not a user message). The note is left out when it would push the prompt over the context.
Conversations without tool results are never flagged. `"loop_guard": false` in a request turns it off.

**Turn end.** A generated `<|im_start|>` ends the reply like the end-of-turn token, so a reply never continues into a
turn the model made up. The log line (`stopping at a generated <|im_start|> ...`) shows the first and last 16 tokens of
the reply.

**Template markers in content.** Marker strings that a conversation only quotes (`<|im_start|>`, `<|im_end|>`,
`<think>`, `<tool_call>`, image placeholders, any added token of the tokenizer) are tokenized as plain text, wherever
they appear: user, system, assistant and tool messages, reasoning, tool-call names and arguments and tool
definitions, also a partial marker at the end of a name that the template would complete. Control tokens appear only
where the chat template puts them, so a conversation that pastes a chat template or server source does not split into
made-up turns, and the model does not copy a real end-of-turn token out of it. One exception follows the template's own
format: a user message that consists only of `<tool_response>...</tool_response>` blocks (tool results sent as a user
message) keeps the tags between and around the blocks. On the output side only the real `<think>`, `</think>` and
`<tool_call>` tokens switch between reasoning, answer and tool call; the same strings written as text stream as part of
the reply, and a tool-call argument that quotes `</parameter>` or `</function>` is not cut there. Conversations
without such strings in their content are tokenized exactly as before.

**Failure handling.**

- A stalled GPU or driver: an engine step that has not finished after `FLASH_RUN_TIMEOUT_S` (120 s, Flash-Next), or no
  scheduler progress for `QE_HANG_S` (180 s, both models), fails the open requests with an error at once, refuses new
  ones, writes bounded diagnostics (stacks, `nvidia-smi`; `QE_HANG_DIAG_S`, 30 s) to the log and exits with code 75 so
  Docker (`--restart`) or systemd restarts it. The step monitor is off under `ncu`, `nsys`, `compute-sanitizer` and a
  debugger. `kill -USR1 <pid>` writes the Python stacks at any time.
- A prompt the engine cannot start fails at once with the reason: an invalid or too-long prompt gets HTTP 400
  (`invalid_request_error`), so clients do not retry it unchanged. Only "no room yet" is retried, while other
  conversations run, for up to `QE_START_RETRY_S` (60 s).
- A request that produces no tokens fails with an error instead of waiting forever.
- Flash-Next's output guard fails a request whose forward produced non-finite values, instead of emitting garbage;
  the server goes on.
- Flash-Next keeps VRAM for its CUDA graphs (`FLASH_GRAPH_MB`, 256 MB per card, next to the expert cache) and frees
  its least recently used graphs to stay within it, prompt-processing graphs first (`/api/stats` -> `graphs` shows the
  use per card). A prompt-shaped graph is kept only when its shape came back within the last 64 prompt steps, and it
  never pushes out the decode graphs (it runs without a graph instead), so many different prompt lengths do not slow
  decoding down. If VRAM still runs short for a new graph, it gives up a small
  reserve, then drops cached graphs, then runs the step without a graph; it does not stop.
- Both engines keep serving when another process (a browser, the desktop, another application) takes GPU memory
  while they run, and take the memory back once it is free again (log lines `VRAM pressure` / `VRAM free again`):
  Flash-Next moves its least-used resident expert groups to host memory (they are read from there, as the other
  host-resident experts; with the pipeline of `FLASH_PAR=pp_prefill`, between pipelined prompts); 27B shrinks its KV
  cache (idle conversations go to the host-RAM archive; a reply that needs more room pauses with its KV in the archive,
  or waits, until the memory is free again; after `QE_PAUSE_TIMEOUT_S` (900 s) of waiting for memory another process
  holds, it ends with an error that says so; a queued request that fits nowhere with nothing running ends after
  `QE_NO_ROOM_S` (10 s) with such an error, and the server keeps serving). `FLASH_VRAM_LOW_MB`,
  `FLASH_VRAM_TARGET_MB`, `QE_VRAM_LOW_MB`, `QE_VRAM_TARGET_MB` set the marks, `FLASH_VRAM_ADAPT=0` / `QE_VRAM_ADAPT=0`
  turn it off. Measured: an application taking 1, 2 and 4 GB of one card during four streams and a 120K-token prompt,
  then freeing it (Flash-Next on 2 cards, also in the pipeline mode; 27B on 3 cards): every request served, speed back
  after the memory was freed.
- A start that finds the GPUs still held by a previous engine waits up to `QE_GPU_LOCK_WAIT_S` (60 s), then names the
  processes holding them.

**Logs.**

- The whole output (engine messages and tracebacks too) is appended to `~/.cache/qengine/logs/<served-name>.log` with
  timestamps, survives crashes and restarts, and ends with the exit status and the kernel's lines about it (OOM kill,
  GPU Xid). Rotated at 64 MB (`--log-mb`); `--log-file` moves it, `--log-file ''` turns it off.
- After every request: one line in vLLM's `SpecDecoding metrics` format and llama.cpp's `prompt eval time` /
  `eval time` format (`QE_STATS_LOG=0` turns them off).
- `WARNING empty reply`: a generated turn with no text and no tool call, with the slot, conversation, finish reason,
  thinking tokens, the `max_tokens` / budget that applied, whether `</think>` came, the last thinking tokens, and
  whether a retry follows; one more line reports the retry's outcome.
- `history: conversation N: the client's history has N assistant turn(s) with reasoning only; left out of the prompt`
  (Flash-Next): once per conversation and again only when the count grows. Such turns make the model end later turns
  empty, so they are left out (`--drop-empty-turns 0` keeps them). This is not an error.
- `/api/stats` counts thinking budgets, forced closes, empty replies and retries, history drops, and shows the parallel
  mode and the CUDA-graph memory per card.

**Monitoring.** Live dashboard and Prometheus metrics on `http://127.0.0.1:9393/` (`--metrics-port`), terminal
monitor `server/qetop.py`. Both show the running version and build in the header (e.g. `QEngine 1.3.6 (build abc1234)`).

## Settings

The defaults are the best configuration I measured; most settings exist to fit a machine or to turn a feature off.
Environment variables are read at start; an empty value counts as unset. A value per card is a comma list in GPU order
(a short list repeats its last value). The long tail is in [docs/advanced.md](docs/advanced.md).

### Both models

| setting | default | what it does |
|---|---|---|
| `server/run.sh <port>` | - | starts the server on `127.0.0.1:<port>`; extra arguments go to `server/app.py` |
| `--model <dir>` (Docker `QE_MODEL`) | `models/Qwen3.8-27B-INT4` | the model directory; the engine is picked from its `config.json` |
| `QE_DEVICES` (Docker: `QE_DEVICES`, else `QE_GPUS`) | all visible GPUs (Docker `0,1,2`) | physical GPU ids, e.g. `0,2` |
| `--max-ctx` | `262144` | context per conversation (`256K`, `512K`, `1M` accepted); above 262,144 turns on YaRN RoPE scaling (factor context / 262,144; Flash-Next also needs `FLASH_KV=ram`; 27B: up to what the VRAM pool holds, 1,232,896 tokens on 3 cards and 623,616 on 2 with the defaults (16 slots; with `QE_MAX_SEQS=32` 1,075,200 and 417,792); `QE_KV_BITS=4` raises it) |
| `--lsa` / `QE_LSA=auto` | off | 27B: sparse attention for long prompts (opt-in, lossy): 512K prompts about 2x faster, 1M about 3x, decode at 1M 2x. The released index map (Prefrontal AI, Apache-2.0) is downloaded on the first start into `~/.cache/qengine/lsa` (Docker: the `qengine-cache` volume); offline without it the server runs with full attention. `--lsa MAP` / `QE_LSA=MAP` takes a map file ([docs](docs/advanced.md#sparse-attention-for-long-prompts-opt-in)) |
| `--think` | `1` | thinking on by default (a request's `chat_template_kwargs.enable_thinking`, or a top-level `enable_thinking`, wins) |
| `--reasoning-effort` / `QE_REASONING_EFFORT` | the template's (`xhigh`) | default effort: `xhigh`, `high` / `max` (= `xhigh`), `medium`, `low`, `minimal` (= `low`), `none` (thinking off) |
| `--thinking-budget` / `QE_THINKING_BUDGET` | `0` (off) | default thinking budget in tokens, or one per effort (`low:4096,medium:16384,xhigh:32768[,default:N]`) |
| `QE_THINK_TAIL` | `auto` | room kept for the answer: 1/8 of `max_tokens`, 64 to 2048 tokens; a number; `0` = off |
| `--empty-retry` / `QE_EMPTY_RETRY` | `2` | resample a non-streamed empty reply up to N times; `0` = off |
| `--vision` | `0` | `1`: image input (Docker: the `vision` image) |
| `--temperature`, `--top-p`, `--top-k`, `--min-p` | model card, per mode: `1.0`, `0.95`, `20`, `0` thinking; `0.7`, `0.80`, `20`, `0` non-thinking | sampling defaults for requests that send none; a set value applies to both modes |
| `--presence-penalty` / `QE_PRESENCE_PENALTY` | model card, per mode: `0` thinking, `1.5` non-thinking | presence penalty for sampled requests that send none; a set value applies to both modes |
| `--frequency-penalty` / `QE_FREQUENCY_PENALTY` | `0` | frequency penalty for sampled requests that send none |
| `--served-name` | per model | the model id in the API |
| `--host`, `--port`, `--metrics-port` | `127.0.0.1`, `8080`, `9393` | where the API and the monitoring listen (`--metrics-port 0` = off) |
| `QE_HANG_S` | `180` | server watchdog in seconds; `0` = off |
| `QE_SSE_KEEPALIVE` | `10` | streamed replies: a keep-alive after this many seconds without an event (OpenAI `: keep-alive` comment, Anthropic `ping` event), e.g. while a reply is queued or waits for its turn; `0` = off |
| `--log-file`, `--log-mb` | `~/.cache/qengine/logs/<name>.log`, `64` | persistent log |
| `QE_STEERING` (Docker) | `0` | `1`: the included steering vector is the default; `<name>`: that vector |

### Qwen3.8-Flash-Next

| setting | default | what it does |
|---|---|---|
| `FLASH_PAR` | `auto` | `auto`, `tp`, `pp`, `pptp` (experimental); see [Parallel mode](#parallel-mode-flash-next-flash_par) |
| `FLASH_PAR_AUTO_P` | `8192` | prompt length `auto` weighs; `32768` favours long-prompt work |
| `FLASH_COMM` | `auto` | prefill transfer path: `auto`, `p2p`, `ce`, `host` (also `pull` / `push`) |
| `QE_DECODE_COMM` | `auto` | decode collectives: `auto` (timed on real decode steps at start-up, cached; keeps P2P or the host path per mode), `remeasure`, `p2p` (SM stores into the peers), `host` (the no-peer host path, the `QE_P2P=0` kernels, with peer access kept for prefill); also for the 27B |
| `QE_COMM_CACHE` | `1` | start-up cache of the link probe and decode calibration: `0` off, `refresh` measure again |
| `FLASH_MAX_SEQS` | `16` | conversations served at once (1 to 32; pipeline mode `pp` at most 16); fewer leave more VRAM for the expert cache. More than 16 step in groups of 16 and page more conversations through the host archive: on 3 cards 32 gave less total output than 16 with parking (`--park`) |
| `FLASH_EC_MARGIN_MB` | `256` | VRAM per card left free next to the expert cache; e.g. `256,2048,256` when GPU1 drives a desktop |
| `FLASH_GRAPH_MB` | `256` | VRAM per card for cached CUDA graphs: 128 MB of it (at most half of `FLASH_EC_MARGIN_MB`) comes out of that margin, the rest is kept on top; `0` = graphs share the margin (1.3.4) |
| `FLASH_PFGRAPH_WINDOW` | `64` | a prompt-shaped graph is captured only when its shape was used within the last N prompt steps, and never evicts a decode graph; `0` = off |
| `FLASH_ATTN_DMIN` | `16` | decode attention of batched steps (more than one wave of blocks): a row uses one key split per N key tiles (at most 8), so short contexts do not pay for 8 splits; `0` = always 8 splits |
| `FLASH_KV` | `vram` | `ram`: KV cache in pinned host RAM |
| `FLASH_KV_RAM_TOKENS` | `1M` | with `FLASH_KV=ram`: KV tokens of all conversations together (`K` / `M` / `G`); the host-RAM plan lowers it when the host is short (at least one window) |
| `FLASH_KV_BITS` | `8` | `4`: 4-bit KV cache (opt-in, not lossless against FP8): 7,488 instead of 13,520 bytes per token, about 1.8x the tokens with `FLASH_KV=ram`; see [Host memory](#host-memory-flash-next) |
| `FLASH_KV_ARCHIVE_GB` | `48` (less on smaller machines) | host-RAM archive of conversations; `0` = off, which on Flash-Next also turns off the pinned prompts and prefix reuse across slots, with `FLASH_KV=ram` too; a set value wins over the [host-RAM plan](#host-memory-flash-next). The archive's memory is pinned in the background, 1 GB at a time, only while the engine is idle and the host has at least 8 GB to spare above `QE_MEM_MARGIN_GB` (never next to a running step: [#32](https://github.com/bpx03/QEngine/issues/32)); a registration slower than 1 s, which is a sign of host memory pressure, pauses pinning for 30 s, and the pause doubles each time it repeats, up to 8 min. Until the pool has grown, new archive copies use plain memory (slower copies, same result) |
| `FLASH_PREFIX_GB` | `8` (less on smaller machines) | host RAM for pinned system prompts; `0` = off |
| `FLASH_NGRAM_PIN` | set by the host-RAM plan (`1` when it fits) | `1`: pin the n-gram table; `0`: read it through the page cache |
| `FLASH_WEIGHTS_COLD` | `1` | after the load, drop the page cache of the model files the engine no longer reads (all but the n-gram table in page-cache mode); `0` = keep it, and also turns off the load-time drop (`QE_LOAD_DROP`) |
| `FLASH_TP_SPIN_US` | `5000` | µs a tensor-parallel helper thread spins after its last job before it sleeps (an idle server then uses no CPU); `-1` = always spin |
| `FLASH_VRAM_ADAPT` | `1` | Flash-Next: when another process takes GPU memory while serving, move resident expert groups to host memory and back later; `0` off |
| `FLASH_VRAM_LOW_MB` / `FLASH_VRAM_TARGET_MB` | `128` / `256` | free memory per card below which groups move to host memory / up to which they move; they come back as soon as the other processes give the memory back (keeping 16 MB above the low mark free) |
| `FLASH_CTX_FIT` | `1` | Flash-Next: lower the context at start to the largest that fits the smallest card (logged); `0` keeps `--max-ctx` as given |
| `FLASH_PRELOAD` | `1` | Flash-Next: when CUDA was started before the engine (the vision encoder) with lazy module loading, load the engine's kernels at start instead of at their first use; `0` off |
| `QE_RAM_WAIT_S` | `20` | seconds the start waits for free host RAM (MemAvailable) when it is short of the full plan (as long as it rises, or while the previous QEngine server on these cards is still exiting with 1 GB or more on them); `0` plans at once |
| `QE_NO_ROOM_S` | `10` | 27B: seconds a queued request that fits nowhere waits with nothing running, then it ends with an error that says why; `0` = no deadline |
| `QE_PAUSE_TIMEOUT_S` | `900` | 27B: seconds a reply may wait for GPU memory another process holds before it ends with an error, counted from when that wait began: one that cannot continue, or a paused one that cannot resume because the KV cache gave memory back; a reply paused only to make room for another start has no deadline |
| `FLASH_NGRAM_SPIN_MS` | `50` | page-cache mode of the n-gram table: ms the row-copy thread spins after the last request, then it polls every 50 µs (1 ms after a second); `-1` = always spin |
| `QE_HOST_RESERVE_GB` | max(8, 8 % of RAM) | at start only: RAM the plan leaves to the system and the desktop (both models). `QE_MEM_MARGIN_GB` is the one that acts while serving |
| `FLASH_RUN_TIMEOUT_S` | `120` | engine step monitor in seconds; `0` = off |

### Qwen3.8-27B

| setting | default | what it does |
|---|---|---|
| `QE_MAX_SEQS` | `16` | conversations kept in the KV pool, up to 32; 1 to 4 on one card. More open requests than slots are parked beyond the seats (`--park`). `32` seats 32 at once, but leaves a smaller KV pool (2 cards: 417,792 against 623,616 tokens; one 262,144-token conversation still fits, two very long ones at once do not) |
| `QE_ADMIT_BY_LIMIT` | `1` | a running conversation is counted at admission with at most its reply's `max_tokens` + 1,024 tokens of growth (else 8,192); `0` = always 8,192 |
| `QE_ARCH_REPLY` | `1` | a conversation's next turn that leaves its cached tokens only inside its last reply (a client that sends another assistant text or drops the reasoning) no longer copies the whole conversation to the host archive first; `0` = copy it, as before |
| `QE_ARCH_SUBS` | `2` | 27B: recurrent-state checkpoints an archived conversation keeps besides its end state (1 to 4; each is about 151 MB of host RAM). With many agents at once, `1` lets the host archive hold about a third more conversations; the second one is what an edited or retried last turn resumes from |
| `QE_ARCH_DEMOTE` | `1` | 27B: an archived conversation whose next turn replaced its last reply (a client that sends another assistant text) goes first when the archive needs room, instead of being kept as recently used; `0` = keep it as before |
| `QE_SNAP_DEMOTE` | `1` | 27B: when a prompt-end state is saved, the same slot's older prompt-end states of that conversation (its earlier turns) go first when the state store needs room, instead of aging like the live ones; with many conversations per slot count a waiting turn no longer loses its own state. The resume point of a repeated prompt and of a fork is kept. `0` = plain least-recently-used order, as before |
| `QE_DF_BATCH` | `2` on one card, `0` on more | batched steps of up to this many conversations draft with [DFlash2](#dflash2-block-drafter-27b-on-when-it-fits) (one context ring each, 43 MB per card); `0`: batched steps draft with MTP |
| `QE_DRAFTER` | `auto` | drafter of a conversation that decodes alone: `auto` loads [DFlash2](#dflash2-block-drafter-27b-on-when-it-fits) when the requested context and slots fit with it (logged, with the `--max-ctx` that would fit; one card with the default `--max-ctx` stays on MTP), `dflash2` always (context shrinks to fit), `mtp` never (no download) |
| `QE_DFLASH` | – | path to a local DFlash2 checkpoint (directory or file) instead of the download |
| `QE_KV_BITS` | `8` | `4`: 4-bit KV cache (opt-in, not lossless against FP8): about 1.5x the context on small cards; `--lsa` is then off (logged); see [Smaller cards](#gpus) |
| `QE_HEAD4R` | `16` | 27B verify head: INT4 screen + exact BF16 rescore of the top N per row (pinned host RAM); `0` = INT8 head |
| `QE_RESERVE_GB` | `1.5` | VRAM per card left outside the KV pool; e.g. `1.5,3,1.5` for a desktop on GPU1 |
| `QE_KV_ARCHIVE_GB` | `32` (less below 64 GB of RAM) | host-RAM archive of conversations; `0` = off; a set value wins |
| `QE_PREFIX_GB` | `8` (less below 64 GB of RAM) | host RAM for pinned system prompts |
| `QE_COMM` | `auto` | prefill transfer path: `auto`, `p2p`, `ce`, `host` |
| `QE_FETCH_ASSETS` | `1` | `0`: no downloads at start: not the tuned MTP head and draft ranking, not the DFlash2 drafter (MTP drafts only), not the `--lsa` index map (full attention) |
| `--disk-cache DIR` | off | persistent prompt cache that survives restarts (Flash-Next: experimental, since 1.3.5.3) |
| `--disk-eager` | off | with `--disk-cache`: also write each conversation after every reply (NVMe wear: the whole file per turn) |

### DeepSeek-V4.1-Flash (preview)

The common settings above apply (`--max-ctx`, `--think`, `--reasoning-effort`, `--thinking-budget`, `--temperature`,
`--top-p`, `--served-name`, `--host`, `--port`, `--metrics-port`); presence / frequency penalties, `--vision` and
steering do not. The full list is in [docs/advanced.md](docs/advanced.md#deepseek-v41-flash-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `Maya-S` directory of the GGUF files; the DSpark drafter is read from `<dir>/../dspark` |
| `--max-ctx` | `262144` | context per conversation; with P2P the long-context memory is taken from the expert pool only while needed (short chats as at 32K); without it, it is reserved at start (halved, down to 32,768, when it would leave less than 8 GiB free on a card; logged); a start-up line says which; an explicit value wins |
| `--archive-gib` | `32` | host RAM for idle conversations (least recently used out first; it shrinks when the host runs short); `0` = off |
| `--max-active` | `8` | requests in service at once (they take turns); more wait in the queue |
| `--route-margin` | off | opt-in, **approximates** the routing: when the router picks an expert that is not in VRAM and a VRAM-resident expert scored within this margin, the resident one runs; `0.05` is the tested value (decode 7-20 % faster on three prompts, negative log-likelihood 1.641 against 1.623 on my gate texts); logged at start |
| `--tier-reset` | off | opt-in: after each request the expert placement goes back to its state after loading, so repeating a greedy request gives the same text; serves one request at a time and turns prefix sharing off; costs decode speed |
| `--no-prefix-share` | off | off-switch: no shared copies of system prompts |
| `--no-dspark` | off | off-switch: decode without the draft model (much slower) |
| `--engine-args "<flags>"` | – | engine flags, e.g. `--cpu-threads N` (default 24), `--reserve-gb X` (VRAM left free per card, default 0.6) |
| `DSV4_EGMIRROR` | off | faster long prompts on btrfs / XFS through reflink copies of the two Engram shards ([docs](docs/advanced.md#deepseek-v41-flash-preview)) |

### GLM-5.3-Flash (preview)

The common settings above apply (`--max-ctx`, `--devices` / `QE_DEVICES`, `--reasoning-effort`, `--thinking-budget`,
`--temperature`, `--top-p`, `--served-name`, `--host`, `--port`, `--metrics-port`); presence / frequency penalties,
`--vision`, `--disk-cache` and steering do not. The engine's own switches are environment variables; a few more are in
[docs/advanced.md](docs/advanced.md#glm-53-flash-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `bpx03/GLM-5.3-Flash-INT4-QEngine` directory |
| `--max-ctx` / `GLM_MAX_CTX` | `262144` | context; the engine lowers it with a warning when its attention cache does not fit (`GLM_MAX_CTX` caps it). The cache of a 262K context leaves room for 32K-row prefill windows instead of 64K: if your prompts stay under 64K, `--max-ctx 65536` prefills prompts above 32K about 6 % faster |
| `--devices` / `QE_DEVICES` | every visible card | the cards: two or three (of more, the first three) |
| `--split` | `704,640,704` | routed expert columns per card, multiples of 64 summing to 2048; by default the 640 goes to the card with the least free VRAM |
| `GLM_PF_SLICE_MS` | `1000` | host slice of a prefill in ms (progress and cancel between slices); `0` = the whole prompt in one call |
| `GLM_CONVS` | `8` | conversations served at once (time-sliced, exact); `1` = one at a time, the others wait in the queue |
| `GLM_SLICE_MS` / `GLM_SLICE_X` | `2000` / `8` | a reply's turn while others wait: at least this many ms and this many times the measured switch cost (park + resume) |
| `GLM_PREEMPT_MS` | the slice | how long the running reply decodes before a new prompt's prefill parks it |
| `GLM_PF_CANCEL_MID` | `1` | `0` = a cancel waits for the next prefill window start (the older rule) |
| `GLM_CPU_ISA` | `auto` | CPU expert lane kernels: `auto` (AVX-512 when present), `avx2`, `avx512` |
| `GLM_LANE_CPUS` / `GLM_CPU_THREADS` | from the CPU topology | the CPU lane's cores (a list of ranges) / its thread count, instead of the derived placement |
| `GLM_P2P` | `1` | `0` = off-switch: the all-reduces go through host memory even when P2P is there |

### gpt-oss-120B (preview)

The common settings above apply (`--max-ctx`, `--reasoning-effort`, `--thinking-budget`, `--temperature`, `--top-p`,
`--served-name`, `--host`, `--port`, `--metrics-port`, `--presence-penalty`, `--frequency-penalty`); `--vision`,
`--disk-cache` and steering do not. The engine's own switches are environment variables; the full list is in
[docs/advanced.md](docs/advanced.md#gpt-oss-120b-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `openai/gpt-oss-120b` directory |
| `--max-ctx` / `GPTOSS_MAX_CTX` | `131072` | context per conversation (the model's limit; `GPTOSS_MAX_CTX` caps it) |
| `GPTOSS_SLOTS` | `8` (context per conversation >= 32,768), else `4` | conversations decoded at once (1-16) |
| `GPTOSS_POOL_ROWS` | every row that fits | K/V rows shared by all slots; a number pins it (never below one conversation) |
| `GPTOSS_POOL_RESERVE_MB` | `320` | VRAM per card left free beside the pool (decode graphs, other processes) |
| `GPTOSS_KV` | `bf16` | opt-in `k12` (12-bit K, 1.56x the rows) or `fp8` (8-bit K and V, 1.94x the rows): lossy, for many agents with long contexts (KL to the bf16 store at 32K / 64K context: k12 1.2e-3 / 2.2e-3, fp8 2.9e-3 / 2.3e-3) |
| `GPTOSS_ADMIT_REPLY_ROWS` | `2048` | rows a new prompt leaves free for the running replies to grow (at most 1/16 of the pool) |
| `GPTOSS_LOOKUP` | `9` | rows per step of the lookup drafts on greedy replies (a token and up to 8 drafts); `0` = off-switch |
| `GPTOSS_COMM` | auto | `host` forces the all-reduce through host memory, `p2p` requires P2P |
| `GPTOSS_CHUNK` | `10240` | prefill chunk in tokens |
| `GPTOSS_ARCHIVE_SYNC_ROWS` | all | an evicted conversation goes to host RAM whatever its size; a number bounds the copy |
| `QE_KV_ARCHIVE_GB` / `QE_KV_ARCHIVE_MARGIN_GB` | `128` / `8` | host RAM for idle conversations / what it leaves free below `MemAvailable` |
| `GPTOSS_BG_REARM` | `1` | after every finished reply an idle conversation is copied to host RAM in the background, so the next long prompt does not wait for that copy; `0` = off-switch |
| `GPTOSS_BROKEN_EXIT` | `1` | after a device fault the process exits with code 75 (for a restart); `0` = off-switch (keeps the failed process for debugging) |

### Inkling-2-Mini (preview)

The common settings above apply (`--max-ctx`, `--reasoning-effort`, `--temperature`, `--top-p`, `--served-name`,
`--host`, `--port`, `--metrics-port`, `--presence-penalty`, `--frequency-penalty`, `--park`); `--vision`,
`--disk-cache`, `--thinking-budget` and steering do not. The engine's other switches are in
[docs/advanced.md](docs/advanced.md#inkling-2-mini-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `thinkingmachines/Inkling-2-Mini` directory (with `dense8.safetensors` and `mtp_qe.safetensors` next to it) |
| `QE_DEVICES` / `--devices` | `0` (one card) | the cards; two (`0,1`) split the model with tensor parallelism (needs GPU P2P). Docker: `QE_GPUS` / `QE_DEVICES`, else one card |
| `--max-ctx` | the model's 1,048,576 | context per conversation, served up to what the K/V pool holds (one start-up `context` line) |
| `--dense-weights` / `INK_DENSE8` | `int8` when `dense8.safetensors` is there, else `g15` | `g15` (`INK_DENSE8=0`) = the original weights, lossless; `int8` (`INK_DENSE8=1`) without the file = RTN INT8 at load (lower quality, logged); the flag wins |
| `INK_SLOTS` | `8` on 2+ cards, `2` on one card | conversations decoded at once (1-16) |
| `INK_DRAFTER` | `1` | `0`: no drafter (plain decoding, about 1 GB more K/V room) |
| `INK_DRAFTER_FILE` | `mtp_qe.safetensors` when present, else `mtp.safetensors` | drafter weights: a file name in the model directory or a path |
| `QE_KV_ARCHIVE_GB` / `QE_KV_ARCHIVE_MARGIN_GB` | `128` / `8` | host RAM for idle conversations / what it leaves free below `MemAvailable` |
| `INK_BG_REARM` | `1` | after every finished reply the idle conversation is copied to host RAM in the background; `0` = off-switch |
| `INK_ADMIT_REPLY_ROWS` | `2048` | rows a new prompt leaves free for the running replies to grow |
| `INK_ARCHIVE_SYNC_ROWS` | all | an evicted conversation goes to host RAM whatever its size; a number bounds the copy |

## Benchmarks

All numbers were measured by me on one machine: 3× RTX 3090 (24 GB, PCIe 4.0 x16, P2P through
the patched driver unless noted), one NUMA node; "2 cards" are two of its cards. QEngine runs with its defaults. They
describe that machine; your numbers depend on the cards, links, driver and host.

Since 1.3.6 the server counts the end token a reply generates in `usage` (see
[Thinking budget and the answer tail](#thinking-budget-and-the-answer-tail)), and benchmark clients take their token
counts from it, so 1.3.6 numbers include one more token per reply that ends on its own than the older numbers below.

### Qwen3.8-Flash-Next

**1.5.0** (November 19, 2026; 3 cards, P2P, the club-3090 `rebench-full.sh` setup: 8 slots, thinking on, effort low,
vision on): single-stream decode narrative / code 310 / 452 tok/s, prefill 10K / 90K 8,254 / 8,390 tok/s (first token
1.23 / 10.5 s), decode at 10K / 90K depth 350 / 306 tok/s, 1 / 2 / 4 requests at once 302 / 421 / 647 tok/s total, soak
median decode 402.7 tok/s, verify-stress 8/8, soak PASS. Concurrency probe (16 slots, greedy, 256-token prompts and
replies): 300 / 642 / 880 / 1,061 tok/s total at 1 / 4 / 8 / 16 streams.

**1.3.6** (November 15, 2026; 3 cards, P2P, the club-3090 `rebench-full.sh` setup: 8 slots, thinking on, effort low,
vision on): single-stream decode narrative / code 305.6 / 449.3 tok/s, prefill 10K / 90K 8,207 / 8,394 tok/s, decode
at 10K / 90K depth 315 / 296 tok/s, 1 / 2 / 4 requests at once 305 / 422 / 653 tok/s total, soak median decode 411.4
tok/s, verify-stress 8/8, soak PASS. The club bench sends no presence penalty, so the server's non-thinking default
1.5 applies; with `QE_PRESENCE_PENALTY=0` (the default before 1.3.5.3) the same run gives 325.0 / 444.4 tok/s
single-stream. Plain defaults (`server/run.sh`, 16 slots): 311 / 454 tok/s single-stream, prefill 8,057 / 8,372, 8
agents at once 760 tok/s (mean of 7 runs, 709-828; real-text prompts 976-993). 2 cards (0, 2), plain defaults: 201 /
226 tok/s single-stream, prefill 4,265 / 7,134, 8 agents at once 198 tok/s (224-256 with presence 0); a 262K-token
prompt with image traffic passes on both.

**Many agents in flight** (1.5.0 code, November 19, 2026; 3 cards, P2P, defaults; the 27B's agent replay below
with 64-token replies: `bench/q27_ttft_conc.py --mix light --gen 64`, warm-up 20 s (64 agents: 60 s), 60 s window,
one run per point). Median / 95th percentile time to first token, total output:

| agents in flight | 1 | 4 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| before (1.3.6 scheduling, same day) | 0.16 / 0.46 s, 131 tok/s | 0.44 / 1.42 s, 154 | 1.91 / 7.2 s, 165 | 3.93 / 9.4 s, 154 | 7.1 / 33 s, 144 |
| prompt-end state + queue-aware split | 0.13 / 0.38 s, 150 tok/s | 0.25 / 0.67 s, 175 | 0.37 / 1.46 s, 191 | 2.83 / 3.54 s, 170 | 8.0 / 9.0 s, 157 |
| 1.5.0 (+ parking beyond the seats) | 0.13 / 0.38 s, 149 tok/s | 0.25 / 0.65 s, 169 | 0.36 / 1.23 s, 197 | 0.26 / 0.79 s, 179 | 0.75 / 2.0 s, 171 |

At 32 and 64 agents the 16 decode seats are full. Without parking a prompt got a seat only when a reply ended; now
it is prefilled on one of four seats kept for prompts, sends its first token and waits parked in host memory (greedy
replies resume bit-identical). At 64 parked replies may hold up to 3/4 of the host archive (`FLASH_KV_ARCHIVE_GB`,
48 GB): past that a reply waits for a seat as before, so a larger archive parks more. Up to 32 seats can be set
(`FLASH_MAX_SEQS`); on 3 cards 32 seats gave less total output than 16 with parking.

**1.3.5** (November 6, 2026; 3 cards, P2P, the club-3090 `rebench-full.sh` setup: 8 slots, thinking on, effort low, vision on): single-stream decode narrative / code 313.6 / 404.8 tok/s, prefill 10K / 90K 6,128 / 7,515 tok/s, decode at 10K / 90K depth 350 / 290 tok/s (one reply each), 1 / 2 / 4 requests at once 293 / 430 / 651 tok/s total, soak median decode 372.4 tok/s, verify-stress 8/8, soak PASS. At the plain defaults, thinking off: 3 cards 312.9 / 410.7 tok/s, prefill 6,186 / 7,249, 8 agents at once (`vllm bench serve`, 500 tokens in, 2,000 out) 867 tok/s, a noisy point, other runs 935 and 956. 2 cards (0, 2): 208.7 / 220.7 tok/s, prefill 3,797 / 6,357, 8 agents at once 285 tok/s, a 262K-token prompt and image traffic pass.

**1.3.4.2** (November 2, 2026; 3 cards, P2P, the club-3090 `rebench-full.sh` setup: 8 slots, thinking on, effort low,
vision on; in brackets 1.3.4.1 on the same day with the same desktop load on GPU1): single-stream decode narrative /
code 314.3 / 401.4 tok/s (292.4 / 368.4), prefill 10K / 90K 6,137 / 6,877 tok/s (6,061 / 6,675), decode at 10K / 90K
depth 328 / 307 tok/s (294 / 275), 1 / 2 / 4 requests at once 288 / 414 / 581 tok/s total (273 / 382 / 555), soak
median decode 359.9 tok/s (333.9), 8 agents at once (`vllm bench serve`, ~500-token real-text prompts, 2,000 output
tokens) 1,022 tok/s (897), verify-stress 8/8, soak PASS. 2 cards (0, 2): 8 agents at once 359 tok/s, a 262K-token
prompt and image traffic pass.

**1.3.4.1** (November 2, 2026; 3 cards, P2P, the club-3090 `rebench-full.sh` setup: 8 slots, thinking on, effort low,
vision on; in brackets the 1.3.4 weights on the same day with the same desktop load on GPU1): single-stream decode
narrative / code 292.4 / 368.4 tok/s (CV 2.7 / 2.2 %; 300.0 / 389.9), prefill 10K / 90K 6,061 / 6,675 tok/s (5,940 /
6,749), decode at 10K / 90K depth 294 / 275 tok/s (318 / 282), 1 / 2 / 4 requests at once 273 / 382 / 555 tok/s total
(293 / 407 / 566), 8 agents at once (`vllm bench serve`, ~500-token real-text prompts, 2,000 output tokens) 897 tok/s
(944 with the 1.3.4 weights, same day), verify-stress 8/8, soak PASS. `flash_run`: 10K / 90K-token prefill 1.561 /
14.995 s (1.554 / 14.96 s). 2 cards (0, 2): 8 agents at once 370 tok/s, a 262K-token prompt and image traffic pass.

Measured during the 1.3.1 release validation (October 30, 2026). Single-stream decode and prefill: club-3090
`scripts/bench.sh` (thinking off); agents: `vllm bench serve` as the client, ~500-token real-text prompts, 2,000
output tokens; long conversations: 4 conversations of ~64K tokens, then rounds where all 4 send a follow-up at once.

| tok/s, higher is better | 3 cards | 2 cards |
|---|---:|---:|
| single-stream decode, narrative | 307 | 210 |
| single-stream decode, code | 388 | 248 |
| prefill, 10K-token prompt | 5,960 | 3,760 |
| prefill, 90K-token prompt | 6,410 | 5,179 |
| agents, 8 at once (total output) | 941 | 390 |
| agents, 16 at once (total output) | 1,104 | 468 |
| 4 × 64K conversations, follow-up rounds (output) | 455 (`FLASH_KV=ram`: 580) | 233 (`FLASH_KV=ram`: 332) |

1.3.4 speed check against 1.3.3 (3 cards, P2P, `flash_run`, November 1, 2026): 90K-token prefill 13.01 s (1.3.3:
13.10 s); decode on a code prompt 343.8 tok/s (deterministic mode 367.6 tok/s; 1.3.3: 365.5).

The parallel modes and the no-P2P path (1.3.4, November 1, 2026) are measured in
[Parallel mode](#parallel-mode-flash-next-flash_par) and [With and without GPU P2P](#with-and-without-gpu-p2p).
Burst prefill, slow links and the KV-in-RAM runs: [docs/advanced.md](docs/advanced.md#measurements).

### Qwen3.8-27B

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults, DFlash2 loaded, the club-3090 `bench.sh` settings: thinking on,
effort low, vision on): single-stream decode narrative / code 275 / 524 tok/s, prefill 10K / 90K 6,188 / 4,994 tok/s
(first token 1.60 / 18.0 s); concurrency probe (greedy, 256-token prompts and replies): 312 / 575 / 848 / 999 tok/s
total at 1 / 4 / 8 / 16 streams.

**1.3.6** (November 15, 2026; 3 cards, P2P, defaults, DFlash2 loaded): single-stream decode narrative / code 278.7 /
515.5 tok/s, prefill 10K / 90K 6,309 / 5,113 tok/s (5,769 / 4,784 right after other runs), 8 agents at
once 1,129 tok/s (1,210 with `QE_PRESENCE_PENALTY=0`). With the club-3090 `bench.sh` settings: 281.2 / 507.0 tok/s,
prefill 6,133 / 4,997. Short single replies (385 tokens, greedy: a Python function, a short story, a proof): 474 / 264
/ 401 tok/s on 3 cards; 212 / 150 / 196 tok/s on one card (`QE_MAX_SEQS=1 --max-ctx 150K`, DFlash2).

**Many agents in flight** (1.5.0 code, November 18, 2026; closed loop, no think time: each agent sends its next
turn when the reply ends; a shared 2K-token system prompt, a first message of 1-3K tokens, then tool results of
30-2,000 tokens (about 650 new tokens per turn) up to 16K tokens of context; 128-token replies, greedy, thinking off;
`bench/q27_ttft_conc.py --mix light --gen 128`, one run per point). Median time to first token / total output:

| agents in flight | 1 | 4 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| 2 cards, 1.3.6 (16 slots, fixed half split) | 0.18 s / 140 tok/s | 0.64 s / 157 | 4.5 s / 231 | 25.0 s / 159 | 73.0 s / 111 |
| 2 cards, 1.5.0 defaults (16 seats, parking beyond them; release build, November 20) | – | – | 1.04 s / 243 tok/s | 0.48 s / 359 | 1.0 s / 285 (after a 90 s warm-up) |
| 2 cards, 1.5.0, `QE_MAX_SEQS=32` | 0.19 s / 142 tok/s | 0.33 s / 173 | 0.86 s / 248 | 0.32 s / 361 | 5.0 s / 155 |
| 3 cards, 1.5.0, `QE_MAX_SEQS=32` | 0.13 s / 193 tok/s | – | 1.31 s / 258 | 0.66 s / 403 | 3.3 s / 242 |

At 64 agents the measured window (after 15 s) still holds the 64 agents' first turns (about 140K new tokens, the
queue is prefill-bound). After a 90 s warm-up, at 64 agents: 2 cards 30.3 s / 147 tok/s without parking and with four checkpoints per archive,
0.27 s / 318 tok/s with 1.5.0 (`QE_MAX_SEQS=32`); 3 cards 1.45 s / 305 tok/s, where past ~10K tokens per agent the default 32 GB host
archive no longer holds all 64 conversations (`QE_ARCH_SUBS=1`: 1.06 s / 317; a larger `QE_KV_ARCHIVE_GB` removes
the limit).

1.3.5 (November 6, 2026; 3 cards, P2P): single-stream decode narrative / code 274.1 / 422.5 tok/s, prefill 10K / 90K 6,437 / 5,159 tok/s, 8 agents at once 1,205 tok/s. With the club-3090 `bench.sh` settings: 276.1 / 418.9 tok/s. Prefill measured on its own.

Serving benchmark (October 2026, QEngine 1.0), 3 cards, `vllm bench serve` as the client; output tokens/s:

| scenario (input / output tokens) | 1 at once | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| short (1024 / 512) | 208 | 794 | 949 | 1,001 |
| agent (500 / 2000) | 287 | 1,080 | 1,302 | 1,423 |
| ShareGPT | 243 | 819 | 935 | 1,110 |

Later changes (W4A8 GEMM for batched passes of 33 rows or more, less admission headroom) raised agent at 32 to 1,703
and short at 32 to 1,039 tok/s in a same-day A/B. Prefill of one new prompt (median time to first token): 32K tokens
5.35 s, 200K tokens 50.7 s.

### DeepSeek-V4.1-Flash (preview)

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults; club-3090 tools): single-stream decode narrative / code 88 /
120 tok/s, prefill 10K / 90K 4,217 / 4,895 tok/s (first token 2.35 / 17.8 s), 70 / 73 / 78 / 78 tok/s total at 1 / 4
/ 8 / 16 streams (requests take turns; `--max-active 16` for 16). Quality: negative log-likelihood 1.635 on my gate
texts (no full-precision reference fits on this machine); 8-pack under [Quality](#quality).

More points (November 20, 2026, release build; 3 cards, defaults unless noted, greedy, DSpark on; one run each):

| DeepSeek-V4.1-Flash, 3x RTX 3090 | P2P | no P2P |
|---|---:|---:|
| single-stream decode, engine, three prompts: git log / code / narrative (tok/s) | 104.0 / 105.5 / 76.3 | 99.8 / 97.9 / 69.7 |
| same, `--route-margin 0.05` | 124.7 / 119.3 / 81.9 | – |
| prefill, 2.5K-token prompt first / second time, 10K-token prompt (s) | 1.44 / 1.28, 2.87 | – |
| prefill, 117K / 233K-token prompt, server (s) | 27.0 / 56.2 | – |
| decode with 117K / 233K tokens of context, server (tok/s) | 103.0 / 91.2 | – |
| 4 agents at once, unrelated prompts, server (total output tok/s; slowest first token) | 65.5; 5.2 s | – |
| 4 / 8 agents at once, shared 7K system prompt, server (total output tok/s; slowest first token) | 99.6 / 97.6; 2.3 / 10.2 s | – |
| ready after start, files in the page cache (s) | 40.9 | – |

The no-P2P column ran with the default 262,144 context, so its reservation is static (`DSV4_NOP2P=1` on the same
machine).

### GLM-5.3-Flash (preview)

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults; club-3090 tools): single-stream decode narrative / code 110 /
92 tok/s, prefill 10K / 90K 2,920 / 3,572 tok/s (first token 3.42 / 25.4 s), 72 / 55 / 50 / 44 tok/s total at 1 / 4 /
8 / 16 streams (replies take turns; `GLM_CONVS=16` for 16). Server, one conversation: prefill of an 8K / 32K /
143K-token prompt 2,996 / 3,443 / 3,917 tok/s, decode at 100K context 94.9 tok/s. Quality: next-token KL against the
official FP8 model 0.448 / 0.541 / 0.772 on three prompts (a README, a git log, an agent turn), argmax agreement 79.4
/ 74.6 / 69.0 %; 8-pack under [Quality](#quality). Ready after start: about 60 s with the files in the page cache.

### gpt-oss-120B (preview)

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults; club-3090 tools): single-stream decode narrative / code 301 /
301 tok/s, prefill 10K / 90K 13,280 / 5,055 tok/s (first token 0.74 / 17.6 s), 306 / 594 / 731 / 835 tok/s total at 1
/ 4 / 8 / 16 streams (`GPTOSS_SLOTS=16` for 16).

More points (November 20, 2026, release build; 3 cards, defaults, greedy, engine tool, mean of two runs; the no-P2P
column with `GPTOSS_COMM=host`):

| gpt-oss-120B, 3x RTX 3090 | P2P | no P2P |
|---|---:|---:|
| single-stream decode, short context (tok/s) | 337 | 321 |
| 8 conversations at once (total output tok/s) | 1,013 | 952 |
| prefill, 10K-token prompt (tok/s) | 13,120 | 12,310 |
| ready after start, server, files in the page cache (s) | 10.6 | – |

In the release stress run a 129K-token prompt with all 8 slots busy found its needle.

Quality (1.5.0 build): every layer's output agrees with an fp32 reference of the model within the bar (worst median
row error 3.96e-4 through the decode path, 5.24e-4 through the prefill path); next-token KL against fp32 0.118 and
top-1 agreement 86.9 % on 11 items through the prefill path (0.112 through the decode path); needles at 90K: 3 of 3.

### Inkling-2-Mini (preview)

**1.6.0 development build** (November 25, 2026; GPU 0, and GPUs 0 + 2 with P2P; defaults: INT8 dense weights, trained
drafter; engine tool, greedy):

| Inkling-2-Mini, RTX 3090 | 2 cards | 1 card |
|---|---:|---:|
| plain decode step at 512 tokens of context, no drafts (ms; tok/s) | 4.99; 200 | 7.10; 141 |
| single-stream decode with drafts, mix of 9 prompt types (prose, code, JSON, tables, edits, math; 128 tokens each, harmonic mean, tok/s) | 320 | 214 |
| prefill, 2K / 8K / 120K-token prompt, engine tool (tok/s) | 10,804 / 12,369 / 9,202 | – |
| context per conversation, default slots (tokens) | 565,216 (8 slots) | 125,920 (2 slots) |
| ready after start, server, files in the page cache (s) | 15 | 9 |

Club-3090 tools (1.6.0 release build, November 26, 2026, server defaults; the club sampler `top_p` 0.95 / `top_k`
20 runs on the device with drafts): single-stream decode narrative / code 230 / 330 tok/s on 2 cards, 162 / 223 on one
card; prefill 10K / 90K 9,430 / 8,860 tok/s (first token 1.07 / 10.2 s) on 2 cards, 6,186 / 4,945 (1.57 / 17.6 s) on
one card; 213 / 362 / 431 tok/s total at 1 / 4 / 8 streams on 2 cards (8 slots), 147 at 1 stream on one card (2 slots).

Quality: next-token KL against an fp32 reference of the model on a 25K-token text set: 0.00539 with the
INT8 dense weights through the decode path (0.00544 through the prefill path; one card 0.00538 / 0.00558), 0.00495
with `--dense-weights g15`; the official bf16 forward measures 0.00592 on the same set. Top-1 agreement 97.6 %.

### Quality

**Closeness to BF16.** Next-token KL divergence against the official BF16 model on a public reference set (21 cases,
32,359 positions: prose, code, Japanese, Chinese, SQL, agent turns and WikiText-2), scored with the set's own
`compare.py`; every position through QEngine's decode path (forwards of up to 16 tokens) on 3 cards (1.3.6 release
build, November 15, 2026, identical to 1.3.5 to every digit; the 1.3.4 row November 2). The last column scores the last 512 positions of the long cases after the context before them went through
the server's batched prefill (my own harness, not part of the command below). Lower is closer to BF16:

| Flash-Next weights | mean KL | top-1 agreement | WikiText-2 KL | KL, context by batched prefill (last 512 positions) |
|---|---:|---:|---:|---:|
| default | 0.114 | 89.6 % | 0.125 | 0.126 |
| 1.3.4 (`--revision v1.3.4`) | 0.267 | 83.3 % | 0.329 | 0.358 |

To measure it yourself (3 cards; ~32 GB download, ~65 GB of scratch space):

```bash
hf download jyohukuchan/qwen38-flash-next-bf16-logits --repo-type dataset \
  --revision 5294aa9ab94b0956742fc4c6528c21c1c3c08e42 --local-dir data/kl-ref \
  --include "reference/*" "corpus/*" compare.py vocab.json manifest.json
python3 tools/flash_kl_ref.py --model models/Qwen3.8-Flash-Next-QE-INT4 --ref data/kl-ref --work /path/to/scratch
```

The same from the Docker image (no source build; the downloaded set at `/ref`, the scratch space at `/work`):

```bash
docker run --rm --gpus all -e QE_GPUS=0,1,2 -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 \
  -v /path/to/models:/app/models:ro -v $PWD/data/kl-ref:/ref:ro -v /path/to/scratch:/work \
  ghcr.io/bpx03/qengine:latest kl --ref /ref --work /work
```

`--cases en_prose_129,python_code_513` scores only those cases (a quick check, not the numbers above); the result
is printed and written to `/work/summary.json`.

club-3090 8-pack (benchlocal, 150 scenarios, 4,096-token answer budget per scenario), QEngine 1.5.0 (November 19-20,
2026), one full run per model on 3 cards, each on a fresh server with its defaults (the two Qwen models with vision on
and the club's launch settings, reasoning effort low). Scenarios passed, thinking off / on (gpt-oss-120B always
reasons: reasoning effort low / medium):

| pack | N | Qwen3.8-27B | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash | GLM-5.3-Flash | gpt-oss-120B | Inkling-2-Mini |
|---|---:|---:|---:|---:|---:|---:|---:|
| toolcall | 15 | 12 / 13 | 13 / 14 | 12 / 13 | 13 / 13 | 11 / 13 | 13 / 14 |
| instructfollow | 15 | 11 / 15 | 13 / 14 | 12 / 15 | 9 / 15 | 14 / 14 | 15 / 15 |
| structoutput | 15 | 12 / 12 | 14 / 14 | 12 / 14 | 12 / 14 | 13 / 14 | 15 / 14 |
| dataextract | 15 | 10 / 13 | 10 / 12 | 12 / 13 | 13 / 14 | 11 / 9 | 13 / 14 |
| reasonmath | 15 | 13 / 15 | 13 / 14 | 12 / 14 | 14 / 15 | 12 / 13 | 13 / 15 |
| bugfind | 15 | 14 / 15 | 15 / 14 | 15 / 15 | 15 / 15 | 11 / 12 | 15 / 15 |
| hermesagent | 20 | 15 / 14 | 12 / 12 | 15 / 14 | 15 / 15 | 12 / 13 | 15 / 13 |
| cli | 40 | 27 / 31 | 30 / 31 | 32 / 35 | 30 / 32 | 26 / 28 | 23 / 24 |
| **total** | 150 | **114 / 128** | **120 / 125** | **122 / 133** | **121 / 133** | **110 / 116** | **122 / 124** |

Inkling-2-Mini: QEngine 1.6.0 development build (November 25, 2026), 2 cards (GPUs 0 and 2), server defaults; with thinking
off it runs at reasoning effort `none` and still writes a short thought, so the club tools mark that leg as not
thinking-free. No engine errors and no graph fallbacks in any of the runs. Earlier releases published the mean of three runs; from
1.5 on it is a single run, because running it for all models takes a long time.

The earlier result, Flash-Next on QEngine 1.3.6 (November 15, 2026), three full runs, each on a fresh server.
Scenarios passed: run 1 / 2 / 3, mean pass@1, mean pass@3 (up to 3 attempts per scenario):

| pack | N | thinking off: runs | mean | pass@3 | thinking on: runs | mean | pass@3 |
|---|---:|---|---:|---:|---|---:|---:|
| toolcall | 15 | 13 / 13 / 13 | 13.0 | 13.0 | 14 / 14 / 13 | 13.7 | 14.0 |
| instructfollow | 15 | 15 / 13 / 13 | 13.7 | 13.7 | 14 / 15 / 15 | 14.7 | 15.0 |
| structoutput | 15 | 13 / 14 / 14 | 13.7 | 14.0 | 13 / 13 / 13 | 13.0 | 13.7 |
| dataextract | 15 | 11 / 9 / 11 | 10.3 | 11.0 | 11 / 13 / 12 | 12.0 | 13.3 |
| reasonmath | 15 | 13 / 15 / 15 | 14.3 | 14.3 | 15 / 15 / 14 | 14.7 | 14.7 |
| bugfind | 15 | 15 / 14 / 15 | 14.7 | 15.0 | 15 / 15 / 14 | 14.7 | 15.0 |
| hermesagent | 20 | 14 / 14 / 14 | 14.0 | 16.7 | 14 / 15 / 14 | 14.3 | 17.0 |
| cli | 40 | 31 / 29 / 30 | 30.0 | 35.0 | 29 / 30 / 29 | 29.3 | 32.3 |
| **total** | 150 | 125 / 121 / 125 | **123.7** | 132.7 | 125 / 130 / 124 | **126.3** | 135.0 |

No engine errors, no empty replies and no graph fallbacks in the three runs. On 1.3.5 (November 6) the same suite gave
120.7 / 126.7 (thinking off / on).

AutomationBench sales (Zapier, 100 tasks, a 1.3.4 pre-release build with default settings, Flash-Next on 3 cards,
16 at once, November 1, 2026): 47 % of tasks passed, mean partial credit 0.78; the 5 zero-score tasks were wrong
decisions of the model, none came from the server. The image encoder was checked against Hugging Face transformers
and the INT8 n-gram table against the BF16 one (next-token KL); see
[docs/advanced.md](docs/advanced.md#measurements).

### Measure your own setup

```bash
python3 -m pip install -r requirements-dev.txt
python3 bench/serving_suite.py --engine-name qengine --base-url http://127.0.0.1:8080 \
    --model qwen3.8-flash-next-qengine --tokenizer models/Qwen3.8-Flash-Next-QE-INT4 --quick
```

`--quick` runs short requests at 1 at once plus two live-traffic scenarios. For the 27B use
`--model qwen3.8-27b-qengine` (no `--tokenizer`). The ShareGPT scenario needs
`ShareGPT_V3_unfiltered_cleaned_split.json` from
[anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
in `bench/data/`.

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
exchanges (see [With and without GPU P2P](#with-and-without-gpu-p2p)). For long-prompt work on Flash-Next try
`FLASH_PAR=pp`, which needs no fast link.

**Long prompts fail with `non-finite values`, or the log shows `GUARD` lines.** Set `FLASH_COMM=ce` (27B:
`QE_COMM=ce`) and report the `[comm]` start-up lines.

**Slow decode, or poor scaling with concurrent requests, on a board with a high P2P co-run tax.** Look at the
`[comm] decode collectives` start-up line: it shows the measured ms per decode step on both paths and the pick for
one and for several sequences. If the two timings are close, or your own runs disagree with the pick, compare
`QE_DECODE_COMM=host` and `QE_DECODE_COMM=p2p` at 1 and 4 concurrent requests and report the `[comm]` lines with
both numbers. Before 1.3.5, decode always used P2P stores when peer access worked; `QE_P2P=0` was the only way off
them, and it also moves prefill to the slower host-staged path. On 1.3.5 to 1.3.5.5 the host candidate was slower than
`QE_P2P=0` itself; on such a board use 1.3.5.6 or later, or `QE_P2P=0` on the older versions.

**`not enough GPU memory for a KV cache` (27B).** Too many slots for the free VRAM, e.g. on one card: lower
`QE_MAX_SEQS`, leave `--vision` off.

**A desktop on one of the cards stutters.** Flash-Next `FLASH_EC_MARGIN_MB=256,2048,256`, 27B
`QE_RESERVE_GB=1.5,3,1.5` (the larger value at that card's position).

**The server says another engine holds the GPU.** Each card serves one engine process. Stop the other one, or wait if
it is still exiting after a crash; check `dmesg` for Xid lines.

**The server restarted with exit code 75.** The engine stalled (`collective timeout`) or the watchdog saw no progress
(`[hang]`). The open requests got an error first. Please report the log, which holds the diagnostics.

**Does it run on Windows / WSL2?** Untested. WSL2 has no GPU P2P and limits pinned host memory.

**Is greedy output reproducible?** Flash-Next: not by default. At temperature 0, two runs can differ where two tokens are nearly tied, depending on what shares a step and on what ran on the cards before. `FLASH_ATTN_CHUNK=1` makes the same request give the same text whatever shares a step and whatever state the machine is in; use it to compare runs, reproduce a report or run tests. It does not make answers closer to BF16, and it costs speed (see [docs/advanced.md](docs/advanced.md)). It holds per setup: a different number of cards can give different text, and I didn't measure it across cache hits and misses. I found the class of the cause (float sums in the decode path), but I can't explain why earlier load on the cards changes the result. For one conversation run through long prefills, add `FLASH_DETERMINISTIC=1 FLASH_LOOKUP=0`. 27B: not bit for bit across cache hits and
misses, batch sizes or shared prefill batches (different kernels split near-ties differently). On every engine this
includes conversations that share their beginning with one the server still holds (the same system prompt, for
example): the shared part is reused, the rest is computed in a call of another size, and where two tokens are nearly
tied the reply can differ from the same request sent to a freshly started server. Greedy output is reproducible within
one installation: the 27B times its GEMM kernels once and keeps the choice in `~/.cache/qengine`, so a fresh tune (a
new version, a cleared cache, or a container without a persistent `~/.cache/qengine` volume) can change tokens where
two are nearly tied.

More symptoms and log lines: [docs/advanced.md](docs/advanced.md#troubleshooting).

## Repository layout and tests

```
src/              27B engine: slots, prompt cache, MTP, scheduling, per-GPU worker, C API; kernels in src/kernels/
src/flash/        Qwen3.8-Flash-Next engine: libqflash.so, flash_run, kernels, pipeline modes
src/dsv4/         DeepSeek-V4.1-Flash engine (libqdsv4.so, build.sh): GGUF reader, CPU expert lane, DSpark drafts
src/glm/          GLM-5.3-Flash engine (libqglm.so, from the CMake build): host-RAM experts, CPU lane, MTP
src/gptoss/       gpt-oss-120B engine (libqgptoss.so, build.sh): MXFP4 experts, harmony format (server/harmony.py)
server/           app.py (OpenAI + Anthropic API, tool calls, grammars, scheduling), engine.py (ctypes),
                  model_profile.py (per-model defaults), think_budget.py, vision.py, steering.py,
                  dashboard + qetop.py (monitoring), templates/ (chat template), run.sh
tests/            regression tests (engines, server, kernels)
tools/            model conversion (convert_flash_next.py, convert_ngram_int8.py, convert_ngram_fp8.py), reference dumps, draft vocabulary,
                  GLM-5.3-Flash data files (tools/glm/)
bench/            serving_suite.py and the benchmark harness
docs/             advanced.md: more settings, measurements, steering, troubleshooting
```

The tests in `tests/` check invariants of the engines (cached, batched, speculative and archived paths reproduce a
fresh computation; sampling keeps its distribution; kernels match torch). CPU-only examples:

```bash
python3 -m pip install -r requirements-dev.txt
python3 tests/test_server_mock.py && python3 tests/test_think_budget.py   # server end to end on a mock engine
python3 tests/test_grammar.py && python3 tests/test_toolcall_parse.py
```

Most other tests need the GPUs and a model (`QE_TEST_MODEL`, `QE_LIB=build/libqflash.so` for Flash-Next).

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
