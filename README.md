# QEngine

![QEngine: fast local AI on RTX 3000, 4000 and 5000 series cards](docs/img/qengine.jpg)

<p align="center">
  <a href="https://buymeacoffee.com/bpx03"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="60"></a>
  <br>
  <sub>Built and tested on one home machine. If QEngine helps you, a coffee keeps it going.</sub>
</p>

QEngine is a source-available inference engine for two models on NVIDIA RTX 3000, 4000 and 5000 series cards, with
nothing extra to install or set for any of them (tuned on the RTX 3090):

- **Qwen3.8-Flash-Next** (routed experts INT4 from Intel's AutoRound checkpoint, the parts every token passes through
  in INT8 or calibrated INT4; converted for QEngine): a sparse mixture-of-experts model with 512 routed experts, 262K
  context (up to 1M opt-in) and image input. Runs on 2 or more cards (tuned for 3 and 2).
- **Qwen3.8-27B** (INT4, the RedHatAI W4A16 checkpoint plus its MTP head): a dense model with 262K context (up to 1M
  opt-in, with optional sparse attention for long prompts) and image input. Tuned for 3 cards, also runs on 2, and on 1 as an experiment.

One server serves either model through an **OpenAI-compatible** and an **Anthropic-compatible** API (it works with
agent clients such as Claude Code). The C++/CUDA engines are built as shared libraries; the Python server loads them
and handles the API, tool calls, structured output, scheduling and caching.

**Measured on our machine** (3× RTX 3090, 24 GB, PCIe 4.0 x16, GPU P2P on; "2 cards" are two of its cards), with
the default settings:

| output tok/s, server, default settings | Flash-Next, 3 cards | Flash-Next, 2 cards | 27B, 3 cards |
|---|---:|---:|---:|
| single-stream decode, narrative / code | 313 / 411 | 209 / 221 | 274 / 423 |
| prefill, 10K / 90K-token prompt (tok/s) | 6,186 / 7,249 | 3,797 / 6,357 | 6,437 / 5,159 |
| 8 agents at once (total output) | 867 | 285 | 1,205 |

(1.3.5, November 6, 2026; method under [Benchmarks](#benchmarks).)

**What is new in 1.3.5:** mostly fixes, from every open issue and my own code reviews: a use-after-free in the 27B archive, disk cache fixes, a reply parser that no longer depends on where a decode step ends, stop strings that no longer leak, one eviction order for both models, and memory taken by other programs no longer takes the server down. New: 27B context up to 1M (`--max-ctx`, opt-in), sparse attention for long prompts (`--lsa`, opt-in), faster Flash-Next prefill, and a decode transport that picks itself per machine. `--disk-cache` on Flash-Next now stops the start with an error ([release notes](https://github.com/bpx03/QEngine/releases/tag/v1.3.5)).

**What is new in 1.3.4.3:** the default thinking budget can be set per reasoning effort
(`QE_THINKING_BUDGET=low:4096,medium:16384,xhigh:32768`, see [Thinking budget](#thinking-budget-and-the-answer-tail);
`/v1/messages` also reads Anthropic's `output_config.effort`). Two fixes: an idle Flash-Next server no longer keeps
CPU cores busy (about 2 cores before, about 0.02 now, on 3 cards), and after the load the engine releases the page
cache of the weight files, so they no longer push the n-gram table out of RAM when it is read through the page cache
([release notes](https://github.com/bpx03/QEngine/releases/tag/v1.3.4.3)).

**What is new in 1.3.4.2:** faster Flash-Next decode (+14 % with 8 agents, +7 to +12 % single-stream; one
conversation's output bit for bit as 1.3.4.1), the web dashboard's agents table and `qetop` work again, and both
show the running version and build ([release notes](https://github.com/bpx03/QEngine/releases/tag/v1.3.4.2)).

**What is new in 1.3.4.1:** new default Flash-Next weights (the DeltaNet projections in INT8, the shared expert and
the n-gram projections calibrated; KL against BF16 0.119 instead of 0.267), the CUDA-graph room under agent load
(`FLASH_GRAPH_MB`), and a steering vector that cannot be loaded no longer stops the server
([release notes](https://github.com/bpx03/QEngine/releases/tag/v1.3.4.1)). The 1.3.4 weights stay available
(`--revision v1.3.4`, see [Model files](#qwen38-flash-next)).

**What is new in 1.3.4** (details in the [release notes](https://github.com/bpx03/QEngine/releases/tag/v1.3.4)):

- fast tensor parallelism **without GPU P2P** (the stock NVIDIA driver) for both models;
- `FLASH_PAR`: Flash-Next can run as a **layer pipeline** over the cards; at start the server picks tensor or pipeline
  parallel from a link probe and a per-card calibration, splits uneven cards by speed, and runs 4 or more cards as a
  pipeline;
- **host-memory defaults that fit the machine**: the start-up plans host RAM from what the machine has, so smaller
  hosts no longer swap;
- a hard **thinking budget** and a reserved **answer tail**, so a reply never ends inside its reasoning;
- **empty replies are sampled again** (non-streamed requests), clearer logs for empty turns and dropped history;
- an overlong prompt gets **OpenAI's context-length error text** on the OpenAI path;
- a stalled engine fails its requests at once and restarts; prompts that can never start fail at once with the reason;
- the Docker images carry code for the **RTX 3000, 4000 and 5000 series** (tuned on the 3090).

**Better on non-standard setups.** QEngine is tuned on 3× RTX 3090 with GPU P2P and plenty of host RAM; 1.3.4 is
aimed at setups that differ from that, and should improve them:

- **no P2P** (stock driver, x4 / x8 slots, dual-socket or NUMA): Flash-Next on 3 cards, 10K-token prefill 3.75 →
  1.83 s, 90K 34.95 → 17.03 s, decode within about 4 % of P2P; 27B, 16K prefill 3.94 → 2.68 s
  ([details](#with-and-without-gpu-p2p));
- **long prompts on slow links**: `FLASH_PAR=pp`, a layer pipeline, picked automatically where it wins
  ([Parallel mode](#parallel-mode-flash-next-flash_par));
- **small host RAM** (e.g. 64-96 GB): the host-memory plan adapts; no swapping with the defaults
  ([Host memory](#host-memory-flash-next));
- **mixed or uneven cards**: a per-card calibration and a layer split by speed;
- **4 or more cards**: pipeline mode; `FLASH_PAR=pptp` is experimental;
- **RTX 3000, 4000 and 5000 series**: built in, nothing to set (tuned on the 3090; [what has run where](#gpus)).

If a setup like this is still slow or misbehaves, please open an issue with the start-up log, started with
`FLASH_BOX_REPORT=1` (it adds `[box]` lines on each card and link; on 4 or more cards they are on by default): those,
the `[qflash] parallel mode` line and the `host RAM plan` line are what we need.

1.3.3.1 fixed a case where a reply could pick up text from another conversation running at the same time; if you run
an older version, please upgrade.

## Contents

- [Quick start](#quick-start)
- [Hardware and interconnect](#hardware-and-interconnect)
- [Models](#models)
- [API](#api)
- [Serving behaviour](#serving-behaviour)
- [Settings](#settings)
- [Benchmarks](#benchmarks)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
- [Repository layout and tests](#repository-layout-and-tests)
- [Support](#support)
- [Changelog, license, disclaimer](#changelog)

More settings, measurements and the troubleshooting table: [docs/advanced.md](docs/advanced.md).

## Quick start

You need Linux, two or three RTX 3000, 4000 or 5000 series cards (see [Hardware](#hardware-and-interconnect)), an
NVIDIA driver with CUDA 13.1
support, and enough host RAM (Flash-Next: about 140 GB for the full defaults; with less, the start-up sizes its host
memory to the machine, from 64 GB up; see [Host memory](#host-memory-flash-next)).

### 1. Get a model

```bash
# Qwen3.8-Flash-Next, ready to run (~129 GB): converted weights, INT8 n-gram table and three small files
hf download bpx03/Qwen3.8-Flash-Next-QEngine --local-dir models/Qwen3.8-Flash-Next-QE-INT4
# (the previous, 1.3.4 weights: add --revision v1.3.4; see "Model files" under Models)

# Qwen3.8-27B (incl. its MTP head); two small tuned files are fetched automatically at the first start
hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir models/Qwen3.8-27B-INT4
```

### 2a. Run with Docker

Requirements on the host: the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/). One image serves both
models; it picks the engine from the model's `config.json`.

```bash
docker run -d --name qengine --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -p 127.0.0.1:9393:9393 \
  -v /path/to/models:/app/models:ro -v qengine-cache:/root/.cache/qengine \
  -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 \
  ghcr.io/bpx03/qengine:latest
```

- Without `QE_MODEL` the container serves `/app/models/Qwen3.8-27B-INT4`.
- Images: `latest` / `1.3.5` (text only, ~0.4 GB download) and `vision` / `1.3.5-vision` (with the image encoder, ~3.2 GB download).
  Earlier releases stay available under their version tags.
- `QE_GPUS=0,2` picks the GPUs (host ids with `--gpus all`; default `0,1,2`; `QE_DEVICES`, if set, wins over `QE_GPUS`). Settings are environment variables
  (`-e FLASH_PAR=pp`); anything after the image name, or `QE_EXTRA_ARGS`, goes to the server (`--think 0`,
  `--vision 1`).
- Keep the `qengine-cache` volume: it holds the GEMM tuning table, the downloaded 27B files and the stats.
- `--ipc=host` and `--ulimit memlock=-1` are needed for the pinned host memory. On SELinux hosts (Fedora, RHEL) add
  `--security-opt label=disable`.
- Port 8080: the API. Port 9393: monitoring dashboard, `/api/stats`, Prometheus `/metrics`.
- Compose: `QE_MODELS_DIR=/path/to/models docker compose up -d` (see `docker-compose.yml`).
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
| 2 cards | `QE_DEVICES=0,2 server/run.sh 8080 --model <dir>` | Flash-Next keeps about half of its experts in host RAM; the 27B serves 32 slots with `QE_MAX_SEQS=32` |
| 1 card (27B only, experimental) | `QE_DEVICES=0 QE_MAX_SEQS=1 server/run.sh 8080` | one conversation, ~112K tokens of context; `QE_MAX_SEQS` up to 4 |
| a card also drives a desktop | Flash-Next `FLASH_EC_MARGIN_MB=256,2048,256`; 27B `QE_RESERVE_GB=1.5,3,1.5` | more free VRAM on that card (values in GPU order) |
| 4 or more cards (Flash-Next) | `QE_DEVICES=0,1,2,3 server/run.sh 8080 --model <dir>` | runs as a pipeline (`FLASH_PAR=pp`); see [4 or more cards](#4-or-more-cards) |
| long documents, short answers (Flash-Next) | `FLASH_PAR=pp` | pipeline mode: faster long prompts, slower single-stream decode |
| many long conversations (Flash-Next) | `FLASH_KV=ram FLASH_KV_RAM_TOKENS=4M` | KV cache in host RAM, ~74K tokens per GB |

The first start on a new card count tunes the GEMMs and caches the table in `~/.cache/qengine`.

### 4. First request

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-qengine",
  "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
  "max_tokens": 1024
}'
```

The served names are `qwen3.8-flash-next-qengine` and `qwen3.8-27b-qengine` (`--served-name` changes it).
For **Claude Code**, set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` and `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.

## Hardware and interconnect

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

**Smaller cards** (measured on our 3090s with all but 12 GB or 8 GB of each card taken by another process before the
start, November 3, 2026; defaults otherwise, 16 conversation slots; c=1 / c=4: greedy decode of 400-token replies,
thinking off; prefill of an 11.5K-token prompt; a 53K-token needle):

| model | cards x free VRAM | context | c=1 / c=4 tok/s | prefill tok/s | needle |
|---|---|---:|---:|---:|---|
| Flash-Next | 3 x 12 GB | 262,144 | 66.5 / 145.1 | 4,612 | pass |
| Flash-Next | 3 x 8 GB | 262,144 | 47.2 / 91.1 | 3,819 | pass |
| Flash-Next | 2 x 12 GB | 262,144 | 40.7 / 78.7 | 3,546 | pass |
| Flash-Next | 2 x 8 GB | 65,536 (fitted at start) | 34.7 / 55.1 | 2,263 | pass |
| 27B | 3 x 12 GB | 178,865 (KV pool fitted at start) | 215.4 / 641.1 | 6,571 | pass |
| 27B | 3 x 8 GB | 44,675 (9 slots, fitted at start) | 220.6 / 619.3 | 6,288 | pass (26K) |
| 27B | 2 x 12 GB | 30,720 (14 slots, fitted at start) | 171.6 / 340.5 | 4,678 | pass (18K) |
| 27B | 2 x 8 GB | does not start (the weights do not fit) | - | - | - |

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

| model | cards | how |
|---|---|---|
| Qwen3.8-Flash-Next | 3 (tuned), 2 | tensor parallel or pipeline, picked at start (`FLASH_PAR=auto`) |
| Qwen3.8-Flash-Next | 4 or more | pipeline (`FLASH_PAR=pp`); tested only as 4 pipeline stages on our 3 cards |
| Qwen3.8-Flash-Next | 1 | not supported |
| Qwen3.8-27B | 3 (tuned), 2 | tensor parallel |
| Qwen3.8-27B | 1 | experimental, `QE_MAX_SEQS` 1 to 4 |

GPU selection: `QE_DEVICES=0,2` (physical ids; Docker `QE_GPUS`), or `--devices` (indices within
`CUDA_VISIBLE_DEVICES`). By default every visible GPU is used. Each engine locks the GPUs it uses: a second engine on
a card in use stops at start with the first one's pid and command line (`QE_ALLOW_SHARED_GPU=1` skips the check).

### With and without GPU P2P

GeForce cards have no GPU-to-GPU P2P with the stock NVIDIA driver. A community patch of the open kernel modules
([tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules), with Resizable BAR on)
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
| tensor parallel | `tp` | every layer split over the cards; all cards work on every token, exchanges in every layer (P2P, or host-staged without it) | 2-3 equal cards: fastest decode and short prompts, with or without P2P | yes: 2-3 equal cards | `FLASH_PAR=tp`; `FLASH_COMM` sets the exchange path |
| pipeline | `pp` | each stage holds a contiguous range of whole layers; prompts flow through in micro-batches, several conversations decode as groups moving stage to stage; only a 40 KB row per token crosses between stages, so it needs no P2P | long uncached prompts with short replies (RAG, long documents); uneven cards; 4 or more cards | yes: 4+ cards, and uneven cards (or 2-3 cards without P2P) when it wins the mix | `FLASH_PAR=pp`; `FLASH_PP_LAYERS=17,17,14` sets layers per stage; a card may hold several stages (`--devices 0,1,2,0`; a test layout, it adds no compute) |
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
| 4 or more cards | leave `auto` (= `pp`); exactly 4 cards with P2P between all of them: try `FLASH_PAR=tp` ([why](#4-or-more-cards)); if your cards pair up on P2P, try `FLASH_PAR=pptp FLASH_PP_MAP=0,1\|2,3` |
| image input, steering vectors or `FLASH_KV=ram` needed | `FLASH_PAR=tp` (2-3 cards) |

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

**What `pp` and `pptp` do not serve**: image input (refused with a clear message), steering vectors (not loaded),
`FLASH_KV=ram` (on 2 or 3 cards the server runs `tp` instead and logs it; on 4 or more cards the start is refused with
a message), activation capture; lookup drafts are off. Structured output and the thinking budget work as in `tp`
(a constrained step runs through the stages as a batch of one).

#### 4 or more cards

With 4 or more cards, `auto` runs Flash-Next as a pipeline (`pp`). We have no 4-card machine: the 4-way split was
tested as 4 pipeline stages on our 3 cards (two stages on one card), where it gave the same output bit for bit as 3
stages. `pptp` was tested only in a 3-card layout (`0,1|2`), where its next-token probabilities match `tp` within the
run-to-run noise. Reports from 4-card machines are welcome.

**On exactly 4 cards with P2P, try `FLASH_PAR=tp`** (`FLASH_PAR=tp server/run.sh ...`, Docker: `-e FLASH_PAR=tp`).
`auto` always picks `pp` on 4 cards, and its log line reads
`[qflash] parallel mode: pp (auto: 4 cards: mode B); probe: 4 cards, peer access no`. Here `peer access no` does not
mean that your cards lack P2P: `auto` does not open it on 4 cards. With the forced mode the line reads
`tp (forced by FLASH_PAR=tp); probe: 4 cards, peer access yes`. One user measured this on one rig on 1.3.4.3
(4× RTX 5080 16 GB behind one PCIe switch): decode at 8K / 32K tokens went from 116 / 123 to 237 / 303 tok/s, about
2×; prefill was about even at 8K and slower at 32K (7,367 to 5,466 tok/s). We cannot test 4 cards ourselves, so
take this as one report, not a guarantee. `tp` may also not fit: on another user's 4 × 20 GB cards it could not load.
In that case go back to `auto`. `tp` runs on at most 4 cards.

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
SM stores into the other cards' memory (P2P) or the same exchange through pinned host memory, with the same results.
Which one is faster depends on the board: peer stores can stall the other kernels running on a card, and only the
real model shows by how much. With `auto` both engines time a few real decode steps on each path at the end of the
start-up (one sequence, and four batched sequences), keep the faster path for each, and print both timings on the
`[comm] decode collectives` line. `QE_DECODE_COMM=p2p` or `host` forces the decode path (both engines; prefill keeps
its own path). The measurements are cached in `~/.cache/qengine` (the Docker volume) and reused while the cards, links, driver and
build stay the same; `QE_COMM_CACHE=refresh` measures again.

Set `FLASH_COMM=ce` (27B: `QE_COMM=ce`) yourself when the `[comm]` line says P2P but:

- long prompts fail with `non-finite values` (HTTP 500) or the log shows `GUARD` lines;
- a card sits behind something the link width does not show (a PCIe switch, a riser or extension cable, a chipset
  slot, a bifurcated slot);
- the cards differ (e.g. a 3090 next to a 3090 Ti) or sit in slots of different PCIe generations, and the line shows a
  P2P co-run tax of about 1.9 or more;
- prefill is clearly slower than the numbers here.

The copy engine gives the same output; on fast x16 links P2P is a few percent faster in prefill.

**NUMA.** On a machine with several NUMA nodes, Flash-Next places pinned host memory by GPU (each card's host-backed
experts and staging buffers on its node, the n-gram table interleaved over the GPUs' nodes).

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
about 85 GB; 27B started with 24 GB free (the lowest we tried). Below its minimum Flash-Next stops at start with a
message naming the shortfall, before it pins anything.

| part | host RAM |
|---|---|
| host-resident experts (what VRAM cannot hold) | about 12 GB (3 cards), about 31 GB (2 cards) |
| n-gram embedding table (INT8, pinned) | 51.8 GB |
| conversation archive (`FLASH_KV_ARCHIVE_GB`) | 48 GB, filled as conversations leave VRAM |
| pinned system prompts (`FLASH_PREFIX_GB`) | 8 GB, as agent kinds appear |
| server process (Python, vision encoder, CUDA) | about 5 GB |
| `FLASH_KV=ram` (opt-in) | 13.5 GB per 1M tokens of KV |

**The defaults fit the machine (since 1.3.4).** At start-up the engine plans its host memory from `MemTotal` and
`MemAvailable` (and a cgroup / `docker --memory` limit, if one is set). It first sets aside a reserve for the system
and the desktop: max(8 GB, 8 % of RAM). The rest goes, in this order, to:

1. the host-resident experts, which are not optional;
2. the n-gram table. It is pinned when it fits next to an 8 GB minimum archive. Otherwise it is read through the page
   cache (as with `FLASH_NGRAM_PIN=0`, below), and part of the remaining RAM stays free for the page cache to hold
   its rows;
3. the archive and the pinned system prompts, which get what is left, up to 48 + 8 GB.

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
set does not fit, the log says so and the value is kept. `QE_HOST_BASE_GB` (6) changes the share counted for the
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
  when other programs need it.
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
- MTP drafts (k = 3) plus lookup drafts, steps launched ahead, CUDA graphs;
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
| attention KV cache | FP8 (E4M3) | |
| MTP draft head | INT8, group 128 (experts and shared expert INT4) | attention and fc BF16 in the files, converted at load |

To build the directory yourself from Intel's checkpoint (169 GB download), see
[docs/advanced.md](docs/advanced.md#build-the-flash-next-model-directory-yourself).

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

- INT4 weights repacked at load (a few seconds, no conversion) into tensor-core layouts; `lm_head` in INT8.
- Tensor parallel over PCIe; the four full-attention KV heads are split by position (3 cards with P2P), so no KV head
  is duplicated. All-reduces run inside the kernels and inside CUDA graphs.
- MTP speculative decoding with 1 to 7 drafts per step, chosen from the measured acceptance; drafts over the 64K most
  frequent tokens, verification over the full vocabulary, so outputs are unchanged. Lookup drafts on the GPU.
- Up to 32 conversations (`QE_MAX_SEQS`, default 16); KV cache in FP8 in one pool per card (about 1M tokens on 3
  cards with P2P); 262,144 tokens per conversation.

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

#### Long documents (27B, opt-in)

Above 262K tokens the 27B needs YaRN (`--max-ctx 512K` or `1M`), and for prompts that long most of the time goes to
attention; `--lsa MAP` makes that attention sparse. Both are off by default. YaRN is static: it applies to every
request of that server, short ones too, so run a separate server for long documents. What `--lsa` changes
(3× RTX 3090, measured November 3, 2026; details in [docs/advanced.md](docs/advanced.md#sparse-attention-for-long-prompts-opt-in)):

| you gain | you give up |
|---|---|
| 512K prompt about 2× faster (225 → 103–114 s), 1M about 3× (736 → 239 s) | exactness: each token reads 2,048–4,096 of its keys, which carry about 60–80 % of the full attention weight in our measurement |
| decode at 1M about 2× (106 → 227 tok/s), at 512K 1.5× | tested only on 2 retrieval tasks at 128K and 512K plus one 1M needle (same answers as full attention): not RULER, not reasoning quality over long documents |
| nothing changes below a prompt position of ~64K or a context of ~192K (full attention runs there) | only one conversation at a time benefits (the last long prompt's); others, and requests on a pinned shared system prompt, use full attention |
| | about 0.5 GB per card less KV pool; needs P2P and the FP8 KV cache; at most 1.87M tokens |

### Both models: context and the KV archive

- `--max-ctx` (default 262,144) is the context of one conversation; the KV pool of all conversations together is
  sized from free VRAM.
- Conversations that leave VRAM go to a **host-RAM archive** (`FLASH_KV_ARCHIVE_GB` 48, `QE_KV_ARCHIVE_GB` 32; smaller
  on machines with less RAM, see [Host memory](#host-memory-flash-next)): returning to a 50K-token conversation is a
  copy, not a recompute.
- **Pinned system prompts**: the KV and state at the end of each agent kind's system prompt and tools are kept apart
  from the archive (`FLASH_PREFIX_GB` / `QE_PREFIX_GB`, 8 GB), so a new agent of that kind starts after its system
  prompt even when the archive is full.
- There is no disk cache by default; the 27B has an optional one (`--disk-cache DIR`). Flash-Next has none, so its
  conversations do not survive a restart: it accepts the flag and warns at start.

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
followed by a literal `</think>`.

**Structured output**: `response_format` with `json_object` or `json_schema`, enforced in the engine (llguidance
grammars) together with speculative decoding. Stop sequences are supported.

**Sampling**: the model's `generation_config.json` defaults (the 27B uses temperature 0.6), overridable per request
or with `--temperature`, `--top-p`, `--top-k`, `--min-p`. `presence_penalty` and `frequency_penalty` are applied on
the GPU (server defaults: `--presence-penalty`, `--frequency-penalty`).

**Thinking and reasoning effort**: thinking is on by default (`--think 0` turns it off by default;
`chat_template_kwargs.enable_thinking` per request). `reasoning_effort` (`xhigh` = the template default, `high` is the
same, `medium`, `low`) per request or as a server default with `--reasoning-effort` / `QE_REASONING_EFFORT`; a
request's value wins. `max` is taken as `xhigh` and `minimal` as `low` (the names other APIs use); other unknown
values render as `medium`. On `/v1/messages` the Anthropic `output_config.effort` is used when the request sends no
`reasoning_effort` (so `max` gets the most reasoning).

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
sentence), and the model goes on with its answer or tool call. The reasoning holds at most the budget plus one decode
step plus the wrap-up.

`max_tokens` counts thinking and answer together. **Answer tail**: when `max_tokens` would run out inside `<think>`,
the same close comes early enough to leave the answer the last eighth of `max_tokens` (64 to 2048 tokens;
`QE_THINK_TAIL`, a token count, `0` = off), with or without a budget (not with `ignore_eos`). `finish_reason` is `length` only when the answer
itself is cut off; `usage.completion_tokens_details.reasoning_tokens` reports the thinking tokens.

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
counts every try; `finish_reason` is the last try's.

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
of the running conversations; short requests can cut in between the chunks of a long prefill. A request that does not
fit the KV pool lets smaller ones start first, until it has waited `--backfill-wait` seconds (30).

**Agents.** Requests are grouped by the client's session id (Claude Code `metadata.user_id`) and by message history
(a client without a session id: by system prompt);
the **main agent thread** (the conversation that receives the others' results) goes first in the prefill queue, on a
long subagent prefill in progress, and is evicted last while it is active. When subagents fill the pool, the main thread is admitted
at once by pausing the lowest-priority decoding conversations; they resume where they stopped. `--main-agent-re`
forces the main thread by a regex on the system prompt or tool names.

**KV pool under pressure** (both models). Idle conversations go to the host-RAM archive first. The archive frees memory
in one order: least recently used whole conversation first, the main thread used in the last 5 minutes after all the
others, with no guarantee: when nothing else is left it goes too, with a log line. On Flash-Next the same order frees
VRAM, a generating conversation never leaves VRAM, and an idle main thread never makes a request wait. On Qwen3.8-27B
VRAM slots free finished conversations first and the main thread last. When the running conversations still do not fit, the lowest-priority one
is paused: its KV goes to the archive and its stream pauses without an end token; it resumes exactly where it stopped
once the others keep their headroom. A reply ends early only when a single conversation is larger than the pool.

**Speculative decoding.** MTP drafts plus lookup drafts (when recent tokens repeat an earlier part of the
conversation, as in file edits). Greedy and sampled outputs keep the model's distribution.

**Loops.** A reply whose last 600 tokens are one short fragment repeated is ended (`--loop-guard-tokens`). When an
agent keeps repeating earlier turns, whether one call or a short cycle of a few calls (same tool, exactly the same
arguments apart from a Bash `description`), each getting exactly the same result as before, for 3 turns
(`--loop-guard-turns`; calls that differ only in an id, such as a batch of deletes, and polls whose output changes are
not a loop), the next turn is generated with reasoning on even when thinking is off (at most 2048 reasoning tokens,
`--loop-guard-think`; the cached prompt is kept), with a short note on the tool result, a presence penalty on the
repeated reply and no lookup drafts. The reasoning comes on top of the request's `max_tokens`, and stop sequences apply
only to the answer. A guarded turn thinks even when the client disabled thinking (`enable_thinking: false`, `thinking: {"type": "disabled"}`,
`thinking_budget: 0`); `--loop-guard-think 0` turns this off. A long stretch of unchanged results can be legitimate waiting, so the guard backs off: it applies at 3,
6 and 12 turns, then every 24 turns. Right after a user message it only turns reasoning on. Conversations without tool results are never flagged. `"loop_guard": false` in a request turns it off.

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
monitor `server/qetop.py`. Both show the running version and build in the header (e.g. `QEngine 1.3.4.3 (build abc1234)`).

## Settings

The defaults are the best configuration we measured; most settings exist to fit a machine or to turn a feature off.
Environment variables are read at start; an empty value counts as unset. A value per card is a comma list in GPU order
(a short list repeats its last value). The long tail is in [docs/advanced.md](docs/advanced.md).

### Both models

| setting | default | what it does |
|---|---|---|
| `server/run.sh <port>` | - | starts the server on `127.0.0.1:<port>`; extra arguments go to `server/app.py` |
| `--model <dir>` (Docker `QE_MODEL`) | `models/Qwen3.8-27B-INT4` | the model directory; the engine is picked from its `config.json` |
| `QE_DEVICES` (Docker: `QE_DEVICES`, else `QE_GPUS`) | all visible GPUs (Docker `0,1,2`) | physical GPU ids, e.g. `0,2` |
| `--max-ctx` | `262144` | context per conversation (`256K`, `512K`, `1M` accepted); above 262,144 turns on YaRN RoPE scaling (factor context / 262,144; Flash-Next also needs `FLASH_KV=ram`; 27B: up to what the VRAM pool holds, about 1.2M on 3 cards, 580K on 2) |
| `--lsa MAP` | off | 27B: sparse attention for long prompts (opt-in, lossy): 512K prompts about 2x faster, 1M about 3x, decode at 1M 2x; MAP is the converted index map ([docs](docs/advanced.md#sparse-attention-for-long-prompts-opt-in)) |
| `--think` | `1` | thinking on by default (a request's `enable_thinking` wins) |
| `--reasoning-effort` / `QE_REASONING_EFFORT` | the template's (`xhigh`) | default effort: `xhigh`, `high` / `max` (= `xhigh`), `medium`, `low`, `minimal` (= `low`) |
| `--thinking-budget` / `QE_THINKING_BUDGET` | `0` (off) | default thinking budget in tokens, or one per effort (`low:4096,medium:16384,xhigh:32768[,default:N]`) |
| `QE_THINK_TAIL` | `auto` | room kept for the answer: 1/8 of `max_tokens`, 64 to 2048 tokens; a number; `0` = off |
| `--empty-retry` / `QE_EMPTY_RETRY` | `2` | resample a non-streamed empty reply up to N times; `0` = off |
| `--vision` | `0` | `1`: image input (Docker: the `vision` image) |
| `--temperature`, `--top-p`, `--top-k`, `--min-p` | the model's | sampling defaults for requests that send none |
| `--presence-penalty`, `--frequency-penalty` | `0` | penalties for sampled requests that send none |
| `--served-name` | per model | the model id in the API |
| `--host`, `--port`, `--metrics-port` | `127.0.0.1`, `8080`, `9393` | where the API and the monitoring listen (`--metrics-port 0` = off) |
| `QE_HANG_S` | `180` | server watchdog in seconds; `0` = off |
| `--log-file`, `--log-mb` | `~/.cache/qengine/logs/<name>.log`, `64` | persistent log |
| `QE_STEERING` (Docker) | `0` | `1`: the included steering vector is the default; `<name>`: that vector |

### Qwen3.8-Flash-Next

| setting | default | what it does |
|---|---|---|
| `FLASH_PAR` | `auto` | `auto`, `tp`, `pp`, `pptp` (experimental); see [Parallel mode](#parallel-mode-flash-next-flash_par) |
| `FLASH_PAR_AUTO_P` | `8192` | prompt length `auto` weighs; `32768` favours long-prompt work |
| `FLASH_COMM` | `auto` | prefill transfer path: `auto`, `p2p`, `ce`, `host` (also `pull` / `push`) |
| `QE_DECODE_COMM` | `auto` | decode collectives: `auto` (timed on real decode steps at start-up, cached), `remeasure`, `p2p` (SM stores into the peers), `host` (through pinned host memory); also for the 27B |
| `QE_COMM_CACHE` | `1` | start-up cache of the link probe and decode calibration: `0` off, `refresh` measure again |
| `FLASH_MAX_SEQS` | `16` | conversations served at once (1 to 16); fewer leave more VRAM for the expert cache |
| `FLASH_EC_MARGIN_MB` | `256` | VRAM per card left free next to the expert cache; e.g. `256,2048,256` when GPU1 drives a desktop |
| `FLASH_GRAPH_MB` | `256` | VRAM per card for cached CUDA graphs: 128 MB of it (at most half of `FLASH_EC_MARGIN_MB`) comes out of that margin, the rest is kept on top; `0` = graphs share the margin (1.3.4) |
| `FLASH_PFGRAPH_WINDOW` | `64` | a prompt-shaped graph is captured only when its shape was used within the last N prompt steps, and never evicts a decode graph; `0` = off |
| `FLASH_ATTN_DMIN` | `16` | decode attention of batched steps (more than one wave of blocks): a row uses one key split per N key tiles (at most 8), so short contexts do not pay for 8 splits; `0` = always 8 splits |
| `FLASH_KV` | `vram` | `ram`: KV cache in pinned host RAM |
| `FLASH_KV_RAM_TOKENS` | `1M` | with `FLASH_KV=ram`: KV tokens of all conversations together (`K` / `M` / `G`) |
| `FLASH_KV_ARCHIVE_GB` | `48` (less on smaller machines) | host-RAM archive of conversations; `0` = off; a set value wins over the [host-RAM plan](#host-memory-flash-next) |
| `FLASH_PREFIX_GB` | `8` (less on smaller machines) | host RAM for pinned system prompts; `0` = off |
| `FLASH_NGRAM_PIN` | set by the host-RAM plan (`1` when it fits) | `1`: pin the n-gram table; `0`: read it through the page cache |
| `FLASH_WEIGHTS_COLD` | `1` | after the load, drop the page cache of the model files the engine no longer reads (all but the n-gram table in page-cache mode); `0` = keep it |
| `FLASH_TP_SPIN_US` | `5000` | µs a tensor-parallel helper thread spins after its last job before it sleeps (an idle server then uses no CPU); `-1` = always spin |
| `FLASH_VRAM_ADAPT` | `1` | Flash-Next: when another process takes GPU memory while serving, move resident expert groups to host memory and back later; `0` off |
| `FLASH_VRAM_LOW_MB` / `FLASH_VRAM_TARGET_MB` | `128` / `256` | free memory per card below which groups move to host memory / up to which they move; they come back as soon as the other processes give the memory back (keeping 16 MB above the low mark free) |
| `FLASH_CTX_FIT` | `1` | Flash-Next: lower the context at start to the largest that fits the smallest card (logged); `0` keeps `--max-ctx` as given |
| `FLASH_PRELOAD` | `1` | Flash-Next: when CUDA was started before the engine (the vision encoder) with lazy module loading, load the engine's kernels at start instead of at their first use; `0` off |
| `QE_RAM_WAIT_S` | `20` | seconds the start waits for free host RAM (MemAvailable) when it is short of the full plan (as long as it rises, or while the previous QEngine server on these cards is still exiting with 1 GB or more on them); `0` plans at once |
| `QE_NO_ROOM_S` | `10` | 27B: seconds a queued request that fits nowhere waits with nothing running, then it ends with an error that says why; `0` = no deadline |
| `QE_PAUSE_TIMEOUT_S` | `900` | 27B: seconds a reply may wait for GPU memory another process holds before it ends with an error, counted from when that wait began: one that cannot continue, or a paused one that cannot resume because the KV cache gave memory back; a reply paused only for a higher-priority one has no deadline |
| `FLASH_NGRAM_SPIN_MS` | `50` | page-cache mode of the n-gram table: ms the row-copy thread spins after the last request, then it polls every 50 µs (1 ms after a second); `-1` = always spin |
| `QE_HOST_RESERVE_GB` | max(8, 8 % of RAM) | at start only: RAM the plan leaves to the system and the desktop (both models). `QE_MEM_MARGIN_GB` is the one that acts while serving |
| `FLASH_RUN_TIMEOUT_S` | `120` | engine step monitor in seconds; `0` = off |

### Qwen3.8-27B

| setting | default | what it does |
|---|---|---|
| `QE_MAX_SEQS` | `16` | conversations kept in the KV pool, up to 32; 1 to 4 on one card |
| `QE_RESERVE_GB` | `1.5` | VRAM per card left outside the KV pool; e.g. `1.5,3,1.5` for a desktop on GPU1 |
| `QE_KV_ARCHIVE_GB` | `32` (less below 64 GB of RAM) | host-RAM archive of conversations; `0` = off; a set value wins |
| `QE_PREFIX_GB` | `8` (less below 64 GB of RAM) | host RAM for pinned system prompts |
| `QE_COMM` | `auto` | prefill transfer path: `auto`, `p2p`, `ce`, `host` |
| `QE_FETCH_ASSETS` | `1` | `0`: do not download the tuned MTP head and draft ranking |
| `--disk-cache DIR` | off | 27B: persistent prompt cache that survives restarts. Flash-Next has none: the flag stops the start with an error |

## Benchmarks

All numbers were measured by us on one machine: 3× RTX 3090 (24 GB, PCIe 4.0 x16, P2P through
the patched driver unless noted), one NUMA node; "2 cards" are two of its cards. QEngine runs with its defaults. They
describe that machine; your numbers depend on the cards, links, driver and host.

### Qwen3.8-Flash-Next

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

1.3.5 (November 6, 2026; 3 cards, P2P): single-stream decode narrative / code 274.1 / 422.5 tok/s, prefill 10K / 90K 6,437 / 5,159 tok/s, 8 agents at once 1,205 tok/s. With the club-3090 `bench.sh` settings: 276.1 / 418.9 tok/s. Prefill measured on its own on cooled cards.

Serving benchmark (October 2026, QEngine 1.0), 3 cards, `vllm bench serve` as the client; output tokens/s:

| scenario (input / output tokens) | 1 at once | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| short (1024 / 512) | 208 | 794 | 949 | 1,001 |
| agent (500 / 2000) | 287 | 1,080 | 1,302 | 1,423 |
| ShareGPT | 243 | 819 | 935 | 1,110 |

Later changes (W4A8 GEMM for batched passes of 33 rows or more, less admission headroom) raised agent at 32 to 1,703
and short at 32 to 1,039 tok/s in a same-day A/B. Prefill of one new prompt (median time to first token): 32K tokens
5.35 s, 200K tokens 50.7 s.

Fewer cards (October 2026): 2 cards ~162 tok/s single stream, a ~387K-token KV pool with 32 slots and the full 262K
context; 1 card ~104 tok/s with one conversation (~112K tokens of context), ~96 tok/s with 4 slots (~85K tokens).

### Quality

**Closeness to BF16.** Next-token KL divergence against the official BF16 model on a public reference set (21 cases,
32,359 positions: prose, code, Japanese, Chinese, SQL, agent turns and WikiText-2), scored with the set's own
`compare.py`; every position through QEngine's decode path (forwards of up to 16 tokens) on 3 cards (November 6,
2026; the 1.3.4 row November 2). The last column scores the last 512 positions of the long cases after the context before them went through
the server's batched prefill (our own harness, not part of the command below). Lower is closer to BF16:

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

club-3090 8-pack (benchlocal, 150 scenarios, 4,096-token answer budget per scenario): Flash-Next on 3 cards, vision
on, the club's launch settings; three full runs on QEngine 1.3.5 (November 6, 2026), each on a fresh server. Scenarios
passed: run 1 / 2 / 3, mean pass@1, mean pass@3 (up to 3 attempts per scenario):

| pack | N | thinking off: runs | mean | pass@3 | thinking on: runs | mean | pass@3 |
|---|---:|---|---:|---:|---|---:|---:|
| toolcall | 15 | 13 / 12 / 13 | 12.7 | 14.0 | 13 / 13 / 13 | 13.0 | 14.0 |
| instructfollow | 15 | 14 / 12 / 14 | 13.3 | 14.0 | 14 / 15 / 15 | 14.7 | 15.0 |
| structoutput | 15 | 13 / 12 / 12 | 12.3 | 14.0 | 13 / 13 / 14 | 13.3 | 14.0 |
| dataextract | 15 | 12 / 12 / 10 | 11.3 | 12.7 | 13 / 13 / 13 | 13.0 | 13.7 |
| reasonmath | 15 | 13 / 14 / 15 | 14.0 | 15.0 | 13 / 15 / 13 | 13.7 | 15.0 |
| bugfind | 15 | 15 / 15 / 15 | 15.0 | 15.0 | 14 / 14 / 14 | 14.0 | 15.0 |
| hermesagent | 20 | 12 / 13 / 14 | 13.0 | 15.7 | 15 / 13 / 15 | 14.3 | 17.0 |
| cli | 40 | 28 / 31 / 28 | 29.0 | 32.3 | 32 / 29 / 31 | 30.7 | 33.7 |
| **total** | 150 | 120 / 121 / 121 | **120.7** | 132.7 | 127 / 125 / 128 | **126.7** | 137.3 |

No engine errors in the three runs; one tool call that didn't parse recovered on its retry, and the loop guard fired once.

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
them, and it also moves prefill to the slower host-staged path.

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
misses, batch sizes or shared prefill batches (different kernels split near-ties differently).

More symptoms and log lines: [docs/advanced.md](docs/advanced.md#troubleshooting).

## Repository layout and tests

```
src/              27B engine: slots, prompt cache, MTP, scheduling, per-GPU worker, C API; kernels in src/kernels/
src/flash/        Qwen3.8-Flash-Next engine: libqflash.so, flash_run, kernels, pipeline modes
server/           app.py (OpenAI + Anthropic API, tool calls, grammars, scheduling), engine.py (ctypes),
                  model_profile.py (per-model defaults), think_budget.py, vision.py, steering.py,
                  dashboard + qetop.py (monitoring), templates/ (chat template), run.sh
tests/            regression tests (engines, server, kernels)
tools/            model conversion (convert_flash_next.py, convert_ngram_int8.py), reference dumps, draft vocabulary
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

Free for non-commercial use. QEngine is licensed under the PolyForm Noncommercial License 1.0.0, see
[LICENSE](LICENSE).

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
