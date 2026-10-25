# QEngine

QEngine is a source-available inference engine for one model on one kind of machine:
**Qwen3.8-27B INT4** (the RedHatAI W4A16 checkpoint plus its MTP head) on **three NVIDIA RTX 3090s**
with tensor parallelism 3 over PCIe.

**The fastest engine we know of for this model on RTX 3090s**: 1.2–2.8× vLLM 0.28's output throughput
in our benchmark (see [Benchmark summary](#benchmark-summary)), and well ahead of SGLang, Ollama and LM Studio
in our quick tests.

- C++/CUDA engine (`src/`, about 26k lines), built as a shared library (`libqengine.so`).
- Python server (`server/`) that loads the library through ctypes and exposes an OpenAI-compatible
  API (`/v1/chat/completions`, `/v1/models`) and an Anthropic-compatible API (`/v1/messages`,
  `/v1/messages/count_tokens`). It works with agentic clients such as Claude Code.
- **Experimental preview on this branch:** a second engine for **Qwen3.8-Flash-Next**, see
  [Qwen3.8-Flash-Next (experimental preview)](#qwen38-flash-next-experimental-preview).

## Hardware requirements and limitations

**Read this first.**

- **Designed and tuned for 3× RTX 3090** (24 GB, sm_86, PCIe 4.0 x16, no NVLink): tensor parallel 3,
  GPUs `0,1,2` by default. The weight and KV-head split, kernels and tuning (82 SMs, tile shapes, split
  counts, wave sizing, memory budgets) target this setup; it is the configuration behind the main
  benchmark.
- **2× RTX 3090 also measured**: 1.2–2.0× vLLM's output throughput on the same two GPUs up to 16
  concurrent requests (short, agent and ShareGPT), at 72–85% of the 3-GPU throughput.
- **1× RTX 3090 is experimental**: it starts only with `QE_MAX_SEQS` ≤ 4 (about 85K–112K tokens of context,
  around 100 tok/s for a single user).
- **Device selection**: `QE_DEVICES=0,2 server/run.sh 8080` (physical GPU ids), `--devices` (indices within
  `CUDA_VISIBLE_DEVICES`), or `QE_GPUS` in Docker. Counts other than 3 start with a warning.
- **sm_86 only**: `CMakeLists.txt` builds for `CMAKE_CUDA_ARCHITECTURES 86`. Other GPU models are untested.
- One model only: Qwen3.8-27B INT4 (RedHatAI compressed-tensors W4A16 layout). Other checkpoints or
  quantizations are not supported.
- Linux only (developed on Fedora 44, kernel 7.2). **Windows is untested**, including Docker Desktop / WSL2:
  there is no GPU P2P under WSL2 (the engine would fall back to the slower host-memory path) and pinned host
  memory, which the KV archive relies on, is limited there.

**Peer-to-peer (P2P).** All published numbers were measured with GPU-to-GPU P2P enabled through a
patched open NVIDIA kernel driver (the community P2P patch for GeForce cards:
https://github.com/tinygrad/open-gpu-kernel-modules, with Resizable BAR enabled). With the stock GeForce
driver, P2P is not available; the engine detects this and falls back to a slower path through host
memory (the same as `QE_P2P=0`): all-reduces go through pinned host memory, the position-split KV
layout is disabled (smaller KV pool) and decode and prefill are slower than the published numbers.

## Features

- **INT4 weights, repacked at load time** (a few seconds, no file conversion) into tensor-core fragment
  layouts; INT4 GEMV for decode and INT4 GEMM for prefill; `lm_head` in INT8.
- **Tensor parallel over PCIe** (3 GPUs by default): every layer is split across the cards (DeltaNet heads, MLP
  groups, vocabulary). The four full-attention KV heads are split by position for heads 1 and 2, so no
  KV head is duplicated. All-reduces run inside the kernels (they work inside CUDA Graphs): a low-latency
  protocol for decode and reduce-scatter overlapped with GEMM for prefill.
- **MTP speculative decoding** with the model's MTP head: 1–7 drafts per step, chosen adaptively from
  the measured acceptance and step cost; drafts over the 64K most frequent tokens, verification over the
  full vocabulary, so greedy and sampled outputs are unchanged.
- **Lookup drafts** on the GPU: when the recent tokens repeat an earlier part of the prompt or reply (file
  edits, quoted code), up to 12 drafts are copied from there instead of the MTP drafts, in single and batched
  decoding (`QE_LOOKUP=0` switches them off).
- **Continuous batching** of up to 32 sequences; batched passes of up to 32 rows give the same result as
  decoding each sequence alone (larger passes use the W4A8 GEMM, `QE_DEC_GEMM`, which is numerically close
  but not bit-identical).
- **KV cache in FP8** (e4m3) in one shared pool per GPU (about 1M tokens with P2P), up to 262,144 tokens
  of context per conversation.
- **Prefix / KV reuse**: live slot continuation, DeltaNet state snapshots, prefix copies across slots,
  shared prefix KV for agents with the same system prompt, and a host-RAM archive of evicted
  conversations (returning to a 50K-token conversation is a copy, not a recompute). Optional disk cache
  (`--disk-cache`).
- **Scheduling**: shortest-prefill-first queue with aging, short requests can preempt long prefills
  between chunks, prefill chunks interleaved with running decodes.
- **Structured output**: `response_format` with `json_object` / `json_schema` through llguidance
  grammars, enforced in the engine together with speculative decoding.
- **Tool calls** (OpenAI `tools` and Anthropic `tool_use` / `tool_result`) and thinking on/off with
  reasoning-effort levels (chat template in `server/templates/`).
- **Vision** (`--vision 1`): images in Anthropic and OpenAI messages; the ViT encoder runs in PyTorch on
  the last GPU, M-RoPE positions as in Hugging Face transformers, and image tokens take part in prefix caching.
- **Monitoring**: live web dashboard and Prometheus metrics on a separate port (default
  `http://127.0.0.1:9393/`), terminal monitor `server/qetop.py`.

## Build

Requirements:

- Linux, RTX 3090 GPUs (3 recommended, see above), NVIDIA driver with CUDA 13.1 support.
- CUDA Toolkit **13.1** (`nvcc`).
- CMake **≥ 3.24** (tested with 4.3) and Ninja (or Make).
- A host C++ compiler supported by nvcc 13.1. nvcc 13.1 rejects GCC 16; `CMakeLists.txt` picks
  `/usr/bin/g++-15` automatically when it exists. Otherwise pass a supported compiler, e.g.
  `-DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14`.
- Python 3.12 for the server.

```bash
cmake -B build -G Ninja
cmake --build build                       # -> build/libqengine.so
python3 -m pip install -r requirements.txt  # torch: install the CUDA build matching your driver (see the file)
```

`requirements-dev.txt` adds what the tests, tools and benchmark scripts need; `requirements-ref.txt`
is a separate (CPU) venv for the Hugging Face reference dumps.

## Model files

Download the checkpoint (includes the MTP head `model_mtp.safetensors`):

```bash
hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir models/Qwen3.8-27B-INT4
```

Two additional files are published as **release assets** of this repository (not in git). Put both into
`models/Qwen3.8-27B-INT4/`:

| file | size | what |
|---|---:|---|
| `token_rank.bin` | 992,308 B | token-frequency ranking for the MTP draft vocabulary (`tools/draft_vocab.py`) |
| `model_mtp_ft.safetensors` | 849,400,392 B | MTP head fine-tuned on the engine's own answers |

Check them with `sha256sum -c SHA256SUMS` (also a release asset). Without them the engine still runs:
without `token_rank.bin` the drafts use the full vocabulary, and without `model_mtp_ft.safetensors` it
uses the checkpoint's original MTP head (`QE_MTP_FILE` selects another head file). Both make it slower
than the published numbers (fewer accepted drafts per step, more expensive drafting).

## Run

```bash
server/run.sh 8080                    # OpenAI + Anthropic API on 127.0.0.1:8080, GPUs 0,1,2
QE_DEVICES=0,2 server/run.sh 8080     # tensor parallel 2 on GPUs 0 and 2
server/run.sh 8080 --vision 1         # with image input (encoder on the last GPU, less KV room)
server/run.sh 8080 --think 0          # thinking off by default
python3 server/app.py --help          # all options
```

### Running on one or two GPUs

Three RTX 3090s are the tuned configuration; two and one card work with these settings:

| setup | command | what to expect (measured) |
|---|---|---|
| 2× RTX 3090 | `QE_DEVICES=0,1 QE_MAX_SEQS=32 server/run.sh 8080` | ~387K-token KV pool, 32 slots, full 262K context; ~162 tok/s single user, 1.2–2.0× vLLM throughput on the same cards up to 16 concurrent requests |
| 1× RTX 3090 | `QE_DEVICES=0 QE_MAX_SEQS=1 server/run.sh 8080` | one conversation, ~112K tokens of context; ~104 tok/s single user |
| 1× RTX 3090, a few chats | `QE_DEVICES=0 QE_MAX_SEQS=4 server/run.sh 8080` | 4 slots, ~85K tokens of context; ~96 tok/s single user |

- Pick the GPUs that are not driving a desktop: a display takes 1–2 GB of VRAM and the KV pool shrinks
  accordingly. If one of the cards also drives a desktop, give it more headroom so the browser and
  compositor keep enough VRAM, e.g. `QE_RESERVE_GB=1.5,3,1.5` when GPU1 runs the display; the KV pool still
  fills the other cards.
- On one card the default `QE_MAX_SEQS=16` does not start ("not enough GPU memory for a KV cache"):
  every slot holds its DeltaNet state (~0.3 GB, about 9K tokens of KV pool each). Use `QE_MAX_SEQS`
  between 1 and 4, leave `--vision` off, and lower `QE_RESERVE_GB` (e.g. `1.0`) only if you need a bit
  more context.
- The first start on a new GPU count tunes the GEMMs for the new matrix shapes and stores the table in
  `~/.cache/qengine`; later starts reuse it.
- The three-card-only optimizations (KV position split, copy-engine all-reduce) switch off by
  themselves on other counts. P2P between two cards works the same way as on three.
- Docker: the same settings as environment variables, e.g. `-e QE_GPUS=0 -e QE_MAX_SEQS=1` or
  `-e QE_GPUS=0,1 -e QE_MAX_SEQS=32`.

Startup loads the weights, pins the host-RAM archive and captures CUDA graphs. The served model name is
`qwen3.8-27b-qengine` (`--served-name`).

Example request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-qengine",
  "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
  "max_tokens": 256,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Anthropic API: `POST /v1/messages` with the usual `messages`, `system`, `tools` and `max_tokens`
fields. For Claude Code, point `ANTHROPIC_BASE_URL` at the server and set
`CLAUDE_CODE_ATTRIBUTION_HEADER=0` (the server strips the per-request attribution line anyway, so the
prompt prefix stays cacheable).

Useful environment variables:

| variable | default | meaning |
|---|---|---|
| `QE_MAX_SEQS` | 16 | sequence slots (conversations kept in the KV pool), up to 32 |
| `QE_POOL_TOKENS` | auto | KV pool size in tokens (default: all free memory) |
| `QE_RESERVE_GB` | 1.5 | VRAM left outside the KV pool; a list sets it per card, e.g. `1.5,3,1.5` (a short list repeats its last value) |
| `QE_KV_ARCHIVE_GB` | 32 | host-RAM archive of evicted conversations (0 = off) |
| `QE_PREFIX_GB` | 8 | host RAM for pinned system-prompt prefixes |
| `QE_P2P` | auto | 0 forces the host-memory communication path |
| `QE_MTP_FILE` | auto | MTP head file (default: `model_mtp_ft.safetensors` if present) |
| `QE_DRAFT_VOCAB` | 65536 | MTP draft vocabulary size (0 = full vocabulary) |
| `QE_DEC_GEMM` | 33 | batched decode passes of at least this many rows use the W4A8 GEMM instead of the GEMV (0 = off) |
| `QE_ADMIT_HEADROOM` | 8192 | KV tokens reserved per running sequence when admitting a new one |
| `QE_LOOKUP` | 1 | lookup drafts from the token history (0 = off) |

## Qwen3.8-Flash-Next (experimental preview)

The repository also contains a second engine, for **Qwen3.8-Flash-Next** (the INT4 AutoRound checkpoint by Intel):
`src/flash/`, built as `build/libqflash.so` (server) and `build/flash_run` (command-line runner). The same server
serves both models; it detects the model from `config.json` and picks the engine library, served name, MTP depth and
chat-history defaults.

> **Experimental preview.** Tested on one machine (3× RTX 3090 with P2P), no quality benchmark yet; up to 16
> requests run at once. See [Known limitations](#known-limitations). Interfaces, defaults and numbers may change
> before a release.

What the engine does for this model (48 layers: 36 Gated DeltaNet + 12 sparse-attention layers, 512 routed experts
with top-10 + a shared expert, 4-stream hyper-connections, a per-layer n-gram embedding table):

- experts split across the three cards in 128-wide groups (weighted 176/160/176, so the middle card can drive a
  desktop), INT4 expert GEMV/GEMM, the dense parts INT8;
- an expert cache: the least-used expert groups live in pinned host memory and are copied into a part of the KV
  pool while they are hot;
- attention KV in FP8, split by position over the three cards, with the indexer's top-512 block selection beyond 2K
  tokens; 262K context;
- MTP drafts (k = 3) plus lookup drafts from the conversation history, steps launched ahead, CUDA graphs;
- 16 sequence slots by default (the maximum) with batched decoding (grouped expert GEMMs from 9 rows), up to two prompts
  prefilled in flight, prompt-prefix reuse, checkpoints and a host-RAM archive of conversations;
- stop sequences and structured output (JSON schema) natively, like the 27B engine.

### Hardware and host memory

- **3× RTX 3090 with GPU P2P** (the patched driver described above). This is the only configuration tested; two or
  one card are not supported yet, and the path without P2P has not been tested with this model.
- **Host RAM**: by default the n-gram embedding table is pinned in host memory, 51.8 GB in its INT8 form (102.4 GB as
  BF16); together with the host-resident expert groups and the embeddings that is **about 64 GB with the archive
  off** (`FLASH_KV_ARCHIVE_GB=0`). The conversation archive (default `FLASH_KV_ARCHIVE_GB=48`) adds up to its size, so
  plan for **about 110 GB** with the defaults, or about 80 GB with `FLASH_KV_ARCHIVE_GB=16`.
- **Low-RAM machines: disk mode** (`FLASH_NGRAM_PIN=0`, INT8 table only). The table is not pinned: it stays in the
  file mapping, and each step a host thread copies the rows it needs (from the page cache, or from disk on a miss)
  into a small pinned buffer. It needs the NVIDIA open kernel modules (the startup check asks for pageable-memory
  access). Measured on 3× RTX 3090 with the table on a PCIe 4.0 NVMe SSD, archive off, output tok/s (October 25, 2026):

  | | short c=1 | agent c=1 | agent c=8 |
  |---|---:|---:|---:|
  | pinned table (default) | 262 | 448 | 805 |
  | disk mode, table fully in the page cache (warm) | 250 | 389 | 793 |
  | disk mode, whole server limited to 12 GB of RAM incl. page cache (cold) | 127 | 356 | 493 |

  Cold, every n-gram row that is not in the page cache is a random 4 KB disk read (median time to first token 1.7 s
  in the cold run, 0.22 s pinned). With more free RAM than the 12 GB of the cold run, the page cache keeps more of the
  table and the numbers move toward the warm row.
- **Disk**: the ready-to-run directory from Hugging Face is about 128 GB (incl. the 52 GB INT8 n-gram table). Building
  it yourself needs the 169 GB Intel download; the converted directory is about 67 GB plus 52 GB for the INT8 table.

### Model files

**Download the ready-to-run directory** from [bpx03/Qwen3.8-Flash-Next-QEngine](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine)
(about 128 GB: the converted weights, the INT8 n-gram table and the three small files below; no conversion needed):

```bash
hf download bpx03/Qwen3.8-Flash-Next-QEngine --local-dir models/Qwen3.8-Flash-Next-QE-INT4
```

#### Build the model yourself (alternative to the download)

Everything needed is in this repository plus Intel's checkpoint (169 GB download):

```bash
# 1. Intel's INT4 AutoRound checkpoint
hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir models/Qwen3.8-Flash-Next-W4A16-AutoRound
# 2. convert -> models/Qwen3.8-Flash-Next-QE-INT4 (experts repacked, dense parts RTN INT4 / BF16, INT4 MTP head)
python3 tools/convert_flash_next.py --mtp-int4
# 3. INT8 n-gram table -> models/Qwen3.8-Flash-Next-QE-INT4/ngram_int8.safetensors (51.8 GB)
python3 tools/convert_ngram_int8.py
# 4. the three small files (shipped in data/flash-next/)
cp data/flash-next/token_rank.bin data/flash-next/expert_prior.bin data/flash-next/cold_experts.json \
   models/Qwen3.8-Flash-Next-QE-INT4/
# 5. optional, to free 169 GB: delete the download and the link to its n-gram shard
#    (the engine opens every .safetensors file in the directory, so a dangling link must go)
rm -rf models/Qwen3.8-Flash-Next-W4A16-AutoRound
rm models/Qwen3.8-Flash-Next-QE-INT4/model-00016-of-00017.safetensors
```

`convert_flash_next.py` links the n-gram shard of the download (`model-00016-of-00017.safetensors`, 102 GB) into
the converted directory; `convert_ngram_int8.py` reads it once. The engine uses the INT8 table when
`ngram_int8.safetensors` is present, so after step 3 the download is no longer needed. Skip step 5 to keep the BF16
table (`FLASH_NGRAM=bf16`, needs the shard and 102 GB of pinned RAM).

The three small files (in `data/flash-next/`, also part of the Hugging Face download):

| file | size | what |
|---|---:|---|
| `token_rank.bin` | 972 KB | MTP draft vocabulary ranking: the same file as for the 27B model (same tokenizer) |
| `expert_prior.bin` | 96 KB | per-token routing probabilities, seed order of the expert cache |
| `cold_experts.json` | 5 KB | the expert groups that start in host memory |

They are optional: without them the engine runs with full-vocabulary drafting and cold groups chosen by index.
The two expert files can also be generated from routing counts of your own prompts (the result is not bit-identical
to the shipped files, which come from a router trace):

```bash
FLASH_STATS=stats.bin build/flash_run --model models/Qwen3.8-Flash-Next-QE-INT4 --ids @prompt.ids --gen 512 --k 3
python3 tools/flash_expert_prior.py stats.bin models/Qwen3.8-Flash-Next-QE-INT4/expert_prior.bin
python3 tools/flash_cold_experts.py stats.bin > models/Qwen3.8-Flash-Next-QE-INT4/cold_experts.json
```

### Run

```bash
server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4          # served as qwen3.8-flash-next-qengine
FLASH_EC_MARGIN_MB=256,2048,256 server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4   # GPU1 drives a desktop
build/flash_run --model models/Qwen3.8-Flash-Next-QE-INT4 --ids @prompt.ids --gen 256 --k 3     # runner (token ids)
```

| variable | default | meaning |
|---|---|---|
| `FLASH_EC_MARGIN_MB` | 256 | VRAM per card left free next to the expert cache; a list sets it per card (`256,2048,256` for a desktop GPU1) |
| `FLASH_NGRAM` | auto | `int8` when `ngram_int8.safetensors` exists, else `bf16`; set either to force it |
| `FLASH_KV_ARCHIVE_GB` | 48 | host-RAM archive of conversations (0 = off) |
| `FLASH_NGRAM_PIN` | 1 | 0: disk mode, the n-gram table read through the file mapping instead of pinned (see above) |
| `FLASH_MAX_SEQS` | 16 | sequence slots (1..16); lower it to free VRAM for the expert cache on a single-user setup |
| `FLASH_MAX_PREFILLS` | 2 | prompts prefilled in flight (a short prompt can run between the chunks of a long one) |
| `FLASH_PFB` | 2048 | rows of one prefill expert (MoE) batch; the server sends 2048-token chunks (`--prefill-chunk`) |
| `FLASH_PF_STAGE` | 1 | prefill batches of ≥ 2048 rows get the cold expert groups of the layer two ahead copied to the GPU by the copy engine while the current layers compute (0 = off: read over PCIe from the kernels; 2 = experimental buffer lent by the expert cache) |
| `FLASH_MOE_GU2` | 1 | prefill expert gate/up kernel that stages each job's token rows once in shared memory (0 = the previous kernel) |
| `FLASH_X16_SEL` | 1 | prefill attention exchanges between cards in fp16 also for long prompts (block selection on) |
| `FLASH_HCSP` | 1 | prefill: each card runs the hyper-connection reads only on the rows it owns in the all-reduce and gathers the results (0 = off: all rows on every card) |
| `FLASH_CKPT_MIN` | 512 | a conversation gets a checkpoint where it diverges from a cached one at least this many tokens in (prefix reuse across conversations) |
| `FLASH_LOOKUP` | 1 | lookup drafts from the history (0 = off) |
| `FLASH_DETERMINISTIC` | 0 | 1: fixed-order reductions in prefill (reproducible runs) |
| `FLASH_GUARD` | 1 | non-finite values reaching the output head stop the process with a message (2: log only, 0: off) |

### Numbers (experimental preview, October 24–25, 2026)

Measured on 3× RTX 3090 with P2P; each paragraph says which build and which tool.

**Server benchmark, October 24, 2026** (build before the prefill work of October 25): `bench/serving_suite.py` (T 0.7,
thinking off, ignore EOS), `FLASH_MAX_SEQS=16`. Output tok/s:

| scenario (input / output tokens) | c=1 | c=8 | c=16 | c=32 |
|---|---:|---:|---:|---:|
| short (1024 / 512) | 270 | 643 | 719 | 723 |
| agent (500 / 2000) | 344 | 867 | 1,084 | 1,091 |
| ShareGPT | 256 | 618 | 688 | 734 |

Prefill of a single prompt in the same run: 4,261 / 3,918 / 3,220 tok/s at 1K / 32K / 200K tokens.

**Server prefill, October 25** (single prompt, after copy-engine staging of cold expert groups and 2048-row expert
batches, `FLASH_PF_STAGE`, `FLASH_PFB`): 1K 3,634, 8K 3,965, 32K 3,996, 100K 3,800 tok/s; with sequence-parallel
hyper-connections (`FLASH_HCSP`) added: 8K 4,365, 100K 4,130 tok/s.

**Engine only, October 25** (`flash_run`, 16 slots, fresh engine, time to prefill one prompt; this release's code, with
`FLASH_MOE_GU2`, 512-row prefill chunks and `FLASH_X16_SEL`): 8K 1.83 s (4.47K tok/s), 26K 5.62 s (4.68K tok/s),
100K 21.8 s (4.58K tok/s).

**Server check, October 25** (build with `FLASH_CKPT_MIN=512`): short c=1 312 tok/s, agent c=8 939 tok/s; the chat
goodput test (1024-token shared prefix, target TTFT 3 s and TPOT 50 ms) met the target for 100% of requests at
1 request/s and 34% at 2 requests/s. Single-stream decode varies with how well the drafts hit: the agent c=1 point
measured 344–494 tok/s across runs.

### Known limitations

- No image input, no two- or one-GPU mode yet.
- **At most 16 sequence slots**: beyond 16 concurrent requests the rest wait in the server's queue.
- No quality benchmark yet. Checked so far: the INT8 n-gram table against BF16 (next-token KL on 72 held-out
  prompts below the engine's run-to-run noise, same greedy agreement and MTP acceptance as BF16, needles found at
  5K-200K tokens) and a garbage-output soak (about 4,000 requests, 33M prompt tokens, no bad reply).
- Greedy output is not bit-reproducible across a prompt-cache hit and a miss, or between single and batched lookup
  steps: the paths use different kernels and split at near-ties. Use `FLASH_DETERMINISTIC=1` with lookup off for
  reproducible runs of one conversation.
- One engine process per set of GPUs: a second engine on the same cards fails at start for lack of memory.
- In testing, one run produced all-zero tokens; it overlapped a second engine process that was being killed on the
  same GPUs and could not be reproduced (VRAM and P2P contention tests passed). The output-head guard
  (`FLASH_GUARD=1`, default) now stops the process with an error instead of emitting such tokens.

## Run with Docker

The image contains the server and `libqengine.so` built for sm_86 (CUDA 13.1.1, Ubuntu 24.04).
The model is **not** in the image: mount the directory that contains `Qwen3.8-27B-INT4/`.

Requirements on the host: NVIDIA driver with CUDA 13.1 support, Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
GPU-to-GPU P2P inside the container works exactly when it works on the host (it depends on the
host driver, not on the image): check for the `P2P:` line in the startup log.

### Prebuilt image

```bash
docker pull ghcr.io/bpx03/qengine:latest          # ~0.27 GB, no image input
docker pull ghcr.io/bpx03/qengine:vision          # ~3 GB, with the vision encoder (torch)
```

Tags: `latest` / `1.2` / `sha-<commit>` and `vision` / `1.2-vision` / `sha-<commit>-vision`. Use
`ghcr.io/bpx03/qengine:latest` in place of `qengine` in the commands below, or build the image yourself.

### Build

```bash
docker build -t qengine .                       # with vision (--vision 1): includes PyTorch, ~3 GB compressed
docker build -t qengine --build-arg VISION=0 .  # text only, no PyTorch: ~0.27 GB compressed
```

### Run

```bash
docker run -d --name qengine --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -p 127.0.0.1:9393:9393 \
  -v /path/to/models:/app/models:ro \
  -v qengine-cache:/root/.cache/qengine \
  qengine --think 0
```

or with Compose (edit `QE_MODELS_DIR`):

```bash
QE_MODELS_DIR=/path/to/models docker compose up -d
docker compose logs -f
```

- Anything after the image name (or `QE_EXTRA_ARGS`) is passed to `server/app.py`, e.g. `--think 0`, `--vision 1`, `--max-ctx 131072`.
- `QE_GPUS` selects the GPUs (ids as the host numbers them with `--gpus all`), default `0,1,2`.
  `-e QE_GPUS=0,2` runs tensor parallel 2 on GPUs 0 and 2. Three GPUs is the tuned and tested
  configuration; other counts start with a warning.
- The `qengine-cache` volume keeps the GEMM tune table, the KV split and the stats between
  restarts. Without it every start tunes the GEMMs again.
- `--ipc=host` and `--ulimit memlock=-1`: the engine pins a large host-memory KV archive.
- SELinux hosts (Fedora, RHEL): add `--security-opt label=disable` (Compose does) so the container
  can read the bind-mounted model without relabelling it.
- Port 8080: OpenAI + Anthropic API. Port 9393: monitoring dashboard, `/api/stats`, `/metrics`.

Check that it is up:

```bash
docker logs qengine | grep -E "P2P|KV pool"
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-qengine","messages":[{"role":"user","content":"Hi"}]}'
```

Performance in the container matches bare metal (single-user decode, same GPUs).

**Flash-Next (experimental preview)** has its own image, `ghcr.io/bpx03/qengine:flash-preview` (also tagged `1.3-preview.1` and
`sha-<commit>-flash-preview`; text only, built from the
`flash-next-preview` branch with `--build-arg VISION=0`; it contains both engine libraries). Point `QE_MODEL` at the converted directory and pass
the `FLASH_*` variables with `-e`:

```bash
docker run -d --name qengine-flash --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -p 127.0.0.1:9393:9393 \
  -v /path/to/models:/app/models:ro -v qengine-cache:/root/.cache/qengine \
  -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 -e FLASH_MAX_SEQS=16 \
  ghcr.io/bpx03/qengine:flash-preview
```

Add `-e FLASH_NGRAM_PIN=0` for disk mode on a low-RAM host (see [Hardware and host memory](#hardware-and-host-memory)).

## Benchmark summary

Serving benchmark (October 2026) against vLLM 0.28.0 (club-3090 "dual-fast" recipe, best of MTP / no-spec per point)
with the same checkpoint, client (`vllm bench serve`), prompts and seeds. QEngine on 3× RTX 3090
(TP=3); vLLM on 2× RTX 3090 (TP=2, because this model's 4 KV heads and 16 DeltaNet key heads do not
divide by 3). Output tokens/s, higher is better:

| scenario (input / output tokens) | c=1 | c=8 | c=16 | c=32 |
|---|---:|---:|---:|---:|
| short (1024 / 512): QEngine / vLLM best | 208 / 93 | 794 / 456 | 949 / 609 | 1,001 / 797 |
| agent (500 / 2000): QEngine / vLLM best | 287 / 128 | 1,080 / 635 | 1,302 / 857 | 1,423 / 1,219 |
| ShareGPT: QEngine / vLLM best | 243 / 87 | 819 / 414 | 935 / 523 | 1,110 / 801 |

After the comparison run, batched decode passes of ≥ 33 rows got faster (W4A8 GEMM, `QE_DEC_GEMM`) and
admission reserves less headroom per sequence: in a same-day A/B on 3 GPUs, agent c=32 went from 1,422 to
1,703 tok/s and short c=32 from 982 to 1,039 tok/s (the table above is the original run).

Prefill of a single new prompt (median time to first token): 32K tokens 5.35 s vs 9.86 s, 200K tokens
50.7 s vs 130.5 s. vLLM is ahead on a few latency points (e.g. RAG 15K-token prompts at c=8: lower
median TTFT, at the cost of 115–165 ms per output token vs 9.6 ms).

![QEngine vs vLLM summary](docs/hero.png)

Numbers from October 2026 (QEngine v1.0). The benchmark is scripted in `bench/serving_suite.py`,
`bench/vllm_cfg_best.sh` and `bench/compare/`. To measure your own setup, run the quick suite (short at c=1
plus two live-traffic scenarios, about 8 minutes):

```bash
python3 bench/serving_suite.py --engine-name qengine --base-url http://127.0.0.1:8080 \
    --model qwen3.8-27b-qengine --quick
```


The ShareGPT scenario needs `ShareGPT_V3_unfiltered_cleaned_split.json` from
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered in `bench/data/`.

## Repository layout

```
src/              engine: slots, prompt cache, MTP, scheduling (engine.*), per-GPU worker (model.*), C API (capi.cpp)
src/kernels/      GEMV / GEMM (INT4, INT8), attention (prefill + FP8 decode), DeltaNet, all-reduce, sampling, MTP
src/flash/        Qwen3.8-Flash-Next engine (preview): libqflash.so (C API, slots, archive), flash_run, kernels
server/           app.py (OpenAI + Anthropic API, tool calls, grammars), engine.py (ctypes), model_profile.py
                  (per-model defaults), qe_capi.h (C API contract), vision.py,
                  metrics.py + dashboard.html + qetop.py (monitoring), templates/ (chat template), run.sh
tests/            regression tests: caches, slots, pool, batching, MTP, sampling, grammars, kernels
tools/            reference.py / reference_vision.py (HF golden dumps), draft_vocab.py (MTP draft vocabulary),
                  convert_flash_next.py / convert_ngram_int8.py / flash_expert_prior.py / flash_cold_experts.py (Flash-Next)
bench/            serving_suite.py (vllm bench serve suite), vllm_cfg_best.sh, compare/ (harness)
```

## Tests

`tests/` holds the engine's regression tests (`test_*.py`); `tests/common.py` and `tests/qe.py` are the
shared ctypes helpers. Most tests need the three GPUs and the model; they check invariants of the engine
itself: cached, batched, speculative and archived paths must reproduce a fresh greedy computation, MTP
sampling must keep the sampling distribution, kernels must match torch or stay bit-identical across row
counts. Tests that compare against Hugging Face golden dumps (`refs/`) are not included in this repository.

```bash
python3 -m pip install -r requirements-dev.txt
python3 tests/test_grammar.py && python3 tests/test_toolcall_parse.py   # CPU only
python3 tests/test_gemv.py && python3 tests/test_gemm.py                # kernels vs torch (one GPU)
python3 tests/test_cache.py && python3 tests/test_slots.py              # prompt cache, slots
python3 tests/test_batch.py 300 3 && python3 tests/test_pool.py         # batching, KV pool under pressure
python3 tests/test_sampling.py 400                                      # MTP sampling == plain sampling
```

Flash-Next tests (CPU only, they need the converted model directory for its tokenizer and config; the 27B directory
is used where a test compares the two):

```bash
python3 tests/test_flash_template.py     # chat template: syssafe == stock, leading system messages merged, reasoning-only turns dropped
python3 tests/test_flash_tokenizer.py    # the server's tokenizer vs the official Flash-Next tokenizer (downloads it once)
python3 tests/test_flash_toolcall.py     # tool calls / reasoning streamed token by token through the parser
python3 tests/test_server_mock.py        # the server end to end (OpenAI + Anthropic APIs) on a mock engine (tests/mock_engine.c, gcc)
```

The engine-level tests above (`test_batch.py`, `test_batch_churn.py`, `test_interleave.py`, `test_fork.py`) also run
on Flash-Next (three GPUs):

```bash
export QE_TEST_MODEL=models/Qwen3.8-Flash-Next-QE-INT4 QE_LIB=build/libqflash.so
python3 tests/test_batch.py 300 3 && python3 tests/test_batch_churn.py
python3 tests/test_interleave.py && python3 tests/test_fork.py
```

To measure a Flash-Next server, start it (see [Run](#run-1)) and run the quick serving suite against it:

```bash
python3 bench/serving_suite.py --engine-name qengine-flash --base-url http://127.0.0.1:8080 \
    --model qwen3.8-flash-next-qengine --tokenizer models/Qwen3.8-Flash-Next-QE-INT4 --quick
```

## License

Free for non-commercial use. QEngine is licensed under the PolyForm Noncommercial License 1.0.0 — see
[LICENSE](LICENSE).

The model weights are not part of this repository; they are distributed by their authors under their own
license. `server/templates/qwen38_syssafe.jinja` is a modified version of the model's chat template.

## Disclaimer

QEngine is provided "as is", without warranty of any kind, express or implied, including but not limited to
the warranties of merchantability, fitness for a particular purpose and non-infringement. As far as the law
allows, the author is not liable for any claim, damages or other liability arising from the use of the
software, including hardware damage, data loss, incorrect model output or downtime. It is experimental
software, tested on a single machine (3× RTX 3090); use it at your own risk. Benchmark results describe that
machine and may not reproduce on other hardware, drivers or software versions. See [LICENSE](LICENSE) for the
binding terms.
