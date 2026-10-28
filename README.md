# QEngine

QEngine is a source-available inference engine for one model on one kind of machine:
**Qwen3.8-27B INT4** (the RedHatAI W4A16 checkpoint plus its MTP head) on **three NVIDIA RTX 3090s**
with tensor parallelism 3 over PCIe.

**The fastest engine we know of for this model on RTX 3090s**: 1.2–2.8× vLLM 0.28's output throughput
in our benchmark (see [Benchmark summary](#benchmark-summary)), and well ahead of SGLang, Ollama and LM Studio
in our quick tests.

> **QEngine 1.3: Qwen3.8-Flash-Next.** QEngine 1.3 runs Qwen3.8-Flash-Next (INT4) on 3 or 2 RTX 3090s:
> see [Qwen3.8-Flash-Next](#qwen38-flash-next) and the ready-to-run model on
> [Hugging Face](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine); the `latest` Docker image runs both models.

- C++/CUDA engine (`src/`, about 26k lines), built as a shared library (`libqengine.so`).
- Python server (`server/`) that loads the library through ctypes and exposes an OpenAI-compatible
  API (`/v1/chat/completions`, `/v1/models`) and an Anthropic-compatible API (`/v1/messages`,
  `/v1/messages/count_tokens`). It works with agentic clients such as Claude Code.
- A second engine for **Qwen3.8-Flash-Next**, see [Qwen3.8-Flash-Next](#qwen38-flash-next).

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
  `CUDA_VISIBLE_DEVICES`), or `QE_GPUS` in Docker. By default every visible GPU is used; counts other than 2 or 3
  start with a warning.
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
layout is disabled (smaller KV pool) and decode and prefill are slower than the published numbers. For
Flash-Next the attention exchanges between the cards also go through pinned host memory then (1.3 release
candidates up to rc3 crashed in the first prefill without P2P); measured with `QE_P2P=0` on 3 GPUs, a 16K prompt
takes 6.0 s instead of 2.8 s and decode runs at about 210 tok/s.

**Interconnect probe (multi-socket / SYS topologies).** At startup both engines measure every GPU pair for a few
milliseconds: stores from SMs into the peer's memory, loads by the receiving card's SMs from the sender's memory,
copy-engine peer copies, and copies staged through pinned host memory (device to host, then host to device). For each
transport the probe also measures the co-run tax: how much a local compute kernel slows down while that transfer
runs next to it. The log shows the topology and the result (3× RTX 3090, PCIe 4.0 x16, one NUMA node):

```
[comm] topology: numa -1,-1,-1 (1 node), P2P rank 0-1:0 0-2:0 1-2:0
[comm] probe GB/s (SM store / SM load by receiver / copy engine / host-staged; tax = local compute slowdown store/load/ce): 0->1 26.2/26.6/26.3/13.2 (tax 15.59/1.36/1.03), ...
[comm] prefill transfers: P2P SM loads / stores, pairs 1<-0:pull 2<-0:pull 0<-1:pull 2<-1:pull 0<-2:pull 1<-2:pull (FLASH_COMM=auto, auto: P2P tax 1.36 (cap 1.30, 1 NUMA node), CE tax 1.03, P2P 27 CE 26 host 13 GB/s)
```

With `FLASH_COMM=auto` (the default) Flash-Next chooses the prefill path as follows:

- **P2P, per pair:** the receiving card pulls the rows with SM loads, unless loads are markedly slower than stores
  on that pair (`QE_COMM_PULL_RATIO`, default 0.7). SM stores over PCIe slow down every kernel next to them on the
  sending card; loads of the same bytes barely do.
- **Copy engine on slow or cross-NUMA links:** the prefill transfers leave P2P for the copy engine when one of these
  holds:
  - the P2P co-run tax is above `QE_COMM_TAX` (default 1.30), markedly above the copy engine's, and the GPUs sit
    on different NUMA nodes. This is the `SYS` case (`nvidia-smi topo -m`, e.g. one GPU per CPU socket), where
    transfers cross the CPU interconnect and stall compute.
  - P2P bandwidth is below half of the best alternative on some pair (`QE_COMM_RATIO`).
- **Host staging** replaces the copy engine only when host-staged bandwidth is far higher.
- **On one NUMA node** the tax alone never moves the transfers off P2P (`QE_COMM_XNUMA=0` lets it decide anywhere),
  and `QE_COMM_CORUN=0` skips the tax probe.

Decode keeps its path.
`FLASH_COMM` (Flash-Next) and `QE_COMM` (27B) force a path: `auto` (default), `p2p`, `ce` (copy engine), `host`
(staged through pinned host memory). For the 27B the prefill all-reduce runs on the copy engine already; `QE_COMM=p2p`
selects the SM-store all-reduce and `QE_COMM=host` turns peer access off (every collective through host memory,
as `QE_P2P=0`). `QE_COMM_PROBE=0` skips the probe.

**NUMA.** On a machine with several NUMA nodes, Flash-Next places pinned host memory by GPU: each card's host-backed
expert groups and its staging buffers on the card's node, the pinned n-gram table interleaved over the nodes of the
GPUs. On a single-node machine nothing changes.

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
  - The **main agent thread** of an agent session goes first: in the prefill queue (`--main-bonus`, a subagent
    passes it only after waiting `main-bonus / prefill-aging` seconds, 50 s by default), on a long subagent
    prefill in progress, and in the KV eviction order. When running subagents fill the pool, the main thread is
    admitted at once by pausing the lowest-priority decoding sequences (never a prompt being prefilled, never
    another main thread); they resume where they stopped.
  - The main thread is recognised by behaviour, not by its prompt text: requests are grouped by the client's
    session id (Claude Code `metadata.user_id`), conversations by their message history, and the main thread is
    the one that receives the other conversations' results. Before that, a Claude Code-like system prompt with an
    agent-spawning tool (`Agent`, `Task`) marks it. `--main-agent-re` (or `QE_MAIN_AGENT_RE`) forces it by a regex
    on the system prompt or the tool names. A session where no main thread could be identified is flagged in
    `qetop` and `/api/stats`.
  - A request that does not fit the KV pool no longer blocks the queue: smaller ones that fit start first, until
    it has waited `--backfill-wait` seconds (default 30; 0 = strict order).
- **KV pool under pressure** (27B):
  - A conversation's region grows by at least `QE_POOL_MIN_HEAD` tokens (default 2048, or 1/16 of its size if
    larger), so a nearly full pool does not re-place regions on every step.
  - When room runs out, idle conversations are archived to host RAM (least recently used first, finished ones
    first, the main thread last) before any running conversation is squeezed.
  - When the running conversations still do not fit, the engine preempts one of them instead of ending its reply.
    The victim is the lowest priority, then the most recently admitted. Its KV and recurrent state go to the RAM
    archive (`QE_KV_ARCHIVE_GB`), and its stream pauses without an end token.
  - Paused conversations resume in priority, then arrival, order, as soon as every running sequence keeps its
    headroom. They continue exactly where they stopped: drafts and lookup history are kept. This works for any
    number of conversations.
  - A reply ends early only when a single conversation is larger than the pool.
- **Repetition control** (both models):
  - `presence_penalty` / `frequency_penalty` request fields (OpenAI and Anthropic API), applied in the GPU
    sampler (also to speculative drafts). Default 0; `--presence-penalty` / `--frequency-penalty` (or
    `QE_PRESENCE_PENALTY` / `QE_FREQUENCY_PENALTY`) set it for sampled requests that send none.
  - **Cross-turn loop guard**: when an agent repeats the same reply with the same result `--loop-guard-turns` times
    in a row (default 3; 0 = off), the next turn gets a short server note appended to the latest tool result (the
    same call returned the same result N times, change the approach; `--loop-guard-note 0` or
    `QE_LOOP_GUARD_NOTE=0` turns the note off), a presence penalty on the repeated reply (`--loop-guard-penalty`,
    1.5; 2.0 from twice the turn count) and no lookup drafts. A repeated "ok", a call that gets new output and a
    turn the user typed never trigger it. `"loop_guard": false` in a request turns it off for that request.
    Triggers show up as a warning in the log, in `/api/stats` and in `qetop`.
  - Within one reply, a reply whose last `--loop-guard-tokens` tokens (default 600) are one short fragment repeated
    is ended.
- **Structured output**: `response_format` with `json_object` / `json_schema` through llguidance
  grammars, enforced in the engine together with speculative decoding.
- **Turn end**: a generated `<|im_start|>` ends the reply like the end-of-turn token (both models; not with
  `ignore_eos`), so a reply never continues into a turn the model made up.
- **Tool calls** (OpenAI `tools` and Anthropic `tool_use` / `tool_result`) and thinking on/off with
  reasoning-effort levels (chat template in `server/templates/`).
- **Vision** (`--vision 1`): images in Anthropic and OpenAI messages; the ViT encoder runs in PyTorch on
  the last GPU, M-RoPE positions as in Hugging Face transformers, and image tokens take part in prefix caching.
- **Activation steering** (both models): control vectors in the llama.cpp control-vector GGUF format, `add` or
  `project` mode, chosen per request (see [Activation steering](#activation-steering)).
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
prompt prefix stays cacheable). A prompt longer than the largest context slot is answered with HTTP 400
(`prompt is too long: N tokens > M maximum`, both APIs) before any streaming starts, so a client such as Claude Code
can compact the conversation and retry.

**Per-request statistics.** After every request the server prints, next to its `[chat]` / `[messages]` line, the
engine's numbers in the formats common benchmark log parsers read (vLLM's `SpecDecoding metrics`, llama.cpp's
`prompt eval time` / `eval time`):

```
[stats] SpecDecoding metrics: Mean acceptance length: 3.21, Accepted throughput: 99.80 tokens/s, Drafted throughput: 131.20 tokens/s, Accepted: 482 tokens, Drafted: 634 tokens, Avg Draft acceptance rate: 76.0% (MTP 401/517, lookup 81/117, 150 steps)
[stats] prompt eval time =     412.31 ms /  8192 tokens (    0.05 ms per token, 19868.31 tokens per second)
[stats]        eval time =    4829.10 ms /   632 tokens (    7.64 ms per token,   130.87 tokens per second)
```

Drafts count both MTP and lookup drafts; the mean acceptance length is tokens per verify step. The prompt line counts
the tokens actually computed (cached prefix tokens excluded). `QE_STATS_LOG=0` turns the lines off.

**Persistent log.** The server (both engines) appends its whole output, including the engine's own messages and any
Python traceback, to `~/.cache/qengine/logs/<served-name>.log` with a timestamp per line, so the reason for a crash
is still there after a restart (a supervisor such as llama-swap usually keeps only the current process's output).
When the process ends, the log gets its exit status (when it can still be read) and the kernel's lines about it from
the last minutes (OOM kill, GPU Xid). The file is rotated at 64 MB with 3 old files kept (`--log-mb`, `QE_LOG_MB`);
`--log-file PATH` (`QE_LOG_FILE`) moves it, `--log-file ''` or `QE_LOG_FILE=0` turns it off.

**One engine process per GPU.** Both engines lock every GPU they use for the lifetime of the process
(`~/.cache/qengine/locks/gpu-<uuid>.lock`, released by the kernel also after a crash). A second engine on a card that
is in use, a server or a tool that loads the engine, stops at start with the first one's pid and command line instead
of both running out of GPU or host memory later. `QE_ALLOW_SHARED_GPU=1` skips the check.

**Host RAM safety.** The host-RAM archive (`QE_KV_ARCHIVE_GB`, `FLASH_KV_ARCHIVE_GB`) never takes the machine below
`MemAvailable` minus a margin of max(8 GB, 5 % of RAM) (`QE_MEM_MARGIN_GB`): pinning stops there and the archive's
budget shrinks, with a log line (`archive: host memory low, budget capped at …`). When another process takes memory
later, the archive gives it back (free pinned memory first, then archived conversations in eviction order) instead of
the kernel's OOM killer ending the server. Pinning at startup that finds no host memory (for example right after a
crash, while the old process is still being written out) is retried for up to a minute, and the log names the
processes holding the most memory.

Useful environment variables:

| variable | default | meaning |
|---|---|---|
| `QE_MAX_SEQS` | 16 | sequence slots (conversations kept in the KV pool), up to 32 |
| `QE_POOL_TOKENS` | auto | KV pool size in tokens (default: all free memory) |
| `QE_POOL_MIN_HEAD` | 2048 | least headroom in tokens a conversation's KV region grows by when the pool is tight (or 1/16 of the conversation, if larger) |
| `QE_RESERVE_GB` | 1.5 | VRAM left outside the KV pool; a list sets it per card, e.g. `1.5,3,1.5` (a short list repeats its last value) |
| `QE_KV_ARCHIVE_GB` | 32 | host-RAM archive of evicted conversations (0 = off); capped at what the host can spare (see Host RAM safety) |
| `QE_PREFIX_GB` | 8 | host RAM for pinned system-prompt prefixes |
| `QE_P2P` | auto | 0 forces the host-memory communication path |
| `QE_COMM` | auto | prefill transfer path: `auto` (startup probe), `p2p`, `ce`, `host` (see [Interconnect probe](#hardware-requirements-and-limitations)) |
| `QE_MTP_FILE` | auto | MTP head file (default: `model_mtp_ft.safetensors` if present) |
| `QE_DRAFT_VOCAB` | 65536 | MTP draft vocabulary size (0 = full vocabulary) |
| `QE_DEC_GEMM` | 33 | batched decode passes of at least this many rows use the W4A8 GEMM instead of the GEMV (0 = off) |
| `QE_ADMIT_HEADROOM` | 8192 | KV tokens reserved per running sequence when admitting a new one |
| `QE_LOOKUP` | 1 | lookup drafts from the token history (0 = off) |

## Activation steering

Both engines (27B and Flash-Next) can steer the model with control vectors: a direction per layer that is added to
the residual stream at a layer's output (`add`), or whose component is removed from it (`project`). A vector changes
one documented trait of the replies (for example the answer language or the tone) without changing the prompt.

### Loading vectors

Vectors are loaded when the server starts; a vector that fails a check stops the start with the reason.

| variable | meaning |
|---|---|
| `QE_STEER=<file.gguf>[:scale]` | one vector file (its sidecar: `<file>.json`, or `QE_STEER_META=<path>`); the optional scale replaces the sidecar's default scale. A single `QE_STEER` vector is also the default vector |
| `QE_STEER_DIR=<dir>` | every `*.gguf` in the directory, each with its sidecar; vector names must be unique |
| `QE_STEER_DEFAULT=<name>` | the vector applied to requests that have no `steering` field (default: none, requests run unsteered) |

```bash
QE_STEER_DIR=/path/to/vectors server/run.sh 8080
```

The log shows one line per vector (`steering: <file>, mode=…, layers lo..hi …`). Without these variables nothing
is loaded and the engine runs exactly as without the feature.

### Vector format

A vector is two files:

- **`<name>.gguf`**: a llama.cpp control-vector file: GGUF version 3, `general.architecture = "controlvector"`,
  `controlvector.layer_count` = the number of directions, and one F32 tensor `direction.<L>` of shape
  `[hidden_size]` per layer `L` (a contiguous range of layers; 27B: hidden size 5120, layers 0..63; Flash-Next:
  2560, layers 0..47). `direction.L` acts at the output of layer `L`.
- **`<name>.json`** (the sidecar): required fields
  - `direction`: how the vector was measured;
  - `effects.intended`: the behavior change it is meant to cause;
  - `mode`: `"add"` or `"project"` (the default mode).

  Optional fields: `name` (default: the file name), `description`, `scale` (default 1), `layer_range` `[lo, hi]`
  and `per_layer_strength` `{"L": alpha}` (defaults of the default mode), `modes` (`{"<other mode>": {"layer_range",
  "scale", "per_layer_strength"}}`: the other mode's defaults; else all layers of the file, scale 1, strength 1),
  `hidden_size`, `compatibility.model` (checkpoint directory name) and `vector_file_sha256` (checked when present),
  `dir` (only `"per-layer"`).

The modes, per row and per layer `L` in the selected range, with `h` the residual after layer `L`:

- `add`: `h ← h + scale · alpha_L · v_L`;
- `project`: `h ← h − scale · alpha_L · (h · v̂_L) v̂_L` with `v̂_L` the unit direction (`scale` 1 removes the
  component completely).

The engine refuses a file with another architecture or GGUF version, a wrong shape or dtype, non-finite values,
missing or repeated layers, a missing sidecar or a sidecar without `direction` / `effects.intended`, an unknown mode,
a negative project strength, a sha256 mismatch, or a vector measured on another checkpoint.

### Per-request selection

Every request of the OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) API may carry a `steering`
field; at most one vector is active per request:

```json
"steering": "french"
"steering": {"vector": "french", "scale": 0.5}
"steering": {"vector": "french", "mode": "project"}
"steering": {"vector": "french", "mode": "add", "scale": 0.5, "layers": [2, 8]}
"steering": false
```

- Without the field (or `true`), the server default (`QE_STEER_DEFAULT`) applies; `false` turns steering off.
- `scale` multiplies the chosen mode's default scale (default 1). `mode` overrides the default mode; without
  `layers`, the chosen mode's default range applies. `layers` (`[lo, hi]`, inclusive) must lie within the layers
  stored in the file.
- An unknown vector name (the reply lists the loaded ones), a bad mode, a layer range outside the file or reversed,
  or an unknown field is answered with HTTP 400 and the reason.
- `GET /v1/steering` lists the loaded vectors: name, description, intended effect, the default mode / layers /
  scale, the layers in the file (`file_layers`) and the default layers and scale of each mode (`modes`).

Requests with different vectors, modes, layers or scales are batched together: every sequence uses its own
settings, and a request without steering gives the same tokens as on a server without vectors. The MTP draft head
is not steered (drafts are verified by the steered model).

### Notes

- **KV reuse is per steering setting.** Cached prefixes, checkpoints, the RAM archive and shared prefixes are kept
  per (vector, mode, layers, scale): a request never reuses KV computed with a different steering setting. Changing
  the steering in the middle of a conversation therefore recomputes the whole prompt once (a fresh KV cache for
  that setting); later turns with the same setting reuse it again. Steered conversations are not written to the
  disk cache.
- **A vector belongs to one checkpoint.** Directions are measured on the residual stream of a specific checkpoint
  and are meaningless for another one: the engine checks the hidden size, the layer range and, when the sidecar
  names it, the checkpoint (`compatibility.model` against the model directory name; `QE_STEER_MODEL_CHECK=0` skips
  the name check). A 27B vector cannot be used with Flash-Next or the reverse; keep one vector directory per model.
- Large `add` scales (or many layers at once) degrade the text; `project` is safe over wide ranges with scale ≤ 1.

## Qwen3.8-Flash-Next

The repository also contains a second engine, for **Qwen3.8-Flash-Next** (the INT4 AutoRound checkpoint by Intel):
`src/flash/`, built as `build/libqflash.so` (server) and `build/flash_run` (command-line runner). The same server
serves both models; it detects the model from `config.json` and picks the engine library, served name, MTP depth and
chat-history defaults.

> Tested on one machine (3× RTX 3090 with P2P, and 2 of its cards), no quality benchmark yet; up to 16 requests run
> at once. See [Known limitations](#known-limitations).

What the engine does for this model (48 layers: 36 Gated DeltaNet + 12 sparse-attention layers, 512 routed experts
with top-10 + a shared expert, 4-stream hyper-connections, a per-layer n-gram embedding table):

- experts split across the cards (3 or 2) in 128-wide groups (on 3 cards weighted 176/160/176, so the middle card
  can drive a desktop; on 2 cards evenly, `--split` sets one weight per card), INT4 expert GEMV/GEMM; dense parts INT4 (RTN, group 128) or INT8 (QSA attention, lm_head, converted at load);
- an expert cache: the least-used expert groups live in pinned host memory and are copied into a part of the KV
  pool while they are hot;
- attention KV in FP8, split by position over the cards, with the indexer's top-512 block selection beyond 2K
  tokens; 262K context;
- MTP drafts (k = 3) plus lookup drafts from the conversation history, steps launched ahead, CUDA graphs;
- 16 sequence slots by default (the maximum) with batched decoding (grouped expert GEMMs from 9 rows), up to two prompts
  prefilled in flight, prompt-prefix reuse, checkpoints and a host-RAM archive of conversations;
- stop sequences and structured output (JSON schema) natively, like the 27B engine;
- the KV pool under pressure works like the 27B one: when the running conversations do not fit, the one with the
  lowest priority (then the most recently admitted) is paused, its KV goes to the RAM archive and its stream pauses
  without an end token; it resumes exactly where it stopped as soon as the others keep their headroom. A reply ends
  early only when a single conversation is larger than the pool.

### Hardware and host memory

- **3× or 2× RTX 3090 (24 GB) with GPU P2P** (the patched driver described above). One card is not supported. The
  number of cards comes from the visible GPUs: `QE_DEVICES=0,2 server/run.sh 8080 --model …` runs on GPUs 0 and 2
  (`build/flash_run` uses every GPU in `CUDA_VISIBLE_DEVICES`). Without P2P the engine uses the copy-engine or
  host-staged transfer path (see [Interconnect probe](#hardware-requirements-and-limitations)).
- **2 cards**: about half of the expert groups do not fit in VRAM and stay in pinned host memory, about 15 GB per card
  (30 GB together), read over PCIe or copied into the expert cache when hot. Host RAM with 2 cards: about 85 GB
  with the archive off, about 130 GB with the defaults. Measured on 2× RTX 3090 (GPUs 0 and 2, October 26, 2026): short c=1
  139 tok/s, agent c=8 264 tok/s (server); single-stream decode 262 tok/s and an 8K prompt in 3.0 s (`flash_run`);
  a 26K-token needle found; next-token KL against the 3-card engine 9.1e-5 (within its run-to-run noise).
  With 8 concurrent requests the step reads about 3 GB of host-backed expert weights over PCIe.
- **Host RAM**: by default the n-gram embedding table is pinned in host memory, 51.8 GB in its INT8 form (102.4 GB as
  BF16); together with the host-resident expert groups and the embeddings that is **about 64 GB with the archive
  off** (`FLASH_KV_ARCHIVE_GB=0`). The conversation archive (default `FLASH_KV_ARCHIVE_GB=48`) adds up to its size, so
  plan for **about 110 GB** with the defaults, or about 80 GB with `FLASH_KV_ARCHIVE_GB=16`. The archive is pinned
  host memory: set `FLASH_KV_ARCHIVE_GB` to what is free after about 64 GB for the engine itself (a tmpfs `/tmp`
  also takes RAM). A budget larger than that is safe but not used: the archive stops at `MemAvailable` minus a
  margin and gives memory back when other programs need it (see Host RAM safety).
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
# 2. convert -> models/Qwen3.8-Flash-Next-QE-INT4 (experts repacked, dense parts RTN INT4 / BF16 (the engine makes QSA attention and lm_head INT8 at load), INT4 MTP head)
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
| `FLASH_KV_ARCHIVE_GB` | 48 | host-RAM archive of conversations (0 = off). When it is full, finished and normal conversations go first (least recently used); the main agent thread gives back its newest part first and keeps a prefix it resumes from. Capped at what the host can spare (see Host RAM safety) |
| `FLASH_NGRAM_PIN` | 1 | 0: disk mode, the n-gram table read through the file mapping instead of pinned (see above) |
| `FLASH_MAX_SEQS` | 16 | sequence slots (1..16); lower it to free VRAM for the expert cache on a single-user setup |
| `FLASH_MAX_PREFILLS` | 2 | prompts prefilled in flight (a short prompt can run between the chunks of a long one) |
| `FLASH_PFB` | 2048 | rows of the prefill batch whose buffers are always allocated; the server sends 2048-token chunks (`--prefill-chunk`) while other requests decode |
| `FLASH_PFB_BIG` | 4096 | larger prefill batches (2048-row expert parts); the extra rows' buffers (~0.2 GB per card) serve as expert-cache slots outside such batches and are taken back only while one runs (0 = off). The server uses them for a prompt with no other request decoding or waiting (`--prefill-chunk-idle`, 4096 for Flash-Next) |
| `FLASH_MOE_Q` | 1 | prefill expert gate/up with INT8 activations (per 64 values, quantized at runtime; the INT4 weights are unchanged) on INT8 tensor cores (0 = fp16) |
| `FLASH_MOE_QW` | 1 | INT8 gate/up kernel variant: half-group blocks (2 per SM). `2` = whole-group blocks, which in a 1.3 release candidate (rc3) gave wrong activations for a few tokens now and then on our machine (not reproduced outside the engine, under investigation); same speed |
| `FLASH_MOE_DQ` | 1 | the same for the expert down projection (0 = fp16) |
| `FLASH_ROWS_Q` | 1 | prefill attention q/k/v and output projections, router and MTP input with INT8 activations on the INT8 weights (0 = previous kernel) |
| `FLASH_ATTN_PFSEL` | 1 | prefill attention with block selection (long prompts): 8 rows share one pass over the union of their selected blocks (0 = per-row kernel; `FLASH_COMM=ce` always uses the per-row kernel) |
| `FLASH_PF_STAGE` | 1 | prefill batches of ≥ 2048 rows get the cold expert groups of the layer two ahead copied to the GPU by the copy engine while the current layers compute (0 = off: read over PCIe from the kernels; 2 = experimental buffer lent by the expert cache) |
| `FLASH_MOE_GU2` | 1 | prefill expert gate/up kernel that stages each job's token rows once in shared memory (0 = the previous kernel) |
| `FLASH_X16_SEL` | 1 | prefill attention exchanges between cards in fp16 also for long prompts (block selection on) |
| `FLASH_COMM` | auto | prefill transfers between cards (all-reduce, row gather, attention exchanges): `auto` (startup probe: P2P with receiver pulls per pair, the copy engine on cross-NUMA / slow links, see [Interconnect probe](#hardware-requirements-and-limitations)), `p2p`, `pull` / `push` (P2P with SM loads by the receiver / SM stores into the peer on every pair), `ce` (copy-engine peer copies), `host` (staged through pinned host memory); decode unchanged |
| `FLASH_QPF_UNION` | 1 | prefill attention with block selection: the union of each 8-row group's selected blocks is built once per group instead of in every attention block (0 = off) |
| `FLASH_HCSP` | 1 | prefill: each card runs the hyper-connection reads only on the rows it owns in the all-reduce and gathers the results (0 = off: all rows on every card) |
| `FLASH_PFU` | 1 | prefill: one all-reduce / row gather per 1024 rows (`FLASH_ARU`) and the expert half run in the same units, so each transfer overlaps the next unit's compute (0 = the previous schedule: transfers per 512-row chunk) |
| `FLASH_MOE_SLOT` | 1 | prefill expert down projection into a per-(token, expert) buffer summed in a fixed order instead of floating-point atomics (0 = atomics) |
| `FLASH_CKPT_MIN` | 512 | a conversation gets a checkpoint where it diverges from a cached one at least this many tokens in (prefix reuse across conversations) |
| `FLASH_LOOKUP` | 1 | lookup drafts from the history (0 = off) |
| `FLASH_LK_UNIFORM` | 1 | every card takes the lookup result of the first card, so all cards run the same draft step (0 = off) |
| `FLASH_DETERMINISTIC` | 0 | 1: fixed-order reductions in prefill (reproducible runs) |
| `CUDA_MODULE_LOADING` | EAGER | set by the engine at load unless already set: every kernel module is loaded at startup (lazy loading could deadlock the first prefill on the copy-engine path, `FLASH_COMM=ce`, and made the first prefill slower); `LAZY` = the CUDA default |
| `FLASH_GUARD` | 1 | non-finite values reaching the output head fail that request with an error (HTTP 500, or an `error` event in a stream) instead of emitting garbage tokens; its conversation is dropped from the cache and the other requests go on. More than `FLASH_GUARD_MAX` (4) such faults within 5 minutes stop the process (2: log only, 3: stop at the first, 0: off) |

### Numbers (October 24–26, 2026)

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

**Engine only, October 26** (`flash_run`, 16 slots, fresh engine, time to prefill one prompt, after the prefill
overlap round: `FLASH_PFU`, `FLASH_MOE_SLOT`): 3 GPUs 8K 1.64 s (5.00K tok/s), 26K 5.34 s (4.99K tok/s), 100K 21.7 s
(4.72K tok/s); 3 GPUs with `FLASH_COMM=host` (transfers through host memory, the slow-link case) 8K 2.14 s, 26K 6.92 s,
100K 28.1 s (the build before: 2.40 / 7.46 / 29.6 s); 2 GPUs 8K 3.03 s, 26K 8.88 s, 100K 34.6 s.

**Engine only, October 26, prefill round 3** (`flash_run`, 16 slots, fresh engine; INT8 activations in the expert
and projection GEMMs, 4096-row batches, the row-tile attention): 3 GPUs 8K 1.33 s (6.17K tok/s), 26K 4.67 s
(5.70K tok/s), 100K 18.9 s (5.43K tok/s); 3 GPUs with `FLASH_COMM=host` 8K 1.60 s, 26K 5.84 s, 100K 23.6 s; 2 GPUs 8K
1.95 s, 26K 6.23 s, 100K 25.0 s (4.09K tok/s). Server, same build: a single 16K prompt reaches its first token in
2.6–3.0 s; agent c=8 decode is unchanged (865–1008 tok/s across runs, run-to-run spread ±8 %).

**Engine only, October 26, prefill round 4** (`flash_run`, 16 slots, fresh engine, 3 GPUs). Changes:

- The all-reduce and the row gather pull rows with SM loads instead of storing them into the peers.
- The n-gram rows are computed in O(1) per token on 8 host threads.
- The attention block unions are built once per row group.

Results: 16K 2.62 s (6.25K tok/s) and 100K 16.08 s (6.37K tok/s). With the round-3 build these took 2.85 s and
18.9 s.

**Server check, October 25** (build with `FLASH_CKPT_MIN=512`): short c=1 312 tok/s, agent c=8 939 tok/s; the chat
goodput test (1024-token shared prefix, target TTFT 3 s and TPOT 50 ms) met the target for 100% of requests at
1 request/s and 34% at 2 requests/s. Single-stream decode varies with how well the drafts hit: the agent c=1 point
measured 344–494 tok/s across runs.

### Known limitations

- No image input, no one-GPU mode.
- **At most 16 sequence slots**: beyond 16 concurrent requests the rest wait in the server's queue.
- No quality benchmark yet. Checked so far: the INT8 n-gram table against BF16 (next-token KL on 72 held-out
  prompts below the engine's run-to-run noise, same greedy agreement and MTP acceptance as BF16, needles found at
  5K-200K tokens) and a garbage-output soak (about 4,000 requests, 33M prompt tokens, no bad reply).
- Greedy output is not bit-reproducible across a prompt-cache hit and a miss, or between single and batched lookup
  steps: the paths use different kernels and split at near-ties. Use `FLASH_DETERMINISTIC=1` with lookup off for
  reproducible runs of one conversation.
- One engine process per set of GPUs: a second engine on the same cards is refused at start (see One engine process
  per GPU).
- In testing, one run produced all-zero tokens; it overlapped a second engine process that was being killed on the
  same GPUs and could not be reproduced (VRAM and P2P contention tests passed). The output-head guard
  (`FLASH_GUARD=1`, default) now fails such a request with an error instead of emitting its tokens.

## Run with Docker

The image contains the server and `libqengine.so` built for sm_86 (CUDA 13.1.1, Ubuntu 24.04).
The model is **not** in the image: mount the directory that contains `Qwen3.8-27B-INT4/`.

Requirements on the host: NVIDIA driver with CUDA 13.1 support, Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
GPU-to-GPU P2P inside the container works exactly when it works on the host (it depends on the
host driver, not on the image): check for the `P2P:` line in the startup log.

### Prebuilt image

```bash
docker pull ghcr.io/bpx03/qengine:latest          # ~0.3 GB, no image input
docker pull ghcr.io/bpx03/qengine:vision          # ~3 GB, with the vision encoder (torch)
```

Tags: `latest` / `1.3` and `vision` / `1.3-vision`, built from `main` (1.3; both engine libraries, so the same
image serves Qwen3.8-27B and Qwen3.8-Flash-Next). The previous release stays available as `1.2` / `1.2-vision`.
Use `ghcr.io/bpx03/qengine:latest` in place of `qengine` in the commands below, or build the image yourself.

### Build

```bash
docker build -t qengine .                       # with vision (--vision 1): includes PyTorch, ~3 GB compressed
docker build -t qengine --build-arg VISION=0 .  # text only, no PyTorch: ~0.3 GB
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
  `-e QE_GPUS=0,2` runs tensor parallel 2 on GPUs 0 and 2. Three GPUs is the tuned configuration; both
  models also run on two (Flash-Next: see its hardware notes); other counts start with a warning.
- The `qengine-cache` volume keeps the GEMM tune table, the KV split and the stats between
  restarts. Without it every start tunes the GEMMs again.
- `--ipc=host` and `--ulimit memlock=-1`: the engine pins a large host-memory KV archive.
- SELinux hosts (Fedora, RHEL): add `--security-opt label=disable` (Compose does) so the container
  can read the bind-mounted model without relabelling it.
- Port 8080: OpenAI + Anthropic API. Port 9393: monitoring dashboard, `/api/stats`, `/metrics`.

Check that it is up:

```bash
docker logs qengine | grep -E "P2P|KV pool|\[comm\]"
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-qengine","messages":[{"role":"user","content":"Hi"}]}'
```

Performance in the container matches bare metal (single-user decode, same GPUs).

**Flash-Next** runs in the same image (`latest` or `1.3`; Flash-Next has no image input). Point `QE_MODEL` at the
Flash-Next model directory and pass the `FLASH_*` variables with `-e`:

```bash
docker run -d --name qengine-flash --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -p 127.0.0.1:9393:9393 \
  -v /path/to/models:/app/models:ro -v qengine-cache:/root/.cache/qengine \
  -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 \
  ghcr.io/bpx03/qengine:latest
```

Add `-e FLASH_NGRAM_PIN=0` for disk mode on a low-RAM host (see [Hardware and host memory](#hardware-and-host-memory)).

If you built the model yourself and kept the BF16 n-gram table (skipped step 5), `model-00016-of-00017.safetensors`
in the converted directory is a link into the original download. Mount that download at the same absolute path
inside the container as well (e.g. `-v /path/to/models/Qwen3.8-Flash-Next-W4A16-AutoRound:/path/to/models/Qwen3.8-Flash-Next-W4A16-AutoRound:ro`),
otherwise the link is dangling in the container and the engine cannot open it. The Hugging Face download and the
INT8 table need no extra mount.

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
src/flash/        Qwen3.8-Flash-Next engine: libqflash.so (C API, slots, archive), flash_run, kernels
server/           app.py (OpenAI + Anthropic API, tool calls, grammars), engine.py (ctypes), model_profile.py
                  (per-model defaults), qe_capi.h (C API contract), vision.py, steering.py (activation steering),
                  engine_stats.py (per-request stats lines),
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
