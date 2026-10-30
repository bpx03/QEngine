# QEngine: advanced settings

The [README](../README.md) lists the settings most setups need. This page has the rest: settings for tuning a
particular workload, and settings that help when something goes wrong. The defaults are the best configuration we
measured; most of the settings below exist to turn a feature off or to trade one kind of speed for another.

Environment variables are read when the server starts. An empty value (for example a Docker Compose pass-through
`FLASH_X: "${FLASH_X:-}"`) counts as unset and uses the default; a value that does not parse logs one warning line
and uses the default too.

- [Tuning](#tuning)
  - [Qwen3.8-Flash-Next](#qwen38-flash-next)
  - [Large prefill batches: `FLASH_PFB_BIG`](#large-prefill-batches-flash_pfb_big)
  - [Image input](#image-input)
  - [Qwen3.8-27B](#qwen38-27b)
  - [How `FLASH_COMM=auto` / `QE_COMM=auto` choose a path](#how-flash_commauto--qe_commauto-choose-a-path)
- [Troubleshooting](#troubleshooting)

## Tuning

### Qwen3.8-Flash-Next

| variable | default | what it does | when to change it |
|---|---|---|---|
| `FLASH_KV_CACHE_BLOCKS` | `16384` | With `FLASH_KV=ram`: the VRAM cache of recently used KV blocks (4 positions each), sized to the running conversations: up to 886 MB per card while long contexts run, ~3 MB idle; the unused part is used by the expert cache. | Lower it only if decode next to very long contexts matters more than their prefill. |
| `FLASH_EC_RECLAIM_S` | `30` | KV cache in VRAM: a conversation idle this many seconds moves to the host-RAM archive while other requests decode, so the expert cache gets its VRAM back; its next turn is restored from host memory. Never the main agent thread; nothing moves with `FLASH_KV_ARCHIVE_GB=0`. | `0` keeps idle conversations in VRAM. |
| `--burst-wait` (server) | `5` | ms: on an idle server the first request waits until none has arrived for this long (at most 30 ms), so a burst of requests shares one prefill batch. | `0` starts at once. |
| `FLASH_KV_RAM_LAZY` | `0` | With `FLASH_KV=ram`: `0` pins the whole host-RAM pool at start; `1` pins it in 443 MB slabs per card as the conversations grow (only on GPUs that can use registered host memory directly; otherwise the pool is pinned at start). | `1` if the pinned RAM should grow only with use. |
| `FLASH_MAX_PREFILLS` | `8` | Prompts that may be in the prefill at the same time. While several are waiting, they share prefill batches (the expert weights are read once for all of them). | Rarely; lower it to limit how many requests start prefilling together. |
| `FLASH_PF_MULTI` | `1` | Several waiting prompts share one prefill batch. `0`: one prompt per prefill batch, as before 1.3.1. | `0` only to rule the feature out while debugging. |
| `FLASH_PF_STAGE_XMB` | `1024` | Size cap (MB per card) of the ring that stages host-backed expert groups for large prefill batches. The part above 512 MB is lent to the expert cache between large prefill batches and taken back for each one. It matters on 2 cards, where about half of the expert groups live in host RAM; on 3 cards 512 MB already hold a layer. | `0` restores the fixed 512 MB ring (to rule it out while debugging); otherwise leave it. |
| `FLASH_NGRAM` | auto | n-gram embedding table: `int8` when `ngram_int8.safetensors` exists in the model directory, else `bf16`. | Set `bf16` to force the BF16 table (needs the original shard and 102 GB of pinned RAM). |
| `FLASH_NGRAM_PIN` | `1` | `0`: disk mode, the INT8 n-gram table is read through the file mapping instead of being pinned (see Hardware and host memory in the README). | `0` on a machine with too little RAM to pin 52 GB. |
| `FLASH_LOOKUP` | `1` | Lookup drafts: when the recent tokens repeat an earlier part of the conversation, drafts are copied from there. | `0` turns them off (with `FLASH_DETERMINISTIC=1` for reproducible runs). |
| `FLASH_CKPT_MIN` | `512` | A conversation gets a checkpoint where it diverges from a cached one at least this many tokens in (prefix reuse across conversations). | Rarely. |
| `FLASH_DETERMINISTIC` | `0` | `1`: fixed-order reductions in prefill, so a run repeats exactly. Slower. | For reproducible runs of one conversation, together with `FLASH_LOOKUP=0`. |

Server-side: `--prefill-chunk` (2048) is the prefill chunk while other requests decode, `--prefill-chunk-idle`
(4096 for Flash-Next) the chunk when nothing else runs.

### Large prefill batches: `FLASH_PFB_BIG`

`FLASH_PFB_BIG` (default `4096`) is the number of rows of a large prefill batch, used for a prompt when no other
request is decoding. `8192` halves the number of batches of a long prompt, so on 2 cards the host-backed expert groups
are copied to the GPUs half as often. The larger batch buffers are lent to the expert cache outside prefill, but
fewer expert groups stay resident, so decode gets slightly slower.

Measured with `flash_run` on 1.3.1 (October 30, 2026), `8192` against the default `4096`:

| setup | 10K prompt | 90K prompt | single-stream decode |
|---|---|---|---|
| 3× RTX 3090 | no change | no change (+0.3 % time) | -0.2 % |
| 2× RTX 3090, PCIe 4.0 x16 | -21 % prefill time | -7 % prefill time | -0.8 % |
| 2× RTX 3090, slow link between the cards (3.3 GB/s, emulated) | -10 % prefill time | -2 % prefill time | -0.5 % |

**Who should turn it on:** 2-card setups where most of the work is long prompts (RAG, large documents, agents that
resend big contexts) and a slightly slower decode is acceptable. Leave the default on 3 cards and for chat-style
use. `0` turns the large batches off (2048-row batches only).

### Image input

With `--vision 1` (both models):

| variable | default | what it does |
|---|---|---|
| `QE_VISION_STREAM` | Flash-Next `1`, 27B `0` | `1` keeps the encoder's block weights in pinned host RAM and copies them to the GPU during each encode; `0` keeps them on the GPU (same encode speed, ~0.77 GB more VRAM on that card, about 4 % lower Flash-Next agent throughput at 8 concurrent requests). |
| `QE_VISION_DTYPE` | Flash-Next `fp16`, 27B `bf16` | Precision of the encoder's matrix products (`fp16` or `bf16`); on Flash-Next `bf16` is less accurate at the same speed. |
| `QE_VISION_RES32` | Flash-Next `1`, 27B `0` | `1` keeps the encoder's residual stream in fp32. `0` only for comparison. |
| `QE_VISION_CACHE_MB` | `2048` | Host-RAM cache of encoded images, so an image repeated in a conversation is encoded once. Raise it when conversations carry many distinct images. |

`--vision-device` picks the GPU of the encoder (default: the last one), `--max-image-pixels` the largest image
(default 4.2 MPix = 4,096 image tokens).

### Qwen3.8-27B

| variable | default | what it does |
|---|---|---|
| `QE_POOL_TOKENS` | auto | KV pool size in tokens (default: all free memory). |
| `QE_POOL_MIN_HEAD` | `2048` | Least headroom in tokens a conversation's KV region grows by when the pool is tight (or 1/16 of the conversation, if larger). |
| `QE_ADMIT_HEADROOM` | `8192` | KV tokens reserved per running sequence when admitting a new one. |
| `QE_P2P` | auto | `0` forces the host-memory communication path (every collective through pinned host memory). |
| `QE_MTP_FILE` | auto | MTP head file (default: `model_mtp_ft.safetensors` if present). |
| `QE_DRAFT_VOCAB` | `65536` | MTP draft vocabulary size (`0` = full vocabulary). |
| `QE_DEC_GEMM` | `33` | Batched decode passes of at least this many rows use the W4A8 GEMM instead of the GEMV (`0` = off). |
| `QE_LOOKUP` | `1` | Lookup drafts from the token history (`0` = off). |
| `QE_MEM_MARGIN_GB` | max(8, 5 % of RAM) | Host RAM the archives never take: pinning stops at `MemAvailable` minus this margin (both models). |

### How `FLASH_COMM=auto` / `QE_COMM=auto` choose a path

At startup both engines measure every GPU pair for a few milliseconds (SM stores into the peer, SM loads by the
receiver, copy-engine copies, copies staged through pinned host memory) and how much a local compute kernel slows
down while each transfer runs next to it (the co-run tax). They also check that data written over P2P really
arrives (a pattern written on one card and read on the other by SM loads, SM stores and the copy engine, also through
the real exchange buffers). Flash-Next then picks its prefill transfer path:

- **P2P, per pair:** the receiving card pulls the rows with SM loads, unless loads are markedly slower than stores on
  that pair (`QE_COMM_PULL_RATIO`, default 0.7). A transport that fails the data check is not used: if SM loads
  fail, that pair pushes; if SM stores fail, peer access is turned off and every transfer goes through host memory;
  if the chosen copy engine fails, the copies are staged through host memory.
- **Copy engine** when one of these holds:
  - the P2P co-run tax is above `QE_COMM_TAX` (default 1.30), markedly above the copy engine's, and the GPUs sit on
    different NUMA nodes (the `SYS` case in `nvidia-smi topo -m`, e.g. one GPU per CPU socket);
  - the P2P co-run tax is severe: at least `QE_COMM_TAX_SEVERE` (default 3.0) and markedly above the copy engine's,
    on any NUMA layout (a slow link, e.g. two cards on PCIe Gen2 x8, can measure ~100x);
  - on one NUMA node, the P2P co-run tax reaches `QE_COMM_TAX_1NODE` (default 1.90), is markedly above the copy
    engine's, and a GPU of the set sits on a narrow link: fewer than `QE_COMM_NARROW_X` (8) PCIe lanes, or P2P probe
    bandwidth under `QE_COMM_NARROW_GBPS` (10 GB/s) (e.g. a card in a gen4 x4 slot: tax ~2.1, P2P ~6.5 GB/s);
  - P2P bandwidth is below half of the best alternative on some pair (`QE_COMM_RATIO`, 0.5).
- **Host staging** replaces the copy engine only when host-staged bandwidth is far higher.
- Otherwise, on one NUMA node, a tax between `QE_COMM_TAX` and `QE_COMM_TAX_SEVERE` keeps P2P: there the probe
  over-reads a fast PCIe link (median 1.36, rarely up to ~2.2 on 3× RTX 3090 at x16, where P2P is the fastest path). `QE_COMM_XNUMA=0` lets `QE_COMM_TAX` decide on
  one node too; `QE_COMM_CORUN=0` skips the tax probe; `QE_COMM_PROBE=0` skips the bandwidth probe.

The `[comm]` startup lines show the measurements, the data check and the reason for the choice. Decode keeps its
path. For the 27B the prefill all-reduce runs on the copy engine already; `QE_COMM=p2p` selects the SM-store
all-reduce and `QE_COMM=host` turns peer access off.

## Troubleshooting

| symptom (log line or behaviour) | what it means | what to try |
|---|---|---|
| `[comm] P2P data check FAILED (...)` at start | Data sent over P2P between two cards arrived corrupt on this machine (a PCIe / chipset problem, not a model problem). With `auto` the engine does not use the failing transport. | Nothing, if the server runs well with `auto`. If you force a path with `FLASH_COMM` / `QE_COMM`, prefer `ce`. Check `sudo dmesg \| grep -iE "aer\|pcie bus error"` for link errors. `QE_COMM_CHECK=0` turns the check off (default 4 rounds per pair and transport). |
| Prefill errors, NaN or garbage output only on long prompts, only on P2P | The P2P link may corrupt data in ways the startup check did not catch. | `FLASH_COMM=ce` (copy engine) as a workaround; compare with `FLASH_COMM=push` and `FLASH_COMM=pull` to find which P2P direction is at fault; `FLASH_COMM=host` as the last resort. Please report it with the `[comm]` startup lines. |
| `[qflash] FAULT non-finite values in the forward ...`; the request gets an error (HTTP 500 or an `error` event in a stream) | The output-head guard (`FLASH_GUARD=1`, default) caught non-finite values before they became tokens. The request fails, its conversation is dropped from the cache, the server goes on. | Note the prompt and the log lines and report them. The same fault repeated (same tokens, position and step) counts once; more than `FLASH_GUARD_MAX` (default 4) distinct faults within 5 minutes stop the process so a supervisor can restart it. `FLASH_GUARD=2` only logs, `FLASH_GUARD=3` stops at the first fault (for debugging), `0` turns the guard off. |
| Non-finite values that appear only in long-prompt prefill | Possibly the prefill attention with block selection. | Diagnostic only: `FLASH_ATTN_PFSEL=0` uses the per-row prefill attention kernel (slower at long prompts). If that fixes it, please report it. |
| `graph instantiate out of memory (N MB free): dropped K cached graphs, retrying` | A CUDA graph for a new conversation shape did not fit next to the expert cache. The engine dropped its cached graphs and retried. | Nothing; graphs are rebuilt as needed. |
| `graph instantiate still out of memory (...): this step and the next 64 new shapes run without graphs` | The retry did not fit either; those steps run without a graph (same output, slower only for those steps). | If it shows up often, raise `FLASH_EC_MARGIN_MB` on that card (e.g. `512`), at a small cost in decode and prefill speed. |
| `[gemm] GPU<n> <buffer>: out of memory growing to ... MB: ..., retrying` | A GEMM workspace had to grow after start and did not fit; the engine dropped cached graphs and retried. | As above. |
| The server does not start: `not enough GPU memory for a KV cache` (27B) | Too many sequence slots for the free VRAM (e.g. one card). | Lower `QE_MAX_SEQS`, leave `--vision` off, or lower `QE_RESERVE_GB`. |
| A desktop on one of the cards stutters or runs out of VRAM | The engine filled the card. | Flash-Next: `FLASH_EC_MARGIN_MB=256,2048,256` (one value per card); 27B: `QE_RESERVE_GB=1.5,3,1.5`. |
