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
  - [Context up to 1M tokens (YaRN, opt-in)](#context-up-to-1m-tokens-yarn-opt-in)
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
| `FLASH_PREFIX_GB` | `8` | Host RAM for pinned system prompts: the KV and state at the end of each agent kind's system prompt (and tools) are kept outside the conversation archive, so a new agent of that kind starts after its system prompt even when the archive is full (a 20K-token system prompt takes about 0.4 GB). Used only as prefixes appear; the least recently used one gives way when it is full. Needs the archive (`FLASH_KV_ARCHIVE_GB` > 0). | Lower it on a machine short of RAM; `0` turns it off. |
| `FLASH_CKPT_MIN` | `512` | A conversation gets a checkpoint where it diverges from a cached one at least this many tokens in (prefix reuse across conversations). | Rarely. |
| `FLASH_DETERMINISTIC` | `0` | `1`: fixed-order reductions in prefill, so a run repeats exactly. Slower. | For reproducible runs of one conversation, together with `FLASH_LOOKUP=0`. |

Server-side: `--prefill-chunk` (2048) is the prefill chunk while other requests decode, `--prefill-chunk-idle`
(4096 for Flash-Next) the chunk when nothing else runs.

### Large prefill batches: `FLASH_PFB_BIG`

`FLASH_PFB_BIG` is the number of rows of a large prefill batch, used for a prompt when no other request is decoding.
The default is `8192` on 2 cards and `4096` on 3 cards. A larger batch means fewer batches for a long prompt, so on
2 cards the host-backed expert groups are copied to the GPUs less often.

The batch buffers above the always-present 2048 rows are lent to the expert cache outside prefill. On 2 cards the rows
past 4096 are mapped onto the end of the KV cache arrays, the part that holds expert-cache slots while the
conversations are shorter than the pool. So the large batch takes no memory from the experts that stay on the GPU,
and decode is as fast as with `4096`. A batch uses those rows only while the KV cache in use ends below them (the
first 75 % of the pool, about 196K tokens on 2 cards with the default 256K context); past that, batches of 4096 rows
are used. `FLASH_PFB_ALIAS=0` turns the mapping off; then `8192` costs about 880 resident expert groups per card
(agent throughput at 8 concurrent requests about 1 % lower). The mapping needs the KV cache in VRAM; with
`FLASH_KV=ram` the default stays `4096`.

Measured on 2× RTX 3090 (PCIe 4.0 x16), October 30, 2026, `8192` (default) against `4096`:

| measurement | 4096 | 8192 |
|---|---|---|
| server prefill, 10K prompt | 3,958 tok/s | 4,163 tok/s (+5 %) |
| server prefill, 90K prompt | 5,180 tok/s | 5,362 tok/s (+4 %) |
| `flash_run` 90K prompt | 16.6 s | 15.1 s (-9 % time) |
| agent, 8 concurrent requests | 452.6 tok/s | 450.9 tok/s (same within noise) |
| single-stream decode (narrative / code) | 218 / 235 tok/s | 218 / 234 tok/s |

On 3 cards `8192` gives no gain (`flash_run` 10K and 90K unchanged): the prefill staging ring already holds a whole
layer's host-backed groups, so the host copies are not the bottleneck. `0` turns the large batches off (2048-row
batches only).

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

### Context up to 1M tokens (YaRN, opt-in)

Qwen3.8-Flash-Next is trained for 262,144 tokens. Its model card extends that to about 1M with YaRN RoPE scaling
(factor 4, original context 262,144). QEngine applies it only on request:

```bash
FLASH_KV=ram server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4 --max-ctx 1M
```

- A `--max-ctx` above 262,144 turns YaRN on with factor `context / 262,144`: 4 for `1M` (1,048,576 tokens), 2 for
  `512K`. The model card suggests a factor that fits your typical length. `FLASH_YARN=<factor>` sets it directly,
  and `FLASH_YARN=0` turns it off.
- It needs `FLASH_KV=ram`, because the VRAM pool holds 262,144 tokens. Without it the server stays at 262,144 and
  logs a line.
- YaRN is static: it applies to every request of that server, short ones too. The card warns that this can lower
  quality on short texts. In our checks at 16K the answers stayed the same (a needle, a 160-token continuation, a
  coding answer that was worded differently but just as good). Still, use it only on a server that is meant for
  long documents. The default (262,144, native RoPE) is unchanged, bit for bit.
- YaRN changes the 12 sparse-attention layers, their indexer and the MTP layer. The DeltaNet layers have no
  positions.

Measured on RTX 3090 cards (October 30, 2026) with `flash_run`, `--ctx 1048576` (factor 4), needle in the middle of
source-code text:

| cards | prompt | needle found | prompt time | decode, single stream |
|---:|---:|---|---:|---:|
| 3 | 524,288 tokens | yes | 114 s (4.6K tok/s) | 175 tok/s |
| 3 | 1,040,000 tokens | yes | 278 s (3.7K tok/s) | 140 tok/s |
| 2 | 1,040,000 tokens | yes | 547 s (1.9K tok/s) | 79 tok/s |

On 2 cards, decode at 1M is limited by the expert cache: with its smaller budget about half of the cold expert
groups a step needs come from host memory.

Memory for one 1M-token conversation:

- **Host RAM:** 13,520 bytes per token, so about **14.2 GB per 1M tokens**. This is the `FLASH_KV=ram` pool
  (`FLASH_KV_RAM_TOKENS`, at least the context), pinned at start. It comes on top of the usual host memory (the
  52 GB n-gram table and the archive).
- **VRAM per card (3 cards):** 291 MB of indexer keys, plus a block cache of up to 886 MB while the long context
  runs. The per-position buffers sized for 1M also cut the expert cache by about 3.5 GB per card, though their
  unused part is lent back to it. Short-request decode on such a server was about 1 % slower (347 vs 351 tok/s).
- **VRAM per card (2 cards):** each card holds half of the positions, so it needs 436 MB of indexer keys plus the
  same block cache, and the expert cache shrinks by about 5.6 GB per card (from 15.8 to 10.1 GB). Short-request
  decode on such a server was about 2 % slower (283 vs 289 tok/s).

### Qwen3.8-27B

| variable | default | what it does |
|---|---|---|
| `QE_POOL_TOKENS` | auto | KV pool size in tokens (default: all free memory). |
| `QE_POOL_MIN_HEAD` | `2048` | Least headroom in tokens a conversation's KV region grows by when the pool is tight (or 1/16 of the conversation, if larger). |
| `QE_ADMIT_HEADROOM` | `8192` | KV tokens reserved per running sequence when admitting a new one. |
| `QE_P2P` | auto | `0` forces the host-memory communication path (every collective through pinned host memory). |
| `QE_MTP_FILE` | auto | MTP head file (default: `model_mtp_ft.safetensors` from the model directory, else the copy the server downloaded into `~/.cache/qengine/assets`). |
| `QE_TOKEN_RANK_FILE` | auto | Token ranking of the MTP draft vocabulary (default: `<model dir>/token_rank.bin`, else the server's downloaded copy in `~/.cache/qengine/assets`). |
| `QE_FETCH_ASSETS` | `1` | `0`: the server does not download missing `model_mtp_ft.safetensors` / `token_rank.bin` from the latest release (it only logs a warning). |
| `QE_FETCH_WAIT_S` | `120` | Seconds the server start waits for the asset download; after that it starts with the stock head and the download finishes in the background for the next start. | Raise it to always start with the tuned files on a slow link; `0` = never wait. |
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
