# QEngine: advanced settings

The [README](../README.md) lists the settings most setups need. This page has the rest: settings for tuning a
particular workload, settings that help when something goes wrong, more measurements, the Flash-Next model build,
activation steering in detail, and a troubleshooting table. The defaults are the best configuration we measured; most
of the settings below exist to turn a feature off or to trade one kind of speed for another.

Environment variables are read when the server starts. An empty value (for example a Docker Compose pass-through
`FLASH_X: "${FLASH_X:-}"`) counts as unset and uses the default; a value that does not parse logs one warning line
and uses the default too. In Docker pass them with `-e`.

- [Tuning](#tuning)
  - [Qwen3.8-Flash-Next](#qwen38-flash-next)
  - [Parallel modes (`FLASH_PAR`)](#parallel-modes-flash_par)
  - [Large prefill batches: `FLASH_PFB_BIG`](#large-prefill-batches-flash_pfb_big)
  - [KV cache in host RAM (Flash-Next)](#kv-cache-in-host-ram-flash-next)
  - [Disk mode for the n-gram table](#disk-mode-for-the-n-gram-table)
  - [Image input](#image-input)
  - [Context up to 1M tokens (YaRN, opt-in)](#context-up-to-1m-tokens-yarn-opt-in)
  - [Qwen3.8-27B](#qwen38-27b)
  - [Server: scheduling, thinking, logs](#server-scheduling-thinking-logs)
  - [Failure handling and host memory](#failure-handling-and-host-memory)
  - [How `FLASH_COMM=auto` / `QE_COMM=auto` choose a path](#how-flash_commauto--qe_commauto-choose-a-path)
- [Build the Flash-Next model directory yourself](#build-the-flash-next-model-directory-yourself)
- [Activation steering](#activation-steering)
- [Measurements](#measurements)
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
| `FLASH_PF_STAGE_XMB` | `1024` (`2048` on `FLASH_PAR=pp` stages) | Size cap (MB per card) of the ring that stages host-backed expert groups for large prefill batches. The part above 512 MB is lent to the expert cache between large prefill batches and taken back for each one. It matters on 2 cards, where about half of the expert groups live in host RAM; on 3 cards 512 MB already hold a layer. | `0` restores the fixed 512 MB ring (to rule it out while debugging); otherwise leave it. |
| `FLASH_NGRAM` | auto | n-gram embedding table: `int8` when `ngram_int8.safetensors` exists in the model directory, else `bf16`. | Set `bf16` to force the BF16 table (needs the original shard and 102 GB of pinned RAM). |
| `FLASH_NGRAM_PIN` | set by the host-RAM plan (`1` when the table fits) | `0`: disk mode, the INT8 n-gram table is read through the file mapping instead of being pinned; `1`: pinned even if the plan says it does not fit (see [Host memory](../README.md#host-memory-flash-next) in the README; measurements under [Disk mode](#disk-mode-for-the-n-gram-table)). | Leave unset: the start-up picks it from the RAM. |
| `QE_NUMA` | sysfs | NUMA nodes for the host-pinned buffers (n-gram table, cold experts, host-staged comm), overriding the per-GPU sysfs `numa_node`. For machines where sysfs reports `-1` (no GPU-to-node mapping): without it every pinned buffer lands on default first-touch placement, silently. Several nodes interleave the pages, one prefers them. | Set it on a multi-node machine whose GPUs report no node (`nvidia-smi topo -m` `GPU NUMA ID: N/A`), e.g. `QE_NUMA=0,1,2,3`. |
| `QE_PIN_THREADS` | `32` / `4` | CPU threads of the load-time pin/copy pools (the n-gram table's 32, the cold experts' 4): read the model files, first-touch the pinned buffers, `cudaHostRegister` them. | Lower it when a load's memory traffic should not use the whole machine. |
| `FLASH_WEIGHTS_COLD` | `1` | After the load, the engine drops the page cache of the model files it no longer reads: every weight file, and the n-gram table file too when the table is pinned. In disk mode the table's file is kept, so the weights' page cache (about 70 GB, read last) no longer pushes the table out. The mappings stay valid; nothing reads them during serving. | `0` keeps the page cache (only to rule it out). |
| `FLASH_TP_SPIN_US` | `5000` | Tensor parallel: µs a helper thread (one per card after the first) spins after its last job before it sleeps until the next one. The jobs of a decode step and consecutive steps come within this window, so serving speed is unchanged, and an idle server uses no CPU (1.3.4.2 and earlier: one core per helper). | `-1` spins all the time (the old behaviour). |
| `FLASH_NGRAM_SPIN_MS` | `50` | Disk mode: ms the row-copy thread of each card spins after the last request; then it only polls, every 50 µs during the first second and every 1 ms after, so an idle server costs almost no CPU (1.3.4.2 and earlier: about a third of a core per card). The first decode step after a pause waits at most one poll. | `-1` spins all the time; `0` polls at every step (slower). |
| `FLASH_LOOKUP` | `1` | Lookup drafts: when the recent tokens repeat an earlier part of the conversation, drafts are copied from there. | `0` turns them off (with `FLASH_DETERMINISTIC=1` for reproducible runs). |
| `FLASH_PREFIX_GB` | `8` | Host RAM for pinned system prompts: the KV and state at the end of each agent kind's system prompt (and tools) are kept outside the conversation archive, so a new agent of that kind starts after its system prompt even when the archive is full (a 20K-token system prompt takes about 0.4 GB). Used only as prefixes appear; the least recently used one gives way when it is full. Needs the archive (`FLASH_KV_ARCHIVE_GB` > 0). | Lower it on a machine short of RAM; `0` turns it off. |
| `FLASH_CKPT_MIN` | `512` | A conversation gets a checkpoint where it diverges from a cached one at least this many tokens in (prefix reuse across conversations). | Rarely. |
| `FLASH_DETERMINISTIC` | `0` | `1`: fixed-order reductions in prefill, so a run repeats exactly. Slower. | For reproducible runs of one conversation, together with `FLASH_LOOKUP=0`. |
| `FLASH_GUARD` | `1` | Output-head guard: a request whose forward produced non-finite values fails with an error instead of emitting tokens. `2` only logs, `3` stops at the first fault (debugging), `0` = off. | Rarely. |
| `FLASH_OOM_RESERVE_MB` | `64` | Emergency VRAM per card, given up before a step runs without a CUDA graph; taken back once there is clearly room. | `0` = off. |
| `FLASH_PIN_VRAM_MB_PER_GB` | `2` | VRAM per GB of host memory registered after start (archive, pinned prompts), kept out of the expert cache (the driver maps registered host memory with 2 MB of GPU page tables per GB on every card). | `0` = the pre-1.3.3 budget. |
| `FLASH_CE_PREWAIT` | `1` | With `FLASH_COMM=ce`: one GPU block waits for the other cards' attention data before the kernel that uses it, only where a kernel runs next to the transfers; `2` = always. | `0` only for debugging. |
| `QE_AR_SPLIT` | `1` | Decode all-reduces without a fused norm (Flash-Next) spread each row over several 128-thread blocks, so more SMs send and poll; same results. | `0` restores one block per row (only for debugging). |
| `QE_AG_LL` | `1` | With peer access (both models): the small all-gathers of each decode step (argmax, top-k, lookup agreement) go GPU to GPU as tagged words instead of through host memory; same results. | `0` restores the host-memory path (only for debugging). |
| `FLASH_QX16_DEC` | `1` | Decode and MTP steps send the attention queries to the other cards as fp16 (the values the attention kernel uses anyway, so results are identical); half the bytes. | `0` sends fp32 (only for debugging). |

Server-side: `--prefill-chunk` (2048) is the prefill chunk while other requests decode, `--prefill-chunk-idle`
(4096 for Flash-Next) the chunk when nothing else runs.

### Parallel modes (`FLASH_PAR`)

How Flash-Next shares the cards; the [README](../README.md#parallel-mode-flash-next-flash_par) explains the modes and the automatic pick, with measurements.

| variable | default | what it does |
|---|---|---|
| `FLASH_PAR` | `auto` (server), `tp` (`flash_run`) | `tp`: every layer split over the cards. `pp`: pipeline, each card holds a contiguous range of layers; prompts flow through in micro-batches and several conversations decode as groups that move card to card (also between the server's calls). `pptp`: pipeline over tensor-parallel groups (experimental, opt-in, untested on 4 cards). `pp_prefill`: pipelined prefill, then the state moves to `tp` for decode; it needs 58 GB or more per card; on smaller cards a 2- or 3-card server runs `tp` instead, and on 4 or more cards the start is refused. `auto`: picks `tp` or `pp` at start. |
| `FLASH_PAR_AUTO_P` | `8192` | Uncached prompt tokens of the typical request `auto` models. `32768` makes long-prompt work count. |
| `FLASH_PAR_AUTO_D` | `512` | Reply tokens of that request. |
| `FLASH_PAR_AUTO_C` | `1,4,8` | Concurrencies `auto` averages over (capped at `FLASH_MAX_SEQS`). |
| `FLASH_PAR_MARGIN` | `0.10` | How much faster per request `pp` must be over the mix to be picked. |
| `FLASH_CARD_CAL` | `1` | The short per-card speed calibration at start (used by `auto` and by the `pp` layer split); `0` = treat the cards as equal. |
| `FLASH_PP_LAYERS` | the planner's | `pp`: layers per stage as counts, e.g. `17,17,14` on 3 cards (48 in all). |
| `FLASH_PP_CARD_SPLIT` | `1` | `pp` on uneven cards: split the layers by the calibrated speeds; `0` = the equal-card split. |
| `FLASH_PP_MAP` | - | `pptp`: the stage groups, `,` inside a stage, `|` between stages (e.g. `0,1|2,3`); device indices as in `--devices`. |
| `FLASH_PP_TP` | `2` | `pptp` without `FLASH_PP_MAP`: group size over consecutive cards. |
| `FLASH_PPB_DSHARE` | `0.5` | `pp`: the decoders' share of the time while a prompt is prefilled. |
| `FLASH_BOX_REPORT` | on for 4+ cards and `pptp` | Logs a `[box]` report at start (build, CUDA archs, driver, per-card PCIe / NUMA / host rate, per-pair peer access and copy rate) for remote diagnosis. `1` turns it on everywhere. |

In `pp`, image input is refused, steering vectors are not loaded (also when `auto` picks `pp` on 4 or more cards),
`FLASH_KV=ram` makes a 2- or 3-card server run `tp` instead and refuses the start on 4 or more cards, activation
capture is refused and lookup drafts are off. Structured output and the thinking budget work; a constrained step runs
through the stages as a batch of one.

In `pp` on 2 cards a prompt's end checkpoint moves to the end of its last prefill batch, so retrying the exact same
prompt can recompute up to one batch.

### Large prefill batches: `FLASH_PFB_BIG`

`FLASH_PFB_BIG` is the number of rows of a large prefill batch, used for a prompt when no other request is decoding.
The default is `8192` on 2 cards and in `pp`, `4096` on 3 cards, and `16384` on a slow host link. A larger batch means fewer batches for a long prompt, so on
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

### KV cache in host RAM (Flash-Next)

By default the attention KV of all conversations shares one VRAM pool of 262,144 tokens. With `FLASH_KV=ram` the K/V
rows of the 12 sparse-attention layers and the MTP layer live in pinned host RAM, and VRAM keeps the indexer keys of
every position plus a cache of recently used KV blocks. The blocks the indexer selects that are not in the cache are
copied from host RAM inside the step (the selection changes little from step to step). Output is bit-identical to the
VRAM pool. One conversation is still limited to 262,144 tokens unless you turn on the 1M context (below).

Host RAM is 13,520 bytes per token for the whole machine, on 3 or 2 cards alike (about 74K tokens per GB), pinned at
start (`FLASH_KV_RAM_LAZY=1` pins it as the conversations grow). The start-up log prints what the setting means
(`KV in RAM: 1.00M tokens = ~13.5 GB pinned ...`); a size that does not fit under `MemAvailable` minus the safety
margin is capped with a log line.

| `FLASH_KV_RAM_TOKENS` | host RAM (all cards) | VRAM per card for indexer keys, 3 cards / 2 cards |
|---:|---:|---:|
| 1M | 13.5 GB | 277 / 416 MB |
| 4M | 54 GB | 1.1 / 1.7 GB |
| 10M | 135 GB | 2.8 / 4.2 GB |

The unused part of the indexer-key memory is lent to the expert cache, but it is reserved at start: with 10M on 3
cards, the 2.8 GB per card come out of the resident experts and a 32K prompt took 11 % longer. The VRAM block cache
takes up to 886 MB per card, only while long contexts run (`FLASH_KV_CACHE_BLOCKS`). Measurements: see
[Measurements](#measurements).

### Disk mode for the n-gram table

`FLASH_NGRAM_PIN=0` (INT8 table only): the 52 GB n-gram table is not pinned. It stays in the file mapping, and each
step a host thread copies the rows it needs (from the page cache, or from disk on a miss) into a small pinned buffer.
Since 1.3.4 the start-up also picks this mode by itself when the table does not fit the machine's RAM (the host-RAM
plan, see [Host memory](../README.md#host-memory-flash-next)); it no longer needs the NVIDIA open kernel modules
(only `FLASH_NGRAM_HMM=1`, where the GPU reads the file mapping itself, does). Measured on 3× RTX 3090
with the table on a PCIe 4.0 NVMe SSD, archive off, output tok/s (October 25, 2026):

| | short, 1 at once | agent, 1 at once | agent, 8 at once |
|---|---:|---:|---:|
| pinned table (default) | 262 | 448 | 805 |
| disk mode, table fully in the page cache (warm) | 250 | 389 | 793 |
| disk mode, whole server limited to 12 GB of RAM incl. page cache (cold) | 127 | 356 | 493 |

Cold, every n-gram row that is not in the page cache is a random 4 KB disk read (median time to first token 1.7 s in
the cold run, 0.22 s pinned). With more free RAM the page cache keeps more of the table and the numbers move toward
the warm row. Since 1.3.4.3 the engine drops the weight files' page cache after the load (`FLASH_WEIGHTS_COLD`), so the table is
the only model file that competes for the page cache.

### Image input

With `--vision 1` (both models):

| variable | default | what it does |
|---|---|---|
| `QE_VISION_STREAM` | Flash-Next `1`, 27B `0` | `1` keeps the encoder's block weights in pinned host RAM and copies them to the GPU during each encode; `0` keeps them on the GPU (same encode speed, ~0.77 GB more VRAM on that card, about 4 % lower Flash-Next agent throughput at 8 concurrent requests). |
| `QE_VISION_DTYPE` | Flash-Next `fp16`, 27B `bf16` | Precision of the encoder's matrix products (`fp16` or `bf16`); on Flash-Next `bf16` is less accurate at the same speed. |
| `QE_VISION_RES32` | Flash-Next `1`, 27B `0` | `1` keeps the encoder's residual stream in fp32. `0` only for comparison. |
| `QE_VISION_CACHE_MB` | `2048` | Host-RAM cache of encoded images, so an image repeated in a conversation is encoded once. Raise it when conversations carry many distinct images. |

`--vision-device` picks the GPU of the encoder (default: the last one), `--max-image-pixels` the largest image
(default 4.2 MPix = 4,096 image tokens; larger images are downscaled).

How it works on Flash-Next: the same 27-block vision tower as the 27B, fp16 matrix products with an fp32 residual
stream. The block weights stay in pinned host memory (~0.8 GB) and are copied to the GPU block by block while the
previous block computes, so the GPU keeps only ~0.65 GB for the encoder. Image tokens get the model's 3-D
(time, height, width) M-RoPE positions in the sparse-attention layers, their indexer and the MTP layer; the n-gram table
sees `<|image_pad|>` for image tokens, as in the reference implementation. Image tokens take part in prefix reuse,
checkpoints and the RAM archive, keyed on the image content. Document / file parts become a short
"[document omitted …]" note. A non-finite fp16 output (not seen in testing) is re-encoded in bf16.

Measured (3× RTX 3090, October 28, 2026): encoder output against Hugging Face transformers in fp32, per-token cosine
min 0.996 / mean 0.99998; a 1,437-token prompt with a 1600×900 screenshot reaches its first token in about 0.7 s;
agent throughput at 8 at once 536 tok/s with `--vision 1` against 537 without it. Against an fp32 reference of the
whole model, the next-token distributions after an image prompt differ more than after a text prompt (mean KL 0.014
vs 0.003 over 33 positions). Video input is not supported. No vision quality benchmark yet.

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
| `QE_HOST_CE` | `2` | Without P2P: the prefill all-reduce copies its rows through pinned host memory by DMA (3 cards). `1` uses the driver's peer copies, `0` the pre-1.3.4 path. |
| `QE_KV_SPLIT` | auto | Position split of the KV heads 1 and 2 (3 cards with P2P only); `0` = the old layout. |
| `QE_STEER`, `QE_STEER_DIR`, `QE_STEER_DEFAULT` | - | Activation steering (both models), see [Activation steering](#activation-steering). |

**One or two cards (27B).** On one card the default `QE_MAX_SEQS=16` does not start ("not enough GPU memory for a
KV cache"): every slot holds its DeltaNet state (~0.3 GB, about 9K tokens of KV pool each). Use `QE_MAX_SEQS` between
1 and 4, leave `--vision` off, and lower `QE_RESERVE_GB` (e.g. `1.0`) only if you need a bit more context. The
three-card-only optimizations (KV position split, copy-engine all-reduce) switch off by themselves on other counts.
Pick cards that do not drive a desktop: a display takes 1-2 GB of VRAM and the KV pool shrinks accordingly.

### Server: scheduling, thinking, logs

| setting | default | what it does |
|---|---|---|
| `--prefill-chunk` | `2048` | prompt chunk (tokens) between decode bursts of the running conversations (`0`: a prompt is prefilled at once) |
| `--prefill-chunk-idle` | Flash-Next `4096`, 27B = `--prefill-chunk` | chunk while nothing else decodes or waits |
| `--prefill-share` | `0.5` | share of the GPU time running conversations keep while another prompt is prefilled |
| `--prefill-aging` | `4000` | queue order: least prefill work first; every second of waiting counts as this many tokens |
| `--burst-wait` | `5` | ms an idle server waits after the first request of a burst (at most 30 ms), so the burst shares one prefill batch; `0` = start at once |
| `--preempt-max` / `--preempt-min` | `1024` / `8192` | a queued request with at most this many tokens to prefill starts between the chunks of a prefill with at least `--preempt-min` tokens left |
| `--main-bonus` | `200000` | the main agent thread counts as this many tokens less prefill work (a subagent passes it only after waiting `main-bonus / prefill-aging` seconds); `0` = no precedence |
| `--backfill-wait` | `30` | seconds a request that does not fit the pool lets smaller ones start first; `0` = strict order |
| `--main-agent-re` / `QE_MAIN_AGENT_RE` | auto | regex for the main agent thread (system prompt or tool names) |
| `--bg-archive` | `64` | MB per GPU per step of archiving idle conversations while the server is idle; `0` = archive only when a slot is taken |
| `--sort-tools` | `1` | render tools sorted by name with sorted JSON keys (`0`: the client's order) |
| `--drop-empty-turns` | Flash-Next `1`, 27B `0` | leave earlier reasoning-only assistant turns out of the prompt |
| `--merge-system` | Flash-Next `1`, 27B `0` | merge several leading system messages into one |
| `--mtp-k` | 27B `7`, Flash-Next `3` | most MTP drafts per step (adaptive below that) |
| `--loop-guard-tokens` | `600` | end a reply whose last N tokens are one short fragment repeated; `0` = off |
| `--loop-guard-turns` / `QE_LOOP_GUARD_TURNS` | `3` | cross-turn loop guard after N identical (reply, result) turns (tool calls must match exactly, ids included); `0` = off |
| `--loop-guard-penalty` / `QE_LOOP_GUARD_PENALTY` | `1.5` | presence penalty of a guarded turn (2.0 from twice the turn count) |
| `--loop-guard-note` / `QE_LOOP_GUARD_NOTE` | `1` | append a short server note to the latest tool result of a guarded turn; `0` = off |
| `QE_THINK_WRAP` | the default sentence | the wrap-up sentence before a forced `</think>` |
| `QE_STATS_LOG` | `1` | per-request statistics lines; `0` = off |
| `--log-file` / `QE_LOG_FILE` | `auto` | persistent log path (`''` or `0` = off) |
| `--log-mb` / `QE_LOG_MB` | `64` | size before the log is rotated (3 old files kept) |
| `--stats-file` | `~/.cache/qengine/stats.json` | all-time monitoring totals |
| `--disk-cache DIR`, `--disk-cache-gb`, `--persist-idle` | off, `200`, `120` | 27B persistent prompt cache, its size, and the idle seconds before live conversations are saved; steered conversations are not saved |

Per-request statistics lines, e.g.:

```
[stats] SpecDecoding metrics: Mean acceptance length: 3.21, Accepted throughput: 99.80 tokens/s, Drafted throughput: 131.20 tokens/s, Accepted: 482 tokens, Drafted: 634 tokens, Avg Draft acceptance rate: 76.0% (MTP 401/517, lookup 81/117, 150 steps)
[stats] prompt eval time =     412.31 ms /  8192 tokens (    0.05 ms per token, 19868.31 tokens per second)
[stats]        eval time =    4829.10 ms /   632 tokens (    7.64 ms per token,   130.87 tokens per second)
```

Drafts count MTP and lookup drafts; the mean acceptance length is tokens per verify step. The prompt line counts the
tokens actually computed (cached prefix tokens excluded).

`/api/stats` counters (monitoring port): `version` and `build` (the release, e.g. `1.3.4.2`, and the engine's build hash;
since 1.3.4.2, shown in the qetop and dashboard headers), `thinking` (`requests_with_budget`, `forced_budget`, `forced_max_tokens`,
`forced_failed`, `empty_replies`, `empty_stop`, `empty_length`, `empty_after_forced_close`, `empty_retried`,
`empty_retry_recovered`, `empty_retry_exhausted`; since 1.3.4.3 `budget_default` (the `--thinking-budget` number or
per-effort map), `effort_rules` (how the chat template names efforts) and `budget_from` (thinking replies per budget
source: `request`, `effort:<level>`, `default`, `off`, `none`; an empty-reply retry counts again)), `history_drops` (reasoning-only turns left out; since 1.3.4.2, earlier `history`), `parallel` (Flash-Next
mode, reason, probe numbers), `graphs` (Flash-Next cached CUDA graphs per card: `est_mb`, `peak_mb`, `budget_mb`,
`budget_evictions`, `evicted`, `too_big`, `oom`, `eager`, `pf_eager` (prompt steps run without a graph because their
shape did not come back within `FLASH_PFGRAPH_WINDOW`), `pf_noroom` (prompt steps without a graph because the decode
graphs keep the room); a fast-growing `budget_evictions` under steady traffic means
`FLASH_GRAPH_MB` is too small for the working set).

### Failure handling and host memory

| setting | default | what it does |
|---|---|---|
| `QE_HANG_S` | `180` | no scheduler progress for this long: open requests fail at once, new ones are refused, diagnostics, exit 75. Also active during a shutdown, with twice the time. A process paused for longer (`docker pause`, SIGSTOP) also exits when it resumes. `0` = off |
| `QE_HANG_DIAG_S` | `30` | time the watchdog gives its diagnostics (stacks, `nvidia-smi`) before the exit |
| `QE_HANG_PERSIST_S` | `3600` | watchdog limit while the 27B saves conversations to `--disk-cache` |
| `QE_EXIT_HOOK_S` | `5` | on a fatal engine error, the time the server gets to send the open requests their error |
| `FLASH_RUN_TIMEOUT_S` | `120` | Flash-Next engine step monitor; off by itself under `ncu`, `nsys`, `compute-sanitizer` or a debugger unless set; `0` = off |
| `QE_START_RETRY_S` | `60` | how long a start refused for lack of room waits for a running conversation to finish |
| `QE_NO_PROGRESS_STEPS` / `QE_NO_PROGRESS_S` | `64` / `10` | a conversation without a token for this many steps and seconds (not paused) fails with an error; either `0` = off |
| `QE_GPU_LOCK_WAIT_S` | `60` | at start, wait for a previous engine that is still exiting, also one in another container (the old name `FLASH_PIN_WAIT_S` is still read when this one is not set); `0` = fail at once |
| `QE_ALLOW_SHARED_GPU` | `0` | `1` skips the one-engine-per-GPU lock (`~/.cache/qengine/locks/`) |
| `QE_CORE_DUMP` | `0` | `1` keeps core dumps of a crash (off: writing the core of a process with tens of GB of pinned memory took minutes and held the GPUs); a crash prints native and Python stacks to the log either way |
| `QE_MEM_MARGIN_GB` | max(8, 5 % of RAM) | host RAM the archives never take |
| `QE_HOST_RESERVE_GB` | max(8, 8 % of RAM) | host-RAM plan at start-up (both models): RAM left to the system and the desktop when the n-gram table, the archive and the pinned prompts are sized ([Host memory](../README.md#host-memory-flash-next)) |
| `QE_HOST_BASE_GB` | `6` | host-RAM plan: RAM counted for the server process (Python, vision encoder, CUDA) |

**Host RAM safety.** The archives (`QE_KV_ARCHIVE_GB`, `FLASH_KV_ARCHIVE_GB`) never take the machine below
`MemAvailable` minus `QE_MEM_MARGIN_GB`: pinning stops there and the archive's budget shrinks, with a log line
(`archive: host memory low, budget capped at …`). When another process takes memory later, the archive gives it back
(free pinned memory first, then archived conversations in eviction order) instead of the kernel's OOM killer ending
the server. Pinning at start that finds no host memory (for example right after a crash) is retried for up to a
minute, and the log names the processes holding the most memory. When the Flash-Next archive is full, finished and
normal conversations go first; the main agent thread gives back its newest part first and keeps a prefix it resumes
from.


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

## Build the Flash-Next model directory yourself

An alternative to the [Hugging Face download](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine). Everything
needed is in this repository plus Intel's checkpoint (169 GB download):

```bash
# 1. Intel's INT4 AutoRound checkpoint
hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir models/Qwen3.8-Flash-Next-W4A16-AutoRound
# 2. convert -> models/Qwen3.8-Flash-Next-QE-INT4 (experts repacked, DeltaNet projections INT8 g128, other dense
#    parts RTN INT4 / BF16, INT4 MTP head). --int8 computes the INT8 values with the engine's own kernel
#    (build/qe_w8q, needs one GPU), so they equal what the engine would compute from BF16
cmake --build build --target qe_w8q
python3 tools/convert_flash_next.py --mtp-int4 --int8 in_proj_qkv,in_proj_z,out_proj
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

The Hugging Face download differs from this build in one step: its shared expert and n-gram key/value projections
are INT4 calibrated with GPTQ on our own calibration text instead of round-to-nearest (next-token KL against BF16 on
the reference set in [Quality](../README.md#quality): 0.119 for the download; with round-to-nearest for those two
classes we measured 0.145). Without `--int8` the DeltaNet projections are INT4 as well, which is the 1.3.4 format
(0.267; about 4-5 % faster decode, up to 9 % at one request, about 1 GB smaller).

`convert_flash_next.py` links the n-gram shard of the download (`model-00016-of-00017.safetensors`, 102 GB) into the
converted directory; `convert_ngram_int8.py` reads it once. The engine uses the INT8 table when
`ngram_int8.safetensors` is present. Skip step 5 to keep the BF16 table (`FLASH_NGRAM=bf16`, needs the shard and
102 GB of pinned RAM). In Docker, a kept BF16 shard link must resolve inside the container: mount the download at the
same absolute path too. The converted directory is about 68 GB plus 52 GB for the INT8 table.

The three small files are optional (without them: full-vocabulary drafting, cold groups chosen by index):

| file | size | what |
|---|---:|---|
| `token_rank.bin` | 972 KB | MTP draft vocabulary ranking (the same file as the 27B's, same tokenizer) |
| `expert_prior.bin` | 96 KB | per-token routing probabilities, seed order of the expert cache |
| `cold_experts.json` | 5 KB | the expert groups that start in host memory |

The two expert files can be generated from routing counts of your own prompts (not bit-identical to the shipped
files, which come from a router trace):

```bash
FLASH_STATS=stats.bin build/flash_run --model models/Qwen3.8-Flash-Next-QE-INT4 --ids @prompt.ids --gen 512 --k 3
python3 tools/flash_expert_prior.py stats.bin models/Qwen3.8-Flash-Next-QE-INT4/expert_prior.bin
python3 tools/flash_cold_experts.py stats.bin > models/Qwen3.8-Flash-Next-QE-INT4/cold_experts.json
```

`build/flash_run --model <dir> --ids @prompt.ids --gen 256 --k 3` is a command-line runner on token ids (it uses every
GPU in `CUDA_VISIBLE_DEVICES`).

## Activation steering

Both engines can steer the model with control vectors: a direction per layer that is added to the residual stream at
a layer's output (`add`), or whose component is removed from it (`project`). A vector changes one documented trait of
the replies (for example the answer language or the tone) without changing the prompt.

### Loading vectors

Vectors are loaded when the server starts; a vector that fails a check stops the start with the reason. Without these
variables nothing is loaded and the engine runs exactly as without the feature.

| variable | meaning |
|---|---|
| `QE_STEER=<file.gguf>[:scale]` | one vector file (its sidecar: `<file>.json`, or `QE_STEER_META=<path>`); the optional scale replaces the sidecar's default. A single `QE_STEER` vector is also the default vector |
| `QE_STEER_DIR=<dir>` | every `*.gguf` in the directory, each with its sidecar; names must be unique |
| `QE_STEER_DEFAULT=<name>` | the vector applied to requests without a `steering` field (default: none) |

The log shows one line per vector (`steering: <file>, mode=…, layers lo..hi …`).

### Vector format

- **`<name>.gguf`**: a llama.cpp control-vector file: GGUF version 3, `general.architecture = "controlvector"`,
  `controlvector.layer_count`, and one F32 tensor `direction.<L>` of shape `[hidden_size]` per layer `L` (a contiguous
  range; 27B: hidden size 5120, layers 0..63; Flash-Next: 2560, layers 0..47). `direction.L` acts at the output of
  layer `L`.
- **`<name>.json`** (the sidecar), required: `direction` (how the vector was measured), `effects.intended` (the
  behaviour change it is meant to cause), `mode` (`"add"` or `"project"`, the default mode). Optional: `name`,
  `description`, `scale` (default 1), `layer_range` `[lo, hi]`, `per_layer_strength` `{"L": alpha}`, `modes`
  (`{"<other mode>": {"layer_range", "scale", "per_layer_strength"}}`), `hidden_size`, `compatibility.model`
  (one checkpoint or a list: matched against the checkpoint's `qengine_checkpoint` in its `config.json`, e.g.
  `bpx03/Qwen3.8-Flash-Next-QEngine@1.3.4.1`; checkpoints without that field are matched by directory name),
  `vector_file_sha256` (checked when present), `dir` (only `"per-layer"`).

Per row and per layer `L` in the selected range, with `h` the residual after layer `L`:

- `add`: `h ← h + scale · alpha_L · v_L`;
- `project`: `h ← h − scale · alpha_L · (h · v̂_L) v̂_L` with `v̂_L` the unit direction (`scale` 1 removes the
  component completely).

The engine refuses a file with another architecture or GGUF version, a wrong shape or dtype, non-finite values,
missing or repeated layers, a missing sidecar or one without `direction` / `effects.intended`, an unknown mode, a
negative project strength, a sha256 mismatch, a name already loaded, or a vector measured on another checkpoint
(`QE_STEER_MODEL_CHECK=0` skips that check). A refused vector is not loaded, and the server starts without it: one
`[steering] WARN` line in the log says which file and why, and `GET /v1/steering` and `/api/stats` (`steering`) list it
under `dropped`. A vector asked for explicitly (`QE_STEER_DEFAULT`, Docker `QE_STEERING=<name>` or `1`, a lone
`QE_STEER` file) that cannot be loaded stops the start with one message instead; a request naming a dropped vector
gets a 400 with the reason. Own vectors: add the checkpoint's `qengine_checkpoint` to `compatibility.model`.

### Per-request selection

Every OpenAI and Anthropic request may carry a `steering` field; at most one vector is active per request:

```json
"steering": "french"
"steering": {"vector": "french", "scale": 0.5}
"steering": {"vector": "french", "mode": "project"}
"steering": {"vector": "french", "mode": "add", "scale": 0.5, "layers": [2, 8]}
"steering": false
```

- Without the field (or `true`) the server default applies; `false` turns steering off.
- `scale` multiplies the chosen mode's default scale. `mode` overrides the default mode; without `layers` the chosen
  mode's default range applies. `layers` (`[lo, hi]`, inclusive) must lie within the file's layers.
- An unknown vector (the reply lists the loaded ones), a bad mode, a bad layer range or an unknown field gets HTTP 400.
- `GET /v1/steering` lists the loaded vectors with their defaults, file layers and per-mode defaults.

Requests with different steering are batched together; a request without steering gives the same tokens as on a
server without vectors. The MTP draft head is not steered (drafts are verified by the steered model). Not available in
Flash-Next's `pp` mode.

### Included vector (Flash-Next)

`vectors/refusal_projection.gguf` with its sidecar (`project` mode, layers 8-33), **off by default**. It projects a
refusal direction out of the residual stream, so the model declines far fewer requests.

- Removing refusals removes a safety behaviour of the model. You are responsible for what the model outputs with it.
- It shifts the model's predictions slightly on ordinary text as well. All published numbers are measured with
  steering off.

**Docker:** `QE_STEERING` picks what is loaded and the default vector:

| `QE_STEERING` | effect |
|---|---|
| `0` (default) | nothing is loaded |
| `1` | the included `refusal_projection` is the default of every request |
| `<name>` | the vector of that name (included or your own) is the default of every request |

With any value but `0`, the included vectors and your own are all loaded, so a request can pick any of them or send
`"steering": false`. Your own vectors: mount a directory at `/vectors` (read-only) with each `*.gguf` and its sidecar;
a mounted vector with the same name as an included one replaces it. A `<name>` that is not loaded stops the container
at start with the list of available names. On the 27B the included vector is not loaded (it is for Flash-Next).
Compose: `QE_STEERING=1 docker compose up -d`; for your own vectors uncomment the `/vectors` volume line and set
`QE_VECTORS_DIR`.

**From source:**

```bash
QE_STEER=vectors/refusal_projection.gguf QE_STEER_DEFAULT=refusal_projection \
  server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4
```

Without `QE_STEER_DEFAULT` the vector is loaded but only requests with `"steering": "refusal_projection"` use it.

Credit: the vector comes from [sergqwer/Strata-pr](https://github.com/sergqwer/Strata-pr) (commit 841f8c9,
`data/experimental-speed-projection`), building on [Niko1221/Strata](https://github.com/Niko1221/Strata). Thanks to
both. That repository is MIT-licensed; the vector itself is derived from the model's activations and falls under the
model's license (Qwen Community License 1.0).

### Notes

- **KV reuse is per steering setting**: cached prefixes, checkpoints, the archive and shared prefixes are kept per
  (vector, mode, layers, scale). Changing the steering in the middle of a conversation recomputes the prompt once.
- **A vector belongs to one checkpoint**: the engine checks the hidden size, the layer range and, when the sidecar
  names it, the checkpoint. Keep one vector directory per model.
- Large `add` scales (or many layers at once) degrade the text; `project` is safe over wide ranges with scale ≤ 1.

## Measurements

All on our machine (3× RTX 3090, PCIe 4.0 x16, P2P unless noted; "2 cards" are two of its cards), Flash-Next.

**Requests that arrive together** share prefill batches since 1.3.1 (October 30, 2026). Burst of N simultaneous
requests with distinct prompts, prefill throughput until the last first token and the slowest time to first token,
one prompt per batch → shared batches:

| burst | 3 cards: prefill tok/s | 3 cards: slowest TTFT | 2 cards: prefill tok/s | 2 cards: slowest TTFT |
|---|---:|---:|---:|---:|
| 4 × 1K tokens | 4,816 → 7,311 | 0.87 → 0.57 s | 2,211 → 5,169 | 1.89 → 0.81 s |
| 8 × 1K tokens | 4,246 → 6,269 | 1.97 → 1.33 s | 1,961 → 3,360 | 4.26 → 2.48 s |
| 8 × 16K tokens | 5,850 → 6,080 | 21.9 → 21.1 s | 3,061 → 4,434 | 41.9 → 28.9 s |

With 8 agents at once on 2 cards the median time to first token went from 1.10 to 0.66 s (p90 3.9 → 0.8 s) at the
same decode speed.

**Slow links.** Two cards whose link carries only 3.3 GB/s per direction (PCIe Gen2 x8 class, emulated with the
copy-engine path): prefill 4.1K tok/s at 10K and 4.5K tok/s at 90K tokens (1.3.1, October 30, 2026).

**KV cache in host RAM** (`FLASH_KV=ram` with 1M tokens against the VRAM pool, 3 cards, October 28-29, 2026):

- 4 agent conversations of 128K each, then rounds where all 4 send a follow-up at once: 2.5 s instead of 3.5 s per
  round; time to first token 0.2-0.4 s instead of 1.0-1.3 s (with the VRAM pool only 2 of them fit);
- 8 agent conversations of 64K each: 3.2-4.7 s instead of 4.3-4.7 s per round;
- a 250K prompt arriving next to 8 decoding streams: started at once and answered in 102 s, against 166 s of waiting
  for room; the 8 streams kept their speed;
- agent throughput at 8 at once with short prompts: 818 against 926 tok/s, about 12 % lower;
- one conversation at 32K: decode step 1.2 % slower, prompt 2-3 % slower.

On 1.3.1 (October 30, 2026), 4 conversations of ~64K tokens, then 3 rounds where all 4 send a follow-up at once:
output 455 → 580 tok/s and time to first token 0.48 → 0.24 s on 3 cards; 233 → 332 tok/s and 0.90 → 0.40 s on 2 cards.
10 of 10 runs with 8 conversations of 64K each completed every request without an error.

**2 cards against 3**: next-token KL of the 2-card engine against the 3-card engine 9.1e-5, within its run-to-run
noise. **INT8 n-gram table against BF16**: next-token KL on 72 held-out prompts below the run-to-run noise, the same
greedy agreement and MTP acceptance, needles found at 5K-200K tokens.

## Troubleshooting

| symptom (log line or behaviour) | what it means | what to try |
|---|---|---|
| `[comm] P2P data check FAILED (...)` at start | Data sent over P2P between two cards arrived corrupt on this machine (a PCIe / chipset problem, not a model problem). With `auto` the engine does not use the failing transport. | Nothing, if the server runs well with `auto`. If you force a path with `FLASH_COMM` / `QE_COMM`, prefer `ce`. Check `sudo dmesg \| grep -iE "aer\|pcie bus error"` for link errors. `QE_COMM_CHECK=0` turns the check off (default 4 rounds per pair and transport). |
| Prefill errors, NaN or garbage output only on long prompts, only on P2P | The P2P link may corrupt data in ways the startup check did not catch. | `FLASH_COMM=ce` (copy engine) as a workaround; compare with `FLASH_COMM=push` and `FLASH_COMM=pull` to find which P2P direction is at fault; `FLASH_COMM=host` as the last resort. Please report it with the `[comm]` startup lines. |
| `[qflash] FAULT non-finite values in the forward ...`; the request gets an error (HTTP 500 or an `error` event in a stream) | The output-head guard (`FLASH_GUARD=1`, default) caught non-finite values before they became tokens. The request fails, its conversation is dropped from the cache, the server goes on. | Note the prompt and the log lines and report them. The same fault repeated (same tokens, position and step) counts once; more than `FLASH_GUARD_MAX` (default 4) distinct faults within 5 minutes stop the process so a supervisor can restart it. `FLASH_GUARD=2` only logs, `FLASH_GUARD=3` stops at the first fault (for debugging), `0` turns the guard off. |
| Non-finite values that appear only in long-prompt prefill | Possibly the prefill attention with block selection. | Diagnostic only: `FLASH_ATTN_PFSEL=0` uses the per-row prefill attention kernel (slower at long prompts). If that fixes it, please report it. |
| `graph instantiate out of memory (N MB free): dropped K cached graphs, retrying` | A CUDA graph for a new conversation shape did not fit next to the expert cache. The engine dropped its cached graphs and retried. Since 1.3.4.1 the graphs have their own room (`FLASH_GRAPH_MB`), so this should be rare. | Nothing; graphs are rebuilt as needed. If it repeats, raise `FLASH_GRAPH_MB`. |
| `graph instantiate still out of memory (...): this step and the next 64 new shapes run without graphs` | The retry did not fit either; those steps run without a graph (same output, slower only for those steps). | If it shows up often, raise `FLASH_GRAPH_MB` (default 256 per card, e.g. `384`), at a small cost in decode and prefill speed. |
| `graph instantiate out of memory ...: freed the 64 MB emergency reserve, retrying`, then `dropped K of N cached graphs (least recently used), retrying` / `still out of memory: dropped the other ...` | Since 1.3.4 the engine first gives up a small reserve (`FLASH_OOM_RESERVE_MB`, default 64 per card; taken back only once there is clearly room again), then drops the least recently used half of its cached graphs, then the rest, before it runs a step without a graph. A graph launch that runs out of memory uses the reserve too. | Nothing; `FLASH_GRAPH_LRU_FRAC=1` drops all graphs at once, `FLASH_OOM_RESERVE_MB=0` turns the reserve off. |
| `collective timeout (host): engine job '<name>' not finished after 120 s ...; ranks still in it: ...`, exit code 75 | An engine step did not finish: a card or the driver stalled (`FLASH_RUN_TIMEOUT_S`). The open requests get an error (`[fatal] the engine is ending the process ...`), then the process ends so that it can be restarted. | Check `dmesg` for Xid lines; report the log with the line. Profiling the server with a tool that is not detected: `FLASH_RUN_TIMEOUT_S=0`. |
| `[hang] the scheduler has not advanced for N s ...` followed by stacks, exit code 75 | The server's watchdog (`QE_HANG_S`) saw no progress. Open requests got an error at once and new ones are refused; the stacks (gdb when installed and no thread is stuck in the driver, nvidia-smi) are bounded by `QE_HANG_DIAG_S` (30 s), then the process exits. | Report the log; it holds the stacks. |
| `[sched] start of N tokens refused after admission (KV pool full ...): requeued until a running sequence finishes` | A prompt that was admitted found no room when it started (for example its cached prefix is still being generated from). It tries again when a running sequence or prompt finishes, for up to `QE_START_RETRY_S` (60) seconds; when nothing is running the refusal is final at once. A prompt the engine rejects as invalid or too long fails at once with that message. | Nothing; it fails with "KV pool full" only if the pool stays full that long. |
| `engine error: no progress on slot ...` | A sequence produced no token for 64 steps and 10 s while not paused (`QE_NO_PROGRESS_STEPS`, `QE_NO_PROGRESS_S`; either `0` = off); the request fails with an error instead of waiting forever. | Report the log line (it includes the pool state). |
| `qflash: ...: cudaErrorInitializationError ... a previous engine process still holds them ...` at start | CUDA could not be used: usually the previous engine is still exiting after a crash, or the driver is in an error state. The start waits up to `QE_GPU_LOCK_WAIT_S` (60) s for a previous engine that is still exiting, also one it cannot see (e.g. in another container). | Wait for the listed process to end; check `dmesg` for Xid lines; reboot if the GPUs stay busy. |
| `[gemm] GPU<n> <buffer>: out of memory growing to ... MB: ..., retrying` | A GEMM workspace had to grow after start and did not fit; the engine dropped cached graphs and retried. | As above. |
| The server does not start: `not enough GPU memory for a KV cache` (27B) | Too many sequence slots for the free VRAM (e.g. one card). | Lower `QE_MAX_SEQS`, leave `--vision` off, or lower `QE_RESERVE_GB`. |
| A desktop on one of the cards stutters or runs out of VRAM | The engine filled the card. | Flash-Next: `FLASH_EC_MARGIN_MB=256,2048,256` (one value per card); 27B: `QE_RESERVE_GB=1.5,3,1.5`. |
