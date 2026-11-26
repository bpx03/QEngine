# Qwen3.8-Flash-Next

Back to the [README](../../README.md). Settings for every model: [common settings](../advanced.md#common-settings).

**Qwen3.8-Flash-Next**, run from my checkpoint [bpx03/Qwen3.8-Flash-Next-QEngine](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine)
(routed experts INT4 from Intel's AutoRound quantization, repacked bit-exact; the parts every token passes through in
INT8 or calibrated INT4): a sparse mixture-of-experts model with 512 routed experts, 262K
context (up to 1M opt-in) and image input. Runs on 2 or more cards (tuned for 3 and 2).

**Weights**: routed experts INT4 (Intel's AutoRound, repacked bit-exact); shared expert and dense parts INT8 or calibrated INT4; DeltaNet projections INT8. Headline speed and quality: the [model table](../../README.md#models); KL against BF16 in detail: [Quality](../benchmarks.md#quality). Docker and the source start: [Quick start](../../README.md#quick-start).

## How it runs

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
- 16 conversations at once by default (`FLASH_MAX_SEQS`, 1 to 32) with batched decoding; prompts that arrive together share prefill batches (the
  expert weights are read once for all of them);
- prefix reuse, checkpoints and a host-RAM archive of conversations; image input (`--vision 1`); structured output.

**Model files.** The [Hugging Face download](https://huggingface.co/bpx03/Qwen3.8-Flash-Next-QEngine) is ready to
run. Since 1.3.4.1 its default revision stores the DeltaNet projections (in_proj_qkv / z / out_proj) as INT8 (group
128) and has the shared expert and the n-gram key/value projections calibrated with GPTQ (INT4); the routed experts
are unchanged. That brings the model about twice as close to BF16 (see
[Quality](../benchmarks.md#quality)) and costs decode speed: about 5 % with 8 conversations, about 4 % single-stream on the club-3090 bench and up to
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
documents; see [docs/advanced.md](../advanced.md#context-up-to-1m-tokens-yarn-opt-in).

**KV cache in host RAM (opt-in)**: by default the attention KV of all conversations shares one VRAM pool of 262,144
tokens. With `FLASH_KV=ram` it lives in pinned host RAM (`FLASH_KV_RAM_TOKENS`, default `1M`, 13,520 bytes per token
for the whole machine) and VRAM keeps the indexer keys and a cache of recent blocks; output is bit-identical. Many
long conversations then stay resident: with 4 conversations of ~64K tokens sending follow-ups at once, output went
from 455 to 580 tok/s on 3 cards and from 233 to 332 tok/s on 2 cards (October 30, 2026). It is not the default
because short-request throughput is about 12 % lower (the indexer keys and block cache take VRAM from the experts).
Details: [docs/advanced.md](../advanced.md#kv-cache-in-host-ram-flash-next).

## Settings

| setting | default | what it does |
|---|---|---|
| `FLASH_PAR` | `auto` | `auto`, `tp`, `pp`, `pptp` (experimental), `pp_prefill` (pipelined prefill, `tp` decode; 58 GB or more per card); see [Parallel mode](../hardware.md#parallel-mode-flash-next-flash_par) |
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
| `FLASH_KV_BITS` | `8` | `4`: 4-bit KV cache (opt-in, not lossless against FP8): 7,488 instead of 13,520 bytes per token, about 1.8x the tokens with `FLASH_KV=ram`; see [Host memory](../hardware.md#host-memory-flash-next) |
| `FLASH_KV_ARCHIVE_GB` | `48` (less on smaller machines) | host-RAM archive of conversations; `0` = off, which on Flash-Next also turns off the pinned prompts and prefix reuse across slots, with `FLASH_KV=ram` too; a set value wins over the [host-RAM plan](../hardware.md#host-memory-flash-next). The archive's memory is pinned in the background, 1 GB at a time, only while the engine is idle and the host has at least 8 GB to spare above `QE_MEM_MARGIN_GB` (never next to a running step: [#32](https://github.com/bpx03/QEngine/issues/32)); a registration slower than 1 s, which is a sign of host memory pressure, pauses pinning for 30 s, and the pause doubles each time it repeats, up to 8 min. Until the pool has grown, new archive copies use plain memory (slower copies, same result) |
| `FLASH_PREFIX_GB` | `8` (less on smaller machines) | host RAM for pinned system prompts; `0` = off |
| `FLASH_NGRAM_PIN` | set by the host-RAM plan (`1` when it fits) | `1`: pin the n-gram table; `0`: read it through the page cache |
| `FLASH_WEIGHTS_COLD` | `1` | after the load, drop the page cache of the model files the engine no longer reads (all but the n-gram table in page-cache mode); `0` = keep it, and also turns off the load-time drop (`QE_LOAD_DROP`) |
| `FLASH_TP_SPIN_US` | `5000` | µs a tensor-parallel helper thread spins after its last job before it sleeps (an idle server then uses no CPU); `-1` = always spin |
| `FLASH_VRAM_ADAPT` | `1` | Flash-Next: when another process takes GPU memory while serving, move resident expert groups to host memory and back later; `0` off |
| `FLASH_VRAM_LOW_MB` / `FLASH_VRAM_TARGET_MB` | `128` / `256` | free memory per card below which groups move to host memory / up to which they move; they come back as soon as the other processes give the memory back (keeping 16 MB above the low mark free) |
| `FLASH_CTX_FIT` | `1` | Flash-Next: lower the context at start to the largest that fits the smallest card (logged); `0` keeps `--max-ctx` as given |
| `FLASH_PRELOAD` | `1` | Flash-Next: when CUDA was started before the engine (the vision encoder) with lazy module loading, load the engine's kernels at start instead of at their first use; `0` off |
| `QE_RAM_WAIT_S` | `20` | seconds the start waits for free host RAM (MemAvailable) when it is short of the full plan (as long as it rises, or while the previous QEngine server on these cards is still exiting with 1 GB or more on them); `0` plans at once |
| `QE_NO_ROOM_S` | `10` | seconds a queued request that fits nowhere waits with nothing running, then it ends with an error that says why; `0` = no deadline |
| `FLASH_NGRAM_SPIN_MS` | `50` | page-cache mode of the n-gram table: ms the row-copy thread spins after the last request, then it polls every 50 µs (1 ms after a second); `-1` = always spin |
| `QE_HOST_RESERVE_GB` | max(8, 8 % of RAM) | at start only: RAM the plan leaves to the system and the desktop (both models). `QE_MEM_MARGIN_GB` is the one that acts while serving |
| `FLASH_RUN_TIMEOUT_S` | `120` | engine step monitor in seconds; `0` = off |

## Benchmarks

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

**Many agents in flight** (1.5.0 code, November 19, 2026; 3 cards, P2P, defaults; the 27B's agent replay ([27B benchmarks](qwen3.8-27b.md#benchmarks))
with 64-token replies, warm-up 20 s (64 agents: 60 s), 60 s window,
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
[Parallel mode](../hardware.md#parallel-mode-flash-next-flash_par) and [With and without GPU P2P](../hardware.md#with-and-without-gpu-p2p).
Burst prefill, slow links and the KV-in-RAM runs: [docs/advanced.md](../advanced.md#measurements).
