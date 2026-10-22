# QEngine on 2× RTX 3090: 2-GPU vs 3-GPU, and QEngine vs vLLM on the same 2 cards

This run repeats the qengine part of the published three-way comparison ([`../compare_2026-10/`](../compare_2026-10/README.md))
with qengine limited to **two** RTX 3090s, GPU0 and GPU2. vLLM used the same two cards in that comparison. It gives two
comparisons:

- **qengine-2gpu vs qengine-now**: the same engine on 2 cards (tensor parallel 2) vs 3 cards (tensor parallel 3).
- **qengine-2gpu vs vllm-best**: qengine vs vLLM 0.28.0 (club-3090 "dual-fast" recipe) on identical 2× RTX 3090
  hardware.

Measured 2026-10-22, 16:50–18:34, on a single server instance. The 3-GPU qengine numbers and all vLLM numbers come
from the 2026-10-21/22 run in `compare_2026-10` and were not re-measured. `vllm-best` = per point the better of
vllm-mtp and vllm-nospec (`../compare_2026-10/best_vllm.csv`).

## Engine

| name | what | GPUs |
|---|---|---|
| `qengine-2gpu` | QEngine build `9494c40` (private development history; the first public commit of this repository has the same engine code). Its engine sources (`src/`) are identical to `1be10df` (= qengine-now). The later commits only touch the monitoring dashboard. Uses the existing `build/libqengine.so`. Bare metal. | 2× RTX 3090 (GPU0 + GPU2), tensor parallel 2 |

```
QE_MAX_SEQS=32 CUDA_VISIBLE_DEVICES=0,2 python3 server/app.py --host 127.0.0.1 --port 8181 --devices 0,1 \
    --think 0 --metrics-port 19394 --stats-file <scratch>/qe_stats_2gpu.json
```
These are the same arguments as qengine-now except the devices. At the time `server/run.sh` hard-coded all three GPUs, so
`app.py` was started directly. At startup the server reported a KV pool of 387,072 tokens shared by 32 sequence slots
(3 GPUs: 991,232). Per GPU: 7.16 GB weights and 13.03 GB cache + activations. The decode all-reduce runs GPU to
GPU. Details are in [`engines.json`](engines.json).

## Hardware and state

- 2 of the 3 RTX 3090 24 GB cards: GPU0 and GPU2. PCIe 4.0 x16, no NVLink. GPU1
  drives the desktop and was not used.
- Same host, driver (615.71.09), CUDA 13.1, vLLM 0.28.0 client, model directory and ShareGPT file as in
  `compare_2026-10`. Nothing else ran on GPU0/GPU2.

## Methodology

Identical to `compare_2026-10`. The same harness (`driver.py` of the public harness, with only the engine entry
added) drove `bench/serving_suite.py` → `vllm bench serve`. Commands, seeds, prompt counts, sampling, thinking off,
`--ignore-eos`, the 2-request warm-up before every point and the 5 s pauses are all the same. Every classic and
goodput command line matches qengine-now's line in `../compare_2026-10/commands.txt` byte for byte, apart from the
engine label. See [`commands.txt`](commands.txt).

Differences from the original run:
- **Prefill: 1 measured run per length** (plus the warm-up). The original took the median of 3 runs. The single run
  uses the same seed as the original's rep 1. `prefill/summary.csv` has `reps=1`, so "median" there is that single
  run.
- **No drift check.** A single pass was run.

## Results

### Classic — output tokens/s

| scenario | c | qengine-2gpu | qengine-now (3 GPUs) | vllm-best (2 GPUs) | 2gpu / 3gpu | 2gpu / vllm-best |
|---|---:|---:|---:|---:|---:|---:|
| short | 1 | 162 | 208 | 93 (mtp) | 0.78x | 1.74x |
| short | 8 | 608 | 794 | 456 (mtp) | 0.76x | 1.33x |
| short | 16 | 738 | 949 | 609 (nospec) | 0.78x | 1.21x |
| short | 32 | 739 | 1,001 | 797 (nospec) | 0.74x | 0.93x |
| rag16k | 1 | 31 | 42 | 23 (mtp) | 0.73x | 1.36x |
| rag16k | 8 | 31 | 43 | 30 (nospec) | 0.73x | 1.02x |
| agent | 1 | 240 | 287 | 128 (mtp) | 0.84x | 1.87x |
| agent | 8 | 827 | 1,080 | 635 (mtp) | 0.77x | 1.30x |
| agent | 16 | 1,036 | 1,302 | 857 (mtp) | 0.80x | 1.21x |
| agent | 32 | 1,031 | 1,423 | 1,219 (mtp) | 0.72x | 0.85x |
| sharegpt | 1 | 178 | 243 | 87 (mtp) | 0.73x | 2.04x |
| sharegpt | 8 | 677 | 819 | 414 (mtp) | 0.83x | 1.64x |
| sharegpt | 16 | 797 | 935 | 523 (mtp) | 0.85x | 1.52x |
| sharegpt | 32 | 883 | 1,110 | 801 (nospec) | 0.80x | 1.10x |

TTFT p50 / TPOT p50 (ms). For vllm-best, the values are those of the variant that had the higher throughput at that point:

| scenario | c | qengine-2gpu | qengine-now (3 GPUs) | vllm-best (2 GPUs) |
|---|---:|---:|---:|---:|
| short | 1 | 243 / 4.4 | 176 / 3.9 | 326 / 10.2 |
| short | 8 | 276 / 10.7 | 199 / 8.3 | 670 / 14.9 |
| short | 16 | 456 / 18.5 | 322 / 14.6 | 3,872 / 18.7 |
| short | 32 | 7,857 / 25.6 | 299 / 28.5 | 7,170 / 26.1 |
| rag16k | 1 | 3,408 / 5.2 | 2,402 / 4.3 | 4,232 / 11.1 |
| rag16k | 8 | 30,541 / 12.2 | 22,307 / 9.6 | 18,365 / 115.0 |
| agent | 1 | 145 / 4.0 | 107 / 3.5 | 170 / 7.5 |
| agent | 8 | 191 / 7.4 | 149 / 5.9 | 412 / 10.8 |
| agent | 16 | 212 / 12.7 | 173 / 10.5 | 461 / 16.5 |
| agent | 32 | 7,053 / 18.3 | 213 / 19.1 | 677 / 22.8 |
| sharegpt | 1 | 81 / 5.1 | 66 / 3.8 | 150 / 11.5 |
| sharegpt | 8 | 193 / 9.2 | 152 / 8.2 | 258 / 14.8 |
| sharegpt | 16 | 146 / 16.0 | 119 / 13.8 | 351 / 22.2 |
| sharegpt | 32 | 2,362 / 24.0 | 243 / 26.0 | 198 / 31.6 |

### Goodput — goodput req/s (% of requests within SLA) · output tok/s

SLA: TTFT ≤ 3000 ms and TPOT ≤ 50 ms per request, Poisson arrivals.

| scenario | rate | qengine-2gpu | qengine-now (3 GPUs) | vllm-best (2 GPUs) |
|---|---:|---:|---:|---:|
| goodput_chat | 0.5 | 0.50 (100%) · 126 | 0.50 (100%) · 126 | 0.49 (100%) · 125 (mtp) |
| goodput_chat | 1 | 0.98 (100%) · 255 | 0.99 (100%) · 256 | 0.96 (100%) · 250 (nospec) |
| goodput_chat | 2 | 0.98 (51%) · 489 | 1.98 (100%) · 509 | 0.27 (15%) · 478 (nospec) |
| goodput_chat | 3 | 0.10 (5%) · 497 | 0.42 (16%) · 667 | 0.00 (0%) · 490 (nospec) |
| goodput_chat | 4 | 0.05 (2%) · 474 | 0.14 (6%) · 645 | 0.01 (0%) · 490 (nospec) |
| goodput_long16k | 0.1 | 0.08 (88%) · 54 | 0.10 (100%) · 55 | 0.07 (79%) · 53 (nospec) |
| goodput_long16k | 0.25 | 0.19 (82%) · 120 | 0.22 (90%) · 123 | 0.18 (77%) · 119 (nospec) |
| goodput_long16k | 0.5 | 0.09 (19%) · 229 | 0.33 (68%) · 243 | 0.09 (18%) · 233 (nospec) |
| goodput_long16k | 1 | 0.00 (0%) · 218 | 0.01 (2%) · 371 | 0.00 (0%) · 280 (nospec) |
| goodput_agent | 0.25 | 0.24 (100%) · 243 | 0.24 (100%) · 246 | 0.23 (97%) · 240 (mtp) |
| goodput_agent | 0.5 | 0.47 (98%) · 497 | 0.49 (100%) · 510 | 0.46 (100%) · 475 (nospec) |
| goodput_agent | 1 | 0.05 (9%) · 631 | 0.18 (22%) · 831 | 0.10 (15%) · 652 (nospec) |
| goodput_agent | 2 | 0.01 (1%) · 633 | 0.06 (7%) · 902 | 0.04 (5%) · 694 (nospec) |

### Prefill — TTFT · prompt tokens/s (c=1, unique prompt)

| prompt tokens | qengine-2gpu (1 run) | qengine-now (3 GPUs, median of 3) | vllm-best (2 GPUs, median of 3) | 2gpu speed vs 3gpu | 2gpu vs vllm-best |
|---:|---:|---:|---:|---:|---:|
| 1037 | 0.24 s · 4,274 | 0.18 s · 5,791 | 0.31 s · 3,393 (nospec) | 0.74x | 1.26x faster |
| 8205 | 1.78 s · 4,617 | 1.27 s · 6,485 | 2.07 s · 3,966 (nospec) | 0.71x | 1.16x faster |
| 32781 | 7.67 s · 4,275 | 5.35 s · 6,122 | 9.86 s · 3,324 (nospec) | 0.70x | 1.29x faster |
| 100012 | 28.83 s · 3,468 | 19.86 s · 5,037 | 44.42 s · 2,252 (nospec) | 0.69x | 1.54x faster |
| 200012 | 74.13 s · 2,698 | 50.67 s · 3,947 | 130.49 s · 1,533 (nospec) | 0.68x | 1.76x faster |

The warm-up runs (kept as `len<L>_warmup.json`) agree with the measured runs to within 3% at 8K–200K. At 1K the
warm-up was 20% slower (292 vs 243 ms).

## Notes and observations

- **Crashes / failed requests:** none. All 27 classic/goodput points and all 10 prefill runs completed, with 0 failed
  requests.
- **2 vs 3 GPUs:** on 2 cards, qengine delivers 72–85% of its 3-card throughput in classic and 68–74% in prefill.
- **qengine vs vLLM on the same 2 cards:** qengine is faster at every classic point up to c=16 (1.2–2.0×). The
  exceptions are c=32 for short (0.93×) and agent (0.85×). qengine-2gpu has lower TPOT at every point. Prefill is
  1.16–1.76× faster, and the gap grows with length. Goodput: qengine-2gpu ≥ vllm-best at every rate except
  goodput_agent 1–2 req/s. goodput_chat at 2 req/s: 51% within SLA vs 15%.
- **c=32 TTFT on 2 GPUs:** TTFT p50 rises sharply at c=32: short 7.9 s, agent 7.1 s, sharegpt 2.4 s (3 GPUs:
  0.2–0.3 s). Throughput from c=16 to c=32 stays flat (short 738 → 739, agent 1,036 → 1,031). The server log shows
  that over the whole run only slots 0–22 were ever used. The 3-GPU run used all 32. So on 2 GPUs about 23 sequences
  decode at once and the rest wait in the queue, even though `QE_MAX_SEQS=32` is set and the KV pool (387K tokens)
  is far from full at these lengths. The cause was not investigated in this run. It was found afterwards: the admission check reserved
  16,384 tokens of growth per running sequence, which on the 2-GPU pool admitted only 22 of 32 sequences. The
  default is now 8,192 (`QE_ADMIT_HEADROOM`), and batched decode passes of ≥ 33 rows use the W4A8 GEMM
  (`QE_DEC_GEMM=33`); these numbers were measured before that change.
- TPOT is the vLLM definition: (E2E latency − TTFT) / (output tokens − 1), per request.

## Files

```
README.md, engines.json, commands.txt
classic/summary.csv   classic/<scenario>/qengine-2gpu/c<C>.json + .log
goodput/summary.csv   goodput/<scenario>/qengine-2gpu/rate<R>.json + .log
prefill/summary.csv   prefill/qengine-2gpu/len<L>_rep1.json + .log (and _warmup)
logs/qengine-2gpu/    full server log
```
The summary CSVs have the same columns as in `compare_2026-10` (same `build_summary.py` logic). The raw JSON
outputs were not edited.
