# QEngine vs vLLM — Qwen3.8-27B INT4 on 3× RTX 3090 (October 2026)

Serving benchmark of **QEngine** (a custom C++/CUDA inference engine) against **vLLM**. Both serve the same
Qwen3.8-27B INT4 checkpoint through the same OpenAI-compatible chat API. Every number below comes from
`vllm bench serve` (vLLM 0.28.0's own benchmark client). The same client, parameters, seeds and prompts were used for
every engine. The raw result JSON and full stdout of every run are in this directory. Raw outputs were not edited;
the summary CSVs are generated from them.

Measured 2026-10-21/22, engine by engine, each on a freshly started server.

## Engines

| name | what | GPUs |
|---|---|---|
| `qengine-now` | QEngine build `1be10df` (2026-10-21) | 3× RTX 3090, tensor parallel 3 |
| `qengine-sunday` | QEngine build `1c3f603` (2026-10-18), the previous reference build | 3× RTX 3090, tensor parallel 3 |
| `vllm-mtp` | vLLM 0.28.0 + club-3090 patches, club-3090 "dual-fast" recipe, MTP speculative decoding (4 draft tokens) | 2× RTX 3090, tensor parallel 2 |
| `vllm-nospec` | same as `vllm-mtp` without speculative decoding | 2× RTX 3090, tensor parallel 2 |
| `vllm-best` | per point, the better of the two vLLM variants (`best_vllm.csv`) | |

The two QEngine commit ids refer to the private development history this repository was published from;
The engine code published here (`src/`) is that of `qengine-now` (comments edited for publication); the server
has since changed only in the monitoring UI and in path and language defaults.

**GPU count.** vLLM uses 2 GPUs and QEngine uses 3. Tensor parallel 3 is not possible in vLLM for this model:
its 4 KV heads and 16 DeltaNet key heads are not divisible by 3. TP=2 on two cards is the strongest vLLM layout
available on this box, and it is the configuration the community recipe uses. Keep this in mind when reading
per-GPU efficiency. Every table here compares whole-system throughput and latency.

**vLLM configuration.** This is the most-starred community recipe for this model on RTX 3090s, club-3090
`qwen3.8-27b dual-fast` (https://github.com/noonghunna/club-3090), ported flag for flag to stock vLLM 0.28.0 (launcher
[`bench/vllm_cfg_best.sh`](../../vllm_cfg_best.sh); deviations: our checkpoint instead of the AutoRound one,
`--disable-custom-all-reduce`, no reasoning/tool parsers). It uses W4A8 Marlin (INT4 weights, INT8 activations), fp8 KV cache, prefix caching,
chunked prefill, 32 sequences and 262,144 context. vLLM was run with the two club-3090
runtime patches that fix MTP crashes under concurrency: FlashInfer decode `pin_memory=False`
([vllm#40756](https://github.com/vllm-project/vllm/issues/40756)) and `gpu_model_runner` waiting for
`num_accepted_tokens_event` ([vllm#50021](https://github.com/vllm-project/vllm/issues/50021),
[vllm#52873](https://github.com/vllm-project/vllm/issues/52873)); both are taken unchanged from
[club-3090](https://github.com/noonghunna/club-3090). vLLM is only the comparison baseline; QEngine does not depend on it.
vLLM 0.28.0 is the newest release that runs with this host's toolchain: 0.29–0.32 pull in FlashInfer 0.7, whose JIT
compilation fails with the host compiler (GCC 16).

Full server command lines, environment variables and versions are in [`engines.json`](engines.json).

## Hardware and software

- 3× NVIDIA GeForce RTX 3090 24 GB, PCIe 4.0 x16, no NVLink. GPU1 also drives the desktop (light load).
- AMD Ryzen Threadripper 7960X (24 cores), 256 GB RAM, Fedora 44, kernel 7.2.5.
- NVIDIA driver 615.71.09 (open kernel modules with a PCIe BAR1 peer-to-peer patch; the same driver for all
  engines), CUDA 13.1.
- vLLM 0.28.0, torch 2.13.0+cu130, FlashInfer 0.6.16.post3, Python 3.12.
- Model: Qwen3.8-27B INT4 (RedHatAI W4A16 compressed-tensors checkpoint + MTP head). All engines load the same
  directory and use the same tokenizer. Both QEngine builds also used the two QEngine release assets in that
  directory (`token_rank.bin`, the draft vocabulary ranking, and `model_mtp_ft.safetensors`, the fine-tuned MTP
  head; see the top-level README). vLLM uses the checkpoint's own MTP head.

### Server commands

```
# qengine (both builds; run from the respective checkout)
QE_MAX_SEQS=32 server/run.sh 8181 --think 0 --metrics-port 19394 --stats-file <scratch>/qe_stats.json
#   server/run.sh = CUDA_VISIBLE_DEVICES=0,1,2 python3 server/app.py --host 127.0.0.1 --port 8181 ...
#   (max context 262144, built-in MTP with up to 7 adaptive drafts: engine defaults)

# vllm-mtp
bench/vllm_cfg_best.sh 8182            # SPEC_N=4 (default)
# vllm-nospec
SPEC_N=0 bench/vllm_cfg_best.sh 8182
```
The expanded `vllm serve` command is in `engines.json`.

## Methodology

- **Client:** `vllm bench serve` from vLLM 0.28.0, `--backend openai-chat --endpoint /v1/chat/completions`, driven
  by [`bench/serving_suite.py`](../../serving_suite.py) (the same script for every engine); the whole run is
  scripted in [`bench/compare/`](../../compare/) (`run_all.sh` → `driver.py`, then `build_summary.py`). Sampling is sent explicitly
  with every request: `--temperature 0.7 --top-p 0.8 --top-k 20`. Thinking is off via
  `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`. `--ignore-eos` is set for all synthetic
  datasets, so output length is fixed; ShareGPT uses its real lengths and stops at EOS.
- **Prompts:** vLLM's `random` dataset (random token ids, a random vocabulary offset per prompt) and ShareGPT V3
  (`ShareGPT_V3_unfiltered_cleaned_split.json` from
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered, placed in `bench/data/`). Every (scenario, point) has its own seed, identical for all engines.
  So all engines receive the same prompts, and no point reuses another point's prompts (no prefix-cache hits
  across points).
- **Warm-up:** before every point, an unsaved 2-request warm-up (1024/512 tokens, separate seed). vLLM's built-in
  `--num-warmups` is not used, because it repeats the first measured prompt.
- **Order:** engine by engine (qengine-sunday, vllm-nospec, qengine-now, vllm-mtp), each block = classic → goodput →
  prefill on one server instance. A drift check comes last. Only one engine runs at a time. There is a 5 s pause between points.
- Every exact command line is in [`commands.txt`](commands.txt).

### Classic (fixed concurrency)
`--max-concurrency C --request-rate inf`, C ∈ {1, 8, 16, 32}.

| scenario | input / output tokens | prompts |
|---|---|---|
| short | 1024 / 512 | 8 at c=1, 4·C otherwise |
| rag16k | 15000 / 128 (c=1, 8 only) | 8 at c=1, 16 at c=8 |
| agent | 500 / 2000 | 8 at c=1, 2·C otherwise |
| sharegpt | ShareGPT conversations: the first user turn is the prompt; max tokens = the reference reply's length, stops at EOS | 16 at c=1, 4·C otherwise |

### Goodput (Poisson arrivals, SLA)
`--request-rate R --burstiness 1.0`, with no client concurrency cap: the server queues, and the queueing shows in TTFT.
**SLA: TTFT ≤ 3000 ms and TPOT ≤ 50 ms per request** (`--goodput ttft:3000 tpot:50`). Goodput = requests per second
that meet both limits; "% in SLA" = goodput / request throughput. The prompt is a shared prefix plus a unique
part, with ±50% length variation (`--random-range-ratio 0.5`).

| scenario | shared prefix + unique input / output | rates (req/s) | duration per rate |
|---|---|---|---|
| goodput_chat | 1024 + 1024 / 256 | 0.5, 1, 2, 3, 4 | 180 s of arrivals |
| goodput_long16k | 12288 + 4096 / 512 | 0.1, 0.25, 0.5, 1 | 240 s |
| goodput_agent | 4096 + 2048 / 1024 | 0.25, 0.5, 1, 2 | 240 s |

### Prefill (TTFT)
A single unique prompt per run: `--dataset-name random --random-input-len L --random-output-len 1
--random-range-ratio 0 --max-concurrency 1 --num-prompts 1`, L ∈ {1024, 8192, 32768, 100000, 200000}. Each length
gets 1 warm-up and 3 measured runs, each with a different seed, so no request can be served from a prefix cache.
Reported: median TTFT of the 3 runs, and prompt tok/s = prompt tokens / TTFT. TTFT covers the whole client-side
time to the first token (HTTP, tokenization, prefill, first sample). Warm-up results are kept as
`len<L>_warmup.json` and are not in the summary. vLLM accepted the 200K prompts (max-model-len 262144).

### Drift
At the end, `short` c=1 and c=32 were repeated for every engine on a freshly started server, with the same prompts as the
classic run. This measures how stable each engine was over the session.

## Results

### Classic — output tokens/s

| scenario | c | qengine-now | qengine-sunday | vllm-mtp | vllm-nospec | vllm-best | qengine-now / vllm-best |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 208 | 218 | 93 | 67 | 93 (mtp) | 2.24x |
| short | 8 | 794 | 681 | 456 | 370 | 456 (mtp) | 1.74x |
| short | 16 | 949 | 796 | 576 | 609 | 609 (nospec) | 1.56x |
| short | 32 | 1,001 | 812 | 720 | 797 | 797 (nospec) | 1.26x |
| rag16k | 1 | 42 | 36 | 23 | 22 | 23 (mtp) | 1.86x |
| rag16k | 8 | 43 | 37 | 29 | 30 | 30 (nospec) | 1.40x |
| agent | 1 | 287 | 270 | 128 | 69 | 128 (mtp) | 2.23x |
| agent | 8 | 1,080 | 972 | 635 | 437 | 635 (mtp) | 1.70x |
| agent | 16 | 1,302 | 1,089 | 857 | 812 | 857 (mtp) | 1.52x |
| agent | 32 | 1,423 | 1,166 | 1,219 | 1,179 | 1,219 (mtp) | 1.17x |
| sharegpt | 1 | 243 | 224 | 87 | 65 | 87 (mtp) | 2.78x |
| sharegpt | 8 | 819 | 742 | 414 | 318 | 414 (mtp) | 1.98x |
| sharegpt | 16 | 935 | 833 | 523 | 476 | 523 (mtp) | 1.79x |
| sharegpt | 32 | 1,110 | 888 | 780 | 801 | 801 (nospec) | 1.39x |


TTFT p50 / TPOT p50 (ms):


| scenario | c | qengine-now | qengine-sunday | vllm-mtp | vllm-nospec |
|---|---:|---:|---:|---:|---:|
| short | 1 | 176 / 3.9 | 205 / 3.8 | 326 / 10.2 | 307 / 14.4 |
| short | 8 | 199 / 8.3 | 238 / 9.2 | 670 / 14.9 | 1,993 / 17.8 |
| short | 16 | 322 / 14.6 | 621 / 17.6 | 743 / 24.9 | 3,872 / 18.7 |
| short | 32 | 299 / 28.5 | 235 / 38.1 | 1,083 / 39.7 | 7,170 / 26.1 |
| rag16k | 1 | 2,402 / 4.3 | 2,896 / 4.7 | 4,232 / 11.1 | 4,024 / 14.7 |
| rag16k | 8 | 22,307 / 9.6 | 25,790 / 10.3 | 12,277 / 164.7 | 18,365 / 115.0 |
| agent | 1 | 107 / 3.5 | 128 / 3.4 | 170 / 7.5 | 143 / 14.4 |
| agent | 8 | 149 / 5.9 | 175 / 6.7 | 412 / 10.8 | 995 / 17.8 |
| agent | 16 | 173 / 10.5 | 202 / 12.6 | 461 / 16.5 | 1,840 / 18.8 |
| agent | 32 | 213 / 19.1 | 294 / 26.4 | 677 / 22.8 | 3,695 / 25.3 |
| sharegpt | 1 | 66 / 3.8 | 75 / 4.3 | 150 / 11.5 | 109 / 14.4 |
| sharegpt | 8 | 152 / 8.2 | 167 / 8.9 | 258 / 14.8 | 146 / 19.2 |
| sharegpt | 16 | 119 / 13.8 | 123 / 16.4 | 351 / 22.2 | 168 / 21.8 |
| sharegpt | 32 | 243 / 26.0 | 154 / 34.2 | 484 / 33.3 | 198 / 31.6 |

### Goodput — goodput req/s (% of requests within SLA) · output tok/s


| scenario | rate | qengine-now | qengine-sunday | vllm-mtp | vllm-nospec |
|---|---:|---:|---:|---:|---:|
| goodput_chat | 0.5 | 0.50 (100%) · 126 | 0.50 (100%) · 126 | 0.49 (100%) · 125 | 0.49 (100%) · 124 |
| goodput_chat | 1 | 0.99 (100%) · 256 | 0.99 (100%) · 255 | 0.93 (97%) · 249 * | 0.96 (100%) · 250 |
| goodput_chat | 2 | 1.98 (100%) · 509 | 1.91 (97%) · 506 | 0.00 (0%) · 341 | 0.27 (15%) · 478 |
| goodput_chat | 3 | 0.42 (16%) · 667 | 0.14 (6%) · 577 | 0.00 (0%) · 340 | 0.00 (0%) · 490 |
| goodput_chat | 4 | 0.14 (6%) · 645 | 0.08 (4%) · 559 | 0.00 (0%) · 336 | 0.01 (0%) · 490 |
| goodput_long16k | 0.1 | 0.10 (100%) · 55 | 0.09 (92%) · 54 | 0.07 (71%) · 54 | 0.07 (79%) · 53 |
| goodput_long16k | 0.25 | 0.22 (90%) · 123 | 0.20 (83%) · 122 | 0.15 (63%) · 118 | 0.18 (77%) · 119 |
| goodput_long16k | 0.5 | 0.33 (68%) · 243 | 0.23 (48%) · 238 | 0.00 (0%) · 190 | 0.09 (18%) · 233 |
| goodput_long16k | 1 | 0.01 (2%) · 371 | 0.00 (1%) · 306 | 0.00 (0%) · 205 | 0.00 (0%) · 280 |
| goodput_agent | 0.25 | 0.24 (100%) · 246 | 0.24 (100%) · 246 | 0.23 (97%) · 240 | 0.23 (100%) · 229 |
| goodput_agent | 0.5 | 0.49 (100%) · 510 | 0.49 (100%) · 506 | 0.33 (72%) · 477 | 0.46 (100%) · 475 |
| goodput_agent | 1 | 0.18 (22%) · 831 | 0.10 (14%) · 745 | 0.00 (0%) · 473 | 0.10 (15%) · 652 |
| goodput_agent | 2 | 0.06 (7%) · 902 | 0.01 (2%) · 713 | 0.00 (0%) · 499 | 0.04 (5%) · 694 |

`*` 1 of 180 requests failed on the client side (`aiohttp ServerDisconnectedError`). The server stayed up and logged
no error; see notes.

### Prefill — median TTFT · prompt tokens/s (c=1, unique prompts)


| prompt tokens | qengine-now | qengine-sunday | vllm-mtp | vllm-nospec | qengine-now vs vllm-best |
|---:|---:|---:|---:|---:|---:|
| 1037 | 0.18 s · 5,791 | 0.20 s · 5,209 | 0.32 s · 3,289 | 0.31 s · 3,393 | 1.71x faster |
| 8205 | 1.27 s · 6,485 | 1.47 s · 5,570 | 2.19 s · 3,739 | 2.07 s · 3,966 | 1.64x faster |
| 32780 | 5.35 s · 6,122 | 6.66 s · 4,926 | 10.38 s · 3,158 | 9.86 s · 3,324 | 1.84x faster |
| 100012 | 19.86 s · 5,037 | 27.83 s · 3,594 | 45.96 s · 2,176 | 44.42 s · 2,252 | 2.24x faster |
| 200013 | 50.67 s · 3,947 | 78.20 s · 2,558 | 134.19 s · 1,491 | 130.49 s · 1,533 | 2.58x faster |

### Drift (short, repeated at the end of the session)


| engine | c | classic tok/s | drift tok/s | Δ |
|---|---:|---:|---:|---:|
| vllm-mtp | 1 | 93 | 97 | +4.9% |
| vllm-mtp | 32 | 720 | 715 | -0.6% |
| vllm-nospec | 1 | 67 | 67 | +0.1% |
| vllm-nospec | 32 | 797 | 796 | -0.2% |
| qengine-sunday | 1 | 218 | 220 | +0.7% |
| qengine-sunday | 32 | 812 | 829 | +2.2% |
| qengine-now | 1 | 208 | 202 | -2.6% |
| qengine-now | 32 | 1,001 | 1,004 | +0.3% |

All engines reproduce within ±5% at c=1 and within ±2.2% at c=32.

## Notes and observations

- **Crashes:** none. With the club-3090 patches, `vllm-mtp` ran every point without a server crash.
- **Failed requests:** one: `vllm-mtp` goodput_chat at 1 req/s, 179/180 completed. The failed request ended with a
  client-side `ServerDisconnectedError`; the server log shows no error at that time and the server kept serving.
  This row is marked `partial (1 failed)` in `goodput/summary.csv`.
- **Where vLLM is ahead:** rag16k c=8 TTFT p50 (vllm-mtp 12.3 s vs qengine-now 22.3 s). vLLM's chunked prefill
  interleaves prefill with decode, which lowers median TTFT but raises TPOT to 115–165 ms (QEngine 9.6 ms). Also
  short c=32 TPOT (vllm-nospec 26.1 ms vs qengine-now 28.5 ms), and sharegpt TTFT p50 at c=8 (146 vs 152 ms) and
  c=32 (198 vs 243 ms).
- **qengine-now vs qengine-sunday (QEngine builds):** faster at every point except short c=1 (208 vs 218 tok/s, reproduced in the
  drift run: 202 vs 220).
- **MTP in vLLM:** it helps most at c=1–8 (up to 1.9× over no-spec). At c=16–32 the gain is small or negative,
  depending on the scenario (short c=32: −10%, agent c=32: +3%). Under Poisson load it reduces goodput capacity at every
  rate (e.g. goodput_chat at 2 req/s: 0% within SLA vs 15% without MTP).
- TPOT is the vLLM definition: (E2E latency − TTFT) / (output tokens − 1), per request.

## Files

```
README.md, engines.json, commands.txt, best_vllm.csv
classic/summary.csv   classic/<scenario>/<engine>/c<C>.json + .log
goodput/summary.csv   goodput/<scenario>/<engine>/rate<R>.json + .log
prefill/summary.csv   prefill/<engine>/len<L>_rep<k>.json + .log (and _warmup)
drift/summary.csv     drift/short/<engine>/c<C>.json + .log
logs/<engine>/        full server logs
```
Summary columns: TTFT/TPOT/ITL/E2EL percentiles in ms; `goodput_frac` = share of requests within the SLA; `crash` =
the server died during that point; `status` = ok / partial / crash.
