# Benchmarks

Back to the [README](../README.md). The numbers of each model are on its page:
[Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md#benchmarks), [Qwen3.8-27B](models/qwen3.8-27b.md#benchmarks),
[DeepSeek-V4.1-Flash](models/deepseek-v4.1-flash.md#benchmarks), [GLM-5.3-Flash](models/glm-5.3-flash.md#benchmarks),
[gpt-oss-120B](models/gpt-oss-120b.md#benchmarks), [Inkling-2-Mini](models/inkling-2-mini.md#benchmarks).

All numbers were measured by me on one machine: 3× RTX 3090 (24 GB, PCIe 4.0 x16, P2P through
the patched driver unless noted), one NUMA node; "2 cards" are two of its cards. QEngine runs with its defaults. They
describe that machine; your numbers depend on the cards, links, driver and host.

Since 1.3.6 the server counts the end token a reply generates in `usage` (see
[Thinking budget and the answer tail](api.md#thinking-budget-and-the-answer-tail)), and benchmark clients take their token
counts from it, so 1.3.6 numbers include one more token per reply that ends on its own than the older numbers on the model pages.

## Headline numbers

How the numbers in the [model table](../README.md#models) of the README were measured:

Single-stream decode: club-3090 `bench.sh` with each model's own sampling; streams: its `concurrency-probe.sh`,
greedy, 256-token prompts and replies, total output of all streams. Each number is the mean of the runs of that model:
Qwen3.8-27B 6, Qwen3.8-Flash-Next 2, DeepSeek-V4.1-Flash 2, GLM-5.3-Flash 4 (single-stream decode 3); gpt-oss-120B
one run (its 10K prefill the mean of 3 in that run, its 90K prefill the mean of 5 runs); Inkling-2-Mini one run per
card count. The 1.5.0 numbers come from the 1.5.0 release-candidate builds of November 19, 2026. The two
Qwen servers ran with thinking on at effort low and vision on. For 16 streams the servers that hold 8 conversations
by default were started with 16: `GLM_CONVS=16`, DeepSeek-V4.1-Flash `--max-active 16`, `GPTOSS_SLOTS=16`.
DeepSeek-V4.1-Flash and GLM-5.3-Flash decode one conversation at a time and give the streams turns, so their total
output stays at about one stream's. gpt-oss-120B at 90K: prompts of 86,786 to 89,999 tokens. Quality: the same
builds (Inkling-2-Mini's KL: a development build, November 25, 2026); the [8-pack](#quality) results are below. Inkling-2-Mini: the 1.6.0 release build, November 26, 2026, its
server defaults (8 slots on 2 cards, GPUs 0 and 2 with P2P; 2 slots on one card, GPU 0), so the stream counts above its
slots are not measured; greedy with the mix of nine prompt types it decodes about 320 tok/s single-stream on 2
cards and 214 on one card ([Inkling-2-Mini](models/inkling-2-mini.md)).

## Quality

**Closeness to BF16.** Next-token KL divergence against the official BF16 model on a public reference set (21 cases,
32,359 positions: prose, code, Japanese, Chinese, SQL, agent turns and WikiText-2), scored with the set's own
`compare.py`; every position through QEngine's decode path (forwards of up to 16 tokens) on 3 cards (1.3.6 release
build, November 15, 2026, identical to 1.3.5 to every digit; the 1.3.4 row November 2). The last column scores the last 512 positions of the long cases after the context before them went through
the server's batched prefill (my own harness, not part of the command below). Lower is closer to BF16. The 1.5.0 build (November 19, 2026) gives the same numbers as 1.3.6 to every digit. The set: [jyohukuchan/qwen38-flash-next-bf16-logits](https://huggingface.co/datasets/jyohukuchan/qwen38-flash-next-bf16-logits).

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
2026), one full run per model on 3 cards (Inkling-2-Mini on 2 cards, below), each on a fresh server with its defaults (the two Qwen models with vision on
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

Inkling-2-Mini: a QEngine development build (November 25, 2026), 2 cards (GPUs 0 and 2), server defaults; with thinking
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
[docs/advanced.md](advanced.md#measurements).

## Measure your own setup

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
