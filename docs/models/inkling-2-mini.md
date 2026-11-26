# Inkling-2-Mini (preview)

Back to the [README](../../README.md). Settings for every model: [common settings](../advanced.md#common-settings).

**Inkling-2-Mini** (preview), run from Thinking Machines' official checkpoint
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (NVFP4 / MXFP8 experts as
shipped) plus two small files of mine ([bpx03/Inkling-2-Mini-QEngine](https://huggingface.co/bpx03/Inkling-2-Mini-QEngine):
INT8 dense attention weights and a trained DSpark drafter): a looped sparse mixture-of-experts model (26.4B
parameters, 3.4B active) on one 24 GB card (about 126K tokens of context) or two with GPU P2P (about 565K), with
exact DSpark drafts and up to 8 conversations decoded together. RTX 3090-class cards (compute capability 8.6) only.

**Weights**: the official checkpoint's NVFP4 / MXFP8 experts as they are; dense attention weights GPTQ INT8, one scale per 128 inputs (`--dense-weights g15`: the original bf16 weights, lossless). Headline speed and quality: the [model table](../../README.md#models).

## How it runs

Thinking Machines' Inkling 2 Mini: 24 layers of which layers 6-17 run three times per token (48 layer applications), a
sparse mixture of experts (26.4B parameters, 3.4B active per token) with NVFP4 experts in layers 0-19 and MXFP8 in
20-23, sliding-window attention (1,024 tokens) on 40 of the 48 applications and full attention on 8, and its own chat
format. QEngine runs the text part with its own engine (`src/inkling`): on one 24 GB card, or split over two cards
with tensor parallelism. It is a preview in 1.6: same server and API as the other models, with the limits below.

**Model files.** The official checkpoint, read as it is (I tested revision `b85c3cfa`, about 19 GB), plus two files of
mine next to it in the same directory (about 1.3 GB):

    hf download thinkingmachines/Inkling-2-Mini --revision b85c3cfa --local-dir models/Inkling-2-Mini
    hf download bpx03/Inkling-2-Mini-QEngine dense8.safetensors mtp_qe.safetensors --local-dir models/Inkling-2-Mini
    server/run.sh 8080 --model models/Inkling-2-Mini                    # one card (GPU 0)
    QE_DEVICES=0,1 server/run.sh 8080 --model models/Inkling-2-Mini     # two cards, needs GPU P2P

- `dense8.safetensors` (0.65 GB): the dense attention weights in INT8 with one scale per 128 inputs, rounded with GPTQ
  (error feedback against activations of public text), so decode reads fewer bytes. Without it the engine uses the
  original bf16 weights (`g15`, see the table below) and says so at start.
- `mtp_qe.safetensors` (0.64 GB): the model's DSpark drafter, fine-tuned by me on the model's own outputs (same shapes,
  less over-confident, with its own stop policy). Without it the engine uses the official `mtp.safetensors` of the
  checkpoint (`INK_DRAFTER_FILE=mtp.safetensors` picks the official one when both are there). Drafts never change the
  text: every drafted token is checked against the model's own choice.

The server recognises the model from `config.json` (`model_type` `inkling2_loop`) and loads the Inkling engine
(`build_srv/libqinkling.so`, built by `OUT=build_srv src/inkling/build.sh` from the repository root; a relative `OUT`
is taken from the directory you run it in; the Docker image has it). Without that
library the start stops and says how to build it. The chat template is the model's own `chat_template.jinja`. The
server is ready about 9 s (one card) or 15 s (2 cards) after start with the files in the page cache.

**What it needs**

- **One or two NVIDIA cards with 24 GB, compute capability 8.6** (RTX 3090 / 3090 Ti; the engine is built for sm_86
  only). One card by default; `QE_DEVICES=a,b` (or `--devices`) splits it over two cards, which needs GPU P2P between
  them: without peer access the start stops with that reason. Three cards work (an uneven head split) but are not part
  of the 1.6 checks.
- **Host RAM** only for the conversation archive (below); the model itself stays in VRAM.
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`.
- **Thinking**: the template's `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or a
  number from 0 to 0.99); thinking off (`enable_thinking: false`) is effort 0; without one the model's default 0.9.
  The reasoning comes back in `reasoning_content` (OpenAI) or thinking blocks (Anthropic).
- **Tool calls** in the OpenAI and Anthropic formats (the model's `content_invoke_tool_json` messages).
- **DSpark drafts** for greedy and plain-temperature replies: the drafter proposes up to 7 tokens, one multi-row step
  checks them, and the text is exactly the text without drafts.
- **Several conversations at once**: 8 slots on 2 cards, 2 on one card (`INK_SLOTS`, 1-16; when they don't fit the
  server serves fewer and says so). They share one K/V pool, and one conversation may use all of it: per conversation
  about 565K tokens on 2 cards and 126K on one card with the default weights. Replies decode together in one step,
  and a reply's tokens are the same as when it runs alone.
- **Prompt cache**: a follow-up turn reuses its conversation's K/V; idle conversations move to host RAM
  (`QE_KV_ARCHIVE_GB`, default 128 GiB, never closer than `QE_KV_ARCHIVE_MARGIN_GB` = 8 GiB to `MemAvailable`) and
  come back with a copy, also in the background after every finished reply (`INK_BG_REARM`); with `--park 1`
  (off by default) replies beyond the seats are parked in host memory.
- **Sampling**: `temperature` (default 1.0, from the model's `generation_config.json`), `top_p`, `top_k`, `min_p`,
  `seed`, presence / frequency penalties. Temperature with `top_k` up to 64, `top_p` and `min_p` is sampled on the
  GPU with drafts on, at about the greedy speed: the engine finds the exact top logits, keeps the same set as the
  usual rule (top_k, then min_p, then the smallest top_p prefix) and draws from it with seeded noise, so drafts never
  change the text and a seed gives the same reply alone or in a batch. Without `top_k` the kept set has to end
  inside the 64 most likely tokens; a token where it doesn't (`top_p` near 1 on very flat text) is picked on the CPU
  with the same rule and seed, and a conversation where that happens often runs without drafts. `top_k` above 64
  and presence / frequency penalties are picked on the CPU without drafts. `INK_SAMP_DEV=0` puts all filters on the
  CPU.

**Known limits (preview)**

- Text only: an image part is answered with `[image omitted: this server runs without vision]` in its place.
- No structured output and no logprobs (HTTP 400); `tool_choice` `required` / named are not supported (`none` works:
  the reply ends at its first call); the thinking budget is ignored.
- Without GPU P2P only one card.

## Thinking off, dense weights and slots

Inkling-2-Mini with thinking off runs at reasoning effort `none`, and the model still writes a short thought (one or two
sentences) before most replies; the server returns it as reasoning (`reasoning_content` / a thinking block), never
mixed into the answer.

Inkling-2-Mini lets you pick the form of its dense attention weights (`--dense-weights`, or `INK_DENSE8=1` / `0`;
the flag wins) on any card count:

| form | what it is | decode step, 2 cards / one card | quality (set-2 KL vs the FP32 reference, decode form; BF16 bar 0.00592) | context per conversation, 2 cards / one card |
|---|---|---|---|---|
| `int8` (default) | GPTQ INT8, one scale per 128 inputs, from `dense8.safetensors` next to the checkpoint: faster, small quality loss | ~4.9 / ~7.1 ms | 0.00539 (one card 0.00538) | ~565K / ~126K |
| `g15` | original weights (lossless, bit-exact to the checkpoint, stored in a 12-bit form) | ~5.5 / ~8.0 ms | 0.00495 | ~544K / ~117K |

`int8` is the default on any card count. Prefill runs at about the same speed in both forms (`int8` within 1 %).
Without `dense8.safetensors` the default is `g15`, and the server logs one line about it. If you ask for `int8` and `dense8.safetensors` is missing, it falls back to RTN INT8 at load (lower quality,
KL 0.00589), and the server says so at start. A start-up line names the active form, the served context and the
switch, e.g. `inkling: dense attention weights int8 (...) on 2 cards, context per conversation 565216 tokens; switch:
--dense-weights g15 (INK_DENSE8=0) = original weights, lossless`.

Inkling-2-Mini serves 8 conversation slots on 2+ cards and 2 on one card. On one card each slot's fixed state
(~0.36 GB) comes out of the K/V pool, so 2 slots leave ~126K tokens per conversation (8 slots: ~62K; with the
lossless `g15` weights ~117K, 8 slots ~53K, 1 slot ~137K). `INK_SLOTS=N` (1-16) sets the count. When the requested slots don't fit, the server serves fewer
and says so in a start-up line (`requested 16 slots, serving 8`).

After every finished reply Inkling-2-Mini copies the idle conversation to host RAM in the background (`INK_BG_REARM=0`
= off-switch). A 109K-token conversation then comes back in about 0.3 s instead of a 12.7 s prefill, and a new long
prompt that needs its room does not wait for the copy.

## Docker

- Inkling-2-Mini (preview): mount the model directory with `dense8.safetensors` and `mtp_qe.safetensors` in it, e.g.
  `-v /path/to/Inkling-2-Mini:/app/models/inkling:ro -e QE_MODEL=/app/models/inkling`. It runs on one card (the first
  of `--gpus`) unless `QE_GPUS` / `QE_DEVICES` name the cards (`-e QE_GPUS=0,1`: two cards, needs GPU P2P; compose
  passes both from your shell and leaves them empty otherwise, so it also runs on one card). The image
  carries the Inkling engine for sm_86 only.

## Settings

The [common settings](../advanced.md#common-settings) apply (`--max-ctx`, `--reasoning-effort`, `--temperature`, `--top-p`, `--served-name`,
`--host`, `--port`, `--metrics-port`, `--presence-penalty`, `--frequency-penalty`, `--park`); `--vision`,
`--disk-cache`, `--thinking-budget` and steering do not. The engine's other switches are in
[docs/advanced.md](../advanced.md#inkling-2-mini-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `thinkingmachines/Inkling-2-Mini` directory (with `dense8.safetensors` and `mtp_qe.safetensors` next to it) |
| `QE_DEVICES` / `--devices` | `0` (one card) | the cards; two (`0,1`) split the model with tensor parallelism (needs GPU P2P). Docker: `QE_GPUS` / `QE_DEVICES`, else one card |
| `--max-ctx` | the model's 1,048,576 | context per conversation, served up to what the K/V pool holds (one start-up `context` line) |
| `--dense-weights` / `INK_DENSE8` | `int8` when `dense8.safetensors` is there, else `g15` | `g15` (`INK_DENSE8=0`) = the original weights, lossless; `int8` (`INK_DENSE8=1`) without the file = RTN INT8 at load (lower quality, logged); the flag wins |
| `INK_SLOTS` | `8` on 2+ cards, `2` on one card | conversations decoded at once (1-16) |
| `INK_DRAFTER` | `1` | `0`: no drafter (plain decoding, about 1 GB more K/V room) |
| `INK_DRAFTER_FILE` | `mtp_qe.safetensors` when present, else `mtp.safetensors` | drafter weights: a file name in the model directory or a path |
| `QE_KV_ARCHIVE_GB` / `QE_KV_ARCHIVE_MARGIN_GB` | `128` / `8` | host RAM for idle conversations / what it leaves free below `MemAvailable` |
| `INK_BG_REARM` | `1` | after every finished reply the idle conversation is copied to host RAM in the background; `0` = off-switch |
| `INK_ADMIT_REPLY_ROWS` | `2048` | rows a new prompt leaves free for the running replies to grow |
| `INK_ARCHIVE_SYNC_ROWS` | all | an evicted conversation goes to host RAM whatever its size; a number bounds the copy |

## Benchmarks

**Development build** (November 25, 2026; GPU 0, and GPUs 0 + 2 with P2P; defaults: INT8 dense weights, trained
drafter; engine tool, greedy):

| Inkling-2-Mini, RTX 3090 | 2 cards | 1 card |
|---|---:|---:|
| plain decode step at 512 tokens of context, no drafts (ms; tok/s) | 4.99; 200 | 7.10; 141 |
| single-stream decode with drafts, mix of 9 prompt types (prose, code, JSON, tables, edits, math; 128 tokens each, harmonic mean, tok/s) | 320 | 214 |
| prefill, 2K / 8K / 120K-token prompt, engine tool (tok/s) | 10,804 / 12,369 / 9,202 | – |
| context per conversation, default slots (tokens) | 565,216 (8 slots) | 125,920 (2 slots) |
| ready after start, server, files in the page cache (s; release candidate, November 26) | 15 | 9 |

Club-3090 tools (1.6.0 release build, November 26, 2026, server defaults; the club sampler `top_p` 0.95 / `top_k`
20 runs on the device with drafts): single-stream decode narrative / code 230 / 330 tok/s on 2 cards, 162 / 223 on one
card; prefill 10K / 90K 9,430 / 8,860 tok/s (first token 1.07 / 10.2 s) on 2 cards, 6,186 / 4,945 (1.57 / 17.6 s) on
one card; 213 / 362 / 431 tok/s total at 1 / 4 / 8 streams on 2 cards (8 slots), 147 at 1 stream on one card (2 slots).

Quality: next-token KL against an fp32 reference of the model on a 25K-token text set: 0.00539 with the
INT8 dense weights through the decode path (0.00544 through the prefill path; one card 0.00538 / 0.00558), 0.00495
with `--dense-weights g15`; the official bf16 forward measures 0.00592 on the same set. Top-1 agreement 97.6 %.
