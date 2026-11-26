# gpt-oss-120B (preview)

Back to the [README](../../README.md). Settings for every model: [common settings](../advanced.md#common-settings).

**gpt-oss-120B** (preview), run from OpenAI's official checkpoint as it is,
[openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) (MXFP4 experts): a sparse
mixture-of-experts model (128 experts per layer, 4 active) on three 24 GB cards, all in VRAM, 128K context, up to 8
conversations decoded together. Tuned and tested on RTX 3090-class cards (compute capability 8.6); RTX 4000 cards (8.9) run it untested.

**Weights**: OpenAI's checkpoint as it is: MXFP4 experts, bf16 dense weights. Headline speed and quality: the [model table](../../README.md#models).

## How it runs

OpenAI's gpt-oss-120B: 36 layers, 128 experts per layer (4 active per token), MXFP4 expert weights, grouped attention
with every second layer on a 128-token window, and the harmony chat format. QEngine runs it on three 24 GB cards, all of
it in VRAM, with its own engine (`src/gptoss`). It is a preview in 1.6: it runs in the same server and API as the other
models, with the limits listed below.

**Model files.** The official checkpoint as OpenAI publishes it, read as it is: the MXFP4 experts and the bf16 dense
weights, nothing converted, nothing of mine to download. I tested revision `b5c939de`; about 65 GB without the
`original/` and `metal/` folders:

    hf download openai/gpt-oss-120b --revision b5c939de --exclude "original/*" "metal/*" \
      --local-dir models/gpt-oss-120b
    server/run.sh 8080 --model models/gpt-oss-120b

The server recognises the model from `config.json` (`model_type` `gpt_oss`) and loads the gpt-oss engine
(`build_gptoss/libqgptoss.so`, built by `src/gptoss/build.sh`; the Docker image has it). Without that library the
start stops and says how to build it. The chat template is the model's own `chat_template.jinja`. The server is ready
about 11 s after start with the files in the page cache.

**What it needs**

- **Three NVIDIA cards with 24 GB each, compute capability 8.6** (RTX 3090 / 3090 Ti). The engine is built for sm_86
  only. RTX 4000 cards (8.9) run the same code untested, with the output head computed in full: its fast two-stage form
  has a rounding margin I measured on 3090s only, so a start-up line says the full head is used (slower decode: about 6 % for one conversation on my 3090s).
  Other cards (A100 8.0, RTX 5000 12.0, ...) are refused at start in 1.6. Two or four cards are accepted but not tested (two
  24 GB cards cannot hold the 65 GB of weights: the start stops with what each card would need); one card is refused.
- **GPU P2P between the cards is not needed.** With it the cards write into each other's memory. Without it (the
  stock GeForce driver) they exchange through pinned host memory, with the same output bit for bit: on my cards the
  host path reaches about 94 % of the P2P speed (2026-11-19, engine tool: decode 314 against 333 tok/s for one
  conversation, 951 against 1,009 tok/s for eight, a 10K-token prompt at 13.1K against 13.8K tok/s). A start-up line
  says which path runs; `GPTOSS_COMM=host` forces the host path. The host path takes the same VRAM as the P2P path and
  about 0.5 GB of pinned host RAM; `GPTOSS_PF_HOSTLANES=0` with `GPTOSS_LM=full` is the older, slower host path with
  about 0.5 GB more K/V room per card. When the host cannot pin that memory at start, the engine falls back by
  itself (the one-lane host form, a smaller prefill chunk, pageable row copies; one WARNING line each, slower prefill)
  and keeps serving.
- **Host RAM** only for the conversation archive (below); the model itself stays in VRAM.
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`.
- **Reasoning** always runs (the model has no off switch). `reasoning_effort` `low` / `medium` (default) / `high`;
  other APIs' names are mapped: `max` and `xhigh` to `high`, `minimal` and `none` to `low`, thinking off
  (`enable_thinking: false`) to `low`. The reasoning comes back in `reasoning_content` (OpenAI) or thinking blocks
  (Anthropic); the thinking budget (`--thinking-budget`, a request's budget) closes it at its count.
- **Tool calls** (the harmony `functions` namespace) in the OpenAI and Anthropic formats, `tool_choice` (`auto`,
  `none`, `required`, a named tool) and **structured output** (`response_format` `json_object` / `json_schema`). A
  history with several tool calls in one assistant turn (from another model) is rendered as one call and its result
  after the other, as harmony expects; several system / developer messages at the start are joined into one.
- **Several conversations at once**: 8 slots when the context per conversation is at least 32,768 tokens (the default
  131,072), else 4 (`GPTOSS_SLOTS`); they share one K/V pool (3x24 GB: 144,611 rows in bf16). Greedy and plain-temperature replies decode together in one step, and a reply's tokens are
  the same as when it runs alone. A reply that cannot grow because the pool is full ends with `finish_reason`
  `length`; a new prompt waits until the running replies have room to grow.
- **Prompt cache**: a follow-up turn reuses its conversation's K/V; a shared system prompt (tools included) is kept
  once for all conversations that start with it; idle conversations move to host RAM (`QE_KV_ARCHIVE_GB`, default
  128 GiB, never closer than `QE_KV_ARCHIVE_MARGIN_GB` = 8 GiB to `MemAvailable`; least recently used out first) and
  come back with a copy. `cache_prompt: false` forces a full prefill.
- **Lookup drafts** for greedy replies (`GPTOSS_LOOKUP`): tokens that repeat earlier text (edits, code) are checked
  in one multi-row step; the text is exactly the plain greedy text.
- **Sampling**: `temperature` (default 1.0, OpenAI's recommendation), `top_p` (default 1.0), `top_k`, `min_p`, `seed`.
  Plain temperature sampling decodes at greedy speed; requests with `top_p` < 1, `top_k` or `min_p` run the full
  output head one row per step (slower: about 6 % for one conversation on my cards, more with several at once).
- **Presence / frequency penalties** (`presence_penalty`, `frequency_penalty`, OpenAI's definition: a token's logit
  drops by `frequency_penalty` times its count in the reply so far, plus `presence_penalty` once it has appeared;
  before the temperature), per conversation, on the GPU and at the plain decode speed, also for several conversations
  at once. A greedy request with a penalty takes the highest penalised logit. A reply with a penalty runs without
  lookup drafts.
- **Stop strings** (`stop`), matched in the final answer only, never in the reasoning.
- **A device fault ends the process** (exit code 75) after the open requests get an error, so llama-swap or Docker
  restarts it instead of a server that answers `/health` and fails every request.

**Known limits (preview)**

- Text only: an image part is answered with `[image omitted: this server runs without vision]` in its place. No
  logprobs (HTTP 400). No activation
  steering, no disk cache (`--disk-cache` is ignored with a start-up line).
- One tool call per reply (the model's format); the built-in `browser` and `python` tools are not run by the server.
- Constrained replies (structured output, `tool_choice` `required` / named / `none`) decode one row per step.
- The DFlash drafter is not used by the server in 1.6 (`GPTOSS_DFLASH` is ignored with a start-up line).
- A greedy or seeded reply in a reused conversation can differ from the same prompt sent fresh where two tokens are
  nearly tied: rows computed by decode and by prefill differ in rounding. The same holds for a conversation that
  shares its beginning with another one the server still holds (the same system prompt, or just the harmony header):
  the shared rows are reused, so the rest of the prompt is computed in a call of another size, which rounds
  differently. The text is equally valid either way; requests that start from the same server state give the same
  reply.
- Context per conversation: 131,072 tokens (the model's own limit, `--max-ctx`); the pool is shared by all slots.

## Docker

- gpt-oss-120B (preview): mount the `openai/gpt-oss-120b` directory, e.g.
  `-v /path/to/gpt-oss-120b:/app/models/gpt-oss-120b:ro -e QE_MODEL=/app/models/gpt-oss-120b`. Engine settings go in
  with `-e` (`-e GPTOSS_SLOTS=4`). The image carries the gpt-oss engine for sm_86 only.

## Settings

The [common settings](../advanced.md#common-settings) apply (`--max-ctx`, `--reasoning-effort`, `--thinking-budget`, `--temperature`, `--top-p`,
`--served-name`, `--host`, `--port`, `--metrics-port`, `--presence-penalty`, `--frequency-penalty`); `--vision`,
`--disk-cache` and steering do not. The engine's own switches are environment variables; the full list is in
[docs/advanced.md](../advanced.md#gpt-oss-120b-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `openai/gpt-oss-120b` directory |
| `--max-ctx` / `GPTOSS_MAX_CTX` | `131072` | context per conversation (the model's limit; `GPTOSS_MAX_CTX` caps it) |
| `GPTOSS_SLOTS` | `8` (context per conversation >= 32,768), else `4` | conversations decoded at once (1-16) |
| `GPTOSS_POOL_ROWS` | every row that fits | K/V rows shared by all slots; a number pins it (never below one conversation) |
| `GPTOSS_POOL_RESERVE_MB` | `320` | VRAM per card left free beside the pool (decode graphs, other processes) |
| `GPTOSS_KV` | `bf16` | opt-in `k12` (12-bit K, 1.56x the rows) or `fp8` (8-bit K and V, 1.94x the rows): lossy, for many agents with long contexts (KL to the bf16 store at 32K / 64K context: k12 1.2e-3 / 2.2e-3, fp8 2.9e-3 / 2.3e-3) |
| `GPTOSS_ADMIT_REPLY_ROWS` | `2048` | rows a new prompt leaves free for the running replies to grow (at most 1/16 of the pool) |
| `GPTOSS_LOOKUP` | `9` | rows per step of the lookup drafts on greedy replies (a token and up to 8 drafts); `0` = off-switch |
| `GPTOSS_COMM` | auto | `host` forces the all-reduce through host memory, `p2p` requires P2P |
| `GPTOSS_CHUNK` | `10240` | prefill chunk in tokens |
| `GPTOSS_ARCHIVE_SYNC_ROWS` | all | an evicted conversation goes to host RAM whatever its size; a number bounds the copy |
| `QE_KV_ARCHIVE_GB` / `QE_KV_ARCHIVE_MARGIN_GB` | `128` / `8` | host RAM for idle conversations / what it leaves free below `MemAvailable` |
| `GPTOSS_BG_REARM` | `1` | after every finished reply an idle conversation is copied to host RAM in the background, so the next long prompt does not wait for that copy; `0` = off-switch |
| `GPTOSS_BROKEN_EXIT` | `1` | after a device fault the process exits with code 75 (for a restart); `0` = off-switch (keeps the failed process for debugging) |

## Benchmarks

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults; club-3090 tools): single-stream decode narrative / code 301 /
301 tok/s, prefill 10K / 90K 13,280 / 5,055 tok/s (first token 0.74 / 17.6 s), 306 / 594 / 731 / 835 tok/s total at 1
/ 4 / 8 / 16 streams (`GPTOSS_SLOTS=16` for 16).

More points (November 20, 2026, release build; 3 cards, defaults, greedy, engine tool, mean of two runs; the no-P2P
column with `GPTOSS_COMM=host`):

| gpt-oss-120B, 3x RTX 3090 | P2P | no P2P |
|---|---:|---:|
| single-stream decode, short context (tok/s) | 337 | 321 |
| 8 conversations at once (total output tok/s) | 1,013 | 952 |
| prefill, 10K-token prompt (tok/s) | 13,120 | 12,310 |
| ready after start, server, files in the page cache (s) | 10.6 | – |

In the release stress run a 129K-token prompt with all 8 slots busy found its needle.

Quality (1.5.0 build): every layer's output agrees with an fp32 reference of the model within the bar (worst median
row error 3.96e-4 through the decode path, 5.46e-4 through the prefill path); next-token KL against fp32 0.118 and
top-1 agreement 86.9 % on 11 items through the prefill path (0.112 through the decode path); needles at 90K: 3 of 3.
