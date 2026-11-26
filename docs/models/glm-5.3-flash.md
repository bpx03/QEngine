# GLM-5.3-Flash (preview)

Back to the [README](../../README.md). Settings for every model: [common settings](../advanced.md#common-settings).

**GLM-5.3-Flash** (preview), run from my INT4 quant [bpx03/GLM-5.3-Flash-INT4-QEngine](https://huggingface.co/bpx03/GLM-5.3-Flash-INT4-QEngine)
(routed experts and KDA layers): a sparse mixture-of-experts
model (288 routed experts per layer, 8 active) on three 24 GB cards plus about 192 GB of host RAM, with the experts in
host RAM and a CPU expert lane next to the cards, MTP drafts, 262K context.

**Weights**: my INT4 quant: routed experts and KDA layers INT4. Headline speed and quality: the [model table](../../README.md#models).

## How it runs

Z.ai's GLM-5.3-Flash: 45 layers, 288 routed experts per layer (8 active per token) plus a shared expert, attention
layers with a sparse indexer for long prompts next to KDA linear-attention layers, and one MTP layer. QEngine runs it
on three 24 GB cards plus host RAM, with its own engine (`src/glm`): the experts used most often sit in VRAM, the full
set in pinned host RAM, and the rest of a step's experts run on the CPU next to the cards. It is a preview in 1.6: it
runs in the same server and API as the other models, with the limits listed below.

**Model files.** My INT4 quant of GLM-5.3-Flash,
[bpx03/GLM-5.3-Flash-INT4-QEngine](https://huggingface.co/bpx03/GLM-5.3-Flash-INT4-QEngine)
(routed experts and KDA layers in INT4; lossy, see the limits). **The GLM engine runs only from this checkpoint**: it
holds my GPTQ INT4 experts and KDA layers in QEngine's own layout, so other GLM-5.3-Flash quants, GGUF files and the
original weights do not load. About 186 GB:

    hf download bpx03/GLM-5.3-Flash-INT4-QEngine --local-dir models/GLM-5.3-Flash-INT4-QEngine
    server/run.sh 8080 --model models/GLM-5.3-Flash-INT4-QEngine

The server recognises the model from `config.json` (`model_type` `glm5_next`) and loads the GLM engine
(`build/libqglm.so`, built by the normal CMake build when `src/glm` is in the tree; the Docker image has it). The MTP
draft vocabulary and the expert prior (`tools/glm/draft_vocab_49k.i32`, `tools/glm/expert_prior_gen.f32`) are read
from next to the library (`<root>/build/libqglm.so` -> `<root>/tools/glm/`); without them the engine still runs, with
the full drafter head and an empty expert cache at start (a start-up line says so). The chat template is the model's
own `chat_template.jinja`. Loading takes about 60 s with the files in the page cache.

**What it needs**

- **Three NVIDIA cards with 24 GB each** (tuned and tested on RTX 3090s). Two cards also work, with fewer expert slots
  per card (slower decode). With four or more cards the engine uses the first three.
- **GPU P2P is optional.** Without it (the stock GeForce driver) the all-reduces go through host memory and the
  prefill indexer runs on every card instead of split across them (a 32K prompt prefills about 10 % slower than with
  P2P; 2K-8K prompts and decode are about level). A start-up line says which path runs; `GLM_P2P=0` forces the host
  path.
- **About 192 GB of host RAM**: the experts (about 157 GB) stay pinned in host RAM, plus the conversation archive
  (below, up to 32 GB by default). Less RAM means a smaller archive; the experts themselves have to fit.
- **An x86-64 CPU with AVX2.** The CPU expert lane is built twice and picks its kernels at start: AVX-512 when the
  CPU has the full Zen 4 set (with VNNI and BF16: Zen 4, Sapphire Rapids and later), else AVX2 (`GLM_CPU_ISA`
  forces one). The lane's thread placement is derived from the CPU topology at start (one log line).
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`.
- **Tool calls** in the GLM format (`<tool_call>` with `<arg_key>` / `<arg_value>`), returned in the OpenAI and
  Anthropic formats; `tool_choice` `auto`, `none`, `required` and a named tool; **structured output**
  (`response_format` `json_schema` / `json_object`).
- **Reasoning** on or off (`enable_thinking`, Anthropic thinking), `reasoning_effort` as the model's template takes it
  (`low`, `high`, `max`); the thinking budget (`--thinking-budget`, a request's budget) closes the reasoning at its
  count.
- **logprobs** / `top_logprobs`, **stop strings** (`stop`) with an exact rollback of the tokens past the match.
- **Prompt cache**: a follow-up turn reuses the prefix it shares with earlier turns; idle conversations go to a
  conversation archive in pinned host RAM and come back with a copy instead of a new prefill.
- **MTP speculative decoding** with the model's own MTP layer.
- **Several conversations at once** (8 by default, `GLM_CONVS`): replies take turns on the GPUs. A switch parks the
  running reply between two decode steps (its state, and its expert cache contents, saved to host RAM) and resumes the
  other one exactly where it stopped, so each reply is bit-identical to the same reply served alone. A new prompt starts
  once the running reply has had its turn (`GLM_PREEMPT_MS`, by default the slice length), so short replies end
  without a switch; long replies are sliced (`GLM_SLICE_MS` 2000, at least 8x the measured cost of a switch: a park,
  ~0.06 s, plus the later resume, ~0.3-0.6 s, so about 4-5 s per turn). With 8 agents (agent replay: 2-16K-token
  prompts, 64-token replies) the first token comes in 13.2 s p50 instead of 28.3 s one at a time, and the total output
  is 22.4 instead of 12.6 tok/s; 4 clients with 256-token replies get the same total output as one at a time (70.5 vs
  70.7 tok/s); one conversation alone runs as before.
- **Long prompts return between layers**: the prefill runs in host slices of about one second, so the server keeps
  reporting progress; a cancel (or a client that disconnects) stops the prompt at the next layer, within about half a
  second, and the cache goes back exactly to the start of the interrupted prefill window.

**Known limits (preview)**

- **One reply decodes at a time**: several conversations share the GPUs in turns (see above), not in one batch, so the
  total output speed does not grow with the number of conversations; a switch between two long replies costs ~0.3 s
  (the expert cache contents of the reply that resumes are copied back).
- Parked replies and the snapshots of up to 3 x `GLM_CONVS` conversations use pinned host RAM (about 150 MB each, plus
  the archived attention rows); under host-RAM pressure a parked reply whose rows were given back is prefilled again
  (logged; that reply is then no longer bit-identical to a run alone).
- Greedy output is deterministic for the same request on the same machine and settings, but it depends on the number
  of expert slots (free VRAM, the context setting, the number of cards) and on how the prompt was split across
  prefill windows: a prompt resumed from the cache can differ in the last bits from one prefilled at once.
- Constrained replies (structured output, forced tool calls) and logprobs are slower: each step reads the full logits
  on the host.
- The default checkpoint is a lossy INT4 quant (routed experts and KDA layers).
- Text only (no image input), no presence / frequency penalties (accepted and ignored), no activation steering, no disk
  cache, no smaller K/V formats.

## Docker

- GLM-5.3-Flash (preview): mount the `bpx03/GLM-5.3-Flash-INT4-QEngine` directory, e.g.
  `-v /path/to/GLM-5.3-Flash-INT4-QEngine:/app/models/glm:ro -e QE_MODEL=/app/models/glm`. The experts are pinned in
  host RAM (about 157 GB), so `--ipc=host --ulimit memlock=-1` matter here. Engine settings go in with `-e`
  (`-e GLM_CPU_ISA=avx2`), server flags in `QE_EXTRA_ARGS` (`--split 704,704,640`).

## Settings

The [common settings](../advanced.md#common-settings) apply (`--max-ctx`, `--devices` / `QE_DEVICES`, `--reasoning-effort`, `--thinking-budget`,
`--temperature`, `--top-p`, `--served-name`, `--host`, `--port`, `--metrics-port`); presence / frequency penalties,
`--vision`, `--disk-cache` and steering do not. The engine's own switches are environment variables; a few more are in
[docs/advanced.md](../advanced.md#glm-53-flash-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `bpx03/GLM-5.3-Flash-INT4-QEngine` directory |
| `--max-ctx` / `GLM_MAX_CTX` | `262144` | context; the engine lowers it with a warning when its attention cache does not fit (`GLM_MAX_CTX` caps it). The cache of a 262K context leaves room for 32K-row prefill windows instead of 64K: if your prompts stay under 64K, `--max-ctx 65536` prefills prompts above 32K about 6 % faster |
| `--devices` / `QE_DEVICES` | every visible card | the cards: one to three (of more, the first three; one card holds every dense weight, VRAM permitting) |
| `--split` | `704,640,704` | routed expert columns per card, multiples of 64 summing to 2048; by default the 640 goes to the card with the least free VRAM |
| `GLM_PF_SLICE_MS` | `1000` | host slice of a prefill in ms (progress and cancel between slices); `0` = the whole prompt in one call |
| `GLM_CONVS` | `8` | conversations served at once (time-sliced, exact); `1` = one at a time, the others wait in the queue |
| `GLM_SLICE_MS` / `GLM_SLICE_X` | `2000` / `8` | a reply's turn while others wait: at least this many ms and this many times the measured switch cost (park + resume) |
| `GLM_PREEMPT_MS` | the slice | how long the running reply decodes before a new prompt's prefill parks it |
| `GLM_PF_CANCEL_MID` | `1` | `0` = a cancel waits for the next prefill window start (the older rule) |
| `GLM_CPU_ISA` | `auto` | CPU expert lane kernels: `auto` (AVX-512 when present), `avx2`, `avx512` |
| `GLM_LANE_CPUS` / `GLM_CPU_THREADS` | from the CPU topology | the CPU lane's cores (a list of ranges) / its thread count, instead of the derived placement |
| `GLM_P2P` | `1` | `0` = off-switch: the all-reduces go through host memory even when P2P is there |

## Benchmarks

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults; club-3090 tools): single-stream decode narrative / code 110 /
92 tok/s, prefill 10K / 90K 2,920 / 3,572 tok/s (first token 3.42 / 25.4 s), 72 / 55 / 50 / 44 tok/s total at 1 / 4 /
8 / 16 streams (replies take turns; `GLM_CONVS=16` for 16). Server, one conversation: prefill of an 8K / 32K /
143K-token prompt 2,996 / 3,443 / 3,917 tok/s, decode at 100K context 94.9 tok/s. Quality: next-token KL against the
official FP8 model 0.448 / 0.541 / 0.772 on three prompts (a README, a git log, an agent turn), argmax agreement 79.4
/ 74.6 / 69.0 %; 8-pack under [Quality](../benchmarks.md#quality). Ready after start: about 60 s with the files in the page cache.
