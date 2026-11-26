# DeepSeek-V4.1-Flash (preview)

Back to the [README](../../README.md). Settings for every model: [common settings](../advanced.md#common-settings).

**DeepSeek-V4.1-Flash** (preview), run from the third-party Maya-S GGUF files as they are
([peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)): a large sparse
mixture-of-experts model (384 routed experts per layer, Engram n-gram memory, compressed sparse attention) on three
24 GB cards plus about 170 GB of host RAM, with a DSpark draft model, 262K context. RTX 3000 / 4000 series only.

**Weights**: the Maya-S GGUF files as they are: experts IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S, dense Q8_0, Engram Q6_K. Headline speed and quality: the [model table](../../README.md#models).

## How it runs

DeepSeek-V4.1-Flash-1017: 40 layers, 384 routed experts per layer plus a shared expert, Engram n-gram memory tables and
compressed sparse attention. QEngine runs it on three 24 GB cards plus host RAM, with the DSpark draft model for
speculative decoding. It is a preview in 1.6: it runs in the same server and API as the other models, with the
limits listed below.

**Model files.** I do not ship weights for this model. QEngine runs from the **Maya-S GGUF files** on Hugging Face
([peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF),
revision `c33d87b8`), read as they are: the routed experts (IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S), the dense weights
(Q8_0), the Engram tables (Q6_K), the tokenizer and the chat format from the GGUF metadata, and the DSpark drafter from
the `dspark/` folder next to `Maya-S/`. Nothing is converted on disk and the model directory is never written. About
347 GB with the drafter (`dspark.*`; the other `dspark-xs*` files there are not read):

    hf download peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF --revision c33d87b8 \
      --include "Maya-S/*" "dspark/dspark.*" --local-dir models/DeepSeek-V4.1-Flash-1017-GGUF
    server/run.sh 8080 --model models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-S

The server recognises the model from the GGUF metadata (architecture `deepseek41`) and loads the DSV4 engine
(`build_dsv4/libqdsv4.so`, built by `src/dsv4/build.sh`; the Docker image has it). Other revisions are not tested: the
start checks the model's dimensions and stops with the reason if they differ. The first start writes a few derived
files (tokenizer, the drafter's experts re-packed for the GPU, Engram hash constants) into
`~/.cache/qengine/dsv4/<model id>` (`--cache-dir` moves it; Docker: the `qengine-cache` volume), so the first start takes longer. After that the server
is ready about 41 s after start with the files in the page cache.

**What it needs**

- **Exactly three NVIDIA cards with 24 GB each**, compute capability 8.6 or 8.9 (RTX 3090 / 3090 Ti, RTX 4090; tuned
  on the 3090). The DSV4 engine is built for sm_86 only, which runs on RTX 4000 (sm_89) cards as is; RTX 5000 cards
  (sm_120) are not supported in 1.6. The engine uses the first three visible cards (`QE_DEVICES`); a fourth is
  ignored, two are not enough (`server/run.sh` and the image stop at start with a message when fewer than three are visible). The expert pool on each card is sized from its free VRAM, so a card that drives a
  desktop works, with fewer experts in VRAM.
- **GPU P2P between every pair of cards is recommended, not required.** With it, the context reservation is lazy:
  `--max-ctx` defaults to 262,144 and short chats run as fast as at 32K, because the long-context memory is taken from
  the expert pool only while a long prompt needs it and given back after. Without it (a stock driver, or
  `DSV4_NOCTXLAZY=1`) the reservation is static: `--max-ctx` still defaults to 262,144, but its memory is held for the
  whole run, so fewer experts fit in VRAM (on my cards without P2P, decode 2-3 % slower at short context than with
  `--max-ctx 32768`). When that context does not
  leave 8 GiB free on the fullest card, the start halves it (down to 32,768) with a WARNING line. Without P2P the cards
  also exchange through pinned host memory. A start-up line says which reservation runs. An explicit `--max-ctx`
  always wins.
- **Host RAM: about 170 GB free** for the full setup (on my machine `MemAvailable` drops by about 165 GB while it
  serves a 233K-token conversation). With less, the engine keeps fewer host copies of the experts and reads the rarest
  ones from the GGUF files; I tested that under a RAM hog taken while it serves, not on a machine with less RAM. It never allocates past `MemAvailable` minus
  a safety margin: it serves slower instead of running into the out-of-memory killer, and gives memory back while
  serving when the host runs short.
- **An NVMe drive for the model files.** The two Engram shards (5 and 6, about 161 GB) stay on disk and are read row by
  row while a prompt is processed. A hard disk or a network share is far too slow for this.
- **A CPU with AVX-512 is recommended; AVX2 is the minimum** (x86-64-v3: AVX2, FMA, F16C, BMI2). The experts that are
  not in VRAM run on the CPU. The engine picks its CPU kernels at start: the AVX-512 kernels need the Zen 4 set
  (AVX-512 with VNNI, VBMI, BITALG, BF16: AMD Zen 4 / Zen 5, Intel Sapphire Rapids and later); other CPUs get the AVX2
  kernels, with the same output but slower decode. The start-up line
  `CPU lane: N threads, AVX-512 kernels` shows which. The CPU part uses 24 threads by default
  (`--engine-args "--cpu-threads N"`).
- Linux and an NVIDIA driver with CUDA 13.1 support, as for the other models.

**What works**

- The same server and API as the other models: OpenAI `POST /v1/chat/completions` (streaming or not), Anthropic
  `POST /v1/messages` and `count_tokens`, `GET /v1/models`, `GET /health`. Responses carry `timings` (prompt and
  decode tokens and seconds).
- **Thinking** on by default (`--think`, a request's `enable_thinking`). `reasoning_effort`: `low`, `medium` / `high`,
  `max` / `xhigh`, or a number 1-100 (default 75); `none` turns thinking off. The reasoning comes back in
  `reasoning_content` (OpenAI) or thinking blocks (Anthropic). The thinking budget (`--thinking-budget`, a request's
  `thinking.budget_tokens` / `thinking_budget`) closes the reasoning at its count.
- **Tool calls** in the OpenAI and Anthropic formats (the model's own DSML call blocks are parsed into tool calls),
  `tool_choice` (`auto`, `none`, `required`, a named tool) and **structured output** (`response_format` `json_object`
  / `json_schema`).
- **Several requests at once** (agents): up to `--max-active 8` are in service, more wait in the queue. They take turns
  on the GPU: a long prompt in chunks of 10,240 tokens, an answer in slices of 16 decode passes, so a short request does
  not wait behind a long prompt or a long answer. The engine decodes one conversation at a time, so the total speed
  stays that of one request; a switch between conversations is exact (each one's text is the same as served alone with
  the same expert state).
- **Shared system prompts** (agent teams): when a second conversation starts with the same system message (tools
  included, at least 1,024 tokens) as an earlier one, that part is kept once, and every later conversation with it
  starts from the kept copy and computes only its own part. `--no-prefix-share` turns it off.
- **Multi-turn reuse and a host-RAM archive**: a follow-up turn reuses the cached prompt, also with thinking on;
  regenerating a reply reuses it too. Idle conversations move to host RAM (`--archive-gib`, default 32 GiB, least
  recently used out first) and come back with a copy instead of a recompute. `cache_prompt: false` in a request forces
  a full prefill. `usage.prompt_tokens_details.cached_tokens` shows what was reused.
- **Stop strings** (`stop`), matched in the answer only, never inside the reasoning.
- **Sampling**: `temperature`, `top_p`, `top_k`, `min_p`, `seed`; defaults from the GGUF metadata (temperature 1.0,
  top_p 1.0) unless set with `--temperature` / `--top-p`. Speculative decoding with DSpark keeps the sampled
  distribution exact.
- **Monitoring**: `qetop` and the dashboard on the metrics port (`/api/stats` has a DSV4 panel: held conversation,
  archive, K/V pool, prefix entries, passes). A conversation that has to be computed again because it left the archive
  shows up as a refill, counted with the prefill.

**Known limits (preview)**

- Text only: no images. No logprobs, no presence / frequency penalty (the DSV4 sampler has none), no activation
  steering (a request that asks for it gets HTTP 400: the server has no steering vector).
- **Constrained replies**: structured output and `tool_choice` `required` / a named tool decode greedy replies
  (temperature 0) on DSpark passes, each pass checked against the grammar (the same text as one masked step per
  token, about 1.6x faster); sampled constrained replies decode one token per step without drafts, so they are slower.
  `QE_DSV4_GRAM_PASS=0` turns the passes off.
- **One conversation decodes at a time.** Several requests share the GPU in turns, so the total output speed with 8
  agents is about that of one (on my cards 66 tok/s for 4 agents with unrelated prompts, 98 tok/s for 8 agents that
  share a system prompt), and each agent gets its share.
- **Greedy text depends on more than the prompt.** Which experts sit in VRAM changes as the model runs, and the GPU and
  CPU paths round differently, so the same greedy request can give a different text after unrelated requests, with
  another amount of free VRAM (a desktop on a card), with another `--cpu-threads`, with another `--max-ctx` and when a
  prompt is fed in chunks (beside other requests, or after a shared system prompt). The last three come from the
  prefill's CPU expert split: `DSV4_NOCPUPF=1` removes them at a prefill cost. `--tier-reset` makes repeats of
  a request give the same text (see Settings).
- Editing an earlier message re-processes the prompt from that point (reuse works when the new prompt extends an old
  one or a kept system prompt).
- Quality is that of the Maya-S quantization (2- to 3-bit experts, Q8_0 dense weights, Q6_K Engram), not of the
  original checkpoint.
- Decode speed at long context (256-token replies, greedy, on my cards): 103 tok/s with 117K tokens of context and 91
  tok/s with 233K, against 76-106 tok/s for three prompts at short context.

## Docker

- DeepSeek-V4.1-Flash (preview): mount the folder that holds both `Maya-S/` and `dspark/` and point `QE_MODEL` at
  `Maya-S`, e.g. `-v /path/to/DeepSeek-V4.1-Flash-1017-GGUF:/app/models/dsv4:ro -e QE_MODEL=/app/models/dsv4/Maya-S`.
  Symbolic links inside the mount must resolve inside the container (mount their targets at the same paths). The
  image carries the DSV4 engine built for sm_86 only (it runs on sm_89 cards as is). Engine settings go in with `-e`
  (`-e DSV4_EGMIRROR=...`), server flags in `QE_EXTRA_ARGS` (`--route-margin 0.05`).

## Settings

The [common settings](../advanced.md#common-settings) apply (`--max-ctx`, `--think`, `--reasoning-effort`, `--thinking-budget`, `--temperature`,
`--top-p`, `--served-name`, `--host`, `--port`, `--metrics-port`); presence / frequency penalties, `--vision` and
steering do not. The full list is in [docs/advanced.md](../advanced.md#deepseek-v41-flash-preview).

| setting | default | what it does |
|---|---|---|
| `--model <dir>` (Docker `QE_MODEL`) | – | the `Maya-S` directory of the GGUF files; the DSpark drafter is read from `<dir>/../dspark` |
| `--max-ctx` | `262144` | context per conversation; with P2P the long-context memory is taken from the expert pool only while needed (short chats as at 32K); without it, it is reserved at start (halved, down to 32,768, when it would leave less than 8 GiB free on a card; logged); a start-up line says which; an explicit value wins |
| `--archive-gib` | `32` | host RAM for idle conversations (least recently used out first; it shrinks when the host runs short); `0` = off |
| `--max-active` | `8` | requests in service at once (they take turns); more wait in the queue |
| `--route-margin` | off | opt-in, **approximates** the routing: when the router picks an expert that is not in VRAM and a VRAM-resident expert scored within this margin, the resident one runs; `0.05` is the tested value (decode 7-20 % faster on three prompts, negative log-likelihood 1.641 against 1.623 on my gate texts); logged at start |
| `--tier-reset` | off | opt-in: after each request the expert placement goes back to its state after loading, so repeating a greedy request gives the same text; serves one request at a time and turns prefix sharing off; costs decode speed |
| `--no-prefix-share` | off | off-switch: no shared copies of system prompts |
| `--no-dspark` | off | off-switch: decode without the draft model (much slower) |
| `--engine-args "<flags>"` | – | engine flags, e.g. `--cpu-threads N` (default 24), `--reserve-gb X` (VRAM left free per card, default 0.6) |
| `DSV4_EGMIRROR` | off | faster long prompts on btrfs / XFS through reflink copies of the two Engram shards ([docs](../advanced.md#deepseek-v41-flash-preview)) |

## Benchmarks

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults; club-3090 tools): single-stream decode narrative / code 88 /
120 tok/s, prefill 10K / 90K 4,217 / 4,895 tok/s (first token 2.35 / 17.8 s), 70 / 73 / 78 / 78 tok/s total at 1 / 4
/ 8 / 16 streams (requests take turns; `--max-active 16` for 16). Quality: negative log-likelihood 1.635 on my gate
texts (no full-precision reference fits on this machine); 8-pack under [Quality](../benchmarks.md#quality).

More points (November 20, 2026, release build; 3 cards, defaults unless noted, greedy, DSpark on; one run each):

| DeepSeek-V4.1-Flash, 3x RTX 3090 | P2P | no P2P |
|---|---:|---:|
| single-stream decode, engine, three prompts: git log / code / narrative (tok/s) | 104.0 / 105.5 / 76.3 | 99.8 / 97.9 / 69.7 |
| same, `--route-margin 0.05` | 124.7 / 119.3 / 81.9 | – |
| prefill, 2.5K-token prompt first / second time, 10K-token prompt (s) | 1.44 / 1.28, 2.87 | – |
| prefill, 117K / 233K-token prompt, server (s) | 27.0 / 56.2 | – |
| decode with 117K / 233K tokens of context, server (tok/s) | 103.0 / 91.2 | – |
| 4 agents at once, unrelated prompts, server (total output tok/s; slowest first token) | 65.5; 5.2 s | – |
| 4 / 8 agents at once, shared 7K system prompt, server (total output tok/s; slowest first token) | 99.6 / 97.6; 2.3 / 10.2 s | – |
| ready after start, files in the page cache (s) | 40.9 | – |

The no-P2P column ran with the default 262,144 context, so its reservation is static (`DSV4_NOP2P=1` on the same
machine).
