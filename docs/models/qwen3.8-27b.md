# Qwen3.8-27B

Back to the [README](../../README.md). Settings for every model: [common settings](../advanced.md#common-settings).

**Qwen3.8-27B** (INT4, the RedHatAI W4A16 checkpoint [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
plus its MTP head; INT8 opt-in: [bpx03/Qwen3.8-27B-INT8-QEngine](https://huggingface.co/bpx03/Qwen3.8-27B-INT8-QEngine)): a dense model with 262K context (up to 1M
opt-in, with optional sparse attention for long prompts) and image input. Tuned for 3 cards, also runs on 2, and on 1 as an experiment.

**Weights**: INT4 W4A16, group 128 (RedHatAI) plus its MTP head. Headline speed and quality: the [model table](../../README.md#models). Docker and the source start: [Quick start](../../README.md#quick-start); 4-bit KV cache and smaller cards: [Smaller cards](../hardware.md#gpus).

## How it runs

- INT4 weights repacked at load (a few seconds, no conversion) into tensor-core layouts.
- Verify head: an INT4 copy of `lm_head` screens the vocabulary, and each row's top 16 candidates (top 64 for sampled
  rows and logprobs) are rescored exactly from the BF16 rows in pinned host RAM, so greedy output takes the BF16
  argmax. It needs 2.5 GB of pinned host RAM on one card (0.85 GB per card on 3) and 0.64 GB less VRAM per card than
  an INT8 head. Without that host RAM, or with `QE_HEAD4R=0`, every card uses the INT8 head.
- Tensor parallel over PCIe; the four full-attention KV heads are split by position (3 cards with P2P), so no KV head
  is duplicated. All-reduces run inside the kernels and inside CUDA graphs.
- Speculative decoding, verified over the full vocabulary, so outputs are unchanged: the [DFlash2 block
  drafter](#dflash2-block-drafter-27b-on-when-it-fits) (7 drafts per step) for a conversation that decodes alone,
  when its memory fits; MTP with 1 to 7 drafts per step, chosen from the measured acceptance, for everything else.
  Lookup drafts on the GPU.
- Up to 32 conversations (`QE_MAX_SEQS`, default 16; more open requests are parked, see
  [Serving behaviour](../api.md#serving-behaviour)); KV cache in FP8 in one pool per card (1,232,896 tokens on 3 cards with P2P
  and 16 slots, 1,075,200 with 32; `QE_KV_BITS=4` stores it in 4 bits for small cards, see [GPUs](../hardware.md#gpus)); 262,144
  tokens per conversation.
- Prompt-cache states (snapshots of the recurrent state, about 150 MB of pinned host RAM each, split over the cards):
  up to 8 per sequence slot while they take at most 8 % of the host's RAM (or of a container's memory limit), at least
  4 per slot and at least 32 (`QE_SNAPSHOTS`); with 16 slots on a 256 GB host that is 128 states, about 19 GB. The
  states within the 8 % are pinned at start while the host has room, before the first request (`QE_ARCHIVE_WAIT_S`);
  the rest only as the store fills. With 32 for every slot count, 8 agent
  conversations on 32 slots dropped each other's states and re-prefilled from the system prompt; with 4 per slot, 16
  conversations on 16 slots still did. The state at the end of each
  prompt is copied to the host beside the first decode step instead of before it (about 6 ms less to the first token
  and per reply on one card, 2 ms on three; 16 MB of VRAM per card for the staged convolution rings;
  `QE_SNAP_ASYNC=0` copies it before the first step, as before).

**Release files.** Two files are attached to every [release](https://github.com/bpx03/QEngine/releases/latest):
`model_mtp_ft.safetensors` (849 MB, an MTP head fine-tuned on the engine's own answers) and `token_rank.bin` (the
draft-vocabulary ranking). The server downloads them on the first start into `~/.cache/qengine/assets`, checks them
against the release's `SHA256SUMS` and reuses them offline (the model directory is never written). Without them the
engine runs, just slower. Offline machines: put them into the model directory by hand:

```bash
cd models/Qwen3.8-27B-INT4
for f in model_mtp_ft.safetensors token_rank.bin SHA256SUMS; do
  curl -LO https://github.com/bpx03/QEngine/releases/latest/download/$f
done
sha256sum -c SHA256SUMS
```

`QE_FETCH_ASSETS=0` turns the download off; the start waits for it at most `QE_FETCH_WAIT_S` (120) seconds.

### DFlash2 block drafter (27B, on when it fits)

Since 1.3.5.8 the 27B can draft with **DFlash2**, a small block drafter (the W4A16 build
[syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), Apache-2.0). It reads the
hidden states of five layers of the model and proposes 7 tokens per step; the model verifies them like MTP drafts, so
greedy output is identical to MTP, and sampled output follows the same distribution (only the random draws differ).

**Where it helps.** It is used for a conversation that decodes alone: one chat, one agent, one long answer. Measured
on this hardware (1.3.5.8; server, one conversation, against the MTP default of the same build):

| setup | short code | short prose | short math | 131K context |
|---|---:|---:|---:|---:|
| 1 card (`QE_DRAFTER=dflash2`) | +38 % | +17 % | +22 % | +10.5 % |
| 3 cards, P2P | +34 % (321 → 432 tok/s) | +6 % (230 → 244) | +15 % (297 → 341) | – |

The gain depends on the text: code and other predictable output gain most; long reasoning that MTP and lookup
drafts already predict well gains little (a 32K prompt with a long thinking part: +2 %).

Second choices on one card: when the drafter is unsure about its first three drafts (greedy replies), the step
verifies four drafts plus the drafter's second choice for each of the first three positions in the same 8 rows; a
right second choice keeps one more token. Output is identical. Measured on one card (engine, same build on / off):
+3.5 % on a 30-prompt mix, +4.6..4.9 % on the three short benchmark prompts, +2.7 % / +1.6 % at 32K / 96K, about level
on file edits. `QE_DF_SIB=0` turns it off.

Two conversations at once on one card: when exactly two conversations decode in the same step, each drafts with its
own DFlash2 context (8 rows each); output stays identical to each conversation decoding alone. Measured on one card
(server, two 512-token replies at once): +2.8 % for the pair as a whole; +8 % per step while both run (engine benchmark). With three or more
conversations in a step, and on 2 or 3 cards, batched steps draft with MTP as before (three at once on one card was
level to slightly slower, four 5 % slower). `QE_DF_BATCH=0` turns it off; `QE_DF_BATCH=n` allows it up to n
conversations per step. Grammar-constrained replies (`response_format`, a forced tool call until its header, `logprobs`) always keep MTP; a `tool_choice: none` reply drafts as usual. Agent turns that edit a file already in the context come out about 3 % slower,
because the model copies text the drafter no longer sees (it keeps the last ~2K positions); turns that write new text
are faster.

**What it costs.** VRAM on every card, plus 0.25 GB of pinned host RAM:

| cards | VRAM per card | KV pool (all conversations together) | context per conversation |
|---|---:|---|---|
| 1 | 1.20 GB (three context rings for the two-conversation steps; `QE_DF_BATCH=0`: 1.16 GB) | about −38K tokens | `QE_MAX_SEQS=1`: **206,848 → about 165,000** (default `QE_RESERVE_GB=1.5`) |
| 2 | 0.82 GB | 675,840 → 623,616 tokens | 262,144, unchanged |
| 3 | 0.70 GB | 1,298,432 → 1,232,896 tokens | 262,144, unchanged |

On 2 and 3 cards the default context still fits. **On one card it does not**, which is why a single card stays on MTP
by default.

**When it is on (`QE_DRAFTER`):**

- `auto` (default): at start the server loads DFlash2 only when the requested context (`--max-ctx`) and slots still
  fit with it, and prints one line saying which drafter it loaded and why (memory needed against memory free). The
  context is never lowered to make room for it. On 2 and 3 cards it is on with the defaults. **On one card the
  default `--max-ctx 262144` does not fit, so a single card stays on MTP**, and the start-up line tells you what this
  machine needs, e.g. `For DFlash2: --max-ctx <= 165K or QE_DRAFTER=dflash2` (the number is computed from your free
  VRAM). Either ask for that context or force it:
  ```bash
  # one card: room for DFlash2 by asking for less context (use the number from your start-up line)
  QE_DEVICES=0 QE_MAX_SEQS=1 server/run.sh 8080 --model <dir> --max-ctx 150K
  # or force it: the context becomes whatever still fits (logged)
  QE_DEVICES=0 QE_MAX_SEQS=1 QE_DRAFTER=dflash2 server/run.sh 8080 --model <dir>
  ```
  With the default 16 slots one card has about 43K tokens per conversation and stays on MTP; DFlash2 on one card is
  meant for `QE_MAX_SEQS` 1 to 4.
- `dflash2`: always load it; the served context shrinks to what fits, with a start-up line.
- `mtp`: never load it (and nothing is downloaded); the previous behaviour, full context.

**Files.** On the first start the server downloads the drafter (1.28 GB, a pinned revision checked against its
sha256) into `~/.cache/qengine/dflash2` (Docker: the `qengine-cache` volume) and reuses it offline. Without network
access and without the file the server starts on MTP and says so in one line; it never stops for it.
`QE_FETCH_ASSETS=0` turns the download off; `QE_DFLASH_URL` points it at a mirror. A local copy: `QE_DFLASH=<dir or
file>`, or a `dflash2/` directory inside the model directory.

### INT8 weights (27B, opt-in, experimental)

INT4 stays the default. Pointing the server at an 8-bit checkpoint runs the INT8 layer weights: closer to the BF16 model
and slower, because a decode step reads twice the weight bytes. Three cards are recommended; one card can't hold them.
The weights are at [bpx03/Qwen3.8-27B-INT8-QEngine](https://huggingface.co/bpx03/Qwen3.8-27B-INT8-QEngine).

```bash
hf download bpx03/Qwen3.8-27B-INT8-QEngine --local-dir models/Qwen3.8-27B-INT8
server/run.sh 8080 --model models/Qwen3.8-27B-INT8     # Docker: -e QE_MODEL=/app/models/Qwen3.8-27B-INT8
```

To build one from the official BF16 weights (no GPU, no calibration data; Python packages in `requirements-ref.txt`):

```bash
hf download Qwen/Qwen3.8-27B --local-dir models/Qwen3.8-27B-BF16
python3 tools/quant_int8.py --src models/Qwen3.8-27B-BF16 --out models/Qwen3.8-27B-INT8 \
    --kv-from models/Qwen3.8-27B-INT4     # copies the FP8 KV scales, which keep the full KV pool
```

INT8 KL and top-1 are from 1.3.5.2. Everything else, the INT4 KL and top-1 included, is from November 1, on older code, and not
rerun since; 3 cards:

| metric | INT8 | INT4 | INT8 / INT4 |
|---|---:|---:|---:|
| KL vs BF16, mean (320 positions) | 6.66e-4 | 1.47e-2 | 22× closer |
| top-1 match with BF16 | 99.7 % | 95.3 % | |
| decode narrative / code, tok/s | 184.6 / 276.7 | 266.6 / 387.7 | 0.69 / 0.71 |
| prefill 10K / 90K, tok/s | 4,816 / 4,098 | 6,077 / 4,866 | 0.79 / 0.84 |
| agent c=8, output tok/s | 810 | 1,215 | 0.67 |
| KV pool, 32 slots, 262K context, tokens | 566,272 | 932,864 | 0.61 |

The 8-pack has not run on INT8, so whether the lower KL gives better answers in real tasks is not known yet.

**Your own checkpoint** goes in through `--model`. The engine reads one `model.safetensors` (not shards),
compressed-tensors pack-quantized, symmetric, group 128: `weight_packed` int32 `[N][K/4]` with bytes LSB-first and the
stored value `w + 128`, `weight_scale` bf16 `[N][K/128]`, tensor names as in the 27B INT4 checkpoint. A format it would
misread stops the start with the reason. Without FP8 KV scales it runs a bf16 KV cache with half the pool and no position
split (`QE_KV_SCALES=<a 27B INT4 model.safetensors>` borrows the scales; this fallback has not run on cards). INT8
checkpoints use their own `model_mtp.safetensors`, not the fine-tuned head (`QE_INT8_MTP_FT=1` uses it). Not tried:
`lued/Qwen3.8-27B-INT8-W8A16-MTP`; it is sharded, so the start refuses it as it is.

### Long documents (27B, opt-in)

Above 262K tokens the 27B needs YaRN (`--max-ctx 512K` or `1M`), and for prompts that long most of the time goes to
attention; `--lsa` makes that attention sparse (the index map is downloaded once and cached, see the docs). Both are off by default. YaRN is static: it applies to every
request of that server, short ones too, so run a separate server for long documents. What `--lsa` changes
(3× RTX 3090, measured November 3, 2026; details in [docs/advanced.md](../advanced.md#sparse-attention-for-long-prompts-opt-in)):

| you gain | you give up |
|---|---|
| 512K prompt about 2× faster (225 → 103–114 s), 1M about 3× (736 → 239 s) | exactness: each token reads 2,048–4,096 of its keys, which carry about 60–80 % of the full attention weight in my measurement |
| decode at 1M about 2× (106 → 227 tok/s), at 512K 1.5× | tested only on 2 retrieval tasks at 128K and 512K plus one 1M needle (same answers as full attention): not RULER, not reasoning quality over long documents |
| nothing changes below a prompt position of ~64K or a context of ~192K (full attention runs there) | only one conversation at a time benefits (the last long prompt's); others, and requests on a pinned shared system prompt, use full attention |
| | about 0.5 GB per card less KV pool; needs P2P and the FP8 KV cache; at most 1.87M tokens |

## Settings

| setting | default | what it does |
|---|---|---|
| `QE_MAX_SEQS` | `16` | conversations kept in the KV pool, 1 to 32 on any card count; a start whose cards cannot hold that many lowers it (logged). For long conversations on one card use 1 to 4. More open requests than slots are parked beyond the seats (`--park`). `32` seats 32 at once, but leaves a smaller KV pool (2 cards: 417,792 against 623,616 tokens; one 262,144-token conversation still fits, two very long ones at once do not) |
| `QE_ADMIT_BY_LIMIT` | `1` | a running conversation is counted at admission with at most its reply's `max_tokens` + 1,024 tokens of growth (else 8,192); `0` = always 8,192 |
| `QE_ARCH_REPLY` | `1` | a conversation's next turn that leaves its cached tokens only inside its last reply (a client that sends another assistant text or drops the reasoning) no longer copies the whole conversation to the host archive first; `0` = copy it, as before |
| `QE_ARCH_SUBS` | `2` | 27B: recurrent-state checkpoints an archived conversation keeps besides its end state (1 to 4; each is about 151 MB of host RAM). With many agents at once, `1` lets the host archive hold about a third more conversations; the second one is what an edited or retried last turn resumes from |
| `QE_ARCH_DEMOTE` | `1` | 27B: an archived conversation whose next turn replaced its last reply (a client that sends another assistant text) goes first when the archive needs room, instead of being kept as recently used; `0` = keep it as before |
| `QE_SNAP_DEMOTE` | `1` | 27B: when a prompt-end state is saved, the same slot's older prompt-end states of that conversation (its earlier turns) go first when the state store needs room, instead of aging like the live ones; with many conversations per slot count a waiting turn no longer loses its own state. The resume point of a repeated prompt and of a fork is kept. `0` = plain least-recently-used order, as before |
| `QE_DF_BATCH` | `2` on one card, `0` on more | batched steps of up to this many conversations draft with [DFlash2](#dflash2-block-drafter-27b-on-when-it-fits) (one context ring each, 43 MB per card); `0`: batched steps draft with MTP |
| `QE_DRAFTER` | `auto` | drafter of a conversation that decodes alone: `auto` loads [DFlash2](#dflash2-block-drafter-27b-on-when-it-fits) when the requested context and slots fit with it (logged, with the `--max-ctx` that would fit; one card with the default `--max-ctx` stays on MTP), `dflash2` always (context shrinks to fit), `mtp` never (no download) |
| `QE_DFLASH` | – | path to a local DFlash2 checkpoint (directory or file) instead of the download |
| `QE_KV_BITS` | `8` | `4`: 4-bit KV cache (opt-in, not lossless against FP8): about 1.5x the context on small cards; `--lsa` is then off (logged); see [Smaller cards](../hardware.md#gpus) |
| `QE_HEAD4R` | `16` | 27B verify head: INT4 screen + exact BF16 rescore of the top N per row (pinned host RAM); `0` = INT8 head |
| `QE_RESERVE_GB` | `1.5` | VRAM per card left outside the KV pool; e.g. `1.5,3,1.5` for a desktop on GPU1 |
| `QE_KV_ARCHIVE_GB` | `32` (less below 64 GB of RAM) | host-RAM archive of conversations; `0` = off; a set value wins |
| `QE_PREFIX_GB` | `8` (less below 64 GB of RAM) | host RAM for pinned system prompts |
| `QE_COMM` | `auto` | prefill transfer path: `auto`, `p2p`, `ce`, `host` |
| `QE_FETCH_ASSETS` | `1` | `0`: no downloads at start: not the tuned MTP head and draft ranking, not the DFlash2 drafter (MTP drafts only), not the `--lsa` index map (full attention) |
| `QE_PAUSE_TIMEOUT_S` | `900` | seconds a reply may wait for GPU memory another process holds before it ends with an error, counted from when that wait began: one that cannot continue, or a paused one that cannot resume because the KV cache gave memory back; a reply paused only to make room for another start has no deadline |
| `QE_DECODE_COMM` | `auto` | decode collectives: `auto` (timed on real decode steps at start-up, cached; keeps P2P or the host path per mode), `remeasure`, `p2p`, `host`; as on [Flash-Next](qwen3.8-flash-next.md#settings) |
| `QE_HOST_RESERVE_GB` | max(8, 8 % of RAM) | at start only: RAM the host-RAM plan leaves to the system and the desktop when it sizes the archive and the pinned prompts |
| `--disk-cache DIR` | off | persistent prompt cache that survives restarts (Flash-Next: experimental, since 1.3.5.3) |
| `--disk-eager` | off | with `--disk-cache`: also write each conversation after every reply (NVMe wear: the whole file per turn) |

## Benchmarks

**1.5.0** (November 19, 2026; 3 cards, P2P, defaults, DFlash2 loaded, the club-3090 `bench.sh` settings: thinking on,
effort low, vision on): single-stream decode narrative / code 275 / 524 tok/s, prefill 10K / 90K 6,188 / 4,994 tok/s
(first token 1.60 / 18.0 s); concurrency probe (greedy, 256-token prompts and replies): 312 / 575 / 848 / 999 tok/s
total at 1 / 4 / 8 / 16 streams.

**1.3.6** (November 15, 2026; 3 cards, P2P, defaults, DFlash2 loaded): single-stream decode narrative / code 278.7 /
515.5 tok/s, prefill 10K / 90K 6,309 / 5,113 tok/s (5,769 / 4,784 right after other runs), 8 agents at
once 1,129 tok/s (1,210 with `QE_PRESENCE_PENALTY=0`). With the club-3090 `bench.sh` settings: 281.2 / 507.0 tok/s,
prefill 6,133 / 4,997. Short single replies (385 tokens, greedy: a Python function, a short story, a proof): 474 / 264
/ 401 tok/s on 3 cards; 212 / 150 / 196 tok/s on one card (`QE_MAX_SEQS=1 --max-ctx 150K`, DFlash2).

**Many agents in flight** (1.5.0 code, November 18, 2026; closed loop, no think time: each agent sends its next
turn when the reply ends; a shared 2K-token system prompt, a first message of 1-3K tokens, then tool results of
30-2,000 tokens (about 650 new tokens per turn) up to 16K tokens of context; 128-token replies, greedy, thinking off;
replayed by my own harness (not in this repository), one run per point). Median time to first token / total output:

| agents in flight | 1 | 4 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| 2 cards, 1.3.6 (16 slots, fixed half split) | 0.18 s / 140 tok/s | 0.64 s / 157 | 4.5 s / 231 | 25.0 s / 159 | 73.0 s / 111 |
| 2 cards, 1.5.0 defaults (16 seats, parking beyond them; release build, November 20) | – | – | 1.04 s / 243 tok/s | 0.48 s / 359 | 1.0 s / 285 (after a 90 s warm-up) |
| 2 cards, 1.5.0, `QE_MAX_SEQS=32` | 0.19 s / 142 tok/s | 0.33 s / 173 | 0.86 s / 248 | 0.32 s / 361 | 5.0 s / 155 |
| 3 cards, 1.5.0, `QE_MAX_SEQS=32` | 0.13 s / 193 tok/s | – | 1.31 s / 258 | 0.66 s / 403 | 3.3 s / 242 |

At 64 agents the measured window (after 15 s) still holds the 64 agents' first turns (about 140K new tokens, the
queue is prefill-bound). After a 90 s warm-up, at 64 agents: 2 cards 30.3 s / 147 tok/s without parking and with four checkpoints per archive,
0.27 s / 318 tok/s with 1.5.0 (`QE_MAX_SEQS=32`); 3 cards 1.45 s / 305 tok/s, where past ~10K tokens per agent the default 32 GB host
archive no longer holds all 64 conversations (`QE_ARCH_SUBS=1`: 1.06 s / 317; a larger `QE_KV_ARCHIVE_GB` removes
the limit).

1.3.5 (November 6, 2026; 3 cards, P2P): single-stream decode narrative / code 274.1 / 422.5 tok/s, prefill 10K / 90K 6,437 / 5,159 tok/s, 8 agents at once 1,205 tok/s. With the club-3090 `bench.sh` settings: 276.1 / 418.9 tok/s. Prefill measured on its own.

Serving benchmark (October 2026, QEngine 1.0), 3 cards, `vllm bench serve` as the client; output tokens/s:

| scenario (input / output tokens) | 1 at once | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| short (1024 / 512) | 208 | 794 | 949 | 1,001 |
| agent (500 / 2000) | 287 | 1,080 | 1,302 | 1,423 |
| ShareGPT | 243 | 819 | 935 | 1,110 |

Later changes (W4A8 GEMM for batched passes of 33 rows or more, less admission headroom) raised agent at 32 to 1,703
and short at 32 to 1,039 tok/s in a same-day A/B. Prefill of one new prompt (median time to first token): 32K tokens
5.35 s, 200K tokens 50.7 s.
