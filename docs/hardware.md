# Hardware and interconnect

Back to the [README](../README.md#hardware-and-interconnect), which has the short version of this page.

## GPUs

QEngine runs on the RTX 3000, 4000 and 5000 series with no extra steps. The Docker images carry the code for all three,
and a source build targets the GPUs it finds (`-DCMAKE_CUDA_ARCHITECTURES` sets others).

| series | code | what has run on it |
|---|---|---|
| RTX 3000 | sm_86 | 3090: all tuning (82 SMs, tile shapes, split counts, memory budgets) targets it, and everything published here was measured on it. Community: 4× modded 3080 20 GB ([#9](https://github.com/bpx03/QEngine/issues/9)) |
| RTX 4000 | sm_89 | built, but no run on record yet; reports are welcome |
| RTX 5000 | sm_120 | community: 2× 5090 ([club-3090 #1826](https://github.com/noonghunna/club-3090/issues/1826)), 4× 5080 16 GB ([#13](https://github.com/bpx03/QEngine/issues/13)) |

The 4000 and 5000 series use the settings tuned for the 3090, so they may have speed left unused. What limits a card
is its memory; see below.

How many cards each model runs on:

| model | cards | how |
|---|---|---|
| Qwen3.8-Flash-Next | 3 (tuned), 2 | tensor parallel or pipeline, picked at start (`FLASH_PAR=auto`) |
| Qwen3.8-Flash-Next | 4 | tensor parallel when peer access between all 4 equal cards passes the start-up check, else pipeline (`FLASH_PAR=auto`); community runs only, see [4 or more cards](#4-or-more-cards) |
| Qwen3.8-Flash-Next | 5 or more | pipeline (`FLASH_PAR=pp`); tested only as 4 pipeline stages on my 3 cards |
| Qwen3.8-Flash-Next | 1 | not supported |
| Qwen3.8-27B | 3 (tuned), 2 | tensor parallel |
| Qwen3.8-27B | 4 | tensor parallel; the engine accepts it, not tested by me |
| Qwen3.8-27B | 1 | experimental; `QE_MAX_SEQS` 1 to 4 for long conversations ([one or two cards](advanced.md#qwen38-27b)) |

**Smaller cards** (Flash-Next; measured on my 3090s with all but 12 GB or 8 GB of each card taken by another process
before the start, November 3, 2026; defaults otherwise, 16 conversation slots; c=1 / c=4: greedy decode of 400-token
replies, thinking off; prefill of an 11.5K-token prompt; a 53K-token needle; the 27B follows below):

| model | cards x free VRAM | context | c=1 / c=4 tok/s | prefill tok/s | needle |
|---|---|---:|---:|---:|---|
| Flash-Next | 3 x 12 GB | 262,144 | 66.5 / 145.1 | 4,612 | pass |
| Flash-Next | 3 x 8 GB | 262,144 | 47.2 / 91.1 | 3,819 | pass |
| Flash-Next | 2 x 12 GB | 262,144 | 40.7 / 78.7 | 3,546 | pass |
| Flash-Next | 2 x 8 GB | 65,536 (fitted at start) | 34.7 / 55.1 | 2,263 | pass |

- Flash-Next keeps its dense weights, slot state and KV cache on the cards and puts the experts that do not fit in
  host RAM (on 8 GB cards nearly all of them, so decode is host-link bound). The minimum per card is about 7.5 GB on
  3 cards and about 8 GB on 2 cards at a 65K context (about 10 GB at 262K); when the requested context does not fit,
  it starts with the largest that does and says so (`--max-ctx` sets it, `FLASH_MAX_SEQS` fewer slots leave room for
  more). Below the minimum it stops with a message naming what to change.
- 27B needs its weights on the cards: about 5 GB per card on 3 cards and 7.2 GB on 2, plus buffers, the conversation
  slots' state (110 MB per slot and card on 3 cards, about 200 MB on 2) and the KV cache. On a small card it starts
  with fewer slots and a smaller VRAM reserve (logged; `QE_MAX_SEQS`, `QE_RESERVE_GB` set them) so the KV cache holds
  at least a 32K context: 8 GB cards work on 3 cards, 12 GB cards on 2. A smaller pool still starts, with a warning;
  it stops with a message (exit code 3) only when the pool cannot hold one 2,048-token prefill chunk plus the
  engine's run-ahead room.
- 27B on one card needs about 15.8 GB for the weights alone, so a single 16 GB card does not start; 2 x 16 GB does.

**27B on smaller cards, and with a 4-bit KV cache (opt-in, `QE_KV_BITS=4`)** for more context. Each key and value
row is rotated (a fixed random Hadamard transform, which spreads the outlier channels) and stored in 4 bits with one
fp16 scale per 32 values: 144 bytes per row instead of 256 with FP8, so the pool holds 1.6-1.7x the tokens (the MTP
layer stays FP8). Measured like the table above (November 14, 2026, same day for both caches, before the INT4
verify head of 1.3.6, which leaves each card more room for the KV pool; the 3 x 8 GB FP8 row is from November 3;
needle at about 60 % of the served context; slots as fitted for FP8):

| cards x free VRAM | KV cache | context | c=1 / c=4 tok/s | prefill tok/s | needle |
|---|---|---:|---:|---:|---|
| 3 x 12 GB | FP8 | 255,649 | 220.2 / 630.1 | 6,372 | pass |
| 3 x 12 GB | 4-bit | 262,144 | 189.8 / 653.4: the context now fits DFlash2, which is slower than MTP on this story prompt (`QE_DRAFTER=mtp`: 212.9 / 664.6) | 6,231 | pass |
| 3 x 8 GB | FP8 | 44,675 (9 slots) | 220.6 / 619.3 | 6,288 | pass |
| 3 x 8 GB | 4-bit | 67,584 (9 slots) | 215.2 / 630.2 | 6,364 | pass |
| 2 x 12 GB | FP8 | 30,720 (14 slots) | 171.5 / 338.5 | 4,568 | pass |
| 2 x 12 GB | 4-bit | 46,080 (14 slots) | 190.7 / 513.3 | 4,537 | pass |
| 1 x 24 GB, `QE_MAX_SEQS=1` | 4-bit | 240,640 with DFlash2 (262,144 with `QE_DRAFTER=mtp`; FP8 on 1.3.5.8: about 191K) | 102.9 / - | 2,550 | pass |
| 2 x 8 GB | either | does not start (the weights do not fit) | - | - | - |

- Quality: next-token KL against a bf16 KV cache over 100K / 32K / 64K-token documents is 4.4e-3 with 4 bits and
  1.5e-3 with FP8 (for scale: the INT4 weights against BF16 are 1.45e-2); top-1 agreement stays at FP8's level, a
  passphrase needle at 130K (3 depths) and 239K passes, MTP / DFlash2 acceptance within 2 %. It is opt-in because it
  is not lossless against FP8; on cards with room the FP8 default is the better choice.
- Speed: decode within 2 % of FP8 at 32K and faster at long context (128K: -9 % time per step); prefill 0.4 % slower
  at 128K and 3 % at 256K (the prompt's keys are converted for the prefill kernels, about 0.5 KB per key and KV head
  of the served context). On 3 x 12 GB the context now fits the DFlash2 drafter, which `QE_DRAFTER=auto` then loads,
  as on 24 GB cards: warm c=1, it is 10 % faster than MTP alone on code and 7 % slower on prose, the same on both
  memory sizes (the c=1 numbers in the table are a first, cold request).
- Also checked with 4 bits: conversations restored from the RAM archive continue bit-identically, the disk cache
  resumes after a restart, the KV pool shrinks and grows back when another process takes and frees 16 GB of a card,
  and `--max-ctx 512K` (YaRN) serves 524,288 tokens on 3 cards with a 331K-token needle found.
- Not with `QE_LSA` (sparse attention reads FP8 keys; it is turned off with a start-up line) and not with
  `QE_KV_FP8=0`. Conversations archived or saved to disk with one cache format are not reused by the other.

GPU selection: `QE_DEVICES=0,2` (physical ids; Docker `QE_GPUS`), or `--devices` (indices within
`CUDA_VISIBLE_DEVICES`). With neither `QE_DEVICES` nor `CUDA_VISIBLE_DEVICES` set, `server/run.sh` uses every GPU
present (counted with `nvidia-smi -L`; `0,1,2` when `nvidia-smi` is missing); Inkling-2-Mini still takes one card
and GLM-5.3-Flash the first three, and DeepSeek-V4.1-Flash needs three. On a machine with more than 3 cards, set
`QE_DEVICES` to pick them. The 27B and Flash-Next lock the GPUs they use: a second one of them on a card in use stops
at start with the first one's pid and command line (`QE_ALLOW_SHARED_GPU=1` skips the check). The other engines take
no lock.

## With and without GPU P2P

GeForce cards have no GPU-to-GPU P2P with the stock NVIDIA driver. A community patch of the open kernel modules
([aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules), a fork of tinygrad's patch
that tracks current driver versions, with Resizable BAR on)
enables it. QEngine detects which one you have and needs no setting.

Since 1.3.4 both engines are fast without P2P: their tensor-parallel exchanges copy through pinned host memory by
DMA (the copy engines), and the GPU cores only wait on flags in their own memory. Measured on 3× RTX 3090 (PCIe 4.0
x16), with P2P forced off (`QE_P2P=0`) against P2P on (November 1, 2026):

| measured (s, lower is better) | without P2P | with P2P | 1.3.3 without P2P |
|---|---:|---:|---:|
| Flash-Next prefill, 10K / 32K / 90K-token prompt (`flash_run`) | 1.83 / 6.06 / 17.03 | 1.35 / 4.54 / 13.06 | 3.75 / 12.55 / 34.95 |
| Flash-Next decode step, one conversation | 9.9 ms | 9.5 ms | 12.7 ms |
| 27B prefill, 16K-token prompt | 2.68 | 2.59 | 3.94 |

The 27B's decode speed does not depend on P2P. Without P2P the 27B uses a smaller KV pool (its position-split KV
layout needs P2P). Inside Docker, P2P works exactly when it works on the host (check the `P2P:` / `[comm]` lines of the
start-up log).

## Parallel mode (Flash-Next, `FLASH_PAR`)

Flash-Next can share the cards in several ways; the 27B always runs tensor parallel. `FLASH_PAR` selects the mode
(source: `FLASH_PAR=pp server/run.sh ...`; Docker: `-e FLASH_PAR=pp`).

| mode | `FLASH_PAR` | what it does | recommended for | picked by `auto`? | how to force / options |
|---|---|---|---|---|---|
| automatic (default) | `auto` | picks `tp` or `pp` at start from a link probe, a short per-card speed calibration and a model of a typical agent request | almost every machine | - | `FLASH_PAR_AUTO_P=32768` weighs 32K-token prompts instead of 8K (favours `pp`) |
| tensor parallel | `tp` | every layer split over the cards; all cards work on every token, exchanges in every layer (P2P, or host-staged without it) | 2-3 equal cards: fastest decode and short prompts, with or without P2P | yes: 2-3 equal cards; exactly 4 equal cards with verified peer access between all of them, if it fits | `FLASH_PAR=tp`; `FLASH_COMM` sets the exchange path |
| pipeline | `pp` | each stage holds a contiguous range of whole layers; prompts flow through in micro-batches, several conversations decode as groups moving stage to stage; only a 40 KB row per token crosses between stages, so it needs no P2P | long uncached prompts with short replies (RAG, long documents); uneven cards; 4 or more cards | yes: 5+ cards; 4 cards when peer access is missing or unverified, the cards are uneven, or tp does not fit; and uneven cards (or 2-3 cards without P2P) when it wins the mix | `FLASH_PAR=pp`; `FLASH_PP_LAYERS=17,17,14` sets layers per stage; a card may hold several stages (`--devices 0,1,2,0`; a test layout, it adds no compute) |
| pipeline over TP groups (experimental, untested on 4 cards) | `pptp` | a pipeline whose stages are tensor-parallel groups of 2-3 cards; per-layer exchanges stay inside a group | 4+ cards where P2P exists only inside card pairs (two PCIe switches, two CPU sockets) | no, opt-in only | `FLASH_PAR=pptp FLASH_PP_MAP=0,1\|2,3` (`,` inside a stage, `\|` between stages); without a map `FLASH_PP_TP=2` pairs consecutive cards |
| pipelined prefill, `tp` decode | `pp_prefill` | prompts run through the layer pipeline, then the state moves to `tp` for decode | cards with 58 GB or more each (on smaller cards a 2- or 3-card server runs `tp` instead; on 4 or more cards the start is refused) | no, opt-in only | `FLASH_PAR=pp_prefill` |

`auto` models an 8K-token uncached prompt and a 512-token reply, averaged over 1, 4 and 8 conversations at once
(capped at `FLASH_MAX_SEQS`); `pp` must beat `tp` by 10 % over that mix to be picked. With P2P on equal cards it is
always `tp`. Settings of the pick and the pipeline: [docs/advanced.md](advanced.md#parallel-modes-flash_par).

**Recipes**

| your machine and workload | setting |
|---|---|
| 2-3 equal cards, with or without P2P, agents / chat | leave `auto` (= `tp`) |
| mostly long uncached prompts (RAG, long documents, short answers) | `FLASH_PAR=pp`, or `FLASH_PAR_AUTO_P=32768` to let `auto` decide on that basis |
| uneven or mixed cards | leave `auto`: it calibrates each card and picks `pp` with a split by speed when that wins |
| exactly 4 cards | leave `auto`: `tp` with verified peer access between all of them, otherwise `pp`; mostly long uncached prompts: `FLASH_PAR=pp` ([why](#4-or-more-cards)) |
| 5 or more cards | leave `auto` (= `pp`); if your cards pair up on P2P, try `FLASH_PAR=pptp FLASH_PP_MAP=0,1\|2,3` |
| steering vectors or `FLASH_KV=ram` needed | `FLASH_PAR=tp` (2-4 cards) |

```bash
FLASH_PAR=pp server/run.sh 8080 --model models/Qwen3.8-Flash-Next-QE-INT4
docker run ... -e FLASH_PAR=pp -e QE_MODEL=/app/models/Qwen3.8-Flash-Next-QE-INT4 ghcr.io/bpx03/qengine:latest
docker run ... -e FLASH_PAR=pptp -e FLASH_PP_MAP='0,1|2,3' -e QE_GPUS=0,1,2,3 ... ghcr.io/bpx03/qengine:latest
```

**What was picked.** The start-up log has one line with the mode and the reason, e.g.
`[qflash] parallel mode: tp (auto: no peer access, but TP not beaten by the margin over the mix); probe: ...`, and a
`[qflash] cards:` line with each card's measured speed and free memory (cards of the same model count as uneven only
when a longer re-calibration confirms the gap). `/api/stats` on the monitoring port shows the same under `parallel`.

**The trade-off.** `pp` prefills 32K-90K-token prompts up to 1.8× faster than `tp` (3 cards with P2P: a 90K prompt
in 9.9 s against 13.6 s; on 2 cards with P2P the two are about even), decodes 4-8 conversations at 0.83-1.03× of
`tp`, but one conversation at only 0.54-0.65×; 10K prompts with P2P are faster in `tp`. Measured on the server, same build, cold prompts end to end and decode of 1 / 4 / 8
conversations (November 1, 2026):

| cards, link | mode | 10K prompt | 32K prompt | 90K prompt | decode tok/s, 1 / 4 / 8 at once |
|---|---|---:|---:|---:|---:|
| 3, P2P | `tp` | 1.52 s | 5.20 s | 13.55 s | 244 / 541 / 773 |
| 3, P2P | `pp` | 1.72 s | 3.89 s | 9.88 s | 132 / 447 / 696 |
| 3, no P2P | `tp` | 2.05 s | 7.09 s | 18.03 s | 232 / 487 / 708 |
| 3, no P2P | `pp` | 1.77 s | 3.95 s | 10.21 s | 136 / 438 / 703 |
| 2, P2P | `tp` | 2.54 s | 5.67 s | 15.96 s | 91 / 200 / 246 |
| 2, P2P | `pp` | 3.08 s | 5.59 s | 14.75 s | 56 / 203 / 245 |
| 2, no P2P | `tp` | 3.78 s | 8.31 s | 23.45 s | 88 / 195 / 236 |
| 2, no P2P | `pp` | 2.97 s | 5.63 s | 14.85 s | 57 / 201 / 241 |

Each row is one fresh server; the decode runs come first with a cold expert cache, which costs 2-card single-stream
decode most. The prompts and method differ from the [headline table](../README.md#models), so compare rows within this table only.

**Uneven cards.** `pp` splits the layers by each card's measured speed and free memory. With one 3090 slowed to a
third of its compute (a test setting), the split moved 4 layers off it: prefill about 30 % and decode 13 % faster than
an equal split.

**What `pp` and `pptp` do not serve**: image input in `pptp` (refused with a clear message; `pp` serves images since 1.3.5.1, experimental), steering vectors (not loaded),
`FLASH_KV=ram` (on 2 or 3 cards the server runs `tp` instead and logs it; on 4 or more cards the start is refused with
a message), activation capture; lookup drafts are off. Structured output and the thinking budget work as in `tp`
(a constrained step runs through the stages as a batch of one).

### 4 or more cards

I have no 4-card machine: the 4-way split was
tested as 4 pipeline stages on my 3 cards (two stages on one card), where it gave the same output bit for bit as 3
stages. `pptp` was tested only in a 3-card layout (`0,1|2`), where its next-token probabilities match `tp` within the
run-to-run noise. A community run on four RTX 5080 cards (`tp`, 1.3.5.7 and 1.3.5.9, [#24](https://github.com/bpx03/QEngine/issues/24))
confirmed Flash-Next's long-context recall there: needles found up to 240,664 tokens at 262K, and at 1M up to 843,147
on 1.3.5.7 and 813,180 on 1.3.5.9 (one ladder rung lower), the same level as on 3 cards. More reports from 4-card machines are welcome.

**Since 1.3.5.4 (experimental):** on exactly 4 equal cards, `auto` picks `tp` when peer access between all of them passes the data check
at start and `tp` fits the free VRAM; otherwise it picks `pp`. With 5 or more cards it is always `pp`. If `tp` can't start
(out of device or host memory during the load, or a CUDA error or a stall in the check), the server restarts itself in
`pp` and says so in the log, at the cost of the partial load. On a card whose free VRAM is badly fragmented, a `tp` that did
load can still stop later on a request: use `FLASH_PAR=pp` there. This rule rests on one user's measurements on one rig
(4× RTX 5080 16 GB behind one PCIe switch, 1.3.5): one 1,000-token stream at 310.3 tok/s in `tp` against 145.4 in `pp`. On
1.3.4.3, prefill at 32K was higher in `pp` (7,367 against 5,466 tok/s), so for mostly long uncached prompts set
`FLASH_PAR=pp`. The rest was tested on 3 cards only. On 1.3.5 to 1.3.5.3, `auto` always picks `pp` on 4 cards and
prints `peer access no` even if your cards have P2P, so set `FLASH_PAR=tp` yourself.

## Link probe and `FLASH_COMM` / `QE_COMM`

At start both engines measure every GPU pair for a few milliseconds (P2P loads and stores, copy-engine copies, copies
staged through host memory), how much each transfer slows a kernel running next to it (the co-run tax), and whether
data sent over P2P really arrives intact:

```
[comm] topology: numa -1,-1,-1 (1 node), PCIe x16,x16,x16, P2P rank 0-1:0 0-2:0 1-2:0
[comm] P2P data check: SM loads, SM stores and copy engine ok on every pair (...)
[comm] prefill transfers: P2P SM loads / stores, ... (FLASH_COMM=auto, auto: ...: P2P kept: tax over the cap, under the 1 NUMA node cap)
[comm] decode collectives (QE_DECODE_COMM=auto, real decode steps, median of 4 interleaved rounds of 12): single sequence: P2P SM stores (P2P ..., host memory ... ms/step, host faster in 0/4); batched (4 sequences): ...; calibration ... ms
```

With `auto` (the default) Flash-Next keeps P2P on fast links, moves the prefill transfers to the copy engine on slow,
narrow (e.g. an x4 slot) or cross-socket links, and never uses a transport that failed the data check. The line ends
with the reason. `FLASH_COMM` (Flash-Next) and `QE_COMM` (27B) force a path: `p2p`, `ce` (copy engine), `host`.
The rule and its thresholds: [docs/advanced.md](advanced.md#how-flash_commauto--qe_commauto-choose-a-path).

Decode sends only small messages, but several per layer and step (all-reduces, the attention exchanges): either as
SM stores into the other cards' memory (P2P) or through the host path, with the same results. The host path is the
one the engines use without peer access (`QE_P2P=0`): each card writes its words once into pinned host memory and the
others read them; peer access stays on for prefill. Which one is faster depends on the board: peer stores can stall the
other kernels running on a card, and only the real model shows by how much. With `auto` both engines time a few real
decode steps on each path at the end of the start-up (one sequence, and four batched sequences), keep P2P or the host
path for each, and print both timings on the `[comm] decode collectives` line. `QE_DECODE_COMM=p2p` or `host` forces
the decode path (both engines; prefill keeps its own path). Before 1.3.5.6 the host candidate was a slower path
(about 45 % more time per step on 3 cards), so boards where P2P stores are expensive decoded slower than on 1.3.4.3. The measurements are cached in `~/.cache/qengine` (the Docker volume) and reused while the cards, links, driver and
build stay the same; `QE_COMM_CACHE=refresh` measures again. A cached decode line says `from the start-up cache, not
measured this boot` and carries the date of the measurement, so compare its ms/step across boots with care; a round
that was not measured (with fewer than 4 slots there is no batched round) prints `n/a`.

Set `FLASH_COMM=ce` (27B: `QE_COMM=ce`) yourself when the `[comm]` line says P2P but:

- long prompts fail with `non-finite values` (HTTP 500) or the log shows `GUARD` lines;
- a card sits behind something the link width does not show (a PCIe switch, a riser or extension cable, a chipset
  slot, a bifurcated slot);
- the cards differ (e.g. a 3090 next to a 3090 Ti) or sit in slots of different PCIe generations, and the line shows a
  P2P co-run tax of about 1.9 or more;
- prefill is clearly slower than the numbers here.

The copy engine gives the same output; on fast x16 links P2P is a few percent faster in prefill.

**NUMA.** On a machine with several NUMA nodes, Flash-Next places pinned host memory by GPU (each card's host-backed
experts and staging buffers on its node, the n-gram table interleaved over the GPUs' nodes). When the system does not
report a card's node (`GPU NUMA ID: N/A`), those buffers are interleaved over all nodes; `QE_NUMA` sets the nodes by hand.

**Slow decode, or poor scaling with concurrent requests, on a board with a high P2P co-run tax.** Look at the
`[comm] decode collectives` start-up line: it shows the measured ms per decode step on both paths and the pick for
one and for several sequences. If the two timings are close, or your own runs disagree with the pick, compare
`QE_DECODE_COMM=host` and `QE_DECODE_COMM=p2p` at 1 and 4 concurrent requests and report the `[comm]` lines with
both numbers. Before 1.3.5, decode always used P2P stores when peer access worked; `QE_P2P=0` was the only way off
them, and it also moves prefill to the slower host-staged path. On 1.3.5 to 1.3.5.5 the host candidate was slower than
`QE_P2P=0` itself; on such a board use 1.3.5.6 or later, or `QE_P2P=0` on the older versions.

## Host memory (Flash-Next)

Both engines plan their host-RAM parts from the machine's size, not from what happens to be free at start. If other
programs hold RAM at start, the plan warns and stays: the archive keeps below `MemAvailable` minus the margin while
memory is short and grows when it comes back. Only the Flash-Next n-gram table is decided at start: when it does not fit
next to the experts as it is read, it comes from the page cache until the next restart (decode about 30 % slower at
c=1). If the free RAM is short at start, the server first waits up to 20 s for it (a previous server still exiting;
`QE_RAM_WAIT_S`). If the model can load it runs, with warnings; the load stops with a clear message only when the
experts really do not fit. The start-up line `host RAM plan: ...` says what was reduced. While serving, when another
program takes RAM, the archive gives memory back (log lines `archive: host memory below the margin ...`) and the server
keeps running.

Measured minimum free host RAM at start (3 cards, November 3, 2026): Flash-Next about 28 GB (the host-backed experts
must be pinned: about 12 GB on 3 cards, about 31 GB on 2, plus a margin), full speed (the n-gram table pinned) from
about 85 GB; 27B started with 24 GB free (the lowest I tried). Below its minimum Flash-Next stops at start with a
message naming the shortfall, before it pins anything.

| part | host RAM |
|---|---|
| host-resident experts (what VRAM cannot hold) | about 12 GB (3 cards), about 31 GB (2 cards) |
| n-gram embedding table (INT8, pinned) | 51.8 GB |
| conversation archive (`FLASH_KV_ARCHIVE_GB`) | 48 GB, filled as conversations leave VRAM |
| pinned system prompts (`FLASH_PREFIX_GB`) | 8 GB, as agent kinds appear |
| server process (Python, vision encoder, CUDA) | about 5 GB |
| `FLASH_KV=ram` (opt-in) | 13.5 GB per 1M tokens of KV, pinned at start; smaller when the host is short (see below) |

**The defaults fit the machine (since 1.3.4).** At start-up the engine plans its host memory from `MemTotal` and
`MemAvailable` (and a cgroup / `docker --memory` limit, if one is set). It first sets aside a reserve for the system
and the desktop: max(8 GB, 8 % of RAM). The rest goes, in this order, to:

1. the host-resident experts, which are not optional;
2. the n-gram table. It is pinned when it fits next to an 8 GB minimum archive. Otherwise it is read through the page
   cache (as with `FLASH_NGRAM_PIN=0`, below), and part of the remaining RAM stays free for the page cache to hold
   its rows;
3. the archive and the pinned system prompts, which get what is left, up to 48 + 8 GB.

With `FLASH_KV=ram` the KV pool in host RAM (`FLASH_KV_RAM_TOKENS`, 13,520 bytes per token, 13.5 GB at the default 1M;
7,488 bytes and 7.5 GB with `FLASH_KV_BITS=4`) is in the plan as well. It is pinned at start, like the experts and the n-gram table, so these three together must fit
both the RAM the machine has and what is free at start (`MemAvailable` minus the margin). When they do not, the plan
gives way in this order: the archive and the pinned prompts first; then the KV pool, down to the seats times the
context (`FLASH_MAX_SEQS` x `--max-ctx`, so every seat still holds a full window); then down to one window; then the
n-gram table moves to the page cache (it costs decode speed for every request, so it goes last). The experts are never
reduced; if even the smallest plan does not fit, the log warns and the server starts anyway. The start-up line shows
the pool and what is pinned at start next to what was free.

**Flash-Next with a 4-bit KV cache (opt-in, `FLASH_KV_BITS=4`).** Each key and value row of the attention layers (the
MTP layer too) is rotated (a fixed random Hadamard transform, which spreads the outlier channels) and stored in 4 bits
with one fp16 scale per 32 values: 576 bytes per layer and token instead of 1,040 with FP8, 7,488 instead of 13,520
bytes per token in all. The VRAM pool keeps its 262,144 tokens and gives the freed memory to the expert cache; with
`FLASH_KV=ram` the same host RAM holds 1.8x the tokens (1.05M tokens: 7.9 GB pinned instead of 14.2). Measured on 3 cards
(November 14, 2026, same day for both caches): KL against the FP8 cache 4.95e-2 on 32K-100K documents (two FP8 runs
that differ only in summation order: 2.52e-2), top-1 agreement 92.2 %; needle at 130K and at 1M (`FLASH_KV=ram`,
YaRN) pass; prefill of a 90K prompt 1-2 % faster, decode steps unchanged, MTP drafts accepted 3 % less (3.00 vs 3.10
tokens per step at 90K). Other values of `FLASH_KV_BITS` and settings that select an attention kernel without a
4-bit reader (`FLASH_ATTN_TC`, `FLASH_ATTN_DEC=0`) keep the FP8 cache and log it. On 2 cards, prompts of about
115K-188K prefill in 4096-row batches with the 4-bit cache (the smaller K/V arrays have less room for the big
batches). Known limit, from one community report on 4 cards
([#25](https://github.com/bpx03/QEngine/issues/25)): on a 1M-token recall ladder the 4-bit cache filled 83 % of the
window (every miss a refusal) against 91 % with FP8, so keep FP8 when maximum reach matters.

Nothing is planned beyond the RAM the machine has, so the defaults do not push it into swap. With about 140 GB or
more (3 cards) or about 160 GB or more (2 cards), the plan is exactly the defaults above. The log has one line with
the plan, and `/api/stats` reports it as `host_memory`:

```
[qflash] host RAM plan: MemTotal 98.8 GB, MemAvailable 92.8 GB, reserve 8.0 GB (OS + desktop), server base 6.0 GB,
engine buffers 1.3 GB; experts in host memory 31.2 GB (estimate 31.3 GB), n-gram table 51.8 GB from the page cache
(12.0 GB kept for it); KV archive 34.5 GB + pinned prefixes 5.7 GB; reduced: n-gram table not pinned: ...
```

Planned sizes by machine (start-up plan, desktop using about 6 GB):

| RAM | cards | n-gram table | archive + pinned prompts |
|---|---|---|---|
| 64 GB | 2 / 3 | page cache | 9.6 + 1.5 / 25.2 + 4.1 GB |
| 92 GB | 2 | page cache | 34.5 + 5.7 GB |
| 92 GB | 3 | pinned | 16.8 + 2.8 GB |
| 128 GB | 2 / 3 | pinned | 30.9 + 5.1 / 47.4 + 7.9 GB |
| 160 GB and more | 2 / 3 | pinned | 48 + 8 GB (the defaults) |

**Overrides.** A value you set wins: `FLASH_NGRAM_PIN=1` pins the table, `=0` keeps it in the page cache;
`FLASH_KV_ARCHIVE_GB` and `FLASH_PREFIX_GB` set the archive and the pinned prompts (`0` turns them off). If a value you
set does not fit, the log says so and the value is kept. On Flash-Next `FLASH_KV_ARCHIVE_GB=0` also turns off the pinned
prompts and every reuse that goes through the archive: no checkpoints, no copy of a cached prefix from another slot, and
a conversation that leaves its slot is prefilled again on its next turn. This holds with `FLASH_KV=ram` too: the KV pool
in host RAM holds only the conversations in the slots, it does not replace the archive. `QE_HOST_BASE_GB` (6) changes the share counted for the
server process.

**Keeping RAM free for other programs: two settings, used at different times.**

| setting | when it acts | what it does | raise it when |
|---|---|---|---|
| `QE_HOST_RESERVE_GB` | once, at start | RAM the start-up plan leaves out when it sizes the n-gram table, the archive and the pinned prompts | other programs will need more RAM later, or the plan comes out too large |
| `QE_MEM_MARGIN_GB` | all the time, while serving | free RAM (`MemAvailable`) the archive keeps: below it, the archive stops growing and gives memory back | programs that start later get squeezed, or swap appears |

On a small machine, lower them to leave more for the archive: with both at their 8 GB default, a 16 GB machine has
almost no archive, so a conversation that leaves VRAM is recomputed when it returns (a planned archive under 2 GB gets a
warning line at start; the start goes on). The lower you go, the closer the
kernel's OOM killer gets, and it would end the server.

- On **2 cards** about half of the expert groups stay in pinned host memory (about 15 GB per card) and are read over
  PCIe or copied into the expert cache when hot.
- While it fills, the archive keeps at least max(8 GB, 5 % of RAM) free (`QE_MEM_MARGIN_GB`), and gives memory back
  when other programs need it. The engine also watches the kernel's memory-pressure and swap-in counters while it runs:
  when other programs squeeze the host, it stops growing, shrinks the archive, drops pinned prompts and, last, moves the
  n-gram table to the page cache, one logged stage at a time, and keeps serving (`QE_MEMP=0` turns this off;
  [details](advanced.md#host-ram-safety)).
- **n-gram table in the page cache** (`FLASH_NGRAM_PIN=0`, or chosen at start-up): the table stays in the file mapping
  and a host thread copies the rows each step needs. With the table fully in the page cache, agent throughput at 8
  conversations was 793 tok/s (pinned: 805); with the whole server limited to 12 GB of RAM, 493 tok/s (3 cards,
  October 25, 2026). After the load the engine drops the page cache of the weight files (it never reads them again),
  so they don't push the table out of the page cache (`FLASH_WEIGHTS_COLD=0` keeps it). Details:
  [docs/advanced.md](advanced.md#disk-mode-for-the-n-gram-table).
- The 27B needs much less: its archive defaults to 32 GB (`QE_KV_ARCHIVE_GB`) plus 8 GB of pinned prompts
  (`QE_PREFIX_GB`), sized the same way. Its defaults fit from 64 GB of RAM.

## Operating system

Linux only (developed on Fedora 44). **Windows, including Docker Desktop / WSL2, is untested**: WSL2 has no GPU P2P
and limits pinned host memory, which the archive and Flash-Next's host-resident parts rely on.
