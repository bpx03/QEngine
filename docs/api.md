# API and serving behaviour

Back to the [README](../README.md). What each model supports: [Feature matrix](../README.md#feature-matrix).

## API

- **OpenAI**: `POST /v1/chat/completions` (streaming or not), `GET /v1/models`.
- **Anthropic**: `POST /v1/messages`, `POST /v1/messages/count_tokens`.
- `GET /health`, `GET /props` (llama.cpp style), `GET /v1/steering`.

`/v1/models` and `/props` report what a client needs to size its requests: `max_model_len` (prompt + reply tokens of
one conversation), `max_num_seqs` (conversations served at once), `max_prompt_tokens`, `max_output_tokens` (`null`:
no cap beyond the context), `kv_pool_tokens` and `kv_cache` (`vram` / `ram`); `/v1/models` adds `input_modalities`,
`/props` adds `modalities.vision`, `total_slots` and `default_generation_settings.n_ctx`.

**Too-long prompts** get HTTP 400 before any streaming starts. The Anthropic path says
`prompt is too long: N tokens > M maximum` (Claude Code compacts the conversation on it). The OpenAI path uses
OpenAI's wording, `This model's maximum context length is M tokens. However, your messages resulted in N tokens ...`
with code `context_length_exceeded`, which harnesses such as verifiers / AutomationBench and LiteLLM recognise as a
context overflow.

**Tool calls**: OpenAI `tools` and Anthropic `tool_use` / `tool_result`. Tools are rendered sorted by name with
sorted JSON keys, so a client that reorders the same tools keeps the prompt cache (`--sort-tools 0`: the client's
order). The chat template is `server/templates/qwen38_syssafe.jinja` (the model's template, accepting several leading
system messages). `--template` replaces it. A reply starts as reasoning only if the template's generation prompt opens
`<think>`: with a custom template that does not, a model reasoning without writing the tag shows that text in `content`,
followed by a literal `</think>`. Since 1.3.6 a `<tool_call>` that does not parse as a call (prose inside the tag,
broken markup), or a call that `tool_choice` filters out, reaches the client as plain text in `content`, streamed or
not, on both APIs, instead of an empty reply.

**`tool_choice`** on both APIs: `auto` (default), `none` (Anthropic `{"type": "none"}`: the tools stay in the prompt, the
tool-call token is masked, so the model answers in text), `required` (Anthropic `any`) and a named function (OpenAI
`{"type": "function", "function": {"name": ...}}`, Anthropic `{"type": "tool", "name": ...}`): the answer starts with
that call after any thinking. Enforced in the engine: `none` bans the tool-call token in the normal decode path, so
such a reply keeps its batching and speculative drafts and runs as fast as `auto` (with `logprobs` it is masked on the
constrained path instead). A forced call runs through its thinking and the call header on the same masks as structured
output (on Flash-Next such a reply steps outside the batch until then). Calls to other tools than a named one are
dropped. Client stop sequences do not cut a forced reply before its call is complete; they
apply after it. Under `none` a call the model writes out as plain text is text and arrives in `content`. An
unknown name, or a forcing choice without tools or together with `response_format`, is a 400.

**Structured output**: `response_format` with `json_object` or `json_schema`, enforced in the engine (llguidance
grammars) together with speculative decoding. Stop sequences are supported.

**Sampling**: values a request doesn't send come from the Qwen3.8 model cards, per reply mode. Thinking: temperature 1.0,
top_p 0.95, top_k 20, min_p 0, presence penalty 0. Non-thinking: temperature 0.7, top_p 0.80, top_k 20, min_p 0, presence
penalty 1.5. `--temperature`, `--top-p`, `--top-k`, `--min-p` and `--presence-penalty` set one value for both modes, and a
value in the request wins. Only a missing value takes the default; `0` is a value (also in the Anthropic API):
`top_k: 0` means no top-k cut (27B, Flash-Next and DeepSeek-V4.1-Flash: the 64 candidates their sampler keeps; the
other engines: the full vocabulary), and `top_p: 0` keeps only the most likely token, so the reply is greedy, as with
temperature 0. `seed: 0` is a seed like any other (a request without `seed` gets a time-based one). Greedy requests (temperature 0) get no penalty. The cards warn that a higher presence penalty may
occasionally mix languages and slightly lower quality: `--presence-penalty 0`, or `presence_penalty: 0` in a request, turns
it off. The speed cost of presence 1.5 was measured on Flash-Next (3 cards, 8 agents at once, November 15, 2026):
with a step's verify rows sampled one after another it took 17-20 % of the throughput; since 1.3.6 they are sampled
in parallel (`QE_SAMPLE_PAR`), which gave +16 % on that test (697 → 808 tok/s). Its effect on answer quality and
tool calls has not been measured here.
`presence_penalty` and `frequency_penalty` are applied on the GPU.

**`logprobs` / `top_logprobs`** on `/v1/chat/completions` (`top_logprobs` 0-20, streamed too; since 1.3.5.4, experimental). The numbers
are the target model's own next-token distribution: raw logits through log-softmax over the full vocabulary, before
temperature, top-p, top-k, min-p and penalties, never a draft head's. Steering, when used, is included. `content` covers
only the tokens of `content`, not reasoning or tool-call tokens; with `logprobs` the non-streamed `content` is not stripped,
so the entries join to it. Such a request decodes one token per step without draft speedup, about 3x slower, and while it
runs a normal reply beside it drops to about 0.4-0.45 of its speed alone. In `pp` the numbers carry the mode's own
rounding, so use `tp` for KL-grade measurements. On the 27B the top 64 values of each row are exact BF16 logits; the
normaliser also sums the INT4 screen's values for the rest of the vocabulary (log-prob error about 0.01 at most).
Prompt logprobs and `/v1/completions` are not available.

**Thinking and reasoning effort**: thinking is on by default (`--think 0` turns it off by default;
`chat_template_kwargs.enable_thinking` per request; a top-level `enable_thinking` is accepted too when
`chat_template_kwargs` does not set it, and a `null` counts as not sent). `reasoning_effort` (`xhigh` = the template default, `high` is the
same, `medium`, `low`) per request or as a server default with `--reasoning-effort` / `QE_REASONING_EFFORT`; a
request's value wins. `max` is taken as `xhigh` and `minimal` as `low` (the names other APIs use); other unknown
values render as `medium`. On `/v1/messages` the Anthropic `output_config.effort` is used when the request sends no
`reasoning_effort` (so `max` gets the most reasoning). The OpenAI Responses / OpenRouter form `reasoning: {"effort": ...}`
is read too, on both paths, after `chat_template_kwargs.reasoning_effort` and the top-level `reasoning_effort`.
`none` (any case, in any of these fields) turns thinking off,
the same as `enable_thinking: false`, also when the same request sets `enable_thinking: true` (an "off" always wins).
`QE_REASONING_EFFORT=none` makes thinking off the default; a request that names another effort, sets
`enable_thinking: true` or the Anthropic `thinking.type` `enabled` / `adaptive` still thinks.
The OpenRouter switches are read last: `reasoning: {"enabled": false}` is thinking off when the request sends no other
thinking setting or effort; `"enabled": true` turns thinking on over the server default (`--think 0`,
`--reasoning-effort none`) at the effort the request names, unless the same request turns thinking off.
`reasoning: {"exclude": true}` on `/v1/chat/completions`: the model reasons as usual (the reasoning tokens are in
`usage`), but the reply carries no `reasoning_content`, streamed or not.

### Thinking budget and the answer tail

Effort only asks the model to think less. A model can still reason until `max_tokens` runs out and return a turn
with reasoning only (no answer, no tool call), which an agent takes as the end of its task and a benchmark scores as
zero. A budget stops that:

- per request: Anthropic `thinking: {"type": "enabled", "budget_tokens": N}`; OpenAI path `thinking_budget` (top
  level or in `chat_template_kwargs`; `thinking_token_budget`, `max_thinking_tokens`,
  `custom_params.thinking_budget` and `reasoning.max_tokens` are accepted too);
- server default for requests that send none: `--thinking-budget N` / `QE_THINKING_BUDGET=N` (0 = off), one budget
  for every request. A request's value wins; a negative one turns the budget off for that request; 0 turns thinking
  off.
- or a default per reasoning effort: `QE_THINKING_BUDGET=low:4096,medium:16384,xhigh:32768` (the same value works
  for `--thinking-budget`), optionally with `default:N` (or `*:N`) for efforts the map does not list. The effort is
  the one the prompt is rendered with (the request's, else `--reasoning-effort`, else the template's `xhigh`), named
  the way the template renders it: `high` and `max` use the `xhigh` entry, `minimal` the `low` entry, other unknown
  values the `medium` entry. An entry of 0 or below means no budget for that effort (as a plain 0 or negative
  number). Order: the request's budget, then its effort's entry, then `default:`, then none. With a custom
  `--template` whose efforts the server does not recognise, a warning is printed at start-up, efforts are matched as
  sent, and requests without one use the `xhigh` entry.
- since 1.3.4.3 a malformed value (e.g. `8k`, `1.5`, `low:4k`, an unknown effort) stops the server at start-up with an
  error naming the entry; 1.3.4.2 and earlier warned and ran without a budget.
- the start-up line prints the map, and `/api/stats` (`thinking.budget_from`) and the per-request log line
  (`think 66/4096 (effort:low)`) show where each budget came from: `request`, `effort:<level>`, `default`, `off` (the
  request turned the budget off) or `none`.

At the budget the server writes a one-sentence wrap-up and `</think>` (27 tokens; `QE_THINK_WRAP` changes the
sentence), and the model goes on with its answer or tool call. The reasoning holds exactly the budget plus the
wrap-up: near the budget (and near the answer tail) the server caps how many drafts a step may verify, so the close
lands at the same token whatever the drafts (before 1.3.6 it could come up to one decode step late, and the
text then depended on the drafter).

`max_tokens` counts thinking and answer together. **Answer tail**: when `max_tokens` would run out inside `<think>`,
the same close comes early enough to leave the answer the last eighth of `max_tokens` (64 to 2048 tokens;
`QE_THINK_TAIL`, a token count, `0` = off), with or without a budget (not with `ignore_eos`). `finish_reason` is `length` only when the answer
itself is cut off; `usage.completion_tokens_details.reasoning_tokens` reports the thinking tokens.
`usage.completion_tokens` (Anthropic `output_tokens`) counts the end-of-turn token the model generated, as vLLM
does: a reply of 7 text tokens that ends on its own reports 8. A reply cut by `max_tokens` or a stop string has no
end token, and neither has structured output that ends when its grammar is complete (the server ends that turn
itself). Before 1.3.6 the end token was never counted.

**Cost of the forced tokens.** They are verified like drafts, up to 4 per step on Flash-Next and up to 8 on the 27B,
together with MTP and lookup drafts. On the 27B a constrained request stays in the batched step with the others. On
Flash-Next a constrained step (a forced close, and structured output in general) runs as a step of its own, so with
several conversations decoding at once it adds steps the others wait for. A request without a budget that never
reaches the tail decodes exactly as before.

### Empty replies are sampled again

Now and then a thinking model ends its turn with reasoning only (end-of-turn sampled inside or right after
`<think>`) or with a tool call that does not parse. Agent harnesses (AutomationBench / verifiers and many OpenAI-API
agent loops) end the task on a turn without text and without a tool call. Such a non-streamed, sampled reply
(finish `stop`) is sampled again up to 2 times (`--empty-retry N` / `QE_EMPTY_RETRY`, `0` = off). The prompt is
cached, so a retry costs only its decode. Never retried: streams (the bytes are already sent), greedy (`temperature`
0) and seeded requests, structured output, `ignore_eos`, requests with stop sequences. `usage.completion_tokens`
counts every try; `finish_reason` is the last try's. A tool call that does not parse is retried the same way; when
the last try still does not parse, its text is returned as `content` (streams: at once), so the client never gets an
empty turn from it.

### Images

`--vision 1` (both models; Docker: the `vision` image) accepts images in OpenAI (`image_url`, data URLs or http(s))
and Anthropic (`image` blocks) messages. The vision encoder runs in PyTorch on the last GPU (Flash-Next: ~0.6 GB of
VRAM there, weights streamed from host RAM; 27B: ~1.3 GB). Image tokens take part in prefix caching, keyed on the image
content. A 1600×900 screenshot is encoded in about 0.14 s on Flash-Next. Unreadable images and video / audio parts get
HTTP 400. No vision quality benchmark yet. Details: [docs/advanced.md](advanced.md#image-input).

### Activation steering

Both engines can steer the model with control vectors in the llama.cpp control-vector GGUF format (plus a JSON
sidecar), in `add` or `project` mode, chosen per request with a `steering` field. QEngine ships one vector for
Flash-Next, **off by default**: `refusal_projection`, which projects a refusal direction out of the residual stream so
the model declines far fewer requests. Removing refusals removes a safety behaviour of the model; you are responsible
for what it outputs. Turn it on in Docker with `-e QE_STEERING=1`, from source with
`QE_STEER=vectors/refusal_projection.gguf QE_STEER_DEFAULT=refusal_projection`. Vector format, loading your own
vectors, per-request selection and credits: [docs/advanced.md](advanced.md#activation-steering).

## Serving behaviour

**Scheduling.** Shortest-prefill-first queue with aging; long prompts are prefilled in chunks between decode steps
of the running conversations; short requests can cut in between the chunks of a long prefill (Flash-Next: requests
up to 8K tokens, e.g. tool results, join the next prefill batch at once while the long prompt keeps half of it, on
servers whose prefill is bound by host-backed expert copies; `--long-share`). 27B: the time split follows the queue:
while the prompt to prefill next is short (up to `--prefill-short`, 4,096 tokens), prefill passes run back to back
with one decode step between them; a longer one leaves the running conversations a share by head count (running /
(running + waiting), between `--prefill-share-min` 0.10 and `--prefill-share` 0.5), and over any 2 seconds they keep
at least 10 % of the time. With many agents in flight the first token comes several times sooner (numbers under
[Benchmarks](benchmarks.md)); `--prefill-share-min 0.5` restores the fixed half split. 27B with all decode seats
taken and more requests open than seats: two seats (Flash-Next on 3 cards: four, whose prompts share prefill passes)
stay for prompts; a reply prefilled there sends its first token and is parked in host memory until a seat frees
(oldest first; `--park 0` off). A request that does not
fit the KV pool lets smaller ones start first, until it has waited `--backfill-wait` seconds (30).

**Agents.** Requests are grouped by the client's session id (Claude Code `metadata.user_id`) and by message history
(a client without a session id: by system prompt);
the **main agent thread** (the conversation that receives the others' results) is recognised for the logs and
`/api/stats` only: the prefill queue, the shared prefill batches and memory treat every conversation alike (least
left to compute, aged by waiting time). When the pool is full, a request is admitted by pausing the most
recently admitted decoding conversations (the pool's own order) once it has waited `--pause-wait` seconds (3), and not
while another one is paused; they resume where they stopped. `--main-agent-re`
forces the main-thread label by a regex on the system prompt or tool names.

**KV pool under pressure** (both models). Idle conversations go to the host-RAM archive first. The archive frees memory
in one order: least recently used whole conversation first, every conversation alike (the main agent thread too), each
loss with a log line. The same order frees VRAM slots on both models; a generating conversation never leaves VRAM, and an
idle one never makes a request wait. When the running conversations still do not fit, the most recently admitted one
is paused: its KV goes to the archive and its stream pauses without an end token; it resumes exactly where it stopped
once the others keep their headroom. A reply ends early only when a single conversation is larger than the pool.
On Flash-Next every slot release, archive drop and restore from the archive prints one line (conversation id, tokens,
seconds idle, where it went and why), and `/api/stats` (`engine.conversations`, also in `qetop`) lists the known
conversations least recently used first, in a slot or in the archive, so the next one to leave is the first.

**Speculative decoding.** MTP drafts plus lookup drafts (when recent tokens repeat an earlier part of the
conversation, as in file edits); on the 27B the [DFlash2 block drafter](models/qwen3.8-27b.md#dflash2-block-drafter-27b-on-when-it-fits)
for a conversation that decodes alone. Greedy and sampled outputs keep the model's distribution.

**Loops.** A reply whose last 600 tokens are one short fragment repeated is ended (`--loop-guard-tokens`). When an
agent keeps repeating earlier turns, whether one call or a short cycle of a few calls (same tool, exactly the same
arguments apart from the `description` of a shell call), each getting exactly the same result as before, for 3 turns
(`--loop-guard-turns`; calls that differ only in an id, such as a batch of deletes, and polls whose output changes are
not a loop), the next turn is generated with reasoning on even when thinking is off (at most 2048 reasoning tokens,
`--loop-guard-think`; the cached prompt is kept), with a short note on the tool result, a presence penalty on the
repeated reply and no lookup drafts. The reasoning comes on top of the request's `max_tokens`, and stop sequences apply
only to the answer. A guarded turn thinks even when the client disabled thinking (`enable_thinking: false`, `thinking: {"type": "disabled"}`,
`thinking_budget: 0`); `--loop-guard-think 0` turns this off. A long stretch of unchanged results can be legitimate waiting, so the guard backs off: it applies at 3,
6 and 12 turns, then every 24 turns (also for a client that sends only its last turns). Right after a user message it only turns reasoning on (harness text such as
`<system-reminder>` or `<task-notification>` blocks is not a user message). The note is left out when it would push the prompt over the context.
Conversations without tool results are never flagged. `"loop_guard": false` in a request turns it off.

**Turn end.** A generated `<|im_start|>` ends the reply like the end-of-turn token, so a reply never continues into a
turn the model made up. The log line (`stopping at a generated <|im_start|> ...`) shows the first and last 16 tokens of
the reply.

**Template markers in content.** Marker strings that a conversation only quotes (`<|im_start|>`, `<|im_end|>`,
`<think>`, `<tool_call>`, image placeholders, any added token of the tokenizer) are tokenized as plain text, wherever
they appear: user, system, assistant and tool messages, reasoning, tool-call names and arguments and tool
definitions, also a partial marker at the end of a name that the template would complete. Control tokens appear only
where the chat template puts them, so a conversation that pastes a chat template or server source does not split into
made-up turns, and the model does not copy a real end-of-turn token out of it. One exception follows the template's own
format: a user message that consists only of `<tool_response>...</tool_response>` blocks (tool results sent as a user
message) keeps the tags between and around the blocks. On the output side only the real `<think>`, `</think>` and
`<tool_call>` tokens switch between reasoning, answer and tool call; the same strings written as text stream as part of
the reply, and a tool-call argument that quotes `</parameter>` or `</function>` is not cut there. Conversations
without such strings in their content are tokenized exactly as before.

**Failure handling.**

- A stalled GPU or driver: an engine step that has not finished after `FLASH_RUN_TIMEOUT_S` (120 s, Flash-Next), or no
  scheduler progress for `QE_HANG_S` (180 s, both models), fails the open requests with an error at once, refuses new
  ones, writes bounded diagnostics (stacks, `nvidia-smi`; `QE_HANG_DIAG_S`, 30 s) to the log and exits with code 75 so
  Docker (`--restart`) or systemd restarts it. The step monitor is off under `ncu`, `nsys`, `compute-sanitizer` and a
  debugger. `kill -USR1 <pid>` writes the Python stacks at any time.
- A prompt the engine cannot start fails at once with the reason: an invalid or too-long prompt gets HTTP 400
  (`invalid_request_error`), so clients do not retry it unchanged. Only "no room yet" is retried, while other
  conversations run, for up to `QE_START_RETRY_S` (60 s).
- A request that produces no tokens fails with an error instead of waiting forever.
- Flash-Next's output guard fails a request whose forward produced non-finite values, instead of emitting garbage;
  the server goes on.
- Flash-Next keeps VRAM for its CUDA graphs (`FLASH_GRAPH_MB`, 256 MB per card, next to the expert cache) and frees
  its least recently used graphs to stay within it, prompt-processing graphs first (`/api/stats` -> `graphs` shows the
  use per card). A prompt-shaped graph is kept only when its shape came back within the last 64 prompt steps, and it
  never pushes out the decode graphs (it runs without a graph instead), so many different prompt lengths do not slow
  decoding down. If VRAM still runs short for a new graph, it gives up a small
  reserve, then drops cached graphs, then runs the step without a graph; it does not stop.
- Both engines keep serving when another process (a browser, the desktop, another application) takes GPU memory
  while they run, and take the memory back once it is free again (log lines `VRAM pressure` / `VRAM free again`):
  Flash-Next moves its least-used resident expert groups to host memory (they are read from there, as the other
  host-resident experts; with the pipeline of `FLASH_PAR=pp_prefill`, between pipelined prompts); 27B shrinks its KV
  cache (idle conversations go to the host-RAM archive; a reply that needs more room pauses with its KV in the archive,
  or waits, until the memory is free again; after `QE_PAUSE_TIMEOUT_S` (900 s) of waiting for memory another process
  holds, it ends with an error that says so; a queued request that fits nowhere with nothing running ends after
  `QE_NO_ROOM_S` (10 s) with such an error, and the server keeps serving). `FLASH_VRAM_LOW_MB`,
  `FLASH_VRAM_TARGET_MB`, `QE_VRAM_LOW_MB`, `QE_VRAM_TARGET_MB` set the marks, `FLASH_VRAM_ADAPT=0` / `QE_VRAM_ADAPT=0`
  turn it off. Measured: an application taking 1, 2 and 4 GB of one card during four streams and a 120K-token prompt,
  then freeing it (Flash-Next on 2 cards, also in the pipeline mode; 27B on 3 cards): every request served, speed back
  after the memory was freed.
- 27B and Flash-Next: a start that finds the GPUs still held by a previous engine waits up to `QE_GPU_LOCK_WAIT_S`
  (60 s), then names the processes holding them. The other engines take no GPU lock.

**Logs.**

- The whole output (engine messages and tracebacks too) is appended to `~/.cache/qengine/logs/<served-name>.log` with
  timestamps, survives crashes and restarts, and ends with the exit status and the kernel's lines about it (OOM kill,
  GPU Xid). Rotated at 64 MB (`--log-mb`); `--log-file` moves it, `--log-file ''` turns it off.
- After every request: one line in vLLM's `SpecDecoding metrics` format and llama.cpp's `prompt eval time` /
  `eval time` format (`QE_STATS_LOG=0` turns them off).
- `WARNING empty reply`: a generated turn with no text and no tool call, with the slot, conversation, finish reason,
  thinking tokens, the `max_tokens` / budget that applied, whether `</think>` came, the last thinking tokens, and
  whether a retry follows; one more line reports the retry's outcome.
- `history: conversation N: the client's history has N assistant turn(s) with reasoning only; left out of the prompt`
  (Flash-Next): once per conversation and again only when the count grows. Such turns make the model end later turns
  empty, so they are left out (`--drop-empty-turns 0` keeps them). This is not an error.
- `/api/stats` counts thinking budgets, forced closes, empty replies and retries, history drops, and shows the parallel
  mode and the CUDA-graph memory per card.

**Monitoring.** Live dashboard and Prometheus metrics on `http://127.0.0.1:9393/` (`--metrics-port`), terminal
monitor `server/qetop.py`. Both show the running version and build in the header (e.g. `QEngine 1.3.6 (build abc1234)`).

### Both models: context and the KV archive

- `--max-ctx` (default 262,144) is the context of one conversation; the KV pool of all conversations together is
  sized from free VRAM. The served context is never larger than the pool: when the pool is smaller than `--max-ctx`,
  the server still starts, serves at most the pool's length (`max_model_len` in `/v1/models`, the length check) and
  says so in one start-up line (27B on 2 cards with `--max-ctx 1M`, a log of 1.3.5.4): `context: requested 1048576 tokens, KV pool 659456 tokens, served 659456 tokens per conversation`.
- Conversations that leave VRAM go to a **host-RAM archive** (`FLASH_KV_ARCHIVE_GB` 48, `QE_KV_ARCHIVE_GB` 32; smaller
  on machines with less RAM, see [Host memory](hardware.md#host-memory-flash-next)): returning to a 50K-token conversation is a
  copy, not a recompute.
- **Restore points at message starts** (Flash-Next): the model carries a recurrent
  state that cannot be cut back to an arbitrary token, so a conversation resumes only where a state was saved. Each
  prompt now also keeps the state at the start of its last two messages. An agent client that appends a short
  message to every request and drops it on the next one (a runtime-context or reminder message), or that drops the
  reasoning of its last reply, resumes at that message instead of the end of the system prompt: on a 125K-token
  Flash-Next conversation 15.5-16.3 s -> 0.4-0.6 s per turn (from the second such turn; the first pays the one-time
  archive copy). Inkling-2-Mini gets the same in a later release. No state is planned when the last
  message starts within 512 tokens of the point the prompt resumed from, so an append-only agent loop is unchanged;
  each state takes about 0.12 GB of host RAM. A changed tool list still recomputes the whole prompt (the chat template
  puts the tools at its start). Flash-Next on 2 cards (more than 10 GB of host-backed
  experts per card, so every prefill batch copies them again) keeps these states off unless `FLASH_CKPT_TAIL_SNAP=1`
  (a tail turn of a 67K conversation 10.3 s -> 0.95 s, a cold multi-message prompt about 0.6 s slower). `FLASH_CKPT_TAIL=0`: off.
- **Pinned system prompts**: the KV and state at the end of each agent kind's system prompt and tools are kept apart
  from the archive (`FLASH_PREFIX_GB` / `QE_PREFIX_GB`, 8 GB), so a new agent of that kind starts after its system
  prompt even when the archive is full.
- There is no disk cache by default. With `--disk-cache DIR` (both models, 200 GB at most, `--disk-cache-gb`) conversations
  are also written to files and survive a restart: a conversation that returns is read back instead of recomputed. Flash-Next
  has it since 1.3.5.3, experimental; details in
  [docs/advanced.md](advanced.md#disk-cache). A failing disk costs only the cache; the server keeps serving. Files are
  written at `--persist-idle` (120 s idle) and at shutdown, so this covers a graceful restart; the opt-in `--disk-eager` also
  writes after every reply (survives a `kill -9`, at the cost of rewriting the file every turn). On the 27B a conversation
  that shares a pinned system prompt is saved together with it, so each file restores on its own.
