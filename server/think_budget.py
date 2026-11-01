"""Hard thinking budget (both models: Qwen3.8-27B and Qwen3.8-Flash-Next share the tokenizer and the template).

The template opens the reply with `<think>\\n` (enable_thinking); the model closes it with `\\n</think>\\n\\n` and
then writes its answer or tool call. Without a limit a model can think until max_tokens, and a max_tokens stop
inside <think> returns a turn with reasoning only: no answer, no tool call (agent harnesses end the task, benchmarks
score zero). This module ends the thinking itself:

  * budget: once the reply has thought `budget` tokens, the server forces a short wrap-up sentence and the closing
    `</think>`, then the model writes its answer as usual;
  * answer tail: when max_tokens would end inside <think>, the same close comes early enough to leave the answer
    a tail of the reply (max_tokens counts thinking + answer, as in other engines); off with QE_THINK_TAIL=0.

The forced tokens go through the engine's grammar tier (server/qe_capi.h TIER 8, the same path as structured
output): the slot is constrained, every forced token is a planned draft and its row mask allows only that token,
so up to min(mtp_k, 7) + 1 forced tokens are emitted per step and they are ordinary verified rows: the KV, the MTP
state and the lookup history stay those of the emitted tokens (qe_constrain undoes a step queued ahead; constrained
slots never run ahead and use no lookup drafts; the engines rebuild the lookup history after them). Then the slot is
unconstrained and decodes freely again.

Request fields (the first one present wins; a negative value: no budget for this request, also over the server
default; 0: no thinking at all, i.e. enable_thinking=false):
  chat_template_kwargs.thinking_budget, thinking_budget (top level), thinking_token_budget (vLLM),
  max_thinking_tokens (SGLang /generate), custom_params.thinking_budget (SGLang chat), reasoning.max_tokens
  (OpenRouter), thinking.budget_tokens (Anthropic {type: enabled}; on both paths), else --thinking-budget /
  QE_THINKING_BUDGET (0 = off).
"""
from qe_env import env_int, env_str

THINK_START, THINK_END = "<think>", "</think>"
# the wrap-up the model reads as its own last thought (a known technique: a plain "out of time, answer now"
# sentence keeps the answer coherent); QE_THINK_WRAP replaces the sentence (the delimiters stay)
WRAP = "Considering the limited time, I have to stop thinking here and give the answer directly, based on my " \
       "reasoning so far."
MIN_ANSWER = 16  # room below this after the full wrap-up: only the bare close `\n</think>\n\n`


def _tail_cfg():
    v = (env_str("QE_THINK_TAIL", "auto") or "auto").strip().lower()
    if v == "auto":
        return -1
    try:
        return max(0, int(v))
    except ValueError:
        print(f"[env] warning: QE_THINK_TAIL={v!r} is not auto or a token count, using auto", flush=True)
        return -1


TAIL = _tail_cfg()


def answer_tail(max_new):
    """Reply tokens kept for the answer when max_tokens would end inside <think> (0: off). auto: 1/8 of the
    reply's max_tokens, at least 64 and at most 2048 tokens."""
    if TAIL >= 0:
        return TAIL
    return max(64, min(2048, max_new // 8))


class BudgetError(ValueError):
    pass


def _int(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise BudgetError(f"{name} must be an integer")
    try:
        f = float(v)
    except ValueError:
        raise BudgetError(f"{name} must be an integer")
    if f != f or abs(f) > 1 << 40:
        raise BudgetError(f"{name} must be an integer")
    # a positive fraction rounds up (0.5 must not turn into 0 = thinking off: review server F9)
    return int(f) if f <= 0 or f == int(f) else int(f) + 1


def request_budget(body, kw, default=0):
    """The request's thinking budget in tokens (None: no budget). Pops thinking_budget from the template kwargs;
    a budget of 0 turns thinking off for the request (kw enable_thinking=false)."""
    cands = [("chat_template_kwargs.thinking_budget", kw.pop("thinking_budget", None))]
    for k in ("thinking_budget", "thinking_token_budget", "max_thinking_tokens"):
        cands.append((k, body.get(k)))
    cp = body.get("custom_params")
    if isinstance(cp, dict):
        cands.append(("custom_params.thinking_budget", cp.get("thinking_budget")))
    rs = body.get("reasoning")
    if isinstance(rs, dict):
        cands.append(("reasoning.max_tokens", rs.get("max_tokens")))
    th = body.get("thinking")
    if isinstance(th, dict) and th.get("type") == "enabled":
        cands.append(("thinking.budget_tokens", th.get("budget_tokens")))
    for name, v in cands:
        if v is None:
            continue
        b = _int(v, name)
        if b < 0:
            return None
        if b == 0:
            kw["enable_thinking"] = False
            return None
        return b
    return default if default and default > 0 else None


class ThinkIds:
    """Token ids of the delimiters and the pre-tokenized forced closes (built once per server)."""

    def __init__(self, tok):
        self.start = tok.token_to_id(THINK_START)
        self.end = tok.token_to_id(THINK_END)
        self.ok = self.start is not None and self.end is not None
        body = env_str("QE_THINK_WRAP", WRAP).strip()
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        # keyed by the newlines put before the wrap-up: none after a paragraph break, one after a newline, else two
        self.wrap = {p: enc(p + body + "\n" + THINK_END + "\n\n") for p in ("", "\n", "\n\n")}
        self.bare = {p: enc(p + THINK_END + "\n\n") for p in ("", "\n", "\n\n")}
        self.wrap_max = max(len(v) for v in self.wrap.values())
        for d in (self.wrap, self.bare):
            for p, ids in d.items():  # the close must be the delimiter token itself, exactly once at the end
                if not self.ok or ids.count(self.end) != 1 or len(ids) < 2 or ids[-2] != self.end:
                    self.ok = False

    def open_at_end(self, ids, im_start):
        """The prompt leaves the reply inside <think> (the generation prompt `<think>\\n`, or a prefilled
        assistant turn that opened one): scans back to the last <|im_start|>."""
        for t in reversed(ids):
            if t == self.end:
                return False
            if t == self.start:
                return True
            if t == im_start:
                return False
        return False


class ThinkControl:
    """Thinking state of one reply and its forced close (driven by Job.feed on the scheduler thread)."""

    def __init__(self, job, budget, ti, can_force):
        self.job, self.budget, self.ti, self.can_force = job, budget, ti, can_force
        self.open = ti.open_at_end(job.ids, job.gen.im_start)
        self.n = 0              # thinking tokens so far (the forced wrap-up included)
        self.forced = None      # why the thinking was closed by the server: "budget" / "max_tokens"
        self.fz, self.fi = None, 0  # forced tokens in flight, how many emitted
        self.n_forced = 0       # forced tokens emitted
        self.failed = None      # the engine emitted something else than a forced token (should not happen)
        # ignore_eos (benchmarks: exactly max_tokens of raw continuation): no answer tail and no hold near it (review
        # server F3); an explicit thinking budget still applies
        self.tail = 0 if getattr(job, "p", {}).get("ignore_eos") else answer_tail(job.max_new)

    def active(self):
        return self.fz is not None

    def track(self, toks):
        """Thinking state over the reply's emitted tokens (in order)."""
        for t in toks:
            if self.open:
                if t == self.ti.end:
                    self.open = False
                else:
                    self.n += 1
            elif t == self.ti.start:  # the model re-opened a think block: it counts against the same budget
                self.open = True

    def after(self, toks):
        """After the job fed a step's tokens (not finished): advance a forced close in flight or start one."""
        j = self.job
        e = j.gen.engine
        if self.fz is not None:
            if not toks:
                # no step of this slot finished (27B: a constrained slot sits out the batched step queued ahead and
                # joins the next one, already launched with the current masks / drafts): the buffers must not be
                # rewritten now, the launched step's asynchronous copies may still read them
                return
            for t in toks:
                if self.fi < len(self.fz) and t == self.fz[self.fi]:
                    self.fi += 1
                    self.n_forced += 1
                else:
                    self.failed = f"token {t} where the forced close has {self.fz[min(self.fi, len(self.fz) - 1)]}"
                    print(f"[think] forced close failed after {self.fi} of {len(self.fz)} tokens: {self.failed}",
                          flush=True)
                    break
            if self.failed or self.fi >= len(self.fz):
                self.fz = None
                e.constrain(j.slot, False)
                return
            self._plan()
            return
        if not self.open or not self.can_force or self.failed:
            return
        room = j.max_new - j.n
        why = None
        if self.budget is not None and self.n >= self.budget:
            why = "budget"
        elif self.tail and room <= self.ti.wrap_max + self.tail:
            why = "max_tokens"
        if why is None:
            return
        text = j.det.tok.decode(j.det.ids[-4:], skip_special_tokens=False) if j.det.ids else ""
        if text.endswith("�"):  # the reply ends inside a multi-byte character: close after it
            return
        p = "" if text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n"  # a blank line before it
        seq = self.ti.wrap[p] if room >= len(self.ti.wrap[p]) + MIN_ANSWER else self.ti.bare[p]
        if room < len(seq) + 1:  # not even the bare close and one answer token fit: max_tokens ends the reply
            return
        self.forced, self.fz, self.fi = why, list(seq), 0
        # masks / drafts first (QFN uploads row 0 inside qe_constrain), then constrain (it clears spec_io and undoes
        # a step of this slot queued ahead), then the drafts again
        self._fill()
        e.constrain(j.slot, True)
        self._plan()

    def _fill(self):
        """Row masks of the next forced tokens (row r: only fz[fi + r]); returns the drafts to verify."""
        j = self.job
        e = j.gen.engine
        rest = self.fz[self.fi:]
        kv = max(0, min(len(rest) - 1, 7, j.gen.args.mtp_k, j.max_new - j.n - 1))
        M = e.mask_buffer(j.slot)
        for r in range(kv + 1):
            t = rest[r]
            M[r].fill(0)
            M[r][t >> 5] = 1 << (t & 31)
        return rest[:kv]

    def _plan(self):
        j = self.job
        e = j.gen.engine
        d = self._fill()
        if j.gen.args.mtp_k == 0:  # QFN without MTP uploads only mask row 0, at qe_constrain: once per token
            e.constrain(j.slot, True)
        io = e.spec_io(j.slot)
        for i, t in enumerate(d):
            io[17 + i] = t
        io[16] = len(d)

    def near(self, margin=64):
        """A forced close may be due within `margin` tokens: the engine should not decode ahead of what the server has
        seen (QFN mode B's persistent pipeline runs up to a few steps past the last returned tokens; a close planned
        from the seen reply must not land after tokens it never saw)."""
        if self.fz is not None or not self.open or not self.can_force or self.failed:
            return False
        j = self.job
        if self.budget is not None and self.n + margin >= self.budget:
            return True
        return bool(self.tail) and j.max_new - j.n <= self.ti.wrap_max + self.tail + margin

    def stats(self):
        return {"tokens": self.n, "budget": self.budget, "forced": self.forced, "forced_tokens": self.n_forced,
                **({"failed": self.failed} if self.failed else {})}


def empty_reply(job, tail_tokens=8):
    """A finished reply with no text and no tool call (reasoning only, or nothing at all) and finish stop / length:
    the fields of its WARNING line (the line itself under "line"), else None. The last thinking tokens tell an
    end-of-turn right after the thinking (the model chose to stop) from a cut-off (max_tokens / a forced close)."""
    if job.finish not in ("stop", "length"):
        return None
    text = job.text_all
    if "<tool_call>" in text:
        return None
    tc = job.tc
    if THINK_END in text:
        answer = text.rsplit(THINK_END, 1)[1]
    else:
        answer = "" if tc is not None else text  # the reply opened inside <think> and never closed it
    if answer.strip():
        return None
    g = job.gen
    ids = job.det.ids
    end = g.think_ids.end
    k = len(ids) - 1 - ids[::-1].index(end) if end is not None and end in ids else len(ids)
    tail = g.tok.decode(ids[max(0, k - tail_tokens):k], skip_special_tokens=False)
    tail = repr(tail if len(tail) <= 120 else "..." + tail[-117:])
    info = getattr(job.ids, "agent", None)
    conv = getattr(getattr(info, "conv", None), "id", None)
    sess = (getattr(getattr(info, "session", None), "key", "") or "")[2:10]
    forced = tc.forced if tc is not None else None
    budget = tc.budget if tc is not None else None
    closed = end is not None and end in ids
    mt = job.p.get("max_tokens")
    out = {"slot": job.slot, "conversation": conv, "session": sess, "finish": job.finish, "forced": forced,
           "thinking_tokens": tc.n if tc is not None else 0, "completion_tokens": job.n, "max_tokens": job.max_new,
           "max_tokens_from": "request" if mt else "context", "thinking_budget": budget,
           "think_closed": closed, "thinking_tail": tail}
    kind = "reasoning only" if out["thinking_tokens"] else "empty"
    fin = job.finish + (f", forced close ({forced})" if forced else "")
    lim = f"max_tokens {job.max_new}{'' if mt else ' (context room)'}, thinking budget {budget or 'none'}"
    if tc is not None and tc.tail and not forced:
        lim += f", answer tail {tc.tail}"
    out["line"] = (f"WARNING empty reply: slot {job.slot}, conversation {conv if conv is not None else '-'}"
                   f"{f' (session {sess})' if sess else ''}: {kind}, no text and no tool call; finish {fin}; "
                   f"{out['thinking_tokens']} thinking tokens of {job.n} ({lim}); </think> "
                   f"{'yes' if closed else 'no'}; thinking ends {tail}")
    k, n = job.p.get("empty_retry") or (0, 0)  # server/app.py complete(): --empty-retry of non-streamed replies
    if n and job.finish == "stop":
        out["retry"] = (k + 1, n) if k < n else None
        out["line"] += f"; retry {k + 1}/{n} follows" if k < n else f"; no retry left ({n}/{n}), returned as is"
    return out
