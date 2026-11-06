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

The server default (BudgetMap) is either one number for every request or a budget per reasoning effort
(`low:4096,medium:16384,xhigh:32768,default:8192`, issue #4). A request that sends no budget gets the entry of the
effort its prompt is rendered with (kw["reasoning_effort"] after template_kwargs and the client aliases of
EffortRules.client_effort: Qwen3.8 max = xhigh, minimal = low), normalised the way the served chat template renders
it (EffortRules, probed from the template at start-up: Qwen3.8 high = xhigh, others = medium; GLM-5.3 low / high,
others = max), else the `default:` / `*:` entry, else no budget.
"""
from qe_env import env_int, env_str
from content_markers import OUT_MARK

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


# effort names a budget map may use (the union of what the templates render)
EFFORT_NAMES = ("low", "medium", "high", "xhigh", "max")


class EffortRules:
    """How a chat template turns the reasoning_effort variable into the effort it renders: `unset` when the
    variable is not passed, a value of `levels` as is, an `alias` to its level, anything else `fallback`. The
    comparison is exact (no lower-casing), as in the templates. `client`: effort names of other APIs that the server
    rewrites before rendering (Anthropic / vLLM `max`, `minimal`), so the template and the budget see the level the
    client meant instead of the template's fallback."""

    def __init__(self, name, levels, unset, fallback, alias=None, client=None):
        self.name, self.levels, self.unset, self.fallback = name, tuple(levels), unset, fallback
        self.alias, self.client = dict(alias or {}), dict(client or {})

    def client_effort(self, kw):
        """Rewrites kw["reasoning_effort"] by the client aliases (before the prompt is rendered)."""
        e = kw.get("reasoning_effort")
        if isinstance(e, str) and e in self.client:
            kw["reasoning_effort"] = self.client[e]
        return kw

    def level(self, effort):
        if effort is None:
            return self.unset
        if not isinstance(effort, str):
            return self.fallback
        if effort in self.levels:
            return effort
        return self.alias.get(effort, self.fallback)

    def describe(self):
        al = "".join(f", {k} = {v}" for k, v in {**self.client, **self.alias}.items())
        return f"{self.name}: {' / '.join(self.levels)}{al}, unset = {self.unset}, others = {self.fallback or 'as sent'}"


# qwen38_syssafe.jinja (the server default for Qwen3.8-27B and -Flash-Next; the models' own chat_template.jinja is
# the same without the 'high' alias and raises on other values, so those requests fail before any budget matters)
# (client aliases: Anthropic effort max / vLLM-SGLang buckets; without them max and minimal would render as medium)
QWEN_RULES = EffortRules("qwen3.8", ("low", "medium", "xhigh"), "xhigh", "medium", {"high": "xhigh"},
                         {"max": "xhigh", "minimal": "low"})
# GLM-5.3-Flash chat_template.jinja: `reasoning_effort in ['low', 'high'] else 'max'` ("Reasoning Effort: Max")
GLM_RULES = EffortRules("glm-5.3", ("low", "high", "max"), "max", "max", client={"minimal": "low"})
# a template whose rendering fits neither: the effort as sent; a request with no effort uses the xhigh entry (the
# Qwen3.8 default), so a map with an xhigh entry still bounds clients that send none
RAW_RULES = EffortRules("not recognised (efforts as sent)", EFFORT_NAMES, "xhigh", None)
PROBE_EFFORTS = (None, "low", "medium", "high", "xhigh", "max", "minimal", "none", "__qe_probe__")


def probe_rules(render, log=None):
    """The EffortRules of a template: `render(effort)` renders a probe prompt with thinking on (None: no
    reasoning_effort variable). A rule set fits when the efforts it maps to one level render the same prompt and
    efforts of different levels render different ones (values the template raises on are left out). Render errors
    go to `log` in one line: template errors (the stock Qwen template refuses unknown efforts) only when no rule set
    fits, any other error always."""
    out, errs = {}, []
    for e in PROBE_EFFORTS:
        try:
            out[e] = render(e)
        except Exception as ex:  # noqa: BLE001 (the stock Qwen template raises on unknown efforts)
            errs.append((e, ex))
    rules = _fit(out)
    odd = [(e, ex) for e, ex in errs if not type(ex).__module__.startswith("jinja2")]
    if log is not None and (odd or (errs and rules is RAW_RULES)):
        e, ex = (odd or errs)[0]
        log(f"[server] warning: reasoning effort probe of the chat template: {len(errs)} of {len(PROBE_EFFORTS)} "
            f"renders failed (first: effort {e!r}: {type(ex).__name__}: {str(ex)[:160]})")
    return rules


def _fit(out):
    for rules in (QWEN_RULES, GLM_RULES):
        seen, ok = {}, True
        for e, text in out.items():
            lv = rules.level(e)
            if seen.setdefault(lv, text) != text:
                ok = False
        if ok and len(set(seen.values())) == len(seen) and len(seen) > 1:
            return rules
    return RAW_RULES


class BudgetMap:
    """The server's default thinking budget: one number (`default`) or a budget per effort level (`by_effort`),
    parsed from --thinking-budget / QE_THINKING_BUDGET. 0 / unset / a negative number: no budget. An entry of 0
    or below means no budget for that effort (the default entry does not apply to it)."""

    def __init__(self, by_effort=None, default=0, plain=True):
        self.by_effort, self.default, self.plain = dict(by_effort or {}), max(0, int(default or 0)), plain

    @classmethod
    def parse(cls, v):
        if isinstance(v, BudgetMap):
            return v
        if v is None:
            return cls()
        if isinstance(v, bool):
            raise ValueError(f"thinking budget {v!r}: not a token count")
        if isinstance(v, int):
            return cls(default=v)
        s = str(v).strip()
        if not s:
            return cls()
        if ":" not in s:
            try:
                return cls(default=int(s, 10))
            except ValueError:
                raise ValueError(f"thinking budget {s!r}: expected a token count (e.g. 8192) or a map like "
                                 f"low:4096,medium:16384,xhigh:32768[,default:8192]") from None
        by, dflt = {}, None
        for part in s.split(","):
            k, sep, n = part.partition(":")
            k = k.strip().lower()
            if not sep or not k:
                raise ValueError(f"thinking budget map {s!r}: entry {part.strip()!r} is not <effort>:<tokens>")
            if k not in EFFORT_NAMES + ("default", "*"):
                raise ValueError(f"thinking budget map {s!r}: unknown effort {k!r} (known: "
                                 f"{', '.join(EFFORT_NAMES)}, default or *)")
            try:
                t = int(n.strip(), 10)
            except ValueError:
                raise ValueError(f"thinking budget map {s!r}: {k}: {n.strip()!r} is not a token count") from None
            t = max(0, t)  # negative = no budget for that effort, as 0 (and as a negative plain number)
            key = "default" if k in ("default", "*") else k
            if key in by or (key == "default" and dflt is not None):
                raise ValueError(f"thinking budget map {s!r}: {key} given twice")
            if key == "default":
                dflt = t
            else:
                by[key] = t
        return cls(by, dflt or 0, plain=False)

    def __bool__(self):
        return self.default > 0 or any(v > 0 for v in self.by_effort.values())

    def resolve(self, effort, rules):
        """(budget or None, source) for a request that sends no budget of its own."""
        if self.by_effort:
            lv = (rules or QWEN_RULES).level(effort)
            if lv in self.by_effort:
                b = self.by_effort[lv]
                return (b, f"effort:{lv}") if b > 0 else (None, f"effort:{lv}")
        if self.default > 0:
            return self.default, "default"
        return None, "none"

    def config(self):
        """For /api/stats: the plain number, or the map ({effort: tokens, "default": tokens})."""
        if self.plain:
            return self.default
        return {**self.by_effort, **({"default": self.default} if self.default else {})}

    def describe(self):
        if self.plain:
            return f"{self.default} tokens" if self.default else "none"
        ent = ", ".join(f"{k} {v or 'none'}" for k, v in self.by_effort.items())
        return f"{ent}{', ' if ent else ''}default {self.default or 'none'}"

    def unreachable(self, rules):
        """Map entries the template never renders (they can never apply)."""
        return [k for k in self.by_effort if k not in rules.levels] if rules is not RAW_RULES else []


def budget_banner(bm, rules):
    """Start-up lines of the server default budget (none when there is none; the plain number as in 1.3.4)."""
    if bm.plain:
        return [f"thinking budget: {bm.default} tokens by default (a request's budget wins)"] if bm else []
    out = [f"thinking budget by effort: {bm.describe()} (template efforts {rules.describe()}; a request's budget "
           f"wins)"]
    if rules is RAW_RULES:
        out.append("[server] warning: thinking budget map: the chat template's reasoning efforts were not recognised "
                   "(neither Qwen3.8 nor GLM-5.3 rendering); efforts are matched as sent and a request with no effort "
                   "uses the xhigh entry")
    dead = bm.unreachable(rules)
    if dead:
        out.append(f"[server] warning: thinking budget map: {', '.join(dead)} never applies (the template renders "
                   f"{rules.describe()})")
    return out


def budget_source(body, kw, default=0, rules=None):
    """(budget or None, source): the request's own budget ("request"), else the server default for the effort the
    prompt is rendered with ("effort:<level>" / "default"), else ("none"). Pops thinking_budget from the template
    kwargs; a budget of 0 turns thinking off for the request (kw enable_thinking=false)."""
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
            return None, "off"
        if b == 0:
            kw["enable_thinking"] = False
            return None, "request"
        return b, "request"
    if kw.get("enable_thinking") is False:  # thinking off (request, --think 0): no server budget (review L3)
        return None, "none"
    return BudgetMap.parse(default).resolve(kw.get("reasoning_effort"), rules)


def request_budget(body, kw, default=0, rules=None):
    """The request's thinking budget in tokens (None: no budget); see budget_source."""
    return budget_source(body, kw, default, rules)[0]


class ThinkIds:
    """Token ids of the delimiters and the pre-tokenized forced closes (built once per server)."""

    def __init__(self, tok):
        self.start = tok.token_to_id(THINK_START)
        self.end = tok.token_to_id(THINK_END)
        self.tok, self.spaces = tok, {}
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

    def is_space(self, t):
        """The token is whitespace only: does not begin the answer (app.Parser, fresh)."""
        v = self.spaces.get(t)
        if v is None:
            v = self.spaces[t] = not self.tok.decode([t], skip_special_tokens=False).strip()
        return v

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

    def __init__(self, job, budget, ti, can_force, src=None):
        self.job, self.budget, self.ti, self.can_force = job, budget, ti, can_force
        self.src = src or ("request" if budget is not None else "none")  # where the budget came from (budget_source)
        self.open = ti.open_at_end(job.ids, job.gen.im_start)
        self.fresh = True       # no answer text yet: as app.Parser, only here a real <think> opens a block
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
            elif self.fresh and t == self.ti.start:  # the model re-opened a think block: it counts against the same budget
                self.open = True
            elif self.fresh and not self.ti.is_space(t):  # the answer began: a later <think> is text (a close there would be too)
                self.fresh = False

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
        return {"tokens": self.n, "budget": self.budget, "budget_from": self.src, "forced": self.forced,
                "forced_tokens": self.n_forced,
                **({"failed": self.failed} if self.failed else {})}


def empty_reply(job, tail_tokens=8):
    """A finished reply with no text and no tool call (reasoning only, or nothing at all) and finish stop / length:
    the fields of its WARNING line (the line itself under "line"), else None. The last thinking tokens tell an
    end-of-turn right after the thinking (the model chose to stop) from a cut-off (max_tokens / a forced close)."""
    if job.finish not in ("stop", "length"):
        return None
    # the real tag tokens (marked in job.text_mk, server/content_markers.py), not the same strings quoted as text
    text = getattr(job, "text_mk", "") or ""
    call, end_tag = OUT_MARK + "tool_call>", OUT_MARK + "/think>"
    if not text and job.text_all:  # a job without the marked text (tests): the plain tags
        text, call, end_tag = job.text_all, "<tool_call>", THINK_END
    if call in text:
        return None
    tc = job.tc
    if end_tag in text:
        answer = text.rsplit(end_tag, 1)[1]
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
