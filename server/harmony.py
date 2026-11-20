"""[srv-gptoss] gpt-oss (harmony format) in the shared server: history preparation for the model's own chat template,
the streaming reply parser and the tool-call parser.

Harmony replies are messages `<|start|>assistant<|channel|>CH[ to=functions.NAME][ <|constrain|>json]<|message|>BODY`
ended by `<|end|>` (another message follows), `<|call|>` (a tool call: the end of the reply) or `<|return|>` (the end
of the reply). The prompt ends with `<|start|>assistant`, so the reply begins inside a header. Channels: analysis =
reasoning, final = the answer, commentary with a recipient = a tool call (its body = the JSON arguments), commentary
without one = preamble text (content). The recipient may sit before the channel (`assistant to=functions.x<|channel|>`)
or after it (`<|channel|>commentary to=functions.x`).

Only the real control tokens count: the detokenizer marks them (their `<` -> OUT_MARK, server/content_markers.py, the
TAGS below added to the marks for this family), so the same strings written as text by the model are text.
"""
import json
import re

from content_markers import mark, unmark

TAGS = ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|constrain|>", "<|call|>", "<|return|>")
START, END, MESSAGE, CHANNEL, CONSTRAIN, CALL, RETURN = (mark(t) for t in TAGS)
_RECIPIENT = re.compile(r"to=([^\s<﷑]+)")


def _partial(s):
    """length of the longest suffix of s that is a proper prefix of an end tag (a tag split across two feeds)"""
    for n in range(min(len(s), len(RETURN) - 1), 0, -1):
        if any(t.startswith(s[-n:]) for t in (END, CALL, RETURN)):
            return n
    return 0


def marks(tok):
    """{token id: marked text} of the harmony control tokens (for Detok)."""
    return {i: mark(t) for t in TAGS for i in [tok.token_to_id(t)] if i is not None}


def _text(c):
    """[gptoss-rel1-parts] message content as one string: the template concatenates content with str, so a list of parts
    (OpenAI text / image parts, Anthropic blocks after conversion) would raise a TypeError (HTTP 500). Text parts are
    joined with a blank line; other parts (an image the server already replaced by its text note) keep their text."""
    if c is None or isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append(p["text"])
            elif isinstance(p, dict) and p.get("type") in ("image", "image_url"):
                out.append("[image omitted: this server runs without vision]")
            elif p is not None:   # (review rel1 L15: another part, e.g. a JSON tool result: its JSON, not nothing)
                out.append(json.dumps(p, ensure_ascii=False, default=str))
        return "\n\n".join(x for x in out if x)
    return str(c)


def _split_calls(msgs, i):
    """[gptoss-rel1-calls] msgs[i] is an assistant turn with several tool calls (another model's history: harmony has one
    call per message, the template renders only the first and labels every result with its name). Returns (pieces, j):
    one assistant call + its result per call, in call order (results matched by tool_call_id, else by position), and
    the index after the turn's result block."""
    m = msgs[i]
    calls = m["tool_calls"]
    j = i + 1
    while j < len(msgs) and msgs[j].get("role") == "tool":
        j += 1
    res = list(msgs[i + 1:j])
    ids = [c.get("id") for c in calls]
    taken = [None] * len(calls)
    rest = []
    for r in res:
        k = ids.index(r.get("tool_call_id")) if r.get("tool_call_id") in ids else -1
        if k >= 0 and taken[k] is None:
            taken[k] = r
        else:
            rest.append(r)
    for k in range(len(calls)):   # results without a matching id: by position
        if taken[k] is None and rest:
            taken[k] = rest.pop(0)
    out = []
    for k, c in enumerate(calls):
        a = {kk: v for kk, v in m.items() if kk not in ("tool_calls", "content", "thinking", "reasoning_content",
                                                          "reasoning")} if k else dict(m)
        a["tool_calls"] = [c]
        if k:
            a["content"] = ""
        out.append(a)
        if taken[k] is not None:
            out.append(taken[k])
    out.extend(rest)   # (more results than calls: after the last call, as the client sent them)
    return out, j


def prepare(messages):
    """OpenAI-shaped history -> what the gpt-oss template reads: assistant reasoning in `thinking` (from
    reasoning_content / reasoning), tool-call arguments as objects (the template writes them with tojson; an OpenAI
    client sends a JSON string), one of content / thinking on a tool-call turn (the template refuses both: the
    thinking goes), tool results' content as text. [gptoss-rel1] Also: content parts joined into text; the leading
    system / developer messages merged into one (the template reads only messages[0]); a later system / developer
    message (the template has no place for it and dropped it) becomes a user message that says what it is; a turn with
    several tool calls split into one call + its result per call."""
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if "content" in m:
            m["content"] = _text(m["content"])
    k = 0
    while k < len(msgs) and msgs[k].get("role") in ("system", "developer"):
        k += 1
    if k > 1:
        parts = [(msgs[x].get("content") or "").strip() for x in range(k)]
        msgs = [{"role": msgs[0]["role"], "content": "\n\n".join(p for p in parts if p)}] + msgs[k:]
        k = 1
    for x in range(k, len(msgs)):
        if msgs[x].get("role") in ("system", "developer"):
            msgs[x] = {"role": "user", "content": f"[{msgs[x]['role']} message] " + (msgs[x].get("content") or "")}
    flat, i = [], 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list) and len(m["tool_calls"]) > 1:
            pieces, i = _split_calls(msgs, i)
            flat.extend(pieces)
        else:
            flat.append(m)
            i += 1
    out = []
    for m in flat:
        m = dict(m)
        if m.get("role") == "assistant":
            r = m.pop("reasoning_content", None)
            r = m.pop("reasoning", None) if r is None else r
            if r and "thinking" not in m:
                m["thinking"] = r
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    c = dict(c)
                    f = dict(c.get("function") or {})
                    a = f.get("arguments")
                    if isinstance(a, str):
                        try:
                            f["arguments"] = json.loads(a) if a.strip() else {}
                        except ValueError:
                            pass   # (kept as the client's text)
                    c["function"] = f
                    calls.append(c)
                m["tool_calls"] = calls
                if m.get("content") and m.get("thinking"):
                    m.pop("thinking")
            if m.get("content") is None:
                m["content"] = ""
        out.append(m)
    return out


class HarmonyParser:
    """Streaming split of a harmony reply into reasoning / content / tool calls (app.Parser's interface: feed(text,
    final, promote) -> [(kind, text)], tool_text). tool_text keeps the tool messages as
    `to=functions.NAME<|message|>ARGS<|call|>` with real-token marks (parse_calls reads them)."""

    def __init__(self, thinking=True):
        self.state = "header"   # header: text up to <|message|>; body: the message text; wait: after <|end|>
        self.head = ""
        self.kind = None        # body kind: reasoning / content / tool
        self.buf = ""
        self.tool_text = ""
        self.promoted = False
        self.mode = "reasoning"   # (app.Parser's attribute: the kind of the current body)
        self.bad = 0              # headers that did not parse (degraded to content)

    def _begin(self):
        h = self.head
        rec = _RECIPIENT.search(unmark(h))
        ch = ""
        k = h.find(CHANNEL)
        if k >= 0:
            ch = re.split(r"[\s﷑]", h[k + len(CHANNEL):].lstrip(), maxsplit=1)[0]
        if rec and rec.group(1).startswith("functions."):
            self.kind = "tool"
            self.tool_text += "to=" + rec.group(1) + MESSAGE
        elif ch == "analysis":
            self.kind = "reasoning"
        elif ch in ("final", "commentary"):
            self.kind = "content"
        else:   # (a header without a known channel: its text is the answer)
            self.kind = "content"
            self.bad += 1
        self.mode = self.kind
        self.head = ""

    def feed(self, text, final=False, promote=False):
        out = []
        self.buf += text
        while self.buf:
            if self.state in ("header", "wait"):
                if self.state == "wait":
                    k = self.buf.find(START)
                    c = self.buf.find(CHANNEL)
                    if c >= 0 and (k < 0 or c < k) and not self.buf[:c].strip():
                        # a header without its <|start|> (review L9): the header, not text
                        self.buf = self.buf[c:]
                        self.state = "header"
                        continue
                    if k < 0:   # text outside a message (a model that skipped <|start|>): content
                        if final or len(self.buf) > 64:
                            out.append(("content", self.buf))
                            self.buf = ""
                        break
                    if k:
                        out.append(("content", self.buf[:k]))
                    self.buf = self.buf[k + len(START):]
                    self.state = "header"
                k = self.buf.find(MESSAGE)
                if k < 0:   # (the header stays in buf until its <|message|>)
                    if final and self.buf.strip():   # a reply cut inside a header: its text, unless it is header syntax
                        if CHANNEL not in self.buf and CONSTRAIN not in self.buf:   # (review L9)
                            out.append(("content", self.buf))
                        self.buf = ""
                    break
                self.head += self.buf[:k]
                self.buf = self.buf[k + len(MESSAGE):]
                self._begin()
                self.state = "body"
                continue
            ends = [k for k in (self.buf.find(t) for t in (END, CALL, RETURN)) if k >= 0]
            k = min(ends) if ends else -1
            hold = 0 if (k >= 0 or final) else _partial(self.buf)
            body = self.buf[:len(self.buf) - hold] if k < 0 else self.buf[:k]
            if body:
                if self.kind == "tool":
                    self.tool_text += body
                else:
                    out.append((self.kind, body))
            if k < 0:
                self.buf = self.buf[len(self.buf) - hold:] if hold else ""
                break
            tag = next(t for t in (END, CALL, RETURN) if self.buf.startswith(t, k))
            if self.kind == "tool":
                self.tool_text += CALL
            self.buf = self.buf[k + len(tag):]
            self.state = "wait"
        if final and self.kind == "tool" and self.state == "body" and not self.tool_text.endswith(CALL):
            self.tool_text += CALL   # (the reply ended at <|call|>, an end token the parser never sees)
        return [(k, unmark(t)) for k, t in out if t]


def parse_calls(tool_text, tools=None):
    """tool_text of HarmonyParser -> [{"name", "arguments": dict}]; arguments that are not a JSON object stay as
    {"input": text} would mislead a client, so a call whose body does not parse is dropped (its text then reaches the
    client as content, app.tool_calls_or_text)."""
    calls = []
    for m in re.finditer(r"to=functions\.([^\s<﷑]+)" + re.escape(MESSAGE) + r"(.*?)" + re.escape(CALL), tool_text, re.S):
        name, body = m.group(1), unmark(m.group(2)).strip()
        try:
            args = json.loads(body) if body else {}
        except ValueError:
            continue
        if isinstance(args, dict):
            calls.append({"name": name, "arguments": args})
    return calls


# ---------------------------------------------------------------- grammar tier (structured output, tool_choice)
# The reply begins inside a header (the prompt ends with `<|start|>assistant`). Constraints work on token automata with
# the llguidance matcher interface grammar.Constraint drives (consume_token, rollback, compute_ff_tokens,
# unsafe_compute_mask_ptr, is_stopped, is_accepting, is_error, get_error).

def _mask_of(tokens, words):
    import numpy as np
    m = np.zeros(words, np.uint32)
    for t in tokens:
        if t is not None and 0 <= t < 32 * words:
            m[t >> 5] |= np.uint32(1 << (t & 31))
    return m


def _all_but(tokens, words):
    import numpy as np
    m = np.full(words, 0xFFFFFFFF, np.uint32)
    for t in tokens:
        if t is not None and 0 <= t < 32 * words:
            m[t >> 5] &= np.uint32(~(1 << (t & 31)) & 0xFFFFFFFF)
    return m


class Forced:
    """Structured output on gpt-oss: the header `<|channel|>final<|message|>` is forced first (no reasoning, as the
    server runs every structured request), then the inner matcher (the JSON schema) owns the message body."""

    def __init__(self, pre, inner):
        self.pre, self.inner, self.i, self.hist = list(pre), inner, 0, []
        self.error = None

    def consume_token(self, t):
        if self.i < len(self.pre):
            if int(t) != self.pre[self.i]:
                self.error = f"structured output: token {t} where the final-channel header continues"
                return False
            self.i += 1
            self.hist.append(0)
            return True
        if not self.inner.consume_token(t):
            return False
        self.hist.append(1)
        return True

    def rollback(self, n):
        k = 0
        for _ in range(n):
            if self.hist.pop():
                k += 1
            else:
                self.i -= 1
        if k:
            self.inner.rollback(k)
        self.error = None

    def compute_ff_tokens(self):
        if self.i < len(self.pre):
            return self.pre[self.i:]
        return self.inner.compute_ff_tokens()

    def unsafe_compute_mask_ptr(self, ptr, nbytes):
        if self.i < len(self.pre):
            import ctypes
            m = _mask_of([self.pre[self.i]], nbytes // 4)
            ctypes.memmove(ptr, m.ctypes.data, nbytes)
            return
        self.inner.unsafe_compute_mask_ptr(ptr, nbytes)

    def is_stopped(self):
        return self.i >= len(self.pre) and self.inner.is_stopped()

    def is_accepting(self):
        return self.i >= len(self.pre) and self.inner.is_accepting()

    def is_error(self):
        return self.error is not None or self.inner.is_error()

    def get_error(self):
        return self.error or self.inner.get_error()


class ReasonFirst(Forced):
    """Structured output on gpt-oss with its reasoning: an optional analysis message first (free body; it may not end
    the reply), then the final-channel header, then the inner matcher. `pre` = the first header's tokens to the
    channel name branch: [<|channel|>] then analysis / final."""

    def __init__(self, tok, inner):
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        self.a_hdr, self.f_hdr = enc("<|channel|>analysis<|message|>"), enc("<|channel|>final<|message|>")
        self.f_hdr2 = enc("<|start|>assistant<|channel|>final<|message|>")
        self.end = tok.token_to_id("<|end|>")
        # (<|endoftext|> too: a structured reply must not end inside its reasoning; review L11)
        self.block = {i for t in ("<|call|>", "<|return|>", "<|start|>", "<|channel|>", "<|endoftext|>") for i in [tok.token_to_id(t)]
                      if i is not None}
        # phase: "hdr" (which first header, by prefix), "think" (analysis body), then Forced's own pre / inner
        self.phase, self.got = "hdr", []
        super().__init__([], inner)
        self.ph_hist = []

    def _hdr_next(self):
        k = len(self.got)
        return {h[k] for h in (self.a_hdr, self.f_hdr) if len(h) > k and h[:k] == self.got}

    def consume_token(self, t):
        t = int(t)
        if self.phase == "hdr":
            if t not in self._hdr_next():
                self.error = f"structured output: token {t} not a header token"
                return False
            self.ph_hist.append(("hdr", list(self.got), self.i, list(self.pre)))
            self.got.append(t)
            if self.got == self.a_hdr:
                self.phase = "think"
            elif self.got == self.f_hdr:
                self.phase, self.pre, self.i = "body", [], 0
            return True
        if self.phase == "think":
            if t in self.block:
                self.error = f"structured output: token {t} inside the reasoning"
                return False
            self.ph_hist.append(("think", list(self.got), self.i, list(self.pre)))
            if t == self.end:
                self.phase, self.pre, self.i = "body", list(self.f_hdr2), 0
            return True
        self.ph_hist.append(("body", list(self.got), self.i, list(self.pre)))
        if not Forced.consume_token(self, t):
            self.ph_hist.pop()
            return False
        return True

    def rollback(self, n):
        for _ in range(n):
            ph, got, i, pre = self.ph_hist.pop()
            if ph == "body" and self.phase == "body" and self.hist:
                Forced.rollback(self, 1)
            self.phase, self.got, self.i, self.pre = ph, got, i, pre
            if ph == "body":
                self.phase = "body"
        self.error = None

    def compute_ff_tokens(self):
        if self.phase == "hdr":
            nx = self._hdr_next()
            return list(nx) if len(nx) == 1 else []
        if self.phase == "think":
            return []
        return Forced.compute_ff_tokens(self)

    def unsafe_compute_mask_ptr(self, ptr, nbytes):
        import ctypes
        if self.phase == "hdr":
            m = _mask_of(self._hdr_next(), nbytes // 4)
        elif self.phase == "think":
            m = _all_but(self.block, nbytes // 4)
        else:
            return Forced.unsafe_compute_mask_ptr(self, ptr, nbytes)
        ctypes.memmove(ptr, m.ctypes.data, nbytes)

    def is_stopped(self):
        return self.phase == "body" and Forced.is_stopped(self)

    def is_accepting(self):
        return self.phase == "body" and Forced.is_accepting(self)


def final_header(tok):
    return tok.encode("<|channel|>final<|message|>", add_special_tokens=False).ids


class HarmonyTier:
    """tool_choice on gpt-oss (server/tool_choice.py's Tier interface). The reply is a series of messages; the matcher
    walks their headers through a trie of the allowed ones and lets the bodies decode freely:
      none        headers: analysis or final only (no commentary: no call, no preamble); `<|call|>` never.
      required /  first an analysis (optional), its body may not end the reply (`<|call|>` / `<|return|>` masked),
      named       then only a call header `<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json
                  <|message|>` (the form the model writes); once complete the slot is released (the arguments decode
                  freely)."""

    def __init__(self, tok, think_ids, eos, im_start):
        self.tok = tok
        ids = {t: tok.token_to_id(t) for t in TAGS}
        self.start, self.end, self.message = ids["<|start|>"], ids["<|end|>"], ids["<|message|>"]
        self.call, self.ret = ids["<|call|>"], ids["<|return|>"]
        self.ok = None not in ids.values()
        self.eos = set(eos)
        self._tries = {}
        self._masks = {}

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def none_ids(self):
        return [self.call]

    def trie(self, key, seqs):
        t = self._tries.get(key)
        if t is None:
            root = {}
            for s in seqs:
                node = root
                for x in self.enc(s):
                    node = node.setdefault(x, {})
                node[None] = s
            if len(self._tries) > 256:
                self._tries.clear()
            t = self._tries[key] = root
        return t

    def headers(self, mode, names, first):
        lead = "" if first else "<|start|>assistant"
        hs = [lead + "<|channel|>analysis<|message|>"]
        if mode == "none":
            hs.append(lead + "<|channel|>final<|message|>")
        else:
            hs += [lead + f"<|channel|>commentary to=functions.{n} <|constrain|>json<|message|>" for n in names]
        return self.trie((mode, tuple(names), first), hs)

    def body_mask(self, mode, words):
        m = self._masks.get((mode, words))
        if m is None:   # none: no call ever; required: an analysis body may only end with <|end|>
            m = _all_but([self.call] if mode == "none" else [self.call, self.ret, *self.eos], words)
            self._masks[(mode, words)] = m
        return m

    def matcher(self, spec, opens_in_think):
        mode = "none" if spec["mode"] == "none" else "required"
        return HarmonyToolMatcher(self, mode, spec.get("names") or [])


class HarmonyToolMatcher:
    """States: ("hdr", node) inside a header (the trie of the allowed ones), ("body", None) in a message body, ("free",
    None) after a complete call header (released)."""

    def __init__(self, tier, mode, names):
        self.t, self.mode, self.names = tier, mode, names
        self.state, self.node = "hdr", tier.headers(mode, names, True)
        self.hist, self.error = [], None

    def _step(self, tok):
        st, node = self.state, self.node
        if st == "free":
            return st, node
        if st == "body":
            if tok == self.t.call or (self.mode != "none" and (tok == self.t.ret or tok in self.t.eos)):
                return None
            if tok == self.t.end:
                return "hdr", self.t.headers(self.mode, self.names, False)
            return st, node
        nxt = node.get(tok)
        if nxt is None:
            return None
        if None in nxt and len(nxt) == 1:
            return ("free", None) if "to=functions." in nxt[None] else ("body", None)
        return "hdr", nxt

    def consume_token(self, tok):
        r = self._step(int(tok))
        if r is None:
            self.error = f"tool_choice: token {tok} not allowed in state {self.state}"
            return False
        self.hist.append((self.state, self.node))
        if len(self.hist) > 1024:   # (srv L8: a rollback deeper than the kept history raised IndexError; drafts and the
            del self.hist[:-512]    # server's rollbacks are far below 512 tokens)
        self.state, self.node = r
        return True

    def rollback(self, n):
        if n > len(self.hist):
            raise ValueError(f"tool_choice: rollback of {n} tokens, only {len(self.hist)} kept")
        for _ in range(n):
            self.state, self.node = self.hist.pop()
        self.error = None

    def compute_ff_tokens(self):
        out, node = [], self.node
        if self.state != "hdr":
            return out
        while node is not None and len(node) == 1 and None not in node:
            (tok, node), = node.items()
            out.append(tok)
        return out

    def unsafe_compute_mask_ptr(self, ptr, nbytes):
        import ctypes
        words = nbytes // 4
        if self.state == "hdr":
            m = _mask_of([t for t in self.node if t is not None], words)
        elif self.state == "body":
            m = self.t.body_mask(self.mode, words)
        else:
            m = _all_but([], words)
        ctypes.memmove(ptr, m.ctypes.data, words * 4)

    def is_stopped(self):
        return False

    def is_accepting(self):
        return self.mode == "none" or self.state == "free"

    def is_error(self):
        return self.error is not None

    def get_error(self):
        return self.error or ""

    def released(self):
        return self.state == "free"


class HarmonyTrack:
    """[srv-gptoss-think] Where a harmony reply is, token by token: in a header ("hdr", collecting it), in an analysis
    body ("think": reasoning, counted), in another body ("body"; final = the answer), or between messages ("out").
    Used by ThinkControl (think_budget.py) for the budget count and by the logprobs filter (final body tokens only)."""

    def __init__(self, ti, ids=()):
        self.ti = ti
        self.state, self.hdr, self.channel = "out", [], None
        self.fresh = True       # no answer (final / commentary body) yet
        self.last_final = False  # the last token fed was a body token of the final channel
        self.answer_toks = 0     # body tokens outside the analysis channel (answer, preamble, tool-call arguments)
        # the prompt's last message: the generation prompt `<|start|>assistant` leaves the reply inside its header; a
        # prefilled assistant message leaves it inside that message's body
        ids = list(ids)
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == ti.start:
                self.state, self.hdr = "hdr", []
                self.feed(ids[i + 1:])
                break
            if ids[i] in (ti.end, ti.call, ti.ret):
                break
        self.last_final = False
        self.answer_toks = 0

    @property
    def open(self):
        return self.state == "think"

    def _channel_of(self, hdr):
        text = self.ti.tok.decode(hdr, skip_special_tokens=False)
        m = re.search(r"<\|channel\|>\s*(\w+)", text)
        return m.group(1) if m else None

    def feed(self, toks):
        """Advance over tokens; returns how many were analysis body tokens."""
        ti, n = self.ti, 0
        for t in toks:
            self.last_final = False
            if self.state == "hdr":
                if t == ti.message:
                    self.channel = self._channel_of(self.hdr)
                    self.state = "think" if self.channel == "analysis" else "body"
                    if self.state == "body":
                        self.fresh = False
                elif t in (ti.end, ti.call, ti.ret):
                    self.state = "out"
                else:
                    self.hdr.append(t)
            elif self.state in ("think", "body"):
                if t in (ti.end, ti.call, ti.ret):
                    self.state = "out"
                elif self.state == "think":
                    n += 1
                else:
                    self.last_final = self.channel == "final"
                    self.answer_toks += 1
            elif t == ti.start:
                self.state, self.hdr = "hdr", []
        return n


class HarmonyThinkIds:
    """[srv-gptoss-think] ThinkIds for harmony (think_budget.py's interface): the reasoning is the analysis channel's
    body; the forced close ends it and opens the answer, `<|end|><|start|>assistant<|channel|>final<|message|>` (after the
    wrap-up sentence when there is room). ThinkControl tracks the state with HarmonyTrack (tracker()) instead of a
    single open / close token."""
    harmony = True

    def __init__(self, tok, wrap_body):
        self.tok, self.spaces = tok, {}
        ids = {t: tok.token_to_id(t) for t in TAGS}
        self.ok = None not in ids.values()
        self.start, self.end, self.message = ids["<|start|>"], ids["<|end|>"], ids["<|message|>"]
        self.call, self.ret = ids["<|call|>"], ids["<|return|>"]
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        close = enc("<|end|><|start|>assistant<|channel|>final<|message|>")
        body = (wrap_body or "").strip()
        # keyed by the newlines put before the wrap-up (think_budget.ThinkControl's rule)
        self.wrap = {p: (enc(p + body) if body else []) + close for p in ("", "\n", "\n\n")}
        self.bare = {p: list(close) for p in ("", "\n", "\n\n")}
        self.wrap_max = max(len(v) for v in self.wrap.values())
        if self.ok and (close.count(self.end) != 1 or close[0] != self.end or close[-1] != self.message):
            self.ok = False

    def tracker(self, ids=()):
        return HarmonyTrack(self, ids)

    def is_space(self, t):
        v = self.spaces.get(t)
        if v is None:
            v = self.spaces[t] = not self.tok.decode([t], skip_special_tokens=False).strip()
        return v

    def open_at_end(self, ids, im_start=None):
        """The reply starts in reasoning: inside the assistant header (gpt-oss reasons first) or an analysis body."""
        tr = HarmonyTrack(self, ids)
        return tr.state == "hdr" or tr.open
