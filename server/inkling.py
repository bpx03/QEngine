"""[ink-srv-capi] Inkling 2 Mini in the shared server: history preparation for the model's own chat template, the
streaming reply parser and the tool-call parser.

A reply is a series of messages (tokenizer_config.json response_template):
  <|message_model|>[NAME]<|content_thinking|>BODY<|end_message|>            reasoning
  <|message_model|><|content_text|>BODY<|end_message|>                       content (also content_tool_error /
                                                                               content_invoke_tool_text)
  <|message_model|>NAME<|content_invoke_tool_json|>{"name": .., "args": {..}}<|end_message|>   a tool call
ended by <|content_model_end_sampling|> (EOS 200006; the server stops there). The prompt ends after the last message
(the template has no generation prompt), so the reply begins with its own <|message_model|>.

Only the real control tokens count: the detokenizer marks them (content_markers.mark, TAGS below added to the marks for
this family), so the same strings written as text by the model are text.
"""
import json

from content_markers import mark, unmark, OUT_MARK

TAGS = ("<|message_model|>", "<|message_user|>", "<|message_system|>", "<|message_tool|>", "<|content_thinking|>",
        "<|content_text|>", "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>", "<|content_tool_error|>",
        "<|content_xml|>", "<|end_message|>", "<|content_model_end_sampling|>", "<|content_image|>")
MODEL, USER, SYSTEM, TOOL, THINK, TEXT, CALL, CALL_TEXT, TOOL_ERR, XML, END, EOS, IMAGE = (mark(t) for t in TAGS)
OPENS = {THINK: "reasoning", TEXT: "content", CALL_TEXT: "content", TOOL_ERR: "content", CALL: "tool", XML: "content"}
HEADS = (MODEL, USER, SYSTEM, TOOL)
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def marks(tok):
    """{token id: marked text} of the control tokens (for Detok)."""
    return {i: mark(t) for t in TAGS for i in [tok.token_to_id(t)] if i is not None}


def _partial(s, tags):
    """length of the longest suffix of s that is a proper prefix of one of the tags (a tag split across two feeds)"""
    k = s.rfind(OUT_MARK)
    if k < 0 or len(s) - k >= max(map(len, tags)):
        return 0
    tail = s[k:]
    return len(tail) if any(t.startswith(tail) and t != tail for t in tags) else 0


def _text(c):
    """message content as one string (the template takes a string or text parts); other parts keep their text"""
    if c is None or isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append(p["text"])
            elif isinstance(p, dict) and p.get("type") in ("image", "image_url", "input_image"):
                out.append("[image omitted: this server runs without vision]")
            elif p is not None:
                out.append(json.dumps(p, ensure_ascii=False, default=str))
        return "\n\n".join(x for x in out if x)
    return str(c)


def effort(kw):
    """The template's reasoning_effort from the server's variables (app.template_kwargs): thinking off = "none" (0.0);
    a known level as is; a number (or a numeric string) as a float (the template checks [0, 0.99]); anything else as is
    (the template refuses it: HTTP 400). None: the template's own default (0.9)."""
    kw = dict(kw)
    think = kw.pop("enable_thinking", True)
    e = kw.get("reasoning_effort")
    if think is False:
        e = "none"
    if isinstance(e, str):
        s = e.strip().lower()
        if s in EFFORTS:
            e = s
        else:
            try:
                e = float(s)
            except ValueError:
                pass
    elif isinstance(e, bool):
        e = None
    if e is None:
        kw.pop("reasoning_effort", None)
    else:
        kw["reasoning_effort"] = e
    return kw


def prepare(messages):
    """OpenAI-shaped history -> what the Inkling template reads: content parts joined into text (it takes strings or
    text parts; images: a text note, this server has no vision), assistant reasoning in reasoning_content (also from
    `reasoning` / `thinking`), tool-call arguments as objects (the template refuses a JSON string: "canonicalize
    upstream"), the developer role as system."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if "content" in m:
            m["content"] = _text(m["content"])
        if m.get("role") == "assistant":
            r = m.get("reasoning_content")
            for k in ("reasoning", "thinking"):
                v = m.pop(k, None)
                if r is None and isinstance(v, str):
                    r = v
            if r is not None:
                m["reasoning_content"] = r
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    c = dict(c)
                    f = dict(c.get("function") or {})
                    a = f.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a) if a.strip() else {}
                        except ValueError:
                            a = {"input": a}   # (not JSON: kept as text under one key; the template needs an object)
                    if not isinstance(a, dict):
                        a = {"input": a}
                    f["arguments"] = a
                    c["function"] = f
                    calls.append(c)
                m["tool_calls"] = calls
            if m.get("content") is None:
                m["content"] = ""
        out.append(m)
    return out


class InklingParser:
    """Streaming split of an Inkling reply into reasoning / content / tool calls (app.Parser's interface: feed(text,
    final, promote) -> [(kind, text)], tool_text). tool_text keeps the call bodies as CALL + JSON + END (marked),
    parse_calls reads them."""

    ALL = tuple(OPENS) + HEADS + (END, EOS)

    def __init__(self, thinking=True):
        self.state = "out"     # out: between messages (a header: <|message_model|> NAME); body: inside one
        self.kind = None
        self.buf = ""
        self.tool_text = ""
        self.promoted = False
        self.mode = "reasoning"
        self.head = False      # a <|message_model|> header is open (its NAME is not text)

    def feed(self, text, final=False, promote=False):
        out = []
        self.buf += text
        while self.buf:
            if self.state == "body":
                ends = [k for k in (self.buf.find(t) for t in (END, EOS) + HEADS) if k >= 0]
                k = min(ends) if ends else -1
                hold = 0 if (k >= 0 or final) else _partial(self.buf, self.ALL)
                body = self.buf[:len(self.buf) - hold] if k < 0 else self.buf[:k]
                if body:
                    if self.kind == "tool":
                        self.tool_text += body
                    else:
                        out.append((self.kind, body))
                if k < 0:
                    self.buf = self.buf[len(self.buf) - hold:] if hold else ""
                    break
                if self.kind == "tool":
                    self.tool_text += END
                if self.buf.startswith(END, k) or self.buf.startswith(EOS, k):
                    self.buf = self.buf[k + len(END if self.buf.startswith(END, k) else EOS):]
                else:
                    self.buf = self.buf[k:]   # (a header without the body's end: a new message)
                self.state, self.head = "out", False
                continue
            # out: up to the next body opening; <|message_model|> NAME before it is a header
            opens = [(k, t) for t in OPENS for k in [self.buf.find(t)] if k >= 0]
            k, tag = min(opens) if opens else (-1, None)
            pre = self.buf if k < 0 else self.buf[:k]
            h = pre.rfind(MODEL)
            stray = pre if h < 0 else pre[:h]
            for t in HEADS + (END, EOS):   # [ink-srv-review-fix] L6: a stray end / EOS / header is never content text
                stray = stray.replace(t, "")
            if k < 0:
                if final:   # text outside a message (not a header's name): the answer
                    s = stray if (h >= 0 or self.head) else pre
                    for t in HEADS + (END, EOS):
                        s = s.replace(t, "")
                    if s.strip() and not (self.head and h < 0):
                        out.append(("content", s))
                    self.buf = ""
                    break
                if h >= 0:   # a header in progress: what was before it is stray text, the rest waits
                    if stray.strip() and not self.head:
                        out.append(("content", stray))
                    self.buf = self.buf[h:]
                    self.head = True
                    if len(self.buf) > 256:   # (a header that never ends: its text)
                        out.append(("content", self.buf.replace(MODEL, "")))
                        self.buf, self.head = "", False
                    break
                if OUT_MARK not in self.buf and len(self.buf) > 64 and not self.head:
                    out.append(("content", self.buf))
                    self.buf = ""
                break
            if stray.strip() and not self.head:
                out.append(("content", stray))
            self.kind = OPENS[tag]
            self.mode = self.kind
            if self.kind == "tool":
                self.tool_text += CALL
            self.buf = self.buf[k + len(tag):]
            self.state, self.head = "body", False
        if final and self.state == "body" and self.kind == "tool" and not self.tool_text.endswith(END):
            self.tool_text += END
        return [(kd, unmark(t)) for kd, t in out if t]


def parse_calls(tool_text, tools=None):
    """tool_text of InklingParser -> [{"name", "arguments": dict}]. The body is {"name": NAME, "args": {...}}; a body
    that does not parse (or has no name) is dropped (its text then reaches the client as content,
    app.tool_calls_or_text)."""
    calls = []
    t = tool_text
    while True:
        i = t.find(CALL)
        if i < 0:
            break
        j = t.find(END, i + len(CALL))
        body = unmark(t[i + len(CALL):j if j >= 0 else len(t)]).strip()
        t = t[j + len(END):] if j >= 0 else ""
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        args = obj.get("args", obj.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": args}
        calls.append({"name": obj["name"], "arguments": args})
    return calls


def plain(tool_text):
    """The tool text of a reply none of whose calls parsed (app.tool_calls_or_text), as client text: the control tokens
    removed, the bodies kept (a model-written body with broken JSON, e.g. a doubled quote after "name", then reaches the client
    as text without <|content_invoke_tool_json|> / <|end_message|>; the other parsers drop such a call the same way)."""
    for t in (MODEL, CALL, CALL_TEXT, END, EOS, TEXT, THINK, TOOL_ERR, XML):
        tool_text = tool_text.replace(t, "")
    return unmark(tool_text)


def answer_of(text_mk):
    """(the answer text (marked), a tool call made) of a whole marked reply (app.Job.answer_call)."""
    p = InklingParser()
    parts = p.feed(text_mk, final=True)
    return "".join(t for k, t in parts if k == "content"), bool(parse_calls(p.tool_text))


CONTENT_OPENS = tuple(t for t, k in OPENS.items() if k == "content")
BODY_ENDS = (END, EOS) + HEADS


def stop_spans(text_mk):
    """[ink-srv-review-fix] L5: the (start, end) spans of the content message bodies (content_text, invoke_tool_text,
    tool_error, xml: what the parser streams as content), in order, and whether the text ends inside one (a stop may
    still begin at its end). Client stop sequences match only there: never in a later reasoning body or a tool call's
    JSON (a stop such as "\n" would cut its arguments). The bodies follow InklingParser: a body opens at a real open
    tag and ends at the next <|end_message|>, EOS or message header."""
    spans, u, n = [], 0, len(text_mk)
    while True:
        k, tag = min(((text_mk.find(t, u), t) for t in OPENS if text_mk.find(t, u) >= 0), default=(-1, None))
        if k < 0:
            return spans, False
        b = k + len(tag)
        e = min((x for x in (text_mk.find(t, b) for t in BODY_ENDS) if x >= 0), default=-1)
        if tag in CONTENT_OPENS:
            spans.append((b, n if e < 0 else e))
        if e < 0:
            return spans, tag in CONTENT_OPENS
        u = e


def answer_start(text_mk):
    """Where client stop sequences may start: the body of the first content message (None: none yet); stop_spans
    gives the bodies a stop may match in."""
    spans, _ = stop_spans(text_mk)
    return spans[0][0] if spans else None
