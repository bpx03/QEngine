"""tool_choice (OpenAI /v1/chat/completions and Anthropic /v1/messages), both models.

The tools stay in the prompt whatever the choice (the prefix cache stays valid; OpenAI semantics: `none` means the
model does not call, not that it cannot see the tools). The choice is enforced on the reply through the engine's
grammar tier (server/qe_capi.h TIER 8, the path of structured output and of the forced close of <think>): the slot is
constrained and every step's rows are masked by ToolMatcher below, a small token automaton with the llguidance
matcher interface, so grammar.Constraint plans its drafts and masks exactly as for a JSON schema.

  none                 `<tool_call>` is masked for the whole reply (thinking included): the model answers in text
  required / any       inside <think>: `<tool_call>` and the end of turn are masked (the model closes its thinking
                       first, as the template renders a call after `</think>\\n\\n`); then the call header
                       `\\n\\n<tool_call>\\n<function=` + a tool name + `>\\n` is the only path (a trie over the
                       tools' tokenized headers: a unique path is forced, drafted and accepted in one step). Once a
                       header is complete the slot is released and the arguments decode freely.
  a named tool         the same with one name: the header is forced token by token.
  auto / absent        nothing (as before).

The constraint for `none` stays for the whole reply (a constrained slot is never queued ahead and uses no lookup
drafts, so such replies decode on that slower path); `required` / named ones only until the call header.

Without the grammar tier (an engine library without TIER 8, or a parallel mode that refuses it): `none` ends the reply
at its first `<tool_call>` token in the answer (the text before it is the reply, finish stop; a call written inside
<think> stays reasoning text), and `required` / named choices are not enforced (one log line at start-up and one at
the first such request); never an error.
"""
import ctypes

import numpy as np

AUTO, NONE, REQUIRED = "auto", "none", "required"


class ToolChoiceError(ValueError):
    """A tool_choice the server cannot serve: HTTP 400 invalid_request_error, param tool_choice."""


def tool_names(tools):
    """Function names of the request's tools (OpenAI form, as anthropic_tools() makes them), in order, unique."""
    out = []
    for t in tools or []:
        f = t.get("function", t) if isinstance(t, dict) else None
        n = f.get("name") if isinstance(f, dict) else None
        if isinstance(n, str) and n and n not in out:
            out.append(n)
    return out


def _named(tc):
    """The tool name of an object form: OpenAI {"type": "function", "function": {"name"}} (also a top-level "name",
    the Responses form), Anthropic {"type": "tool", "name"}."""
    f = tc.get("function")
    n = f.get("name") if isinstance(f, dict) else tc.get("name")
    if not isinstance(n, str) or not n:
        raise ToolChoiceError(f"tool_choice type {tc.get('type')!r} needs a function name")
    return n


def parse(tc, tools):
    """The request's tool_choice -> None (auto: nothing to enforce) or {"mode": "none"} / {"mode": "required",
    "names": [...]}; ToolChoiceError for a value the server cannot serve. Both APIs' forms are read on both paths:
    OpenAI "none" / "auto" / "required", {"type": "function", "function": {"name": X}}, {"type": "allowed_tools",
    "allowed_tools": {"mode": "auto" | "required", "tools": [...]}}; Anthropic {"type": "auto" | "any" | "none"},
    {"type": "tool", "name": X} (disable_parallel_tool_use is accepted and has no effect, as parallel_tool_calls).
    A choice that forces a call needs tools, and a named tool must be one of them; `none` / `auto` without tools are
    accepted (nothing to do)."""
    if tc is None:
        return None
    if isinstance(tc, str):
        kind, names = tc.strip().lower(), None
    elif isinstance(tc, dict):
        kind = tc.get("type")
        kind = kind.strip().lower() if isinstance(kind, str) else kind
        names = None
        if kind in ("function", "tool"):
            names, kind = [_named(tc)], REQUIRED
        elif kind == "allowed_tools":
            at = tc.get("allowed_tools")
            at = at if isinstance(at, dict) else tc  # (the fields are also accepted at the top level)
            mode = at.get("mode", AUTO)
            lst = at.get("tools")
            if mode not in (AUTO, REQUIRED) or not isinstance(lst, list) or not lst:
                raise ToolChoiceError("tool_choice allowed_tools needs mode auto or required and a list of tools")
            kind = mode
            names = []
            for t in lst:
                if not isinstance(t, dict):
                    raise ToolChoiceError("tool_choice allowed_tools: each tool must be an object")
                n = _named(t)
                if n not in names:
                    names.append(n)
    else:
        raise ToolChoiceError("tool_choice must be a string or an object")
    if kind == "any":
        kind = REQUIRED
    if kind not in (AUTO, NONE, REQUIRED):
        raise ToolChoiceError(f"unsupported tool_choice {tc!r}: use none, auto, required (Anthropic any) or a named "
                              f"function")
    have = tool_names(tools)
    if names is not None:
        unknown = [n for n in names if n not in have]
        if unknown:
            raise ToolChoiceError(f"tool_choice names {', '.join(repr(n) for n in unknown)}, which is not in tools"
                                  f"{' (' + ', '.join(have) + ')' if have else ' (no tools sent)'}")
    if kind == AUTO:
        return None  # (allowed_tools in auto mode: the model may still call any of the tools it sees)
    if kind == NONE:
        # without tools the template renders no tool instructions: nothing to mask (no constrained decode, review L1);
        # a call the model writes anyway still ends the reply ("cut") and is never returned
        return {"mode": NONE} if have else {"mode": NONE, "cut": True}
    if not have:
        raise ToolChoiceError("tool_choice requires a call but the request has no tools")
    return {"mode": REQUIRED, "names": names or have, "subset": names is not None}


def filter_calls(spec, calls):
    """The parsed calls a reply may return under its tool_choice: none under `none`; under a named choice or an
    allowed_tools list only calls to those tools (the first call's header is forced, a later call is the model's own:
    one to a tool outside the choice is dropped, review L4)."""
    if not spec or not calls:
        return calls
    if spec["mode"] == NONE:
        return []
    if spec.get("subset"):
        return [c for c in calls if c.get("name") in spec["names"]]
    return calls


# matcher states
S_NOCALL, S_THINK, S_TRIE, S_FREE = "nocall", "think", "trie", "free"


class Tier:
    """Token ids and masks of the tool_choice constraints, built once per server (tokenizer, ThinkIds, end tokens)."""

    def __init__(self, tok, think_ids, eos, im_start):
        self.tok, self.ti = tok, think_ids
        self.call = tok.token_to_id("<tool_call>")
        self.ok = self.call is not None
        self.blocked_think = {self.call, *eos} | ({im_start} if im_start is not None else set())
        self._masks = {}
        self._tries = {}

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def header(self, lead, name):
        """Tokens of the call header the template renders (`<tool_call>\\n<function=NAME>\\n`, after `lead`), cut back
        to the part whose tokenization does not depend on what follows (checked against a header followed by a
        parameter and by the end of the function)."""
        h = self.enc(lead + "<tool_call>\n<function=" + name + ">\n")
        for tail in ("<parameter=x>\nv\n</parameter>\n", "</function>\n"):
            f = self.enc(lead + "<tool_call>\n<function=" + name + ">\n" + tail)
            k = 0
            while k < len(h) and k < len(f) and h[k] == f[k]:
                k += 1
            h = h[:k]
        return h

    def trie(self, lead, names):
        key = (lead, tuple(names))
        t = self._tries.get(key)
        if t is None:
            root = {}
            for n in names:
                node = root
                for x in self.header(lead, n):
                    node = node.setdefault(x, {})
                node[None] = True  # a complete header (a node may also lead on: a name that prefixes another)
            if len(self._tries) > 256:
                self._tries.clear()
            t = self._tries[key] = root
        return t

    def mask(self, kind, words):
        """Cached rows: "all" (every token), "nocall" (all but <tool_call>), "think" (all but <tool_call>, the end
        tokens and <|im_start|>)."""
        m = self._masks.get((kind, words))
        if m is None:
            m = np.full(words, 0xFFFFFFFF, np.uint32)
            for t in ({S_NOCALL: {self.call}, S_THINK: self.blocked_think}.get(kind) or ()):
                if t is not None and 0 <= t < 32 * words:
                    m[t >> 5] &= np.uint32(~(1 << (t & 31)) & 0xFFFFFFFF)
            self._masks[(kind, words)] = m
        return m

    def matcher(self, spec, opens_in_think):
        """A fresh ToolMatcher for one reply (spec from parse(); opens_in_think: the prompt leaves the reply inside
        <think>, ThinkIds.open_at_end)."""
        if spec["mode"] == NONE:
            return ToolMatcher(self, S_NOCALL, None, None)
        names = spec["names"]
        if opens_in_think and self.ti is not None and self.ti.ok:
            return ToolMatcher(self, S_THINK, None, self.trie("\n\n", names))
        return ToolMatcher(self, S_TRIE, self.trie("", names), None)


class ToolMatcher:
    """The llguidance LLMatcher calls grammar.Constraint makes (consume_token, rollback, compute_ff_tokens,
    unsafe_compute_mask_ptr, is_stopped, is_accepting, is_error, get_error) over the tool_choice automaton:
    nocall (all but <tool_call>, for good), think (until </think>, then the trie after it), trie (only the next header
    tokens), free (the header is complete: every token; the job then releases the slot, released())."""

    def __init__(self, tier, state, node, after_think):
        self.t = tier
        self.state, self.node, self.after_think = state, node, after_think
        self.hist = []
        self.error = None

    def _step(self, tok):
        """(state, node) after tok, or None when the automaton refuses it."""
        st, node = self.state, self.node
        if st == S_FREE:
            return st, node
        if st == S_NOCALL:
            return None if tok == self.t.call else (st, node)
        if st == S_THINK:
            if tok in self.t.blocked_think:
                return None
            if tok == self.t.ti.end:
                return S_TRIE, self.after_think
            return st, node
        nxt = node.get(tok)
        if nxt is None:
            return None
        if nxt.get(None) and len(nxt) == 1:  # a complete header that leads nowhere else
            return S_FREE, None
        return S_TRIE, nxt

    def consume_token(self, tok):
        r = self._step(int(tok))
        if r is None:
            self.error = f"tool_choice: token {tok} not allowed in state {self.state}"
            return False
        self.hist.append((self.state, self.node))
        if len(self.hist) > 64:  # rollback() undoes at most one step's drafts (< 16): a bounded history (review L6)
            del self.hist[:-32]
        self.state, self.node = r
        return True

    def rollback(self, n):
        for _ in range(n):
            self.state, self.node = self.hist.pop()
        self.error = None

    def compute_ff_tokens(self):
        """Tokens forced from here: the trie's unique path (stops at a branch or at a complete header)."""
        out, node = [], self.node
        if self.state != S_TRIE:
            return out
        while node is not None and len(node) == 1 and None not in node:
            (tok, node), = node.items()
            out.append(tok)
        return out

    def unsafe_compute_mask_ptr(self, ptr, nbytes):
        words = nbytes // 4
        if self.state == S_TRIE:
            m = np.zeros(words, np.uint32)
            for t in self.node:
                if t is not None and 0 <= t < 32 * words:
                    m[t >> 5] |= np.uint32(1 << (t & 31))
            if self.node.get(None):  # a complete header that a longer name extends: anything may follow too
                m = self.t.mask("all", words)
        else:
            m = self.t.mask({S_FREE: "all"}.get(self.state, self.state), words)
        ctypes.memmove(ptr, m.ctypes.data, words * 4)

    def is_stopped(self):
        return False  # never complete: the reply goes on after the call header (or ends with the model's EOS)

    def is_accepting(self):
        return self.state in (S_NOCALL, S_FREE)  # an end in <think> or in the header: required call not made

    def is_error(self):
        return self.error is not None

    def get_error(self):
        return self.error or ""

    def released(self):
        """The forced part is done: the job may drop the constraint (the rest of the reply decodes freely)."""
        return self.state == S_FREE or (self.state == S_TRIE and bool(self.node.get(None)))
