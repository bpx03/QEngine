"""Template markers inside message content stay plain text (issue #7).

The chat prompt is rendered to text and tokenized with added-token parsing, so a marker string that a
message merely quotes (`<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>`, `<|image_pad|>`, any added
token) became the control token itself: fake turns in the prompt, a model that copies a real
`<|im_start|>` when it quotes one (the server ends the reply there), an image placeholder count that no
longer matches.

Input side: every string that comes from the request (message content, text parts, reasoning, tool-call
names and arguments, tool definitions) has its added-token strings replaced by sentinel pairs before the
template runs (SENT + one private-use code per marker; a literal SENT is escaped the same way). The
rendered text then holds real markers only where the template wrote them. Generator.encode tokenizes a
segment without sentinels exactly as before (so a conversation without markers in its content gives the
same ids as without this module), and a segment with sentinels piece by piece: the template's markers as
their tokens, the text between them restored and tokenized with the added tokens switched off (plain
byte-level BPE, as HF `split_special_tokens`).

Partial markers at the ends of a string (review M1): the template writes text right next to request strings
(`'<parameter=' + key + '>'`), so a key ending in `<|im_start|` would be completed into the token. A string
whose end (before trailing whitespace) is a proper prefix of a marker, or whose start (after leading whitespace)
is a proper suffix of one, gets an empty sentinel (EDGE) there: the marker cannot form across it, and restore()
drops it. Its ids are the master ids unless the template really completed a marker there.

Exception (the template's own protocol): a user message that is wholly `<tool_response>` blocks
(`<tool_response>...</tool_response>`, one or more, as the template writes consecutive tool results) keeps the
tags at the block boundaries (the template's multi-step-tool check looks at the outer two); the insides are
protected.

Output side: the detokenizer writes the real `<think>`, `</think>`, `<tool_call>`, `</tool_call>` tokens
with their `<` replaced by OUT_MARK (same length), so the reply parser and the tool-call parser react to
those tokens only; the same strings written as plain text stream as text. unmark() restores the text. A literal
OUT_MARK in the decoded text (the model wrote U+FDD1) is escaped (OUT_LIT), so only a real tag token carries the mark."""
import json
import re
import threading

SENT = "﷐"  # Unicode noncharacters: never in real text
OUT_MARK = "﷑"
PARSER_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>")
_BASE = 0xE000  # code chars (private use) after SENT


# a U+FDD1 the model wrote itself (decoded from ordinary tokens) stands for itself, not a tag (review L5): Detok writes
# it as OUT_LIT, a lone surrogate (same length; a UTF-8 decode never yields one), and unmark() turns it back
OUT_LIT = "\udfd1"


def unmark(s):
    if OUT_MARK in s:
        s = s.replace(OUT_MARK, "<")
    return s.replace(OUT_LIT, OUT_MARK) if OUT_LIT in s else s


def escape_out(s):
    """Decoded text of ordinary tokens with a literal OUT_MARK escaped (Detok, review L5)."""
    return s.replace(OUT_MARK, OUT_LIT) if OUT_MARK in s else s


def mark(tag):
    return OUT_MARK + tag[1:]


def out_marks(tok):
    """{token id: marked text} of the reply parser's tags (Detok)."""
    return {i: mark(t) for t in PARSER_TAGS for i in [tok.token_to_id(t)] if i is not None}


class Markers:
    def __init__(self, tok, tok_json=None):
        self.tok, self.tok_json = tok, tok_json
        dec = tok.get_added_tokens_decoder()
        self.atom_id = {a.content: i for i, a in dec.items()}
        names = sorted(self.atom_id, key=len, reverse=True)
        self.code = {m: SENT + chr(_BASE + k) for k, m in enumerate(names)}
        self.code[SENT] = SENT + chr(_BASE + len(names))
        self.edge = SENT + chr(_BASE + len(names) + 1)  # restores to nothing
        self.back = {v[1]: m for m, v in self.code.items()}
        self.back[self.edge[1]] = ""
        self.prefixes = {m[:i] for m in names for i in range(1, len(m))}
        self.suffixes = {m[i:] for m in names for i in range(1, len(m))}
        self.plen = max(map(len, names)) - 1 if names else 0
        self.blocks_rx = re.compile(r"(</tool_response>\s*<tool_response>)")
        self.rx = re.compile("|".join(map(re.escape, [SENT] + names)))
        self.unrx = re.compile(SENT + "(.)", re.S)
        self.atoms_rx = re.compile("(" + "|".join(map(re.escape, names)) + ")")
        self.out = out_marks(tok)
        self._plain, self._mu = None, threading.Lock()
        self.protected = 0  # strings with a whole marker replaced (diagnostics)
        self.edges = 0  # edge sentinels set (diagnostics)

    # ---- input
    def protect_str(self, s):
        if self.rx.search(s):
            self.protected += 1
            s = self.rx.sub(lambda m: self.code[m.group(0)], s)
        n = len(s)
        e = n
        while e > 0 and s[e - 1].isspace():
            e -= 1
        b = 0
        while b < e and s[b].isspace():
            b += 1
        if b == e:
            return s
        tail = next((k for k in range(min(self.plen, e - b), 0, -1) if s[e - k:e] in self.prefixes), 0)
        head = next((k for k in range(min(self.plen, e - b), 0, -1) if s[b:b + k] in self.suffixes), 0)
        if tail:
            s = s[:e] + self.edge + s[e:]
        if head:
            s = s[:b] + self.edge + s[b:]
        if tail or head:
            self.edges += 1
        return s

    def restore(self, s):
        return self.unrx.sub(lambda m: self.back.get(m.group(1), m.group(0)), s) if SENT in s else s

    def protect(self, x):
        if isinstance(x, str):
            return self.protect_str(x)
        if isinstance(x, list):
            return [self.protect(v) for v in x]
        if isinstance(x, tuple):
            return tuple(self.protect(v) for v in x)
        if isinstance(x, dict):
            return {(self.protect_str(k) if isinstance(k, str) else k): self.protect(v) for k, v in x.items()}
        return x

    def protect_messages(self, messages):
        out = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            c, wrapped = m.get("content"), None
            if m.get("role") == "user" and isinstance(c, str):
                s = c.strip()
                o, e = "<tool_response>", "</tool_response>"
                if len(s) >= len(o) + len(e) and s.startswith(o) and s.endswith(e):
                    i, j = c.index(o) + len(o), c.rindex(e)
                    parts = self.blocks_rx.split(c[i:j])  # odd: block boundaries, kept as written
                    wrapped = c[:i] + "".join(x if k % 2 else self.protect_str(x) for k, x in enumerate(parts)) + c[j:]
            p = {k: (v if k == "role" or (k == "content" and wrapped is not None) else self.protect(v))
                 for k, v in m.items()}
            if wrapped is not None:
                p["content"] = wrapped
            out.append(p)
        return out

    def plain_tok(self):
        """The tokenizer without its added tokens (~0.4 s; the server builds it at start, review M2). The
        tokenizer.json text is dropped once it is built."""
        with self._mu:
            if self._plain is None:
                from tokenizers import Tokenizer
                d = json.loads(self.tok_json)
                d["added_tokens"] = []
                self._plain = Tokenizer.from_str(json.dumps(d, ensure_ascii=False))
                self.tok_json = None
            return self._plain

    def encode_segment(self, seg):
        """Ids of a rendered segment that holds sentinels: the template's markers as their tokens, the
        rest (content with its markers restored) as plain text."""
        ids = []
        plain = None
        for k, piece in enumerate(self.atoms_rx.split(seg)):
            if k % 2:
                ids.append(self.atom_id[piece])
            elif piece:
                if SENT in piece:
                    plain = plain or self.plain_tok()
                    ids += plain.encode(self.restore(piece), add_special_tokens=False).ids
                else:
                    ids += self.tok.encode(piece, add_special_tokens=False).ids
        return ids
