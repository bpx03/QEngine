"""Structured output (OpenAI response_format / Anthropic output_format) with llguidance.

The engine masks a constrained sequence's logits with a per-slot bitmask (engine.mask_buffer);
this module turns the request's schema into an llguidance grammar, advances a matcher with every
emitted token and writes the next allowed-token mask straight into that pinned buffer.
Measured on the Qwen3.8 tokenizer: grammar compile ~0.5 ms per
request, mask ~13 us per token (vs a ~9 ms decode step), so the Python path costs ~0.2 %.
"""
import json
import os
import threading

# JSON layout the grammar allows. Schemas that name their properties: json.dumps style (", " /
# ": "), no free whitespace: ~30 % fewer tokens than the model's pretty-printed layout (10-chunk
# RAG batch: 915 vs 1330 tokens) and no whitespace runs under greedy decoding. Free-form objects
# (json_object, no "properties"): any whitespace, since there the model picks the keys itself and
# a forced compact layout pushed it off its usual `{\n  "key"` path into nonsense keys.
# QE_JSON_WS=flexible / compact forces one layout for every schema.
# \uXXXX escapes in strings (QE_JSON_UESC). In the compact (json.dumps) layout the model often
# escapes non-ASCII text the json.dumps way (ensure_ascii: "wi\u0105\u017c\u0105"); llguidance by
# default allows \u only for control characters (\u0000-\u001f, \u007f), which bent "\u0105" into
# "\u0019" and "\u00f4" into "\u007f" and the model looped on them.
#   none    (default) no \u escapes at all (\" \\ \b \f \n \r \t only): the model writes the UTF-8
#           character; GPU A/B (14 RAG requests): 14/14 valid, 0 escapes, text identical to "all"
#           wherever "all" did not escape, 1.4 % fewer tokens (-22 % on the escaping batch)
#   all     any \uXXXX (complete surrogate pairs): valid JSON, but escaped Polish text costs ~2x tokens
#   control llguidance's default (the bug; for reference only)
UESC = {"all": {"json_allow_general_unicode_escapes": True},
        "none": {"json_allowed_escapes": '"\\bfnrt'},
        "control": {}}


def uesc():
    mode = os.environ.get("QE_JSON_UESC", "none")
    if mode not in UESC:
        raise GrammarError(f"QE_JSON_UESC={mode!r}: use all, none or control")
    return UESC[mode]


COMPACT = {"whitespace_flexible": False, "item_separator": ", ", "key_separator": ": "}
FLEXIBLE = {"whitespace_flexible": True}


def json_options(schema):
    mode = os.environ.get("QE_JSON_WS", "")
    if mode in ("flexible", "compact"):
        layout = FLEXIBLE if mode == "flexible" else COMPACT
    else:
        layout = COMPACT if isinstance(schema, dict) and schema.get("properties") else FLEXIBLE
    return {**layout, **uesc()}


class GrammarError(ValueError):
    pass


def schema_of(body):
    """The JSON schema a request asks for (None: unconstrained). OpenAI: response_format
    {"type": "json_schema", "json_schema": {"schema": ...}} or {"type": "json_object"};
    Anthropic: output_format / output_config.format {"type": "json_schema", "schema": ...}."""
    rf = body.get("response_format")
    if rf is None:
        rf = body.get("output_format") or (body.get("output_config") or {}).get("format")
    if not rf:
        return None
    if not isinstance(rf, dict):
        raise GrammarError("response_format must be an object")
    t = rf.get("type")
    if t in (None, "text"):
        return None
    if t == "json_object":
        return {"type": "object"}
    if t == "json_schema":
        js = rf.get("json_schema")
        sch = js.get("schema") if isinstance(js, dict) else rf.get("schema")
        if sch is None:
            return {"type": "object"}  # name only: any JSON object
        if not isinstance(sch, dict):
            raise GrammarError("json_schema.schema must be an object")
        return sch
    raise GrammarError(f"unsupported response_format type {t!r}")


class Grammars:
    """llguidance tokenizer (built once, ~1 s) + per-request matchers."""

    def __init__(self, tokenizer_json, vocab, eos_ids):
        self.tokenizer_json, self.vocab, self.eos = tokenizer_json, vocab, list(eos_ids)
        self._lt = None
        self._lock = threading.Lock()
        try:
            import llguidance  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def prepare(self):
        """Build the llguidance tokenizer in the background (first request would wait ~1 s)."""
        if self.available:
            threading.Thread(target=self.tokenizer, daemon=True).start()

    def tokenizer(self):
        with self._lock:
            if self._lt is None:
                import llguidance
                self._lt = llguidance.LLTokenizer(self.tokenizer_json, n_vocab=self.vocab, eos_token=self.eos)
            return self._lt

    def matcher(self, schema):
        """A fresh matcher for the schema (GrammarError for unusable schemas)."""
        if not self.available:
            raise GrammarError("structured output needs the llguidance package (pip install llguidance)")
        import llguidance
        try:
            g = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults=json_options(schema))
        except Exception as ex:
            raise GrammarError(f"invalid JSON schema: {ex}")
        m = llguidance.LLMatcher(self.tokenizer(), g, log_level=0)
        if m.is_error():
            raise GrammarError(f"unsupported JSON schema: {m.get_error()}")
        return Constraint(m, self.eos)


def allowed(mask, t):
    return bool((int(mask[t >> 5]) >> (t & 31)) & 1)


class Constraint:
    """Grammar state of one generation; fill() writes the next-token mask into a uint32 buffer,
    plan() the drafts and per-row masks of the next speculative step."""

    def __init__(self, m, eos=()):
        self.m = m
        self.eos = set(eos)
        self.error = None
        self.n_forced = 0  # drafts proposed from grammar-forced tokens (statistics)

    def plan(self, io, masks, kmax):
        """Next speculative step of a constrained sequence (engine spec_io / mask_buffer layout):
        candidates are the tokens the grammar forces from here (always accepted: their row mask
        holds only them), followed by the model's drafts io[1 : 1 + io[0]] when those agree with
        them (else the model's drafts alone); the grammar-valid prefix (at most kmax) goes to
        io[16] / io[17..], and masks[j] = allowed tokens after drafts 1..j (masks[0]: now)."""
        m = self.m
        self.fill(masks[0])
        n = 0
        if not m.is_stopped() and kmax > 0:
            drafts = [int(x) for x in io[1:1 + max(0, min(int(io[0]), 15))]]
            ff = m.compute_ff_tokens()
            if ff:
                cand = list(ff) + drafts[len(ff):] if drafts[:len(ff)] == list(ff) else list(ff)
                self.n_forced += min(len(ff), kmax)
            else:
                cand = drafts
            for t in cand[:kmax]:
                # the row mask is exact: a token outside it would put the matcher in its error state
                if t in self.eos or not allowed(masks[n], t) or not m.consume_token(t):
                    break
                n += 1
                io[16 + n] = t
                self.fill(masks[n])
                if m.is_stopped():
                    break
            if n:
                m.rollback(n)
        io[16] = n
        return n

    def fill(self, buf):
        self.m.unsafe_compute_mask_ptr(buf.ctypes.data, buf.nbytes)

    def consume(self, t):
        """Advance by one emitted token; False when the grammar rejects it (error state)."""
        if self.m.consume_token(t):
            return True
        self.error = self.m.get_error()
        return False

    def done(self):
        """The grammar is complete: nothing but the end token can follow."""
        return self.m.is_stopped()

    def accepting(self):
        return self.m.is_accepting()
