"""CPU-only: json_schema + non-ASCII text.

Bug 1: llguidance allowed \\uXXXX escapes only for control characters, so a model escaping
"wiążą" / "L'Hôpital" the json.dumps way ("wi\\u0105...", "L'H\\u00f4pital") was bent into
"\\u0019..." / "\\u007f..." and looped. Bug 2: the loop guard then ended the turn with
finish_reason "stop" on a cut-off JSON. Replays the two failing RAG batches' answer texts (direct
UTF-8 and ensure_ascii escapes, surrogate pairs; every QE_JSON_UESC mode) through the server path minus the GPU: the
grammar masks, the stage-2 speculative plan with per-row masks, and Job.feed (finish reasons).
python tests/test_json_utf8.py"""
import json, os, sys, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.path.insert(0, HERE)
import test_grammar as tg  # noqa: E402  (runs the base grammar tests too)
import grammar  # noqa: E402
# test_grammar re-imports grammar.allowed(mask, t) at its end; walk() calls allowed(t) on buf
tg.allowed = lambda *a: grammar.allowed(tg.buf, a[0]) if len(a) == 1 else grammar.allowed(*a)
from test_grammar import G, tok, walk, simulate, allowed, buf, V, EOS  # noqa: E402
from grammar import schema_of  # noqa: E402
import llguidance  # noqa: E402


def body(n):  # the failing requests' response_format (failed_ed197f7c_b0: 8 items, 684e7072: 7)
    return {"response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": {
        "type": "object", "properties": {"items": {"type": "array", "minItems": n, "maxItems": n, "items": {
            "type": "object", "properties": {
                "chunk": {"type": "integer", "minimum": 1, "maximum": n}, "context": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
            "required": ["chunk", "context", "questions"], "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False}}}}


S8, S7 = schema_of(body(8)), schema_of(body(7))
PL = {"items": [{"chunk": i, "context": "Rozdział o etyce sztucznej inteligencji, fragment %d." % i,
                 "questions": ["Jakie etyczne wyzwania wiążą się z autonomicznymi systemami?",
                               "Czym jest zażółć gęślą jaźń — test polskich znaków?"]} for i in range(1, 9)]}
EN = {"items": [{"chunk": i, "context": "Calculus chapter on limits, part %d." % i,
                 "questions": ["How is the limit evaluated by applying L'Hôpital's rule?",
                               "Why does ∫ e^x dx = e^x + C (and 😀 / 𝔼[X] survive)?"]} for i in range(1, 8)]}
CASES = [("ed197f7c (PL, wiążą)", S8, PL), ("684e7072 (EN, L'Hôpital)", S7, EN)]


def compact(obj, ascii_):  # the layout the grammar forces for schemas with properties (json.dumps)
    return json.dumps(obj, ensure_ascii=ascii_)


def with_uesc(mode, fn):
    old = os.environ.get("QE_JSON_UESC")
    os.environ["QE_JSON_UESC"] = mode
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("QE_JSON_UESC")
        else:
            os.environ["QE_JSON_UESC"] = old


# ---- bug 1, per QE_JSON_UESC mode: which spellings pass the grammar and the stage-2 plan
# all: direct UTF-8 and ensure_ascii escapes; none: direct UTF-8 only; control: direct UTF-8 only
for mode, ascii_ok in (("all", True), ("none", False), ("control", False)):
    for name, sch, obj in CASES:
        for ascii_ in (False, True):
            text = compact(obj, ascii_)
            assert ("\\u" in text) == ascii_
            if ascii_ and not ascii_ok:
                assert with_uesc(mode, lambda: walk(sch, text, expect_ok=False)) is False, (mode, name)
                continue
            with_uesc(mode, lambda: walk(sch, text))  # every token in the mask computed before it
            for q in (0.0, 0.7, 1.0):  # per-row masks of speculative steps (forced tokens + drafts)
                with_uesc(mode, lambda: simulate(sch, text, q))
            assert json.loads(text) == obj
    print(f"QE_JSON_UESC={mode}: direct UTF-8 ok, ensure_ascii {'ok' if ascii_ok else 'rejected'}")
    # the other escapes stay in every mode
    with_uesc(mode, lambda: walk({"type": "string"}, json.dumps('a "q" \\ b\nc\td\re\x08\x0c /')))

# token cost of the escaped spelling
for name, _, obj in CASES:
    d, a = (len(tok.encode(compact(obj, x), add_special_tokens=False).ids) for x in (False, True))
    print(f"{name}: direct {d} tokens, ensure_ascii {a} tokens (+{100 * (a - d) / d:.0f} %)")


def mask_after(mode, prefix):
    c = with_uesc(mode, lambda: G.matcher(S8))
    for t in tok.encode(prefix, add_special_tokens=False).ids:
        assert c.consume(t), c.error
    c.fill(buf)
    return c


# the failing spots: after '...wi' / "...L'H" the model reaches for ą / ô
for word, want, direct in (("Jakie etyczne wyzwania wi", "0105", "ąż"), ("by applying L'H", "00f4", "ô")):
    prefix = '{"items": [{"chunk": 1, "context": "x", "questions": ["' + word
    got = {}
    for mode in ("all", "none", "control"):
        c = mask_after(mode, prefix)
        assert allowed(buf, tok.encode(direct, add_special_tokens=False).ids[0]), (mode, direct)  # always
        bs, bu = tok.token_to_id("\\"), tok.token_to_id("\\u")
        assert allowed(buf, bs)  # \" \\ \n ... stay
        if mode == "none":
            assert not allowed(buf, bu), "\\u token allowed in none"
            assert c.consume(bs)
            c.fill(buf)
            assert not allowed(buf, tok.token_to_id("u")), "u after \\ allowed in none"
            got[mode] = "blocked at u"
            continue
        assert allowed(buf, bu) and c.consume(bu)
        s = ""
        for ch in want:
            c.fill(buf)
            if not allowed(buf, tok.token_to_id(ch)):
                break
            assert c.consume(tok.token_to_id(ch))
            s += ch
        assert (s == want) == (mode == "all"), (mode, s)
        got[mode] = "\\u" + s + ("" if s == want else "<" + want[len(s)] + " blocked>")
    print(f"{word[-3:]}+\\u{want}: " + ", ".join(f"{k}: {v}" for k, v in got.items()))

# surrogate pairs (all): a pair is valid, a lone surrogate is not (json.loads would take it, but it is not UTF-8)
with_uesc("all", lambda: walk({"type": "string"}, '"\\ud83d\\ude00 \\ud835\\udd3c"'))
assert with_uesc("all", lambda: walk({"type": "string"}, '"a\\udc00b"', expect_ok=False)) is False
assert with_uesc("all", lambda: walk({"type": "string"}, '"a\\ud83d b"', expect_ok=False)) is False
print("surrogate pairs ok, lone surrogates rejected")
try:
    with_uesc("bogus", lambda: G.matcher(S8))
    raise AssertionError("bad QE_JSON_UESC accepted")
except grammar.GrammarError:
    pass

# ---- bug 2: Job.feed finish reasons of a constrained request
import app  # noqa: E402  (imports only: Engine loads libqengine.so in its constructor)


class FakeEngine:
    def __init__(self):
        self.io = np.zeros(32, dtype=np.int32)
        self.masks = np.zeros((8, (V + 31) // 32), dtype=np.uint32)
        self.slots = {0: 1 << 20}

    def spec_io(self, slot):
        return self.io

    def mask_buffer(self, slot):
        return self.masks


def job(schema, max_new=4000, loop_guard=600, stops=()):
    g = types.SimpleNamespace(eos=set(EOS), engine=FakeEngine(), tok=tok,
                              args=types.SimpleNamespace(mtp_k=3, loop_guard_tokens=loop_guard))
    j = app.Job.__new__(app.Job)
    j.gen, j.ids, j.p, j.slot, j.finished = g, [1] * 100, {}, 0, False
    j.on_text, j.max_new, j.stops = (lambda s: None), max_new, list(stops)
    j.grammar, j.grammar_s, j.gsteps = G.matcher(schema), 0.0, 0
    j.det = app.Detok(tok)
    j.n, j.finish, j.text_all, j.cut, j.loop_checked = 0, "length", "", False, 0
    j.grammar.fill(g.engine.masks[0])
    return j


def run(j, ids, step=4):
    for i in range(0, len(ids), step):
        j.feed(ids[i:i + step])
        if j.finished:
            break
    return j


# complete JSON + EOS: stop, valid
text = compact(PL, False)
j = run(job(S8), tok.encode(text, add_special_tokens=False).ids + [EOS[1]], step=1)
assert j.finished and j.finish == "stop" and json.loads(j.text_all) == PL, (j.finish, j.text_all[-80:])
# the reported failure: '...wi' then a \u0019 loop -> the loop guard ends it: length, not stop
head = '{"items": [{"chunk": 1, "context": "x", "questions": ["Jakie etyczne wyzwania wi'
ids = tok.encode(head, add_special_tokens=False).ids + tok.encode("\\u0019" * 200, add_special_tokens=False).ids
j = run(job(S8), ids, step=1)
assert j.finished and j.finish == "length", j.finish
assert j.n < len(ids), "the loop guard did not fire"
print(f"loop guard on a constrained request: stopped after {j.n} tokens, finish {j.finish}")
# max_tokens inside the JSON: length (unchanged)
j = run(job(S8, max_new=40), tok.encode(text, add_special_tokens=False).ids, step=3)
assert j.finished and j.finish == "length" and j.n == 40
# a stop sequence the client asked for keeps 'stop' (OpenAI semantics: the client cut it)
j = run(job(S8, stops=["wiążą"]), tok.encode(text, add_special_tokens=False).ids, step=2)
assert j.finished and j.finish == "stop" and j.cut
# a grammar-rejected token (should never happen: the mask is exact): length, logged
j = job(S8)
j.feed(tok.encode('{"items": [{"chunk": 1, "context": "x"', add_special_tokens=False).ids)
j.feed(tok.encode("}}", add_special_tokens=False).ids)
assert j.finished and j.finish == "length" and j.grammar.error, (j.finish, j.grammar.error)
print("finish reasons ok (stop only for complete JSON or a client stop sequence)")
print("ok (json utf8)")
