"""CPU-only: structured output plumbing (response_format -> llguidance grammar -> uint32 token
masks in the engine's layout). Needs llguidance + jsonschema.  python tests/test_grammar.py"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from grammar import GrammarError, Grammars, schema_of
from tokenizers import Tokenizer
import jsonschema

MODEL = os.environ.get("QE_MODEL", os.path.join(os.path.dirname(__file__), "..", "models/Qwen3.8-27B-INT4"))
V = int(json.load(open(os.path.join(MODEL, "config.json"))).get("text_config", {}).get("vocab_size", 248320))
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
EOS = [tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|im_end|>")]
G = Grammars(open(os.path.join(MODEL, "tokenizer.json")).read(), V, EOS)
buf = np.zeros((V + 31) // 32, dtype=np.uint32)  # same layout as Engine.mask_buffer


def allowed(t):
    return bool((buf[t >> 5] >> np.uint32(t & 31)) & 1)


def walk(schema, text, expect_ok=True):
    """Emit text's tokens one by one like the server: every token must be in the mask that was
    computed before it; at the end the grammar is complete (only EOS allowed)."""
    c = G.matcher(schema)
    c.fill(buf)
    ids = tok.encode(text, add_special_tokens=False).ids
    t0 = time.perf_counter()
    for i, t in enumerate(ids):
        if not allowed(t):
            assert not expect_ok, f"token {i} {tok.decode([t])!r} not allowed in {text[:80]!r}"
            return False
        assert c.consume(t), c.error
        if c.done():
            assert i == len(ids) - 1, "grammar complete before the text ends"
            break
        c.fill(buf)
    dt = (time.perf_counter() - t0) / max(1, len(ids))
    assert expect_ok, "an invalid text was accepted"
    # complete: only the end tokens remain (or done() already)
    if not c.done():
        assert c.accepting()
        assert any(allowed(e) for e in EOS)
        n_allowed = int(np.unpackbits(buf.view(np.uint8)).sum())
        assert n_allowed <= len(EOS) + 1, n_allowed
    jsonschema.validate(json.loads(text), schema)
    return dt


# ---- request parsing
BATCH = {"type": "object", "properties": {"items": {"type": "array", "minItems": 10, "maxItems": 10, "items": {
    "type": "object", "properties": {"chunk": {"type": "integer", "minimum": 1, "maximum": 10},
                                     "context": {"type": "string"},
                                     "questions": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
    "required": ["chunk", "context", "questions"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}
assert schema_of({}) is None
assert schema_of({"response_format": {"type": "text"}}) is None
assert schema_of({"response_format": {"type": "json_object"}}) == {"type": "object"}
assert schema_of({"response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True,
                                                                             "schema": BATCH}}}) == BATCH
assert schema_of({"output_format": {"type": "json_schema", "schema": BATCH}}) == BATCH
assert schema_of({"output_config": {"format": {"type": "json_schema", "schema": BATCH}}}) == BATCH
for bad in ({"response_format": "json"}, {"response_format": {"type": "xml"}}):
    try:
        schema_of(bad)
        raise AssertionError(bad)
    except GrammarError:
        pass

# ---- the contextualize batch schema (10 items)
item = lambda i: {"chunk": i, "context": "Section 3.2 on write-ahead logging: \"it\" refers to the log; "
                  "zażółć gęślą jaźń \\ ok.", "questions": ["Jak baza zapewnia trwałość?", "Why log first?"]}
good = json.dumps({"items": [item(i) for i in range(1, 11)]}, ensure_ascii=False)
dt = walk(BATCH, good)
print(f"batch schema: {len(tok.encode(good).ids)} tokens ok, {1e6 * dt:.0f} us/token (consume + mask)")
assert walk(BATCH, json.dumps([item(i) for i in range(1, 11)]), expect_ok=False) is False  # bare list
assert walk(BATCH, json.dumps({"items": [item(i) for i in range(1, 10)]}), expect_ok=False) is False  # 9 items
bad = [item(i) for i in range(1, 11)]
bad[3]["chunk"] = 11
assert walk(BATCH, json.dumps({"items": bad}), expect_ok=False) is False  # maximum
bad = [item(i) for i in range(1, 11)]
bad[0]["extra"] = 1
assert walk(BATCH, json.dumps({"items": bad}), expect_ok=False) is False  # additionalProperties
c = G.matcher(BATCH)
c.fill(buf)
assert not any(allowed(e) for e in EOS), "EOS allowed before the grammar is complete"
assert allowed(tok.encode("{\"", add_special_tokens=False).ids[0])
assert not allowed(tok.encode("[", add_special_tokens=False).ids[0])

# ---- query-path schemas: enum, boolean, number, union types with null
# optional: a real answer schema (question / language / answer / insufficient_evidence ...) via QE_ANSWER_SCHEMA
ANSWER = json.load(open(os.environ["QE_ANSWER_SCHEMA"])) if os.environ.get("QE_ANSWER_SCHEMA") else None
UNION = {"type": "object", "properties": {
    "language": {"type": "string", "enum": ["pl", "en"]},
    "ok": {"type": "boolean"}, "score": {"type": "number"},
    "title": {"type": ["string", "null"]}, "page": {"type": ["integer", "string", "null"]},
    "conf": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["language", "ok", "score", "title", "page", "conf"], "additionalProperties": False}
walk(UNION, json.dumps({"language": "pl", "ok": True, "score": -1.5e-3, "title": None, "page": "xii", "conf": 100}))
walk(UNION, json.dumps({"language": "en", "ok": False, "score": 3, "title": "T", "page": 7, "conf": 0}))
assert walk(UNION, json.dumps({"language": "de", "ok": True, "score": 1, "title": None, "page": 1, "conf": 1}),
            expect_ok=False) is False
assert walk(UNION, json.dumps({"language": "pl", "ok": True, "score": 1, "title": None, "page": 1, "conf": 101}),
            expect_ok=False) is False
if ANSWER:  # properties are emitted in schema order (llguidance, like xgrammar / vLLM)
    walk(ANSWER, json.dumps({"question": "q", "language": "pl", "answer": "a [1]", "insufficient_evidence": False,
                             "confidence": 80, "citations": [{"n": 1, "doc": "d", "title": None, "printed_page": "iv",
                                                              "pdf_page": 3, "quote": "x", "chunk_id": "c1"}],
                             "retrieved": [{"chunk_id": "c1", "score": 0.5, "doc": None}],
                             "meta": {"level": "03-agentic", "latency_ms": 12.5}}))
    print("answer.schema.json (union types with null) ok")

# ---- json_object: any object
walk({"type": "object"}, json.dumps({"a": [1, 2, {"b": None}], "c": "x"}))
assert walk({"type": "object"}, "[1, 2]", expect_ok=False) is False

# ---- unusable schema -> GrammarError (HTTP 400)
try:
    G.matcher({"type": "object", "properties": {"a": {"$ref": "#/nope"}}, "required": ["a"]})
    raise AssertionError("bad $ref accepted")
except GrammarError as ex:
    print("bad schema ->", str(ex)[:80])

# ---- per-request cost
t0 = time.perf_counter()
for _ in range(20):
    G.matcher(BATCH).fill(buf)
print(f"grammar compile + first mask: {1e3 * (time.perf_counter() - t0) / 20:.2f} ms")
print("ok")

# ---- stage 2: speculative plan (engine spec_io / mask_buffer layout), simulated verification
import random
masks = np.zeros((8, (V + 31) // 32), dtype=np.uint32)
io = np.zeros(32, dtype=np.int32)


def simulate(schema, text, draft_quality, seed=0):
    """The 'model' emits text's tokens; its MTP drafts are the true next tokens with probability
    draft_quality each (else a wrong token). Every step verifies the planned drafts with the row
    masks like the engine (accept while draft == model token, then one bonus token)."""
    rnd = random.Random(seed)
    ids = tok.encode(text, add_special_tokens=False).ids
    c = G.matcher(schema)
    c.fill(masks[0])
    pos, steps, forced = 0, 0, 0

    def model_drafts(p):
        d = []
        for j in range(7):
            if p + j >= len(ids):
                break
            d.append(ids[p + j] if rnd.random() < draft_quality else rnd.randrange(1000, 200000))
        return d

    first = True
    while pos < len(ids):
        steps += 1
        if first:  # prompt end: no drafts verified yet, the first token from mask 0
            planned, first = [], False
        else:
            planned = [int(x) for x in io[17:17 + io[16]]]
        emitted = []
        for j in range(len(planned) + 1):  # row j: the model's token (must be allowed by mask j)
            t = ids[pos + j]
            assert allowed(masks[j], t), f"row {j} mask rejects the model token {tok.decode([t])!r}"
            emitted.append(t)
            if j < len(planned) and planned[j] != t:
                break
            if pos + j + 1 >= len(ids):
                break
        for t in emitted:
            assert c.consume(t), c.error
        pos += len(emitted)
        if c.done():
            break
        d = model_drafts(pos)
        io[0] = len(d)
        io[1:1 + len(d)] = d
        n = c.plan(io, masks, 7)
        for j in range(n):  # planned drafts are grammar-valid prefixes
            assert allowed(masks[j], int(io[17 + j]))
    assert pos == len(ids) and c.done() or c.accepting(), (pos, len(ids))
    jsonschema.validate(json.loads(text), schema)
    return steps, c.n_forced


from grammar import allowed
n_tok = len(tok.encode(good, add_special_tokens=False).ids)
for q in (0.0, 0.7, 1.0):
    t0 = time.perf_counter()
    steps, forced = simulate(BATCH, good, q)
    print(f"spec plan, draft quality {q}: {n_tok} tokens in {steps} steps ({n_tok / steps:.2f} tok/step), "
          f"{forced} forced drafts, {1e6 * (time.perf_counter() - t0) / steps:.0f} us/step")
steps, _ = simulate(UNION, json.dumps({"language": "pl", "ok": True, "score": -1.5e-3, "title": None, "page": "xii", "conf": 100}), 0.8)
print("spec plan with null unions ok")
print("ok (stage 2)")
