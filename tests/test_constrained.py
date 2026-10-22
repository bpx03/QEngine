"""GPU: grammar-constrained decoding in the engine (response_format, stage 2: speculative).
1. an all-allowed mask with the model's drafts passed through reproduces plain greedy decoding
   on the constrained path (host-planned drafts, per-row masks);
2. a JSON-schema grammar (llguidance) yields valid JSON, alone and batched with a normal sequence,
   identical with drafts (forced tokens + grammar-valid MTP drafts) and without (kmax 0);
3. the normal sequence batched with a constrained one keeps its output and its drafts;
4. after release the slot decodes on the normal (drafted) path again.
Needs llguidance + jsonschema.  python tests/test_constrained.py [tokens]"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
from grammar import Grammars
from tokenizers import Tokenizer
import jsonschema

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 160
K = 7
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
EOS = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}


def chat(user):  # Qwen3.8 chat template without thinking
    return tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      add_special_tokens=False).ids


P_NORMAL = chat("Explain in detail how a CPU branch predictor works, including TAGE.")
P_JSON = chat("List the three largest Polish cities with their population and voivodeship. "
              "Answer in JSON.")
SCHEMA = {"type": "object", "properties": {"cities": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
    "type": "object", "properties": {"name": {"type": "string"}, "population": {"type": "integer", "minimum": 0},
                                     "voivodeship": {"type": ["string", "null"]},
                                     "capital": {"type": "boolean"}},
    "required": ["name", "population", "voivodeship", "capital"], "additionalProperties": False}}},
    "required": ["cities"], "additionalProperties": False}

e = Engine(MODEL)
e.set_sampling(0)
e.set_concurrent(True)
G = Grammars(open(os.path.join(MODEL, "tokenizer.json")).read(), e.vocab(), sorted(EOS))


class Seq:
    def __init__(self, ids, schema=None, all_mask=False, affinity=0, kmax=K):
        self.slot = e.start_begin(ids, K, affinity)
        self.g = G.matcher(schema) if schema else None
        self.all_mask, self.kmax = all_mask, kmax
        self.steps = self.multi = 0
        if self.g or all_mask:
            self.masks, self.io = e.mask_buffer(self.slot), e.spec_io(self.slot)
            if self.g:
                self.g.fill(self.masks[0])
            else:
                self.masks[:] = 0xFFFFFFFF
            e.constrain(self.slot, True)
        while not e.prefill_more(1 << 30, self.slot):
            pass
        t, _, _ = e.start_finish(self.slot)
        self.out, self.done = [], False
        self.feed([t])

    def feed(self, toks):
        if not toks or self.done:
            return
        self.steps += 1
        self.multi += len(toks) > 1
        for t in toks:
            if t in EOS or len(self.out) >= N:
                self.done = True
                break
            if self.g:
                assert self.g.consume(t), self.g.error
            self.out.append(t)
            if self.g and self.g.done():
                self.done = True
                break
        if self.g and not self.done:
            self.g.plan(self.io, self.masks, self.kmax)
        elif self.all_mask and not self.done:  # the model's drafts as they are
            n = min(int(self.io[0]), self.kmax)
            self.io[17:17 + n] = self.io[1:1 + n]
            self.io[16] = n

    def text(self):
        return tok.decode(self.out)


def run(seqs):
    while not all(s.done for s in seqs):
        live = [s for s in seqs if not s.done]
        if len(live) == 1:
            e.resume(live[0].slot)
            live[0].feed(e.step())
        else:
            for s, r in zip(live, e.step_batch([s.slot for s in live])):
                s.feed(r)
        for s in seqs:
            if s.done and s.slot >= 0:
                e.release(s.slot)
                s.slot_done, s.slot = s.slot, -1


ok = True
# 1. plain greedy reference vs the constrained path with an all-allowed mask
ref = Seq(P_NORMAL)
t0 = time.time(); run([ref]); t_ref = time.time() - t0
print(f"normal alone: {len(ref.out)} tokens in {ref.steps} steps ({len(ref.out) / ref.steps:.2f} tok/step), "
      f"{len(ref.out) / t_ref:.0f} tok/s")
al = Seq(P_NORMAL, all_mask=True)
t0 = time.time(); run([al]); t_al = time.time() - t0
same = al.out == ref.out
first = next((i for i, (a, b) in enumerate(zip(al.out, ref.out)) if a != b), None)
print(f"all-allowed mask: {len(al.out)} tokens in {al.steps} steps ({len(al.out) / al.steps:.2f} tok/step), "
      f"{len(al.out) / t_al:.0f} tok/s, "
      f"same as greedy: {same}" + ("" if same else f" (first difference at {first})"))
ok &= al.multi > 0  # drafts verified on the constrained path
ok &= same or (first is not None and first > 40)  # near-tie flips only far into the text

# 2. grammar alone
js = Seq(P_JSON, SCHEMA)
t0 = time.time(); run([js]); t_js = time.time() - t0
try:
    jsonschema.validate(json.loads(js.text()), SCHEMA)
    print(f"json alone: valid, {len(js.out)} tokens in {js.steps} steps ({len(js.out) / js.steps:.2f} tok/step, "
          f"{js.g.n_forced} forced drafts), {len(js.out) / t_js:.0f} tok/s: {js.text()[:200]}")
except Exception as ex:
    ok = False
    print(f"json alone: INVALID ({ex!r}): {js.text()[:400]!r}")
js0 = Seq(P_JSON, SCHEMA, kmax=0)  # stage-1 behaviour: one token per step
t0 = time.time(); run([js0]); t_js0 = time.time() - t0
print(f"json without drafts: {len(js0.out)} tokens in {js0.steps} steps, {len(js0.out) / t_js0:.0f} tok/s, "
      f"same as with drafts: {js0.out == js.out}")
ok &= js0.out == js.out and js.steps < js0.steps

# 3. normal + constrained batched together
a, b = Seq(P_NORMAL), Seq(P_JSON, SCHEMA, affinity=777)
t0 = time.time(); run([a, b]); t_ab = time.time() - t0
try:
    jsonschema.validate(json.loads(b.text()), SCHEMA)
    vb = True
except Exception as ex:
    vb = False
print(f"batched: normal {len(a.out)} tokens in {a.steps} steps ({len(a.out) / a.steps:.2f} tok/step, "
      f"same as alone: {a.out == ref.out}); json valid: {vb}, same as alone: {b.out == js.out}, "
      f"{b.steps} steps")
ok &= vb and a.out == ref.out and len(a.out) / a.steps > 1.5

# 4. the constrained slot returns to the normal path (same prompt as the json request: same slot)
c = Seq(P_JSON, affinity=777)
reuse = c.slot == b.slot_done
run([c])
print(f"normal on the former constrained slot ({'same slot' if reuse else 'other slot'}): "
      f"{len(c.out)} tokens in {c.steps} steps ({len(c.out) / c.steps:.2f} tok/step)")
ok &= c.multi > 0 and not e.constrained(b.slot_done)
print("OK" if ok else "FAILED")
e.close()
sys.exit(0 if ok else 1)
