"""Prefill interleaved with decoding (server --prefill-chunk): while C's prompt is prefilled in
small chunks, A and B keep decoding between the chunks. A's and B's greedy outputs must equal
their outputs decoded without C; C's output is compared with C prefilled at once (other chunk
boundaries: numerically close, may flip at a near-tie).  python3 test_interleave.py [chunk] [N]"""
import os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
A = tok.encode("Explain this code briefly:\n" + text[:6000]).ids
B = tok.encode("List the functions defined here:\n" + text[6000:12000]).ids
C = tok.encode("Summarize this file in detail:\n" + text[12000:90000]).ids
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
e.set_concurrent(True)
print("prompts", len(A), len(B), len(C), flush=True)


def alone(ids):
    t, _, s = e.start(ids, 3)
    out = [t]
    while len(out) < N:
        out += e.step()
    e.release(s)
    return out[:N]


ref = {k: alone(v) for k, v in (("A", A), ("B", B), ("C", C))}
e.reset()  # no cached prefixes: everything is computed again below

outs, slots = {}, {}
for k, ids in (("A", A), ("B", B)):
    t, _, s = e.start(ids, 3)
    outs[k], slots[k] = [t], s
for _ in range(10):  # A and B are decoding when C arrives
    for k, r in zip("AB", e.step_batch([slots["A"], slots["B"]])):
        outs[k] += r
sc = e.start_begin(C, 3)
e.set_lookahead(False)
t0, steps_during, chunks = time.time(), 0, 0
while True:
    chunks += 1
    if e.prefill_more(CHUNK):
        break
    for k, r in zip("AB", e.step_batch([slots["A"], slots["B"]])):
        outs[k] += r
    steps_during += 1
e.set_lookahead(True)
tc, _, _ = e.start_finish()
prefill_s = time.time() - t0
outs["C"], slots["C"] = [tc], sc
gen_during = {k: len(outs[k]) for k in "AB"}
while min(len(o) for o in outs.values()) < N:
    live = [k for k in "ABC" if len(outs[k]) < N]
    res = e.step_batch([slots[k] for k in live]) if len(live) > 1 else None
    if res is None:
        e.resume(slots[live[0]])
        res = [e.step()]
    for k, r in zip(live, res):
        outs[k] += r
print(f"C: {len(C)} tokens prefilled in {chunks} chunks of {CHUNK} in {prefill_s:.1f}s "
      f"({len(C) / prefill_s:.0f} tok/s) while A and B made {steps_during} decode steps "
      f"(A had {gen_during['A']} tokens, B {gen_during['B']} when C's prefill ended)")
ok = True
for k in "ABC":
    d = next((j for j in range(N) if outs[k][j] != ref[k][j]), N)
    print(f"{k}: identical to its uninterleaved run for {d}/{N} tokens")
    if k != "C":
        ok &= d == N
print("PASS" if ok else "FAIL")
e.close()
