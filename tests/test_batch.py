"""Batched speculative steps for concurrent sequences must reproduce each sequence decoded alone
(greedy). Usage: python3 test_batch.py [tokens] [nseq]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
from transformers import AutoTokenizer

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
tok = AutoTokenizer.from_pretrained(MODEL)
prompts = [
    "Write a Python function that parses an INI file into a nested dict, with tests.",
    "Explain in detail how a CPU branch predictor works, including TAGE.",
    "Napisz krótkie opowiadanie o latarniku na Bałtyku.",
    "List 30 countries with their capitals as a JSON array of objects.",
    "Implement a thread-safe LRU cache in C++17 with a short explanation.",
    "Describe the causes and consequences of the 1929 stock market crash.",
    "Write a bash script that backs up a directory with rotation of 7 copies.",
    "Explain how TCP congestion control works (Reno, CUBIC, BBR).",
    "Write a SQL schema for a library system and three example queries with joins.",
    "Wyjaśnij, jak działa szyfrowanie asymetryczne i podpis cyfrowy, z przykładem.",
    "Implement binary search in Rust with unit tests and explain the edge cases.",
    "Summarize the plot of Hamlet act by act in a numbered list.",
]
# more than 12 sequences (QE_MAX_SEQS >= NS): the prompts again with a variant suffix
prompts = [prompts[i % 12] + (f" (variant {i // 12})" if i >= 12 else "") for i in range(NS)]
ids = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                               enable_thinking=False) for p in prompts]
e = Engine(MODEL)
e.set_sampling(0)

def alone(x):
    t, _, _ = e.start(x, 6)
    out = [t]
    while len(out) < N:
        out += e.step()
    return out[:N]

ref = []
t0 = time.time()
for x in ids:
    ref.append(alone(x))
ta = time.time() - t0

e.set_concurrent(True)
outs, slots = [], []
t0 = time.time()
for x in ids:
    t, _, s = e.start(x, 6)
    outs.append([t]); slots.append(s)
assert len(set(slots)) == len(slots), slots
tp = time.time() - t0
steps = 0
t0 = time.time()
while min(len(o) for o in outs) < N:
    res = e.step_batch(slots)
    steps += 1
    for o, r in zip(outs, res):
        o += r
tb = time.time() - t0
for s in slots:
    e.release(s)
ok = True
for i, (r, o) in enumerate(zip(ref, outs)):
    o = o[:N]
    d = next((j for j in range(N) if r[j] != o[j]), N)
    print(f"seq {i} slot {slots[i]}: match {d}/{N}")
    if d < N:
        ok = False
        print("  alone:", repr(tok.decode(r[max(0, d - 5):d + 10])))
        print("  batch:", repr(tok.decode(o[max(0, d - 5):d + 10])))
tot = sum(min(len(o), N) for o in outs)
print(f"alone (incl prefill) {ta:.2f}s; batch prefill {tp:.2f}s, decode {tb:.2f}s, "
      f"{steps} steps, {tot / tb:.0f} tok/s aggregate, {tb / steps * 1000:.2f} ms/step")
print("PASS" if ok else "FAIL")
e.close()
