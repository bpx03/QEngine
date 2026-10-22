"""Sequences joining and leaving a batch while a batched step is queued ahead (server-like: at
most W running, a new prompt prefilled with start_begin / prefill_more / start_finish as soon as
one ends). The queued step must survive both (release undoes only the ended sequence, a prefill
keeps the others' step), and every greedy output must equal the sequence decoded alone.
Usage: python3 test_batch_churn.py [W] [rounds]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
from transformers import AutoTokenizer

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
W = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
K = 7
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
]
jobs = []  # (ids, tokens to generate): different lengths, so they end at different steps
for r in range(ROUNDS):
    for i, p in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": f"[{r}] {p}"}],
                                      add_generation_prompt=True, enable_thinking=False)
        jobs.append((ids, 60 + 37 * ((i * 5 + r * 3) % 8)))
e = Engine(MODEL)
e.set_sampling(0)
e.set_concurrent(True)


def alone(ids, n):
    t, _, s = e.start(ids, K)
    out = [t]
    while len(out) < n:
        out += e.step()
    e.release(s)
    return out[:n]


ref = [alone(ids, n) for ids, n in jobs]
e.reset()  # no cached prefixes below

outs, live, nxt, joins, leaves, steps = {}, [], 0, 0, 0, 0
t0 = time.time()
while nxt < len(jobs) or live:
    if nxt < len(jobs) and len(live) < W:  # a step of the others may be queued ahead here
        j = nxt
        nxt += 1
        s = e.start_begin(jobs[j][0], K)
        while not e.prefill_more(1 << 30):
            pass
        t, _, s = e.start_finish()
        outs[j] = [t]
        live.append((j, s))
        joins += 1
    if not live:
        continue
    e.set_lookahead(True)
    if len(live) == 1:
        j, s = live[0]
        e.resume(s)
        outs[j] += e.step()
    else:
        for (j, s), r in zip(live, e.step_batch([s for _, s in live])):
            outs[j] += r
    steps += 1
    for j, s in list(live):
        if len(outs[j]) >= jobs[j][1]:
            e.release(s)  # mid-flight: a step including s is queued ahead
            live.remove((j, s))
            leaves += 1
tb = time.time() - t0

ok = True
for j, (ids, n) in enumerate(jobs):
    o, r = outs[j][:n], ref[j]
    d = next((i for i in range(n) if r[i] != o[i]), n)
    if d < n:
        ok = False
        print(f"job {j}: match {d}/{n}")
        print("  alone:", repr(tok.decode(r[max(0, d - 5):d + 10])))
        print("  churn:", repr(tok.decode(o[max(0, d - 5):d + 10])))
tot = sum(min(len(outs[j]), n) for j, (_, n) in enumerate(jobs))
print(f"{len(jobs)} jobs, {joins} joins, {leaves} leaves, {steps} steps, {tb:.1f}s, {tot / tb:.0f} tok/s")
print("PASS" if ok else "FAIL")
e.close()
