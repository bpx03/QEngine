"""Two prefills in flight (branch cache-priority): while C's long prompt is prefilled in chunks
(A and B decoding between the chunks), D - a conversation with a short new turn - is started,
prefilled to completion and decoded, then C's prefill continues. A's and B's greedy outputs must
equal their runs without C and D; D's and C's must equal their uninterleaved runs (chunk
boundaries differ: compared with tolerance like test_interleave). Also checks probe_cached()
against the reuse start() reports and that prefill_pos() tracks the long prompt.
usage: python3 tests/test_preempt.py [chunk] [N]"""
import os, sys, time
os.environ.setdefault("QE_MTP_ADAPT", "0")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
chat = lambda s: tok.encode(f"<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
A = chat("Explain this code briefly:\n" + text[:6000])
B = chat("List the functions defined here:\n" + text[6000:12000])
C = chat("Summarize this file in detail:\n" + (text * 3)[12000:150000])
D1 = chat("Describe this code:\n" + text[20000:30000])
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
e.set_concurrent(True)
print("prompts", len(A), len(B), len(C), len(D1), flush=True)


def alone(ids, n=N):
    t, r, s = e.start(ids, 3)
    out = [t]
    while len(out) < n:
        out += e.step()
    e.release(s)
    return out, r


ref = {k: alone(v)[0][:N] for k, v in (("A", A), ("B", B), ("C", C))}
d1_out, _ = alone(D1, 40)  # D's first turn (its reply stays in the slot's live state)
D2 = D1 + d1_out + tok.encode("<|im_end|>\n<|im_start|>user\nNow list its functions.<|im_end|>\n"
                              "<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
ref["D"], rD_alone = alone(D2)
ref["D"] = ref["D"][:N]
e.reset()
d1_out2, _ = alone(D1, 40)  # again (reset forgot everything): D2 will extend this live state
assert d1_out2 == d1_out

outs, slots = {}, {}
for k, ids in (("A", A), ("B", B)):
    t, _, s = e.start(ids, 3)
    outs[k], slots[k] = [t], s
for _ in range(10):
    for k, r in zip("AB", e.step_batch([slots["A"], slots["B"]])):
        outs[k] += r
sc = e.start_begin(C, 3)
e.set_lookahead(False)
chunks_c = 0
for _ in range(3):  # a few chunks of C, A and B decoding in between
    assert not e.prefill_more(CHUNK, sc)
    chunks_c += 1
    for k, r in zip("AB", e.step_batch([slots["A"], slots["B"]])):
        outs[k] += r
pos_c = e.prefill_pos(sc)
probe_d = e.probe_cached(D2)
print(f"C: {pos_c} of {len(C)} prefilled when D arrives; probe(D) = {probe_d} of {len(D2)} "
      f"(alone it reused {rD_alone})", flush=True)
t0 = time.time()
sd = e.start_begin(D2, 3)
assert e.num_prefills() == 2
while not e.prefill_more(CHUNK, sd):
    for k, r in zip("AB", e.step_batch([slots["A"], slots["B"]])):
        outs[k] += r
td, rD, _ = e.start_finish(sd)
t_d = time.time() - t0
outs["D"], slots["D"] = [td], sd
print(f"D: reused {rD}, first token after {t_d * 1e3:.0f} ms (C's prefill suspended at {e.prefill_pos(sc)})", flush=True)
assert e.prefill_pos(sc) == pos_c and e.num_prefills() == 1
live = ["A", "B", "D"]
while True:
    if e.prefill_more(CHUNK, sc):
        break
    chunks_c += 1
    for k, r in zip(live, e.step_batch([slots[k] for k in live])):
        outs[k] += r
e.set_lookahead(True)
tc, rC, _ = e.start_finish(sc)
outs["C"], slots["C"] = [tc], sc
print(f"C: done in {chunks_c} chunks, reused {rC}", flush=True)
while min(len(o) for o in outs.values()) < N:
    lv = [k for k in "ABCD" if len(outs[k]) < N]
    if len(lv) > 1:
        res = e.step_batch([slots[k] for k in lv])
    else:
        e.resume(slots[lv[0]])
        res = [e.step()]
    for k, r in zip(lv, res):
        outs[k] += r
ok = probe_d == rD
for k in "ABCD":
    d = next((j for j in range(N) if outs[k][j] != ref[k][j]), N)
    print(f"{k}: identical to its uninterleaved run for {d}/{N} tokens")
    if k in "AB":
        ok &= d == N
    else:
        ok &= d >= 20
print("PASS" if ok else "FAIL")
e.close()
