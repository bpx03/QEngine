"""KV pool under pressure (small pool, 4 slots): regions grow, move (compaction), get trimmed and
evicted while sequences decode. Greedy results must not depend on any of it:
  1. 4 conversations decoded alone (each grows past its first region),
  2. the same 4 decoded together (batched steps; regions move between steps) == alone,
  3. a long 5th conversation evicts the idle ones to host memory; conversation 0 continued
     afterwards resumes from the archive with the same next-token logits as a fresh prefill.
Usage: QE_POOL_TOKENS=65536 QE_MAX_SEQS=4 python3 test_pool.py"""
import ctypes, os, sys
import numpy as np
os.environ.setdefault("QE_POOL_TOKENS", "65536")
os.environ.setdefault("QE_MAX_SEQS", "4")
os.environ.setdefault("QE_POOL_HEADROOM", "256")   # small regions: they grow (and move) often
os.environ.setdefault("QE_POOL_GROW_FRAC", "0")
# fixed draft count: adaptive k follows acceptance statistics shared by all sequences, so the
# alone and batched runs would verify different row counts and may flip a near-tie
os.environ.setdefault("QE_MTP_ADAPT", "0")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
N = int(os.environ.get("POOL_GEN", "1200"))
prompts = [tok.encode(f"Conversation {i}. Explain this code in detail, function by function:\n"
                      + text[i * 15000:(i + 1) * 15000]).ids for i in range(4)]
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
print("pool", e.pool_tokens, "tokens; prompts", [len(p) for p in prompts], flush=True)

def alone(ids):
    t, _, s = e.start(ids, 3)
    out = [t]
    while len(out) < N:
        out += e.step()
    e.release(s) if hasattr(e, "release") else None
    return out  # every generated token (a step may overshoot N; the engine processed them all)

ok = True
e.set_concurrent(True)
ref = [alone(p) for p in prompts]
print("alone: done")
outs, slots = [], []
fresh = [p + tok.encode(" (second run)").ids for p in prompts]  # not served from the caches
ref2 = []
for p in fresh:  # alone references of the fresh prompts
    ref2.append(alone(p))
for p in fresh:
    t, _, s = e.start(p, 3)
    outs.append([t]); slots.append(s)
while min(len(o) for o in outs) < N:
    for o, r in zip(outs, e.step_batch(slots)):
        o += r
for s in slots:
    e.release(s)
for i, (a, b) in enumerate(zip(ref2, outs)):
    d = next((j for j in range(N) if a[j] != b[j]), N)
    print(f"seq {i}: batched == alone for {d}/{N} tokens")
    ok &= d == N

# 3. eviction: a long conversation needs most of the pool
big = tok.encode("Summarize:\n" + text * 4).ids[:52000]  # more than the idle slots leave free
t, _, s = e.start(big, 3)
for _ in range(20):
    e.step()
e.release(s)
cont = prompts[0] + ref[0] + tok.encode(" Continue.").ids
V = 248320
e.lib.qe_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

def logits(ids):
    _, reused, s = e.start(ids, 0)
    a = np.zeros((8, V), np.float32)
    e.lib.qe_logits(e.h, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    e.release(s)
    return a[0], reused, e.last_from_archive()

la, reused, arch = logits(cont)
cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
# reference 1: the same conversation continued from its live state (never evicted): the KV of
# the reply rows comes from the same decode steps, so the archive round trip must be exact
e.reset()
alone(prompts[0])
ll, reused_l, arch_l = logits(cont)
c_live = cos(la, ll)
# reference 2: everything prefilled from scratch. The reply rows then come from the prefill
# GEMM instead of the decode GEMV: with the INT8 prefill (default) the two differ by design
# (cos ~0.9997, the accepted INT8/fp difference); with QE_PREFILL_INT8=0 they must agree
e.reset()
lb, _, _ = logits(cont)
c_fresh = cos(la, lb)
int8 = os.environ.get("QE_PREFILL_INT8", "1") != "0"
print(f"resume after eviction: reused {reused}/{len(cont)} (from archive: {arch}), logits cos vs live "
      f"continuation {c_live:.6f} (reused {reused_l}, archive {arch_l}), vs fresh prefill {c_fresh:.6f} "
      f"(INT8 prefill {'on' if int8 else 'off'})")
ok &= reused > len(prompts[0]) and arch and reused_l >= reused and not arch_l
ok &= c_live > 0.9999 and c_fresh > (0.999 if int8 else 0.9999)
print("PASS" if ok else "FAIL")
e.close()
