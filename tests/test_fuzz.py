"""Randomized engine stress: conversations with random append sizes, new conversations, slot
switches, batched steps and sampling; checks for CUDA errors and, periodically, that a cached
continuation's next-token logits match a fresh computation. usage: python3 test_fuzz.py [rounds] [seed]"""
import ctypes, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
R = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
M = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
e = Engine(M)
e.lib.qe_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
V = 248320
def logits():
    a = np.zeros(V, np.float32)
    e.lib.qe_logits(e.h, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    return a
def rand_tokens(n):
    return [rng.randrange(1000, 150000) for _ in range(n)]
convs = [rand_tokens(rng.choice([50, 400, 3000, 9000])) for _ in range(4)]
hist = [[] for _ in range(64)]  # operations per conversation (by object id)
H = {}
def note(i, what):
    H.setdefault(id(convs[i]), []).append(what)
e.set_concurrent(True)
bad = 0
t0 = time.time()
for r in range(R):
    op = rng.random()
    if op < 0.15:  # batched decode of 2-3 conversations
        n = rng.choice([2, 3])
        idx = rng.sample(range(len(convs)), n)
        slots = []
        e.set_sampling(rng.choice([0, 0.7]))
        for i in idx:
            t, _, s = e.start(convs[i], 7)
            slots.append(s); convs[i].append(t)
            note(i, f"batch-start s{s}")
        for _ in range(rng.randrange(5, 40)):
            res = e.step_batch(slots)
            for i, toks in zip(idx, res): convs[i] += toks
        # release one early (batch set changes) then continue the rest
        e.release(slots[0])
        note(idx[0], "released-mid-batch")
        for _ in range(rng.randrange(1, 10)):
            res = e.step_batch(slots[1:])
            for i, toks in zip(idx[1:], res): convs[i] += toks
        for s in slots[1:]: e.release(s)
        for i in idx[1:]: note(i, "batch-end")
        # the generated tokens are only valid as part of the conversation up to what was emitted
        continue
    i = rng.randrange(len(convs))
    if op < 0.25:
        convs.append(rand_tokens(rng.choice([20, 300, 2500])))
        i = len(convs) - 1
        if len(convs) > 8: convs.pop(0); i -= 1
    convs[i] += rand_tokens(rng.choice([1, 3, 9, 17, 33, 64, 100, 127, 129, 200, 255, 256, 513, 1000, 2047, 2049, 3000]))
    e.set_sampling(rng.choice([0, 0.6, 1.0]))
    verify = rng.random() < 0.3
    # (with drafts the prompt's MTP pass overwrites the logits buffer: verify with k = 0)
    kk = 0 if verify else rng.choice([0, 3, 7])
    t, reused, s = e.start(convs[i], kk)
    note(i, f"start k{kk} s{s} reused{reused} arch{int(e.last_from_archive())} copy{e.last_copied()}")
    if verify:  # against a fresh computation
        a = logits()
        e.release(s)
        e.set_concurrent(False); e.reset(); e.set_sampling(0)
        e.start(convs[i], 0); b = logits()
        e.set_concurrent(True)
        cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        print(f"round {r}: conv {i} len {len(convs[i])} reused {reused} slot {s}: cos vs fresh {cos:.5f}")
        bad += cos < 0.995
        if cos < 0.995: print("   history:", H.get(id(convs[i]), [])[-6:])
        continue
    out = [t]
    ns = rng.randrange(1, 30)
    for _ in range(ns): out += e.step()
    convs[i] += out
    note(i, f"single {ns} steps")
    e.release(s)
print(f"{R} rounds in {time.time() - t0:.0f}s; {'PASS' if not bad else f'FAIL ({bad} mismatches)'}")
e.close()
