"""Prompt cache: a prompt served from cached state must match a fresh prefill.

Cached and fresh runs process some tokens through different kernels (decode GEMV/attention vs
prefill GEMM/flash attention, summing in different orders), so greedy text can flip at a
near-tie; the check compares next-token logits after the prompt (a broken cache gives cos << 1)
and reports how many generated tokens agree."""
import ctypes, os, time
os.environ.setdefault("QE_KV_FP8", "0")  # bf16 cache: tightest comparison
import numpy as np
import common as T
from tokenizers import Tokenizer

lib = T.lib
lib.qe_last_reused.argtypes = [ctypes.c_void_p]
tok = Tokenizer.from_file(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4/tokenizer.json"))
e = lib.qe_create(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4").encode(), b"0,1,2", 16384, b"")
text = open(os.path.join(T.ROOT, "tests/data/cache_text.txt")).read()  # fixed text: repo sources change
A = tok.encode("Here is some C++ code:\n" + text[:6000] + "\nQuestion: what does load_w4 do?").ids
V = 248320
buf = (ctypes.c_int * 16)()


def logits_after(ids, fresh):
    """next-token logits right after the prompt (k=0 decode step), plus reuse count"""
    if fresh:
        lib.qe_reset(e)
    lib.qe_start(e, T.ints(ids), len(ids), 0)
    r = lib.qe_last_reused(e)
    lib.qe_spec_step(e, buf)
    a = np.zeros(V, np.float32)
    lib.qe_logits(e, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    return a, r


def run(ids, fresh, n=48):
    if fresh:
        lib.qe_reset(e)
    t0 = time.time()
    out, _, _ = T.generate(e, ids, 3, n, full=True)
    return out, lib.qe_last_reused(e), time.time() - t0


def compare(name, ids, prep):
    prep()
    la, ra = logits_after(ids, False)
    lb, rb = logits_after(ids, True)
    prep()
    oa, _, ta = run(ids, False)
    ob, _, tb = run(ids, True)
    cos = float(la @ lb / np.linalg.norm(la) / np.linalg.norm(lb))
    agree = next((i for i, (x, y) in enumerate(zip(oa, ob)) if x != y), min(len(oa), len(ob)))
    print(f"{name}: reused {ra}/{len(ids)} ({ta:.2f}s vs fresh {tb:.2f}s), logits cos {cos:.6f}, "
          f"argmax {la.argmax()}/{lb.argmax()}, greedy agrees for {agree} tokens")
    return cos > 0.999 and la.argmax() == lb.argmax() and rb == 0, ra


ok = True
outA, _, _ = run(A, True)
B = A + outA + tok.encode(" Thanks. Now explain load_w16.").ids          # conversation continues
C = A + tok.encode(" Actually, explain the Worker class instead.").ids  # branch after prompt A
D = A[:800] + tok.encode(" totally different text").ids             # diverges mid-prompt


def prep_B():
    run(A, True)


def prep_C():
    run(A, True)
    run(B, False)  # live state at B; C must fall back to the snapshot at the end of A


def prep_D():
    run(A, True)


r_ok, rB = compare("B (extends A+answer)", B, prep_B)
ok &= r_ok and rB > len(A)
r_ok, rC = compare("C (branches after A)", C, prep_C)
ok &= r_ok and rC >= len(A) - 1  # prompt-end state at n or n-1 (QE_END_CUT)
r_ok, _ = compare("D (diverges at 800)", D, prep_D)
ok &= r_ok
lib.qe_destroy(e)
print("OK" if ok else "MISMATCH")
