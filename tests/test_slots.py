"""Sequence slots: main agent + subagent interleaved; each keeps its prompt cache and every
result must equal a fresh computation."""
import ctypes, os
import common as T
from tokenizers import Tokenizer

lib = T.lib
for f in ("qe_last_slot", "qe_last_reused", "qe_num_slots"):
    getattr(lib, f).argtypes = [ctypes.c_void_p]
lib.qe_start2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                          ctypes.c_ulonglong]
lib.qe_slot_capacity.argtypes = [ctypes.c_void_p, ctypes.c_int]
tok = Tokenizer.from_file(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4/tokenizer.json"))
e = lib.qe_create(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4").encode(), b"0,1,2", 262144, b"")
print("slots:", [lib.qe_slot_capacity(e, i) for i in range(lib.qe_num_slots(e))])
buf = (ctypes.c_int * 16)()

def gen(ids, aff, n=40):
    out = [lib.qe_start2(e, T.ints(ids), len(ids), 3, aff)]
    slot, reused = lib.qe_last_slot(e), lib.qe_last_reused(e)
    while len(out) < n:
        c = lib.qe_spec_step(e, buf)
        out += list(buf[:c])
    return out, slot, reused

src = open(os.path.join(T.ROOT, "tests/data/cache_text.txt")).read()  # fixed text: repo sources change
MAIN_SYS, SUB_SYS = 101, 202
main = tok.encode("System: you are the main agent.\n" + src[:9000] + "\nUser: summarize load_w4.").ids
sub = tok.encode("System: you are a search subagent.\n" + src[9000:15000] + "\nTask: list functions.").ids

ok = True
o1, s1, r1 = gen(main, MAIN_SYS)
print(f"main t1: slot {s1} reused {r1}/{len(main)}")
main2 = main + o1 + tok.encode(" Now spawn a subagent.").ids
o2, s2, r2 = gen(main2, MAIN_SYS)
print(f"main t2: slot {s2} reused {r2}/{len(main2)}")
ok &= s2 == s1 and r2 > len(main)
os1, ss1, rs1 = gen(sub, SUB_SYS)
sub2 = sub + os1 + tok.encode(" Continue.").ids
os2, ss2, rs2 = gen(sub2, SUB_SYS)
print(f"sub  t1: slot {ss1} reused {rs1} | sub t2: slot {ss2} reused {rs2}/{len(sub2)}")
ok &= ss1 != s1 and ss2 == ss1 and rs2 > len(sub)
main3 = main2 + o2 + tok.encode(" The subagent finished; continue.").ids
o3, s3, r3 = gen(main3, MAIN_SYS)
print(f"main t3 (after subagent): slot {s3} reused {r3}/{len(main3)}")
ok &= s3 == s1 and r3 > len(main2)
# Recomputation must agree. Prefill and decode kernels sum in different orders, so greedy text
# may flip at a near-tie; compare the next-token logits instead (a broken cache gives cos << 1).
import numpy as np
V = 248320

def first_logits(ids, aff):
    # logits at the prompt end (same input on both paths; after a decode step a near-tie first
    # token could differ and make the comparison meaningless)
    lib.qe_start2(e, T.ints(ids), len(ids), 0, aff)
    slot, reused = lib.qe_last_slot(e), lib.qe_last_reused(e)
    a = np.zeros(V, np.float32)
    lib.qe_logits(e, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    return a, slot, reused

la, sa, ra = first_logits(main3, MAIN_SYS)          # cached (live / snapshot) in the main slot
lib.qe_reset(e)
lb, sb, rb = first_logits(main3, MAIN_SYS)          # fresh
cos = float(la @ lb / np.linalg.norm(la) / np.linalg.norm(lb))
print(f"cached (slot {sa}, reused {ra}) vs fresh (reused {rb}): logits cos {cos:.6f}, "
      f"argmax {la.argmax()} vs {lb.argmax()}")
srt = np.sort(lb)
tie = srt[-1] - srt[-2] < 0.15  # a near-tie may flip between prefill paths
ok &= cos > 0.999 and rb == 0 and (tie or la.argmax() == lb.argmax())
lib.qe_destroy(e)
print("OK" if ok else "MISMATCH")
