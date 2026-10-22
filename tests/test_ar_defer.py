"""Deferred all-reduce tail (QE_AR_DEFER) must give bit-identical logits: run with QE_AR_DEFER=0 and =1
(separate processes) on prompts of several sizes (no shared prefixes), then compare.
usage: QE_AR_DEFER=0 python3 tests/test_ar_defer.py save a.npz; QE_AR_DEFER=1 ... save b.npz;
       python3 tests/test_ar_defer.py compare a.npz b.npz"""
import ctypes, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
if sys.argv[1] == "compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    ok = True
    for k in a.files:
        same = np.array_equal(a[k], b[k])
        ok &= same
        d = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()
        print(f"{k}: {'identical' if same else f'max abs diff {d:.3e}'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
import common as T
T.lib.qe_start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
sizes = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "300,1500,2500,4500").split(",")]
e = T.lib.qe_create(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4").encode(), b"0,1,2", 8192, b"")
res = {}
for i, n in enumerate(sizes):
    rng = np.random.default_rng(100 + i)
    ids = [int(v) for v in rng.integers(1000, 150000, size=n)]
    T.lib.qe_reset(e)
    T.lib.qe_start(e, T.ints(ids), n, 0)
    lg = np.zeros(248320, dtype=np.float32)
    T.lib.qe_logits(e, lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    res[f"lg{n}"] = lg
    print(f"{n}: argmax {lg.argmax()}", flush=True)
np.savez(sys.argv[2], **res)
T.lib.qe_destroy(e)
