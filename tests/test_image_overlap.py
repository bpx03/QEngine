"""GPU, 27B: two image prompts prefilling at once (the engine allows kMaxPrefills = 2; the server keeps image requests
apart today) must each embed their own image rows. Prompt A starts, prefills its text head (its image rows not yet
embedded), prompt B (other image rows) starts and finishes, then A finishes: the last-position logits of both must
equal those of A and B run alone (before the per-prefill image table ranges, A embedded B's rows).
Usage: QE_LIB=build/libqengine.so QE_TEST_MODE=alone python3 tests/test_image_overlap.py [model dir] [devices], then the
same without QE_TEST_MODE (the overlap; compares with the saved run)."""
import ctypes, os, sys
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
DEVS = sys.argv[2] if len(sys.argv) > 2 else "0,1"
IMG = 1 << 30
V = 248320
e = Engine(MODEL, devices=DEVS, max_t=32768)
e.set_sampling(0)
H = int(__import__("json").load(open(os.path.join(MODEL, "config.json"))).get("text_config", {}).get("hidden_size", 5120))
e.lib.qe_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
rng = np.random.default_rng(7)


def prompt(seed, head, rows, tail):
    r = np.random.default_rng(seed)
    txt = lambda n: [int(x) for x in r.integers(1000, 100000, n)]  # noqa: E731
    base = int(r.integers(0, 1 << 20))
    ids = txt(head) + [IMG | (base + j) for j in range(rows)] + txt(tail)
    emb = (torch.from_numpy(r.standard_normal((rows, H)).astype(np.float32)) * 0.02).to(torch.bfloat16)
    rope3 = np.repeat(np.arange(len(ids), dtype=np.int32)[:, None], 3, axis=1).copy()
    return ids, (emb, rope3, 0)


def logits():
    lg = np.zeros((8, V), np.float32)
    e.lib.qe_logits(e.h, lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    return lg[0].copy()


A = prompt(1, 2048, 400, 64)   # the image after the first chunk (2048 rows) of A
B = prompt(2, 300, 600, 64)    # more rows than A: the table grows while A is in flight


def alone(p):
    s = e.start_begin(p[0], 0, mm=p[1])
    while not e.prefill_more(1 << 20, s):
        pass
    tok, _, _ = e.start_finish(s)
    lg = logits()
    e.release(s)
    return tok, lg


MODE = os.environ.get("QE_TEST_MODE", "")
OUT = os.environ.get("QE_TEST_OUT", "/tmp/qe_image_overlap.npz")
if MODE == "alone":  # one process per mode: the prompt cache of the other run would restore the image rows
    ta, la = alone(A)
    tb, lb = alone(B)
    np.savez(OUT, ta=ta, tb=tb, la=la, lb=lb)
    e.close()
    sys.exit(0)
sa = e.start_begin(A[0], 0, mm=A[1])
done_a = e.prefill_more(2048, sa)
assert not done_a and e.prefill_pos(sa) <= 2048, e.prefill_pos(sa)
sb = e.start_begin(B[0], 0, mm=B[1])
while not e.prefill_more(1 << 20, sb):
    pass
tb2, _, _ = e.start_finish(sb)
lb2 = logits()
while not e.prefill_more(1 << 20, sa):
    pass
ta2, _, _ = e.start_finish(sa)
la2 = logits()
e.close()
r = np.load(OUT)
ta, tb, la, lb = int(r["ta"]), int(r["tb"]), r["la"], r["lb"]
da, db = float(np.abs(la2 - la).max()), float(np.abs(lb2 - lb).max())
print(f"A alone tok {ta}, overlapped {ta2}, max |dlogit| {da:.4f}; B alone tok {tb}, overlapped {tb2}, max |dlogit| {db:.4f}")
ok = ta == ta2 and tb == tb2 and da < 1e-3 and db < 1e-3  # (same chunks, same kernels: bit-identical)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
