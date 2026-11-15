#!/usr/bin/env python3
"""kv4_27b L7: the 27B 4-bit KV row writer (kv4_store) on injected edge cases, through qe_test_kv4_row (one row, no
model). Checks: a normal row decodes within half a step of the rotated input; a NaN input dim gives the bytes of the
same row with 0 there, an inf dim too (the rest of the row is kept, no scale is NaN / inf / 0); a tiny row (subnormal fp16 scales) still decodes within half a step. K and V layouts. Exit 0 = PASS.
usage: QE_LIB=build/libqengine.so tests/test_kv4_row.py"""
import ctypes, os, sys
import numpy as np

lib = ctypes.CDLL(os.environ.get("QE_LIB", os.path.join(os.path.dirname(__file__), "..", "build", "libqengine.so")))
lib.qe_test_kv4_row.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
ROW = 144


def store(x, krow):
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros(ROW, dtype=np.uint8)
    assert lib.qe_test_kv4_row(x.ctypes.data, krow, out.ctypes.data) == ROW
    return out


def sign(d):
    return -1.0 if ((0x9E3779B9 * (d + 1)) & 0xFFFFFFFF) >> 31 else 1.0


def rotate(x):  # kv4_rotate: signs, the kernel's butterfly order, / 16
    h = np.array([x[d] * sign(d) for d in range(256)], dtype=np.float64)
    ln = 1
    while ln < 256:
        a = h.copy()
        for d in range(256):
            h[d] = a[d ^ ln] - a[d] if d & ln else a[d] + a[d ^ ln]
        ln <<= 1
    return h * 0.0625


def decode(row, krow):
    sc = row[128:].view(np.float16).astype(np.float64)
    n = np.zeros(256, dtype=np.int64)
    if krow:
        for u in range(64):
            b0, b1 = int(row[2 * u]), int(row[2 * u + 1])
            n[4 * u], n[4 * u + 2], n[4 * u + 1], n[4 * u + 3] = b0 & 15, b0 >> 4, b1 & 15, b1 >> 4
    else:
        for i in range(128):
            n[2 * i], n[2 * i + 1] = row[i] & 15, row[i] >> 4
    return (n - 7.5) * np.repeat(sc, 32), sc


def bound(r, sc):  # 16 half-offset levels (n - 7.5) s: half a step inside +-7.5 s, the clip beyond
    s = np.repeat(sc, 32)
    return 0.5 * s + np.maximum(0.0, np.abs(r) - 7.5 * s)


fails = 0


def check(name, ok, msg=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + (" " + msg if msg else ""), flush=True)
    fails += not ok


rng = np.random.default_rng(7)
for krow in (1, 0):
    lay = "K" if krow else "V"
    x = rng.standard_normal(256).astype(np.float32)
    y, sc = decode(store(x, krow), krow)
    r = rotate(x)
    err = np.abs(y - r) - bound(r, sc)
    check(f"{lay} normal row within half a step (or the clip)", np.all(np.isfinite(sc)) and np.all(sc > 0) and err.max() <= 1e-3 * sc.max(),
          f"max excess {err.max():.2e}")
    xn = x.copy(); xn[5] = np.nan
    x0 = x.copy(); x0[5] = 0.0
    bn = store(xn, krow)
    check(f"{lay} NaN dim = 0 dim", np.array_equal(bn, store(x0, krow)))
    _, s2 = decode(bn, krow)
    check(f"{lay} NaN row scales finite and > 0", np.all(np.isfinite(s2)) and np.all(s2 > 0))
    for v in (np.inf, -np.inf):
        xi = x.copy(); xi[9] = v
        xc = x.copy(); xc[9] = 0.0
        bi = store(xi, krow)
        _, s3 = decode(bi, krow)
        check(f"{lay} {v} dim = 0 dim", np.array_equal(bi, store(xc, krow)) and np.all(np.isfinite(s3)) and np.all(s3 > 0))
    xt = (rng.standard_normal(256) * 1e-6).astype(np.float32)
    yt, st = decode(store(xt, krow), krow)
    rt = rotate(xt)
    et = np.abs(yt - rt) - bound(rt, st)
    check(f"{lay} tiny row (subnormal scales) within half a step (or the clip)", np.all(st > 0) and np.all(np.isfinite(st)) and
          et.max() <= 1e-2 * st.max(), f"scale {st.max():.2e}, max excess {et.max():.2e}")
print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
