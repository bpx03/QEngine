"""27B RoPE / YaRN table (rope_table_host via qe_rope_table, host only, no GPU) vs an independent reference of the
published YaRN definition (HF _compute_yarn_parameters / vLLM YaRNScalingRotaryEmbedding: NTK-by-parts, beta_fast 32,
beta_slow 1, truncated correction range, mscale 0.1 ln s + 1, base ** (arange / d) in fp32).
usage: QE_LIB=<build>/libqengine.so python3 tests/test_rope_table.py"""
import ctypes, math, os, sys
import numpy as np

lib = ctypes.CDLL(os.environ.get("QE_LIB", os.path.join(os.path.dirname(__file__), "..", "build", "libqengine.so")))
lib.qe_rope_table.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
THETA, ROT, NATIVE = 1e7, 64, 262144


def engine(factor):
    out = (ctypes.c_float * 33)()
    lib.qe_rope_table(factor, THETA, ROT, NATIVE, out)
    return np.array(out[:], dtype=np.float32)


def reference(factor, dim=ROT, base=THETA, orig=NATIVE, beta_fast=32, beta_slow=1):
    pos_freqs = np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))
    if factor <= 1:
        return (1.0 / base ** (np.arange(0, dim, 2) / dim)).astype(np.float32), 1.0, None
    extra = np.float32(1.0) / pos_freqs
    inter = np.float32(1.0) / (np.float32(factor) * pos_freqs)
    cdim = lambda nrot: dim * math.log(orig / (nrot * 2 * math.pi)) / (2 * math.log(base))
    lo, hi = max(math.floor(cdim(beta_fast)), 0), min(math.ceil(cdim(beta_slow)), dim - 1)
    if lo == hi:
        hi += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - lo) / (hi - lo), 0, 1).astype(np.float32)
    ef = np.float32(1) - ramp
    inv = inter * (np.float32(1) - ef) + extra * ef
    return inv.astype(np.float32), 0.1 * math.log(factor) + 1.0, (lo, hi)


fails = 0
for f in (1.0, 2.0, 578560 / 262144, 4.0):
    e = engine(f)
    r, ms, rng = reference(f)
    if f <= 1:
        ok = np.array_equal(e[:32], r) and e[32] == 1.0  # native: bit for bit (the double pow of master)
    else:
        ok = np.allclose(e[:32], r, rtol=4e-7, atol=0) and abs(e[32] - ms * ms) < 1e-6
    rel = float(np.max(np.abs(e[:32] / r - 1)))
    print(f"factor {f:.4f}: max rel diff {rel:.2e}, q factor {e[32]:.6f} (mscale {ms:.6f}^2 = {ms * ms:.6f}), "
          f"correction dims {rng}: {'PASS' if ok else 'FAIL'}")
    fails += not ok
# q-only placement: (mscale^2 q_rot) . k_rot == (mscale q_rot) . (mscale k_rot) up to rounding, pass-through untouched
rs = np.random.default_rng(1)
q, k = rs.standard_normal((8, 256)), rs.standard_normal((1024, 256))
ms = 0.1 * math.log(4) + 1
qa, ka, qb = q.copy(), k.copy(), q.copy()
qa[:, :64] *= ms; ka[:, :64] *= ms; qb[:, :64] *= ms * ms
d = float(np.max(np.abs(qa @ ka.T - qb @ k.T)))
print(f"scores, q-only vs q and k placement: max abs diff {d:.2e}: {'PASS' if d < 1e-9 else 'FAIL'}")
fails += d >= 1e-9
print("ALL PASS" if not fails else f"{fails} FAIL")
sys.exit(1 if fails else 0)
