"""skinny_w4 must be bit-identical to gemv_w4 (same summation order) for every row count 1..32,
plain and GLU epilogue, on real checkpoint weights and on the per-GPU decode shapes."""
import os
import sys

import torch
from safetensors import safe_open

import qe
import test_gemv as G

MODEL = os.path.join(os.path.dirname(__file__), "..", "models/Qwen3.8-27B-INT4/model.safetensors")
MS = [1, 2, 5, 8, 9, 12, 15, 16, 17, 20, 24, 25, 31, 32]


def check(dq, ds, n, k, label, ms=MS, wref=None):
    worst = 0.0
    for glu in (0, 1):
        for m in ms:
            x = torch.randn(m, k, device="cuda").bfloat16()
            if glu:
                y0 = torch.full((m, n // 2), float("nan"), device="cuda").bfloat16()
                y1 = y0.clone()
            else:
                y0 = torch.full((m, n), float("nan"), device="cuda")
                y1 = y0.clone()
            qe.call("qe_gemv_w4g", dq, ds, n, k, x, y0, m, glu)
            ok = torch.zeros(1, dtype=torch.int32)
            r = qe._LIB.qe_skinny_w4(qe._p(dq), qe._p(ds), n, k, qe._p(x), qe._p(y1), m, glu)
            torch.cuda.synchronize()
            if r == 0:
                print(f"  {label}: skinny unsupported (fallback to gemv_w4), skipped")
                return True
            same = torch.equal(y0.view(torch.int32 if not glu else torch.int16),
                               y1.view(torch.int32 if not glu else torch.int16))
            if not same:
                d = (y0.float() - y1.float()).abs()
                nan = torch.isnan(y1.float()).sum().item()
                print(f"  MISMATCH {label} m={m} glu={glu}: max|d|={d.nan_to_num(1e9).max().item():.3e} "
                      f"n_diff={(d > 0).sum().item()} nan={nan}")
                return False
            if wref is not None and not glu:
                ref = x.float() @ wref.T
                worst = max(worst, ((y1 - ref).abs().max() / ref.abs().max()).item())
    print(f"  {label:40s} {n:6d}x{k:5d} bit-exact for m={ms} (plain+GLU)" +
          (f" relerr={worst:.2e}" if wref is not None else ""))
    return True


def real(name):
    with safe_open(MODEL, "pt", device="cuda") as f:
        wp = f.get_tensor(name + ".weight_packed")
        sc = f.get_tensor(name + ".weight_scale")
    n, k = wp.shape[0], wp.shape[1] * 8
    wref = (G.unpack(wp, n, k) * sc.float().repeat_interleave(128, dim=1)).bfloat16().float()
    dq = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w4", wp, sc, n, k, dq, ds)
    return check(dq, ds, n, k, name.split("layers.")[-1], wref=wref)


def synth(n, k):
    wp = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
    sc = (torch.rand(n, k // 128, device="cuda") * 0.01).bfloat16()
    dq = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w4", wp, sc, n, k, dq, ds)
    return check(dq, ds, n, k, f"synth {n}x{k}")


if __name__ == "__main__":
    torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    L = "model.language_model.layers."
    ok = True
    for name in ["0.mlp.gate_proj", "0.mlp.down_proj", "0.linear_attn.in_proj_qkv",
                 "0.linear_attn.out_proj", "3.self_attn.q_proj", "3.self_attn.k_proj"]:
        ok &= real(L + name)
    for n, k in [(11776, 5120), (5120, 5888), (5632, 5120), (5120, 5120), (5120, 2048), (48, 5120),
                 (2048, 10240), (82784, 5120)]:
        ok &= synth(n, k)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
