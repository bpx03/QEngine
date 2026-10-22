"""W4A8 prefill GEMM (gemm_w4_i8): correctness vs an fp64 emulation of the same int8 quantization,
quantization error vs exact fp64, and timing vs the fp16-accumulate path on the per-GPU shapes.
Usage: python3 tests/test_gemm_i8.py [--ts 512,1024,2048] [--gs 32,64,128] [--cis 2,3] [--real]
QE_PREFILL_INT8_DIRECT=1 selects the register-light flush variant of the old kernel (set before loading);
the last argument of qe_gemm_w4_i8 is the kernel variant (1 = gemm_i8t_kernel, 0 = gemm_i8_kernel)."""
import argparse
import os
import sys

import torch

import qe
import test_gemv as G

SHAPES = {"gate_up": (11776, 5120), "down": (5120, 5888), "dn_in": (5632, 5120),
          "qkv_split": (7168, 5120), "o_split": (5120, 3072),  # KV position-split layout
          "qkv": (5120, 5120), "o/out": (5120, 2048)}
MODEL = os.path.join(os.path.dirname(__file__), "..", "models/Qwen3.8-27B-INT4/model.safetensors")
REAL = {"gate_up": "0.mlp.gate_proj", "down": "0.mlp.down_proj", "dn_in": "0.linear_attn.in_proj_qkv",
        "qkv": "3.self_attn.q_proj", "o/out": "0.linear_attn.out_proj"}


def make_w(n, k, real=None):
    if real:
        from safetensors import safe_open
        with safe_open(MODEL, "pt", device="cuda") as f:
            wp = f.get_tensor("model.language_model.layers." + real + ".weight_packed")
            sc = f.get_tensor("model.language_model.layers." + real + ".weight_scale")
        n, k = wp.shape[0], wp.shape[1] * 8
    else:
        torch.manual_seed(1)
        wp = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
        sc = (torch.rand(n, k // 128, device="cuda") * 0.01 + 0.001).bfloat16()
    wref = (G.unpack(wp, n, k).double() * sc.double().repeat_interleave(128, 1))  # exact INT4 math
    dq = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w4", wp, sc, n, k, dq, ds)
    return n, k, wref, dq, ds


def make_x(t, k, seed=0):
    """bf16 activations with channel outliers: amax/rms per row ~ 15-60 like the real residual."""
    torch.manual_seed(seed)
    ch = torch.exp(torch.randn(k, device="cuda") * 0.6)
    idx = torch.randperm(k, device="cuda")[: max(1, k // 200)]
    ch[idx] *= 12.0
    x = torch.randn(t, k, device="cuda") * ch
    x *= torch.exp(torch.randn(t, 1, device="cuda") * 0.3)
    return x.bfloat16()


def quant_emul(x, g):
    """the kernel's quantization in fp32 (scale with 22 significant bits, q = rn(x / sc))"""
    t, k = x.shape
    xg = x.float().reshape(t, k // g, g)
    amax = xg.abs().amax(-1, keepdim=True)
    sc = (amax / 127.0).view(torch.int32) & ~3
    sc = torch.where(amax > 0, sc.view(torch.float32), torch.ones_like(amax))
    inv = torch.where(amax > 0, 1.0 / sc, torch.zeros_like(amax))
    q = torch.round(xg * inv).clamp(-127, 127)
    return (q.double() * sc.double()).reshape(t, k)


def relrms(a, b):
    return ((a - b).double().norm() / b.double().norm()).item()


def glu_ref(y):  # y [t][n] fp64 in GLU tile order -> act [t][n/2]
    t, n = y.shape
    yt = y.reshape(t, n // 16, 2, 8)
    gate, up = yt[:, :, 0, :], yt[:, :, 1, :]
    return (gate / (1 + torch.exp(-gate)) * up).reshape(t, n // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="512,1024,2048")
    ap.add_argument("--gs", default="32,64,128")
    ap.add_argument("--cis", default="2,3")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--real", action="store_true", help="real checkpoint weights (precision)")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--glu", action="store_true", help="also check the GLU epilogue")
    ap.add_argument("--var", type=int, default=100, help="kernel variant (gemm_i8_dispatch; 100 = the engine default of QE_PREFILL_INT8_KERNEL=2)")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    ts = [int(v) for v in a.ts.split(",")]
    gs = [int(v) for v in a.gs.split(",")]
    cis = [int(v) for v in a.cis.split(",")]
    worst_match = 0.0
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        n, k, wref, dq, ds = make_w(n, k, REAL.get(name) if a.real else None)
        for t in ts:
            x = make_x(t, k)
            y = torch.zeros(t, n, device="cuda")
            ref = x.double() @ wref.T
            amax_rms = (x.float().abs().amax(1) / x.float().pow(2).mean(1).sqrt())
            # fp16-accumulate path (whatever gemm_w4 selects: tuned)
            qe.call("qe_gemm_w4", dq, ds, n, k, x, y, t)
            qe.sync()
            e16 = relrms(y, ref)
            us16 = G.bench(lambda: qe.call("qe_gemm_w4", dq, ds, n, k, x, y, t), a.iters)
            tf = 2 * n * k * t / 1e6
            line = (f"{name:8s} t={t:5d} amax/rms {amax_rms.mean():5.1f} | fp16: err {e16:.1e} "
                    f"{us16:7.1f}us {tf / us16:5.1f}TF")
            for g in gs:
                emul = quant_emul(x, g) @ wref.T
                best = None
                for ci in cis:
                    if ci < 2 and t > 16 * (32 if ci == 0 else 64):
                        continue
                    qe.call("qe_gemm_w4_i8", dq, ds, n, k, x, y, t, g, ci, a.var)
                    qe.sync()
                    em = relrms(y, emul)
                    worst_match = max(worst_match, em)
                    eq = relrms(y, ref)
                    us = G.bench(lambda: qe.call("qe_gemm_w4_i8", dq, ds, n, k, x, y, t, g, ci, a.var),
                                 a.iters)
                    if best is None or us < best[0]:
                        best = (us, ci, em, eq)
                us, ci, em, eq = best
                line += f" | g{g}: ci{ci} {us:7.1f}us {tf / us:5.1f}TF x{us16 / us:.2f} err {eq:.1e} (vs emul {em:.0e})"
            print(line, flush=True)
        if a.glu:
            t = ts[0]
            x = make_x(t, k)
            act = torch.zeros(t, n // 2, dtype=torch.bfloat16, device="cuda")
            ref = glu_ref(x.double() @ wref.T)
            w_glu = (n, k)
            qe.call("qe_gemm_w4_i8_glu", dq, ds, n, k, x, act, t, gs[0], cis[0], a.var)
            qe.sync()
            print(f"{name:8s} GLU t={t} g{gs[0]}: err vs exact {relrms(act, ref):.1e}")
    print(f"max mismatch vs int8 emulation: {worst_match:.1e}")
    assert worst_match < 1e-5, worst_match
    print("OK")


if __name__ == "__main__":
    main()
