"""INT8 layer weights (Q8 layout, src/kernels/q8_layout.h) - kernel checks on one GPU:
  1. repack: compressed-tensors 8-bit -> Q8, dequantized through the decode GEMV of an identity-like probe;
  2. decode GEMV, H16 (default) and bf16 (QE_GEMV_H16=0) paths: every row bit-identical across m = 1..64
     (batch == alone), plain and GLU (+ the XH activation copy); error vs fp64 below the bf16 output rounding;
  3. prefill GEMM W8A8 (G 32 / 64 / 128, candidates 0..3): equal to an fp64 emulation of the int8 activation
     quantization (only fp32 rounding apart) and error vs exact fp64; GLU epilogue; stream-K remainders (odd t);
  4. timing vs the INT4 kernels on the per-GPU shapes (decode m = 1 / 8 / 16, prefill t = 2048).
The fp16-accumulate prefill path runs with QE_PREFILL_INT8=0 (set before the library loads):
  python3 tests/test_q8.py [--real DIR] [--quick]          (W8A8 + decode)
  QE_PREFILL_INT8=0 python3 tests/test_q8.py --gemm-only   (fp16-accumulate prefill)
--real DIR: matrices from an INT8 checkpoint (tools/quant_int8.py output) instead of random ones."""
import argparse
import os
import sys

import torch

import qe
import test_gemv as G
from test_gemm_i8 import glu_ref, make_x, quant_emul, relrms

SHAPES = {"gate_up": (17408, 5120), "down": (5120, 8704), "dn_in": (8192, 5120), "qkv": (7168, 5120),
          "o/out": (5120, 3072)}  # 2-GPU slices (n, k)
REAL = {"gate_up": "0.mlp.gate_proj", "down": "0.mlp.down_proj", "dn_in": "0.linear_attn.in_proj_qkv",
        "qkv": "3.self_attn.q_proj", "o/out": "0.linear_attn.out_proj"}
MS = [1, 2, 3, 5, 8, 9, 12, 16, 17, 24, 25, 32, 33, 40, 48, 64]


def make_w(n, k, real=None):
    """(n, k, fp64 reference [n][k], Q8 weights, scales, compressed-tensors source (int32, bf16 scales))"""
    if real:
        from safetensors import safe_open
        with safe_open(os.path.join(real, "model.safetensors"), "pt", device="cuda") as f:
            nm = "model.language_model.layers." + REAL_NAME + ".weight_"
            wp, sc = f.get_tensor(nm + "packed"), f.get_tensor(nm + "scale")
        n, k = wp.shape[0], wp.shape[1] * 4
    else:
        torch.manual_seed(1)
        w = torch.randint(-127, 128, (n, k), dtype=torch.int16, device="cuda")
        wp = (w + 128).to(torch.uint8).view(torch.int32)
        sc = (torch.rand(n, k // 128, device="cuda") * 0.01 + 0.001).bfloat16()
    q = wp.contiguous().view(torch.uint8).view(n, k).to(torch.int16) - 128
    wref = q.double() * sc.double().repeat_interleave(128, 1)
    dq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_q8", wp, sc, n, k, dq, ds)
    return n, k, wref, dq, ds, (wp, sc)


def glu_w(n, k, src):
    """GLU (gate/up interleaved) Q8 copy of an [n][k] matrix whose rows [0, n/2) are gate and [n/2, n) up"""
    wp, sc = src
    gq = torch.empty_like(wp)
    gs = torch.empty_like(sc)
    qe.call("qe_glu_rows8", wp, sc, n, k, gq, gs)
    dq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_q8", gq, gs, n, k, dq, ds)
    return dq, ds


def gemv(dq, ds, n, k, x, m, glu=0, h16=1, yh=None):
    xh = torch.empty(m, k, dtype=torch.float16, device="cuda")
    y = torch.full((m, n // 2 if glu else n), float("nan"), device="cuda")
    if glu:
        y = y.bfloat16()
    qe.call("qe_gemv_q8", dq, ds, n, k, x, y, m, glu, h16, xh, yh)
    return y


def bits(t):
    return t.view(torch.int16 if t.dtype in (torch.bfloat16, torch.float16) else torch.int32)


def interleave_ref(wref):
    """fp64 reference rows in the GLU tile order (8 gate rows, 8 up rows per 16-row tile)"""
    n = wref.shape[0]
    g, u = wref[: n // 2].reshape(-1, 8, wref.shape[1]), wref[n // 2:].reshape(-1, 8, wref.shape[1])
    return torch.stack([g, u], 1).reshape(n, -1)


def check_decode(name, n, k, wref, dq, ds, src):
    ok = True
    x = make_x(64, k, seed=3)
    ref = x.double() @ wref.T
    rb = relrms(ref.float().bfloat16(), ref)  # bf16 rounding of the output: the bar
    gdq, gds = glu_w(n, k, src)
    gref = glu_ref(x.double() @ interleave_ref(wref).T)
    for h16 in (1, 0):
        full = gemv(dq, ds, n, k, x, 64, 0, h16)
        yh64 = torch.empty(64, n // 2, dtype=torch.float16, device="cuda") if h16 else None
        gfull = gemv(gdq, gds, n, k, x, 64, 1, h16, yh64)
        bad = []
        for m in MS:
            y = gemv(dq, ds, n, k, x[:m].contiguous(), m, 0, h16)
            yh = torch.empty(m, n // 2, dtype=torch.float16, device="cuda") if h16 else None
            g = gemv(gdq, gds, n, k, x[:m].contiguous(), m, 1, h16, yh)
            same = torch.equal(bits(y), bits(full[:m])) and torch.equal(bits(g), bits(gfull[:m]))
            if h16:
                same = same and torch.equal(bits(yh), bits(yh64[:m]))
            if not same:
                bad.append(m)
        e, eg = relrms(full, ref), relrms(gfull, gref)
        ok &= not bad and e < 2 * rb + 1e-4 and eg < 1e-2
        print(f"  {name:8s} {'H16 ' if h16 else 'bf16'} GEMV: rows bit-identical m=1..64 "
              f"{'OK' if not bad else 'FAIL at m=' + str(bad)}; err {e:.1e} (bf16 out {rb:.1e}), GLU {eg:.1e}")
    return ok


def check_gemm(name, n, k, wref, dq, ds, src, ts, gs, a8):
    ok = True
    gdq, gds = glu_w(n, k, src)
    for t in ts:
        x = make_x(t, k)
        ref = x.double() @ wref.T
        y = torch.zeros(t, n, device="cuda")
        if not a8:  # fp16-accumulate path (QE_PREFILL_INT8=0)
            qe.call("qe_gemm_q8", dq, ds, n, k, x, y, t, 0, 0, 0)
            qe.sync()
            e = relrms(y, ref)
            ok &= e < 2e-3
            print(f"  {name:8s} t={t:5d} fp16-acc GEMM: err {e:.1e}")
            continue
        for g in gs:
            emul = quant_emul(x, g) @ wref.T
            for ci in range(4):
                y.zero_()
                qe.call("qe_gemm_q8", dq, ds, n, k, x, y, t, 0, g, ci)
                qe.sync()
                em, eq = relrms(y, emul), relrms(y, ref)
                ok &= em < 1e-5
                print(f"  {name:8s} t={t:5d} W8A8 g{g} ci{ci}: vs int8 emulation {em:.1e}, vs exact {eq:.1e}"
                      + ("" if em < 1e-5 else "  FAIL"))
        act = torch.zeros(t, n // 2, dtype=torch.bfloat16, device="cuda")
        qe.call("qe_gemm_q8", gdq, gds, n, k, x, act, t, 1, gs[0], 1)
        qe.sync()
        eg = relrms(act, glu_ref(x.double() @ interleave_ref(wref).T))
        ok &= eg < 2e-2
        print(f"  {name:8s} t={t:5d} W8A8 GLU g{gs[0]}: err vs exact {eg:.1e}")
    return ok


def bench(name, n, k, dq, ds, iters):
    """Q8 vs INT4 (random INT4 of the same shape) decode GEMV and prefill GEMM times"""
    wp4 = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
    sc4 = (torch.rand(n, k // 128, device="cuda") * 0.01 + 0.001).bfloat16()
    dq4 = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
    ds4 = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w4", wp4, sc4, n, k, dq4, ds4)
    line = f"  {name:8s}"
    for m in (1, 8, 16):
        x = make_x(m, k)
        xh = torch.empty(m, k, dtype=torch.float16, device="cuda")
        y = torch.empty(m, n, device="cuda")
        us8 = G.bench(lambda: qe.call("qe_gemv_q8", dq, ds, n, k, x, y, m, 0, 1, xh, None), iters)
        us4 = G.bench(lambda: qe.call("qe_gemv_w4h", dq4, ds4, n, k, x, y, m, 0, xh, None), iters)
        line += f" | m={m}: Q8 {us8:6.1f}us ({n * k / us8 / 1e3:4.0f} GB/s) INT4 {us4:6.1f}us"
    t = 2048
    x = make_x(t, k)
    y = torch.empty(t, n, device="cuda")
    us8 = G.bench(lambda: qe.call("qe_gemm_q8", dq, ds, n, k, x, y, t, 0, 0, 0), max(5, iters // 10))
    us4 = G.bench(lambda: qe.call("qe_gemm_w4_t", dq4, ds4, n, k, x, y, t), max(5, iters // 10))
    line += f" | t=2048: Q8 {us8:7.1f}us ({2 * n * k * t / us8 / 1e6:5.1f} TOPS) INT4 {us4:7.1f}us"
    print(line, flush=True)


REAL_NAME = None


def main():
    global REAL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="", help="INT8 checkpoint dir (tools/quant_int8.py output)")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ts", default="33,100,1000,2048")
    ap.add_argument("--gs", default="64,32,128")
    ap.add_argument("--quick", action="store_true", help="one shape, t = 100 / 1000, G = 64")
    ap.add_argument("--gemm-only", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    a8 = os.environ.get("QE_PREFILL_INT8", "1").strip() not in ("0",)
    torch.cuda.set_device(0)
    qe.call("qe_gemv_workspace_init")
    shapes = a.shapes.split(",")
    ts, gs = [int(v) for v in a.ts.split(",")], [int(v) for v in a.gs.split(",")]
    if a.quick:
        shapes, ts, gs = shapes[:1], [100, 1000], [64]
    ok = True
    for name in shapes:
        n, k = SHAPES[name]
        REAL_NAME = REAL[name]
        n, k, wref, dq, ds, src = make_w(n, k, a.real or None)
        print(f"{name}: n={n} k={k}", flush=True)
        if not a.gemm_only:
            ok &= check_decode(name, n, k, wref, dq, ds, src)
        ok &= check_gemm(name, n, k, wref, dq, ds, src, ts, gs, a8)
        if a.bench:
            bench(name, n, k, dq, ds, a.iters)
    print("ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
