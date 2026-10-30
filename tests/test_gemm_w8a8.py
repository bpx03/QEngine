"""W8A8 prefill GEMM (gemm_w8 / gemm_w8_cfg): correctness vs an fp64 emulation of the same int8 activation
quantization (every tile candidate and group, odd token counts: partial tiles and stream-K splits), the GLU
epilogue, quantization error vs exact fp64 next to the W8A16 GEMV, and timing vs the GEMV in 64-row pieces
(the engine's path before this kernel) and the W4A8 GEMM on the same shapes.
Usage: python3 tests/test_gemm_w8a8.py [--ts 65,512,2048] [--gs 32,64,128] [--cis 0,1,2] [--real] [--quick]"""
import argparse
import json
import os

import torch

import qe
import test_gemm_i8 as I8
import test_gemv as G

# per-GPU shapes at TP=2 (Qwen3.8-27B): gate+up, down, DeltaNet in (qkv+z), attention qkv, o / out
SHAPES = {"gate_up": (17408, 5120), "down": (5120, 8704), "dn_in": (8192, 5120), "qkv": (7168, 5120),
          "o/out": (5120, 3072)}
MODEL = os.path.join(os.path.dirname(__file__), "..", "models/Qwen3.8-27B-INT8")
REAL = {"gate_up": ("0.mlp.gate_proj", (0, 8704), None), "down": ("0.mlp.down_proj", (0, 5120), (0, 8704)),
        "qkv": ("3.self_attn.q_proj", (0, 7168), None), "o/out": ("0.linear_attn.out_proj", (0, 5120), (0, 3072))}


def ct_pack(q):
    """int8 [n, k] -> compressed-tensors int32 [n, k/4] (u = q + 128, bytes LSB-first)"""
    return (q.to(torch.int16) + 128).to(torch.uint8).contiguous().view(torch.int32)


def make_w(n, k, real=None):
    if real:
        from safetensors import safe_open
        name, rows, cols = real
        name = "model.language_model.layers." + name
        wm = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(MODEL, wm[name + ".weight_packed"]), "pt") as f:
            qp, sc = f.get_tensor(name + ".weight_packed"), f.get_tensor(name + ".weight_scale")
        c0, c1 = cols or (0, qp.shape[1] * 4)
        u = qp[rows[0]:rows[1]].contiguous().view(torch.uint8)[:, c0:c1].contiguous().cuda()
        sc = sc[rows[0]:rows[1], c0 // 128:c1 // 128].contiguous().cuda()
        n, k = u.shape
        q = u.to(torch.int32) - 128
        qp = u.view(torch.int32)
    else:
        g = torch.Generator(device="cuda").manual_seed(n * 7 + k)
        q = torch.randint(-128, 128, (n, k), device="cuda", generator=g, dtype=torch.int32)
        sc = (torch.rand(n, k // 128, device="cuda", generator=g) * 2e-4 + 1e-5).bfloat16()
        qp = ct_pack(q.to(torch.int8))
    wref = (q.double().view(n, k // 128, 128) * sc.double().unsqueeze(-1)).view(n, k)
    dq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w8", qp, sc, n, k, dq, ds)
    return n, k, wref, dq, ds


def gemv_pieces(dq, ds, n, k, x, y, t):
    for r in range(0, t, 64):
        m = min(64, t - r)
        qe.call("qe_gemv_w8", dq, ds, n, k, x[r:r + m], y[r:r + m], m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="65,100,512,777,2048")
    ap.add_argument("--gs", default="32,64,128")
    ap.add_argument("--cis", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--real", action="store_true", help="real INT8 checkpoint slices (precision)")
    ap.add_argument("--quick", action="store_true", help="correctness only, no timing")
    ap.add_argument("--kernel", type=int, default=1, help="1 = gemm_w8v_kernel (tiles 0..3), 0 = the first W8A8 kernel (0..2)")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    qe.call("qe_gemv_workspace_init")
    qe.call("qe_gemm_w8_set_kernel", a.kernel)
    ts =[int(v) for v in a.ts.split(",")]
    gs = [int(v) for v in a.gs.split(",")]
    cis = [int(v) for v in a.cis.split(",")]
    worst = 0.0
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        n, k, wref, dq, ds = make_w(n, k, REAL.get(name) if a.real else None)
        for t in ts:
            x = I8.make_x(t, k)
            y = torch.zeros(t, n, device="cuda")
            ref = x.double() @ wref.T
            gemv_pieces(dq, ds, n, k, x, y, t)
            qe.sync()
            egv = I8.relrms(y, ref)
            tf = 2 * n * k * t / 1e6
            line = f"{name:8s} t={t:5d} | gemv: err {egv:.1e}"
            if not a.quick:
                usg = G.bench(lambda: gemv_pieces(dq, ds, n, k, x, y, t), a.iters)
                line += f" {usg:8.1f}us {tf / usg:5.1f}TF"
            for g in gs:
                emul = I8.quant_emul(x, g) @ wref.T
                best = None
                for ci in cis:
                    y.zero_()
                    qe.call("qe_gemm_w8a8", dq, ds, n, k, x, y, t, g, ci, 0)
                    qe.sync()
                    em = I8.relrms(y, emul)
                    worst = max(worst, em)
                    eq = I8.relrms(y, ref)
                    us = 0.0 if a.quick else G.bench(
                        lambda: qe.call("qe_gemm_w8a8", dq, ds, n, k, x, y, t, g, ci, 0), a.iters)
                    if best is None or (us, em) < (best[0], best[2]):
                        best = (us, ci, em, eq)
                us, ci, em, eq = best
                line += f" | g{g}: err {eq:.1e} (vs emul {em:.0e})"
                if not a.quick:
                    line += f" ci{ci} {us:7.1f}us {tf / us:5.1f}TF x{usg / us:.2f}"
            print(line, flush=True)
        # GLU epilogue: rows in GLU tile order (8 gate, 8 up, ...), bf16 act [t][n / 2]
        t = ts[-1]
        x = I8.make_x(t, k)
        act = torch.zeros(t, n // 2, dtype=torch.bfloat16, device="cuda")
        for ci in cis:
            qe.call("qe_gemm_w8a8", dq, ds, n, k, x, act, t, gs[0], ci, 1)
            qe.sync()
            e = I8.relrms(act, I8.glu_ref(I8.quant_emul(x, gs[0]) @ wref.T))
            assert e < 1e-2, (name, "glu", ci, e)
        print(f"{name:8s} GLU t={t} g{gs[0]}: OK (bf16 output vs emulation {e:.1e})")
        for m in (1, 8, 12, 40):  # GLU epilogue of the decode GEMV (fp16 activations, exact weights)
            act = torch.zeros(m, n // 2, dtype=torch.bfloat16, device="cuda")
            qe.call("qe_gemv_w8_glu", dq, ds, n, k, x[:m], act, m)
            qe.sync()
            e = I8.relrms(act, I8.glu_ref(x[:m].half().double() @ wref.T))
            assert e < 1e-2, (name, "gemv glu", m, e)
        print(f"{name:8s} GEMV GLU m=1,8,12,40: OK (vs exact {e:.1e})")
    if not a.quick:  # the W4A8 GEMM on the same shapes (engine default kernel and group) for reference
        for name in a.shapes.split(","):
            n, k = SHAPES[name]
            n, k, _, dq4, ds4 = I8.make_w(n, k)
            for t in ts:
                if t < 512:
                    continue
                x = I8.make_x(t, k)
                y = torch.zeros(t, n, device="cuda")
                us = G.bench(lambda: qe.call("qe_gemm_w4", dq4, ds4, n, k, x, y, t), a.iters)
                print(f"{name:8s} t={t:5d} | W4A8 (gemm_w4) {us:7.1f}us {2 * n * k * t / 1e6 / us:5.1f}TF")
    print(f"max mismatch vs int8 emulation: {worst:.1e}")
    assert worst < 1e-5, worst
    print("OK")


if __name__ == "__main__":
    main()
