"""In-process A/B of the W8A8 GEMM kernels (rotating rounds, medians): the first gemm_w8a8_kernel (tiles 0..2) vs
gemm_w8v_kernel (tiles 0..3), activation group G, on the per-GPU shapes, against the INT4 engine's W4A8 GEMM.
Usage: python3 tests/ab_gemm_w8a8.py [--ts 64,512,2048] [--g 32] [--rounds 5]"""
import argparse
import statistics

import torch

import qe
import test_gemm_i8 as I8
import test_gemv as G
from test_gemm_w8a8 import SHAPES, make_w

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default="64,512,2048")
ap.add_argument("--g", type=int, default=32)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--shapes", default=",".join(SHAPES))
a = ap.parse_args()
torch.cuda.set_device(0)
qe.call("qe_gemv_workspace_init")
cands = [(0, ci) for ci in range(3)] + [(1, ci) for ci in range(4)]
for name in a.shapes.split(","):
    n, k = SHAPES[name]
    n, k, _, dq, ds = make_w(n, k)
    _, _, _, dq4, ds4 = I8.make_w(n, k)
    for t in [int(v) for v in a.ts.split(",")]:
        x = I8.make_x(t, k)
        y = torch.zeros(t, n, device="cuda")
        ts = {c: [] for c in cands + ["w4"]}

        def run(c):
            if c == "w4":
                qe.call("qe_gemm_w4", dq4, ds4, n, k, x, y, t)
            else:
                qe.call("qe_gemm_w8_set_kernel", c[0])
                qe.call("qe_gemm_w8a8", dq, ds, n, k, x, y, t, a.g, c[1], 0)
        for c in ts:
            run(c)  # warm-up / first-use buffers
        order = list(ts)
        for r in range(a.rounds):
            for i in range(len(order)):
                c = order[(i + r) % len(order)]
                ts[c].append(G.bench(lambda: run(c), a.iters))
        med = {c: statistics.median(v) for c, v in ts.items()}
        tf = 2 * n * k * t / 1e6
        b0 = min((c for c in cands if c[0] == 0), key=med.get)
        b1 = min((c for c in cands if c[0] == 1), key=med.get)
        v1 = " ".join(f"ci{c[1]} {med[c]:.0f}" for c in cands if c[0] == 1)
        print(f"{name:8s} t={t:5d} | w8a8 ci{b0[1]} {med[b0]:7.1f} us {tf / med[b0]:5.1f} TF | w8v ci{b1[1]} {med[b1]:7.1f} us "
              f"{tf / med[b1]:5.1f} TF x{med[b0] / med[b1]:.2f} [{v1}] | W4A8 {med['w4']:7.1f} us {tf / med['w4']:5.1f} TF",
              flush=True)
qe.call("qe_gemm_w8_set_kernel", 1)
