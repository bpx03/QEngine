"""A/B of gemv_w8's <= 8-row kernels in one process (rotating rounds, medians): the per-lane bf16 kernel vs the
packed-activation one (QE_W8P1_MIN). Usage: python3 tests/ab_gemv_w8p1.py [--ms 2,4,6,8] [--rounds 9]"""
import argparse
import statistics

import torch

import qe
import test_gemv as G
from bench_gemv_w8 import SHAPES, w8

ap = argparse.ArgumentParser()
ap.add_argument("--ms", default="2,4,6,8")
ap.add_argument("--rounds", type=int, default=9)
ap.add_argument("--iters", type=int, default=200)
a = ap.parse_args()
torch.cuda.set_device(0)
qe.call("qe_gemv_workspace_init")
for name, (n, k) in SHAPES.items():
    dq, ds = w8(n, k)
    b8 = n * k + n * (k // 128) * 2
    for m in [int(v) for v in a.ms.split(",")]:
        x = torch.randn(m, k, device="cuda").bfloat16()
        y = torch.empty(m, n, device="cuda")
        ts = {9: [], 1: []}
        for r in range(a.rounds):
            for mode in ((9, 1) if r % 2 == 0 else (1, 9)):
                qe.call("qe_gemv_w8_set_p1_min", mode)
                ts[mode].append(G.bench(lambda: qe.call("qe_gemv_w8", dq, ds, n, k, x, y, m), a.iters))
        old, new = statistics.median(ts[9]), statistics.median(ts[1])
        print(f"{name:8s} m={m}: bf16-x kernel {old:6.1f} us {b8 / old / 1e3:4.0f} GB/s | packed {new:6.1f} us "
              f"{b8 / new / 1e3:4.0f} GB/s | x{old / new:.2f}", flush=True)
qe.call("qe_gemv_w8_set_p1_min", 0)
