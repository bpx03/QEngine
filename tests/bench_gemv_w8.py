"""Decode GEMV bandwidth: gemv_w8 (INT8 layers) vs the INT4 H16 kernel (gemv_w4h) on the per-GPU shapes of
Qwen3.8-27B at TP=2, against a device-to-device copy as the DRAM ceiling.
Usage: python3 tests/bench_gemv_w8.py [--ms 1,2,4,8,16,32,64] [--shapes gate_up,down,...] [--iters 200]"""
import argparse

import torch

import qe
import test_gemv as G

SHAPES = {"gate_up": (17408, 5120), "down": (5120, 8704), "dn_in": (8192, 5120), "qkv": (7168, 5120),
          "o/out": (5120, 3072)}


def w8(n, k):
    q = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int16)
    qp = (q + 128).to(torch.uint8).contiguous().view(torch.int32)
    sc = (torch.rand(n, k // 128, device="cuda") * 2e-4 + 1e-5).bfloat16()
    dq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w8", qp, sc, n, k, dq, ds)
    return dq, ds


def w4(n, k):
    wp = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
    sc = (torch.rand(n, k // 128, device="cuda") * 0.01 + 0.001).bfloat16()
    dq = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w4", wp, sc, n, k, dq, ds)
    return dq, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8,16,32,64")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-w4", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    qe.call("qe_gemv_workspace_init")
    ms = [int(v) for v in a.ms.split(",")]
    src = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    us = G.bench(lambda: dst.copy_(src), 50)
    peak = 2 * src.numel() / us / 1e3
    print(f"device copy: {peak:.0f} GB/s (read + write)")
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        dq8, ds8 = w8(n, k)
        b8 = n * k + n * (k // 128) * 2
        if not a.no_w4:
            dq4, ds4 = w4(n, k)
            b4 = n * k // 2 + n * (k // 128) * 2
        for m in ms:
            x = torch.randn(m, k, device="cuda").bfloat16()
            y = torch.empty(m, n, device="cuda")
            t8 = G.bench(lambda: qe.call("qe_gemv_w8", dq8, ds8, n, k, x, y, m), a.iters)
            line = f"{name:8s} {n:5d}x{k:<5d} m={m:2d} | W8 {t8:7.1f} us {b8 / t8 / 1e3:5.0f} GB/s ({b8 / t8 / 1e3 / peak * 100:3.0f}%)"
            if not a.no_w4 and m <= 32:
                xh = torch.empty(m, k, dtype=torch.float16, device="cuda")
                qe.call("qe_xh_pack", x, k, m, xh, k)
                t4 = G.bench(lambda: qe.call("qe_gemv_w4hx", dq4, ds4, n, k, xh, y, m, 0, k), a.iters)
                line += f" | W4H {t4:7.1f} us {b4 / t4 / 1e3:5.0f} GB/s | W8/W4 time {t8 / t4:.2f}x (bytes 1.9x)"
            print(line, flush=True)


if __name__ == "__main__":
    main()
