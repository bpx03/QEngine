"""Skinny INT8 GEMM (skinny_w8) for 1..32 decode rows: bit-identical to the single-row gemv_w8 on every row
(batch == alone), close to torch, and timed against gemv_w8 on the per-GPU shapes of Qwen3.8-27B at TP=2.
Usage: python3 tests/test_skinny_w8.py [--ms 1,2,...] [--shapes gate_up,...] [--no-bench]"""
import argparse

import torch

import qe
import test_gemv as G

SHAPES = {"gate_up": (17408, 5120), "down": (5120, 8704), "dn_in": (8192, 5120), "qkv": (7168, 5120),
          "o/out": (5120, 3072)}


def make(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(-128, 128, (n, k), device="cuda", generator=g, dtype=torch.int16)
    qp = (q + 128).to(torch.uint8).contiguous().view(torch.int32)
    sc = (torch.rand(n, k // 128, device="cuda", generator=g) * 2e-4 + 1e-5).bfloat16()
    dq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w8", qp, sc, n, k, dq, ds)
    wref = (q.float().view(n, k // 128, 128) * sc.float().unsqueeze(-1)).view(n, k)
    return dq, ds, wref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default=",".join(str(m) for m in range(1, 33)))
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--bench-ms", default="1,2,4,8,12,16,24,32")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    qe.call("qe_gemv_workspace_init")
    ms = [int(v) for v in a.ms.split(",")]
    for si, name in enumerate(a.shapes.split(",")):
        n, k = SHAPES[name]
        dq, ds, wref = make(n, k, si + 1)
        x = (torch.randn(32, k, device="cuda") * torch.exp(torch.randn(k, device="cuda") * 0.5)).bfloat16()
        alone = torch.empty(32, n, device="cuda")
        for r in range(32):  # the single-row GEMV of every row: the reference for batch == alone
            qe.call("qe_gemv_w8", dq, ds, n, k, x[r:r + 1], alone[r:r + 1], 1)
        qe.sync()
        ref = x[:32].half().float() @ wref.T
        rel = ((alone - ref).abs().max() / ref.abs().max()).item()
        assert rel < 1e-2, (name, rel)
        bad = []
        for m in ms:
            y = torch.full((m, n), float("nan"), device="cuda")
            ok = qe._LIB.qe_skinny_w8(qe._p(dq), qe._p(ds), n, k, qe._p(x), qe._p(y), m)
            qe.sync()
            assert ok, (name, m, "unsupported")
            if not torch.equal(y, alone[:m]):
                d = (y - alone[:m]).abs().max().item()
                bad.append((m, d))
        print(f"{name:8s} {n}x{k}: gemv_w8 vs torch {rel:.1e}; skinny_w8 == alone for m in {ms[0]}..{ms[-1]}: "
              f"{'OK' if not bad else 'MISMATCH ' + str(bad[:6])}", flush=True)
        assert not bad, (name, bad)
        if a.no_bench:
            continue
        b8 = n * k + n * (k // 128) * 2
        for m in [int(v) for v in a.bench_ms.split(",")]:
            y = torch.empty(m, n, device="cuda")
            tg = G.bench(lambda: qe.call("qe_gemv_w8", dq, ds, n, k, x, y, m), a.iters)
            ts = G.bench(lambda: qe._LIB.qe_skinny_w8(qe._p(dq), qe._p(ds), n, k, qe._p(x), qe._p(y), m), a.iters)
            print(f"   m={m:2d}: gemv_w8 {tg:6.1f} us {b8 / tg / 1e3:4.0f} GB/s | skinny_w8 {ts:6.1f} us "
                  f"{b8 / ts / 1e3:4.0f} GB/s | x{tg / ts:.2f}", flush=True)
    print("OK")


if __name__ == "__main__":
    main()
