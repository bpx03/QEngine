"""Batch == alone for the decode GEMVs of INT8 setups: gemv_w8 (layer matrices, plain and GLU) and the bf16-input
gemv_w4 (MTP / draft head without H16) must give every row bit for bit what the same row gives alone, at every row
count 1..64. Prints the first mismatching m per shape.
Usage: python3 tests/test_w8_rows.py [--ms 1,2,...]"""
import argparse

import torch

import qe
from test_skinny_w8 import SHAPES, make


def make_w4(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    u = torch.randint(0, 16, (n, k), device="cuda", generator=g, dtype=torch.int32)
    packed = torch.zeros(n, k // 8, dtype=torch.int32, device="cuda")
    for i in range(8):
        packed |= u[:, i::8] << (4 * i)
    sc = (torch.rand(n, k // 128, device="cuda", generator=g) * 2e-3 + 1e-4).bfloat16()
    dq = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w4", packed, sc, n, k, dq, ds)
    return dq, ds


def check(name, fn, n_out, k, ms, dt=torch.float32):
    x = (torch.randn(64, k, device="cuda") * 0.5).bfloat16()
    alone = torch.empty(64, n_out, dtype=dt, device="cuda")
    for r in range(64):
        y = torch.empty(1, n_out, dtype=dt, device="cuda")
        fn(x[r:r + 1].contiguous(), y, 1)
        alone[r] = y[0]
    bad = []
    for m in ms:
        y = torch.empty(m, n_out, dtype=dt, device="cuda")
        fn(x[:m].contiguous(), y, m)
        if not torch.equal(y.view(torch.int32 if dt == torch.float32 else torch.int16),
                           alone[:m].view(torch.int32 if dt == torch.float32 else torch.int16)):
            rows = [r for r in range(m) if not torch.equal(y[r], alone[r])]
            bad.append((m, rows[:4]))
    print(f"{name:24s} {'OK' if not bad else 'MISMATCH ' + str(bad[:6])}", flush=True)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default=",".join(str(m) for m in range(1, 65)))
    a = ap.parse_args()
    ms = [int(v) for v in a.ms.split(",")]
    torch.cuda.set_device(0)
    qe.call("qe_gemv_workspace_init")
    ok = True
    for si, (name, (n, k)) in enumerate(SHAPES.items()):
        dq, ds, _ = make(n, k, si + 1)
        ok &= check("w8 " + name, lambda x, y, m: qe.call("qe_gemv_w8", dq, ds, n, k, x, y, m), n, k, ms)
        if name == "gate_up":
            ok &= check("w8 glu " + name, lambda x, y, m: qe.call("qe_gemv_w8_glu", dq, ds, n, k, x, y, m),
                        n // 2, k, ms, torch.bfloat16)
        q4, s4 = make_w4(n, k, si + 11)
        ok &= check("w4 bf16 " + name, lambda x, y, m: qe.call("qe_gemv_w4", q4, s4, n, k, x, y, m),
                    n, k, [m for m in ms if m <= 64])
    # DeltaNet a/b gate projection without H16: bf16 GEMV with 16 K-splits (partials per split)
    n, k, ks = 48, 5120, 16
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    wp = torch.empty(n * k, dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w16", w, n, k, wp)
    x = (torch.randn(64, k, device="cuda") * 0.5).bfloat16()
    alone = []
    for r in range(64):
        y = torch.empty(ks, 1, n, device="cuda")
        qe.call("qe_gemv_w16k", wp, n, k, x[r:r + 1].contiguous(), y, 1, ks, n)
        alone.append(y[:, 0])
    bad = []
    for m in ms:
        y = torch.empty(ks, m, n, device="cuda")
        qe.call("qe_gemv_w16k", wp, n, k, x[:m].contiguous(), y, m, ks, m * n)
        rows = [r for r in range(m) if not torch.equal(y[:, r], alone[r])]
        if rows: bad.append((m, rows[:4]))
    print(f"{'w16 ab ksplit16':24s} {'OK' if not bad else 'MISMATCH ' + str(bad[:6])}", flush=True)
    ok &= not bad
    print("OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
