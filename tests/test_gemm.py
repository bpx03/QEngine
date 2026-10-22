"""Prefill GEMM correctness vs torch (bf16-dequantized weights) and TFLOPS."""
import sys
import torch
import qe
import test_gemv as G

def run(n, k, t, w4=True):
    torch.manual_seed(0)
    if w4:
        wp = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
        sc = (torch.rand(n, k // 128, device="cuda") * 0.01 + 0.001).bfloat16()
        wref = (G.unpack(wp, n, k) * sc.float().repeat_interleave(128, 1)).bfloat16().float()
        dq = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
        ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
        qe.call("qe_repack_w4", wp, sc, n, k, dq, ds)
        fn = lambda x, y: qe.call("qe_gemm_w4", dq, ds, n, k, x, y, t)
    else:
        w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
        wref = w.float()
        dw = torch.empty(n * k, dtype=torch.bfloat16, device="cuda")
        qe.call("qe_repack_w16", w, n, k, dw)
        fn = lambda x, y: qe.call("qe_gemm_w16", dw, n, k, x, y, t)
    x = torch.randn(t, k, device="cuda").bfloat16()
    y = torch.zeros(t, n, device="cuda")
    fn(x, y)
    ref = x.float() @ wref.T
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    us = G.bench(lambda: fn(x, y), 20)
    tf = 2 * n * k * t / us / 1e6
    print(f"{'W4 ' if w4 else 'W16'} N={n:6d} K={k:6d} T={t:5d} relerr={err:.2e} {us:8.1f}us {tf:6.1f} TFLOPS")
    return err

if __name__ == "__main__":
    torch.cuda.set_device(0)
    errs = []
    for (n, k) in [(11776, 5120), (5120, 5888), (5632, 5120), (5120, 2048)]:
        for t in (100, 512, 1024):
            errs.append(run(n, k, t))
    errs.append(run(32, 5120, 300, w4=False))
    errs.append(run(5120, 10240 // 3 // 128 * 128, 512, w4=False))
    assert max(errs) < 1e-2, errs
    print("OK")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "small":
    for t in (9, 17, 33, 50, 64, 100, 200):
        run(5120, 5888, t); run(11776, 5120, t)
