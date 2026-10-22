"""GEMV W4/W16 correctness vs torch and bandwidth benchmark, on real checkpoint weights."""
import os
import sys

import torch
from safetensors import safe_open

import qe

MODEL = os.path.join(os.path.dirname(__file__), "..", "models/Qwen3.8-27B-INT4/model.safetensors")


def unpack(wp, n, k):
    shifts = torch.arange(0, 32, 4, device=wp.device, dtype=torch.int32)
    return (((wp.unsqueeze(-1) >> shifts) & 15).reshape(n, k) - 8).to(torch.float32)


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def test_w4(name):
    with safe_open(MODEL, "pt", device="cuda") as f:
        wp = f.get_tensor(name + ".weight_packed")
        sc = f.get_tensor(name + ".weight_scale")
    n, k = wp.shape[0], wp.shape[1] * 8
    # reference decompression: (q - 8) * scale rounded to bf16, like the kernels
    wref = (unpack(wp, n, k) * sc.float().repeat_interleave(128, dim=1)).bfloat16().float()
    dq = torch.empty(n * k // 2, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w4", wp, sc, n, k, dq, ds)
    worst = 0.0
    for m in list(range(1, 9)) + [9, 12, 16, 17, 24, 32]:
        x = torch.randn(m, k, device="cuda").bfloat16()
        y = torch.empty(m, n, device="cuda")
        qe.call("qe_gemv_w4", dq, ds, n, k, x, y, m)
        ref = x.float() @ wref.T
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, err)
    x = torch.randn(1, k, device="cuda").bfloat16()
    y = torch.empty(8, n, device="cuda")
    us1 = bench(lambda: qe.call("qe_gemv_w4", dq, ds, n, k, x, y, 1))
    x8 = torch.randn(8, k, device="cuda").bfloat16()
    us8 = bench(lambda: qe.call("qe_gemv_w4", dq, ds, n, k, x8, y, 8))
    gbs = (dq.numel() + ds.numel() * 2) / us1 / 1e3
    print(f"W4  {name.split('layers.')[-1]:40s} {n:6d}x{k:6d} relerr={worst:.2e}  "
          f"m=1 {us1:7.1f}us ({gbs:5.0f} GB/s)  m=8 {us8:7.1f}us")
    return worst


def test_w16(n, k):
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    dw = torch.empty(n * k, dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w16", w, n, k, dw)
    worst = 0.0
    for m in (1, 3, 8):
        x = torch.randn(m, k, device="cuda").bfloat16()
        y = torch.empty(m, n, device="cuda")
        qe.call("qe_gemv_w16", dw, n, k, x, y, m)
        ref = x.float() @ w.float().T
        worst = max(worst, ((y - ref).abs().max() / ref.abs().max()).item())
    x = torch.randn(1, k, device="cuda").bfloat16()
    y = torch.empty(1, n, device="cuda")
    us = bench(lambda: qe.call("qe_gemv_w16", dw, n, k, x, y, 1))
    print(f"W16 {n:6d}x{k:6d} relerr={worst:.2e}  m=1 {us:7.1f}us ({n*k*2/us/1e3:5.0f} GB/s)")
    return worst


if __name__ == "__main__":
    torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    L = "model.language_model.layers."
    errs = [test_w4(L + n) for n in [
        "0.mlp.gate_proj", "0.mlp.down_proj", "0.linear_attn.in_proj_qkv",
        "0.linear_attn.out_proj", "3.self_attn.q_proj", "3.self_attn.k_proj"]]
    errs += [test_w16(82784, 5120), test_w16(48, 5120)]
    assert max(errs) < 1e-2, errs
    print("OK")
