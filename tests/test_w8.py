"""INT8 GEMV (lm_head path) vs torch with the same RTN quantization."""
import torch, qe, test_gemv as G
torch.cuda.set_device(0)
for n, k in [(82816, 5120), (256, 5120)]:
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    g = w.float().view(n, k // 128, 128)
    sc = (g.abs().amax(-1) / 127).bfloat16().float()
    qref = torch.clamp(torch.round(g / torch.where(sc > 0, sc, 1).unsqueeze(-1)), -127, 127)
    wref = (qref * sc.unsqueeze(-1)).view(n, k)
    dq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_quantize_w8", w, n, k, dq, ds)
    worst = 0
    for m in (1, 4, 8):
        x = torch.randn(m, k, device="cuda").bfloat16()
        y = torch.empty(m, n, device="cuda")
        qe.call("qe_gemv_w8", dq, ds, n, k, x, y, m)
        ref = x.half().float() @ wref.T
        worst = max(worst, ((y - ref).abs().max() / ref.abs().max()).item())
    x = torch.randn(1, k, device="cuda").bfloat16(); y = torch.empty(1, n, device="cuda")
    us = G.bench(lambda: qe.call("qe_gemv_w8", dq, ds, n, k, x, y, 1), 100)
    print(f"W8 {n}x{k}: relerr {worst:.2e}, {us:.1f} us ({n*k*1.03/us/1e3:.0f} GB/s)  [bf16 head ~950 us]")
    assert worst < 1e-2
print("OK")
