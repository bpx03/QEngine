"""repack_w8 (compressed-tensors INT8 -> W8 layout): bit-exact vs quantize_w8, gemv_w8 vs torch incl. w = -128,
GLU row interleave at 8 bits, and real INT8-checkpoint tensors vs torch dequant (when the checkpoint is present)."""
import os
import torch
from safetensors import safe_open

import qe

torch.cuda.set_device(0)
MODEL = os.path.join(os.path.dirname(__file__), "..", "models/Qwen3.8-27B-INT8")


def ct_pack(q):
    """int8 [n, k] -> compressed-tensors int32 [n, k/4] (u = q + 128, bytes LSB-first)"""
    return (q.to(torch.int16) + 128).to(torch.uint8).contiguous().view(torch.int32)


def repack(qp, sc, n, k):
    dq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    ds = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_repack_w8", qp, sc, n, k, dq, ds)
    return dq, ds


def gemv_err(dq, ds, n, k, wref):
    worst = 0
    for m in (1, 3, 8):
        x = torch.randn(m, k, device="cuda").bfloat16()
        y = torch.empty(m, n, device="cuda")
        qe.call("qe_gemv_w8", dq, ds, n, k, x, y, m)
        ref = x.half().float() @ wref.T
        worst = max(worst, ((y - ref).abs().max() / ref.abs().max()).item())
    return worst


g = torch.Generator(device="cuda").manual_seed(8)
for n, k in [(256, 5120), (1024, 17408), (8704, 5120)]:
    # 1) bit-exact vs quantize_w8: power-of-two group scales and |q| <= 127 with a 127 in every group make
    #    w = q * s exact in bf16, so RTN recovers the same q and s
    q = torch.randint(-127, 128, (n, k), device="cuda", generator=g, dtype=torch.int32)
    q.view(n, k // 128, 128)[:, :, 0] = 127
    s = torch.pow(2.0, torch.randint(-12, -4, (n, k // 128), device="cuda", generator=g).float())
    w = (q.float().view(n, k // 128, 128) * s.unsqueeze(-1)).view(n, k).bfloat16()
    rq = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    rs = torch.empty(n * (k // 128), dtype=torch.bfloat16, device="cuda")
    qe.call("qe_quantize_w8", w, n, k, rq, rs)
    dq, ds = repack(ct_pack(q.to(torch.int8)), s.bfloat16().contiguous(), n, k)
    qe.sync()
    assert torch.equal(dq, rq), f"{n}x{k}: weight bytes differ from quantize_w8"
    assert torch.equal(ds, rs), f"{n}x{k}: scales differ from quantize_w8"
    # 2) full int8 range (w = -128 occurs in the checkpoint), random bf16 scales, through gemv_w8
    q = torch.randint(-128, 128, (n, k), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(n, k // 128, device="cuda", generator=g) * 2e-4 + 1e-5).bfloat16()
    wref = (q.float().view(n, k // 128, 128) * s.float().unsqueeze(-1)).view(n, k)
    dq, ds = repack(ct_pack(q.to(torch.int8)), s, n, k)
    err = gemv_err(dq, ds, n, k, wref)
    print(f"synthetic {n:>5}x{k:<5}: bit-exact vs quantize_w8, gemv_w8 relerr {err:.2e} (q in [-128, 127])")
    assert err < 1e-2

# 3) GLU interleave at 8 bits: row order 8 gate, 8 up, ... with whole K-byte rows and their scales
n, k = 512, 1024
q = torch.randint(-128, 128, (n, k), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
s = torch.rand(n, k // 128, device="cuda", generator=g).bfloat16()
gq = torch.empty(n, k // 4, dtype=torch.int32, device="cuda")
gs = torch.empty_like(s)
qe.call("qe_glu_interleave_rows", ct_pack(q), s, n, k, gq, gs, 8)
qe.sync()
order = torch.tensor([t * 8 + j if j < 8 else n // 2 + t * 8 + j - 8 for t in range(n // 16) for j in range(16)],
                     device="cuda")
assert torch.equal(gq, ct_pack(q)[order]) and torch.equal(gs, s[order]), "glu interleave (8-bit)"
print("glu_interleave_rows bits=8: OK")

# 4) real tensors from the INT8 checkpoint (row and column slices as a TP shard would take them)
idx = os.path.join(MODEL, "model.safetensors.index.json")
if os.path.exists(idx):
    import json
    wm = json.load(open(idx))["weight_map"]
    for name, rows, cols in [("model.language_model.layers.0.mlp.down_proj", (0, 5120), (0, 8704)),
                             ("model.language_model.layers.0.mlp.gate_proj", (8704, 17408), None),
                             ("model.language_model.layers.3.self_attn.q_proj", (6144, 12288), None)]:
        with safe_open(os.path.join(MODEL, wm[name + ".weight_packed"]), "pt") as f:
            qp, sc = f.get_tensor(name + ".weight_packed"), f.get_tensor(name + ".weight_scale")
        K = qp.shape[1] * 4
        c0, c1 = cols or (0, K)
        u = qp[rows[0]:rows[1]].contiguous().view(torch.uint8)[:, c0:c1].contiguous()
        sc = sc[rows[0]:rows[1], c0 // 128:c1 // 128].contiguous()
        n, k = u.shape
        wref = ((u.cuda().to(torch.int32) - 128).float().view(n, k // 128, 128) * sc.cuda().float().unsqueeze(-1)).view(n, k)
        dq, ds = repack(u.cuda().view(torch.int32), sc.cuda(), n, k)
        err = gemv_err(dq, ds, n, k, wref)
        print(f"checkpoint {name.split('layers.')[1]:<22} rows {rows} cols {c0}:{c1}: gemv_w8 relerr {err:.2e}")
        assert err < 1e-2
else:
    print(f"skip checkpoint tensors ({MODEL} not found)")
print("OK")
