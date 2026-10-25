#!/usr/bin/env python3
"""Convert Intel/Qwen3.8-Flash-Next-W4A16-AutoRound into the qengine INT4 checkpoint.

  python tools/convert_flash_next.py [--src DIR] [--dst DIR] [--workers N] [--layers 0,1,..] [--verify-only]

Output (row-major "compressed-tensors pack-quantized" layout, the one repack_w4 / load_w4 consume):
  <name>.weight_packed  int32 [N, K/8]   nibble j of word c = k = 8c + j, stored u = w + 8 (sym, LSB first)
  <name>.weight_scale   [N, K/128]        group scale along K (g128)
  w = (u - 8) * scale

* Routed experts (AutoRound INT4, auto_gptq v1 packing: qweight [K/8, N] packed along K, qzeros = 7 -> zero 8,
  fp16 scales [K/128, N]) are repacked losslessly: weight_packed = qweight.T (the int32 words are identical,
  only transposed), weight_scale = scales.T kept in **fp16** (bf16 would round the scale). Stacked per layer:
    layers.L.mlp.experts.gate_up_proj.weight_packed [512, 1280, 320] (rows 0..639 gate, 640..1279 up), scale f16
    layers.L.mlp.experts.down_proj.weight_packed    [512, 2560, 80], scale f16 [512, 2560, 5]
* BF16 dense matrices on the decode critical path are RTN-quantized here (sym INT4 g128, per-group scale search
  over the clip ratio minimizing the squared error): GDN in_proj_qkv / in_proj_z / out_proj, QSA q/k/v/o,
  PLE key/value proj, shared expert (scales f16, it runs in the expert kernel), MTP experts + shared (f16),
  MTP q/k/v/o and fc_embedding / fc_hidden (bf16 scales).
* Kept BF16 (the engine decides INT8 at load): hyper-connection matrices,
  router, lm_head, embed_tokens, indexer, gates (shared_expert_gate, in_proj_a/b, block_inject), norms, conv,
  A_log / dt_bias, vision. The n-gram table shard is symlinked, not copied.
Files: base.safetensors (embed, lm_head, final mixer, vision), layer-LL.safetensors, mtp.safetensors, n-gram
symlink, model.safetensors.index.json, config/tokenizer copies, convert_report.json (RTN error stats).
"""
import argparse, glob, json, os, shutil, struct, sys, time
from concurrent.futures import ProcessPoolExecutor

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-W4A16-AutoRound")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4")
P = "model.language_model."
NE, H, I = 512, 2560, 640

# QSA q/k/v/o stay BF16 (engine: INT8 at load): RTN INT4 of them lengthened reasoning ~2.9x in a community
# W4A16 (community report); MTP stays BF16 (fp8 MTP halved acceptance, HF #47)
RTN_BF16 = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
            "ple.key_proj", "ple.value_proj")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


class Src:
    """tensor name -> file, lazily opened safe_open handles"""
    def __init__(self, d):
        self.d = d
        self.where = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            for k in header(f):
                self.where[k] = f
        self.h = {}

    def get(self, k):
        f = self.where[k]
        if f not in self.h:
            self.h[f] = safe_open(f, "pt", device="cpu")
        return self.h[f].get_tensor(k)

    def keys(self, prefix):
        return [k for k in self.where if k.startswith(prefix)]


# ---------------------------------------------------------------- INT4 helpers
def pack_rows(u):
    """u: uint8-valued int32 [N, K] in 0..15 -> int32 [N, K/8], nibble j of word c = column 8c + j"""
    N, K = u.shape
    u = u.reshape(N, K // 8, 8).to(torch.int64)
    w = torch.zeros(N, K // 8, dtype=torch.int64)
    for j in range(8):
        w |= u[:, :, j] << (4 * j)
    w = torch.where(w >= 2**31, w - 2**32, w)
    return w.to(torch.int32)


def unpack_rows(q):
    """int32 [..., N, K/8] -> int32 [..., N, K] values 0..15"""
    q = q.to(torch.int64) & 0xFFFFFFFF
    sh = torch.arange(8, dtype=torch.int64) * 4
    u = (q.unsqueeze(-1) >> sh) & 15
    return u.reshape(*q.shape[:-1], q.shape[-1] * 8).to(torch.int32)


def dequant_ours(q, s):
    """our layout -> fp32 [..., N, K]"""
    u = unpack_rows(q).float() - 8.0
    sc = s.float().repeat_interleave(128, dim=-1)
    return u * sc


def dequant_gptq(qweight, qzeros, scales):
    """auto_gptq v1 (zero stored minus 1) -> fp32 [N, K] (nn.Linear orientation)"""
    Kp, N = qweight.shape
    q = qweight.to(torch.int64) & 0xFFFFFFFF
    sh = torch.arange(8, dtype=torch.int64) * 4
    u = ((q.unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(Kp * 8, N)  # [K, N], k = 8r + j
    z = qzeros.to(torch.int64) & 0xFFFFFFFF
    zu = ((z.unsqueeze(-1) >> sh) & 15).reshape(z.shape[0], z.shape[1] * 8) + 1  # [K/128, N]
    zz = zu.repeat_interleave(128, dim=0)
    sc = scales.float().repeat_interleave(128, dim=0)
    return ((u - zz).float() * sc).t().contiguous()


def rtn_int4(w, scale_dtype, search=True):
    """w [N, K] float -> (packed int32 [N, K/8], scale [N, K/128] scale_dtype, stats). Symmetric INT4 g128,
    stored u = q + 8, q in [-8, 7]. Scale = absmax / 7 * c, c from a small grid minimizing the group's
    squared error (c = 1: plain absmax RTN)."""
    N, K = w.shape
    g = w.float().reshape(N, K // 128, 128)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    cands = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75] if search else [1.0]
    best_err, best_s = None, None
    for c in cands:
        s = (amax / 7.0 * c).to(scale_dtype).float()
        qv = torch.clamp(torch.round(g / s), -8, 7)
        err = ((qv * s - g) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s, best_s)
    qv = torch.clamp(torch.round(g / best_s), -8, 7)
    deq = (qv * best_s).reshape(N, K)
    u = (qv + 8).to(torch.int32).reshape(N, K)
    wf = w.float()
    rel = ((deq - wf).norm() / wf.norm().clamp_min(1e-12)).item()
    # plain absmax RTN for comparison
    s0 = (amax / 7.0).to(scale_dtype).float()
    d0 = (torch.clamp(torch.round(g / s0), -8, 7) * s0).reshape(N, K)
    rel0 = ((d0 - wf).norm() / wf.norm().clamp_min(1e-12)).item()
    return pack_rows(u), best_s.reshape(N, K // 128).to(scale_dtype), {"rel_err": rel, "rel_err_absmax": rel0}


# ---------------------------------------------------------------- per-unit conversion
def conv_layer(args):
    L, src_dir, dst_dir, nverify = args
    t0 = time.time()
    torch.set_num_threads(6)
    S = Src(src_dir)
    pre = f"{P}layers.{L}."
    out, stats = {}, {}
    # routed experts: lossless repack
    gu_q = torch.empty(NE, 2 * I, H // 8, dtype=torch.int32)
    gu_s = torch.empty(NE, 2 * I, H // 128, dtype=torch.float16)
    dn_q = torch.empty(NE, H, I // 8, dtype=torch.int32)
    dn_s = torch.empty(NE, H, I // 128, dtype=torch.float16)
    for e in range(NE):
        b = f"{pre}mlp.experts.{e}."
        for nm, (qd, sd, r0) in (("gate_proj", (gu_q, gu_s, 0)), ("up_proj", (gu_q, gu_s, I)), ("down_proj", (dn_q, dn_s, 0))):
            qw, qz, sc = S.get(b + nm + ".qweight"), S.get(b + nm + ".qzeros"), S.get(b + nm + ".scales")
            assert sc.dtype == torch.float16
            assert bool((qz == 0x77777777).all()), f"non-symmetric zero point in {b}{nm}"
            n = qw.shape[1]
            qd[e, r0:r0 + n] = qw.t()
            sd[e, r0:r0 + n] = sc.t()
    out[pre + "mlp.experts.gate_up_proj.weight_packed"] = gu_q
    out[pre + "mlp.experts.gate_up_proj.weight_scale"] = gu_s
    out[pre + "mlp.experts.down_proj.weight_packed"] = dn_q
    out[pre + "mlp.experts.down_proj.weight_scale"] = dn_s
    # bit-exactness check on sample experts (dequantized repack == dequantized original, fp32 exact)
    g = torch.Generator().manual_seed(L)
    ver = []
    for e in torch.randperm(NE, generator=g)[:nverify].tolist():
        b = f"{pre}mlp.experts.{e}."
        for nm, qd, sd, r0 in (("gate_proj", gu_q, gu_s, 0), ("up_proj", gu_q, gu_s, I), ("down_proj", dn_q, dn_s, 0)):
            ref = dequant_gptq(S.get(b + nm + ".qweight"), S.get(b + nm + ".qzeros"), S.get(b + nm + ".scales"))
            n = ref.shape[0]
            mine = dequant_ours(qd[e, r0:r0 + n], sd[e, r0:r0 + n])
            ver.append(bool(torch.equal(ref, mine)))
    stats["experts_verified"] = len(ver)
    stats["experts_bitexact"] = all(ver)
    # shared expert: RTN, fp16 scales (runs in the expert kernel); gate_up stacked like the experts
    sh = {}
    for nm in ("gate_proj", "up_proj", "down_proj"):
        w = S.get(f"{pre}mlp.shared_expert.{nm}.weight")
        q, s, st = rtn_int4(w, torch.float16)
        sh[nm] = (q, s)
        stats[f"shared_expert.{nm}"] = st
    out[pre + "mlp.shared_expert.gate_up_proj.weight_packed"] = torch.cat([sh["gate_proj"][0], sh["up_proj"][0]])
    out[pre + "mlp.shared_expert.gate_up_proj.weight_scale"] = torch.cat([sh["gate_proj"][1], sh["up_proj"][1]])
    out[pre + "mlp.shared_expert.down_proj.weight_packed"] = sh["down_proj"][0]
    out[pre + "mlp.shared_expert.down_proj.weight_scale"] = sh["down_proj"][1]
    # everything else of the layer
    for k in S.keys(pre):
        rest = k[len(pre):]
        if rest.startswith("mlp.experts.") or rest.startswith("mlp.shared_expert.") or "ngram_embedding" in rest:
            continue
        base = rest[:-len(".weight")] if rest.endswith(".weight") else None
        if base in RTN_BF16:
            q, s, st = rtn_int4(S.get(k), torch.bfloat16)
            out[pre + base + ".weight_packed"] = q
            out[pre + base + ".weight_scale"] = s
            stats[base] = st
        else:
            out[k] = S.get(k).contiguous()
    fn = f"layer-{L:02d}.safetensors"
    save_file(out, os.path.join(dst_dir, fn))
    return L, fn, list(out.keys()), stats, time.time() - t0


def conv_mtp_bf16(args):
    src_dir, dst_dir = args
    t0 = time.time()
    S = Src(src_dir)
    out = {k: S.get(k).contiguous() for k in S.keys("mtp.")}
    save_file(out, os.path.join(dst_dir, "mtp.safetensors"))
    return "mtp.safetensors", list(out.keys()), {"mtp": "bf16 passthrough"}, time.time() - t0


def conv_mtp(args):
    src_dir, dst_dir = args
    t0 = time.time()
    torch.set_num_threads(16)
    S = Src(src_dir)
    out, stats = {}, {}
    pre = "mtp.layers.0."
    gu_q = torch.empty(NE, 2 * I, H // 8, dtype=torch.int32)
    gu_s = torch.empty(NE, 2 * I, H // 128, dtype=torch.float16)
    dn_q = torch.empty(NE, H, I // 8, dtype=torch.int32)
    dn_s = torch.empty(NE, H, I // 128, dtype=torch.float16)
    errs = {"gate_proj": [], "up_proj": [], "down_proj": []}
    errs0 = {"gate_proj": [], "up_proj": [], "down_proj": []}
    for e in range(NE):
        b = f"{pre}mlp.experts.{e}."
        for nm, qd, sd, r0 in (("gate_proj", gu_q, gu_s, 0), ("up_proj", gu_q, gu_s, I), ("down_proj", dn_q, dn_s, 0)):
            q, s, st = rtn_int4(S.get(b + nm + ".weight"), torch.float16)
            qd[e, r0:r0 + q.shape[0]] = q
            sd[e, r0:r0 + q.shape[0]] = s
            errs[nm].append(st["rel_err"])
            errs0[nm].append(st["rel_err_absmax"])
    for nm in errs:
        v = torch.tensor(errs[nm]); v0 = torch.tensor(errs0[nm])
        stats[f"mtp.experts.{nm}"] = {"rel_err_mean": v.mean().item(), "rel_err_max": v.max().item(),
                                      "rel_err_absmax_mean": v0.mean().item()}
    out[pre + "mlp.experts.gate_up_proj.weight_packed"] = gu_q
    out[pre + "mlp.experts.gate_up_proj.weight_scale"] = gu_s
    out[pre + "mlp.experts.down_proj.weight_packed"] = dn_q
    out[pre + "mlp.experts.down_proj.weight_scale"] = dn_s
    sh = {}
    for nm in ("gate_proj", "up_proj", "down_proj"):
        q, s, st = rtn_int4(S.get(f"{pre}mlp.shared_expert.{nm}.weight"), torch.float16)
        sh[nm] = (q, s)
        stats[f"mtp.shared_expert.{nm}"] = st
    out[pre + "mlp.shared_expert.gate_up_proj.weight_packed"] = torch.cat([sh["gate_proj"][0], sh["up_proj"][0]])
    out[pre + "mlp.shared_expert.gate_up_proj.weight_scale"] = torch.cat([sh["gate_proj"][1], sh["up_proj"][1]])
    out[pre + "mlp.shared_expert.down_proj.weight_packed"] = sh["down_proj"][0]
    out[pre + "mlp.shared_expert.down_proj.weight_scale"] = sh["down_proj"][1]
    rtn = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    for k in S.keys("mtp."):
        if ".mlp.experts." in k or ".mlp.shared_expert." in k:
            continue
        nm = k[len(pre):-len(".weight")] if k.startswith(pre) and k.endswith(".weight") else None
        top = k[len("mtp."):-len(".weight")] if k.endswith(".weight") else None
        if nm in rtn or top in ("fc_embedding", "fc_hidden"):
            base = k[:-len(".weight")]
            q, s, st = rtn_int4(S.get(k), torch.bfloat16)
            out[base + ".weight_packed"] = q
            out[base + ".weight_scale"] = s
            stats[base] = st
        else:
            out[k] = S.get(k).contiguous()
    save_file(out, os.path.join(dst_dir, "mtp-int4.safetensors"))
    return "mtp-int4.safetensors", list(out.keys()), stats, time.time() - t0


def conv_base(args):
    src_dir, dst_dir = args
    t0 = time.time()
    S = Src(src_dir)
    out = {}
    for k in S.where:
        if k.startswith(P + "layers.") or k.startswith("mtp."):
            continue
        out[k] = S.get(k).contiguous()
    save_file(out, os.path.join(dst_dir, "base.safetensors"))
    return "base.safetensors", list(out.keys()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layers", default="")
    ap.add_argument("--verify", type=int, default=4, help="sample experts per layer checked bit-exact")
    ap.add_argument("--skip-mtp", action="store_true")
    ap.add_argument("--mtp-int4", action="store_true", help="also write mtp-int4.safetensors: RTN INT4 MTP experts/shared (fp16 "
                    "scales, stacked like the trunk) + attention/fc (default: only the BF16 passthrough mtp.safetensors)")
    ap.add_argument("--skip-base", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    S = Src(a.src)
    ngram_files = sorted({f for k, f in S.where.items() if "ngram_embedding" in k})
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(48))
    rep_path = os.path.join(a.dst, "convert_report.json")
    report = json.load(open(rep_path)) if os.path.exists(rep_path) else {"layers": {}}
    wmap = {}
    idx_path = os.path.join(a.dst, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path))["weight_map"]
    for f in ngram_files:  # n-gram table: symlink
        dst = os.path.join(a.dst, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(f, dst)
        for k in header(f):
            wmap[k] = os.path.basename(f)
    for fn in ("config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json",
               "tokenizer_config.json", "preprocessor_config.json", "processor_config.json"):
        if os.path.exists(os.path.join(a.src, fn)):
            shutil.copy(os.path.join(a.src, fn), os.path.join(a.dst, fn))
    # the AutoRound repo ships the 27B tokenizer.json (pre-tokenizer \p{L}+ instead of Flash-Next's [\p{L}\p{M}]+):
    # use the official one when it is in the HF cache
    off = os.path.expanduser("~/.cache/qengine/hf/Qwen3.8-Flash-Next/tokenizer.json")
    if os.path.exists(off):
        shutil.copy(os.path.join(a.src, "tokenizer.json"), os.path.join(a.dst, "tokenizer.autoround.json"))
        shutil.copy(off, os.path.join(a.dst, "tokenizer.json"))
    # config: drop the AutoRound quantization_config, describe ours
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg.pop("quantization_config", None)
    cfg["qengine_quant"] = {"format": "pack-quantized-int4-sym-g128", "experts": "AutoRound lossless repack, fp16 scales",
                            "dense_rtn": list(RTN_BF16) + ["shared_expert"] + (["mtp"] if a.mtp_int4 else []), "bf16_kept": "qsa q/k/v/o, mtp, hc, router, lm_head, embed, indexer, gates, norms"}
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=1)

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(conv_layer, (L, a.src, a.dst, a.verify)) for L in layers]
        fb = None if a.skip_base else ex.submit(conv_base, (a.src, a.dst))
        fm = None if a.skip_mtp else ex.submit(conv_mtp if a.mtp_int4 else conv_mtp_bf16, (a.src, a.dst))
        fm2 = ex.submit(conv_mtp_bf16, (a.src, a.dst)) if (a.mtp_int4 and not a.skip_mtp) else None
        for fu in futs:
            L, fn, keys, st, dt = fu.result()
            for k in keys:
                wmap[k] = fn
            report["layers"][str(L)] = st
            print(f"layer {L:2d} {dt:6.1f}s bitexact={st['experts_bitexact']} ({st['experts_verified']} matrices)", flush=True)
        if fb:
            fn, keys, dt = fb.result()
            for k in keys:
                wmap[k] = fn
            print(f"base {dt:.1f}s", flush=True)
        if fm:
            fn, keys, st, dt = fm.result()
            for k in keys:
                wmap[k] = fn
            report["mtp"] = st
            print(f"mtp {dt:.1f}s", flush=True)
        if fm2:
            fn, keys, _, dt = fm2.result()
            for k in keys:
                wmap.setdefault(k, fn)
    json.dump({"metadata": {}, "weight_map": wmap}, open(idx_path, "w"), indent=1)
    json.dump(report, open(rep_path, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
