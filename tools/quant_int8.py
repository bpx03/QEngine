#!/usr/bin/env python3
"""Our INT8 checkpoint of Qwen3.8-27B for the engine's opt-in INT8 path.

Source: the official BF16 weights (Qwen/Qwen3.8-27B, Apache-2.0, sharded). Output: one model.safetensors in the
compressed-tensors pack-quantized layout the engine loads (int32 [N][K/4], bytes LSB-first, u = w + 128; bf16
scales [N][K/128]; weight_shape), plus model_mtp.safetensors (the stock BF16 MTP head) and the configs.

- Quantized: the layer projections the INT4 checkpoint quantizes (q/k/v/o, in_proj_qkv / in_proj_z / out_proj,
  gate / up / down). Kept BF16: embeddings, lm_head (the engine makes its own INT8 copy at load), norms, conv1d,
  A_log / dt_bias, in_proj_a / in_proj_b, the vision tower.
- Weights: symmetric INT8, group 128 along K. Default RTN: scale = absmax / 127 rounded to bf16, q = rn(w / scale)
  clamped to [-127, 127] (data-free, deterministic). --search N: per-group scale search over N clip ratios in
  [0.80, 1] minimising the group's squared error (also data-free; the clip ratio 1 is always a candidate).
- FP8 KV scales: k_scale / v_scale of the full-attention layers copied from a checkpoint of the same model
  (--kv-from, default the INT4 checkpoint: K / V activations do not depend on the weight format). Without them the
  engine keeps a bf16 KV cache (half the pool).
- Tokenizer / template / generation config: from --kv-from when it has them (the INT4 checkpoint the server's
  tokenizer handling is tested with; the BF16 repo's tokenizer.json is formatted differently), else from --src.
- No calibration data, no GPU needed (torch on CPU; --dev cuda:N runs the per-tensor math on a GPU instead).

Usage:
  .venv-ref/bin/python tools/quant_int8.py --src models/Qwen3.8-27B-BF16 --out models/Qwen3.8-27B-INT8
  .venv-ref/bin/python tools/quant_int8.py --check models/Qwen3.8-27B-INT8 --src models/Qwen3.8-27B-BF16
The output is written tensor by tensor (header first): peak RAM ~ one BF16 tensor, not the model.
"""
import argparse
import copy
import json
import os
import shutil
import struct
import sys
import time

import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
LM = "model.language_model."
QUANT = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
DT = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int32: "I32", torch.int64: "I64",
      torch.uint8: "U8", torch.int8: "I8", torch.float8_e4m3fn: "F8_E4M3"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Src:
    """lazy reader over a sharded (index.json) or single-file checkpoint"""

    def __init__(self, d, single="model.safetensors"):
        self.d, self.h = d, {}
        ix = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(ix) and not os.path.exists(os.path.join(d, single)):
            self.where = json.load(open(ix))["weight_map"]
        else:
            with safe_open(os.path.join(d, single), "pt") as f:
                self.where = {k: single for k in f.keys()}

    def keys(self):
        return list(self.where)

    def get(self, k):
        f = self.where[k]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(k)


def is_quant(name):
    return name.startswith(LM + "layers.") and name.endswith(".weight") and any(
        name[: -len(".weight")].endswith(q) for q in QUANT)


def quant_int8(w, search=0, dev="cpu"):
    """w [N, K] -> (q int8 [N, K], scale bf16 [N, K/128]); symmetric, group 128"""
    N, K = w.shape
    assert K % 128 == 0, (N, K)
    wf = w.to(dev, torch.float32).view(N, K // 128, 128)
    amax = wf.abs().amax(-1, keepdim=True)

    def q_err(ratio):
        s = (amax * ratio / 127.0).to(torch.bfloat16).float()
        s = torch.where(s > 0, s, torch.ones_like(s))
        q = torch.round(wf / s).clamp_(-127, 127)
        return q, s, ((q * s - wf) ** 2).sum(-1, keepdim=True)

    q, s, err = q_err(1.0)
    for i in range(1, search):
        r = 1.0 - 0.20 * i / max(1, search - 1)
        q2, s2, e2 = q_err(r)
        better = e2 < err
        q = torch.where(better, q2, q)
        s = torch.where(better, s2, s)
        err = torch.where(better, e2, err)
    return q.view(N, K).to(torch.int8).cpu(), s.view(N, K // 128).to(torch.bfloat16).cpu()


def pack_int8(q):
    """int8 [N, K] -> int32 [N, K/4]: byte b of word j = k 4j + b, stored u = w + 128 (compressed-tensors)"""
    u = (q.to(torch.int16) + 128).to(torch.uint8)
    return u.contiguous().view(torch.int32)


class Writer:
    """safetensors writer: header from the planned (name, dtype, shape) list, then the tensors in that order"""

    def __init__(self, path, plan, meta=None):
        self.plan, off, hdr = plan, 0, {}
        for name, dt, shape, nbytes in plan:
            hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nbytes]}
            off += nbytes
        if meta:
            hdr["__metadata__"] = meta
        h = json.dumps(hdr, separators=(",", ":")).encode()
        h += b" " * (-len(h) % 8)
        self.f = open(path + ".part", "wb")
        self.f.write(struct.pack("<Q", len(h)) + h)
        self.path, self.i = path, 0

    def put(self, name, t):
        pn, dt, shape, nbytes = self.plan[self.i]
        assert pn == name and DT[t.dtype] == dt and list(t.shape) == list(shape), (name, pn, t.dtype, t.shape)
        b = t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else t.numpy().tobytes()
        assert len(b) == nbytes, (name, len(b), nbytes)
        self.f.write(b)
        self.i += 1

    def close(self):
        assert self.i == len(self.plan), (self.i, len(self.plan))
        self.f.close()
        os.replace(self.path + ".part", self.path)


def esize(dt):
    return {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1}[dt]


def cmd_make(a):
    src = Src(a.src)
    kv = Src(a.kv_from) if a.kv_from else None
    os.makedirs(a.out, exist_ok=True)
    names = src.keys()
    main = [n for n in names if not n.startswith("mtp.")]
    mtp = [n for n in names if n.startswith("mtp.")]
    # plan the main file (shapes from the source headers; no tensor loaded yet)
    shapes = {}
    for n in names:
        f = src.where[n]
        if f not in src.h:
            src.h[f] = safe_open(os.path.join(src.d, f), "pt", device="cpu")
        sl = src.h[f].get_slice(n)
        shapes[n] = (sl.get_shape(), sl.get_dtype())
    plan = []
    kv_names = []
    for n in sorted(main):
        shape, dt = shapes[n]
        if is_quant(n):
            N, K = shape
            base = n[: -len(".weight")]
            plan += [(base + ".weight_packed", "I32", [N, K // 4], N * K),
                     (base + ".weight_scale", "BF16", [N, K // 128], N * (K // 128) * 2),
                     (base + ".weight_shape", "I64", [2], 16)]
        else:
            plan.append((n, dt, shape, esize(dt) * int(torch.tensor(shape).prod().item() if shape else 1)))
        if n.endswith("self_attn.q_norm.weight") and kv is not None:
            a_ = n[: -len("q_norm.weight")]
            for s in ("k_scale", "v_scale"):
                if a_ + s in kv.where:
                    t = kv.get(a_ + s)
                    kv_names.append((a_ + s, t))
                    plan.append((a_ + s, DT[t.dtype], list(t.shape), t.numel() * t.element_size()))
    plan.sort(key=lambda p: p[0])
    kvd = dict(kv_names)
    meta = {"format": "pt", "qengine": f"quant_int8.py rtn g128 search={a.search}"}
    w = Writer(os.path.join(a.out, "model.safetensors"), plan, meta)
    t0, done, nq = time.time(), 0, 0
    pending = {}
    for name, dt, shape, nbytes in plan:
        if name in kvd:
            w.put(name, kvd[name])
            continue
        if name.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
            base = name.rsplit(".", 1)[0]
            if base not in pending:
                wt = src.get(base + ".weight")
                q, s = quant_int8(wt, a.search, a.dev)
                pending = {base: (pack_int8(q), s, torch.tensor(list(wt.shape), dtype=torch.int64))}
                nq += 1
            pk, sc, sh = pending[base]
            w.put(name, pk if name.endswith("_packed") else sc if name.endswith("_scale") else sh)
        else:
            w.put(name, src.get(name))
        done += nbytes
        if w.i % 200 == 0:
            log(f"{w.i}/{len(plan)} tensors, {done / 1e9:.1f} GB, {nq} quantized, {time.time() - t0:.0f} s")
    w.close()
    log(f"model.safetensors: {len(plan)} tensors, {nq} INT8 matrices, {len(kv_names)} KV scales")
    # the stock MTP head, BF16 (the engine RTN-quantizes it to INT4 for drafting, QE_MTP_W4)
    if mtp:
        mp = [(n, shapes[n][1], shapes[n][0], esize(shapes[n][1]) * int(torch.tensor(shapes[n][0]).prod().item()))
              for n in sorted(mtp)]
        wm = Writer(os.path.join(a.out, "model_mtp.safetensors"), mp, {"format": "pt"})
        for n, *_ in mp:
            wm.put(n, src.get(n))
        wm.close()
        log(f"model_mtp.safetensors: {len(mp)} tensors (stock BF16 head)")
    # configs: the INT4 checkpoint's config (the engine's tested reference) with an 8-bit quantization_config
    ref_cfg = os.path.join(a.kv_from, "config.json") if a.kv_from else None
    cfg = json.load(open(ref_cfg if ref_cfg and os.path.exists(ref_cfg) else os.path.join(a.src, "config.json")))
    q = copy.deepcopy(cfg.get("quantization_config") or {})
    q.update({"format": "pack-quantized", "quant_method": "compressed-tensors", "quantization_status": "compressed"})
    grp = {"targets": ["Linear"], "input_activations": None, "output_activations": None, "format": "pack-quantized",
           "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128,
                       "dynamic": False, "actorder": None, "observer": "minmax" if a.search <= 1 else "mse",
                       "observer_kwargs": {}, "block_structure": None, "scale_dtype": None, "zp_dtype": None}}
    q["config_groups"] = {"group_0": grp}
    if not kv_names:
        q["kv_cache_scheme"] = None
    cfg["quantization_config"] = q
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "LICENSE",
              "vocab.json", "merges.txt"):
        for d in (a.kv_from or "", a.src):  # the INT4 checkpoint's files first: the ones the server is tested with
            p = os.path.join(d, f)
            if d and os.path.exists(p):
                shutil.copy(p, os.path.join(a.out, f))
                break
    with open(os.path.join(a.out, "QENGINE_INT8.md"), "w") as f:
        f.write(f"INT8 (W8A16, symmetric, group 128) of {os.path.basename(os.path.abspath(a.src))} by "
                f"tools/quant_int8.py (search={a.search}); KV scales from {a.kv_from or 'none'}; MTP head: stock BF16.\n")
    log("done", a.out)


def cmd_check(a):
    """read back: unpack + dequantize a sample of matrices and compare with the BF16 source"""
    out = Src(a.check)
    src = Src(a.src)
    names = sorted(n[: -len(".weight_packed")] for n in out.keys() if n.endswith(".weight_packed"))
    pick = names[:: max(1, len(names) // a.n)][: a.n]
    for b in pick:
        pk, sc = out.get(b + ".weight_packed"), out.get(b + ".weight_scale").float()
        N, K = (int(x) for x in out.get(b + ".weight_shape"))
        u = pk.contiguous().view(torch.uint8).view(N, K).to(torch.int16) - 128
        deq = (u.float().view(N, K // 128, 128) * sc.view(N, K // 128, 1)).view(N, K)
        w = src.get(b + ".weight").float()
        rel = ((deq - w).norm() / w.norm()).item()
        q, s = quant_int8(src.get(b + ".weight"), a.search)
        same = torch.equal(q.to(torch.int16), u) and torch.equal(s.float(), sc)
        log(f"{b}: rel err {rel:.2e}, max |q| {u.abs().max().item()}, repro {'exact' if same else 'DIFFERS'}")
    ks = [n for n in out.keys() if n.endswith("k_scale")]
    log(f"{len(names)} INT8 matrices, {len(ks)} k_scale tensors")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="models/Qwen3.8-27B-BF16", help="BF16 checkpoint (sharded or single)")
    ap.add_argument("--out", default="")
    ap.add_argument("--kv-from", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                      "models", "Qwen3.8-27B-INT4"),
                    help="checkpoint whose k_scale / v_scale are copied ('' = none: bf16 KV)")
    ap.add_argument("--search", type=int, default=0, help="per-group clip-ratio candidates (0/1: plain RTN)")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--check", default="", help="verify an output dir against --src")
    ap.add_argument("--n", type=int, default=8, help="matrices sampled by --check")
    a = ap.parse_args()
    if a.kv_from and not os.path.exists(a.kv_from):
        log(f"--kv-from {a.kv_from} missing: no KV scales")
        a.kv_from = ""
    if a.check:
        cmd_check(a)
    elif a.out:
        cmd_make(a)
    else:
        ap.error("--out or --check")


if __name__ == "__main__":
    sys.exit(main())
