#!/usr/bin/env python3
"""Offline INT8 weight conversion for Qwen3.8-27B and Qwen3.8-Flash-Next (QFN).

The 27B route preserves the pinned upstream converter: symmetric INT8, group 128, with BF16 MTP passthrough.
The QFN route converts the explicitly selected attention, shared-expert, fused-expert, PLE, and MTP matrices;
all other source tensors are preserved. Its 128 n-gram tables are excluded and refer to --ngram-from instead.
The current QEngine does not support QFN INT8 expert weights, so QFN output is not executable by that runtime.

RTN uses BF16(absmax / 127), nearest-even rounding, and clamp [-127, 127]. --search N searches N clip ratios in
[0.80, 1], with ratio 1 always included. No calibration data or GPU required.
For QFN, --dev cuda uses every visible GPU with bounded row-block concurrency; --dev cuda:0 selects one GPU.

For 27B, --kv-from defaults to the upstream INT4 checkpoint and supplies KV scales/config/tokenizer when present.
QFN does not import 27B KV scales or assets. Its config and tokenizer assets always come from --src.

Usage:
  python tools/quant_int8.py --src models/Qwen3.8-27B-BF16 --out models/Qwen3.8-27B-INT8
  python tools/quant_int8.py --src models/Qwen3.8-Flash-Next-BF16 --out models/Qwen3.8-Flash-Next-INT8 \
      --ngram-from models/Qwen3.8-Flash-Next-QE-INT4/ngram_int8.safetensors --kv-from ''
  python tools/quant_int8.py --check models/Qwen3.8-Flash-Next-INT8 --src models/Qwen3.8-Flash-Next-BF16
Writes safetensors tensor-by-tensor using .part files and atomic rename.
"""
import argparse
import copy
import json
import os
import filecmp
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open


torch.set_grad_enabled(False)
LM = "model.language_model."
QUANT = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
DT = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
      torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
      torch.uint8: "U8", torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Src:
    """Lazy reader over indexed or single-file safetensors checkpoints."""

    def __init__(self, d, single="model.safetensors"):
        self.d, self.h = d, {}
        ix = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(ix) and not os.path.exists(os.path.join(d, single)):
            with open(ix) as f:
                self.where = json.load(f)["weight_map"]
        else:
            with safe_open(os.path.join(d, single), "pt") as f:
                self.where = {k: single for k in f.keys()}

    def keys(self):
        return list(self.where)

    def _file(self, k):
        f = self.where[k]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f]

    def slice(self, k):
        return self._file(k).get_slice(k)

    def get(self, k):
        return self._file(k).get_tensor(k)

    def block(self, k, index):
        return self.slice(k)[index]


def is_quant(name):
    return name.startswith(LM + "layers.") and name.endswith(".weight") and any(
        name[: -len(".weight")].endswith(q) for q in QUANT)
QFN_BODY = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)\.weight|"
    r"mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)\.weight|"
    r"ple\.(?:key_proj|value_proj)\.weight|"
    r"mlp\.experts\.(?:gate_up_proj|down_proj))$")
QFN_MTP = re.compile(
    r"^mtp\.layers\.0\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight|"
    r"mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)\.weight|"
    r"mlp\.experts\.(?:gate_up_proj|down_proj))$|^mtp\.fc_(?:embedding|hidden)\.weight$")
NGRAM_BF16 = re.compile(
    r"^model\.language_model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
NGRAM_Q8 = "model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_"


def model_config(directory):
    path = os.path.join(directory, "config.json")
    if not os.path.isfile(path):
        raise ValueError(f"missing config.json: {directory}")
    with open(path) as f:
        return json.load(f)


def family(cfg):
    model_type = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type") or ""
    if model_type.startswith("qwen4_exp"):
        return "qwen4_exp"
    if model_type.startswith("qwen3_5"):
        return "qwen3_5"
    raise ValueError(f"unsupported model family: {model_type!r}")


def qfn_dims(cfg):
    text = cfg.get("text_config") or cfg
    hidden = int(text.get("hidden_size") or 0)
    layers = int(text.get("num_hidden_layers") or 0)
    experts = int(text.get("num_experts") or text.get("num_local_experts") or
                  text.get("n_routed_experts") or cfg.get("num_experts") or
                  cfg.get("num_local_experts") or cfg.get("n_routed_experts") or 0)
    intermediate = int(text.get("moe_intermediate_size") or text.get("intermediate_size") or
                       cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or 0)
    if min(hidden, layers, experts, intermediate) <= 0:
        raise ValueError("QFN config lacks hidden_size, num_hidden_layers, expert count, or MoE intermediate size")
    return hidden, layers, experts, intermediate


def qfn_descriptor(name):
    if QFN_BODY.fullmatch(name):
        part = "body"
    elif QFN_MTP.fullmatch(name):
        part = "mtp"
    else:
        return None
    base = name[:-len(".weight")] if name.endswith(".weight") else name
    return {"source": name, "base": base, "file": "model_mtp.safetensors" if part == "mtp" else "model.safetensors",
            "part": part}

def quant_int8(w, search=0, dev="cpu"):
    """[..., N, K] -> int8 codes and BF16 scales grouped over K in 128-wide blocks."""
    if w.ndim < 2:
        raise ValueError(f"INT8 matrix needs rank >= 2, got {tuple(w.shape)}")
    shape = tuple(w.shape)
    N, K = shape[-2:]
    if K % 128:
        raise ValueError(f"K must be divisible by 128, got shape {shape}")
    wf = w.to(dev, torch.float32).reshape(-1, N, K // 128, 128)
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
    q = q.reshape(shape).to(torch.int8).cpu()
    s = s.reshape(*shape[:-1], K // 128).to(torch.bfloat16).cpu()
    return q, s




def pack_int8(q):
    """int8 [N, K] -> int32 [N, K/4]: byte b of word j = k 4j + b, stored u = w + 128 (compressed-tensors)"""
    u = (q.to(torch.int16) + 128).to(torch.uint8)
    return u.contiguous().view(torch.int32)


class Writer:
    """Header-first safetensors writer with byte-checked incremental tensor writes."""

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
        self.path, self.i, self.active = path, 0, None

    def begin(self, name):
        if self.active is not None or self.i >= len(self.plan):
            raise ValueError("writer tensor order/state mismatch")
        pn, dt, shape, nbytes = self.plan[self.i]
        if pn != name:
            raise ValueError(f"writer expected {pn}, got {name}")
        self.active = (name, dt, shape, nbytes, 0)

    def put_chunk(self, t):
        if self.active is None:
            raise ValueError("writer has no active tensor")
        name, dt, shape, nbytes, written = self.active
        if DT.get(t.dtype) != dt:
            raise ValueError(f"{name}: expected {dt}, got {t.dtype}")
        size = t.numel() * esize(dt)
        if size == 0 or written + size > nbytes:
            raise ValueError(f"{name}: invalid incremental byte count")
        flat = t.contiguous().reshape(-1)
        raw = flat.view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else flat.numpy().tobytes()
        if len(raw) != size:
            raise ValueError(f"{name}: encoded {len(raw)} bytes, expected {size}")
        self.f.write(raw)
        self.active = (name, dt, shape, nbytes, written + size)

    def end(self):
        if self.active is None:
            raise ValueError("writer has no active tensor")
        name, dt, shape, nbytes, written = self.active
        expected = esize(dt)
        for dim in shape:
            expected *= dim
        if written != nbytes or written != expected:
            raise ValueError(f"{name}: wrote {written} bytes, expected {nbytes}")
        self.active = None
        self.i += 1

    def put(self, name, t):
        pn, dt, shape, _ = self.plan[self.i]
        if list(t.shape) != list(shape):
            raise ValueError(f"{name}: shape {list(t.shape)}, expected {shape}")
        self.begin(name)
        self.put_chunk(t)
        self.end()

    def close(self):
        if self.active is not None or self.i != len(self.plan):
            raise ValueError(f"incomplete safetensors write: {self.i}/{len(self.plan)}")
        self.f.close()
        os.replace(self.path + ".part", self.path)


def esize(dt):
    return {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2,
            "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1}[dt]
def product(shape):
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def tensor_nbytes(dtype, shape):
    return esize(dtype) * product(shape)


def tensor_info(src, name):
    sl = src.slice(name)
    return tuple(sl.get_shape()), sl.get_dtype()


def qfn_table_names(src):
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    expected = {prefix + str(i) + ".weight" for i in range(128)}
    actual = {name for name in src.keys() if name.startswith(prefix) and name.endswith(".weight")}
    if actual != expected:
        raise ValueError(f"source n-gram index must contain shards 0..127, found {len(actual)}")
    return [prefix + str(i) + ".weight" for i in range(128)]


def validate_ngram_file(path, full_model, expected_rows=None):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"external n-gram table does not exist: {path}")
    prefix = NGRAM_Q8
    with safe_open(path, "pt", device="cpu") as f:
        expected = {prefix + str(i) for i in range(128)}
        if set(f.keys()) != expected:
            raise ValueError(f"external n-gram table must contain exactly 128 shards; found {len(f.keys())}")
        rows = None
        for i in range(128):
            name = prefix + str(i)
            sl = f.get_slice(name)
            shape, dtype = tuple(sl.get_shape()), sl.get_dtype()
            if dtype != "U8" or len(shape) != 2 or shape[0] <= 0 or shape[1] != 162:
                raise ValueError(f"{name}: expected U8 [positive rows,162], got {dtype} {shape}")
            if rows is None:
                rows = shape[0]
            elif shape[0] != rows:
                raise ValueError(f"{name}: inconsistent n-gram rows {shape[0]} != {rows}")
        if full_model and rows != 2500012:
            raise ValueError(f"external n-gram rows {rows}, expected 2500012 for pinned QFN")
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"external n-gram rows {rows}, expected {expected_rows}")
    return {"path": path, "size_bytes": os.path.getsize(path), "rows_per_shard": rows,
            "width_bytes": 162, "shards": 128,
            "layout": "U8 [rows,162]: 160 signed INT8 values then one FP16 scale",
            "keys": [prefix + str(i) for i in range(128)],
            "equivalence_verified": False}


def configured_dims(text, names):
    values = [text.get(name) for name in names]
    if not any(value is not None for value in values):
        return None
    if any(value is None or int(value) <= 0 for value in values):
        raise ValueError(f"QFN config requires positive {', '.join(names)}")
    return tuple(int(value) for value in values)


def validate_qfn_matrix(desc, shape, dtype, dims, cfg):
    hidden, layers, experts, intermediate = dims
    name = desc["source"]
    text = cfg.get("text_config") or cfg
    if dtype != "BF16":
        raise ValueError(f"{name}: QFN selected matrix must be BF16, got {dtype}")
    if len(shape) not in (2, 3) or shape[-1] % 128:
        raise ValueError(f"{name}: expected rank 2/3 and K divisible by 128, got {shape}")
    if desc["part"] == "body":
        layer = int(QFN_BODY.fullmatch(name).group(1))
        if layer >= layers:
            raise ValueError(f"{name}: layer {layer} outside configured layer count {layers}")
        layer_types = text.get("layer_types")
        if layer_types is not None:
            if len(layer_types) != layers:
                raise ValueError(f"QFN layer_types count {len(layer_types)} differs from num_hidden_layers {layers}")
            kind = "linear_attention" if ".linear_attn." in name else (
                "full_attention" if ".self_attn." in name else None)
            if kind and layer_types[layer] != kind:
                raise ValueError(f"{name}: config layer {layer} is {layer_types[layer]}, expected {kind}")
    if len(shape) == 3:
        if name.endswith("mlp.experts.gate_up_proj"):
            expected = (experts, 2 * intermediate, hidden)
        elif name.endswith("mlp.experts.down_proj"):
            expected = (experts, hidden, intermediate)
        else:
            raise ValueError(f"{name}: only fused expert matrices may have rank 3")
        if tuple(shape) != expected:
            raise ValueError(f"{name}: shape {shape}, expected {expected} from text_config")
        return

    expected = None
    if name.endswith("self_attn.q_proj.weight"):
        heads = configured_dims(text, ("num_attention_heads", "head_dim"))
        if heads:
            expected = (2 * heads[0] * heads[1], hidden)
    elif name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        heads = configured_dims(text, ("num_key_value_heads", "head_dim"))
        if heads:
            expected = (heads[0] * heads[1], hidden)
    elif name.endswith("self_attn.o_proj.weight"):
        heads = configured_dims(text, ("num_attention_heads", "head_dim"))
        if heads:
            expected = (hidden, heads[0] * heads[1])
    elif name.endswith("linear_attn.in_proj_qkv.weight"):
        heads = configured_dims(text, ("linear_num_key_heads", "linear_key_head_dim",
                                        "linear_num_value_heads", "linear_value_head_dim"))
        if heads:
            expected = (2 * heads[0] * heads[1] + heads[2] * heads[3], hidden)
    elif name.endswith("linear_attn.in_proj_z.weight"):
        heads = configured_dims(text, ("linear_num_value_heads", "linear_value_head_dim"))
        if heads:
            expected = (heads[0] * heads[1], hidden)
    elif name.endswith("linear_attn.out_proj.weight"):
        heads = configured_dims(text, ("linear_num_value_heads", "linear_value_head_dim"))
        if heads:
            expected = (hidden, heads[0] * heads[1])
    elif name.endswith("ple.key_proj.weight"):
        hc_count = configured_dims(text, ("hc_count",))
        if hc_count:
            expected = (hc_count[0] * hidden, hidden)
    elif name.endswith("ple.value_proj.weight"):
        expected = (hidden, hidden)

    if expected is not None:
        if tuple(shape) != expected:
            raise ValueError(f"{name}: shape {shape}, expected {expected} from text_config")
        return
    n, k = shape
    if (".mlp.shared_expert.gate_proj.weight" in name or
            ".mlp.shared_expert.up_proj.weight" in name):
        shared = int(text.get("shared_expert_intermediate_size") or intermediate)
        expected = (shared, hidden)
    elif ".mlp.shared_expert.down_proj.weight" in name:
        shared = int(text.get("shared_expert_intermediate_size") or intermediate)
        expected = (hidden, shared)
    elif name.endswith("mtp.fc_embedding.weight"):
        if n != hidden or k not in (hidden, 2 * hidden):
            raise ValueError(f"{name}: shape {shape} incompatible with hidden_size {hidden}")
        return
    elif name.endswith("mtp.fc_hidden.weight"):
        expected = (hidden, hidden)
    else:
        if k != hidden:
            raise ValueError(f"{name}: input dimension {k}, expected hidden_size {hidden}")
        return
    if tuple(shape) != expected:
        raise ValueError(f"{name}: shape {shape}, expected {expected} from text_config")


def slice_blocks(shape, dtype, max_bytes=8 << 20):
    """Yield contiguous rectangular slices, each no larger than max_bytes."""
    if not shape:
        yield ()
        return
    cap = max(1, max_bytes // esize(dtype))
    chunks, remaining = [1] * len(shape), cap
    for axis in range(len(shape) - 1, -1, -1):
        chunks[axis] = min(shape[axis], max(1, remaining))
        remaining = max(1, remaining // chunks[axis])
    starts = [0] * len(shape)
    def walk(axis):
        if axis == len(shape):
            yield tuple(slice(start, min(start + chunk, dim))
                        for start, chunk, dim in zip(starts, chunks, shape))
            return
        for start in range(0, shape[axis], chunks[axis]):
            starts[axis] = start
            yield from walk(axis + 1)
    yield from walk(0)


def add_tensor_plan(plan, name, dtype, shape):
    plan.append((name, dtype, list(shape), tensor_nbytes(dtype, shape)))


def qfn_plans(source_keys, descriptors, headers):
    plans, source_for_output, packed_for_name = {}, {}, {}
    desc_by_source = {d["source"]: d for d in descriptors}
    for part, filename in (("body", "model.safetensors"), ("mtp", "model_mtp.safetensors")):
        plan = []
        keys = [k for k in source_keys if (k.startswith("mtp.") == (part == "mtp"))]
        for key in sorted(keys):
            shape, dtype = headers[key]
            desc = desc_by_source.get(key)
            if desc:
                n, k = shape[-2:]
                packed = desc["base"] + ".weight_packed"
                scale = desc["base"] + ".weight_scale"
                shaped = desc["base"] + ".weight_shape"
                add_tensor_plan(plan, packed, "I32", (*shape[:-1], k // 4))
                add_tensor_plan(plan, scale, "BF16", (*shape[:-1], k // 128))
                add_tensor_plan(plan, shaped, "I64", (len(shape),))
                source_for_output[scale] = desc
                source_for_output[shaped] = desc
                packed_for_name[packed] = desc
            else:
                add_tensor_plan(plan, key, dtype, shape)
                source_for_output[key] = key
        plan.sort(key=lambda item: item[0])
        for i, entry in enumerate(plan):
            if entry[0] in packed_for_name:
                if [p[0] for p in plan[i:i + 3]] != [
                    entry[0], entry[0][:-len("_packed")] + "_scale", entry[0][:-len("_packed")] + "_shape"]:
                    raise ValueError(f"non-adjacent QFN quantization outputs for {entry[0]}")
        plans[filename] = plan
    return plans, source_for_output, packed_for_name


def validate_qfn_source(src, cfg, ngram_path, out):
    if os.path.realpath(src.d) == os.path.realpath(out):
        raise ValueError("--src and --out resolve to the same directory")
    if family(cfg) != "qwen4_exp":
        raise ValueError("QFN conversion requires model_type qwen4_exp")
    if cfg.get("quantization_config"):
        raise ValueError("QFN source config is already quantized")
    dims, names, table_names, descriptors, headers, full_model = inspect_qfn_source(src, cfg)
    ngram = validate_ngram_file(ngram_path, full_model, 2500012 if full_model else None)
    out = os.path.abspath(out)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise ValueError(f"QFN output directory must be empty: {out}")
    os.makedirs(out, exist_ok=True)
    return dims, names, table_names, descriptors, headers, ngram

def write_quant_tensor(writer, src, desc, shape, search, dev, block_rows=256):
    source = desc["source"]
    packed_name = desc["base"] + ".weight_packed"
    scale_name = desc["base"] + ".weight_scale"
    shape_name = desc["base"] + ".weight_shape"
    n, k = shape[-2:]
    scale_shape = (*shape[:-1], k // 128)
    scales = torch.empty(scale_shape, dtype=torch.bfloat16)
    devices = ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
               if dev == "cuda" and torch.cuda.is_available() else [dev])
    rank3 = len(shape) == 3
    writer.begin(packed_name)

    def store(result, expert, r0, r1):
        packed, scale = result
        writer.put_chunk(packed)
        if rank3:
            scales[expert, r0:r1] = scale
        else:
            scales[r0:r1] = scale

    def quantize(block, device):
        q, scale = quant_int8(block, search, device)
        return pack_int8(q), scale

    if len(devices) == 1:
        for expert in range(shape[0] if rank3 else 1):
            for r0 in range(0, n, block_rows):
                r1 = min(r0 + block_rows, n)
                index = ((expert, slice(r0, r1), slice(None)) if rank3 else
                         (slice(r0, r1), slice(None)))
                store(quantize(src.block(source, index), devices[0]), expert, r0, r1)
    else:
        pending = []
        block_index = 0
        with ThreadPoolExecutor(max_workers=len(devices)) as workers:
            for expert in range(shape[0] if rank3 else 1):
                for r0 in range(0, n, block_rows):
                    r1 = min(r0 + block_rows, n)
                    index = ((expert, slice(r0, r1), slice(None)) if rank3 else
                             (slice(r0, r1), slice(None)))
                    block = src.block(source, index)
                    device = devices[block_index % len(devices)]
                    pending.append((workers.submit(quantize, block, device), expert, r0, r1))
                    block_index += 1
                    if len(pending) == len(devices):
                        future, old_expert, old_r0, old_r1 = pending.pop(0)
                        store(future.result(), old_expert, old_r0, old_r1)
            for future, expert, r0, r1 in pending:
                store(future.result(), expert, r0, r1)

    writer.end()
    writer.put(scale_name, scales)
    writer.put(shape_name, torch.tensor(shape, dtype=torch.int64))


def write_passthrough(writer, src, name, dtype, shape):
    writer.begin(name)
    if not shape:
        writer.put_chunk(src.get(name))
    else:
        for index in slice_blocks(shape, dtype):
            writer.put_chunk(src.block(name, index))
    writer.end()


def cmd_make_qfn(a, cfg):
    if a.kv_from not in (None, ""):
        raise ValueError("QFN route does not import KV scales from the 27B checkpoint; omit --kv-from")
    if not a.ngram_from:
        raise ValueError("QFN requires --ngram-from with the existing INT8 table")
    src = Src(a.src)
    dims, source_keys, excluded, descriptors, headers, ngram = validate_qfn_source(
        src, cfg, a.ngram_from, a.out)
    plans, source_for_output, packed_for_name = qfn_plans(source_keys, descriptors, headers)
    t0 = time.time()
    for filename in ("model.safetensors", "model_mtp.safetensors"):
        plan = plans[filename]
        writer = Writer(os.path.join(a.out, filename), plan,
                        {"format": "pt", "qengine": f"quant_int8.py symmetric g128 search={a.search}"})
        while writer.i < len(plan):
            name, dtype, shape, _ = plan[writer.i]
            desc = packed_for_name.get(name)
            if desc:
                write_quant_tensor(writer, src, desc, headers[desc["source"]][0],
                                   a.search, a.dev, 256)
            else:
                source_name = source_for_output[name]
                source_shape, source_dtype = headers[source_name]
                write_passthrough(writer, src, source_name, source_dtype, source_shape)
        writer.close()
        log(f"{filename}: {len(plan)} tensors written")

    cfg_out = copy.deepcopy(cfg)
    q = copy.deepcopy(cfg_out.get("quantization_config") or {})
    targets = sorted({d["base"] for d in descriptors})
    q.update({"format": "pack-quantized", "quant_method": "compressed-tensors",
              "quantization_status": "compressed", "target_modules": targets})
    q["config_groups"] = {"group_0": {
        "targets": targets, "input_activations": None, "output_activations": None,
        "format": "pack-quantized",
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "group",
                    "group_size": 128, "dynamic": False, "actorder": None,
                    "observer": "minmax" if a.search <= 1 else "mse",
                    "observer_kwargs": {}, "block_structure": None, "scale_dtype": None,
                    "zp_dtype": None}}}
    q["kv_cache_scheme"] = None
    cfg_out["quantization_config"] = q
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    assets = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "LICENSE",
              "vocab.json", "merges.txt")
    for name in assets:
        path = os.path.join(a.src, name)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(a.out, name))

    converted = []
    converted_names = {d["source"] for d in descriptors}
    for desc in descriptors:
        converted.append({"source_key": desc["source"], "base": desc["base"], "file": desc["file"],
                          "shape": list(headers[desc["source"]][0]), "dtype": headers[desc["source"]][1],
                          "outputs": {"packed": desc["base"] + ".weight_packed",
                                      "scale": desc["base"] + ".weight_scale",
                                      "shape": desc["base"] + ".weight_shape"}})
    preserved = {"body": sorted(n for n in source_keys if not n.startswith("mtp.") and n not in converted_names),
                 "mtp": sorted(n for n in source_keys if n.startswith("mtp.") and n not in converted_names)}
    manifest = {
        "format_version": 1, "family": "qwen4_exp",
        "source": {"repo": "orcarouter/Qwen3.8-Flash-Next-Uncensored",
                   "revision": "da29941aee972c7eeab497e05a292c569da1f1d6"},
        "recipe": {"format": "symmetric INT8", "group_size": 128, "rounding": "nearest-even",
                   "scale_dtype": "BF16", "zero_point": 0, "search": a.search,
                   "algorithm": "RTN; per-group BF16 absmax/127; zero groups use scale 1"},
        "outputs": {"body": "model.safetensors", "mtp": "model_mtp.safetensors",
                    "config": "config.json", "manifest": "qfn-int8-manifest.json"},
        "tensors": {"converted": converted, "preserved": preserved, "excluded": excluded},
        "ngram_external": ngram,
        "runtime_supported": False,
        "runtime_support_reason": "Current QFN engine does not support INT8 expert weights; checkpoint is not executable by current QEngine."
    }
    with open(os.path.join(a.out, "qfn-int8-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(a.out, "QENGINE_INT8.md"), "w") as f:
        f.write("QFN INT8 symmetric group-128 weights. Current QEngine does not support INT8 QFN experts; "
                "this checkpoint is not executable by the current engine.\n")
    log(f"QFN: {len(converted)} matrices, {len(excluded)} n-gram source tensors excluded; "
        f"runtime unsupported; {time.time() - t0:.0f}s")


def cmd_make(a):
    cfg = model_config(a.src)
    model_family = family(cfg)
    if model_family == "qwen4_exp":
        return cmd_make_qfn(a, cfg)
    if a.ngram_from:
        raise ValueError("--ngram-from is only valid for QFN")
    return cmd_make_27b(a, cfg)


def cmd_make_27b(a, src_cfg):
    if os.path.realpath(a.src) == os.path.realpath(a.out):
        raise ValueError("--src and --out resolve to the same directory")
    if a.kv_from is None:
        a.kv_from = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-27B-INT4")
    if a.kv_from and not os.path.exists(a.kv_from):
        log(f"--kv-from {a.kv_from} missing: no KV scales")
        a.kv_from = ""
    if a.kv_from and family(model_config(a.kv_from)) != family(src_cfg):
        raise ValueError("--kv-from must use same model family as --src")
    src = Src(a.src)
    kv = Src(a.kv_from) if a.kv_from else None
    os.makedirs(a.out, exist_ok=True)
    names = src.keys()
    main = [n for n in names if not n.startswith("mtp.")]
    mtp = [n for n in names if n.startswith("mtp.")]
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
    if mtp:
        mp = [(n, shapes[n][1], shapes[n][0], esize(shapes[n][1]) * int(torch.tensor(shapes[n][0]).prod().item()))
              for n in sorted(mtp)]
        wm = Writer(os.path.join(a.out, "model_mtp.safetensors"), mp, {"format": "pt"})
        for n, *_ in mp:
            wm.put(n, src.get(n))
        wm.close()
        log(f"model_mtp.safetensors: {len(mp)} tensors (stock BF16 head)")
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
        for d in (a.kv_from or "", a.src):
            p = os.path.join(d, f)
            if d and os.path.exists(p):
                shutil.copy(p, os.path.join(a.out, f))
                break
    with open(os.path.join(a.out, "QENGINE_INT8.md"), "w") as f:
        f.write(f"INT8 (W8A16, symmetric, group 128) of {os.path.basename(os.path.abspath(a.src))} by "
                f"tools/quant_int8.py (search={a.search}); KV scales from {a.kv_from or 'none'}; MTP head: stock BF16.\n")
    log("done", a.out)


def inspect_qfn_source(src, cfg):
    dims = qfn_dims(cfg)
    table_names = qfn_table_names(src)
    excluded_set = set(table_names)
    names = [n for n in src.keys() if n not in excluded_set]
    descriptors = [d for n in names if (d := qfn_descriptor(n)) is not None]
    body_count = sum(d["part"] == "body" for d in descriptors)
    mtp_count = sum(d["part"] == "mtp" for d in descriptors)
    full_model = dims == (2560, 48, 512, 640)
    if not descriptors:
        raise ValueError("QFN source contains no selected matrices")
    if mtp_count != 11:
        raise ValueError(f"QFN MTP selection must contain 11 matrices, found {mtp_count}")
    if full_model and (body_count, len(table_names)) != (398, 128):
        raise ValueError(f"pinned QFN inventory mismatch: body={body_count}, MTP={mtp_count}, n-gram={len(table_names)}")
    headers = {name: tensor_info(src, name) for name in names}
    for desc in descriptors:
        validate_qfn_matrix(desc, *headers[desc["source"]], dims, cfg)
    return dims, names, table_names, descriptors, headers, full_model


def tensor_blocks(src, name, shape, dtype, max_bytes=8 << 20):
    if not shape:
        yield (), src.get(name)
        return
    for index in slice_blocks(shape, dtype, max_bytes):
        yield index, src.block(name, index)


def assert_tensor_equal(src_a, src_b, name, shape, dtype):
    for index, tensor_a in tensor_blocks(src_a, name, shape, dtype):
        tensor_b = src_b.get(name) if not index else src_b.block(name, index)
        if not torch.equal(tensor_a, tensor_b):
            raise ValueError(f"passthrough differs: {name} at {index}")


def check_qfn(a, manifest):
    src_cfg = model_config(a.src)
    if (family(src_cfg) != "qwen4_exp" or src_cfg.get("quantization_config") or
            manifest.get("family") != "qwen4_exp"):
        raise ValueError("QFN manifest/source family mismatch or source is already quantized")
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported QFN manifest format")
    source_info = manifest.get("source", {})
    if (source_info.get("repo") != "orcarouter/Qwen3.8-Flash-Next-Uncensored" or
            source_info.get("revision") != "da29941aee972c7eeab497e05a292c569da1f1d6"):
        raise ValueError("unexpected pinned source in manifest")
    recipe = manifest.get("recipe", {})
    if int(recipe.get("search", -1)) != a.search or recipe.get("group_size") != 128:
        raise ValueError("--search/group_size does not match QFN manifest")
    if manifest.get("runtime_supported") is not False:
        raise ValueError("QFN manifest must state runtime_supported=false")
    ngram_info = manifest.get("ngram_external", {})
    source_dims = qfn_dims(src_cfg)
    full_source = source_dims == (2560, 48, 512, 640)
    ngram = validate_ngram_file(ngram_info.get("path", ""), full_source,
                                2500012 if full_source else ngram_info.get("rows_per_shard"))
    if (not os.path.isabs(ngram_info.get("path", "")) or
            ngram["size_bytes"] != ngram_info.get("size_bytes") or
            ngram["keys"] != ngram_info.get("keys") or ngram["layout"] != ngram_info.get("layout") or
            ngram_info.get("equivalence_verified") is not False):
        raise ValueError("external n-gram dependency differs from manifest")
    if os.path.exists(os.path.join(a.check, "ngram_int8.safetensors")):
        raise ValueError("n-gram table must remain an external dependency, not be copied into output")
    src = Src(a.src)
    dims, source_keys, table_names, descriptors, headers, full_model = inspect_qfn_source(src, src_cfg)
    manifest_tensors = manifest.get("tensors", {})
    converted_records = {d.get("source_key"): d for d in manifest_tensors.get("converted", [])}
    expected_converted = {d["source"] for d in descriptors}
    targets = sorted({d["base"] for d in descriptors})
    if (expected_converted != set(converted_records) or
            set(table_names) != set(manifest_tensors.get("excluded", []))):
        raise ValueError("manifest tensor inventory does not match source")
    for desc in descriptors:
        record = converted_records[desc["source"]]
        shape, dtype = headers[desc["source"]]
        expected_outputs = {"packed": desc["base"] + ".weight_packed",
                            "scale": desc["base"] + ".weight_scale",
                            "shape": desc["base"] + ".weight_shape"}
        if (record.get("base") != desc["base"] or record.get("file") != desc["file"] or
                record.get("shape") != list(shape) or record.get("dtype") != dtype or
                record.get("outputs") != expected_outputs):
            raise ValueError(f"manifest descriptor differs for {desc['source']}")
    expected_preserved = {
        "body": sorted(n for n in source_keys if not n.startswith("mtp.") and n not in expected_converted),
        "mtp": sorted(n for n in source_keys if n.startswith("mtp.") and n not in expected_converted)}
    if manifest_tensors.get("preserved") != expected_preserved:
        raise ValueError("manifest passthrough inventory does not match source")
    cfg_out = model_config(a.check)
    if (family(cfg_out) != "qwen4_exp" or
            cfg_out.get("quantization_config", {}).get("target_modules") != targets):
        raise ValueError("output config family or descriptive quantization targets differ")
    if manifest.get("outputs") != {"body": "model.safetensors", "mtp": "model_mtp.safetensors",
                                   "config": "config.json", "manifest": "qfn-int8-manifest.json"}:
        raise ValueError("manifest output file inventory differs")
    plans, _, _ = qfn_plans(source_keys, descriptors, headers)
    outputs = {"model.safetensors": Src(a.check),
               "model_mtp.safetensors": Src(a.check, "model_mtp.safetensors")}
    for filename, plan in plans.items():
        output = outputs[filename]
        expected = {entry[0]: entry[1:3] for entry in plan}
        if set(output.keys()) != set(expected):
            raise ValueError(f"{filename}: output key inventory differs from plan")
        for name, (dtype, shape) in expected.items():
            actual_shape, actual_dtype = tensor_info(output, name)
            if actual_dtype != dtype or list(actual_shape) != list(shape):
                raise ValueError(f"{filename}:{name}: header {actual_dtype} {actual_shape}, expected {dtype} {shape}")

    report = {}
    for desc in descriptors:
        source_name, base = desc["source"], desc["base"]
        shape, _ = headers[source_name]
        output = outputs[desc["file"]]
        packed_name, scale_name, shape_name = (base + ".weight_packed", base + ".weight_scale",
                                                base + ".weight_shape")
        stored_shape = tuple(int(x) for x in output.get(shape_name).tolist())
        if stored_shape != shape:
            raise ValueError(f"{base}: stored weight_shape {stored_shape}, expected {shape}")
        n, k = shape[-2:]
        totals = report.setdefault(("experts" if len(shape) == 3 else desc["part"]), [0.0, 0.0, 0])
        for expert in range(shape[0] if len(shape) == 3 else 1):
            for r0 in range(0, n, 256):
                r1 = min(r0 + 256, n)
                index = ((expert, slice(r0, r1), slice(None)) if len(shape) == 3 else
                         (slice(r0, r1), slice(None)))
                original = src.block(source_name, index)
                packed = output.block(packed_name, index)
                scales = output.block(scale_name, index)
                codes = packed.contiguous().view(torch.uint8).reshape(r1 - r0, k).to(torch.int16) - 128
                expected_q, expected_s = quant_int8(original, a.search)
                if not torch.equal(codes, expected_q.to(torch.int16)) or not torch.equal(scales, expected_s):
                    raise ValueError(f"{base}: RTN/search reproduction differs at expert={expert}, rows={r0}:{r1}")
                if int(codes.abs().max()) > 127:
                    raise ValueError(f"{base}: code outside symmetric INT8 range")
                deq = (codes.float().reshape(r1 - r0, k // 128, 128) *
                       scales.float().unsqueeze(-1)).reshape(r1 - r0, k)
                original_f = original.float()
                totals[0] += float((deq - original_f).square().sum())
                totals[1] += float(original_f.square().sum())
                totals[2] += 1
        log(f"{base}: RTN/search exact, shape={shape}")

    converted = {d["source"] for d in descriptors}
    for name in source_keys:
        if name in converted:
            continue
        shape, dtype = headers[name]
        output = outputs["model_mtp.safetensors" if name.startswith("mtp.") else "model.safetensors"]
        assert_tensor_equal(src, output, name, shape, dtype)
    for key, (error_sq, source_sq, blocks) in sorted(report.items()):
        rel = (error_sq / source_sq) ** 0.5 if source_sq else 0.0
        log(f"{key}: {blocks} blocks, relative L2 error={rel:.6g}")
    if full_model and (len(descriptors), len(table_names)) != (409, 128):
        raise ValueError("full QFN verification inventory must be 398 body + 11 MTP matrices and 128 excluded tables")
    if any(name.endswith(".part") for name in os.listdir(a.check)):
        raise ValueError("incomplete .part file remains in output")
    assets = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "LICENSE",
              "vocab.json", "merges.txt")
    for name in assets:
        source = os.path.join(a.src, name)
        output = os.path.join(a.check, name)
        if os.path.isfile(source) and (not os.path.isfile(output) or
                                       not filecmp.cmp(source, output, shallow=False)):
            raise ValueError(f"source asset missing or content differs: {name}")
    log(f"QFN check passed: {sum(d['part'] == 'body' for d in descriptors)} body + "
        f"{sum(d['part'] == 'mtp' for d in descriptors)} MTP matrices; {len(table_names)} external n-gram shards")


def cmd_check(a):
    manifest_path = os.path.join(a.check, "qfn-int8-manifest.json")
    if os.path.isfile(manifest_path) and a.kv_from not in (None, ""):
        raise ValueError("QFN check does not accept --kv-from")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        return check_qfn(a, manifest)
    return cmd_check_27b(a)


def cmd_check_27b(a):
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
    ap.add_argument("--kv-from", default=None,
                    help="27B checkpoint supplying KV scales/config/tokenizer; omitted uses the upstream default")
    ap.add_argument("--ngram-from", default=None, help="QFN external INT8 n-gram table (required for QFN)")
    ap.add_argument("--search", type=int, default=0, help="per-group clip-ratio candidates (0/1: plain RTN)")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--check", default="", help="verify an output dir against --src")
    ap.add_argument("--n", type=int, default=8, help="matrices sampled by --check")
    a = ap.parse_args()
    if a.check:
        cmd_check(a)
    elif a.out:
        cmd_make(a)
    else:
        ap.error("--out or --check")


if __name__ == "__main__":
    sys.exit(main())
