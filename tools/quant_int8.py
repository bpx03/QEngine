#!/usr/bin/env python3
"""Offline INT8 weight conversion for Qwen3.8-27B and Qwen3.8-Flash-Next (QFN).

The 27B route preserves the pinned upstream converter: symmetric INT8, group 128, with BF16 MTP passthrough.
The QFN route converts the explicitly selected attention, shared-expert, fused-expert, PLE, and MTP matrices;
all other source tensors are preserved. Its 128 n-gram tables are excluded from the body; by default the
existing INT8 table is materialized INSIDE the output as an independent full copy (--link-ngram hard links
it instead) so the checkpoint is self-contained. --external-ngram keeps it outside: the manifest then
records only table identity plus a sha256 digest, and --check takes the table path via --external-ngram
PATH or QENGINE_INT8_NGRAM_PATH.
The QFN body is written as shards (model-00001-of-000NN.safetensors, each <= --max-shard-bytes, default
20 GiB, grouped per transformer layer; PLE/n-gram-adjacent tensors go to the final shard); every manifest
tensor names its shard. The manifest records a sha256 digest of the source index (or of the sorted source
weight-file inventory) plus optional --source-repo/--source-revision describing what was ACTUALLY converted.
The Flash-Next engine (src/flash/) loads QFN output; the 27B engine (libqengine) reads INT4 weights only.

RTN uses BF16(absmax / 127), nearest-even rounding, and clamp [-127, 127]. --search N searches N clip ratios in
[0.80, 1], with ratio 1 always included. NaN or Inf anywhere in a quantized matrix is a hard error naming the
tensor. No calibration data or GPU required.
For QFN, --dev cuda uses every visible GPU with bounded row-block concurrency; --dev cuda:0 selects one GPU.

For 27B, --kv-from defaults to the upstream INT4 checkpoint and supplies KV scales/config/tokenizer when present.
QFN does not import 27B KV scales or assets. Its config and tokenizer assets always come from --src.

Usage:
  python tools/quant_int8.py --src models/Qwen3.8-27B-BF16 --out models/Qwen3.8-27B-INT8
  python tools/quant_int8.py --src models/Qwen3.8-Flash-Next-BF16 --out models/Qwen3.8-Flash-Next-INT8 \
      --ngram-from models/Qwen3.8-Flash-Next-QE-INT4/ngram_int8.safetensors --kv-from ''
  python tools/quant_int8.py --src models/Qwen3.8-Flash-Next-BF16 --out models/Qwen3.8-Flash-Next-INT8 \
      --ngram-from .../ngram_int8.safetensors --kv-from '' --external-ngram \
      --source-repo acme/fork --source-revision 0123456789abcdef
  python tools/quant_int8.py --check models/Qwen3.8-Flash-Next-INT8                 # structural check
  python tools/quant_int8.py --check models/Qwen3.8-Flash-Next-INT8 \
      --src models/Qwen3.8-Flash-Next-BF16                                          # + RTN/source reproduction
  python tools/quant_int8.py --check ... --external-ngram /srv/tables/ngram_int8.safetensors
Writes safetensors tensor-by-tensor using .part files and atomic rename.
"""
import argparse
import copy
import hashlib
import json
import os
import filecmp
import re
import shutil
import stat
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open


torch.set_grad_enabled(False)
LM = "model.language_model."
# Constants shared with src/flash/checkpoint.h (keep both sides in sync; the C++ constants are the spec).
MAX_SHARD_BYTES = 20 << 30  # mirror of kMaxShardBytes in src/flash/checkpoint.h
MTP_FC_EMBEDDING_K_MULTIPLIERS = (1, 2)  # mirror of kMtpFcEmbeddingKMultipliers in src/flash/checkpoint.h
MANIFEST_SCHEMA = "qfn-int8-v2"  # mirror of kManifestSchema in src/flash/checkpoint.h
BODY_MATRICES = 398  # mirror of kBodyMatrices in src/flash/checkpoint.h
MTP_MATRICES = 11  # mirror of kMtpMatrices in src/flash/checkpoint.h
LAYER_COUNT = 48  # mirror of kLayerCount in src/flash/checkpoint.h
NGRAM_ROWS_PER_SHARD = 2500012  # mirror of kNgramRowsPerShard in src/flash/checkpoint.h
NGRAM_OUT_NAME = "ngram_int8.safetensors"
HEADER_MARGIN = 8 << 20  # safetensors header/padding headroom kept below every shard byte cap
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

def quant_int8(w, search=0, dev="cpu", name="tensor"):
    """[..., N, K] -> int8 codes and BF16 scales grouped over K in 128-wide blocks."""
    if w.ndim < 2:
        raise ValueError(f"INT8 matrix needs rank >= 2, got {tuple(w.shape)}")
    shape = tuple(w.shape)
    N, K = shape[-2:]
    if K % 128:
        raise ValueError(f"K must be divisible by 128, got shape {shape}")
    wf = w.to(dev, torch.float32).reshape(-1, N, K // 128, 128)
    finite = torch.isfinite(wf)
    if not bool(finite.all()):
        # never let NaN/Inf poison the scale: absmax(NaN) would silently fall back to scale 1
        raise ValueError(f"{name}: contains NaN or Inf values; refusing to quantize")
    amax = wf.abs().amax(-1, keepdim=True)

    def q_err(ratio):
        s = (amax * ratio / 127.0).to(torch.bfloat16).float()
        # A positive absmax whose BF16 scale underflows to zero is NOT a zero group: floor such
        # scales at the smallest positive BF16, 2**-133 (bit pattern 0x0001, subnormal; the
        # smallest POSITIVE NORMAL 2**-126 = 0x0080 would still round tiny groups to code 0), so
        # they keep a nonzero code instead of collapsing silently to all-zero. The recipe's
        # scale-1 fallback stays reserved for true-zero groups (manifest recipe string unchanged).
        # Any future C++ RTN mirror of this converter must reuse the 0x0001 floor bit-for-bit to
        # stay bit-compatible with checkpoints produced here.
        s = torch.where(s > 0, s,
                        torch.where(amax > 0, torch.full_like(s, 2.0 ** -133), torch.ones_like(s)))
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


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def source_digest(src):
    """Deterministic traceability record for the source checkpoint (NOT an integrity proof — that
    comes from --check --src): the raw bytes of its
    model.safetensors.index.json when the reader used one, else a digest over the sorted
    (filename, size) inventory of the source weight files. The weight_files form is BEST-EFFORT:
    it records only names and sizes, so size-preserving edits inside a weight file are invisible
    to it (the index_json form hashes raw bytes and has no such blind spot)."""
    ix = os.path.join(src.d, "model.safetensors.index.json")
    if os.path.exists(ix) and not os.path.exists(os.path.join(src.d, "model.safetensors")):
        return "index_json", sha256_file(ix)
    h = hashlib.sha256()
    for name in sorted(set(src.where.values())):
        h.update(f"{name}\0{os.path.getsize(os.path.join(src.d, name))}\n".encode())
    return "weight_files", h.hexdigest()


def qfn_table_names(src):
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    expected = {prefix + str(i) + ".weight" for i in range(128)}
    actual = {name for name in src.keys() if name.startswith(prefix) and name.endswith(".weight")}
    if actual != expected:
        raise ValueError(f"source n-gram index must contain shards 0..127, found {len(actual)}")
    return [prefix + str(i) + ".weight" for i in range(128)]


def validate_ngram_file(path, full_model, expected_rows=None):
    """Header-only identity validation of an INT8 n-gram table (128 U8 [rows,162] shards)."""
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
        if full_model and rows != NGRAM_ROWS_PER_SHARD:
            raise ValueError(f"external n-gram rows {rows}, expected {NGRAM_ROWS_PER_SHARD} for pinned QFN")
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"external n-gram rows {rows}, expected {expected_rows}")
    return {"size_bytes": os.path.getsize(path), "rows_per_shard": rows,
            "width_bytes": 162, "shards": 128,
            "layout": "U8 [rows,162]: 160 signed INT8 values then one FP16 scale",
            "keys": [prefix + str(i) for i in range(128)]}


def materialize_ngram(table_path, out, link=False):
    """Self-contained default: place the table next to the weights as an independent, byte-identical
    copy. A hard link would share the file with the source table, so a later write through the
    source name would mutate the DEPLOYED checkpoint; copying is therefore the default and
    --link-ngram opts back into the zero-copy hard link (falling back to a full copy across
    filesystems, where the shortcut is impossible)."""
    dst = os.path.join(out, NGRAM_OUT_NAME)
    if os.path.lexists(dst):
        os.remove(dst)
    if link:
        try:
            os.link(table_path, dst)
            return dst
        except OSError:
            pass  # different filesystems: the hard link is impossible, copy below
    total = os.path.getsize(table_path)
    # copy across filesystems lands on <name>.part first: os.replace publishes the table atomically
    # once all bytes are written, so an interrupted 52 GiB copy can never leave a half-written table
    # under the final name; a .part that remains in an output dir is itself a --check rejection
    part = dst + ".part"
    log(f"n-gram: copying {total} bytes to {dst} (--link-ngram hard links instead)")
    t0 = time.time()
    try:
        with open(table_path, "rb") as fin, open(part, "wb") as fout:
            done = 0
            while chunk := fin.read(1 << 30):
                fout.write(chunk)
                done += len(chunk)
                log(f"n-gram: {done / (1 << 30):.1f}/{total / (1 << 30):.1f} GiB copied")
        os.replace(part, dst)
    except BaseException:
        try:
            os.unlink(part)
        except OSError:
            pass
        raise
    log(f"n-gram: copied {total} bytes in {time.time() - t0:.0f}s")
    return dst


def sandbox_relative(name):
    """Manifest path fields are relative to the checkpoint directory: never absolute, never a drive/UNC
    root, never '..'."""
    if not name or os.path.isabs(name) or name.startswith("~") or name.startswith("\\\\"):
        return False
    if len(name) >= 2 and name[0].isalpha() and name[1] == ":":  # Windows drive root (POSIX sees it as relative)
        return False
    parts = re.split(r"[/\\]", name)
    return ".." not in parts and "\0" not in name


def strict_json_object(pairs):
    """object_pairs_hook rejecting duplicate keys. The C++ reader parses the manifest with
    Json::parse_strict, so --check must refuse the same malformed input, never read it leniently."""
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key {key!r} in manifest (the strict C++ loader rejects it too)")
        seen.add(key)
    return dict(pairs)


def sandbox_regular_file(path, root):
    """Symlink gate for sandboxed checkpoint paths, mirroring require_no_symlink in
    src/flash/checkpoint.h: lstat EVERY component from the checkpoint root downward, so neither a
    manifest-named file nor an intermediate directory may be a symlink; any symlink component is
    rejected, never followed (a symlinked directory would otherwise smuggle outside files into the
    sandbox while the final component looks like a regular file). Operator-supplied absolute inputs
    (--external-ngram / QENGINE_INT8_NGRAM_PATH) keep their documented allowance and are not gated.
    Missing components end the walk; callers report missing files with their own message."""
    walk = os.path.abspath(root)
    for part in os.path.relpath(os.path.abspath(path), walk).split(os.sep):
        walk = os.path.join(walk, part)
        try:
            mode = os.lstat(walk).st_mode
        except (FileNotFoundError, NotADirectoryError):
            return
        if stat.S_ISLNK(mode):
            raise ValueError(f"{walk}: symbolic link rejected inside the sandbox (must be a regular file)")


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
        if n != hidden or k not in tuple(m * hidden for m in MTP_FC_EMBEDDING_K_MULTIPLIERS):
            raise ValueError(f"{name}: shape {shape} incompatible with hidden_size {hidden} "
                             f"(K multiplier must be one of {MTP_FC_EMBEDDING_K_MULTIPLIERS}; "
                             f"shared rule with kMtpFcEmbeddingKMultipliers in src/flash/checkpoint.h)")
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
    """Per-part tensor plans; the body plan is a flat list, sharded later by plan_shards()."""
    plans, source_for_output, packed_for_name = {}, {}, {}
    desc_by_source = {d["source"]: d for d in descriptors}
    for part in ("body", "mtp"):
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
        plans[part] = plan
    return plans, source_for_output, packed_for_name


def body_group(name):
    """Shard grouping key: one group per transformer layer; everything else one group per tensor.
    PLE/n-gram-adjacent tensors are deferred by plan_shards into the final shard."""
    layer = re.match(r"^model\.language_model\.layers\.(\d+)\.", name)
    return ("layer", int(layer.group(1))) if layer else ("tensor", name)


def plan_shards(plan, max_bytes):
    """Group entries per layer, pack groups into shards <= max_bytes without splitting a group,
    then append n-gram-adjacent tensors to the final shard. Returns [entry, ...] per shard."""
    # header headroom scales down for tiny caps so --max-shard-bytes stays testable;
    # the default 20 GiB cap always keeps the full 8 MiB margin
    limit = max_bytes - min(HEADER_MARGIN, max(8192, max_bytes // 16))
    if limit <= 0:
        raise ValueError(f"--max-shard-bytes {max_bytes} leaves no room for the safetensors header")
    groups, deferred = {}, []
    for entry in plan:
        if "ple_embedding." in entry[0]:  # PLE metadata lives next to the n-gram table
            deferred.append(entry)
            continue
        groups.setdefault(body_group(entry[0]), []).append(entry)
    shards, cur, cur_bytes = [], [], 0
    for key in sorted(groups, key=lambda g: (g[0] != "layer", g)):
        group = groups[key]
        gbytes = sum(e[3] for e in group)
        if gbytes > limit and len(group) > 1:
            raise ValueError(f"layer group {key} needs {gbytes} bytes, over --max-shard-bytes {max_bytes}; "
                             f"a layer is never split across shards")
        if cur and cur_bytes + gbytes > limit:
            shards.append(cur)
            cur, cur_bytes = [], 0
        if gbytes > limit:  # single tensor larger than the cap: both readers reject an oversized
                            # shard, so refuse at plan time instead of emitting unreadable output
            raise ValueError(f"tensor {group[0][0]} serializes to {gbytes} bytes, which exceeds "
                             f"--max-shard-bytes {max_bytes}; a single tensor must fit one shard")
        cur.extend(group)
        cur_bytes += gbytes
    if deferred:
        dbytes = sum(e[3] for e in deferred)
        # the deferred set rides the final shard as one block: the cap applies to it like to any
        # shard, so an oversized entry (or an oversized set) is refused at plan time instead of
        # emitting a shard both readers reject
        if dbytes > limit:
            over = [e for e in deferred if e[3] > limit]
            if over:
                raise ValueError(f"tensor {over[0][0]} serializes to {over[0][3]} bytes, which exceeds "
                                 f"--max-shard-bytes {max_bytes}; a single tensor must fit one shard")
            raise ValueError(f"deferred PLE/n-gram-adjacent tensors together need {dbytes} bytes, which "
                             f"exceeds --max-shard-bytes {max_bytes}; they share the final shard and "
                             f"are never split across shards")
        if cur and cur_bytes + dbytes > limit:
            shards.append(cur)
            cur = []
        cur.extend(deferred)
    if cur:
        shards.append(cur)
    for shard in shards:
        shard.sort(key=lambda item: item[0])
    return [shard for shard in shards] or [[]]


def shard_names(count):
    return [f"model-{i + 1:05d}-of-{count:05d}.safetensors" for i in range(count)]


def validate_qfn_source(src, cfg, ngram_path, out):
    if os.path.realpath(src.d) == os.path.realpath(out):
        raise ValueError("--src and --out resolve to the same directory")
    if family(cfg) != "qwen4_exp":
        raise ValueError("QFN conversion requires model_type qwen4_exp")
    if cfg.get("quantization_config"):
        raise ValueError("QFN source config is already quantized")
    dims, names, table_names, descriptors, headers, full_model = inspect_qfn_source(src, cfg)
    ngram = validate_ngram_file(ngram_path, full_model, NGRAM_ROWS_PER_SHARD if full_model else None)
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
        q, scale = quant_int8(block, search, device, source)
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
    if a.external_ngram not in (None, ""):
        raise ValueError("--external-ngram takes no PATH during conversion; it only takes a path with --check")
    if a.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be positive")
    src = Src(a.src)
    dims, source_keys, excluded, descriptors, headers, ngram = validate_qfn_source(
        src, cfg, a.ngram_from, a.out)
    plans, source_for_output, packed_for_name = qfn_plans(source_keys, descriptors, headers)
    body_shards = plan_shards(plans["body"], a.max_shard_bytes)
    files = shard_names(len(body_shards))
    plans_out = list(zip(files, body_shards)) + [("model_mtp.safetensors", plans["mtp"])]
    # every output tensor names the shard it lives in
    tensor_file = {}
    for filename, plan in plans_out:
        for entry in plan:
            tensor_file[entry[0]] = filename
    t0 = time.time()
    for filename, plan in plans_out:
        plan = list(plan)
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
        "targets": targets, "input_activations": None, "output_activations": None, "format": "pack-quantized",
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

    if a.external_ngram is None:
        materialize_ngram(a.ngram_from, a.out, link=a.link_ngram)
    else:
        log(f"external n-gram: table stays at {os.path.abspath(a.ngram_from)}; manifest records identity only")
    ngram["sha256"] = sha256_file(a.ngram_from)
    ngram["equivalence_verified"] = False
    kind, digest = source_digest(src)

    converted = []
    converted_names = {d["source"] for d in descriptors}
    for desc in descriptors:
        converted.append({"source_key": desc["source"], "base": desc["base"],
                          "file": tensor_file[desc["base"] + ".weight_packed"],
                          "shape": list(headers[desc["source"]][0]), "dtype": headers[desc["source"]][1],
                          "outputs": {"packed": desc["base"] + ".weight_packed",
                                      "scale": desc["base"] + ".weight_scale",
                                      "shape": desc["base"] + ".weight_shape"}})
    preserved = {"body": {n: tensor_file[n] for n in source_keys
                          if not n.startswith("mtp.") and n not in converted_names},
                 "mtp": sorted(n for n in source_keys if n.startswith("mtp.") and n not in converted_names)}
    outputs = {"body": files, "mtp": "model_mtp.safetensors",
               "config": "config.json", "manifest": "qfn-int8-manifest.json"}
    if a.external_ngram is None:
        outputs["ngram"] = NGRAM_OUT_NAME
    source = {"index_sha256_kind": kind, "index_sha256": digest}
    if kind == "weight_files":
        # honest labeling: the inventory digest is best-effort, not a content hash (see source_digest)
        source["index_sha256_note"] = ("best-effort: digest over the sorted (filename, size) inventory; "
                                       "size-preserving edits of a weight file are invisible to it")
    if a.source_repo is not None or a.source_revision is not None:
        if a.source_repo is None or a.source_revision is None:
            raise ValueError("--source-repo and --source-revision must be given together")
        source["repo"], source["revision"] = a.source_repo, a.source_revision
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "format_version": 1, "family": "qwen4_exp",
        "source": source,
        "recipe": {"format": "symmetric INT8", "group_size": 128, "rounding": "nearest-even",
                   "scale_dtype": "BF16", "zero_point": 0, "search": a.search,
                   "algorithm": "RTN; per-group BF16 absmax/127; zero groups use scale 1"},
        "outputs": outputs,
        "shard_limit_bytes": a.max_shard_bytes,
        "tensors": {"converted": converted, "preserved": preserved, "excluded": excluded},
        "ngram": ngram,
        "runtime_supported": False,
        "runtime_support_reason": ("no runtime in this repository executes INT8 W8 checkpoints yet; "
                                   "the runtime is proposed separately")
    }
    with open(os.path.join(a.out, "qfn-int8-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"QFN: {len(converted)} matrices in {len(files)} body shards, {len(excluded)} n-gram source tensors excluded; "
        f"{'self-contained' if a.external_ngram is None else 'external'} n-gram; {time.time() - t0:.0f}s")


def cmd_make(a):
    if not a.src:
        raise ValueError("--src is required for conversion")
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
                q, s = quant_int8(wt, a.search, a.dev, name=base + ".weight")
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
    full_model = dims == (2560, LAYER_COUNT, 512, 640)
    if not descriptors:
        raise ValueError("QFN source contains no selected matrices")
    if mtp_count != MTP_MATRICES:
        raise ValueError(f"QFN MTP selection must contain {MTP_MATRICES} matrices, found {mtp_count}")
    if full_model and (body_count, len(table_names)) != (BODY_MATRICES, 128):
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


def expected_shard_inventory(manifest):
    """tensor -> shard file from the manifest mapping; rejects duplicates and unknown shards."""
    tensors = manifest.get("tensors", {})
    body = manifest.get("outputs", {}).get("body")
    if not isinstance(body, list) or not body or not all(isinstance(n, str) and n for n in body):
        raise ValueError("outputs.body must be a nonempty list of shard files")
    if len(set(body)) != len(body):
        raise ValueError("outputs.body lists a shard twice")
    if not isinstance(tensors.get("preserved"), dict) or not isinstance(tensors.get("preserved", {}).get("body"), dict):
        raise ValueError("tensors.preserved.body must be a tensor -> shard mapping")
    preserved_mtp = tensors["preserved"].get("mtp", [])
    if not isinstance(preserved_mtp, list) or not all(isinstance(n, str) and n for n in preserved_mtp):
        raise ValueError("tensors.preserved.mtp must be a list of tensor names")
    mapping = {}
    def put(name, shard):
        if name in mapping:
            raise ValueError(f"{name}: recorded in more than one shard")
        mapping[name] = shard
    for name, shard in tensors["preserved"]["body"].items():
        if shard not in body:
            raise ValueError(f"{name}: shard {shard} is not an output body shard")
        put(name, shard)
    for name in preserved_mtp:
        put(name, manifest.get("outputs", {}).get("mtp"))
    for record in tensors.get("converted", []):
        outputs = record.get("outputs", {})
        for suffix in (".weight_packed", ".weight_scale", ".weight_shape"):
            name = record.get("base", "") + suffix
            if record.get("file") not in body and record.get("file") != "model_mtp.safetensors":
                raise ValueError(f"{name}: shard {record.get('file')} is not an output file")
            put(name, record.get("file"))
    return mapping, body


def check_qfn(a, manifest_path, manifest):
    # Clean break, no compat reader: pre-sharding checkpoints are re-converted, never half-validated.
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{manifest_path}: schema updated, reconvert with the current tools/quant_int8.py "
                         f"(manifest schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r})")
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported QFN manifest format_version")
    if manifest.get("family") != "qwen4_exp":
        raise ValueError("QFN manifest family mismatch")
    recipe = manifest.get("recipe", {})
    if int(recipe.get("search", -1)) != a.search or recipe.get("group_size") != 128:
        raise ValueError("--search/group_size does not match QFN manifest")
    if not isinstance(manifest.get("runtime_supported"), bool):
        raise ValueError("QFN manifest must carry a boolean runtime_supported (review R2-8: presence and type are the contract; "
                         "true states the flash INT8 W8 runtime executes the checkpoint)")
    if not isinstance(manifest.get("source"), dict):
        raise ValueError("manifest is missing the source traceability record")
    source = manifest["source"]
    if (source.get("index_sha256_kind") not in ("index_json", "weight_files") or
            not re.fullmatch(r"[0-9a-f]{64}", str(source.get("index_sha256", "")))):
        raise ValueError("manifest source index digest is missing or malformed")
    # source honesty: repo/revision exist only when recorded; an unlisted source is never a rejection
    if bool("repo" in source) != bool("revision" in source):
        raise ValueError("manifest source repo/revision must be recorded together")

    outputs = manifest["outputs"]
    for key in ("body", "mtp", "config"):
        values = outputs[key] if key == "body" else [outputs.get(key)]
        for value in values:
            if not sandbox_relative(str(value)):
                raise ValueError(f"outputs.{key} entry {value!r} is not a sandboxed relative path")
    if len(set(outputs["body"]) | {outputs["mtp"]}) != len(outputs["body"]) + 1:
        raise ValueError("body shards and the MTP file must be distinct")
    limit = manifest.get("shard_limit_bytes", 0)
    if not isinstance(limit, int) or limit <= 0 or limit > MAX_SHARD_BYTES:
        raise ValueError(f"shard_limit_bytes {limit} is not within 1..{MAX_SHARD_BYTES}")
    mapping, body = expected_shard_inventory(manifest)
    if outputs.get("ngram") is not None and not sandbox_relative(outputs["ngram"]):
        raise ValueError(f"outputs.ngram {outputs['ngram']!r} is not a sandboxed relative path")
    if any(name.endswith(".part") for name in os.listdir(a.check)):
        raise ValueError("incomplete .part file remains in output")

    # every shard readable, within the cap, and its header keys exactly the tensors it claims
    outputs_by_file = {}
    for filename in body + [outputs["mtp"]]:
        path = os.path.join(a.check, filename)
        sandbox_regular_file(path, a.check)
        if not os.path.isfile(path):
            raise ValueError(f"{path}: missing output file")
        if os.path.getsize(path) > min(limit, MAX_SHARD_BYTES):
            raise ValueError(f"{path}: {os.path.getsize(path)} bytes exceeds the shard limit {limit}")
        outputs_by_file[filename] = Src(a.check, filename)
    for filename in body + [outputs["mtp"]]:
        reader = outputs_by_file[filename]
        claimed = {name for name, shard in mapping.items() if shard == filename}
        if filename in body and not claimed:  # parity with the C++ reader: an empty shard is never readable
            raise ValueError(f"{filename}: shard maps zero tensors")
        if set(reader.keys()) != claimed:
            missing = sorted(claimed - set(reader.keys()))[:3]
            extra = sorted(set(reader.keys()) - claimed)[:3]
            raise ValueError(f"{filename}: header inventory differs from manifest (missing {missing}, extra {extra})")

    # n-gram: identity + sha256 re-verification, self-contained or external
    identity = manifest.get("ngram", {})
    self_contained = outputs.get("ngram") is not None
    if self_contained:
        if a.external_ngram is not None:
            raise ValueError("checkpoint is self-contained; --external-ngram is not applicable")
        table_path = os.path.join(a.check, outputs["ngram"])
        sandbox_regular_file(table_path, a.check)
        if not os.path.isfile(table_path):
            raise ValueError(f"{table_path}: self-contained n-gram table is missing next to the weights")
    else:
        table_path = a.external_ngram or os.environ.get("QENGINE_INT8_NGRAM_PATH", "")
        if not table_path:
            raise ValueError("external n-gram checkpoint: pass the table path with --external-ngram PATH "
                             "or set QENGINE_INT8_NGRAM_PATH")
        table_path = os.path.abspath(table_path)
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", ""))):
        raise ValueError("manifest n-gram sha256 digest is missing or malformed")
    ngram = validate_ngram_file(table_path, False, identity.get("rows_per_shard"))
    if (ngram["size_bytes"] != identity.get("size_bytes") or ngram["keys"] != identity.get("keys") or
            ngram["layout"] != identity.get("layout") or identity.get("equivalence_verified") is not False):
        raise ValueError("n-gram table identity differs from the manifest record")
    actual_digest = sha256_file(table_path)
    if actual_digest != identity["sha256"]:
        raise ValueError(f"n-gram table sha256 {actual_digest} differs from the manifest {identity['sha256']}")

    sandbox_regular_file(os.path.join(a.check, str(outputs["config"])), a.check)
    cfg_out = model_config(a.check)
    if family(cfg_out) != "qwen4_exp":
        raise ValueError("output config family differs from the manifest")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not os.path.isfile(os.path.join(a.check, name)):
            raise ValueError(f"required asset missing from output: {name}")

    # converted outputs must carry the exact packed/scale/shape dtype+shape the record implies; this
    # contract is source-independent, so the structural check (without --src) verifies it too
    manifest_tensors = manifest.get("tensors", {})
    records_by_output = {}
    for record in manifest_tensors.get("converted", []):
        logical, recorded = record.get("shape"), record.get("outputs") or {}
        if (not isinstance(logical, list) or not logical or
                not all(isinstance(dim, int) and dim > 0 for dim in logical) or
                not all(isinstance(recorded.get(k), str) for k in ("packed", "scale", "shape"))):
            raise ValueError(f"converted record for {record.get('source_key')!r} carries no usable "
                             "shape/outputs to verify the tensor headers against")
        records_by_output[recorded["packed"]] = ("I32", logical[:-1] + [logical[-1] // 4])
        records_by_output[recorded["scale"]] = ("BF16", logical[:-1] + [logical[-1] // 128])
        records_by_output[recorded["shape"]] = ("I64", [len(logical)])
    for name, (dtype, shape) in records_by_output.items():
        if name not in mapping:
            raise ValueError(f"{name}: a converted output mapped to no shard")
        actual_shape, actual_dtype = tensor_info(outputs_by_file[mapping[name]], name)
        if actual_dtype != dtype or list(actual_shape) != list(shape):
            raise ValueError(f"{name}: header {actual_dtype} {list(actual_shape)}, expected {dtype} {list(shape)}")

    if not a.src:
        log(f"QFN structural check passed: {len(body)} body shards, "
            f"{len(manifest['tensors']['converted'])} matrices, "
            f"{'self-contained' if self_contained else 'external'} n-gram "
            f"(sha256 {identity['sha256'][:12]}..., re-hashed {ngram['size_bytes']} bytes); "
            "pass --src for source/RTN reproduction")
        return

    src_cfg = model_config(a.src)
    if (family(src_cfg) != "qwen4_exp" or src_cfg.get("quantization_config") or
            manifest.get("family") != "qwen4_exp"):
        raise ValueError("QFN manifest/source family mismatch or source is already quantized")
    kind, digest = source_digest(Src(a.src))
    if digest != source["index_sha256"]:
        raise ValueError("source index digest differs from the manifest: this checkpoint was not made from --src")
    ngram_src = validate_ngram_file(table_path, False, None)
    source_dims = qfn_dims(src_cfg)
    full_source = source_dims == (2560, LAYER_COUNT, 512, 640)
    if full_source and ngram_src["rows_per_shard"] != NGRAM_ROWS_PER_SHARD:
        raise ValueError(f"pinned QFN requires {NGRAM_ROWS_PER_SHARD} n-gram rows")
    src = Src(a.src)
    dims, source_keys, table_names, descriptors, headers, full_model = inspect_qfn_source(src, src_cfg)
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
        if (record.get("base") != desc["base"] or record.get("file") not in outputs_by_file or
                record.get("shape") != list(shape) or record.get("dtype") != dtype or
                record.get("outputs") != expected_outputs):
            raise ValueError(f"manifest descriptor differs for {desc['source']}")
    expected_preserved = {
        "body": {n: mapping[n] for n in source_keys
                 if not n.startswith("mtp.") and n not in expected_converted},
        "mtp": sorted(n for n in source_keys if n.startswith("mtp.") and n not in expected_converted)}
    if manifest_tensors.get("preserved") != expected_preserved:
        raise ValueError("manifest passthrough inventory does not match source")
    if (cfg_out.get("quantization_config", {}).get("target_modules") != targets):
        raise ValueError("output config descriptive quantization targets differ")
    if (outputs["mtp"], outputs["config"], outputs["manifest"]) != (
            "model_mtp.safetensors", "config.json", "qfn-int8-manifest.json"):
        raise ValueError("manifest output file inventory differs")
    report = {}
    for desc in descriptors:
        source_name, base = desc["source"], desc["base"]
        shape, _ = headers[source_name]
        output = outputs_by_file[converted_records[source_name]["file"]]
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
                expected_q, expected_s = quant_int8(original, a.search, name=source_name)
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
        output = outputs_by_file[mapping[name] if name in mapping else
                                 ("model_mtp.safetensors" if name.startswith("mtp.") else body[0])]
        assert_tensor_equal(src, output, name, shape, dtype)
    for key, (error_sq, source_sq, blocks) in sorted(report.items()):
        rel = (error_sq / source_sq) ** 0.5 if source_sq else 0.0
        log(f"{key}: {blocks} blocks, relative L2 error={rel:.6g}")
    if full_model and (len(descriptors), len(table_names)) != (BODY_MATRICES + MTP_MATRICES, 128):
        raise ValueError(f"full QFN verification inventory must be {BODY_MATRICES} body + {MTP_MATRICES} MTP "
                         f"matrices and 128 excluded tables")
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
        f"{sum(d['part'] == 'mtp' for d in descriptors)} MTP matrices; {len(table_names)} n-gram shards "
        f"({'self-contained' if self_contained else 'external'})")


def cmd_check(a):
    manifest_path = os.path.join(a.check, "qfn-int8-manifest.json")
    if os.path.isfile(manifest_path) and a.kv_from not in (None, ""):
        raise ValueError("QFN check does not accept --kv-from")
    if os.path.isfile(manifest_path):
        sandbox_regular_file(manifest_path, a.check)
        with open(manifest_path) as f:
            manifest = json.load(f, object_pairs_hook=strict_json_object)
        return check_qfn(a, manifest_path, manifest)
    if not a.src:
        raise ValueError("27B check requires --src")
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
        q, s = quant_int8(src.get(b + ".weight"), a.search, name=b + ".weight")
        same = torch.equal(q.to(torch.int16), u) and torch.equal(s.float(), sc)
        log(f"{b}: rel err {rel:.2e}, max |q| {u.abs().max().item()}, repro {'exact' if same else 'DIFFERS'}")
    ks = [n for n in out.keys() if n.endswith("k_scale")]
    log(f"{len(names)} INT8 matrices, {len(ks)} k_scale tensors")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="BF16 checkpoint (sharded or single); with --check enables source reproduction")
    ap.add_argument("--out", default="")
    ap.add_argument("--kv-from", default=None,
                    help="27B checkpoint supplying KV scales/config/tokenizer; omitted uses the upstream default")
    ap.add_argument("--ngram-from", default=None, help="QFN existing INT8 n-gram table (required for QFN)")
    ap.add_argument("--external-ngram", nargs="?", const="", default=None, metavar="PATH",
                    help="conversion: record table identity only, keep it outside the output; "
                         "--check: external table path (or set QENGINE_INT8_NGRAM_PATH)")
    ap.add_argument("--link-ngram", action="store_true",
                    help="self-contained conversion: hard link the n-gram table into the output "
                         "instead of the default independent copy (zero-copy, but a later write "
                         "through the source table path would mutate the deployed checkpoint)")
    ap.add_argument("--source-repo", default=None, help="record the ACTUAL source repository in the manifest")
    ap.add_argument("--source-revision", default=None, help="record the ACTUAL source revision in the manifest")
    ap.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES,
                    help=f"body shard cap in bytes (default {MAX_SHARD_BYTES} = 20 GiB)")
    ap.add_argument("--search", type=int, default=0, help="per-group clip-ratio candidates (0/1: plain RTN)")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--check", default="", help="verify an output dir; --src additionally checks source reproduction")
    ap.add_argument("--n", type=int, default=8, help="matrices sampled by the 27B --check")
    a = ap.parse_args()
    if a.check:
        cmd_check(a)
    elif a.out:
        cmd_make(a)
    else:
        ap.error("--out or --check")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:  # one clean line, never a traceback:
        # OSError covers PermissionError / FileNotFoundError when the filesystem refuses the operation
        print(f"quant_int8: {error}", file=sys.stderr)
        sys.exit(1)
