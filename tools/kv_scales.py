"""FP8 KV cache scales: a side file of per-layer k_scale / v_scale for checkpoints that have none.

The engine stores e4m3(x / scale) for every key (after k_norm and RoPE) and value of the full-attention
layers. Without self_attn.k_scale / v_scale tensors it falls back to a bf16 cache (half the KV pool).
The side file holds just those scalars, under the checkpoint's tensor names; the engine reads
<model>/kv_scales.safetensors when present (QE_KV_SCALES=<file> elsewhere, =off to ignore it).

  extract    copy the scales of a checkpoint that has them (e.g. the RedHat INT4):
             python tools/kv_scales.py extract --src models/Qwen3.8-27B-INT4 --out models/Qwen3.8-27B-INT8/kv_scales.safetensors
  calibrate  run prompts through the engine on the target checkpoint (bf16 cache, QE_KV_AMAX=1) and
             set scale = margin * max|x| / 448 per layer:
             python tools/kv_scales.py calibrate --model models/Qwen3.8-27B-INT8 --corpus DIR|FILE ... \
                 [--ref models/Qwen3.8-27B-INT4] [--out <model>/kv_scales.safetensors]
  show       print the scales of a checkpoint or side file
"""
import argparse
import ctypes
import json
import os
import random
import struct
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
E4M3_MAX = 448.0
SUFFIXES = ("self_attn.k_scale", "self_attn.v_scale")


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_files(d):
    """the checkpoint's safetensors files (a directory: model.safetensors or the index's shards), or d itself"""
    if os.path.isfile(d):
        return [d]
    if os.path.exists(os.path.join(d, "model.safetensors")):
        return [os.path.join(d, "model.safetensors")]
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    return [os.path.join(d, f) for f in sorted(set(idx["weight_map"].values()))]


def read_scales(src):
    """{tensor name: float} of every *.self_attn.k_scale / v_scale scalar in src"""
    out = {}
    for path in st_files(src):
        h, base = st_header(path)
        with open(path, "rb") as f:
            for name, t in h.items():
                if name == "__metadata__" or not name.endswith(SUFFIXES):
                    continue
                f.seek(base + t["data_offsets"][0])
                raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
                if t["dtype"] == "F32":
                    v = np.frombuffer(raw, np.float32)
                elif t["dtype"] == "BF16":
                    v = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
                else:
                    sys.exit(f"{name}: dtype {t['dtype']} not supported")
                if v.size != 1:
                    sys.exit(f"{name}: {v.size} values, expected a per-tensor scalar")
                out[name] = float(v[0])
    return out


def write_scales(path, scales, meta):
    """F32 scalars (shape []), the format the engine's loader reads"""
    header, data = {"__metadata__": {k: str(v) for k, v in meta.items()}}, b""
    for name in sorted(scales, key=layer_key):
        header[name] = {"dtype": "F32", "shape": [], "data_offsets": [len(data), len(data) + 4]}
        data += struct.pack("<f", scales[name])
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + data)
    print(f"wrote {len(scales)} scales to {path}")


def layer_key(name):
    p = name.split(".")
    i = p.index("layers") + 1 if "layers" in p else -1
    return (int(p[i]) if i > 0 else 1 << 30, name)


def name_of(layer, kind):
    return f"model.language_model.layers.{layer}.self_attn.{kind}_scale"


def show(scales, ref=None):
    layers = sorted({layer_key(n)[0] for n in scales})
    for i in layers:
        k, v = scales.get(name_of(i, "k")), scales.get(name_of(i, "v"))
        line = f"layer {i:2d}  k {k:.6g}  v {v:.6g}"
        if ref and name_of(i, "k") in ref:
            line += f"   (ref k {ref[name_of(i, 'k')]:.6g} x{k / ref[name_of(i, 'k')]:.2f}," \
                    f" v {ref[name_of(i, 'v')]:.6g} x{v / ref[name_of(i, 'v')]:.2f})"
        print(line)


def corpus(roots, seed):
    """texts of the given files / directories (recursively), shuffled"""
    exts = (".py", ".md", ".txt", ".rst", ".c", ".cpp", ".h", ".cu", ".js", ".ts", ".go", ".rs", ".sh",
            ".json", ".jsonl", ".html", ".tex", ".yaml")
    files = []
    for r in roots:
        r = os.path.expanduser(r)
        if os.path.isfile(r):
            files.append(r)
            continue
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in (".git", "node_modules", "build", "models") and not d.startswith(".")]
            files += [os.path.join(dp, f) for f in fn if f.endswith(exts)]
    random.Random(seed).shuffle(files)
    for f in files:
        try:
            if 2000 < os.path.getsize(f) < 4_000_000:
                yield f, open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            pass


def calibrate(args):
    from tokenizers import Tokenizer

    os.environ["QE_KV_AMAX"] = "1"
    os.environ["QE_KV_FP8"] = "0"  # bf16 cache: the stored values are the model's, no saturation upstream
    os.environ["QE_KV_SCALES"] = "off"
    os.environ.pop("QE_DISK_CACHE", None)
    lib = ctypes.CDLL(os.environ.get("QE_LIB") or os.path.join(ROOT, "build", "libqengine.so"))
    lib.qe_create.restype = ctypes.c_void_p
    lib.qe_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    lib.qe_reset.argtypes = [ctypes.c_void_p]
    lib.qe_destroy.argtypes = [ctypes.c_void_p]
    lib.qe_start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    lib.qe_kv_amax.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    nl = cfg.get("text_config", cfg)["num_hidden_layers"]
    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    rng = random.Random(args.seed)
    lens = [int(x) for x in args.lengths.split(",")]
    e = lib.qe_create(os.path.abspath(args.model).encode(), args.devices.encode(), max(lens) + 256, b"")
    if not e:
        sys.exit("engine creation failed")
    amax = np.zeros((nl + 1, 2), np.float32)
    buf = np.zeros((nl + 1) * 2, np.float32)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    n, ntok, t0 = 0, 0, time.time()
    head = "<|im_start|>user\n"
    tail = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    for path, text in corpus(args.corpus, args.seed):
        L = lens[n % len(lens)]
        ids = tok.encode(text).ids
        if len(ids) < L // 2:
            continue
        s = rng.randrange(0, max(1, len(ids) - L))
        ids = tok.encode(head).ids + ids[s:s + L] + tok.encode(tail).ids
        lib.qe_reset(e)  # no prefix reuse: every prompt row goes through the layers
        lib.qe_start(e, (ctypes.c_int * len(ids))(*ids), len(ids), 1)  # k = 1: the MTP layer too
        if not lib.qe_kv_amax(e, ptr, 1):
            sys.exit("calibration is off in this libqengine (needs QE_KV_AMAX support)")
        a = buf.reshape(nl + 1, 2)
        grew = (a > amax).sum()
        amax = np.maximum(amax, a)
        n += 1
        ntok += len(ids)
        if args.verbose or n % 20 == 0:
            print(f"{n:4d} prompts {ntok:8d} tokens {time.time() - t0:6.0f} s  {os.path.basename(path)[:40]:40s}"
                  f" +{grew} max k {amax[:nl, 0].max():.3g} v {amax[:nl, 1].max():.3g}", flush=True)
        if n >= args.prompts:
            break
    lib.qe_destroy(e)
    if n == 0:
        sys.exit("no usable text in the corpus")
    full = [i for i in range(nl) if amax[i].min() > 0]
    print(f"{n} prompts, {ntok} tokens; full-attention layers {full}")
    print(f"MTP layer max |k| {amax[nl, 0]:.4g} |v| {amax[nl, 1]:.4g} (its cache uses the max scale over the layers)")
    scales = {}
    for i in full:
        scales[name_of(i, "k")] = float(amax[i, 0]) * args.margin / E4M3_MAX
        scales[name_of(i, "v")] = float(amax[i, 1]) * args.margin / E4M3_MAX
    mk, mv = max(scales[name_of(i, "k")] for i in full), max(scales[name_of(i, "v")] for i in full)
    if amax[nl, 0] > mk * E4M3_MAX or amax[nl, 1] > mv * E4M3_MAX:
        print("note: the MTP layer exceeds the largest layer scale and will saturate a little (QE_MTP_SCALE)")
    show(scales, read_scales(args.ref) if args.ref else None)
    out = args.out or os.path.join(args.model, "kv_scales.safetensors")
    write_scales(out, scales, {"source": "tools/kv_scales.py calibrate", "prompts": n, "tokens": ntok,
                               "margin": args.margin, "lengths": args.lengths})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("--src", required=True, help="checkpoint directory or safetensors file with the scales")
    x.add_argument("--out", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--model", required=True)
    c.add_argument("--corpus", nargs="+", required=True, help="text files / directories to sample prompts from")
    c.add_argument("--out", default=None, help="default <model>/kv_scales.safetensors")
    c.add_argument("--ref", default=None, help="checkpoint or side file to compare the scales with")
    c.add_argument("--devices", default="0,1")
    c.add_argument("--prompts", type=int, default=256)
    c.add_argument("--lengths", default="512,2048,8192", help="prompt lengths in tokens (cycled)")
    c.add_argument("--margin", type=float, default=1.2, help="headroom over the observed max |x|")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("-v", "--verbose", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("src")
    s.add_argument("--ref", default=None)
    args = ap.parse_args()
    if args.cmd == "extract":
        scales = read_scales(args.src)
        if not scales:
            sys.exit(f"no k_scale / v_scale in {args.src}")
        show(scales)
        write_scales(args.out, scales, {"source": f"extract {os.path.abspath(args.src)}"})
    elif args.cmd == "calibrate":
        calibrate(args)
    else:
        show(read_scales(args.src), read_scales(args.ref) if args.ref else None)


if __name__ == "__main__":
    main()
