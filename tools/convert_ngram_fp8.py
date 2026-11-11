#!/usr/bin/env python3
"""FP8 (E4M3 + global scale) n-gram table -> QEngine INT8 row layout.

vcerny/ThinkingCap-...-FP8NGram stores the PLE table as
  model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_<i>.weight  F8_E4M3 [2500012,160]
  model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale     F32 [1]   (global)
QEngine wants ngram_int8.safetensors with per-row symmetric INT8:
  ...ngram_q8.shard_<i>  U8 [2500012,162]   160 x int8 then the row's fp16 scale  (x = q * scale)

Usage: convert_ngram_fp8.py --model DIR [--out FILE] [--workers N]
"""
import argparse, json, os, struct, sys, time
from multiprocessing import Pool
import numpy as np
import torch
from safetensors import safe_open

ROWS, DIM, RB = 2500012, 160, 162
PRE = "model.language_model.layers.1.ple.ple_embedding."
NSHARD = 128

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
OUT = a.out or os.path.join(a.model, "ngram_int8.safetensors")

idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
shard_files = {i: idx[f"{PRE}ngram_embedding.shard_{i}.weight"] for i in range(NSHARD)}
with safe_open(os.path.join(a.model, idx[f"{PRE}ngram_embedding.weight_scale"]),
               framework="pt", device="cpu") as f:
    GSCALE = float(f.get_tensor(f"{PRE}ngram_embedding.weight_scale").reshape(-1)[0])
print("global FP8 scale:", GSCALE, flush=True)

# output header
hdr = {"__metadata__": {"format": "pt"}}
off = 0
for i in range(NSHARD):
    hdr[f"{PRE}ngram_q8.shard_{i}"] = {"dtype": "U8", "shape": [ROWS, RB],
                                       "data_offsets": [off, off + ROWS * RB]}
    off += ROWS * RB
hb = json.dumps(hdr).encode()
with open(OUT, "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
OBASE = 8 + len(hb)
with open(OUT, "r+b") as f:
    f.truncate(OBASE + off)


def shard(i):
    with safe_open(os.path.join(a.model, shard_files[i]), framework="pt", device="cpu") as f:
        w = f.get_tensor(f"{PRE}ngram_embedding.shard_{i}.weight")          # [ROWS,160] fp8
    x = w.to(torch.float32).numpy() * GSCALE
    sf = np.abs(x).max(axis=1) / 127.0
    sf[sf == 0] = 1.0
    q = np.clip(np.rint(x / sf[:, None]), -127, 127).astype(np.int8)
    o = np.empty((ROWS, RB), np.uint8)
    o[:, :DIM] = q.view(np.uint8)
    o[:, DIM:] = sf.astype(np.float16).view(np.uint8).reshape(-1, 2)
    with open(OUT, "r+b") as f:
        f.seek(OBASE + i * ROWS * RB)
        f.write(o.tobytes())
    return i


if __name__ == "__main__":
    t0 = time.time()
    with Pool(a.workers) as p:
        for k, _ in enumerate(p.imap_unordered(shard, range(NSHARD))):
            if (k + 1) % 16 == 0:
                print(f"  {k + 1}/{NSHARD} shards, {time.time() - t0:.0f}s", flush=True)
    print("wrote", OUT, f"({os.path.getsize(OUT)/1e9:.1f} GB) in {time.time()-t0:.0f}s")
