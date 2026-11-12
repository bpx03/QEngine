#!/usr/bin/env python3
"""FP8 (E4M3 + global scale) n-gram table -> QEngine INT8 row layout.

vcerny/ThinkingCap-...-FP8NGram stores the PLE table as
  model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_<i>.weight  F8_E4M3 [2500012,160]
  model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale     F32 [1]   (global)
QEngine wants ngram_int8.safetensors with per-row symmetric INT8:
  ...ngram_q8.shard_<i>  U8 [2500012,162]   160 x int8 then the row's fp16 scale  (x = q * scale)

Usage: convert_ngram_fp8.py --model DIR --out FILE [--workers N]
  --out is required: write it into the converter's output directory (DST/ngram_int8.safetensors; create DST first),
  then run convert_flash_next.py with QE_NGRAM_EXTERNAL=1 (its index points at DST/ngram_int8.safetensors and it
  checks the file). Not into the source DIR: the converter reads every *.safetensors there.
The source shards are memory-mapped and converted in chunks of 250K rows (numpy only, spawn workers).
"""
import argparse, json, os, struct, sys, time
import multiprocessing as mp
import numpy as np

ROWS, DIM, RB = 2500012, 160, 162
PRE = "model.language_model.layers.1.ple.ple_embedding."
NSHARD = 128
CHUNK = 250000


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def e4m3_lut():
    """uint8 -> float32 of every F8_E4M3 (fn) code: bias 7, subnormals at exponent 0, S.1111.111 = NaN"""
    b = np.arange(256)
    s = np.where(b >> 7, -1.0, 1.0)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return (s * v).astype(np.float32)


def shard(job):
    i, src, soff, out, obase, rows, chunk, gscale = job
    lut = e4m3_lut()
    w = np.memmap(src, dtype=np.uint8, mode="r", offset=soff, shape=(rows, DIM))
    o = np.memmap(out, dtype=np.uint8, mode="r+", offset=obase + i * rows * RB, shape=(rows, RB))
    for r0 in range(0, rows, chunk):
        x = lut[np.asarray(w[r0:r0 + chunk])] * gscale
        sf = np.abs(x).max(axis=1) / 127.0
        sf[sf == 0] = 1.0
        q = np.clip(np.rint(x / sf[:, None]), -127, 127).astype(np.int8)
        o[r0:r0 + chunk, :DIM] = q.view(np.uint8)
        o[r0:r0 + chunk, DIM:] = sf.astype(np.float16).view(np.uint8).reshape(-1, 2)
    o.flush()
    del o
    return i


def main():  # everything with side effects stays here: the spawn workers re-import this module
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="source model dir (FP8 n-gram shards + model.safetensors.index.json)")
    ap.add_argument("--out", required=True, help="output file, normally DST/ngram_int8.safetensors of convert_flash_next.py")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = a.out

    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    sk = f"{PRE}ngram_embedding.weight_scale"
    sp = os.path.join(a.model, idx[sk])
    h, base = header(sp)
    t = h[sk]
    assert t["dtype"] == "F32" and int(np.prod(t["shape"])) == 1, (sk, t["dtype"], t["shape"], "expected one F32 scale")
    gscale = float(np.fromfile(sp, dtype=np.float32, count=1, offset=base + t["data_offsets"][0])[0])
    print("global FP8 scale:", gscale, flush=True)

    jobs, srcs = [], {os.path.realpath(sp)}
    hb_cache = {}
    for i in range(NSHARD):
        k = f"{PRE}ngram_embedding.shard_{i}.weight"
        src = os.path.join(a.model, idx[k])
        if src not in hb_cache:
            hb_cache[src] = header(src)
        sh, sbase = hb_cache[src]
        t = sh[k]
        assert t["dtype"] == "F8_E4M3" and t["shape"] == [ROWS, DIM], (k, t["dtype"], t["shape"], f"expected F8_E4M3 [{ROWS}, {DIM}]")
        srcs.add(os.path.realpath(src))
        jobs.append([i, src, sbase + t["data_offsets"][0], out, None, ROWS, CHUNK, gscale])
    if os.path.realpath(out) in srcs:
        raise SystemExit(f"refusing to overwrite the source {out}")

    # output header (padded to 8 bytes like tools/convert_ngram_int8.py)
    hdr = {"__metadata__": {"format": "pt", "layout": "per row: 160 x int8 then fp16 scale (x = q * scale)"}}
    off = 0
    for i in range(NSHARD):
        hdr[f"{PRE}ngram_q8.shard_{i}"] = {"dtype": "U8", "shape": [ROWS, RB], "data_offsets": [off, off + ROWS * RB]}
        off += ROWS * RB
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.truncate(8 + len(hb) + off)
    for j in jobs:
        j[4] = 8 + len(hb)

    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as p:  # no fork: a caller that ran torch first deadlocked forked children
        for k, _ in enumerate(p.imap_unordered(shard, [tuple(j) for j in jobs])):
            if (k + 1) % 16 == 0:
                print(f"  {k + 1}/{NSHARD} shards, {time.time() - t0:.0f}s", flush=True)
    print("wrote", out, f"({os.path.getsize(out)/1e9:.1f} GB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
