#!/usr/bin/env python3
"""INT8 copy of the Flash-Next n-gram (PLE) embedding table: ngram_int8.safetensors next to the BF16 source.

  python3 tools/convert_ngram_int8.py [--model DIR] [--out FILE] [--workers N] [--verify N]

The source (128 shards [2500012][160] BF16, 102.4 GB; model-00016-of-00017.safetensors via the model directory) is
only read. Per-row symmetric INT8: scale = fp16(absmax / 127), q = clip(rint(x / scale), -127, 127). A row is stored
as 160 int8 then its fp16 scale (162 bytes, one gather reads one row): tensors
  model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_<i>   U8 [2500012, 162]
51.8 GB in all. Per-row vs per-group-32 on 160K sampled rows: relative error 0.66 % vs 0.53 % (rows have no outliers:
absmax / rms median 2.8, p99 3.9), so the per-row layout (+1.2 % bytes for scales vs +6 % for g32) is kept.
--verify N: re-reads N random rows of every shard and checks the dequantized error (prints max / mean relative error).
"""
import argparse, json, os, struct, sys, time
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
ap.add_argument("--out", default=None)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--verify", type=int, default=0)
a = ap.parse_args()
ROWS, DIM, RB = 2500012, 160, 162
PRE = "model.language_model.layers.1.ple.ple_embedding."


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def find_source(model):
    for name in sorted(os.listdir(model)):
        if not name.endswith(".safetensors"):
            continue
        p = os.path.join(model, name)
        h, _ = header(p)
        if PRE + "ngram_embedding.shard_0.weight" in h:
            return p
    raise SystemExit("no n-gram shard in " + model)


SRC = find_source(a.model)
OUT = a.out or os.path.join(a.model, "ngram_int8.safetensors")
SH, SBASE = header(SRC)


def out_layout():
    hdr = {"__metadata__": {"format": "pt", "source": os.path.basename(os.path.realpath(SRC)),
                            "layout": "per row: 160 x int8 then fp16 scale (x = q * scale)"}}
    off = 0
    for i in range(128):
        hdr[PRE + f"ngram_q8.shard_{i}"] = {"dtype": "U8", "shape": [ROWS, RB], "data_offsets": [off, off + ROWS * RB]}
        off += ROWS * RB
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return hb, 8 + len(hb), off


HB, OBASE, TOTAL = out_layout()


def quant(x):  # x float32 [n][160] -> uint8 [n][162]
    s = (np.abs(x).max(1) / 127.0).astype(np.float16)
    sf = s.astype(np.float32)
    sf[sf == 0] = 1.0
    q = np.clip(np.rint(x / sf[:, None]), -127, 127).astype(np.int8)
    o = np.empty((x.shape[0], RB), np.uint8)
    o[:, :DIM] = q.view(np.uint8)
    o[:, DIM:] = s.view(np.uint8).reshape(-1, 2)
    return o


def shard(i):
    t = SH[PRE + f"ngram_embedding.shard_{i}.weight"]
    assert t["dtype"] == "BF16" and t["shape"] == [ROWS, DIM]
    src = np.memmap(SRC, dtype=np.uint16, mode="r", offset=SBASE + t["data_offsets"][0], shape=(ROWS, DIM))
    dst = np.memmap(OUT, dtype=np.uint8, mode="r+", offset=OBASE + i * ROWS * RB, shape=(ROWS, RB))
    C = 250000
    for r0 in range(0, ROWS, C):
        x = (np.asarray(src[r0:r0 + C]).astype(np.uint32) << 16).view(np.float32)
        dst[r0:r0 + C] = quant(x)
    dst.flush()
    del dst
    return i


def verify(i, n=a.verify):
    t = SH[PRE + f"ngram_embedding.shard_{i}.weight"]
    src = np.memmap(SRC, dtype=np.uint16, mode="r", offset=SBASE + t["data_offsets"][0], shape=(ROWS, DIM))
    dst = np.memmap(OUT, dtype=np.uint8, mode="r", offset=OBASE + i * ROWS * RB, shape=(ROWS, RB))
    idx = np.sort(np.random.default_rng(i).choice(ROWS, n, replace=False))
    x = (np.asarray(src[idx]).astype(np.uint32) << 16).view(np.float32)
    d = np.asarray(dst[idx])
    y = d[:, :DIM].view(np.int8).astype(np.float32) * d[:, DIM:].copy().view(np.float16).astype(np.float32)
    e = np.linalg.norm(y - x, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-12)
    return float(e.max()), float(e.mean())


if __name__ == "__main__":
    print(f"source {SRC} -> {OUT}: {TOTAL / 1e9:.2f} GB", flush=True)
    if not a.verify:
        if os.path.realpath(OUT) == os.path.realpath(SRC):
            raise SystemExit("refusing to overwrite the source")
        with open(OUT, "wb") as f:
            f.write(struct.pack("<Q", len(HB)))
            f.write(HB)
            f.truncate(OBASE + TOTAL)
        t0 = time.time()
        with Pool(a.workers) as p:
            for k, i in enumerate(p.imap_unordered(shard, range(128))):
                if k % 16 == 15:
                    print(f"  {k + 1}/128 shards, {time.time() - t0:.0f} s", flush=True)
        print(f"done in {time.time() - t0:.0f} s", flush=True)
    else:
        with Pool(a.workers) as p:
            r = p.map(verify, range(128))
        print("verify: max relative row error %.4f %%, mean %.4f %%" % (100 * max(x[0] for x in r), 100 * np.mean([x[1] for x in r])))
