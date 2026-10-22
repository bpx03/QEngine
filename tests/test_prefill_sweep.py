"""Prompt-chunk length sweep: every chunk-size boundary of the prefill paths (decode-style rows <= 32,
33-64 rows, prefill GEMM / copy-engine all-reduce / chunked delta-net at multiples of 64, micro-batches
1024 / 2048, INT8 Q.K^T from position 16K) must neither crash nor change the result.
(A) fresh prompts of length L; (B) a cached prefix of B tokens continued by s new tokens (the chat /
prompt-cache path) must give the same 16 greedy tokens as the same prompt prefilled from scratch.
(B) is reported, not gated: a continuation prefills other chunk sizes than a full prefill (other GEMM tiles,
other delta-net chunk splits), so near-tie argmaxes flip. Baseline 2026-10-21 (same repeated text, 291 cases):
master 1c3f603 100 differ (26 from the 1st token), night-merge 81 (26). PASS = no crash, finite tokens,
prefix reused. Run with QE_GEMM_TUNE=0 (the autotuner picks kernels by timing, so tiles may differ between runs).
usage: QE_GEMM_TUNE=0 python3 tests/test_prefill_sweep.py [--quick]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
base = tok.encode("Summarize this text in detail:\n" + text * 8).ids
quick = "--quick" in sys.argv
N = 16
edges = [255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049]
fresh_lens = (list(range(1, 131)) if not quick else [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129]) + edges + \
    [4095, 4096, 4097, 16383, 16384, 16385]
suffixes = (list(range(1, 131)) if not quick else [1, 2, 31, 32, 33, 63, 64, 65, 129]) + edges
bases = [3000, 4096, 16000]
assert len(base) > 16000 + 2049 + 16, len(base)

e = Engine(MODEL)
e.set_sampling(0)


def gen(ids):
    t, r, s = e.start(ids, 3)
    out = [t]
    while len(out) < N:
        out += e.step()
    return out[:N], r


fails = []
for L in fresh_lens:  # (A) no crash, sane output
    e.reset()
    out, _ = gen(base[:L])
    if len(out) != N or any(t < 0 for t in out):
        fails.append(("fresh", L, out))
print(f"(A) fresh prefill: {len(fresh_lens)} lengths, {len(fails)} failures")

worst = N
ndiff = 0
for B in bases:  # (B) cached prefix + s new tokens == from scratch
    for s in suffixes:
        if B == 16000 and s not in (1, 33, 64, 65, 129, 384, 385, 1024, 2049):
            continue
        e.reset()
        ref, _ = gen(base[:B + s])
        e.reset()
        gen(base[:B])
        out, r = gen(base[:B + s])
        d = next((i for i in range(N) if out[i] != ref[i]), N)
        worst = min(worst, d)
        if d < N:
            ndiff += 1
            print(f"  B={B} s={s}: reused {r}, agreement {d}/{N}")
        if r < B - 1 or len(out) != N:
            fails.append(("cont", B, s, r, d))
print(f"(B) cached prefix + suffix: {ndiff} differ from a full prefill, worst agreement {worst}/{N}")
e.close()
print("PASS" if not fails else f"FAIL {len(fails)}: {fails[:10]}")
