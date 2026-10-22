"""Sampling: MTP speculative sampling must match the plain sampling distribution."""
import collections, ctypes, os, sys, time
import common as T
from tokenizers import Tokenizer

lib = T.lib
lib.qe_set_sampling.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                ctypes.c_float, ctypes.c_ulonglong]
tok = Tokenizer.from_file(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4/tokenizer.json"))
e = lib.qe_create(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4").encode(), b"0,1,2", 4096, b"")
ids = tok.encode("Once upon a time, in a small village,").ids
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

def dist(k):
    c2, c3 = collections.Counter(), collections.Counter()
    for seed in range(N):
        lib.qe_set_sampling(e, 1.0, 1.0, 40, 0.0, seed * 7919 + 1)
        out, _, _ = T.generate(e, ids, k, 4)
        c2[tuple(out[:2])] += 1
        c3[tuple(out[:3])] += 1
    return c2, c3

p2, p3 = dist(0)
s2, s3 = dist(3)
def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[x] - b[x]) for x in keys) / N
# expected TV between two independent samples of size N from the same distribution is
# roughly sqrt(K / N) scale; compare against a plain-vs-plain baseline with other seeds
print(f"TV(first 2 tokens) plain vs MTP: {tv(p2, s2):.3f}   TV(first 3): {tv(p3, s3):.3f}")
top = p2.most_common(5)
print("top pairs plain:", [(tok.decode(list(k)), v) for k, v in top])
print("top pairs MTP:  ", [(tok.decode(list(k)), s2[k]) for k, _ in top])
lib.qe_set_sampling(e, 0.7, 0.95, 40, 0.0, 1234)
for k in (0, 3):
    out, rate, tps = T.generate(e, tok.encode("Write a detailed explanation of how a hash map works, with a C++ implementation.").ids, k, 400)
    print(f"temp 0.7 k={k}: {rate:.1f} tok/s, {tps:.2f} tok/step | {tok.decode(out[:30])!r}")
lib.qe_set_sampling(e, 0.0, 1.0, 40, 0.0, 0)
out, rate, tps = T.generate(e, ids, 3, 64)
print(f"greedy again: {rate:.1f} tok/s | {tok.decode(out[:20])!r}")
lib.qe_destroy(e)
