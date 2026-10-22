"""Speculative (MTP) sampling must leave the sampling distribution unchanged at every position,
not just the first tokens: per-position token histograms of MTP vs plain sampling, compared with
the plain-vs-plain noise level (independent seeds).  python3 test_sampling_deep.py [N] [T]"""
import os, sys, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
L = int(sys.argv[2]) if len(sys.argv) > 2 else 12
PROMPTS = ["Once upon a time, in a small village by the sea,",
           "<|im_start|>user\nWrite a bash command that lists the 5 largest files in a directory.<|im_end|>\n"
           "<|im_start|>assistant\n<think>\n\n</think>\n\n"]
e = Engine(MODEL, max_t=8192)


def sample(ids, k, seed0, top_p, top_k):
    out = []
    for s in range(N):
        e.set_sampling(1.0, top_p, top_k, 0.0, seed0 + s * 7919)
        t, _, _ = e.start(ids, k)
        o = [t]
        while len(o) < L:
            o += e.step()
        out.append(o[:L])
    return out


def tv(a, b, pos):
    ca = collections.Counter(x[pos] for x in a)
    cb = collections.Counter(x[pos] for x in b)
    return 0.5 * sum(abs(ca[t] - cb[t]) for t in set(ca) | set(cb)) / N


for top_p, top_k in ((1.0, 40), (0.95, 20)):
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False).ids
        plain_a = sample(ids, 0, 1, top_p, top_k)
        plain_b = sample(ids, 0, 500001, top_p, top_k)
        mtp = sample(ids, 3, 900001, top_p, top_k)
        base = [tv(plain_a, plain_b, i) for i in range(L)]
        test = [tv(plain_a, mtp, i) for i in range(L)]
        print(f"top_p={top_p} top_k={top_k} prompt={p[:30]!r}")
        print("  TV plain/plain (noise):", " ".join(f"{x:.2f}" for x in base), f" mean {sum(base) / L:.3f}")
        print("  TV plain/MTP          :", " ".join(f"{x:.2f}" for x in test), f" mean {sum(test) / L:.3f}", flush=True)
e.close()
