"""Token frequency ranking for the MTP draft vocabulary (FR-Spec style).

Drafts are only proposed from the most frequent tokens, so the draft lm_head reads a fraction
of the vocabulary; verification still uses the full vocabulary, so outputs are unchanged.
Counts tokens of a text corpus you provide (directories of source code, markdown, chat logs, ...;
it should resemble the traffic the server will see). Only the ranking of token ids is written.

usage: python tools/draft_vocab.py --corpus DIR [DIR|FILE ...] [--mb 260]
                                   [--out models/Qwen3.8-27B-INT4/token_rank.bin]
"""
import argparse
import collections
import os
import random

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def corpus_files(roots, limit_bytes):
    """Text of the files under the given directories (or the given files), shuffled, up to limit_bytes."""
    exts = (".py", ".ts", ".tsx", ".js", ".cpp", ".h", ".cu", ".md", ".go", ".rs", ".sh", ".yaml", ".txt",
            ".json", ".jsonl")
    files = []
    for root in roots:
        if os.path.isfile(root):
            files.append(root)
            continue
        for dp, dn, fn in os.walk(os.path.expanduser(root)):
            dn[:] = [d for d in dn if d not in ("node_modules", ".git", "models", "dist", "build")
                     and not d.startswith(".venv")]
            files += [os.path.join(dp, f) for f in fn if f.endswith(exts)]
    random.Random(0).shuffle(files)
    total = 0
    for f in files:
        try:
            if os.path.getsize(f) > 1_000_000:
                continue
            t = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        total += len(t)
        yield t
        if total > limit_bytes:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen3.8-27B-INT4"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="directories (walked recursively) and/or files of text to count")
    ap.add_argument("--mb", type=int, default=260, help="read at most this many MB of text")
    args = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    counts = collections.Counter()
    for name, gen in (("corpus", lambda lim: corpus_files(args.corpus, lim)),):
        batch, n = [], 0
        for t in gen(args.mb * 1_000_000):
            batch.append(t)
            if len(batch) == 256:
                for e in tok.encode_batch(batch, add_special_tokens=False):
                    counts.update(e.ids)
                    n += len(e.ids)
                batch = []
        for e in tok.encode_batch(batch, add_special_tokens=False):
            counts.update(e.ids)
            n += len(e.ids)
        print(f"{name}: {n / 1e6:.1f}M tokens")
    vocab = tok.get_vocab_size()
    # rank: by count, unseen tokens last (in id order)
    rank = sorted(range(vocab), key=lambda i: (-counts.get(i, 0), i))
    out = args.out or os.path.join(args.model, "token_rank.bin")
    np.asarray(rank, dtype=np.int32).tofile(out)
    tot = sum(counts.values())
    for n in (16384, 32768, 65536):
        cov = sum(counts[i] for i in rank[:n]) / tot
        print(f"top {n}: covers {cov * 100:.2f}% of counted tokens")
    print("wrote", out)


if __name__ == "__main__":
    main()
