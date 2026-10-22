"""Cross-slot prefix reuse: B shares a long prefix with A (still active in its slot), lands in
another slot and must resume from A's snapshot + copied KV, matching a fresh computation."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
from tokenizers import Tokenizer
import transformers
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
sysp = open(os.path.join(os.path.dirname(transformers.__file__), "generation", "logits_process.py")).read()
pre = tok.encode("<|im_start|>system\n" + sysp[:20000] + "<|im_end|>\n").ids
A = pre + tok.encode("<|im_start|>user\nSummarize the first class.<|im_end|>\n<|im_start|>assistant\n").ids
B = pre + tok.encode("<|im_start|>user\nWhat does TopK warper do?<|im_end|>\n<|im_start|>assistant\n").ids
e = Engine(MODEL)
e.set_sampling(0)
N = 60
e.set_concurrent(True)
ta, ra, sa = e.start(A, 3, 1, len(pre))
outa = [ta] + e.step()
tb, rb, sb = e.start(B, 3, 1, len(pre))
print(f"A slot {sa} reused {ra}; B slot {sb} reused {rb}, copied {e.last_copied()} (prefix {len(pre)})")
outb = [tb]
while len(outb) < N:
    outb += e.step_batch([sb])[0]
e.release(sa); e.release(sb)
# fresh B: new engine state (no snapshots usable)
e.close()
e = Engine(MODEL)
e.set_sampling(0)
tf, rf, sf = e.start(B, 3)
outf = [tf]
while len(outf) < N:
    outf += e.step()
d = next((i for i in range(N) if outb[i] != outf[i]), N)
print(f"fresh B reused {rf}; greedy agreement copied vs fresh: {d}/{N}")
print("PASS" if rb >= len(pre) - 8 and e is not None and d >= 20 else "FAIL")
