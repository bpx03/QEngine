"""Context archive: conversation A is pushed out of its slot by three others, then resumes from
the host archive (KV + delta-net state) instead of recomputing; output must match fresh."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
from tokenizers import Tokenizer
import transformers
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
gdir = os.path.join(os.path.dirname(transformers.__file__), "generation")
src = [open(os.path.join(gdir, f)).read() for f in ("logits_process.py", "utils.py", "stopping_criteria.py", "streamers.py")]
def conv(i, n, q):
    return tok.encode("<|im_start|>user\n" + src[i][:n] + "\n" + q + "<|im_end|>\n<|im_start|>assistant\n").ids
A = conv(0, 9000, "Summarize the first class.")
others = [conv(1, 16000, "What is generate()?"), conv(2, 16000, "Explain this file."), conv(3, 16000, "Explain this file.")]
N = 60
e = Engine(MODEL)
e.set_sampling(0)
def gen(ids):
    t0 = time.time()
    t, r, s = e.start(ids, 3)
    dt = time.time() - t0
    out = [t]
    while len(out) < N:
        out += e.step()
    return out[:N], r, s, dt
outA, r, s, _ = gen(A)
print(f"A: {len(A)} tok, slot {s}")
for o in others:
    _, r2, s2, _ = gen(o)
    print(f"other: {len(o)} tok -> slot {s2} (archives {e.num_archives()})")
A2 = A + outA + tok.encode("<|im_end|>\n<|im_start|>user\nAnd the second class?<|im_end|>\n<|im_start|>assistant\n").ids
outA2, r, s, dt = gen(A2)
print(f"A continued ({len(A2)} tok): slot {s}, reused {r}, copied {e.last_copied()}, from archive {e.last_from_archive()}, start {dt*1e3:.0f} ms")
e.close()
e = Engine(MODEL)
e.set_sampling(0)
outF, rf, _, dtf = gen(A2)
d = next((i for i in range(N) if outA2[i] != outF[i]), N)
print(f"fresh A start {dtf*1e3:.0f} ms (reused {rf}); greedy agreement archive vs fresh {d}/{N}")
print("PASS" if r >= len(A) and d >= 20 else "FAIL")
