"""A prompt refused for lack of KV room must leave the running sequence untouched: its next steps
equal those of a run without the refused prompt. (The engine used to switch its current slot to
the refused prompt's slot first, so the caller's next step decoded that other slot.)
usage: python tests/test_admit_fail.py [lib_path]"""
import os, sys
os.environ.update(QE_POOL_TOKENS="65536", QE_POOL_HEADROOM="0", QE_POOL_GROW_FRAC="0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from engine import Engine
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
M = os.path.join(ROOT, "models/Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(M, "tokenizer.json"))
e = Engine(M, "0,1,2", 262144, lib_path=sys.argv[1] if len(sys.argv) > 1 else None)
e.set_concurrent(True)
e.set_sampling(0.0)
src = open(os.path.join(ROOT, "tests/data/cache_text.txt")).read()
chat = lambda s: tok.encode(f"<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
A = chat(src[20000:38000] + "\nIgnore the code above. Count from 1 to 300, numbers separated by single spaces.")
B = chat(src[:20000] + "\nSummarize the code above.")
big = chat((src * 60)[:600000] + "\nSummarize.")[:62000]  # fits the pool alone, not next to A
print(f"pool {e.pool_tokens} tokens; A {len(A)}, idle B {len(B)}, refused prompt {len(big)}")


def run(refuse):
    e.reset()
    sb = e.start(B)[2]
    for _ in range(5):
        e.step()
    e.release(sb)
    first, _, sa = e.start(A)
    out = [first]
    while len(out) < 60:
        out += e.step()
    refused = None
    if refuse:
        try:
            e.start_begin(big)
            refused = False
        except RuntimeError:
            refused = True
    while len(out) < 400:
        out += e.step()
    e.release(sa)
    return out[:400], refused


ref, _ = run(False)
got, refused = run(True)
same = got == ref
d = next((i for i in range(400) if got[i] != ref[i]), None)
print(f"refused: {refused}; continuation {'identical' if same else f'differs at token {d}'}")
print(repr(tok.decode(got[55:75])))
ok = same and refused

# 2) the only room is held by an idle slot sharing a prefix with the prompt (a copy source, e.g.
# every idle Claude Code agent shares the system prompt and tools): evict it instead of refusing
e.reset()
sb = e.start(B)[2]
e.step()
e.release(sb)
first, _, sa = e.start(A)
out = [first]
while len(out) < 60:
    out += e.step()
shared = B[:-8] + chat((src * 60)[:400000])[:45100]  # B's text (minus its question) as a prefix
try:
    e.start_begin(shared)
    while not e.prefill_more(1 << 30):
        pass
    t2, reused2, s2 = e.start_finish()
    print(f"prompt sharing the idle slot's prefix ({len(shared)} tokens): admitted, reused {reused2}")
    adm = True
except RuntimeError as ex:
    print(f"prompt sharing the idle slot's prefix ({len(shared)} tokens): refused ({ex})")
    adm = False
e.resume(sa)
while len(out) < 400:
    out += e.step()
same2 = out[:400] == ref
print(f"running sequence after it: {'identical' if same2 else 'DIFFERS'}")
ok &= adm and same2
print("PASS" if ok else "FAIL")
