"""QE_DEC_GEMM consistency: the same sequences decoded (greedy) in a batch of 32 (64 rows per pass) and in a
batch of 17 (34 rows) - both through the W4A8 GEMM path - must be bit-identical when the GEMM plan is pinned
(QE_SK_MAXPARTS=1: no stream-K split of K, QE_GEMM_FORCE=1,1: one 64-token tile), since then a row's
arithmetic does not depend on the other rows. Identical -> the path is correct and batch != alone (test_batch)
is the W4A8 vs H16 numerics; an early divergence -> a bug in the path.
usage: QE_MAX_SEQS=32 QE_SK_MAXPARTS=1 QE_GEMM_FORCE=1,1 python3 tests/check_decgemm_rows.py [tokens] [big] [small]"""
import os, sys
os.environ.setdefault("QE_MTP_ADAPT", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
from transformers import AutoTokenizer

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BIG = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SMALL = int(sys.argv[3]) if len(sys.argv) > 3 else 17
tok = AutoTokenizer.from_pretrained(MODEL)
base = [
    "Write a Python function that parses an INI file into a nested dict, with tests.",
    "Explain in detail how a CPU branch predictor works, including TAGE.",
    "Napisz krótkie opowiadanie o latarniku na Bałtyku.",
    "List 30 countries with their capitals as a JSON array of objects.",
    "Implement a thread-safe LRU cache in C++17 with a short explanation.",
    "Describe the causes and consequences of the 1929 stock market crash.",
    "Write a bash script that backs up a directory with rotation of 7 copies.",
    "Explain how TCP congestion control works (Reno, CUBIC, BBR).",
    "Write a SQL schema for a library system and three example queries with joins.",
    "Wyjaśnij, jak działa szyfrowanie asymetryczne i podpis cyfrowy, z przykładem.",
    "Implement binary search in Rust with unit tests and explain the edge cases.",
    "Summarize the plot of Hamlet act by act in a numbered list.",
]
prompts = [base[i % 12] + (f" (variant {i // 12})" if i >= 12 else "") for i in range(BIG)]
ids = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                               enable_thinking=False) for p in prompts]
e = Engine(MODEL)
e.set_sampling(0)
e.set_concurrent(True)


def batch(xs):
    outs, slots = [], []
    for x in xs:
        t, _, s = e.start(x, 6)
        outs.append([t])
        slots.append(s)
    while min(len(o) for o in outs) < N:
        for o, r in zip(outs, e.step_batch(slots)):
            o += r
    for s in slots:
        e.release(s)
    return [o[:N] for o in outs]


big = batch(ids)[:SMALL]
small = batch(ids[:SMALL])
same = 0
for i, (a, b) in enumerate(zip(big, small)):
    d = next((j for j in range(N) if a[j] != b[j]), N)
    same += d == N
    print(f"seq {i}: match {d}/{N}" + ("" if d == N else f"  {BIG}: {tok.decode(a[max(0, d - 3):d + 6])!r} | {SMALL}: {tok.decode(b[max(0, d - 3):d + 6])!r}"))
print(f"{same}/{SMALL} identical")
print("PASS" if same == SMALL else "FAIL")
e.close()
