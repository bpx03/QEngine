"""Forks of a live conversation (Claude Code's subagent-progress summary: the whole history plus
a short question, right after the reply). The engine snapshots the live end before a turn
extends it, so (1) a fork that takes the idle slot and (2) the conversation's own next turn
afterwards both resume from prompt + reply (nothing recomputed); (3) a fork arriving while the
next turn is already being prefilled copies the rows and that snapshot from the active slot.
Greedy outputs must equal fresh computations. usage: python3 tests/test_fork.py"""
import os, sys
os.environ.setdefault("QE_MTP_ADAPT", "0")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.environ.get("QE_TEST_MODEL") or os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")  # QE_TEST_MODEL: another model (e.g. Flash-Next with QE_LIB)
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
A = enc(f"<|im_start|>user\nExplain this code:\n{text[:9000]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
e.set_concurrent(True)
N = 40


def gen(ids, n=N):
    t, r, s = e.start(ids, 3)
    out = [t]
    while len(out) < n:
        out += e.step()
    e.release(s)
    return out, r, s


reply, _, sA = gen(A)
base = A + reply
fork = base + enc("<|im_end|>\n<|im_start|>user\nIn one short sentence: what has been done so far?<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
nxt = base + enc("<|im_end|>\n<|im_start|>user\nNow list every function name.<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")
print(f"A {len(A)} + reply {len(reply)} = {len(base)}; fork {len(fork)}, next turn {len(nxt)}", flush=True)
ok = True
# 1. fork takes the idle slot: live extension
f_out, rf, sf = gen(fork, 12)
print(f"fork: slot {sf} (A's {sA}), reused {rf} (live end {len(base)})")
ok &= rf >= len(base) - 1  # the live end: the last generated token is still pending
# 2. the conversation's own next turn after the fork: the live end snapshot, not the prompt end
n_out, rn, sn = gen(nxt)
print(f"next turn after the fork: slot {sn}, reused {rn} (live end {len(base)}, old prompt end {len(A)}), "
      f"copied {e.last_copied()}")
ok &= rn >= len(base) - 1
ONE_PF = os.environ.get("QE_TEST_ONE_PREFILL") == "1"  # engines with one prompt in flight (Flash-Next): no part 3
if not ONE_PF:
    # 3. fork while the next turn is being prefilled (A's slot active): rows + snapshot copied
    e.reset()
    reply2, _, sA2 = gen(A)
    assert reply2 == reply
    sx = e.start_begin(nxt, 3)  # the next turn starts (live extension, snapshot of the live end taken)
    e.prefill_more(8, sx)  # a first chunk of the next turn is in
    f2_out, rf2, sf2 = gen(fork, 12)
    print(f"fork during the next turn's prefill: slot {sf2} (active slot {sx}), reused {rf2}, copied {e.last_copied()}")
    ok &= rf2 >= len(base) - 1 and sf2 != sx
    while not e.prefill_more(1 << 30, sx):
        pass
    tx, rx, _ = e.start_finish(sx)
    x_out = [tx]
    while len(x_out) < N:
        e.resume(sx)
        x_out += e.step()
    e.release(sx)

# references: the same turns on the live state with no fork in between (the reply rows come
# from the same decode steps; a from-scratch prefill of the reply differs by the INT8-prefill /
# decode precision and may flip a near-tie)
e.reset()
reply3, _, _ = gen(A)
assert reply3 == reply
ref_n, rn_live, _ = gen(nxt)
e.reset()
gen(A)
ref_f, rf_live, _ = gen(fork, 12)
d1 = next((j for j in range(12) if f_out[j] != ref_f[j]), 12)
d2 = next((j for j in range(N) if n_out[j] != ref_n[j]), N)
d3 = 12 if ONE_PF else next((j for j in range(12) if f2_out[j] != ref_f[j]), 12)
d4 = N if ONE_PF else next((j for j in range(N) if x_out[j] != ref_n[j]), N)
print(f"identical to the live-state turns: fork {d1}/12, next turn after the fork {d2}/{N}, fork during prefill "
      f"{d3}/12, interrupted next turn {d4}/{N}")
ok &= d1 == 12 and d2 == N and d3 == 12 and d4 == N
print("PASS" if ok else "FAIL")
e.close()
