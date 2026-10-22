"""Reply limit (max_tokens) with MTP: an MTP step can accept drafts past the client's max_tokens;
the server cut the reply, but the slot kept the extra tokens, so the next turn (history + the cut
reply + a new message) missed the slot's live end and resumed from the prompt-end checkpoint
(n - 1), recomputing the whole reply. With set_limit the steps
verify only the drafts that fit: the reply ends exactly at max_tokens and the next turn resumes
from the live end. Checks, single-sequence steps and batched steps (3 sequences):
 - no step emits past the limit (the tokens are never cut),
 - greedy output = the first N tokens of the same prompt without a limit,
 - the next turn reuses >= prompt + N - 1 tokens (the last token is pending, not yet in the slot),
 - control without a limit: some N overshoot by >= 2 tokens and lose the live end (the bug path
   is exercised),
 - stop sequences (trim): the slot cut inside / at the start of its last step (single and batched,
   no step queued ahead) resumes the next turn from the cut, and that turn decodes as from scratch.
usage: python3 tests/test_reply_limit.py"""
import os, random, sys

from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine  # noqa: E402

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
W = "cache state token layer prefix kernel memory batch window queue model vector index chapter".split()
K = 7  # the server's --mtp-k


def prompt(seed, n=2500):
    r = random.Random(seed)
    doc = " ".join(r.choice(W) for _ in range(n))
    return tok.encode(f"<|im_start|>system\nYou are assistant {seed}.<|im_end|>\n<|im_start|>user\n"
                      f"<document>\n{doc}\n</document>\nDescribe the document in detail.<|im_end|>\n"
                      f"<|im_start|>assistant\n<think>\n\n</think>\n\n").ids


NEXT = tok.encode("<|im_end|>\n<|im_start|>user\nOne more sentence.<|im_end|>\n<|im_start|>assistant\n",
                  add_special_tokens=False).ids
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
e.set_concurrent(True)
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)


def next_turn(ids, reply):
    """Reused prompt tokens of the conversation's next turn."""
    _, r, s = e.start(ids + reply + NEXT, K)
    e.release(s)
    return r


def single(ids, n, limit):
    t, _, s = e.start(ids, K)
    if limit:
        e.set_limit(s, n - 1)
    out, over = [t], False
    while len(out) < n:
        out += e.step()
        over |= len(out) > n
    e.release(s)
    return out, over


print("1. single-sequence steps", flush=True)
lost = 0
for i, n in enumerate((33, 47, 61, 77, 95, 120)):
    ids = prompt(100 + i)
    e.reset()
    ctl, _ = single(ids, n, False)
    r_ctl = next_turn(ids, ctl[:n])
    e.reset()
    out, over = single(ids, n, True)
    r = next_turn(ids, out)
    lost += r_ctl < len(ids) + n - 1
    print(f"  N {n}: prompt {len(ids)}; no limit: {len(ctl)} tokens emitted, next turn reused {r_ctl}; "
          f"limit: {len(out)} emitted, next turn reused {r}", flush=True)
    check(not over and len(out) == n, f"N {n}: exactly N tokens emitted")
    check(out == ctl[:n], f"N {n}: greedy output = the first N tokens without a limit")
    check(r >= len(ids) + n - 1, f"N {n}: next turn resumes from the live end ({r} >= {len(ids) + n - 1})")
check(lost > 0, f"control: {lost} of 6 replies without a limit lost the live end (bug path exercised)")

print("2. batched steps (3 sequences)", flush=True)
e.reset()
lims = (41, 66, 87)
convs = [prompt(200 + i) for i in range(3)]
live = []
for ids, n in zip(convs, lims):
    t, _, s = e.start(ids, K)
    e.set_limit(s, n - 1)
    live.append([ids, n, [t], s])
over = False
run = list(live)
while run:
    for c, toks in zip(run, e.step_batch([c[3] for c in run])):
        c[2] += toks
        over |= len(c[2]) > c[1]
    for c in [c for c in run if len(c[2]) >= c[1]]:
        e.release(c[3])
        run.remove(c)
check(not over, "no batched step emitted past its sequence's limit")
for ids, n, out, s in live:
    r = next_turn(ids, out[:n])
    print(f"  N {n}: prompt {len(ids)}, slot {s}, next turn reused {r}", flush=True)
    check(r >= len(ids) + n - 1, f"N {n}: next turn resumes from the live end ({r} >= {len(ids) + n - 1})")


def gen(ids, n=16):
    t, r, s = e.start(ids, K)
    out = [t]
    while len(out) < n:
        out += e.step()
    e.release(s)
    return out[:n], r


def cut_reply(slots, min_len, want_rows):
    """Decode the slots (stop watch on, no step queued ahead) until each has >= min_len tokens and
    its last step committed >= want_rows rows; returns per slot (reply, tokens before that step,
    rows of that step)."""
    st = {s: [o, 0, 0] for s, o in slots}
    for s in st:
        e.watch_stops(s)
        e.set_no_ahead(s, True)
    run = list(st)
    while run:
        res = [e.step()] if len(run) == 1 else e.step_batch(run)
        for s, toks in zip(list(run), res):
            o = st[s][0]
            if not toks:
                continue
            st[s][1], st[s][2] = len(o), len(toks)
            o += toks
            if len(o) >= min_len and len(toks) >= want_rows:
                run.remove(s)
                if len(run) == 1:
                    e.resume(run[0])
    return st


print("3. stop sequences: undo of the last step's tokens past the cut (trim)", flush=True)
for label, nseq, pick in (("single, cut before the step's pending token (r = 0)", 1, lambda L0, c: L0 - 1),
                          ("single, cut inside the step (r = 2)", 1, lambda L0, c: L0 + 1),
                          ("batched, cut inside the step (r = 1)", 2, lambda L0, c: L0)):
    e.reset()
    convs = [prompt(300 + i) for i in range(nseq)]
    slots = []
    for ids in convs:
        t, _, s = e.start(ids, K)
        slots.append((s, [t]))
    st = cut_reply(slots, 40, 3)
    turns = []
    for ids, (s, _) in zip(convs, slots):
        out, L0, c = st[s]
        keep = pick(L0, c)  # reply tokens kept: the slot ends at len(ids) + keep
        done = e.trim(s, len(ids) + keep)
        e.release(s)
        check(done, f"{label}: trim to reply {keep} of {len(out)} (last step {c} rows from {L0})")
        turns.append(ids + out[:keep] + NEXT)
    live = [gen(t) for t in turns]
    e.reset()
    ref = [gen(t)[0] for t in turns]
    for (o, r), rf, t in zip(live, ref, turns):
        need = len(t) - len(NEXT)
        print(f"  next turn reused {r} of {len(t)} (cut reply end {need})", flush=True)
        check(r >= need, f"{label}: next turn resumes from the cut reply's end ({r} >= {need})")
        check(o == rf, f"{label}: next turn's greedy output = from scratch")

e.close()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
