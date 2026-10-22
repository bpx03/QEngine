"""B1 repro (archive-hold-fix): a conversation restored from the RAM archive while making room
for it evicts other slots, whose archiving needs RAM: the archive being restored is the least
recently used one, so before the fix ram_make_room dropped it between start_begin's choice and
the copy, its buffers went to the evicted slot's new archive, and the restore loaded another
conversation's KV / delta-net state.

Setup: K conversations (common system prompt = pinned prefix hint, a private secret code, ~L
tokens of private filler), a pool holding ~2 of them and an archive holding ~1.5. Rounds of
round-robin turns: every turn restores its conversation from the archive while evicting (and
archiving) another. Each turn asks for the conversation's code; the greedy reply must equal a
from-scratch computation of the same prompt (engine reset) and contain its own code and no
other conversation's code.
Before the fix (AH_SYS=distinct): every restore answered with another conversation's code.
With a common system prompt (default) a new conversation copies the system rows from another
slot (OTHER); before the fix ensure_cap could evict that source slot and the copy aborted
(CHECK in copy_region_rows).
usage: python3 tests/test_archive_hold.py [rounds]
  env: QE_SHARE_PREFIX=0/1, AH_SYS=distinct, QE_LIB=path/libqengine.so, AH_CONVS, AH_TOKENS"""
import os, sys, random
os.environ.setdefault("QE_POOL_TOKENS", "40000")
os.environ.setdefault("QE_MAX_SEQS", "4")
os.environ.setdefault("QE_POOL_HEADROOM", "256")
os.environ.setdefault("QE_POOL_GROW_FRAC", "0")
os.environ.setdefault("QE_MTP_ADAPT", "0")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
K = int(os.environ.get("AH_CONVS", "4"))
L = int(os.environ.get("AH_TOKENS", "13000"))
N = 24
W = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
     "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
NAMES = ["MARIGOLD", "OBSIDIAN", "PELICAN", "QUASAR", "RHUBARB", "SAFFRON", "TUNDRA", "UMBRELLA", "VORTEX", "WALNUT"]
CODES = [f"{NAMES[i]}-{random.Random(i).randint(1000, 9999)}" for i in range(K)]
SYSTEM = "You are a careful assistant. Reference notes:\n" + text[:6000]


def filler(seed, n):
    r = random.Random(seed)
    return " ".join(r.choice(W) for _ in range(n))


def enc(s):
    return tok.encode(s, add_special_tokens=False).ids


def turn(user):
    return enc(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


# AH_SYS=distinct: a system prompt per conversation (no prefix rows copied between slots)
DISTINCT = os.environ.get("AH_SYS") == "distinct"
sys_of = [enc(f"<|im_start|>system\n{f'Agent {i}. ' if DISTINCT else ''}{SYSTEM}<|im_end|>\n") for i in range(K)]
sys_ids = sys_of[0]
convs = []
for i in range(K):
    sys_ids = sys_of[i]
    first = (f"Remember this: the secret code of this conversation is {CODES[i]}.\nLog:\n" + filler(100 + i, L // 2)
             + f"\nWhat is the secret code of this conversation? Answer with the code only.")
    convs.append(sys_ids + turn(first))

# the archive holds ~1.5 conversations (KV ~65 KB/token of the own rows + up to 4 checkpoint
# states of ~0.17 GB): with K=4 and a pool of 3, the evicted one; its restore then evicts
# another one, whose archive has to push out the only (least recently used) archive: the
# restored one
share = os.environ.get("QE_SHARE_PREFIX", "1") != "0"
own = len(convs[0]) - (len(sys_ids) if share else 0)
os.environ.setdefault("QE_KV_ARCHIVE_GB", f"{float(os.environ.get('AH_ARCHIVE_CONVS', '1.5')) * (65e3 * own + 4 * 0.17e9) / 2**30:.2f}")
lib = os.environ.get("QE_LIB")
e = Engine(MODEL, max_t=65536, lib_path=lib)
e.set_sampling(0)
print(f"lib {lib or 'build'}; pool {e.pool_tokens} tokens, archive {os.environ['QE_KV_ARCHIVE_GB']} GB, "
      f"share_prefix {os.environ.get('QE_SHARE_PREFIX', 'default')}, {'distinct' if DISTINCT else 'common'} system, {K} conversations of ~{len(convs[0])} tokens, "
      f"system part {len(sys_ids)}", flush=True)


IM_END = tok.token_to_id("<|im_end|>")


def gen(ids, aff):
    t, reused, s = e.start(ids, 3, aff, len(sys_of[aff - 0x100]))
    arch = e.last_from_archive()
    out = [t]
    while len(out) < N:
        out += e.step()
    e.release(s)
    out = out[:N]
    if IM_END in out:
        out = out[:out.index(IM_END) + 1]
    return out, reused, s, arch


log = []  # (conversation, prompt, output, reused, from archive)
for rnd in range(ROUNDS):
    for i in range(K):
        out, reused, s, arch = gen(convs[i], 0x100 + i)
        info = e.pool_info()["archives"]
        reply = tok.decode(out)
        print(f"round {rnd} conv {i}: {len(convs[i])} tok slot {s} reused {reused} archive {arch} "
              f"(archives {info['ram']}, {info['ram_bytes'] / 1e9:.2f} GB) -> {reply[:40]!r}", flush=True)
        log.append((i, list(convs[i]), out, reused, arch))
        # next turn: the reply (as generated) and the same question again
        convs[i] = convs[i] + out + enc(("" if out[-1] == IM_END else "<|im_end|>") + "\n") + turn(
            f"Round {rnd + 1}. " + filler(1000 * rnd + i, 40) + "\nRepeat the secret code of this conversation, code only.")

print("reference: every prompt computed from scratch", flush=True)
bad = mism = restored = 0
for i, ids, out, reused, arch in log:
    e.reset()
    ref, _, _, _ = gen(ids, 0x100 + i)
    d = next((j for j in range(min(len(out), len(ref))) if out[j] != ref[j]), N if out == ref else min(len(out), len(ref)))
    reply = tok.decode(out)
    foreign = [c for j, c in enumerate(CODES) if j != i and c.split("-")[0] in reply]
    own = CODES[i] in reply
    restored += arch
    if d < N:
        mism += 1
    if foreign or (not own and CODES[i] in tok.decode(ref)):
        bad += 1
    if d < N or foreign:
        print(f"  conv {i} ({len(ids)} tok, reused {reused}, archive {arch}): agreement {d}/{N}, own code {own}, "
              f"foreign {foreign}: {reply[:60]!r} vs ref {tok.decode(ref)[:60]!r}", flush=True)
print(f"{len(log)} turns, {restored} from the archive: {mism} differ from scratch, {bad} with a wrong/foreign code")
e.close()
print("PASS" if mism == 0 and bad == 0 and restored > 0 else "FAIL")
