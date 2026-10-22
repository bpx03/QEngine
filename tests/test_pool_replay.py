"""Replay a real agent context from the disk cache under KV-pool pressure. Greedy output of every
copy decoded concurrently (regions growing / moving / trimmed between steps) must equal the
reference decoded alone in an unpressured pool.
  python3 test_pool_replay.py ref <archive.qkv> <prompt_len> <ntok> <out.json>
  QE_POOL_TOKENS=... QE_POOL_HEADROOM=256 QE_POOL_GROW_FRAC=0 \\
  python3 test_pool_replay.py pressure <archive.qkv> <prompt_len> <ntok> <ref.json> [copies]"""
import json, os, struct, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")


def archive_tokens(path):
    with open(path, "rb") as f:
        assert f.read(8)[:4] == b"QKVA"
        fl, = struct.unpack("<I", f.read(4)); f.read(fl)
        n, = struct.unpack("<i", f.read(4))
        ns, = struct.unpack("<I", f.read(4)); f.read(4 * ns)
        nz, = struct.unpack("<I", f.read(4)); f.read(8 * nz)
        return list(struct.unpack(f"<{n}i", f.read(4 * n)))


mode, arch, plen, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompt = archive_tokens(arch)[:plen]
assert all(t < (1 << 30) for t in prompt)
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
print("pool", e.pool_tokens, "prompt", len(prompt), flush=True)
if mode == "ref":
    t, _, s = e.start(prompt, 3)
    out = [t]
    while len(out) < ntok:
        out += e.step()
    json.dump(out[:ntok], open(sys.argv[5], "w"))
    print("reference saved", flush=True)
else:
    ref = json.load(open(sys.argv[5]))
    copies = int(sys.argv[6]) if len(sys.argv) > 6 else 6
    e.set_concurrent(True)
    slots, outs = [], []
    t0 = time.time()
    for c in range(copies):
        t, _, s = e.start(prompt, 3)
        slots.append(s); outs.append([t])
    live = list(range(copies))
    while live:
        res = e.step_batch([slots[i] for i in live]) if len(live) > 1 else [e.step()]
        nxt = []
        for i, r in zip(live, res):
            outs[i] += r
            if len(outs[i]) >= ntok:
                e.release(slots[i])
            else:
                nxt.append(i)
        if len(nxt) == 1 and len(live) > 1:
            e.resume(slots[nxt[0]])
        live = nxt
    ok = True
    for c, o in enumerate(outs):
        d = next((j for j in range(ntok) if o[j] != ref[j]), ntok)
        print(f"copy {c}: identical to the reference for {d}/{ntok} tokens", flush=True)
        ok &= d == ntok
    print(f"{time.time() - t0:.0f}s;", "PASS" if ok else "FAIL", flush=True)
    if os.environ.get("DUMP"):
        json.dump(outs, open(os.environ["DUMP"], "w"))
e.close()
