"""How often the model breaks tool-call syntax under different sampling settings, on real
agent contexts taken from the disk cache (prompt = an archived conversation up to its last
assistant turn). Each setting: N samples per context, decoded 8 at a time (batched).
  python3 test_toolcall_sampling.py <archive.qkv> [more.qkv ...]      (env N=16, MAXTOK=700)"""
import os, re, struct, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
N = int(os.environ.get("N", "16"))
MAXTOK = int(os.environ.get("MAXTOK", "700"))
IM_END = tok.token_to_id("<|im_end|>")
ASSIST = tok.encode("<|im_start|>assistant\n", add_special_tokens=False).ids


def archive_tokens(path):
    with open(path, "rb") as f:
        assert f.read(8)[:4] == b"QKVA"
        fl, = struct.unpack("<I", f.read(4)); f.read(fl)
        n, = struct.unpack("<i", f.read(4))
        ns, = struct.unpack("<I", f.read(4)); f.read(4 * ns)
        nz, = struct.unpack("<I", f.read(4)); f.read(8 * nz)
        return list(struct.unpack(f"<{n}i", f.read(4 * n)))


def prompt_of(path):
    ids = archive_tokens(path)
    assert all(t < (1 << 30) for t in ids), "image rows: not supported here"
    for i in range(len(ids) - len(ASSIST), 0, -1):  # last assistant turn start
        if ids[i:i + len(ASSIST)] == ASSIST:
            return ids[:i + len(ASSIST)]
    raise ValueError("no assistant turn")


CALL = re.compile(r"<tool_call>\n<function=[A-Za-z_][\w.-]*>\n(<parameter=[A-Za-z_][\w-]*>\n.*?\n</parameter>\n)+"
                  r"</function>\n</tool_call>", re.S)


def verdict(text):
    """ok: no tool call or every tool call well formed; bad: a malformed one; open: cut by MAXTOK."""
    body = text.split("</think>")[-1]
    if "<tool_call>" not in body:
        return "none"
    rest = body[body.index("<tool_call>"):]
    if "</tool_call>" not in rest:
        return "open" if len(rest) > 200 else "bad"
    first = rest[:rest.index("</tool_call>") + len("</tool_call>")]
    return "ok" if CALL.fullmatch(first) else "bad"


def run(e, prompt, temp, top_p, top_k, k):
    res = {"ok": 0, "bad": 0, "open": 0, "none": 0}
    bad = []
    e.set_concurrent(True)
    for b0 in range(0, N, 8):
        slots, outs = [], []
        for i in range(b0, min(N, b0 + 8)):
            e.set_sampling(temp, top_p, top_k, 0.0, 1000 + i)
            t, _, s = e.start(prompt, k)
            slots.append(s); outs.append([t])
        live = list(range(len(slots)))
        while live:
            res_b = e.step_batch([slots[i] for i in live]) if len(live) > 1 else [e.step()]
            nxt = []
            for i, r in zip(live, res_b):
                outs[i] += r
                txt = tok.decode(outs[i], skip_special_tokens=False)
                done = IM_END in outs[i] or "</tool_call>" in txt or len(outs[i]) >= MAXTOK
                if done:
                    e.release(slots[i])
                else:
                    nxt.append(i)
            if len(nxt) == 1 and len(live) > 1:
                e.resume(slots[nxt[0]])
            live = nxt
        for o in outs:
            txt = tok.decode(o, skip_special_tokens=False)
            v = verdict(txt)
            res[v] += 1
            if v == "bad":
                body = txt.split("</think>")[-1]
                bad.append(body[body.index("<tool_call>"):][:160])
    return res, bad


if __name__ == "__main__":
    e = Engine(MODEL, max_t=262144)
    prompts = [prompt_of(p) for p in sys.argv[1:]]
    print("contexts:", [len(p) for p in prompts], flush=True)
    # batched steps need MTP (k > 0); MTP vs plain sampling itself: test_sampling_deep.py
    settings = [("T=1 top_p=1 top_k=40 (old)", 1.0, 1.0, 40, 3),
                ("T=1 top_p=0.95 top_k=20 (generation_config)", 1.0, 0.95, 20, 3),
                ("T=0.6 top_p=0.95 top_k=20 (Qwen: precise coding)", 0.6, 0.95, 20, 3)]
    for name, t, p, kk, k in settings:
        tot = {"ok": 0, "bad": 0, "open": 0, "none": 0}
        bads = []
        for pr in prompts:
            r, b = run(e, pr, t, p, kk, k)
            for x in tot:
                tot[x] += r[x]
            bads += b
        calls = tot["ok"] + tot["bad"]
        print(f"{name}: {tot}  -> malformed {tot['bad']}/{calls} tool calls", flush=True)
        for b in bads[:3]:
            print("    e.g.", repr(b))
    e.close()
