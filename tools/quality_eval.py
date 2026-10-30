"""Quality of a quantized setup against a BF16 reference: perplexity and KLD on a fixed text set.

Every sequence is L tokens of plain text (no chat template). The first C tokens are the prompt (prefilled), and
the next-token distributions at positions C-1 .. L-2 are scored (C-1 from the prefill, the rest from
teacher-forced decode steps on the engines; the reference runs one forward over the sequence). Outputs per
position: the reference's top-K tokens with their log-probs, and for a candidate its log-probs at those
tokens, its argmax and its log-prob of the actual next token. KLD(ref || cand) is summed over the reference's
top-K plus one "rest" bucket (a tight lower bound for K = 64: the rest holds ~1e-4 of the mass).

  make-set   python tools/quality_eval.py make-set --out SET.npz --prose FILE... --code DIR|FILE... [--n 16 --len 2048]
  ref-hf     python tools/quality_eval.py ref-hf --set SET.npz --model <bf16 dir> --out REF.npz      (transformers >= 5.18)
  engine     python tools/quality_eval.py engine --set SET.npz --ref REF.npz --model <dir> --out X.npz  (env = engine config)
  vllm       python tools/quality_eval.py vllm --set SET.npz --ref REF.npz --url http://host:port --model NAME --out X.npz
  report     python tools/quality_eval.py report --ref REF.npz X.npz [Y.npz ...]
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
K = 64


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


# ---------------------------------------------------------------- text set
def make_set(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer))

    def files(roots, exts):
        out = []
        for r in roots:
            if os.path.isfile(r):
                out.append(r)
                continue
            for dp, dn, fn in os.walk(r):
                dn[:] = sorted(d for d in dn if not d.startswith((".", "__")))
                out += [os.path.join(dp, f) for f in sorted(fn) if f.endswith(exts)]
        return sorted(out)

    seqs, kinds, srcs = [], [], []
    for kind, roots, exts in (("prose", a.prose, (".txt", ".md", ".raw")), ("code", a.code, (".py",))):
        ids = []
        for f in files(roots, exts):
            ids += tok.encode(open(f, encoding="utf-8", errors="ignore").read()).ids
            if len(ids) >= a.n * a.len * 4:
                break
        if len(ids) < a.n * a.len:
            sys.exit(f"{kind}: only {len(ids)} tokens, need {a.n * a.len}")
        step = len(ids) // a.n  # spread the windows over the whole text
        for i in range(a.n):
            seqs.append(ids[i * step:i * step + a.len])
            kinds.append(kind)
            srcs.append(f"{kind}:{i * step}")
    np.savez(a.out, ids=np.array(seqs, np.int32), kinds=np.array(kinds), ctx=np.int32(a.ctx))
    print(f"{len(seqs)} sequences x {a.len} tokens (prompt {a.ctx}, scored {a.len - a.ctx} each) -> {a.out}")


# ---------------------------------------------------------------- reference (HF transformers, BF16)
def dequant_into(model, ckpt):
    """replace every Linear weight that ckpt stores as compressed-tensors pack-quantized (INT4 or INT8,
    symmetric, group 128) by its dequantized bf16 value (q * scale)"""
    import torch
    from safetensors import safe_open
    files = [os.path.join(ckpt, "model.safetensors")] if os.path.exists(os.path.join(ckpt, "model.safetensors")) else         [os.path.join(ckpt, f) for f in sorted(set(json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
                                                     ["weight_map"].values())) if f != "model_mtp.safetensors"]
    params = dict(model.named_parameters())
    n = 0
    for f in files:
        with safe_open(f, "pt") as h:
            for name in h.keys():
                if not name.endswith(".weight_packed"):
                    continue
                base = name[: -len(".weight_packed")]
                if base + ".weight" not in params:
                    continue
                qp, sc = h.get_tensor(name), h.get_tensor(base + ".weight_scale").float()
                N, K = params[base + ".weight"].shape
                if qp.shape[1] * 4 == K:  # INT8: 4 bytes per int32, u = w + 128
                    q = qp.contiguous().view(torch.uint8).to(torch.int16) - 128
                else:  # INT4: 8 nibbles per int32 (LSB first), u = w + 8
                    sh = torch.arange(0, 32, 4, dtype=torch.int32)
                    q = ((qp.unsqueeze(-1) >> sh) & 15).reshape(N, K).to(torch.int16) - 8
                w = (q.float().view(N, K // 128, 128) * sc.unsqueeze(-1)).view(N, K)
                params[base + ".weight"].data.copy_(w.to(torch.bfloat16))
                n += 1
    return n


def ref_hf(a):
    import torch
    from transformers import AutoModelForImageTextToText
    S = np.load(a.set)
    ids, C = S["ids"], int(S["ctx"])
    n, L = ids.shape
    mm = {i: f"{a.gpu_gib}GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "200GiB"
    t0 = time.time()
    if a.dequant_from:  # same model with the quantized checkpoint's layer weights (HF math, quantized weights)
        from accelerate import dispatch_model, infer_auto_device_map
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16)
        n_q = dequant_into(model, a.dequant_from)
        dm = infer_auto_device_map(model, max_memory=mm, no_split_module_classes=model._no_split_modules)
        model = dispatch_model(model, dm)
        print(f"{n_q} layer weights replaced by the dequantized {a.dequant_from}", flush=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
                                                            max_memory=mm)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    P = L - C  # scored positions per sequence
    top_i = np.zeros((n, P, K), np.int32)
    top_l = np.zeros((n, P, K), np.float32)
    act = np.zeros((n, P), np.float32)
    R = dict(np.load(a.ref)) if a.ref else None  # also score this run against another reference
    at_ref = np.zeros((n, P, K), np.float32)
    amax = np.zeros((n, P), np.int32)
    with torch.no_grad():
        for s in range(n):
            t0 = time.time()
            x = torch.tensor(ids[s:s + 1], device="cuda:0")
            out = model(input_ids=x, logits_to_keep=P + 1)
            lg = out.logits[0, :P].float()  # rows C-1 .. L-2
            lp = torch.log_softmax(lg, -1)
            v, i = lp.topk(K, -1)
            top_i[s], top_l[s] = i.cpu().numpy(), v.cpu().numpy()
            tgt = torch.tensor(ids[s, C:], device=lp.device, dtype=torch.long)
            act[s] = lp.gather(1, tgt[:, None])[:, 0].cpu().numpy()
            if R is not None:
                ri = torch.tensor(R["top_i"][s], device=lp.device, dtype=torch.long)
                at_ref[s] = lp.gather(1, ri).cpu().numpy()
                amax[s] = lp.argmax(-1).cpu().numpy()
            del out, lg, lp, v, i
            torch.cuda.empty_cache()
            print(f"seq {s}: ppl {np.exp(-act[s].mean()):.3f} ({time.time() - t0:.1f} s)", flush=True)
    if R is not None:  # usable both as a reference and as a candidate against --ref
        np.savez(a.out, top_i=top_i, top_l=top_l, act=act, at_ref=at_ref, amax=amax, approx=np.int32(0))
    else:
        np.savez(a.out, top_i=top_i, top_l=top_l, act=act)


# ---------------------------------------------------------------- QEngine (C API)
def engine(a):
    S, R = np.load(a.set), dict(np.load(a.ref))  # arrays in memory (an NpzFile re-reads per access)
    ids, C = S["ids"], int(S["ctx"])
    if a.seqs:  # quick check on the first sequences of each kind
        k = S["kinds"]
        sel = np.concatenate([np.where(k == kd)[0][:a.seqs] for kd in ("prose", "code")])
        ids, R = ids[sel], {kk: v[sel] if v.ndim else v for kk, v in R.items()}
    n, L = ids.shape
    P = L - C
    os.environ.pop("QE_DISK_CACHE", None)
    lib = ctypes.CDLL(os.environ.get("QE_LIB") or os.path.join(ROOT, "build", "libqengine.so"))
    lib.qe_create.restype = ctypes.c_void_p
    lib.qe_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    for f in ("qe_reset", "qe_destroy"):
        getattr(lib, f).argtypes = [ctypes.c_void_p]
    lib.qe_start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    lib.qe_decode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.qe_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    e = lib.qe_create(os.path.abspath(a.model).encode(), a.devices.encode(), L + 64, b"")
    V = 248320
    lg = np.zeros(V, np.float32)
    lgp = lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    at_ref = np.zeros((n, P, K), np.float32)
    act = np.zeros((n, P), np.float32)
    amax = np.zeros((n, P), np.int32)
    top_i = R["top_i"]
    scored = np.zeros((n, P), bool)  # positions with a result (--prefill-stride scores a subset)
    for s in range(n):
        t0 = time.time()
        seq = [int(v) for v in ids[s]]
        lib.qe_reset(e)
        if not a.prefill_stride:
            lib.qe_start(e, (ctypes.c_int * C)(*seq[:C]), C, 0)
        for j in range(P):
            if a.prefill_stride:  # prefill path only: a fresh prefill of the prefix, its last row
                if j % a.prefill_stride:
                    continue
                lib.qe_reset(e)
                lib.qe_start(e, (ctypes.c_int * (C + j))(*seq[:C + j]), C + j, 0)
            elif j > 0:
                lib.qe_decode(e, seq[C + j - 1])
            scored[s, j] = True
            lib.qe_logits(e, lgp, 1)
            lp = logsoftmax(lg)
            at_ref[s, j] = lp[top_i[s, j]]
            act[s, j] = lp[seq[C + j]]
            amax[s, j] = int(lg.argmax())
        print(f"seq {s}: ppl {np.exp(-act[s][scored[s]].mean()):.3f} ({time.time() - t0:.1f} s)", flush=True)
    lib.qe_destroy(e)
    np.savez(a.out, at_ref=at_ref, act=act, amax=amax, approx=np.int32(0), scored=scored, sel=sel if a.seqs else np.arange(n))


# ---------------------------------------------------------------- vLLM / OpenAI-compatible server (prompt_logprobs)
def vllm(a):
    import requests
    S, R = np.load(a.set), dict(np.load(a.ref))  # arrays in memory (an NpzFile re-reads per access)
    ids, C = S["ids"], int(S["ctx"])
    n, L = ids.shape
    P = L - C
    at_ref = np.zeros((n, P, K), np.float32)
    act = np.zeros((n, P), np.float32)
    amax = np.zeros((n, P), np.int32)
    for s in range(n):
        t0 = time.time()
        body = {"model": a.model, "prompt": [int(v) for v in ids[s]], "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": a.topk}
        r = requests.post(a.url.rstrip("/") + "/v1/completions", json=body, timeout=600)
        r.raise_for_status()
        js = r.json()
        pl = js.get("prompt_logprobs") or js["choices"][0]["prompt_logprobs"]  # [L] (first None): {token id: {logprob, rank}}
        for j in range(P):
            d = pl[C + j]  # distribution that predicted token C + j (computed at position C + j - 1)
            m = {int(t): v["logprob"] for t, v in d.items()}
            floor = min(m.values())  # tokens outside the returned top-k: at most the k-th log-prob
            at_ref[s, j] = [m.get(int(t), floor) for t in R["top_i"][s, j]]
            act[s, j] = m[int(ids[s, C + j])]
            amax[s, j] = min(m, key=lambda t: (d[str(t)]["rank"] if str(t) in d else 1 << 30))
        print(f"seq {s}: ppl {np.exp(-act[s].mean()):.3f} ({time.time() - t0:.1f} s)", flush=True)
    np.savez(a.out, at_ref=at_ref, act=act, amax=amax, approx=np.int32(a.topk))


# ---------------------------------------------------------------- report
def kld(ref_l, cand_l):
    """KL(ref || cand) over the reference's top-K plus the rest bucket, per position"""
    p = np.exp(ref_l.astype(np.float64))
    q = np.exp(cand_l.astype(np.float64))
    kl = (p * (ref_l - cand_l)).sum(-1)
    pr, qr = np.clip(1 - p.sum(-1), 1e-12, 1), np.clip(1 - q.sum(-1), 1e-12, 1)
    return np.maximum(kl + pr * np.log(pr / qr), 0)


def report(a):
    R = np.load(a.ref)
    S = np.load(a.set) if a.set else None
    kinds = S["kinds"] if S is not None else None
    print(f"{'setup':28s} {'PPL':>7s} {'PPL prose':>9s} {'PPL code':>8s} {'KLD mean':>9s} {'p99':>7s} {'max':>6s} "
          f"{'top1 %':>6s}")
    rows = [("BF16 reference", R["act"], None, None)]
    for f in a.cands:
        X = np.load(f)
        rows.append((os.path.basename(f).replace(".npz", ""), X["act"], kld(R["top_l"], X["at_ref"]),
                     (X["amax"] == R["top_i"][..., 0]).mean() * 100))
    for name, act, kl, t1 in rows:
        ppl = np.exp(-act.mean())
        pp = [np.exp(-act[kinds == k].mean()) if kinds is not None else float("nan") for k in ("prose", "code")]
        line = f"{name:28s} {ppl:7.3f} {pp[0]:9.3f} {pp[1]:8.3f}"
        if kl is not None:
            line += f" {kl.mean():9.5f} {np.percentile(kl, 99):7.4f} {kl.max():6.2f} {t1:6.2f}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-set")
    m.add_argument("--out", required=True)
    m.add_argument("--tokenizer", required=True, help="tokenizer.json")
    m.add_argument("--prose", nargs="+", required=True)
    m.add_argument("--code", nargs="+", required=True)
    m.add_argument("--n", type=int, default=16, help="sequences per kind")
    m.add_argument("--len", type=int, default=2048)
    m.add_argument("--ctx", type=int, default=1024)
    r = sub.add_parser("ref-hf")
    r.add_argument("--set", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gpu-gib", type=int, default=18)
    r.add_argument("--ref", default=None, help="also score against this reference (candidate fields)")
    r.add_argument("--dequant-from", default=None, help="quantized checkpoint whose layer weights replace the BF16 ones")
    e = sub.add_parser("engine")
    e.add_argument("--set", required=True)
    e.add_argument("--ref", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--devices", default="0,1")
    e.add_argument("--seqs", type=int, default=0, help="only the first N sequences of each kind (quick check)")
    e.add_argument("--prefill-stride", type=int, default=0,
                   help="score every N-th position by a fresh prefill of its prefix (prefill path only)")
    v = sub.add_parser("vllm")
    v.add_argument("--set", required=True)
    v.add_argument("--ref", required=True)
    v.add_argument("--url", required=True)
    v.add_argument("--model", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--topk", type=int, default=K)
    p = sub.add_parser("report")
    p.add_argument("--ref", required=True)
    p.add_argument("--set", default=None)
    p.add_argument("cands", nargs="*")
    a = ap.parse_args()
    {"make-set": make_set, "ref-hf": ref_hf, "engine": engine, "vllm": vllm, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
