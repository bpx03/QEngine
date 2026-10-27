"""Decode probe: tokens accepted per step and decode speed, QEngine vs an OpenAI-compatible vLLM server, on the
same token ids (greedy, thinking off).

  make    build the prompt set (club-3090 bench prompts, its filler haystack at 10K / 90K, and this repo's source
          code as a realistic long context at 10K / 90K) -> prompts.json
  qe      run QEngine in-process (server/engine.py): per step the number of tokens it returned and its wall time
          (QE_LOOKUP=0 for MTP alone); --profile N: after the prefill of each prompt, N decode steps inside
          cudaProfilerStart/Stop (run under nsys --capture-range=cudaProfilerApi)
  vllm    the same ids against /v1/completions; mean accepted length from the server's spec-decode counters
  report  table of every results file

Usage (box): python3 tools/decode_probe.py make --tokenizer models/Qwen3.8-27B-INT8 --out p.json
             python3 tools/decode_probe.py qe --model models/Qwen3.8-27B-INT8 --prompts p.json --out r_qe.jsonl
             python3 tools/decode_probe.py vllm --url http://127.0.0.1:8110 --model qwen3.8-27b --prompts p.json ...
             python3 tools/decode_probe.py report r_*.jsonl"""
import argparse
import glob
import itertools
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NARR = "Write a detailed 800-word essay explaining transformer attention."
CODE = "Write a Python implementation of quicksort with comments explaining each step."
FILLER = ("club3090 prompt processing calibration filler with stable token shape. "
          "This sentence is intentionally plain so tokenizer variance stays modest. ")
# multi-prompt sets that stop at the end of the answer (no tokens past EOS)
CODE_SET = [CODE,
            "Implement a thread-safe LRU cache in Python with get, put and delete, type hints and docstrings, "
            "followed by unit tests using unittest.",
            "Write a complete Rust command-line program that reads a CSV file, computes per-column statistics "
            "(count, mean, min, max, standard deviation) and prints them as a table, with error handling.",
            "Write a TypeScript React component for a sortable, filterable data table with pagination, then "
            "explain its design."]
PROSE_SET = [NARR,
             "Explain how a modern CPU executes instructions: pipelining, branch prediction, out-of-order "
             "execution and caches, in about 800 words.",
             "Write a short story of about 700 words about a lighthouse keeper who finds a message in a bottle.",
             "Describe the history of the printing press and its effects on European society, in about 800 words."]
SRC_Q = ("\n\nUsing the source files above, explain in detail how one decoding step with speculative drafts "
         "works, function by function, citing the functions involved.")


def chat(tok, text):
    return tok.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      add_special_tokens=False).ids


def make(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    out = {"narr": {"ids": chat(tok, NARR), "gen": a.gen}, "code": {"ids": chat(tok, CODE), "gen": a.gen}}
    eos = [tok.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")]
    for i, q in enumerate(CODE_SET):
        out[f"code{i}"] = {"ids": chat(tok, q), "gen": a.gen_stop, "eos": eos}
    for i, q in enumerate(PROSE_SET):
        out[f"prose{i}"] = {"ids": chat(tok, q), "gen": a.gen_stop, "eos": eos}
    head = "Read the following calibration text. Reply with one concise sentence summarizing its purpose.\n\n"
    per = len(tok.encode(FILLER, add_special_tokens=False).ids)
    files = sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.c*"), recursive=True) +
                   glob.glob(os.path.join(ROOT, "src", "**", "*.h"), recursive=True) +
                   glob.glob(os.path.join(ROOT, "server", "*.py")))
    src = "".join(f"=== {os.path.relpath(f, ROOT)}\n{open(f, encoding='utf-8', errors='replace').read()}\n"
                  for f in files)
    src_ids = tok.encode(src, add_special_tokens=False).ids
    for d, name in ((10000, "10k"), (90000, "90k")):
        out["filler" + name] = {"ids": chat(tok, head + FILLER * (d // per)), "gen": a.gen_depth}
        body = tok.decode(src_ids[:d])
        out["src" + name] = {"ids": chat(tok, body + SRC_Q), "gen": a.gen_depth}
    for k, v in out.items():
        print(f"{k:10s} {len(v['ids']):6d} prompt tokens, {v['gen']} generated")
    json.dump(out, open(a.out, "w"))


def load_prompts(a):
    p = json.load(open(a.prompts))
    names = a.only.split(",") if a.only else list(p)
    return [(n, p[n]) for n in names]


def run_qe(a):
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from engine import Engine
    e = Engine(a.model, devices=a.devices)
    e.set_sampling(0)
    e.warmup(greedy=True, sampling=False)
    tag = a.tag or ("qe-lookup" if os.environ.get("QE_LOOKUP", "1") != "0" else "qe-mtp")
    prof = None
    if a.profile:
        import torch
        prof = torch.cuda.profiler
    with open(a.out, "a") as f:
        ks = [int(v) for v in a.k.split(",")]
        for (name, p), k, rep in itertools.product(load_prompts(a), ks, range(a.reps)):
            ktag = tag if len(ks) == 1 else f"{tag}-k{k}"
            t0 = time.perf_counter()
            tok, _, slot = e.start(p["ids"], k)
            ttft = time.perf_counter() - t0
            if prof:
                e.step()
                prof.start()
                for _ in range(a.profile):
                    e.step()
                prof.stop()
                e.release(slot)
                print(f"{name}: profiled {a.profile} steps", flush=True)
                continue
            sizes, times, out = [], [], [tok]
            eos = set(p.get("eos", []))
            while len(out) < p["gen"] + 1 and not (eos & set(out)):
                s0 = time.perf_counter()
                r = e.step()
                times.append(time.perf_counter() - s0)
                if eos & set(r):  # count the answer up to and including its end
                    r = r[:min(r.index(t) for t in eos if t in r) + 1]
                sizes.append(len(r))
                out += r
            e.release(slot)
            n = sum(sizes)
            rec = {"engine": ktag, "prompt": name, "rep": rep, "prompt_tokens": len(p["ids"]), "ttft": ttft,
                   "steps": len(sizes), "tokens": n, "tok_per_step": n / len(sizes),
                   "ms_per_step": 1e3 * sum(times) / len(times), "decode_tps": n / sum(times),
                   "hist": [sizes.count(i) for i in range(1, max(sizes) + 1)], "text_ids": out[:64]}
            med_t = sorted(times)[len(times) // 2]
            slow = [t - med_t for t in times if t > med_t + 2e-3]  # stalls (graph captures, host waits)
            rec["slow_steps"], rec["slow_ms"] = len(slow), 1e3 * sum(slow)
            rec["slow_at"] = [(i, round(1e3 * t, 1)) for i, t in enumerate(times) if t > med_t + 2e-3][:12]
            f.write(json.dumps(rec) + "\n")
            print(f"{ktag} {name} rep {rep}: {rec['tok_per_step']:.2f} tok/step, {rec['ms_per_step']:.1f} ms/step, "
                  f"{rec['decode_tps']:.1f} tok/s, ttft {ttft:.2f} s, {rec['slow_steps']} slow steps "
                  f"({rec['slow_ms']:.0f} ms) at {rec['slow_at']}", flush=True)


def spec_counters(url):
    import urllib.request
    txt = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    c = {}
    for line in txt.splitlines():
        m = re.match(r"^(vllm:spec_decode_[a-z_]+?)(_total)?(\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if m:
            c[m.group(1)] = c.get(m.group(1), 0.0) + float(m.group(4))
    return c


def run_vllm(a):
    import urllib.request
    with open(a.out, "a") as f:
        for name, p in load_prompts(a):
            for rep in range(a.reps):
                c0 = spec_counters(a.url)
                body = {"model": a.model, "prompt": p["ids"], "max_tokens": p["gen"], "temperature": 0,
                        "ignore_eos": not p.get("eos"), "stream": True, "stream_options": {"include_usage": True}}
                req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                t0 = time.perf_counter()
                first, n = None, 0
                with urllib.request.urlopen(req, timeout=3600) as r:
                    for raw in r:
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        d = json.loads(line[5:])
                        if d.get("choices") and d["choices"][0].get("text") and first is None:
                            first = time.perf_counter()
                        if d.get("usage"):
                            n = d["usage"]["completion_tokens"]
                t1 = time.perf_counter()
                c1 = spec_counters(a.url)
                dc = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
                drafts = dc.get("vllm:spec_decode_num_drafts", 0)
                acc = dc.get("vllm:spec_decode_num_accepted_tokens", 0)
                rec = {"engine": a.tag, "prompt": name, "rep": rep, "prompt_tokens": len(p["ids"]),
                       "ttft": first - t0, "tokens": n, "decode_tps": (n - 1) / (t1 - first),
                       "steps": drafts, "tok_per_step": 1 + acc / drafts if drafts else None,
                       "ms_per_step": 1e3 * (t1 - first) / drafts if drafts else None, "counters": dc}
                f.write(json.dumps(rec) + "\n")
                print(f"{a.tag} {name} rep {rep}: {rec['tok_per_step'] or 0:.2f} tok/step, "
                      f"{rec['ms_per_step'] or 0:.1f} ms/step, {rec['decode_tps']:.1f} tok/s, ttft {rec['ttft']:.2f} s",
                      flush=True)


def report(a):
    rows, pool = {}, {}
    for path in a.files:
        for line in open(path):
            r = json.loads(line)
            rows.setdefault((r["prompt"], r["engine"]), []).append(r)
            fam = re.sub(r"\d+$", "", r["prompt"])
            if fam != r["prompt"] and r.get("steps") and r["rep"] == 0:
                q = pool.setdefault((fam, r["engine"]), [0, 0, 0.0])
                q[0] += r["tokens"]; q[1] += r["steps"]; q[2] += r["tokens"] / r["decode_tps"]
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"{'prompt':10s} {'engine':14s} {'ctx':>6s} {'tok/step':>8s} {'ms/step':>8s} {'tok/s':>7s} {'ttft s':>7s}")
    for (p, e), rs in sorted(rows.items()):
        g = lambda k: med([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) for r in rs) else float("nan")
        print(f"{p:10s} {e:14s} {rs[0]['prompt_tokens']:6d} {g('tok_per_step'):8.2f} {g('ms_per_step'):8.1f} "
              f"{g('decode_tps'):7.1f} {g('ttft'):7.2f}  ({rs[0]['tokens']} tok)")
    for (fam, e), (n, steps, t) in sorted(pool.items()):
        print(f"{fam + '*':10s} {e:14s} {'pooled':>6s} {n / steps:8.2f} {1e3 * t / steps:8.1f} {n / t:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--tokenizer", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--gen", type=int, default=512)
    m.add_argument("--gen-depth", type=int, default=256)
    m.add_argument("--gen-stop", type=int, default=1536, help="cap for the prompts that stop at EOS")
    for name in ("qe", "vllm"):
        s = sp.add_parser(name)
        s.add_argument("--prompts", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--only", default="")
        s.add_argument("--reps", type=int, default=2)
        s.add_argument("--tag", default="")
    sp.choices["qe"].add_argument("--model", required=True)
    sp.choices["qe"].add_argument("--devices", default="0,1")
    sp.choices["qe"].add_argument("--k", default="7", help="max drafts per step; a list (0,1,3,7) runs each")
    sp.choices["qe"].add_argument("--profile", type=int, default=0)
    sp.choices["vllm"].add_argument("--url", default="http://127.0.0.1:8110")
    sp.choices["vllm"].add_argument("--model", default="qwen3.8-27b")
    r = sp.add_parser("report")
    r.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"make": make, "qe": run_qe, "vllm": run_vllm, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
