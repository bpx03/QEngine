#!/usr/bin/env python3
"""Tidy summaries from the raw vllm bench serve JSONs in bench/results/compare_2026-10 (raw files are only read)."""
import csv, glob, json, os, re, statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEL = os.environ.get("OUT", f"{ROOT}/bench/results/compare_2026-10")
CR = os.path.join(os.environ.get("STAGE", f"{ROOT}/bench/compare/stage"), "crashes.jsonl")  # written by driver.py
ENG = ["vllm-mtp", "vllm-nospec", "qengine-sunday", "qengine-now"]
COLS = ["engine", "scenario", "concurrency", "request_rate", "num_prompts", "completed", "failed", "crash", "status",
        "mean_input_tokens", "mean_output_tokens", "output_tok_s", "total_tok_s", "req_s",
        "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
        "itl_p50_ms", "itl_p99_ms", "e2el_p50_ms", "e2el_p99_ms", "goodput_req_s", "goodput_frac", "duration_s",
        "raw_file"]
CLASSIC = [("short", [1, 8, 16, 32]), ("rag16k", [1, 8]), ("agent", [1, 8, 16, 32]), ("sharegpt", [1, 8, 16, 32])]
GOODPUT = [("goodput_chat", [0.5, 1, 2, 3, 4]), ("goodput_long16k", [0.1, 0.25, 0.5, 1]),
           ("goodput_agent", [0.25, 0.5, 1, 2])]
DRIFT = [("short", [1, 32])]
crashes = [json.loads(l) for l in open(CR)] if os.path.exists(CR) else []


def r2(x, d=2):
    return "" if x is None else round(x, d)


def crash_for(engine, scen, path):
    blk = os.path.relpath(path, DEL).split("/")[0]
    pt = os.path.basename(path)[:-5]
    return [c for c in crashes if c["engine"] == engine and c["scenario"] == scen and c["block"] == blk
            and c["point"] == pt]


def row(path, engine, scen, conc="", rate=""):
    base = dict(engine=engine, scenario=scen, concurrency=conc, request_rate=rate)
    if not os.path.exists(path):
        cr = crash_for(engine, scen, path)
        return {**base, "crash": bool(cr), "status": "crash (no result)" if cr else "missing",
                "raw_file": os.path.relpath(path, DEL)}
    r = json.load(open(path))
    n = r["completed"] + r["failed"]
    good = r.get("request_goodput")
    st = "ok" if not r["failed"] else f"partial ({r['failed']} failed)"
    cr = [c for c in crash_for(engine, scen, path) if not c["server_alive"]]
    if cr:
        st += ", server crashed"
    return {**base, "num_prompts": n, "completed": r["completed"], "failed": r["failed"], "crash": bool(cr),
            "status": st,
            "mean_input_tokens": r2(r["total_input_tokens"] / max(r["completed"], 1), 1),
            "mean_output_tokens": r2(r["total_output_tokens"] / max(r["completed"], 1), 1),
            "output_tok_s": r2(r["output_throughput"]), "total_tok_s": r2(r["total_token_throughput"]),
            "req_s": r2(r["request_throughput"], 4),
            **{f"{m}_p{p}_ms": r2(r.get(f"p{p}_{m}_ms")) for m in ("ttft", "tpot") for p in (50, 95, 99)},
            **{f"{m}_p{p}_ms": r2(r.get(f"p{p}_{m}_ms")) for m in ("itl", "e2el") for p in (50, 99)},
            "goodput_req_s": r2(good, 4), "goodput_frac": r2(good / r["request_throughput"], 4)
            if good is not None and r["request_throughput"] else "", "duration_s": r2(r["duration"], 1),
            "raw_file": os.path.relpath(path, DEL)}


def write(path, rows, cols=COLS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


cl = [row(f"{DEL}/classic/{s}/{e}/c{c}.json", e, s, conc=c) for e in ENG for s, cs in CLASSIC for c in cs]
gp = [row(f"{DEL}/goodput/{s}/{e}/rate{r:g}.json", e, s, rate=r) for e in ENG for s, rs in GOODPUT for r in rs]
dr = [row(f"{DEL}/drift/{s}/{e}/c{c}.json", e, s, conc=c) for e in ENG for s, cs in DRIFT for c in cs]
write(f"{DEL}/classic/summary.csv", cl)
write(f"{DEL}/goodput/summary.csv", gp)

# drift: second run vs the classic run of the same point (same prompts, fresh server)
dcols = ["engine", "scenario", "concurrency", "classic_output_tok_s", "drift_output_tok_s", "output_delta_pct",
         "classic_ttft_p50_ms", "drift_ttft_p50_ms", "classic_tpot_p50_ms", "drift_tpot_p50_ms", "tpot_delta_pct",
         "drift_raw_file"]
drows = []
for d in dr:
    c = next(x for x in cl if x["engine"] == d["engine"] and x["scenario"] == d["scenario"]
             and x["concurrency"] == d["concurrency"])
    pct = lambda a, b: round(100 * (b - a) / a, 2) if isinstance(a, float) and isinstance(b, float) and a else ""
    drows.append(dict(engine=d["engine"], scenario=d["scenario"], concurrency=d["concurrency"],
                      classic_output_tok_s=c.get("output_tok_s"), drift_output_tok_s=d.get("output_tok_s"),
                      output_delta_pct=pct(c.get("output_tok_s"), d.get("output_tok_s")),
                      classic_ttft_p50_ms=c.get("ttft_p50_ms"), drift_ttft_p50_ms=d.get("ttft_p50_ms"),
                      classic_tpot_p50_ms=c.get("tpot_p50_ms"), drift_tpot_p50_ms=d.get("tpot_p50_ms"),
                      tpot_delta_pct=pct(c.get("tpot_p50_ms"), d.get("tpot_p50_ms")), drift_raw_file=d["raw_file"]))
os.makedirs(f"{DEL}/drift", exist_ok=True)
write(f"{DEL}/drift/summary.csv", drows, dcols)

# prefill: per engine x length, 3 reps (warm-up excluded)
pcols = ["engine", "target_len", "prompt_tokens", "reps", "ttft_ms_rep1", "ttft_ms_rep2", "ttft_ms_rep3",
         "ttft_ms_median", "ttft_ms_min", "ttft_ms_max", "prompt_tok_s_median", "status", "raw_files"]
prow = []
for e in ENG:
    for L in [1024, 8192, 32768, 100000, 200000]:
        t, pt, files, st = [], [], [], "ok"
        for k in (1, 2, 3):
            p = f"{DEL}/prefill/{e}/len{L}_rep{k}.json"
            if not os.path.exists(p):
                st = "missing/crash"
                t.append(None)
                continue
            r = json.load(open(p))
            files.append(os.path.relpath(p, DEL))
            if r["completed"] != 1:
                st = "failed"
                t.append(None)
                continue
            t.append(r["median_ttft_ms"])
            pt.append(r["total_input_tokens"])
        ok = [x for x in t if x is not None]
        med = statistics.median(ok) if ok else None
        prow.append(dict(engine=e, target_len=L, prompt_tokens=round(statistics.mean(pt)) if pt else "",
                         reps=len(ok), **{f"ttft_ms_rep{k}": r2(t[k - 1], 1) for k in (1, 2, 3)},
                         ttft_ms_median=r2(med, 1), ttft_ms_min=r2(min(ok), 1) if ok else "",
                         ttft_ms_max=r2(max(ok), 1) if ok else "",
                         prompt_tok_s_median=r2(statistics.mean(pt) / med * 1000, 0) if ok else "",
                         status=st if ok or st != "ok" else "missing", raw_files=";".join(files)))
os.makedirs(f"{DEL}/prefill", exist_ok=True)
write(f"{DEL}/prefill/summary.csv", prow, pcols)

# best vLLM per point: classic by output tok/s; goodput by goodput req/s (tie -> output tok/s); prefill by TTFT
best = []
def pick(rows, key):
    a = [x for x in rows if isinstance(x.get(key), (int, float))]
    return max(a, key=lambda x: (x[key], x.get("output_tok_s") or 0)) if a else None
for s, cs in CLASSIC:
    for c in cs:
        rr = [x for x in cl if x["scenario"] == s and x["concurrency"] == c and x["engine"].startswith("vllm")]
        b = pick(rr, "output_tok_s")
        best.append(dict(block="classic", scenario=s, point=f"c={c}", criterion="max output_tok_s",
                         best_variant=b["engine"] if b else "none", **({k: b[k] for k in COLS[COLS.index("output_tok_s"):-1]} if b else {}),
                         **{f"{x['engine']}_status": x["status"] for x in rr}))
for s, rs in GOODPUT:
    for r in rs:
        rr = [x for x in gp if x["scenario"] == s and x["request_rate"] == r and x["engine"].startswith("vllm")]
        b = pick(rr, "goodput_req_s")
        best.append(dict(block="goodput", scenario=s, point=f"rate={r:g}", criterion="max goodput_req_s",
                         best_variant=b["engine"] if b else "none", **({k: b[k] for k in COLS[COLS.index("output_tok_s"):-1]} if b else {}),
                         **{f"{x['engine']}_status": x["status"] for x in rr}))
for L in [1024, 8192, 32768, 100000, 200000]:
    rr = [x for x in prow if x["target_len"] == L and x["engine"].startswith("vllm") and x["ttft_ms_median"] != ""]
    b = min(rr, key=lambda x: x["ttft_ms_median"]) if rr else None
    best.append(dict(block="prefill", scenario="prefill", point=f"len={L}", criterion="min median TTFT",
                     best_variant=b["engine"] if b else "none", ttft_p50_ms=b["ttft_ms_median"] if b else "",
                     prompt_tok_s=b["prompt_tok_s_median"] if b else ""))
bcols = ["block", "scenario", "point", "criterion", "best_variant", *COLS[COLS.index("output_tok_s"):-1], "prompt_tok_s",
         "vllm-mtp_status", "vllm-nospec_status"]
write(f"{DEL}/best_vllm.csv", best, bcols)
print("ok", len(cl), len(gp), len(prow), len(drows))
