"""Markdown tables of bench/results/compare_2026-10 (from the summary CSVs written by build_summary.py)."""
import csv, os
R = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "compare_2026-10"))
E=["qengine-now","qengine-sunday","vllm-mtp","vllm-nospec"]
rd=lambda p: list(csv.DictReader(open(f"{R}/{p}")))
cl, gp, pf, dr = rd("classic/summary.csv"), rd("goodput/summary.csv"), rd("prefill/summary.csv"), rd("drift/summary.csv")
f=lambda x,d=0: f"{float(x):,.{d}f}" if x not in ("",None) else "–"
def get(rows, e, s, k, v): return next(r for r in rows if r["engine"]==e and r["scenario"]==s and r[k]==v)
out=[]
out.append("| scenario | c | " + " | ".join(E) + " | vllm-best | qengine-now / vllm-best |")
out.append("|---|---:|" + "---:|"*(len(E)+2))
for s,cs in [("short",[1,8,16,32]),("rag16k",[1,8]),("agent",[1,8,16,32]),("sharegpt",[1,8,16,32])]:
    for c in cs:
        rr={e:get(cl,e,s,"concurrency",str(c)) for e in E}
        vb=max(("vllm-mtp","vllm-nospec"), key=lambda e: float(rr[e]["output_tok_s"]))
        out.append(f"| {s} | {c} | " + " | ".join(f(rr[e]["output_tok_s"]) for e in E) + f" | {f(rr[vb]['output_tok_s'])} ({vb.split('-')[1]}) | {float(rr['qengine-now']['output_tok_s'])/float(rr[vb]['output_tok_s']):.2f}x |")
out.append("\nTTFT p50 / TPOT p50 (ms):\n")
out.append("| scenario | c | " + " | ".join(E) + " |")
out.append("|---|---:|" + "---:|"*len(E))
for s,cs in [("short",[1,8,16,32]),("rag16k",[1,8]),("agent",[1,8,16,32]),("sharegpt",[1,8,16,32])]:
    for c in cs:
        out.append(f"| {s} | {c} | " + " | ".join(f"{f(get(cl,e,s,'concurrency',str(c))['ttft_p50_ms'])} / {f(get(cl,e,s,'concurrency',str(c))['tpot_p50_ms'],1)}" for e in E) + " |")
out.append("\nGOODPUT (goodput req/s, share of requests within SLA, output tok/s):\n")
out.append("| scenario | rate | " + " | ".join(E) + " |")
out.append("|---|---:|" + "---:|"*len(E))
for s,rs in [("goodput_chat",["0.5","1","2","3","4"]),("goodput_long16k",["0.1","0.25","0.5","1"]),("goodput_agent",["0.25","0.5","1","2"])]:
    for r in rs:
        cells=[]
        for e in E:
            x=next(y for y in gp if y["engine"]==e and y["scenario"]==s and float(y["request_rate"])==float(r))
            cells.append(f"{float(x['goodput_req_s']):.2f} ({100*float(x['goodput_frac']):.0f}%) · {f(x['output_tok_s'])}" + (" *" if x["failed"]!="0" else ""))
        out.append(f"| {s} | {r} | " + " | ".join(cells) + " |")
out.append("\nPREFILL (median TTFT of 3 reps, prompt tok/s):\n")
out.append("| prompt tokens | " + " | ".join(E) + " | qengine-now vs vllm-best |")
out.append("|---:|" + "---:|"*(len(E)+1))
for L in ["1024","8192","32768","100000","200000"]:
    rr={e:next(y for y in pf if y["engine"]==e and y["target_len"]==L) for e in E}
    vb=min(("vllm-mtp","vllm-nospec"), key=lambda e: float(rr[e]["ttft_ms_median"]))
    out.append(f"| {rr['qengine-now']['prompt_tokens']} | " + " | ".join(f"{float(rr[e]['ttft_ms_median'])/1000:.2f} s · {f(rr[e]['prompt_tok_s_median'])}" for e in E) + f" | {float(rr[vb]['ttft_ms_median'])/float(rr['qengine-now']['ttft_ms_median']):.2f}x faster |")
out.append("\nDRIFT:\n")
out.append("| engine | c | classic tok/s | drift tok/s | Δ |")
out.append("|---|---:|---:|---:|---:|")
for x in dr:
    out.append(f"| {x['engine']} | {x['concurrency']} | {f(x['classic_output_tok_s'])} | {f(x['drift_output_tok_s'])} | {float(x['output_delta_pct']):+.1f}% |")
print("\n".join(out))
