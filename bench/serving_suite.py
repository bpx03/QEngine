"""Serving benchmark suite (vllm bench serve) for any OpenAI-compatible server; same settings for every engine.
usage: python bench/serving_suite.py --engine-name qengine --base-url http://127.0.0.1:8181 \
           --model qwen3.8-27b-qengine [--scenarios short,rag16k] [--concurrency 1,8,16,32] [--quick] [--dry-run]
Per run: <out-dir>/<engine>/<scenario>_c<C>.json (vllm result), log next to it; one row per run appended to
<out-dir>/summary.csv; meta_<engine>_<time>.json with GPUs, versions and the exact command lines.
Notes on the random dataset (vllm 0.28): --random-prefix-len defaults to 0 and each prompt starts at its own
random offset in the vocab, so prompts within a run share no prefix. The prompts depend only on the seed, so
runs with the same seed repeat the same prompts and a server-side prefix cache (vLLM APC, SGLang radix, qengine
RAM archive) would serve them from cache: each (scenario, concurrency) gets its own seed (--same-seed: old style).
Built-in --num-warmups is not used: it repeats the first prompt of the measured run (cache hit)."""
import argparse, csv, datetime, json, os, shlex, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VLLM = os.environ.get("VLLM", "vllm")  # the vllm CLI (vllm bench serve); set VLLM=<path> for a venv
# name: (input len, output len); the chat template adds ~10-20 tokens on top of the input
SCENARIOS = {"short": (1024, 512), "rag16k": (15000, 128), "agent": (500, 2000),
             "long16k_decode": (16000, 1024), "sharegpt": (None, None)}
# goodput (SLA) scenarios: Poisson arrivals at fixed request rates, no client concurrency cap (the server
# queues, so the queueing shows in TTFT). name: (shared prefix, unique input, output, rates, seconds per rate,
# quick rates, quick seconds); prompt = prefix + input (+-50%), output +-50%
GOODPUT = {"goodput_chat": (1024, 1024, 256, (0.5, 1, 2, 3, 4), 180, (1, 2, 4), 60),
           "goodput_long16k": (12288, 4096, 512, (0.1, 0.25, 0.5, 1), 240, (0.25, 0.5), 90),
           "goodput_agent": (4096, 2048, 1024, (0.25, 0.5, 1, 2), 240, (0.5, 1), 90)}
# the fixed engine comparison (--classic): scenario -> (concurrencies, prompts for a concurrency)
CLASSIC = {"short": ((1, 8, 16, 32), lambda c: 8 if c == 1 else 4 * c),
           "rag16k": ((1, 8), lambda c: 8 if c == 1 else 16),
           "agent": ((1, 8, 16, 32), lambda c: 8 if c == 1 else 2 * c),
           "sharegpt": ((1, 8, 16, 32), lambda c: 16 if c == 1 else 4 * c)}
SHAREGPT = os.path.join(ROOT, "bench/data/ShareGPT_V3_unfiltered_cleaned_split.json")  # download: see README
FIELDS = ["timestamp", "engine", "scenario", "concurrency", "request_rate", "num_prompts", "completed", "failed",
          "input_len", "output_len", "seed", "output_tok_s", "total_tok_s", "req_s",
          "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "itl_p50_ms", "itl_p99_ms",
          "e2el_p50_ms", "e2el_p99_ms", "ttft_p95_ms", "tpot_p95_ms", "goodput_req_s", "goodput_frac",
          "users_little", "duration_s", "server_cpu_cores", "git_commit", "result_file"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as ex:
        return f"error: {ex}"


def server_cpu(pid):
    """CPU seconds (user + system) of a process and its children (vLLM runs several processes)"""
    if not pid:
        return 0.0
    import psutil
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0.0
    tot = 0.0
    for q in procs:
        try:
            t = q.cpu_times()
            tot += t.user + t.system
        except psutil.Error:
            pass
    return tot


def git_commit():
    c = sh(f"git -C {ROOT} rev-parse --short HEAD")
    return c + ("-dirty" if sh(f"git -C {ROOT} status --porcelain --untracked-files=no") else "")


def bench_cmd(a, scen, c, n, seed, path):
    """c: client concurrency, or the request rate for a goodput scenario"""
    cmd = [VLLM, "bench", "serve", "--backend", a.backend, "--base-url", a.base_url, "--endpoint", a.endpoint,
           "--model", a.model, "--tokenizer", a.tokenizer, "--num-prompts", str(n), "--seed", str(seed),
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,95,99", "--disable-tqdm"]
    if scen in GOODPUT:
        pre, il, ol = GOODPUT[scen][:3]
        cmd += ["--request-rate", str(c), "--burstiness", "1.0", "--goodput", *a.sla.split(),
                "--dataset-name", "random", "--random-prefix-len", str(pre), "--random-input-len", str(il),
                "--random-output-len", str(ol), "--random-range-ratio", "0.5"]
    else:
        cmd += ["--max-concurrency", str(c), "--request-rate", "inf"]
    if scen == "sharegpt":
        cmd += ["--dataset-name", "sharegpt", "--dataset-path", a.sharegpt]  # real input/output lengths
    elif scen not in GOODPUT:
        il, ol = SCENARIOS[scen]
        cmd += ["--dataset-name", "random", "--random-input-len", str(il), "--random-output-len", str(ol),
                "--random-range-ratio", "0", "--random-prefix-len", "0"]
    if a.ignore_eos and scen != "sharegpt":
        cmd += ["--ignore-eos"]  # fixed output length; qengine also skips its loop guard with it
    for k in ("temperature", "top_p", "top_k"):  # explicit sampling: server defaults differ between engines
        if getattr(a, k) is not None:
            cmd += ["--" + k.replace("_", "-"), str(getattr(a, k))]
    if a.extra_body:
        cmd += ["--extra-body", a.extra_body]
    if path:
        cmd += ["--save-result", "--result-dir", os.path.dirname(path), "--result-filename", os.path.basename(path),
                "--label", a.engine_name, "--metadata", f"engine={a.engine_name}", f"scenario={scen}"]
    return cmd + a.extra


def summary_row(a, scen, c, n, seed, path, commit):
    r = json.load(open(path))
    gp = scen in GOODPUT
    il, ol = (GOODPUT[scen][0] + GOODPUT[scen][1], GOODPUT[scen][2]) if gp else SCENARIOS[scen]
    g = lambda k: r.get(k)
    good = g("request_goodput")
    return {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "engine": a.engine_name,
            "scenario": scen, "concurrency": "" if gp else c, "request_rate": c if gp else "", "num_prompts": n, "completed": g("completed"), "failed": g("failed"),
            "input_len": il if il else round(g("total_input_tokens") / max(g("completed"), 1)),
            "output_len": ol if ol else round(g("total_output_tokens") / max(g("completed"), 1)), "seed": seed,
            "output_tok_s": g("output_throughput"), "total_tok_s": g("total_token_throughput"),
            "req_s": g("request_throughput"),
            **{f"{m}_p{p}_ms": g(f"p{p}_{m}_ms") for m in ("ttft", "tpot", "itl", "e2el") for p in (50, 99)},
            "ttft_p95_ms": g("p95_ttft_ms"), "tpot_p95_ms": g("p95_tpot_ms"), "goodput_req_s": good,
            # share of requests within the SLA; users = goodput x mean latency (Little's law)
            "goodput_frac": good / g("request_throughput") if good is not None and g("request_throughput") else "",
            "users_little": good * g("mean_e2el_ms") / 1000 if good is not None and g("mean_e2el_ms") else "",
            "duration_s": g("duration"), "git_commit": commit, "result_file": os.path.relpath(path, a.out_dir)}


def meta(a, cmds, commit):
    gpus = sh("nvidia-smi --query-gpu=index,name,pci.bus_id,"
              "memory.total,driver_version,vbios_version --format=csv,noheader")
    return {"time": datetime.datetime.now().isoformat(timespec="seconds"), "engine": a.engine_name,
            "base_url": a.base_url, "model": a.model, "tokenizer": a.tokenizer, "git_commit": commit,
            "git_branch": sh(f"git -C {ROOT} rev-parse --abbrev-ref HEAD"),
            "vllm_version": sh(f"{VLLM} --version 2>/dev/null | tail -1"), "host": os.uname().nodename,
            "kernel": os.uname().release, "gpus": gpus.splitlines(), "p2p": sh("nvidia-smi topo -p2p r"),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("QE_", "CUDA_VISIBLE"))},
            "engine_cmd": a.engine_cmd, "suite_argv": sys.argv, "commands": cmds}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine-name", required=True, help="label: qengine / vllm-tp2 / sglang / exllamav3 ...")
    ap.add_argument("--base-url", default="http://127.0.0.1:8181")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models/Qwen3.8-27B-INT4"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "bench/results", datetime.date.today().isoformat()))
    ap.add_argument("--scenarios", default="short,rag16k,agent,long16k_decode,sharegpt",
                    help=f"comma list of {','.join(SCENARIOS)} (sharegpt only with --sharegpt)")
    ap.add_argument("--concurrency", default="1,8,16,32")
    ap.add_argument("--prompts-per-conc", type=int, default=4, help="num_prompts = max(k * c, --min-prompts)")
    ap.add_argument("--min-prompts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=908)
    ap.add_argument("--same-seed", action="store_true", help="one seed for all runs (prompts repeat between runs)")
    ap.add_argument("--sharegpt", help="ShareGPT json (ShareGPT_V3_unfiltered_cleaned_split.json)")
    ap.add_argument("--backend", default="openai-chat", help="qengine serves only /v1/chat/completions")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    # no-think sampling (Qwen recommendation for non-thinking mode); thinking off via the chat template
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--extra-body", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    ap.add_argument("--warmup-prompts", type=int, default=4, help="unsaved short run (own seed) before the suite")
    ap.add_argument("--engine-cmd", default="", help="server command line, stored in meta.json")
    ap.add_argument("--pause", type=float, default=5, help="seconds between runs (server drains, GPUs cool)")
    ap.add_argument("--quick", action="store_true",
                    help="quick comparison (~8 min): short at c=1 + goodput_chat, goodput_long16k at quick rates; "
                         "--scenarios / --rates still apply")
    ap.add_argument("--classic", action="store_true",
                    help="the fixed engine comparison: short (1024/512) at c = 1, 8, 16, 32 with 8 / 4c prompts, "
                         "results in bench/results/classic/<engine>/, points already measured are skipped")
    ap.add_argument("--server-pid", type=int, help="server process: its CPU use (with children) is recorded per run")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs whose result file exists")
    ap.add_argument("--sla", default="ttft:3000 tpot:50", help="goodput SLO (vllm bench --goodput), ms")
    ap.add_argument("--rates", help="request rates for goodput scenarios (default: per scenario)")
    ap.add_argument("--dry-run", action="store_true", help="only print the commands")
    ap.add_argument("extra", nargs="*", help="after --: extra args for vllm bench serve")
    a = ap.parse_args()
    if a.quick:  # an explicit --scenarios still selects (quick durations and prompt counts stay)
        if a.scenarios == ap.get_default("scenarios"):
            a.scenarios = "short,goodput_chat,goodput_long16k"
        a.concurrency, a.prompts_per_conc, a.min_prompts, a.warmup_prompts, a.pause = "1", 8, 8, 2, 2
    if a.classic:  # identical prompts for every engine: same seeds, same counts; never re-measured
        if a.scenarios == ap.get_default("scenarios"):
            a.scenarios = ",".join(CLASSIC)
        a.skip_existing, a.warmup_prompts = True, 2
        if not a.sharegpt and os.path.exists(SHAREGPT):
            a.sharegpt = SHAREGPT
        if a.out_dir == ap.get_default("out_dir"):
            a.out_dir = os.path.join(ROOT, "bench/results/classic")
    a.out_dir = os.path.abspath(a.out_dir)
    scens = [s for s in a.scenarios.split(",") if s]
    for s in scens:
        if s not in SCENARIOS and s not in GOODPUT:
            sys.exit(f"unknown scenario {s}")
    if "sharegpt" in scens and not a.sharegpt:
        scens.remove("sharegpt")
        print("# sharegpt skipped (no --sharegpt path)", file=sys.stderr)
    concs = [int(c) for c in a.concurrency.split(",")]
    commit = git_commit()
    rdir = os.path.join(a.out_dir, a.engine_name)

    runs = []
    for si, s in enumerate([*SCENARIOS, *GOODPUT]):
        if s not in scens:
            continue
        gp = GOODPUT.get(s)
        rates = [float(r) for r in a.rates.split(",")] if gp and a.rates else gp and (gp[5] if a.quick else gp[3])
        classic_c = [x for x in CLASSIC[s][0] if a.concurrency == ap.get_default("concurrency") or x in concs] if a.classic else []
        for c in rates or (classic_c if a.classic else concs):
            if gp:
                n = max(round(c * (gp[6] if a.quick else gp[4])), 8)
                seed = a.seed if a.same_seed else a.seed + 1000 * (si + 1) + round(c * 100)
                path = os.path.join(rdir, f"{s}_r{c:g}.json")
            else:
                n = CLASSIC[s][1](c) if a.classic else max(a.prompts_per_conc * c, a.min_prompts)
                seed = a.seed if a.same_seed else a.seed + 1000 * (si + 1) + c
                path = os.path.join(rdir, f"{s}_c{c}.json")
            if a.skip_existing and os.path.exists(path):
                print(f"# skip {a.engine_name} {s} {c:g}: {os.path.relpath(path, ROOT)} exists", file=sys.stderr)
                continue
            runs.append((s, c, n, seed, path, bench_cmd(a, s, c, n, seed, path)))
    warm = []
    if a.warmup_prompts and runs:  # graph capture / autotune / compile; different seed than any measured run
        warm = bench_cmd(a, "short", min(a.warmup_prompts, 8), a.warmup_prompts, a.seed + 999_999, None)
    cmds = ([shlex.join(warm)] if warm else []) + [shlex.join(r[5]) for r in runs]
    if a.dry_run:
        print("\n".join(cmds))
        return

    os.makedirs(rdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H%M%S")
    with open(os.path.join(a.out_dir, f"meta_{a.engine_name}_{stamp}.json"), "w") as f:
        json.dump(meta(a, cmds, commit), f, indent=1)
    if warm:
        print("# warm-up", flush=True)
        subprocess.run(warm, stdout=subprocess.DEVNULL)
    csv_path = os.path.join(a.out_dir, "summary.csv")
    for s, c, n, seed, path, cmd in runs:
        print(f"# {a.engine_name} {s} {'rate' if s in GOODPUT else 'c'}={c:g} n={n}", flush=True)
        if os.path.exists(path):
            os.rename(path, path[:-5] + f".old{int(time.time())}.json")  # never mix old and new results
        cpu0, t0 = server_cpu(a.server_pid), time.time()
        with open(path[:-5] + ".log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        cores = (server_cpu(a.server_pid) - cpu0) / (time.time() - t0) if a.server_pid else ""
        if rc or not os.path.exists(path):
            print(f"  FAILED (rc {rc}), see {path[:-5]}.log", flush=True)
            continue
        row = summary_row(a, s, c, n, seed, path, commit)
        row["server_cpu_cores"] = cores
        if os.path.exists(csv_path):  # older header (fewer fields): rewrite with the current one
            with open(csv_path) as f:
                rd = csv.DictReader(f)
                old = list(rd) if rd.fieldnames != FIELDS else None
            if old is not None:
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, FIELDS)
                    w.writeheader()
                    w.writerows({k: r.get(k, "") for k in FIELDS} for r in old)
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"  {row['output_tok_s']:.0f} tok/s out, TTFT p50/p95 {row['ttft_p50_ms']:.0f}/{row['ttft_p95_ms']:.0f} ms, "
              f"TPOT p50/p95 {row['tpot_p50_ms']:.1f}/{row['tpot_p95_ms']:.1f} ms, failed {row['failed']}"
              + (f", goodput {row['goodput_req_s']:.2f} req/s ({100 * row['goodput_frac']:.0f}% in SLA), "
                 f"~{row['users_little']:.1f} users" if row["goodput_req_s"] is not None else ""), flush=True)
        time.sleep(a.pause)


if __name__ == "__main__":
    main()
