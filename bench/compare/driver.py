#!/usr/bin/env python3
"""compare_2026-10 driver: one engine block (server start, runs, server stop). Run one engine at a time.
usage: driver.py <engine> <blocks: classic,goodput,prefill | drift>
env:   STAGE (staging dir for raw runs, default <repo>/bench/compare/stage), OUT (result tree, default
       <repo>/bench/results/compare_2026-10), VLLM (vllm CLI, default vllm), QENGINE_REF (checkout of the
       reference qengine build used as qengine-sunday, commit 1c3f603), SHAREGPT (ShareGPT json)"""
import json, os, signal, subprocess, sys, time, shutil, datetime, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE = os.environ.get("STAGE", f"{ROOT}/bench/compare/stage")
WT = os.environ.get("QENGINE_REF", ROOT)
DEL = os.environ.get("OUT", f"{ROOT}/bench/results/compare_2026-10")
SUITE = f"{ROOT}/bench/serving_suite.py"
VLLM = os.environ.get("VLLM", "vllm")
TOK = f"{ROOT}/models/Qwen3.8-27B-INT4"
SHAREGPT = os.environ.get("SHAREGPT", f"{ROOT}/bench/data/ShareGPT_V3_unfiltered_cleaned_split.json")
PROG = f"{STAGE}/progress.log"
os.makedirs(STAGE, exist_ok=True)

ENGINES = {
    "vllm-mtp": dict(cwd=ROOT, cmd=["bench/vllm_cfg_best.sh", "8182"], env={"SPEC_N": "4"}, port=8182),
    "vllm-nospec": dict(cwd=ROOT, cmd=["bench/vllm_cfg_best.sh", "8182"], env={"SPEC_N": "0"}, port=8182),
    "qengine-sunday": dict(cwd=WT, cmd=["server/run.sh", "8181", "--think", "0", "--metrics-port", "19394",
                                        "--stats-file", f"{STAGE}/qe_stats_sunday.json"],
                           env={"QE_MAX_SEQS": "32"}, port=8181),
    "qengine-now": dict(cwd=ROOT, cmd=["server/run.sh", "8181", "--think", "0", "--metrics-port", "19394",
                                       "--stats-file", f"{STAGE}/qe_stats_now.json"],
                        env={"QE_MAX_SEQS": "32"}, port=8181),
}
CLASSIC = [("short", [1, 8, 16, 32]), ("rag16k", [1, 8]), ("agent", [1, 8, 16, 32]), ("sharegpt", [1, 8, 16, 32])]
GOODPUT = [("goodput_chat", [0.5, 1, 2, 3, 4]), ("goodput_long16k", [0.1, 0.25, 0.5, 1]),
           ("goodput_agent", [0.25, 0.5, 1, 2])]
PREFILL = [1024, 8192, 32768, 100000, 200000]
DRIFT = [("short", [1, 32])]
POINT_TIMEOUT = 4 * 3600

eng_name, blocks = sys.argv[1], sys.argv[2].split(",")
E = ENGINES[eng_name]
url = f"http://127.0.0.1:{E['port']}"
server = None
server_n = 0


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def prog(msg):
    with open(PROG, "a") as f:
        f.write(f"{now()} {msg}\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def stop_server():
    global server
    if server is None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(90):
        if server.poll() is not None:
            break
        time.sleep(1)
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server.wait()
    server = None
    for _ in range(60):  # GPU memory released
        used = [int(x) for x in sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()]
        if max(used) < 1500:
            break
        time.sleep(2)


def cleanup(*_):
    stop_server()
    if _:
        prog(f"ABORT {eng_name} signal {_[0]}")
        sys.exit(1)


for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, cleanup)

def healthy():
    if server is None or server.poll() is not None:
        return False
    try:
        urllib.request.urlopen(url + "/v1/models", timeout=15).read()
        return True
    except Exception:
        return False


def start_server():
    global server, server_n
    server_n += 1
    logp = f"{DEL}/logs/{eng_name}/server_{'-'.join(blocks)}_{server_n}.log"
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    env = dict(os.environ, **E["env"])
    with open(logp, "w") as lf:
        server = subprocess.Popen(E["cmd"], cwd=E["cwd"], env=env, stdout=lf, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 2400:
        if server.poll() is not None:
            prog(f"SERVER_DIED_AT_START {eng_name} rc={server.returncode} log={logp}")
            server = None
            return None
        if healthy():
            m = json.load(urllib.request.urlopen(url + "/v1/models", timeout=15))["data"][0]["id"]
            prog(f"SERVER_UP {eng_name} model={m} after {time.time() - t0:.0f}s log={os.path.relpath(logp, DEL)}")
            return m
        time.sleep(3)
    prog(f"SERVER_START_TIMEOUT {eng_name}")
    stop_server()
    return None


def ensure_server():
    global model
    if healthy():
        return True
    stop_server()
    for _ in range(2):
        model = start_server()
        if model:
            return True
    return False


def run_group(cmd, log):
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=POINT_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return "timeout"


def fmt(x, d=0):
    return "n/a" if x is None else f"{x:.{d}f}"


def record_crash(block, scen, point, rc, extra=""):
    with open(f"{STAGE}/crashes.jsonl", "a") as f:
        f.write(json.dumps(dict(time=now(), engine=eng_name, block=block, scenario=scen, point=point, rc=str(rc),
                                server_alive=healthy(), note=extra)) + "\n")


def suite_point(block, scen, c):
    """one point through bench/serving_suite.py, raw files copied into the layout"""
    gp = scen.startswith("goodput")
    sdir = f"{STAGE}/raw/{block}"
    tagp = f"rate{c:g}" if gp else f"c{c}"
    dst = f"{DEL}/{block}/{scen}/{eng_name}/{tagp}"
    if os.path.exists(dst + ".json"):
        prog(f"SKIP {eng_name} {block} {scen} {tagp} exists")
        return
    if not ensure_server():
        prog(f"ABORT {eng_name} server not starting before {block} {scen} {tagp}")
        record_crash(block, scen, tagp, "server-start")
        cleanup()
        sys.exit(3)
    cmd = ["python3", SUITE, "--engine-name", eng_name, "--base-url", url, "--model", model, "--out-dir", sdir,
           "--scenarios", scen, "--skip-existing", "--server-pid", str(server.pid),
           "--engine-cmd", " ".join(f"{k}={v}" for k, v in E["env"].items()) + " " + " ".join(E["cmd"])]
    if gp:
        cmd += ["--rates", f"{c:g}", "--warmup-prompts", "2"]
    else:
        cmd += ["--classic", "--concurrency", str(c)]
        if scen == "sharegpt":
            cmd += ["--sharegpt", SHAREGPT]
    src = f"{sdir}/{eng_name}/{scen}_{'r%g' % c if gp else 'c%d' % c}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rc = run_group(cmd, f"{STAGE}/suite_{eng_name}_{block}_{scen}_{tagp}.log")
    if os.path.exists(src + ".log"):
        shutil.copy2(src + ".log", dst + ".log")
    alive = healthy()
    if os.path.exists(src + ".json"):
        shutil.copy2(src + ".json", dst + ".json")
        r = json.load(open(src + ".json"))
        good = r.get("request_goodput")
        sla = f", SLA {100 * good / r['request_throughput']:.0f}% ({good:.2f} req/s)" if good is not None and r.get("request_throughput") else ""
        prog(f"RESULT {eng_name} {block} {scen} {tagp}: n={r['completed']}/{r['num_prompts'] if 'num_prompts' in r else r['completed'] + r['failed']} "
             f"failed={r['failed']} out {fmt(r['output_throughput'])} tok/s, TTFT p50 {fmt(r.get('median_ttft_ms'))} ms, "
             f"TPOT p50 {fmt(r.get('median_tpot_ms'), 1)} ms{sla}" + ("" if alive else " | SERVER DEAD after run"))
        if not alive or r["failed"]:
            record_crash(block, scen, tagp, rc, f"result saved, failed={r['failed']}, server_alive={alive}")
    else:
        prog(f"CRASH {eng_name} {block} {scen} {tagp}: no result (rc {rc}), server_alive={alive}; log {dst}.log")
        record_crash(block, scen, tagp, rc, "no result file")
    if not alive:
        tail = f"{DEL}/logs/{eng_name}/server_{'-'.join(blocks)}_{server_n}.log"
        prog(f"RESTART {eng_name} after {scen} {tagp} (server log {os.path.relpath(tail, DEL)})")
        stop_server()
    time.sleep(5)


def prefill_cmd(L, seed, d, fn):
    return [VLLM, "bench", "serve", "--backend", "openai-chat", "--base-url", url, "--endpoint",
            "/v1/chat/completions", "--model", model, "--tokenizer", TOK, "--num-prompts", "1", "--seed", str(seed),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,95,99", "--disable-tqdm",
            "--max-concurrency", "1", "--request-rate", "inf", "--dataset-name", "random",
            "--random-input-len", str(L), "--random-output-len", "1", "--random-range-ratio", "0",
            "--random-prefix-len", "0", "--ignore-eos", "--temperature", "0.7", "--top-p", "0.8", "--top-k", "20",
            "--extra-body", '{"chat_template_kwargs": {"enable_thinking": false}}',
            "--save-result", "--result-dir", d, "--result-filename", fn, "--label", eng_name,
            "--metadata", f"engine={eng_name}", "scenario=prefill"]


def prefill():
    d = f"{DEL}/prefill/{eng_name}"
    os.makedirs(d, exist_ok=True)
    for li, L in enumerate(PREFILL):
        res = []
        for rep in ["warmup", 1, 2, 3]:
            fn = f"len{L}_{'warmup' if rep == 'warmup' else 'rep%d' % rep}"
            if os.path.exists(f"{d}/{fn}.json"):
                continue
            seed = 500000 + li * 100 + (0 if rep == "warmup" else rep)
            if not ensure_server():
                prog(f"ABORT {eng_name} server not starting before prefill {fn}")
                record_crash("prefill", "prefill", fn, "server-start")
                cleanup()
                sys.exit(3)
            cmd = prefill_cmd(L, seed, d, fn + ".json")
            with open(f"{STAGE}/prefill_cmds.txt", "a") as f:
                f.write(" ".join(cmd) + "\n")
            rc = run_group(cmd, f"{d}/{fn}.log")
            alive = healthy()
            if os.path.exists(f"{d}/{fn}.json"):
                r = json.load(open(f"{d}/{fn}.json"))
                if r["completed"]:
                    pt, ttft = r["total_input_tokens"], r["median_ttft_ms"]
                    res.append(f"{rep}: {pt} tok, TTFT {ttft:.0f} ms = {pt / ttft * 1000:.0f} tok/s")
                else:
                    res.append(f"{rep}: FAILED (completed 0)")
                    record_crash("prefill", "prefill", fn, rc, "completed 0")
            else:
                res.append(f"{rep}: no result rc {rc}")
                record_crash("prefill", "prefill", fn, rc, "no result file")
            if not alive:
                res.append("SERVER DEAD -> restart")
                record_crash("prefill", "prefill", fn, rc, "server dead after run")
                stop_server()
            time.sleep(3)
        if res:
            prog(f"RESULT {eng_name} prefill len{L}: " + "; ".join(res))


model = None
try:
    prog(f"BLOCK_START {eng_name} {blocks}")
    if not ensure_server():
        prog(f"ABORT {eng_name} server does not start")
        sys.exit(3)
    for b in blocks:
        if b == "classic":
            for s, cs in CLASSIC:
                for c in cs:
                    suite_point("classic", s, c)
        elif b == "goodput":
            for s, rs in GOODPUT:
                for r in rs:
                    suite_point("goodput", s, r)
        elif b == "prefill":
            prefill()
        elif b == "drift":
            for s, cs in DRIFT:
                for c in cs:
                    suite_point("drift", s, c)
    prog(f"BLOCK_END {eng_name} {blocks}")
finally:
    stop_server()
