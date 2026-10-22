"""Live monitoring of the QEngine server: throughput, prefill, agents and queue, KV pool, GPUs.

Counters are kept since the server start ("session") and across restarts ("global", saved to a
JSON file every 30 s and at shutdown). Everything is served on a separate fixed port (default
127.0.0.1:9393), independent of the API port, so watching never touches the model server:
  /               live dashboard (server/dashboard.html)
  /api/stats      JSON snapshot            /api/stream   the same as server-sent events (2 Hz)
  /metrics        Prometheus text format
The scheduler thread reports events (cheap host-side bookkeeping); a 1 Hz thread polls NVML."""
import collections
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COUNTERS = ("requests", "errors", "prompt_tokens", "cached_tokens", "prefill_tokens", "prefill_s",
            "gen_tokens", "decode_s", "steps", "seq_steps", "images", "image_tokens", "energy_j",
            "uptime_s")


class Metrics:
    def __init__(self, path, served_name=""):
        self.lock = threading.Lock()
        self.path = path
        self.name = served_name
        self.t0 = time.time()
        self.session = dict.fromkeys(COUNTERS, 0)
        self.session["cache_src"] = {}
        self.base = {k: 0 for k in COUNTERS}  # global totals before this run
        self.base["cache_src"] = {}
        self.base_records = {"peak_tok_s": 0.0, "max_agents": 0, "longest_context": 0}
        self.records = dict(self.base_records)  # all time
        self.srec = dict.fromkeys(self.base_records, 0)  # this session (dashboard reset clears it)
        self.runs = 1
        try:
            g = json.load(open(path))
            for k in COUNTERS:
                self.base[k] = g.get(k, 0)
            self.base["cache_src"] = g.get("cache_src", {})
            self.base_records.update(g.get("records", {}))
            self.records = dict(self.base_records)
            self.runs = g.get("runs", 0) + 1
            self.first_start = g.get("first_start", self.t0)
        except Exception:
            self.first_start = self.t0
        self.decode_ev = collections.deque()  # (t, tokens) of decode steps, last 60 s
        self.hist = collections.deque(maxlen=600)  # 1 s buckets, last 10 min
        self.cur = self._bucket(int(self.t0))
        self.jobs = {}  # id -> live info (queued / prefill / decode)
        self.last_done = collections.deque(maxlen=12)
        self.ended = collections.deque()  # (t, frozen job view) of requests finished in the last 5 s
        self.engine = {}
        self.gpus = []
        self.step_ms = collections.deque(maxlen=200)
        self._energy = {}
        self._busy_since = None
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------ events (scheduler thread)
    @staticmethod
    def _bucket(sec):
        return {"t": sec, "dec": 0, "pre": 0, "active": 0, "queued": 0, "power": 0.0}

    def queued(self, job, label, images=0, image_tokens=0):
        with self.lock:
            self.jobs[id(job)] = {"label": label, "state": "queued", "prompt": len(job.ids),
                                  "t_q": time.time(), "gen": 0, "cached": 0, "slot": -1,
                                  "ev": collections.deque()}  # (t, tokens) of its steps, last 10 s
            self.session["requests"] += 1
            self.session["images"] += images
            self.session["image_tokens"] += image_tokens

    def prefill_start(self, job):
        with self.lock:
            j = self.jobs.get(id(job))
            if j:
                j["state"], j["t_p"] = "prefill", time.time()

    def prefill_done(self, job):
        now = time.time()
        with self.lock:
            j = self.jobs.get(id(job))
            if not j:
                return
            dt = max(now - j["t_p"], 1e-6)
            comp = max(len(job.ids) - job.reused, 0)
            j.update(state="decode", t_d=now, cached=job.reused, slot=job.slot, src=job.src or
                     ("live" if job.reused else ""), prefill_tok=comp, prefill_tps=comp / dt, prefill_s=dt)
            s = self.session
            s["prompt_tokens"] += len(job.ids)
            s["cached_tokens"] += job.reused
            s["prefill_tokens"] += comp
            s["prefill_s"] += dt
            src = j["src"] or "none"
            s["cache_src"][src] = s["cache_src"].get(src, 0) + 1
            self.records["longest_context"] = max(self.records["longest_context"], len(job.ids))
            self.srec["longest_context"] = max(self.srec["longest_context"], len(job.ids))
            # spread the prefilled tokens over the seconds it took (history chart)
            t0, t1 = int(j["t_p"]), int(now)
            n = t1 - t0 + 1
            for b in list(self.hist) + [self.cur]:
                if t0 <= b["t"] <= t1:
                    b["pre"] += comp / n

    def step(self, jobs, counts, dt):
        """One decode step of the given jobs (counts: tokens each got)."""
        now = time.time()
        tot = sum(counts)
        with self.lock:
            self.decode_ev.append((now, tot))
            self.cur["dec"] += tot
            self.step_ms.append(dt * 1000)
            s = self.session
            s["gen_tokens"] += tot
            s["steps"] += 1
            s["seq_steps"] += len(jobs)
            for job, c in zip(jobs, counts):
                j = self.jobs.get(id(job))
                if j:
                    j["gen"] += c
                    ev = j["ev"]
                    ev.append((now, c))
                    while ev[0][0] < now - 10:
                        ev.popleft()
            self.records["max_agents"] = max(self.records["max_agents"], len(jobs))
            self.srec["max_agents"] = max(self.srec["max_agents"], len(jobs))

    def finished(self, job, error=None):
        with self.lock:
            j = self.jobs.pop(id(job), None)
            if j:  # dashboards keep showing it (greyed, stats frozen) for 5 s
                self.ended.append((time.time(), self._job_view(j, time.time())))
            if error is not None:
                self.session["errors"] += 1
            if j and j["state"] == "decode":
                d = time.time() - j["t_d"]
                self.last_done.append({"label": j["label"], "prompt": j["prompt"], "cached": j["cached"],
                                       "gen": j["gen"], "tps": j["gen"] / max(d, 1e-6),
                                       "prefill_tps": j.get("prefill_tps", 0), "t": time.time(),
                                       "error": str(error) if error else ""})

    def reset_session(self):
        """Dashboard "reset": the session's counters move into the global totals (nothing is lost
        there), then the session, its charts and recent requests start over. Records stay."""
        now = time.time()
        with self.lock:
            for k in COUNTERS:
                self.base[k] += (now - self.t0) if k == "uptime_s" else self.session[k]
            for k, v in self.session["cache_src"].items():
                self.base["cache_src"][k] = self.base["cache_src"].get(k, 0) + v
            self.session = dict.fromkeys(COUNTERS, 0)
            self.session["cache_src"] = {}
            self.t0 = now
            self.decode_ev.clear()
            self.hist.clear()
            self.cur = self._bucket(int(now))
            self.last_done.clear()
            self.step_ms.clear()
            self.srec = dict.fromkeys(self.srec, 0)
            for j in self.jobs.values():  # running requests count from now on
                j["gen"] = 0
                j["ev"].clear()
                j["t_d"] = now if j["state"] == "decode" else j.get("t_d")
        self.save()

    def engine_snapshot(self, info):
        with self.lock:
            self.engine = info

    # ------------------------------------------------------------ background: NVML + buckets
    def _loop(self):
        nv = None
        try:
            import warnings
            with warnings.catch_warnings():  # "pynvml is deprecated" (it still works)
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit()
            nv = pynvml
        except Exception:
            pass
        last_save = time.time()
        while True:
            time.sleep(1.0 - (time.time() % 1.0))
            now = time.time()
            gpus, power = [], 0.0
            if nv:
                try:
                    for i in range(nv.nvmlDeviceGetCount()):
                        h = nv.nvmlDeviceGetHandleByIndex(i)
                        m = nv.nvmlDeviceGetMemoryInfo(h)
                        u = nv.nvmlDeviceGetUtilizationRates(h)
                        p = nv.nvmlDeviceGetPowerUsage(h) / 1000
                        e = nv.nvmlDeviceGetTotalEnergyConsumption(h) / 1000  # J since driver load
                        with self.lock:
                            if i in self._energy:
                                self.session["energy_j"] += max(e - self._energy[i], 0)
                            self._energy[i] = e
                        gpus.append({"i": i, "mem": m.used, "mem_total": m.total, "util": u.gpu,
                                     "power": p, "limit": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
                                     "temp": nv.nvmlDeviceGetTemperature(h, 0),
                                     "clock": nv.nvmlDeviceGetClockInfo(h, 1)})
                        power += p
                except Exception:
                    pass
            with self.lock:
                self.gpus = gpus
                c = self.cur
                c["power"] = power
                c["active"] = sum(1 for j in self.jobs.values() if j["state"] != "queued")
                c["queued"] = sum(1 for j in self.jobs.values() if j["state"] == "queued")
                if c["active"]:
                    self.session["decode_s"] += 1
                self.hist.append(c)
                self.cur = self._bucket(int(now))
                while self.decode_ev and self.decode_ev[0][0] < now - 60:
                    self.decode_ev.popleft()
                tps10 = sum(n for t, n in self.decode_ev if t >= now - 10) / 10
                self.records["peak_tok_s"] = max(self.records["peak_tok_s"], tps10)
                self.srec["peak_tok_s"] = max(self.srec["peak_tok_s"], tps10)
            if now - last_save > 30:
                self.save()
                last_save = now

    # ------------------------------------------------------------ views
    def _rate(self, sec, now):
        return sum(n for t, n in self.decode_ev if t >= now - sec) / sec

    @staticmethod
    def _job_view(j, now):
        x = dict(j)
        ev = x.pop("ev", ())
        x["age_s"] = now - x["t_q"]  # since the request arrived
        if x["state"] == "queued":
            x["wait_s"] = now - x["t_q"]
        elif x["state"] == "prefill":
            x["elapsed_s"] = now - x["t_p"]
        else:
            x["tps"] = x["gen"] / max(now - x["t_d"], 1e-6)  # average since decode start
            win = min(10, max(now - x["t_d"], 1e-6))
            x["tps_now"] = sum(c for t, c in ev if t >= now - win) / win  # last 10 s
        for k in ("t_q", "t_p", "t_d"):
            x.pop(k, None)
        return x

    def snapshot(self):
        now = time.time()
        with self.lock:
            s = dict(self.session)
            s["uptime_s"] = now - self.t0
            s["cache_src"] = dict(self.session["cache_src"])
            g = {k: self.base[k] + s[k] for k in COUNTERS}
            g["cache_src"] = {k: self.base["cache_src"].get(k, 0) + s["cache_src"].get(k, 0)
                              for k in set(self.base["cache_src"]) | set(s["cache_src"])}
            jobs = []
            for j in self.jobs.values():
                jobs.append(self._job_view(j, now))
            while self.ended and self.ended[0][0] < now - 5:
                self.ended.popleft()
            sm = sorted(self.step_ms)
            return {
                "name": self.name, "time": now, "runs": self.runs, "first_start": self.first_start,
                "rate": {"tok_s_1": self._rate(1, now), "tok_s_10": self._rate(10, now),
                         "tok_s_60": self._rate(60, now),
                         "step_ms": sm[len(sm) // 2] if sm else 0,
                         "tokens_per_step": s["gen_tokens"] / s["seq_steps"] if s["seq_steps"] else 0},
                "jobs": jobs, "ended": [x for _, x in self.ended], "recent": list(self.last_done)[::-1],
                "session": s, "global": g, "records": dict(self.records), "session_records": dict(self.srec),
                "engine": self.engine, "gpus": self.gpus, "history": list(self.hist)[-300:],
            }

    def save(self):
        try:
            snap = self.snapshot()
            g = snap["global"]
            g["records"] = snap["records"]
            g["runs"] = self.runs
            g["first_start"] = self.first_start
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            json.dump(g, open(tmp, "w"), indent=1)
            os.replace(tmp, self.path)
        except Exception as ex:
            print(f"metrics: cannot save {self.path}: {ex}", flush=True)

    def prometheus(self):
        s = self.snapshot()
        se, en = s["session"], s["engine"] or {}
        out = []

        def m(name, val, help_, typ="gauge", labels=""):
            out.append(f"# HELP qengine_{name} {help_}\n# TYPE qengine_{name} {typ}\n"
                       f"qengine_{name}{labels} {val}")
        m("generated_tokens_total", se["gen_tokens"], "tokens generated since start", "counter")
        m("prompt_tokens_total", se["prompt_tokens"], "prompt tokens since start", "counter")
        m("cached_tokens_total", se["cached_tokens"], "prompt tokens served from the cache", "counter")
        m("prefill_tokens_total", se["prefill_tokens"], "prompt tokens computed", "counter")
        m("requests_total", se["requests"], "requests since start", "counter")
        m("energy_joules_total", round(se["energy_j"], 1), "GPU energy since start", "counter")
        m("decode_tokens_per_second", round(s["rate"]["tok_s_10"], 2), "decode rate (10 s)")
        m("active_sequences", sum(1 for j in s["jobs"] if j["state"] != "queued"), "sequences running")
        m("queued_requests", sum(1 for j in s["jobs"] if j["state"] == "queued"), "requests waiting")
        if en:
            m("kv_pool_tokens", en["pool_tokens"], "KV pool capacity (tokens)")
            m("kv_used_tokens", sum(x["pos"] for x in en["slots"]), "tokens held in the KV pool")
        for gpu in s["gpus"]:
            lb = f'{{gpu="{gpu["i"]}"}}'
            out.append(f'qengine_gpu_memory_used_bytes{lb} {gpu["mem"]}\nqengine_gpu_power_watts{lb} {gpu["power"]}\n'
                       f'qengine_gpu_temperature_celsius{lb} {gpu["temp"]}\nqengine_gpu_utilization_percent{lb} {gpu["util"]}')
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------ HTTP (separate port)
    def serve(self, host, port):
        me = self
        page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                b = body.encode() if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                if self.path.split("?")[0] == "/api/reset":
                    me.reset_session()
                    self._send(200, json.dumps({"ok": True}), "application/json")
                else:
                    self._send(404, "not found", "text/plain")

            def do_GET(self):
                p = self.path.split("?")[0]
                if p in ("/", "/dashboard"):
                    self._send(200, open(page, "rb").read(), "text/html; charset=utf-8")
                elif p == "/api/stats":
                    self._send(200, json.dumps(me.snapshot()), "application/json")
                elif p == "/metrics":
                    self._send(200, me.prometheus(), "text/plain; version=0.0.4")
                elif p == "/api/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            self.wfile.write(f"data: {json.dumps(me.snapshot())}\n\n".encode())
                            self.wfile.flush()
                            time.sleep(0.5)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                else:
                    self._send(404, "not found", "text/plain")

        try:
            srv = ThreadingHTTPServer((host, port), H)
        except OSError as ex:
            print(f"metrics: port {port} unavailable ({ex}); monitoring off", flush=True)
            return
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"monitoring: http://{host}:{port}/  (qetop, /api/stats, /metrics)", flush=True)
