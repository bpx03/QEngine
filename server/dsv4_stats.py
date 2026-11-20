"""[dsv4-qetop] Live monitoring of the DSV4 server for qetop: GET /api/stats (JSON) and POST /api/reset on a separate
port (--metrics-host / --metrics-port, default 127.0.0.1:9393 as server/app.py; 0 = off), never through llama-swap.

The snapshot has the keys qetop reads from the 27B / Flash-Next server (rate, session, global, records, jobs, recent,
ended, gpus, history, ...) plus engine.dsv4 (held context, archive, KV pool, shared prefixes, TTFT). DSV4 has one active
sequence, so a job's state is "queued" (not yet served), "prefill" (prompt being fed, in chunks while others are in
service) or "decode" (in service: decoding or parked for another request's turn); slot 0 = the request whose turn the
engine is on, -1 = the others.

Threading: the hooks are called by the server's worker thread (and queued() by the event loop), take a short lock and
do host-side bookkeeping only; no hook calls the engine. The engine's counters (archive, KV pool, prefixes) are read
by the worker thread where it already reads them (end of a request, idle ticks) and handed over with engine();
the HTTP thread only copies Python data. A failing hook is reported once and never reaches the request path.
"""
import collections
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COUNTERS = ("requests", "errors", "prompt_tokens", "cached_tokens", "prefill_tokens", "prefill_s", "gen_tokens",
            "decode_s", "steps", "pass_tokens", "ttft_s", "ttft_n", "energy_j", "uptime_s")
RECORDS = ("peak_tok_s", "max_agents", "longest_context")


def _safe(f):
    """a hook never raises into the server (the first failure of each hook is printed)"""
    def g(self, *a, **k):
        try:
            return f(self, *a, **k)
        except Exception as ex:
            if f.__name__ not in self._failed:
                self._failed.add(f.__name__)
                print(f"[dsv4] stats: {f.__name__} failed ({ex!r}); monitoring may be incomplete", flush=True)
    g.__name__ = f.__name__
    return g


class Off:
    """--metrics-port 0: every hook is a no-op"""

    def __getattr__(self, name):
        return lambda *a, **k: None


class Stats:
    def __init__(self, path, name, max_ctx, version=("dev", "unknown"), mode="", nvml=True):
        self.lock = threading.Lock()
        self._failed = set()
        self.path, self.name, self.max_ctx = path, name, max_ctx
        self.version, self.build = version
        self.mode = mode
        self.t0 = time.time()
        self.session = dict.fromkeys(COUNTERS, 0)
        self.base = dict.fromkeys(COUNTERS, 0)   # all-time totals before this run (the stats file)
        self.base_records = dict.fromkeys(RECORDS, 0)
        self.runs, self.first_start = 1, self.t0
        try:
            g = json.load(open(path))
            for k in COUNTERS:
                self.base[k] = g.get(k, 0)
            self.base_records.update({k: g.get("records", {}).get(k, 0) for k in RECORDS})
            self.runs = g.get("runs", 0) + 1
            self.first_start = g.get("first_start", self.t0)
        except Exception:
            pass
        self.records = dict(self.base_records)
        self.srec = dict.fromkeys(RECORDS, 0)
        self.decode_ev = collections.deque()   # (t, tokens), last 60 s
        self.hist = collections.deque(maxlen=600)   # 1 s buckets
        self.cur = self._bucket(int(self.t0))
        self.jobs = {}   # id(job) -> live view
        self.n = 0   # arrivals (labels)
        self.current = None   # id(job) of the request whose turn the engine is on
        self.last_done = collections.deque(maxlen=12)
        self.ended = collections.deque()   # (t, frozen view), last 10 s
        self.eng = {}
        self.gpus = []
        self._energy = {}
        self._nvml = nvml
        threading.Thread(target=self._loop, daemon=True, name="dsv4-stats").start()

    @staticmethod
    def _bucket(sec):
        return {"t": sec, "dec": 0, "pre": 0, "active": 0, "queued": 0, "power": 0.0}

    # ------------------------------------------------------------ hooks (event loop: queued; the rest: worker thread)
    @_safe
    def queued(self, job):
        with self.lock:
            self.n += 1
            # (a job dict lives until its request ends, so id() is unique while it is here: review L5)
            self.jobs[id(job)] = {"label": f"#{self.n}", "state": "queued", "prompt": len(job["ids"]), "t_q": time.time(),
                                  "gen": 0, "cached": 0, "slot": -1, "switches": 0, "ev": collections.deque()}
            self.session["requests"] += 1

    @_safe
    def turn(self, job):
        """the engine works on job now (its prompt chunk or decode slice)"""
        with self.lock:
            self.current = id(job)

    @_safe
    def prefill_start(self, job):
        with self.lock:
            j = self.jobs.get(id(job))
            if j and j["state"] == "queued":
                j["state"], j["t_p"], j["prefill_tok"] = "prefill", time.time(), 0

    @_safe
    def prefill_progress(self, job, fed, reused):
        """a prompt chunk was fed: fed tokens of the prompt are in, reused of them came from the cache"""
        with self.lock:
            j = self.jobs.get(id(job))
            if j and j["state"] == "prefill":
                j["prefill_tok"] = max(fed - (reused or 0), 0)
                j["cached"] = reused or 0

    @_safe
    def prefill_done(self, job, reused, pf_s=None):
        """pf_s: seconds of the job's own prefill calls (time-sliced: other requests' turns excluded)"""
        now = time.time()
        with self.lock:
            j = self.jobs.get(id(job))
            if not j:
                return
            t_p = j.get("t_p", now)
            dt = max(pf_s if pf_s is not None else now - t_p, 1e-6)
            comp = max(j["prompt"] - reused, 0)
            ttft = now - j["t_q"]   # from the arrival, queue wait included (the reply's timings.ttft_s: from its first turn)
            j.update(state="decode", t_d=now, cached=reused, prefill_tok=comp, prefill_tps=comp / dt, prefill_s=dt,
                     ttft=ttft)
            s = self.session
            s["prompt_tokens"] += j["prompt"]
            s["cached_tokens"] += reused
            s["prefill_tokens"] += comp
            s["prefill_s"] += dt
            s["ttft_s"] += ttft
            s["ttft_n"] += 1
            for r in (self.records, self.srec):
                r["longest_context"] = max(r["longest_context"], j["prompt"])
            t0, t1 = int(t_p), int(now)   # the prefilled tokens spread over the seconds it took (history chart)
            for b in list(self.hist) + [self.cur]:
                if t0 <= b["t"] <= t1:
                    b["pre"] += comp / (t1 - t0 + 1)

    @_safe
    def tokens(self, job, k):
        """k tokens of job's reply were emitted"""
        if k <= 0:
            return
        now = time.time()
        with self.lock:
            self.decode_ev.append((now, k))
            self.cur["dec"] += k
            self.session["gen_tokens"] += k
            j = self.jobs.get(id(job))
            if j:
                j["gen"] += k
                ev = j["ev"]
                ev.append((now, k))
                while ev[0][0] < now - 10:
                    ev.popleft()

    @_safe
    def pass_done(self, n):
        """one engine decode pass gave n tokens (DSpark: accepted drafts + 1)"""
        with self.lock:
            self.session["steps"] += 1
            self.session["pass_tokens"] += n

    @_safe
    def switched(self, job):
        with self.lock:
            j = self.jobs.get(id(job))
            if j:
                j["switches"] += 1

    @_safe
    def finished(self, job, error=None):
        now = time.time()
        with self.lock:
            j = self.jobs.pop(id(job), None)
            if self.current == id(job):
                self.current = None
            if error is not None:
                self.session["errors"] += 1
            if not j:
                return
            self.ended.append((now, self._job_view(j, now)))
            if j["state"] == "decode":
                self.last_done.append({"label": j["label"], "prompt": j["prompt"], "cached": j["cached"], "gen": j["gen"],
                                       "tps": j["gen"] / max(now - j["t_d"], 1e-6), "prefill_tps": j.get("prefill_tps", 0),
                                       "ttft": j.get("ttft"), "t": now, "error": str(error) if error else ""})

    @_safe
    def engine(self, info):
        """the engine's counters, read by the worker thread (a dict, see Server.engine_counters)"""
        with self.lock:
            self.eng = dict(info, updated=time.time())

    # ------------------------------------------------------------ reset (POST /api/reset, qetop Ctrl+R)
    def reset_session(self):
        """the session's counters move into the all-time totals, then the session starts over (records stay)"""
        now = time.time()
        with self.lock:
            for k in COUNTERS:
                self.base[k] += (now - self.t0) if k == "uptime_s" else self.session[k]
            self.session = dict.fromkeys(COUNTERS, 0)
            self.t0 = now
            self.decode_ev.clear()
            self.hist.clear()
            self.cur = self._bucket(int(now))
            self.last_done.clear()
            self.srec = dict.fromkeys(RECORDS, 0)
            for j in self.jobs.values():   # requests in service count from now on
                j["gen"] = 0
                j["ev"].clear()
                if j["state"] == "decode":
                    j["t_d"] = now
        self.save()

    # ------------------------------------------------------------ background: 1 s buckets, NVML, the stats file
    def _gpu_poll(self, nv):
        gpus, power = [], 0.0
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            m = nv.nvmlDeviceGetMemoryInfo(h)
            p = nv.nvmlDeviceGetPowerUsage(h) / 1000
            e = nv.nvmlDeviceGetTotalEnergyConsumption(h) / 1000   # J since the driver load
            with self.lock:
                if i in self._energy:
                    self.session["energy_j"] += max(e - self._energy[i], 0)
                self._energy[i] = e
            gpus.append({"i": i, "mem": m.used, "mem_total": m.total, "util": nv.nvmlDeviceGetUtilizationRates(h).gpu,
                         "power": p, "limit": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
                         "temp": nv.nvmlDeviceGetTemperature(h, 0), "clock": nv.nvmlDeviceGetClockInfo(h, 1)})
            power += p
        return gpus, power

    def _loop(self):
        nv = None
        if self._nvml:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import pynvml
                pynvml.nvmlInit()
                nv = pynvml
            except Exception:
                pass
        last_save = time.time()
        while True:
            time.sleep(1.0 - (time.time() % 1.0))
            try:
                self.tick(nv)
            except Exception as ex:
                if "tick" not in self._failed:
                    self._failed.add("tick")
                    print(f"[dsv4] stats: background tick failed ({ex!r})", flush=True)
            if time.time() - last_save > 30:
                self.save()
                last_save = time.time()

    def tick(self, nv=None):
        now = time.time()
        gpus, power = [], 0.0
        if nv:
            try:
                gpus, power = self._gpu_poll(nv)
            except Exception:
                pass
        with self.lock:
            self.gpus = gpus
            c = self.cur
            c["power"] = power
            c["active"] = sum(1 for j in self.jobs.values() if j["state"] != "queued")
            c["queued"] = len(self.jobs) - c["active"]
            if c["active"]:
                self.session["decode_s"] += 1
            self.hist.append(c)
            self.cur = self._bucket(int(now))
            while self.decode_ev and self.decode_ev[0][0] < now - 60:
                self.decode_ev.popleft()
            tps10 = self._rate(10, now)
            act = c["active"]
            for r in (self.records, self.srec):
                r["peak_tok_s"] = max(r["peak_tok_s"], tps10)
                r["max_agents"] = max(r["max_agents"], act)

    # ------------------------------------------------------------ views
    def _rate(self, sec, now):
        return sum(n for t, n in self.decode_ev if t >= now - sec) / sec

    def _job_view(self, j, now):
        x = {k: v for k, v in j.items() if k not in ("ev", "t_q", "t_p", "t_d")}
        x["age_s"] = now - j["t_q"]
        if j["state"] == "queued":
            x["wait_s"] = now - j["t_q"]
        elif j["state"] == "prefill":
            x["elapsed_s"] = now - j.get("t_p", now)
        else:
            d = max(now - j["t_d"], 1e-6)
            x["tps"] = j["gen"] / d   # since its first token (time-sliced: other requests' turns included)
            win = min(10.0, d)
            x["tps_now"] = sum(c for t, c in j["ev"] if t >= now - win) / win
        return x

    def snapshot(self):
        now = time.time()
        with self.lock:
            s = dict(self.session)
            s["uptime_s"] = now - self.t0
            g = {k: self.base[k] + s[k] for k in COUNTERS}
            jobs = []
            for key, j in self.jobs.items():
                x = self._job_view(j, now)
                x["slot"] = 0 if key == self.current and j["state"] != "queued" else -1
                jobs.append(x)
            while self.ended and self.ended[0][0] < now - 10:
                self.ended.popleft()
            ended = [x for _, x in self.ended]
            recent = list(self.last_done)[::-1]
            eng = dict(self.eng)
            cj = self.jobs.get(self.current)
            if cj and cj["state"] == "decode":   # review M2: the held context live during a long request
                eng["held"] = cj["prompt"] + cj["gen"]
            hist = list(self.hist)[-300:]
            gpus = list(self.gpus)
            records, srec = dict(self.records), dict(self.srec)
            rate = {"tok_s_1": self._rate(1, now), "tok_s_10": self._rate(10, now), "tok_s_60": self._rate(60, now),
                    "step_ms": 0,
                    "tokens_per_step": s["pass_tokens"] / s["steps"] if s["steps"] else 0}
        last_ttft = next((r["ttft"] for r in recent if r.get("ttft") is not None), None)
        eng.update(max_ctx=self.max_ctx, mode=self.mode, last_ttft_s=last_ttft,
                   avg_ttft_s=s["ttft_s"] / s["ttft_n"] if s["ttft_n"] else None)
        return {"name": self.name, "version": self.version, "build": self.build, "time": now, "runs": self.runs,
                "first_start": self.first_start, "max_ctx": self.max_ctx, "rate": rate, "jobs": jobs, "ended": ended,
                "recent": recent, "session": s, "global": g, "records": records, "session_records": srec,
                "engine": {"dsv4": eng}, "gpus": gpus, "history": hist}

    def save(self):
        try:
            snap = self.snapshot()
            g = dict(snap["global"], records=snap["records"], runs=self.runs, first_start=self.first_start)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.{threading.get_ident()}.tmp"   # review L4: save() may run on several threads
            with open(tmp, "w") as f:
                json.dump(g, f, indent=1)
            os.replace(tmp, self.path)
        except Exception as ex:
            print(f"[dsv4] stats: cannot save {self.path}: {ex}", flush=True)

    # ------------------------------------------------------------ HTTP (separate port)
    def serve(self, host, port):
        """GET /api/stats, POST /api/reset; a taken port is logged and the API serves on without monitoring"""
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                b = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path.split("?")[0] == "/api/stats":
                    self._send(200, json.dumps(me.snapshot()), "application/json")
                else:
                    self._send(404, "not found", "text/plain")

            def do_POST(self):
                if self.path.split("?")[0] == "/api/reset":
                    me.reset_session()
                    self._send(200, json.dumps({"ok": True}), "application/json")
                else:
                    self._send(404, "not found", "text/plain")

        try:
            srv = ThreadingHTTPServer((host, port), H)
        except OSError as ex:
            print(f"[dsv4] monitoring: port {port} unavailable ({ex}); monitoring off, the API serves on", flush=True)
            return None
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True, name="dsv4-metrics").start()
        print(f"[dsv4] monitoring: http://{host}:{port}/api/stats (qetop --url http://{host}:{port})", flush=True)
        return srv
