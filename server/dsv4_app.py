"""DSV4 server: DeepSeek-V4.1-Flash from a GGUF split set (e.g. Maya-S) on the QEngine DSV4 engine (libqdsv4.so).

OpenAI-compatible /v1/chat/completions (streaming and not), /v1/models, /health. The engine holds one active sequence;
several requests are served time-sliced ([dsv4-rel6-sched]: prompt chunks and decode slices in turn, idle ones parked in
the KV pool / host archive and back without a prefill); a new prompt continues the held sequence or an idle
conversation from the host archive ([dsv4-rel1-arch]), anything else starts over. DeepSeek-V4.1 prompt format (dsv4_chat.py), reasoning in reasoning_content, DSML tool calls as tool_calls.
Started by server/app.py when --model is a DSV4 GGUF directory (the other models' server is unchanged).
"""
import argparse
import asyncio
import ctypes
import json
import os
import queue
import sys
import threading
import time
import uuid

import dsv4_chat as C
import dsv4_gguf as G
import dsv4_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 1 << 30


def mem_available():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


from dsv4_engine import CLib as Engine  # noqa: E402  (the ctypes wrapper lives in dsv4_engine.py: [shared-srv])


class Server:
    def __init__(self, a, version_info=None):
        self.a = a
        self.tok, _, self.kv = G.load_tokenizer(a.model, a.cache_dir)
        self.eos = self.tok.token_to_id(C.EOS)
        smp = (self.kv.get("general.sampling.temp"), self.kv.get("general.sampling.top_p"))
        self.def_t = a.temperature if a.temperature is not None else (smp[0] if smp[0] is not None else 1.0)
        self.def_p = a.top_p if a.top_p is not None else (smp[1] if smp[1] is not None else 1.0)
        args = f"--gguf {a.model} --ctx {a.max_ctx} --dspark {0 if a.no_dspark else 1}"
        if a.tier_reset:
            args += " --tier-reset 1"
        args += f" --archive-gib {a.archive_gib}"   # [dsv4-rel1-arch]
        if a.kv_pool_mib:   # [dsv4-rel4-p2]
            args += f" --kv-pool-mib {int(a.kv_pool_mib)}"
        if a.cache_dir:
            args += f" --cache-dir {a.cache_dir}"
        if a.route_margin is not None:   # [dsv4-dec7-mg] opt-in, approximates the routing; <= 0 = off (wins over the env)
            if not (a.route_margin <= 1.0):
                raise SystemExit("--route-margin: a value <= 1 (<= 0 = off)")
            args += f" --car-margin {a.route_margin if a.route_margin > 0 else -1}"
        if a.drop_w is not None:   # [dsv4-rel2-dropw] 0 = exact routing (every top-6 choice runs)
            if not (0.0 <= a.drop_w <= 1.5):
                raise SystemExit("--drop-w: a value in [0, 1.5] (0 = exact routing)")
            args += f" --drop-w {a.drop_w}"
        if a.engine_args:
            args += " " + a.engine_args
        # a fixed expert ranking at every start (greedy text = f(prompt, settings, files)); --engine-args "--profile
        # learned" opts in to the persisted learned profile
        prof = os.path.join(ROOT, "data", "dsv4", "expert_profile_maya_s.txt")
        if "--profile" not in (a.engine_args or "") and os.path.exists(prof):
            args += f" --profile {prof}"
        ma = mem_available()
        if 0 <= ma < 24 * GIB:   # start check: the engine sizes its host tier to what is free, but this is too little
            print(f"[dsv4] warning: MemAvailable {ma / GIB:.1f} GiB: the engine keeps an 8 GiB margin and reads the "
                  "experts without a host copy from the GGUF files (slow below ~100 GiB)", flush=True)
        t0 = time.time()
        self.e = Engine(a.engine_lib, args)
        print(f"[dsv4] engine ready in {time.time() - t0:.0f} s; context {self.e.max_ctx}; sampling defaults "
              f"temperature {self.def_t} top_p {self.def_p}", flush=True)
        self.cur = None   # [dsv4-rel6-sched] the job whose conversation the engine holds
        self.jid = 0
        self.pfx_cache = {}   # [dsv4-rel6-xprefix] system part -> its tokens
        # review rel6 M3: --tier-reset promises greedy text independent of other requests: only one at a time can keep it
        self.sched = self.e.has_sched and a.archive_gib > 0 and not a.serial and not a.tier_reset
        a.slice_passes = max(1, a.slice_passes)   # (review rel6 L1)
        a.max_active = max(1, a.max_active)
        print(f"[dsv4] requests: {'time-sliced (prompt chunks of %d tokens, decode slices of %d passes)' % (a.prefill_chunk, a.slice_passes) if self.sched else 'one at a time'}", flush=True)
        # [dsv4-qetop] /api/stats for qetop on --metrics-port (host-side counters fed by the worker thread)
        self.st = dsv4_stats.Off()
        if a.metrics_port:
            try:
                ver = version_info(a.engine_lib) if version_info else ("dev", "unknown")
            except Exception:
                ver = ("dev", "unknown")
            mode = (f"time-sliced, up to {a.max_active} in service" if self.sched else "one at a time")
            try:   # review L7 / L8: monitoring never stops the start
                if not 0 < a.metrics_port < 65536:
                    raise ValueError(f"--metrics-port {a.metrics_port}: 1..65535 (0 = off)")
                self.st = dsv4_stats.Stats(os.path.expanduser(a.stats_file), a.served_name, self.e.max_ctx, ver, mode)
                self.engine_counters()
                self.st.serve(a.metrics_host, a.metrics_port)
            except Exception as ex:
                print(f"[dsv4] monitoring off: {ex}", flush=True)
                self.st = dsv4_stats.Off()
        self.last_info = 0.0
        self.q = queue.Queue()
        threading.Thread(target=self.worker, daemon=True).start()

    def engine_counters(self, info=None):
        """[dsv4-qetop] the engine's counters for /api/stats; worker thread only (the info calls walk engine state).
        info: the tuple finish() already read, else read here (idle ticks)"""
        if isinstance(self.st, dsv4_stats.Off):
            return
        if info is None:
            info = (self.e.archive_info(), self.e.kvpool_info(), self.e.kvhost_info(), self.e.prefix_info(), self.e.stats())
        (na, ab, src, cnt), (kn, kv, kh, kc), (hp, hc, hn), (pm, pu), es = info
        d = {"held": self.e.L.dsv4_pos(self.e.h),
             "archive": {"entries": na, "bytes": ab, "last_src": src, "saved": cnt[0], "loaded": cnt[1],
                         "dropped": cnt[2], "skipped": cnt[3], "cap_gib": self.a.archive_gib},
             "kvpool": {"parked_vram": kn, "vram_mb_card": kv, "host_mb": kh, "in_place": kc[0], "moved": kc[1],
                        "from_host": kc[2], "evicted": kc[3], "dropped": kc[4]},
             "kvhost": {"pinned_mb": hp, "cache_mb": hc, "reused": hn[0], "new_pinned": hn[1], "new_pageable": hn[2]},
             "prefix": {"made": pm, "used": pu}}
        if es is not None:   # DSV4_STATS=1: engine totals since the start
            d["passes"] = dict(zip(("passes", "tokens", "pass_ms", "lane_ms", "route_ms", "host_gb", "lane_gb"), es))
        self.st.engine(d)

    # ---------------------------------------------------------------- generation (engine thread)
    def worker(self):
        """[dsv4-rel6-sched] Time-sliced service of several requests on the one-active-sequence engine: the jobs in
        service take turns, each turn one prompt chunk (--prefill-chunk tokens) or one decode slice (--slice-passes
        passes); a switch parks the active conversation in the archive and brings the next one back exactly
        (dsv4_resume, no prefill). Alone, a request makes the calls of the one-at-a-time server (one dsv4_start for the
        whole prompt, steps until it ends), so c = 1 is unchanged. Without the archive (or with --serial) requests run
        one after another as before."""
        active = []
        while True:
            try:
                if len(active) >= self.a.max_active:   # review rel6 M2: the rest waits in the queue
                    raise queue.Empty
                job = self.q.get(timeout=1.0) if not active else self.q.get_nowait()
                self.jid += 1
                job["jid"] = self.jid
                active.append(job)
                continue   # (take every queued request before the next turn)
            except queue.Empty:
                if not active:   # [dsv4-rel1-arch] idle: the archive gives host memory back under pressure
                    try:
                        self.e.archive_pressure()
                    except Exception as ex:   # review rel1 L: the worker never dies here
                        print(f"[dsv4] archive pressure check failed: {ex}", flush=True)
                    if time.time() - self.last_info >= 1.0:   # [dsv4-qetop] idle only (never between passes)
                        self.last_info = time.time()
                        try:
                            self.engine_counters()
                        except Exception:
                            pass
                    continue
            job = active.pop(0)
            err = None
            self.st.turn(job)
            try:
                done = self.turn(job, lambda: bool(active) or not self.q.empty())
            except Exception as ex:   # the request ends with an error event, the server goes on
                job["emit"](("error", str(ex)))
                done, err = True, str(ex)
                self.cur = None
                if self.sched:
                    self.e.end_live(job["jid"])
            if done:
                self.st.finished(job, err)
            if not done:
                active.append(job)
            elif job.get("served") and (not self.sched or (not active and self.q.empty())):   # review rel6 L4: only after
                # engine work; one at a time (--serial / --tier-reset): after every request, as before
                try:
                    self.e.end_request()   # [dsv4-rel-tierreset]: after the last response in service (no-op without --tier-reset 1)
                except Exception:   # review rel3 L2: the tier reset also after a failed request
                    pass

    def activate(self, job):
        """[dsv4-rel6-sched] make job's conversation the engine's active one; False = it had to start over (its K/V left
        the archive): then the tokens so far are prefilled again and the next token comes from that prefill"""
        if self.cur is job:
            return True
        sp = dict(job["sampling"])
        job["sw"] = job.get("sw", 0) + 1   # another seed per switch: the sampler restarts from its seed (no repeated draws)
        sp["seed"] = (sp["seed"] + 0x9E3779B1 * job["sw"]) & 0xFFFFFFFF
        self.st.switched(job)
        self.e.set_sampling(sp["temperature"], sp["top_p"], sp["top_k"], sp["min_p"], sp["seed"])
        self.cur = job
        ok = self.e.resume(job["ids"] + job["out"])
        self.e.set_live(job["jid"])
        return ok

    def turn(self, job, others):
        """one turn of a job; True = finished (done or error emitted)"""
        ids, mx, emit, cancel = job["ids"], job["max_tokens"], job["emit"], job["cancel"]
        if "out" not in job:   # first turn
            if cancel.is_set():   # review rel L3: the client left while queued: no prefill (the held sequence stays)
                emit(("done", {"prompt": len(ids), "reused": 0, "completion": 0, "ttft": 0.0, "tok_s": 0.0, "finish": "cancelled"}))
                return True
            if len(ids) + 16 > self.e.max_ctx:
                emit(("error", f"prompt of {len(ids)} tokens exceeds the context ({self.e.max_ctx})"))
                self.st.finished(job, "context overflow")   # (the worker's finished() then finds nothing: review L6)
                return True
            job.update(out=[], fed=0, reuse=None, t0=time.time(), ttft=None, stop_hit=None, stop=False,
                       sm=StopMatch(self.tok, job.get("stop") or [], job.get("thinking", False)) if job.get("stop") else None)
        sched = self.sched
        if cancel.is_set() and job["ttft"] is not None:   # review rel6 M1: no activation (no park / prefill) for a cancelled job
            return self.finish(job)
        job["served"] = True
        if job["ttft"] is None:   # prompt
            if cancel.is_set():
                return self.finish(job)
            self.st.prefill_start(job)
            if self.cur is not job:
                sp = job["sampling"]
                self.e.set_sampling(sp["temperature"], sp["top_p"], sp["top_k"], sp["min_p"], sp["seed"])
                self.cur = job
            ch = self.a.prefill_chunk
            if sched and others() and ch > 0 and job["reuse"] is None and job.get("cache", True):
                job["fed"] = self.e.reusable(ids)   # review rel6 H2: the chunks start after what the cache holds
            if sched and others() and ch > 0 and len(ids) - job["fed"] > ch:   # a chunk while others are in service
                if not job.get("cache", True) and job["reuse"] is None:
                    self.e.L.dsv4_set_nocache(self.e.h)
                if job["reuse"] is not None:   # review rel6x H1: its own earlier part is reusable for this call
                    self.e.end_live(job["jid"])
                self.e.prefix_hint(job.get("pfx", 0))   # [dsv4-rel6-xprefix]
                tp = time.time()
                r = self.e.prefill_part(ids[:job["fed"] + ch])
                job["pf_s"] = job.get("pf_s", 0.0) + time.time() - tp   # [dsv4-qetop] review M1: own prefill time only
                self.e.set_live(job["jid"])
                if job["reuse"] is None:
                    job["reuse"] = r
                job["fed"] += ch
                self.st.prefill_progress(job, job["fed"], job["reuse"])
                return False
            if not job.get("cache", True) and job["reuse"] is None:   # cache_prompt false: no reuse for this request
                self.e.L.dsv4_set_nocache(self.e.h)
            if job["reuse"] is not None:   # review rel6x H1: its own earlier part is reusable for this call
                self.e.end_live(job["jid"])
            self.e.prefix_hint(job.get("pfx", 0))   # [dsv4-rel6-xprefix]
            tp = time.time()
            t, reuse = self.e.start(ids, True)
            job["pf_s"] = job.get("pf_s", 0.0) + time.time() - tp
            if sched:
                self.e.set_live(job["jid"])
            job["ttft"] = time.time() - job["t0"]
            if job["reuse"] is None:
                job["reuse"] = reuse
            self.st.prefill_done(job, job["reuse"], job["pf_s"])
            self.emit_tokens(job, [t])
            job["t1"] = time.time()
        elif not self.activate(job):   # its K/V is gone: prefill what it has, the reply goes on from that token
            if len(ids) + len(job["out"]) + 16 > self.e.max_ctx:
                return self.finish(job)
            t, _ = self.e.start(ids + job["out"], True)
            self.e.set_live(job["jid"])
            self.emit_tokens(job, [t])
        passes = 0
        while not job["stop"] and len(job["out"]) < mx and not cancel.is_set():
            if sched and passes >= self.a.slice_passes and others():
                return False
            new = self.e.step(mx - len(job["out"]))
            passes += 1
            self.st.pass_done(len(new))
            if not new:
                break
            self.emit_tokens(job, new)
        return self.finish(job)

    def emit_tokens(self, job, new):
        out, mx, sm = job["out"], job["max_tokens"], job["sm"]
        k = 0
        for x in new:
            if len(out) >= mx:
                break
            out.append(x)
            k += 1
            if x == self.eos:
                job["stop"] = True
                break
        if sm and (hit := sm.feed(out)) is not None:   # [dsv4-srv-stop]: tokens up to the one that completes the match
            drop = min(k, len(out) - hit[0])   # (only this pass's tokens: earlier ones were emitted)
            k -= drop
            del out[len(out) - drop:]
            job["stop"], job["stop_hit"] = True, hit
        job["emit"](("tokens", new[:k]))
        self.st.tokens(job, k)   # review rel M4: only the tokens kept (max_tokens, EOS), as usage counts them

    def finish(self, job):
        ids, out, sm, hit = job["ids"], job["out"], job["sm"], job["stop_hit"]
        if self.sched:
            self.e.end_live(job["jid"])
        if sm and hit is None and out and (hit := sm.feed(out, final=True)) is not None:   # a pending match at the end
            job["stop"] = True                                                              # (its tokens are out already)
        ttft = job["ttft"] or 0.0
        dt = time.time() - job.get("t1", time.time())
        na, ab, src, cnt = self.e.archive_info()
        kn, kv, kh, kc = self.e.kvpool_info()
        hp, hc, hn = self.e.kvhost_info()
        pm, pu = self.e.prefix_info()
        reuse = job["reuse"] or 0
        print(f"[dsv4] request: prompt {len(ids)}, reused {reuse} ({src or 'none'}), completion {len(out)}, ttft {ttft:.3f} s; "
              f"archive {na} conversations {ab / 1e9:.2f} GB (saved {cnt[0]} loaded {cnt[1]} dropped {cnt[2]} skipped {cnt[3]}); "
              f"KV pool: {kn} parked in VRAM {kv:.0f} MB/card, host {kh:.0f} MB (in place {kc[0]} moved {kc[1]} from host {kc[2]} evicted {kc[3]} "
              f"dropped {kc[4]}; pinned {hp:.0f} MB, cache {hc:.0f} MB, buffers reused {hn[0]} new {hn[1]} pageable {hn[2]}); "
              f"switches {job.get('sw', 0)}; shared prefixes made {pm} used {pu}", flush=True)
        if (st := self.e.stats()) is not None:   # [dsv4-rel6-sched] DSV4_STATS=1: engine totals since the start
            print("[dsv4] stats: passes %d tokens %d pass_ms %.1f lane_ms %.1f route_ms %.1f host_gb %.3f lane_gb %.3f" % tuple(st), flush=True)
        try:   # [dsv4-qetop] the counters just read (no extra engine call but dsv4_pos)
            self.engine_counters(((na, ab, src, cnt), (kn, kv, kh, kc), (hp, hc, hn), (pm, pu), st))
        except Exception:
            pass
        fin = "cancelled" if job["ttft"] is None else ("stop" if job["stop"] else "length")
        job["emit"](("done", {"prompt": len(ids), "reused": reuse, "completion": len(out), "ttft": ttft,
                              "tok_s": (len(out) - 1) / dt if dt > 0 else 0.0, "finish": fin}))
        # [dsv4-srv-stop]: the hold ends at the last token wholly before the stop string (after "done": review L5);
        # not in thinking mode without tools: the next turn drops this reply's reasoning, the snapshot serves it
        if hit is not None and not (job.get("thinking") and not job.get("tools")) and self.cur is job:
            self.e.trim(len(ids) + hit[1])
        return True

    # ---------------------------------------------------------------- request handling
    def prepare(self, body):
        kw = dict(body.get("chat_template_kwargs") or {})
        effort = body.get("reasoning_effort", kw.get("reasoning_effort"))
        thinking = kw.get("enable_thinking", True)
        if effort in ("none", "off", 0):
            thinking, effort = False, None
        if isinstance(body.get("thinking"), dict) and body["thinking"].get("type") == "disabled":
            thinking = False
        prompt = C.encode(body.get("messages") or [], body.get("tools") or None, thinking=thinking, effort=effort)
        ids = self.tok.encode(prompt, add_special_tokens=False).ids
        # [dsv4-rel6-xprefix] the shared part: the system message (with the tools), as tokens of this prompt
        msgs = body.get("messages") or []
        pfx = 0
        if not self.a.no_prefix_share and not self.a.tier_reset and msgs and msgs[0].get("role") == "system" and len(msgs) > 1:
            try:
                key = hash((json.dumps(msgs[0], sort_keys=True), json.dumps(body.get("tools"), sort_keys=True), thinking, effort))
                sp = self.pfx_cache.get(key)   # review rel6x L8: the system part tokenized once
                if sp is None:
                    sp = self.tok.encode(C.encode(msgs[:1], body.get("tools") or None, thinking=thinking, effort=effort),
                                         add_special_tokens=False).ids
                    if len(self.pfx_cache) >= 64:
                        self.pfx_cache.pop(next(iter(self.pfx_cache)))
                    self.pfx_cache[key] = sp
                while pfx < min(len(sp), len(ids)) and sp[pfx] == ids[pfx]:
                    pfx += 1
            except Exception:
                pfx = 0
        pfx = pfx if pfx >= 1024 else 0
        t = body.get("temperature", self.def_t)
        sampling = dict(temperature=0.0 if t is None else float(t),
                        top_p=float(body.get("top_p", self.def_p) or 1.0), top_k=int(body.get("top_k") or 0),
                        min_p=float(body.get("min_p") or 0.0), seed=int(body["seed"]) if body.get("seed") is not None else uuid.uuid4().int & 0xFFFFFFFF)   # review rel L2: seed 0 is a seed
        mx = int(body.get("max_completion_tokens") or body.get("max_tokens") or (self.e.max_ctx - len(ids) - 16))
        return ids, thinking, sampling, max(1, mx), parse_stop(body.get("stop")), pfx

    def submit(self, ids, sampling, mx, stop=None, thinking=False, tools=False, cache=True, pfx=0):
        loop = asyncio.get_running_loop()
        aq = asyncio.Queue()
        cancel = threading.Event()
        job = dict(ids=ids, pfx=pfx, sampling=sampling, max_tokens=mx, cancel=cancel, stop=stop or [], thinking=thinking, tools=tools, cache=cache,
                   emit=lambda ev: loop.call_soon_threadsafe(aq.put_nowait, ev))
        self.st.queued(job)   # [dsv4-qetop] (before the worker can see it)
        self.q.put(job)
        return aq, cancel


class Stream:
    """Incremental text of a completion: reasoning until </think>, then content; a DSML calls block is held back and
    parsed at the end."""

    def __init__(self, tok, thinking, stops=()):
        self.tok, self.thinking, self.stops = tok, thinking, list(stops)
        self.ids, self.text = [], ""
        self.sent_r = self.sent_c = 0

    def add(self, ids):
        self.ids += ids
        t = self.tok.decode(self.ids, skip_special_tokens=False)
        if t.endswith("�"):   # an incomplete UTF-8 sequence: wait for the next token
            return None
        self.text = t
        return self.split(final=False)

    def split(self, final):
        t = self.text.split(C.EOS)[0]
        if self.thinking:
            k = t.find(C.ETHINK)
            if k < 0:
                r, c = t, ""
                if not final:   # a partial "</think>" at the end stays back
                    for j in range(len(C.ETHINK) - 1, 0, -1):   # the longest partial tag
                        if r.endswith(C.ETHINK[:j]):
                            r = r[:-j]
                            break
            else:
                r, c = t[:k], t[k + len(C.ETHINK):]
        else:
            r, c = "", t
        if self.stops:   # [dsv4-srv-stop] (content only, never in a tool-call block)
            k, _ = stop_scan(c, self.stops, final)
            if k >= 0:
                c = c[:k]
            elif not final:   # a partial stop string at the end stays back
                c = c[:len(c) - partial_stop(c, self.stops)]
        start = "\n\n<" + C.DSML
        k = c.find(start)
        if k >= 0:
            c = c[:k]
        elif not final:
            for j in range(len(start) - 1, 0, -1):
                if c.endswith(start[:j]):
                    c = c[:-j]
                    break
        dr, dc = r[self.sent_r:], c[self.sent_c:]
        self.sent_r, self.sent_c = len(r), len(c)
        return dr, dc


def parse_stop(v):
    """[dsv4-srv-stop] body["stop"]: null, a string or a list of up to 4 non-empty strings (OpenAI)."""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or len(v) > 4 or not all(isinstance(x, str) for x in v):
        raise ValueError("stop: a string or a list of up to 4 strings")
    return [x for x in v if x]


def first_stop(text, stops):
    """Index of the earliest stop string in text, -1 = none."""
    ks = [k for k in (text.find(x) for x in stops) if k >= 0]
    return min(ks) if ks else -1


def partial_stop(text, stops):
    """Length of the longest suffix of text that is a proper prefix of a stop string (held back while streaming)."""
    best = 0
    for x in stops:
        for j in range(min(len(x) - 1, len(text)), best, -1):
            if text.endswith(x[:j]):
                best = j
                break
    return best


DSTART = "\n\n<" + C.DSML   # a tool-call block starts with this


def stop_scan(c, stops, final):
    """[dsv4-srv-stop] The earliest stop in the content text c -> (index or -1, pending). Never inside a tool-call block
    (review dec4 M2): the text from its start on is not matched; a match that may still turn into the block's start
    (its text so far is a prefix of DSTART) is pending until more text comes (final: a match)."""
    kd = c.find(DSTART)
    if kd >= 0:
        c = c[:kd]
    k = first_stop(c, stops)
    if k >= 0 and kd < 0 and not final and DSTART.startswith(c[k:]):
        return -1, True
    return k, False


def cut_text(t, thinking, stops):
    """The completion text cut at the first stop string of its content (after </think> in thinking mode)."""
    if not stops:
        return t
    t = t.split(C.EOS)[0]
    c0 = 0
    if thinking:
        k = t.find(C.ETHINK)
        if k < 0:
            return t
        c0 = k + len(C.ETHINK)
    k, _ = stop_scan(t[c0:], stops, True)
    return t if k < 0 else t[:c0 + k]


class StopMatch:
    """[dsv4-srv-stop] Stop strings in the reply's content (after </think> in thinking mode, found in the text as the
    response does; the reasoning and tool-call blocks are never matched), checked after each pass on the decoded tail.
    feed(out) -> None, or (n_emit, n_keep): the tokens through the one that completes the earliest match (emitted; the
    text is cut at the match in the response), and the tokens wholly before the match (what the held sequence keeps)."""

    def __init__(self, tok, stops, thinking):
        self.tok, self.stops = tok, stops
        self.c0 = -1 if thinking else 0   # first content token (-1: still reasoning)
        self.done = 0                     # tokens already checked
        self.w = 4 * max(len(x) for x in stops) + 16   # tokens back (a char can be up to 4 byte tokens: review L2)

    def dec(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False).split(C.EOS)[0]   # review L1: nothing after EOS

    def feed(self, out, final=False):
        if self.c0 < 0:
            lo = max(0, self.done - 16)
            if C.ETHINK not in self.dec(out[lo:]):
                self.done = len(out)
                return None
            self.c0 = next(j for j in range(lo + 1, len(out) + 1) if C.ETHINK in self.dec(out[:j]))
        lo = max(self.c0, self.done - self.w)   # every char of a match that ends in the new tokens
        prev = self.done
        self.done = len(out)
        if first_stop(self.dec(out[lo:]), self.stops) < 0:
            return None
        c = out[self.c0:]
        text = self.dec(c)
        k, pend = stop_scan(text, self.stops, final)
        if pend:   # maybe a tool-call block's start: look again after the next pass
            self.done = prev
            return None
        if k < 0:
            return None
        hit = lambda j: stop_scan(self.dec(c[:j]), self.stops, True)[0] >= 0
        for j0 in (max(0, lo - self.c0 - 1), 0):   # the scan from the window start, else from the first content token
            e = next((j for j in range(j0 + 1, len(c) + 1) if hit(j)), None)
            if e is None:
                continue
            b = next((j for j in range(e, j0 - 1, -1) if len(p := self.dec(c[:j])) <= k and text.startswith(p)), None)
            if b is not None:
                return self.c0 + e, self.c0 + b
        return len(out), self.c0   # (not reached: dec(0) = "")


def build_app(s):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    app = FastAPI()
    name = s.a.served_name

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "qengine",
                                            "max_model_len": s.e.max_ctx}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        try:
            ids, thinking, sampling, mx, stop, pfx = s.prepare(body)
        except (ValueError, KeyError, TypeError) as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error"}}, status_code=400)
        aq, cancel = s.submit(ids, sampling, mx, stop, thinking, bool(body.get("tools")), body.get("cache_prompt", True) is not False, pfx)
        st = Stream(s.tok, thinking, stop)
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

        def finish_msg(info):
            st.text = s.tok.decode(st.ids, skip_special_tokens=False)
            m = C.parse_completion(cut_text(st.text, thinking, stop), thinking=thinking)
            fin = "tool_calls" if m["tool_calls"] else info["finish"]
            calls = [dict(c, id="call_" + uuid.uuid4().hex[:24], index=i) for i, c in enumerate(m["tool_calls"])]
            usage = {"prompt_tokens": info["prompt"], "completion_tokens": info["completion"],
                     "total_tokens": info["prompt"] + info["completion"],
                     "prompt_tokens_details": {"cached_tokens": info["reused"]}}
            return m, fin, calls, usage

        if not body.get("stream"):
            # review rel M5: a client that leaves stops the generation (polled, and on handler cancellation)
            try:
                while True:
                    try:
                        ev, val = await asyncio.wait_for(aq.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if await req.is_disconnected():
                            cancel.set()
                        continue
                    if ev == "tokens":
                        st.ids += val
                    elif ev == "error":
                        return JSONResponse({"error": {"message": val, "type": "server_error"}}, status_code=500)
                    elif ev == "done":
                        m, fin, calls, usage = finish_msg(val)
                        msg = {"role": "assistant", "content": m["content"]}
                        if thinking:
                            msg["reasoning_content"] = m["reasoning_content"]
                        if calls:
                            msg["tool_calls"] = calls
                        return {"id": rid, "object": "chat.completion", "created": created, "model": name,
                                "choices": [{"index": 0, "message": msg, "finish_reason": fin}], "usage": usage,
                                "timings": {"ttft_s": round(val["ttft"], 3), "tok_s": round(val["tok_s"], 2)}}
            finally:
                cancel.set()

        async def gen():
            def chunk(delta, fin=None, **extra):
                d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                d.update(extra)
                return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
            yield chunk({"role": "assistant", "content": ""})
            try:
                ka = float(os.environ.get("QE_SSE_KEEPALIVE") or 10.0)  # (as server/app.py: a keep-alive comment)
                while True:
                    if ka > 0:
                        try:
                            ev, val = await asyncio.wait_for(aq.get(), ka)
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"
                            continue
                    else:
                        ev, val = await aq.get()
                    if ev == "tokens":
                        r = st.add(val)
                        if r:
                            dr, dc = r
                            if dr:
                                yield chunk({"reasoning_content": dr})
                            if dc:
                                yield chunk({"content": dc})
                    elif ev == "error":
                        yield "data: " + json.dumps({"error": {"message": val}}) + "\n\n"
                        break
                    elif ev == "done":
                        st.text = s.tok.decode(st.ids, skip_special_tokens=False)
                        dr, dc = st.split(final=True)
                        if dr:
                            yield chunk({"reasoning_content": dr})
                        if dc:
                            yield chunk({"content": dc})
                        m, fin, calls, usage = finish_msg(val)
                        if calls:
                            yield chunk({"tool_calls": calls})
                        yield chunk({}, fin, usage=usage)
                        break
            finally:
                cancel.set()
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main(argv, version_info=None):
    ap = argparse.ArgumentParser(description="QEngine DSV4 server (DeepSeek-V4.1-Flash GGUF)")
    ap.add_argument("--model", required=True, help="the GGUF split set's directory (e.g. Maya-S)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-ctx", type=int, default=32768)
    ap.add_argument("--served-name", default="deepseek-v4.1-flash-qengine")
    ap.add_argument("--engine-lib", default=os.environ.get("QE_DSV4_LIB") or os.path.join(ROOT, "build_dsv4/libqdsv4.so"))
    ap.add_argument("--engine-args", default="", help="extra engine flags (--dspark-dir, --profile, --cpu-threads ...)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-dspark", action="store_true", help="off-switch: plain decode without the DSpark drafter")
    ap.add_argument("--archive-gib", type=float, default=32.0,
                    help="host RAM for idle conversations (their KV state, LRU; it shrinks when MemAvailable runs low); 0 = off: "
                         "a different conversation drops the held one")
    ap.add_argument("--kv-pool-mib", type=int, default=0,
                    help="VRAM per card for the compressed K/V of idle conversations beyond one --max-ctx sequence (they switch "
                         "back without a host copy); default 0: idle ones use what the active conversation does not (fewer "
                         "expert slots otherwise)")
    ap.add_argument("--tier-reset", action="store_true",
                    help="greedy text independent of earlier unrelated requests (a prompt that continues the held sequence reuses its "
                         "KV): the expert tier back to its post-load state after each request (in the background; -2..-10 %% tok/s)")
    ap.add_argument("--route-margin", type=float, default=None,
                    help="opt-in, default off. APPROXIMATES the routing: a cold expert choice whose router score is within this margin of a "
                         "VRAM-resident expert is replaced by that expert, so a different expert runs than the model chose (decode, DSpark "
                         "verify passes and prompts of <= 8 tokens; the tier then learns from the replaced ids). Measured 11-17 (3 draws, "
                         "tflp NLL gate mean): off 1.623 (sd 0.015), 0.05 1.641, 0.1 1.641; speed 256 tokens +6..+11 %% (0.05), +6..+22 %% "
                         "(0.1). <= 0 = off (also over DSV4_ROUTE_MARGIN, which the engine reads when the flag is absent)")
    ap.add_argument("--drop-w", type=float, default=None,
                    help="adaptive top-k at decode: a 5th / 6th expert choice whose routing weight (of 1.5) is below this value is "
                         "left out (lossy: NLL +0.016 at 0.1, no speed gain measured). Default 0 = exact routing (every chosen expert "
                         "runs); also DSV4_DROP_W")
    ap.add_argument("--prefill-chunk", type=int, default=10240,
                    help="time-sliced service: a prompt is fed in chunks of this many tokens while other requests are in service "
                         "(alone: in one call); 0 = whole prompts. Default = the engine's prefill pass (a prefill call streams most "
                         "experts whatever its size: 2,048-token chunks cost -20 / -10 %% aggregate at c = 4 / 8)")
    ap.add_argument("--slice-passes", type=int, default=16,
                    help="time-sliced service: decode passes of one request before the next one in service takes its turn")
    ap.add_argument("--no-prefix-share", action="store_true",
                    help="off-switch: no shared prompt-prefix entries (by default the system message of a second prompt that "
                         "shares it is kept once and reused by later prompts)")
    ap.add_argument("--serial", action="store_true",
                    help="off-switch: one request at a time to its end (the pre-1.4 behaviour; also with --tier-reset or --archive-gib 0)")
    ap.add_argument("--max-active", type=int, default=8,
                    help="time-sliced service: requests in service at once (more wait in the queue)")
    ap.add_argument("--metrics-port", type=int, default=9393,
                    help="live monitoring for qetop (/api/stats, /api/reset) on this port; 0 = off; a taken port turns "
                         "monitoring off, the API serves on")
    ap.add_argument("--metrics-host", default="127.0.0.1")
    ap.add_argument("--stats-file", default="~/.cache/qengine/stats-dsv4.json",
                    help="all-time monitoring totals of the DSV4 server (qetop's all-time column)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    a, _unknown = ap.parse_known_args(argv)
    s = Server(a, version_info)
    import uvicorn
    try:
        uvicorn.run(build_app(s), host=a.host, port=a.port, log_level="warning")
    finally:
        if isinstance(s.st, dsv4_stats.Stats):   # [dsv4-qetop] all-time totals
            s.st.save()


if __name__ == "__main__":
    main(sys.argv[1:])
