"""[shared-srv] DSV4 (DeepSeek-V4.1-Flash GGUF) behind server/app.py's engine interface.

app.py's Generator drives a slot engine (server/engine.py: start_begin / prefill_more / start_finish, step / step_batch,
set_limit, trim, release, pool_info, the grammar buffers). The DSV4 C API (src/dsv4/dsv4_capi.cpp) holds one active
sequence and parks the others (VRAM K/V pool, host archive), so Dsv4Engine presents that interface in Python:

- virtual slots = the requests in service (--max-active; one with --serial / --tier-reset / --archive-gib 0);
- a prompt alone is one dsv4_start (the calls of the one-at-a-time server, so c = 1 is unchanged); beside others in
  service it goes in chunks of the scheduler's budget through dsv4_prefill_part, starting after what the cache holds;
- decode passes are time-sliced: step_batch gives one pass to the slot whose turn it is (the others get no tokens and
  report "paused", so the no-progress guard waits), --slice-passes passes per turn; a switch is dsv4_resume (no prefill),
  or a re-prefill of what the reply has when its K/V left the archive;
- sampling per slot (applied at every activation; another seed per switch, as dsv4_app), set_limit = the pass cap
  ([dsv4-dec3-cap]), trim = dsv4_trim of the held sequence ([dsv4-srv-stop]), the last release ends the request
  (--tier-reset runs after the response), the system-part hint for cross-sequence prefix sharing ([dsv4-rel6-xprefix]);
- grammar (structured output, tool_choice, the thinking budget): a constrained slot decodes one token per pass through
  dsv4_step_logits (a plain step whose logits come back; the DSpark window stays current); the mask is applied and the
  token sampled here, then fed as the pending token (dsv4_set_next). Without that C call the grammar group is off.

The model's prompt format (dsv4_chat.py) and tokenizer (dsv4_gguf.py) reach app.py through model_profile.py's "dsv4"
family. Extension point for other single-sequence engines (gpt-oss, src/gptoss): a family names its adapter class
(model_profile FAMILIES[...]["adapter"]) and its renderer; Generator builds that adapter instead of engine.Engine.
"""
import ctypes
import os
import time

import numpy as np

from engine import SequenceFault  # (CtxNoRoom: an engine fault, a stream ends with its error event; review kb M1)
from steering import SteeringMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 1 << 30
# [dsv4-rel8-optgram] a constrained slot with a grammar checker (app.py: Constraint.legal_prefix) decodes on DSpark passes:
# the pass's tokens are checked in order on the host, the first illegal one is redone through the mask (greedy only: the
# same tokens as one masked step per token; sampled replies keep the masked steps: review M1).
# QE_DSV4_GRAM_PASS=0: one masked step per token (the path before)
GRAM_PASS = os.environ.get("QE_DSV4_GRAM_PASS", "1") != "0"


def mem_available():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


class CtxNoRoom(SequenceFault):
    """[kb-dsv4-ctxgrow] the request's context K/V could not grow (lazy reservation, VRAM taken by another process):
    this request fails, the engine goes on (it aborted before)"""


class PrefillCancelled(SequenceFault):
    """[srv-disconnect] the request's cancel flag stopped its prefill at a chunk start (dsv4_start / dsv4_prefill_part
    returned -5): the engine holds nothing now; the scheduler releases the request"""


class CLib:
    """ctypes wrapper of src/dsv4/dsv4_capi.cpp (one sequence); used by Dsv4Engine and the legacy server/dsv4_app.py."""

    def __init__(self, lib, args):
        self.L = L = ctypes.CDLL(lib)
        L.dsv4_create.restype = ctypes.c_void_p
        L.dsv4_create.argtypes = [ctypes.c_char_p]
        for f in ("dsv4_start", "dsv4_step", "dsv4_pos", "dsv4_max_ctx"):
            getattr(L, f).restype = ctypes.c_int
        L.dsv4_start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        L.dsv4_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.dsv4_pos.argtypes = [ctypes.c_void_p]
        if hasattr(L, "dsv4_stats"):   # [dsv4-rel6-sched] DSV4_STATS=1 (benches)
            L.dsv4_stats.restype = ctypes.c_int
            L.dsv4_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.has_sched = hasattr(L, "dsv4_resume")   # [dsv4-rel6-sched] (an older library: one request at a time)
        if self.has_sched:
            L.dsv4_resume.restype = ctypes.c_int
            L.dsv4_resume.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            L.dsv4_prefill_part.restype = ctypes.c_int
            L.dsv4_prefill_part.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            L.dsv4_reusable.restype = ctypes.c_int
            L.dsv4_reusable.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            L.dsv4_set_live.argtypes = [ctypes.c_void_p, ctypes.c_int]
            L.dsv4_end_live.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.has_abort = hasattr(L, "dsv4_set_abort")   # [srv-disconnect] a client gone stops the prefill at a chunk
        if self.has_abort:
            L.dsv4_set_abort.restype = None
            L.dsv4_set_abort.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.has_follow = hasattr(L, "dsv4_prefill_follow")   # [dsv4-xcall]
        if self.has_follow:
            L.dsv4_prefill_follow.restype = None
            L.dsv4_prefill_follow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.has_pfx = hasattr(L, "dsv4_set_prefix_hint")   # [dsv4-rel6-xprefix]
        if self.has_pfx:
            L.dsv4_set_prefix_hint.argtypes = [ctypes.c_void_p, ctypes.c_int]
            L.dsv4_prefix_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long)]
        L.dsv4_max_ctx.argtypes = [ctypes.c_void_p]
        L.dsv4_set_sampling.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                        ctypes.c_ulonglong]
        L.dsv4_reset.argtypes = [ctypes.c_void_p]
        L.dsv4_end_request.restype = ctypes.c_double   # [dsv4-rel-tierreset]
        L.dsv4_end_request.argtypes = [ctypes.c_void_p]
        L.dsv4_trim.restype = ctypes.c_int   # [dsv4-srv-stop]
        L.dsv4_trim.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.dsv4_archive_info.restype = ctypes.c_int   # [dsv4-rel1-arch]
        L.dsv4_archive_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_long)]
        L.dsv4_archive_pressure.argtypes = [ctypes.c_void_p]
        L.dsv4_set_nocache.argtypes = [ctypes.c_void_p]
        L.dsv4_kvpool_info.restype = ctypes.c_int   # [dsv4-rel4-p2]
        L.dsv4_kvpool_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_long)]
        self.has_kvhost = hasattr(L, "dsv4_kvhost_info")   # [dsv4-rel5-p3]
        if self.has_kvhost:
            L.dsv4_kvhost_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                           ctypes.POINTER(ctypes.c_long)]
        # [shared-srv] grammar: a plain step whose logits come back, and the pending token set by the caller
        self.has_logits = hasattr(L, "dsv4_step_logits") and hasattr(L, "dsv4_set_next") and hasattr(L, "dsv4_vocab")
        if self.has_logits:
            L.dsv4_step_logits.restype = ctypes.c_int
            L.dsv4_step_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
            L.dsv4_set_next.restype = ctypes.c_int
            L.dsv4_set_next.argtypes = [ctypes.c_void_p, ctypes.c_int]
        for f in ("dsv4_vocab", "dsv4_eos"):
            if hasattr(L, f):
                getattr(L, f).restype = ctypes.c_int
                getattr(L, f).argtypes = []
        self.has_idle = hasattr(L, "dsv4_ctx_idle")   # [dsv4-rel8-giveback]
        if self.has_idle:
            L.dsv4_ctx_idle.restype = ctypes.c_int
            L.dsv4_ctx_idle.argtypes = [ctypes.c_void_p]
        L.dsv4_reset.restype = None
        self.h = L.dsv4_create(args.encode())
        if not self.h:
            raise RuntimeError("dsv4_create failed (see the messages above)")
        self.max_ctx = L.dsv4_max_ctx(self.h)
        self.out = (ctypes.c_int * 32)()

    def set_sampling(self, t, top_p, top_k, min_p, seed):
        self.L.dsv4_set_sampling(self.h, float(t), float(top_p), int(top_k), float(min_p), int(seed) & ((1 << 64) - 1))

    def start(self, ids, cache=True):
        if not cache:   # [dsv4-rel1-arch] cache_prompt false: no reuse for this request
            self.L.dsv4_set_nocache(self.h)
        a = (ctypes.c_int * len(ids))(*ids)
        reuse = ctypes.c_int(0)
        t = self._armed(self.L.dsv4_start, self.h, a, len(ids), ctypes.byref(reuse))
        if t == -2:
            self._lost()
        if t == -5:
            self._cancelled()
        return t, reuse.value

    abort = None   # [srv-disconnect] the cancel flag (ctypes.c_int) of the prompt being prefilled (Dsv4Engine.prefill_more)

    def _armed(self, fn, *a):
        """fn(*a) with the engine reading self.abort before every prefill chunk (another thread sets it to 1 when the
        client leaves); only start / prefill_part: a step that completes a pending part never stops (review M1: an
        abort there would escape dsv4_step)"""
        if not self.has_abort or self.abort is None:
            return fn(*a)
        self.L.dsv4_set_abort(self.h, ctypes.addressof(self.abort))
        try:
            return fn(*a)
        finally:
            self.L.dsv4_set_abort(self.h, None)

    def _cancelled(self):
        if self.on_lost:   # (the engine holds the prefix done, no request owns it: as after a failed start)
            self.on_lost()
        raise PrefillCancelled("cancelled while its prompt was prefilled (the client left)")

    on_lost = None   # [kb-dsv4-ctxgrow] the scheduler's hook: the engine holds no conversation after a failed start

    def _lost(self):
        if self.on_lost:
            self.on_lost()
        raise CtxNoRoom("no VRAM for this request's context K/V (another process took it; a smaller --max-ctx avoids it)")

    def stats(self):   # [dsv4-rel6-sched] passes, tokens, pass ms, lane ms, route ms, host GB, lane GB; None = off
        if not hasattr(self.L, "dsv4_stats"):
            return None
        v = (ctypes.c_double * 7)()
        return list(v) if self.L.dsv4_stats(self.h, v) else None

    def resume(self, ids):   # [dsv4-rel6-sched] ids = tokens + the pending one of a parked reply: active again, no prefill
        a = (ctypes.c_int * len(ids))(*ids)
        return bool(self.L.dsv4_resume(self.h, a, len(ids)))

    def reusable(self, ids):   # [dsv4-rel6-sched] tokens a start of ids would reuse (requests in service excluded)
        a = (ctypes.c_int * len(ids))(*ids)
        return self.L.dsv4_reusable(self.h, a, len(ids))

    def prefix_hint(self, n):   # [dsv4-rel6-xprefix] the next prompt's first n tokens are a part other prompts may share
        if self.has_pfx:   # (review rel6x L1: 0 too)
            self.L.dsv4_set_prefix_hint(self.h, int(n))

    def prefix_info(self):
        if not self.has_pfx:
            return 0, 0
        m, u = ctypes.c_long(0), ctypes.c_long(0)
        self.L.dsv4_prefix_info(self.h, ctypes.byref(m), ctypes.byref(u))
        return m.value, u.value

    def set_live(self, jid):   # [dsv4-rel6-sched] the held conversation belongs to request jid while it is in service
        self.L.dsv4_set_live(self.h, int(jid))

    def end_live(self, jid):
        self.L.dsv4_end_live(self.h, int(jid))

    def prefill_part(self, ids, follow=0):   # [dsv4-rel6-sched] a prompt prefix fed, nothing pending; -> tokens reused
        # [dsv4-xcall] follow = prompt tokens later prefill calls still feed (the engine may defer layers 21-39)
        if self.has_follow:
            self.L.dsv4_prefill_follow(self.h, max(0, int(follow)))
        a = (ctypes.c_int * len(ids))(*ids)
        reuse = ctypes.c_int(0)
        r = self._armed(self.L.dsv4_prefill_part, self.h, a, len(ids), ctypes.byref(reuse))
        if r == -2:
            self._lost()
        if r == -5:
            self._cancelled()
        if r < 0:
            raise RuntimeError("context overflow")
        return reuse.value

    def archive_info(self):   # [dsv4-rel1-arch] (entries, bytes, last start's source, [saved, loaded, dropped, skipped])
        b, src, c = ctypes.c_double(0), ctypes.c_int(0), (ctypes.c_long * 4)()
        n = self.L.dsv4_archive_info(self.h, ctypes.byref(b), ctypes.byref(src), c)
        return n, b.value, ("", "held", "snapshot", "archive")[src.value], list(c)

    def kvpool_info(self):   # [dsv4-rel4-p2] (parked in VRAM, their MB per card, host MB, [in place, moved, from host, evicted, dropped])
        v, hm, c = ctypes.c_double(0), ctypes.c_double(0), (ctypes.c_long * 5)()
        n = self.L.dsv4_kvpool_info(self.h, ctypes.byref(v), ctypes.byref(hm), c)
        return n, v.value, hm.value, list(c)

    def kvhost_info(self):   # [dsv4-rel5-p3] (pinned MB in use + cache, cache MB, [reused, new pinned, new pageable])
        if not self.has_kvhost:
            return 0.0, 0.0, [0, 0, 0]
        pm, cm, c = ctypes.c_double(0), ctypes.c_double(0), (ctypes.c_long * 3)()
        self.L.dsv4_kvhost_info(self.h, ctypes.byref(pm), ctypes.byref(cm), c)
        return pm.value, cm.value, list(c)

    def archive_pressure(self):
        self.L.dsv4_archive_pressure(self.h)

    def ctx_idle(self):   # [dsv4-rel8-giveback] between requests: a stale lazy-context loan back (1 = at the base)
        return self.L.dsv4_ctx_idle(self.h) if self.has_idle else 1

    def end_request(self):   # the tier reset (--tier-reset 1) runs in the background after the response
        return self.L.dsv4_end_request(self.h)

    def trim(self, keep):   # [dsv4-srv-stop]: hold only the first `keep` tokens (prompt + reply); False = hold dropped
        return bool(self.L.dsv4_trim(self.h, int(keep)))

    def step(self, cap=32):   # [dsv4-dec3-cap]: the engine accepts at most cap tokens (what max_tokens still allows)
        n = self.L.dsv4_step(self.h, self.out, max(1, min(32, cap)))
        if n == -2:   # (nothing fed: the held conversation stays as it was)
            raise CtxNoRoom("no VRAM for this request's context K/V (another process took it; a smaller --max-ctx avoids it)")
        return [self.out[i] for i in range(n)]

    def step_logits(self):   # [shared-srv] feeds the pending token (no drafts); its step's logits [V] (None: context full)
        if not hasattr(self, "_lg"):
            import numpy as np
            self._lg = np.empty(self.L.dsv4_vocab(), dtype=np.float32)
        n = self.L.dsv4_step_logits(self.h, self._lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if n == -2:
            raise CtxNoRoom("no VRAM for this request's context K/V (another process took it; a smaller --max-ctx avoids it)")
        return self._lg[:n] if n > 0 else None

    def set_next(self, t):   # [shared-srv] the pending token (after a prefill_part, or the token sampled from step_logits)
        return self.L.dsv4_set_next(self.h, int(t))


def engine_args(a):
    """The dsv4_create argument string of the server flags (as dsv4_app.Server built it)."""
    args = f"--gguf {a.model} --ctx {a.max_ctx} --dspark {0 if a.no_dspark else 1}"
    if getattr(a, "ctx_static", 0):   # [dsv4-rel8-ctxdef] the default context with the static reservation (equal to the lazy one since 11-19)
        args += f" --ctx-static {int(a.ctx_static)}"
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
    return args


class _Seq:
    """One virtual slot: a request in service."""
    __slots__ = ("ids", "jid", "hint", "fed", "reuse", "first", "out", "samp", "sw", "limit", "done", "served",
                 "cache", "mask", "sio", "constrained", "ban", "pf_s", "trim_ok", "rng", "banmask", "trim_lost", "check",
                 "gbuf", "abort")

    def __init__(self, ids, jid, hint, limit):
        self.ids, self.jid, self.hint = list(ids), jid, hint
        self.fed, self.reuse, self.first, self.out = 0, None, None, []
        self.samp = (0.0, 1.0, 0, 0.0, 0)
        self.sw, self.limit, self.done, self.served, self.cache = 0, limit, False, False, True
        self.mask = self.sio = None
        self.constrained, self.ban, self.pf_s = False, (), 0.0
        self.trim_ok, self.rng, self.banmask = True, None, None
        self.trim_lost = False   # [dsv4-rel8-hang] a banned-token redo's trim failed once (its hold may go again)
        self.abort = None        # [srv-disconnect] the request's cancel flag (ctypes.c_int; set_abort_flag)
        self.check, self.gbuf = None, None   # [dsv4-rel8-optgram] the grammar checker; its mask scratch


class Dsv4Engine(SteeringMixin):
    """The slot interface of server/engine.Engine over libqdsv4 (see the module docstring). Scheduler thread only."""

    def __init__(self, a, vocab=None):
        self.a = a
        self.lib_path = a.engine_lib
        a.slice_passes = max(1, int(getattr(a, "slice_passes", 16)))   # (review rel6 L1)
        a.max_active = max(1, int(getattr(a, "max_active", 8)))
        ma = mem_available()
        if 0 <= ma < 24 * GIB:   # start check: the engine sizes its host tier to what is free, but this is too little
            print(f"[dsv4] warning: MemAvailable {ma / GIB:.1f} GiB: the engine keeps an 8 GiB margin and reads the "
                  "experts without a host copy from the GGUF files (slow below ~100 GiB)", flush=True)
        t0 = time.time()
        self.c = c = CLib(a.engine_lib, engine_args(a))
        self.lib, self.h = c.L, c.h   # (SteeringMixin: the library has no steering calls: none, default only)
        print(f"[dsv4] engine ready in {time.time() - t0:.0f} s; context {c.max_ctx}", flush=True)
        if getattr(a, "ctx_static", 0) and a.ctx_static == a.max_ctx:   # one default for both reservations (user 11-19)
            print(f"[dsv4] max context {c.max_ctx} (default without --max-ctx: {a.max_ctx} with either context reservation; "
                  f"the engine's 'context:' line names the one this start uses); --max-ctx N sets it", flush=True)
            if c.max_ctx < a.max_ctx:
                print(f"[dsv4] WARNING: max context {c.max_ctx} instead of the default {a.max_ctx} (the load could not map "
                      f"the context buffers in full: see the engine's warning)", flush=True)
        elif getattr(a, "ctx_static", 0):   # [dsv4-rel8-ctxdef] the start-up line: which default and why
            why = ("the lazy context reservation (peer access between every pair of cards)" if c.max_ctx >= a.max_ctx else
                   "the static context reservation (no peer access between every pair of cards, or DSV4_NOCTXLAZY=1)"
                   if c.max_ctx == a.ctx_static else "the VRAM the load could map (see the engine's warning)")
            print(f"[dsv4] max context {c.max_ctx} (default without --max-ctx: {a.max_ctx} with the lazy reservation, "
                  f"{a.ctx_static} with the static one; this start: {why}); --max-ctx N sets it", flush=True)
        elif c.max_ctx < a.max_ctx:   # [dsv4-rel8-ctxcap] an explicit --max-ctx the load could not map in full
            print(f"[dsv4] WARNING: max context {c.max_ctx} instead of the requested {a.max_ctx} (the load could not map the "
                  f"context buffers in full: see the engine's warning)", flush=True)
        # review rel6 M3: --tier-reset promises greedy text independent of other requests: only one at a time keeps it
        self.sched = c.has_sched and a.archive_gib > 0 and not a.serial and not a.tier_reset
        print(f"[dsv4] requests: {'time-sliced (prompt chunks of %d tokens, decode slices of %d passes, up to %d in service)' % (a.prefill_chunk, a.slice_passes, a.max_active) if self.sched else 'one at a time'}", flush=True)
        self.max_t = self.max_t_engine = c.max_ctx
        self.pool_tokens = c.max_ctx
        self.kv_in_ram = False
        # the engine refuses a step 16 tokens before its context end: the scheduler's end check (+8) lands there
        self.slots = [max(64, c.max_ctx - 9)] * (a.max_active if self.sched else 1)
        self._vocab = c.L.dsv4_vocab() if hasattr(c.L, "dsv4_vocab") else vocab
        self.eos = c.L.dsv4_eos() if hasattr(c.L, "dsv4_eos") else 1
        self.par = {"mode": "dsv4", "why": "one active sequence; requests time-sliced" if self.sched else "one request at a time"}
        self.refused = {}
        self.fns = set()
        self.caps = {"chunked": True, "batch": True, "cache": True, "archive": True, "monitor": True, "stops": True,
                     "grammar": c.has_logits, "logprobs": False, "vision": False,
                     "qe_ban_tokens": c.has_logits}   # tool_choice none: DSpark steps, a banned token redone masked
        self.seqs = {}         # slot -> _Seq (in service)
        self.jid = 0
        self.held = None       # the slot whose conversation the engine holds (None: none of the ones in service)
        c.on_lost = self._lost_hold
        self.slot_faults = {}  # [kb-dsv4-ctxgrow] slot -> why its reply failed in a decode pass (app.faults ends it)
        self.n_faults = 0
        self.want = None       # resume(slot): the single-step path's slot
        self.turn, self.passes = None, 0
        self.bound = None      # the last start_begin's slot (set_sampling applies to it)
        self.last_src = ""
        self.counters = {"switches": 0, "reprefills": 0, "chunks": 0, "constrained_steps": 0,
                         "gram_passes": 0, "gram_tokens": 0, "gram_viol": 0,   # [dsv4-rel8-optgram]
                         "refill_tokens": 0, "refill_s": 0.0}
        self.refill_new = []   # [dsv4-rel8-refill] (slot, tokens, seconds, in a prompt's prefill) since take_refills
        self._in_pf = False    # prefill_more is running (its redo's time is already in the prompt's prefill time)
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------ capabilities (engine.Engine's interface)
    def has(self, feature):
        return bool(self.caps.get(feature, False))

    def missing_optional(self):
        return [g for g, ok in self.caps.items() if not ok]

    def par_info(self):
        return dict(self.par)

    def set_exit_hook(self, fn):
        return False

    def vocab(self):
        return self._vocab

    def warmup(self, greedy=False, sampling=True):
        return 0.0

    def set_concurrent(self, on=True):
        pass

    def set_lookahead(self, on):
        pass

    def can_prefill_multi(self):
        return False

    def prefill_pass_bound(self):
        return False

    def prefill_multi(self, slots, budget, caps=None):
        return None

    def add_corpus(self, ids):
        pass

    def set_no_lookup(self, slot, off=True):
        pass

    def set_prio(self, slot, prio):
        pass

    def set_penalty(self, presence=0.0, frequency=0.0, seed=()):
        return False   # the DSV4 sampler has no presence / frequency penalty

    def watch_stops(self, slot, on=True):
        pass

    def set_no_ahead(self, slot, on):
        pass

    def drafts(self, slot):
        return []   # (DSpark drafts are made inside the pass: nothing to look at ahead)

    def fault_count(self):
        return self.n_faults

    def slot_fault(self, slot):
        return self.slot_faults.get(slot)

    def _lost_hold(self):
        self.held = None

    def pause_state(self):
        return None

    def pause_for(self, n, prio):
        return False

    def pool_now(self):
        return self.pool_tokens

    def persist(self):
        pass

    def disk_flush(self, seconds):
        return True

    def num_disk_archives(self):
        return 0

    def num_pinned(self):
        return 0

    def last_copied(self):
        return 0

    def last_from_archive(self):
        return self.last_src == "archive"

    def num_archives(self):
        return self.c.archive_info()[0]

    def archive_more(self, budget=64 << 20):
        return True

    def mem_guard(self):
        self.c.archive_pressure()   # [dsv4-rel1-arch] the archive gives host memory back under pressure

    def idle_tick(self):
        """[dsv4-rel8-giveback] (rel7 review L5) the server is idle (nothing in service or queued; app.py calls it from
        its idle wait only, review rel8 L2): the lazy context's stale loan goes back now, not inside the next short
        request's first call (--tier-reset: its reset after each request gives it back)"""
        if not self.seqs and not self.a.tier_reset and hasattr(self.c, "ctx_idle"):
            self.c.ctx_idle()

    def reset(self):
        self.c.L.dsv4_reset(self.h)
        self.held = None

    def close(self):
        pass   # (the process exits; dsv4_destroy is not called, as dsv4_app)

    # ------------------------------------------------------------------ admission
    def can_admit(self, n):
        return len(self.seqs) < len(self.slots) and n <= self.max_t

    def probe_cached(self, ids):
        return self.c.reusable(list(ids)) if ids and self.c.has_sched else 0

    # ------------------------------------------------------------------ prefill
    def start_begin(self, ids, k=3, affinity=0, hint=0, mm=None):
        if mm is not None:
            raise RuntimeError("this engine has no image input")
        from engine import StartRefused
        if len(ids) + 16 > self.max_t:
            raise StartRefused(StartRefused.TOO_LONG)
        slot = next((i for i in range(len(self.slots)) if i not in self.seqs), None)
        if slot is None:
            raise StartRefused(StartRefused.NO_ROOM)
        self.jid += 1
        self.seqs[slot] = _Seq(ids, self.jid, int(hint or 0), self.max_t)
        self.bound = slot
        return slot

    def set_trim(self, slot, on):
        """[dsv4-srv-stop] False: no stop trim for this reply (thinking without tools: the next turn drops the reasoning,
        the prompt-end snapshot serves it; a failed trim would drop the hold and the snapshot with it)"""
        s = self.seqs.get(slot)
        if s is not None:
            s.trim_ok = bool(on)

    def set_no_cache(self, slot):
        """cache_prompt false: the request's first engine call reuses nothing."""
        s = self.seqs.get(slot)
        if s is not None:
            s.cache = False

    def set_sampling(self, temperature, top_p=1.0, top_k=40, min_p=0.0, seed=0):
        s = self.seqs.get(self.bound)
        if s is not None:
            s.samp = (float(temperature), float(top_p), int(top_k or 0), float(min_p or 0.0), int(seed) & 0xFFFFFFFF)

    def _apply_sampling(self, s, bump):
        t, p, k, m, seed = s.samp
        if bump:   # another seed per switch: the sampler restarts from its seed (no repeated draws)
            s.sw += 1
            seed = (seed + 0x9E3779B1 * s.sw) & 0xFFFFFFFF
        self.c.set_sampling(t, p, k, m, seed)

    def call_tokens(self, kind, slots=()):
        """[dsv4-rel8-hang] (review L4) tokens the scheduler's next engine call may prefill, for its hang deadline (app.py
        call_limit: QE_HANG_S + tokens / QE_HANG_PREFILL_TPS; 0 = QE_HANG_S). "prefill": the longest prompt rest in
        flight (one dsv4_start may run all of it). "step": a pass of a conversation the engine does not hold may
        re-prefill all of it (its K/V left the archive), a tool_choice none slot's banned-token redo too once a trim of it
        failed (review rel8 L7: trims normally succeed);
        a pass of the held one is a decode pass."""
        if kind == "prefill":
            return max((len(s.ids) - s.fed for s in self.seqs.values() if s.first is None), default=0)
        live = [x for x in slots if x in self.seqs and self.seqs[x].first is not None]
        if len(live) > 1:   # step_batch: the pass runs the slot whose turn it is
            live = [self._next_turn(live)[0]]
        n = 0
        for x in live:
            s = self.seqs[x]
            # (review rel8 L7: a redo re-prefills only after a failed trim; optgram review L4: a grammar redo too)
            if x != self.held or ((s.ban or s.check is not None) and s.trim_lost):
                n = max(n, len(s.ids) + len(s.out))
        return n

    def prefill_pos(self, slot):
        s = self.seqs.get(slot)
        return -1 if s is None or s.first is not None else s.fed

    def num_prefills(self):
        return sum(1 for s in self.seqs.values() if s.first is None)

    def set_abort_flag(self, slot, flag):
        """[srv-disconnect] the request's cancel flag (ctypes.c_int, 1 = its client left): its prefill calls stop at the
        next chunk start (one dsv4_start runs a whole prompt alone in service: without it a cancel waited for all of it)"""
        s = self.seqs.get(slot)
        if s is not None:
            s.abort = flag

    def prefill_more(self, budget, slot=None):
        self._in_pf = True   # [dsv4-rel8-refill] review rel8 L3
        if slot is None:
            slot = max((i for i, s in self.seqs.items() if s.first is None), key=lambda i: self.seqs[i].jid)
        s = self.seqs.get(slot)
        ab = s.abort if s is not None and getattr(self.c, "has_abort", False) else None
        self.c.abort = ab
        try:
            return self._prefill_more(budget, slot)
        except PrefillCancelled:   # [srv-disconnect] the scheduler releases the request next (the engine keeps the prefix)
            if s is not None:   # (the rows done, for the cancel's log line: one dsv4_start does not advance s.fed)
                s.fed = max(s.fed, min(len(s.ids), int(self.c.L.dsv4_pos(self.c.h))))
            return False
        finally:
            self._in_pf = False
            self.c.abort = None

    def _prefill_more(self, budget, slot=None):
        """[dsv4-rel6-sched] the next chunk of the slot's prompt; True once the prompt is done (its first token
        sampled). Alone in service (or the rest fits the budget): the whole rest in one dsv4_start."""
        if slot is None:
            slot = max((i for i, s in self.seqs.items() if s.first is None), key=lambda i: self.seqs[i].jid)
        s = self.seqs[slot]
        if s.first is not None:
            return True
        c, ids = self.c, s.ids
        s.served = True
        if self.held != slot:
            self._apply_sampling(s, False)
        others = any(i != slot for i in self.seqs)
        ch = int(budget) if budget and budget > 0 else 0
        if self.sched and others and 0 < ch < (1 << 30):
            if s.reuse is None and s.cache:
                s.fed = c.reusable(ids)   # review rel6 H2: the chunks start after what the cache holds
            if len(ids) - s.fed > ch:
                if not s.cache and s.reuse is None:
                    c.L.dsv4_set_nocache(c.h)
                if s.reuse is not None:   # review rel6x H1: its own earlier part is reusable for this call
                    c.end_live(s.jid)
                c.prefix_hint(s.hint)   # [dsv4-rel6-xprefix]
                tp = time.time()
                # [dsv4-xcall] what the later calls feed: the rest (a constrained slot's last token goes through a step)
                r = c.prefill_part(ids[:s.fed + ch], len(ids) - (s.fed + ch) - (1 if s.constrained else 0))
                s.pf_s += time.time() - tp
                c.set_live(s.jid)
                self.held = slot
                if s.reuse is None:
                    s.reuse = r
                s.fed += ch
                self.counters["chunks"] += 1
                return False
        if not s.cache and s.reuse is None:   # cache_prompt false: no reuse for this request
            c.L.dsv4_set_nocache(c.h)
        if s.reuse is not None:   # review rel6x H1
            c.end_live(s.jid)
        c.prefix_hint(s.hint)
        tp = time.time()
        if s.constrained:   # the first token goes through the mask too: the prompt without its last token, then a
            r = c.prefill_part(ids[:-1]) if len(ids) > 1 else 0   # constrained step feeds that token
            self.held = slot   # (shared_server review L2: the engine holds this prompt now, also if the masked step raises)
            c.set_next(ids[-1])
            t = self._masked_step(s)
            reuse = r
        else:
            t, reuse = c.start(ids, True)
            if t >= 0 and s.ban and t in s.ban:   # a banned first token: the prompt's end decoded again through the mask
                self.held = slot
                s.out = []
                self._back_to(s, len(ids))
                t = self._masked_step(s, s.banmask)
                self.counters["ban_redo"] = self.counters.get("ban_redo", 0) + 1
        s.pf_s += time.time() - tp
        if t < 0:
            raise RuntimeError("context overflow")
        if self.sched:
            c.set_live(s.jid)
        self.held = slot
        if s.reuse is None:
            s.reuse = reuse
        s.fed, s.first, s.out = len(ids), t, [t]
        self.last_src = c.archive_info()[2]
        return True

    def start_finish(self, slot=None):
        s = self.seqs[slot]
        self.want = slot   # (the engine holds this prompt's conversation now, as the slot engines bind it)
        if self.held == slot:   # its first slice right away, no park / unpark (review L5; as dsv4_app)
            self.turn, self.passes = slot, 0
        return s.first, s.reuse or 0, slot

    # ------------------------------------------------------------------ decode
    def set_limit(self, slot, tokens):
        s = self.seqs.get(slot)
        if s is not None:
            s.limit = max(0, int(tokens))

    def resume(self, slot):
        self.want = slot

    def _activate(self, slot):
        """[dsv4-rel6-sched] the slot's conversation becomes the engine's active one: None = it was there or came back
        exactly (dsv4_resume), else the token of a re-prefill of what it has (its K/V left the archive), -1 = no room"""
        if self.held == slot:
            return None
        s, c = self.seqs[slot], self.c
        self._apply_sampling(s, True)
        self.counters["switches"] += 1
        self.held = slot
        seq = s.ids + s.out
        ok = c.resume(seq)
        if ok:   # (on a miss the engine still holds another conversation: its live id stays: review L1)
            if self.sched:   # (review rel9 L2: end_live runs only in the time-sliced mode)
                c.set_live(s.jid)
            return None
        self.counters["reprefills"] += 1
        if len(seq) + 16 > self.max_t:
            return -1
        tp = time.time()
        if s.constrained:   # the state back with its pending token; the pass then steps through the mask
            r = c.prefill_part(seq[:-1])
            c.set_next(seq[-1])
            if self.sched:
                c.set_live(s.jid)
            self._refilled(slot, len(seq) - 1 - max(0, r or 0), tp)
            return None
        t, reuse = c.start(seq, True)
        if self.sched:
            c.set_live(s.jid)
        self._refilled(slot, len(seq) - max(0, reuse or 0), tp)
        return t

    def _refilled(self, slot, n, tp):
        """[dsv4-rel8-refill] (qetop review L2) a re-prefill is prefill work: its tokens (those not reused) and seconds go
        to the prefill counters (app.py takes them after the call: take_refills) and to the DSV4 panel's refills"""
        dt = time.time() - tp
        self.counters["refill_tokens"] += max(0, n)
        self.counters["refill_s"] += dt
        self.refill_new.append((slot, max(0, n), dt, self._in_pf))

    def take_refills(self):
        """the re-prefills since the last call: [(slot, tokens, seconds, inside a prefill_more)] (scheduler thread)"""
        r, self.refill_new = self.refill_new, []
        return r

    def _pass(self, slot):
        s = self.seqs.get(slot)
        if s is None or s.first is None or s.limit <= 0:
            return []
        try:
            return self._pass_i(slot, s)
        except CtxNoRoom as ex:   # [kb-dsv4-ctxgrow] only this reply fails (app.faults); the others in service go on
            self.slot_faults[slot] = str(ex)
            self.n_faults += 1
            s.limit = 0
            return []

    def _pass_i(self, slot, s):
        r = self._activate(slot)
        if r is not None:
            toks = [] if r < 0 else [r]
        elif s.constrained and s.check is not None and GRAM_PASS and (s.samp[0] <= 0 or s.samp[2] == 1):
            # (optgram review M1: greedy only; sampled, the engine's top-k / top-p cut over all tokens differs from the
            # masked distribution, so sampled replies keep one masked step per token)
            toks = self._gram_pass(s)
        elif s.constrained:
            t = self._masked_step(s)
            toks = [] if t < 0 else [t]
        else:
            toks = self.c.step(min(32, s.limit))
            if s.ban and any(t in s.ban for t in toks):
                toks = self._unban(s, toks)
        toks = toks[:s.limit]
        s.out += toks
        s.limit -= len(toks)
        return toks

    def step(self):
        slot = self.want if self.want in self.seqs else self.turn
        if slot not in self.seqs:
            slot = next((i for i, s in self.seqs.items() if s.first is not None), None)
            if slot is None:
                return []
        self.turn = slot
        return self._pass(slot)

    def step_batch(self, slots):
        """[dsv4-rel6-sched] one decode pass of the slot whose turn it is; --slice-passes passes, then the next one"""
        live = [x for x in slots if x in self.seqs and self.seqs[x].first is not None]
        if not live:
            return [[] for _ in slots]
        self.turn, self.passes = self._next_turn(live)
        self.passes += 1
        toks = self._pass(self.turn)
        return [toks if x == self.turn else [] for x in slots]

    def _next_turn(self, live):
        """(slot, passes) of step_batch's next pass: the turn moves on after --slice-passes passes"""
        if self.turn not in live or self.passes >= self.a.slice_passes:
            nxt = [x for x in sorted(live) if self.turn is None or x > self.turn]
            return (nxt[0] if nxt else sorted(live)[0]), 0
        return self.turn, self.passes

    def trim(self, slot, pos):
        """[dsv4-srv-stop] the held sequence keeps its first pos tokens (prompt + reply up to the stop)"""
        if self.held != slot or slot not in self.seqs or not self.seqs[slot].trim_ok:
            return False
        return self.c.trim(int(pos))

    def release(self, slot):
        self.slot_faults.pop(slot, None)
        s = self.seqs.pop(slot, None)
        if s is None:
            return
        if self.want == slot:
            self.want = None
        if self.turn == slot:
            self.turn, self.passes = None, 0
        if self.sched:
            self.c.end_live(s.jid)
        if self.held == slot:
            self.held = None   # (the engine still holds it: a continuation reuses it)
        self._log(s)
        if s.served and not self.seqs:
            try:
                self.c.end_request()   # [dsv4-rel-tierreset]: after the last response in service (no-op without --tier-reset 1)
            except Exception:
                pass

    def _log(self, s):
        """the per-request line of dsv4_app (smoke scripts and the archive check read it)"""
        try:
            na, ab, src, cnt = self.c.archive_info()
            kn, kv, kh, kc = self.c.kvpool_info()
            hp, hc, hn = self.c.kvhost_info()
            pm, pu = self.c.prefix_info()
            print(f"[dsv4] request: prompt {len(s.ids)}, reused {s.reuse or 0} ({self.last_src or 'none'}), completion "
                  f"{max(0, len(s.out))}, prefill {s.pf_s:.3f} s; archive {na} conversations {ab / 1e9:.2f} GB (saved {cnt[0]} "
                  f"loaded {cnt[1]} dropped {cnt[2]} skipped {cnt[3]}); KV pool: {kn} parked in VRAM {kv:.0f} MB/card, host "
                  f"{kh:.0f} MB (in place {kc[0]} moved {kc[1]} from host {kc[2]} evicted {kc[3]} dropped {kc[4]}; pinned "
                  f"{hp:.0f} MB, cache {hc:.0f} MB, buffers reused {hn[0]} new {hn[1]} pageable {hn[2]}); switches {s.sw}; "
                  f"shared prefixes made {pm} used {pu}", flush=True)
            k = self.counters
            if k["gram_passes"] or k["constrained_steps"]:   # [dsv4-rel8-optgram] totals since the start
                print(f"[dsv4] grammar: passes {k['gram_passes']} tokens {k['gram_tokens']} redone {k['gram_viol']} masked steps "
                      f"{k['constrained_steps']} reprefills {k['reprefills']}", flush=True)
            if (st := self.c.stats()) is not None:
                print("[dsv4] stats: passes %d tokens %d pass_ms %.1f lane_ms %.1f route_ms %.1f host_gb %.3f lane_gb %.3f"
                      % tuple(st), flush=True)
        except Exception as ex:
            print(f"[dsv4] request line: {ex!r}", flush=True)

    # ------------------------------------------------------------------ monitoring
    def pool_info(self, convs=True):
        """engine.dsv4 (qetop's DSV4 panel) + the slot list the scheduler reads (slots not on turn: paused)"""
        c = self.c
        (na, ab, src, cnt), (kn, kv, kh, kc), (hp, hc, hn), (pm, pu) = (c.archive_info(), c.kvpool_info(),
                                                                       c.kvhost_info(), c.prefix_info())
        cur = self.turn if self.turn in self.seqs else self.held
        slots = []
        for i in range(len(self.slots)):
            s = self.seqs.get(i)
            slots.append({"off": 0, "cap": self.slots[i], "pos": len(s.ids) + len(s.out) if s else 0, "active": s is not None,
                          "used": 0, "prio": 1, "paused": bool(s is not None and s.first is not None and i != cur),
                          "vram": False})
        d = {"held": c.L.dsv4_pos(c.h), "max_ctx": self.max_t,
             "mode": (f"time-sliced, up to {self.a.max_active} in service" if self.sched else "one at a time"),
             "archive": {"entries": na, "bytes": ab, "last_src": src, "saved": cnt[0], "loaded": cnt[1],
                         "dropped": cnt[2], "skipped": cnt[3], "cap_gib": self.a.archive_gib},
             "kvpool": {"parked_vram": kn, "vram_mb_card": kv, "host_mb": kh, "in_place": kc[0], "moved": kc[1],
                        "from_host": kc[2], "evicted": kc[3], "dropped": kc[4]},
             "kvhost": {"pinned_mb": hp, "cache_mb": hc, "reused": hn[0], "new_pinned": hn[1], "new_pageable": hn[2]},
             "prefix": {"made": pm, "used": pu}, "sched": dict(self.counters),
             "refills": {"count": self.counters["reprefills"], "tokens": self.counters["refill_tokens"],
                         "s": round(self.counters["refill_s"], 3)}}
        es = c.stats()
        if es is not None:   # DSV4_STATS=1: engine totals since the start
            d["passes"] = dict(zip(("passes", "tokens", "pass_ms", "lane_ms", "route_ms", "host_gb", "lane_gb"), es))
        return {"pool_tokens": 0, "max_context": self.max_t, "slots": slots, "conversations": [],
                "archives": {"ram": na, "ram_bytes": ab, "disk": 0, "disk_bytes": 0, "pinned_bytes": hp * 1e6, "pinned_used": 0},
                "events": {"grow": 0, "move": kc[1], "evict": kc[3], "pause": 0, "resume": cnt[1]}, "paused": 0, "dsv4": d}

    # ------------------------------------------------------------------ grammar (structured output, tool_choice, budget)
    def mask_buffer(self, slot):
        """uint32 bitmasks [8][words] (bit t = token t allowed); row 0 is the next token's mask (one token per pass)"""
        s = self.seqs[slot]
        if s.mask is None:
            s.mask = np.full((8, (self._vocab + 31) // 32), 0xFFFFFFFF, np.uint32)
        return s.mask

    def spec_io(self, slot):
        """[0] drafts the model made (none: a constrained pass is one token), [16] drafts to verify (ignored)"""
        s = self.seqs[slot]
        if s.sio is None:
            s.sio = np.zeros(32, np.int32)
        return s.sio

    def constrain(self, slot, on=True):
        if on and not self.caps["grammar"]:
            raise RuntimeError("this DSV4 library has no dsv4_step_logits (grammar off)")
        self.mask_buffer(slot)
        self.seqs[slot].constrained = bool(on)
        self.seqs[slot].check = None   # [dsv4-rel8-optgram] set again by set_check (any other mask writer: masked steps)

    def set_check(self, slot, fn):
        """[dsv4-rel8-optgram] fn(tokens, buf) -> (k, bad): how many of tokens the slot's grammar takes in order from its
        current state (state unchanged); bad = token k is refused, and buf then holds the mask at k. None = masked steps"""
        s = self.seqs.get(slot)
        if s is not None:
            s.check = fn
            if fn is not None and s.gbuf is None:
                s.gbuf = np.zeros((self._vocab + 31) // 32, np.uint32)

    def _gram_pass(self, s):
        """[dsv4-rel8-optgram] a DSpark pass of a constrained slot: the legal prefix of its tokens, then (at the first
        illegal one) the held sequence back there and one masked step"""
        base = len(s.ids) + len(s.out)   # (s.out's last token is this pass's input)
        toks = self.c.step(min(32, s.limit))
        if not toks:
            return []
        k, bad = s.check(toks, s.gbuf)
        self.counters["gram_passes"] += 1
        self.counters["gram_tokens"] += k
        if not bad:
            if k < len(toks) and not self.c.trim(base + k):
                # the reply ends inside the pass (end token, grammar complete): the hold ends there too. A failed trim
                # dropped the hold (optgram review L8): a later pass (ignore_eos) re-prefills it through _activate.
                # (trim_ok is never False with a grammar: response_format turns thinking off, tool_choice has tools)
                s.trim_lost = True
                self.held = None
            return toks[:k]
        self.counters["gram_viol"] += 1
        s.out += toks[:k]
        self._back_to(s, base + k)
        del s.out[len(s.out) - k:]
        t = self._masked_step(s, s.gbuf)
        return toks[:k] + ([] if t < 0 else [t])

    def constrained(self, slot):
        s = self.seqs.get(slot)
        return bool(s is not None and s.constrained)

    def ban_tokens(self, slot, ids):
        """[tool-choice-ban] tool_choice none: tokens the reply never gets. The slot keeps its DSpark passes; a pass that
        returns a banned token keeps the tokens before it, the held sequence goes back there (dsv4_trim, else a
        re-prefill) and that position is decoded once more through the mask (review M1)"""
        s = self.seqs[slot]
        s.ban = frozenset(int(t) for t in ids)
        s.banmask = None
        if s.ban:
            m = np.full((self._vocab + 31) // 32, 0xFFFFFFFF, np.uint32)
            for t in s.ban:
                m[t >> 5] &= np.uint32(~(1 << (t & 31)) & 0xFFFFFFFF)
            s.banmask = m

    def _back_to(self, s, n):
        """the held sequence = the first n tokens of the slot's (prompt + reply), the last one pending"""
        seq = s.ids + s.out
        if self.c.trim(n):
            return
        tp = time.time()
        s.trim_lost = True
        r = self.c.prefill_part(seq[:n - 1])   # (the trim could not move back exactly: the hold was dropped)
        self.c.set_next(seq[n - 1])
        self.counters["reprefills"] += 1
        self._refilled(self.held, n - 1 - max(0, r or 0), tp)

    def _unban(self, s, toks):
        k = next(i for i, t in enumerate(toks) if t in s.ban)
        base = len(s.ids) + len(s.out)   # (s.out not yet extended by toks: its last token was this pass's input)
        s.out += toks[:k]
        self._back_to(s, base + k)
        del s.out[len(s.out) - k:]
        t = self._masked_step(s, s.banmask)
        self.counters["ban_redo"] = self.counters.get("ban_redo", 0) + 1
        return toks[:k] + ([] if t < 0 else [t])

    def _masked_step(self, s, mask=None):
        """one pass of a constrained slot: the logits of the pending token's step, masked by row 0, sampled here with
        the slot's parameters (greedy at temperature <= 0); the token becomes the pending one. -1 = context full"""
        x = self.c.step_logits()
        if x is None:
            return -1
        self.counters["constrained_steps"] += 1
        m = mask if mask is not None else (s.mask[0] if s.mask is not None else None)
        if m is not None:
            bits = np.unpackbits(m.view(np.uint8), bitorder="little")[:x.shape[0]].astype(bool)
            x = np.where(bits, x, -np.inf)
        if s.rng is None:   # per slot from the request's seed (review L3)
            s.rng = np.random.default_rng(s.samp[4] + 0x9E3779B1 * s.sw)
        t = sample_host(x, s.samp, s.rng)
        self.c.set_next(t)
        return t


def sample_host(x, samp, rng):
    """temperature / top_k (0 = 64, the engine sampler's candidates) / top_p / min_p over float logits x; greedy at
    temperature <= 0. A fully masked row (never: the grammar always allows something) gives the argmax."""
    temp, top_p, top_k, min_p, _seed = samp
    if not np.isfinite(x).any():
        return int(np.argmax(x))
    if temp <= 0:
        return int(np.argmax(x))
    k = top_k if 0 < top_k <= 64 else 64
    idx = np.argpartition(-x, min(k, x.shape[0]) - 1)[:k]
    idx = idx[np.isfinite(x[idx])]
    v = x[idx].astype(np.float64) / temp
    o = np.argsort(-v)
    idx, v = idx[o], v[o]
    p = np.exp(v - v[0])
    p /= p.sum()
    if min_p > 0:
        keep = p >= min_p * p[0]
        idx, p = idx[keep], p[keep]
    if top_p < 1.0:
        cut = int(np.searchsorted(np.cumsum(p), top_p) + 1)
        idx, p = idx[:cut], p[:cut]
    p = p / p.sum()
    return int(rng.choice(idx, p=p))


class Dsv4Template:
    """[shared-srv] the DeepSeek-V4.1 prompt format (dsv4_chat.encode) behind app.Template's render(messages, tools, **kw):
    enable_thinking (only False turns it off, as the jinja templates test `is false`), reasoning_effort (low / medium /
    high / xhigh / max or 1-100; minimal = low)."""

    def __init__(self, profile=None):
        import dsv4_chat
        import threading
        self.C = dsv4_chat
        self.hints = {}
        self.mu = threading.Lock()   # (request threads: review L9)

    def effort(self, e):
        if e is None:
            return None
        if isinstance(e, str):
            e = e.strip().lower()
            if e == "minimal":
                return "low"
            if e in ("none", "off", ""):   # (thinking off is enable_thinking false; the effort line then goes)
                return None
            if e.isdigit():
                return int(e)
        return e

    def render(self, messages, tools=None, **kw):
        import jinja2
        thinking = kw.get("enable_thinking", True) is not False
        try:
            return self.C.encode(messages or [], tools or None, thinking=thinking, effort=self.effort(kw.get("reasoning_effort")))
        except ValueError as ex:   # an unknown effort / role: HTTP 400 through prompt()'s template error path
            raise jinja2.TemplateError(str(ex)) from ex

    def system_hint(self, messages, tools, kw, ids, encode):
        """[dsv4-rel6-xprefix] tokens of ids that are the system part (the system message with the tools) when other
        prompts may share it: a system message and more after it, >= 1024 tokens; else 0 (dsv4_app's rule)"""
        if not messages or messages[0].get("role") != "system" or len(messages) < 2:
            return 0
        import json
        thinking = kw.get("enable_thinking", True) is not False
        try:
            key = hash((json.dumps(messages[0], sort_keys=True, default=str), json.dumps(tools, sort_keys=True, default=str),
                        thinking, str(kw.get("reasoning_effort"))))
            with self.mu:
                sp = self.hints.get(key)
            if sp is None:
                sp = list(encode(self.render(messages[:1], tools, **kw)))
                with self.mu:
                    if len(self.hints) >= 64:
                        self.hints.pop(next(iter(self.hints)))
                    self.hints[key] = sp
        except Exception:
            return 0
        n = 0
        while n < min(len(sp), len(ids)) and sp[n] == ids[n]:
            n += 1
        return n if n >= 1024 else 0


import tool_choice as _tc  # noqa: E402


class DsmlTier(_tc.Tier):
    """[shared-srv] tool_choice for DSV4: the call header is `\\n\\n<｜DSML｜ calls>\\n<｜DSML｜ invoke name="NAME">\\n`
    (dsv4_chat), its marker the ｜DSML｜ token: `none` masks that token (no block can start), required / named force
    the header (after </think> in thinking mode), as tool_choice.Tier does for <tool_call>."""
    START = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\""

    def __init__(self, tok, think_ids, eos, im_start):
        self.tok, self.ti = tok, think_ids
        self.call = tok.token_to_id("｜DSML｜")
        self.ok = self.call is not None
        self.blocked_think = {self.call, *eos}
        self._masks = {}
        self._tries = {}

    def header(self, lead, name):
        h = self.enc(lead + self.START + name + "\">\n")
        for tail in ("<｜DSML｜ parameter name=\"x\" string=\"true\">v</｜DSML｜ parameter>\n", "</｜DSML｜ invoke>\n"):
            f = self.enc(lead + self.START + name + "\">\n" + tail)
            k = 0
            while k < len(h) and k < len(f) and h[k] == f[k]:
                k += 1
            h = h[:k]
        return h

    def matcher(self, spec, opens_in_think):
        if spec["mode"] == _tc.NONE:
            return _tc.ToolMatcher(self, _tc.S_NOCALL, None, None)
        names = spec["names"]
        if opens_in_think and self.ti is not None and self.ti.ok:   # (the header starts with its own blank line)
            return _tc.ToolMatcher(self, _tc.S_THINK, None, self.trie("", names))
        return _tc.ToolMatcher(self, _tc.S_TRIE, self.trie("", names), None)
