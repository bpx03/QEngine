"""OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) compatible server for qengine.

usage:
  CUDA_VISIBLE_DEVICES=0,1,2 python server/app.py --port 8080 [--template FILE] [--max-ctx 262144]

One engine, one request at a time (requests queue up); prompt caching is automatic: every
request is a full prompt and the engine reuses the longest cached prefix.
"""
import os

# First, before anything imports numpy: its OpenBLAS pool (one thread per CPU, the server does no BLAS work) would be alive
# at every later environment write: parse_args below and libqflash's constructor (issue #11). A value set by the user stays.
if not os.environ.get("OPENBLAS_NUM_THREADS", "").strip():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import array
import hashlib
import asyncio
import collections
import ctypes
import faulthandler
import functools
import json
import re
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime

import jinja2
import jinja2.ext
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agents import Tracker
from qe_env import env_float, env_int, env_str, set_env
import loop_guard
from content_markers import OUT_MARK, SENT, Markers, escape_out, mark, out_marks, unmark
from engine import Engine, EngineRefused, SequenceFault, StartRefused, drain_logs
import engine_stats
from steering import DEFAULT as STEER_DEFAULT, resolve_steering
from grammar import Constraint, GrammarError, Grammars, TokenLookup, schema_of
import tool_choice
from think_budget import (BudgetError, BudgetMap, ThinkControl, ThinkIds, budget_banner, budget_source, empty_reply,
                          probe_rules, request_budget)
from model_profile import DroppedTurnLog, ParamNotes, Profile, drop_empty_turns, engine_caps, ignored_settings, merge_leading_system

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _hash7(h):
    """a git hash cut to 7 characters, a "-dirty" suffix kept"""
    m = re.fullmatch(r"([0-9a-f]{7,40})(-dirty)?", h)
    return m.group(1)[:7] + (m.group(2) or "") if m else h


def _tree_hash():
    """HEAD of the server's own git tree (+ "-dirty": tracked files changed), or "" outside one (also when ROOT is a
    tree without .git unpacked inside another repository)"""
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, timeout=3).stdout.strip()
    try:
        top = git("rev-parse", "--show-toplevel")
        if not top or os.path.realpath(top) != os.path.realpath(ROOT):
            return ""
        h = git("rev-parse", "--short=7", "HEAD")
        return h + ("-dirty" if h and git("status", "--porcelain", "-uno") else "")
    except Exception:
        return ""


@functools.lru_cache(maxsize=None)
def _version_info(lib):
    try:
        with open(os.path.join(ROOT, "VERSION")) as f:
            version = f.read().strip()
    except OSError:
        version = ""
    build = ""
    try:  # written by the build next to the libraries (CMakeLists.txt target qe_build_id; the Docker image copies it)
        with open(os.path.join(os.path.dirname(lib), "qe_build.txt")) as f:
            build = f.read().strip()
    except OSError:
        pass
    if build in ("", "unknown"):  # a library copied out of its build: libqflash-<hash>.so
        m = re.search(r"-([0-9a-f]{7,40})\.so$", os.path.basename(lib))
        build = m.group(1) if m else ""
    tree = _tree_hash()
    if not build and not tree:  # an old build directory without qe_build.txt, named qengine-<hash>
        m = re.fullmatch(r"qengine-([0-9a-f]{7,40})", os.path.basename(os.path.realpath(ROOT)))
        build = m.group(1) if m else ""
    build = _hash7(build) if build else "unknown"
    if tree and tree != build:  # the library was built from other code than the tree the server runs from
        build = f"{build}, tree {tree}"
    return version or "dev", build


def version_info(lib=None):
    """(version, build) of this install for the start-up log and /api/stats (qetop, web dashboard); computed once per
    library. version: the VERSION file at the tree root (bumped by the release flow; in the git tree, the public
    export and the Docker image), else "dev". build: the hash the loaded engine library was built from (qe_build.txt
    next to it, else a -<hash>.so file name), "unknown" if neither; ", tree <hash>" is added when the server's git
    tree is at another commit or has uncommitted changes."""
    return _version_info(os.path.realpath(lib or os.path.join(ROOT, "build", "libqengine.so")))


# QE_STEP_PROF=1: per-step host timing of the single-sequence decode loop (ctypes step call, metrics,
# feed = detokenize + parser hand-off, check_done, the rest of the loop iteration), one JSON line per
# request to $QE_STEP_PROF_FILE (default step_prof.jsonl) with the prompt ids, plus a log summary
STEP_PROF = env_str("QE_STEP_PROF", "0") != "0"
REQ_PROF = env_str("QE_REQ_PROF", "0") != "0"  # [ink-srv-gap] per streamed request: server-side stamps (chat handler)
# [srv-gptoss-gap] QE_LOOP_PROF=1: scheduler iterations that only decode (no prefill), split by named marks (each mark
# takes the time since the previous one; "call" = the engine step); a summary line each time the server goes idle
LOOP_PROF = env_str("QE_LOOP_PROF", "0") != "0"


class LoopProf:
    """[srv-gptoss-gap] QE_LOOP_PROF accumulator over decode-only scheduler iterations."""

    def __init__(self):
        self.acc, self.cur, self.t, self.ok = {}, {}, 0.0, False
        self.it = self.toks = self.rows = 0
        self.n = (0, 0)

    def top(self, active):
        """Start of a loop iteration: the previous one counts when it decoded and prefilled nothing."""
        now = _pc()
        if self.ok and "call" in self.cur:
            self.mark("rest", now)
            for k, v in self.cur.items():
                self.acc[k] = self.acc.get(k, 0.0) + v
            self.it += 1
            self.toks += self.n[0]
            self.rows += self.n[1]
        self.cur, self.t, self.ok, self.n = {}, now, True, (0, 0)
        if not active and self.it:
            self.report()

    def mark(self, k, now=None):
        now = now or _pc()
        self.cur[k] = self.cur.get(k, 0.0) + now - self.t
        self.t = now

    def report(self):
        tot = sum(self.acc.values())
        parts = " ".join(f"{k} {1e3 * v / self.it:.3f}" for k, v in self.acc.items())
        print(f"[loopprof] {self.it} decode iterations, {self.rows / self.it:.2f} slots, {self.toks / self.it:.2f} tok "
              f"each: ms/iter {1e3 * tot / self.it:.3f} = {parts}; host {100 * (tot - self.acc.get('call', 0)) / tot:.1f} %, "
              f"{self.toks / tot:.1f} tok/s", flush=True)
        self.acc, self.it, self.toks, self.rows = {}, 0, 0, 0


class _NoProf:
    ok = False

    def top(self, active):
        pass

    def mark(self, k, now=None):
        pass
# forward progress (issue #3): the scheduler loop beats a heartbeat; no beat for QE_HANG_S seconds (an engine call that
# never returns) -> Python + native stacks, the requests end with an error, exit 75 (docker / systemd restart it).
# A start refused after admission for lack of room (KV pool full, prompts in flight) is requeued while sequences run,
# for up to QE_START_RETRY_S seconds, before it fails; bad input and too-long refusals fail at once.
# A queued request that fits nowhere while nothing runs (a KV pool shrunk under VRAM pressure: nothing here will free
# room) ends with an error after QE_NO_ROOM_S seconds, 0 = no deadline; the server goes on.
# A sequence that gets no token for QE_NO_PROGRESS_STEPS steps and QE_NO_PROGRESS_S seconds while not paused fails.
HANG_S = env_float("QE_HANG_S", 180.0)
START_RETRY_S = env_float("QE_START_RETRY_S", 60.0)
# streamed replies: an SSE keep-alive after this many seconds without an event (a reply queued, parked or waiting for its
# turn: GLM / DSV4 time slicing can hold one for 30-40 s), so proxies and clients with read timeouts keep the stream;
# OpenAI streams get the comment ": keep-alive", Anthropic streams a ping event (0 = off)
SSE_KEEPALIVE_S = env_float("QE_SSE_KEEPALIVE", 10.0)
KEEPALIVE = object()  # stream_bridge item: nothing new for SSE_KEEPALIVE_S seconds


def keepalive_due(last):
    """review M1: a keep-alive is due when nothing was written for SSE_KEEPALIVE_S (items that the parser holds back,
    a tool call or hidden reasoning, write nothing); last: [monotonic time of the last write], kept by stamped()"""
    return SSE_KEEPALIVE_S > 0 and time.monotonic() - last[0] >= SSE_KEEPALIVE_S


async def stamped(it, last):
    """the SSE generator's writes, each stamping last[0] (keepalive_due)"""
    async for x in it:
        last[0] = time.monotonic()
        yield x
NO_ROOM_S = env_float("QE_NO_ROOM_S", 10.0)
NOPROG_STEPS = env_int("QE_NO_PROGRESS_STEPS", 64)
NOPROG_S = env_float("QE_NO_PROGRESS_S", 10.0)
PAUSE_TIMEOUT_S = env_float("QE_PAUSE_TIMEOUT_S", 900.0)  # deadline of a reply waiting for VRAM another process holds (preemption pauses: none)
# the hang deadline while the slots are persisted to the disk cache (--disk-cache: idle and at shutdown)
HANG_PERSIST_S = env_float("QE_HANG_PERSIST_S", 3600.0)
# [dsv4-rel8-hang] an engine call that may (re-)prefill (DSV4: a whole prompt is one call, a switch may re-prefill the
# conversation) gets QE_HANG_S plus its tokens at this rate (a floor about 10x below the measured 2,645 tok/s at 236K, review
# rel8 L6: 1M tokens -> 70 min, 262K -> 20 min); decode passes QE_HANG_S
HANG_PREFILL_TPS = max(1.0, env_float("QE_HANG_PREFILL_TPS", 250.0))
DISK_FLUSH_S = env_float("QE_DISK_FLUSH_S", 600.0)  # Flash-Next --disk-cache: how long the exit waits for the queued files


QE_HANG_DIAG_S = env_float("QE_HANG_DIAG_S", 30.0)  # hang monitor: time for the diagnostics before the hard exit


def run_bounded(cmd, timeout):
    """stdout of cmd, or None. Never blocks past `timeout`: on expiry the child is killed and abandoned (no wait():
    a child stuck in the driver (D state) ignores SIGKILL until the driver call returns, and CPython's
    subprocess.run(timeout=) waits for it without a bound)."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             text=True, start_new_session=True)
    except Exception as ex:
        print(f"[hang] {cmd[0]}: {ex}", flush=True)
        return None
    try:
        out, _ = p.communicate(timeout=max(0.1, timeout))
        return out
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        print(f"[hang] {os.path.basename(cmd[0])} did not finish in {timeout:.0f} s: abandoned", flush=True)
        return None
    except Exception as ex:
        print(f"[hang] {cmd[0]}: {ex}", flush=True)
        return None


def task_states():
    """[(tid, comm, state, wchan, syscall)] of this process's threads (from /proc, never blocks on the driver)."""
    out = []
    for t in sorted(os.listdir("/proc/self/task"), key=int):
        def rd(f):
            try:
                with open(f"/proc/self/task/{t}/{f}") as h:
                    return h.read().strip()
            except OSError:
                return "?"
        stat = rd("stat")
        out.append((t, rd("comm"), stat[stat.rfind(")") + 2:][:1], rd("wchan"), rd("syscall")[:60]))
    return out


def native_stacks(timeout=20.0):
    """Native stacks of every thread of this process to stdout: the /proc task states, then gdb (when installed and no
    thread is in uninterruptible sleep: gdb's attach waits for such a thread, and a killed gdb can leave the process
    stopped). Bounded by `timeout`."""
    ts = task_states()
    for t, comm, st, wchan, sc in ts:
        print(f"[hang] task {t} {comm}: state {st}, wchan {wchan}, syscall {sc}", flush=True)
    gdb = shutil.which("gdb")
    if not gdb:
        return
    if any(st == "D" for _, _, st, _, _ in ts):
        print("[hang] a thread is in uninterruptible sleep (D, stuck in the driver): no gdb", flush=True)
        return
    try:  # Yama ptrace_scope 1: allow the child gdb to attach to its parent
        import ctypes
        ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF), 0, 0, 0)
    except Exception:
        pass
    out = run_bounded([gdb, "-p", str(os.getpid()), "-batch", "-ex", "thread apply all bt"], timeout)
    if out:
        print(out[-200000:], flush=True)


def hard_exit(code):
    """os._exit after the engine's buffered log lines are out (os._exit skips the atexit drain; qe_log.h)."""
    drain_logs()
    os._exit(code)


def hard_exit_guard(after_s, code=75):
    """The hard bound of the hang path: an in-process timer exits with `code` after `after_s`, and an outside watcher
    process SIGKILLs this one (SIGCONT first: a killed gdb can leave it stopped) after_s + 10 s later if it still
    runs. The watcher ends quietly when this process exits first (its pipe closes)."""
    t = threading.Timer(after_s, lambda: hard_exit(code))
    t.daemon = True
    t.start()
    try:
        r, w = os.pipe()  # the watcher reads r: EOF = this process is gone
        src = ("import os, select, signal, sys\n"
               "r = select.select([0], [], [], float(sys.argv[2]))[0]\n"
               "if not r:\n"
               "    p = int(sys.argv[1])\n"
               "    for s in (signal.SIGCONT, signal.SIGKILL):\n"
               "        try: os.kill(p, s)\n"
               "        except OSError: pass\n")
        subprocess.Popen([sys.executable, "-I", "-c", src, str(os.getpid()), str(after_s + 10.0)], stdin=r,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        os.close(r)
        os.set_inheritable(w, False)
        hard_exit_guard.keep = w  # held open until the process exits
    except Exception as ex:
        print(f"[hang] outside watcher not started: {ex}", flush=True)
_pc = time.perf_counter

# ----------------------------------------------------------------------------- template


class _GenerationTag(jinja2.ext.Extension):
    """HF's {% generation %} ... {% endgeneration %} (marks the assistant tokens for training masks): renders its body
    as is. [ink-srv-capi] Inkling 2's template uses it; the other templates never do."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.Scope(body, lineno=lineno)


class Template:
    def __init__(self, path):
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                 extensions=["jinja2.ext.loopcontrols", _GenerationTag])

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        self.t = env.from_string(open(path).read())

    def render(self, messages, tools=None, **kw):
        return self.t.render(messages=messages, tools=tools or None, add_generation_prompt=True,
                             **{k: v for k, v in kw.items() if v is not None})


# ----------------------------------------------------------------------------- detokenizer


# GPT-2 byte-level alphabet of the tokenizers we serve (ByteLevel decoder): token text -> the token's own bytes
_BYTES = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
_BYTE_OF = {chr(b): b for b in _BYTES}
_BYTE_OF.update({chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in _BYTES)})


class LpText(str):
    """Reply text that carries the logprobs entries (attribute lp) of its tokens to the HTTP side."""


def token_bytes(tok, t):
    """The UTF-8 bytes of one token (a token may hold part of a character)."""
    s = tok.id_to_token(int(t)) or ""
    return bytes(_BYTE_OF[c] for c in s) if all(c in _BYTE_OF for c in s) else s.encode()


def logprob_entry(tok, logits, t, top):
    """OpenAI logprobs entry of token t: raw logits of the target model through log-softmax over the whole vocabulary,
    before temperature, top-p / top-k / min-p and penalties. -9999.0 (OpenAI's value): a token the head does not reach."""
    import numpy as np
    x = logits.astype(np.float64)
    bad = ~np.isfinite(x)
    if bad.any():  # (the 27B has no guard on the head's input) never NaN in the JSON: those tokens count as unreachable
        print(f"WARNING logprobs: {int(bad.sum())} non-finite logits at token {t}", flush=True)
        x[bad] = -np.inf
    m = x.max()
    lp = x - (m + np.log(np.exp(x - m).sum())) if np.isfinite(m) else np.full_like(x, -np.inf)

    def one(i):
        b = token_bytes(tok, i)
        return {"token": b.decode("utf-8", "replace"), "logprob": max(float(lp[i]), -9999.0), "bytes": list(b)}

    tops = []
    if top > 0:
        ids = np.argpartition(-x, top - 1)[:top]
        tops = [one(i) for i in ids[np.argsort(-x[ids], kind="stable")]]
    return dict(one(t), top_logprobs=tops)


def logprob_params(body):
    """(logprobs, top_logprobs) of a chat request: None without logprobs, else the alternatives per token; a string:
    what is wrong with the fields."""
    lp, top = body.get("logprobs"), body.get("top_logprobs")
    if lp is not None and not isinstance(lp, bool):
        return "logprobs must be a boolean"
    if top is not None:
        if isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= 20:
            return f"top_logprobs must be an integer between 0 and 20, got {top!r}"
        if top and not lp:
            return "top_logprobs requires logprobs to be true"
    return (top or 0) if lp else None


class Detok:
    """Incremental detokenization that never emits half of a multi-byte character. marks: {token id:
    text} for the reply parser's tags (server/content_markers.py): those tokens come out with their `<`
    replaced by OUT_MARK (same length), so a tag written as plain text is never taken for the token
    (default: the tokenizer's tag tokens; {} = plain decode)."""

    def __init__(self, tok, marks=None):
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0
        self.marks = out_marks(tok) if marks is None else marks

    def _dec(self, ids):
        mk = self.marks
        if not mk:  # (marks off: still escaped, parse_tool_calls keys on OUT_MARK, review L10)
            return escape_out(self.tok.decode(ids, skip_special_tokens=False))
        if not any(t in mk for t in ids):  # (a U+FDD1 the model wrote: escaped, only the tag tokens are marked, L5)
            return escape_out(self.tok.decode(ids, skip_special_tokens=False))
        out, run = [], []
        for t in ids:
            if t in mk:
                if run:
                    out.append(escape_out(self.tok.decode(run, skip_special_tokens=False)))
                    run = []
                out.append(mk[t])
            else:
                run.append(t)
        if run:
            out.append(escape_out(self.tok.decode(run, skip_special_tokens=False)))
        return "".join(out)

    def push(self, ids):
        self.ids += ids
        prev = self._dec(self.ids[self.prefix:self.read])
        cur = self._dec(self.ids[self.prefix:])
        if cur.endswith("�"):
            return ""
        out = cur[len(prev):]
        self.prefix, self.read = max(0, len(self.ids) - 6), len(self.ids)
        return out

    def flush(self):
        """The end of the reply: what push still holds back for a character that will not complete now, without that
        partial character (a step can end anywhere, so complete text may sit in front of it)."""
        prev = self._dec(self.ids[self.prefix:self.read])
        cur = self._dec(self.ids[self.prefix:]).rstrip("�")
        self.prefix, self.read = max(0, len(self.ids) - 6), len(self.ids)
        return cur[len(prev):]


# ----------------------------------------------------------------------------- output parser


def bad_param(body, stop_key, keys):
    """(field, message) of a request field that would fail on the scheduler thread, where an error can end other clients'
    replies too (Job.feed: all active jobs) or leave a started prefill behind (Job.begin_start). None: all fine. The
    conversions are the ones the engine call makes, so what worked before still does."""
    stop = body.get(stop_key)
    if stop is not None and not isinstance(stop, str) and not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        return stop_key, f"{stop_key} must be a string or a list of strings"
    conv = {"temperature": None, "max_tokens": None, "max_completion_tokens": None,
            "top_p": float, "min_p": float, "presence_penalty": float, "frequency_penalty": float, "top_k": int, "seed": int}
    for k in keys:
        v = body.get(k)
        if v is None:
            continue
        if k == "temperature" or k.startswith("max_"):
            if isinstance(v, int) or (k == "temperature" and isinstance(v, float)):
                continue
            return k, f"{k} must be {'a number' if k == 'temperature' else 'an integer'}"
        try:
            conv[k](v)
        except (TypeError, ValueError, OverflowError):
            return k, f"{k} must be a number"
    return None


def context_line(asked, e):
    """The start-up line of the context: requested (--max-ctx), the KV pool and the context served per conversation
    (/v1/models max_model_len, /props n_ctx, the request length check). The served one is never above the pool: a
    smaller pool lowers it, the server starts anyway (club-3090 #1966)."""
    ln = f"context: requested {asked} tokens, KV pool {e.pool_tokens} tokens, served {e.max_t} tokens per conversation"
    if e.max_t >= asked:
        return ln
    if getattr(e, "max_t_engine", e.max_t) > e.max_t:
        why = "the engine reported more than its KV pool"
    else:
        why = "by the engine (KV pool, GPU memory or the YaRN factor; see its start-up lines)"
    return f"{ln} (lowered: {why}; longer requests get 400 context_length_exceeded)"


def reply_thinking(gen, ids, kw):
    """The reply starts in reasoning: what the template really rendered (`enable_thinking` null or 0 still opens <think>,
    the template's test is `is false`), as ThinkControl decides it. Without the tag tokens: the request's setting."""
    ti = getattr(gen, "think_ids", None)
    if ti is not None and ti.ok:
        return ti.open_at_end(ids, gen.im_start)
    return kw.get("enable_thinking", True)


class Parser:
    """Splits generated text into reasoning / content / tool calls, holding back partial tags. The tags
    are the real tokens (marked by Detok, see server/content_markers.py); the same strings written as
    plain text are text. tool_text keeps the marks (parse_tool_calls reads them)."""

    THINK, END_THINK, CALL = mark("<think>"), mark("</think>"), mark("<tool_call>")
    TAGS = (THINK, END_THINK, CALL)
    END_CALL = mark("</tool_call>")
    CALL_BODY = "\n<function="  # what follows the tag in a call (the template's form, what parse_tool_calls needs)

    def __init__(self, thinking):
        self.mode = "reasoning" if thinking else "content"
        self.buf = ""
        self.tool_text = ""
        self.lead = False  # right after </think>: newlines before the answer are dropped, also in later chunks
        # no content text yet (the reply's start or right after a </think>): only here a real <think> opens reasoning.
        # Once the answer has begun it is text, as a stray </think> is. Kept as state, because a step can end anywhere
        # in the tokens (draft counts follow measured step times): whitespace waits here until the next thing is known.
        self.fresh = True
        self.promoted = False  # the reply's last calls, inside an open block, went to tool_text

    HOLD = 12  # longest tag

    def _hold(self, s):
        # length of the longest suffix of s that is a prefix of a tag we may still see
        for n in range(min(len(s), self.HOLD), 0, -1):
            if any(t.startswith(s[-n:]) for t in self.TAGS):
                return n
        return 0

    def feed(self, text, final=False, promote=False):
        """Returns list of (kind, text) with kind in reasoning/content. promote (with final): Job.call_in_block said yes."""
        out = []
        self.buf += text
        while True:
            if self.mode == "tool":
                self.tool_text += self.buf
                self.buf = ""
                break
            if self.lead and self.mode == "content":
                self.buf = self.buf.lstrip("\n")
                if not self.buf:
                    break
                self.lead = False
            if self.mode == "content" and self.fresh:
                b = self.buf.lstrip()
                if b.startswith(self.THINK):  # model re-opened a think block
                    self.buf = b[len(self.THINK):]
                    self.mode = "reasoning"
                    continue
                if not final and self.THINK.startswith(b):  # whitespace or part of the tag so far: undecided
                    break
                self.fresh = False
            tag = self.END_THINK if self.mode == "reasoning" else self.CALL
            i = self.buf.find(tag)
            if self.mode == "reasoning" and i < 0 and self.CALL in self.buf:
                # a real <tool_call> inside an open block. Not followed by the call's form it is reasoning text at once.
                # A call waits for what follows: calls that end a reply the model ended itself are its calls (`promote`:
                # Job.call_in_block; the template writes them after the block, but an agent harness would see a stop and no
                # call). More reasoning, a later </think> or any other end means quoted: reasoning text, as before.
                j = self.buf.find(self.CALL)
                if j:
                    out.append((self.mode, self.buf[:j]))
                    self.buf = self.buf[j:]
                p = 0  # end of the complete calls so far
                tag_text = False
                while True:
                    b = self.buf[p:].lstrip()
                    q = len(self.buf) - len(b)
                    if not b.startswith(self.CALL):
                        break
                    r = b[len(self.CALL):]
                    if not r.startswith(self.CALL_BODY):
                        if self.CALL_BODY.startswith(r):
                            b = None  # the form can still come
                        else:
                            tag_text = True
                        break
                    e = self.buf.find(self.END_CALL, q)
                    if e < 0:
                        b = None  # a call in progress
                        break
                    p = e + len(self.END_CALL)
                if tag_text:
                    out.append((self.mode, self.buf[:q + len(self.CALL)]))
                    self.buf = self.buf[q + len(self.CALL):]
                    continue
                if not final:
                    if b is None or self.CALL.startswith(b) or self.END_THINK.startswith(b):
                        break  # undecided (whitespace or part of a tag counts as "startswith")
                    out.append((self.mode, self.buf[:q]))  # other text follows: those calls were quoted
                    self.buf = self.buf[q:]
                    continue
                if promote and p and b == "":
                    self.promoted = True
                    self.mode, self.tool_text, self.buf = "tool", self.buf, ""
                    continue
            if i >= 0:
                if i:
                    out.append((self.mode, self.buf[:i]))
                if self.mode == "reasoning":
                    self.mode = "content"
                    self.buf = self.buf[i + len(tag):].lstrip("\n")
                    self.lead = True
                else:
                    self.mode = "tool"
                    self.tool_text = self.buf[i:]
                    self.buf = ""
                continue
            keep = 0 if final else self._hold(self.buf)
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                out.append((self.mode, emit))
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return [(k, unmark(t)) for k, t in out if t]


class DsmlParser(Parser):
    """[shared-srv] DSV4 (DeepSeek-V4.1): reasoning ends at the real </think> token, a tool-call block starts with the
    text "\n\n<｜DSML｜ calls>" (as dsv4_app's Stream / dsv4_chat.parse_completion)."""
    CALL = "<｜DSML｜ calls>"  # (the blank line before it stays in the content: Parser's lead strip may eat it)
    TAGS = (Parser.THINK, Parser.END_THINK, CALL)
    END_CALL = "</｜DSML｜ calls>"
    CALL_BODY = "\n<｜DSML｜ invoke"
    HOLD = len(CALL)
    LEAD = "\n\n"  # the blank line the format puts before a block: held back, dropped when the block follows (review M3)

    def _hold(self, s):
        n = super()._hold(s)  # a partial tag at the end, and the newlines right before it (or at the very end)
        if self.mode == "content":
            head = s[:len(s) - n]
            n += min(len(head) - len(head.rstrip("\n")), len(self.LEAD))
        return n

    def feed(self, text, final=False, promote=False):
        self.buf += text
        if self.mode == "content" and self.LEAD + self.CALL in self.buf:
            self.buf = self.buf.replace(self.LEAD + self.CALL, self.CALL, 1)
        return super().feed("", final, promote)


PARSER = Parser  # the reply parser of the model's format (resolve_model_args)


GENERIC = re.compile(r"^(you are claude code, anthropic's official cli|you are a claude agent, built on|"
                     r"x-anthropic-billing-header)", re.I)
ROLE = re.compile(r"^\s*you are (?:an? |the )?\*{0,2}([A-Za-z0-9][\w.\-]*(?: [\w.\-]+){0,3}?)\*{0,2}"
                  r"(?=[,.:;!\n]| -| —| the | an? )", re.I)


def agent_label(sys_txt, messages):
    """Short name of the agent behind a request (monitoring): the main Claude Code agent, a
    subagent named in its task ("You are **lumen-pm**, ..."), or the first specific line of the
    system prompt (background tasks such as session titles)."""
    lines = [l.strip() for l in sys_txt.splitlines() if l.strip()]
    specific = [l for l in lines if not GENERIC.match(l)]
    if any(GENERIC.match(l) and "official CLI" in l for l in lines) and any(
            "interactive agent" in l or "software engineering tasks" in l for l in specific[:3]):
        return "main agent"
    first_user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    if isinstance(first_user, list):
        first_user = " ".join(p.get("text", "") for p in first_user if isinstance(p, dict))
    m = ROLE.match(first_user or "")
    if m and len(m.group(1)) <= 40 and m.group(1).lower() not in ("claude", "a helpful assistant"):
        return m.group(1)
    for l in specific:
        l = re.sub(r"^you are (an? |the )?", "", l, flags=re.I).rstrip(".")
        if l:
            return l[:40] + ("…" if len(l) > 40 else "")
    task = " ".join((first_user or "").split())
    if lines and task:  # a subagent without a name: the start of its task
        return task[:36] + ("…" if len(task) > 36 else "")
    return "chat" if not lines else "agent"


# (not env_str: an empty or blank QE_DIAG_LOG turns the file off, it does not mean unset)
DIAG_PATH = os.path.expanduser(os.environ.get("QE_DIAG_LOG", "~/.local/state/qengine/diag.log").strip())
DIAG_PATH = os.path.abspath(DIAG_PATH) if DIAG_PATH else ""  # [kb-server] a bare file name: in the working directory (was never written)


def diag(kind, msg):
    """Anomalies worth a look (QE_DIAG_LOG, default ~/.local/state/qengine/diag.log; "" = off):
    own reply re-prefilled, reply sent back different from what was generated, tool calls
    without their required arguments. Also printed to the server log."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {kind} {msg}"
    print("diag: " + line, flush=True)
    if DIAG_PATH:
        try:
            os.makedirs(os.path.dirname(DIAG_PATH), exist_ok=True)
            with open(DIAG_PATH, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def diag_dump(kind, text):
    """Full text of an anomaly next to the diagnostics log; returns its path."""
    if not DIAG_PATH:
        return ""
    d = os.path.join(os.path.dirname(DIAG_PATH), "raw")
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{kind}.txt")
        with open(path, "w") as f:
            f.write(text)
        return path
    except OSError:
        return ""


def lcp_len(a, b):
    """Length of the common prefix of two token lists (binary search on C-speed slice compares)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        m = (lo + hi + 1) // 2
        if a[:m] == b[:m]:
            lo = m
        else:
            hi = m - 1
    return lo


_TOOL_HEAD = {"type": 0, "function": 1, "name": 2, "description": 3, "parameters": 4}


def _canon_json(x, head=False):
    """x with every mapping's keys sorted (head: the stock tool keys first, in the order clients send them)."""
    if isinstance(x, dict):
        keys = sorted(x, key=(lambda k: (_TOOL_HEAD.get(k, 9), str(k))) if head else str)
        return {k: _canon_json(x[k], head and k == "function") for k in keys}
    if isinstance(x, list):
        return [_canon_json(v) for v in x]
    return x


def canonical_tools(tools):
    """The tools in a canonical form, so an equal tool set renders a byte-identical prompt whatever order the client
    sends it in (clients such as omp re-inject MCP tools between turns; the tools block sits at the very start of the
    prompt, so a reordering used to invalidate the whole prompt cache): sorted by function name (stable: equal names
    keep their order), every key of each tool's JSON sorted (the head keys type / function / name / description /
    parameters keep the stock order). Tool-call parsing looks tools up by name, so it does not depend on the order."""
    if not tools or not isinstance(tools, list):
        return tools

    def name(t):
        f = t.get("function", t) if isinstance(t, dict) else None
        n = f.get("name") if isinstance(f, dict) else None
        return n if isinstance(n, str) else ""

    return [_canon_json(t, head=True) for t in sorted(tools, key=name)]


TOOL_FORMAT = "qwen"  # set from the model profile at startup ("glm": <arg_key> / <arg_value> pairs)


def parse_tool_calls_glm(text, tools):
    """GLM-5.3: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>"""
    schemas = {}
    for t in tools or []:
        f = t.get("function", t)
        props = (f.get("parameters") or {}).get("properties")
        schemas[f.get("name")] = props if isinstance(props, dict) and props else None
    calls = []
    if OUT_MARK in text:  # from Parser (master 99bce1ac): only the real <tool_call> / </tool_call> tokens delimit a call
        blocks = [unmark(b) for b in re.findall(re.escape(mark("<tool_call>")) + "(.*?)(?:" +
                                                re.escape(mark("</tool_call>")) + "|$)", text, re.S)]
    else:
        blocks = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)
    for block in blocks:
        name = block.split("<arg_key>", 1)[0].strip()
        # round 17: a name is one identifier (OpenAI's [A-Za-z0-9_-], plus . and :); prose after <tool_call> (the model
        # answering inside the tag) is no call, so tool_calls_or_text returns it as text
        if not re.fullmatch(r"[A-Za-z0-9_.:\-]{1,128}", name):
            continue
        props = schemas.get(name)
        args = {}
        for key, val in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)(?:</arg_value>|$)", block, re.S):
            key = key.strip()
            ptype = ((props or {}).get(key) or {}).get("type")
            if ptype != "string":
                try:
                    val = json.loads(val)
                except Exception:
                    pass
            args[key] = val
        calls.append({"name": name, "arguments": args})
    return calls


def tool_calls_or_text(parser, tools, tchoice, no_calls):
    """The calls of a reply's tool text that tool_choice allows, and that tool text as plain content when no call is
    left (nothing parses, tool_choice filters the calls out, or tool_choice is none; raw, marks removed; "" otherwise). A model-emitted <tool_call> whose body is not a
    call (prose, broken markup) then reaches the client as text instead of an empty reply, as vLLM's tool parsers
    return the model output as content when the parse fails."""
    if not parser.tool_text:
        return [], ""
    parsed = [] if no_calls else parse_tool_calls(parser.tool_text, tools)
    calls = tool_choice.filter_calls(tchoice, parsed) if not no_calls else []
    # review r17 L10: calls that parse but tool_choice filters out also leave the reply empty otherwise
    if calls:
        return calls, ""
    if TOOL_FORMAT == "inkling":  # never the control tokens of a call body that did not parse as text (rel160)
        import inkling
        return calls, inkling.plain(parser.tool_text)
    return calls, unmark(parser.tool_text)


def parse_tool_calls_dsml(text, tools):
    """[shared-srv] DSV4: "\n\n<｜DSML｜ calls>" blocks (dsv4_chat.parse_completion); arguments as a dict"""
    import dsv4_chat
    calls = []
    t = unmark(text)
    t = t if t.startswith("\n\n") else "\n\n" + t.lstrip()  # (the parser's tool text starts at the block's tag)
    for c in dsv4_chat.parse_calls(t):  # per parameter: a value that is not JSON stays its raw string (review M4)
        args = {}
        for k, v, is_s in c["params"]:
            if is_s:
                args[k] = v
                continue
            try:
                args[k] = json.loads(v)
            except ValueError:
                diag("TOOL_ARG_RAW", f"{c['name']}.{k}: not JSON, kept as text ({v[:120]!r})")
                args[k] = v
        calls.append({"name": c["name"], "arguments": args})
    return calls


def parse_tool_calls(text, tools):
    if TOOL_FORMAT == "glm":
        return parse_tool_calls_glm(text, tools)
    if TOOL_FORMAT == "dsml":
        return parse_tool_calls_dsml(text, tools)
    if TOOL_FORMAT == "harmony":  # [srv-gptoss]
        import harmony
        return harmony.parse_calls(text, tools)
    if TOOL_FORMAT == "inkling":  # [ink-srv-capi] <|content_invoke_tool_json|>{"name", "args"}<|end_message|>
        import inkling
        return inkling.parse_calls(text, tools)
    schemas = {}  # name -> properties of its parameter schema (None: the schema lists none)
    for t in tools or []:
        f = t.get("function", t)
        props = (f.get("parameters") or {}).get("properties")
        schemas[f.get("name")] = props if isinstance(props, dict) and props else None
    calls = []
    if OUT_MARK in text:  # from Parser: only the real <tool_call> / </tool_call> tokens delimit a call
        blocks = [unmark(b) for b in re.findall(re.escape(mark("<tool_call>")) + "(.*?)(?:" +
                                                re.escape(mark("</tool_call>")) + "|$)", text, re.S)]
    else:
        blocks = [unmark(b) for b in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)]  # (L5 escapes)
    for block in blocks:
        m = re.search(r"<function=([^>\n]+)>(.*)", block, re.S)
        if not m:
            continue
        # the body ends at the LAST </function> and a value at a </parameter> followed by the next parameter or the
        # end of the body: a value that quotes </parameter> / </function> (a file with a chat template or this
        # server's source) is not cut there (review L6)
        name, body = m.group(1).strip(), m.group(2)
        k = body.rfind("</function>")
        body = body[:k] if k >= 0 else body
        props = schemas.get(name)
        args = {}
        ms = list(re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>(?=\s*(?:<parameter=|$))", body, re.S))
        found = [(pm.group(1).strip(), pm.group(2)) for pm in ms]
        end = ms[-1].end() if ms else 0
        # the last parameter left open (the model went straight to </function>): its value runs
        # to the end of the function body, as in Qwen's own tool parser -- but only for a parameter
        # the tool declares and that is not set yet (an unknown open one, e.g. a stray `section`
        # after a complete Write, would make the client reject the whole call; drop it instead)
        tail = re.search(r"<parameter=([^>\n]+)>\n?(.*)", body[end:], re.S)
        if tail and "</parameter>" not in tail.group(2):
            key, val = tail.group(1).strip(), tail.group(2)
            if (props is None or key in props) and key not in (k for k, _ in found):
                found.append((key, val.rstrip("\n")))
                diag("TOOL_PARAM_UNCLOSED", f"{name}.{key} ({len(val)} chars)")
            else:
                diag("TOOL_PARAM_DROPPED", f"{name}.{key} unclosed, "
                     f"{'duplicate' if props is None or key in props else 'not in schema'} ({len(val)} chars)")
        for key, val in found:
            if props is not None and key not in props:  # the client rejects unknown parameters
                diag("TOOL_PARAM_DROPPED", f"{name}.{key} not in schema ({len(val)} chars)")
                continue
            ptype = ((props or {}).get(key) or {}).get("type")
            if ptype != "string":
                try:
                    val = json.loads(val)
                except Exception:
                    pass
            args[key] = val
        calls.append({"name": name, "arguments": args})
        params = next((t.get("function", t).get("parameters") or {} for t in tools or []
                       if t.get("function", t).get("name") == name), None)
        if params is None:
            diag("TOOL_UNKNOWN", f"{name!r}: {block[:300]!r}")
        else:
            miss = [r for r in params.get("required", []) if r not in args]
            if miss:
                raw = diag_dump('tool', unmark(text))   # "" with the diag file off: no empty "(raw: )"
                diag("TOOL_ARGS_MISSING", f"{name} missing {miss}: {block[:300]!r}" + (f" (raw: {raw})" if raw else ""))
    return calls


# ----------------------------------------------------------------------------- generation


WS = re.compile(r"\s+")


# generated ids of recent assistant turns (Generator.remember_turn): 8M tokens = 32 MB
GEN_TURNS_TOKENS = int(os.environ.get("QE_GEN_TURNS_TOKENS", 8_000_000))


def turn_key(text):
    return hashlib.blake2b(WS.sub("", text).encode(), digest_size=16).digest()


class Ids(list):
    """Prompt token ids plus the end of the system part (see Generator.encode) and, for prompts
    with images, mm = (image embeddings, M-RoPE positions, position shift) for the engine."""
    hint = 0
    mm = None
    label = ""
    img = (0, 0)
    agent = None  # agents.Info: main-thread detection of the request (prio, tracker feedback)


class QaSplit:
    """Queue-aware prefill / decode time split (Admission.qa step A). Prefill and decode time over the
    last ~QA_TAU seconds (exponential decay); window() = the decode time after a prefill pass of tc seconds."""

    def __init__(self, smin, smax, short_s=0.0):
        self.smin, self.smax, self.short_s = smin, smax, short_s
        self.acc = [0.0, 0.0]  # decayed seconds of prefill passes / decode steps
        self.t = time.time()

    def add(self, k, dt):
        now = time.time()
        f = math.exp(-(now - self.t) / QA_TAU)
        self.acc = [self.acc[0] * f, self.acc[1] * f]
        self.acc[k] += dt
        self.t = now

    def window(self, tc, adm, inflight, queue, active):
        """Prompts in flight / queued: the next pass serves the one with the least left (shortest first). Short (at most
        --prefill-short rows): no decode window beyond the one step the loop runs anyway; else the decoders' share by head
        count A / (A + W) within [smin, smax]. Either way the decoders' share over the window stays >= smin."""
        e = adm.e
        if self.short_s > 0 and any(j.est is None for j in queue):
            # (shared batches: a request that arrived during the pass has no estimate yet; counted as its whole prompt it
            # made a short turn look long, and the decoders got a head-count window while it waited: 16 in flight TTFT
            # p50 0.61 s vs 0.35 s for the fixed split)
            adm.estimate(time.time())
        lefts = [len(j.ids) - max(0, e.prefill_pos(j.slot)) for j in inflight]
        lefts += [j.est if j.est is not None else len(j.ids) for j in queue]
        A = sum(1 for j in active if not j.finished)
        W = len(inflight) + len(queue)
        s = 0.0
        if lefts and min(lefts) > adm.short and A > 0:
            s = min(self.smax, max(self.smin, A / (A + W)))
        elif self.short_s > 0 and A > 0:
            # shared prefill batches, a short prompt next: a fixed short window after every pass and no catch-up window
            # (arrivals during it join the next shared batch; the catch-up of the floor came in bursts that held the
            # queued prompts: 16 in flight TTFT mean 0.78-0.90 s vs 0.43 s with the fixed 0.10 split)
            return tc * self.short_s / (1 - self.short_s)
        win = tc * s / (1 - s) if 0 < s < 1 else 0.0
        pf, dec = self.acc
        if self.smin > 0 and pf + dec > 0:  # floor: the decoders' share of the recent time
            win = max(win, (self.smin * (pf + dec) - dec) / (1 - self.smin))
        return win


class Admission:
    """Which queued request the scheduler starts next (engine thread; called under gen.cv)."""

    def __init__(self, gen):
        self.g, self.e, a = gen, gen.engine, gen.args
        self.aging, self.pre_min = a.prefill_aging, a.preempt_min
        self.backfill_wait = getattr(a, "backfill_wait", 0)
        # a request pauses others only after waiting this long (a short turn does not pay two full KV transfers of a
        # sequence that would end in seconds; review M1)
        self.pause_wait = getattr(a, "pause_wait", PAUSE_WAIT)
        self.k = a.mtp_k
        self.last_note = 0.0
        # several prompts in one prefill batch (Flash-Next): every admitted text prompt joins the next batches
        self.multi = gen.engine.can_prefill_multi() if hasattr(gen.engine, "can_prefill_multi") else False
        # passes bound by the PCIe copies of the host-backed experts (2 cards): the large batch also while requests wait
        self.pass_bound = self.multi and gen.engine.prefill_pass_bound()
        # shared batches: cut-ins at any time, the longest prompt in flight keeps this share of the rows (0: the old
        # cut-in rounds with the preempt_min gap). Default (None): 0.5 where passes are copy-bound (measured there,
        #), else 0; an explicit value applies on any Flash-Next server
        ls = getattr(a, "long_share", None)
        if ls is None:
            ls = LONG_SHARE if self.pass_bound else 0.0
        elif not 0.0 <= ls <= 0.95:
            print(f"[sched] --long-share {ls} out of [0, 0.95]: {min(0.95, max(0.0, ls))} used", flush=True)
        self.long_share = min(0.95, max(0.0, ls)) if self.multi else 0.0
        self.long_job, self.credit, self.pos0 = None, 0.0, None
        # --fill-passes 1 (opt-in, with a share): the other prompts in flight take each shared batch first, oldest
        # first; the longest prompt fills the rest of every pass and keeps at least long_share of its rows (it leads
        # capped to that share when the others want more). Default 0: the per-pass credit rule; full
        # passes did not beat it on short appends in the user profile
        # (not in mode B, FLASH_PAR=pp: its stage queues take whole prompts and ignore row caps)
        self.fill = bool(getattr(a, "fill_passes", 0)) and (getattr(gen.engine, "par", None) or {}).get("mode", "tp") != "pp"
        self.caps = {}
        # queue-aware split (--prefill-share-min < --prefill-share; engines that prefill one prompt per pass, time split):
        # short prompts never wait for a decode window, a long one shares the time by head count, the decoders keep a
        # floor share over the last QA_TAU seconds; cut-ins up to --prefill-short rows (step A).
        # Default on for the 27B and Flash-Next in TP mode with passes not bound by expert copies (3 cards; measured
        # there; Flash-Next: the time split only, a fixed 0.10 window after every pass while short prompts are next, its
        # shared batches keep their own cut-in rules); an explicit --prefill-share-min applies to any such engine
        smin = getattr(a, "prefill_share_min", None)
        fam = getattr(getattr(gen, "profile", None), "family", None)
        smax = getattr(a, "prefill_share", 0.5)
        if smin is None:
            # (Flash-Next with passes bound by host-backed expert copies (2 cards) or in mode B: not measured, the fixed split)
            smin = (QA_SHARE_MIN if fam == "qwen38-27b" else
                    QA_SHARE_MIN_MULTI if fam == "qwen38-flash-next" and not self.pass_bound and
                    (getattr(gen.engine, "par", None) or {}).get("mode", "tp") == "tp" else smax)
        self.share_min = min(max(0.0, smin), smax)
        self.short = max(1, int(getattr(a, "prefill_short", 4096)))
        self.qa = (self.share_min < smax and getattr(a, "prefill_chunk", 0) > 0 and
                   getattr(a, "prefill_turns", "time") != "slices")
        # (the cut-in rule between the chunks of a long prompt: engines that prefill one prompt per pass only)
        self.qa_cut = self.qa and not self.multi
        if self.qa:
            short_txt = (f"prompts up to {self.short} rows next: the decoders get {self.share_min:.2f} of the time after every "
                         f"pass" if self.multi else
                         f"prompts up to {self.short} rows run back to back with one decode step between passes")
            print(f"[sched] queue-aware prefill split: {short_txt}; longer ones leave the decoders A / (A + W) of the time within "
                  f"[{self.share_min:.2f}, {smax:.2f}]; decoders' floor {self.share_min:.2f} over {QA_TAU:.0f} s",
                  flush=True)
        # [q27-park-c] park replies beyond the seats: default on for the 27B (measured there), any engine with qe_park when
        # --park 1 is given
        pk = getattr(a, "park", None)
        if pk is None:
            # ([qfn-ttft-park] Flash-Next: measured on 3 cards in TP mode with shared batches; 2 cards (passes bound by
            # expert copies) and mode B keep it off)
            pk = 1 if fam == "qwen38-27b" or (fam == "qwen38-flash-next" and self.multi and not self.pass_bound and
                                              (getattr(gen.engine, "par", None) or {}).get("mode", "tp") == "tp") else 0
        self.seats = len(getattr(self.e, "slots", []) or [])
        rv = getattr(a, "park_reserve", None)
        if rv is None:  # (Flash-Next: the reserved seats' prompts share prefill batches; 4 measured over 2)
            rv = PARK_RESERVE_MULTI if self.multi else PARK_RESERVE
        self.reserve = max(1, int(rv))
        self.park = bool(pk) and hasattr(self.e, "can_park") and self.e.can_park() and self.seats > self.reserve + 1
        if self.park:
            print(f"[sched] replies beyond {self.seats - self.reserve} decoding sequences are parked in host memory after "
                  f"their first token ({self.reserve} seats kept for prompts) and resume oldest first", flush=True)
        # --preempt-max default: the profile's (Flash-Next 8192) with the shared-batch cut-ins, else the old 1024
        if getattr(a, "preempt_max", None) is None:
            d = getattr(getattr(gen, "profile", None), "defaults", None) or {}
            a.preempt_max = d.get("preempt_max", 1024) if self.long_share > 0 else 1024
        self.pre_max = a.preempt_max
        # pass-bound (a pass costs ~0.65 s of PCIe expert copies whatever its rows): the big batch also while others decode,
        # ~2x the prompt rows per decode stall of +0.13 s (agents + long prompts: short-turn TTFT p90 -41..-58 %,
        #), at most the measured 4096; an explicit --prefill-chunk keeps its value
        busy = min(getattr(a, "prefill_chunk_idle", 0), PASS_BOUND_CHUNK)
        if self.pass_bound and getattr(a, "prefill_chunk_auto", False) and busy > a.prefill_chunk > 0:
            a.prefill_chunk = busy
            print(f"[sched] prefill chunk {a.prefill_chunk} also while sequences decode (passes bound by expert copies)",
                  flush=True)
        if self.long_share > 0:
            print(f"[sched] cut-ins up to {self.pre_max} tokens join the shared prefill batches at once; the longest prompt "
                  f"keeps {self.long_share:.2f} of the rows", flush=True)
        if self.multi:
            print(f"[sched] several queued prompts share prefill batches"
                  f"{' (passes bound by host-backed expert copies: large batches also while requests wait)' if self.pass_bound else ''}",
                  flush=True)

    def note(self, msg):  # scheduler notes that would repeat every loop: at most one per second
        if time.time() - self.last_note > 1.0:
            self.last_note = time.time()
            print(msg, flush=True)

    def estimate(self, now):
        # prefill cost of the queued requests: prompt tokens minus what the caches hold
        # (engine probe, no side effects; refreshed at most twice a second)
        e = self.e
        for j in self.g.queue:
            if j.est is None or now - j.t_est > 0.5:
                e.set_steering(*j.p.get("steering_sel", STEER_DEFAULT))  # (cached states are keyed by it)
                j.est = len(j.ids) - e.probe_cached(j.ids)
                j.t_est = now

    def pick_next(self, now, prefills):
        """The queued requests that may be prefilled next, best first (empty: keep waiting): the
        least left to compute, aged so that nothing starves (every second of waiting is worth
        `aging` tokens); every conversation alike (no main-thread rule, user decision 2026-11-12).
        While a long prompt is in progress only a request with at most preempt_max tokens left may
        cut in (this cut-in rule only with --long-share 0; with a share the shared-batch rule below
        applies)."""
        q = self.g.queue
        if not q:
            return []
        self.estimate(now)
        order = sorted(range(len(q)), key=lambda i: q[i].est - (now - q[i].t_enq) * self.aging)
        if not prefills:
            return order
        join = []
        if self.multi and len(prefills) < MULTI_MAX and self.long_share <= 0:
            # text prompts join the shared batches of the ones in flight while all of them fit one batch (a long prompt
            # gains nothing from sharing and admitting it early only claims its KV region ahead); else the cut-in rule
            cap = self.join_cap()
            left = sum(len(j.ids) - max(0, self.e.prefill_pos(j.slot)) for j in prefills)
            join = [i for i in order if not getattr(q[i].ids, "mm", None) and left + q[i].est <= cap]
            if join:
                return join
        left_of = lambda j: len(j.ids) - max(0, self.e.prefill_pos(j.slot))
        long_ = max(prefills, key=left_of)
        pos = max(0, self.e.prefill_pos(long_.slot))
        if self.long_share > 0:
            # (this includes the join rule above: a prompt that fits one batch with the ones in flight also joins)
            # shared batches with a guaranteed share for the longest prompt (batch_order): a text prompt with at most
            # preempt_max tokens left joins the next batch at once, whatever the long one has done since the last
            # cut-in. Before, it waited for preempt_min tokens of the long prompt between rounds, and a 1-8K tool result
            # (above preempt_max 1024) waited for the whole long prompt
            if len(prefills) >= MULTI_MAX:
                return []
            cap = self.join_cap()
            left = sum(left_of(j) for j in prefills)
            # (by size and age only, no main-thread rule: user decision 2026-11-12, every conversation alike)
            elig = lambda i: not getattr(q[i].ids, "mm", None) and (q[i].est <= self.pre_max or left + q[i].est <= cap)
            # nothing starves: once the best queued request by the aged order cannot join (too large, images) and has
            # waited backfill_wait, nothing more joins, the prompts in flight drain and it starts next (before: shorts
            # arriving every few seconds kept the batch busy and a 20K prompt or an image turn waited without bound)
            if not elig(order[0]) and now - q[order[0]].t_enq > self.backfill_wait:
                return []
            return [i for i in order if elig(i)]
        if self.pre_max <= 0 or (len(prefills) >= 2 and not self.multi):
            return []
        if self.qa_cut:
            # queue-aware split: a queued text request with at most --prefill-short rows (and fewer than the prompt in
            # flight has left) starts between its chunks, whatever it has left; the long prompt gets at least one pass
            # between two cut-ins (cut_at: its position at the last one): at least a third of the passes with 4K cut-ins (two
            # passes each), half with cut-ins of one pass
            left_long = len(long_.ids) - pos
            if long_.cut_at is not None and pos <= long_.cut_at:
                return []
            return [i for i in order if not getattr(q[i].ids, "mm", None) and q[i].est <= self.short and q[i].est < left_long]
        if len(long_.ids) - pos < self.pre_min:
            return []
        cuts = [j for j in prefills if j is not long_]
        if cuts:
            # a cut-in round (shared batches): more short requests join it while any of its cut-ins is in flight, up to
            # one batch of prompt tokens per round in all (before: one cut-in at a time, the rest of a burst waited
            # preempt_min tokens of the long prompt each). The long prompt still advances preempt_min tokens between
            # rounds, so it keeps a bounded share and finishes in bounded time however many requests keep arriving
            if len(prefills) >= MULTI_MAX:
                return []
            room = self.join_cap() - long_.round_tok
            return [i for i in order if not getattr(q[i].ids, "mm", None) and q[i].est <= min(self.pre_max, room)]
        # fairness: between cut-in rounds the long prompt advances preempt_min tokens per join-cap batch of cut-in tokens
        # in the last round (a full round: preempt_min; a lone 600-token cut-in: ~1K), so cut-ins take at most
        # join_cap / (join_cap + preempt_min) of the prompt rows (~36 % with the defaults) and the long one finishes
        if long_.cut_at is not None and pos - long_.cut_at < min(self.pre_min, long_.round_tok * self.pre_min / self.join_cap()):
            return []
        return [i for i in order if not getattr(q[i].ids, "mm", None) and q[i].est <= self.pre_max]

    def join_cap(self):  # prompt rows in flight that still share one prefill batch (the idle batch + a tail chunk)
        # (the idle batch only where prompts share batches: gpt-oss's idle chunk is one prompt's call, review pfidle M1)
        return max(self.g.args.prefill_chunk, self.g.args.prefill_chunk_idle if self.multi else 0) + 512

    def batch_order(self, now, prefills, budget=None):
        """Prompts in flight in the order they get rows of a shared prefill batch: least left first, aged (as
        pick_next); a long prompt gets what is left of the batch. Prompts that end within one batch (cut-ins) come
        before longer ones: aged by its wait in the queue, a long prompt in progress for ~45 s sorted ahead of a
        cut-in, took the whole batch and the cut-in waited for all of it (KV in RAM, t8 + 262K: TTFT p50 58 s).
        With --long-share the longest prompt is placed explicitly: first while it is owed rows, else last."""
        cap = self.join_cap()
        left = {id(j): len(j.ids) - max(0, self.e.prefill_pos(j.slot)) for j in prefills}
        def key(j):
            return (left[id(j)] > cap, left[id(j)] - (now - j.t_enq) * self.aging)
        order = sorted(prefills, key=key)
        self.pos0 = None
        self.caps = {}
        if self.long_share > 0 and self.fill and len(prefills) >= 2:
            # full passes: the others by age (oldest first), the longest prompt last, so it
            # takes whatever rows they leave. When they want more than (1 - long_share) of the batch, it leads instead,
            # capped to long_share of the rows (whole 512-row chunks), and they share the rest: every pass is full and
            # the long prompt gets at least its share of every pass (no pass of its own that the others wait out)
            long_ = max(prefills, key=lambda j: left[id(j)])
            others = sorted((j for j in prefills if j is not long_), key=lambda j: j.t_enq)
            T = min(int(budget), cap - 512) if budget else cap - 512  # (the engine's batch: min(budget, its batch rows))
            keep = max(512, int(self.long_share * T) // 512 * 512)
            self.long_job, self.credit = None, 0.0
            if sum(left[id(j)] for j in others) <= T - keep or left[id(long_)] <= keep:
                return others + [long_]
            self.caps = {id(long_): keep}
            return [long_] + others
        if self.long_share > 0 and len(prefills) >= 2:
            # the longest prompt keeps long_share of the prompt rows: the batch rows it got short of its share since the
            # cut-ins began (credit) put it first in the next batch (the engine then gives it the whole batch), else it
            # goes last and gets what the cut-ins leave (placed last explicitly: by the aged key a long prompt in flight
            # for a while sorted ahead of a 5-8K cut-in, which then waited for all of it)
            long_ = max(prefills, key=lambda j: left[id(j)])
            if long_ is not self.long_job:
                self.long_job, self.credit = long_, 0.0
            order.remove(long_)
            if self.credit > 0:
                order.insert(0, long_)
            else:
                order.append(long_)
            self.pos0 = {id(j): len(j.ids) - left[id(j)] for j in prefills}
        else:
            self.long_job, self.credit = None, 0.0
        return order

    def account(self, prefills, budget):
        """After a batch of batch_order's prompts: the long prompt's credit grows by its share of the batch and shrinks
        by the part of the batch's rows it got, per pass (bounded by one pass either way, so neither side banks a long
        run). Per pass, not per row: a pass costs about the same whatever its rows where passes are copy-bound, and a
        row count let small cut-in batches (a few hundred rows, without the long prompt) take most of the passes (62K
        prompt 28 -> 56 s next to agents)."""
        if self.pos0 is None or self.long_job is None:  # (--fill-passes: no credit to keep)
            return
        if len(prefills) < 2:  # the long prompt ran alone: no cut-in to share with, nothing owed either way
            self.credit, self.pos0 = 0.0, None
            return
        rows = {}
        for j in prefills:
            p0 = self.pos0.get(id(j))
            if p0 is None:
                continue
            p = self.e.prefill_pos(j.slot)
            rows[id(j)] = max(0, (len(j.ids) if p < 0 else min(p, len(j.ids))) - p0)
        tot = sum(rows.values())
        if tot > 0:
            self.credit = min(1.0, max(-1.0, self.credit + self.long_share - rows.get(id(self.long_job), 0) / tot))
        self.pos0 = None
        p = self.e.prefill_pos(self.long_job.slot)
        if p < 0 or p >= len(self.long_job.ids):  # (it ended in this batch: not kept referenced until the next one)
            self.long_job, self.credit = None, 0.0

    def admit_next(self, now, prefills):
        """Index of the queued request to start now, or None. The best candidate that fits the pool
        goes; one that does not fit lets the next ones that fit start before it (backfill) until it
        has waited backfill_wait seconds: then nothing passes it any more.
        A request that does not fit and has waited pause_wait seconds pauses decoding sequences
        (never a prefill) when that makes it fit: the most recently admitted ones first
        (src/pool_order.h), whoever they are. Never while a sequence is paused: it resumes first
        (the engines' pause_for does not count paused rooms, can_admit does: a newcomer would take
        the room the paused one waits for, again for every arrival)."""
        g, e = self.g, self.e
        blocked = None
        for i in self.pick_next(now, prefills):
            j = g.queue[i]
            if blocked is not None and now - blocked.t_enq > self.backfill_wait:
                continue  # starvation guard: nothing passes it any more
            need = len(j.ids) + self.k + 2
            if e.can_admit(need):
                if blocked is not None:
                    g.sched["backfills"] += 1
                    self.note(f"[sched] backfill: {need} tokens start while {len(blocked.ids)} tokens "
                              f"(waiting {now - blocked.t_enq:.1f} s) do not fit")
                return i
            ps = e.pause_state() if hasattr(e, "pause_state") else None
            if ps is not None and ps[1] == 0 and now - j.t_enq >= self.pause_wait:
                if hasattr(e, "protect_ids"):  # [ink-archprot] its archived conversation stays while the copies make room
                    e.protect_ids(j.ids, j)
                ok = e.pause_for(need, j.prio)  # (any request, by the pool order: the most recently admitted first)
                ps2 = e.pause_state()  # (also after a False: pause_slot may have failed after pausing some)
                if ps2 is not None and ps2[0] > ps[0]:  # (counted and logged only when something was paused)
                    g.sched["pauses"] += 1
                    g.cur_slot = -1  # a paused slot may have been the current single sequence
                    print(f"[sched] {need} tokens (waited {now - j.t_enq:.1f} s): paused {ps2[0] - ps[0]} of the most "
                          f"recently admitted decoding sequences{' to admit it' if ok else ' (then the pause failed: not admitted)'}",
                          flush=True)
                if ok:
                    return i
            if blocked is None:
                blocked = j
        if blocked is not None:
            self.note(f"[sched] no room for {len(blocked.ids) + self.k + 2} tokens next to the running sequences")
        return None


class Generator:
    def __init__(self, args):
        self.args = args
        self.profile = prof = getattr(args, "profile", None) or Profile(args.model)
        self.tok, tok_json, tok_note = prof.tokenizer(getattr(args, "tokenizer", None))
        print(f"model {prof.family} ({prof.model_type or 'no model_type'}), tokenizer {tok_note}", flush=True)
        # [shared-srv] a family with its own prompt format renders through its renderer (DSV4: dsv4_chat.encode)
        if prof.defaults.get("renderer") == "dsv4":
            from dsv4_engine import Dsv4Template
            self.tmpl = Dsv4Template(prof)
        else:
            self.tmpl = Template(args.template)
        # encode(): the prompt is tokenized segment by segment at this atomic special token
        self.seg_mark = "<｜User｜>" if prof.defaults.get("renderer") == "dsv4" else prof.defaults.get("seg_mark", "<|im_start|>")
        self.marks = Markers(self.tok, tok_json)  # markers in content stay plain text (issue #7)
        self.marks.plain_tok()  # at start, not on the event loop of the first request that needs it (review M2)
        if TOOL_FORMAT == "harmony":  # [srv-gptoss] the harmony control tokens come out marked (only real ones count)
            import harmony
            self.marks.out.update(harmony.marks(self.tok))
        if TOOL_FORMAT == "inkling":  # [ink-srv-capi] Inkling's message / content tokens likewise
            import inkling
            self.marks.out.update(inkling.marks(self.tok))
        self.im_start_user = self.tok.encode("<|im_start|>user", add_special_tokens=False).ids
        self.seg_cache = collections.OrderedDict()
        self.seg_tokens = 0
        self.turns = collections.OrderedDict()  # affinity -> recent (prompt, generated) (see check_reuse)
        self.gen_turns = collections.OrderedDict()  # see remember_turn
        self.gen_turn_tokens = 0
        self.im_start = self.tok.token_to_id("<|im_start|>")
        # image rows in the grammar lookup history (grammar.TokenLookup): the pad, as in the engines' own lookup;
        # a tokenizer without it has no image prompts, eos is then only a valid placeholder
        self.image_pad = self.tok.token_to_id("<|image_pad|>")
        if self.image_pad is None:
            self.image_pad = min(prof.eos_ids) if prof.eos_ids else 0
        self.im_end_nl = self.tok.encode("<|im_end|>\n", add_special_tokens=False).ids
        # loop guard (server/loop_guard.py): the repeated reply's tokens seed the penalty, without special tokens
        # (ids from <|endoftext|> on) and whitespace-only pieces
        self.asst_ids = self.tok.encode("assistant\n", add_special_tokens=False).ids
        self.im_end = self.tok.token_to_id("<|im_end|>")
        self.special_min = self.tok.token_to_id("<|endoftext|>") or 1 << 30
        wsch = set("ĠĊĉčĀ \t\n")  # byte-level BPE spellings of whitespace
        self.ws_ids = {i for t, i in self.tok.get_vocab().items() if i < self.special_min and t and set(t) <= wsch}
        self.guards = {"triggers": 0, "recent": []}  # /api/stats loop_guard
        self.guard_seen = collections.OrderedDict()  # loop guard: stretch -> (total, repeats) for guard_total
        self.seg_lock = threading.Lock()
        # end of turn: generation_config.json eos_token_id ([<|im_end|>, <|endoftext|>] for both
        # models) plus those two tokens by name
        self.eos = set(prof.eos_ids) | {self.tok.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")
                                        if self.tok.token_to_id(t) is not None}
        self.sampling_defaults = {t: prof.sampling(t, args.temperature, args.top_p, args.top_k, args.min_p, args.presence_penalty)
                                  for t in (True, False)}
        for t, name in ((True, "thinking"), (False, "non-thinking")):
            sd = self.sampling_defaults[t]
            print("sampling defaults, %s (when the request has none): temperature %.2f, top_p %.2f, top_k %d, min_p %.2f, "
                  "presence_penalty %.2f" % (name, sd["temperature"], sd["top_p"], sd["top_k"], sd["min_p"], sd["presence_penalty"]),
                  flush=True)
        self.vision = None
        if args.vision:  # before the engine, so that its KV cache planning sees the memory used
            from vision import VisionEncoder
            dev = args.vision_device or f"cuda:{len(args.devices.split(',')) - 1}"
            t0 = time.time()
            self.vision = VisionEncoder(args.model, dev, args.max_image_pixels)
            self.vision_lock = threading.Lock()
            print(f"vision encoder on {dev} in {time.time() - t0:.1f}s "
                  f"(max {args.max_image_pixels / 1e6:.1f} MPix per image)", flush=True)
        if prof.defaults.get("adapter"):  # [shared-srv] an engine with its own Python adapter (DSV4: server/dsv4_engine.py)
            import importlib
            mod, cls = prof.defaults["adapter"].split(":")
            self.engine = getattr(importlib.import_module(mod), cls)(args, vocab=prof.vocab_size)
        else:
            self.engine = Engine(args.model, args.devices, args.max_ctx, getattr(args, "split", "") or "",
                                 lib_path=getattr(args, "engine_lib", None) or None,
                                 vocab=prof.vocab_size)
        self.dstats = engine_stats.DraftStats(self.engine)  # per-slot draft counters (qe_draft_stats)
        asked = args.max_ctx  # review rel8 L5: the DSV4 default served at its static value is no "lowered" request
        if getattr(args, "max_ctx_model", False):  # [ink-ctx-cap] review L1: the model's limit, not a request
            asked = min(asked, self.engine.max_t)
        if getattr(args, "ctx_static", 0) and self.engine.max_t == args.ctx_static:
            asked = args.ctx_static
        print(context_line(asked, self.engine), flush=True)
        if not self.engine.has("stops"):  # no undo of the steps past a stop sequence
            global STOP_ROLLBACK
            STOP_ROLLBACK = 0
        if getattr(self.profile, "is_gptoss", False) and "qe_set_penalty" not in getattr(self.engine, "fns", ()):
            print("[server] gpt-oss: presence / frequency penalties are not applied by this engine (requests are accepted, "
                  "the penalty is ignored; the loop guard keeps its note and thinking turn)", flush=True)
        missing = self.engine.missing_optional()
        if missing:
            print(f"engine: optional C API not provided (fallbacks in use): {', '.join(missing)}", flush=True)
        # structured output (response_format): llguidance matchers, masks written into the engine
        self.grammars = None
        print(f"engine parallel mode: {self.engine.par.get('mode', 'tp')} ({self.engine.par.get('why', '')})", flush=True)
        if self.vision is not None and not self.engine.has("vision"):  # (PP x TP: no image input)
            print(f"--vision ignored: {self.engine.refused.get('vision', 'the engine has no image input')}", flush=True)
            self.vision = None
        if self.vision is not None and hasattr(self.engine, "driver_gate"):  # [qfn-stress2-lazy] (no-op without qe_drv_gate)
            self.vision.gate = self.engine.driver_gate  # the encoder's launches between engine jobs (vision.py embed)
        if self.engine.has("grammar"):
            self.grammars = Grammars(tok_json, self.engine.vocab(), sorted(self.eos))
            self.grammars.prepare()
        # thinking budget (server/think_budget.py): the forced close of <think> goes through the grammar tier
        self.think_ids = ThinkIds(self.tok, self.profile.defaults.get("think_close", "\n\n")
                                  if getattr(self, "profile", None) is not None else "\n\n")
        if TOOL_FORMAT == "harmony":  # [srv-gptoss-think] the analysis channel; the close opens the final channel
            from harmony import HarmonyThinkIds
            from think_budget import WRAP
            self.think_ids = HarmonyThinkIds(self.tok, env_str("QE_THINK_WRAP", WRAP))
        self.tool_tier_call = self.tok.token_to_id("<tool_call>")
        if TOOL_FORMAT == "inkling":  # (tool_choice none without masks: the reply ends at the call's body token)
            self.tool_tier_call = self.tok.token_to_id("<|content_invoke_tool_json|>")
        self.can_force = self.engine.has("grammar") and self.think_ids.ok
        self.think = {"requests_with_budget": 0, "forced_budget": 0, "forced_max_tokens": 0, "forced_failed": 0,
                      "empty_replies": 0, "empty_stop": 0, "empty_length": 0, "empty_after_forced_close": 0,
                      "empty_retried": 0, "empty_retry_recovered": 0, "empty_retry_exhausted": 0,
                      "budget_from": {}}
        self.think_mu = threading.Lock()  # (counted from the scheduler thread and the event loop: review server F9)
        self.drop_log = DroppedTurnLog()  # --drop-empty-turns (HF #50): logged once per conversation
        # tool_choice (server/tool_choice.py): masks through the same grammar tier
        tier = tool_choice.Tier
        if TOOL_FORMAT == "dsml":  # [shared-srv] DSV4: the call header is a DSML block
            from dsv4_engine import DsmlTier as tier
        elif TOOL_FORMAT == "harmony":  # [srv-gptoss] gpt-oss: message headers (server/harmony.py)
            from harmony import HarmonyTier as tier
        elif TOOL_FORMAT == "glm":  # [glm-rel-tools] GLM: <tool_call>NAME<arg_key>..
            from tool_choice import GlmTier as tier
        self.tool_tier = tier(self.tok, self.think_ids, self.eos, self.im_start) \
            if self.engine.has("grammar") else None
        if self.tool_tier is not None and not self.tool_tier.ok:
            self.tool_tier = None
        self.tool_choice_noted = False  # the first request whose required / named tool_choice is not enforced
        # [tool-choice-ban] tool_choice none through the engine's static token ban (no constrained path); QE_TOOL_BAN=0:
        # the mask path as before (off-switch)
        self.tool_ban = self.tool_tier is not None and self.engine.has("qe_ban_tokens") and env_int("QE_TOOL_BAN", 1) != 0
        if self.tool_tier is None:
            print("tool_choice: not enforced through masks (the engine has no grammar C API"
                  f"{': ' + self.engine.refused['grammar'] if getattr(self.engine, 'refused', {}).get('grammar') else ''}"
                  "): none ends a reply at its first tool call, required / any / a named tool decode as auto",
                  flush=True)
        # [feature-matrix] what the loaded engine can do: start-up lines for flags / variables that need a capability
        # it lacks, and the request parameters it does not support (model_profile.FLAGS / ENV / PARAMS)
        self.caps = engine_caps(self.engine, prof.family, grammar=self.grammars is not None, force=self.can_force,
                                vision=self.vision is not None, tool_tier=self.tool_tier is not None)
        for line in ignored_settings(prof.family, getattr(args, "given", None), getattr(args, "env0", None),
                                     caps=self.caps, args=args):
            print(line, flush=True)
        self.param_notes = ParamNotes(prof.family, self.caps)
        # [gptoss-lows] gpt-oss's idle-chunk rules (pfidle L1 / L2); pf_pass: rows of one engine prefill pass (0: unknown)
        self.gptoss = prof.family == "gptoss"
        pc = getattr(self.engine, "prefill_chunk", None) if self.gptoss else None
        self.pf_pass = int(pc()) if pc is not None else 0
        if not self.can_force:
            print("thinking budget: not enforced (the engine has no grammar C API or the tokenizer no <think> "
                  "tokens): budgets are ignored and max_tokens may end inside <think>", flush=True)
        # how the template renders reasoning_effort (the key of a per-effort budget map; issue #4)
        probe = [{"role": "user", "content": "probe"}]
        self.effort_rules = probe_rules(
            lambda e: self.tmpl.render([dict(m) for m in probe], None, enable_thinking=True, reasoning_effort=e),
            log=lambda m: print(m, flush=True))
        self.budget_map = BudgetMap.parse(getattr(args, "thinking_budget", 0))
        if self.can_force:
            for line in budget_banner(self.budget_map, self.effort_rules):
                print(line, flush=True)
        from metrics import Metrics
        stats = os.path.expanduser(args.stats_file)
        self.metrics = Metrics(stats, args.served_name)
        # main agent thread detection (server/agents.py) and the scheduler's counters, both in /api/stats
        self.agents = Tracker(override=getattr(args, "main_agent_re", None))
        self.sched = {"pauses": 0, "backfills": 0}
        snap = self.metrics.snapshot
        version, build = version_info(getattr(self.engine, "lib_path", None))  # qetop / dashboard header
        # parallel mode of the engine (FLASH_PAR / the boot-time pick: mode, why, probe numbers)
        par_info = getattr(self.engine, "par_info", None)
        host_mem = getattr(self.engine, "host_mem", None)  # the host-RAM plan of the boot (src/host_budget.h)
        graph_info = getattr(self.engine, "graph_info", None)  # cached CUDA graphs per card (FLASH_GRAPH_MB)
        def wrapped():
            base = snap()
            extra = dict(agents=self.agents.stats(), sched=dict(self.sched), loop_guard=self.guard_stats(),
                         parallel=par_info() if par_info else {"mode": "tp"},
                         host_memory=host_mem() if host_mem else {},
                         graphs=graph_info() if graph_info else {},
                         thinking=self.think_snapshot(), history_drops=self.drop_log.stats(),
                         steering=self.steer_stats(), version=version, build=build)
            clash = base.keys() & extra.keys()  # a shadowed metrics key breaks the dashboards (see history_drops)
            assert not clash, f"/api/stats keys shadow metrics keys: {sorted(clash)}"
            return dict(base, **extra)
        self.metrics.snapshot = wrapped
        if args.metrics_port:
            self.metrics.serve(args.metrics_host, args.metrics_port)
        if env_str("QE_WARMUP", "1") != "0":  # pre-build decode graphs (no first-use stalls)
            # sampling mode: requests normally carry a temperature (Claude Code does), and the
            # server defaults sample too
            print(f"graph warm-up: {self.engine.warmup(greedy=self.sampling_defaults[True]['temperature'] <= 0):.1f}s", flush=True)
        self.engine.set_concurrent(True)
        self.cv = threading.Condition()
        self.queue = []
        self.stopping = False  # SIGTERM / SIGINT: requests end with an error event (stop_requests)
        self.ctrl = []  # (fn, done event) run by the scheduler thread when nothing is generating
        self.n_faults = 0  # engine NaN guard faults seen (faults())
        self.cur_slot = -1
        self.dirty = False  # slots changed since the last persist
        self.hb, self.hb_what, self.hb_limit = time.monotonic(), "start", HANG_S  # scheduler heartbeat (_hang_monitor)
        self.jobs_ref = ([], [])  # the scheduler's (active, prefills), for the hang monitor
        self.retry_q = []  # (time, job, retry_gen): starts refused after admission (see _loop)
        self.retry_gen = 0  # +1 whenever a sequence or a prefill ends: the refused starts try again
        self.starting = None  # the job inside begin_start (hang monitor)
        self.engine.set_exit_hook(self._on_engine_exit)
        threading.Thread(target=self._loop, daemon=True).start()
        if HANG_S > 0:
            threading.Thread(target=self._hang_monitor, daemon=True).start()

    def idle_budget(self, budget, busy, prefills, adm):
        """The rows of the next prefill call: the larger idle chunk when nothing decodes (see the rules below), else
        `budget` (--prefill-chunk)."""
        idle = self.args.prefill_chunk_idle
        # the larger idle batch when nothing decodes: one prompt and nothing waiting, or several sharing it; on
        # 2 cards (pass_bound: a pass costs ~0.6 s of PCIe expert copies whatever its rows) also while requests
        # wait (8 x 16K burst 3.0K -> 4.4K tok/s; 3 cards: no gain)
        # gpt-oss (review pfidle L2): a queued request that can never cut in (larger than any cut-in: it waits for the
        # whole prompt anyway) does not take the idle chunk away from it (2,048-row calls slowed both). One that may cut
        # in later (the fairness gap after a cut-in: review lows M1) keeps the small chunk, as before. (pick_next only
        # reads the queue here; the engine thread is its only remover.)
        gpo = getattr(self, "gptoss", False)
        waiting = bool(self.queue)
        if waiting and gpo and not adm.pick_next(time.time(), prefills):
            cut = max(adm.pre_max or 0, adm.short if adm.qa_cut else 0)
            waiting = any(j.est is None or j.est <= cut for j in list(self.queue))
        if idle > budget and not busy and ((len(prefills) == 1 and not waiting) or
                                           (adm.multi and (len(prefills) >= 2 or adm.pass_bound))):
            budget = idle
            if gpo and self.pf_pass > 0 and budget > self.pf_pass:
                # a call splits into the fewest equal engine passes: keep it a multiple of the engine's chunk
                # where that is cut below the idle chunk (no P2P without host lanes, low pinned RAM: pfidle L1)
                budget = budget // self.pf_pass * self.pf_pass
        return budget

    def count_refills(self):
        """[dsv4-rel8-refill] (qetop review L2) re-prefills the engine ran inside its last call (DSV4: a switch to a
        conversation whose K/V left the archive) are prefill: their tokens and seconds go to the prefill counters of
        /api/stats (the count and totals: engine.dsv4.refills). Other engines have no take_refills: nothing."""
        f = getattr(self.engine, "take_refills", None)
        ev = f() if f is not None else None
        if not ev:
            return
        m = self.metrics  # (server/metrics.py stays as it is: its session counters under its lock)
        try:
            with m.lock:
                for _slot, n, dt, in_pf in ev:  # (review rel8 L3: a redo inside a prompt's prefill: its time is in
                    m.session["prefill_tokens"] += n  # that prefill's already, only the extra tokens count)
                    if not in_pf:
                        m.session["prefill_s"] += dt
                    m.cur["pre"] += n
        except Exception as ex:
            print(f"[stats] re-prefill counters: {ex!r}", flush=True)

    def call_limit(self, kind, slots=()):
        """[dsv4-rel8-hang] the hang deadline of the next engine call (None: QE_HANG_S). An engine whose calls can prefill
        a whole prompt or re-prefill a conversation (DSV4) says how many tokens the call may prefill; the others: None."""
        f = getattr(self.engine, "call_tokens", None)
        n = f(kind, slots) if f is not None else 0
        return HANG_S + n / HANG_PREFILL_TPS if n > 0 and HANG_S > 0 else None

    def beat(self, what, limit=None):
        """Scheduler heartbeat. limit: the hang deadline until the next beat (a known long operation, e.g. the disk
        persist), else HANG_S."""
        self.hb, self.hb_what, self.hb_limit = time.monotonic(), what, limit or HANG_S

    def fail_open(self, ex):
        """Every open request (queued, waiting for a retry, starting, prefilling, generating) ends with `ex` now, without
        an engine call and without the scheduler's lock (the engine or the scheduler is the stuck part; the process
        exits next). Twice, a moment apart: a request that was being queued during the first pass."""
        for k in range(2):
            active, prefills = self.jobs_ref
            for j in list(active) + list(prefills) + list(getattr(self, "parked_ref", [])) + list(self.queue) + \
                    [r[1] for r in self.retry_q] + \
                    ([self.starting] if self.starting is not None else []):
                if not j.finished:
                    j.error, j.finished = ex, True
                    j.done.set()
            if k == 0:
                time.sleep(0.2)

    def _on_engine_exit(self, reason):
        """The engine ends the process (a stall seen by its run() monitor, a CUDA error, a failed check): the open
        requests get an error event first (engine thread, bounded by QE_EXIT_HOOK_S)."""
        self.stopping = True
        print(f"[fatal] the engine is ending the process ({reason or 'fatal error'}): the open requests end with an "
              f"error", flush=True)
        self.fail_open(SequenceFault(f"{reason or 'fatal error'}; the server restarts"))  # (the API adds "engine error:")
        time.sleep(2.0)  # the error events reach the clients

    def _hang_monitor(self):
        """No scheduler heartbeat for HANG_S seconds: an engine call does not return (a worker stuck in the driver, a
        stalled GPU). The requests end with an error, new ones are refused and a hard exit is armed first; then the
        Python and native stacks and the GPUs' state are logged (bounded), and the server exits (75) so it is
        restarted, instead of hanging silently (issue #3). Also during a shutdown (SIGTERM), with twice the deadline."""
        while True:
            time.sleep(2.0)
            age = time.monotonic() - self.hb  # monotonic: a wall-clock step (NTP) is no hang
            limit = self.hb_limit * (2 if self.stopping else 1)
            if age < limit:
                continue
            t_hang = time.monotonic()
            # 1. no new work, every open request gets its error now (no engine call: the engine is the stuck part;
            # no lock: the stuck scheduler may hold self.cv), and the hard exit is armed before any diagnostics
            self.stopping = True
            hard_exit_guard(QE_HANG_DIAG_S + 3.0)
            what = self.hb_what
            print(f"[hang] the scheduler has not advanced for {age:.0f} s (deadline {limit:.0f} s, QE_HANG_S {HANG_S:.0f}); "
                  f"last: {what}. The open requests end with an error; stacks follow (at most {QE_HANG_DIAG_S:.0f} s), "
                  f"then the server exits (75).", flush=True)
            self.fail_open(SequenceFault(f"engine hang: no progress for {age:.0f} s (last: {what}); the server restarts"))
            # 2. diagnostics, each bounded by what is left of QE_HANG_DIAG_S
            def left():
                return QE_HANG_DIAG_S - (time.monotonic() - t_hang)
            try:
                faulthandler.dump_traceback(all_threads=True)
                native_stacks(timeout=max(1.0, left() - 10.0))
                if left() > 2.0:
                    out = run_bounded(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,pstate",
                                       "--format=csv,noheader"], min(10.0, left() - 1.0))
                    if out is not None:
                        print(f"[hang] nvidia-smi: {out.strip()}", flush=True)
            except Exception as e2:
                print(f"[hang] diagnostics: {e2!r}", flush=True)
            # 3. the error events reach the clients (at least 3 s after they were set), then exit
            time.sleep(max(0.0, min(3.0 - (time.monotonic() - t_hang), left())))
            hard_exit(75)

    def prompt(self, messages, tools, kw, meta=None, track=False, guard_on=True, guard_think=True):
        """Returns (ids, affinity): affinity = hash of the system prompt, so the main agent and
        each kind of subagent keep returning to their own sequence slot. ids.hint = position of
        the first user turn (end of the system part). track: a generation request (not a token
        count): the main-thread tracker sees it (meta: the request's metadata, its session id). guard_think: a guarded
        turn may be generated with reasoning on (kw["enable_thinking"] is then set True: the caller's parser follows
        it; False for structured output, constrained from the first token)."""
        info = None
        if track:
            sys0 = "".join(str(m.get("content")) for m in messages[:1] if m.get("role") == "system")
            aff0 = int.from_bytes(hashlib.blake2b(sys0.encode(), digest_size=8).digest(), "little")
            try:
                info = self.agents.observe(messages, tools, meta, aff0)
            except Exception as ex:  # monitoring / priority only: never fail a request for it
                print(f"agents: {ex!r}", flush=True)
        # (before the images are rewritten: without vision they become one fixed text, not a repeated result)
        guard = self.loop_check(messages, info) if track and guard_on else None
        images = []
        for m in messages:  # image parts in prompt order -> placeholders (+ their sources)
            c = m.get("content")
            if not isinstance(c, list):
                continue
            parts = []
            for p in c:
                if p.get("type") != "image":
                    parts.append(p)
                elif self.vision is not None:
                    images.append(p["_src"])
                    parts.append({"type": "image"})
                else:
                    parts.append({"type": "text", "text": "[image omitted: this server runs without vision]"})
            m["content"] = parts
        plain, kw0 = messages, dict(kw)  # (a note that would not fit: the turn goes without it, below)
        if guard and self.args.loop_guard_note and not guard["human"]:  # the repeated results tell it it is looping
            messages, guard["note"] = loop_guard.add_note(messages, guard["turns"], guard["distinct"])

        def build(messages, tools, kw):
            messages = self.clean_history(messages, info, track)
            if getattr(self.args, "sort_tools", 1):
                tools = canonical_tools(tools)
            mk = getattr(self, "marks", None)
            r_msgs, r_tools = (mk.protect_messages(messages), mk.protect(tools)) if mk is not None else (messages, tools)
            if self.profile.is_glm:
                # GLM template: reasoning_effort in (low, high), anything else -> max; no enable_thinking variable: the
                # generation prompt ends with <think>, so thinking off closes it right away
                kw_in, kw = kw, dict(kw)
                eff = kw.get("reasoning_effort")
                if eff is not None:
                    kw["reasoning_effort"] = {"minimal": "low", "low": "low", "medium": "high", "high": "max"}.get(str(eff), eff)
                think = kw.pop("enable_thinking", True)
                text = self.tmpl.render(r_msgs, r_tools, **kw)
                if not think and guard and guard_think and getattr(self.args, "loop_guard_think", 0) > 0 \
                        and self.can_force and text.endswith("<think>"):
                    # the loop guard's thinking turn on GLM (master e4489b18, PLAN.md GLM merge list): the same prompt with
                    # the generation prompt's <think> left open, so this turn reasons first
                    kw["enable_thinking"] = kw_in["enable_thinking"] = True  # review r17 L8: the caller sees it too
                    guard["think"] = self.args.loop_guard_think
                else:
                    kw["enable_thinking"] = think
                    text += "" if think else "</think>"
            elif getattr(self.profile, "is_gptoss", False):
                # gpt-oss always reasons: thinking off = the lowest effort (the template has no off switch)
                kw = dict(kw)
                if kw.pop("enable_thinking", True) is False and kw.get("reasoning_effort") is None:
                    kw["reasoning_effort"] = self.profile.defaults.get("no_effort", "low")
                text = self.tmpl.render(r_msgs, r_tools, **kw)
            elif getattr(self.profile, "is_inkling", False):
                # [ink-srv-capi] Inkling 2: thinking is the template's reasoning_effort (none / minimal / low / medium /
                # high / xhigh / max or a number in [0, 0.99]); thinking off = none (0); no effort = its default 0.9
                import inkling
                text = self.tmpl.render(r_msgs, r_tools, **inkling.effort(kw))
            elif getattr(self.profile, "is_dsv4", False):
                # DeepSeek-V4.1 (dsv4_chat): enable_thinking ends the prompt with <think> or </think>; the loop guard's
                # thinking turn renders the same messages with thinking on
                text = self.tmpl.render(r_msgs, r_tools, **kw)
                if guard and guard_think and getattr(self.args, "loop_guard_think", 0) > 0 and kw.get("enable_thinking") is False \
                        and self.can_force and text.endswith("</think>"):
                    # (the same non-thinking render, so the cached prefix holds: only the generation prompt opens <think>;
                    # a thinking render would add the effort line to the system turn: review M5)
                    text = text[:-len("</think>")] + "<think>"
                    kw["enable_thinking"] = True
                    guard["think"] = self.args.loop_guard_think
            else:
                text = self.tmpl.render(r_msgs, r_tools, **kw)
                if guard and guard_think and getattr(self.args, "loop_guard_think", 0) > 0 and kw.get("enable_thinking") is False \
                        and self.can_force and text.endswith(EMPTY_THINK):  # (review 2 L6: only with the budget enforced)
                    # a looping agent on a no-think server: this turn reasons first (session 92997599: 10/10 replays broke the
                    # loop, presence / note alone did not). The prompt is rendered without thinking (same system header and
                    # history, so the cached prefix holds) and only the generation prompt's empty think block is opened.
                    text = text[:len(text) - len(EMPTY_THINK)] + "<think>\n"
                    kw["enable_thinking"] = True
                    guard["think"] = self.args.loop_guard_think
            sys_txt = "".join(str(m.get("content")) for m in messages[:1] if m.get("role") == "system")
            label = "main agent" if info is not None and info.main else agent_label(sys_txt, messages)
            affinity = int.from_bytes(hashlib.blake2b(sys_txt.encode(), digest_size=8).digest(), "little")
            if env_str("QE_DUMP_PROMPTS"):
                d = env_str("QE_DUMP_PROMPTS")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"{time.time():.3f}.txt"), "w") as f:
                    f.write(mk.restore(text) if mk is not None else text)
            ids = self.encode(text)
            if getattr(self.profile, "is_dsv4", False):  # [dsv4-rel6-xprefix] the shared part: the system message with the tools
                ids.hint = 0 if (self.args.tier_reset or self.args.no_prefix_share) else \
                    self.tmpl.system_hint(r_msgs, r_tools, kw, ids, self.encode)
            return ids, label, affinity

        try:
            ids, label, affinity = build(messages, tools, kw)
        except jinja2.TemplateError as ex:  # the template's own checks (no messages, an unknown role, ...): HTTP 400
            raise InputError(f"the chat template refused the messages: {ex}") from ex
        except TypeError as ex:  # [gptoss-rel1-parts] gpt-oss's template joins strings: a message field of another type
            if not getattr(self.profile, "is_gptoss", False):
                raise
            print(f"[server] gpt-oss: render TypeError (HTTP 400): {traceback.format_exc(limit=3)!r}", flush=True)  # (review rel1 L14)
            raise InputError(f"the chat template could not render the messages (a field of an unexpected type): {ex}") from ex
        raw = []  # the request's images, read once

        def finish(ids, label):  # the GLM system-part hint, the labels, the image tokens
            if getattr(self.profile, "is_gptoss", False) and (self.args.tier_reset or self.args.no_prefix_share):
                ids.hint = 0  # [srv-gptoss-cache] no shared system-part snapshot
            if getattr(self.profile, "is_inkling", False):  # [ink-srv-capi] the system part (tools, system, effort)
                u = self.tok.token_to_id("<|message_user|>")
                ids.hint = ids.index(u) if u is not None and u in ids else 0
            if self.profile.is_glm:
                ids.hint = 0  # (encode()'s Qwen rule would match a literal "<|im_start|>user" inside the text)
                # GLM: one segment (no <|im_start|>); the end of the system part = the first <|user|> token (round 18: the
                # engine keeps a prefix snapshot there, shared by conversations with the same system prompt and tools)
                u = self.tok.token_to_id("<|user|>")
                if u is not None:
                    try:
                        ids.hint = ids.index(u)
                    except ValueError:
                        pass
            ids.label, ids.agent, ids.guard = label, info, guard
            if images:
                from vision import load_image_bytes
                t0 = time.time()
                try:
                    raw[:] = raw or [load_image_bytes(s) for s in images]
                    with self.vision_lock:
                        exp, emb, rope3, delta = self.vision.expand(list(ids), raw)
                except InputError:
                    raise
                except Exception as ex:  # undecodable base64 / not an image / URL not reachable / placeholder mismatch
                    raise InputError(f"image input: {ex}") from ex
                hint = ids.hint
                ids = Ids(exp)
                ids.hint, ids.mm, ids.label, ids.agent, ids.guard = hint, (emb, rope3, delta), label, info, guard
                ids.img = (len(images), emb.shape[0])
                print(f"{len(images)} image(s): {emb.shape[0]} tokens, encoded in {time.time() - t0:.2f}s", flush=True)
            return ids

        ids = finish(ids, label)
        if guard and guard.get("note") and self.too_long(ids):
            # the note would push the prompt over the context (review L6): the guarded turn goes without it (reasoning
            # and penalty stay), rendered again from the client's own messages (checked with the image tokens)
            kw.clear()
            kw.update(kw0)
            guard.pop("think", None)
            guard["note"] = False
            print(f"loop guard: the note would make the prompt {len(ids)} tokens, over the context: left out", flush=True)
            ids, label, affinity = build(plain, tools, kw)
            ids = finish(ids, label)
        return ids, affinity

    def loop_check(self, messages, info):
        """Cross-turn loop guard (server/loop_guard.py): None, or the guard of this request's turn."""
        a = self.args
        if getattr(a, "loop_guard_turns", 0) <= 0:
            return None
        try:
            st = loop_guard.state(messages)
        except Exception as ex:  # a guard must never fail a request
            print(f"loop guard: {ex!r}", flush=True)
            return None
        k = st["turns"]
        if k >= loop_guard.WINDOW:
            st = dict(st, total=self.guard_total(st, getattr(getattr(info, "conv", None), "id", None)))
        # a human just spoke: guarded at once, unless the turn before had human text too (a client that adds text after
        # every result: the back-off then holds, review 2 L2)
        human_now = st["human"] and not st.get("human_prev")
        if k < a.loop_guard_turns or not (human_now or loop_guard.due(st, a.loop_guard_turns)):
            return None
        # the human just spoke (maybe "run it again"): only think first, no penalty, no note
        pres = 0.0 if human_now else a.loop_guard_penalty if k < 2 * a.loop_guard_turns else max(a.loop_guard_penalty, 2.0)
        conv = getattr(getattr(info, "conv", None), "id", None)
        sess = getattr(getattr(info, "session", None), "key", "") or ""
        return {"turns": k, "distinct": st["distinct"], "human": human_now, "presence": pres, "conv": conv,
                "session": sess[2:10], "main": bool(getattr(info, "main", False))}

    def guard_total(self, st, conv):
        """The turn count due() keys on past the window (review 2 L3): the history's own, while it grows; a client that
        keeps only its last turns sends the same count on every turn, so a repeat of the same stretch at the same count
        adds one (a re-sent identical request counts as a turn too). Bounded: the last 256 stretches."""
        key = (conv, st.get("key"))
        with self.seg_lock:
            seen = self.guard_seen.get(key)
            n = seen[1] + 1 if seen is not None and seen[0] == st["total"] else 0
            self.guard_seen[key] = (st["total"], n)
            self.guard_seen.move_to_end(key)
            while len(self.guard_seen) > 256:
                self.guard_seen.popitem(last=False)
        return st["total"] + n

    def guard_note(self, guard):
        """A guarded turn starts: warning line, diag log, /api/stats (qetop)."""
        ev = {"t": time.time(), "turns": guard["turns"], "conversation": guard["conv"], "session": guard["session"],
              "main": guard["main"], "presence": guard["presence"]}
        with self.seg_lock:
            self.guards["triggers"] += 1
            self.guards["recent"] = (self.guards["recent"] + [ev])[-8:]
        print(f"WARNING loop guard: conversation {guard['conv']} (session {guard['session'] or '-'}) repeated the "
              f"same turns {guard['turns']} times ({guard['distinct']} different): next turn with presence penalty "
              f"{guard['presence']:.2f} on the repeated reply ({guard.get('seed', 0)} tokens), lookup drafts off"
              f"{', note appended to the tool result' if guard.get('note') else ''}"
              f"{', reasoning on (budget %d)' % guard['think'] if guard.get('think') else ''}"
              f"{', after a human message' if guard.get('human') else ''}", flush=True)
        diag("LOOP_GUARD", f"conversation {guard['conv']} session {guard['session']} turns {guard['turns']} "
             f"distinct {guard['distinct']} presence {guard['presence']:.2f} note {int(bool(guard.get('note')))} "
             f"think {guard.get('think', 0)} human {int(bool(guard.get('human')))}")

    def steer_stats(self):
        """/api/stats steering: loaded vector names, the default, and the vectors dropped at start (with the reason)"""
        st = getattr(self, "steer", None) or {}
        return {"enabled": bool(st.get("vectors")), "default": st.get("default"), "default_active": st.get("default") is not None,
                "vectors": [v.get("name") for v in st.get("vectors", [])], "dropped": st.get("dropped", [])}

    def pf_progress(self, jobs, others=()):
        """Live prefill progress for /api/stats (qetop, dashboard), after each prefill chunk / shared batch: the prompt rows
        computed so far (prefill_tok), the rows to compute (prefill_total), the rate since the prefill began (prefill_tps,
        as prefill_done counts it, which overwrites them with the final numbers). Before, the prefill row had only its
        elapsed time until the whole prompt was done. One prefill_pos read per prompt per chunk; never raises. others:
        the other prompts in flight, whose rate only decays while they wait (no engine read: Flash-Next pf_of ignores
        the slot without the batch API)."""
        m, now = self.metrics, time.time()
        for j in others:
            if any(j is k for k in jobs):
                continue
            try:
                with m.lock:
                    x = m.jobs.get(id(j))
                    if x is not None and x.get("state") == "prefill" and "prefill_total" in x:
                        x["prefill_tps"] = x["prefill_tok"] / max(now - x["t_p"], 1e-6)
            except Exception:
                pass
        for j in jobs:
            try:
                pos = self.engine.prefill_pos(j.slot)
                if pos <= getattr(j, "pf0", 0):  # done (-1: finish_start counts it), or no rows yet (Flash-Next mode B
                    continue  # moves them only at its finish): the row keeps its elapsed time
                p0 = getattr(j, "pf0", 0)
                with m.lock:
                    x = m.jobs.get(id(j))
                    if x is None or x.get("state") != "prefill":
                        continue
                    done = max(pos - p0, 0)
                    x["prefill_tok"], x["prefill_total"] = done, max(len(j.ids) - p0, 0)
                    x["prefill_tps"] = done / max(now - x["t_p"], 1e-6)
            except Exception:
                pass

    def guard_stats(self):
        with self.seg_lock:
            return {"turns": getattr(self.args, "loop_guard_turns", 0), "triggers": self.guards["triggers"],
                    "recent": list(self.guards["recent"])}

    def clean_history(self, messages, info=None, track=False):
        """Model-specific history fixes before the template (--merge-system, --drop-empty-turns). track: a
        generation request (info: the agent tracker's view of it); the HF #50 drop is logged once per
        conversation and again only when its count grows."""
        if getattr(self.profile, "is_gptoss", False):  # [srv-gptoss] reasoning -> thinking, tool arguments as objects
            import harmony
            messages = harmony.prepare(messages)
        if getattr(self.profile, "is_inkling", False):  # [ink-srv-capi] text parts, reasoning_content, arguments as objects
            import inkling
            messages = inkling.prepare(messages)
        if self.args.merge_system:
            messages = merge_leading_system(messages)
        if self.args.drop_empty_turns:
            messages, n = drop_empty_turns(messages)
            if n and track:
                conv = getattr(getattr(info, "conv", None), "id", None)
                if conv is not None:
                    key, who = ("conv", conv), f"conversation {conv}"
                else:  # no tracker conversation (--main-agent-re): the conversation's root (system + first user)
                    root = json.dumps(messages[:2], sort_keys=True, default=str)
                    key = ("root", hashlib.blake2b(root.encode(), digest_size=8).hexdigest())
                    who = f"conversation {key[1][:8]}"
                sess = (getattr(getattr(info, "session", None), "key", "") or "")[2:10]
                prev = self.drop_log.note(key, n)
                if prev is not None:
                    print(f"history: {who}{f' (session {sess})' if sess else ''}: the client's history has {n} "
                          f"assistant turn(s) with reasoning only{f' (was {prev})' if prev else ''}; left out of "
                          f"the prompt (HF #50, --drop-empty-turns 0 keeps them)", flush=True)
        return messages

    def encode(self, text):
        """Tokenize chat text segment by segment (each starts with <|im_start|>, an atomic special
        token, so the result equals tokenizing the whole text) with a cache: every turn of a long
        conversation only tokenizes its new messages (84K tokens: ~85 ms -> <1 ms). A segment with
        content markers (sentinels, see prompt()) is tokenized with those markers as plain text."""
        ids = Ids()
        mk = getattr(self, "marks", None)
        with self.seg_lock:
            sm = getattr(self, "seg_mark", "<|im_start|>")
            for i, part in enumerate(text.split(sm)):
                seg = part if i == 0 else sm + part
                if not seg:
                    continue
                t = self.seg_cache.get(seg)
                prot = mk is not None and SENT in seg
                if t is None and seg.startswith("<|im_start|>assistant"):
                    t = self.gen_turns.get(turn_key(mk.restore(seg) if prot else seg))
                    if t is not None:
                        t = list(t) + self.im_end_nl if seg.endswith("<|im_end|>\n") else None
                    if t is not None:
                        self.seg_cache[seg] = t
                        self.seg_tokens += len(t)
                if t is None:
                    t = mk.encode_segment(seg) if prot else self.tok.encode(seg, add_special_tokens=False).ids
                    self.seg_cache[seg] = t
                    self.seg_tokens += len(t)
                    while self.seg_tokens > 2_000_000 and self.seg_cache:
                        _, old = self.seg_cache.popitem(last=False)
                        self.seg_tokens -= len(old)
                else:
                    self.seg_cache.move_to_end(seg)
                # (harmony, segments at <|start|>: the system and developer messages end at the first user message)
                if not ids.hint and (seg.startswith("<|im_start|>user") or (sm == "<|start|>" and seg.startswith("<|start|>user"))):
                    ids.hint = len(ids)
                ids.extend(t)
        return ids

    def remember(self, affinity, prompt, out):
        with self.seg_lock:
            lst = self.turns.setdefault(affinity, [])
            lst.append((len(prompt), len(out), array.array("i", list(prompt) + list(out))))
            del lst[:-16]
            self.turns.move_to_end(affinity)
            while sum(len(t[2]) for v in self.turns.values() for t in v) > 30_000_000:  # ~120 MB
                k = next(iter(self.turns))
                if len(self.turns) == 1:
                    del self.turns[k][:-1]
                    break
                self.turns.popitem(last=False)

    def check_reuse(self, job):
        """Diagnostics: a request continuing one of our recent replies should resume from the live
        state (cached ~ previous prompt + reply). Logs when the reply came back different from
        what was generated, or when it matched but the engine still re-prefilled it."""
        ids = array.array("i", job.ids)
        with self.seg_lock:
            cand = list(self.turns.get(job.affinity, []))
        best = None
        for np_, no, po in cand:
            l = lcp_len(po, ids)
            if l >= np_ and (best is None or np_ > best[0]):
                best = (np_, no, po, l)
        if best is None:
            return
        np_, no, po, l = best
        if l < np_ + no:
            if l == len(ids):
                return  # a retry / shorter continuation
            dec = lambda t: self.tok.decode(list(t), skip_special_tokens=False)
            diag("REPLY_CHANGED", f"slot {job.slot} prompt {len(ids)} cached {job.reused}: reply of {no} tok "
                 f"differs at +{l - np_} after {dec(po[max(np_, l - 12):l])!r}: generated "
                 f"{dec(po[l:l + 24])!r} sent back {dec(ids[l:l + 24])!r}")
        elif job.reused < np_ + no - 2:
            diag("REPLY_REPREFILLED", f"slot {job.slot} prompt {len(ids)} cached {job.reused} "
                 f"(src {job.src or 'live/snapshot'}): previous prompt {np_} + reply {no} matched "
                 f"but was computed again")
    def remember_turn(self, prompt, out):
        """The engine's live state holds the tokens the model generated; re-tokenizing the reply
        when the client sends it back often gives other ids (merged whitespace, trimmed reasoning,
        re-serialized tool arguments), so the next request would roll back to the end of the
        previous prompt and prefill the whole reply again. Keep the generated ids (with the
        assistant header from the prompt) keyed by the turn's text without whitespace; encode()
        uses them for a matching assistant segment.
        Bounded by tokens (GEN_TURNS_TOKENS), not by turns: a turn whose ids differ from its re-tokenized text (a
        real tag token the Parser passed through as text comes back as plain text, review L2; merged whitespace)
        must keep its ids as long as its KV can still be reused (the live pool and the RAM archive hold far fewer
        reply tokens than the bound), or the same history would tokenize differently once the turn was evicted
        (up to 1.3.4.3: 1024 turns, which agent sessions pass while the main thread's KV is still archived)."""
        if self.im_start not in prompt[-16:]:
            return
        head = prompt[len(prompt) - prompt[::-1].index(self.im_start) - 1:]
        t = array.array("i", list(head) + list(out))
        k = turn_key(self.tok.decode(list(t), skip_special_tokens=False) + "<|im_end|>")
        with self.seg_lock:
            old = self.gen_turns.pop(k, None)
            n = getattr(self, "gen_turn_tokens", 0) - (len(old) if old is not None else 0) + len(t)
            self.gen_turns[k] = t
            while n > GEN_TURNS_TOKENS and len(self.gen_turns) > 1:
                _, x = self.gen_turns.popitem(last=False)
                n -= len(x)
            self.gen_turn_tokens = n

    def system_end(self, ids):
        """Position of the first user turn: a snapshot there lets new sessions / subagents with
        the same system prompt and tools skip re-computing it."""
        if isinstance(ids, Ids):
            return ids.hint
        pat = self.im_start_user
        for i in range(len(ids) - len(pat) + 1):
            if ids[i:i + len(pat)] == pat:
                return i
        return 0

    def too_long(self, ids):
        """The error message when the prompt does not fit the largest context slot (checked before a stream starts:
        Anthropic clients compact the conversation on a 400 'prompt is too long'), else None."""
        cap = max(self.engine.slots) - self.args.mtp_k - 2
        return f"prompt is too long: {len(ids)} tokens > {cap} maximum" if len(ids) > cap else None

    def think_snapshot(self):
        with self.think_mu:
            out = dict(self.think)
            out["budget_from"] = dict(out.get("budget_from") or {})
        bm = getattr(self, "budget_map", None)
        if bm is not None:
            out["budget_default"] = bm.config()
            out["effort_rules"] = self.effort_rules.describe()
        return out

    def think_src(self, src):
        with self.think_mu:
            d = self.think.setdefault("budget_from", {})
            d[src] = d.get(src, 0) + 1

    def think_add(self, key, n=1):
        with self.think_mu:
            self.think[key] = self.think.get(key, 0) + int(n)

    def no_room_msg(self, n, k, waited):
        """Why a request that fits nowhere while nothing runs ends. Another process is named only when the pool is smaller
        than at start and the request would have fit at that size (the pool shrinks only under VRAM pressure)."""
        e = self.engine
        cur, start = e.pool_now(), e.pool_tokens
        facts = f"requested {n} tokens, KV pool now {cur} tokens (at start {start}), nothing else running, waited {waited:.0f} s"
        gpus = ", ".join(f"GPU{x['i']} {(x['mem_total'] - x['mem']) / 2**30:.1f} GB" for x in self.metrics.gpus)
        if gpus:
            facts += f", free VRAM {gpus}"
        if cur < start and n + k + 2 <= min(start, e.max_t):
            mem = "host RAM" if getattr(e, "kv_in_ram", False) else "GPU memory"  # FLASH_KV=ram (lazy pinning) shrinks with host RAM
            return (f"no room in the KV pool: another process took the {mem} and this request can't be finished ({facts}). "
                    f"Free the {mem} and send it again.")
        return f"no room in the KV pool: this request doesn't fit ({facts}). No sign of another process taking memory."

    def cancel_note(self, j, where):
        """[srv-disconnect] one line per request whose client left before its end: where it was stopped and how long
        after the disconnect its slot was free (the latency of the cancel path)"""
        c = j.cancelled
        why = getattr(c, "why", "client")
        if why != "client":
            return
        self.sched["cancelled"] = self.sched.get("cancelled", 0) + 1
        t = getattr(c, "t", None)
        dt = f"{time.monotonic() - t:.2f} s" if t is not None else "?"
        print(f"[sched] client gone: request of {len(j.ids)} tokens cancelled while {where}; slot free {dt} after the "
              f"disconnect", flush=True)

    def drop_cancelled_queued(self):
        """[srv-disconnect] requests whose client left while they waited (queue, refused-start retries) end now, before
        any engine work (they were admitted and prefilled in full before)"""
        with self.cv:
            gone = [j for j in self.queue if j.cancelled()] + [r[1] for r in self.retry_q if r[1].cancelled()]
            if not gone:
                return
            self.queue[:] = [j for j in self.queue if not j.cancelled()]
            self.retry_q = [r for r in self.retry_q if not r[1].cancelled()]
        for j in gone:
            j.slot = -1
            j.fail(Cancelled("cancelled while queued"))
            self.cancel_note(j, "queued")

    def stop_requests(self):
        """Shutdown (SIGTERM / SIGINT, before uvicorn waits for the open streams): queued requests fail now, prompts
        being prefilled at the scheduler's next pass, generating ones at their next step; each ends with an error
        event, so the server exits instead of waiting for them (docker stop without SIGKILL)."""
        with self.cv:
            self.stopping = True
            q, self.queue = self.queue, []
            q += [r[1] for r in self.retry_q]  # starts refused for room, waiting to be requeued (review server F6)
            self.retry_q = []
            for j in q:
                j.fail(SequenceFault("server is shutting down"))
            self.cv.notify()
        if q:
            print(f"shutdown: {len(q)} queued requests ended", flush=True)

    def run(self, ids, p, on_text, cancelled, affinity=0):
        """Blocking generation; calls on_text(str) for new text. Returns stats dict."""
        if self.stopping:
            raise SequenceFault("server is shutting down")
        if len(ids) + self.args.mtp_k + 2 > max(self.engine.slots):
            raise ValueError(f"prompt has {len(ids)} tokens; the largest context slot holds "
                             f"{max(self.engine.slots)}")
        job = Job(self, ids, p, on_text, cancelled, affinity)
        img = getattr(ids, "img", (0, 0))
        self.metrics.queued(job, getattr(ids, "label", "") or "chat", img[0], img[1])
        with self.cv:
            self.queue.append(job)
            self.cv.notify()
        job.done.wait()
        if job.error:
            raise job.error
        if getattr(job, "shut", False):
            raise SequenceFault("server is shutting down: the reply was cut")
        return job.stats

    # ------------------------------------------------------------ scheduler (engine thread)
    def faults(self, jobs):
        """The engine's NaN guard failed sequences (non-finite values): those requests end with an error, the
        others go on. True when one of `jobs` failed."""
        n = self.engine.fault_count()
        if n == self.n_faults:
            return False
        self.n_faults = n
        hit = False
        for j in jobs:
            msg = None if j.finished else self.engine.slot_fault(j.slot)
            # (rel160: Inkling too: its C API is the gpt-oss port and logs "the reply ends with finish_reason length", but
            # the server answered HTTP 500, 1.6.0 stress gate)
            if msg and (getattr(self.profile, "is_gptoss", False) or getattr(self.profile, "is_inkling", False)) \
                    and msg.startswith("KV pool full"):
                # [gptoss-rel1-full] no room for the reply to grow (or its context is full): the reply ends as at
                # max_tokens (finish_reason length), not with an error; the engine keeps the conversation's rows
                print(f"engine: slot {j.slot} ({len(j.ids)} prompt tokens, {getattr(j, 'n', 0)} generated): {msg}; the "
                      f"reply ends with finish_reason length", flush=True)
                j.max_new = getattr(j, "n", 0)  # (review rel1 L6: through feed, which flushes the held-back tail text)
                j.feed([])
                j.finish, j.finished = "length", True
                continue
            if msg:
                print(f"engine fault: request on slot {j.slot} ({len(j.ids)} prompt tokens, {getattr(j, 'n', 0)} "
                      f"generated) failed: {msg}", flush=True)
                j.fail(SequenceFault(msg))
                hit = True
        return hit

    def step_logprobs(self, j, active, tr):
        """One step of a logprobs reply, alone (single-sequence path): the engine keeps its raw logits per slot until its
        next step. Tokens are fed one by one: each reads its own row. Returns the tokens of the step."""
        e = self.engine
        if self.cur_slot != j.slot:
            e.resume(j.slot)
            self.cur_slot = j.slot
        ts = time.time()
        self.beat("step")
        toks = e.step()
        if not toks:
            self.cur_slot = -1
        tr("step", ts, 1, len(toks))
        self.metrics.step([j], [len(toks)], time.time() - ts)
        self.count_refills()
        self.faults(active)  # (all of them: a settle in pp can file another slot's fault)
        if j.finished:
            return 0
        for i, t in enumerate(toks):
            j.lp_row = i
            j.feed([t])
        self.no_progress(j, len(toks))
        return len(toks)

    def no_progress(self, j, n):
        """A sequence that gets no token for NOPROG_STEPS steps and NOPROG_S seconds while it is not paused fails
        with an error (and the pool state is logged) instead of looping silently (issue #3 D2)."""
        if n or j.finished or NOPROG_STEPS <= 0 or NOPROG_S <= 0:  # QE_NO_PROGRESS_STEPS / _S = 0: off
            j.empty_n, j.t_tok, j.t_vwait = 0, time.time(), None
            return
        j.empty_n = getattr(j, "empty_n", 0) + 1
        t_tok = getattr(j, "t_tok", None) or time.time()
        j.t_tok = t_tok
        if j.empty_n < NOPROG_STEPS or time.time() - t_tok < NOPROG_S:
            return
        try:
            pi = self.engine.pool_info(convs=False)
            pz = pi["slots"][j.slot]["paused"]
            if pz:  # paused: preempted by a reply worth more (waits for it, no deadline, as before), or waiting for
                # GPU memory another process holds (vram): a long, explicit deadline (QE_PAUSE_TIMEOUT_S), counted from
                # when the VRAM wait began, not from the last token (a preemption before it does not count, R6-L1)
                if not pi["slots"][j.slot].get("vram"):
                    j.t_vwait = None
                    return
                if getattr(j, "t_vwait", None) is None:
                    j.t_vwait = time.time()
                if time.time() - j.t_vwait < PAUSE_TIMEOUT_S:
                    return
                why = "no GPU memory for this reply's context: another process holds VRAM the KV cache gave up"
                print(f"engine: slot {j.slot} waited {time.time() - j.t_vwait:.0f} s for VRAM ({why}): the request ends with an error",
                      flush=True)
                j.fail(SequenceFault(f"the reply waited {time.time() - j.t_vwait:.0f} s for GPU memory: {why}; try again later"))
                return
        except Exception:
            pi = None
        print(f"engine error: no progress on slot {j.slot} ({len(j.ids)} prompt tokens, {getattr(j, 'n', 0)} generated): "
              f"{j.empty_n} steps without a token in {time.time() - t_tok:.0f} s; pool: {json.dumps(pi)[:2000]}", flush=True)
        j.fail(SequenceFault("engine error: no progress (no token for "
                             f"{j.empty_n} steps); see the server log"))

    def run_on_engine(self, fn):
        """Run fn() on the scheduler thread between requests and wait for it."""
        ev = threading.Event()
        with self.cv:
            self.ctrl.append((fn, ev))
            self.cv.notify()
        ev.wait()

    def _loop(self):
        e = self.engine
        active = []
        idle_persist = self.args.persist_idle
        idle_since = None  # start of the idle time before the idle disk persist
        last_snap = 0.0
        # prompts being prefilled chunk by chunk: at most two, the newest first; a short prompt
        # (little left to compute) may be started while a long one is in progress and runs to
        # completion between its chunks (see pick_next)
        prefills = []
        decode_until = 0.0  # while another prompt is prefilled, decode steps run until this time
        # --prefill-turns slices (DSV4's profile default, dsv4_app's policy): after a prompt chunk every decoding request
        # gets one slice of --slice-passes passes before the next chunk, instead of the --prefill-share time split
        turns = getattr(self.args, "prefill_turns", "time") == "slices"
        dec_left = 0
        bg_done = True  # idle archiving has nothing left (reset only when a request ends without a decode step; else the idle persist)
        # [ink-ctx-arch] Inkling, [gptoss-rearm] gpt-oss: re-armed after every finished reply too (their archive is host RAM
        # only; before, a reply that ended after decode steps was archived only on the next prompt's path: Inkling +1.6 s
        # TTFT per 109K rows on TP2)
        ink_rearm = bg_rearm_after_reply(self.profile, os.environ, self.args)
        burst_wait = max(0.0, getattr(self.args, "burst_wait", 5.0)) / 1000.0

        adm = Admission(self)
        # (shared prefill batches: short prompts next leave the decoders the floor share after every pass too, not only over
        # the window: arrivals during that short decode window join the next shared batch; back to back passes at 16 in
        # flight: TTFT p50 0.55 s vs 0.35 s)
        qa = QaSplit(adm.share_min, self.args.prefill_share, adm.share_min if adm.multi else 0.0)
        parked = []  # [q27-park-c] replies parked after their first token (engine handle in j.park_h), oldest first

        trace = open(env_str("QE_SCHED_TRACE"), "w", buffering=1) if env_str("QE_SCHED_TRACE") else None
        prof_last = 0.0  # QE_STEP_PROF: return time of the last single step (0: none to chain from)

        def tr(kind, t0, n=0, toks=0, j=None):  # scheduler trace (QE_SCHED_TRACE=file): one JSON line per event
            if trace:  # (j: the job, as its id; finish_start also its enqueue time, its slot, the cached rows)
                x = [kind, round(t0, 6), round(time.time() - t0, 6), n, toks]
                if j is not None:
                    x.append(id(j))
                    if kind == "finish_start":
                        x += [round(j.t_enq, 6), j.slot, getattr(j, "reused", 0)]
                trace.write(json.dumps(x) + "\n")
        last_mg = 0.0
        stuck_since = None  # since when requests have been queued with nothing running and nothing admitted
        lp = LoopProf() if LOOP_PROF else _NoProf()  # [srv-gptoss-gap]
        while True:
            lp.top(active)
            self.beat("scheduler loop")
            self.jobs_ref = (active, prefills)
            self.parked_ref = parked  # (fail_open: parked replies are open requests too)
            if active or prefills:
                stuck_since = None
            if self.retry_q:  # refused starts back to the queue (in order): a sequence or prefill ended since the
                # refusal (retry_gen), the backoff ran out, or nothing runs any more (the last try decides)
                tn = time.monotonic()
                idle = not active and not prefills
                with self.cv:
                    keep = []
                    for r in self.retry_q:
                        if self.stopping:  # (shutdown: never admitted again)
                            r[1].fail(SequenceFault("server is shutting down"))
                        elif idle or r[2] != self.retry_gen or r[0] <= tn:
                            self.queue.append(r[1])
                        else:
                            keep.append(r)
                    self.retry_q = keep
            if self.queue or self.retry_q:  # [srv-disconnect] a client gone while its request waits: out of the queue now
                self.drop_cancelled_queued()
            lp.mark("retry")
            if time.time() - last_mg > 1.0:  # host memory pressure: the archive gives memory back (engine)
                last_mg = time.time()
                try:
                    e.mem_guard()
                except Exception:
                    pass
            if not active or time.time() - last_snap > 0.5:  # monitoring: KV pool / archives
                try:
                    self.metrics.engine_snapshot(e.pool_info())
                except Exception:
                    pass
                last_snap = time.time()
            lp.mark("memsnap")
            ctrl = []
            with self.cv:
                while not self.queue and not active and not self.ctrl and not prefills and not parked:
                    self.beat("idle")
                    if self.retry_q:  # (released at the top of the loop when idle)
                        break
                    # idle: archive the idle conversations to host memory piece by piece (~64 MB
                    # per GPU per call, a few ms each), so that evicting them later for another
                    # prompt costs nothing on its path; then wait for work
                    if self.args.bg_archive and not bg_done:
                        self.cv.release()
                        try:
                            bg_done = e.archive_more(self.args.bg_archive << 20)
                        finally:
                            self.cv.acquire()
                        continue
                    # long idle: save the conversations to the disk cache (they then survive a
                    # restart / model swap even without a clean shutdown)
                    # (waited in slices of at most HANG_S / 2 with a beat in between: --persist-idle above QE_HANG_S
                    # must not look like a hang)
                    # an archiver stopped for host memory retries once a second here too, also while the disk
                    # persist waits for --persist-idle (qfn_archive_loss re-review L1)
                    if bg_done == 2 and time.time() - last_mg > 1.0:
                        last_mg = time.time()
                        bg_done = False
                        continue
                    if self.dirty and idle_persist > 0 and self.args.disk_cache:
                        if idle_since is None:
                            idle_since = time.monotonic()
                        rest = idle_persist - (time.monotonic() - idle_since)
                        if rest > 0:
                            w = min(rest, HANG_S / 2) if HANG_S > 0 else rest
                            self.cv.wait(min(w, 1.0))
                            # the host memory check once a second here too, as in the idle wait below (before, an idle
                            # server with --disk-cache skipped it until the persist: a hog got no pressure response for
                            # up to --persist-idle); [disk-eager] the engine also collects its finished writes there
                            # (Flash-Next dk_reap: a superseded file goes, the index and the disk budget follow)
                            if time.time() - last_mg > 1.0:
                                last_mg = time.time()
                                self.cv.release()
                                try:
                                    # (only the host-memory part, review M2: no idle reclaim / VRAM tick before the persist)
                                    mp = getattr(e, "mem_pressure", None)
                                    if mp is not None:
                                        mp()
                                except Exception:
                                    pass
                                finally:
                                    self.cv.acquire()
                                if bg_done == 2:  # (review M1) the archiver stopped for host memory retries, as below
                                    bg_done = False
                            continue
                        idle_since = None
                        if not self.queue and not self.ctrl:
                            self.ctrl.append((e.persist, None))
                            self.dirty = False
                    elif not self.cv.wait(1.0) and time.time() - last_mg > 1.0:
                        # idle: the host memory check still runs (another process may grow meanwhile)
                        last_mg = time.time()
                        self.cv.release()
                        try:
                            e.mem_guard()
                            tick = getattr(e, "idle_tick", None)  # [dsv4-rel8-giveback] (DSV4) only here: never while queued
                            if tick is not None and not self.queue and not self.ctrl:
                                tick()
                        except Exception:
                            pass
                        finally:
                            self.cv.acquire()
                        if bg_done == 2:  # the archiver stopped for host memory: try again (it says 2 again while short)
                            bg_done = False
                if self.queue or active or prefills or self.ctrl:
                    idle_since = None
                if not active and not prefills:
                    ctrl, self.ctrl = self.ctrl, []
            lp.mark("idlewait")
            for fn, ev in ctrl:
                try:
                    t0 = time.time()
                    # the disk persist (27B --disk-cache) can take minutes: its own, longer hang deadline
                    self.beat("persist" if fn == e.persist else f"ctrl {getattr(fn, '__name__', '?')}",
                              max(HANG_S, HANG_PERSIST_S) if fn == e.persist else None)
                    fn()
                    if fn == e.persist:
                        print(f"persisted slots to the disk cache in {time.time() - t0:.1f}s "
                              f"({e.num_disk_archives()} archives on disk)", flush=True)
                finally:
                    if ev:
                        ev.set()
            lp.mark("ctrl")
            resume_due = False  # parked replies wait for room a seat has: new prompts do not take it first (review M3)
            if parked:  # [q27-park-c] resume parked replies (oldest first) while a decode seat is free
                if self.stopping:
                    for j in parked:
                        e.park_drop(j.park_h)
                        j.slot = -1
                        j.fail(SequenceFault("server is shutting down"))
                    parked = []
                else:
                    for j in [j for j in parked if j.cancelled()]:  # client gone: no resume just to release (review L4)
                        e.park_drop(j.park_h)
                        parked.remove(j)
                        j.slot = -1
                        j.fail(Cancelled("cancelled while parked"))
                        self.cancel_note(j, "parked")
                    dec = sum(1 for x in active if not x.finished)
                    ps = e.pause_state() if hasattr(e, "pause_state") else None
                    paused_now = bool(ps and ps[1] > 0)  # a paused reply resumes first (as admit_next; review M4)
                    # (shared prefill batches keep up to --park-reserve prompts in flight: no 2-prompt gate there, the
                    # engine checks its own limit; nothing waiting for a seat: any free seat, qfn_park_review M4 / H1)
                    # (a resume prefilled in pieces holds its slot and region: its next piece comes first, whatever the
                    # seat conditions, else it and a paused reply could wait on each other; review kb M3)
                    while parked and (getattr(parked[0], "resuming", False) or (
                            (adm.multi or len(prefills) < 2) and not paused_now and
                            (dec < adm.seats - adm.reserve or not active or (not self.queue and dec + len(prefills) < adm.seats)))):
                        j = parked[0]
                        tu = time.time()
                        try:
                            r = e.unpark(j.park_h, self.args.mtp_k)
                        except Exception as ex:  # (review L3: one reply fails, the scheduler goes on)
                            parked.pop(0)
                            self.cur_slot = -1
                            e.park_drop(j.park_h)
                            j.slot = -1
                            j.fail(ex)
                            continue
                        if r is False:  # a resume prefilled in pieces: the next piece after the others' turn
                            j.resuming = True
                            self.cur_slot = -1  # (the engine bound the resuming slot)
                            resume_due = True
                            break
                        if r is None:
                            resume_due = bool(active or prefills)
                            if not active and not prefills:
                                time.sleep(0.02)  # (nothing runs that could free room: no hot spin)
                            break
                        parked.pop(0)
                        j.resuming = False
                        j.slot, extra = r
                        j.park_h, j.ds0 = None, self.dstats.read(j.slot)  # (draft counters are per slot: from here on)
                        j.empty_n, j.t_tok = 0, time.time()  # (the no-progress guard restarts: review L4)
                        self.cur_slot = -1
                        tr("unpark", tu, 1, 0, j)
                        active.append(j)
                        dec += 1
                        if extra is not None:  # (the held state did not serve: the engine prefilled the pending token)
                            j.feed([extra])
            while True:  # several prompts in one batch: every request that can join is admitted before the next batch
                with self.cv:
                    if not self.queue and not active and not prefills:
                        break
                    if burst_wait > 0 and self.queue and not active and not prefills:
                        # idle server, a request arrived: wait until no new one has come for burst_wait (at most 30 ms),
                        # so a burst shares the first prefill batch. Without it the first request of an 8-request burst
                        # was often prefilled alone and the rest queued behind it (1 + 4 + 3 batches, TTFT p90 2.2 vs 0.8 s)
                        t_cap = time.time() + 0.03
                        while True:
                            rest = min(max(j.t_enq for j in self.queue) + burst_wait, t_cap) - time.time()
                            if rest <= 0:
                                break
                            self.cv.wait(rest)
                    # admit the next request (shortest remaining prefill first, aged) while a
                    # sequence slot is free and the pool has room for it
                    admit = None
                    i = None if resume_due else adm.admit_next(time.time(), prefills)
                    if i is not None:
                        admit = self.queue.pop(i)
                        long_ = max(prefills, key=lambda j: len(j.ids) - max(0, e.prefill_pos(j.slot))) if prefills else None
                        if long_ is not None and not getattr(admit, "n_refused", 0) and \
                                (not adm.multi or len(long_.ids) - max(0, e.prefill_pos(long_.slot)) +
                                 admit.est > adm.join_cap()):  # a cut-in on the long prompt in progress (once per request)
                            if not any(j is not long_ for j in prefills):
                                long_.round_tok = 0  # a new cut-in round
                            long_.round_tok += admit.est
                            long_.cut_at = max(0, e.prefill_pos(long_.slot))
                            print(f"[sched] {admit.est} tokens to prefill cut in on {len(long_.ids)} at {long_.cut_at}",
                                  flush=True)
                if admit is None:
                    break
                self.dirty = True
                prof_last = 0.0  # QE_STEP_PROF: no gap across a prefill
                try:
                    tb = time.time()
                    self.beat("start")
                    self.starting = admit  # (the hang monitor fails it too: out of the queue, not in prefills yet)
                    admit.begin_start()
                    self.starting = None
                    # the engine is bound to the new prompt's slot now: the next single step must resume its own
                    # sequence first (without this, a step in a --prefill-share window decoded the slot of the prompt
                    # just admitted, and the running request streamed another conversation's tokens)
                    self.cur_slot = -1
                    tr("begin_start", tb, len(admit.ids), 0, admit)
                    prefills.append(admit)
                except Exception as ex:
                    # a prompt refused after the engine's checks of its input may have switched the engine's current
                    # slot: the next single-sequence step must resume its own slot first
                    if not (isinstance(ex, StartRefused) and ex.code != StartRefused.NO_ROOM):
                        self.cur_slot = -1
                    self.starting = None
                    # admitted (can_admit said yes) but the start found no room, e.g. a donor conversation that is
                    # generating cannot be shared in place: wait for the running sequences instead of failing at once.
                    # Only a refusal that can go away (no room, prompts in flight) and only while something runs (idle,
                    # nothing can change: the refusal is final); bad input / too long fail at once with their message
                    t_ref = getattr(admit, "t_refused", None) or time.monotonic()
                    # (a disk cache read ends by itself, also with nothing running)
                    retry = isinstance(ex, StartRefused) and ex.retry and (active or prefills or ex.code == StartRefused.DISK) and not self.stopping
                    if retry and time.monotonic() - t_ref < START_RETRY_S:
                        admit.t_refused = t_ref
                        admit.n_refused = getattr(admit, "n_refused", 0) + 1
                        admit.slot = -1
                        if admit.n_refused == 1:  # logged once per request
                            try:
                                pi = e.pool_info(convs=False)
                                busy = sum(1 for x in pi["slots"] if x["active"])
                                note = f"{busy} active slots, {pi['events']['evict']} evictions, {pi['paused']} paused"
                            except Exception:
                                note = "pool info unavailable"
                            print(f"[sched] start of {len(admit.ids)} tokens refused after admission ({ex}; {note}): "
                                  f"requeued until a running sequence finishes (at most {START_RETRY_S:.0f} s)", flush=True)
                        # back when a sequence or prefill ends (retry_gen), else after a backoff of 0.5 .. 4 s
                        back = min(4.0, 0.5 * 2 ** (admit.n_refused - 1))
                        self.retry_q.append((time.monotonic() + back, admit, self.retry_gen))
                    else:
                        if getattr(admit, "n_refused", 0):
                            print(f"[sched] start of {len(admit.ids)} tokens failed after {admit.n_refused} refusals in "
                                  f"{time.monotonic() - t_ref:.1f} s: {ex}", flush=True)
                        admit.fail(ex)
                if not adm.multi:
                    break
            lp.mark("admit")
            if self.stopping and prefills:  # shutdown: prompts being prefilled end now (stop_requests)
                for j in prefills:
                    j.fail(SequenceFault("server is shutting down"))
                prefills = []
            if prefills:
                # [srv-disconnect] a client that left while its prompt is prefilled: the prompt stops here, between two
                # engine calls (27B / Flash-Next / gpt-oss: one chunk; DSV4: a chunk too, its engine reads the cancel
                # flag inside a call; GLM [glm-rel-pfgate]: its prefill returns inside a prompt, qe_prefill_cancel stops
                # the window job, the engine keeps the whole windows done). The release drops the prompt in flight
                # (every engine's qe_release handles a prompt abandoned before start_finish)
                for j in [j for j in prefills if j.cancelled()]:
                    pos = e.prefill_pos(j.slot)
                    pc = getattr(e, "prefill_cancel", None)
                    if pc is not None:
                        pc(j.slot)
                    self.cur_slot = -1  # (the engine may be bound to this slot: the next single step resumes its own)
                    j.fail(Cancelled("cancelled while its prompt was prefilled"))
                    prefills.remove(j)
                    self.retry_gen += 1  # (a slot freed: refused starts try again)
                    self.cancel_note(j, f"its prompt was prefilled ({max(0, pos)}/{len(j.ids)} rows done)")
            if not self.queue and not active and not prefills:
                continue
            # no step queued ahead of a prefill chunk, nor next to a logprobs reply's own steps (it would be undone each time)
            e.set_lookahead(not prefills and not any(j.lp is not None for j in active))
            if prefills:
                lp.ok = False
                # prompt chunks interleaved with decode steps of the running sequences: small
                # chunks while others decode (they keep generating), whole ones otherwise
                cur = prefills[-1]
                busy = any(not j.finished for j in active)
                # always chunked (the engine prefills in 2048-token chunks anyway): between the
                # chunks the queue is checked, so a short request can cut in even when nothing
                # decodes; the decoders' share of the time only applies while they run
                budget = self.args.prefill_chunk if self.args.prefill_chunk > 0 else 1 << 30
                # adaptive chunk: no other slot decoding or waiting -> the larger idle chunk (Flash-Next: its
                # 4096-row batch); decided per chunk, so a request that arrives meanwhile gets 2048 again
                budget = self.idle_budget(budget, busy, prefills, adm)
                share = self.args.prefill_share if busy and self.args.prefill_chunk > 0 else 0.0
                try:
                    finished = []
                    if busy and (dec_left > 0 if turns else time.time() < decode_until):  # the decoders' share
                        dec_left -= 1  # (slices: one engine decode call per loop below)
                    else:
                        tc = time.time()
                        ready = None
                        self.beat("prefill", self.call_limit("prefill"))  # (DSV4: a whole prompt is one call)
                        if adm.multi and len(prefills) >= 2:  # one batch over the prompts in flight
                            order = adm.batch_order(tc, prefills, budget)
                            slots = [j.slot for j in order]
                            # (row caps only with --fill-passes: the long prompt leads capped to its share)
                            ready = (e.prefill_multi(slots, budget, caps=[adm.caps.get(id(j), 0) for j in order])
                                     if adm.caps else e.prefill_multi(slots, budget))
                            if ready is not None:
                                self.cur_slot = -1  # a piece prefilled alone binds its own slot: the next step resumes
                                tr("prefill_multi", tc, len(order))
                                finished = [j for j in order if j.slot in ready]
                                self.sched["multi_batches"] = self.sched.get("multi_batches", 0) + 1
                            else:
                                cur = order[0]  # no shared batch (layout, image rows ...): the first one alone
                        if ready is None:
                            p0 = max(0, e.prefill_pos(cur.slot)) if trace else 0
                            done = e.prefill_more(budget, cur.slot)
                            self.count_refills()  # (DSV4: a banned first token's redo whose trim failed)
                            tr("prefill", tc, len(cur.ids),
                               ((len(cur.ids) if done else max(0, e.prefill_pos(cur.slot))) - p0) if trace else 0, cur)
                            self.cur_slot = cur.slot
                            if done:
                                finished = [cur]
                        adm.account(prefills, budget)  # (a no-op unless batch_order set a long prompt's share)
                        self.pf_progress(order if ready is not None else [cur], prefills)  # /api/stats: rows done so far
                        tc = time.time() - tc
                        if adm.qa and share > 0:  # queue-aware split (busy, chunked)
                            qa.add(0, tc)
                            decode_until = time.time() + qa.window(tc, adm, [j for j in prefills if j not in finished],
                                                                   self.queue, active)
                        else:
                            decode_until = time.time() + (tc * share / (1 - share) if 0 < share < 1 else 0)
                        if turns and share > 0:  # (busy, chunked)
                            dec_left = max(1, int(getattr(self.args, "slice_passes", 16))) * sum(1 for j in active if not j.finished)
                    if finished:
                        self.retry_gen += 1  # a prefill ended (in-flight limit, donor done)
                    for cur in finished:
                        tf = time.time()
                        try:
                            cur.finish_start()
                            tr("finish_start", tf, len(cur.ids), 0, cur)
                            # [q27-park-c] beyond the decode seats while others wait for a prompt seat: park after the
                            # first token (the waiting moves behind it, as the reply would wait for the seat anyway)
                            # ([qfn-ttft-park] only beyond the seats: while every open request fits one, nobody waits
                            # for a seat and a park only takes decode time from the others: Flash-Next 16 agents on 16
                            # seats with reserve 4, out -14 %)
                            ndec = sum(1 for x in active if not x.finished)
                            if adm.park and cur.parkable() and ndec >= adm.seats - adm.reserve and \
                                    (self.queue or parked or any(x is not cur for x in prefills)) and \
                                    ndec + len(prefills) + len(self.queue) + len(parked) > adm.seats:
                                tpk = time.time()
                                if self.queue and hasattr(e, "protect_ids"):  # [ink-archprot] review2 L4: the queue head's source stays
                                    try:  # (review3 nit: never a failed start of cur)
                                        q0 = self.queue[0]
                                        e.protect_ids(q0.ids, q0)
                                    except Exception:
                                        pass
                                h = e.park(cur.slot)
                                if h is not None:
                                    cur.park_h, cur.t_park = h, time.time()
                                    self.cur_slot = -1
                                    tr("park", tpk, 1, 0, cur)
                                    parked.append(cur)
                                    prefills.remove(cur)
                                    self.retry_gen += 1  # (a slot freed: refused starts try again)
                                    continue
                            active.append(cur)
                        except Exception as ex:
                            # the engine may be bound to this slot now (start_finish binds it before it fails or
                            # before a later call raises): the next single step must resume its own sequence
                            self.cur_slot = -1
                            cur.fail(ex)
                        prefills.remove(cur)
                except Exception as ex:
                    self.retry_gen += 1
                    self.cur_slot = -1  # (as above: the engine's bound slot is unknown after a failed prefill)
                    for j in list(prefills) if adm.multi and len(prefills) >= 2 else [cur]:
                        j.fail(ex)
                        prefills.remove(j)
            n_active = len(active)
            active = [j for j in active if not j.check_done()]
            if len(active) < n_active:
                bg_done = False  # a conversation went idle: worth archiving
                self.retry_gen += 1  # room may have freed: refused starts try again
            if not active:
                prof_last = 0.0
                if self.queue and not prefills:  # nothing running and nothing admitted
                    # nothing will free room: the pool is smaller than the request (shrunk under VRAM pressure; it grows back
                    # if the memory returns, every can_admit tries). A bounded wait, then a clear error (review M4)
                    now = time.time()
                    stuck_since = stuck_since or now
                    with self.cv:
                        # (the last can_admit: a request that fits was only not tried yet, e.g. behind a refused start)
                        for j in [j for j in self.queue if NO_ROOM_S > 0 and now - max(stuck_since, j.t_enq) >= NO_ROOM_S
                                  and not e.can_admit(len(j.ids) + adm.k + 2)]:
                            self.queue.remove(j)
                            msg = self.no_room_msg(len(j.ids), adm.k, now - max(stuck_since, j.t_enq))
                            print(f"[sched] ERROR, the request ends, the server goes on: {msg}", flush=True)
                            j.fail(SequenceFault(msg))
                        self.cv.wait(0.05)
                continue
            lp.mark("lookahead")
            if any(j.cancelled() and not j.finished for j in active):
                # [srv-disconnect] a reply whose client left ends before the next step (review M3: on DSV4 that step
                # could be a turn switch or a whole re-prefill for nobody)
                active = [j for j in active if not (j.cancelled() and j.check_done())]
                self.retry_gen += 1
                bg_done = False
            t_dec = time.time()
            try:
                for j in active:
                    if j.finished:
                        continue
                    stop = bool(j.stops and STOP_ROLLBACK) and j.stop_risk()
                    near = j.tc is not None and j.tc.near()  # a forced close may come: nothing decoded ahead of it
                    if (j.stops and STOP_ROLLBACK) or j.tc is not None:
                        e.set_no_ahead(j.slot, stop or near)
                lp.mark("noahead")
                others = active
                if any(j.lp is not None for j in active):  # logprobs replies step alone, the others as usual
                    others = [j for j in active if j.lp is None]
                    got = 0
                    for j in active:
                        if j.lp is not None and not j.finished:
                            got += self.step_logprobs(j, active, tr)
                    others = [j for j in others if not j.finished]
                    if not others and not got and not prefills:  # paused / waiting: no hot spin (N-L7)
                        time.sleep(0.02)
                if not others:
                    pass
                elif len(others) == 1:
                    j = others[0]
                    if self.cur_slot != j.slot:
                        e.resume(j.slot)
                        self.cur_slot = j.slot
                    ts = time.time()
                    self.beat("step", self.call_limit("step", (j.slot,)))  # (DSV4: a switch may re-prefill)
                    if STEP_PROF:
                        p0 = _pc()
                    lp.mark("pre")
                    toks = e.step()
                    lp.mark("call")
                    lp.n = (len(toks), 1)
                    if not toks:  # no token (paused, refused by the engine's slot check ...): resume explicitly next time
                        self.cur_slot = -1
                    if STEP_PROF:
                        p1 = _pc()
                    tr("step", ts, 1, len(toks))
                    self.metrics.step(others, [len(toks)], time.time() - ts)
                    self.count_refills()
                    if STEP_PROF:
                        p2 = _pc()
                    lp.mark("metrics")
                    if self.faults(others) or j.finished:
                        continue
                    j.feed(toks)
                    self.no_progress(j, len(toks))
                    lp.mark("feed")
                    if not toks and not prefills:  # paused / waiting for room: no hot spin (N-L7; never while a prompt prefills, R4-L2)
                        time.sleep(0.02)
                    if STEP_PROF:
                        p3 = _pc()
                        gap = p0 - prof_last if prof_last else 0.0  # previous step's return -> this call
                        j.prof.append((p1 - p0, len(toks), p2 - p1, p3 - p2, gap, j.prof_detok, j.prof_emit))
                        j.prof_detok = j.prof_emit = 0.0
                        prof_last = p1
                else:
                    self.cur_slot = -1  # batch steps undo the queued-ahead single step
                    ts = time.time()
                    self.beat("batch step", self.call_limit("step", [j.slot for j in others]))
                    lp.mark("pre")
                    res = e.step_batch([j.slot for j in others])
                    lp.mark("call")
                    lp.n = (sum(len(r) for r in res), len(others))
                    tr("batch", ts, len(others), sum(len(r) for r in res))
                    self.metrics.step(others, [len(r) for r in res], time.time() - ts)
                    self.count_refills()
                    lp.mark("metrics")
                    tf = time.time()
                    self.faults(others)
                    for j, toks in zip(others, res):
                        if not j.finished:
                            j.feed(toks)
                            self.no_progress(j, len(toks))
                    lp.mark("feed")
                    tr("feed", tf, len(others))
                    if not any(res) and not prefills:  # every slot paused / waiting: no hot spin (N-L7; not while a prompt prefills, R4-L2)
                        time.sleep(0.02)
            except Exception as ex:
                for j in active:
                    j.fail(ex)
                active = []
            if adm.qa:
                qa.add(1, time.time() - t_dec)
            n_active = len(active)
            active = [j for j in active if not j.check_done()]
            lp.mark("done")
            if len(active) < n_active:
                self.retry_gen += 1  # room may have freed: refused starts try again
                if ink_rearm:  # [ink-ctx-arch] / [gptoss-rearm] a finished reply goes to host RAM while the server idles (no disk files)
                    bg_done = False


def repetition_period(ids, max_period=200, min_span=600, min_reps=5):
    """Period p of a degenerate loop at the end of ids (the last max(min_span, min_reps * p)
    tokens repeat with period p), else 0. Real text (even lists, tables, code) does not repeat
    exactly that long."""
    n = len(ids)
    for p in range(1, max_period + 1):
        span = max(min_span, min_reps * p)
        if span + p > n:
            break
        if ids[n - span:] == ids[n - span - p:n - p]:
            return p
    return 0


STOP_ROLLBACK = env_int("QE_STOP_ROLLBACK", 2)
QA_SHARE_MIN = 0.10  # --prefill-share-min default for the 27B (the decoders' floor share; 0.15 measured slower at 16 in flight)
QA_SHARE_MIN_MULTI = 0.10  # Flash-Next (shared prefill batches): fixed 0.10 / 0.15 equal at 4 / 16 in flight (0.10 mean 0.43 vs 0.50 s), 0.25 slower
PARK_RESERVE = 2  # --park-reserve default: seats kept for prompts while replies wait parked ([q27-park-c])
PARK_RESERVE_MULTI = 4  # the same with shared prefill batches (Flash-Next, [qfn-ttft-park]: c32 / c64 p95 0.87 / 0.80 -> 0.64 / 0.65 s)
QA_TAU = 2.0  # s: the window of the decoders' floor share (queue-aware split, --prefill-share-min)
MULTI_MAX = 8  # prompts in flight that share prefill batches (the engine's FLASH_MAX_PREFILLS default)
PAUSE_WAIT = 3.0  # --pause-wait default (s): how long a request that does not fit waits before it pauses others
LONG_SHARE = 0.5  # --long-share default where passes are copy-bound (the knee for short appends)
PASS_BOUND_CHUNK = 4096  # the prefill chunk while others decode on pass-bound servers (measured)


class Job:
    """One generation request driven by the scheduler thread."""

    text_mk = ""  # the reply with the real tag tokens marked (server/content_markers.py); text_all: the text itself
    eos_end = False  # the model ended the reply itself
    stop_sent = 0  # stop sequences: the length of text_all that went to the client (stop_scan)
    answer_at = 0  # stop sequences match from here; None: a guard-opened thinking turn whose answer has not begun (stop_from)
    answer_max = None  # a guard-opened thinking turn: the client's max_tokens, for the answer alone
    tool_banned = False  # [tool-choice-ban] tool_choice none through the engine's static ban (begin_start)
    tool_hold = False  # tool_choice required / named: client stops wait until the forced call is complete (stop_from)
    cut_think = (False, True)  # tool_choice none, no grammar tier, no ThinkControl: (inside <think>, no answer yet)
    lp = None  # OpenAI logprobs: top_logprobs of each token, None: off (no engine or token work at all)
    lp_row = 0  # logprobs: the row of the token being fed

    def n_out(self):
        """Generated tokens as usage reports them: the reply's tokens plus the end token the model generated (EOS /
        <|im_end|>, as vLLM counts it; DECISIONS 11-14). Not when the reply ended by length, a stop string, a tool call
        cut or the server's own stops: no end token was generated; structured output that ends when its grammar
        completes counts none either (vLLM would sample one more step). self.n stays the reply's text tokens
        (max_tokens, trims, context checks)."""
        return self.n + (1 if self.eos_end and not self.cut else 0)

    def __init__(self, gen, ids, p, on_text, cancelled, affinity):
        self.gen, self.ids, self.p, self.on_text, self.cancelled = gen, ids, p, on_text, cancelled
        self.affinity = affinity
        self.lp = p.get("logprobs")
        self.grammar, self.grammar_s, self.gsteps = None, 0.0, 0  # structured output (begin_start)
        self.glk = None  # TokenLookup of a constrained reply (grammar.py)
        self.done = threading.Event()
        self.error, self.stats, self.slot = None, None, -1
        self.penalty, self.guarded = None, 0  # (presence, frequency) in force; loop guard: repeated turns before
        self.finished = False
        self.t_enq, self.est, self.t_est = time.time(), None, 0.0  # scheduler: prefill-cost estimate
        self.cut_at = None  # prefill position at the last cut-in by a short request (fairness)
        self.round_tok = 0  # prompt tokens of the cut-in round in progress on this (long) prompt
        # the main-thread label (server/agents.py): logs, /api/stats and the CACHE_PARTIAL check only. Neither the
        # scheduler nor the engines order by it (no main-thread priority, user decision 2026-11-12)
        info = getattr(ids, "agent", None)
        self.prio = 2 if (gen.agents.is_main_now(info) if info is not None
                          else getattr(ids, "label", "") == "main agent") else 1
        self.prof, self.prof_detok, self.prof_emit = [], 0.0, 0.0  # QE_STEP_PROF

    def begin_start(self):
        g, e, p = self.gen, self.gen.engine, self.p
        # values the client does not send (Claude Code sends none): the model card's for this reply's mode, or
        # --temperature / --top-p / ... Sampling at temperature 1 without them (top_p 1, top_k 40) lets tail tokens
        # break tool-call syntax.
        d = g.sampling_defaults[bool(p.get("thinking", True))]
        temp = p.get("temperature")
        temp = d["temperature"] if temp is None else temp
        # an explicit 0 is a value, only a missing one takes the default; top_k 0 = no top-k cut (the engine's candidate
        # list: 27B / Flash-Next / DSV4 64, the others the full vocabulary); top_p <= 0 = only the top token: greedy
        # (every engine reads top_p <= 0 as "no top-p cut", so it becomes temperature 0 here)
        top_p, top_k, min_p = (d[k] if p.get(k) is None else p[k] for k in ("top_p", "top_k", "min_p"))
        if float(top_p) <= 0:
            temp = 0
        self.max_new = p.get("max_tokens") or (e.max_t - len(self.ids) - 16)
        self.stops = [s for s in (p.get("stop") or []) if s]
        self.t0 = time.time()
        g.metrics.prefill_start(self)
        e.set_steering(*p.get("steering_sel", STEER_DEFAULT))  # this request's steering (keys the cached prefixes)
        self.slot = e.start_begin(self.ids, g.args.mtp_k, self.affinity, g.system_end(self.ids),
                                  getattr(self.ids, "mm", None))
        flag = getattr(self.cancelled, "flag", None)
        if flag is not None and hasattr(e, "set_abort_flag"):  # [srv-disconnect] DSV4: the cancel stops a prefill inside its call
            e.set_abort_flag(self.slot, flag)
        if p.get("cache_prompt") is False and hasattr(e, "set_no_cache"):  # [shared-srv] DSV4: no reuse for this request
            e.set_no_cache(self.slot)
        if hasattr(e, "set_trim"):  # DSV4 [dsv4-srv-stop]: no stop trim in thinking mode without tools (the next turn drops
            e.set_trim(self.slot, not (p.get("thinking") and not p.get("has_tools")))  # the reasoning: the snapshot serves it)
        self.pf0 = max(e.prefill_pos(self.slot), 0)  # rows cached at the start (live prefill progress: Generator.pf_progress)
        # after start_begin: the engine applies sampling parameters to its current slot, which
        # is this request's only now (before, they landed on whichever sequence was current)
        # presence / frequency penalty: the request's, else the server default for sampled requests (greedy ones
        # stay greedy: deterministic checks); the loop guard raises presence and seeds it with the repeated reply
        pres, freq = p.get("presence_penalty"), p.get("frequency_penalty")
        explicit = pres is not None or freq is not None
        pres = float(pres if pres is not None else (d["presence_penalty"] if temp > 0 else 0.0))
        freq = float(freq if freq is not None else (getattr(g.args, "frequency_penalty", 0.0) if temp > 0 else 0.0))
        guard = getattr(self.ids, "guard", None)
        seed = []
        if guard is not None:
            pres = max(pres, guard["presence"])
            seed = loop_guard.last_reply_tokens(self.ids, g.im_start, g.asst_ids, g.im_end, g.special_min, g.ws_ids)
        can_pen = not getattr(g.profile, "is_gptoss", False) or "qe_set_penalty" in getattr(e, "fns", ())
        if temp <= 0 and (pres or freq) and (explicit or guard is not None) and can_pen:
            # greedy with a penalty: the penalised argmax ([gptoss-rel1] not on an engine without penalties: the
            # top_k = 1 path would only make the reply slower, filtered sampling takes the full head)
            temp, top_p, top_k, min_p = 1.0, 1.0, 1, 0.0
        # an explicit seed is kept, 0 included (`or` replaced seed 0 with a time seed); the GLM engine takes 0 for "a
        # random seed": an explicit 0 goes there as another fixed value, so seed 0 repeats there too
        seed_v = p.get("seed")
        seed_v = int(time.time() * 1e6) if seed_v is None else int(seed_v)
        if seed_v == 0 and getattr(g.profile, "is_glm", False):
            seed_v = 0x9E3779B97F4A7C15
        e.set_sampling(temp, top_p, top_k, min_p, seed_v)
        self.penalty = (pres, freq) if temp > 0 and (pres or freq) else None
        if self.penalty and not e.set_penalty(pres, freq, seed):
            self.penalty = None
        if guard is not None:
            e.set_no_lookup(self.slot, True)
            self.guarded = guard["turns"]
            g.guard_note(dict(guard, seed=len(seed)))
        e.set_prio(self.slot, self.prio)
        self.grammar = self.p.get("grammar")
        tcs = self.p.get("tool_choice")
        self.tool_banned = False
        if self.grammar is None and tcs and not tcs.get("cut") and g.tool_tier is not None:
            if tcs["mode"] == tool_choice.NONE and g.tool_ban and self.lp is None:
                # tool_choice none: the engine never picks <tool_call> for this reply (static ban, set per start after
                # start_begin, which clears it); the slot stays batched, runs ahead and keeps its drafts
                e.ban_tokens(self.slot, g.tool_tier.none_ids())
                self.tool_banned = True
            else:
                # tool_choice none (with logprobs, or no engine ban) / required / named (server/tool_choice.py): a fresh
                # automaton per start, through the grammar path; it knows from the prompt whether the reply opens
                # inside <think>
                opens = g.think_ids.ok and g.think_ids.open_at_end(self.ids, g.im_start)
                self.grammar = Constraint(g.tool_tier.matcher(tcs, opens), sorted(g.eos))
        self.grammar_s = 0.0
        if self.grammar is not None:  # constrained from the first token on (prompt's last chunk)
            tg = time.perf_counter()
            self.grammar.fill(e.mask_buffer(self.slot)[0])
            e.constrain(self.slot, True)
            if self.lp is None and hasattr(e, "set_check"):   # [dsv4-rel8-optgram] DSV4: constrained DSpark passes
                e.set_check(self.slot, self.grammar.legal_prefix)
            self.grammar_s += time.perf_counter() - tg
        if self.lp is not None:  # the raw logits per token: a constrained slot with every token allowed (engine.capture_logits)
            if self.grammar is None:
                e.mask_buffer(self.slot).fill(0xFFFFFFFF)
                e.constrain(self.slot, True)
            e.capture_logits(self.slot)
            self.lp_mode = "reasoning" if p["thinking"] else "content"  # (as Parser)
            self.lp_lead, self.lp_fresh, self.lp_held, self.lp_out = False, True, [], []

    def lp_token(self, t):
        """The entry of a reply token that goes to choices[].logprobs.content, else None: OpenAI has no field for reasoning
        or tool-call tokens. By token ids, as Parser by the marked tags: <think> .. </think> and the newlines after it, a
        real <tool_call> and all that follows."""
        g = self.gen
        ti = g.think_ids
        if getattr(ti, "harmony", False):  # [srv-gptoss-think] entries for the final channel's body tokens only
            if getattr(self, "lp_hs", None) is None:
                self.lp_hs = ti.tracker(self.ids)
            self.lp_hs.feed([t])
            if not self.lp_hs.last_final:
                return None
            x = g.engine.logits_row(self.slot, self.lp_row)
            return logprob_entry(g.tok, x, t, self.lp) if x is not None else None
        if self.lp_mode == "tool":
            return None
        if self.lp_mode == "reasoning":
            if t == ti.end:
                self.lp_mode, self.lp_lead, self.lp_fresh = "content", True, True
            return None
        if self.lp_fresh and t == ti.start:
            self.lp_mode = "reasoning"
            return None
        if t == g.tok.token_to_id("<tool_call>"):
            self.lp_mode = "tool"
            return None
        text = g.tok.decode([t], skip_special_tokens=False)
        if self.lp_lead and not text.strip("\n"):
            return None
        self.lp_lead = False
        if text.strip():
            self.lp_fresh = False
        x = g.engine.logits_row(self.slot, self.lp_row)
        if x is None:
            print(f"WARNING logprobs: no row {self.lp_row} for token {t}: no entry", flush=True)
            return None
        return logprob_entry(g.tok, x, t, self.lp)

    def lp_attach(self, s, entry):
        """Entries go with the text that reached the client: an entry whose token ends past a stop's cut, or in text held
        back for a stop that may still come, waits (and past the cut never goes)."""
        end = len(self.text_all)
        if entry is not None:
            self.lp_held.append((entry, end))
        sent = self.stop_sent if self.stops else end
        while self.lp_held and self.lp_held[0][1] <= sent:
            self.lp_out.append(self.lp_held.pop(0)[0])
        if s and self.lp_out:
            s = LpText(s)
            s.lp, self.lp_out = self.lp_out, []
        return s

    def finish_start(self):
        """The prompt is prefilled: first token, then this job decodes with the others."""
        g, e = self.gen, self.gen.engine
        first, self.reused, self.slot = e.start_finish(self.slot)
        self.ds0 = g.dstats.read(self.slot)  # draft counters of the slot (engine_stats: the per-request stats line)
        # max_tokens in the engine: an MTP step accepting drafts past it would leave tokens in the
        # slot that the reply does not have, and the next turn would miss the live end (resume
        # from the prompt-end checkpoint and recompute the whole reply)
        guard = getattr(self.ids, "guard", None)
        if guard and guard.get("think") and self.p.get("max_tokens") and g.think_ids.ok:
            # the client sized max_tokens for an answer and never asked for reasoning: those tokens come on top (as far
            # as the context has room: ThinkControl plans its close from max_new, not from the slot, so the limit must
            # not pass the slot end, or the reply ends there inside <think>)
            self.answer_max = self.max_new
            room = e.slots[self.slot] - len(self.ids) - 8
            top = self.max_new + self.p["think_budget"] + g.think_ids.wrap_max
            self.max_new = min(top, room) if room > 0 else self.max_new
        e.set_limit(self.slot, self.max_new - 1)
        if self.p.get("ignore_eos") and hasattr(e, "set_ignore_eos"):  # review r25-M2 (GLM: no EOS cut in the steps)
            e.set_ignore_eos(self.slot, True)
        # stop sequences: the reply is cut where its stop starts; the engine undoes the accepted
        # tokens past it (trim) when no step ran since, so the next turn resumes from the cut reply's
        # end. A step is queued ahead only while no stop can start before the step in flight's
        # last token (stop_risk; QE_STOP_ROLLBACK=1: never ahead, 0: off)
        self.keep = None
        if self.stops and STOP_ROLLBACK:
            e.watch_stops(self.slot)
        self.copied = e.last_copied()
        self.src = "archive" if e.last_from_archive() else ("other slot" if self.copied else "")
        n = len(self.ids)
        conv = getattr(getattr(self.ids, "agent", None), "conv", None)
        later = conv is not None and conv.turns >= 2  # a later turn of a known conversation (not a cold start)
        if n >= 16384 and 2 * self.reused < n and (self.src == "archive" or (self.prio >= 2 and later)):
            # a long prompt that came back mostly uncached: a restore that found only an early checkpoint, or
            # the main thread's context lost from the archive (an eviction it should have survived)
            diag("CACHE_PARTIAL", f"slot {self.slot} prompt {n} cached {self.reused} ({100 * self.reused // n} %, "
                 f"src {self.src or 'none'}, {'main thread' if self.prio >= 2 else 'prio %d' % self.prio})")
        g.cur_slot = self.slot
        self.t1 = time.time()
        g.metrics.prefill_done(self)
        if not getattr(self.ids, "mm", None):  # off the scheduler thread: long prompts take ms
            def check():
                try:
                    g.check_reuse(self)
                except Exception as ex:
                    print(f"check_reuse: {ex!r}", flush=True)
            threading.Thread(target=check, daemon=True).start()
        self.det = Detok(g.tok, g.marks.out if getattr(g, "marks", None) is not None else None)  # (tags marked)
        self.n, self.finish, self.text_all, self.cut = 0, "length", "", False
        self.text_mk = ""
        self.loop_checked = 0
        self.tc = None  # thinking state / forced close (server/think_budget.py)
        if g.think_ids.ok and g.think_ids.open_at_end(self.ids, g.im_start):
            self.tc = ThinkControl(self, self.p.get("think_budget"), g.think_ids, g.can_force,
                                   self.p.get("think_budget_src"))
        if guard and guard.get("think") and self.tc is not None:  # (also without a client max_tokens)
            self.answer_at = None
        elif getattr(getattr(g, "profile", None), "is_gptoss", False):  # harmony: client stops match the final channel only (stop_from)
            self.answer_at = None
        elif getattr(getattr(g, "profile", None), "is_inkling", False):  # Inkling: the content messages only (stop_from)
            self.answer_at = None
        elif self.tc is not None and getattr(getattr(g, "profile", None), "is_dsv4", False):
            # DSV4 (dsv4_app's [dsv4-srv-stop] rule): client stops match the answer only, never the reasoning
            self.answer_at = None
        self.cut_think = (False, g.think_ids.ok)  # tool_choice none without the grammar tier and without ThinkControl
        # tool_choice required / named (review L2): a client stop ("\n\n", "\n") must not cut the thinking or the forced
        # call header (the reply would end with no call); stops match from the end of the header on (stop_from)
        # (DSV4: never; its calls are text, dsml_spans keeps stops out of the block, review rel8 L8)
        self.tool_hold = bool(self.stops) and isinstance(getattr(self.grammar, "m", None), tool_choice.ToolMatcher) \
            and (self.p.get("tool_choice") or {}).get("mode") == tool_choice.REQUIRED and TOOL_FORMAT != "dsml"
        self.feed([first])

    def feed(self, pending):
        if self.finished:
            return
        g = self.gen
        stop_now = False
        pending = pending[:max(0, self.max_new - self.n)]  # a step may overshoot max_tokens
        gdone = False
        if self.grammar is not None and pending:  # advance the grammar, then the next step's mask
            tg = time.perf_counter()
            c, ok = self.grammar, []
            for t in pending:
                if t in g.eos:  # only allowed by the mask once the grammar accepts
                    ok.append(t)
                    break
                if not c.consume(t):
                    print(f"structured output: grammar rejected token {t}: {c.error}", flush=True)
                    gdone = True
                    break
                ok.append(t)
                if c.done():  # complete: end the turn without another step
                    gdone = True
                    break
                if getattr(c, "released", None) is not None and c.released():
                    # (optgram review L3) the forced header is complete: the step's later tokens are free text, never
                    # run through the matcher (a mixed trie node would refuse them)
                    ok.extend(pending[len(ok):])
                    break
            pending = ok
            self.gsteps += 1
            if self.glk is None:  # token history for lookup drafts (prompt + emitted so far)
                self.glk = TokenLookup(list(self.ids) + list(self.det.ids), g.image_pad)
            self.glk.append(ok)
            if not gdone and getattr(c, "released", None) is not None and c.released():
                # tool_choice: the forced call header is complete, the arguments decode freely (as after a forced
                # close of <think>: unconstrained, or with logprobs constrained with every token allowed)
                self.grammar_s += time.perf_counter() - tg
                self.release_grammar()
            elif not gdone:
                # drafts for the next step: forced tokens / lookup / the grammar-valid model drafts
                e = g.engine
                c.plan(e.spec_io(self.slot), e.mask_buffer(self.slot), g.args.mtp_k, self.glk)
            if self.grammar is not None:
                self.grammar_s += time.perf_counter() - tg
        cut_call = (self.p.get("tool_choice") or {}).get("cut") and not self.p.get("ignore_eos")
        # (a block the model re-opens: as ThinkControl.track; without one, a reply that starts outside <think>, as Parser)
        think_open, think_fresh = ((self.tc.open, self.tc.fresh) if self.tc is not None else self.cut_think) \
            if cut_call else (False, False)
        for i, t in enumerate(pending):
            if cut_call:  # tool_choice none without the grammar tier: the reply ends at a call in the answer
                if think_open:
                    think_open = t != g.think_ids.end
                elif think_fresh and t == g.think_ids.start:  # re-opened <think> (review L8): a call in it stays reasoning
                    think_open = True
                elif t == g.tool_tier_call:
                    print(f"tool_choice none: the reply ends at the tool call the model began after {self.n + i} "
                          f"tokens (no grammar tier to mask it)", flush=True)
                    self.finish, stop_now = "stop", True
                    pending = pending[:i]
                    break
                elif think_fresh and not g.think_ids.is_space(t):
                    think_fresh = False
            if cut_call and self.tc is None:
                self.cut_think = (think_open, think_fresh)
            if t in g.eos and not self.p.get("ignore_eos"):
                self.finish, stop_now = "stop", True
                self.eos_end = True
                pending = pending[:i]
                break
            if t == g.im_start and not self.p.get("ignore_eos"):
                # <|im_start|> is never valid inside a reply: the model skipped <|im_end|> and began a
                # turn of its own (seen at 130K+ contexts). End the turn here: the client gets the reply
                # so far, cleanly closed, instead of a hallucinated user / tool turn. (Markers quoted in
                # the prompt's content are plain text since issue #7, so the model does not copy this
                # token from there.) The log line shows the turn's first and last 16 tokens.
                turn = list(self.det.ids) + list(pending[:i])
                dec = lambda x: g.tok.decode(x, skip_special_tokens=False)  # noqa: E731
                print(f"stopping at a generated <|im_start|> after {self.n + i} tokens (prompt {len(self.ids)}): "
                      f"turn starts {dec(turn[:16])!r}, ends {dec(turn[-16:]) if len(turn) > 16 else ''!r}",
                      flush=True)
                diag("IM_START_STOP", f"slot {self.slot} prompt {len(self.ids)} after {self.n + i} tokens: "
                     f"head {dec(turn[:16])!r} tail {dec(turn[-16:])!r}")
                self.finish, stop_now = "stop", True
                pending = pending[:i]
                break
        self.n += len(pending)
        entry = self.lp_token(pending[0]) if self.lp is not None and pending else None
        if self.lp is not None and len(pending) > 1 and getattr(self, "lp_hs", None) is not None:
            self.lp_hs.feed(pending[1:])  # (harmony: the channel tracker sees every token of the step, review srv L10)
        if self.tc is not None:
            self.tc.track(pending)
            # (harmony: the reply's first header comes before its analysis opens: not "done" yet; review M5)
            hs = getattr(self.tc, "hs", None)
            first_hdr = hs is not None and hs.state == "hdr" and hs.fresh
            if self.answer_max is not None and not self.tc.open and not first_hdr:  # thinking done: max_tokens for the answer from here
                self.max_new = min(self.max_new, self.tc.n + 1 + self.answer_max)
                self.answer_max = None
                # the engine's limit follows (review 2 L5): a reply that ends by length leaves no verified drafts past
                # its end in the slot, so the next turn resumes from the live end
                g.engine.set_limit(self.slot, max(0, self.max_new - self.n))
        if STEP_PROF:
            q0 = _pc()
        s = self.det.push(pending) if pending else ""  # (parser tags marked: server/content_markers.py)
        if not stop_now and g.args.loop_guard_tokens and not self.p.get("ignore_eos") and self.n - self.loop_checked >= 64:
            self.loop_checked = self.n
            p = repetition_period(self.det.ids, min_span=g.args.loop_guard_tokens)
            if p:  # degenerate loop: end the turn instead of streaming up to max_tokens of it
                print(f"stopping a repetition loop (period {p} tokens) after {self.n} tokens", flush=True)
                self.finish, stop_now = "stop", True
        if stop_now or gdone or self.n >= self.max_new or len(self.ids) + self.n + 8 >= g.engine.slots[self.slot]:
            s += self.det.flush()  # the last text of the reply, whichever step it arrived in
        if STEP_PROF:
            self.prof_detok += _pc() - q0
        if self.stops:
            s, cut = self.stop_scan(s, stop_now or gdone or self.n >= self.max_new or
                                    len(self.ids) + self.n + 8 >= g.engine.slots[self.slot])
            if cut >= 0:
                if STOP_ROLLBACK:
                    self.keep = self.kept_tokens(self.text_all[:cut])
                self.finish, stop_now = "stop", True
                self.cut = True
        elif s:
            self.text_all += unmark(s)  # the text itself (stops); on_text gets the marked text
            self.text_mk += s
        if self.lp is not None:
            s = self.lp_attach(s, entry)
        if s:
            if STEP_PROF:
                q0 = _pc()
            self.on_text(s)
            if STEP_PROF:
                self.prof_emit += _pc() - q0
        if gdone and not stop_now:
            self.finish, stop_now = "stop", True
        c = self.grammar
        if stop_now and c is not None and not self.cut and (c.error is not None or not (
                c.done() or (self.finish == "stop" and c.accepting()))):
            # structured output ended before its grammar was complete (loop guard, rejected token):
            # the JSON is cut off, so say "length" (clients retry / raise) and never "stop"
            why = c.error.splitlines()[0] if c.error else "loop guard" if self.finish == "stop" else self.finish
            what = "tool_choice: the required call was not made" if isinstance(getattr(c, "m", None),
                                                                               tool_choice.ToolMatcher) \
                else "structured output: incomplete"
            print(f"{what} after {self.n} tokens ({why}), finish_reason length", flush=True)
            self.finish = "length"
        if (stop_now or self.n >= self.max_new or
                len(self.ids) + self.n + 8 >= g.engine.slots[self.slot]):
            self.finished = True
        elif self.tc is not None:
            self.tc.after(pending)  # a forced close of <think> in flight, or one due now (budget / answer tail)

    def release_grammar(self):
        """Drop a tool_choice constraint whose forced part is done (Constraint.released): the slot decodes freely again;
        a logprobs reply stays constrained for its logits with every token allowed (as ThinkControl after its close)."""
        e = self.gen.engine
        self.grammar = None
        if self.lp is None:
            e.constrain(self.slot, False)
        else:
            e.mask_buffer(self.slot).fill(0xFFFFFFFF)
            e.constrain(self.slot, True)

    def call_in_block(self):
        """The model ended the reply itself with complete tool calls inside an open <think>: they are its calls. Not after a
        stop string, max_tokens, the context limit or the loop guard: such a cut can land right after a quoted call. The
        call's tokens are then not thinking tokens. Never under tool_choice none."""
        g = self.gen
        if (self.p.get("tool_choice") or {}).get("mode") == tool_choice.NONE:
            return False
        if not self.eos_end or self.cut or mark("<tool_call>") not in self.text_mk or not g.think_ids.ok:
            return False
        p = Parser(g.think_ids.open_at_end(self.ids, g.im_start))
        p.feed(self.text_mk)
        p.feed("", final=True, promote=True)
        if not p.promoted or not re.search(r"<function=[^>\n]+>", p.tool_text):  # (as parse_tool_calls needs)
            return False
        print("WARNING: the reply ended inside <think> with a complete tool call: running the call", flush=True)
        if self.tc is not None:
            tid, left, k = g.tok.token_to_id("<tool_call>"), p.tool_text.count(mark("<tool_call>")), len(self.det.ids)
            while left and k > 0:
                k -= 1
                left -= self.det.ids[k] == tid
            self.tc.n -= min(self.tc.n, len(self.det.ids) - k)
        return True

    def answer_call(self):
        """The reply's answer (marked text: after the real </think>, before the first real tool call) and whether the reply
        made a tool call. Qwen / GLM: a real <tool_call> token, not the text (review L3). DSV4 ([dsv4-rel8-dsml]): the
        calls are plain text, so the reply's own parser decides (a DSML block in the answer, not one quoted in the
        reasoning), as the HTTP side does."""
        ans = self.text_mk
        if TOOL_FORMAT == "inkling":  # [ink-srv-capi] the content messages; a call = a parsed invoke body
            import inkling
            return inkling.answer_of(ans)
        if mark("</think>") in ans:
            ans = ans.split(mark("</think>"), 1)[1]
        if TOOL_FORMAT == "dsml":
            g = self.gen
            p = DsmlParser(bool(g.think_ids.ok and g.think_ids.open_at_end(self.ids, g.im_start)))
            p.feed(self.text_mk)
            p.feed("", final=True)
            call = False
            if p.mode == "tool" and p.tool_text:  # review rel8 L9: a call only if the block parses (as the HTTP side)
                import dsv4_chat
                t = unmark(p.tool_text)
                try:  # (the calls the reply returns: under its tool_choice, rel re-review L3)
                    call = bool(tool_choice.filter_calls(self.p.get("tool_choice"),
                                                         dsv4_chat.parse_calls(t if t.startswith("\n\n") else "\n\n" + t.lstrip())))
                except Exception:
                    call = False
            return (ans.split(DsmlParser.CALL, 1)[0] if call else ans), call
        return ans.split(mark("<tool_call>"), 1)[0], mark("<tool_call>") in self.text_mk

    def stop_from(self):
        """Where client stop sequences may match in text_all, None: not yet. On a guard-opened thinking turn they start at
        the answer's first character: the client asked for no reasoning, and its stops (ReAct "Observation:", "\\n\\n") must
        not end the reply mid-thought or in the blank line after </think>. The real </think> (marked): a block can reopen,
        and the same string in the text is text. tool_choice required / named (tool_hold): not before the forced call is
        complete (thinking, header and arguments: a stop such as "\n" or "\n\n" would cut its arguments, and the client
        would run a call with truncated values), then from the end of its real </tool_call> on."""
        if self.tool_hold:
            i = self.text_mk.find(mark("<tool_call>")) if self.grammar is None else -1  # (the first real one: forced)
            j = self.text_mk.find(mark("</tool_call>"), i) if i >= 0 else -1
            if j < 0:
                return None
            self.tool_hold = False
            self.answer_at = j + len("</tool_call>")
        if self.answer_at is None and getattr(getattr(self.gen, "profile", None), "is_inkling", False):
            import inkling
            self.answer_at = inkling.answer_start(self.text_mk)   # [ink-srv-capi] the body of the latest content message
            return self.answer_at
        if self.answer_at is None and getattr(getattr(self.gen, "profile", None), "is_gptoss", False):
            i = self.text_mk.rfind(mark("<|channel|>") + "final")  # [srv-gptoss] the answer = the final message's body
            j = self.text_mk.find(mark("<|message|>"), i) if i >= 0 else -1
            if j >= 0:
                self.answer_at = j + len("<|message|>")
            return self.answer_at
        if self.answer_at is None and not self.tc.open:
            tag = mark("</think>")
            i = self.text_mk.rfind(tag)
            rest = self.text_all[i + len(tag):]
            if i >= 0 and rest.strip():
                self.answer_at = len(self.text_all) - len(rest.lstrip())
        return self.answer_at

    def stop_scan(self, s, ending):
        """A request with stop sequences: takes new text, returns what may go to the client now and where the reply is cut
        (-1: not cut). Text that could still become a stop waits, so what the client gets does not depend on where steps
        end. The cut is the leftmost stop of the reply: a stop found while an earlier-starting one can still complete waits
        too. `ending`: nothing more comes, the waiting text goes out."""
        self.text_all += unmark(s)  # the text itself (stops); on_text gets the marked text
        self.text_mk += s
        t, sent = self.text_all, self.stop_sent
        a = self.stop_from()
        if a is None:  # nothing to scan yet: all of it goes out (the parser holds its own tags)
            self.stop_sent = len(t)
            return self.text_mk[sent:], -1
        spans, hold, tail = [(a, len(t))], len(t), True
        if TOOL_FORMAT == "dsml":
            spans, hold, tail = self.dsml_spans(t, a, ending)
        elif getattr(getattr(self.gen, "profile", None), "is_inkling", False):
            import inkling  # [ink-srv-review-fix] L5: content bodies only (no later reasoning, no tool-call JSON)
            spans, tail = inkling.stop_spans(self.text_mk)
            spans = [(u, v) for u, v in spans if v > a] or [(a, a)]
        a = max(sent, a)  # (the text in front of the answer goes out with what follows)
        found = [p for u, v in spans for p in (t.find(x, max(u, a), v) for x in self.stops) if p >= 0]
        pos = min(found) if found else -1
        # hold: start of the longest end of the text that begins a stop (inside a DSML block nothing can begin one)
        h0 = hold
        for x in self.stops if tail else ():
            for n in range(min(len(x) - 1, h0 - max(a, spans[-1][0])), 0, -1):
                if x.startswith(t[h0 - n:h0]):
                    hold = min(hold, h0 - n)
                    break
        if pos >= 0 and (ending or hold >= pos):
            self.stop_sent = pos
            return self.text_mk[sent:pos], pos
        end = len(t) if ending else hold
        self.stop_sent = end
        return self.text_mk[sent:end], -1

    def dsml_spans(self, t, a, ending=False):
        """[dsv4-rel8-think] DSV4 (shared_server_review L7): client stops never match inside a DSML call block (the
        blank line before it counts as its start; dsv4_app scanned the content only), as tool_hold waits for the forced
        call's end. Returns the (start, end) spans of t from a on where a stop may match, the end of the text a stop
        may still begin in (an end of the text that can still become a block start waits), and whether that end is
        free text (False: inside an open block)."""
        lead, call, end = DsmlParser.LEAD, DsmlParser.CALL, DsmlParser.END_CALL
        spans, u = [], a
        while True:
            k = t.find(call, u)
            if k < 0:
                break
            b = k
            while b > u and k - b < len(lead) and t[b - 1] == "\n":
                b -= 1
            spans.append((u, b))
            e = t.find(end, k)
            if e < 0:
                return spans, len(t), False
            u = e + len(end)
        head = lead + call
        n = 0 if ending else next((n for n in range(min(len(head) - 1, len(t) - u), 0, -1)
                                   if head.startswith(t[-n:]) or call.startswith(t[-n:])), 0)
        spans.append((u, len(t) - n))
        return spans, len(t) - n, True

    def kept_tokens(self, text):
        """Reply tokens wholly inside the text before the stop (trailing whitespace dropped: the
        reply comes back stripped)."""
        dec, ids = self.gen.tok.decode, self.det.ids
        t = text.rstrip()
        k = len(ids)
        while k > 0 and not t.startswith(dec(ids[:k], skip_special_tokens=False)):
            k -= 1
        return k

    def stop_risk(self):
        """A stop sequence may start at the step in flight's pending token or in its drafts and end
        past them: the step must stay undoable (no step queued after it)."""
        if STOP_ROLLBACK == 1:
            return True
        dec = self.gen.tok.decode
        tail = self.det.ids[-4:]
        d = self.gen.engine.drafts(self.slot)
        t0 = len(dec(tail[:-1], skip_special_tokens=False))
        for j in range(len(d) + 1):
            u = dec(tail + d[:j], skip_special_tokens=False)[t0:]
            for x in self.stops:
                for q in range(len(u)):
                    w = u[q:]
                    if w.startswith(x) or x.startswith(w):
                        return True
        return False

    def parkable(self):
        """[q27-park-c] A reply the engine can park and resume bit-identically: no grammar / logprobs / token ban / images,
        not finished (the engine refuses steering, LSA and the rest itself)."""
        return (not self.finished and self.grammar is None and self.lp is None and not self.tool_banned and
                not getattr(self.ids, "mm", None))

    def check_done(self):
        """Release the slot and wake the caller once finished (or cancelled)."""
        if not self.finished and (self.cancelled() or self.gen.stopping):
            self.shut = self.gen.stopping and not self.cancelled()  # shutdown: run() raises (error event)
            self.finished = True
            if self.cancelled() and getattr(self.cancelled, "why", "client") == "client":
                self.client_gone = True
        if not self.finished:
            return False
        if not self.done.is_set():
            promoted = self.call_in_block()
            ans, call = self.answer_call()
            if self.prio == 1 and self.finish == "stop" and not call:
                self.gen.engine.set_prio(self.slot, 0)  # a subagent that is done (no tool to run)
            if getattr(self, "det", None) is not None:  # its final reply is recognised in a parent's history
                self.gen.agents.finished(getattr(self.ids, "agent", None), unmark(ans), call, split=False)
            ds = engine_stats.delta(getattr(self, "ds0", None), self.gen.dstats.read(self.slot))
            trimmed = None
            if getattr(self, "keep", None) is not None:
                trimmed = self.gen.engine.trim(self.slot, len(self.ids) + self.keep)
                if trimmed:  # (hotfix_1331 review F4: Flash-Next's qe_trim binds this slot: the next single step resumes its own)
                    self.gen.cur_slot = -1
            t2 = time.time()
            self.stats = {"prompt_tokens": len(self.ids), "cached_tokens": self.reused,
                          "cache_src": self.src,
                          "completion_tokens": self.n_out(), "finish": self.finish, "slot": self.slot,
                          "prefill_s": self.t1 - self.t0, "decode_s": t2 - self.t1,
                          "queue_s": max(0.0, self.t0 - self.t_enq)}
            if self.lp is not None:
                self.stats["logprobs"] = self.lp_out  # (entries no text carried: the HTTP side's last chunk)
            if ds is not None:
                self.stats["drafts"] = ds  # [steps, MTP drafted, accepted, lookup drafted, accepted]
            if trimmed is not None:
                self.stats["stop_trim"] = trimmed
            if getattr(self, "client_gone", False):
                self.stats["cancelled"] = True  # (complete(): no empty-reply retry for a client that left)
            if promoted:
                self.stats["promote_call"] = True
            if self.penalty:
                self.stats["penalty"] = self.penalty
            if self.guarded:
                self.stats["loop_guard"] = self.guarded
            tc = getattr(self, "tc", None)
            if tc is not None:
                self.stats["think"] = tc.stats()
                g = self.gen
                g.think_add("requests_with_budget", tc.budget is not None)
                g.think_src(tc.src)
                if tc.forced:
                    g.think_add("forced_" + tc.forced)
                if tc.failed:
                    g.think_add("forced_failed")
            if getattr(self, "det", None) is not None and not self.cancelled() and not getattr(self, "shut", False):
                er = empty_reply(self)  # a turn with no text and no tool call: say why (EOS after think / cut-off)
                if er is not None:
                    g = self.gen
                    g.think_add("empty_replies")
                    g.think_add("empty_" + self.finish)
                    g.think_add("empty_after_forced_close", bool(er["forced"]))
                    self.stats["empty"] = {k: v for k, v in er.items() if k != "line"}
                    print(er["line"], flush=True)
            if self.p.get("tool_choice"):
                self.stats["tool_choice"] = self.p["tool_choice"]["mode"] + (" (cut)" if self.p["tool_choice"].get("cut")
                                                                             else " (ban)" if self.tool_banned else "")
            if self.grammar is not None:
                self.stats["grammar_s"] = self.grammar_s
                self.stats["grammar_steps"] = self.gsteps
                self.stats["grammar_forced"] = self.grammar.n_forced
            self.gen.metrics.finished(self)
            if STEP_PROF and self.prof:
                prof_write(self)
            # [q27-end-drain] the reply is done before the release: release() waits for the speculative step queued
            # ahead (one step, ~21 ms on one 27B card), which the client should not wait for. The slot is still released
            # right here, on this (the generator's) thread, before it takes the next request: the order of slot reuse
            # is unchanged. remember / remember_turn feed encode() of the NEXT request, which runs on the HTTP thread
            # as soon as the client has the reply, so they run before done (review q27_1card3 M1: a fast agent turn
            # inside the release window would otherwise re-tokenize the reply text and miss the prefix cache)
            keep_out = None
            if getattr(self, "det", None) and not getattr(self.ids, "mm", None) and (
                    not self.cut or getattr(self, "keep", None) is not None):
                keep_out = self.det.ids if not self.cut else self.det.ids[:self.keep]
                if not promoted:  # (the next turn renders a </think> the tokens don't have: not a changed reply)
                    self.gen.remember(self.affinity, self.ids, keep_out)
                self.gen.remember_turn(self.ids, keep_out)
            self.done.set()
            self.gen.engine.release(self.slot)
            if getattr(self, "client_gone", False):
                self.gen.cancel_note(self, f"it generated ({getattr(self, 'n', 0)} tokens out)")
            if keep_out is not None:
                self.gen.engine.add_corpus(keep_out)
        return True

    def fail(self, ex):
        if self.slot >= 0:
            try:
                self.gen.engine.release(self.slot)
            except Exception:
                pass
        self.error = ex
        self.finished = True
        self.gen.metrics.finished(self, None if isinstance(ex, Cancelled) else ex)  # (a client that left: no error)
        self.done.set()


def _pstats(v):
    v = sorted(v)
    return {"mean": sum(v) / len(v), "p50": v[len(v) // 2], "p90": v[len(v) * 9 // 10], "max": v[-1]}


def prof_write(job):
    """QE_STEP_PROF: per-request step profile (seconds) -> JSONL + one summary line."""
    P = job.prof
    keys = ("step", "ntok", "metrics", "feed", "gap", "detok", "emit")
    cols = {k: [r[i] for r in P] for i, k in enumerate(keys)}
    gaps = [g for g in cols["gap"] if g > 0]
    rec = {"prompt_n": len(job.ids), "gen": job.n, "steps": len(P), "decode_s": time.time() - job.t1,
           "prefill_s": job.t1 - job.t0, "ids": list(job.ids), "cols": cols,
           "sum": {k: _pstats(v) for k, v in cols.items() if v}}
    if gaps:
        rec["sum"]["gap"] = _pstats(gaps)
    with open(env_str("QE_STEP_PROF_FILE", "step_prof.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    S = rec["sum"]
    print(f"[prof] {len(P)} steps, {job.n / len(P):.2f} tok/step | step call mean {S['step']['mean'] * 1e3:.3f} ms "
          f"p90 {S['step']['p90'] * 1e3:.3f} | between steps mean {S.get('gap', S['step'])['mean'] * 1e6:.0f} us "
          f"p90 {S.get('gap', S['step'])['p90'] * 1e6:.0f} (metrics {S['metrics']['mean'] * 1e6:.0f}, feed "
          f"{S['feed']['mean'] * 1e6:.0f} = detok {S['detok']['mean'] * 1e6:.0f} + emit {S['emit']['mean'] * 1e6:.0f} + rest)",
          flush=True)


class Cancelled(SequenceFault):
    """[srv-disconnect] the request ended because its client left (queued, prefilling, parked): no engine error (not
    counted as one; the reply, written to nobody, is a 499)"""


class Cancel:
    """[srv-disconnect] A request's cancel. The HTTP side sets it (the client disconnected, or the reply's stream was
    closed); the scheduler reads it (calling it: Job.cancelled) at its safe points: queued, between prefill chunks /
    windows, between decode steps, parked. flag: the same as a C int, for an engine that reads it inside one call (DSV4
    prefills a prompt alone in one dsv4_start: it stops at the next chunk). t: monotonic time of the disconnect (the
    scheduler logs the time until the slot was free); why: 'client' (a disconnect seen) or '' (the reply's end)."""
    __slots__ = ("ev", "flag", "t", "why", "gen")

    def __init__(self, gen=None):
        self.ev = threading.Event()
        self.flag = ctypes.c_int(0)
        self.t, self.why, self.gen = None, "", gen

    def __call__(self):
        return self.ev.is_set()

    is_set = __call__

    def set(self, why=""):
        if self.ev.is_set():
            return
        self.t, self.why = time.monotonic(), why
        self.ev.set()
        self.flag.value = 1
        g = self.gen
        if g is not None and why and g.cv.acquire(blocking=False):  # wake an idle / waiting scheduler (never blocks the
            try:                                                      # event loop: a busy one sees it at its next pass)
                g.cv.notify()
            finally:
                g.cv.release()


async def watch_disconnect(req, cancel):
    """[srv-disconnect] Waits for the client's http.disconnect on the request's receive channel (the body was read
    before): a client that leaves while its request is queued or its prompt is prefilled (nothing is written then, so
    a write never fails) cancels it at the scheduler's next safe point. Non-streamed replies write nothing until the
    end: without this they ran to completion for nobody."""
    try:
        while not cancel():
            msg = await req.receive()
            if msg.get("type") == "http.disconnect":
                cancel.set("client")
                return
            await asyncio.sleep(0.05)  # (another message: a server that repeats empty bodies must not spin)
    except asyncio.CancelledError:
        raise
    except Exception:  # (no receive channel / a closed one: no watch, the write path still notices)
        return


def stream_bridge(gen, ids, params, affinity=0, req=None, keepalive=0.0):
    """Run generation in a thread; yields text chunks then the stats dict (async). req: the HTTP request, watched for
    a client disconnect while the generation runs (watch_disconnect). keepalive > 0 (streams only): KEEPALIVE after that
    many seconds without an item."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    cancel = Cancel(gen)

    def work():
        try:
            st = gen.run(ids, params, lambda s: loop.call_soon_threadsafe(q.put_nowait, s),
                         cancel, affinity)
            loop.call_soon_threadsafe(q.put_nowait, st)
        except Exception as ex:  # surface errors to the client
            loop.call_soon_threadsafe(q.put_nowait, ex)

    threading.Thread(target=work, daemon=True).start()

    async def agen():
        watch = asyncio.ensure_future(watch_disconnect(req, cancel)) if req is not None else None
        ended = False
        try:
            while True:
                if keepalive > 0:
                    try:
                        item = await asyncio.wait_for(q.get(), keepalive)
                    except asyncio.TimeoutError:  # (a cancelled Queue.get loses no item)
                        yield KEEPALIVE
                        continue
                else:
                    item = await q.get()
                if isinstance(item, (Exception, dict)):
                    ended = True
                    if watch is not None:  # (review L1: a completed response reads as http.disconnect, never a cancel)
                        watch.cancel()
                if isinstance(item, Exception):
                    raise item
                yield item
                if isinstance(item, dict):
                    return
        except (asyncio.CancelledError, GeneratorExit):
            # the stream was closed before its end (a write to a gone client failed, or the server cancelled the
            # response on a disconnect)
            if not ended:
                cancel.set("client")
            raise
        finally:
            if watch is not None:
                watch.cancel()
            cancel.set()

    return agen()


def bad_input(ex):
    """A final start refusal (bad input, too long for the engine): the client's request is at fault -> 400, not an
    engine error 500 that a client would retry unchanged (review server F5)."""
    return isinstance(ex, StartRefused) and not ex.retry


async def fault_events(it):
    """stream_bridge items, with an engine fault (SequenceFault) as the last item instead of an exception."""
    try:
        async for item in it:
            yield item
    except SequenceFault as ex:
        yield ex


async def complete(gen, ids, params, aff, thinking, tools, tchoice=None, req=None):
    """A non-streamed reply: (parser, parts, stats, tool calls), or the SequenceFault of an engine error. A sampled
    reply that ends by itself (finish stop) with no text and no tool call -- reasoning only (EOS inside or right after
    <think>, HF Qwen3.8-Flash-Next #50) or a tool call that does not parse -- is sampled again, up to --empty-retry
    times: agent harnesses (AutomationBench, verifiers, Claude Code) end the task on such a turn, so one bad sample
    costs the whole task. The prompt is cached, so a retry costs only its decode. Off for greedy (temperature 0) and
    seeded requests (a resample would repeat the output or break reproducibility), structured output (single-use
    grammar), ignore_eos and client stop sequences (the stop may have cut the text). Streams are never retried: their
    bytes are already sent. usage.completion_tokens counts the tokens of every try; finish_reason is the last try's.
    tchoice (the request's tool_choice, tool_choice_of): the calls it allows (tool_choice.filter_calls; under none they
    are not parsed: none are left, masked or cut without the grammar tier)."""
    tries = max(0, getattr(gen.args, "empty_retry", 0))
    temp = params.get("temperature")
    temp = gen.sampling_defaults[bool(thinking)]["temperature"] if temp is None else temp
    if (params.get("grammar") is not None or temp <= 0 or params.get("seed") is not None or params.get("ignore_eos")
            or params.get("stop")):
        tries = 0
    spent, spent_s, first = 0, 0.0, None
    for attempt in range(tries + 1):
        parser = PARSER(thinking)
        parts = {"reasoning": "", "content": "", "logprobs": []}
        st = None
        p = dict(params, empty_retry=(attempt, tries)) if tries else params  # the empty-reply WARNING says retry k/N
        async for item in fault_events(stream_bridge(gen, ids, p, aff, req)):
            if isinstance(item, SequenceFault):
                return item
            if isinstance(item, dict):
                st = item
                for kind, t in parser.feed("", final=True, promote=st.get("promote_call", False)):
                    parts[kind] += t
                parts["logprobs"] += st.get("logprobs", ())
            else:
                for kind, t in parser.feed(item):
                    parts[kind] += t
                if type(item) is LpText:
                    parts["logprobs"] += item.lp
        if first is None:
            first = st
        no_calls = tchoice is not None and tchoice["mode"] == tool_choice.NONE
        calls, raw_tool = tool_calls_or_text(parser, tools, tchoice, no_calls)
        empty = not calls and not parts["content"].strip() and st["finish"] == "stop"
        if st.get("cancelled"):  # [srv-disconnect] the client left: no retry
            break
        if not empty:
            if attempt:
                gen.think_add("empty_retry_recovered")
                print(f"empty reply: recovered on retry {attempt}/{tries}: "
                      f"{f'{len(calls)} tool call(s)' if calls else 'text answer' if parts['content'].strip() else st['finish']}",
                      flush=True)
            break
        if attempt == tries:
            if tries:
                gen.think_add("empty_retry_exhausted")
                print(f"empty reply: still empty after {tries}/{tries} retries, returned to the client as is",
                      flush=True)
            break
        spent += st["completion_tokens"]
        spent_s += st.get("decode_s", 0.0)
        gen.think_add("empty_retried")
        if parser.tool_text:  # the empty-reply WARNING skips replies with <tool_call>: this is the line for it
            raw = diag_dump("tool", unmark(parser.tool_text))
            print(f"WARNING empty reply: a tool call that does not parse ({st['completion_tokens']} tokens"
                  f"{', raw: ' + raw if raw else ''}); retry {attempt + 1}/{tries} follows", flush=True)
    if raw_tool:  # the last try's tool text did not parse: returned as text, never an empty reply
        parts["content"] += raw_tool
    if spent:
        # usage over all tries (review server F4): the client's prompt usage and the prefill are the first try's (the
        # retries come from the cache), tokens and decode time the sum, so the logged tok/s stays true
        st = dict(st, completion_tokens=st["completion_tokens"] + spent, empty_retries=attempt,
                  decode_s=st.get("decode_s", 0.0) + spent_s)
        for k in ("cached_tokens", "prefill_s", "cache_src"):
            if k in first:
                st[k] = first[k]
    return parser, parts, st, calls


def think_note(st):
    th = st.get("think")
    if not th or (th["budget"] is None and not th["forced"]):
        return ""
    b = f"/{th['budget']} ({th.get('budget_from') or 'request'})" if th["budget"] is not None else ""
    f = f", forced close ({th['forced']}, {th['forced_tokens']} tokens)" if th["forced"] else ""
    return f" | think {th['tokens']}{b}{f}{' FAILED' if th.get('failed') else ''}"


def timings(st):
    """[dsv4-rel8-timings] llama.cpp's `timings` object of a response (DSV4 only, as dsv4_app returned one; review L9):
    prompt_n = the tokens prefilled (the cached ones in cache_n), prompt_ms its wall time (prompt chunks beside other
    requests' decode included), predicted_* the reply; ttft_s (queue wait + prefill) and tok_s as dsv4_app had them."""
    pn = max(0, st["prompt_tokens"] - st["cached_tokens"])
    pms, dms, dn = 1000.0 * st["prefill_s"], 1000.0 * st["decode_s"], st["completion_tokens"]
    return {"cache_n": st["cached_tokens"], "prompt_n": pn, "prompt_ms": round(pms, 3),
            "prompt_per_token_ms": round(pms / pn, 3) if pn else 0.0,
            "prompt_per_second": round(1000.0 * pn / pms, 2) if pms > 0 else 0.0,
            "predicted_n": dn, "predicted_ms": round(dms, 3), "predicted_per_token_ms": round(dms / dn, 3) if dn else 0.0,
            "predicted_per_second": round(1000.0 * dn / dms, 2) if dms > 0 else 0.0,
            "ttft_s": round(st.get("queue_s", 0.0) + st["prefill_s"], 3),
            "tok_s": round(1000.0 * dn / dms, 2) if dms > 0 else 0.0}


def log_stats(kind, st):
    tps = st["completion_tokens"] / max(st["decode_s"], 1e-6)
    pps = (st["prompt_tokens"] - st["cached_tokens"]) / max(st["prefill_s"], 1e-6)
    src = f" from {st['cache_src']}" if st.get("cache_src") else ""
    gr = (f" | grammar {1e6 * st['grammar_s'] / max(st['completion_tokens'], 1):.0f} us/token, "
          f"{st['completion_tokens'] / max(st['grammar_steps'], 1):.2f} tok/step, "
          f"{st['grammar_forced']} forced drafts" if "grammar_s" in st else "")
    print(f"[{kind}] slot {st['slot']} prompt {st['prompt_tokens']} (cached {st['cached_tokens']}{src}, prefill "
          f"{pps:.0f} tok/s) | gen {st['completion_tokens']} @ {tps:.1f} tok/s | {st['finish']}{gr}"
          f"{' | stop trim ' + ('ok' if st['stop_trim'] else 'missed') if 'stop_trim' in st else ''}{think_note(st)}"
          f"{' | tool_choice ' + st['tool_choice'] if st.get('tool_choice') else ''}",
          flush=True)
    engine_stats.log(st, st.get("drafts"))  # [stats] lines for benchmark log parsers (QE_STATS_LOG=0: off)


# ----------------------------------------------------------------------------- request mapping


# reasoning efforts qwen38_syssafe.jinja knows: 'high' renders as 'xhigh', 'medium' adds no instruction, and
# any other value (typos) renders as 'medium' (the model's own template raises instead); the server rewrites
# 'max' to 'xhigh' and 'minimal' to 'low' before rendering (think_budget.QWEN_RULES.client), and 'none' (the
# OpenAI "no reasoning" value, any case) to enable_thinking=false before any template sees it (template_kwargs)
KNOWN_EFFORTS = ("xhigh", "high", "medium", "low")
CLIENT_EFFORTS = ("max", "minimal")
NO_EFFORT = "none"


def is_no_effort(e):
    """The effort means "no reasoning" (OpenAI `none`; case and surrounding blanks ignored)."""
    return isinstance(e, str) and e.strip().lower() == NO_EFFORT


EMPTY_THINK = "<think>\n\n</think>\n\n"  # the template's generation prompt with enable_thinking=false


def guard_budget(ids, budget, src):
    """(think budget, source) of a request whose turn the loop guard generates with reasoning on (prompt():
    guard["think"]): at most --loop-guard-think tokens; otherwise the request's own."""
    g = getattr(ids, "guard", None)
    if not g or not g.get("think"):
        return budget, src
    return (g["think"] if budget is None else min(budget, g["think"])), "loop_guard"


def template_kwargs(body, default_think, default_effort=None, anthropic=False):
    """Template variables of a request. Effort: the request's chat_template_kwargs.reasoning_effort, else its
    top-level reasoning_effort (OpenAI field, also read on the Anthropic path), else reasoning.effort (OpenAI
    Responses / OpenRouter object, both paths), else (anthropic: /v1/messages) output_config.effort, else
    --reasoning-effort (default_effort; None = the template's own default, xhigh).
    A null value counts as not sent. enable_thinking: chat_template_kwargs.enable_thinking, else a top-level
    enable_thinking (a real bool only; issue #31), else --think.

    Effort `none` (any case, surrounding blanks ignored): thinking off, i.e. the variables of enable_thinking=false
    (reasoning_effort is dropped, so the prompt is byte for byte that of enable_thinking=false). A `none` the request
    sends (any of the fields above) wins over its own enable_thinking: true, as enable_thinking: false wins over an
    effort (any "off" wins). The server default `none` (--reasoning-effort) only sets the default: a request that
    turns thinking on itself (chat_template_kwargs.enable_thinking: true, thinking.type enabled / adaptive) thinks with
    the template's own default effort, one that names an effort thinks at that effort.

    reasoning.enabled (OpenRouter; true / false only) has the lowest precedence. false (= effort none) applies only when
    the request sends none of the fields above (no effort, no enable_thinking, no thinking.type); true turns thinking on
    over the server default (--think 0, --reasoning-effort none) unless the request turns it off itself, at the effort
    the request names, else the server's, else the template's own default."""
    kw = dict(body.get("chat_template_kwargs") or {})
    if kw.get("enable_thinking") is None:  # [kb-server] a null counts as not sent (server-small N2: it kept the default off)
        kw.pop("enable_thinking", None)
        if isinstance(body.get("enable_thinking"), bool):  # issue #31: the top-level form (only a real bool), was ignored
            kw["enable_thinking"] = body["enable_thinking"]
    th = body.get("thinking")
    th_type = th.get("type") if isinstance(th, dict) else None
    rs = body.get("reasoning")  # OpenAI Responses / OpenRouter: reasoning: {"effort": ..., "enabled": ...}
    rs = rs if isinstance(rs, dict) else {}
    oc = body.get("output_config")
    # OpenRouter reasoning.enabled (a real bool only), the lowest-precedence switch: false only when the request sends no
    # other thinking switch or effort (fields above); true unless the request turns thinking off itself (enable_thinking
    # false, thinking.type disabled, an effort none), with the effort the request names (OpenRouter clients send both)
    en = rs.get("enabled")
    efforts = [kw.get("reasoning_effort"), body.get("reasoning_effort"), rs.get("effort")]
    if anthropic and isinstance(oc, dict):
        efforts.append(oc.get("effort"))
    sent = kw.get("enable_thinking") is not None or th_type is not None or any(x is not None for x in efforts)
    asked_off = kw.get("enable_thinking") is False or th_type == "disabled" or any(is_no_effort(x) for x in efforts)
    en = en if isinstance(en, bool) and not (sent if en is False else asked_off) else None
    if th_type == "disabled":
        kw.setdefault("enable_thinking", False)
    from_request = True
    if kw.get("reasoning_effort") is None:
        kw.pop("reasoning_effort", None)
        eff = body.get("reasoning_effort")
        if eff is None:
            eff = rs.get("effort")
        if eff is None and anthropic and isinstance(oc, dict):
            eff = oc.get("effort")
        if eff is None:
            eff, from_request = default_effort, False
        if eff is not None:
            kw["reasoning_effort"] = eff
    if is_no_effort(kw.get("reasoning_effort")):
        del kw["reasoning_effort"]
        # (only a real bool is a switch: the template tests `is true` / `is false`, so 0 or "false" still think)
        asked_on = kw.get("enable_thinking") is True or th_type in ("enabled", "adaptive")
        if from_request or not asked_on:
            kw["enable_thinking"] = False
    if en is False:  # as effort none: the prompt of enable_thinking=false
        kw.pop("reasoning_effort", None)
        kw["enable_thinking"] = False
    elif en is True:  # on even under --think 0 / --reasoning-effort none (then at the template's own default effort)
        kw["enable_thinking"] = True
    kw.setdefault("enable_thinking", default_think)
    return kw


def reasoning_excluded(body):
    """OpenRouter reasoning.exclude: true: the model reasons as usual (same prompt, same budget, reasoning tokens in
    usage), the reply carries no reasoning_content (/v1/chat/completions; streamed reasoning deltas are dropped)."""
    rs = body.get("reasoning")
    return isinstance(rs, dict) and rs.get("exclude") is True


def structured(gen, body, kw):
    """Grammar of a structured-output request (None if unconstrained). The grammar applies from
    the first generated token, so such requests run without thinking (enable_thinking=false)."""
    schema = schema_of(body)
    if schema is None:
        return None
    if gen.grammars is None:
        why = getattr(gen.engine, "refused", {}).get("grammar")
        raise GrammarError(why or "structured output is not supported by this engine (no grammar C API)")
    kw["enable_thinking"] = False
    c = gen.grammars.matcher(schema)
    if TOOL_FORMAT == "harmony":  # [srv-gptoss] gpt-oss: its analysis first (free), then the final header, then the
        import harmony               # schema's body (forcing the final header at once broke the JSON on the GPU: city " Warsaw\xa0?...")
        c.m = harmony.ReasonFirst(gen.tok, c.m)
    return c


def tool_choice_of(gen, body, tools, grammar):
    """The tool_choice of a request for its job (params["tool_choice"]; None: auto, nothing to enforce). Raises
    tool_choice.ToolChoiceError (HTTP 400). With the grammar tier the job masks the reply (server/tool_choice.py);
    without it `none` ends the reply at its first tool call ("cut") and a required / named choice is not enforced
    (one log line at the first such request). A structured-output request (response_format) cannot also force a call;
    its JSON grammar already allows no tool call, so `none` needs nothing more."""
    spec = tool_choice.parse(body.get("tool_choice"), tools)
    if spec is None:
        return None
    if grammar is not None:
        if spec["mode"] == tool_choice.REQUIRED:
            raise tool_choice.ToolChoiceError("a tool_choice that forces a call cannot be combined with structured "
                                              "output (response_format / output_format)")
        return None
    if gen.tool_tier is not None:
        return spec
    if spec["mode"] == tool_choice.NONE:
        return dict(spec, cut=True)
    gen.tool_choice_noted = True  # (logged once by the feature table: ParamNotes)
    return None


class InputError(ValueError):
    """A request input the server cannot use (unreadable image, unsupported content part): HTTP 400."""


UNSUPPORTED_PARTS = {"video_url": "video", "video": "video", "input_video": "video", "input_audio": "audio",
                     "audio_url": "audio", "audio": "audio"}
# documents / files (e.g. a PDF an agent read): a text note instead of an error, so that a conversation that
# carries one in its history goes on
OMITTED_PARTS = {"file", "document", "input_file"}
OMITTED_NOTE = "[document omitted: this server reads only text and images]"


def image_part(p):
    """OpenAI / Anthropic image content part -> {"type": "image", "_src": url or base64} or None."""
    if not isinstance(p, dict):
        return None
    t = p.get("type")
    if t == "image_url":
        u = p.get("image_url")
        return {"type": "image", "_src": u.get("url") if isinstance(u, dict) else u}
    if t == "image":
        s = p.get("source")
        if isinstance(s, dict):  # Anthropic
            return {"type": "image", "_src": s.get("data") if s.get("type") == "base64" else s.get("url")}
        if isinstance(p.get("image"), str):
            return {"type": "image", "_src": p["image"]}
    return None


def content_parts(parts):
    """Content list -> plain string if it has no images, else [text / image parts]."""
    out = []
    for p in parts or []:
        if isinstance(p, dict) and p.get("type") in UNSUPPORTED_PARTS:
            raise InputError(f"{UNSUPPORTED_PARTS[p['type']]} input is not supported (content part type {p['type']!r}); "
                             "only text and images")
        if isinstance(p, dict) and p.get("type") in OMITTED_PARTS:
            out.append({"type": "text", "text": OMITTED_NOTE})
            continue
        im = image_part(p)
        if im and im["_src"]:
            out.append(im)
        elif isinstance(p, dict) and p.get("type") == "text":
            out.append({"type": "text", "text": p.get("text", "")})
    if not any(p["type"] == "image" for p in out):
        return "".join(p["text"] for p in out)
    return out


def _parts_error(where, c, anthropic):
    """The error of a content value (None: fine): a string, or a list of parts whose text fields are strings
    (Anthropic: every block an object; OpenAI also takes null, and skips list items that are no object, as before)."""
    if c is None and not anthropic:
        return None
    if isinstance(c, str):
        return None
    if not isinstance(c, list):
        return f"{where} must be a string or a list of content {'blocks' if anthropic else 'parts'}"
    for k, b in enumerate(c):
        if not isinstance(b, dict):
            if anthropic:
                return f"{where}[{k}] must be an object"
            continue  # (OpenAI parts that are no object are skipped, as before)
        f = {"text": "text", "thinking": "thinking"}.get(b.get("type"))  # (other parts: their text is not read)
        if f and f in b and not isinstance(b[f], str):
            return f"{where}[{k}].{f} must be a string"
        if b.get("type") == "tool_use" and b.get("input") and not isinstance(b["input"], dict):  # (falsy: {} as before)
            return f"{where}[{k}].input must be an object"
        if b.get("type") == "tool_use" and b.get("name") is not None and not isinstance(b["name"], str):
            return f"{where}[{k}].name must be a string"
        if b.get("type") == "tool_result" and b.get("content") is not None:
            e = _parts_error(f"{where}[{k}].content", b["content"], False)
            if e:
                return e
    return None


def request_shape_error(body, anthropic=False):
    """(param, message) for a request whose fields have the wrong JSON types (HTTP 400 invalid_request_error instead of
    an exception in the message conversion or the template: a 500), None: fine. Checks only what the conversion reads;
    values that worked before still do (review PR-500)."""
    if not isinstance(body, dict):
        return None, "the request body must be a JSON object"
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return "messages", "messages is required and must be a list of messages"
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            return "messages", f"messages[{i}] must be an object"
        if not isinstance(m.get("role"), str):
            return "messages", f"messages[{i}].role must be a string"
        if anthropic or "content" in m:
            e = _parts_error(f"messages[{i}].content", m.get("content"), anthropic)
            if e:
                return "messages", e
        tc = m.get("tool_calls")
        if not anthropic and tc is not None:
            if not isinstance(tc, list):
                return "messages", f"messages[{i}].tool_calls must be a list"
            for k, c in enumerate(tc):
                f = c.get("function", {}) if isinstance(c, dict) else None
                if not isinstance(f, dict):
                    return "messages", f"messages[{i}].tool_calls[{k}] must be an object with a function object"
                if f.get("name") is not None and not isinstance(f["name"], str):
                    return "messages", f"messages[{i}].tool_calls[{k}].function.name must be a string"
                if f.get("arguments") is not None and not isinstance(f["arguments"], (str, dict)):
                    return "messages", f"messages[{i}].tool_calls[{k}].function.arguments must be a string or an object"
    tools = body.get("tools")
    if tools is not None and not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
        return "tools", "tools must be a list of objects"
    ck = body.get("chat_template_kwargs")
    if ck is not None and not isinstance(ck, dict):
        return "chat_template_kwargs", "chat_template_kwargs must be an object"
    clash = sorted(set(ck or ()) & {"messages", "tools", "add_generation_prompt"})
    if clash:  # (the server sets these template variables itself)
        return "chat_template_kwargs", f"chat_template_kwargs may not set {', '.join(clash)}"
    if anthropic:
        sy = body.get("system")
        if sy is not None and not isinstance(sy, str):
            e = _parts_error("system", sy, True)
            if e:
                return "system", e
    return None


def openai_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = dict(c.get("function", {}))
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"] or "{}")
                    except Exception:
                        pass
                calls.append({**c, "function": f})
            m["tool_calls"] = calls
        if isinstance(m.get("content"), list):  # text and image parts
            m["content"] = content_parts(m["content"])
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


class ReasoningMemory:
    """Claude Code sends our thinking back empty (display: omitted). Re-inserting the reasoning we
    generated makes the rendered history identical to what the engine processed, so the live
    state is reused instead of rolling back to the previous prompt's snapshot. Replies with tool
    calls are keyed by their unique tool id; others by the message they answered plus their full
    text, so a short or empty reply never picks up another conversation's reasoning."""

    def __init__(self, cap=512):
        self.d, self.cap = {}, cap

    @staticmethod
    def key(prev, text, tool_ids):
        if tool_ids:
            return tool_ids[0]
        c = (prev or {}).get("content")
        if isinstance(c, list):
            c = [p.get("text", p.get("type")) for p in c]
        ctx = json.dumps([(prev or {}).get("role"), (prev or {}).get("tool_call_id"), c], ensure_ascii=False)
        return "t:" + hashlib.sha1((ctx + "\0" + (text or "").strip()).encode()).hexdigest()

    def put(self, prev, text, tool_ids, reasoning):
        if not (reasoning or "").strip():
            return
        if len(self.d) >= self.cap:
            self.d.pop(next(iter(self.d)))
        self.d[self.key(prev, text, tool_ids)] = reasoning

    def get(self, prev, text, tool_ids):
        return self.d.get(self.key(prev, text, tool_ids))


REASONING = ReasoningMemory()
BILLING_RE = re.compile(r"^x-anthropic-billing-header:.*$\n?", re.M)


def anthropic_to_messages(body):
    msgs = []
    sys = body.get("system")
    if isinstance(sys, list):
        sys = "\n".join(b.get("text", "") for b in sys if b.get("type") == "text")
    # Claude Code without CLAUDE_CODE_ATTRIBUTION_HEADER=0 prepends a per-request hash line,
    # which would invalidate the prompt cache from the very first token
    sys = BILLING_RE.sub("", sys or "")
    if sys:
        msgs.append({"role": "system", "content": sys})

    def text_of(c):
        if isinstance(c, str):
            return c
        return "".join(b.get("text", "") for b in c or [] if b.get("type") == "text")

    for m in body.get("messages", []):
        role, content = m["role"], m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if role == "system":  # mid-conversation system messages (same rendering in any form)
            msgs.append({"role": "system", "content": text_of(content)})
            continue
        if role == "user":
            texts = []
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    res = content_parts(c) if isinstance(c, list) else (c or "")
                    if b.get("is_error"):
                        res = "Error: " + res if isinstance(res, str) else [{"type": "text", "text": "Error: "}] + res
                    msgs.append({"role": "tool", "content": res, "tool_call_id": b.get("tool_use_id")})
                elif b.get("type") == "text":
                    texts.append({"type": "text", "text": b.get("text", "")})
                elif image_part(b):
                    texts.append(image_part(b))
                elif b.get("type") in OMITTED_PARTS:
                    texts.append({"type": "text", "text": OMITTED_NOTE})
                elif b.get("type") in UNSUPPORTED_PARTS:
                    raise InputError(f"{UNSUPPORTED_PARTS[b['type']]} input is not supported (content block type "
                                     f"{b['type']!r}); only text and images")
            if texts:
                if any(t["type"] == "image" for t in texts):  # text blocks joined by newlines
                    parts = []
                    for t in texts:
                        if t["type"] == "text" and parts and parts[-1]["type"] == "text":
                            parts[-1] = {"type": "text", "text": parts[-1]["text"] + "\n" + t["text"]}
                        else:
                            parts.append(t)
                    msgs.append({"role": "user", "content": parts})
                else:
                    msgs.append({"role": "user", "content": "\n".join(t["text"] for t in texts)})
        else:
            a = {"role": "assistant", "content": "", "reasoning_content": ""}
            calls = []
            for b in content:
                t = b.get("type")
                if t == "text":
                    a["content"] += b.get("text", "")
                elif t == "thinking":
                    a["reasoning_content"] += b.get("thinking", "")
                elif t == "tool_use":
                    calls.append({"id": b.get("id"), "type": "function",
                                  "function": {"name": b.get("name"), "arguments": b.get("input") or {}}})
            if calls:
                a["tool_calls"] = calls
            if not a["reasoning_content"].strip():
                r = REASONING.get(msgs[-1] if msgs else None, a["content"], [c["id"] for c in calls])
                if r:
                    a["reasoning_content"] = r
            msgs.append(a)
    return msgs


def anthropic_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema", {})}}
            for t in tools or [] if "name" in t]


# ----------------------------------------------------------------------------- app


def build_app(gen: Generator, args):
    app = FastAPI()
    model_name = args.served_name
    gen.steer = gen.engine.steer_info()  # activation steering vectors loaded by the engine (server/steering.py)

    @app.get("/v1/steering")
    async def steering():
        return gen.steer

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    def limits():
        """What a client needs to size its requests (vLLM / llama.cpp field names where they exist)."""
        e = gen.engine
        return {"max_model_len": e.max_t,                   # prompt + reply tokens of one conversation (vLLM)
                "max_num_seqs": len(e.slots),               # conversations served at once: sequence slots (vLLM)
                "max_prompt_tokens": max(e.slots) - gen.args.mtp_k - 2,  # longer prompts: 400 context_length_exceeded
                "max_output_tokens": None,                  # no server cap: a reply may fill the rest of the context
                "kv_pool_tokens": e.pool_tokens,            # KV pool of all slots together
                "kv_cache": "ram" if e.kv_in_ram else "vram"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "qengine",
                                            "input_modalities": ["text", "image"] if gen.vision is not None else ["text"],
                                            **limits()}]}

    @app.get("/props")
    async def props():  # llama.cpp-style server properties (slot count and context for tools that read /props)
        lim = limits()
        return {"model_alias": model_name, "total_slots": lim["max_num_seqs"],
                "default_generation_settings": {"n_ctx": lim["max_model_len"]},
                "modalities": {"vision": gen.vision is not None},
                "metrics_port": args.metrics_port, **lim}   # live monitoring port (/api/stats), 0 = off

    # ---------------- OpenAI chat completions ----------------
    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        t_in = time.time()  # [ink-srv-gap] QE_REQ_PROF=1: one line of server-side stamps per streamed request
        body = await req.json()
        t_json = time.time()
        bad = request_shape_error(body)
        if bad:
            return JSONResponse({"error": {"message": bad[1], "type": "invalid_request_error", "param": bad[0]}},
                                status_code=400)
        gen.param_notes.check(body)  # [feature-matrix] unsupported parameters: one log line each per process
        bad = bad_param(body, "stop", ("max_completion_tokens", "max_tokens", "temperature", "top_p", "top_k", "min_p", "seed",
                                       "presence_penalty", "frequency_penalty"))
        if bad:
            return JSONResponse({"error": {"message": bad[1], "type": "invalid_request_error", "param": bad[0]}}, status_code=400)
        logprobs = logprob_params(body)
        if isinstance(logprobs, str) or (logprobs is not None and not gen.engine.has("logprobs")):
            msg = logprobs if isinstance(logprobs, str) else "logprobs: this engine library has no logits read-back"
            return JSONResponse({"error": {"message": msg, "type": "invalid_request_error", "param": "top_logprobs"
                                           if "top_logprobs" in msg else "logprobs"}}, status_code=400)
        tools = body.get("tools")
        kw = gen.effort_rules.client_effort(template_kwargs(body, args.think, args.reasoning_effort))
        try:
            budget, budget_src = budget_source(body, kw, gen.budget_map, gen.effort_rules)
        except BudgetError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error",
                                           "param": "thinking_budget"}}, status_code=400)
        try:
            grammar = structured(gen, body, kw)
        except GrammarError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error",
                                           "param": "response_format"}}, status_code=400)
        try:
            tchoice = tool_choice_of(gen, body, tools, grammar)
        except tool_choice.ToolChoiceError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error",
                                           "param": "tool_choice"}}, status_code=400)
        no_calls = tchoice is not None and tchoice["mode"] == tool_choice.NONE
        try:
            steer = resolve_steering(gen.steer, body)
        except ValueError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error",
                                           "param": "steering"}}, status_code=400)
        t_pr0 = time.time()
        try:
            ids, aff = gen.prompt(openai_messages(body["messages"]), tools, kw, body.get("metadata"), track=True,
                                  guard_on=body.get("loop_guard") is not False, guard_think=grammar is None)
        except InputError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error", "param": "messages"}},
                                status_code=400)
        t_pr1 = time.time()
        if gen.too_long(ids):
            # OpenAI's wording: harnesses detect an overlong prompt by it (verifiers / AutomationBench end the rollout
            # cleanly and score it instead of aborting it, LiteLLM raises ContextWindowExceededError)
            cap = max(gen.engine.slots) - gen.args.mtp_k - 2
            return JSONResponse({"error": {"message": f"This model's maximum context length is {cap} tokens. However, "
                                                      f"your messages resulted in {len(ids)} tokens ({gen.too_long(ids)}).",
                                           "type": "invalid_request_error",
                                           "code": "context_length_exceeded"}}, status_code=400)
        if body.get("loop_guard") is False:  # per request off-switch of the cross-turn loop guard
            ids.guard = None
        budget, budget_src = guard_budget(ids, budget, budget_src)
        stop = body.get("stop")
        params = {"temperature": body.get("temperature"), "top_p": body.get("top_p"),
                  "top_k": body.get("top_k"), "min_p": body.get("min_p"), "seed": body.get("seed"),
                  "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
                  "stop": [stop] if isinstance(stop, str) else stop,
                  "ignore_eos": bool(body.get("ignore_eos")),  # benchmarks (vllm bench serve --ignore-eos)
                  "grammar": grammar, "tool_choice": tchoice, "steering_sel": steer, "think_budget": budget,
                  "think_budget_src": budget_src,
                  "presence_penalty": body.get("presence_penalty"), "frequency_penalty": body.get("frequency_penalty"),
                  "thinking": reply_thinking(gen, ids, kw),
                  "cache_prompt": body.get("cache_prompt", True) is not False, "has_tools": bool(tools)}
        if logprobs is not None:
            params["logprobs"] = logprobs
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        parser = PARSER(reply_thinking(gen, ids, kw))
        hide_rs = reasoning_excluded(body)  # reasoning.exclude: generated, not returned

        def tool_calls_out(calls):
            return [{"index": i, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for i, c in enumerate(calls)]

        dsv4 = getattr(getattr(gen, "profile", None), "is_dsv4", False)  # [dsv4-rel8-timings]

        def usage(st):
            u = {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
                 "total_tokens": st["prompt_tokens"] + st["completion_tokens"],
                 "prompt_tokens_details": {"cached_tokens": st["cached_tokens"]}}
            if st.get("think"):
                u["completion_tokens_details"] = {"reasoning_tokens": st["think"]["tokens"]}
            return u

        if body.get("stream"):
            async def sse():
                lp_buf = []  # logprobs entries that wait for a content chunk (the parser may hold text back)

                def chunk(delta, finish=None, extra=None):
                    d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if logprobs is not None:
                        d["choices"][0]["logprobs"] = None
                        if lp_buf and ("content" in delta and delta["content"] or finish):
                            d["choices"][0]["logprobs"] = {"content": list(lp_buf)}
                            lp_buf.clear()
                    if extra:
                        d.update(extra)
                    return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                started = False
                text_seen, t_role = [False], t_in
                sse_s = [0.0, 0]  # QE_STEP_PROF: event-loop time of parser + chunk JSON, chunks
                async for item in fault_events(stream_bridge(gen, ids, params, aff, req, SSE_KEEPALIVE_S)):
                    if item is KEEPALIVE or (type(item) in (str, LpText) and keepalive_due(last_w)):
                        # (an SSE comment: clients skip it; the role chunk still comes with the first token; also while
                        # the parser holds every item back: a tool call, hidden reasoning)
                        if keepalive_due(last_w):
                            yield ": keep-alive\n\n"
                        if item is KEEPALIVE:
                            continue
                    if isinstance(item, SequenceFault):  # the engine failed this request: an error event, then the end
                        if bad_input(item):
                            yield "data: " + json.dumps({"error": {"message": str(item), "type": "invalid_request_error",
                                                                   "code": 400}}) + "\n\n"
                        else:
                            yield "data: " + json.dumps({"error": {"message": f"engine error: {item}",
                                                                   "type": "server_error", "code": 500}}) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    if not started:  # role chunk with the first token: clients measure TTFT on it
                        started = True
                        t_role = time.time()
                        yield chunk({"role": "assistant", "content": ""})
                    if isinstance(item, dict):
                        st = item
                        for kind, t in parser.feed("", final=True, promote=st.get("promote_call", False)):
                            if kind == "reasoning" and hide_rs:
                                continue
                            yield chunk({"reasoning_content" if kind == "reasoning" else "content": t})
                        calls, raw_tool = tool_calls_or_text(parser, tools, tchoice, no_calls)
                        if raw_tool:
                            yield chunk({"content": raw_tool})
                        if calls:
                            yield chunk({"tool_calls": tool_calls_out(calls)})
                        finish = "tool_calls" if calls else st["finish"]
                        lp_buf.extend(st.get("logprobs", ()))
                        yield chunk({}, finish)
                        # usage in its own chunk with empty choices (OpenAI stream_options.include_usage)
                        last = {"id": rid, "object": "chat.completion.chunk", "created": created,
                                "model": model_name, "choices": [], "usage": usage(st)}
                        if dsv4:
                            last["timings"] = timings(st)
                        yield "data: " + json.dumps(last) + "\n\n"
                        yield "data: [DONE]\n\n"
                        log_stats("chat", st)
                        if STEP_PROF and sse_s[1]:
                            print(f"[prof] sse: {sse_s[1]} chunks, parser + JSON {sse_s[0] / sse_s[1] * 1e6:.0f} us/chunk "
                                  f"(event-loop thread)", flush=True)
                        return
                    if STEP_PROF:
                        a0 = _pc()
                    if type(item) is LpText:
                        lp_buf.extend(item.lp)
                    for kind, t in parser.feed(item):
                        if kind == "reasoning" and hide_rs:
                            continue
                        c = chunk({"reasoning_content" if kind == "reasoning" else "content": t})
                        if STEP_PROF:
                            sse_s[0] += _pc() - a0
                            sse_s[1] += 1
                        if REQ_PROF and not text_seen[0]:
                            text_seen[0] = True
                            print(f"[reqprof] t_in {t_in:.6f}: json {1e3 * (t_json - t_in):.1f} ms, prompt (template + "
                                  f"tokenize, {len(ids)} tokens) {1e3 * (t_pr1 - t_pr0):.1f} ms (at +{1e3 * (t_pr0 - t_in):.1f}), "
                                  f"role chunk +{1e3 * (t_role - t_in):.1f} ms, first text chunk +{1e3 * (time.time() - t_in):.1f} ms",
                                  flush=True)
                        yield c
                        if STEP_PROF:
                            a0 = _pc()
            last_w = [time.monotonic()]
            return StreamingResponse(stamped(sse(), last_w), media_type="text/event-stream")

        r = await complete(gen, ids, params, aff, reply_thinking(gen, ids, kw), tools, tchoice, req)
        if isinstance(r, Cancelled):  # [srv-disconnect] the client left (nobody reads it; the access log says 499)
            return JSONResponse({"error": {"message": str(r), "type": "client_closed_request"}}, status_code=499)
        if isinstance(r, SequenceFault):
            if bad_input(r):
                return JSONResponse({"error": {"message": str(r), "type": "invalid_request_error"}}, status_code=400)
            return JSONResponse({"error": {"message": f"engine error: {r}", "type": "server_error"}},
                                status_code=500)
        parser, parts, st, calls = r
        raw = PARSER is DsmlParser  # DSV4 renders the history verbatim: the reply as generated (review M3)
        msg = {"role": "assistant", "content": (parts["content"] if raw else parts["content"].strip()) or (None if calls else "")}
        if logprobs is not None and msg["content"]:  # as streamed: the entries join to exactly this text
            msg["content"] = parts["content"]
        if parts["reasoning"] and not hide_rs:
            msg["reasoning_content"] = parts["reasoning"] if raw else parts["reasoning"].strip()
        if calls:
            msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in tool_calls_out(calls)]
        log_stats("chat", st)
        choice = {"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else st["finish"]}
        if logprobs is not None:
            choice["logprobs"] = {"content": parts["logprobs"]}
        out = {"id": rid, "object": "chat.completion", "created": created, "model": model_name,
               "choices": [choice], "usage": usage(st)}
        if dsv4:
            out["timings"] = timings(st)
        return out

    # ---------------- Anthropic messages ----------------
    @app.post("/v1/messages/count_tokens")
    async def count_tokens(req: Request):
        body = await req.json()
        bad = request_shape_error(body, anthropic=True)
        if bad:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": bad[1]}},
                                status_code=400)
        try:
            kw = gen.effort_rules.client_effort(template_kwargs(body, args.think, args.reasoning_effort,
                                                                anthropic=True))
            request_budget(body, kw)  # (budget 0: rendered without thinking, as the request would be)
            ids, _ = gen.prompt(anthropic_to_messages(body), anthropic_tools(body.get("tools")), kw)
        except (InputError, BudgetError) as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        return {"input_tokens": len(ids)}

    @app.post("/v1/messages")
    async def messages(req: Request):
        body = await req.json()
        bad = request_shape_error(body, anthropic=True)
        if bad:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": bad[1]}},
                                status_code=400)
        gen.param_notes.check(body)  # [feature-matrix]
        bad = bad_param(body, "stop_sequences", ("max_tokens", "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty"))
        if bad:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": bad[1]}}, status_code=400)
        if env_str("QE_DUMP_PROMPTS"):  # the request as received (prompt() writes the rendered text next to it)
            os.makedirs(env_str("QE_DUMP_PROMPTS"), exist_ok=True)
            with open(os.path.join(env_str("QE_DUMP_PROMPTS"), f"{time.time():.3f}.json"), "w") as f:
                json.dump(body, f, ensure_ascii=False, indent=1)
        tools = anthropic_tools(body.get("tools"))
        kw = gen.effort_rules.client_effort(template_kwargs(body, args.think, args.reasoning_effort, anthropic=True))
        try:
            budget, budget_src = budget_source(body, kw, gen.budget_map, gen.effort_rules)
        except BudgetError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        try:
            grammar = structured(gen, body, kw)
        except GrammarError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        try:
            tchoice = tool_choice_of(gen, body, tools, grammar)
        except tool_choice.ToolChoiceError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        no_calls = tchoice is not None and tchoice["mode"] == tool_choice.NONE
        try:
            steer = resolve_steering(gen.steer, body)
        except ValueError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        try:
            msgs = anthropic_to_messages(body)
            prev = dict(msgs[-1]) if msgs else None  # before prompt() rewrites image parts
            ids, aff = gen.prompt(msgs, tools, kw, body.get("metadata"), track=True,
                                  guard_on=body.get("loop_guard") is not False, guard_think=grammar is None)
        except InputError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        if gen.too_long(ids):
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": gen.too_long(ids)}},
                                status_code=400)
        if body.get("loop_guard") is False:  # per request off-switch of the cross-turn loop guard
            ids.guard = None
        budget, budget_src = guard_budget(ids, budget, budget_src)
        stop = body.get("stop_sequences")
        params = {"temperature": body.get("temperature"), "top_p": body.get("top_p"),
                  "top_k": body.get("top_k"), "max_tokens": body.get("max_tokens"),
                  "stop": [stop] if isinstance(stop, str) else stop, "grammar": grammar, "tool_choice": tchoice,
                  "steering_sel": steer, "think_budget": budget,
                  "think_budget_src": budget_src,
                  "presence_penalty": body.get("presence_penalty"), "frequency_penalty": body.get("frequency_penalty"),
                  "thinking": reply_thinking(gen, ids, kw),
                  # [kb-server] as /v1/chat/completions (shared_server review L9; has_tools: DSV4's stop trim with tools)
                  "cache_prompt": body.get("cache_prompt", True) is not False, "has_tools": bool(body.get("tools"))}
        mid = "msg_" + uuid.uuid4().hex[:24]
        parser = PARSER(reply_thinking(gen, ids, kw))
        stop_map = {"stop": "end_turn", "length": "max_tokens"}

        def usage(st):
            return {"input_tokens": st["prompt_tokens"] - st["cached_tokens"],
                    "cache_read_input_tokens": st["cached_tokens"], "cache_creation_input_tokens": 0,
                    "output_tokens": st["completion_tokens"]}

        if body.get("stream"):
            async def sse():
                def ev(name, data):
                    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                yield ev("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model_name, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": len(ids), "output_tokens": 0}}})
                state = {"idx": -1, "kind": None}
                seen = {"reasoning": "", "content": ""}

                def emit(kind, text):
                    seen[kind] += text
                    out = []
                    if state["kind"] != kind:
                        if state["kind"] is not None:
                            if state["kind"] == "reasoning":
                                out.append(ev("content_block_delta", {"type": "content_block_delta", "index": state["idx"],
                                                                      "delta": {"type": "signature_delta", "signature": ""}}))
                            out.append(ev("content_block_stop", {"type": "content_block_stop", "index": state["idx"]}))
                        state["idx"] += 1
                        state["kind"] = kind
                        block = {"type": "thinking", "thinking": "", "signature": ""} if kind == "reasoning" \
                            else {"type": "text", "text": ""}
                        out.append(ev("content_block_start", {"type": "content_block_start", "index": state["idx"],
                                                              "content_block": block}))
                    delta = {"type": "thinking_delta", "thinking": text} if kind == "reasoning" \
                        else {"type": "text_delta", "text": text}
                    out.append(ev("content_block_delta", {"type": "content_block_delta", "index": state["idx"],
                                                          "delta": delta}))
                    return out

                async for item in fault_events(stream_bridge(gen, ids, params, aff, req, SSE_KEEPALIVE_S)):
                    if item is KEEPALIVE or (type(item) in (str, LpText) and keepalive_due(last_w)):
                        if keepalive_due(last_w):  # (Anthropic's own keep-alive event; also while text is held back)
                            yield ev("ping", {"type": "ping"})
                        if item is KEEPALIVE:
                            continue
                    if isinstance(item, SequenceFault):  # the engine failed this request: an error event, then the end
                        yield ev("error", {"type": "error", "error": {
                            "type": "invalid_request_error" if bad_input(item) else "api_error",
                            "message": str(item) if bad_input(item) else f"engine error: {item}"}})
                        return
                    if isinstance(item, dict):
                        st = item
                        for kind, t in parser.feed("", final=True, promote=st.get("promote_call", False)):
                            for x in emit(kind, t):
                                yield x
                        calls, raw_tool = tool_calls_or_text(parser, tools, tchoice, no_calls)
                        if raw_tool:
                            for x in emit("content", raw_tool):
                                yield x
                        if state["kind"] is not None:
                            yield ev("content_block_stop", {"type": "content_block_stop", "index": state["idx"]})
                        ids_out = ["toolu_" + uuid.uuid4().hex[:24] for _ in calls]
                        REASONING.put(prev, seen["content"], ids_out, seen["reasoning"])
                        for c, tid in zip(calls, ids_out):
                            state["idx"] += 1
                            yield ev("content_block_start", {"type": "content_block_start", "index": state["idx"],
                                                             "content_block": {"type": "tool_use", "id": tid,
                                                                               "name": c["name"], "input": {}}})
                            yield ev("content_block_delta", {"type": "content_block_delta", "index": state["idx"],
                                                             "delta": {"type": "input_json_delta",
                                                                       "partial_json": json.dumps(c["arguments"], ensure_ascii=False)}})
                            yield ev("content_block_stop", {"type": "content_block_stop", "index": state["idx"]})
                        reason = "tool_use" if calls else stop_map.get(st["finish"], "end_turn")
                        md = {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                              "usage": usage(st)}
                        if getattr(getattr(gen, "profile", None), "is_dsv4", False):  # [dsv4-rel8-timings] (review L10)
                            md["timings"] = timings(st)
                        yield ev("message_delta", md)
                        yield ev("message_stop", {"type": "message_stop"})
                        log_stats("messages", st)
                        return
                    for kind, t in parser.feed(item):
                        for x in emit(kind, t):
                            yield x
            last_w = [time.monotonic()]
            return StreamingResponse(stamped(sse(), last_w), media_type="text/event-stream")

        r = await complete(gen, ids, params, aff, reply_thinking(gen, ids, kw), tools, tchoice, req)
        if isinstance(r, Cancelled):  # [srv-disconnect] the client left
            return JSONResponse({"type": "error", "error": {"type": "client_closed_request", "message": str(r)}},
                                status_code=499)
        if isinstance(r, SequenceFault):
            if bad_input(r):
                return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(r)}},
                                    status_code=400)
            return JSONResponse({"type": "error", "error": {"type": "api_error",
                                                            "message": f"engine error: {r}"}}, status_code=500)
        parser, parts, st, calls = r
        content = []
        if parts["reasoning"].strip():
            content.append({"type": "thinking", "thinking": parts["reasoning"].strip(), "signature": ""})
        if parts["content"].strip():
            content.append({"type": "text", "text": parts["content"].strip()})
        ids_out = ["toolu_" + uuid.uuid4().hex[:24] for _ in calls]
        REASONING.put(prev, parts["content"].strip(), ids_out, parts["reasoning"])
        for c, tid in zip(calls, ids_out):
            content.append({"type": "tool_use", "id": tid, "name": c["name"], "input": c["arguments"]})
        log_stats("messages", st)
        out = {"id": mid, "type": "message", "role": "assistant", "model": model_name, "content": content,
               "stop_reason": "tool_use" if calls else stop_map.get(st["finish"], "end_turn"),
               "stop_sequence": None, "usage": usage(st)}
        if getattr(getattr(gen, "profile", None), "is_dsv4", False):  # [dsv4-rel8-timings] (review L10)
            out["timings"] = timings(st)
        return out

    return app


INK_CTX_MAX = 1 << 20  # the Inkling engine's own cap (qe_create)


def model_ctx_limit(v):
    """[ink-ctx-cap] review L6: a config's context limit as a sane int (None: none / a "no limit" sentinel above the
    engine's cap counts as the cap, a non-number as none)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not v == v or v < 1024:
        return None
    return int(min(v, INK_CTX_MAX))


def bg_rearm_after_reply(profile, env, args=None):
    """[ink-ctx-arch] idle archiving re-armed after every finished reply: Inkling and ([gptoss-rearm]) gpt-oss (their
    archive is host RAM only; the 27B / Flash-Next / GLM / DSV4 keep the 1.3.5 rule: re-arm only after a reply without
    decode steps). Off-switches: INK_BG_REARM=0 (Inkling), GPTOSS_BG_REARM=0 (gpt-oss); 0 / false / off / no
    (any case) mean off ([gptoss-rearm] review L3).
    [disk-eager] the 27B and Flash-Next with --disk-cache and the opt-in --disk-eager: re-armed too, so the idle archiver
    copies the conversation after each reply and its file is written right away (the 27B writes a file whenever an
    archive is complete; Flash-Next FLASH_DISK_EAGER, set by the server)."""
    if (args is not None and getattr(args, "disk_eager", False) and getattr(args, "disk_cache", "")
            and getattr(profile, "family", None) in ("qwen38-27b", "qwen38-flash-next")):  # (model_profile Q27 / QFN)
        return True
    if getattr(profile, "is_inkling", False):
        key = "INK_BG_REARM"
    elif getattr(profile, "is_gptoss", False):
        key = "GPTOSS_BG_REARM"
    else:
        return False
    return (env.get(key) or "1").strip().lower() not in ("0", "false", "off", "no")


def _c_atoi(v):
    """C atoi: the leading integer of v (else 0), as the Inkling engine parses INK_DENSE8"""
    m = re.match(r"\s*[+-]?\d+", v or "")
    return int(m.group()) if m else 0


def ink_dense8_lines(model, env):
    """[ink-pf8] review L3 / review2 L2: the start-up line when INT8 dense weights are asked for explicitly (INK_DENSE8 != 0
    by the engine's atoi, or INK_DENSE8_RTN) but will not come from dense8.safetensors. Engine rules (ink.cu load):
    INK_DENSE8 set and atoi 0 = g15; INK_DENSE8_RTN (non-empty, not starting with 0) = RTN at load; a set INK_DENSE8_FILE
    that does not exist stops the start; else the file next to the checkpoint, RTN at load when it is missing."""
    e8 = env.get("INK_DENSE8")
    rtn = env.get("INK_DENSE8_RTN") or ""
    rtn_on = bool(rtn) and not rtn.startswith("0")
    if (e8 and not _c_atoi(e8)) or not ((e8 and _c_atoi(e8)) or rtn_on) or rtn_on:
        return []  # g15 / the engine's default (its own line) / RTN asked for by name
    f = env.get("INK_DENSE8_FILE")
    if f:
        return [] if os.path.exists(f) else [f"[settings] INK_DENSE8_FILE={f} does not exist: the Inkling engine will not start "
                                             "(unset it to use dense8.safetensors next to the checkpoint)"]
    d8f = os.path.join(model, "dense8.safetensors")
    if os.path.exists(d8f):
        return []
    return [f"[settings] INT8 dense weights asked for but {d8f} is missing: RTN INT8 at load (lower quality than GPTQ: "
            "set-2 KL 0.00589 vs 0.00539); put dense8.safetensors next to the checkpoint, or --dense-weights g15 for the "
            "original weights"]


def inkling_devices(given, env):
    """[ink-ctx-dev] Inkling's cards (indices within CUDA_VISIBLE_DEVICES) and start-up lines. given: --devices from the
    command line (None / "": not given); env: os.environ. One card unless asked: an explicit --devices, QE_DEVICES
    through server/run.sh (mapped to --devices), or QE_DEVICES on a direct start (the engine's old way, indices).
    run.sh's own every-visible-GPU list (QE_RUNSH_DEVICES=all) is not a request: tensor parallelism needs P2P and two
    cards are the measured best, so it is never implicit. Raises ValueError for a malformed or repeated list."""
    src = (env.get("QE_RUNSH_DEVICES") or "").strip()
    q = (env.get("QE_DEVICES") or "").strip()
    given = (given or "").strip() or None
    lines = []
    if given is not None and src != "all":
        dev = given
        if q and not src:  # a direct start: QE_DEVICES would have been the engine's own list
            lines.append(f"[settings] QE_DEVICES={q} is set but has no effect for Inkling-2-Mini (--devices {given} wins)")
    elif q and not src:
        dev = q
    else:
        dev = "0"
        if src == "all" and given and given != "0":
            lines.append(f"[ink] one card (index 0 of the visible GPUs); tensor parallel: QE_DEVICES=a,b server/run.sh or --devices 0,1")
    if not re.fullmatch(r"\d+(,\d+)*", dev.replace(" ", "")):
        raise ValueError(f"--devices / QE_DEVICES '{dev}': a comma list of GPU indices (0,1)")
    ids = [int(x) for x in dev.replace(" ", "").split(",")]
    if len(set(ids)) < len(ids) and (env.get("INK_VIRTUAL_TP") or "").strip() in ("", "0"):
        raise ValueError(f"--devices '{dev}' repeats a card (several ranks on one card are an engine test: INK_VIRTUAL_TP=1)")
    vis = [x for x in (env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if x.strip()]
    if vis and max(ids) >= len(vis):
        raise ValueError(f"--devices '{dev}': index {max(ids)} but only {len(vis)} visible GPU(s) (CUDA_VISIBLE_DEVICES={','.join(vis)})")
    return ",".join(map(str, ids)), lines


def resolve_model_args(args):
    """Fill the per-model defaults (None on the command line) from the model profile."""
    if args.model in ("glm", "glm53-flash"):  # the GLM engine's default checkpoint (model_profile.py default_dir)
        from model_profile import FAMILIES
        args.model = os.environ.get("QE_GLM_MODEL") or FAMILIES["glm53-flash"]["default_dir"]
        if not os.path.isdir(args.model):  # [glm-rel-host] a clear message instead of a load error
            sys.exit(f"GLM-5.3-Flash checkpoint not found at {args.model}. Download the QEngine quant "
                     f"({FAMILIES['glm53-flash']['hf_repo']}, e.g. `hf download {FAMILIES['glm53-flash']['hf_repo']} "
                     f"--local-dir <dir>`) and start with --model <dir> (or set QE_GLM_MODEL=<dir>).")
    prof = args.profile = Profile(args.model)
    d = prof.defaults
    if prof.family == "dsv4":  # [shared-srv] the DSV4 server's own defaults (dsv4_app): context, chunk, library, stats
        if not getattr(args, "max_ctx_given", True):
            if d["max_ctx"] == "auto":  # [dsv4-rel8-ctxdef] 262144 with the lazy and the static reservation (user 11-19): the engine knows at load
                args.max_ctx, args.ctx_static = d["max_ctx_lazy"], d["max_ctx_static"]
            else:
                args.max_ctx = d["max_ctx"]
        if args.prefill_chunk is None:
            args.prefill_chunk = d["prefill_chunk"]
        if not args.engine_lib:
            args.engine_lib = os.environ.get("QE_DSV4_LIB") or os.path.join(ROOT, d["lib"])
        if args.stats_file == STATS_DEFAULT:
            args.stats_file = d["stats_file"]
    args.ink_lines = []
    if d.get("max_ctx") == "model":  # [ink-ctx-cap] Inkling: the model's limit (capped at the KV pool)
        mx = model_ctx_limit(prof.max_position)
        if not getattr(args, "max_ctx_given", True):
            args.max_ctx = mx or args.max_ctx
            args.max_ctx_model = True  # (review L1: the start-up context line says nothing is "lowered")
        elif mx and args.max_ctx > mx:  # (review L2)
            args.ink_lines.append(f"[settings] --max-ctx {args.max_ctx} is above what Inkling-2-Mini serves: "
                                  f"capped at its {mx:,} tokens (and the KV pool)")
            args.max_ctx = mx
    if getattr(args, "prefill_turns", "auto") == "auto":
        args.prefill_turns = d.get("prefill_turns", "time")
    if args.served_name is None:
        args.served_name = d["served_name"]
    if args.mtp_k is None:
        args.mtp_k = d["mtp_k"]
    if prof.family == "inkling":  # [ink-ctx-dev] review H1 / M1 / M2 / L4 / L5
        try:
            args.devices, lines = inkling_devices(args.devices if "devices" in getattr(args, "given", {}) else None, os.environ)
        except ValueError as ex:
            sys.exit(f"Inkling-2-Mini: {ex}")
        args.ink_lines += lines
        # [ink-pf8] --dense-weights (any card count): the flag wins over INK_DENSE8
        dw = getattr(args, "dense_weights", None)
        if dw:
            os.environ["INK_DENSE8"] = "1" if dw == "int8" else "0"
        for ln in ink_dense8_lines(args.model, os.environ):
            args.ink_lines.append(ln)
    if args.devices is None:
        vis = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        args.devices = ",".join(str(i) for i in range(len(vis))) if vis else "0,1,2"
    if args.split is None:
        # the profile's split is per GPU (Flash-Next: 3 values); another GPU count uses the engine's default
        sp = d["split"] or ""
        args.split = sp if len([x for x in sp.split(",") if x]) in (0, len(args.devices.split(","))) else ""
    if args.prefill_chunk_idle is None:
        args.prefill_chunk_idle = d.get("prefill_chunk_idle", 0)
    args.prefill_chunk_auto = args.prefill_chunk is None
    if args.prefill_chunk is None:
        args.prefill_chunk = 2048
    for k in ("drop_empty_turns", "merge_system"):
        v = getattr(args, k)
        setattr(args, k, bool(d[k]) if v is None else bool(v))
    if not args.engine_lib and not env_str("QE_LIB"):
        lib = os.path.join(ROOT, d["lib"])
        if not os.path.exists(lib) and prof.family == "inkling":
            sys.exit(f"Inkling engine library not built: {lib} is missing (run OUT={os.path.normpath(os.path.dirname(lib))} "
                     f"{os.path.normpath(os.path.join(ROOT, 'src', 'inkling', 'build.sh'))})")
        if not os.path.exists(lib) and prof.family == "gptoss":
            # [gptoss-rel1-lib] the 27B library cannot run this model: stop with the reason instead of a later failure
            sys.exit(f"gpt-oss engine library not built: {lib} is missing (run src/gptoss/build.sh)")
        args.engine_lib = lib if os.path.exists(lib) else os.path.join(ROOT, "build", "libqengine.so")
    if d.get("own_template"):  # GLM: the model's own template unless one was given explicitly
        own = os.path.join(args.model, "chat_template.jinja")
        if os.path.exists(own) and (args.template is None or os.path.basename(args.template) == "qwen38_syssafe.jinja"):
            args.template = own
    global TOOL_FORMAT, PARSER
    TOOL_FORMAT = d.get("tool_format", "qwen")
    if TOOL_FORMAT == "harmony":
        from harmony import HarmonyParser
        PARSER = HarmonyParser
    elif TOOL_FORMAT == "inkling":
        from inkling import InklingParser
        PARSER = InklingParser
    else:
        PARSER = DsmlParser if TOOL_FORMAT == "dsml" else Parser
    if args.vision and not d["vision"]:  # (logged by the feature table: ignored_settings)
        args.vision = 0
    return args


STATS_DEFAULT = "~/.cache/qengine/stats.json"


def dsv4_flags(ap):
    """[shared-srv] the DSV4 engine's flags (dsv4_app's, same names and defaults); other models ignore them"""
    g = ap.add_argument_group("DSV4 (DeepSeek-V4.1-Flash GGUF)")
    g.add_argument("--engine-args", default="", help="DSV4: extra engine flags (--dspark-dir, --profile, --cpu-threads ...)")
    g.add_argument("--cache-dir", default=None, help="DSV4: the engine's cache directory")
    g.add_argument("--no-dspark", action="store_true", help="DSV4 off-switch: plain decode without the DSpark drafter")
    g.add_argument("--archive-gib", type=float, default=32.0,
                   help="DSV4: host RAM for idle conversations (their KV state, LRU; it shrinks when MemAvailable runs low); "
                        "0 = off: a different conversation drops the held one")
    g.add_argument("--kv-pool-mib", type=int, default=0,
                   help="DSV4: VRAM per card for the compressed K/V of idle conversations beyond one --max-ctx sequence "
                        "(default 0: idle ones use what the active conversation does not)")
    g.add_argument("--tier-reset", action="store_true",
                   help="DSV4: greedy text independent of earlier unrelated requests: the expert tier back to its post-load "
                        "state after each request (in the background; -2..-10 %% tok/s); requests then one at a time")
    g.add_argument("--route-margin", type=float, default=None,
                   help="DSV4 opt-in, default off. APPROXIMATES the routing: a cold expert choice whose router score is within "
                        "this margin of a VRAM-resident expert is replaced by that expert (measured 11-17: NLL 1.623 -> 1.641 "
                        "at 0.05, +6..+11 %% speed at 256 tokens); <= 0 = off (also over DSV4_ROUTE_MARGIN)")
    g.add_argument("--drop-w", type=float, default=None,
                   help="DSV4: adaptive top-k at decode (lossy); default 0 = exact routing; also DSV4_DROP_W")
    g.add_argument("--slice-passes", type=int, default=16,
                   help="DSV4: decode passes of one request before the next one in service takes its turn")
    g.add_argument("--no-prefix-share", action="store_true",
                   help="DSV4 off-switch: no shared prompt-prefix entries (by default the system message of a second "
                        "prompt that shares it is kept once and reused by later prompts)")
    g.add_argument("--serial", action="store_true",
                   help="DSV4 off-switch: one request at a time to its end (also with --tier-reset or --archive-gib 0)")
    g.add_argument("--prefill-turns", choices=("auto", "time", "slices"), default="auto",
                   help="prompt chunks beside decoding requests: slices = every decoder gets one --slice-passes slice "
                        "between two chunks (DSV4's default, as its old server); time = --prefill-share of the time; auto = "
                        "the model profile's (other models: time)")
    g.add_argument("--max-active", type=int, default=8,
                   help="DSV4: requests in service at once (time-sliced; more wait in the queue)")


def given_flags(ap, argv=None):
    """{dest: value} of the flags given on the command line (argparse's own parse with every default suppressed, so
    abbreviations and --x=v count as argparse reads them)."""
    saved = [(a, a.default) for a in ap._actions]
    try:
        for a, _ in saved:
            a.default = argparse.SUPPRESS
        ns, _ = ap.parse_known_args(argv)
    except SystemExit:
        return {}
    finally:
        for a, d in saved:
            a.default = d
    return {k: v for k, v in vars(ns).items() if k != "help"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    dsv4_flags(ap)
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen3.8-27B-INT4"))
    default_tmpl = os.path.join(ROOT, "server/templates/qwen38_syssafe.jinja")
    ap.add_argument("--template", default=default_tmpl if os.path.exists(default_tmpl) else None,
                    help="chat template (default: server/templates/qwen38_syssafe.jinja, else the model's own)")
    ap.add_argument("--devices", default=None,
                    help="GPU indices within CUDA_VISIBLE_DEVICES (default: every visible GPU, else 0,1,2); "
                         "Flash-Next: 2 or 3 cards: tp; 4 cards: tp with peer access between all of them, else pp; "
                         "5+ cards: pp (pipeline stages; see the README)")
    def ctx_arg(v):  # 262144, 256K, 1M (K = 1024 tokens)
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", v)
        if not m:
            raise argparse.ArgumentTypeError("a token count (262144, 512K, 1M)")
        return int(float(m.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[m.group(2).lower()])
    ap.add_argument("--max-ctx", type=ctx_arg, default=262144,
                    help="context per conversation (default 262144; Inkling-2-Mini: the model's 1,048,576); above 262144 (up to 1M) turns on YaRN RoPE "
                         "scaling, factor context / 262144 (27B: QE_YARN, Flash-Next: FLASH_YARN, and needs FLASH_KV=ram). "
                         "Never above the KV pool: a smaller pool lowers it and the server starts anyway (start-up line "
                         "'context: requested ..., KV pool ..., served ...')")
    ap.add_argument("--dense-weights", choices=("int8", "g15"), default=None,
                    help="Inkling-2-Mini: form of the dense attention weights, on any card count. "
                         "g15 = original weights (lossless, bit-exact to the checkpoint; set-2 KL vs the FP32 reference "
                         "0.00495); int8 = GPTQ INT8 g128 from dense8.safetensors next to the checkpoint: faster decode, "
                         "small quality loss (set-2 KL 0.00539 decode form, 0.00544 prefill form on 2 cards; one card "
                         "0.00538 / 0.00558); without that file int8 is RTN INT8 at load (lower quality, KL 0.00589; logged "
                         "at start). Default: int8 on any card count when dense8.safetensors is present, else g15. "
                         "Env: INK_DENSE8=1 (int8) / 0 (g15); the flag wins")
    ap.add_argument("--mtp-k", type=int, default=None,
                    help="max MTP drafts per step (adaptive; default per model: 27B 7, Flash-Next 3)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default when request has none, both modes (model card: thinking 1.0, non-thinking 0.7)")
    ap.add_argument("--top-p", type=float, default=None, help="default when request has none, both modes (model card: thinking 0.95, non-thinking 0.80)")
    ap.add_argument("--top-k", type=int, default=None, help="default when request has none, both modes (model card: 20)")
    ap.add_argument("--min-p", type=float, default=None, help="default when request has none, both modes (model card: 0)")
    ap.add_argument("--mtp-file", default=None,
                    help="MTP head weights for the engine (sets $QE_MTP_FILE; default: the engine's own choice, "
                         "for 27B model_mtp_ft.safetensors when present, else downloaded once from the latest "
                         "release into ~/.cache/qengine/assets; QE_FETCH_ASSETS=0: no download)")
    ap.add_argument("--lsa", nargs="?", const="auto", default=None, metavar="MAP",
                    help="27B: Latent Sparse Attention for long contexts (opt-in, lossy). --lsa (or QE_LSA=auto): "
                         "the released D-LSA index map, downloaded once (pinned URL + sha256), converted and "
                         "cached in $QE_LSA_DIR / $QE_DISK_CACHE/lsa / ~/.cache/qengine/lsa; offline without a cached "
                         "map the server starts with full attention. --lsa MAP (or QE_LSA=MAP): a safetensors map, or "
                         "the released .pt (converted into the cache). Sets $QE_LSA ($QE_LSA_MODE block / dlsa: default block below "
                         "--max-ctx 1M, dlsa from 1M on). Rows at positions >= $QE_LSA_PF_MIN (prefill) / "
                         "$QE_LSA_DEC_MIN (decode) attend sparsely")
    ap.add_argument("--think", type=int, default=1, help="default enable_thinking")
    ap.add_argument("--reasoning-effort", default=env_str("QE_REASONING_EFFORT"),
                    help="default reasoning effort for requests that send none (xhigh, high = xhigh, max = xhigh, "
                         "medium, low, minimal = low, none = thinking off unless a request turns it on or names an "
                         "effort; "
                         "a request's reasoning_effort, chat_template_kwargs.reasoning_effort or reasoning.effort "
                         "wins, and a "
                         "request's none turns thinking off (as enable_thinking=false, also over its own "
                         "enable_thinking=true); OpenRouter reasoning.enabled false / true is read last; other values "
                         "render as medium); env QE_REASONING_EFFORT; default: unset = the template's own (xhigh)")
    ap.add_argument("--thinking-budget", default=env_str("QE_THINKING_BUDGET"),
                    help="default thinking budget in tokens for requests that send none (0 = off): at the budget the "
                         "server closes <think> with a short wrap-up and the model answers. One number for every "
                         "request, or a budget per reasoning effort, e.g. low:4096,medium:16384,xhigh:32768 with an "
                         "optional default:N (or *:N) for the rest; the effort is the one the prompt is rendered with "
                         "(Qwen3.8: high / max = xhigh, minimal = low, unknown = medium; none = thinking off, no "
                         "server budget; an entry < 0 = no budget for that effort). A malformed value stops the start-up. A request's thinking.budget_tokens (Anthropic) / "
                         "thinking_budget (OpenAI, also in chat_template_kwargs; aliases thinking_token_budget, "
                         "max_thinking_tokens, custom_params.thinking_budget, reasoning.max_tokens) wins, a negative "
                         "one turns it off; env QE_THINKING_BUDGET")
    ap.add_argument("--served-name", default=None,
                    help="model id in the API (default per model: qwen3.8-27b-qengine / qwen3.8-flash-next-qengine)")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json to use as is (default: the model's, Flash-Next pre-tokenizer patched in)")
    ap.add_argument("--engine-lib", default=None,
                    help="engine shared library (default: $QE_LIB, else build/libqengine.so for 27B, "
                         "build/libqflash.so for Flash-Next if present)")
    ap.add_argument("--split", default=None,
                    help="per-GPU weight split passed to qe_create (Flash-Next default 176,160,176 expert groups)")
    ap.add_argument("--drop-empty-turns", type=int, default=None,
                    help="1: drop earlier assistant turns with only reasoning from the history "
                         "(default per model: Flash-Next 1, 27B 0)")
    ap.add_argument("--merge-system", type=int, default=None,
                    help="1: merge several leading system messages into one before the template "
                         "(default per model: Flash-Next 1, 27B 0; qwen38_syssafe.jinja merges them anyway)")
    ap.add_argument("--vision", type=int, default=0,
                    help="1: load the vision encoder (images in prompts; 27B: ~1.3 GB on one GPU, less KV cache; "
                         "Flash-Next: ~0.6 GB, weights streamed from host RAM)")
    ap.add_argument("--vision-device", default="", help="torch device of the encoder (default: last GPU)")
    ap.add_argument("--max-image-pixels", type=int, default=4 * 1024 * 1024,
                    help="larger images are downscaled (1 token per 32x32 pixels)")
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="prompt chunk (tokens) between decode bursts of the running sequences "
                         "(0: a prompt is prefilled at once, the others wait; default 2048, Flash-Next with passes bound "
                         "by host-backed expert copies: --prefill-chunk-idle, its big batch)")
    ap.add_argument("--prefill-chunk-idle", type=int, default=None,
                    help="prompt chunk (tokens) while no other sequence decodes or waits (default per "
                         "model: Flash-Next 4096 = its big prefill batch, gpt-oss 10240 = its engine chunk, 27B 0 = --prefill-chunk)")
    ap.add_argument("--prefill-share", type=float, default=0.5,
                    help="share of the GPU time the running sequences keep while another prompt "
                         "is prefilled (0.5: they decode at ~half their speed, the prefill takes "
                         "~2x longer; with the INT8 prefill, 8 agents + a 30K prompt: TTFT 11.5 s "
                         "as with 0.3 before it, at 69 instead of 41 tok/s per agent)")
    ap.add_argument("--prefill-share-min", type=float, default=None,
                    help="queue-aware split (engines without shared prefill batches: 27B, gpt-oss, GLM): while the "
                         "prompt to prefill next has at most --prefill-short rows, passes run back to back with one decode "
                         "step between them; a longer one leaves the decoders A / (A + W) of the time (A decoding, W waiting), "
                         "within [this, --prefill-share]; over the last ~2 s the decoders never get less than this share "
                         "(default 0.10 for the 27B, = --prefill-share elsewhere; = --prefill-share: the fixed split of earlier releases)")
    ap.add_argument("--park", type=int, default=None,
                    help="requests beyond the decode seats (27B, Flash-Next on 3+ cards in TP): prefill on one of --park-reserve seats kept for prompts, "
                         "send the first token, park the reply in host memory (its KV rows and state, bit-identical resume) "
                         "and resume it, oldest first, when a decode seat frees (default 1 for those; 0: such requests "
                         "wait for a seat before their prefill, as before)")
    ap.add_argument("--park-reserve", type=int, default=None,
                    help=f"seats kept for prompts while replies are parked (--park; default {PARK_RESERVE}, Flash-Next "
                         f"{PARK_RESERVE_MULTI})")
    ap.add_argument("--prefill-short", type=int, default=4096,
                    help="rows to prefill that count as short for --prefill-share-min (also the cut-in limit of the "
                         "queue-aware split)")
    ap.add_argument("--bg-archive", type=int, default=64,
                    help="MB per GPU per step of archiving idle conversations to host memory while "
                         "the server is idle (0 = off: a conversation is archived only when its slot "
                         "is taken, on that request's path)")
    ap.add_argument("--prefill-aging", type=float, default=4000,
                    help="queue order: least prefill work first, every second of waiting counts "
                         "as this many tokens (0 = pure shortest-first)")
    ap.add_argument("--burst-wait", type=float, default=5.0,
                    help="ms: on an idle server, the first request of a burst waits until none has arrived for this long "
                         "(at most 30 ms) so the burst shares one prefill batch (0 = start at once)")
    ap.add_argument("--preempt-max", type=int, default=None,
                    help="a queued request with at most this many tokens to prefill (a tool turn "
                         "of a cached conversation) is started between the chunks of a long "
                         "prefill in progress (0 = off; default 1024, Flash-Next with --long-share > 0: 8192)")
    ap.add_argument("--preempt-min", type=int, default=8192,
                    help="only a prefill with at least this many tokens left is cut in on, and "
                         "at most once per this many tokens of its progress (27B; Flash-Next with --long-share 0)")
    ap.add_argument("--long-share", type=float, default=None,
                    help="Flash-Next shared prefill batches: a request up to --preempt-max tokens joins the next batch "
                         "at once and the longest prompt in flight keeps this share of the batch rows (default 0.5 where "
                         "passes are bound by host-backed expert copies, else 0; 0 = the old rule: cut-ins up to 1024 "
                         "tokens at most once per --preempt-min tokens of the long prompt)")
    ap.add_argument("--fill-passes", type=int, default=0,
                    help="Flash-Next shared prefill batches with --long-share > 0: 1 = the other prompts in flight take "
                         "each batch first, oldest first, and the longest prompt fills the rest of every pass, keeping at "
                         "least --long-share of its rows; 0 (default) = the per-pass credit rule (the long prompt leads "
                         "whole batches while it is owed passes)")
    ap.add_argument("--main-bonus", type=float, default=None, help=argparse.SUPPRESS)  # removed 11-12: accepted, ignored
    ap.add_argument("--pause-wait", type=float, default=PAUSE_WAIT,
                    help="a request that does not fit the KV pool pauses the most recently admitted decoding "
                         "sequences only after waiting this many seconds (and never while one is paused already)")
    ap.add_argument("--backfill-wait", type=float, default=30,
                    help="a queued request that does not fit the KV pool lets smaller ones that fit start "
                         "first until it has waited this many seconds (0 = strict order: it blocks the queue)")
    ap.add_argument("--main-agent-re", default=None,
                    help="regex for the main agent thread, searched in the system prompt and the tool names "
                         "(overrides the automatic detection; env QE_MAIN_AGENT_RE)")
    ap.add_argument("--sort-tools", type=int, default=1,
                    help="render the tools sorted by name with sorted JSON keys, so a client that reorders the same "
                         "tools between turns keeps the prompt cache (0 = the client's order)")
    ap.add_argument("--empty-retry", type=int, default=env_int("QE_EMPTY_RETRY", 2),
                    help="a non-streamed sampled reply that ends (EOS) with no text and no tool call (reasoning only, "
                         "or a tool call that does not parse) is sampled again up to N times: agent harnesses end the "
                         "task on such a turn. Never for streams, greedy (temperature 0) or seeded requests, "
                         "structured output or client stop sequences (0 = off); env QE_EMPTY_RETRY")
    ap.add_argument("--loop-guard-tokens", type=int, default=600,
                    help="end a reply whose last N+ tokens are one short fragment repeated (0 = off)")
    ap.add_argument("--presence-penalty", type=float, default=env_float("QE_PRESENCE_PENALTY", None),
                    help="presence penalty of sampled requests that send none (Claude Code's Anthropic API never "
                         "does), both modes (model card: thinking 0, non-thinking 1.5); env QE_PRESENCE_PENALTY")
    ap.add_argument("--frequency-penalty", type=float, default=env_float("QE_FREQUENCY_PENALTY", 0.0),
                    help="frequency penalty of sampled requests that send none; env QE_FREQUENCY_PENALTY")
    ap.add_argument("--loop-guard-turns", type=int, default=env_int("QE_LOOP_GUARD_TURNS", 3),
                    help="cross-turn loop guard: after N turns that repeat earlier (reply, tool result) turns of a "
                         "conversation the next turn runs with --loop-guard-penalty on the repeated reply's tokens "
                         "and no lookup drafts, again at 2N and 4N turns, then every 24 turns (0 = off, else 2 to 24); "
                         "env QE_LOOP_GUARD_TURNS")
    ap.add_argument("--loop-guard-penalty", type=float,
                    default=env_float("QE_LOOP_GUARD_PENALTY", 1.5),
                    help="presence penalty of a guarded turn (2.0 from 2N repeated turns); env QE_LOOP_GUARD_PENALTY")
    ap.add_argument("--loop-guard-think", type=int, default=env_int("QE_LOOP_GUARD_THINK", 2048),
                    help="a guarded turn of a request without reasoning is generated with reasoning on, at most this "
                         "many reasoning tokens, on top of the client's max_tokens (0 = off); env QE_LOOP_GUARD_THINK")
    ap.add_argument("--loop-guard-note", type=int, default=env_int("QE_LOOP_GUARD_NOTE", 1),
                    help="guarded turn: append a server note to the latest tool result (the same call returned the "
                         "same result N times, change the approach); 0 = off; env QE_LOOP_GUARD_NOTE")
    ap.add_argument("--metrics-port", type=int, default=9393,
                    help="live monitoring (dashboard, qetop, /metrics) on this port; 0 = off")
    ap.add_argument("--metrics-host", default="127.0.0.1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--disk-cache", default="",
                    help="directory of a persistent prompt cache surviving restarts (default off: "
                         "conversations live in GPU slots and host memory only)")
    ap.add_argument("--stats-file", default=STATS_DEFAULT,
                    help="all-time monitoring totals (dashboard)")
    ap.add_argument("--disk-cache-gb", type=float, default=200)
    ap.add_argument("--persist-idle", type=float, default=120,
                    help="seconds idle before live conversations are saved to the disk cache")
    ap.add_argument("--disk-eager", action="store_true",
                    help="with --disk-cache (Qwen3.8-27B / Flash-Next): write each conversation to the disk cache in the "
                         "background right after every finished reply (a kill -9 then keeps the last turn), instead of "
                         "only at --persist-idle and at shutdown. Off by default: it writes the whole conversation file "
                         "after every turn (NVMe wear)")
    ap.add_argument("--log-file", default=env_str("QE_LOG_FILE", "auto"),
                    help="persistent log: the server's whole output (also the engine's) is appended to this file with "
                         "a timestamp per line and survives crashes and restarts (auto: "
                         "~/.cache/qengine/logs/<served-name>.log; '' or 0: off); env QE_LOG_FILE (0: off, empty: unset)")
    ap.add_argument("--log-mb", type=float, default=env_float("QE_LOG_MB", 64.0),
                    help="size of the persistent log before it is rotated (3 old files are kept); env QE_LOG_MB")
    env0 = dict(os.environ)  # [feature-matrix] the environment as started (before the server sets its own variables)
    args = ap.parse_args(argv)
    args.given = given_flags(ap, argv)
    # (argparse also takes an unambiguous abbreviation: --max-c, --max-ct; review shared_server L9)
    args.max_ctx_given = any((lambda k: k.startswith("--max-c") and "--max-ctx".startswith(k))(x.split("=", 1)[0])
                             for x in (argv or sys.argv[1:]))
    if args.loop_guard_turns > 0 and not 2 <= args.loop_guard_turns <= loop_guard.WINDOW:
        # 1 flags every reply with a call (a reply that repeats nothing is a stretch of one), above the window nothing
        # ever fires (review 2 L7). The flag is refused; the env value warns and keeps the default (as qe_env does)
        if any(x == "--loop-guard-turns" or x.startswith("--loop-guard-turns=") for x in (argv or sys.argv[1:])):
            ap.error(f"--loop-guard-turns {args.loop_guard_turns}: 0 (off) or 2 to {loop_guard.WINDOW}")
        print(f"[env] warning: QE_LOOP_GUARD_TURNS={args.loop_guard_turns} is not 0 (off) or 2 to {loop_guard.WINDOW}, "
              f"using the default 3", file=sys.stderr, flush=True)
        args.loop_guard_turns = 3
    if args.main_bonus is not None:  # (review M2: old command lines keep starting)
        print("[sched] --main-bonus is ignored: no main-thread priority since 2026-11-12 (every conversation alike)", flush=True)
    resolve_model_args(args)
    # [feature-matrix] flags / variables that have no effect for this model: logged by main() after the log starts
    args.ignored_start = ignored_settings(args.profile.family, args.given, env0, args=args) + getattr(args, "ink_lines", [])
    args.env0 = env0
    if not args.template:
        args.template = os.path.join(args.model, "chat_template.jinja")
    args.think = bool(args.think)
    try:
        args.thinking_budget = BudgetMap.parse(args.thinking_budget)
    except ValueError as ex:
        ap.error(f"--thinking-budget / QE_THINKING_BUDGET: {ex}")
    if args.reasoning_effort is not None:
        args.reasoning_effort = args.reasoning_effort.strip().lower() or None  # '' = unset, like the env var
    if args.reasoning_effort is not None and args.reasoning_effort not in KNOWN_EFFORTS + CLIENT_EFFORTS + (NO_EFFORT,):
        print(f"[server] warning: --reasoning-effort {args.reasoning_effort!r} is not one of "
              f"{', '.join(KNOWN_EFFORTS + CLIENT_EFFORTS + (NO_EFFORT,))}: the template renders it as medium",
              flush=True)
    # the MTP head fine-tuned on the engine's own answers: +8% accepted drafts
    ft = os.path.join(args.model, "model_mtp_ft.safetensors")
    # --lsa wins over $QE_LSA (the engine reads $QE_LSA as the map path); auto / a .pt resolve in main() (lsa_map.py)
    from_env = args.lsa is None
    if from_env:
        args.lsa = os.environ.get("QE_LSA")
    if args.lsa is not None:
        import lsa_map
        spec = args.lsa.strip()
        if spec.lower() in lsa_map.OFF:
            os.environ.pop("QE_LSA", None)
            args.lsa = None
        elif spec.lower() in lsa_map.AUTO:
            os.environ.pop("QE_LSA", None)
            args.lsa = "auto"
        elif args.profile.is_flash:  # (Flash-Next ignores the map; an explicit missing / wrong file stops as before)
            if not from_env and not os.path.isfile(spec):
                sys.exit(f"--lsa: no such file {spec}")
            if not from_env and not lsa_map.is_safetensors(spec):
                sys.exit(f"--lsa: {spec} is not a safetensors file")
            os.environ["QE_LSA"] = os.path.abspath(spec)
        elif from_env and not lsa_map.is_pt(spec):
            pass  # $QE_LSA=<path> as before: the engine checks it (a bad map: `LSA off: map ...`, full attention)
        else:
            try:  # --lsa PATH: checked now, SystemExit as before; a released .pt is converted into the cache
                args.lsa = lsa_map.resolve(spec, args.model)
            except SystemExit as ex:
                if not from_env:
                    raise
                print(f"[server] LSA: off: $QE_LSA {ex}; the server starts with full attention", flush=True)
                args.lsa = None
                os.environ.pop("QE_LSA", None)
            else:
                os.environ["QE_LSA"] = args.lsa
    if args.mtp_file:
        os.environ["QE_MTP_FILE"] = args.mtp_file
    elif env_str("QE_MTP_FILE") is None and os.path.exists(ft):
        import assets  # 27B INT8 checkpoints keep their own head unless QE_INT8_MTP_FT=1 (Flash-Next's D2 has INT8 groups too: unchanged)
        if args.profile.is_flash or assets.use_mtp_ft(args.model):
            os.environ["QE_MTP_FILE"] = ft
    if args.disk_cache:  # (also the directory of the start-up cache, comm_cache.h: on Flash-Next too)
        os.environ["QE_DISK_CACHE"] = args.disk_cache
        os.environ["QE_DISK_CACHE_GB"] = str(args.disk_cache_gb)
    # Flash-Next: its disk tier has its own variables. QE_DISK_CACHE alone only moves the start-up cache there (documented),
    # so setting it never turns the tier on; without the flag it is off
    os.environ.pop("FLASH_DISK_CACHE", None)
    os.environ.pop("FLASH_DISK_EAGER", None)
    os.environ.pop("QE_DISK_EAGER", None)
    if args.disk_cache and args.disk_eager and not args.profile.is_flash:  # (27B: engine.cpp disk_eager_)
        os.environ["QE_DISK_EAGER"] = "1"
    if args.disk_cache and args.profile.is_flash:
        os.environ["FLASH_DISK_CACHE"] = args.disk_cache
        os.environ["FLASH_DISK_CACHE_GB"] = str(args.disk_cache_gb)
        if args.disk_eager:
            os.environ["FLASH_DISK_EAGER"] = "1"
    if args.disk_eager and not args.disk_cache:
        print("[settings] --disk-eager has no effect without --disk-cache DIR", flush=True)
    elif args.disk_eager and not args.bg_archive:  # (review L3: the copy after a reply is the idle archiver's)
        print("[settings] --disk-eager has no effect with --bg-archive 0 (the idle archiver makes the copy it writes)", flush=True)
    return args


def start_log(args):
    """The persistent log (server/logtee.py), before the engine loads; a header with the build and settings."""
    lf = args.log_file
    if not lf or lf in ("0", "off", "none"):
        return
    import logtee
    import sys
    path = logtee.default_path(args.served_name or "qengine") if lf == "auto" else os.path.expanduser(lf)
    version, build = version_info(args.engine_lib or env_str("QE_LIB"))
    env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                   if k.startswith(("QE_", "FLASH_", "CUDA_VISIBLE", "CUDA_MODULE")))
    header = (f"==== qengine server start: QEngine {version}, pid {os.getpid()}, build {build} ({os.path.realpath(ROOT)}), log {path}\n"
              f"argv: {' '.join(sys.argv)}\nenv: {env}")
    try:
        if env_str("FLASH_PAR_FALLBACK"):  # (the engine restarted itself in pp: the first start's tee is still running)
            logtee.reattach(header)
        else:
            logtee.install(path, args.log_mb, 3, header)
    except Exception as ex:  # never keep the server from starting
        print(f"persistent log off: {ex}", flush=True)


def main():
    # DSV4 (DeepSeek-V4.1-Flash from a GGUF split set, e.g. Maya-S): its own server module (one sequence, DeepSeek
    # prompt format); the 27B / Flash-Next path below is unchanged
    # [shared-srv] since 2026-11-18 DSV4 runs in this server (model profile "dsv4", server/dsv4_engine.py);
    # QE_DSV4_LEGACY=1 = the off-switch: the old module as before
    if "--model" in sys.argv[1:-1] and env_str("QE_DSV4_LEGACY", "0") not in ("", "0"):
        md = sys.argv[sys.argv.index("--model") + 1]
        try:   # (the DSV4 modules are not in the public export: without them this path is skipped)
            import dsv4_gguf
        except ImportError:
            dsv4_gguf = None
        if dsv4_gguf is not None and dsv4_gguf.is_dsv4_dir(md):
            import dsv4_app
            return dsv4_app.main(sys.argv[1:], version_info)   # (version + build for qetop's header: [dsv4-qetop])
    args = parse_args()
    start_log(args)
    print("QEngine %s (build %s)" % version_info(args.engine_lib or env_str("QE_LIB")), flush=True)
    if not env_str("FLASH_PAR_FALLBACK"):  # (the pp restart: the first start logged them)
        for line in args.ignored_start:
            print(line, flush=True)
    if args.profile.is_glm:  # GLM (round 18 fix of the master merge): only the assets step it had on its branch; the
        import assets          # 27B checkpoint-format check refused GLM's sharded checkpoint, DFlash2 / LSA are 27B-only
        assets.ensure(args.model)
    elif getattr(args.profile, "is_dsv4", False):  # DSV4: GGUF weights, no assets
        pass
    elif getattr(args.profile, "is_inkling", False):  # [ink-srv-capi] Inkling 2: its own checkpoint (+ mtp.safetensors), no assets
        pass
    elif args.profile.is_gptoss:  # [srv-gptoss] gpt-oss: its own MXFP4 checkpoint (the engine checks it), no assets
        # [gptoss-rel1] features the gpt-oss engine does not offer in the server (1.5): said at the start, never a stop
        if env_str("GPTOSS_DFLASH"):  # (blocker 18) the server never drafts with it: loading it would only cost pool rows
            print(f"[server] gpt-oss: GPTOSS_DFLASH={env_str('GPTOSS_DFLASH')} ignored: the server does not use the DFlash "
                  "drafter yet (engine tool only); lookup drafts stay on", flush=True)
            os.environ.pop("GPTOSS_DFLASH", None)
        if getattr(args, "disk_cache", ""):
            print(f"[server] gpt-oss: --disk-cache {args.disk_cache} ignored: the gpt-oss engine keeps its prompt cache in "
                  "VRAM and host RAM only (lost at a restart)", flush=True)
            args.disk_cache = ""
    elif not args.profile.is_flash:  # 27B: tuned MTP head + draft vocabulary from the latest release if missing
        import assets
        why = assets.check_27b_format(args.model)
        if why:
            sys.exit(print(f"[server] {why}", flush=True) or 3)
        assets.ensure(args.model)
        import dflash_fetch  # DFlash2 drafter checkpoint: found or fetched once (pinned revision + sha256); never a stop
        dflash_fetch.resolve(args.model)
        if args.lsa == "auto":  # find / fetch / convert the released map; any failure: full attention, never a stop
            import lsa_map
            path = lsa_map.resolve("auto", args.model)
            if path:
                set_env("QE_LSA", path)  # (a fetch thread may still run: ENV_LOCK, review FX11-L1)
    try:  # kill -USR1 <pid>: Python stacks of every thread (the native ones: gdb -p <pid>)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass
    try:
        gen = Generator(args)
    except EngineRefused as ex:  # (exit 3 = a refused start, as the engine's own load-time stops; review Flash-Next M7)
        print(f"qengine: {ex}", file=sys.stderr, flush=True)
        hard_exit(3)
    class Server(uvicorn.Server):  # SIGTERM / SIGINT: end the requests first, then uvicorn's wait for the streams
        def handle_exit(self, sig, frame):
            if not gen.stopping:
                gen.stop_requests()
            super().handle_exit(sig, frame)

    # uvicorn (>= 0.29) re-raises the SIGTERM / SIGINT it caught once its run() ends, against the handlers that were
    # in place before it: with the default ones the process died right there, so the conversations were never saved
    # to --disk-cache at shutdown (docker stop) and the stats were not written. A no-op handler takes that re-raise;
    # a signal during the save below ends the process at once (a second docker stop / Ctrl-C).
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *a: None)
    Server(uvicorn.Config(build_app(gen, args), host=args.host, port=args.port, log_level="warning")).run()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda n, f: hard_exit(128 + n))
    gen.metrics.save()
    # uvicorn returns after SIGTERM / SIGINT: save the live conversations before exiting
    if args.disk_cache:
        t0 = time.time()
        gen.run_on_engine(gen.engine.persist)
        # Flash-Next queues its files: wait for them (bounded: a hung disk must not hold the exit), then what a full archive
        # could not take before those writes let go of their memory
        flushed = gen.engine.disk_flush(DISK_FLUSH_S)
        if flushed and "qe_disk_flush" in gen.engine.fns:
            gen.run_on_engine(gen.engine.persist)
            flushed = gen.engine.disk_flush(DISK_FLUSH_S)
        if not flushed:
            print(f"shutdown: disk cache writes still running after {DISK_FLUSH_S:.0f}s, not waiting (the unfinished file is dropped at the next start)", flush=True)
        print(f"shutdown: slots saved in {time.time() - t0:.1f}s", flush=True)
    gen.engine.set_exit_hook(None)
    # end here, as the re-raised signal did before: no interpreter / CUDA teardown while engine threads still run
    sys.stdout.flush()
    sys.stderr.flush()
    hard_exit(0)


if __name__ == "__main__":
    main()
