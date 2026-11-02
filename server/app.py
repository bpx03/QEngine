"""OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) compatible server for qengine.

usage:
  CUDA_VISIBLE_DEVICES=0,1,2 python server/app.py --port 8080 [--template FILE] [--max-ctx 262144]

One engine, one request at a time (requests queue up); prompt caching is automatic: every
request is a full prompt and the engine reuses the longest cached prefix.
"""
import argparse
import array
import hashlib
import asyncio
import collections
import faulthandler
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime

import jinja2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agents import Tracker
from qe_env import env_float, env_int, env_str
import loop_guard
from engine import Engine, SequenceFault, StartRefused
import engine_stats
from steering import DEFAULT as STEER_DEFAULT, resolve_steering
from grammar import GrammarError, Grammars, TokenLookup, schema_of
from think_budget import BudgetError, ThinkControl, ThinkIds, empty_reply, request_budget
from model_profile import DroppedTurnLog, Profile, drop_empty_turns, merge_leading_system

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
# forward progress (issue #3): the scheduler loop beats a heartbeat; no beat for QE_HANG_S seconds (an engine call that
# never returns) -> Python + native stacks, the requests end with an error, exit 75 (docker / systemd restart it).
# A start refused after admission for lack of room (KV pool full, prompts in flight) is requeued while sequences run,
# for up to QE_START_RETRY_S seconds, before it fails; bad input and too-long refusals fail at once.
# A sequence that gets no token for QE_NO_PROGRESS_STEPS steps and QE_NO_PROGRESS_S seconds while not paused fails.
HANG_S = env_float("QE_HANG_S", 180.0)
START_RETRY_S = env_float("QE_START_RETRY_S", 60.0)
NOPROG_STEPS = env_int("QE_NO_PROGRESS_STEPS", 64)
NOPROG_S = env_float("QE_NO_PROGRESS_S", 10.0)
# the hang deadline while the slots are persisted to the disk cache (27B --disk-cache: idle and at shutdown)
HANG_PERSIST_S = env_float("QE_HANG_PERSIST_S", 3600.0)


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


def hard_exit_guard(after_s, code=75):
    """The hard bound of the hang path: an in-process timer exits with `code` after `after_s`, and an outside watcher
    process SIGKILLs this one (SIGCONT first: a killed gdb can leave it stopped) after_s + 10 s later if it still
    runs. The watcher ends quietly when this process exits first (its pipe closes)."""
    t = threading.Timer(after_s, lambda: os._exit(code))
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


class Template:
    def __init__(self, path):
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                 extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
        self.t = env.from_string(open(path).read())

    def render(self, messages, tools=None, **kw):
        return self.t.render(messages=messages, tools=tools or None, add_generation_prompt=True,
                             **{k: v for k, v in kw.items() if v is not None})


# ----------------------------------------------------------------------------- detokenizer


class Detok:
    """Incremental detokenization that never emits half of a multi-byte character."""

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0

    def push(self, ids):
        self.ids += ids
        prev = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        cur = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if cur.endswith("�"):
            return ""
        out = cur[len(prev):]
        self.prefix, self.read = max(0, len(self.ids) - 6), len(self.ids)
        return out


# ----------------------------------------------------------------------------- output parser


class Parser:
    """Splits generated text into reasoning / content / tool calls, holding back partial tags."""

    TAGS = ("<think>", "</think>", "<tool_call>")

    def __init__(self, thinking):
        self.mode = "reasoning" if thinking else "content"
        self.buf = ""
        self.tool_text = ""
        self.lead = False  # right after </think>: newlines before the answer are dropped, also in later chunks

    def _hold(self, s):
        # length of the longest suffix of s that is a prefix of a tag we may still see
        for n in range(min(len(s), 12), 0, -1):
            if any(t.startswith(s[-n:]) for t in self.TAGS):
                return n
        return 0

    def feed(self, text, final=False):
        """Returns list of (kind, text) with kind in reasoning/content."""
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
            tag = "</think>" if self.mode == "reasoning" else "<tool_call>"
            i = self.buf.find(tag)
            if self.mode == "content" and self.buf.lstrip().startswith("<think>") and not out:
                j = self.buf.find("<think>")  # model re-opened a think block
                self.buf = self.buf[j + 7:]
                self.mode = "reasoning"
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
        return [(k, t) for k, t in out if t]


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


DIAG_PATH = os.path.expanduser(env_str("QE_DIAG_LOG", "~/.local/state/qengine/diag.log"))


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


def parse_tool_calls(text, tools):
    schemas = {}  # name -> properties of its parameter schema (None: the schema lists none)
    for t in tools or []:
        f = t.get("function", t)
        props = (f.get("parameters") or {}).get("properties")
        schemas[f.get("name")] = props if isinstance(props, dict) and props else None
    calls = []
    for block in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S):
        m = re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", block, re.S)
        if not m:
            continue
        name, body = m.group(1).strip(), m.group(2)
        props = schemas.get(name)
        args = {}
        ms = list(re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S))
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
                diag("TOOL_ARGS_MISSING", f"{name} missing {miss}: {block[:300]!r}"
                     f" (raw: {diag_dump('tool', text)})")
    return calls


# ----------------------------------------------------------------------------- generation


WS = re.compile(r"\s+")


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


class Admission:
    """Which queued request the scheduler starts next (engine thread; called under gen.cv)."""

    def __init__(self, gen):
        self.g, self.e, a = gen, gen.engine, gen.args
        self.aging, self.pre_min, self.pre_max = a.prefill_aging, a.preempt_min, a.preempt_max
        self.main_bonus = getattr(a, "main_bonus", 0)
        self.backfill_wait = getattr(a, "backfill_wait", 0)
        self.k = a.mtp_k
        self.last_note = 0.0
        # several prompts in one prefill batch (Flash-Next): every admitted text prompt joins the next batches
        self.multi = gen.engine.can_prefill_multi() if hasattr(gen.engine, "can_prefill_multi") else False
        # passes bound by the PCIe copies of the host-backed experts (2 cards): the large batch also while requests wait
        self.pass_bound = self.multi and gen.engine.prefill_pass_bound()
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
        `aging` tokens), the main agent thread ahead by `main_bonus` tokens (it waits for nobody
        else; a subagent overtakes it only after waiting main_bonus / aging seconds). While a long
        prompt is in progress only a request with at most preempt_max tokens left may cut in (and
        one at a time), or a main-thread request of any size on a long prompt that is not main."""
        q = self.g.queue
        if not q:
            return []
        self.estimate(now)
        order = sorted(range(len(q)), key=lambda i: q[i].est - (now - q[i].t_enq) * self.aging
                       - (self.main_bonus if q[i].prio >= 2 else 0))
        if not prefills:
            return order
        join = []
        if self.multi and len(prefills) < MULTI_MAX:
            # text prompts join the shared batches of the ones in flight while all of them fit one batch (a long prompt
            # gains nothing from sharing and admitting it early only claims its KV region ahead); else the cut-in rule
            cap = self.join_cap()
            left = sum(len(j.ids) - max(0, self.e.prefill_pos(j.slot)) for j in prefills)
            join = [i for i in order if not getattr(q[i].ids, "mm", None) and left + q[i].est <= cap]
            if join:
                return join
        if self.pre_max <= 0 or (len(prefills) >= 2 and not self.multi):
            return []
        left_of = lambda j: len(j.ids) - max(0, self.e.prefill_pos(j.slot))
        long_ = max(prefills, key=left_of)
        pos = max(0, self.e.prefill_pos(long_.slot))
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
        return [i for i in order if not getattr(q[i].ids, "mm", None) and
                (q[i].est <= self.pre_max or (q[i].prio >= 2 and long_.prio < 2))]

    def join_cap(self):  # prompt rows in flight that still share one prefill batch (the idle batch + a tail chunk)
        return max(self.g.args.prefill_chunk, self.g.args.prefill_chunk_idle) + 512

    def batch_order(self, now, prefills):
        """Prompts in flight in the order they get rows of a shared prefill batch: least left first, aged, main thread
        ahead (as pick_next); a long prompt gets what is left of the batch. Prompts that end within one batch (cut-ins)
        always come before longer ones: aged by its wait in the queue, a long prompt in progress for ~45 s sorted ahead
        of a cut-in, took the whole batch and the cut-in waited for all of it (KV in RAM, t8 + 262K: TTFT p50 58 s)."""
        cap = self.join_cap()
        def key(j):
            left = len(j.ids) - max(0, self.e.prefill_pos(j.slot))
            return (left > cap, left - (now - j.t_enq) * self.aging - (self.main_bonus if j.prio >= 2 else 0))
        return sorted(prefills, key=key)

    def admit_next(self, now, prefills):
        """Index of the queued request to start now, or None. The best candidate that fits the pool
        goes; one that does not fit lets the next ones that fit start before it (backfill) until it
        has waited backfill_wait seconds: then only a request of a higher prio may still pass it.
        A main-thread request that does not fit pauses decoding sequences worth less (never a
        prefill, never another main thread) when that makes it fit."""
        g, e = self.g, self.e
        blocked = None
        for i in self.pick_next(now, prefills):
            j = g.queue[i]
            if blocked is not None and now - blocked.t_enq > self.backfill_wait and j.prio <= blocked.prio:
                continue  # starvation guard: nothing of the same or a lower class passes it any more
            need = len(j.ids) + self.k + 2
            if e.can_admit(need):
                if blocked is not None:
                    g.sched["backfills"] += 1
                    self.note(f"[sched] backfill: {need} tokens start while {len(blocked.ids)} tokens "
                              f"(waiting {now - blocked.t_enq:.1f} s) do not fit")
                return i
            if j.prio >= 2 and e.pause_for(need, j.prio):
                g.sched["main_pauses"] += 1
                g.cur_slot = -1  # a paused slot may have been the current single sequence
                print(f"[sched] main thread ({need} tokens): paused lower-prio sequences to admit it", flush=True)
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
        self.tmpl = Template(args.template)
        self.im_start_user = self.tok.encode("<|im_start|>user", add_special_tokens=False).ids
        self.seg_cache = collections.OrderedDict()
        self.seg_tokens = 0
        self.turns = collections.OrderedDict()  # affinity -> recent (prompt, generated) (see check_reuse)
        self.gen_turns = collections.OrderedDict()  # see remember_turn
        self.im_start = self.tok.token_to_id("<|im_start|>")
        self.im_end_nl = self.tok.encode("<|im_end|>\n", add_special_tokens=False).ids
        # loop guard (server/loop_guard.py): the repeated reply's tokens seed the penalty, without special tokens
        # (ids from <|endoftext|> on) and whitespace-only pieces
        self.asst_ids = self.tok.encode("assistant\n", add_special_tokens=False).ids
        self.im_end = self.tok.token_to_id("<|im_end|>")
        self.special_min = self.tok.token_to_id("<|endoftext|>") or 1 << 30
        wsch = set("ĠĊĉčĀ \t\n")  # byte-level BPE spellings of whitespace
        self.ws_ids = {i for t, i in self.tok.get_vocab().items() if i < self.special_min and t and set(t) <= wsch}
        self.guards = {"triggers": 0, "recent": []}  # /api/stats loop_guard
        self.seg_lock = threading.Lock()
        # end of turn: generation_config.json eos_token_id ([<|im_end|>, <|endoftext|>] for both
        # models) plus those two tokens by name
        self.eos = set(prof.eos_ids) | {self.tok.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")
                                        if self.tok.token_to_id(t) is not None}
        sd = prof.sampling(args.temperature, args.top_p, args.top_k, args.min_p)
        args.temperature = sd.pop("temperature")
        self.sampling_defaults = sd
        print("sampling defaults (when the request has none): temperature %.2f, top_p %.2f, top_k %d, min_p %.2f"
              % (args.temperature, self.sampling_defaults["top_p"], self.sampling_defaults["top_k"],
                 self.sampling_defaults["min_p"]), flush=True)
        self.vision = None
        if args.vision:  # before the engine, so that its KV cache planning sees the memory used
            from vision import VisionEncoder
            dev = args.vision_device or f"cuda:{len(args.devices.split(',')) - 1}"
            t0 = time.time()
            self.vision = VisionEncoder(args.model, dev, args.max_image_pixels)
            self.vision_lock = threading.Lock()
            print(f"vision encoder on {dev} in {time.time() - t0:.1f}s "
                  f"(max {args.max_image_pixels / 1e6:.1f} MPix per image)", flush=True)
        self.engine = Engine(args.model, args.devices, args.max_ctx, getattr(args, "split", "") or "",
                             lib_path=getattr(args, "engine_lib", None) or None,
                             vocab=prof.vocab_size)
        self.dstats = engine_stats.DraftStats(self.engine)  # per-slot draft counters (qe_draft_stats)
        if not self.engine.has("stops"):  # no undo of the steps past a stop sequence
            global STOP_ROLLBACK
            STOP_ROLLBACK = 0
        missing = self.engine.missing_optional()
        if missing:
            print(f"engine: optional C API not provided (fallbacks in use): {', '.join(missing)}", flush=True)
        # structured output (response_format): llguidance matchers, masks written into the engine
        self.grammars = None
        print(f"engine parallel mode: {self.engine.par.get('mode', 'tp')} ({self.engine.par.get('why', '')})", flush=True)
        if self.vision is not None and not self.engine.has("vision"):  # (mode B: no image input on the pipeline stages)
            print(f"--vision ignored: {self.engine.refused.get('vision', 'the engine has no image input')}", flush=True)
            self.vision = None
        if self.engine.has("grammar"):
            self.grammars = Grammars(tok_json, self.engine.vocab(), sorted(self.eos))
            self.grammars.prepare()
        # thinking budget (server/think_budget.py): the forced close of <think> goes through the grammar tier
        self.think_ids = ThinkIds(self.tok)
        self.can_force = self.engine.has("grammar") and self.think_ids.ok
        self.think = {"requests_with_budget": 0, "forced_budget": 0, "forced_max_tokens": 0, "forced_failed": 0,
                      "empty_replies": 0, "empty_stop": 0, "empty_length": 0, "empty_after_forced_close": 0,
                      "empty_retried": 0, "empty_retry_recovered": 0, "empty_retry_exhausted": 0}
        self.think_mu = threading.Lock()  # (counted from the scheduler thread and the event loop: review server F9)
        self.drop_log = DroppedTurnLog()  # --drop-empty-turns (HF #50): logged once per conversation
        if not self.can_force:
            print("thinking budget: not enforced (the engine has no grammar C API or the tokenizer no <think> "
                  "tokens): budgets are ignored and max_tokens may end inside <think>", flush=True)
        elif getattr(args, "thinking_budget", 0):
            print(f"thinking budget: {args.thinking_budget} tokens by default (a request's budget wins)", flush=True)
        from metrics import Metrics
        stats = os.path.expanduser(args.stats_file)
        self.metrics = Metrics(stats, args.served_name)
        # main agent thread detection (server/agents.py) and the scheduler's counters, both in /api/stats
        self.agents = Tracker(override=getattr(args, "main_agent_re", None))
        self.sched = {"main_pauses": 0, "main_cutins": 0, "backfills": 0}
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
            # server default is 0.6
            print(f"graph warm-up: {self.engine.warmup(greedy=args.temperature <= 0):.1f}s", flush=True)
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
            for j in list(active) + list(prefills) + list(self.queue) + [r[1] for r in self.retry_q] + \
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
            os._exit(75)

    def prompt(self, messages, tools, kw, meta=None, track=False, guard_on=True):
        """Returns (ids, affinity): affinity = hash of the system prompt, so the main agent and
        each kind of subagent keep returning to their own sequence slot. ids.hint = position of
        the first user turn (end of the system part). track: a generation request (not a token
        count): the main-thread tracker sees it (meta: the request's metadata, its session id)."""
        info = None
        if track:
            sys0 = "".join(str(m.get("content")) for m in messages[:1] if m.get("role") == "system")
            aff0 = int.from_bytes(hashlib.blake2b(sys0.encode(), digest_size=8).digest(), "little")
            try:
                info = self.agents.observe(messages, tools, meta, aff0)
            except Exception as ex:  # monitoring / priority only: never fail a request for it
                print(f"agents: {ex!r}", flush=True)
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
        guard = self.loop_check(messages, info) if track and guard_on else None
        if guard and self.args.loop_guard_note:  # the repeated call's result tells the model it is looping
            messages, guard["note"] = loop_guard.add_note(messages, guard["turns"])
        messages = self.clean_history(messages, info, track)
        if getattr(self.args, "sort_tools", 1):
            tools = canonical_tools(tools)
        text = self.tmpl.render(messages, tools, **kw)
        sys_txt = "".join(str(m.get("content")) for m in messages[:1] if m.get("role") == "system")
        label = "main agent" if info is not None and info.main else agent_label(sys_txt, messages)
        affinity = int.from_bytes(hashlib.blake2b(sys_txt.encode(), digest_size=8).digest(), "little")
        if env_str("QE_DUMP_PROMPTS"):
            d = env_str("QE_DUMP_PROMPTS")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{time.time():.3f}.txt"), "w") as f:
                f.write(text)
        ids = self.encode(text)
        ids.label, ids.agent, ids.guard = label, info, guard
        if images:
            from vision import load_image_bytes
            t0 = time.time()
            try:
                raw = [load_image_bytes(s) for s in images]
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
        return ids, affinity

    def loop_check(self, messages, info):
        """Cross-turn loop guard (server/loop_guard.py): None, or the guard of this request's turn."""
        a = self.args
        if getattr(a, "loop_guard_turns", 0) <= 0:
            return None
        try:
            k = loop_guard.streak(messages)
        except Exception as ex:  # a guard must never fail a request
            print(f"loop guard: {ex!r}", flush=True)
            return None
        if k < a.loop_guard_turns:
            return None
        pres = a.loop_guard_penalty if k < 2 * a.loop_guard_turns else max(a.loop_guard_penalty, 2.0)
        conv = getattr(getattr(info, "conv", None), "id", None)
        sess = getattr(getattr(info, "session", None), "key", "") or ""
        return {"turns": k, "presence": pres, "conv": conv, "session": sess[2:10],
                "main": bool(getattr(info, "main", False))}

    def guard_note(self, guard):
        """A guarded turn starts: warning line, diag log, /api/stats (qetop)."""
        ev = {"t": time.time(), "turns": guard["turns"], "conversation": guard["conv"], "session": guard["session"],
              "main": guard["main"], "presence": guard["presence"]}
        with self.seg_lock:
            self.guards["triggers"] += 1
            self.guards["recent"] = (self.guards["recent"] + [ev])[-8:]
        print(f"WARNING loop guard: conversation {guard['conv']} (session {guard['session'] or '-'}) repeated the "
              f"same turn {guard['turns']} times: next turn with presence penalty {guard['presence']:.2f} on the "
              f"repeated reply ({guard.get('seed', 0)} tokens), lookup drafts off"
              f"{', note appended to the tool result' if guard.get('note') else ''}", flush=True)
        diag("LOOP_GUARD", f"conversation {guard['conv']} session {guard['session']} turns {guard['turns']} "
             f"presence {guard['presence']:.2f} note {int(bool(guard.get('note')))}")

    def steer_stats(self):
        """/api/stats steering: loaded vector names, the default, and the vectors dropped at start (with the reason)"""
        st = getattr(self, "steer", None) or {}
        return {"enabled": bool(st.get("vectors")), "default": st.get("default"), "default_active": st.get("default") is not None,
                "vectors": [v.get("name") for v in st.get("vectors", [])], "dropped": st.get("dropped", [])}

    def guard_stats(self):
        with self.seg_lock:
            return {"turns": getattr(self.args, "loop_guard_turns", 0), "triggers": self.guards["triggers"],
                    "recent": list(self.guards["recent"])}

    def clean_history(self, messages, info=None, track=False):
        """Model-specific history fixes before the template (--merge-system, --drop-empty-turns). track: a
        generation request (info: the agent tracker's view of it); the HF #50 drop is logged once per
        conversation and again only when its count grows."""
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
        conversation only tokenizes its new messages (84K tokens: ~85 ms -> <1 ms)."""
        ids = Ids()
        with self.seg_lock:
            for i, part in enumerate(text.split("<|im_start|>")):
                seg = part if i == 0 else "<|im_start|>" + part
                if not seg:
                    continue
                t = self.seg_cache.get(seg)
                if t is None and seg.startswith("<|im_start|>assistant"):
                    t = self.gen_turns.get(turn_key(seg))
                    if t is not None:
                        t = t + self.im_end_nl if seg.endswith("<|im_end|>\n") else None
                    if t is not None:
                        self.seg_cache[seg] = t
                        self.seg_tokens += len(t)
                if t is None:
                    t = self.tok.encode(seg, add_special_tokens=False).ids
                    self.seg_cache[seg] = t
                    self.seg_tokens += len(t)
                    while self.seg_tokens > 2_000_000 and self.seg_cache:
                        _, old = self.seg_cache.popitem(last=False)
                        self.seg_tokens -= len(old)
                else:
                    self.seg_cache.move_to_end(seg)
                if not ids.hint and seg.startswith("<|im_start|>user"):
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
        uses them for a matching assistant segment."""
        if self.im_start not in prompt[-16:]:
            return
        head = prompt[len(prompt) - prompt[::-1].index(self.im_start) - 1:]
        t = list(head) + list(out)
        k = turn_key(self.tok.decode(t, skip_special_tokens=False) + "<|im_end|>")
        with self.seg_lock:
            self.gen_turns[k] = t
            self.gen_turns.move_to_end(k)
            while len(self.gen_turns) > 1024:
                self.gen_turns.popitem(last=False)

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
            return dict(self.think)

    def think_add(self, key, n=1):
        with self.think_mu:
            self.think[key] = self.think.get(key, 0) + int(n)

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
            if msg:
                print(f"engine fault: request on slot {j.slot} ({len(j.ids)} prompt tokens, {getattr(j, 'n', 0)} "
                      f"generated) failed: {msg}", flush=True)
                j.fail(SequenceFault(msg))
                hit = True
        return hit

    def no_progress(self, j, n):
        """A sequence that gets no token for NOPROG_STEPS steps and NOPROG_S seconds while it is not paused fails
        with an error (and the pool state is logged) instead of looping silently (issue #3 D2)."""
        if n or j.finished or NOPROG_STEPS <= 0 or NOPROG_S <= 0:  # QE_NO_PROGRESS_STEPS / _S = 0: off
            j.empty_n, j.t_tok = 0, time.time()
            return
        j.empty_n = getattr(j, "empty_n", 0) + 1
        t_tok = getattr(j, "t_tok", None) or time.time()
        j.t_tok = t_tok
        if j.empty_n < NOPROG_STEPS or time.time() - t_tok < NOPROG_S:
            return
        try:
            pi = self.engine.pool_info()
            if pi["slots"][j.slot]["paused"]:
                j.empty_n = 0
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
        bg_done = True  # idle archiving has nothing left (reset whenever a request finishes)
        pre_min, pre_max = self.args.preempt_min, self.args.preempt_max
        burst_wait = max(0.0, getattr(self.args, "burst_wait", 5.0)) / 1000.0

        adm = Admission(self)

        trace = open(env_str("QE_SCHED_TRACE"), "w", buffering=1) if env_str("QE_SCHED_TRACE") else None
        prof_last = 0.0  # QE_STEP_PROF: return time of the last single step (0: none to chain from)

        def tr(kind, t0, n=0, toks=0):  # scheduler trace (QE_SCHED_TRACE=file): one JSON line per event
            if trace:
                trace.write(json.dumps([kind, round(t0, 6), round(time.time() - t0, 6), n, toks]) + "\n")
        last_mg = 0.0
        while True:
            self.beat("scheduler loop")
            self.jobs_ref = (active, prefills)
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
            ctrl = []
            with self.cv:
                while not self.queue and not active and not self.ctrl and not prefills:
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
                    if self.dirty and idle_persist > 0 and self.args.disk_cache:
                        if idle_since is None:
                            idle_since = time.monotonic()
                        rest = idle_persist - (time.monotonic() - idle_since)
                        if rest > 0:
                            self.cv.wait(min(rest, HANG_S / 2) if HANG_S > 0 else rest)
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
                        except Exception:
                            pass
                        finally:
                            self.cv.acquire()
                if self.queue or active or prefills or self.ctrl:
                    idle_since = None
                if not active and not prefills:
                    ctrl, self.ctrl = self.ctrl, []
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
                    i = adm.admit_next(time.time(), prefills)
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
                            if admit.prio >= 2 and long_.prio < 2 and admit.est > pre_max:
                                self.sched["main_cutins"] += 1
                            print(f"[sched] {admit.est} tokens to prefill cut in on {len(long_.ids)} at "
                                  f"{long_.cut_at}{' (main thread)' if admit.prio >= 2 else ''}", flush=True)
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
                    tr("begin_start", tb, len(admit.ids))
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
                    retry = isinstance(ex, StartRefused) and ex.retry and (active or prefills) and not self.stopping
                    if retry and time.monotonic() - t_ref < START_RETRY_S:
                        admit.t_refused = t_ref
                        admit.n_refused = getattr(admit, "n_refused", 0) + 1
                        admit.slot = -1
                        if admit.n_refused == 1:  # logged once per request
                            try:
                                pi = e.pool_info()
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
            if self.stopping and prefills:  # shutdown: prompts being prefilled end now (stop_requests)
                for j in prefills:
                    j.fail(SequenceFault("server is shutting down"))
                prefills = []
            if not self.queue and not active and not prefills:
                continue
            e.set_lookahead(not prefills)  # no step queued ahead of a prefill chunk
            if prefills:
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
                idle = self.args.prefill_chunk_idle
                # the larger idle batch when nothing decodes: one prompt and nothing waiting, or several sharing it; on
                # 2 cards (pass_bound: a pass costs ~0.6 s of PCIe expert copies whatever its rows) also while requests
                # wait (8 x 16K burst 3.0K -> 4.4K tok/s; 3 cards: no gain)
                if idle > budget and not busy and ((len(prefills) == 1 and not self.queue) or
                                                   (adm.multi and (len(prefills) >= 2 or adm.pass_bound))):
                    budget = idle
                share = self.args.prefill_share if busy and self.args.prefill_chunk > 0 else 0.0
                try:
                    finished = []
                    if busy and time.time() < decode_until:  # the decoders' share of the GPU time
                        pass
                    else:
                        tc = time.time()
                        ready = None
                        self.beat("prefill")
                        if adm.multi and len(prefills) >= 2:  # one batch over the prompts in flight
                            order = adm.batch_order(tc, prefills)
                            ready = e.prefill_multi([j.slot for j in order], budget)
                            if ready is not None:
                                tr("prefill_multi", tc, len(order))
                                finished = [j for j in order if j.slot in ready]
                                self.sched["multi_batches"] = self.sched.get("multi_batches", 0) + 1
                            else:
                                cur = order[0]  # no shared batch (layout, image rows ...): the first one alone
                        if ready is None:
                            done = e.prefill_more(budget, cur.slot)
                            tr("prefill", tc, len(cur.ids))
                            self.cur_slot = cur.slot
                            if done:
                                finished = [cur]
                        tc = time.time() - tc
                        decode_until = time.time() + (tc * share / (1 - share) if 0 < share < 1 else 0)
                    if finished:
                        self.retry_gen += 1  # a prefill ended (in-flight limit, donor done)
                    for cur in finished:
                        tf = time.time()
                        try:
                            cur.finish_start()
                            tr("finish_start", tf)
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
                    with self.cv:
                        self.cv.wait(0.05)
                continue
            try:
                for j in active:
                    if j.finished:
                        continue
                    stop = bool(j.stops and STOP_ROLLBACK) and j.stop_risk()
                    near = j.tc is not None and j.tc.near()  # a forced close may come: nothing decoded ahead of it
                    if (j.stops and STOP_ROLLBACK) or j.tc is not None:
                        e.set_no_ahead(j.slot, stop or near)
                if len(active) == 1:
                    j = active[0]
                    if self.cur_slot != j.slot:
                        e.resume(j.slot)
                        self.cur_slot = j.slot
                    ts = time.time()
                    self.beat("step")
                    if STEP_PROF:
                        p0 = _pc()
                    toks = e.step()
                    if not toks:  # no token (paused, refused by the engine's slot check ...): resume explicitly next time
                        self.cur_slot = -1
                    if STEP_PROF:
                        p1 = _pc()
                    tr("step", ts, 1, len(toks))
                    self.metrics.step(active, [len(toks)], time.time() - ts)
                    if STEP_PROF:
                        p2 = _pc()
                    if self.faults(active) or j.finished:
                        continue
                    j.feed(toks)
                    self.no_progress(j, len(toks))
                    if STEP_PROF:
                        p3 = _pc()
                        gap = p0 - prof_last if prof_last else 0.0  # previous step's return -> this call
                        j.prof.append((p1 - p0, len(toks), p2 - p1, p3 - p2, gap, j.prof_detok, j.prof_emit))
                        j.prof_detok = j.prof_emit = 0.0
                        prof_last = p1
                else:
                    self.cur_slot = -1  # batch steps undo the queued-ahead single step
                    ts = time.time()
                    self.beat("batch step")
                    res = e.step_batch([j.slot for j in active])
                    tr("batch", ts, len(active), sum(len(r) for r in res))
                    self.metrics.step(active, [len(r) for r in res], time.time() - ts)
                    tf = time.time()
                    self.faults(active)
                    for j, toks in zip(active, res):
                        if not j.finished:
                            j.feed(toks)
                            self.no_progress(j, len(toks))
                    tr("feed", tf, len(active))
            except Exception as ex:
                for j in active:
                    j.fail(ex)
                active = []
            active = [j for j in active if not j.check_done()]


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
MULTI_MAX = 8  # prompts in flight that share prefill batches (the engine's FLASH_MAX_PREFILLS default)


class Job:
    """One generation request driven by the scheduler thread."""

    def __init__(self, gen, ids, p, on_text, cancelled, affinity):
        self.gen, self.ids, self.p, self.on_text, self.cancelled = gen, ids, p, on_text, cancelled
        self.affinity = affinity
        self.grammar, self.grammar_s, self.gsteps = None, 0.0, 0  # structured output (begin_start)
        self.glk = None  # TokenLookup of a constrained reply (grammar.py)
        self.done = threading.Event()
        self.error, self.stats, self.slot = None, None, -1
        self.penalty, self.guarded = None, 0  # (presence, frequency) in force; loop guard: repeated turns before
        self.finished = False
        self.t_enq, self.est, self.t_est = time.time(), None, 0.0  # scheduler: prefill-cost estimate
        self.cut_at = None  # prefill position at the last cut-in by a short request (fairness)
        self.round_tok = 0  # prompt tokens of the cut-in round in progress on this (long) prompt
        # eviction value of the conversation (engine prio) and its scheduling class: the main agent
        # thread (server/agents.py) is kept (it returns after its subagents) and goes first; a
        # subagent that ended its turn without a tool call is finished and goes first on eviction
        info = getattr(ids, "agent", None)
        self.prio = 2 if (gen.agents.is_main_now(info) if info is not None
                          else getattr(ids, "label", "") == "main agent") else 1
        self.prof, self.prof_detok, self.prof_emit = [], 0.0, 0.0  # QE_STEP_PROF

    def begin_start(self):
        g, e, p = self.gen, self.gen.engine, self.p
        temp = p.get("temperature")
        temp = g.args.temperature if temp is None else temp
        # top_p / top_k / min_p the client does not send (Claude Code sends none): the model's
        # recommended values (generation_config.json, --top-p / --top-k). Sampling at temperature
        # 1 without them (top_p 1, top_k 40) lets tail tokens break tool-call syntax.
        d = g.sampling_defaults
        self.max_new = p.get("max_tokens") or (e.max_t - len(self.ids) - 16)
        self.stops = [s for s in (p.get("stop") or []) if s]
        self.t0 = time.time()
        g.metrics.prefill_start(self)
        e.set_steering(*p.get("steering_sel", STEER_DEFAULT))  # this request's steering (keys the cached prefixes)
        self.slot = e.start_begin(self.ids, g.args.mtp_k, self.affinity, g.system_end(self.ids),
                                  getattr(self.ids, "mm", None))
        # after start_begin: the engine applies sampling parameters to its current slot, which
        # is this request's only now (before, they landed on whichever sequence was current)
        # presence / frequency penalty: the request's, else the server default for sampled requests (greedy ones
        # stay greedy: deterministic checks); the loop guard raises presence and seeds it with the repeated reply
        pres, freq = p.get("presence_penalty"), p.get("frequency_penalty")
        explicit = pres is not None or freq is not None
        pres = float(pres if pres is not None else (getattr(g.args, "presence_penalty", 0.0) if temp > 0 else 0.0))
        freq = float(freq if freq is not None else (getattr(g.args, "frequency_penalty", 0.0) if temp > 0 else 0.0))
        guard = getattr(self.ids, "guard", None)
        seed = []
        if guard is not None:
            pres = max(pres, guard["presence"])
            seed = loop_guard.last_reply_tokens(self.ids, g.im_start, g.asst_ids, g.im_end, g.special_min, g.ws_ids)
        top_p, top_k, min_p = p.get("top_p") or d["top_p"], p.get("top_k") or d["top_k"], p.get("min_p") or d["min_p"]
        if temp <= 0 and (pres or freq) and (explicit or guard is not None):
            temp, top_p, top_k, min_p = 1.0, 1.0, 1, 0.0  # greedy with a penalty: the penalised argmax
        e.set_sampling(temp, top_p, top_k, min_p, p.get("seed") or int(time.time() * 1e6))
        self.penalty = (pres, freq) if temp > 0 and (pres or freq) else None
        if self.penalty and not e.set_penalty(pres, freq, seed):
            self.penalty = None
        if guard is not None:
            e.set_no_lookup(self.slot, True)
            self.guarded = guard["turns"]
            g.guard_note(dict(guard, seed=len(seed)))
        e.set_prio(self.slot, self.prio)
        self.grammar = self.p.get("grammar")
        self.grammar_s = 0.0
        if self.grammar is not None:  # constrained from the first token on (prompt's last chunk)
            tg = time.perf_counter()
            self.grammar.fill(e.mask_buffer(self.slot)[0])
            e.constrain(self.slot, True)
            self.grammar_s += time.perf_counter() - tg

    def finish_start(self):
        """The prompt is prefilled: first token, then this job decodes with the others."""
        g, e = self.gen, self.gen.engine
        first, self.reused, self.slot = e.start_finish(self.slot)
        self.ds0 = g.dstats.read(self.slot)  # draft counters of the slot (engine_stats: the per-request stats line)
        # max_tokens in the engine: an MTP step accepting drafts past it would leave tokens in the
        # slot that the reply does not have, and the next turn would miss the live end (resume
        # from the prompt-end checkpoint and recompute the whole reply)
        e.set_limit(self.slot, self.max_new - 1)
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
        self.det = Detok(g.tok)
        self.n, self.finish, self.text_all, self.cut = 0, "length", "", False
        self.loop_checked = 0
        self.tc = None  # thinking state / forced close (server/think_budget.py)
        if g.think_ids.ok and g.think_ids.open_at_end(self.ids, g.im_start):
            self.tc = ThinkControl(self, self.p.get("think_budget"), g.think_ids, g.can_force)
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
            pending = ok
            self.gsteps += 1
            if self.glk is None:  # token history for lookup drafts (prompt + emitted so far)
                self.glk = TokenLookup(list(self.ids) + list(self.det.ids))
            self.glk.append(ok)
            if not gdone:
                # drafts for the next step: forced tokens / lookup / the grammar-valid model drafts
                e = g.engine
                c.plan(e.spec_io(self.slot), e.mask_buffer(self.slot), g.args.mtp_k, self.glk)
            self.grammar_s += time.perf_counter() - tg
        for i, t in enumerate(pending):
            if t in g.eos and not self.p.get("ignore_eos"):
                self.finish, stop_now = "stop", True
                pending = pending[:i]
                break
            if t == g.im_start and not self.p.get("ignore_eos"):
                # <|im_start|> is never valid inside a reply: the model skipped <|im_end|> and began a
                # turn of its own (seen at 130K+ contexts). End the turn here: the client gets the reply
                # so far, cleanly closed, instead of a hallucinated user / tool turn.
                print(f"stopping at a generated <|im_start|> after {self.n + i} tokens", flush=True)
                self.finish, stop_now = "stop", True
                pending = pending[:i]
                break
        self.n += len(pending)
        if self.tc is not None:
            self.tc.track(pending)
        if STEP_PROF:
            q0 = _pc()
        s = self.det.push(pending) if pending else ""
        if STEP_PROF:
            self.prof_detok += _pc() - q0
        if s:
            self.text_all += s
            hit = next((x for x in self.stops if x in self.text_all[-len(s) - 64:]), None)
            if hit:
                cut = self.text_all.rfind(hit)
                if STOP_ROLLBACK:
                    self.keep = self.kept_tokens(self.text_all[:cut])
                s = s[: max(0, len(s) - (len(self.text_all) - cut))]
                self.finish, stop_now = "stop", True
                self.cut = True
            if s:
                if STEP_PROF:
                    q0 = _pc()
                self.on_text(s)
                if STEP_PROF:
                    self.prof_emit += _pc() - q0
        if not stop_now and g.args.loop_guard_tokens and not self.p.get("ignore_eos") and self.n - self.loop_checked >= 64:
            self.loop_checked = self.n
            p = repetition_period(self.det.ids, min_span=g.args.loop_guard_tokens)
            if p:  # degenerate loop: end the turn instead of streaming up to max_tokens of it
                print(f"stopping a repetition loop (period {p} tokens) after {self.n} tokens", flush=True)
                self.finish, stop_now = "stop", True
        if gdone and not stop_now:
            self.finish, stop_now = "stop", True
        c = self.grammar
        if stop_now and c is not None and not self.cut and (c.error is not None or not (
                c.done() or (self.finish == "stop" and c.accepting()))):
            # structured output ended before its grammar was complete (loop guard, rejected token):
            # the JSON is cut off, so say "length" (clients retry / raise) and never "stop"
            why = c.error.splitlines()[0] if c.error else "loop guard"
            print(f"structured output: incomplete after {self.n} tokens ({why}), finish_reason length",
                  flush=True)
            self.finish = "length"
        if (stop_now or self.n >= self.max_new or
                len(self.ids) + self.n + 8 >= g.engine.slots[self.slot]):
            self.finished = True
        elif self.tc is not None:
            self.tc.after(pending)  # a forced close of <think> in flight, or one due now (budget / answer tail)

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

    def check_done(self):
        """Release the slot and wake the caller once finished (or cancelled)."""
        if not self.finished and (self.cancelled() or self.gen.stopping):
            self.shut = self.gen.stopping and not self.cancelled()  # shutdown: run() raises (error event)
            self.finished = True
        if not self.finished:
            return False
        if not self.done.is_set():
            if self.prio == 1 and self.finish == "stop" and "<tool_call>" not in self.text_all:
                self.gen.engine.set_prio(self.slot, 0)  # a subagent that is done (no tool to run)
            if getattr(self, "det", None) is not None:  # its final reply is recognised in a parent's history
                self.gen.agents.finished(getattr(self.ids, "agent", None), self.text_all,
                                         "<tool_call>" in self.text_all)
            ds = engine_stats.delta(getattr(self, "ds0", None), self.gen.dstats.read(self.slot))
            trimmed = None
            if getattr(self, "keep", None) is not None:
                trimmed = self.gen.engine.trim(self.slot, len(self.ids) + self.keep)
            self.gen.engine.release(self.slot)
            if getattr(self, "det", None) and not getattr(self.ids, "mm", None) and (
                    not self.cut or getattr(self, "keep", None) is not None):
                out = self.det.ids if not self.cut else self.det.ids[:self.keep]
                self.gen.remember(self.affinity, self.ids, out)
                self.gen.remember_turn(self.ids, out)
                self.gen.engine.add_corpus(out)
            t2 = time.time()
            self.stats = {"prompt_tokens": len(self.ids), "cached_tokens": self.reused,
                          "cache_src": self.src,
                          "completion_tokens": self.n, "finish": self.finish, "slot": self.slot,
                          "prefill_s": self.t1 - self.t0, "decode_s": t2 - self.t1}
            if ds is not None:
                self.stats["drafts"] = ds  # [steps, MTP drafted, accepted, lookup drafted, accepted]
            if trimmed is not None:
                self.stats["stop_trim"] = trimmed
            if self.penalty:
                self.stats["penalty"] = self.penalty
            if self.guarded:
                self.stats["loop_guard"] = self.guarded
            tc = getattr(self, "tc", None)
            if tc is not None:
                self.stats["think"] = tc.stats()
                g = self.gen
                g.think_add("requests_with_budget", tc.budget is not None)
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
            if self.grammar is not None:
                self.stats["grammar_s"] = self.grammar_s
                self.stats["grammar_steps"] = self.gsteps
                self.stats["grammar_forced"] = self.grammar.n_forced
            self.gen.metrics.finished(self)
            if STEP_PROF and self.prof:
                prof_write(self)
            self.done.set()
        return True

    def fail(self, ex):
        if self.slot >= 0:
            try:
                self.gen.engine.release(self.slot)
            except Exception:
                pass
        self.error = ex
        self.finished = True
        self.gen.metrics.finished(self, ex)
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


def stream_bridge(gen, ids, params, affinity=0):
    """Run generation in a thread; yields text chunks then the stats dict (async)."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    cancel = threading.Event()

    def work():
        try:
            st = gen.run(ids, params, lambda s: loop.call_soon_threadsafe(q.put_nowait, s),
                         cancel.is_set, affinity)
            loop.call_soon_threadsafe(q.put_nowait, st)
        except Exception as ex:  # surface errors to the client
            loop.call_soon_threadsafe(q.put_nowait, ex)

    threading.Thread(target=work, daemon=True).start()

    async def agen():
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                yield item
                if isinstance(item, dict):
                    return
        finally:
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


async def complete(gen, ids, params, aff, thinking, tools):
    """A non-streamed reply: (parser, parts, stats, tool calls), or the SequenceFault of an engine error. A sampled
    reply that ends by itself (finish stop) with no text and no tool call -- reasoning only (EOS inside or right after
    <think>, HF Qwen3.8-Flash-Next #50) or a tool call that does not parse -- is sampled again, up to --empty-retry
    times: agent harnesses (AutomationBench, verifiers, Claude Code) end the task on such a turn, so one bad sample
    costs the whole task. The prompt is cached, so a retry costs only its decode. Off for greedy (temperature 0) and
    seeded requests (a resample would repeat the output or break reproducibility), structured output (single-use
    grammar), ignore_eos and client stop sequences (the stop may have cut the text). Streams are never retried: their
    bytes are already sent. usage.completion_tokens counts the tokens of every try; finish_reason is the last try's."""
    tries = max(0, getattr(gen.args, "empty_retry", 0))
    temp = params.get("temperature")
    temp = gen.args.temperature if temp is None else temp
    if (params.get("grammar") is not None or temp <= 0 or params.get("seed") is not None or params.get("ignore_eos")
            or params.get("stop")):
        tries = 0
    spent, spent_s, first = 0, 0.0, None
    for attempt in range(tries + 1):
        parser = Parser(thinking)
        parts = {"reasoning": "", "content": ""}
        st = None
        p = dict(params, empty_retry=(attempt, tries)) if tries else params  # the empty-reply WARNING says retry k/N
        async for item in fault_events(stream_bridge(gen, ids, p, aff)):
            if isinstance(item, SequenceFault):
                return item
            if isinstance(item, dict):
                st = item
                for kind, t in parser.feed("", final=True):
                    parts[kind] += t
            else:
                for kind, t in parser.feed(item):
                    parts[kind] += t
        if first is None:
            first = st
        calls = parse_tool_calls(parser.tool_text, tools) if parser.tool_text else []
        empty = not calls and not parts["content"].strip() and st["finish"] == "stop"
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
            raw = diag_dump("tool", parser.tool_text)
            print(f"WARNING empty reply: a tool call that does not parse ({st['completion_tokens']} tokens"
                  f"{', raw: ' + raw if raw else ''}); retry {attempt + 1}/{tries} follows", flush=True)
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
    b = f"/{th['budget']}" if th["budget"] is not None else ""
    f = f", forced close ({th['forced']}, {th['forced_tokens']} tokens)" if th["forced"] else ""
    return f" | think {th['tokens']}{b}{f}{' FAILED' if th.get('failed') else ''}"


def log_stats(kind, st):
    tps = st["completion_tokens"] / max(st["decode_s"], 1e-6)
    pps = (st["prompt_tokens"] - st["cached_tokens"]) / max(st["prefill_s"], 1e-6)
    src = f" from {st['cache_src']}" if st.get("cache_src") else ""
    gr = (f" | grammar {1e6 * st['grammar_s'] / max(st['completion_tokens'], 1):.0f} us/token, "
          f"{st['completion_tokens'] / max(st['grammar_steps'], 1):.2f} tok/step, "
          f"{st['grammar_forced']} forced drafts" if "grammar_s" in st else "")
    print(f"[{kind}] slot {st['slot']} prompt {st['prompt_tokens']} (cached {st['cached_tokens']}{src}, prefill "
          f"{pps:.0f} tok/s) | gen {st['completion_tokens']} @ {tps:.1f} tok/s | {st['finish']}{gr}"
          f"{' | stop trim ' + ('ok' if st['stop_trim'] else 'missed') if 'stop_trim' in st else ''}{think_note(st)}",
          flush=True)
    engine_stats.log(st, st.get("drafts"))  # [stats] lines for benchmark log parsers (QE_STATS_LOG=0: off)


# ----------------------------------------------------------------------------- request mapping


# reasoning efforts qwen38_syssafe.jinja knows: 'high' renders as 'xhigh', 'medium' adds no instruction, and
# any other value ('minimal', 'max', 'none', typos) renders as 'medium' (the model's own template raises instead)
KNOWN_EFFORTS = ("xhigh", "high", "medium", "low")


def template_kwargs(body, default_think, default_effort=None):
    """Template variables of a request. Effort: the request's chat_template_kwargs.reasoning_effort, else its
    top-level reasoning_effort (OpenAI field, also read on the Anthropic path), else --reasoning-effort
    (default_effort; None = the template's own default, xhigh). A null value counts as not sent."""
    kw = dict(body.get("chat_template_kwargs") or {})
    th = body.get("thinking")
    if isinstance(th, dict) and th.get("type") == "disabled":
        kw.setdefault("enable_thinking", False)
    if kw.get("reasoning_effort") is None:
        kw.pop("reasoning_effort", None)
        eff = body.get("reasoning_effort")
        if eff is None:
            eff = default_effort
        if eff is not None:
            kw["reasoning_effort"] = eff
    kw.setdefault("enable_thinking", default_think)
    return kw


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
    return gen.grammars.matcher(schema)


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
                "modalities": {"vision": gen.vision is not None}, **lim}

    # ---------------- OpenAI chat completions ----------------
    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        tools = body.get("tools")
        kw = template_kwargs(body, args.think, args.reasoning_effort)
        try:
            budget = request_budget(body, kw, args.thinking_budget)
        except BudgetError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error",
                                           "param": "thinking_budget"}}, status_code=400)
        try:
            grammar = structured(gen, body, kw)
        except GrammarError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error",
                                           "param": "response_format"}}, status_code=400)
        try:
            steer = resolve_steering(gen.steer, body)
        except ValueError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error",
                                           "param": "steering"}}, status_code=400)
        try:
            ids, aff = gen.prompt(openai_messages(body["messages"]), tools, kw, body.get("metadata"), track=True,
                                  guard_on=body.get("loop_guard") is not False)
        except InputError as ex:
            return JSONResponse({"error": {"message": str(ex), "type": "invalid_request_error", "param": "messages"}},
                                status_code=400)
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
        stop = body.get("stop")
        params = {"temperature": body.get("temperature"), "top_p": body.get("top_p"),
                  "top_k": body.get("top_k"), "min_p": body.get("min_p"), "seed": body.get("seed"),
                  "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
                  "stop": [stop] if isinstance(stop, str) else stop,
                  "ignore_eos": bool(body.get("ignore_eos")),  # benchmarks (vllm bench serve --ignore-eos)
                  "grammar": grammar, "steering_sel": steer, "think_budget": budget,
                  "presence_penalty": body.get("presence_penalty"), "frequency_penalty": body.get("frequency_penalty")}
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        parser = Parser(kw.get("enable_thinking", True))

        def tool_calls_out(calls):
            return [{"index": i, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for i, c in enumerate(calls)]

        def usage(st):
            u = {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
                 "total_tokens": st["prompt_tokens"] + st["completion_tokens"],
                 "prompt_tokens_details": {"cached_tokens": st["cached_tokens"]}}
            if st.get("think"):
                u["completion_tokens_details"] = {"reasoning_tokens": st["think"]["tokens"]}
            return u

        if body.get("stream"):
            async def sse():
                def chunk(delta, finish=None, extra=None):
                    d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if extra:
                        d.update(extra)
                    return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                started = False
                sse_s = [0.0, 0]  # QE_STEP_PROF: event-loop time of parser + chunk JSON, chunks
                async for item in fault_events(stream_bridge(gen, ids, params, aff)):
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
                        yield chunk({"role": "assistant", "content": ""})
                    if isinstance(item, dict):
                        st = item
                        for kind, t in parser.feed("", final=True):
                            yield chunk({"reasoning_content" if kind == "reasoning" else "content": t})
                        calls = parse_tool_calls(parser.tool_text, tools) if parser.tool_text else []
                        if calls:
                            yield chunk({"tool_calls": tool_calls_out(calls)})
                        finish = "tool_calls" if calls else st["finish"]
                        yield chunk({}, finish)
                        # usage in its own chunk with empty choices (OpenAI stream_options.include_usage)
                        yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                                     "model": model_name, "choices": [], "usage": usage(st)}) + "\n\n"
                        yield "data: [DONE]\n\n"
                        log_stats("chat", st)
                        if STEP_PROF and sse_s[1]:
                            print(f"[prof] sse: {sse_s[1]} chunks, parser + JSON {sse_s[0] / sse_s[1] * 1e6:.0f} us/chunk "
                                  f"(event-loop thread)", flush=True)
                        return
                    if STEP_PROF:
                        a0 = _pc()
                    for kind, t in parser.feed(item):
                        c = chunk({"reasoning_content" if kind == "reasoning" else "content": t})
                        if STEP_PROF:
                            sse_s[0] += _pc() - a0
                            sse_s[1] += 1
                        yield c
                        if STEP_PROF:
                            a0 = _pc()
            return StreamingResponse(sse(), media_type="text/event-stream")

        r = await complete(gen, ids, params, aff, kw.get("enable_thinking", True), tools)
        if isinstance(r, SequenceFault):
            if bad_input(r):
                return JSONResponse({"error": {"message": str(r), "type": "invalid_request_error"}}, status_code=400)
            return JSONResponse({"error": {"message": f"engine error: {r}", "type": "server_error"}},
                                status_code=500)
        parser, parts, st, calls = r
        msg = {"role": "assistant", "content": parts["content"].strip() or (None if calls else "")}
        if parts["reasoning"]:
            msg["reasoning_content"] = parts["reasoning"].strip()
        if calls:
            msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in tool_calls_out(calls)]
        log_stats("chat", st)
        return {"id": rid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "tool_calls" if calls else st["finish"]}],
                "usage": usage(st)}

    # ---------------- Anthropic messages ----------------
    @app.post("/v1/messages/count_tokens")
    async def count_tokens(req: Request):
        body = await req.json()
        try:
            kw = template_kwargs(body, args.think, args.reasoning_effort)
            request_budget(body, kw)  # (budget 0: rendered without thinking, as the request would be)
            ids, _ = gen.prompt(anthropic_to_messages(body), anthropic_tools(body.get("tools")), kw)
        except (InputError, BudgetError) as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        return {"input_tokens": len(ids)}

    @app.post("/v1/messages")
    async def messages(req: Request):
        body = await req.json()
        if env_str("QE_DUMP_PROMPTS"):
            with open(os.path.join(env_str("QE_DUMP_PROMPTS"), f"{time.time():.3f}.json"), "w") as f:
                json.dump(body, f, ensure_ascii=False, indent=1)
        tools = anthropic_tools(body.get("tools"))
        kw = template_kwargs(body, args.think, args.reasoning_effort)
        try:
            budget = request_budget(body, kw, args.thinking_budget)
        except BudgetError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        try:
            grammar = structured(gen, body, kw)
        except GrammarError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        try:
            steer = resolve_steering(gen.steer, body)
        except ValueError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        try:
            msgs = anthropic_to_messages(body)
            prev = dict(msgs[-1]) if msgs else None  # before prompt() rewrites image parts
            ids, aff = gen.prompt(msgs, tools, kw, body.get("metadata"), track=True,
                                  guard_on=body.get("loop_guard") is not False)
        except InputError as ex:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(ex)}},
                                status_code=400)
        if gen.too_long(ids):
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": gen.too_long(ids)}},
                                status_code=400)
        if body.get("loop_guard") is False:  # per request off-switch of the cross-turn loop guard
            ids.guard = None
        params = {"temperature": body.get("temperature"), "top_p": body.get("top_p"),
                  "top_k": body.get("top_k"), "max_tokens": body.get("max_tokens"),
                  "stop": body.get("stop_sequences"), "grammar": grammar, "steering_sel": steer, "think_budget": budget,
                  "presence_penalty": body.get("presence_penalty"), "frequency_penalty": body.get("frequency_penalty")}
        mid = "msg_" + uuid.uuid4().hex[:24]
        parser = Parser(kw.get("enable_thinking", True))
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

                async for item in fault_events(stream_bridge(gen, ids, params, aff)):
                    if isinstance(item, SequenceFault):  # the engine failed this request: an error event, then the end
                        yield ev("error", {"type": "error", "error": {
                            "type": "invalid_request_error" if bad_input(item) else "api_error",
                            "message": str(item) if bad_input(item) else f"engine error: {item}"}})
                        return
                    if isinstance(item, dict):
                        st = item
                        for kind, t in parser.feed("", final=True):
                            for x in emit(kind, t):
                                yield x
                        if state["kind"] is not None:
                            yield ev("content_block_stop", {"type": "content_block_stop", "index": state["idx"]})
                        calls = parse_tool_calls(parser.tool_text, tools) if parser.tool_text else []
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
                        yield ev("message_delta", {"type": "message_delta",
                                                   "delta": {"stop_reason": reason, "stop_sequence": None},
                                                   "usage": usage(st)})
                        yield ev("message_stop", {"type": "message_stop"})
                        log_stats("messages", st)
                        return
                    for kind, t in parser.feed(item):
                        for x in emit(kind, t):
                            yield x
            return StreamingResponse(sse(), media_type="text/event-stream")

        r = await complete(gen, ids, params, aff, kw.get("enable_thinking", True), tools)
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
        return {"id": mid, "type": "message", "role": "assistant", "model": model_name, "content": content,
                "stop_reason": "tool_use" if calls else stop_map.get(st["finish"], "end_turn"),
                "stop_sequence": None, "usage": usage(st)}

    return app


def resolve_model_args(args):
    """Fill the per-model defaults (None on the command line) from the model profile."""
    prof = args.profile = Profile(args.model)
    d = prof.defaults
    if args.served_name is None:
        args.served_name = d["served_name"]
    if args.mtp_k is None:
        args.mtp_k = d["mtp_k"]
    if args.devices is None:
        vis = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        args.devices = ",".join(str(i) for i in range(len(vis))) if vis else "0,1,2"
    if args.split is None:
        # the profile's split is per GPU (Flash-Next: 3 values); another GPU count uses the engine's default
        sp = d["split"] or ""
        args.split = sp if len([x for x in sp.split(",") if x]) in (0, len(args.devices.split(","))) else ""
    if args.prefill_chunk_idle is None:
        args.prefill_chunk_idle = d.get("prefill_chunk_idle", 0)
    for k in ("drop_empty_turns", "merge_system"):
        v = getattr(args, k)
        setattr(args, k, bool(d[k]) if v is None else bool(v))
    if not args.engine_lib and not env_str("QE_LIB"):
        lib = os.path.join(ROOT, d["lib"])
        args.engine_lib = lib if os.path.exists(lib) else os.path.join(ROOT, "build", "libqengine.so")
    if args.vision and not d["vision"]:
        print(f"vision is not available for {prof.family} yet: --vision ignored", flush=True)
        args.vision = 0
    return args


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen3.8-27B-INT4"))
    default_tmpl = os.path.join(ROOT, "server/templates/qwen38_syssafe.jinja")
    ap.add_argument("--template", default=default_tmpl if os.path.exists(default_tmpl) else None,
                    help="chat template (default: server/templates/qwen38_syssafe.jinja, else the model's own)")
    ap.add_argument("--devices", default=None,
                    help="GPU indices within CUDA_VISIBLE_DEVICES (default: every visible GPU, else 0,1,2); "
                         "Flash-Next: 2 or 3 cards: tp; 4+ cards: pp (pipeline stages; see the README)")
    def ctx_arg(v):  # 262144, 256K, 1M (K = 1024 tokens)
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", v)
        if not m:
            raise argparse.ArgumentTypeError("a token count (262144, 512K, 1M)")
        return int(float(m.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[m.group(2).lower()])
    ap.add_argument("--max-ctx", type=ctx_arg, default=262144,
                    help="context per conversation (default 262144); Flash-Next above 262144 (up to 1M) turns on "
                         "YaRN RoPE scaling and needs FLASH_KV=ram")
    ap.add_argument("--mtp-k", type=int, default=None,
                    help="max MTP drafts per step (adaptive; default per model: 27B 7, Flash-Next 3)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default when request has none (27B 0.6, Flash-Next generation_config 1.0)")
    ap.add_argument("--top-p", type=float, default=None, help="default when request has none (generation_config)")
    ap.add_argument("--top-k", type=int, default=None, help="default when request has none (generation_config)")
    ap.add_argument("--min-p", type=float, default=None, help="default when request has none (generation_config)")
    ap.add_argument("--mtp-file", default=None,
                    help="MTP head weights for the engine (sets $QE_MTP_FILE; default: the engine's own choice, "
                         "for 27B model_mtp_ft.safetensors when present, else downloaded once from the latest "
                         "release into ~/.cache/qengine/assets; QE_FETCH_ASSETS=0: no download)")
    ap.add_argument("--think", type=int, default=1, help="default enable_thinking")
    ap.add_argument("--reasoning-effort", default=env_str("QE_REASONING_EFFORT"),
                    help="default reasoning effort for requests that send none (xhigh, high = xhigh, medium, low; "
                         "a request's reasoning_effort or chat_template_kwargs.reasoning_effort wins; other values "
                         "render as medium); env QE_REASONING_EFFORT; default: unset = the template's own (xhigh)")
    ap.add_argument("--thinking-budget", type=int, default=env_int("QE_THINKING_BUDGET", 0),
                    help="default thinking budget in tokens for requests that send none (0 = off): at the budget the "
                         "server closes <think> with a short wrap-up and the model answers; a request's "
                         "thinking.budget_tokens (Anthropic) / thinking_budget (OpenAI, also in chat_template_kwargs; "
                         "aliases thinking_token_budget, max_thinking_tokens, custom_params.thinking_budget, "
                         "reasoning.max_tokens) wins, a negative one turns it off; env QE_THINKING_BUDGET")
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
    ap.add_argument("--prefill-chunk", type=int, default=2048,
                    help="prompt chunk (tokens) between decode bursts of the running sequences "
                         "(0: a prompt is prefilled at once, the others wait)")
    ap.add_argument("--prefill-chunk-idle", type=int, default=None,
                    help="prompt chunk (tokens) while no other sequence decodes or waits (default per "
                         "model: Flash-Next 4096 = its big prefill batch, 27B 0 = --prefill-chunk)")
    ap.add_argument("--prefill-share", type=float, default=0.5,
                    help="share of the GPU time the running sequences keep while another prompt "
                         "is prefilled (0.5: they decode at ~half their speed, the prefill takes "
                         "~2x longer; with the INT8 prefill, 8 agents + a 30K prompt: TTFT 11.5 s "
                         "as with 0.3 before it, at 69 instead of 41 tok/s per agent)")
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
    ap.add_argument("--preempt-max", type=int, default=1024,
                    help="a queued request with at most this many tokens to prefill (a tool turn "
                         "of a cached conversation) is started between the chunks of a long "
                         "prefill in progress (0 = off)")
    ap.add_argument("--preempt-min", type=int, default=8192,
                    help="only a prefill with at least this many tokens left is cut in on, and "
                         "at most once per this many tokens of its progress")
    ap.add_argument("--main-bonus", type=float, default=200000,
                    help="queue order: the main agent thread counts as this many tokens less prefill work "
                         "(it goes first; a subagent passes it only after waiting main-bonus / prefill-aging "
                         "seconds; 0 = no precedence)")
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
    ap.add_argument("--presence-penalty", type=float, default=env_float("QE_PRESENCE_PENALTY", 0.0),
                    help="presence penalty of sampled requests that send none (Claude Code's Anthropic API never "
                         "does); env QE_PRESENCE_PENALTY")
    ap.add_argument("--frequency-penalty", type=float, default=env_float("QE_FREQUENCY_PENALTY", 0.0),
                    help="frequency penalty of sampled requests that send none; env QE_FREQUENCY_PENALTY")
    ap.add_argument("--loop-guard-turns", type=int, default=env_int("QE_LOOP_GUARD_TURNS", 3),
                    help="cross-turn loop guard: after N consecutive identical (reply, result) turns of a "
                         "conversation the next turn runs with --loop-guard-penalty on the repeated reply's tokens "
                         "and no lookup drafts (0 = off); env QE_LOOP_GUARD_TURNS")
    ap.add_argument("--loop-guard-penalty", type=float,
                    default=env_float("QE_LOOP_GUARD_PENALTY", 1.5),
                    help="presence penalty of a guarded turn (2.0 from 2N repeated turns); env QE_LOOP_GUARD_PENALTY")
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
    ap.add_argument("--stats-file", default="~/.cache/qengine/stats.json",
                    help="all-time monitoring totals (dashboard)")
    ap.add_argument("--disk-cache-gb", type=float, default=200)
    ap.add_argument("--persist-idle", type=float, default=120,
                    help="seconds idle before live conversations are saved to the disk cache")
    ap.add_argument("--log-file", default=env_str("QE_LOG_FILE", "auto"),
                    help="persistent log: the server's whole output (also the engine's) is appended to this file with "
                         "a timestamp per line and survives crashes and restarts (auto: "
                         "~/.cache/qengine/logs/<served-name>.log; '' or 0: off); env QE_LOG_FILE (0: off, empty: unset)")
    ap.add_argument("--log-mb", type=float, default=env_float("QE_LOG_MB", 64.0),
                    help="size of the persistent log before it is rotated (3 old files are kept); env QE_LOG_MB")
    args = ap.parse_args(argv)
    resolve_model_args(args)
    if not args.template:
        args.template = os.path.join(args.model, "chat_template.jinja")
    args.think = bool(args.think)
    if args.reasoning_effort is not None:
        args.reasoning_effort = args.reasoning_effort.strip().lower() or None  # '' = unset, like the env var
    if args.reasoning_effort is not None and args.reasoning_effort not in KNOWN_EFFORTS:
        print(f"[server] warning: --reasoning-effort {args.reasoning_effort!r} is not one of "
              f"{', '.join(KNOWN_EFFORTS)}: the template renders it as medium", flush=True)
    # the MTP head fine-tuned on the engine's own answers: +8% accepted drafts
    ft = os.path.join(args.model, "model_mtp_ft.safetensors")
    if args.mtp_file:
        os.environ["QE_MTP_FILE"] = args.mtp_file
    elif env_str("QE_MTP_FILE") is None and os.path.exists(ft):
        os.environ["QE_MTP_FILE"] = ft
    if args.disk_cache:
        os.environ["QE_DISK_CACHE"] = args.disk_cache
        os.environ["QE_DISK_CACHE_GB"] = str(args.disk_cache_gb)
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
    try:
        logtee.install(path, args.log_mb, 3, f"==== qengine server start: QEngine {version}, pid {os.getpid()}, build {build} "
                                             f"({os.path.realpath(ROOT)}), log {path}\n"
                                             f"argv: {' '.join(sys.argv)}\nenv: {env}")
    except Exception as ex:  # never keep the server from starting
        print(f"persistent log off: {ex}", flush=True)


def main():
    args = parse_args()
    start_log(args)
    print("QEngine %s (build %s)" % version_info(args.engine_lib or env_str("QE_LIB")), flush=True)
    if not args.profile.is_flash:  # 27B: tuned MTP head + draft vocabulary from the latest release if missing
        import assets
        assets.ensure(args.model)
    try:  # kill -USR1 <pid>: Python stacks of every thread (the native ones: gdb -p <pid>)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass
    gen = Generator(args)
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
        signal.signal(sig, lambda n, f: os._exit(128 + n))
    gen.metrics.save()
    # uvicorn returns after SIGTERM / SIGINT: save the live conversations before exiting
    if args.disk_cache:
        t0 = time.time()
        gen.run_on_engine(gen.engine.persist)
        print(f"shutdown: slots saved in {time.time() - t0:.1f}s", flush=True)
    gen.engine.set_exit_hook(None)
    # end here, as the re-raised signal did before: no interpreter / CUDA teardown while engine threads still run
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
