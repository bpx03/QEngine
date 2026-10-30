"""Which conversation is the main agent thread (scheduler priority, KV eviction value, monitoring).

The decision does not depend on the client's prompt wording. Layers, strongest first:

  override   QE_MAIN_AGENT_RE / --main-agent-re: a regex searched in the system prompt and in the tool names;
             when set, it alone decides (a match is the main thread, anything else is not)
  behaviour  requests are grouped by session (Anthropic metadata.user_id -> JSON session_id; without it the
             system-prompt hash) and, within a session, into conversations by the hash chain of their messages:
             a request whose message chain extends a known request's chain continues that conversation; one
             that shares only its opening messages is a branch of it (a fork, a summary / compaction task).
             Conversations with the same opening form a group. A group whose request carries the final reply of
             another conversation (a subagent's / fork's result coming back as a tool result) is a parent; the
             main group is the parent nobody consumed (tier 3). Without such evidence: the first group of the
             session that kept going for at least two turns (tier 1).
  structure  until behaviour says more: a Claude Code-like system prompt together with a
             tool that spawns agents ('Agent', 'Task') makes the earliest such group the main one (tier 2)

The choice is sticky: it moves to another group only for a stronger tier, or when that group consumed the output
of the current main group (the main guess was a subagent, or a compaction handed the session over to a new
conversation). Inside the main group the main conversation is the one with the most turns that is no side task
(a request adding only a user message to the tip of a history: a fork's directive, a summary) and that did not
deliver its final reply to another conversation.

alarm: a session with a client session id that saw `alarm_turns` requests without a main thread is reported once
(log + stats()['alarm']), so a client change that breaks the detection shows up in qetop."""
import hashlib
import json
import os
import re
import threading
import time

from qe_env import env_int, env_str

SPAWN_TOOLS = ("Agent", "Task")
CLI_RE = re.compile(r"(official cli|claude agent sdk|claude agent, built on|claude code)", re.I)
WS = re.compile(r"\s+")
SNIP = 96        # characters (whitespace removed) of a final reply kept to recognise it in a parent's history
SNIP_MIN = 24    # shorter replies ("Done.") are not distinctive enough


def _h(b):
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def _strip(x):
    """Message content without volatile parts: cache markers, image bytes."""
    if isinstance(x, dict):
        return {k: (len(v) if k in ("_src", "data") and isinstance(v, (str, bytes)) else _strip(v))
                for k, v in x.items() if k not in ("cache_control",)}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def msg_hash(m):
    """Hash of one message: role + content (+ tool calls / tool result id), whitespace-insensitive; the
    reasoning is left out (clients trim it or the server fills it from its cache)."""
    d = {"r": m.get("role"), "c": _strip(m.get("content"))}
    if m.get("tool_calls"):
        d["t"] = _strip(m["tool_calls"])
    if m.get("tool_call_id"):
        d["i"] = m["tool_call_id"]
    s = json.dumps(d, ensure_ascii=False, sort_keys=True, default=str)
    return _h(WS.sub("", s).encode())


def chain(messages):
    """Prefix hashes: chain[i] identifies messages[0..i]."""
    out, h = [], 0
    for m in messages:
        h = _h(h.to_bytes(8, "little") + msg_hash(m).to_bytes(8, "little"))
        out.append(h)
    return out


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(text_of(p.get("text", p.get("content", ""))) if isinstance(p, dict) else str(p) for p in c)
    if isinstance(c, dict):
        return text_of(c.get("text", c.get("content", "")))
    return "" if c is None else str(c)


def norm(s):
    return WS.sub("", s or "")


def session_of(meta, affinity):
    """Client session id (Claude Code: metadata.user_id is a JSON string with session_id; older clients put
    '..._session_<uuid>' there), else None (the caller groups by the system-prompt hash)."""
    uid = (meta or {}).get("user_id") if isinstance(meta, dict) else None
    if isinstance(uid, str) and uid:
        try:
            d = json.loads(uid)
            if isinstance(d, dict) and d.get("session_id"):
                return "s:" + str(d["session_id"])
        except ValueError:
            pass
        m = re.search(r"session_([0-9a-f-]{8,})", uid)
        if m:
            return "s:" + m.group(1)
    sid = (meta or {}).get("session_id") if isinstance(meta, dict) else None
    if sid:
        return "s:" + str(sid)
    return None


class Conv:
    """One conversation (a chain of requests that extend each other)."""
    __slots__ = ("id", "group", "chain", "turns", "t0", "t_last", "branch", "side", "consumed", "final")

    def __init__(self, cid, group, ch, now, branch, side=False, turns=1):
        self.id, self.group, self.chain, self.turns = cid, group, ch, turns
        self.t0 = self.t_last = now
        self.branch = branch      # forked off another conversation of its group
        self.side = side          # a side task on another conversation's history (only a new user message added)
        self.consumed = False     # its final reply came back in another conversation
        self.final = None         # snippet of its last final reply (no tool call)


class Group:
    """Conversations with the same opening (system prompt + first message)."""
    __slots__ = ("key", "t0", "turns", "convs", "hint", "parent_t", "consumed_by")

    def __init__(self, key, now):
        self.key, self.t0, self.turns, self.convs = key, now, 0, []
        self.hint = False         # Claude Code-like system prompt + an agent-spawning tool
        self.parent_t = 0.0       # last time one of its requests carried another conversation's final reply
        self.consumed_by = None   # group that consumed one of its (non-branch) conversations' final reply


class Session:
    __slots__ = ("key", "client", "groups", "convs", "main", "tier", "n", "alarmed", "t_last", "switches")

    def __init__(self, key, client):
        self.key, self.client = key, client
        self.groups, self.convs = {}, []
        self.main, self.tier, self.n, self.alarmed, self.t_last, self.switches = None, 0, 0, False, 0.0, 0


class Info:
    """What the tracker says about one request (kept on the prompt ids; finished() takes it back)."""
    __slots__ = ("main", "why", "conv", "session", "group")

    def __init__(self, main=False, why="", conv=None, session=None, group=None):
        self.main, self.why, self.conv, self.session, self.group = main, why, conv, session, group


class Tracker:
    def __init__(self, override=None, alarm_turns=None, max_sessions=64, max_convs=256, log=print):
        override = env_str("QE_MAIN_AGENT_RE", "") if override is None else override
        self.override = re.compile(override, re.I | re.M) if override else None
        self.alarm_turns = env_int("QE_MAIN_ALARM_TURNS", 8) if alarm_turns is None else alarm_turns
        self.max_sessions, self.max_convs = max_sessions, max_convs
        self.sessions = {}
        self.lock = threading.Lock()
        self.log = log
        self.alarms = []          # sessions that reported the alarm (stats)
        self.next_id = 1

    # ------------------------------------------------------------ requests
    def observe(self, messages, tools=None, meta=None, affinity=0, now=None):
        """A new request (messages in the server's chat form, system first; tools: names or tool dicts)."""
        now = time.time() if now is None else now
        names = [self._tool_name(t) for t in tools or []]
        sys_txt = "".join(str(m.get("content")) for m in messages[:1] if m.get("role") == "system")
        if self.override is not None:
            hit = bool(self.override.search(sys_txt) or any(self.override.fullmatch(n) or self.override.search(n)
                                                            for n in names if n))
            return Info(hit, "override" if hit else "")
        ch = chain(messages)
        if not ch:
            return Info()
        sk = session_of(meta, affinity)
        client = sk is not None
        sk = sk or f"a:{affinity:x}"
        with self.lock:
            s = self.sessions.get(sk)
            if s is None:
                s = self.sessions[sk] = Session(sk, client)
                while len(self.sessions) > self.max_sessions:
                    old = min(self.sessions.values(), key=lambda x: x.t_last)
                    del self.sessions[old.key]
            s.n += 1
            s.t_last = now
            c, new_from = self._place(s, messages, ch, now)
            g = c.group
            if not g.hint and any(n in SPAWN_TOOLS for n in names) and CLI_RE.search(sys_txt):
                g.hint = True
            self._consumption(s, c, messages[new_from:], now)
            self._decide(s, now)
            main = self._is_main(s, c)
            if s.main is None and client and not s.alarmed and s.n >= self.alarm_turns:
                s.alarmed = True
                self.alarms.append({"session": sk[2:], "requests": s.n, "t": now})
                del self.alarms[:-8]
                self.log(f"[agents] WARNING: session {sk[2:]} sent {s.n} requests and no main agent thread was "
                         f"identified (scheduler priority is off for it); set QE_MAIN_AGENT_RE to force it")
            why = ("tier %d" % s.tier) if main else ""
            return Info(main, why, c, s, g)

    @staticmethod
    def _tool_name(t):
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            return t.get("name") or (t.get("function") or {}).get("name") or ""
        return ""

    def _place(self, s, messages, ch, now):
        """The conversation this request belongs to (continuation, branch or a new one)."""
        best, bl = None, 0
        for c in s.convs:
            n = len(c.chain)
            if n <= len(ch) and ch[n - 1] == c.chain[-1]:
                l = n
            else:  # common prefix (prefix hashes: binary search)
                lo, hi = 0, min(n, len(ch))
                while lo < hi:
                    m = (lo + hi + 1) // 2
                    if c.chain[m - 1] == ch[m - 1]:
                        lo = m
                    else:
                        hi = m - 1
                l = lo
            if l > bl or (l == bl and best is not None and l and c.t_last > best.t_last):
                best, bl = c, l
        if best is not None and bl == len(best.chain) and len(ch) > bl and self._continues(messages, bl):
            best.chain, best.turns, best.t_last = ch, best.turns + 1, now
            best.group.turns += 1
            return best, bl
        if best is not None and bl == len(ch) == len(best.chain):  # a retry of the same request
            best.t_last = now
            return best, len(ch)
        # opening = the system prompt and the first non-system message
        k = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(ch) - 1)
        key = ch[k]
        g = s.groups.get(key)
        if g is None:
            g = s.groups[key] = Group(key, now)
        branch = best is not None and bl > k
        # a side task (fork directive, summary) adds only a user message to the tip of a history; a branch that
        # diverges inside it (an edited / rewritten history) competes with its source and inherits its turns
        side = branch and bl == len(best.chain)
        c = Conv(self.next_id, g, ch, now, branch, side, best.turns if branch and not side else 1)
        self.next_id += 1
        g.turns += 1
        g.convs.append(c)
        s.convs.append(c)
        if len(s.convs) > self.max_convs:  # forget the least recently used conversations
            s.convs.sort(key=lambda x: x.t_last)
            for x in s.convs[:len(s.convs) - self.max_convs]:
                x.group.convs.remove(x)
            del s.convs[:len(s.convs) - self.max_convs]
        return c, (bl if branch else 0)

    @staticmethod
    def _continues(messages, at):
        """A next turn adds the reply to the previous request first (an assistant message) and, when that reply
        called tools, their results (protocol: tool calls are answered by tool results). A request that adds only
        a user message to an existing history, or answers a tool call with plain text, is a side task on it (a
        fork's directive, a summary)."""
        if at >= len(messages) or messages[at].get("role") != "assistant":
            return False
        ids = [c.get("id") for c in messages[at].get("tool_calls") or [] if c.get("id")]
        if ids and at + 1 < len(messages):
            nxt = messages[at + 1]
            return nxt.get("role") == "tool" and nxt.get("tool_call_id") in ids
        return True

    def _consumption(self, s, c, new, now):
        """Final replies of other conversations that appear in this request's new messages: c's group is their
        parent. (A conversation's own earlier final replies stay in its history: not evidence.)"""
        pend = [x for x in s.convs if x.final and x is not c and not x.consumed and (x.branch or x.group is not c.group)]
        if not pend:
            return
        txt = norm(" ".join(text_of(m.get("content")) for m in new if m.get("role") in ("tool", "user", "system")))
        if not txt:
            return
        for x in pend:
            if x.final in txt:
                x.consumed = True
                c.group.parent_t = now
                if x.group is not c.group:
                    x.group.consumed_by = c.group

    def _decide(self, s, now):
        cur = s.main
        cand, tier = None, 0
        # a parent group with a few turns, or one that took over the current main's output (compaction)
        roots = [g for g in s.groups.values() if g.parent_t and g.consumed_by is None and
                 (g.turns >= 2 or (cur is not None and cur.consumed_by is g))]
        if roots:
            # the parent nobody consumed; among several, the one that consumed most recently
            cand, tier = max(roots, key=lambda g: g.parent_t), 3
        else:
            hinted = [g for g in s.groups.values() if g.hint]
            if hinted:
                cand, tier = min(hinted, key=lambda g: g.t0), 2
            else:
                multi = [g for g in s.groups.values() if g.turns >= 2 and any(not c.branch for c in g.convs)
                         and any(c.turns >= 2 for c in g.convs)]
                if multi:
                    cand, tier = min(multi, key=lambda g: g.t0), 1
        if cand is None:
            return
        if cur is None or cur.key not in s.groups:
            s.main, s.tier = cand, tier
            return
        if cand is cur:
            s.tier = max(s.tier, tier)
            return
        # sticky: a stronger tier, or the candidate consumed the current main's output
        if tier > s.tier or (tier == 3 and cur.consumed_by is cand):
            s.main, s.tier = cand, tier
            s.switches += 1

    @staticmethod
    def _is_main(s, c):
        """c is the main group's main conversation: the one with the most turns among its conversations that are
        no side task and did not deliver a result to another one (ties: the most recent)."""
        if s.main is None or c.group is not s.main or c.side or c.consumed:
            return False
        best = max((x for x in s.main.convs if not x.side and not x.consumed), key=lambda x: (x.turns, x.t_last),
                   default=None)
        return best is c

    # ------------------------------------------------------------ results
    def finished(self, info, reply_text, tool_call):
        """The request's reply: a final one (no tool call) is remembered to recognise it in a parent's history."""
        if info is None or info.conv is None:
            return
        txt = reply_text or ""
        if "</think>" in txt:
            txt = txt.split("</think>", 1)[1]
        txt = norm(txt.split("<tool_call>", 1)[0])
        with self.lock:
            if tool_call or len(txt) < SNIP_MIN:
                info.conv.final = None
            else:
                info.conv.final = txt[:SNIP]

    def is_main_now(self, info):
        """Re-evaluated for a queued request (main may have been identified since it arrived)."""
        if info is None:
            return False
        if info.conv is None:
            return info.main
        with self.lock:
            return self._is_main(info.session, info.conv)

    # ------------------------------------------------------------ monitoring
    def stats(self):
        with self.lock:
            ss = sorted(self.sessions.values(), key=lambda x: -x.t_last)[:8]
            out = []
            for s in ss:
                out.append({"session": s.key[2:] if s.client else "", "requests": s.n,
                            "groups": len(s.groups), "conversations": len(s.convs),
                            "main": bool(s.main), "tier": s.tier, "switches": s.switches,
                            "main_turns": s.main.turns if s.main else 0})
            missing = [a for a in self.alarms if a["session"] in {s.key[2:] for s in self.sessions.values()
                                                                  if s.client and s.main is None}]
            return {"override": bool(self.override), "sessions": out, "alarm": missing[-1] if missing else None}
