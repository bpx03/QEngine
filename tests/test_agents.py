"""Main-agent detection (server/agents.py) on synthetic Anthropic bodies, no GPU:
main vs subagent vs fork vs title task with the old and the new Claude Code wording, the real request body
(a dump file named by QE_TEST_DUMP, optional), a changed prompt with a renamed spawn tool (behaviour only), the
QE_MAIN_AGENT_RE override and the alarm. Run: python tests/test_agents.py"""
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from agents import Tracker  # noqa: E402
import app  # noqa: E402  (anthropic_to_messages, agent_label)

OLD_MAIN = ("You are Claude Code, Anthropic's official CLI for Claude.\n"
            "You are an interactive agent that helps users with software engineering tasks.")
NEW_MAIN = ("You are Claude Code, Anthropic's official CLI for Claude.\n"
            "You are an agent working with the user toward their goals, using your own judgment along the way.")
SUB = "You are an agent for Claude Code, Anthropic's official CLI for Claude. Given the user's message, ..."
TITLE = "Generate a concise, sentence-case title (3-7 words) that captures the main topic of this session."
TOOLS = ["Agent", "Bash", "Read", "Edit", "Write"]
SUB_TOOLS = ["Bash", "Read", "Edit", "Write"]


def tools(names):
    return [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}} for n in names]


class Client:
    """A Claude Code-like client: builds /v1/messages bodies and feeds the tracker like the server does."""

    def __init__(self, tr, session=None):
        self.tr, self.sid = tr, session or str(uuid.uuid4())
        self.n = 0

    def ask(self, system, msgs, tool_names, reply, tool_call=False):
        body = {"system": [{"type": "text", "text": system}], "messages": msgs, "tools": tools(tool_names),
                "metadata": {"user_id": json.dumps({"device_id": "d", "account_uuid": "", "session_id": self.sid})}}
        m = app.anthropic_to_messages(body)
        self.n += 1
        info = self.tr.observe(m, app.anthropic_tools(body["tools"]), body["metadata"], affinity=hash(system) & 0xffff,
                               now=float(self.n))
        self.tr.finished(info, reply, tool_call)
        return info


def tool_use(name, i):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": {"x": i}}]}


def tool_result(i, text):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_{i}",
                                         "content": [{"type": "text", "text": text}]}]}


FAILS = []


def check(what, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {what}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILS.append(what)


def session(main_sys, sub_sys, spawn="Agent", main_tools=TOOLS, sub_tools=SUB_TOOLS, fork=True, title=True):
    """main turn 1 spawns two subagents (3 and 2 turns, final replies), a fork side task and a title task, then main
    reads their results (turn 2) and goes on (turn 3). Returns {request: is_main}."""
    tr = Tracker(override="")
    c = Client(tr)
    out = {}
    if title:
        out["title"] = c.ask(TITLE, [{"role": "user", "content": "Build a website with react"}], [],
                             "React website build").main
    u0 = {"role": "user", "content": "Build a website with react and tailwind. Delegate to 2 subagents."}
    out["main1"] = c.ask(main_sys, [u0], main_tools, "<tool_call>spawn</tool_call>", True).main
    h = [u0, tool_use(spawn, 1)]
    a_final = "Subagent A finished: the design system lives in src/design with tokens, Button and Card components."
    b_final = "Subagent B finished: routing is set up with react-router v7, pages Home, About and Pricing exist."
    a = [{"role": "user", "content": "Task A: create the design system in src/design."}]
    b = [{"role": "user", "content": "Task B: set up routing and the three pages."}]
    out["subA1"] = c.ask(sub_sys, a, sub_tools, "", True).main
    a += [tool_use("Bash", 10), tool_result(10, "ok")]
    out["subB1"] = c.ask(sub_sys, b, sub_tools, "", True).main
    b += [tool_use("Bash", 20), tool_result(20, "ok")]
    out["subA2"] = c.ask(sub_sys, a, sub_tools, "", True).main
    a += [tool_use("Write", 11), tool_result(11, "written")]
    out["subB2"] = c.ask(sub_sys, b, sub_tools, b_final).main
    out["subA3"] = c.ask(sub_sys, a, sub_tools, a_final).main
    if fork:  # a fork: the parent's history plus a directive (a side task), one final reply
        f_final = "Fork finished: package.json dependencies are pinned to exact versions now."
        out["fork1"] = c.ask(main_sys, h + [{"role": "user", "content": "You are the fork: pin the deps."}],
                             main_tools, "", True).main
        out["fork2"] = c.ask(main_sys, h + [{"role": "user", "content": "You are the fork: pin the deps."},
                                            tool_use("Bash", 30), tool_result(30, "ok")], main_tools, f_final).main
    h2 = h + [tool_result(1, a_final + "\n" + b_final + ("\n" + f_final if fork else ""))]
    out["main2"] = c.ask(main_sys, h2, main_tools, "<tool_call>bash</tool_call>", True).main
    h3 = h2 + [tool_use("Bash", 2), tool_result(2, "build ok")]
    out["main3"] = c.ask(main_sys, h3, main_tools, "Done: the website builds.").main
    # a new subagent after main has been identified by behaviour
    out["subC1"] = c.ask(sub_sys, [{"role": "user", "content": "Task C: add tests."}], sub_tools, "", True).main
    return out, tr


def expect(out, mains):
    for k, v in out.items():
        check(k, v, k in mains)


print("new Claude Code wording (2.1.310), Agent tool:")
out, tr = session(NEW_MAIN, SUB)
expect(out, {"main1", "main2", "main3"})
check("stats: main tier", tr.stats()["sessions"][0]["tier"], 3)
check("old label of the new wording", app.agent_label(NEW_MAIN, [{"role": "user", "content": "x"}]) == "main agent", False)

print("old wording (interactive agent / software engineering tasks), Task tool:")
out, _ = session(OLD_MAIN, SUB, spawn="Task", main_tools=["Task", "Bash", "Read"])
expect(out, {"main1", "main2", "main3"})

print("subagents also carry the Agent tool (general-purpose): earliest hinted group stays main")
out, _ = session(NEW_MAIN, SUB, sub_tools=TOOLS)
expect(out, {"main1", "main2", "main3"})

print("changed prompt + renamed spawn tool (no structural hint): behaviour finds main once it reads results")
other = "You are Nova, a coding assistant made by ExampleCorp.\nWork carefully."
out, _ = session(other, "You are a helper of Nova.", spawn="Delegate", main_tools=["Delegate", "Shell"],
                 sub_tools=["Shell"])
# main1: one turn and no hint -> nothing known yet; subagent A is the first conversation with a 2nd turn, so
# it is the guess (tier 1) until main brings the results back (main2: behaviour, tier 3)
for k in ("main1", "subA1", "subB1", "subB2", "fork1", "fork2", "title", "subC1"):
    check(k, out[k], False)
check("subA2 (tier-1 guess)", out["subA2"], True)
check("main2", out["main2"], True)
check("main3", out["main3"], True)

print("server restart mid-session: the first request seen is a subagent's 3rd turn (it becomes the guess),")
print("then main brings that subagent's result back -> main takes over")
tr = Tracker(override="")
c = Client(tr)
a = [{"role": "user", "content": "Task A: create the design system."}, tool_use("Bash", 10), tool_result(10, "ok"),
     tool_use("Bash", 11), tool_result(11, "ok")]
fin = "Subagent A finished: tokens, Button, Card and the theme provider are in src/design."
check("subA (3rd turn) alone", c.ask(other, a, ["Shell"], "", True).main, False)
check("subA (4th turn)", c.ask(other, a + [tool_use("Bash", 12), tool_result(12, "ok")], ["Shell"], fin).main, True)
u0 = {"role": "user", "content": "Build a website. Delegate."}
h = [u0, tool_use("Delegate", 1), tool_result(1, fin)]
check("main (history 3, consumes A)", c.ask("You are Nova.", h, ["Delegate", "Shell"], "", True).main, True)
check("main next turn", c.ask("You are Nova.", h + [tool_use("Shell", 2), tool_result(2, "ok")],
                                ["Delegate", "Shell"], "", True).main, True)

print("compaction: main's history is summarized by a side task, a new conversation continues with the summary")
tr = Tracker(override="")
c = Client(tr)
u0 = {"role": "user", "content": "Long task."}
h = [u0, tool_use("Agent", 1), tool_result(1, "ok"), tool_use("Bash", 2), tool_result(2, "ok")]
c.ask(NEW_MAIN, h[:1], TOOLS, "", True)
c.ask(NEW_MAIN, h[:3], TOOLS, "", True)
check("main before compaction", c.ask(NEW_MAIN, h, TOOLS, "", True).main, True)
summ = "Summary of the session so far: the user wants a long task done, steps 1 and 2 are finished, next is step 3."
check("summary side task", c.ask(NEW_MAIN, h + [{"role": "user", "content": "Summarize the conversation."}], TOOLS,
                                 summ).main, False)
n0 = [{"role": "user", "content": "This session is being continued from a previous conversation.\n" + summ}]
check("continued conversation, 1st request", c.ask(NEW_MAIN, n0, TOOLS, "", True).main, True)
check("continued conversation, 2nd request", c.ask(NEW_MAIN, n0 + [tool_use("Bash", 3), tool_result(3, "ok")],
                                                   TOOLS, "", True).main, True)

print("no session id (OpenAI-style client): grouped by the system prompt, no alarm")
tr = Tracker(override="", alarm_turns=3)
for i in range(5):
    tr.observe([{"role": "system", "content": "s"}, {"role": "user", "content": f"q{i}"}], [], None, affinity=7)
check("alarm without client session ids", tr.stats()["alarm"], None)

print("alarm: Claude Code session ids but only single-request conversations")
tr = Tracker(override="", alarm_turns=3)
logs = []
tr.log = logs.append
c = Client(tr)
for i in range(4):
    c.ask(f"system {i}", [{"role": "user", "content": f"one-off {i}"}], [], "x" * 40)
check("alarm reported", tr.stats()["alarm"] is not None, True)
check("warning logged once", len(logs), 1)

print("override QE_MAIN_AGENT_RE (system prompt or tool name)")
tr = Tracker(override=r"^Delegate$")
check("tool match", tr.observe([{"role": "user", "content": "x"}], ["Delegate", "Shell"]).main, True)
check("no match", tr.observe([{"role": "user", "content": "x"}], ["Shell"]).main, False)
tr = Tracker(override=r"agent working with the user")
check("system match", tr.observe([{"role": "system", "content": NEW_MAIN}, {"role": "user", "content": "x"}], []).main, True)

dump = os.environ.get("QE_TEST_DUMP", "")  # a captured Claude Code main-thread request body (optional)
if os.path.exists(dump):
    print("real request body (Claude Code 2.1.310 main thread, 81 messages):")
    body = json.load(open(dump))
    tr = Tracker(override="")
    m = app.anthropic_to_messages(body)
    info = tr.observe(m, app.anthropic_tools(body["tools"]), body["metadata"])
    check("dump: main", info.main, True)
    check("dump: session id parsed", info.session.key, "s:c3a78932-a1f9-4112-a3bf-93a5f6957f66")
    # its next turn (reply + tool result appended) stays main
    m2 = app.anthropic_to_messages(dict(body, messages=body["messages"] + [tool_use("Bash", 99), tool_result(99, "ok")]))
    check("dump: next turn", tr.observe(m2, app.anthropic_tools(body["tools"]), body["metadata"]).main, True)
    # the same body with the prompt reworded and the spawn tool renamed: one request is not enough
    # (no hint), the next turn makes it the session's multi-turn conversation
    body3 = dict(body, system=[{"type": "text", "text": "You are Nova."}],
                 tools=[t for t in body["tools"] if t["name"] != "Agent"] + [dict(body["tools"][0], name="Delegate")],
                 metadata={"user_id": json.dumps({"session_id": "other"})})
    tr = Tracker(override="")
    m3 = app.anthropic_to_messages(body3)
    check("dump reworded: 1st request", tr.observe(m3, app.anthropic_tools(body3["tools"]), body3["metadata"]).main, False)
    m4 = app.anthropic_to_messages(dict(body3, messages=body3["messages"] + [tool_use("Bash", 99), tool_result(99, "ok")]))
    check("dump reworded: next turn", tr.observe(m4, app.anthropic_tools(body3["tools"]), body3["metadata"]).main, True)

print("FAIL: " + ", ".join(FAILS) if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
