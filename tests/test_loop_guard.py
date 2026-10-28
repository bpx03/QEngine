"""CPU test of the cross-turn loop guard detection (server/loop_guard.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import loop_guard as lg  # noqa: E402


def call(cmd, text=""):
    return {"role": "assistant", "content": text,
            "tool_calls": [{"id": "x", "type": "function", "function": {"name": "Bash", "arguments": {"command": cmd}}}]}


def tool(res):
    return {"role": "tool", "content": res, "tool_call_id": "x"}


SYS = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "find the start time"}]


def test_identical_tool_call():
    m = SYS + [call("ps -o lstart= -p 4242"), tool("Exit code 1")] * 3
    assert lg.streak(m) == 3
    m = SYS + [call("ps -o lstart= -p 4242"), tool("Exit code 1")] * 25
    assert lg.streak(m) == 25


def test_near_identical_probes():
    m = list(SYS)
    for i in range(69, 74):
        script = f"cat > /tmp/probe{i}.sh <<'EOF'\nls {'../' * (i - 60)}build/lib/libqflash.so && echo found\nEOF\nbash /tmp/probe{i}.sh"
        m += [call(script), tool(f"ls: cannot access '{'../' * (i - 60)}build/lib/libqflash.so': No such file or directory")]
    assert lg.streak(m) >= 3


def test_short_replies_never():
    m = SYS + [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "done"}] * 6
    assert lg.streak(m) == 0
    m = SYS + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}] * 6
    assert lg.streak(m) == 0


def test_new_results_are_no_loop():
    m = SYS + sum(([call("tail -n 3 build.log"), tool(f"[{i}/200] compiling module_{chr(97 + i)} ... warnings: x{chr(97 + i) * 5}")]
                   for i in range(6)), [])
    assert lg.streak(m) <= 1
    m = SYS + [call("cat a.py"), tool("print(1)"), call("cat b.py"), tool("print(2)"), call("cat c.py"), tool("x = 3")]
    assert lg.streak(m) == 1


def test_human_steps_in():
    m = SYS + [call("ps -o lstart= -p 4242"), tool("Exit code 1")] * 4 + [{"role": "user", "content": "stop that"}]
    assert lg.streak(m) == 0
    m = SYS + [call("ps -o lstart= -p 4242"), tool("Exit code 1"), {"role": "user", "content": "try again"},
               call("ps -o lstart= -p 4242"), tool("Exit code 1"), call("ps -o lstart= -p 4242"), tool("Exit code 1")]
    assert lg.streak(m) == 2


def test_last_reply_tokens():
    # <|im_start|>=1, assistant\n = [7, 8], <|im_end|>=2, specials >= 100, whitespace id 9
    ids = [1, 7, 8, 50, 9, 51, 2, 1, 5, 6, 2, 1, 7, 8, 60, 101, 61, 9, 62, 2, 1, 5, 2, 1, 7, 8]
    assert lg.last_reply_tokens(ids, 1, [7, 8], 2, 100, {9}) == [60, 61, 62]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
