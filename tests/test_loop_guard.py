"""CPU test of the cross-turn loop guard detection (server/loop_guard.py)."""
import json
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
    # tool calls repeat only exactly (1.3.4): near-copies with changing arguments are not a streak
    m = list(SYS)
    for i in range(69, 74):
        script = f"cat > /tmp/probe{i}.sh <<'EOF'\nls {'../' * (i - 60)}build/lib/libqflash.so && echo found\nEOF\nbash /tmp/probe{i}.sh"
        m += [call(script), tool(f"ls: cannot access '{'../' * (i - 60)}build/lib/libqflash.so': No such file or directory")]
    assert lg.streak(m) == 1


def mem(action, text, said=""):
    a = json.dumps({"action": action, "target": "memory", "old_text": text})  # a JSON string, as clients send it
    return {"role": "assistant", "content": said,
            "tool_calls": [{"id": f"call_{abs(hash(text)) % 10**8}", "type": "function",
                            "function": {"name": "memory", "arguments": a}}]}


def test_batch_of_ids_is_no_loop():
    # HA-02 (1.3.4-rc gate): 29 removes of "Legacy note N: ..." that differ only in N, results only in numbers
    m = [{"role": "system", "content": "You are Hermes."},
         {"role": "user", "content": "Memory is near capacity: drop the 29 legacy notes, then add the deployment facts."}]
    used = 1960
    for n in range(1, 30):
        note = f"Legacy note {n}: keep deployment observations concise and current."
        used -= len(note) + 3
        m += [mem("remove", note, "Removing the next legacy note."),
              tool(json.dumps({"success": True, "message": "Entry removed.", "usage": f"{used:,}/2,200 chars",
                               "entries": 30 - n}))]
        assert lg.streak(m) == 1, n


def test_exact_repeat_with_same_result():
    note = "Legacy note 7: keep deployment observations concise and current."
    res = tool(json.dumps({"success": False, "error": "No entry matched 'Legacy note 7'."}))
    m = SYS + [mem("remove", note), res] * 4
    assert lg.streak(m) == 4
    # the same exact call, the result differs only in numbers: still a loop (results compare normalised)
    m = SYS + sum(([mem("remove", note), tool(f"No entry matched (attempt {i}, {1424 + i} chars used)")]
                   for i in range(4)), [])
    assert lg.streak(m) == 4
    # different text around the same call does not hide the loop
    m = SYS + sum(([mem("remove", note, f"Attempt {w}: removing it again."), res] for w in "abcd"), [])
    assert lg.streak(m) == 4


def test_same_call_new_output_is_no_loop():
    m = SYS + sum(([call("squeue -u kris"), tool(f"JOB {i}: {'RUNNING' if i % 2 else 'PENDING'} step {chr(97 + i) * 6}")]
                   for i in range(5)), [])
    assert lg.streak(m) <= 1


def test_text_reply_repeat():
    txt = "I could not find the configuration file 42 times; please check the path /etc/app/conf.d."
    m = SYS + [{"role": "assistant", "content": txt}, {"role": "tool", "content": "error 7"}] * 4
    assert lg.streak(m) == 4
    # digit-normalised text and results still match
    m = SYS + sum(([{"role": "assistant", "content": txt.replace("42", str(40 + i))},
                    {"role": "tool", "content": f"error {i}"}] for i in range(4)), [])
    assert lg.streak(m) == 4


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
