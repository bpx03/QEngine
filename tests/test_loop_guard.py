"""CPU test of the cross-turn loop guard detection (server/loop_guard.py)."""
import json
import os
import random
import sys
import time

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
    assert lg.streak(m) == 24  # capped by the 24-turn window


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
    # the same exact call, the result differs only in numbers: new output, no loop (since agent_loops: a build
    # log's progress counter looks the same with digits normalised; the loops seen have byte-identical results)
    m = SYS + sum(([mem("remove", note), tool(f"No entry matched (attempt {i}, {1424 + i} chars used)")]
                   for i in range(4)), [])
    assert lg.streak(m) == 1
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
    # digit-normalised text with the same result still matches; new digits in the result are new output (a call's rule)
    m = SYS + sum(([{"role": "assistant", "content": txt.replace("42", str(40 + i))},
                    {"role": "tool", "content": "error 7"}] for i in range(4)), [])
    assert lg.streak(m) == 4
    m = SYS + sum(([{"role": "assistant", "content": txt}, {"role": "tool", "content": f"error {i}"}]
                   for i in range(4)), [])
    assert lg.streak(m) == 1


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
    # a human message does not reset the stretch (session 92997599 looped straight past "stop looping"); when it is
    # the newest message the state says so (the server then only turns reasoning on)
    m = SYS + [call("ps -o lstart= -p 4242"), tool("Exit code 1")] * 4 + [{"role": "user", "content": "stop that"}]
    st = lg.state(m)
    assert st["turns"] == 4 and st["human"]
    m = SYS + [call("ps -o lstart= -p 4242"), tool("Exit code 1"), {"role": "user", "content": "try again"},
               call("ps -o lstart= -p 4242"), tool("Exit code 1")]
    st = lg.state(m)
    assert st["turns"] == 2 and not st["human"]  # one re-run on request: below the 3-turn threshold


def bash(cmd, desc):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "x", "type": "function",
                            "function": {"name": "Bash", "arguments": {"command": cmd, "description": desc}}}]}


def test_polling_progress_is_no_loop():
    # corpus false alarm (0dc67cdb): the same poll of a build log, only the counters move
    poll = "grep -c '^\\[' /tmp/mesa-build32.log; tail -1 /tmp/mesa-build32.log | cut -c1-90"
    m = SYS + sum(([call(poll), tool(f"{812 + 9 * i}\n[{812 + 9 * i}/2304] Compiling C object src/vulkan/runtime/libvulkan_lite")]
                   for i in range(6)), [])
    assert lg.streak(m) == 1
    # corpus false alarm (5ddc3ba3): date + nvidia-smi between reports
    m = SYS + sum(([call("date; nvidia-smi --query-gpu=index,power.draw --format=csv"),
                    tool(f"Wed Oct 21 00:{10 + i}:05 CEST 2026\n0, {100 + i}.5 W")] for i in range(5)), [])
    assert lg.streak(m) == 1
    # the same poll with the same output (nothing moved) is still counted: the guard then only adds thinking + note
    m = SYS + [call(poll), tool("812\n[812/2304] Compiling")] * 3
    assert lg.streak(m) == 3


def test_non_text_result_is_unknown():
    # corpus false alarm (e0a4b0a6): two Reads of the same image, the result has no text
    rd = {"role": "assistant", "content": "", "tool_calls": [{"id": "r", "type": "function", "function": {
        "name": "Read", "arguments": {"file_path": "/tmp/out.png"}}}]}
    m = SYS + [rd, {"role": "tool", "content": [{"type": "image"}], "tool_call_id": "r"}] * 3
    assert lg.streak(m) == 1
    # fixed text + a changing image (review L2): the image part makes the result unknown
    shot = {"role": "tool", "content": [{"type": "text", "text": "Captured"}, {"type": "image"}], "tool_call_id": "r"}
    m = SYS + [rd, shot] * 3
    assert lg.streak(m) == 1


def test_cycle_period_4():
    # session 92997599: A A A B A A A B ... (A = ref= script, B = live= script), identical outputs
    A, B = (bash("python3 - <<'EOF'\nref=...\nEOF", "Decode ref branches"), tool("0x48505e0 b -> 0x48505e0")), \
        (bash("python3 - <<'EOF'\nlive=...\nEOF", "Verify live block"), tool("cap block: fmov s1, #13.0"))
    m = list(SYS)
    seen = []
    for k in range(4):
        for turn in (A, A, A, B):
            m += list(turn)
            seen.append(lg.streak(m))
    # the old consecutive streak dropped to 1 at every B; the stretch keeps growing once B repeats
    assert seen[:4] == [1, 2, 3, 1], seen
    assert seen[4:] == list(range(2, 14)), seen
    assert lg.state(m)["distinct"] == 2


def test_description_is_cosmetic():
    cmd = "python3 decode.py --imm 19"
    m = SYS + sum(([bash(cmd, d), tool("0x48508ac b -> 0x48508ac")] for d in
                   ("Decode branch targets with imm*4", "Decode branch targets with correct imm*4 scaling",
                    "Re-decode branch targets, fixed syntax")), [])
    assert lg.streak(m) == 3
    # a different command with the same description is a different call
    m = SYS + sum(([bash(f"python3 decode.py --imm {i}", "Decode"), tool("same")] for i in range(3)), [])
    assert lg.streak(m) <= 1


def test_edit_test_chain_is_no_loop():
    m = list(SYS)
    for i in range(6):
        m += [call(f"sed -i 's/x{i}/x{i + 1}/' a.py"), tool(""), call("pytest -q"), tool("1 failed: test_a AssertionError")]
        assert lg.streak(m) <= 2, i


def test_repeated_check_between_new_actions():
    m = list(SYS)
    for i in range(5):
        m += [call("git status --short"), tool(" M a.py"), call(f"cat part{i}.txt"), tool(f"part {chr(97 + i) * 30}")]
    m += [call("git status --short"), tool(" M a.py")]
    assert lg.streak(m) <= 2


def test_no_tool_result_is_no_loop():
    # review H1: few-shot / batch extraction, same-shape JSON answers to different human messages
    m = list(SYS)
    for i in range(6):
        m += [{"role": "assistant", "content": '{"label": "positive", "score": 0.9, "x": %d}' % i},
              {"role": "user", "content": f"classify item {i}"}]
    st = lg.state(m)
    assert st["turns"] == 1 and st["human"], st
    # Cline style: results sent as user messages, with numbered paths and different content
    m = list(SYS)
    for i in range(6):
        m += [{"role": "assistant", "content": f"<read_file><path>src/file{i}.py</path></read_file> reading next"},
              {"role": "user", "content": f"[read_file] Result: def f{i}(): return {i * i}"}]
    assert lg.streak(m) == 1
    # the identical text reply, no results at all, a reminder in between is no result either
    txt = "I could not find the configuration file; please check the path /etc/app/conf.d."
    m = SYS + [{"role": "assistant", "content": txt}, {"role": "user", "content": "<system-reminder>x</system-reminder>"}] * 5
    assert lg.streak(m) == 1
    # a tool call whose tool returned nothing, with a reminder in the same turn: nothing repeats
    m = SYS + [call("true"), tool(""), {"role": "user", "content": "<system-reminder>x</system-reminder>"}] * 4
    assert lg.streak(m) == 1
    # a text turn followed by a human message is not stale, even with a repeating tool result
    m = SYS + [{"role": "assistant", "content": txt}, tool("error 7"), {"role": "user", "content": "and now?"}] * 5
    assert lg.streak(m) == 1
    # a varying reminder does not hide a real loop of calls (review L4, first part): it is not part of the result
    m = list(SYS)
    for i in range(4):
        m += [call("ps -p 1"), tool("Exit code 1"), {"role": "user", "content": f"<system-reminder>Tokens {i}</system-reminder>"}]
    assert lg.streak(m) == 4


def test_whole_result_counts():
    # review M3: new progress after the first 4000 characters is new output
    head = "x" * 5000
    m = SYS + sum(([call("cat build.log"), tool(f"{head}\n[{i}/200] done")] for i in range(4)), [])
    assert lg.streak(m) == 1
    m = SYS + [call("cat build.log"), tool(head + "\n[7/200] done")] * 4
    assert lg.streak(m) == 4
    # whitespace alone is no new output
    m = SYS + [call("ps"), tool("a  b\n"), call("ps"), tool("a b"), call("ps"), tool(" a\tb ")]
    assert lg.streak(m) == 3


def test_near_copy_text():
    base = " ".join(f"word{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(200))
    w = base.split()

    def var(i):  # 4 words of 200 replaced: a near copy at the review's temperature-1 loop
        b = list(w)
        for j in range(4):
            b[(i * 37 + j * 11) % 200] = f"new{chr(97 + i)}{chr(97 + j)}"
        return " ".join(b)

    m = SYS + sum(([{"role": "assistant", "content": var(i)}, tool("same error")] for i in range(4)), [])
    assert lg.streak(m) == 4
    # half of the words are different: no near copy
    m = SYS + sum(([{"role": "assistant", "content": " ".join(w[:100]) + f" other{chr(97 + i)} " * 100}, tool("same error")]
                   for i in range(4)), [])
    assert lg.streak(m) == 1
    # compact JSON has no whitespace: the punctuation splits it
    js = lambda i: json.dumps({"k%d" % k: "v%d" % k for k in range(60)} | {"status": "retry" + chr(97 + i)}, separators=(",", ":"))  # noqa: E731
    m = SYS + sum(([{"role": "assistant", "content": js(i)}, tool("same error")] for i in range(4)), [])
    assert lg.streak(m) == 4


def test_state_cost():
    # review M2 probes, on this machine 2,422 ms / 5,078 ms / 34 ms before: state() runs on the event loop
    words = [f"w{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(500)]  # (digit runs normalise to one word)

    def rep(seed, n=650):  # the same vocabulary, different text
        r = random.Random(seed)
        return " ".join(r.choice(words) for _ in range(n))[:4000]

    def var(i):
        b = rep(1).split()
        for j in range(12):
            b[(i * 53 + j * 47) % len(b)] = f"x{chr(97 + i)}{chr(97 + j)}"
        return " ".join(b)

    def best(m):
        out = 1e9
        for _ in range(3):
            t = time.perf_counter()
            st = lg.state(m)
            out = min(out, time.perf_counter() - t)
        return st, out * 1e3

    limit = float(os.environ.get("QE_LG_COST_MS", 100))
    loop = SYS + sum(([{"role": "assistant", "content": var(i)}, tool("same error")] for i in range(12)), [])
    st, ms = best(loop)
    assert st["turns"] == 12 and ms < limit, (st, ms)
    distinct = SYS + sum(([{"role": "assistant", "content": rep(i)}, tool("same error")] for i in range(24)), [])
    st, ms = best(distinct)
    assert st["turns"] <= 2 and ms < limit, (st, ms)
    big = SYS + sum(([call(f"cat /a/{i}"), tool(rep(i, 400) + "\n" * 3)] for i in range(400)), [])
    st, ms = best(big)
    assert st["turns"] == 1 and ms < limit, (st, ms)


def firing_turns(seq, turns=120, n=3):
    """Drives a real history turn by turn through state() and due(): the history lengths the guard fires at."""
    m, out = [{"role": "user", "content": "go"}], []
    for t in range(1, turns + 1):
        c = seq[(t - 1) % len(seq)]
        m += [call(c), tool("result of " + c)]
        if lg.due(lg.state(m), n):
            out.append(t)
    return out


def test_schedule_is_the_same_for_every_period():
    # review of 6d0453fb, H1: `turns` must grow to the window for any period, or a cycle goes silent or fires every turn
    names = [f"cmd{chr(97 + i)}" for i in range(24)]
    for p in (1, 2, 3, 4, 13, 19, 22, 24):
        f = firing_turns(names[:p])
        # stale from turn p + 1: `turns` is 3, 6, 12 at p + 2, p + 5, p + 11, then every 24 turns of the conversation
        want = [p + 2, p + 5, p + 11] + [t for t in range(24, 121, 24) if t >= p + 23]
        assert f == want, (p, f, want)
    # A A A B, the module doc's example: the first B resets, then it is one stretch
    f = firing_turns(["a", "a", "a", "b"])
    assert f == [3, 6, 9, 15, 48, 72, 96, 120], f
    # every firing is thinking too: never more than a few turns of 120, whatever the period
    assert all(len(firing_turns(names[:p])) <= 8 for p in range(1, 25))
    # a period the window cannot see is no repeat
    assert firing_turns([f"cmd{chr(97 + i)}{chr(97 + j)}" for i in range(5) for j in range(6)]) == []
    assert not lg.due({"turns": 2, "total": 24}, 3) and not lg.due({"turns": 24, "total": 24}, 0)


def test_add_note():
    m = SYS + [call("x"), tool("same"), call("x"), tool("same")]
    before = json.dumps(m)
    out, ok = lg.add_note(m, 5, 2)
    assert ok and json.dumps(m) == before  # the history is not changed in place
    assert "your last 5 turns" in out[-1]["content"] and "(2 different call(s)" in out[-1]["content"], out[-1]
    assert out[-1]["content"].startswith("same\n\n[server note") and out[:-1] == m[:-1]
    # a long conversation: the number is the count of calls, not the message index (review M1)
    m = SYS + [call("x"), tool("same")] * 200
    out, ok = lg.add_note(m, 3, 1)
    assert ok and "(1 different call(s)" in out[-1]["content"] and "402" not in out[-1]["content"]
    # list content: joined to the last text part, or a new part after an image
    m = SYS + [call("x"), {"role": "tool", "content": [{"type": "text", "text": "same"}]}]
    out, ok = lg.add_note(m, 3, 1)
    assert ok and out[-1]["content"][0]["text"].startswith("same\n\n[server note") and "server note" not in str(m)
    m = SYS + [call("x"), {"role": "tool", "content": [{"type": "image"}]}]
    out, ok = lg.add_note(m, 3, 1)
    assert ok and out[-1]["content"][0] == {"type": "image"} and "server note" in out[-1]["content"][1]["text"]
    # no result to carry it
    assert lg.add_note(SYS + [call("x")], 3, 1) == (SYS + [call("x")], False)
    assert lg.add_note(SYS, 3, 1) == (SYS, False)


def test_last_reply_tokens():
    # <|im_start|>=1, assistant\n = [7, 8], <|im_end|>=2, specials >= 100, whitespace id 9
    ids = [1, 7, 8, 50, 9, 51, 2, 1, 5, 6, 2, 1, 7, 8, 60, 101, 61, 9, 62, 2, 1, 5, 2, 1, 7, 8]
    assert lg.last_reply_tokens(ids, 1, [7, 8], 2, 100, {9}) == [60, 61, 62]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
