"""[rel160] CPU test of Inkling tool-call text: a call whose JSON body does not parse (seen in the 1.6.0 8-pack, cli
pack: a doubled quote after "name") is dropped, and the text that reaches the client has no control tokens; a good call parses.
No GPU, no tokenizer; well under a second.

  python3 tests/test_inkling_toolcall.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import inkling  # noqa: E402
from content_markers import mark  # noqa: E402

fails = 0


def check(name, ok, info=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + (f": {info}" if info and not ok else ""))
    fails += 0 if ok else 1


CALL, END = mark("<|content_invoke_tool_json|>"), mark("<|end_message|>")
bad = CALL + '{"name"""bash","args":{"command":"cat /workspace/install.sh"}}' + END
good = CALL + '{"name": "bash", "args": {"command": "ls"}}' + END
check("broken body: no call", inkling.parse_calls(bad) == [])
t = inkling.plain(bad)
check("broken body: no control tokens in the text", "<|" not in t and "content_invoke" not in t, repr(t))
check("broken body: the body text kept", '"bash"' in t and "install.sh" in t, repr(t))
check("good body parses", inkling.parse_calls(good) == [{"name": "bash", "arguments": {"command": "ls"}}])
# through app.tool_calls_or_text with the inkling tool format
sys.argv = ["app"]
import app  # noqa: E402
app.TOOL_FORMAT = "inkling"
p = types.SimpleNamespace(tool_text=bad)
calls, text = app.tool_calls_or_text(p, None, None, False)
check("app: broken call -> text without control tokens", calls == [] and "<|" not in text and "install.sh" in text, repr(text))
p = types.SimpleNamespace(tool_text=good)
calls, text = app.tool_calls_or_text(p, None, None, False)
check("app: good call", len(calls) == 1 and text == "", repr((calls, text)))
print(f"{fails} failed")
sys.exit(1 if fails else 0)
