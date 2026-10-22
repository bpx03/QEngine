"""CPU-only: tool-call parsing (closed / unclosed / unknown parameters). python tests/test_toolcall_parse.py"""
import os, sys, tempfile
LOG = os.path.join(tempfile.mkdtemp(), "diag.log")
os.environ["QE_DIAG_LOG"] = LOG
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from app import parse_tool_calls

TOOLS = [{"type": "function", "function": {"name": "Write", "parameters": {
    "type": "object", "required": ["file_path", "content"],
    "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "Free", "parameters": {"type": "object"}}}]


def call(body, fn="Write"):
    return parse_tool_calls(f"<tool_call>\n<function={fn}>\n{body}</function>\n</tool_call>", TOOLS)


def log():
    return open(LOG).read() if os.path.exists(LOG) else ""


FP = "<parameter=file_path>\n/a.txt\n</parameter>\n"
c = call(FP + "<parameter=content>\nhello\n</parameter>\n")
assert c == [{"name": "Write", "arguments": {"file_path": "/a.txt", "content": "hello"}}], c

c = call(FP + "<parameter=content>\nhello\nworld\n")
assert c[0]["arguments"] == {"file_path": "/a.txt", "content": "hello\nworld"}, c
assert "TOOL_PARAM_UNCLOSED Write.content" in log()

c = call(FP + "<parameter=content>\nbody\n</parameter>\n<parameter=section>\n" + "x" * 2079 + "\n")
assert c[0]["arguments"] == {"file_path": "/a.txt", "content": "body"}, c
assert "TOOL_PARAM_DROPPED Write.section" in log()

c = call(FP + "<parameter=content>\nuse <parameter=x> here\n</parameter>\n")
assert c[0]["arguments"]["content"] == "use <parameter=x> here", c

c = call(FP + "<parameter=content>\nuse <parameter=x> open\n")  # unclosed, literal inside
assert c[0]["arguments"]["content"] == "use <parameter=x> open", c

c = call(FP + "<parameter=content>\nbody\n</parameter>\n<parameter=mode>\nfast\n</parameter>\n")
assert c[0]["arguments"] == {"file_path": "/a.txt", "content": "body"}, c
assert "TOOL_PARAM_DROPPED Write.mode not in schema" in log()

c = call("<parameter=anything>\n{\"n\": 1}\n</parameter>\n<parameter=open>\ntail\n", fn="Free")
assert c[0]["arguments"] == {"anything": {"n": 1}, "open": "tail"}, c  # no properties: unchanged
print("ok")
