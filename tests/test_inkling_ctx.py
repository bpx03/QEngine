"""[ink-ctx] CPU test of the Inkling server defaults (review-T4): the card choice (inkling_devices,
parse_args), the device-string checks, the Inkling-only idle-archive re-arm, the context-cap edges (model_ctx_limit,
parse_args). No GPU, no tokenizer; about a second.

  python3 tests/test_inkling_ctx.py
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.modules.setdefault("pynvml", types.ModuleType("pynvml"))
import app  # noqa: E402
import model_profile as mp  # noqa: E402

FAILS = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(info)[:400] if info else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def devs(given, **env):
    try:
        return app.inkling_devices(given, env)
    except ValueError as ex:
        return "ERR", str(ex)


def devices():
    check("no --devices, nothing set: one card", devs(None) == ("0", []))
    check("direct start, QE_DEVICES=0,1 (the engine's old way)", devs(None, QE_DEVICES="0,1") == ("0,1", []))
    d, ln = devs("0,1", QE_DEVICES="0")
    check("direct start, --devices wins over QE_DEVICES, one [settings] line", d == "0,1" and len(ln) == 1
          and ln[0].startswith("[settings] QE_DEVICES=0"), (d, ln))
    d, ln = devs("0,1,2", QE_RUNSH_DEVICES="all", CUDA_VISIBLE_DEVICES="0,1,2")
    check("run.sh every visible GPU: still one card (no implicit TP)", d == "0" and len(ln) == 1, (d, ln))
    check("run.sh, one visible GPU: one card, no line", devs("0", QE_RUNSH_DEVICES="all", CUDA_VISIBLE_DEVICES="0") == ("0", []))
    check("run.sh QE_DEVICES=0,2 -> --devices 0,1: TP2",
          devs("0,1", QE_RUNSH_DEVICES="qe", QE_DEVICES="0,2", CUDA_VISIBLE_DEVICES="0,2") == ("0,1", []))
    check("run.sh explicit --devices 1", devs("1", QE_RUNSH_DEVICES="given", CUDA_VISIBLE_DEVICES="0,1,2") == ("1", []))
    check("empty --devices = not given", devs("", QE_DEVICES="0,1") == ("0,1", []) and devs("  ") == ("0", []))
    for bad in ("1,x", "0,,1", "-1", "a", "0;1"):
        check(f"bad list '{bad}' refused", devs(bad)[0] == "ERR", devs(bad))
    check("repeated card refused", devs("0,0")[0] == "ERR")
    check("repeated card with INK_VIRTUAL_TP=1", devs("0,0", INK_VIRTUAL_TP="1") == ("0,0", []))
    check("index beyond the visible GPUs refused", devs("0,2", CUDA_VISIBLE_DEVICES="0,2")[0] == "ERR")


def rearm():
    tmp = tempfile.mkdtemp(prefix="qe_ic_")
    prof = {}
    for fam, mt in ((mp.INK, "inkling2_loop"), (mp.GPTOSS, "gpt_oss"), (mp.Q27, "qwen3_5"), (mp.GLM, "glm5_next")):
        d = os.path.join(tmp, fam)
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"model_type": mt, "vocab_size": 1000}, f)
        prof[fam] = mp.Profile(d)
    check("re-arm after a reply: Inkling", app.bg_rearm_after_reply(prof[mp.INK], {}) is True)
    check("re-arm: INK_BG_REARM=0 off", app.bg_rearm_after_reply(prof[mp.INK], {"INK_BG_REARM": "0"}) is False)
    check("re-arm: gpt-oss too ([gptoss-rearm]); the 27B / GLM keep their rule",
          app.bg_rearm_after_reply(prof[mp.GPTOSS], {}) is True
          and not any(app.bg_rearm_after_reply(prof[f], {}) for f in (mp.Q27, mp.GLM)))


def ctx():
    L = app.model_ctx_limit
    check("model_ctx_limit", (L(1048576), L(None), L("x"), L(1e30), L(2 ** 40), L(float("nan")), L(100))
          == (1048576, None, None, 1 << 20, 1 << 20, None, None), [L(1048576), L(None), L("x"), L(1e30), L(2 ** 40)])


def parse():
    tmp = tempfile.mkdtemp(prefix="qe_ic_")

    def mk(name, text):
        d = os.path.join(tmp, name)
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"model_type": "inkling2_loop", "text_config": dict(text, model_type="inkling2_loop_text", vocab_size=1000)}, f)
        return d

    ink = mk("ink", {"model_max_length": 1048576})
    nolen = mk("nolen", {})
    huge = mk("huge", {"model_max_length": 1e30})
    oss = os.path.join(tmp, "oss")
    os.makedirs(oss)
    with open(os.path.join(oss, "config.json"), "w") as f:
        json.dump({"model_type": "gpt_oss", "vocab_size": 1000}, f)
    saved = dict(os.environ)
    try:
        for k in ("QE_DEVICES", "QE_RUNSH_DEVICES", "INK_VIRTUAL_TP", "FLASH_KV", "INK_DENSE8", "INK_DENSE8_FILE",
                  "INK_DENSE8_RTN", "INK_SLOTS"):
            os.environ.pop(k, None)
        os.environ["QE_LIB"] = "/nonexistent/lib.so"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        a = app.parse_args(["--model", ink])
        check("Inkling default: one card, context 1,048,576 (not 'lowered')", a.devices == "0" and a.max_ctx == 1048576
              and a.max_ctx_model and a.ignored_start == [], (a.devices, a.max_ctx, a.ignored_start))
        a = app.parse_args(["--model", nolen])
        check("no model_max_length: 262,144", a.max_ctx == 262144, a.max_ctx)
        a = app.parse_args(["--model", huge])
        check("model_max_length 1e30: capped at 1,048,576", a.max_ctx == 1 << 20, a.max_ctx)
        a = app.parse_args(["--model", ink, "--max-ctx", "2M"])
        check("--max-ctx 2M on Inkling: capped + one line", a.max_ctx == 1 << 20 and len(a.ignored_start) == 1
              and "capped" in a.ignored_start[0], (a.max_ctx, a.ignored_start))
        a = app.parse_args(["--model", ink, "--max-ctx", "300000"])
        check("--max-ctx 300000 on Inkling: kept, quiet", a.max_ctx == 300000 and a.ignored_start == []
              and not getattr(a, "max_ctx_model", False))
        os.environ["QE_RUNSH_DEVICES"] = "all"
        a = app.parse_args(["--model", ink, "--devices", "0,1,2"])
        check("run.sh default on 3 cards: Inkling one card", a.devices == "0", a.devices)
        a = app.parse_args(["--model", oss, "--devices", "0,1,2"])
        check("run.sh default on 3 cards: gpt-oss unchanged", a.devices == "0,1,2", a.devices)
        os.environ.pop("QE_RUNSH_DEVICES")
        a = app.parse_args(["--model", oss])
        check("gpt-oss without --devices: every visible card (unchanged)", a.devices == "0,1,2", a.devices)
        os.environ["QE_RUNSH_DEVICES"] = "qe"
        os.environ["QE_DEVICES"] = "0,2"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
        a = app.parse_args(["--model", ink, "--devices", "0,1"])
        check("run.sh QE_DEVICES=0,2: Inkling TP2", a.devices == "0,1", a.devices)
        # [ink-pf8] --dense-weights on any card count: int8 -> INK_DENSE8=1, g15 -> INK_DENSE8=0 (flag wins), quiet
        os.environ["INK_DENSE8"] = "0"
        a = app.parse_args(["--model", ink, "--devices", "0,1", "--dense-weights", "int8"])
        check("TP2 --dense-weights int8: INK_DENSE8=1 (wins over env 0); no dense8 file: the RTN line only",
              os.environ.get("INK_DENSE8") == "1" and len(a.ignored_start) == 1 and "RTN INT8" in a.ignored_start[0],
              (os.environ.get("INK_DENSE8"), a.ignored_start))
        a = app.parse_args(["--model", ink, "--devices", "0,1", "--dense-weights", "g15"])
        check("TP2 --dense-weights g15: INK_DENSE8=0, quiet", os.environ.get("INK_DENSE8") == "0" and a.ignored_start == [],
              (os.environ.get("INK_DENSE8"), a.ignored_start))
        os.environ.pop("INK_DENSE8", None)
        os.environ["QE_RUNSH_DEVICES"] = "given"
        a = app.parse_args(["--model", ink, "--devices", "0", "--dense-weights", "int8"])
        check("one card --dense-weights int8: INK_DENSE8=1, no 'no effect' line", os.environ.get("INK_DENSE8") == "1"
              and not any("no effect" in x for x in a.ignored_start), (os.environ.get("INK_DENSE8"), a.ignored_start))
        check("int8 without dense8.safetensors: one RTN line", os.environ.get("INK_DENSE8") == "1"
              and len(a.ignored_start) == 1 and "RTN INT8" in a.ignored_start[0], a.ignored_start)
        d8 = os.path.join(ink, "dense8.safetensors")
        try:  # review2 L5: the file never outlives the check
            open(d8, "w").close()
            a = app.parse_args(["--model", ink, "--devices", "0,1", "--dense-weights", "int8"])
            check("int8 with dense8.safetensors: quiet", a.ignored_start == [], a.ignored_start)
        finally:
            if os.path.exists(d8):
                os.remove(d8)
        os.environ.pop("INK_DENSE8", None)
        a = app.parse_args(["--model", ink, "--devices", "0"])
        check("no flag: INK_DENSE8 untouched", "INK_DENSE8" not in os.environ and a.ignored_start == [], a.ignored_start)
        a = app.parse_args(["--model", oss, "--dense-weights", "g15"])
        check("gpt-oss --dense-weights: warns", len(a.ignored_start) == 1 and "--dense-weights" in a.ignored_start[0],
              a.ignored_start)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def dense8_lines():
    """review2 L2: the server's INT8-fallback line follows the engine's rules"""
    tmp = tempfile.mkdtemp(prefix="qe_ic_d8_")
    L = lambda **e: app.ink_dense8_lines(tmp, e)  # noqa: E731
    check("no INK_DENSE8: no line (the engine's default)", L() == [])
    check("INK_DENSE8=0 / off / yes: g15 (atoi 0), no line", L(INK_DENSE8="0") == L(INK_DENSE8="off") == L(INK_DENSE8="yes") == [])
    check("INK_DENSE8=1 without the file: RTN line", len(L(INK_DENSE8="1")) == 1 and "RTN INT8" in L(INK_DENSE8="1")[0])
    check("INK_DENSE8=2: on (atoi), RTN line", len(L(INK_DENSE8="2")) == 1)
    check("INK_DENSE8_RTN=1: RTN by name, no line", L(INK_DENSE8="1", INK_DENSE8_RTN="1") == [] and L(INK_DENSE8_RTN="1") == [])
    ln = L(INK_DENSE8="1", INK_DENSE8_FILE=os.path.join(tmp, "nope.safetensors"))
    check("missing INK_DENSE8_FILE: 'will not start', not RTN", len(ln) == 1 and "will not start" in ln[0] and "RTN" not in ln[0], ln)
    f = os.path.join(tmp, "d8.safetensors")
    try:
        open(f, "w").close()
        check("INK_DENSE8_FILE present: no line", L(INK_DENSE8="1", INK_DENSE8_FILE=f) == [])
        open(os.path.join(tmp, "dense8.safetensors"), "w").close()
        check("dense8.safetensors next to the checkpoint: no line", L(INK_DENSE8="1") == [])
    finally:
        for x in (f, os.path.join(tmp, "dense8.safetensors")):
            if os.path.exists(x):
                os.remove(x)


def slots_default():
    """[ink-pf8-1c2] the C API's slot rules (qe_create calls these): default 2 on one card / 8 on 2+ distinct cards,
    INK_SLOTS wins (clamped 1..16), and the start-up line for an explicit count that does not run as asked"""
    import ctypes
    lib = os.path.join(HERE, "..", "build_srv", "libqinkling.so")
    if not os.path.exists(lib):
        print("SKIP slot rules: build_srv/libqinkling.so is not built (OUT=build_srv src/inkling/build.sh); the GPU server "
              "gate covers them")
        return
    so = ctypes.CDLL(lib)
    f, g, note = so.qe_ink_default_slots, so.qe_ink_slots, so.qe_ink_slot_note
    f.argtypes, f.restype = [ctypes.c_char_p], ctypes.c_int
    g.argtypes, g.restype = [ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int
    note.argtypes, note.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int
    saved = os.environ.pop("QE_DEVICES", None)
    try:
        got = [f(b"0"), f(b"1"), f(b"0,1"), f(b"0,1,2"), f(b"10,11"), f(b""), f(None), f(b"0,0")]
        check("slot default: one card 2, TP 8, none given = one card, virtual TP 0,0 = one card",
              got == [2, 2, 8, 8, 8, 2, 2, 2], got)
        os.environ["QE_DEVICES"] = "0,1"
        check("slot default: no list, QE_DEVICES=0,1 -> 8; a list wins", (f(b""), f(b"0")) == (8, 2), (f(b""), f(b"0")))
        os.environ.pop("QE_DEVICES")
        got = [g(b"0", None), g(b"0", b""), g(b"0", b"8"), g(b"0,1", b"3"), g(b"0", b"32"), g(b"0", b"0")]
        check("INK_SLOTS wins over the default (clamped 1..16; empty = unset)", got == [2, 2, 8, 3, 16, 1], got)

        def msg(raw, served):
            b = ctypes.create_string_buffer(200)
            return b.value.decode() if note(raw, served, b, 200) else ""
        check("halving line: raw request and served count", msg(b"16", 8) == "inkling: requested 16 slots, serving 8 "
              "(memory: each slot's fixed state ~0.36 GB)", msg(b"16", 8))
        check("clamp line shows the raw value", msg(b"32", 16).startswith("inkling: requested 32 slots, serving 16 (at most 16)"),
              msg(b"32", 16))
        check("no line when it runs as asked / no INK_SLOTS", msg(b"8", 8) == "" and msg(None, 2) == "" and msg(b"", 2) == "")
    finally:
        os.environ.pop("QE_DEVICES", None)
        if saved is not None:
            os.environ["QE_DEVICES"] = saved


if __name__ == "__main__":
    devices()
    rearm()
    ctx()
    parse()
    dense8_lines()
    slots_default()
    print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
    sys.exit(1 if FAILS else 0)
