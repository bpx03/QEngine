"""[gptoss-rearm] CPU test of the idle-archive re-arm rule per model (server/app.py bg_rearm_after_reply): after a
finished reply with decode steps the idle loop archives the conversation to host RAM again on Inkling and gpt-oss
(RAM-only archives); the 27B, Flash-Next, GLM and DSV4 keep the 1.3.5 rule (re-arm only after a reply without decode
steps). Off-switches INK_BG_REARM=0 / GPTOSS_BG_REARM=0, each for its own model only. No GPU; well under a second.

  python3 tests/test_bg_rearm.py
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


def fam_profile(fam):  # a profile of that family without a model directory (DSV4 needs GGUF files otherwise)
    p = object.__new__(mp.Profile)
    p.family = fam
    return p


def main():
    rule = app.bg_rearm_after_reply
    # [disk-eager] --disk-eager with --disk-cache re-arms the 27B and Flash-Next too; without the cache, or other models: as before
    ns = types.SimpleNamespace
    for fam, w in {mp.Q27: True, mp.QFN: True, mp.GLM: False, mp.DSV4: False}.items():
        check(f"{fam}: --disk-eager --disk-cache re-arms = {w}",
              rule(fam_profile(fam), {}, ns(disk_eager=True, disk_cache="/x")) is w)
        check(f"{fam}: --disk-eager without --disk-cache = default",
              rule(fam_profile(fam), {}, ns(disk_eager=True, disk_cache="")) is False)
        check(f"{fam}: --disk-cache without --disk-eager = default",
              rule(fam_profile(fam), {}, ns(disk_eager=False, disk_cache="/x")) is False)
    want = {mp.Q27: False, mp.QFN: False, mp.GLM: False, mp.DSV4: False, mp.GPTOSS: True, mp.INK: True}
    for fam, w in want.items():
        check(f"{fam}: re-arm after a reply = {w}",
              rule(fam_profile(fam), {}) is w)
    # the families as the server detects them from config.json
    tmp = tempfile.mkdtemp(prefix="qe_rearm_")
    for fam, mt in ((mp.GPTOSS, "gpt_oss"), (mp.INK, "inkling2_loop"), (mp.Q27, "qwen3_5"), (mp.QFN, "qwen4_exp"),
                    (mp.GLM, "glm5_next")):
        d = os.path.join(tmp, fam)
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"model_type": mt, "vocab_size": 1000}, f)
        p = mp.Profile(d)
        check(f"config {mt}: family {fam}, re-arm {want[fam]}", p.family == fam and rule(p, {}) is want[fam])
    g, i = fam_profile(mp.GPTOSS), fam_profile(mp.INK)
    check("GPTOSS_BG_REARM=0 / false / OFF / no: gpt-oss off (review L3)",
          all(rule(g, {"GPTOSS_BG_REARM": v}) is False for v in ("0", "false", "OFF", " no ")))
    check("INK_BG_REARM=false: Inkling off (review L3)", rule(i, {"INK_BG_REARM": "false"}) is False)
    check("GPTOSS_BG_REARM=1 / empty / ' 1 ': gpt-oss on",
          all(rule(g, {"GPTOSS_BG_REARM": v}) is True for v in ("1", "", " 1 ")))
    check("GPTOSS_BG_REARM=0 leaves Inkling on", rule(i, {"GPTOSS_BG_REARM": "0"}) is True)
    check("INK_BG_REARM=0: Inkling off, gpt-oss on", rule(i, {"INK_BG_REARM": "0"}) is False
          and rule(g, {"INK_BG_REARM": "0"}) is True)
    check("switches = 1 never turn it on for the other models",
          not any(rule(fam_profile(f), {"GPTOSS_BG_REARM": "1", "INK_BG_REARM": "1"}) for f in (mp.Q27, mp.QFN, mp.GLM, mp.DSV4)))
    # (the scheduler's use of the rule: tests/test_bg_rearm_sched.py, mock engine, review L4)
    print(f"{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
