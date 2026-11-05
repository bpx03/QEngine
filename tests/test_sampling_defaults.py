"""CPU-only, < 1 s: per-mode sampling defaults (model_profile.sampling with thinking=) and the
CLI > mode recommendation > generation_config resolution.  python tests/test_sampling_defaults.py"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from model_profile import Profile  # noqa: E402


def profile(gen, model_type="qwen4_exp"):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"model_type": model_type}, f)
        with open(os.path.join(d, "generation_config.json"), "w") as f:
            json.dump(gen, f)
        return Profile(d)


fails = 0


def expect(c):
    global fails
    if not c:
        print("FAIL (sampling defaults)")
        fails += 1


# Flash-Next: thinking falls through to generation_config, non-thinking gets the family rec
flash = profile({"temperature": 1.0, "top_p": 0.95, "top_k": 20})
t = flash.sampling()
expect(abs(t["temperature"] - 1.0) < 1e-9 and abs(t["top_p"] - 0.95) < 1e-9 and t["top_k"] == 20)
i = flash.sampling(thinking=False)
expect(abs(i["temperature"] - 0.7) < 1e-9 and abs(i["top_p"] - 0.80) < 1e-9 and i["top_k"] == 20)

# the command line wins per side; unset fields keep the mode's recommendation
c = flash.sampling(0.5, None, None, None, thinking=False)
expect(abs(c["temperature"] - 0.5) < 1e-9 and abs(c["top_p"] - 0.80) < 1e-9)
expect(abs(flash.sampling(0.9)["temperature"] - 0.9) < 1e-9)

# 27B: its own thinking rec (0.6, the server's long-standing default) and the instruct rec
q27 = profile({"temperature": 1.0}, model_type="qwen3_dense")
expect(abs(q27.sampling()["temperature"] - 0.6) < 1e-9)
expect(abs(q27.sampling(thinking=False)["temperature"] - 0.7) < 1e-9)

# a model without a generation_config temperature: the 1.0 fallback, per mode
bare = profile({})
expect(abs(bare.sampling()["temperature"] - 1.0) < 1e-9)
expect(abs(bare.sampling(thinking=False)["temperature"] - 0.7) < 1e-9)

if fails:
    sys.exit(1)
print("sampling defaults: ok")
