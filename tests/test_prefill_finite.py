"""Prefill of real chat sequences (some have delta-net decays that underflow to 0, which once made
the chunked recurrence produce NaN): every prompt row's final hidden state must be finite."""
import ctypes, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from engine import Engine
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
seqs = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "data", "dn_decay_seqs.jsonl"))]
e = Engine(MODEL, max_t=16384)
e.lib.qe_set_dump_hidden.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
bad = 0
for s in seqs:
    ids = s["prompt"] + s["answer"]
    buf = np.zeros((len(ids), 5120), dtype=np.uint16)
    e.reset()
    e.lib.qe_set_dump_hidden(e.h, buf.ctypes.data)
    e.start(ids, 1)
    e.lib.qe_set_dump_hidden(e.h, None)
    # bf16 NaN / inf: exponent bits all ones
    nonfinite = int(((buf & 0x7F80) == 0x7F80).any(1).sum())
    bad += nonfinite > 0
    print(f"{len(ids)} tokens: non-finite rows {nonfinite}")
e.close()
print("PASS" if not bad else "FAIL")
