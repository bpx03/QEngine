"""Shared helpers for the engine tests: the libqengine handle with the C API signatures, the repository
root, ctypes int arrays and greedy / speculative generation."""
import ctypes
import os
import time

import qe

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
lib = qe._LIB
lib.qe_create.restype = ctypes.c_void_p
lib.qe_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
for f in ("qe_reset", "qe_destroy"):
    getattr(lib, f).argtypes = [ctypes.c_void_p]
lib.qe_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.qe_decode.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.qe_debug_hidden.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                ctypes.POINTER(ctypes.c_float)]
lib.qe_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.qe_start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
lib.qe_spec_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]


def ints(a):
    return (ctypes.c_int * len(a))(*[int(x) for x in a])


def generate(e, ids, k, ntok, full=False):
    """MTP speculative generation with up to k drafts: (tokens, tok/s, tokens per step)"""
    buf = (ctypes.c_int * 16)()
    out = [lib.qe_start(e, ints(ids), len(ids), k)]
    steps = 0
    t0 = time.time()
    while len(out) < ntok:
        c = lib.qe_spec_step(e, buf)
        out += list(buf[:c])
        steps += 1
    dt = time.time() - t0
    return (out if full else out[:ntok]), (len(out) - 1) / dt, (len(out) - 1) / max(steps, 1)
