"""ctypes bindings to libqengine for tests (pass torch CUDA tensors)."""
import ctypes
import os

import torch

_LIB = ctypes.CDLL(os.environ.get("QE_LIB") or os.path.join(os.path.dirname(__file__), "..", "build", "libqengine.so"))


def _p(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)


def call(name, *args):
    conv = []
    for a in args:
        if isinstance(a, torch.Tensor):
            conv.append(_p(a))
        elif isinstance(a, float):
            conv.append(ctypes.c_float(a))
        elif isinstance(a, int):
            conv.append(ctypes.c_longlong(a) if abs(a) > 2**31 - 1 else ctypes.c_int(a))
        else:
            conv.append(a)
    getattr(_LIB, name)(*conv)


def sync():
    torch.cuda.synchronize()
