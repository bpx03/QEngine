"""Activation steering in the server, shared by the 27B and the Flash-Next engine wrappers (engine.py) and their
app.py: the engine calls (qe_capi.h tier 10) as a mixin, and the request field "steering".

Vectors are loaded by the engine at start (QE_STEER=<file.gguf>[:scale], QE_STEER_DIR=<dir>,
QE_STEER_DEFAULT=<name>). A request picks one with
    "steering": false | "<name>" | {"vector": name, "scale": s, "mode": "add"|"project", "layers": [lo, hi]}
(absent or true: the server default). GET /v1/steering lists the loaded vectors and their defaults.
"""
import ctypes
import json

_SIGS = {
    "qe_set_steering": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]),
    "qe_set_steering_ex": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int]),
    "qe_steer_info": (ctypes.c_char_p, [ctypes.c_void_p]),
    "qe_capture": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_float)]),
}
DEFAULT = (-2, 1.0, -1, -1, -1)  # the server default vector with its defaults
OFF = (-1, 1.0, -1, -1, -1)


class SteeringMixin:
    """Engine methods for steering; needs self.lib (ctypes.CDLL) and self.h (the engine handle). Engines without
    the calls: no vectors, set_steering only accepts the default / off, capture raises."""

    def _steer_fn(self, name):
        f = getattr(self.lib, name, None)
        if f is not None:
            f.restype, f.argtypes = _SIGS[name]
        return f

    def steer_info(self):
        """{"default": name or None, "vectors": [{name, mode, layers, scale (the defaults), file_layers,
        modes: {add|project: {layers, scale}}, description, intended, file}]}"""
        f = self._steer_fn("qe_steer_info")
        if f is None:
            return {"default": None, "vectors": []}
        return json.loads(f(self.h).decode())

    def set_steering(self, vec=-2, scale=1.0, mode=-1, lo=-1, hi=-1):
        """Steering of the next start: vector index (steer_info order), -1 off, -2 the server default; scale
        multiplies the mode's sidecar scale; mode -1 (the vector's default) / 0 add / 1 project; lo, hi the layer
        range (-1, -1: the mode's default; else inside the vector's file_layers). The engine keys its cached
        prefixes / checkpoints / archive by (vector, mode, layers, scale)."""
        ex, one = self._steer_fn("qe_set_steering_ex"), self._steer_fn("qe_set_steering")
        if ex is not None:
            r = ex(self.h, int(vec), float(scale), int(mode), int(lo), int(hi))
        elif one is not None:
            if (mode, lo, hi) != (-1, -1, -1):
                raise ValueError("this engine has no per-request steering mode / layers (qe_set_steering_ex)")
            r = one(self.h, int(vec), float(scale))
        else:
            if vec not in (-1, -2):
                raise ValueError("this engine has no activation steering")
            return
        if r != 0:
            raise ValueError("bad steering selection: vector %d, mode %d, layers %d..%d" % (vec, mode, lo, hi))

    def capture(self, ids, r0, layers, hidden):
        """Mean layer-output residual over the positions [r0, len(ids)) of a fresh unsteered prefill: a flat ctypes
        float array of layers * hidden (layer-major)."""
        f = self._steer_fn("qe_capture")
        if f is None:
            raise RuntimeError("this engine has no activation capture (qe_capture)")
        out = (ctypes.c_float * (layers * hidden))()
        arr = (ctypes.c_int * len(ids))(*ids)
        if f(self.h, arr, len(ids), int(r0), out) != 0:
            raise RuntimeError("capture failed (prompts in flight or no free slot)")
        return out


def resolve_steering(info, body):
    """The request's "steering" field -> (vector index, scale, mode, lo, hi) for Engine.set_steering. Absent / true:
    the server default (-2); false / null: off (-1); "name" or {"vector": name, "scale": s, "mode": "add"|"project",
    "layers": [lo, hi]}: that vector; the scale multiplies the mode's sidecar scale, mode and layers default to the
    sidecar's (the layers to the chosen mode's default range) and the layers must lie in the vector's file. info: the
    engine's steer_info(). Raises ValueError (HTTP 400) for an unknown name or a bad value."""
    if "steering" not in body or body["steering"] is True:
        return DEFAULT
    st = body["steering"]
    if st is False or st is None:
        return OFF
    if isinstance(st, str):
        st = {"vector": st}
    if not isinstance(st, dict):
        raise ValueError('"steering" must be false, a vector name or {"vector": name, "scale": number, '
                         '"mode": "add"|"project", "layers": [lo, hi]}')
    unknown = sorted(set(st) - {"vector", "scale", "mode", "layers"})
    if unknown:
        raise ValueError('"steering": unknown field(s) %s (vector, scale, mode, layers)' % ", ".join(unknown))
    vecs = info["vectors"]
    names = [v["name"] for v in vecs]
    name = st.get("vector")
    if name is None:
        if info["default"] is None:
            raise ValueError('"steering": no default vector; name one of: %s' % (", ".join(names) or "(none loaded)"))
        name = info["default"]
    if name not in names:
        raise ValueError('unknown steering vector "%s"; loaded: %s' % (name, ", ".join(names) or "(none)"))
    v = vecs[names.index(name)]
    scale = st.get("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale != scale or abs(scale) == float("inf"):
        raise ValueError('"steering.scale" must be a finite number')
    mode = st.get("mode")
    if mode is None:
        mi = -1
    elif mode in ("add", "project"):
        mi = 0 if mode == "add" else 1
    else:
        raise ValueError('"steering.mode" must be "add" or "project"')
    lo = hi = -1
    layers = st.get("layers")
    if layers is not None:
        fl = v.get("file_layers", v.get("layers"))
        if (not isinstance(layers, (list, tuple)) or len(layers) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in layers)):
            raise ValueError('"steering.layers" must be [lo, hi] (integers, inclusive)')
        lo, hi = int(layers[0]), int(layers[1])
        if lo > hi or lo < fl[0] or hi > fl[1]:
            raise ValueError('"steering.layers" [%d, %d] not within the layers of vector "%s" (%d..%d)'
                             % (lo, hi, name, fl[0], fl[1]))
    return (names.index(name), float(scale), mi, lo, hi)
