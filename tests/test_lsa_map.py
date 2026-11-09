"""CPU-only, no network: bare --lsa / QE_LSA=auto (server/lsa_map.py) against a local HTTP server, with a synthetic map .pt.
download + sha + conversion, cached reuse without network, offline / 404 / sha mismatch / QE_FETCH_ASSETS=0 -> off
(None, never an exception), a .pt put into the cache by hand, a checkpoint without a known map, an explicit .pt path
converted, an explicit safetensors path unchanged, a pickle naming another global refused, the converter writing the
same bytes as safetensors' save_file (the released map's pinned sha256, with QE_LSA_REAL_PT=<the .pt>).
  python tests/test_lsa_map.py
"""
import contextlib
import hashlib
import http.server
import io
import json
import os
import pickle
import struct
import sys
import tempfile
import threading
import types
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import lsa_map  # noqa: E402

REAL_PT = os.environ.get("QE_LSA_REAL_PT", "")  # the released .pt, to check the pinned converted sha256
CFG_27B = {"text_config": {"num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
                           "head_dim": 256,
                           "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]}}


def _fake_torch():
    """Stand-in modules so the stdlib pickler writes the globals a torch map pickle names (this process only)."""
    t, u = types.ModuleType("torch"), types.ModuleType("torch._utils")

    class FloatStorage:
        pass
    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"

    def _rebuild_tensor_v2(*a):
        pass
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    t.FloatStorage, u._rebuild_tensor_v2 = FloatStorage, _rebuild_tensor_v2
    return t, u


def make_pt(path, extra_global=None, vals=None, numel=768 * 64, state=None):
    """A torch-style zip: <root>/data.pkl (dict of 32 tensors, persistent storages) + one storage file per tensor (<root>, then data, then <i>)."""
    t, u = _fake_torch()
    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = t, u
    try:
        class St:
            def __init__(self, key):
                self.key = key

        class Tn:
            def __init__(self, key):
                self.st = St(key)

            def __reduce__(self):
                import collections
                a = (self.st, 0, (768, 64), (64, 1), False, collections.OrderedDict())
                return (u._rebuild_tensor_v2, a) if state is None else (u._rebuild_tensor_v2, a, state)

        class P(pickle.Pickler):
            def persistent_id(self, o):
                return ("storage", t.FloatStorage, o.key, "cpu", numel) if isinstance(o, St) else None

        names = [f"{l}.W_{w}_idx" for l in range(3, 64, 4) for w in "QK"]
        d = {n: Tn(str(i)) for i, n in enumerate(names)}
        if extra_global is not None:
            d["x"] = extra_global
        b = io.BytesIO()
        P(b, protocol=2).dump(d)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            z.writestr("m/data.pkl", b.getvalue())
            z.writestr("m/byteorder", "little")
            for i, n in enumerate(names):
                v = (vals or (lambda i, j: (i * 7 + j % 13) * 0.25))
                z.writestr(f"m/data/{i}", struct.pack("<49152f", *(v(i, j) for j in range(768 * 64)))
                           + bytes(4 * (numel - 768 * 64)), zipfile.ZIP_DEFLATED)
    finally:
        for k, m in saved.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m


class Handler(http.server.BaseHTTPRequestHandler):
    files = {}
    hits = []

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        Handler.hits.append(name)
        if name not in self.files:
            self.send_error(404)
            return
        b = self.files[name]
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(spec, model, env):
    out = io.StringIO()
    old = dict(os.environ)
    os.environ.update(env)
    try:
        with contextlib.redirect_stdout(out):
            r = lsa_map.resolve(spec, model, env=os.environ)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return r, out.getvalue()


def app_cases(model, tmp, ref, pt):
    import app
    old = dict(os.environ)

    def pa(argv, env=None):
        os.environ.pop("QE_LSA", None)
        os.environ.update(env or {})
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                a = app.parse_args(["--model", model, "--metrics-port", "0"] + argv)
            return a.lsa, os.environ.get("QE_LSA"), out.getvalue()
        finally:
            os.environ.clear()
            os.environ.update(old)
    os.environ["QE_LSA_DIR"] = os.path.join(tmp, "appc")
    old["QE_LSA_DIR"] = os.environ["QE_LSA_DIR"]
    assert pa([])[:2] == (None, None)
    assert pa(["--lsa"])[:2] == ("auto", None)
    assert pa(["--lsa", "--max-ctx", "1M"])[:2] == ("auto", None)
    assert pa([], {"QE_LSA": "auto"})[:2] == ("auto", None)  # the documented forms: bare --lsa, QE_LSA=auto
    for alias in ("1", "on", "ON", "true", "yes", " auto "):  # silent aliases, unchanged behaviour
        assert pa([], {"QE_LSA": alias})[:2] == ("auto", None), alias
    assert pa(["--lsa", "auto"])[:2] == ("auto", None)
    assert pa(["--lsa", "on"])[:2] == ("auto", None)
    for off in ("0", "off", ""):
        assert pa([], {"QE_LSA": off})[:2] == (None, None), off
    assert pa(["--lsa", "0"], {"QE_LSA": "1"})[:2] == (None, None)
    assert pa(["--lsa", ref])[:2] == (os.path.abspath(ref), os.path.abspath(ref))
    assert pa([], {"QE_LSA": "/no/such.safetensors"})[:2] == ("/no/such.safetensors", "/no/such.safetensors")
    a, e, _ = pa([], {"QE_LSA": pt})
    assert a == e and a.endswith(".safetensors") and sha(a) == sha(ref), (a, e)
    bad = os.path.join(tmp, "bad.pt")
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("m/data.pkl", b"junk")
    a, e, log = pa([], {"QE_LSA": bad})
    assert (a, e) == (None, None) and "full attention" in log, log
    for argv in (["--lsa", "/no/such.safetensors"], ["--lsa", bad]):
        try:
            pa(argv)
            raise AssertionError(argv)
        except SystemExit:
            pass


def main():
    tmp = tempfile.mkdtemp(prefix="qe_lsa_")
    model = os.path.join(tmp, "m27")
    os.makedirs(model)
    json.dump(CFG_27B, open(os.path.join(model, "config.json"), "w"))
    other = os.path.join(tmp, "other")
    os.makedirs(other)
    json.dump({"num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128},
              open(os.path.join(other, "config.json"), "w"))

    pt = os.path.join(tmp, "src", "map.pt")
    os.makedirs(os.path.dirname(pt))
    make_pt(pt)
    ref = os.path.join(tmp, "ref.safetensors")
    lsa_map.convert(pt, ref, "map.pt")
    st = json.loads(open(ref, "rb").read()[8:8 + int.from_bytes(open(ref, "rb").read(8), "little")])
    assert st["3.W_Q_idx"] == {"dtype": "F32", "shape": [768, 64], "data_offsets": [st["3.W_Q_idx"]["data_offsets"][0],
                                                                                       st["3.W_Q_idx"]["data_offsets"][0] + 196608]}
    with open(ref, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        f.seek(8 + n + st["3.W_Q_idx"]["data_offsets"][0])
        assert struct.unpack("<2f", f.read(8)) == (0.0, 0.25)
    pt_b = open(pt, "rb").read()
    lsa_map.MAP_27B = dict(lsa_map.MAP_27B, name="map.pt", size=len(pt_b), sha256=hashlib.sha256(pt_b).hexdigest(),
                           out="m.safetensors", out_sha256=sha(ref))

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/rel/map.pt"
    Handler.files = {"map.pt": pt_b}

    # 1 download + convert, log names the map and the cache path
    c1 = os.path.join(tmp, "c1")
    r, log = run("auto", model, {"QE_LSA_DIR": c1, "QE_LSA_URL": url})
    assert r == os.path.join(c1, "m.safetensors") and sha(r) == sha(ref), (r, log)
    assert "cached at " + r in log and "Apache-2.0" in log, log
    assert not [f for f in os.listdir(c1) if ".part" in f or f.endswith(".pt")], os.listdir(c1)
    # 2 cached: no network
    Handler.hits = []
    r, log = run("1", model, {"QE_LSA_DIR": c1, "QE_LSA_URL": url, "QE_FETCH_ASSETS": "0"})
    assert r and Handler.hits == [] and "from the cache" in log, (r, log)
    # 3 offline (connection refused), 404, sha mismatch, QE_FETCH_ASSETS=0: off, server goes on
    for env, want in (({"QE_LSA_URL": "http://127.0.0.1:9/x.pt"}, "could not get"),
                      ({"QE_LSA_URL": url + "x"}, "could not get"),
                      ({"QE_FETCH_ASSETS": "0"}, "QE_FETCH_ASSETS=0")):
        c = tempfile.mkdtemp(dir=tmp)
        r, log = run("auto", model, dict(env, QE_LSA_DIR=c))
        assert r is None and want in log and "full attention" in log and "curl -L" in log, (env, log)
        assert not [f for f in os.listdir(c) if ".part" in f], os.listdir(c)
    bad = bytearray(pt_b)
    bad[-100] ^= 1
    Handler.files = {"map.pt": bytes(bad)}
    c = tempfile.mkdtemp(dir=tmp)
    r, log = run("auto", model, {"QE_LSA_DIR": c, "QE_LSA_URL": url})
    assert r is None and "sha256 mismatch" in log and os.listdir(c) == [], (log, os.listdir(c))
    Handler.files = {"map.pt": pt_b}
    # 4 QE_DISK_CACHE/lsa when QE_LSA_DIR is unset
    dc = os.path.join(tmp, "dc")
    r, log = run("on", model, {"QE_DISK_CACHE": dc, "QE_LSA_URL": url})
    assert r == os.path.join(dc, "lsa", "m.safetensors"), (r, log)
    # 5 a .pt put into the cache by hand: converted without network
    c = os.path.join(tmp, "hand")
    os.makedirs(c)
    open(os.path.join(c, "map.pt"), "wb").write(pt_b)
    Handler.hits = []
    r, log = run("auto", model, {"QE_LSA_DIR": c, "QE_FETCH_ASSETS": "0"})
    assert r and sha(r) == sha(ref) and Handler.hits == [], log
    # 6 a checkpoint with no known map: off
    r, log = run("auto", other, {"QE_LSA_DIR": c1})
    assert r is None and "no released index map" in log, log
    # 7 explicit paths: safetensors unchanged; .pt converted into the cache; bad / missing exit as before
    assert run(ref, model, {})[0] == os.path.abspath(ref)
    c = os.path.join(tmp, "exp")
    r, log = run(pt, model, {"QE_LSA_DIR": c})
    assert r.startswith(c) and sha(r) == sha(ref), (r, log)
    for p in (os.path.join(tmp, "none.safetensors"), os.path.join(model, "config.json")):
        try:
            run(p, model, {"QE_LSA_DIR": c})
            raise AssertionError(p)
        except SystemExit:
            pass
    # 8 off values
    for s in ("", "0", "off", None):
        assert run(s, model, {})[0] is None
    # 9 a pickle naming another global is refused (nothing from the file runs)
    evil = os.path.join(tmp, "evil.pt")
    make_pt(evil, extra_global=os.getcwd)
    try:
        lsa_map.read_pt(evil)
        raise AssertionError("evil pickle accepted")
    except pickle.UnpicklingError as ex:
        assert "not allowed" in str(ex), ex
    try:
        run(evil, model, {"QE_LSA_DIR": c})
        raise AssertionError("evil .pt accepted")
    except SystemExit as ex:
        assert "not a D-LSA index map" in str(ex), ex
    # 10 a non-finite value is refused
    nan = os.path.join(tmp, "nan.pt")
    make_pt(nan, vals=lambda i, j: float("nan") if (i, j) == (5, 9) else 0.0)
    try:
        lsa_map.read_pt(nan)
        raise AssertionError("nan accepted")
    except ValueError as ex:
        assert "not finite" in str(ex), ex
    # 10b crafted storages / state: refused as ValueError (never a crash, never a large read)
    for kw, want in (({"numel": 768 * 64 * 4}, "storage of"), ({"state": {"offset": 0.5}}, "state on a tensor")):
        f = os.path.join(tmp, "craft.pt")
        make_pt(f, **kw)
        try:
            lsa_map.convert(f, os.path.join(tmp, "craft.safetensors"))
            raise AssertionError(kw)
        except ValueError as ex:
            assert want in str(ex), (kw, ex)
    # 10c a stalled server: the start waits QE_FETCH_WAIT_S, then goes on with full attention
    import socket
    ls = socket.socket()
    ls.bind(("127.0.0.1", 0))
    ls.listen(1)
    c = tempfile.mkdtemp(dir=tmp)
    import time
    t0 = time.time()
    r, log = run("auto", model, {"QE_LSA_DIR": c, "QE_FETCH_WAIT_S": "1",
                                  "QE_LSA_URL": f"http://127.0.0.1:{ls.getsockname()[1]}/map.pt"})
    assert r is None and "still downloading" in log and time.time() - t0 < 5, (log, time.time() - t0)
    ls.close()
    # 10d server/app.py: flag / env handling (no engine start)
    app_cases(model, tmp, ref, pt)
    # 11 the released map, when present: the pinned converted sha256 (= tools/lsa/convert_map.py's output)
    if REAL_PT and os.path.exists(REAL_PT):
        out = os.path.join(tmp, "real.safetensors")
        lsa_map.convert(REAL_PT, out, os.path.basename(REAL_PT))
        assert sha(out) == "8c44eab910b91816abd067bfaeb2b7aa13611728eb3e24cbb297933fc40c9d15", sha(out)
        print("released map: converted sha256 matches the pin")
    srv.shutdown()
    print("test_lsa_map: all passed")


if __name__ == "__main__":
    main()
