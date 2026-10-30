"""CPU-only, no network: the 27B release-asset download on first start (server/assets.py) against a local HTTP server.
download + sha ok, sha mismatch refused, cached reuse without network, offline warning, QE_FETCH_ASSETS=0,
the model dir left untouched, an explicit QE_MTP_FILE respected.
  python tests/test_assets.py
"""
import contextlib
import hashlib
import http.server
import io
import os
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import assets  # noqa: E402

FILES = {"model_mtp_ft.safetensors": os.urandom(12 << 20), "token_rank.bin": os.urandom(4096)}


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


def serve(files, sums=None):
    sums = sums or {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    Handler.files = dict(files, SHA256SUMS="".join(f"{h}  {n}\n" for n, h in sums.items()).encode())
    Handler.hits = []


def run(model_dir, cache, base, env=None):
    env = {} if env is None else env
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        used = assets.ensure(model_dir, env=env, cache=cache, base=base, timeout=5)
    return used, env, out.getvalue()


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/rel"
    tmp = tempfile.mkdtemp(prefix="qe_assets_")
    model = os.path.join(tmp, "model")
    os.makedirs(model)
    os.chmod(model, 0o555)  # read-only, as the Docker mount
    cache = os.path.join(tmp, "cache")
    os.environ.pop("QE_FETCH_ASSETS", None)

    # 1. download + sha ok: both files land in the cache, the env points at them, progress lines for the big one
    serve(FILES)
    used, env, log = run(model, cache, base)
    assert set(used) == set(FILES), (used, log)
    for n, b in FILES.items():
        assert open(os.path.join(cache, n), "rb").read() == b
    assert env["QE_MTP_FILE"] == os.path.join(cache, "model_mtp_ft.safetensors")
    assert env["QE_TOKEN_RANK_FILE"] == os.path.join(cache, "token_rank.bin")
    assert "downloaded model_mtp_ft.safetensors (13 MB)" in log and "sha256 ok" in log, log
    assert log.count("% of") >= 5, log
    assert os.listdir(model) == [] and not [f for f in os.listdir(cache) if ".part" in f]
    print("download + sha ok: pass")

    # 2. cached reuse: no request at all, even with the server gone from the URL
    used, env, log = run(model, cache, "http://127.0.0.1:9/none")
    assert set(used) == set(FILES) and log.count("using cached") == 2 and "warning" not in log, log
    print("cached reuse (no network): pass")

    # 3. sha mismatch: the corrupted cached copy is not used, the re-download with a wrong sum is refused
    with open(os.path.join(cache, "token_rank.bin"), "r+b") as f:
        f.write(b"\0\0\0\0")
    bad = {n: hashlib.sha256(b).hexdigest() for n, b in FILES.items()}
    bad["token_rank.bin"] = "0" * 64
    serve(FILES, bad)
    used, env, log = run(model, cache, base)
    assert "token_rank.bin" not in used and "QE_TOKEN_RANK_FILE" not in env, (used, log)
    assert "sha256 mismatch" in log and "full draft vocabulary" in log and "stock MTP head" not in log, log
    assert used.get("model_mtp_ft.safetensors"), log  # the good one still used from the cache
    assert not os.path.exists(os.path.join(cache, "token_rank.bin.part%d" % os.getpid()))
    print("sha mismatch refused: pass")

    # 4. offline (connection refused): a warning with the manual commands, nothing set, no exception
    used, env, log = run(model, os.path.join(tmp, "cache2"), "http://127.0.0.1:9/none")
    assert used == {} and env == {}, (used, env)
    assert "warning" in log and "stock MTP head" in log and "curl -LO" in log and "sha256sum -c" in log, log
    print("offline warning: pass")

    # 5. QE_FETCH_ASSETS=0: no request, the warning
    serve(FILES)
    os.environ["QE_FETCH_ASSETS"] = "0"
    try:
        used, env, log = run(model, os.path.join(tmp, "cache3"), base)
    finally:
        del os.environ["QE_FETCH_ASSETS"]
    assert used == {} and Handler.hits == [] and "QE_FETCH_ASSETS=0" in log and "warning" in log, (Handler.hits, log)
    print("QE_FETCH_ASSETS=0: pass")

    # 6. files present in the model dir / QE_MTP_FILE given: nothing to do
    serve(FILES)
    used, env, log = run(model, os.path.join(tmp, "cache4"), base, {"QE_MTP_FILE": "/x", "QE_TOKEN_RANK_FILE": "/y"})
    assert used == {} and Handler.hits == [] and log == "", log
    os.chmod(model, 0o755)
    for n in FILES:
        open(os.path.join(model, n), "wb").close()
    used, env, log = run(model, os.path.join(tmp, "cache4"), base)
    assert used == {} and Handler.hits == [] and log == "", log
    print("present / explicit: pass")
    srv.shutdown()
    print("ALL PASS")


if __name__ == "__main__":
    main()
