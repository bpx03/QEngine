"""No device used, < 10 s: libqflash sets its environment defaults when it is loaded, before any thread exists (issue
#11: a setenv in flash::Worker::Worker, which runs on one thread per card, moved environ under another card's getenv).
- Worker::Worker has no setenv, and qe_create's pp setenvs wait for the n-gram thread (source checks, need no build);
- QE_GEMM_TUNE after loading build/libqflash.so: 0 when unset or empty, a value set by the user is kept. The library is
  only loaded, with CUDA_VISIBLE_DEVICES empty; skipped when it, or the CUDA libraries it links to, can't be loaded.
  python tests/test_flash_load_env.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
LIB = os.path.join(ROOT, "build", "libqflash.so")

PROBE = r"""
import ctypes, sys
libc = ctypes.CDLL(None)
libc.getenv.restype = ctypes.c_char_p
ctypes.CDLL(sys.argv[1])
v = libc.getenv(b"QE_GEMM_TUNE")
print("unset" if v is None else v.decode())
"""


def test_no_setenv_in_worker_ctor():
    src = open(os.path.join(ROOT, "src", "flash", "flash.cu")).read()
    start = src.index("\nWorker::Worker(")
    body = src[start:src.index("\n}\n", start)]
    assert "setenv(" not in body, "a setenv in Worker::Worker races the other cards' getenv (issue #11)"


def test_pp_setenvs_wait_for_ngram_thread():
    src = open(os.path.join(ROOT, "src", "flash", "flash_capi.cpp")).read()
    start = src.index("std::thread ng_th([&] {")
    first = min(src.index(name, start) for name in ('setenv("FLASH_PP_P2P"', 'setenv("FLASH_PP_H2D_GBPS"', 'setenv("FLASH_LOOKUP"'))
    assert "ng_done.wait();" in src[start:first], "qe_create's pp setenvs run while the n-gram thread reads the environment (issue #11)"


def load(value):
    env = {k: v for k, v in os.environ.items() if k != "QE_GEMM_TUNE"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    if value is not None:
        env["QE_GEMM_TUNE"] = value
    out = subprocess.run([sys.executable, "-c", PROBE, LIB], env=env, capture_output=True, text=True, timeout=60)
    if out.returncode and "cannot open shared object" in out.stderr:
        return None
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_defaults_at_load():
    if not os.path.exists(LIB):
        print("skip: no build/libqflash.so")
        return
    for value, want in ((None, "0"), ("", "0"), ("   ", "0"), ("0", "0"), ("1", "1")):
        got = load(value)
        if got is None:
            print("skip: the CUDA libraries are not on this host")
            return
        assert got == want, f"QE_GEMM_TUNE={value!r}: {got!r}, wanted {want!r}"


if __name__ == "__main__":
    test_no_setenv_in_worker_ctor()
    test_pp_setenvs_wait_for_ngram_thread()
    test_defaults_at_load()
    print("ok")
