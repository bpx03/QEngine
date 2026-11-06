"""CPU-only, < 10 s: environment switch helpers (empty = unset, unparsable = warning + default).
- server/qe_env.py (the server's switches);
- src/env.h (the engines' switches): tests/env_unit.cpp compiled with the host C++ compiler and run;
- the FLASH_COMM=auto tax rule (src/kernels/comm_rule.h): tests/comm_rule_unit.cpp, same way;
- the sampling random numbers (src/kernels/sample_rng.h): tests/sample_rng_unit.cpp, same way;
- the graph-instantiate OOM policy (src/flash/graph_oom.h): tests/graph_oom_unit.cpp, same way;
- the NaN guard abort rule (src/flash/guard_rule.h): tests/guard_rule_unit.cpp, same way;
- the host-RAM budget at boot (src/host_budget.h): tests/host_budget_unit.cpp, same way;
- the Flash-Next eviction order and archive-copy rules (src/flash/kv_evict.h): tests/kv_evict_unit.cpp, same way (property test);
- SafeTensors::page_out (src/safetensors.cpp, issue #6): tests/page_out_unit.cpp, on a disk directory (not tmpfs);
- environment writes in src/ only in the Flash-Next constructor and the 27B bench entry; `import app` starts no
  thread (issue #11).
  python tests/test_env.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from qe_env import env_float, env_int, env_str  # noqa: E402


ENV_WRITE = re.compile(r"\b(?:setenv|putenv|unsetenv|clearenv)\s*\(\s*([^,)]*)")
CTOR = "__attribute__((constructor)) static void qe_process_env"


def env_write_problems(src):
    """Environment writes under `src`, against the two allowed: the Flash-Next constructor and the 27B bench entry."""
    want = {("flash/flash.cu", '"CUDA_MODULE_LOADING"'): 1, ("capi.cpp", '"QE_DN_CHUNK"'): 1}
    got, problems = {}, []
    for root, _, names in os.walk(src):
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src)
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if rel == "flash/flash.cu":  # (the constructor counts, the rest of the file has no write at all)
                at = text.find(CTOR)
                if at < 0:
                    problems.append("flash/flash.cu: no qe_process_env")
                    continue
                end = text.index("\n}\n", at)
                ctor, text = text[at:end], text[:at] + text[end:]
                for m in ENV_WRITE.finditer(ctor):
                    key = (rel, m.group(1).strip())
                    got[key] = got.get(key, 0) + 1
                for m in ENV_WRITE.finditer("\n".join(x.split("//")[0] for x in text.split("\n"))):
                    problems.append(f"{rel}: write outside the constructor: {m.group(0)}")
                continue
            for m in ENV_WRITE.finditer("\n".join(x.split("//")[0] for x in text.split("\n"))):
                key = (rel, m.group(1).strip())
                if key in want:
                    got[key] = got.get(key, 0) + 1
                else:
                    problems.append(f"{rel}: {m.group(0)}")
    if got != want:
        problems.append(f"writes {got}, expected {want}")
    return problems


def test_env_writes():
    """setenv may realloc environ and a getenv on another thread then walks freed memory (issue #11)."""
    problems = env_write_problems(os.path.join(ROOT, "src"))
    assert not problems, problems


def test_server_import_threads():
    """Importing the server starts no thread: its environment writes and libqflash's constructor run alone (issue #11)."""
    code = ("import re, sys; sys.path.insert(0, 'server'); import app; "
            "print(re.search(r'Threads:\\s*(\\d+)', open('/proc/self/status').read()).group(1))")
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    if "ModuleNotFoundError" in out.stderr:
        print("test_server_import_threads: skipped, server dependencies missing")
        return
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.split()[-1] == "1", out.stdout


def test_python():
    n = "QE_TEST_ENV"
    os.environ.pop(n, None)
    assert env_int(n, 64) == 64 and env_float(n, 1.5) == 1.5 and env_str(n, "auto") == "auto"
    for empty in ("", "   ", "\t"):  # compose "${X:-}" pass-through
        os.environ[n] = empty
        assert env_int(n, 64) == 64 and env_float(n, 1.5) == 1.5 and env_str(n, "auto") == "auto"
        assert env_str(n) is None
    os.environ[n] = "0"
    assert env_int(n, 64) == 0 and env_float(n, 1.5) == 0.0 and env_str(n, "1") == "0"
    os.environ[n] = " 12 "
    assert env_int(n, 64) == 12 and env_float(n, 1.5) == 12.0
    os.environ[n] = "abc"  # int("abc") raised at import before; now a warning and the default
    assert env_int(n, 64) == 64 and env_float(n, 1.5) == 1.5 and env_str(n) == "abc"
    os.environ[n] = "2.5"
    assert env_int(n, 64) == 64 and env_float(n, 1.5) == 2.5
    os.environ.pop(n)


def run_cpp(name):
    cxx = next((c for c in ("g++-15", "g++", "c++", "clang++") if shutil.which(c)), None)
    assert cxx, "no C++ compiler"
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, name)
        subprocess.run([cxx, "-std=c++20", "-O1", "-I", os.path.join(ROOT, "src"),
                        os.path.join(HERE, name + ".cpp"), "-o", exe], check=True)
        r = subprocess.run([exe], capture_output=True, text=True)
        sys.stderr.write(r.stderr)
        assert r.returncode == 0, r.stdout + r.stderr
        return r


def test_cpp():
    r = run_cpp("env_unit")
    assert r.stderr.count("[env] warning: QE_T=") == 1, "one warning per variable"


def test_comm_rule():
    run_cpp("comm_rule_unit")


def test_sample_rng():
    run_cpp("sample_rng_unit")


def test_graph_oom():
    run_cpp("graph_oom_unit")


def test_guard_rule():
    run_cpp("guard_rule_unit")


def test_host_budget():
    run_cpp("host_budget_unit")



def test_kv_evict():
    r = run_cpp("kv_evict_unit")
    assert "all passed" in r.stderr


def test_page_out():
    cxx = next((c for c in ("g++-15", "g++", "c++", "clang++") if shutil.which(c)), None)
    assert cxx, "no C++ compiler"
    # a directory on the repository's filesystem: tmpfs pages cannot be paged out
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        exe = os.path.join(d, "page_out_unit")
        subprocess.run([cxx, "-std=c++20", "-O1", "-I", os.path.join(ROOT, "src"), os.path.join(HERE, "page_out_unit.cpp"),
                        os.path.join(ROOT, "src", "safetensors.cpp"), "-o", exe], check=True)
        # a root-owned file nothing maps (MADV_PAGEOUT skips files the process neither owns nor may write)
        foreign = [f for f in ("/usr/share/nvidia-kmod-615.71.09/nvidia-kmod-615.71.09-x86_64.tar.xz",
                               "/usr/share/proj/us_noaa_g2012bu0.tif", "/usr/share/mythes/th_en_US_v2.dat")
                   if os.path.isfile(f) and os.stat(f).st_uid != os.geteuid() and not os.access(f, os.W_OK)]
        r = subprocess.run([exe, d] + foreign[:1], capture_output=True, text=True)
        sys.stderr.write(r.stderr)
        assert r.returncode == 0, r.stdout + r.stderr

if __name__ == "__main__":
    test_env_writes()
    test_server_import_threads()
    test_python()
    test_cpp()
    test_comm_rule()
    test_sample_rng()
    test_graph_oom()
    test_guard_rule()
    test_host_budget()
    test_page_out()
    print("test_env: ok")
