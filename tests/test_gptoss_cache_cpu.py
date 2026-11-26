"""[gptoss-rearm] review2 L5: builds and runs tests/gptoss_cache_cpu.cpp (src/gptoss/capi_cache.cpp with a stub Engine:
the idle-archive fit, back-off, partial copies; no GPU, needs g++ and ~2 GiB of free host RAM; a few seconds).

  python3 tests/test_gptoss_cache_cpu.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cxx = shutil.which("g++-15") or shutil.which("g++")
    if not cxx:
        print("SKIP: no g++")
        return 0
    tmp = tempfile.mkdtemp(prefix="qe_gcache_")
    exe = os.path.join(tmp, "gptoss_cache_cpu")
    subprocess.check_call([cxx, "-O1", "-std=c++20", "-o", exe, os.path.join(HERE, "gptoss_cache_cpu.cpp"),
                           os.path.join(HERE, "..", "src", "gptoss", "capi_cache.cpp")])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QE_KV_ARCHIVE", "GPTOSS_"))}
    p = subprocess.run([exe], env=env, capture_output=True, text=True, timeout=60)
    print(p.stdout + p.stderr, end="")
    shutil.rmtree(tmp, ignore_errors=True)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
