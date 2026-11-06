"""Disk cache fingerprint vs RoPE scaling (q27_yarn review H1): a conversation saved under QE_YARN=4 (persist) must not
be reused after a restart with native RoPE; a restart with the same QE_YARN=4 (control) reuses it.
Phases run in separate processes. usage: QE_LIB=<build>/libqengine.so python3 tests/test_yarn_fingerprint.py <cache_dir>"""
import os, shutil, subprocess, sys
sys.path.insert(0, os.environ.get("QE_SERVER_DIR", os.path.join(os.path.dirname(__file__), "..", "server")))
MODEL = os.path.join(os.path.realpath(os.path.join(os.path.expanduser("~"), "GIT", "QEngine")), "models", "Qwen3.8-27B-INT4")


def prompt():
    from tokenizers import Tokenizer
    import transformers
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    txt = open(os.path.join(os.path.dirname(transformers.__file__), "generation", "utils.py")).read()[:20000]
    return tok.encode("<|im_start|>user\nSummarize:\n" + txt + "<|im_end|>\n<|im_start|>assistant\n").ids


def phase(name, cache):
    from engine import Engine
    os.environ["QE_DISK_CACHE"] = cache
    e = Engine(MODEL, max_t=65536, lib_path=os.environ.get("QE_LIB"))
    e.set_sampling(0)
    A = prompt()
    if name == "save":
        t, r, s = e.start(A, 3, 0, 0)
        for _ in range(8):
            e.step()
        e.release(s)
        e.persist()
        print(f"RESULT save: {len(A)} tok, {e.num_disk_archives()} archives on disk")
    else:
        n0 = e.num_disk_archives()
        t, r, s = e.start(A, 3, 0, 0)
        print(f"RESULT {name}: {n0} disk archives matched at start, reused {r} of {len(A)}, from archive {e.last_from_archive()}")
    e.close()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("save", "native", "yarn4"):
        phase(sys.argv[1], sys.argv[2])
        sys.exit()
    cache = sys.argv[1]
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache)
    ok = True
    for ph, env in (("save", {"QE_YARN": "4"}), ("native", {}), ("yarn4", {"QE_YARN": "4"})):
        en = {k: v for k, v in os.environ.items() if k != "QE_YARN"}
        en.update(env)
        p = subprocess.run([sys.executable, __file__, ph, cache], capture_output=True, text=True, env=en)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        print(*(lines or [f"{ph}: no result, exit {p.returncode}: {p.stderr[-400:]}"]), sep="\n", flush=True)
        if ph != "save" and lines:
            reused = int(lines[0].split("reused ")[1].split()[0])
            ok &= (reused == 0) if ph == "native" else (reused > 0)
        elif ph != "save":
            ok = False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
