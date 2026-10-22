"""Disk prompt cache (QE_DISK_CACHE): a conversation is saved by persist(), the engine is
recreated (as after a server restart) and
  1. the conversation continues from the disk archive (output must match a fresh computation),
  2. a new conversation with the same system prompt resumes from the archived checkpoint at the
     end of the system prompt (hint).
Phases run in separate processes. Usage: python3 test_disk_cache.py [cache_dir]"""
import os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
# fixed GEMM configurations: "new B" compares two different computations of the same rows (fresh:
# GEMM chunks of 1024 / 256 / 207 tokens; load: the archived prefix from the save phase's chunks plus
# a 13-token GEMV tail), and the autotuner's per-bucket kernel choices (tile 32 / 64 / 128) round
# differently. Even with one shared tuning table (bit-identical across processes) a
# near-tie flips: new B 46/60 for one table, 12/60 for another; 60/60 with the cost model alone
os.environ.setdefault("QE_GEMM_TUNE", "0")
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
N = 60


def prompts():
    from tokenizers import Tokenizer
    import transformers
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    gdir = os.path.join(os.path.dirname(transformers.__file__), "generation")
    sys_txt = open(os.path.join(gdir, "utils.py")).read()[:30000]
    system = tok.encode("<|im_start|>system\nYou are a code assistant. Reference:\n" + sys_txt + "<|im_end|>\n").ids
    turn = lambda q: tok.encode("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n").ids
    return tok, system, turn


def gen(e, ids, hint=0):
    t0 = time.time()
    t, r, s = e.start(ids, 3, 0, hint)
    dt = time.time() - t0
    out = [t]
    while len(out) < N:
        out += e.step()
    return out, r, dt  # all accepted tokens (MTP may accept past N): the slot's live end


def phase(name, cache):
    from engine import Engine
    tok, system, turn = prompts()
    A = system + turn("Summarize what generate() does.")
    if cache:
        os.environ["QE_DISK_CACHE"] = cache
    e = Engine(MODEL, max_t=65536)
    e.set_sampling(0)
    if name == "save":
        outA, r, dt = gen(e, A, hint=len(system))
        open(os.path.join(cache, "outA.txt"), "w").write(" ".join(map(str, outA)))
        t0 = time.time()
        e.persist()
        print(f"save: A {len(A)} tok in {dt:.1f}s, persist {time.time() - t0:.1f}s, "
              f"{e.num_disk_archives()} archives on disk")
    else:
        outA = [int(x) for x in open(os.path.join(cache if cache else sys.argv[2], "outA.txt")).read().split()]
        A2 = A + outA + tok.encode("<|im_end|>\n").ids + turn("And what does the streamer do?")
        B = system + turn("Which logits processors exist?")
        # B first: only the archived checkpoint can supply the system prompt's state
        oB, rB, dtB = gen(e, B, hint=len(system))
        print(f"{name}: new conversation B {len(B)} tok: reused {rB} in {dtB:.2f}s")
        o2, r2, dt2 = gen(e, A2, hint=len(system))
        print(f"{name}: A continued {len(A2)} tok: reused {r2} in {dt2:.2f}s, from archive {e.last_from_archive()}")
        print("RESULT", " ".join(map(str, o2[:N])), "|", " ".join(map(str, oB[:N])), "|", r2, rB, len(A) + len(outA), len(system))
    e.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("save", "load", "fresh"):
        phase(sys.argv[1], sys.argv[2] if sys.argv[1] != "fresh" else "")
        sys.exit()
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "qengine-test-cache")
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache)
    run = lambda *a: subprocess.run([sys.executable, __file__, *a], capture_output=True, text=True)
    res = {}
    for ph in ("save", "load", "fresh"):
        p = run(ph, cache)
        lines = [l for l in p.stdout.splitlines() if l.startswith(("save", "load", "fresh", "RESULT", "disk cache"))]
        for l in lines:
            if not l.startswith("RESULT"):
                print(l)
        r = [l for l in lines if l.startswith("RESULT")]
        if ph != "save":
            if not r:
                print(p.stdout[-2000:], p.stderr[-2000:])
                sys.exit("FAIL")
            res[ph] = r[0][7:].split("|")
    load, fresh = res["load"], res["fresh"]
    agree = lambda x, y: next((i for i, (a, b) in enumerate(zip(x.split(), y.split())) if a != b), N)
    r2, rB, nA, nsys = map(int, load[2].split())  # nA: A + its whole reply (the archived live end)
    dA, dB = agree(load[0], fresh[0]), agree(load[1], fresh[1])
    print(f"greedy agreement disk vs fresh: continued A {dA}/{N}, new B {dB}/{N}")
    # the continuation resumes at the archived live end: A + reply minus the last generated token,
    # which was never fed (it is the next input); before the reply was saved in full (MTP accepts
    # past N) only the prompt-end checkpoint matched, which sits at len(A) - 1 since ce7443b
    ok = r2 >= nA - 1 and rB >= nsys and dA >= 20 and dB >= 20
    shutil.rmtree(cache, ignore_errors=True)
    print("PASS" if ok else "FAIL")
