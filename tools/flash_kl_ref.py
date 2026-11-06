#!/usr/bin/env python3
"""Next-token KL of a Flash-Next model directory against the official BF16 model, on jyohukuchan's public reference
set (huggingface.co/datasets/jyohukuchan/qwen38-flash-next-bf16-logits), scored with the set's own compare.py.

  hf download jyohukuchan/qwen38-flash-next-bf16-logits --repo-type dataset \
      --revision 5294aa9ab94b0956742fc4c6528c21c1c3c08e42 --local-dir data/kl-ref \
      --include "reference/*" "corpus/*" compare.py vocab.json manifest.json       # ~32 GB
  python3 tools/flash_kl_ref.py --model models/Qwen3.8-Flash-Next-QE-INT4 --ref data/kl-ref --work /path/with/65GB
  ... --cases en_prose_129,ja_prose_129      # a quick check on a few cases (not the published full-set numbers)

Steps: the files used from the download are checked against the dataset's manifest.json (sha256); every position of
every case goes through the engine's teacher-forced mode (flash_run FLASH_TF, FLASH_TF_ROW0=1: decode-path forwards
of up to 16 tokens, 3 cards by default), the full-vocabulary log-softmax rows are written as the set's candidate layout
(<case>.f32 + manifest.json), then compare.py runs. Prints the macro case-mean KL, the top-1 agreement (both from
compare.py) and the WikiText-2 KL (the wiki2_* rows after their 55-token chat-template prefix, the set's own
WikiText-2 definition). Needs ~32 GB for the engine output and ~32 GB for the candidate files under --work.
"""
import argparse, hashlib, json, os, subprocess, sys
import numpy as np

V = 248320
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--ref", required=True, help="the downloaded dataset directory")
ap.add_argument("--work", required=True, help="scratch directory (~64 GB)")
ap.add_argument("--flash-run", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "flash_run"))
ap.add_argument("--devices", default="0,1,2")
ap.add_argument("--keep", action="store_true", help="keep the engine output and candidate files")
ap.add_argument("--cases", default="", help="comma-separated case ids: score only these (a quick check; the full-set "
                "numbers need every case)")
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
man_ref = {x["path"]: x["sha256"] for x in json.load(open(os.path.join(a.ref, "manifest.json")))["files"]}
need = ["compare.py", "vocab.json", "corpus/all-cases.json", "reference/manifest.json"]
ref_cases = json.load(open(os.path.join(a.ref, "reference", "manifest.json")))["cases"]
pick = [x for x in a.cases.split(",") if x]
if pick:
    unknown = sorted(set(pick) - {c["id"] for c in ref_cases})
    assert not unknown, f"--cases: not in the set: {', '.join(unknown)}"
    ref_cases = [c for c in ref_cases if c["id"] in pick]
need += ["reference/" + c["logits_file"] for c in ref_cases]
for rel in need:
    h = hashlib.sha256()
    with open(os.path.join(a.ref, rel), "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    assert man_ref.get(rel) == h.hexdigest(), f"{rel}: sha256 does not match the dataset manifest"
corpus = os.path.join(a.ref, "corpus", "all-cases.json")
cases = json.load(open(corpus))["cases"]
refdir = os.path.join(a.ref, "reference")
if pick:  # compare.py scores an identical case set: the slice's inputs and reference manifest (files linked)
    cases = [c for c in cases if c["id"] in pick]
    corpus = os.path.join(a.work, "cases.json")
    json.dump({"cases": cases}, open(corpus, "w"))
    refdir = os.path.join(a.work, "reference")
    os.makedirs(refdir, exist_ok=True)
    rm = json.load(open(os.path.join(a.ref, "reference", "manifest.json")))
    rm["cases"] = ref_cases
    json.dump(rm, open(os.path.join(refdir, "manifest.json"), "w"))
    for c in ref_cases:
        dst = os.path.join(refdir, c["logits_file"])
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(a.ref, "reference", c["logits_file"])), dst)
tf = os.path.join(a.work, "tf.txt")
with open(tf, "w") as f:  # one case per line, "N:ids" with N = n (every position)
    for c in cases:
        f.write(f"{len(c['token_ids'])}:" + ",".join(map(str, c["token_ids"])) + "\n")
out = os.path.join(a.work, "engine.f32")
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.devices, FLASH_TF=tf, FLASH_TF_OUT=out, FLASH_TF_ROW0="1")
subprocess.run([a.flash_run, "--model", a.model, "--ids", "1,2", "--gen", "1", "--k", "3"], env=env, check=True)
tot = sum(len(c["token_ids"]) for c in cases)
x = np.memmap(out, dtype="<f4", mode="r")
assert x.size == tot * V, (x.size, tot * V)
x = x.reshape(tot, V)
cand = os.path.join(a.work, "candidate")
os.makedirs(cand, exist_ok=True)
man = {"model": os.path.basename(os.path.normpath(a.model)), "engine": "QEngine flash_run (teacher-forced decode)",
       "dtype": "log-softmax fp32 (full vocabulary)", "cases": []}
off = 0
for c in cases:
    n = len(c["token_ids"])
    nf = 0
    with open(os.path.join(cand, c["id"] + ".f32"), "wb") as f:
        for b in range(0, n, 256):
            r = np.asarray(x[off + b:off + min(n, b + 256)])
            nf += int((~np.isfinite(r)).sum())
            f.write(r.tobytes())
    off += n
    h = hashlib.sha256(json.dumps(c["token_ids"], separators=(",", ":")).encode()).hexdigest()
    man["cases"].append({"id": c["id"], "logits_file": c["id"] + ".f32", "shape": [n, V], "nonfinite_count": nf,
                         "positions": list(range(n)), "input_token_ids_sha256": "sha256:" + h})
json.dump(man, open(os.path.join(cand, "manifest.json"), "w"))
del x
res = os.path.join(a.work, "compare.json")
subprocess.run([sys.executable, os.path.join(a.ref, "compare.py"), "--reference", refdir,
                "--candidate", cand, "--inputs", corpus, "--vocab", os.path.join(a.ref, "vocab.json"), "--output", res],
               check=True, stdout=subprocess.DEVNULL)
r = json.load(open(res))
wiki = [k for c in r["cases"] if c["id"].startswith("wiki2") for p, k in zip(c["positions"], c["position_kld"]) if p >= 55]
summary = {"macro_case_mean_kl": r["primary_macro_case_mean_kld"], "top1_agreement": r["primary_top1_agreement"],
           "wikitext2_kl": float(np.mean(wiki)) if wiki else None, "positions": r["primary_excluding_repeat_control"]["count"]}
if pick:
    summary["cases"] = pick
json.dump(summary, open(os.path.join(a.work, "summary.json"), "w"), indent=1)
print(json.dumps(summary))
if not a.keep:
    os.remove(out)
    for c in cases:
        os.remove(os.path.join(cand, c["id"] + ".f32"))
