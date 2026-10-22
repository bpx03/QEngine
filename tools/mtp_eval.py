"""Draft acceptance of an MTP head in the engine on held-out prompts (not in the training data):
tokens per speculative step with a fixed draft count, greedy and sampled.
  QE_MTP_FILE=<head.safetensors> python tools/mtp_eval.py [n_prompts] [k]
  MTP_EVAL_TEMPS (default "0,0.7"), MTP_SAMPLE_SEED (sampling seed, default 7), MTP_EVAL_SEED (prompts)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QE_MTP_ADAPT", "0")  # fixed k: acceptance, not the cost model
import mtp_data as D
from tokenizers import Tokenizer
from app import Template
from engine import Engine

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
k = int(sys.argv[2]) if len(sys.argv) > 2 else 7
tok = Tokenizer.from_file(os.path.join(D.MODEL, "tokenizer.json"))
tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "templates", "qwen38_syssafe.jinja")
tmpl = Template(tp if os.path.exists(tp) else os.path.join(D.MODEL, "chat_template.jinja"))
eos = {tok.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")}
prompts = D.make_prompts(n, int(os.environ.get("MTP_EVAL_SEED", "12345")))
e = Engine(D.MODEL, max_t=32768)
for temp in [float(t) for t in os.environ.get("MTP_EVAL_TEMPS", "0,0.7").split(",")]:
    e.set_sampling(temp, 0.95, 40, 0.0, int(os.environ.get("MTP_SAMPLE_SEED", "7")))
    toks = steps = 0
    t0 = time.time()
    for p in prompts:
        ids = tok.encode(tmpl.render(p["messages"], None, enable_thinking=p["think"]), add_special_tokens=False).ids
        e.start(ids, k)
        got = 1
        while got < 300:
            out = e.step()
            got += len(out)
            steps += 1
            if any(t in eos for t in out):
                break
        toks += got - 1
    print(f"temp {temp}: {toks / steps:.3f} tokens/step over {steps} steps ({toks / (time.time() - t0):.0f} tok/s incl. prefill)", flush=True)
e.close()
