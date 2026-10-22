"""Golden reference for an image prompt (run with the reference venv, on CPU like reference.py):
  .venv-ref/bin/python tools/reference_vision.py <image> [out.pt]
Saves prompt ids (image tokens expanded), M-RoPE position ids, last-position logits and a greedy
continuation."""
import sys, time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "models/Qwen3.8-27B-INT4"
img = Image.open(sys.argv[1]).convert("RGB")
out = sys.argv[2] if len(sys.argv) > 2 else "refs/vision_ref.pt"
Q = "What does this screenshot show? List the error, the warning and the three resource percentages."
proc = AutoProcessor.from_pretrained(MODEL)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
inp = proc(text=[text], images=[img], return_tensors="pt")
print({k: tuple(v.shape) for k, v in inp.items()})
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu").eval()
t0 = time.time()
with torch.no_grad():
    pos, delta = model.model.get_rope_index(inp["input_ids"], inp["mm_token_type_ids"], inp["image_grid_thw"])
    r = model(**inp, use_cache=False)
    logits = r.logits[0, -1].float()
    print(f"prefill {time.time() - t0:.0f}s; top1", logits.argmax().item())
    g = model.generate(**inp, max_new_tokens=40, do_sample=False)
gen = g[0, inp["input_ids"].shape[1]:].tolist()
print(proc.tokenizer.decode(gen))
torch.save({"text": text, "ids": inp["input_ids"][0].tolist(), "pos": pos[:, 0].T.tolist(), "delta": int(delta[0]),
            "logits": logits, "greedy": gen, "grid": inp["image_grid_thw"][0].tolist()}, out)
print(f"saved {out} in {time.time() - t0:.0f}s")
