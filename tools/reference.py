"""Golden reference dump from HF transformers for engine verification.

Run with the reference venv:  .venv-ref/bin/python tools/reference.py
Writes refs/ref_<i>.pt with: prompt ids, prefill logits (last pos), per-layer hidden
states for the prompt, greedy continuation ids and the logits of every greedy step.
"""
import argparse
import os

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "Write a Python function that checks whether a number is prime.\n\ndef is_prime(n):",
    "Opowiedz krótko, czym jest mechanika kwantowa.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.8-27B-INT4")
    ap.add_argument("--out", default="refs")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        # HF decompresses INT4 to BF16 (~54 GB); accelerate's multi-GPU placement breaks on this
        # multimodal model, so the golden run happens on CPU RAM
        args.model, dtype=torch.bfloat16, device_map="cpu",
    ).eval()
    print(model.hf_device_map if hasattr(model, "hf_device_map") else "")

    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hidden = [h[0].float().cpu() for h in out.hidden_states]  # emb + after each layer (last is normed)
            prefill_logits = out.logits[0].float().cpu()

            gen = model.generate(
                ids, max_new_tokens=args.steps, do_sample=False, output_logits=True,
                return_dict_in_generate=True,
            )
        gen_ids = gen.sequences[0, ids.shape[1]:].cpu()
        step_logits = torch.stack([l[0].float().cpu() for l in gen.logits])
        torch.save(
            dict(prompt=prompt, ids=ids[0].cpu(), prefill_logits=prefill_logits,
                 hidden=hidden, gen_ids=gen_ids, step_logits=step_logits),
            f"{args.out}/ref_{i}.pt",
        )
        print(f"[{i}] {prompt!r} -> {tok.decode(gen_ids)!r}")


if __name__ == "__main__":
    main()
