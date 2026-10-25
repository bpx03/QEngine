#!/usr/bin/env python3
"""Per-token routing probability of every (layer, expert) for the Flash-Next expert cache (seed order and initial
access counts of the cold expert groups): [48][512] float32 -> MODEL_DIR/expert_prior.bin.

Input: routing counts written by flash_run with FLASH_STATS=stats.bin ([49][512] int32 per run, appended: 48 trunk
layers + the MTP layer), e.g. over a few prompts that resemble your traffic:
  FLASH_STATS=stats.bin build/flash_run --model MODEL_DIR --ids @prompt.ids --gen 512 --k 3
  python3 tools/flash_expert_prior.py stats.bin MODEL_DIR/expert_prior.bin
The same stats.bin gives the cold expert list (tools/flash_cold_experts.py). Both files are optional: without them
the engine picks cold groups by index and seeds the cache without a prior."""
import sys

import numpy as np

c = np.fromfile(sys.argv[1], dtype=np.int32).reshape(-1, 49, 512)[:, :48].sum(0).astype(np.float64)  # [48][512]
T = c[0].sum() / 10.0  # 10 routed experts per token
f = c / max(T, 1.0)
f.astype(np.float32).tofile(sys.argv[2])
print(f"{T:.0f} tokens, 48 layers -> {sys.argv[2]}; never routed: {(f == 0).sum()} of {f.size}")
