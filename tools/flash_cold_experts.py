#!/usr/bin/env python3
"""Coldest routed experts per layer from flash_run routing counts (FLASH_STATS=file, [49][512] int32: 48 trunk layers +
MTP, one block per run) -> JSON {"frac": f, "layers": [[expert ids] x 48]} for the engine (MODEL_DIR/cold_experts.json):
their groups start in pinned host memory (read over PCIe, zero copy, until the expert cache brings them in).
  python tools/flash_cold_experts.py stats.bin [frac=0.04] > MODEL_DIR/cold_experts.json"""
import json, sys
import numpy as np
c = np.fromfile(sys.argv[1], dtype=np.int32).reshape(-1, 49, 512)[:, :48].sum(0).astype(np.float64)  # runs appended
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
k = int(round(512 * frac))
layers = []
mass = []
for l in range(48):
    order = np.lexsort((np.arange(512), c[l]))  # ascending count, ties by id
    sel = sorted(int(e) for e in order[:k])
    layers.append(sel)
    mass.append(c[l, sel].sum() / c[l].sum())
print(json.dumps({"frac": frac, "routing_mass_mean": float(np.mean(mass)), "layers": layers}))
