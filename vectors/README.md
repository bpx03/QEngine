# Steering vectors

Activation-steering vectors shipped with QEngine. They are loaded only when you ask for them
(`QE_STEERING=1` or `QE_STEERING=<name>` in Docker, or `QE_STEER` / `QE_STEER_DIR` for `server/run.sh`);
without that the server runs exactly as if this directory did not exist. In Docker your own vectors
(`*.gguf` + sidecar `.json`) mounted at `/vectors` are loaded next to these.

## refusal_projection (Qwen3.8-Flash-Next only)

| file | what |
|---|---|
| `refusal_projection.gguf` | control-vector GGUF, fp32 per-layer directions for layers 8-33 (hidden size 2560) |
| `refusal_projection.json` | sidecar: mode `project`, layers 8-33, scale 1.0, sha256 of the GGUF, provenance |

Projects out a refusal direction from the residual stream: the model declines far fewer requests.
Removing refusals removes a safety behaviour; you are responsible for the output. It is OFF by default.
It is valid only for Qwen3.8-Flash-Next (QE INT4); the 27B refuses it at start (other hidden size).

Use (Docker): `QE_STEERING=1`. With it on, every request is steered unless it sends
`"steering": false` (or picks another loaded vector by name); `GET /v1/steering` lists the vectors.

**Credit.** The direction was measured and published by the Strata project: Niko1221/Strata
(https://github.com/Niko1221/Strata) and its fork sergqwer/Strata-pr (https://github.com/sergqwer/Strata-pr,
commit 841f8c9, `data/experimental-speed-projection`, MIT). The fp32 bytes are unchanged; only the file name
and the sidecar are ours.

**License.** The vector is derived from Qwen3.8-Flash-Next activations, so the model's license applies:
Qwen Community License 1.0. It is not covered by QEngine's own license.
