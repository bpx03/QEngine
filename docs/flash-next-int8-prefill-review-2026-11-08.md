# Flash-Next INT8 prefill — tail ownership review (2026-11-08)

## Scope and chronology

This report distinguishes the original November 8 review from the later upstream integration. Historical resource constraints below describe that review, not the current deployment.

The original candidate was `pr/flash-int8` at `21e30b8` plus uncommitted changes. No end-to-end performance measurements were taken during that review: the protected `qengine-flash` service occupied all three GPUs. A subsequent review covered runtime and loader changes, corrected build provenance and the description of main's execution order, and built a candidate without a staging-only workaround. Those historical builds and CPU checks did not exercise GPU execution.

## Tail ownership behavior

`Worker::prefill_borrow_tails(format, n, base, staged, extension_lent)` uses:

```cpp
n > base || (format == WeightFormat::Int8 && staged && extension_lent)
```

`prefill_batch` and `prefill_multi` determine staging before assigning `bt_want_` and calling `ec_tick`. When a staged, multichunk INT8 prefill needs the ring extension currently lent to the expert cache, the next tick reclaims the tails. `pfs_full_` can then use the full ring rather than the `pfs_cap0_` limit.

The INT4 predicate remains `n > base`. The ownership change did not alter `prefill_batch_mtp`, checks, `ec_tick`, the `pfs_full_` formula, graph keys, copy events, MTP, quantization, or ring capacities.

In the 1.3.6 integration, staging decisions use upstream's adaptive `pfs_min_rows()` threshold consistently; expert staging addresses and byte accounting retain the format-dependent `group_bytes_` stride.

## Original review evidence

| Check | Observed result |
| --- | --- |
| Isolated candidate build | Exit 0; tag `qengine:flash-int8-prefill-review-20261108T152059Z-r1`, image `sha256:8d439d355c80e72ef784bcb27da4e01282a1481201db6cb4fd3102649cddc316` |
| CPU checkpoint fixture | `flash_checkpoint_unit: ok`, exit 0 |
| INT8 CUDA fixture | Compiled, not executed during the original review |
| Independent prefill diff review | No blocking or should-fix findings; 40 predicate cases checked individually; job offsets, addresses and touches checked manually against `k_jobs_pf` |

The independent review excluded pre-existing loader changes. The later expanded review covered those changes and found no demonstrated runtime or loader defect in the 256-row buffers, cold scratch, or reclaim protocol. This is a review finding, not GPU execution evidence.

Historical staging: `/home/cristian/qe-staging/prefill-review-20261108T152059Z`. Candidate source SHA256 prefixes: `flash.h c1ecf1cf`, `flash.cu fe4895a2`, `flash_int8_unit.cu fa59b7e9`; pre-edit prefixes: `150909be`, `f8660f56`, `e6e8e636` respectively.

## Historical build-context defect and expanded review

The committed `.dockerignore` omitted `!tests/` although `Dockerfile.flash-int8` copied `tests`. The original build therefore required adding that exception in remote staging only. Its successful build must not be presented as a build of the unmodified repository.

The expanded review fixed `.dockerignore` in the repository and made the CUDA fixture's `require` fail immediately by throwing `std::runtime_error`. Invalid counts, descriptors and token indices can no longer continue into allocation, copies, kernels or CPU indexing. Placement equivalence is checked separately for deterministic and nondeterministic `dn_pf` and for `dn_q` tiles 1 and 2, starting each `dn_q` case with zeroed slots.

| Expanded-review check | Observed result |
| --- | --- |
| Throwaway host harness using the actual `require` macro | Before: all three invalid-input paths reached consumers. After: exception, exit 1, consumer unreachable |
| Exact 177-file snapshot | SHA256 verification passed for 177/177 files; archive `c0589335497acf5ba55da50ef6e90b358854c7040c4213a29fca11434bb20c70` |
| Build without remote staging patches | Exit 0; `COPY tests ./tests` executed; image `sha256:571ad83955641db14328b3e15722a0b8f85eea944629d7ef369d101bb7859d25`, tag `qengine:flash-int8-prefill-review-20261108T153717Z-r1` |
| CPU checkpoint fixture | `flash_checkpoint_unit: ok`, exit 0 |
| Protected service before and after | Container `c51d60c9a4f4` remained healthy, with unchanged three-GPU ownership |

Expanded-review staging: `/home/cristian/qe-staging/prefill-review-20261108T153717Z`. Exact source hashes:

- `tests/flash_int8_unit.cu`: `7e47f399da1f973d32321002a77cb52f06afae4569f897e725acfbf32cadc2b3`
- `.dockerignore`: `e015f6271e6bf12886fd10552df4a52df2a4d7bf8f286a0493c69a0d11efc565`
- `Dockerfile.flash-int8`: `023aaae79c940487090c6a78d28ea17653533d5ea953bf86f2adcc4f3f305`

Raw historical artifacts include `build-r1.log`, service-state snapshots, the snapshot archive and checksums, and `require-before.txt` / `require-after.txt` harness transcripts.

## Historical version context and remaining coverage

At the original review, main was 1.3.5 (`c07bfc9`), without the INT8 fixture. Main assigned `bt_want_ = n > pfbn_base_` before `ec_tick`, then decided staging afterward. The earlier report incorrectly reversed that order. Structural portability was not integration proof at that time.

The protected service used image `1.3.5.2-vision` with INT4 checkpoint `Qwen3.8-Flash-Next-QE-1.3.4.1`; the checkpoint suffix was not the engine version.

Original resource-blocked checks included GPU INT8 execution, paired INT8 baseline/candidate and INT4 performance, multi-GPU reclaim/restore and graph replay, loader boundaries 256/272/512/528 and cold boundaries 63/64/65, and the 1M-context/four-slot profile in `flash-next-int8-rollout-2026-11-08.md`. Later smoke tests or smaller-context campaigns do not establish that 1M/four-slot profile or a paired performance improvement.

## Upstream 1.3.6 integration

PR [#26](https://github.com/bpx03/QEngine/pull/26) incorporates upstream through merge `5f41ec5`, followed by correction `0502d78`. The integration preserves native INT8 and TP5 support alongside upstream KV4, packed down-projection, n-gram worker and slab-set changes.

The correction uses `NKV` consistently and keeps one `k_dn_rec_b` definition in `flash_kernels.cuh`, callable from production batch code and the standalone fixture. Its body retains upstream's padded shared-memory bank-conflict fix.

Observed checks for the merged code:

- `qflash`, `flash_run`, checkpoint, INT8 and TP5 fixture targets compiled.
- CPU checkpoint and GPU INT8 fixtures printed `ok`.
- The first TP5 fixture execution aborted with CUDA out-of-memory while Maya occupied the GPUs. The wrapper subsequently printed its final marker and exited zero; that wrapper exit is **not** a TP5 pass.
- Runtime image `qengine:qfn-int8-20261115T161932Z-r4` built successfully with CUDA architecture 86 and vision disabled.
- GitHub reported head `0502d7883cf1c046a9a0f5f2a696c9671f5d21ca` as `MERGEABLE`, with merge state `CLEAN`.

No throughput or latency improvement is inferred from compilation, fixture results, or mergeability.

### Final merged-build verification (2026-11-16)

With Maya stopped and all five GPUs available, `flash_tp5_unit --ranks 5`
passed host LL reduce/gather (70 graph units), FP32/FP16 staged exchanges,
and DeltaNet. The same build passed `compute-sanitizer --tool memcheck
--error-exitcode 1 ... --ranks 5 --small`, with `ERROR SUMMARY: 0 errors`.
P2P-specific transport was not exercised.

The deployed API identified version `1.3.6`, build `0502d78`, forced TP,
two slots, maximum context 131072, and host-staged/pinned-host collectives.
Concurrent nonstream `7*8` and streaming `12*12` returned `56` and `144`;
the streaming response included `[DONE]`.

The full Club campaign used three warmups and five measured runs per
narrative/code workload, plus three fresh measured prompts at each prefill
depth. Thinking was disabled; sampling was temperature 0.6, top_p 0.95,
top_k 20 and min_p 0.0. Hardware: five RTX 3090 cards at 230 W each,
PCIe Gen3 (three x16, two x8), 188.6 GiB system RAM.

| Workload | Mean wall tok/s | Mean decode tok/s | Decode CV | Mean TTFT |
| --- | ---: | ---: | ---: | ---: |
| Narrative | 88.72 | 90.06 | 2.7% | 166 ms |
| Code | 49.39 | 50.42 | 9.7% | 232 ms |

| Prefill target | Actual prompt tokens | Mean prefill tok/s | CV | Mean TTFT |
| --- | --- | ---: | ---: | ---: |
| 10K | 10367 / 10367 / 9997 | 1387.62 | 0.7% | 7382 ms |
| 90K | 93331 / 89998 / 93331 | 1485.41 | 0.3% | 62087 ms |

All five runs per decode workload were usable. Client/server decode
cross-checks agreed to displayed precision (0.0% divergence). Capture
integrity and serving-process swap checks passed. Post-run VRAM usage was
23801–23803 MiB/card. The 1060 MiB aggregate increase from the pre-benchmark
snapshot is not by itself evidence of a memory leak.

These are single-configuration observations, not an interleaved A/B test
against the previous build, INT4 or BF16, nor a quality or long-duration
soak evaluation.

Raw evidence is under
`/home/cristian/qe-int8-results-20261115T161932Z/r4-auto256/`:
`unit-tp5.log`, `memcheck-tp5.log`, `readiness.log`,
`api-smoke-results.json`, API identity snapshots, `benchmark.log`,
`container-after.json` and `server.log`.

After the campaign, QEngine was stopped cleanly (exit 0, not OOM-killed).
Maya was restored using the original `/Proyectos/Strata-958` working
directory, Python executable, configuration and command arguments.
Its `/health` endpoint returned `status=ok`, context 1048576 and images
enabled. The desktop QEngine launcher now selects the verified r4 image;
its shell syntax and executable permission were checked.
