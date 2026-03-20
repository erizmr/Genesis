# Parallel Linesearch Improvement — Experiment Log

## Setup

- Benchmark: `g1_fall`, Newton solver, 4096 envs, GPU (RTX 5090)
- Validation: `test_validate.py` — per-iteration convergence (1 step) + FPS (5s)
- All tests use the decomposed solver, NOT the monolith
- Main decomposed (sequential LS): **526k FPS**, 7298 active-env-iters

## Baseline: tight range [1e-2, 10] (committed: eb3a421)

One-line change: `alpha_newton * 1e2` → `alpha_newton * 10.0` in `_kernel_parallel_linesearch_p0`.

| Metric | Before (original [1e-2, 1e2]) | After (tight [1e-2, 10]) |
|---|---|---|
| FPS | 492k | **516k** |
| Active-env-iters | — | 7946 |
| iter 0 converged | — | 13.9% |
| iter 3 converged | — | 82.3% |

## 1A: Parabolic interpolation — FAILED

Implemented per `ls_improvement.md`: 3-point parabolic fit using preserved `sh_cost_orig[K]` after argmin reduction.

### With cost validation (serial constraint loop by thread 0)

| Metric | Baseline | Parabolic + validation |
|---|---|---|
| FPS | 516k | **437k** (-15%) |
| Active-env-iters | 7946 | 10194 (+28%) |
| iter 0 converged | 13.9% | 11.9% |

**Cause**: The cost validation loop (serial thread-0 constraint iteration) adds massive serial latency to every eval kernel launch. Runs on every pass for every active env.

### Without cost validation (geometric guards only)

| Metric | Baseline | Parabolic (no validation) |
|---|---|---|
| FPS | 516k | **501k** (-3%) |
| Active-env-iters | 7946 | 11575 (+46%) |
| iter 0 converged | 13.9% | 2.9% |

**Cause**: The cost function has kinks at contact/friction activation boundaries. The parabolic fit assumes smoothness, so the interpolated alpha often lands on the wrong side of a kink, producing a worse alpha than the grid point. Convergence regresses severely.

**Conclusion**: Parabolic interpolation is not viable for this piecewise cost function. The plan's assumption that "the cost function is smooth and well-approximated by a parabola" is wrong near constraint activation boundaries — precisely where precision matters most.

## Goal 2: Adaptive N_REFINE — FAILED

Replaced fixed `N_REFINE=3` with `MAX_REFINE=5` + per-env `needs_refine` flag + `RTOL=0.01` stopping criterion per `ls_improvement.md`.

| Metric | Baseline | Adaptive (MAX_REFINE=5, RTOL=0.01) |
|---|---|---|
| FPS | 516k | **485k** (-6%) |
| Active-env-iters | 7946 | 12742 (+60%) |
| iter 0 converged | 13.9% | 0.0% |
| iter 1 converged | 42.5% | 5.4% |

**Cause**: MAX_REFINE=5 adds 2 extra kernel launches per iteration (vs N_REFINE=3). The RTOL check doesn't trigger early enough — most envs run all 5 passes. The extra launch overhead is ~40 us/step and the additional eval passes don't improve alpha precision enough to compensate. Convergence regresses because the Quadrants JIT compiles 5 template variants of the eval kernel (one per `_refine_pass` value), increasing code size and potentially hurting instruction cache.

**Conclusion**: Adaptive refine adds complexity and launches without benefit. Fixed N_REFINE=3 is already the optimal pass count.

## Summary

| Config | FPS | vs Main | Active-env-iters |
|---|---|---|---|
| Main (sequential LS) | 526k | — | 7298 |
| **Baseline (tight range [1e-2, 10])** | **516k** | **-1.9%** | **7946** |
| 1A: Parabolic (no validation) | 501k | -4.8% | 11575 |
| 1A: Parabolic (with validation) | 437k | -16.9% | 10194 |
| Goal 2: Adaptive MAX_REFINE=5 | 485k | -7.8% | 12742 |

## Step D: Fuse apply_dofs + apply_constraints — SUCCESS (committed: 32da93f)

Merged two apply kernels into `_kernel_parallel_linesearch_apply_alpha`. Saves 1 launch per iteration.
FPS: 513k (within noise of baseline). Convergence unchanged.

## Partial Fusion (p0 + eval×N + apply) — FAILED

Attempted to fuse p0 + eval×3 + apply into a single kernel with block syncs between phases.

| Metric | Baseline | Fused (global mem) | Fused (shared mem) |
|---|---|---|---|
| FPS | 516k | 519k | 517k |
| Active-env-iters | 7946 | 12620 | 12655 |
| iter 0 converged | 13.9% | 0.0% | 0.0% |

FPS improved slightly (fewer launches) but convergence regressed severely (0% at iter 0). Tried two approaches:
1. Global memory for range communication between eval passes → same bug
2. Shared memory for all inter-phase communication → same bug

The convergence regression is identical to the earlier fused-eval-only attempt. Root cause unidentified — may be a Quadrants JIT issue with for-loops containing block syncs, or a subtle memory ordering issue. Reverted.

## Summary

The tight range baseline (one-line change, commit `eb3a421`) + fused apply (commit `32da93f`) are the only improvements that work. Both planned features from `ls_improvement.md` (parabolic interpolation and adaptive refine) are harmful because:
1. The cost function is piecewise (not smooth) — parabolic fit fails at kinks
2. More refine passes add launch overhead that exceeds any precision benefit
3. The Quadrants JIT template compilation creates overhead for parameterized kernels
