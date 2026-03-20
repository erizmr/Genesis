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
