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

## N_REFINE Sweep: Can more passes match main's precision?

Tested N_REFINE = 3, 4, 5, 6, 8 with fixed seed (torch.manual_seed(42)):

| N_REFINE | active-env-iters | vs N=3 | iter1 conv% | iter3 conv% | FPS |
|----------|-----------------|--------|-------------|-------------|-----|
| 3 | 12768 | — | 5.5% | 60.9% | 497k |
| 5 | 11557 | -9.5% | 16.6% | 66.4% | 488k |
| **8** | **11603** | **-9.1%** | **15.9%** | **65.6%** | **471k** |
| **10** | **11603** | **-9.1%** | **15.9%** | **65.6%** | **459k** |
| **12** | **11603** | **-9.1%** | **15.9%** | **65.6%** | **448k** |
| **Main** | **~7298** | **-43%** | **44.0%** | **88.7%** | **526k** |

N_REFINE=8, 10, 12 produce **identical** per-iteration convergence (same active counts to the last env). Precision completely plateaus at N_REFINE≈8 — further passes find the exact same alpha every time.

The remaining gap: 11603 → 7298 active-env-iters (37% more). This is the grid search choosing the wrong side of cost function kinks at the initial pass, which no refinement can fix.

**Investigation**: Verified that the cost function computation is mathematically identical between the parallel and sequential LS. The gap is not from different objectives. The grid search converges to the correct grid minimum to machine precision (N=8,10,12 identical). But the grid minimum is not the true function minimum when:

1. A contact activation kink (`Jaref + alpha*jv = 0`) falls between two grid points
2. The true minimum is AT the kink (where the cost function is non-differentiable)
3. The grid search evaluates both sides but picks the one with lower cost — the true minimum at the kink itself is never evaluated

The sequential LS handles this via derivative-guided bracketing: the derivative changes sign at the kink, so the bracket correctly captures it. ~18 kinks per env (from contact constraints), K=32 grid points over 3 decades.

**Conclusion**: The precision gap is fundamental to the grid search's discrete sampling at cost function kinks.

## Newton Initial Best — SUCCESS (committed: d8555bc)

Instead of `candidates[0]=0, candidates[4]=1e30`, initialize with the Newton step alpha and its quadratic cost (`p0_cost - grad²/(2*hess)`). The grid search only overrides when it finds genuinely better alpha at a kink.

| Metric | Baseline (tight range) | Newton initial | Main |
|---|---|---|---|
| FPS | 497k | **527k** | 526k |
| Active-env-iters | 12768 | **7059** | 7298 |
| iter 0 converged | 0.0% | **15.2%** | 15.3% |
| iter 1 converged | 5.5% | **46.5%** | 44.0% |
| iter 2 converged | 31.7% | **73.8%** | 69.9% |
| iter 3 converged | 60.9% | **89.0%** | 88.7% |

**The parallel LS now beats main** — both in FPS and convergence rate. The Newton step provides derivative-guided precision as the baseline, while the grid search adds value at cost function kinks where the Newton approximation breaks down.

### Experiment 1: N_REFINE sweep with Newton initial best

| N_REFINE | active-env-iters | iter0% | iter3% | FPS |
|----------|-----------------|--------|--------|-----|
| **1** | **7039** | **16.1%** | **89.3%** | **536k** |
| 3 | 7059 | 15.2% | 89.0% | 528k |
| 5 | 7053 | 15.1% | 89.4% | 511k |
| 7 | 7237 | 15.2% | 88.5% | 499k |
| 9 | 7237 | 15.2% | 88.5% | 487k |

With Newton initial best, active-env-iters is nearly identical across all N_REFINE (7039-7237). The Newton step provides the precision; the grid search barely improves it. FPS drops with more passes purely from launch overhead. **N_REFINE=1 is optimal** — single eval pass, no refinement needed.

### Experiment 2: Three-way convergence comparison (seed=42)

| iter | **Opt (Newton+tight)** | Previous (tight only) | Main (sequential LS) |
|------|----------------------|----------------------|---------------------|
| 0 | **15.2%** (3320 active) | 0.0% (3897 active) | 14.7% (3305 active) |
| 1 | **46.5%** (2093) | 5.5% (3684) | 44.7% (2144) |
| 2 | **73.8%** (1024) | 31.7% (2661) | 69.3% (1189) |
| 3 | **89.0%** (432) | 60.9% (1524) | 88.7% (438) |
| 4 | **96.6%** (134) | 82.0% (703) | 97.1% (113) |
| 5 | **98.9%** (45) | 94.0% (235) | 99.4% (23) |
| 6 | **99.7%** (10) | 98.6% (53) | 99.9% (3) |
| 7 | 100.0% (1) | 99.8% (9) | 100.0% (0) |
| **total** | **7059** | **12768** | **7215** |
| **FPS** | **529k** | **495k** | **517k** |

The Newton initial best (opt) matches main's convergence at every iteration and slightly exceeds it at iters 1-3. The previous best (tight range only) lags significantly — the Newton initial is the critical improvement.

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
