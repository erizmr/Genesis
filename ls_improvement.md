# Parallel Linesearch Improvement Plan

## Background

The parallel grid-search linesearch (K=32 candidates, N_REFINE=3 passes) replaces the sequential Newton-based linesearch in the rigid body constraint solver. However, **it is currently slower than main**, primarily because ~30% of environments fail to converge within 10 solver iterations (vs 0% on main). The root cause: grid search finds imprecise alpha values that produce small-but-above-tolerance improvements, keeping envs active without making real progress toward convergence. The unconverged envs force extra work in hessian/gradient kernels across all iterations.

### Key data from profiling (g1_fall, 4096 envs, RTX 5090)

| Metric | Main (decomposed solver) | Parallel LS (K=32, R=3) |
|--------|----------------------|------------------------|
| FPS | **521,226** | 511,750 |
| GPU kernel time | **2,560 us** | 7,345 us |
| Convergence rate (10 iters) | **100%** | ~70% |

Main uses a fused decomposed kernel (`kernel_step_1`) that runs the entire solver iteration (including linesearch) in a single kernel launch per step. This avoids inter-kernel launch overhead and enables per-env early exit within the kernel via `break`. The parallel LS decomposes this into separate kernels (linesearch, hessian, gradient, etc.) with 16+ launches per iteration, and its imprecise alpha selection causes 30% of envs to remain active through all 10 iterations.

### Trace-level observations

- Main's decomposed kernel: total ~2,560 us for the full solver across all iterations, with aggressive per-env early exit.
- Parallel LS: total ~7,345 us — the decomposed kernels cannot early-exit as efficiently (each kernel launch has fixed overhead), and unconverged envs keep all per-iteration kernels busy.
- The parallel version has +134 extra kernel launches vs main's ~49, causing significant inter-kernel gap overhead.
- The linesearch itself is faster in the parallel version (~828 us vs the linesearch portion embedded in main's decomposed), but the convergence regression negates this advantage and more.

---

## Goal 1: Improve convergence rate of parallel linesearch

### 1A. Parabolic interpolation after grid search (primary improvement)

After the grid evaluation identifies the best candidate `alpha_best` at index `best_idx`, use the three neighboring cost values to fit a parabola and compute the analytic minimum:

```
a_l, a_b, a_r = alpha[best_idx-1], alpha[best_idx], alpha[best_idx+1]
c_l, c_b, c_r = cost[best_idx-1],  cost[best_idx],  cost[best_idx+1]

num = (a_b - a_l)**2 * (c_b - c_r) - (a_b - a_r)**2 * (c_b - c_l)
den = (a_b - a_l)   * (c_b - c_r) - (a_b - a_r)   * (c_b - c_l)
alpha_refined = a_b - 0.5 * num / den
```

**Where to implement:** At the end of the final refinement pass in `_kernel_parallel_linesearch_eval`, after the shared-memory argmin reduction finds `best_idx`. The refined alpha replaces `alpha_best` before being written to the output. This is thread 0's existing post-reduction block — the natural insertion point.

#### Obtaining the three neighbor costs

The shared-memory argmin reduction overwrites `sh_cost` during the tree reduction, so the original per-candidate costs are destroyed before the parabolic fit can read them. Two options:

**Option A (preferred): Preserve costs in a second shared array.** Before the tree reduction begins, copy `sh_cost` to a parallel `sh_cost_orig[K]` array. Cost: K×8 bytes extra shared memory (256 bytes for K=32, float64) — negligible. Thread 0 reads `sh_cost_orig[best_tid-1]`, `sh_cost_orig[best_tid]`, `sh_cost_orig[best_tid+1]` after reduction.

**Option B: Recompute three costs.** Thread 0 recomputes cost at the three alpha values using the same formula as the eval loop. This avoids extra shared memory but requires re-executing the constraint loop (friction + contact terms) for 3 alphas. Cost: `3 × n_constraints` FLOPs by thread 0 alone — acceptable for typical constraint counts but adds serial work.

Option A is simpler and faster. The 256 bytes of extra shared memory is well within limits (current usage is already K×16 bytes for cost+index).

#### Boundary handling

When `best_idx == 0` or `best_idx == K-1`, the best candidate is at the edge of the search grid. In this case:
- `a_l == a_b` (or `a_r == a_b`), making the parabolic denominator zero.
- The safety guard catches this and falls back to the grid-point alpha.
- **Note:** A boundary-hitting best candidate often indicates the optimal alpha lies outside the search range entirely. This is acceptable — the grid search still finds the best available alpha, and the solver will correct in subsequent iterations.

#### Safety guard

Only accept `alpha_refined` if:
1. `|den| > EPS` (the parabola is non-degenerate)
2. `alpha_refined` falls within `[a_l, a_r]` (interpolation, not extrapolation)
3. **Cost validation:** Compute the cost at `alpha_refined` using the same formula used for grid candidates and verify it is lower than `c_b`. This is more robust than checking the parabola's concavity sign, as it catches cases where the cost function has kinks at constraint activation boundaries (friction/contact regime changes) that make the parabolic approximation unreliable.

Otherwise, fall back to the grid-point `alpha_best`.

The cost validation adds ~`n_constraints` FLOPs for one alpha evaluation by thread 0. This is the same cost as Option B for a single alpha and is worthwhile for robustness.

**Total cost:** 256 bytes extra shared memory + ~`n_constraints` FLOPs for the validation eval. Zero additional kernel launches.

**Expected impact:** The parabolic fit locates the near-exact minimum within the bracket, approximating what the sequential Newton linesearch achieves with derivative information. This should significantly reduce the 30% non-convergent env count, especially in later solver iterations where the cost function is smooth and well-approximated by a parabola. Near constraint activation boundaries, the safety guard ensures we never do worse than the grid-point alpha.

---

## Goal 2: Adaptive N_REFINE based on precision threshold

### Motivation

Fixed `N_REFINE=3` is wasteful: envs with large steps (early iterations) converge with 1 pass, while envs near the tolerance boundary may need 4-5 passes. The sweep data confirms diminishing returns: `N=2→3` saves 251 us in non-LS time, but `N=3→4` loses 96 us.

### Design: Relative alpha change stopping criterion

Use a simple, intuitive stopping condition: **refine until the best alpha stops changing significantly between passes.**

After each refinement pass, compare the new best alpha to the previous pass's best alpha:

```
stop_refining = |alpha_new - alpha_prev| / max(|alpha_prev|, EPS) < rtol
```

where `rtol` is a relative tolerance parameter (e.g., `0.01` — alpha has stabilized to within 1%).

This is simpler than the bracket-cost-difference heuristic (one fewer tunable parameter, no dependency on `p0_cost`) and has a more intuitive interpretation: "stop when additional passes aren't moving the answer."

#### Implementation

Store `alpha_prev` in a spare slot of the existing `candidates` array (slots 5-11 are available). No new fields needed in `ConstraintState`.

```python
# In _kernel_parallel_linesearch_eval, thread 0 post-reduction:
alpha_prev = candidates[5, i_b]   # from previous pass (0.0 on first pass)
alpha_new  = sh_alpha[best_tid]

if refine_pass > 0:
    rel_change = abs(alpha_new - alpha_prev) / max(abs(alpha_prev), EPS)
    needs_refine = (rel_change > rtol) and improved[i_b]
else:
    needs_refine = True  # always do at least 2 passes

candidates[5, i_b] = alpha_new  # save for next pass
```

The Python-side loop:

```python
LS_PARALLEL_MAX_REFINE = 5  # hard cap

for refine_pass in range(LS_PARALLEL_MAX_REFINE):
    _kernel_parallel_linesearch_eval(..., refine_pass=refine_pass)
    # Each thread block checks needs_refine[i_b] at top; returns immediately if false
```

Inside `_kernel_parallel_linesearch_eval`, the `needs_refine` state is stored in `candidates[6, i_b]` (as 0.0/1.0) and checked at the kernel entry point. Thread blocks for converged envs return after a single flag read.

#### Interaction with parabolic interpolation (1A)

The parabolic interpolation runs only on the **final** refinement pass (when `needs_refine` becomes false or `refine_pass == MAX_REFINE - 1`). This way, the grid narrows to a tight bracket first, then the parabola provides sub-grid precision within that bracket.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LS_PARALLEL_MAX_REFINE` | 5 | Hard cap on refinement passes |
| `LS_PARALLEL_RTOL` | 0.01 | Relative alpha change threshold for stopping |

These replace the current `LS_PARALLEL_N_REFINE = 3` constant.

#### Cost model

| Scenario | Avg passes | LS time (est.) | vs fixed R=3 |
|----------|-----------|----------------|--------------|
| Fixed R=3 | 3.0 | 828 us | baseline |
| Adaptive (rtol=0.01) | ~2.2 | ~780 us | -6% |
| Adaptive + parabolic (1A) | ~1.8 | ~760 us | -8% |

The main win is not LS time savings (modest) but **convergence improvement**: envs near the tolerance boundary can get up to 5 passes for high precision, while envs making large steps stop at 2 passes. This should reduce the unconverged env count without increasing average LS cost.

---

## Deferred: Adaptive search range (1B)

Narrowing the search range based on convergence state was considered but is **deferred** pending results from 1A + Goal 2. Reasons:
- Adds conditional branching in the hot path
- Requires passing `prev_cost` between solver iterations (cross-iteration coupling)
- The `improvement_ratio` thresholds (100, 10) are arbitrary without empirical validation
- Parabolic interpolation (1A) already provides sub-grid precision, which may make range narrowing redundant

If convergence remains below 95% after implementing 1A + Goal 2, this can be revisited with data-driven threshold selection.

---

## Implementation Order

1. **1A (parabolic interpolation)** — Highest expected impact on convergence, minimal cost. Implement and measure convergence rate change.
2. **Goal 2 (adaptive refine)** — Replace fixed N_REFINE with alpha-change-based adaptive loop. Measure both convergence and throughput.
3. **Validation** — Run `GS_PROFILING=1 pytest -m benchmarks -v -k g1_fall --profile-wait 275` and compare against:

   **Baseline (main, decomposed solver): FPS = 521,226, GPU kernel time = 2,560 us, 100% convergence**

   Targets:
   - **FPS: >521k** (must beat main to justify the parallel LS approach)
   - **Convergence rate: >95%**, ideally matching main's 100%
   - **GPU kernel time: <2,560 us** (main's decomposed total)
   - Per-iteration alpha values vs main's sequential linesearch (sanity check)

   The parallel LS currently loses to main (511k vs 521k FPS, 7,345 vs 2,560 us GPU time) entirely due to the convergence regression. If convergence is fixed, the per-iteration linesearch speedup (~828 us parallel vs embedded sequential) should translate to overall wins — but only if the decomposed kernel launch overhead doesn't eat the savings. This is the critical question the validation must answer.

4. **Extended validation** — Test on scenes with different constraint profiles (many equality constraints, friction-dominated, few contacts with high DOF counts) to ensure improvements generalize beyond the single g1_fall benchmark.
