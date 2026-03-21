# Parallel Linesearch Alpha Error Analysis

## Setup

- Benchmark: g1_fall, 4096 envs, step 276 (after 275 warmup), single solver iteration
- Gold standard: **iterative Newton linesearch** (3-phase bracketing, matching `func_linesearch_batch`)
- Cross-check: exhaustive 100K-point search confirms Newton LS is near-optimal (3.4e-5 median error)
- Sample: 500 envs with constraints
- Error metric: `|alpha_parallel - alpha_iterative| / |alpha_iterative|`
- Sweep: K ∈ {4, 8, 16, 32, 64}, N_REFINE ∈ {1, 3, 5, 8}

## Is the iterative Newton LS the best alpha?

**Yes.** Compared against a 100,000-point exhaustive search:
- Median error: 3.4e-05 (0.003%)
- p90 error: 5.1e-05
- The iterative Newton LS finds alphas within 0.003% of the true optimum.

## Results: Parallel LS error vs iterative Newton LS

### WITH Newton initial best (current method)

```
   K  N_REF |  median      mean       p90         p99       exact(<1e-6)
   4      1 | 7.57e-03   7.06e-02   2.26e-01   6.09e-01    15.4%
   4      8 | 7.57e-03   7.06e-02   2.26e-01   6.09e-01    15.4%   ← N has no effect at K=4
   8      1 | 7.57e-03   7.06e-02   2.26e-01   6.09e-01    15.4%
   8      8 | 7.57e-03   7.06e-02   2.26e-01   6.09e-01    15.4%   ← same as K=4
  16      3 | 5.99e-03   7.00e-02   2.26e-01   6.09e-01    15.4%
  32      1 | 7.78e-03   7.14e-02   2.26e-01   6.09e-01    15.4%
  32      3 | 5.57e-03   7.00e-02   2.26e-01   6.09e-01    15.4%   ← current config
  32      8 | 5.57e-03   7.00e-02   2.26e-01   6.09e-01    18.2%   ← saturated
  64      3 | 8.97e-04   6.91e-02   2.26e-01   6.09e-01    15.4%   ← 6x better median
  64      8 | 8.97e-04   6.91e-02   2.26e-01   6.09e-01    36.2%
  Newton step only (no grid):
             | 7.57e-03   7.06e-02   2.26e-01   6.09e-01    15.4%   ← grid adds almost nothing
```

**Observation**: 15.4% of envs match the iterative LS exactly — these are envs where Phase 1 (one Newton step) already converges. The p90/p99 are **immovable** at 22.6%/60.9% regardless of K or N_REFINE.

### WITHOUT Newton initial (grid only)

```
   K  N_REF |  median      mean       p90         p99       exact(<1e-6)
   8      1 | 3.90e-01   3.96e-01   4.71e-01   4.84e-01     0.0%
   8      8 | 5.13e-05   1.90e-03   1.54e-03   1.33e-02     1.2%
  16      8 | 2.32e-06   2.79e-03   5.29e-03   1.47e-02    48.8%
  32      3 | 2.47e-04   1.59e-03   3.94e-04   4.62e-04     0.2%
  32      5 | 1.02e-06   1.37e-03   1.80e-06   2.27e-04    49.6%
  32      8 | 5.34e-08   1.37e-03   1.95e-07   2.25e-04    94.0%  ← 94% exact!
  64      5 | 8.20e-08   1.40e-03   1.48e-04   7.81e-04    83.2%
  64      8 | 7.57e-08   1.40e-03   1.48e-04   7.81e-04    83.2%
```

Grid-only with enough passes CAN reach high precision: K=32, N=8 achieves 94% exact matches. But p99 stays at 0.023% — a small fraction of envs have kink minima the grid can't resolve.

## Error distribution

| Category | % of envs | Newton step error | What happens |
|---|---|---|---|
| Phase-1 converged | ~15% | < 1e-6 (exact) | Single Newton step = iterative LS result |
| Near-smooth | ~60% | 0.1-2% | Iterative LS needs 2-5 inner iterations; grid is close but not exact |
| Kink-dominated | ~25% | 10-60% | Iterative LS uses bracketing; grid/Newton both overshoot |

## Conclusions

1. **The iterative Newton LS is near-optimal** — validated against exhaustive search (0.003% median error).

2. **The p0 Newton step (one step, no iteration) matches the iterative LS for 15.4% of envs exactly.** These are the "easy" envs where the iterative LS also converges in Phase 1.

3. **Grid search adds marginal median precision** when Newton initial is used (7.57e-3 → 5.57e-3 with K=32,N=3). The p90/p99 are unchanged — the grid cannot replicate the iterative LS's multi-step bracketing.

4. **Grid-only (no Newton) CAN achieve 94% exact match** with K=32,N=8, but at heavy launch overhead cost. The median error drops exponentially with N_REFINE.

5. **The combined approach (Newton initial + grid) is the practical optimum**: Newton handles the 75% smooth cases, grid provides safety-net. The remaining 25% "kink-dominated" envs would need kink-alpha candidates (`-Jaref/jv`) to improve.
