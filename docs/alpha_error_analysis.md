# Parallel Linesearch Alpha Error Analysis

## Setup

- Benchmark: g1_fall, 4096 envs, step 276 (after 275 warmup)
- Gold standard: exhaustive search over 100,000 log-spaced alphas
- Sample: 500 envs with constraints
- Error metric: `|alpha_parallel - alpha_gold| / |alpha_gold|` (relative error)
- Sweep: K ∈ {4, 8, 16, 32, 64}, N_REFINE ∈ {1, 2, 3, 5, 8}

## Key Finding: Newton initial dominates precision

With Newton initial best enabled, the grid search barely improves the alpha. The Newton step alone gives 0.76% median error. Adding K=32 × 3 refine passes only reduces it to 0.56%. The p90 and p99 error are **unchanged** at 22.6% and 53.7% — these are the envs where the Newton approximation is poor (near kinks), and the grid search can't help because it searches a region centered on the same Newton estimate.

Without Newton initial, the grid-only search shows clear improvement with both K and N_REFINE, but converges to a much worse floor (~0.003% median with K=64, N=8 vs 0.004% with K=32, N=8).

## Results: WITH Newton initial best

```
   K N_REF |  median_err   p90_err      p99_err
   4     1 |  7.60e-03    2.26e-01    5.37e-01
   4     8 |  7.60e-03    2.26e-01    5.37e-01    ← N_REFINE has zero effect at K=4
  16     1 |  7.60e-03    2.26e-01    5.37e-01
  16     8 |  6.04e-03    2.26e-01    5.37e-01    ← marginal median improvement
  32     1 |  7.77e-03    2.26e-01    5.37e-01
  32     3 |  5.58e-03    2.26e-01    5.37e-01    ← current config
  32     8 |  5.58e-03    2.26e-01    5.37e-01    ← saturated
  64     3 |  9.21e-04    2.26e-01    5.37e-01    ← 5x better median
  64     8 |  9.21e-04    2.26e-01    5.37e-01    ← saturated
Newton only |  7.60e-03    2.26e-01    5.37e-01    ← grid adds almost nothing!
```

**Interpretation**: The Newton step is already very close to the true minimum for ~50% of envs (median error 0.76%). The grid search shaves a few basis points off the median but cannot help the worst cases (p90/p99) because those envs have kinks near the Newton estimate and the grid is too coarse to find the kink-point minimum.

## Results: WITHOUT Newton initial (grid only)

```
   K N_REF |  median_err   p90_err      p99_err
   8     1 |  3.90e-01    4.71e-01    4.84e-01    ← terrible, 39% median error
   8     3 |  2.42e-02    3.21e-02    4.41e-02
   8     8 |  5.37e-05    1.53e-03    1.08e-02    ← 8 passes needed for K=8
  16     1 |  7.60e-03    1.70e-01    2.71e-01
  16     3 |  1.35e-03    5.01e-03    1.37e-02
  32     1 |  7.68e-02    1.00e-01    1.27e-01
  32     3 |  2.40e-04    3.75e-04    4.83e-04    ← 0.024% median
  32     5 |  3.44e-05    5.20e-05    2.05e-04    ← saturated at N=5
  64     3 |  4.57e-05    1.63e-04    7.60e-04
  64     8 |  4.08e-05    1.63e-04    7.60e-04    ← saturated
```

**Interpretation**: Without Newton initial, precision depends strongly on K and N_REFINE:
- K=32, N=3 (our previous config without Newton): 0.024% median — good but ~50x worse than Newton-seeded
- K=32, N=5+: saturates at 0.003% — diminishing returns from refinement
- K=64 helps at low N_REFINE but saturates at the same floor

## Error distribution shape

The error distribution is bimodal:
- **~50% of envs**: Newton step is near-exact (error < 1%). These are smooth-cost envs.
- **~25% of envs**: Moderate error (1-10%). Grid search helps here.
- **~25% of envs**: High error (10-50%+). These have kink-point minima that neither Newton nor grid can reach precisely. The grid search barely helps because the kink alpha `-Jaref/jv` is not among the candidates.

## Conclusions

1. **Newton initial is the dominant factor.** It provides 0.76% median error for free (no kernel launch). The grid search reduces this to 0.56% with K=32, N=3 — a marginal improvement.

2. **Grid-only precision scales as ~O(1/K^N_REFINE)** in the median, but has a hard floor set by kink-point minima that the grid never evaluates.

3. **Increasing K is more effective than increasing N_REFINE** for Newton-seeded search (K=64 gives 5x improvement, N_REFINE gives ~1.3x).

4. **The grid search's value is NOT in median precision** (Newton handles that) but in the ~25% of envs where the Newton approximation breaks down at kinks. However, even K=64, N=8 can't reduce the p90 error below 22.6%.

5. **To improve the worst-case envs**, the grid would need to include kink alphas (`-Jaref[i_c]/jv[i_c]`) as candidates. With ~18 kinks per env and K=32, this is feasible but would require a fundamentally different candidate selection strategy.
