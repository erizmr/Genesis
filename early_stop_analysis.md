# Early-Stop Analysis: Parallel Linesearch Convergence

## Summary

There are **two separate issues** causing the parallel linesearch to have slower
overall solver convergence than the sequential linesearch on `main`:

1. **Bug (fixed):** `_kernel_parallel_linesearch_p0` unconditionally reset `improved[i_b] = True`,
   resurrecting converged environments. Fixed by adding `improved` guards and removing the reset.

2. **Algorithmic (remaining):** The parallel linesearch finds **suboptimal step sizes**
   compared to the sequential Newton-based linesearch, causing ~30% of environments to
   not converge within 10 iterations (vs 0% on main). This is inherent to the grid-search
   approach and cannot be fixed by the `improved` flag alone.

---

## Issue 1: `improved` Flag Bug (FIXED)

**File:** `genesis/engine/solvers/rigid/constraint/solver_breakdown.py`

The `_kernel_parallel_linesearch_p0` set `improved[i_b] = True` every iteration for envs
with `snorm >= EPS`, undoing convergence decisions from `func_terminate_or_update_descent_batch`.

**Fix applied:** Added `improved[i_b]` guards to `_mv`, `_jv`, `_p0` kernels and removed
the `improved = True` reset line.

---

## Issue 2: Suboptimal Alpha Selection (REMAINING)

### Evidence

At step 276 with 4096 envs + random forces (matching profiler capture conditions):

| Branch | Envs with constraints | Converged in 10 iters | Rate |
|--------|----------------------|----------------------|------|
| **main** (sequential) | 3909 | 3909 | **100%** |
| **opt (fixed)** (parallel) | 3885 | 2706 | **69.7%** |

### Root Cause: Grid Search vs Newton-Based Linesearch

The **sequential linesearch** (main) uses:
- Gradient (derivative) information to do Newton steps within the linesearch
- A sophisticated 3-phase algorithm: init → bracketing → refinement with 3-alpha batched evaluation
- Up to `ls_iterations=20` inner iterations with derivative-guided convergence
- Convergence criterion: `|derivative| < gtol` where `gtol = tolerance * ls_tolerance * snorm * scale`

The **parallel linesearch** (opt) uses:
- 16 log-spaced candidates over `[alpha_newton * 1e-2, alpha_newton * 1e2]`
- Cost-only evaluation (no gradient/derivative information)
- 1 refinement pass (narrowing around best candidate)
- Acceptance criterion: `best_cost < p0_cost`

### Per-Iteration Alpha Behavior (non-converging env)

Looking at Env 0 (non-converging) across 10 solver iterations:

```
iter 0: alpha=0.736  improvement=3.977e+06    (large step, good progress)
iter 1: alpha=0.736  improvement=3.277e+05
iter 2: alpha=0.736  improvement=2.689e+04
iter 3: alpha=0.736  improvement=2.733e+03
iter 4: alpha=0.736  improvement=1.912e+02
iter 5: alpha=0.736  improvement=1.337e+01
iter 6: alpha=0.736  improvement=9.375e-01
iter 7: alpha=1.359  improvement=5.469e-02
iter 8: alpha=1.359  improvement=7.813e-03
iter 9: alpha=0.398  improvement=3.906e-03    ← still improving, doesn't converge
```

The convergence tolerance is:
```
tol_scaled = meaninertia * max(1, n_dofs) * tolerance
           = 3.34 * 35 * 1e-5
           = 0.00117
```

At iter 7-9, improvement (0.055, 0.008, 0.004) is still above `tol_scaled` (0.00117),
so the env stays active. The parallel linesearch keeps finding small improvements
that prevent convergence but are less effective than the sequential linesearch would find.

The sequential linesearch, using derivative information, would find the near-exact
minimum of the cost function along the search direction, allowing the solver to
converge much faster.

### Why the Parallel Linesearch is Imprecise

1. **No derivative information:** The 16 log-spaced candidates can only bracket the minimum,
   not locate it precisely. The sequential linesearch uses `derivative/hessian` Newton steps.

2. **Fixed search range:** `[alpha_newton * 1e-2, alpha_newton * 1e2]` is a 4-order-of-magnitude
   range with only 16 log-spaced points. This gives ~0.27 decades between adjacent candidates.
   The optimal alpha could fall between two candidates.

3. **Single refinement pass:** `LS_PARALLEL_N_REFINE=1` means only one narrowing pass.
   More passes would improve precision at the cost of more kernel launches.

### Convergence Histogram (4096 envs, step 276)

```
iter  0:   442 envs converged  (those with no contacts / already at minimum)
iter  1:     0 envs converged
iter  2:     0 envs converged
iter  3:     0 envs converged
iter  4:     1 envs converged
iter  5:   302 envs converged
iter  6:   575 envs converged
iter  7:   729 envs converged
iter  8:  1045 envs converged
iter  9:   632 envs converged
iter 10:   370 envs still active  ← would need more iterations
```

---

## Performance Impact

### Trace comparison: main vs fixed opt

| Metric | main | opt (fixed) | delta |
|--------|------|-------------|-------|
| **FPS** | 218,592 | 451,457 | **+106.5%** |
| **GPU kernel time** | 10,832 us | 8,539 us | **−21.2%** |

The fixed opt branch is already 2x faster despite the convergence issue.

### Per-iteration kernel timings (fixed opt)

| iter | linesearch (us) | hessian (us) | gradient (us) |
|------|----------------|-------------|---------------|
| 0 | 74.0 | 122.2 | 65.3 |
| 7 | 73.2 | 121.7 | 65.3 |
| 8 | 72.7 | 118.7 | 64.7 |
| 9 | 69.4 | 92.4 | 54.3 |

Some convergence visible at iter 8-9, but not as aggressive as main (which drops to ~5 us).

---

## Potential Further Improvements

1. **Increase `LS_PARALLEL_N_REFINE`** from 1 to 2-3: each pass narrows the search range
   around the best candidate, improving alpha precision. Cost: 1 extra `_eval` kernel per
   refinement pass (~7 us each).

2. **Tighten search range:** Instead of `[alpha * 1e-2, alpha * 1e2]`, use a narrower range
   like `[alpha * 0.1, alpha * 10]` to concentrate candidates near the Newton estimate.

3. **Increase `LS_PARALLEL_K`** from 16 to 32: doubles the candidate density. Cost: doubles
   the `_eval` kernel work per pass.

4. **Hybrid approach:** Use parallel linesearch for early iterations (where large steps dominate
   and precision matters less), switch to sequential for later iterations near convergence.

5. **Add derivative approximation:** Evaluate cost at `alpha ± epsilon` for the best candidate
   to estimate the derivative, then do a Newton correction step.
