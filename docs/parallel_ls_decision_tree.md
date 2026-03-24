# Hybrid Linesearch Decision Tree

## Definitions

```
gtol  = tolerance × ls_tolerance × snorm × scale
snorm = ||search||
scale = meaninertia × max(1, n_dofs)
```

---

## Phase 0 — Shared Initialization *(always runs)*

Compute `mv`, `jv`, `eq_sum`, `quad_gauss`, `snorm`, and `p0_cost` exactly as in the
existing parallel linesearch `_kernel_parallel_linesearch_p0`. These quantities are
reused by all subsequent phases without recomputation.

---

## Phase 1 — Parallel Pass 1 *(always runs)*

Evaluate `K` candidates log-spaced in `[lo, hi]`. Record `best_p1` (lowest-cost
candidate) and `hi` (right boundary of the initial range).

Branch on `cost(best_p1) < p0_cost`:

---

### Branch A — No improvement found (`cost(best_p1) >= p0_cost`)

1. Evaluate `grad(0)` — **2 cost evals**.

2. If `grad(0) >= -gtol`:
   - `α = 0` is locally optimal; any positive step increases cost.
   - **→ Return `alpha = 0`. Done.**

3. If `grad(0) < -gtol`:
   - Cost is still decreasing at `α = 0`, so `α*` lies to the right of the entire
     search range.
   - **→ Proceed to [Phase 4 — Range Expansion](#phase-4--range-expansion), `a_start = hi`.**

---

### Branch B — Improvement found (`cost(best_p1) < p0_cost`)

1. Evaluate `grad(best_p1)` — **2 cost evals**.

2. If `|grad(best_p1)| < gtol`:
   - First-order optimality satisfied at `best_p1`.
   - **→ Return `alpha = best_p1`. Done.**

3. If `grad(best_p1) < -gtol`:
   - Cost is still decreasing at `best_p1`; `α*` lies to its right.
   - Evaluate `grad(hi)` — **2 cost evals**.
   - If `grad(hi) > 0`:
     - `α*` is bracketed within `[best_p1, hi]`.
     - **→ Proceed to [Phase 3b — Bisection](#phase-3b--bisection) on `[best_p1, hi]`.**
   - If `grad(hi) <= 0`:
     - Cost is still decreasing at the right boundary; `α*` lies beyond the current range.
     - **→ Proceed to [Phase 4 — Range Expansion](#phase-4--range-expansion), `a_start = hi`.**

4. If `grad(best_p1) > gtol`:
   - `best_p1` has overshot `α*`; the range is correct but pass 1 resolution is too coarse.
   - **→ Proceed to [Phase 2 — Parallel Passes 2 and 3](#phase-2--parallel-passes-2-and-3).**

---

## Phase 2 — Parallel Passes 2 and 3 *(conditional)*

> Reached only when `grad(best_p1) > gtol`: range is correct, resolution insufficient.

Run passes 2 and 3 as in the existing parallel linesearch, narrowing the search range
around `best_p1` each time.

After pass 3, evaluate `grad(best_final)` — **2 cost evals**.

1. If `|grad(best_final)| < gtol`:
   - **→ Return `alpha = best_final`. Done.**

2. If `grad(best_final) > gtol`:
   - Still overshot after refinement; attempt a Newton correction.
   - **→ Proceed to [Phase 3 — Newton Correction](#phase-3--newton-correction).**

---

## Phase 3 — Newton Correction *(conditional)*

> Reached only from Phase 2 when `grad(best_final) > gtol`.

1. Evaluate `hess(best_final)` — **2 cost evals**.

2. If `hess(best_final) <= 0`:
   - Local quadratic approximation is invalid (non-convex segment).
   - **→ Proceed to [Phase 3b — Bisection](#phase-3b--bisection) on `[lo_final, best_final]`.**

3. Compute Newton step:
   ```
   alpha_c = best_final - grad(best_final) / hess(best_final)
   alpha_c = clamp(alpha_c, 0, AMAX)
   ```

4. Evaluate `cost(alpha_c)` — **1 cost eval**.

5. If `cost(alpha_c) >= cost(best_final)`:
   - Newton step crossed a fold point, making cost worse.
   - **→ Proceed to [Phase 3b — Bisection](#phase-3b--bisection) on `[alpha_c, best_final]`.**

6. Evaluate `grad(alpha_c)` — **2 cost evals**.

7. If `|grad(alpha_c)| < gtol`:
   - Newton correction converged precisely.
   - **→ Return `alpha = alpha_c`. Done.**

8. If `|grad(alpha_c)| >= gtol`:
   - Newton correction improved but did not fully converge.
   - **→ Proceed to [Phase 3b — Bisection](#phase-3b--bisection) on `[alpha_c, hi_c]`**,
     where `hi_c` is the neighboring boundary around `alpha_c`.

---

## Phase 3b — Bisection *(conditional)*

> Receives a valid bracket `[a, b]` where `grad(a) < 0` and `grad(b) > 0`.
> Maximum iterations: `LS_BISECT_STEPS` (recommended: 12).

Repeat until stopping condition:

1. `mid = (a + b) / 2`
2. Evaluate `grad(mid)` — **2 cost evals**.
3. If `|grad(mid)| < gtol` or `|b - a| < EPS`:
   - **→ Return `alpha = mid`. Done.**
4. If `grad(mid) < 0`: set `a = mid`.
5. If `grad(mid) > 0`: set `b = mid`.

---

## Phase 4 — Range Expansion *(conditional)*

> Receives `a_start` where `grad(a_start) < 0`. Finds a bracket `[a, b]` with
> `grad(a) < 0` and `grad(b) > 0` by exponential stepping.
> Maximum steps: `LS_EXPANSION_STEPS` (recommended: 8), factor: `LS_EXPANSION_FACTOR = 4.0`.

1. Set `a = a_start`, `b = a_start`.

2. Repeat up to `LS_EXPANSION_STEPS` times:
   - `b = min(b × LS_EXPANSION_FACTOR, AMAX)`
   - Evaluate `grad(b)` — **2 cost evals**.
   - If `grad(b) > -gtol`:
     - Valid bracket `[a, b]` found.
     - **→ Proceed to [Phase 3b — Bisection](#phase-3b--bisection) on `[a, b]`.**
   - Else: set `a = b` and continue.

3. If `b >= AMAX` with no bracket found:
   - Cost is monotonically decreasing across the entire valid range.
   - Evaluate `cost(AMAX)` — **1 cost eval**.
   - If `cost(AMAX) < p0_cost`:
     - **→ Return `alpha = AMAX`. Done.**
   - Else:
     - **→ Return `alpha = a_start`. Done.**

---

## Eval Cost Summary

| Path | Evals | Condition |
|---|---|---|
| Branch A → skip | 2 | `grad(0) >= -gtol` |
| Branch A → expand → bisect | 2 + 16 + 24 = 42 | `grad(0) < -gtol` |
| Branch B → skip (pass 1) | 2 | pass 1 already converged |
| Branch B → bisect `[best, hi]` | 2 + 2 + 24 = 28 | `grad(best) < -gtol`, bracket valid |
| Branch B → expand → bisect | 2 + 2 + 16 + 24 = 44 | `grad(best) < -gtol`, bracket invalid |
| Branch B → pass 2,3 → skip | 2 + 64 + 2 = 68 | pass 3 converged |
| Branch B → pass 2,3 → newton → skip | 68 + 7 = 75 | newton converged |
| Branch B → pass 2,3 → newton → bisect | 68 + 7 + 24 = 99 | newton failed |

> **Baseline**: original fixed three-pass cost = `K × 3 = 96` evals (K = 32).
> The hybrid strategy matches or improves upon this in most paths while correctly
> handling cases where the original parallel linesearch returns a suboptimal result.


Note:

Commit and push after each important change to prevent code lost

## Test correctness

After each important features you added, run `pytest -xsv --backend gpu "tests/test_rigid_physics.py"` to ensure every test case are passed.


## Test perf

You can use the following command to test the performance, in which <version_name> and <branch_name> is something you need to fill according to current implementing feature 
`python /home/mingrui/workspace/perso_hugh/ai/scripts/bench_cluster_wandb.py --ref 260323_<version_name> --branch <branch_name> --solver auto`

You can use `python /home/mingrui/workspace/perso_hugh/ai/scripts/bench_cluster_wandb.py --ref 260323_<version_name> --branch <branch_name> --solver decomposed` to enforce the perf dispatch to choose the `decomposed` solver

Note you need to push your lastest commit so that it can be effective in this perf test.

The command will trigger a cluster run, and the results are stored in `/home/mingrui/workspace/tmp/260323_<version_name>/results.csv`, which show the perf comaprison of curent branch to main. In the results.csv, pay attention to `g1_fall` and `box_pyramid_6`, i.e., branch should expect better performance on these two cases comparing to main.