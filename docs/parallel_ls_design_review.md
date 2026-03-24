# Design Review: Hybrid Linesearch Decision Tree

**Document reviewed:** `/home/mingrui/workspace/gs-core-genesis/Genesis/docs/parallel_ls_decision_tree.md`
**Reviewer date:** 2026-03-24

---

## Design Summary

The design proposes enhancing the current parallel linesearch (which does a K=32 grid search with no gradient refinement) with a multi-phase decision tree. After the initial grid search (Phase 1), the system evaluates finite-difference gradients (2 cost evals each) to decide whether to: (a) accept the result, (b) run additional parallel grid passes for resolution, (c) apply Newton correction, (d) bisect a bracketed interval, or (e) expand the search range. All post-grid-search logic runs in thread 0 of the CUDA block. The goal is to recover the accuracy benefits of gradient-guided refinement that was previously removed due to Newton divergence on piecewise-quadratic cost landscapes.

---

## Existing Patterns Reviewed

- `solver_breakdown.py` (lines 253-389): Current parallel linesearch eval kernel. Uses K=32 threads for grid search, shared-memory argmin reduction, thread-0 acceptance check. Comment at line 384-386 explicitly states gradient refinement was removed to avoid overshooting on piecewise-quadratic landscapes.

- `solver.py` `func_ls_point_fn_opt` (lines 2122-2189): Sequential linesearch cost/gradient/hessian evaluator. Returns `(alpha, cost, grad, hess)` by iterating over all constraints. This is the established pattern for "evaluate cost at a point." The gradient is computed analytically as `2 * alpha * quad_total_2 + quad_total_1`, NOT via finite differences.

- `solver.py` `func_linesearch_batch` (lines 2394-2564): Sequential linesearch. Uses analytical gradient from `func_ls_point_fn_opt`, Newton steps (`alpha - grad/hess`), bracket tracking with `update_bracket_no_eval_local`, and a 3-alpha batched refinement phase. This is the reference implementation that the parallel version aims to match.

- `solver_breakdown.py` `_kernel_parallel_linesearch_eval` (lines 253-389): The cost evaluation kernel already computes cost inline per thread using the same quadratic + piecewise formulas. The gradient could be computed analytically here too (as in `func_ls_point_fn_opt`), making the "2 cost evals" approach unnecessary.

- `solver_breakdown.py` `_kernel_parallel_linesearch_p0` (lines 79-251): Phase 0 already computes and stores `quad_gauss` and `eq_sum` -- the precomputed quadratic coefficients that enable cheap analytical gradient evaluation at any alpha.

---

## Elegance Assessment

- Score: **Needs work**
- The decision tree has 6+ distinct code paths (Branch A skip, Branch A expand, Branch B skip, Branch B bisect, Branch B expand, Branch B pass2-3, Branch B Newton, Branch B Newton-bisect). Each path requires careful state tracking within a single thread-0 block of a CUDA kernel. This level of branching complexity inside a GPU kernel is a maintenance hazard. The sequential linesearch (`func_linesearch_batch`) manages similar logic more cleanly because it can use local variables and function calls freely, but that luxury does not translate well to the thread-0 section of a cooperative kernel.

- The design proposes computing gradients via **2 cost evaluations (finite differences)** when the existing codebase computes gradients **analytically** everywhere. The sequential `func_ls_point_fn_opt` returns analytical `grad = 2 * alpha * quad_total_2 + quad_total_1` at zero extra cost beyond the cost evaluation itself. Since thread 0 already has access to `quad_gauss`, `eq_sum`, and all constraint data (it loops over constraints for cost), computing the analytical gradient requires adding ~5 lines to the existing cost evaluation loop, not 2x the evaluations.

---

## Minimalism Assessment

- Score: **Some bloat**
- The "2 cost evals for gradient" approach doubles the work for every gradient query unnecessarily. Analytical gradients are free once you have the quadratic coefficients, which Phase 0 already computes. This is a significant source of unnecessary complexity and eval count inflation.

- The eval cost table shows worst-case paths reaching 99 evaluations. The baseline 3-pass grid search is 96. Some paths (Branch B -> pass 2,3 -> newton -> bisect) barely improve on the baseline while adding substantial code complexity. The question is whether a simpler 2-phase approach (grid search + analytical-gradient bisection) could achieve the same results.

- Phase 2 (passes 2 and 3) plus Phase 3 (Newton) plus Phase 3b (bisection) is three layers of refinement fallback. The sequential linesearch achieves good results with just Newton steps and bracket refinement -- no grid search at all.

---

## Hackiness Assessment

- Score: **Minor concerns**

- The design is structurally sound and the decision tree logic is correct in isolation. However:

  1. Using finite-difference gradients when analytical gradients are available and already implemented is a workaround rather than using the proper tool.

  2. The Phase 4 range expansion condition `grad(b) > -gtol` (line 149) uses a negative tolerance threshold. This means it triggers when the gradient is merely "not very negative" rather than when it is positive. This is likely intentional (to find a bracket where grad changes sign), but the asymmetry with bisection's entry condition (`grad(a) < 0, grad(b) > 0`) creates a gap: expansion could hand bisection a bracket where `grad(b)` is in `(-gtol, 0]`, which violates bisection's precondition that `grad(b) > 0`.

  3. The `candidates` array is used as a multi-purpose communication channel between phases (indices 0-5 with different meanings). This is an existing pattern in the codebase, but extending it further for more phases increases the risk of index collision bugs.

---

## Specific Issues

### Issue 1: Finite-Difference Gradients Are Unnecessary

The design says "Evaluate grad(best_p1) -- 2 cost evals" throughout. But `func_ls_point_fn_opt` in the sequential path computes analytical gradients as part of cost evaluation. The parallel kernel's cost evaluation loop at lines 296-336 of `solver_breakdown.py` already iterates over all constraints and computes cost as `alpha^2 * quad_total_2 + alpha * quad_total_1 + quad_total_0`. The gradient is simply `2 * alpha * quad_total_2 + quad_total_1`. Thread 0 can compute this with negligible additional work during a single cost evaluation pass. Using finite differences doubles the number of constraint-loop iterations per gradient query and introduces epsilon-dependent numerical error.

### Issue 2: Branch A Logic Flaw -- "grad(0) < -gtol implies alpha* is beyond search range"

Branch A states: if no improvement was found AND `grad(0) < -gtol`, then "alpha* lies to the right of the entire search range." This reasoning is flawed. If `grad(0) < 0` (cost is decreasing at alpha=0), it means there exist positive alphas with lower cost. But the grid search already evaluated K=32 points in `[lo, hi]` and found none better than alpha=0. This means either: (a) the optimal alpha is between grid points (resolution issue), or (b) the cost decreased and then increased, but the decrease was smaller than numerical noise. Jumping directly to range expansion (Phase 4, starting from `hi`) skips the entire `[lo, hi]` range where the optimum likely lies. A bisection within `[0, lo]` or a finer grid in `[lo, hi]` would be more appropriate.

### Issue 3: Phase 3 Newton Correction Safety

The design addresses the previous divergence problem by adding a cost-improvement check (step 5: "If cost(alpha_c) >= cost(best_final), proceed to bisection"). This is a sound safeguard. However, on piecewise-quadratic landscapes, the Hessian computed at `best_final` may not represent the local curvature at the Newton step destination (the step may cross a kink). The cost check catches this, but the subsequent bisection fallback on `[alpha_c, best_final]` may have issues because:
- If Newton overshot left (`alpha_c < best_final`), the bracket is reversed
- The bracket `[alpha_c, best_final]` may not satisfy `grad(a) < 0, grad(b) > 0`

Step 5's fallback bracket assumes `alpha_c < best_final` (since `grad(best_final) > gtol` implies overshoot, Newton steps left), but this is not guaranteed if the Hessian is wrong.

### Issue 4: Phase 3b Bisection Precondition Not Always Met

The bisection requires `grad(a) < 0` and `grad(b) > 0`. Several entry points do not guarantee this:
- Branch B step 3: bracket `[best_p1, hi]` with `grad(best_p1) < -gtol` and `grad(hi) > 0`. This is valid.
- Phase 3 step 2 (`hess <= 0`): bracket `[lo_final, best_final]`. Nothing guarantees `grad(lo_final) < 0`.
- Phase 3 step 5 (Newton cost worse): bracket `[alpha_c, best_final]`. `grad(alpha_c)` is unknown.
- Phase 3 step 8 (Newton did not converge): bracket `[alpha_c, hi_c]`. `hi_c` is undefined ("neighboring boundary around alpha_c").

### Issue 5: Phase 4 Expansion Could Reach Very Large Alphas

With `LS_EXPANSION_FACTOR = 4.0` and `LS_EXPANSION_STEPS = 8`, starting from `hi` (which could be ~10 based on the current Newton-step range), the maximum expansion reaches `hi * 4^8 = hi * 65536`. If `hi = 10`, that is `alpha = 655360`. The subsequent bisection would then search in `[a, 655360]`. Such large step sizes could cause the qacc update (`qacc += search * alpha`) to produce extreme accelerations. The clamping to `AMAX` is mentioned but `AMAX` is not defined in the document.

### Issue 6: Missing Cost-Improvement Guard on Bisection/Expansion Results

The current parallel linesearch (line 372) has an explicit check: `if best_cost < p0_cost and best_cost < best_cost_prev`. The bisection and expansion phases in the proposed design return alpha values without a final cost-improvement check against `p0_cost`. If bisection converges to a gradient-zero point that has higher cost than alpha=0 (possible at a local maximum of a non-convex segment), the linesearch would accept a cost-increasing step.

---

## Recommendations

1. **Use analytical gradients instead of finite differences.** The infrastructure is already in place. Thread 0 can accumulate `quad_total_1` and `quad_total_2` during any cost evaluation loop and compute `grad = 2*alpha*quad_total_2 + quad_total_1` for free. This halves the eval count for every gradient query and eliminates FD epsilon sensitivity. The existing `func_ls_point_fn_opt` is the reference implementation.

2. **Simplify the decision tree.** Consider a 2-phase approach: (Phase 1) K=32 grid search as today, (Phase 2) analytical-gradient bisection starting from a bracket derived from the grid results. The grid search naturally provides bracket candidates (the two grid points flanking the minimum). This eliminates Phases 2 (extra grid passes), 3 (Newton), and 4 (expansion) entirely while still improving on pure grid search.

3. **Add a universal cost-improvement guard.** Every exit path should verify `cost(result) < p0_cost` before accepting, matching the existing pattern at line 372 of `solver_breakdown.py`.

4. **Define AMAX explicitly** and ensure it is consistent with the physics (e.g., related to `meaninertia` or `snorm`). The expansion phase needs a hard upper bound that prevents physically unreasonable step sizes.

5. **Validate bisection preconditions at entry.** Each call site for Phase 3b should explicitly verify `grad(a) < 0` and `grad(b) > 0`, falling back to the current grid-search result if the bracket is invalid.

6. **Re-examine Branch A logic.** When the grid finds no improvement but `grad(0) < 0`, the optimum is likely in `[0, lo]` (below the grid range), not beyond `hi`. Consider bisecting `[0, grid_point_1]` instead of expanding beyond `hi`.

---

## Questions for the Author

1. Why use finite-difference gradients (2 cost evals) when the analytical gradient is available at essentially zero cost? Is there a concern about the analytical gradient being incorrect on piecewise segments? If so, note that the sequential linesearch uses analytical gradients on the same landscapes without issue.

2. What is the definition of `AMAX`? Is it a fixed constant, or derived from the problem state? How was the value chosen to prevent instability?

3. The previous Newton correction was removed because it "caused divergence on piecewise-quadratic cost landscapes." The cost-improvement check in Phase 3 step 5 addresses this, but what about the case where Newton produces a cost improvement but lands on the wrong side of a kink, leading to an incorrect gradient sign and subsequent bisection failure?

4. In Branch A step 3, why is the conclusion "alpha* lies to the right of the entire search range" rather than "alpha* lies in [0, lo], below the grid's lower bound"? The grid range `[lo, hi]` is set from the Newton estimate; if `grad(0) < 0` and no grid point improved, the Newton estimate may have been too large.

5. What is `hi_c` in Phase 3 step 8 ("neighboring boundary around alpha_c")? This needs a precise definition.

6. Has the eval cost analysis accounted for the factor-of-2 savings from switching to analytical gradients? With analytical gradients, bisection becomes 12 evals (not 24), expansion becomes 8 (not 16), and gradient checks become 1 (not 2), substantially changing the cost comparison.

---

## Files Reviewed

- `/home/mingrui/workspace/gs-core-genesis/Genesis/genesis/engine/solvers/rigid/constraint/solver_breakdown.py` -- current parallel linesearch
- `/home/mingrui/workspace/gs-core-genesis/Genesis/genesis/engine/solvers/rigid/constraint/solver.py` -- sequential linesearch reference, `func_ls_point_fn_opt`, `func_linesearch_batch`
- `/home/mingrui/workspace/gs-core-genesis/Genesis/docs/parallel_ls_decision_tree.md` -- the design under review
