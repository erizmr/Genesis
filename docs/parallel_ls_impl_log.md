# Parallel Linesearch Enhancement — Implementation Log

## Design Review Summary (2026-03-24)

See `docs/parallel_ls_design_review.md` for full review. Key findings:

1. **Use analytical gradients** — `func_ls_point_fn_opt` computes `grad = 2*alpha*quad_total_2 + quad_total_1` at zero extra cost. FD approach (2 cost evals per gradient) is unnecessary.
2. **Branch A logic flaw** — when grid finds no improvement but `grad(0)<0`, optimum is likely in `[0, lo]` (below grid range), not beyond `hi`.
3. **Missing cost-improvement guard** — bisection/expansion results must check `cost < p0_cost` before acceptance.
4. **Bisection preconditions not always guaranteed** — entry points must verify `grad(a)<0, grad(b)>0`.
5. **Simplify** — recommended 2-phase approach: (1) K=32 grid search, (2) analytical-gradient bisection from grid-derived bracket.

## Implementation Plan

Simplified approach based on review recommendations:

### Phase 0 — p0 kernel (unchanged)
Compute mv, jv, quad_gauss, eq_sum, snorm, p0_cost, search range. No changes needed.

### Phase 1 — Grid search (eval kernel, K threads)
Same K=32 grid search with Newton alpha on thread 0. Argmin reduction → `best_alpha`, `best_cost`.

### Phase 2 — Analytical gradient bisection (eval kernel, thread 0)
After grid search, thread 0 runs gradient-guided refinement:

1. Compute analytical `grad(best_alpha)` and `cost(best_alpha)` using accumulated quad coefficients.
2. If `|grad| < gtol` → accept `best_alpha` (already converged).
3. If `grad > gtol` (overshot, optimum is to the left):
   - Bisect `[left_neighbor, best_alpha]` using analytical gradients.
4. If `grad < -gtol` (undershot, optimum is to the right):
   - Bisect `[best_alpha, right_neighbor]` using analytical gradients.
5. **Universal cost guard**: always verify `cost(result) < p0_cost` before accepting.

### Constants
- `LS_BISECT_STEPS = 12` — maximum bisection iterations
- `LS_ALPHA_MAX = 1e4` — hard upper bound on step size

---

## Change Log

### Change 1: Baseline (current state)
- Grid search only, no gradient refinement
- 262 passed, 7 skipped, 1 xfailed (test_mesh_primitive_COM fails on main too)
