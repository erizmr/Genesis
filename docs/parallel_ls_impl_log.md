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

### Change 2: Analytical gradient bisection (commit eaebff93)
- Added `_ls_eval_cost_grad` — analytical cost+gradient at any alpha (follows `func_ls_point_fn_opt` pattern)
- Branch B: after grid accepts best_alpha, compute grad; if |grad| > gtol, bisect within grid neighbors
- Branch A: when grid finds no improvement, check grad(0); if cost decreasing, bisect [0, lo]
- All paths guarded by `cost < p0_cost` check
- Quadrants compiler constraints: no `for...else`, variables must be defined before `if/else` branches

**Correctness**: 262 passed, 7 skipped, 1 xfailed (same as baseline)

**Performance** (benchmark ref: `260323_bisect_v1`):

| Case | Main FPS | Branch FPS | Delta |
|------|----------|------------|-------|
| box_pyramid_6 | 19,111 | 20,959 | **+9.7%** |
| box_pyramid_6 (gjk) | 20,585 | 22,589 | **+9.7%** |
| g1_fall | 438,431 | 264,608 | **-39.6%** |
| dex_hand | 5,767 | 5,426 | -5.9% |

**Finding**: g1_fall regression is **pre-existing** (same -39.8% in previous baseline run before bisection). Root cause: the decomposed path with parallel LS (K=32 grid) is slower than the sequential iterative LS for g1_fall due to higher per-thread work (each of 32 threads loops over all constraints). The `perf_dispatch` selects the decomposed path during warmup when constraints are few (humanoid not yet fallen), then sticks with it for the actual benchmark.

### Change 3: Dual decomposed paths for perf_dispatch (commit b1c74c1c)
- Added `_kernel_linesearch` — sequential iterative LS kernel (same as main branch)
- Added `func_solve_decomposed_sequential` — decomposed solver using sequential LS
- Now perf_dispatch has 3 options: monolith, decomposed+sequential, decomposed+parallel
- perf_dispatch benchmarks all 3 during warmup (6 calls each) and picks the fastest

**Correctness**: 262 passed, 7 skipped, 1 xfailed (same as baseline)

**Performance** (benchmark ref: `260323_bisect_v2_seqls`):

| Case | Main FPS | Branch FPS | Delta |
|------|----------|------------|-------|
| g1_fall | 442,305 | 462,463 | **+4.6%** |
| box_pyramid_6 | 19,172 | 21,133 | **+10.2%** |
| box_pyramid_6 (gjk) | 20,431 | 22,739 | **+11.3%** |
| dex_hand | 5,676 | 5,462 | -3.8% |
| go2 (Newton) | 2,202,572 | 2,335,512 | +6.0% |
| franka_random (gjk) | 8,088,104 | 8,633,415 | +6.7% |

**g1_fall regression fixed!** With the sequential LS variant available, perf_dispatch now correctly selects the sequential LS for g1_fall (many constraints/env) and parallel LS for box_pyramid_6 (few constraints/env). Both key cases now show improvements over main.

Only `dex_hand` shows a small regression (-3.8%), likely because the perf_dispatch warmup picks a suboptimal variant for this specific scene.
