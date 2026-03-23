## Design Summary

The design proposes three improvements to the parallel grid-search linesearch in the rigid body constraint solver: (1A) parabolic interpolation after the grid argmin to refine alpha precision at near-zero cost, (1B) adaptive search range narrowing based on convergence progress, and (Goal 2) adaptive refinement pass count driven by a bracket-cost-difference criterion. The aim is to close the 30% convergence gap relative to the sequential Newton-based linesearch while preserving the 2.3x throughput advantage.

## Existing Patterns Reviewed

- `solver_breakdown.py` `_kernel_parallel_linesearch_eval`: Uses shared-memory argmin reduction with `_K` threads per env (one CUDA block per env). Thread 0 performs post-reduction bookkeeping (acceptance check, range narrowing). Only thread 0 writes to global state after reduction. This is the natural insertion point for parabolic interpolation.
- `solver_breakdown.py` `_kernel_parallel_linesearch_p0`: Uses a two-phase shared-memory reduction pattern (Phase 1 over dofs, Phase 2 over constraints) with `_P0_BLOCK=32` threads. Demonstrates the codebase convention for fused multi-accumulator reductions within a single kernel.
- `solver_breakdown.py` `func_solve_decomposed`: The Python-level dispatch loop launches kernels sequentially. The refinement loop is a simple `for _refine in range(LS_PARALLEL_N_REFINE)` with no inter-pass communication beyond global memory. This is the site for adaptive loop termination.
- `solver.py` `func_linesearch_batch`: The sequential linesearch uses derivative information (gradient + Hessian of the 1D cost) to perform Newton steps, with a 3-phase algorithm (init, bracketing, refinement). This is the precision target the parallel version must approximate.
- `array_class.py`: `candidates` is a `(12, _B)` float array used as a per-env scratchpad. Slots 0-4 are currently used (alpha, p0_cost, lo, hi, best_cost_prev). Slots 5-11 are available for new fields like `needs_refine` without schema changes.
- `solver_breakdown.py` registration: The decomposed solver registers via `func_solve_body.register(is_compatible=...)`, gating on `gs.backend in {gs.cuda}` and `not requires_grad`. This pattern cleanly separates the parallel path from the monolithic fallback.

## Elegance Assessment

- Score: green

The parabolic interpolation (1A) is particularly elegant: it reuses the three cost values already in shared memory, adds roughly 10 FLOPs in thread 0's existing post-reduction block, requires zero new kernel launches, and mirrors what the sequential linesearch achieves with derivative-based Newton steps but without needing derivative computation. The mathematical formulation is standard (three-point parabolic fit) and the safety guards (non-degeneracy, interpolation-not-extrapolation, upward-opening) are the textbook checks.

The adaptive refine loop (Goal 2) follows the established pattern: the Python loop launches the same kernel repeatedly, and per-env early-exit is handled by a flag check at the top of each thread block, consistent with the existing `improved[i_b]` guard pattern used throughout every kernel in `solver_breakdown.py`.

## Minimalism Assessment

- Score: yellow

1A (parabolic interpolation) is minimal and well-scoped. However, proposing all three improvements (1A, 1B, Goal 2) simultaneously before measuring the impact of 1A alone introduces unnecessary design surface area. The document acknowledges this via the phased implementation order, but the adaptive search range (1B) is underspecified: the `improvement_ratio` heuristic (`prev_cost / cost`) conflates per-iteration improvement with proximity to convergence, and the three-tier threshold values (100, 10, else) appear arbitrary without justification from the data.

Goal 2 introduces two new tunable parameters (`LS_PARALLEL_MAX_REFINE`, `LS_PARALLEL_PRECISION_FRACTION`) replacing one (`LS_PARALLEL_N_REFINE`). The precision criterion based on bracket cost difference relative to total improvement is reasonable, but the `precision_fraction = 0.01` default needs empirical validation. The `MAX_REFINE = 6` hard cap is 2x the current value, which is fine as a safety bound, but the document's cost model estimates only ~1.5 average passes with parabolic interpolation -- if that holds, the adaptive loop machinery is doing very little work for its complexity.

## Hackiness Assessment

- Score: green

The design follows established patterns well:
- Thread 0 post-reduction logic in the eval kernel is exactly where the parabolic fit belongs.
- The `needs_refine` flag mirrors the existing `improved` flag pattern -- per-env booleans checked at thread-block top, avoiding cross-env synchronization.
- No new kernel signatures are introduced for 1A.
- No Python-GPU synchronization points are added (the adaptive loop still launches fixed kernel calls; it does not read back `needs_refine` to Python).
- The `candidates` array has spare slots, so no schema changes are needed to store `needs_refine` or intermediate values.

One minor concern: storing `needs_refine` in the `candidates` array (a float buffer) to hold a boolean is a pattern already used in the codebase (e.g., `candidates[4]` stores `best_cost_prev` as a sentinel value), so it is consistent, but adding a dedicated `qd.i32` field to `ConstraintState` would be cleaner if the array is not capacity-constrained.

## Specific Issues

1. **Parabolic interpolation boundary handling at `best_idx == 0` or `best_idx == K-1`**: The document references `alpha[best_idx-1]` and `alpha[best_idx+1]` but does not address what happens when the best candidate is at the boundary of the grid. In `_kernel_parallel_linesearch_eval`, the range-narrowing already clamps `lo_idx = max(0, best_tid - 1)` and `hi_idx = min(K-1, best_tid + 1)`, but if `best_tid == 0`, then `a_l == a_b` and the parabolic denominator becomes zero. The safety guard (`den != 0`) catches this, but it means boundary winners silently skip refinement. This is fine but should be documented, and it may be worth noting that boundary-hitting often indicates the optimal alpha lies outside the search range entirely.

2. **The parabolic safety guard `den > 0` for upward-opening is incorrect as written.** The standard three-point parabolic minimum formula has the denominator `2 * [(a_b - a_l)(c_b - c_r) - (a_b - a_r)(c_b - c_l)]`. For a minimum (upward-opening parabola), the condition is that this denominator is positive, but the document writes the check as `den > 0` where `den` is the un-doubled denominator. This is equivalent, but the narrative says "den > 0 for a minimum" which is only correct if `den` has a consistent sign convention. Recommend explicitly checking the second derivative of the fitted parabola or simply verifying `alpha_refined` produces a lower cost than `alpha_best` (which you can do for free since the cost at `alpha_best` is already known).

3. **Adaptive range (1B) uses `prev_cost / cost` but `prev_cost` is the cost from the previous solver iteration, not the previous linesearch pass.** At solver iteration 0, `prev_cost` may be uninitialized or zero. The document should specify initialization behavior and guard against division by zero or meaningless ratios at iteration 0.

4. **The precision threshold derivation (Goal 2) has a gap.** The document derives grid precision in terms of `relative_precision = interval_width / |alpha_best|` but then switches to a completely different criterion (`bracket_cost_diff / total_improvement`) without connecting the two. The cost-based criterion is more practical, but the precision derivation section is then misleading -- it creates the impression that the threshold is analytically derived when it is actually an empirical heuristic.

5. **The adaptive refine loop does not implement a global early-exit reduction.** The document notes this explicitly ("skip global check -- individual envs skip via needs_refine guard") and this is the right call for now. However, if most envs converge in 1-2 passes, the remaining 4 kernel launches at ~7 us each (28 us) still add up. A single-word atomic-or reduction in the eval kernel could enable the Python loop to break early via a device-to-host copy of one integer. This is worth considering as a follow-up but is correctly deferred.

6. **Cost values in shared memory are overwritten during the argmin reduction.** The tree reduction `sh_cost[tid] = sh_cost[tid + stride]` destroys the original per-thread costs. To read `cost[best_idx-1]` and `cost[best_idx+1]` after reduction, the parabolic interpolation would need the original unreduced costs. This requires either: (a) a second shared array to preserve costs before reduction, (b) re-computing the three costs from the formula (cheap: ~6 FLOPs each since `quad_gauss` and `eq_sum` are in global memory), or (c) storing the three neighbor costs during the reduction. Option (b) is cleanest and consistent with the existing pattern of recomputing alpha from `best_tid`. The design document does not address this, but it is a critical implementation detail.

## Recommendations

1. **Implement 1A first, measure, and only proceed to 1B/Goal 2 if needed.** The parabolic interpolation is the highest-value, lowest-risk change. The analysis data suggests that imprecise alpha is the primary convergence bottleneck, and parabolic interpolation directly addresses it. If 1A alone brings convergence above 95%, the added complexity of 1B and Goal 2 may not be justified.

2. **For the parabolic interpolation, recompute the three neighbor costs rather than trying to preserve them through the reduction.** The cost formula is `alpha^2 * quad_gauss[2] + alpha * quad_gauss[1] + quad_gauss[0] + (equality terms) + (friction loop) + (contact loop)`. For the equality terms, use the precomputed `eq_sum`. For friction and contact, the loop must be re-executed for the three alphas, but this is only done once by thread 0, so the cost is bounded by `3 * n_constraints` FLOPs -- acceptable for typical constraint counts.

3. **Alternatively (and more cheaply), store the three neighbor costs during the reduction.** Before the tree reduction begins, have thread `best_idx-1`, `best_idx`, and `best_idx+1` write their costs to three dedicated shared memory slots. This requires knowing `best_idx` before the reduction, which you don't have. So option (b) -- recomputing three costs -- or having each thread also write its cost to a separate non-reduced shared array (doubling shared memory from K to 2K floats, still small at 256 bytes for K=32 float64) are the practical options.

4. **Add an explicit fallback check for the parabolic result**: after computing `alpha_refined`, verify that the cost at `alpha_refined` (which can be computed with the same formula used for grid candidates) is actually lower than `cost[best_idx]`. This is more robust than the geometric safety guards alone and catches edge cases where the cost function is not well-approximated by a parabola (e.g., near friction/contact activation boundaries).

5. **Consider deferring 1B entirely.** The adaptive search range adds conditional branching in the hot path and introduces coupling between solver iterations (via `prev_cost`). If 1A + increased `N_REFINE` (or adaptive `N_REFINE` from Goal 2) achieves the target convergence rate, 1B is unnecessary complexity.

6. **For Goal 2, consider a simpler formulation: just run until the alpha changes by less than a relative tolerance between passes, or cap at N=4.** The bracket-cost-difference criterion is sound but requires careful tuning of `precision_fraction`. A simpler `|alpha_new - alpha_prev| / |alpha_prev| < rtol` check achieves the same goal with one fewer tunable parameter and a more intuitive stopping condition.

## Questions for the Author

1. Has the parabolic interpolation been prototyped even in a CPU-side NumPy test to verify the expected precision improvement? A quick test with the actual cost values from the non-converging envs (e.g., Env 0 from the early-stop analysis) would validate the approach before GPU implementation.

2. The shared-memory argmin reduction in `_kernel_parallel_linesearch_eval` overwrites `sh_cost` during the tree reduction. How do you plan to access the three neighbor costs for the parabolic fit? This is not addressed in the design and is a non-trivial implementation constraint (see Specific Issue 6).

3. For the friction and contact constraint loops in the eval kernel -- these have data-dependent branching (active/inactive contacts, friction linear regime). Has the impact of this branching on parabolic fit quality been considered? The cost function may have kinks at constraint activation boundaries, making the parabola fit unreliable precisely when it matters most (near convergence, where contacts are on the boundary of activation). The safety guards will catch obvious failures, but subtle curvature mismatches could still produce worse alphas than the grid point.

4. What is the plan for validation beyond the single benchmark (g1_fall, 4096 envs, step 276)? The convergence regression was discovered on this specific scenario; it would be valuable to test on scenes with different constraint profiles (many equality constraints, friction-dominated, few contacts with high DOF counts) to ensure the improvements generalize.

5. The design mentions that non-linesearch kernels (hessian, gradient) are ~20% slower in the parallel version "likely because imprecise alpha leads to solver states that require more work." Is there evidence for this causal claim, or could the slowdown be due to other factors (e.g., different memory access patterns from the decomposed kernel structure, cache effects from the additional kernel launches)? If the latter, improving alpha precision alone may not recover this 20%.
