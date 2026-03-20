# Design Review: Parallel Linesearch Fused Kernel (ls_plan_v2)

## Design Summary

The proposal fuses all 6 parallel linesearch sub-kernels (mv, jv, p0, eval x N_REFINE, apply_dofs, apply_constraints) into a single kernel launch per solver iteration. This targets ~140 us of kernel launch overhead (70 extra launches x 2 us each) which, combined with potential convergence improvements from cheap N_REFINE increase, could close the 2.1% FPS gap vs main's sequential linesearch.

## Existing Patterns Reviewed

- `solver_breakdown.py` (lines 30-79): `mv` and `jv` use `qd.ndrange(n_dofs, _B)` / `qd.ndrange(len_constraints, _B)` — one thread per (dof, env) or (constraint, env) pair. No block structure, no shared memory.
- `solver_breakdown.py` (lines 82-253): `p0` uses `block_dim=_T` (32), one block per env, K threads stride over dofs/constraints with shared-memory tree reductions. 6 shared arrays of size 32.
- `solver_breakdown.py` (lines 255-376): `eval` uses `block_dim=_K` (32), one block per env. 2 shared arrays (cost + idx) of size 32. Each thread evaluates one candidate alpha, then argmin reduction.
- `solver_breakdown.py` (lines 378-413): `apply_alpha_dofs` / `apply_alpha_constraints` use `qd.ndrange` — one thread per (dof, env) or (constraint, env), same pattern as mv/jv.
- `solver.py` (line 2300): Main's `func_linesearch_and_apply_alpha` is a single `@qd.func` called from a monolithic kernel — already fused by design, but serial per env (one thread does entire linesearch).

## Elegance Assessment

- Score: Acceptable

The design identifies the right problem (launch overhead) and proposes a reasonable solution (fusion). However, it papers over a fundamental mismatch: `mv` and `jv` are naturally parallel over `n_dofs x B` and `n_constraints x B` respectively, while `p0`/`eval` are parallel over `K x B`. Forcing mv/jv into K=32 threads striding over dofs/constraints makes them serial within each env. This is a step backward from the current design where mv/jv exploit full dof-level parallelism across all envs simultaneously. The current separate-kernel design has a clean separation: "embarrassingly parallel over work items" (mv, jv, apply) vs "block-cooperative reductions per env" (p0, eval). The fused kernel collapses this distinction.

## Minimalism Assessment

- Score: Some bloat

The proposal fuses 6 kernels with 3 distinct parallelism patterns into one. This creates a monolithic kernel that does matrix-vector products, reductions, candidate evaluation with for-loop refinement, and state updates — all in one function body. The register pressure concern acknowledged in the Risk section is real but understated. With all local variables from mv (entity lookup, dot product accumulators), jv (sparse/dense branching, dot product), p0 (6 shared arrays, 6 local accumulators, constraint type branching), eval (log-space alpha computation, per-constraint cost loop, 2 shared arrays), and apply (alpha read, dof/constraint updates) — this is a large kernel. On the other hand, the approach IS minimal in the sense that it's a single, well-defined transformation (fuse launches) to address a single, measured bottleneck (launch overhead).

## Hackiness Assessment

- Score: Minor concerns

The design follows the existing block-cooperative pattern established by `p0` and `eval`. The for-loop refinement inside the kernel (step 4) with block syncs between passes is a known-safe pattern — Quadrants translates `qd.simt.block.sync()` to `__syncthreads()`, and a for-loop with syncs inside is standard CUDA. No divergent sync issues since all threads in the block execute the same loop iteration count (N_REFINE is a compile-time constant).

However, the `batch_dofs_info` template branch in `mv` (line 50: `[i_d1, i_b] if qd.static(...)`) is currently a compile-time specialization. Inside the fused kernel, this would need to work with the strided-thread pattern, which may interact poorly with the Quadrants JIT if the indexing pattern changes from `qd.ndrange` to manual thread-stride loops.

## Specific Issues

### 1. mv/jv performance regression is likely

Currently, `mv` launches `n_dofs * B` threads — on the benchmark (4096 envs, say ~50 dofs), that's ~200K threads, saturating the GPU. The fused kernel launches `B * K = 4096 * 32 = 131K` threads, where each block of 32 threads serially strides over ~50 dofs. For `jv`, the regression is worse: `n_constraints` can be much larger than `n_dofs`, so 32 threads striding over potentially hundreds of constraints per env is significant serialization. The plan claims the kernel is "memory bandwidth limited" so occupancy doesn't matter, but mv/jv are compute-bound (dot products), not bandwidth-bound.

### 2. Shared memory accounting is tight but feasible

The fused kernel needs shared memory from both p0 (6 arrays x 32 x 8 bytes = 1536B for float64) and eval (2 arrays x 32 x 8 bytes = 512B for float64). If reused across phases (which the plan implies), peak is 1536B for the p0 phase. This is well under the 48KB default shared memory limit. Not a concern.

### 3. The 140 us savings estimate may be optimistic

The plan estimates 70 launches x 2 us = 140 us. But not all launch overhead is eliminated — the fused kernel still has one launch per iteration, and the remaining non-LS kernels (constraint_forces, qfrc, cost, hessian, gradient, search_direction) still launch separately. The 2 us per launch figure includes GPU idle time between kernels, which partially overlaps with Python dispatch of the next kernel. The real saving may be closer to 70-100 us.

### 4. N_REFINE > 3 inside the fused kernel is the real win

The experiment log shows adaptive N_REFINE=5 failed because of extra launch overhead (+40 us) and JIT template bloat. Inside a fused kernel, increasing N_REFINE from 3 to 6 costs essentially nothing — it's just more iterations of a for-loop. This is the most compelling argument for the fused approach. If higher N_REFINE can close the convergence gap (8052 vs 7298 active-env-iters), the payoff could be 200+ us — larger than the launch savings.

### 5. The `sparse_solve` branching in jv needs careful handling

The jv kernel has a `qd.static(static_rigid_sim_config.sparse_solve)` branch that uses indirect indexing (`jac_relevant_dofs`). In a strided-thread fusion, threads within the same block would need to handle variable-length sparse rows, potentially causing warp divergence if some constraints have many relevant dofs and others have few.

## Recommendations

### A. Consider partial fusion: fuse only p0 + eval + apply (same parallelism pattern)

Instead of fusing all 6 kernels, fuse only the 3 that already share the block_dim=K pattern: `p0`, `eval` (x N_REFINE), and `apply`. This eliminates 5 launches per iteration (p0 + 3 eval + 1 combined apply = 5, down from 6) while keeping mv/jv as separate high-parallelism kernels. Savings: ~50 launches x 2 us = ~100 us. This avoids the mv/jv serialization risk entirely and is a much simpler kernel to write and debug.

For the apply step in this partial fusion: after the final eval pass picks the best alpha, the same K=32 threads can stride over dofs and constraints to apply alpha. This is slightly serialized vs the current n_dofs*B parallelism, but apply is trivially cheap (one multiply-add per dof) so the serialization cost is negligible.

### B. If full fusion, benchmark mv/jv serialization in isolation first

Before writing the full fused kernel, write a standalone test that runs mv and jv with block_dim=32 (K threads striding over dofs/constraints) and compare wall time against the current ndrange versions. If the strided versions are more than ~30% slower, the full fusion will not pay for itself.

### C. Prioritize the N_REFINE experiment

The largest potential win is increasing N_REFINE inside the kernel. Even the partial fusion (Recommendation A) enables this. Try N_REFINE=4, 5, 6 with the partial-fused kernel and measure active-env-iters convergence.

### D. Consider fusing apply_dofs + apply_constraints as a trivial first step

These two kernels are identical in structure (read alpha, multiply-add) and always launch back-to-back. Fusing them into a single kernel that strides over max(n_dofs, n_constraints) saves 1 launch per iteration (10 over 10 iters = ~20 us) with zero risk. This is a 5-minute change that validates the launch-overhead hypothesis.

## Questions for the Author

1. What is the typical `n_dofs` and `n_constraints` for the g1_fall benchmark? This determines whether the mv/jv serialization from K=32 threads is acceptable. If n_dofs ~ 6 (single rigid body), striding is free. If n_dofs ~ 100+, it's a real concern.

2. Has the launch overhead been measured directly (e.g., by replacing kernel bodies with no-ops)? The 2 us/launch figure is cited but it would be useful to confirm that eliminating launches actually recovers the expected time, since Python-side dispatch and GPU scheduling overlap can mask launch costs.

3. The plan mentions the previous fused eval attempt "broke convergence because of a communication bug between thread 0 and other threads." Was the root cause identified? If it was a shared-memory race condition, the same pattern exists in the proposed fused kernel (thread 0 writes range narrowing in eval, all threads read it on the next iteration of the refine loop — this requires a sync between the thread-0 write and the all-thread read at the top of the next loop iteration).

4. The `batch_dofs_info` conditional indexing in mv (`[i_d1, i_b] if qd.static(...)`) — does Quadrants handle this correctly when the loop structure changes from `qd.ndrange` to a manual thread-stride loop?
