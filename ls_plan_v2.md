# Parallel Linesearch Improvement Plan v2

## Current State

Branch `mingrui/parallel-ls-parabolic-adaptive` at `eb3a421`: tight range `[1e-2, 10]`.
FPS: **515k** (main decomposed: **526k**, gap: **-2.1%**)

## Root Cause Analysis

### Where the time goes (kernel trace, single step)

| Category | Main (us) | Opt (us) | Delta |
|---|---|---|---|
| Linesearch | 1349.8 | 823.5 | **-526.3** |
| Non-LS solver (hessian, gradient, etc.) | 2171.0 | 2581.6 | **+410.6** |
| func_solve_init | 733.0 | 780.0 | +47.1 |
| Total kernel exec | 7329.7 | 7463.9 | +134.2 |
| Total launches | 248 | 382 | +134 |

### Two independent problems

**Problem 1: Non-LS solver overhead (+411 us)**

Caused by 10.2% more active-env-iterations (8052 vs 7319). The gap concentrates at iters 3-5 where opt has 2-8x more active envs than main. The hessian scales super-linearly (1.24x cost for 1.10x work) due to Cholesky factorization.

The convergence gap is inherent to the grid search — K=32 over 3 decades gives limited precision. Parabolic interpolation and Newton correction both failed (cost function kinks). Increasing K or N_REFINE adds more launch overhead than it saves.

**Problem 2: Kernel launch overhead (~268 us)**

134 extra launches × ~2 us gap each. The parallel LS uses 8 kernel launches per solver iteration (mv, jv, p0, eval×3, apply_dofs, apply_constraints) vs main's 1 (`_kernel_linesearch`). Over 10 iterations: 80 vs 10 LS launches = 70 extra, plus the eval kernel launches 3 times per iteration.

### Budget

To beat main, we need to save more than 526.3 + 268 = ~795 us. Currently saving 526.3 on LS exec. Need to save ~268 us on launches (Problem 2) OR save enough on Non-LS by improving convergence (Problem 1).

## Plan: Reduce kernel launches by fusing LS sub-kernels

The parallel LS launches 8 kernels per iteration for the linesearch alone. Several of these can be fused:

### Approach: Fuse mv + jv + p0 + eval×N into a single kernel

Currently per iteration:
- `_kernel_parallel_linesearch_mv` — M @ search (parallel over dofs × envs)
- `_kernel_parallel_linesearch_jv` — J @ search (parallel over constraints × envs)
- `_kernel_parallel_linesearch_p0` — snorm + quad_gauss + eq_sum + p0_cost (block reduction per env, K threads)
- `_kernel_parallel_linesearch_eval` × 3 — evaluate K candidates + argmin (K threads per env)
- `_kernel_parallel_linesearch_apply_alpha_dofs` — apply alpha to qacc/Ma (parallel over dofs × envs)
- `_kernel_parallel_linesearch_apply_alpha_constraints` — apply alpha to Jaref (parallel over constraints × envs)

**Key observation**: `p0` already uses block_dim=32 (=K) with threads striding over dofs and constraints for shared-memory reductions. The `eval` kernel also uses block_dim=K. These share the same thread block structure. `mv` and `jv` can also be done with K threads striding over dofs/constraints.

**Proposed fusion**: One kernel `_kernel_parallel_linesearch_fused` with block_dim=K, one block per env:
1. K threads cooperatively compute mv (stride over dofs) — replaces `_kernel_mv`
2. K threads cooperatively compute jv (stride over constraints) — replaces `_kernel_jv`
3. Shared-memory reductions for snorm, quad_gauss, eq_sum, p0_cost — replaces `_kernel_p0`
4. For-loop over N_REFINE: K threads evaluate candidates + argmin reduction — replaces `_kernel_eval` × 3
5. K threads cooperatively apply alpha to dofs and constraints — replaces `_kernel_apply_*`

This reduces **8 launches to 1** per solver iteration, saving 70 launches over 10 iterations → **~140 us** of launch overhead.

Additionally, fusing the eval refine loop inside the kernel (step 4) eliminates inter-pass launch overhead and allows increasing N_REFINE to 6+ at near-zero cost, which could improve precision and reduce the convergence gap (Problem 1).

### Why this is different from the previous failed fused attempt

The previous fused attempt only fused the eval refine loop and used `sh_range` shared memory to communicate between passes. It broke convergence because of a communication bug between thread 0 and other threads.

This new approach fuses ALL LS sub-kernels into one kernel. The eval refine loop communicates via shared memory within the same kernel — no global memory round-trips between passes. The mv/jv/p0 steps write to `constraint_state` fields (global memory) which the eval step reads, maintaining the same data flow as the separate kernels.

### Revised approach: partial fusion (p0 + eval×N + apply)

Per reviewer feedback, full fusion risks serializing mv/jv (which need n_dofs×B / n_constraints×B parallelism). Instead, fuse only the kernels that already share block_dim=K:

**Keep separate**: `mv` (dof-parallel), `jv` (constraint-parallel)
**Fuse into one kernel**: `p0` + `eval`×N_REFINE + `apply_dofs` + `apply_constraints`

Per iteration, this goes from 8 launches → 3 (mv, jv, fused), saving 5 launches per iteration = 50 over 10 iterations = **~100 us**.

The apply step uses K=32 threads striding over dofs/constraints. Apply is trivially cheap (one multiply-add per element), so the serialization cost is negligible.

The eval refine loop runs inside the fused kernel as a for-loop with block syncs, enabling N_REFINE=6+ at zero extra launch cost.

### Implementation steps

1. **Step D (trivial)**: Fuse `apply_dofs` + `apply_constraints` into one kernel. Test. (saves 10 launches, ~20 us)
2. **Step A**: Create `_kernel_parallel_linesearch_p0_eval_apply` that combines p0 + eval×N_REFINE + apply. Test with N_REFINE=3.
3. **Step C**: Try N_REFINE=6 in the fused kernel. Measure convergence improvement.
4. Validate FPS.

### Expected impact

- Launch savings: 50 launches × 2 us = ~100 us
- If N_REFINE=6 closes convergence gap: additional ~100-200 us from reduced hessian/gradient
- Target: 515k + savings → potentially beating main's 526k
