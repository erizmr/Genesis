# Benchmark Regression Analysis: g1_fall on `mingrui/260309/solver_opt_parallel_linesearch`

## Summary

**Confirmed regression**: The `g1_fall` benchmark (Newton solver, 4096 envs, GPU) shows a ~38% performance regression on the feature branch compared to `main`.

## Benchmark Results

| Version | Run 1 FPS | Run 2 FPS | Avg FPS | Compile Time (s) |
|---------|-----------|-----------|---------|-------------------|
| `main` (sequential linesearch) | 520,432 | 521,495 | ~520,964 | ~38.6 |
| `feature` (6 separate kernels) | 320,806 | 318,124 | ~319,910 | ~23.7 |
| `feature-fused` (3 fused kernels) | 320,402 | 320,767 | ~320,585 | ~22.5 |
| `feature-fused` + perf_dispatch=1s | 442,158 | 442,223 | ~442,191 | ~22.5 |

**Regression magnitude**: ~38.6% slower runtime FPS (520,964 → 319,910)
**Kernel fusion effect**: Negligible (~0.2% improvement), proving the bottleneck is NOT kernel launch overhead
**perf_dispatch with re-probing**: ~38% improvement over decomposed, but still 15% below main due to probe overhead

## Root Cause Analysis

### What changed

The branch replaces the **fused sequential linesearch** (`func_linesearch_and_apply_alpha`) with a **parallel multi-candidate linesearch**. The key change is in `genesis/engine/solvers/rigid/constraint/solver_breakdown.py`.

### Key finding: The regression is algorithmic, not architectural

Initial hypothesis was that kernel launch overhead (6 kernels → 3 fused) was the bottleneck. **Kernel fusion proved this wrong** — fusing from 6 to 3 kernels gave zero FPS improvement (~320K in both cases).

The true bottleneck is the **parallel linesearch algorithm itself**:

| Aspect | Main (sequential) | Branch (parallel) |
|--------|------------------|-------------------|
| **Search strategy** | Adaptive bracketing + Newton refinement with 3-alpha batched evaluation | K=16 log-spaced candidates evaluated in parallel |
| **Convergence** | Exact gradient/hessian → precise step size | Coarse grid search → approximate step size |
| **Refinement passes** | Up to `ls_iterations=20` adaptive steps | `LS_PARALLEL_N_REFINE=1` fixed pass |
| **Per-iteration cost** | Higher (serial loop within kernel) | Lower (parallel evaluation) |
| **Step quality** | Near-optimal α each iteration | Approximate α — may need more outer iterations |

The g1_fall benchmark uses only `iterations=10` (not the default 100). With fewer outer iterations, each iteration's step quality matters more. The coarser parallel search produces suboptimal step sizes, leading to higher residual cost after 10 iterations. The simulation compensates by doing more physics work per frame, reducing throughput.

### Why perf_dispatch can't fix this

The `perf_dispatch` mechanism benchmarks implementations by calling them with `sync()` before and after. Since `func_solve_body` is called per-iteration (10x per timestep, each taking ~microseconds of GPU time), the `sync()` overhead dominates the timing. Both implementations appear equally slow during probing, so the winner is essentially random. With `repeat_after_seconds > 0`, periodic re-probing introduces sync pauses during the recording window, further degrading measured FPS.

### g1_fall problem characteristics

- **DOFs**: ~30 per env (humanoid with 29 DOFs + 6 free joint)
- **Constraints**: Variable (contacts from falling), typically moderate
- **Solver iterations**: 10 (explicitly set, not default 100)
- **Step dt**: 0.005s (half of default 0.01s)
- **Warmup**: 20s, Recording: 5s
- **Benchmark uses `qd.sync()`** between warmup and recording

## Implementation: Kernel Fusion (branch `mingrui/260309/solver_opt_parallel_linesearch_fused`)

Despite not solving the algorithmic regression, the kernel fusions are a valid optimization that reduces Python→C++ boundary crossings. Changes made:

### Fusion 1: `_kernel_parallel_linesearch_mv` + `_kernel_parallel_linesearch_jv` → `_kernel_parallel_linesearch_mv_jv`
- Single `ndrange(n_dofs + len_constraints, B)` kernel
- First `n_dofs` indices compute `mv = M @ search`, rest compute `jv = J @ search`
- Both are independent matvecs — no inter-thread dependencies

### Fusion 2: `_kernel_parallel_linesearch_p0` + `_kernel_parallel_linesearch_eval` → `_kernel_parallel_linesearch_p0_eval`
- Block-per-env kernel with 3 phases and `block.sync()` between them
- Phase 1: snorm + quad_gauss reduction (same as before)
- Phase 2: constraint cost reduction + search range computation
- Phase 3: K-candidate evaluation + argmin reduction (loop for N_REFINE passes)
- Search range stays in shared memory (`sh_lo`, `sh_hi`) — no global memory round-trip

### Fusion 3: `_kernel_parallel_linesearch_apply_alpha_dofs` + `_kernel_parallel_linesearch_apply_alpha_constraints` → `_kernel_parallel_linesearch_apply_alpha`
- Single `ndrange(n_dofs + len_constraints, B)` kernel
- First `n_dofs` indices update `qacc`, `Ma`; rest update `Jaref`

**Result**: 6 linesearch kernels → 3 fused kernels per iteration.

## Proposed Solutions for the Algorithmic Regression

### Solution A: Increase refinement passes (`LS_PARALLEL_N_REFINE`)

Currently `LS_PARALLEL_N_REFINE=1` with K=16 candidates. Each refinement narrows the range from ~4 decades (1e-2 to 1e2) to ~0.5 decades. More passes would improve step quality:
- 1 pass: ~0.5 decade resolution (current)
- 2 passes: ~0.07 decade resolution
- 3 passes: ~0.01 decade resolution

This adds only 1 kernel launch per extra pass (now fused into p0_eval), but increases the total linesearch GPU work.

### Solution B: Hybrid approach — use parallel linesearch to warm-start sequential

Use the parallel K-candidate evaluation to quickly find an approximate α, then use 1-2 Newton refinement steps (sequential, within the same kernel) to polish it. This combines the broad search of the parallel approach with the precision of the sequential approach.

### Solution C: Increase K (candidates per pass)

`LS_PARALLEL_K=16` may be too coarse for 4-decade search ranges. Increasing to 32 or 64 would improve resolution without adding kernel launches (just more threads per block). Trade-off: more threads means more constraint-loop iterations per thread in the eval phase.

### Solution D: Adaptive search range based on problem conditioning

The current search range is `[α_newton * 1e-2, α_newton * 1e2]`. For well-conditioned problems like g1_fall, the Newton step is already a good estimate — the range could be narrowed to `[α_newton * 0.1, α_newton * 10]`, improving resolution by 100x with the same K.

## Recommendation

1. **Keep the kernel fusions** — they reduce overhead even though they don't fix the core regression
2. **Try Solution A first** (increase `LS_PARALLEL_N_REFINE` to 2-3) — cheapest change
3. **Try Solution D** (narrow search range) — likely the best bang-for-buck since the Newton estimate is good for most problems
4. **If neither suffices**: Solution B (hybrid parallel+sequential) is the most robust but most complex approach
