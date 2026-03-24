# Parallel Linesearch v3 — Experiment Log

Base: branch `mingrui/parallel-ls-grad-check-v2` (commit 5b304f30)

## Audit Findings

Full audit identified these optimization opportunities (ordered by expected impact):

1. **Cooperative bisection** (HIGH) — bisection runs thread-0 only, 31 threads idle.
   Each `_ls_eval_cost_grad` call is O(n_constraints) on 1 thread. With cooperative
   reduction, becomes O(n_constraints/32) per thread per step.

2. **Batch candidate evaluation** (MEDIUM) — 9 candidates evaluated sequentially in
   Phase 1. Could partition 32 threads into groups to evaluate 2-4 candidates per
   iteration, reducing loop iterations from 9 to ~3.

3. **Fuse shared iteration kernels** (LOW) — forces + qfrc + cost are 3 separate
   launches that could be 1. Saves ~2 launches × ~2μs = ~4μs per iteration.

4. **Skip redundant Newton alpha=0** (TRIVIAL) — when total_hess <= 0, alpha_newton=0
   duplicates p0_cost evaluation.

5. **Remove redundant sync** (TRIVIAL) — line 576 syncs right after line 573.

Edge cases audited (all correct):
- active=False: thread 0 sets candidates[0]=0, Phase 4 skips
- No grid improvement: best_alpha stays 0, Phase 3 skipped, Phase 4 sets improved=False
- |grad| < gtol: bisection skipped, Phase 2 result retained
- Invalid bracket (g_a >= 0 or g_b <= 0): bisection skipped, Phase 2 result retained
- alpha_newton == 0: redundant eval but correct (minor waste)

---

## Experiment 1: Cooperative Bisection (commit 753fc7e4)

Rewrote Phase 3 bisection to use all 32 threads for cost+gradient reduction at each
midpoint. Thread 0 broadcasts the evaluation alpha via shared memory, all threads
reduce constraints cooperatively, thread 0 reads the result and updates the bracket.

Changes:
- Replaced `_ls_eval_cost_grad` (thread-0 only) with cooperative reduction pattern
  matching Phase 1's grid search
- Each bisection step: all 32 threads reduce cost + qt1 + qt2, then thread 0 computes
  grad and cost from the reduced values
- Thread 0 signals termination via sh_cand_alpha[1] <= 0
- Cost-guarded acceptance: only accepts if c_eval < p0_cost

**Correctness**: 262 passed, 7 skipped, 1 xfailed

**Performance** (`260323_coop_bisect`, --solver decomposed):

| Case | fused_v1 (N=8) | coop_bisect (N=8) | Change |
|------|----------------|-------------------|--------|
| g1_fall | +8.7% | +9.0% | +0.3pp |
| dex_hand | +15.9% | +15.2% | -0.7pp |
| box_pyramid_6 | -7.9% | -7.8% | +0.1pp |

**Verdict: REVERTED.** No measurable gain — bisection fires rarely (grid usually converges).

---

## Experiment 2: LS_N_CANDIDATES sweep (commits 3245ef9e, 9e87ca6a)

Fewer candidates = less cooperative reduction work per iteration. Bisection handles
refinement, so fewer candidates should suffice for an initial bracket.

**Correctness**: 262 passed for all values (4, 6, 8)

**Performance** (`--solver decomposed`):

| N_CAND | g1_fall | box_pyramid_6 | dex_hand | Notes |
|--------|---------|---------------|----------|-------|
| 4 | +6.7% | **-1.8%** | -4.3% | Too coarse for dex_hand |
| **6** | **+10.0%** | -5.8% | **+13.4%** | Best overall balance |
| 8 | +8.7% | -7.9% | +15.9% | Best for dex_hand only |

**Verdict: KEEP N=6.** Best g1_fall (+10.0%), good dex_hand (+13.4%), better box_pyramid_6
than N=8. The trade-off is 2 fewer cooperative reductions per solver iteration (7 vs 9 total
candidates including Newton), saving ~2 × n_constraints/32 work per thread.
