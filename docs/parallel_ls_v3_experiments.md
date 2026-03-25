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

---

## Experiment 3: repeat_after_count for --solver auto (commit 66403087)

**Problem**: With `repeat_after_seconds=0` (disabled on main via PR #2599), perf_dispatch
locks its decision at warmup. For g1_fall, warmup has few contacts → monolith wins →
locked forever → -42% regression in steady state.

**Root cause investigation**: Compared before/after rebase benchmarks:
- Before rebase (`260324_new_par_auto`): g1_fall **+10.3%**, box_pyramid_6 -0.4%
- After rebase (`260324_new_par_auto_double_check`): g1_fall **-42.2%**, box_pyramid_6 +6.0%

The rebase brought PR #2599 which changed `repeat_after_seconds` from 1.0 to 0.0. Before
the rebase, periodic re-benchmarking allowed perf_dispatch to detect that decomposed+parallel
became faster after the humanoid fell. After the rebase, the warmup decision is permanent.

**Solution**: Use `repeat_after_count=3000` — re-benchmark after 3000 calls (~300 steps).
By step 300, g1_fall's humanoid has landed and has many contacts. The re-benchmark switches
to decomposed+parallel. For fast scenes, the sync overhead is ~120μs every 3000 calls
(0.04% for g1_fall, ~1.7% for fast 30k scenes).

**Correctness**: 262 passed, 7 skipped, 1 xfailed

**Performance** (`260323_rebench3000_auto`, --solver auto):

| Case | Main FPS | Branch FPS | Delta |
|------|----------|------------|-------|
| **g1_fall** | 460,065 | 505,651 | **+9.9%** |
| box_pyramid_6 | 21,032 | 20,985 | -0.2% |
| dex_hand | 5,514 | 5,502 | -0.2% |
| anymal_zero 30k | 13,730,507 | 13,498,325 | -1.7% |
| franka 30k | 15,183,542 | 14,915,836 | -1.8% |

**Verdict: KEEP.** g1_fall +9.9% with `--solver auto`. box_pyramid_6 at parity.
Fast scenes show ~1.7% regression from periodic re-benchmark sync — acceptable tradeoff.

---

## Experiment 4: Branch A2 + B2 expansion (commit 5879fb78) — REVERTED

Implemented the missing decision tree branches identified in `docs/comments_from_claude.md`:
- Branch A2: grad(0) < -gtol → exponential expansion (lo, 4×lo, 16×lo, 64×lo) + bisect
- Branch B2: grad(best) < -gtol → expansion from best_alpha (4x, 16x, 64x) + bisect
- Branch B3: grad(best) > gtol → bisect [best/2, best]

All expansion/bisection runs on thread-0 using `_ls_eval_cost_grad`.

**Correctness**: 262 passed

**Performance** (`260323_expansion_decomposed`, --solver decomposed):

| Case | Before | After | Change |
|------|--------|-------|--------|
| g1_fall | +10.0% | +8.6% | -1.4pp |
| box_pyramid_6 | -5.8% | **-19.5%** | -13.7pp |
| dex_hand | +13.4% | +13.7% | +0.3pp |

**Verdict: REVERTED.** The expansion runs on thread 0 only (31 threads idle), making
each expansion step O(n_constraints) with no parallelism. For box_pyramid_6 (many
constraints), this adds massive overhead. The expansion fires more often than expected
and the alpha quality improvement doesn't translate to fewer outer solver iterations.

The review's suggestion is theoretically sound but the thread-0-only implementation
is too expensive. Would need cooperative expansion (all 32 threads) to be viable, but
that adds significant kernel complexity for marginal quality gains.

---

## Experiment 5: Conditional expansion kernel (commit f7fbd306) — REVERTED

Attempted a middle ground: expansion in a SEPARATE kernel that checks a per-env flag.
The eval kernel sets `candidates[6]` when expansion is needed (Branch A2 or B2). The
expansion kernel reads the flag, skips instantly for flag=0 envs, and only runs
expansion+bisection+delta-apply for flagged envs.

**Correctness**: 262 passed

**Performance** (`260323_cond_expand`, --solver decomposed):

| Case | No expansion | Cond. expansion | Change |
|------|-------------|-----------------|--------|
| g1_fall | +10.0% | +4.9% | -5.1pp |
| box_pyramid_6 | -5.8% | -20.8% | -15.0pp |
| dex_hand | +13.4% | +8.5% | -4.9pp |

**Verdict: REVERTED.** Even with conditional execution, the extra kernel launch (~2μs ×
10 iterations = ~20μs per step) plus the expansion work for flagged envs hurts ALL cases.
The flag fires more often than expected for box_pyramid_6.

**Final conclusion on expansion**: The uncovered branches (A2, B2) are real but the
performance cost of covering them exceeds the benefit in ALL tested implementations:
1. In-kernel thread-0 expansion: -19.5% box_pyramid (Experiment 4)
2. Separate conditional kernel: -20.8% box_pyramid (Experiment 5)
The alpha quality improvement from expansion doesn't reduce outer solver iteration
count enough to offset the per-iteration overhead. The current grid search + bisection
is the right trade-off.
