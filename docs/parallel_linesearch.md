# Parallel vs Iterative Linesearch in the Constraint Solver

## Context

The constraint solver minimizes a piecewise-quadratic cost function over joint accelerations, subject to contact/friction/equality constraints. Each solver iteration picks a search direction, then performs a **linesearch** to find the optimal step size `alpha` along that direction:

```
qacc_new = qacc + alpha * search
```

The cost function `C(alpha)` is composed of:
- **Gauss term**: quadratic in alpha (smooth)
- **Equality constraints**: quadratic in alpha (smooth, always active)
- **Friction constraints**: quadratic when `|x| < rf`, linear when `|x| >= rf` (piecewise, kink at threshold)
- **Contact constraints**: quadratic when `x < 0`, zero when `x >= 0` (piecewise, kink at `x = 0`)

where `x = Jaref + alpha * jv` for each constraint. The kink points occur at `alpha_kink = -Jaref/jv`.

---

## Iterative (Sequential) Linesearch

**File**: `solver.py`, `func_linesearch_batch`

### Algorithm

A derivative-guided 3-phase Newton linesearch, running **one thread per env**:

**Phase 1 — Init + Newton step:**
1. Compute cost, gradient, and hessian at `alpha = 0` (the "p0" point)
2. Take a Newton step: `p1 = -gradient / hessian`
3. Evaluate cost/gradient/hessian at `p1`
4. If `|gradient| < gtol` → converged, return `p1`

**Phase 2 — Bracketing:**
1. Repeatedly take Newton steps until the derivative changes sign
2. This establishes a bracket `[p1, p2]` containing the minimum
3. Up to `ls_iterations` (default 20) inner iterations

**Phase 3 — Refinement:**
1. Evaluate 3 candidates per iteration: Newton step, bracket endpoint, midpoint
2. Check convergence: `|gradient| < gtol` for any candidate
3. Update brackets toward the minimum using derivative information
4. Up to `ls_iterations` total inner iterations

### Key properties
- Uses **derivative (gradient + hessian)** at each candidate alpha
- Serial execution: one thread handles all inner iterations for one env
- Handles kinks correctly: derivative sign changes bracket the kink
- Convergence criterion: `|gradient| < tolerance * ls_tolerance * snorm * scale`
- Typical: converges in 3-5 inner iterations for most envs

---

## Parallel Linesearch

**File**: `solver_breakdown.py`

### Algorithm

A grid-search linesearch using **K=32 threads per env** evaluating candidates in parallel:

**Step 1 — Precompute (`_kernel_mv`, `_kernel_jv`):**
- `mv = M @ search` — mass matrix × search direction
- `jv = J @ search` — Jacobian × search direction
- Fully parallel over (dof × env) and (constraint × env)

**Step 2 — Init + Newton seed (`_kernel_p0`):**
1. Compute snorm, quad_gauss coefficients, eq_sum, p0_cost (K threads per env, shared-memory reduction)
2. Compute Newton step: `alpha_newton = |grad / hess|`
3. **Initialize best alpha to the Newton step** with its quadratic-approximated cost. This gives derivative-guided precision as the starting point.
4. Set search range: `[alpha_newton * 1e-2, alpha_newton * 10]` (3 decades, centered on Newton)

**Step 3 — Grid evaluation + refinement (`_kernel_eval` × N_REFINE):**
For each of N_REFINE=3 passes:
1. Each of K=32 threads evaluates cost at one log-spaced candidate alpha within `[lo, hi]`
2. Shared-memory argmin reduction finds the best candidate
3. Accept if `cost < p0_cost` AND `cost < best_so_far`
4. Narrow range to `[best-1, best+1]` for next pass

**Step 4 — Apply (`_kernel_apply_alpha`):**
- Update `qacc += search * alpha`, `Ma += mv * alpha`, `Jaref += jv * alpha`

### Key properties
- **No derivatives** computed — pure cost comparison
- **Newton step as initial best**: the grid search only overrides the Newton alpha when it finds a genuinely lower cost (at kink points)
- K=32 candidates evaluated in parallel per pass
- 3 refinement passes → effective resolution: `3 decades / 32³ ≈ 9e-5` decades
- Each kernel launch has fixed overhead (~2 us) regardless of active env count

---

## Comparison

### Per-iteration cost

| | Iterative | Parallel |
|---|---|---|
| Kernel launches per solver iter | 1 | 6 (mv, jv, p0, eval×3, apply) |
| GPU threads per env | 1 (serial) | 32 (parallel) |
| Constraint loop iterations | 3-5 × n_constraints (sequential Newton) | K × n_constraints × N_REFINE (parallel grid) |
| Derivative computation | Yes (gradient + hessian per alpha) | No (cost only) |

### Alpha precision

| | Iterative | Parallel |
|---|---|---|
| Starting point | Newton step from alpha=0 | Same Newton step (used as initial best) |
| Refinement | Derivative-guided Newton iterations | Grid narrowing around best |
| Kink handling | Derivative sign change → exact bracket | Grid samples both sides; Newton seed handles smooth cases |
| Final precision | ~1e-8 relative (converges to |grad| < gtol) | ~1e-4 relative (grid resolution) but **starts from Newton**, so effective precision is high |

### Performance profile (g1_fall, 4096 envs, RTX 5090)

| Metric | Iterative | Parallel |
|---|---|---|
| Linesearch kernel time | 1350 us (20 launches) | 821 us (160 launches) |
| Active-env-iters (convergence) | 7215 | 7059 |
| Total FPS | 517k | **528k** |

### Trade-offs

**Parallel wins when:**
- Many envs are active (maximizes GPU occupancy in the eval kernel)
- The cost function is smooth near the Newton step (Newton initial dominates)
- Batch size is large (amortizes kernel launch overhead)

**Iterative wins when:**
- Few envs are active (serial per-env is fine, less launch overhead)
- Cost function has many kinks near the optimum (derivative-guided bracketing is precise)
- `ls_iterations` budget is sufficient (typically is)

**Why parallel is faster overall:**
1. The Newton initial best gives the parallel LS the same starting precision as the iterative LS
2. The grid search costs ~7 us per eval pass (K threads in parallel) vs ~60 us per iterative Newton step (serial constraint loop)
3. The grid search adds value at kink points where the Newton approximation is inaccurate, slightly improving convergence over the iterative LS at iters 1-3
4. The main cost is kernel launch overhead (6 launches vs 1), but the faster per-eval cost more than compensates

### Architecture summary

```
Iterative (1 thread per env):          Parallel (32 threads per env):
┌─────────────────────────┐            ┌─────────────────────────────────┐
│ for env in parallel:    │            │ _kernel_mv:  mv = M @ search   │ ← n_dofs × B threads
│   compute mv, jv        │            │ _kernel_jv:  jv = J @ search   │ ← n_con × B threads
│   compute p0, Newton    │            │ _kernel_p0:                    │ ← K threads per env
│   while not converged:  │            │   snorm, quad_gauss, eq_sum    │
│     eval cost+grad+hess │            │   Newton seed → initial best   │
│     Newton step         │            │ _kernel_eval × 3:              │ ← K threads per env
│     update bracket      │            │   32 candidates in parallel    │
│   apply alpha           │            │   argmin → narrow range        │
│                         │            │ _kernel_apply:                 │ ← n_items × B threads
│ 1 kernel launch total   │            │   apply best alpha             │
└─────────────────────────┘            │ 6 kernel launches total        │
                                       └─────────────────────────────────┘
```
