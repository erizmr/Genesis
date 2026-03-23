# Decision Tree Implementation Plan

## Analysis of the Proposed Decision Tree

### What we currently have (545k FPS, all tests pass)

```
Phase 0: mv, jv, p0 (compute Newton alpha, search range, gtol)
Phase 1: 1 eval pass (K=32 candidates, thread 0 = Newton alpha)
         → argmin → accept if cost < p0_cost
         → gradient check at best_alpha
         → Newton correction if |grad| > gtol
Phase apply: apply best alpha
```

### What the decision tree proposes

The full tree adds 4 conditional phases after Phase 1:
- **Branch A** (no improvement): check grad(0), expand if still descending
- **Branch B.3** (grad < -gtol): bracket check, bisect or expand
- **Phase 2** (grad > gtol): 2 more refine passes + Newton correction
- **Phase 3b** (bisection): up to 12 iterations of binary search
- **Phase 4** (expansion): exponential stepping to find bracket + bisection

### Is it reasonable?

**Yes, the logic is sound.** Each branch handles a distinct failure mode:

| Current failure | Decision tree fix | Frequency (est.) |
|---|---|---|
| Grid finds nothing, Newton at alpha=0 is optimal | Branch A: check grad(0), confirm | ~5% of envs |
| Grid finds improvement but not at minimum (grad < -gtol) | Branch B.3: bracket + bisect | ~10% of envs |
| Grid overshot (grad > gtol) | Phase 2: refine + Newton | ~10% of envs (current Newton correction handles most) |
| Minimum outside search range | Phase 4: expand + bisect | ~2% of envs |

### Concerns

1. **Complexity vs benefit.** The current implementation handles ~85% of envs optimally. The decision tree improves the remaining ~15% at the cost of significant code complexity and conditional kernel launches.

2. **Gradient computation via "2 cost evals" (finite difference).** The plan says gradient is computed with 2 cost evaluations. But we already showed finite difference is unreliable at kinks. **We should use analytical gradient instead** (as in current implementation) — same serial constraint loop, ~0 extra cost.

3. **Bisection requires multiple serial iterations.** Phase 3b does up to 12 bisection steps, each needing a gradient evaluation (serial constraint loop). That's 12 × ~18 constraints × ~8 FLOPs = ~1700 FLOPs by thread 0 per env. For the 10% of envs that need it, this adds ~170 FLOPs average per env — negligible.

4. **Conditional kernel launches.** Phases 2, 3, 3b, 4 are conditional. If implemented as separate kernels, each adds ~5 us launch overhead even when most envs skip. If integrated into the eval kernel, the JIT template variants add compilation overhead.

### Key insight: integrate everything into thread 0's post-reduction block

The current gradient check + Newton correction runs in thread 0 after the argmin. All the additional phases (bisect, expand) can also run in thread 0 — they're serial per-env operations. Thread 0 is idle after the reduction anyway. The constraint loop for gradient evaluation is ~0.1 us per env.

**No extra kernel launches needed.** The entire decision tree runs inside the existing eval kernel's thread 0 block on the last (and only) pass.

## Implementation Plan

### Step 1: Refactor gradient computation into a reusable function

Extract the current gradient+hessian computation (lines 402-428) into a `@qd.func`:

```python
@qd.func
def _compute_grad_hess(alpha, i_b, constraint_state):
    """Returns (grad, hess) at alpha for env i_b."""
    # ... same constraint loop as current ...
    return grad, hess
```

This is reused by Newton correction, bisection, and expansion.

### Step 2: Implement bisection in thread 0

After the Newton correction, if `|grad| > gtol`, run bisection:

```python
# Bisection: bracket is [a, b] where grad(a) < 0, grad(b) > 0
a = alpha_with_negative_grad   # from Newton correction or expansion
b = alpha_with_positive_grad   # from grid search overshoot
for _ in range(LS_BISECT_STEPS):
    mid = (a + b) * 0.5
    grad_mid, _ = _compute_grad_hess(mid, i_b, ...)
    if abs(grad_mid) < gtol:
        break
    if grad_mid < 0: a = mid
    else: b = mid
final_alpha = mid
```

### Step 3: Implement expansion in thread 0

When grad(best) < -gtol (still descending) and grad(hi) <= 0 (no bracket within range):

```python
# Expansion: find bracket by exponential stepping
a = best_alpha
b = hi
for _ in range(LS_EXPANSION_STEPS):
    b = min(b * LS_EXPANSION_FACTOR, AMAX)
    grad_b, _ = _compute_grad_hess(b, i_b, ...)
    if grad_b > -gtol:
        # Found bracket [a, b] → bisect
        break
    a = b
```

### Step 4: Complete decision tree in thread 0

Integrate all phases into the `_is_last_pass` block:

```python
if _is_last_pass:
    final_alpha = candidates[0]
    if abs(final_alpha) < EPS:
        # Branch A: check if grad(0) indicates descending
        grad_0, _ = _compute_grad_hess(0, ...)
        if grad_0 < -gtol:
            # Expand from hi
            final_alpha = _expand_and_bisect(hi, ...)
    else:
        grad, hess = _compute_grad_hess(final_alpha, ...)
        if abs(grad) < gtol:
            pass  # converged
        elif grad > gtol:
            # Overshot → Newton correction
            alpha_c = final_alpha - grad / hess
            if alpha_c > 0:
                final_alpha = alpha_c
                # Verify Newton result
                grad_c, _ = _compute_grad_hess(alpha_c, ...)
                if abs(grad_c) > gtol:
                    # Newton didn't converge → bisect
                    final_alpha = _bisect(alpha_c, final_alpha, ...)
        else:  # grad < -gtol
            # Still descending → check hi for bracket
            grad_hi, _ = _compute_grad_hess(hi, ...)
            if grad_hi > 0:
                final_alpha = _bisect(final_alpha, hi, ...)
            else:
                final_alpha = _expand_and_bisect(hi, ...)
    candidates[0] = final_alpha
```

### Constants

| Name | Value | Description |
|---|---|---|
| `LS_BISECT_STEPS` | 8 | Max bisection iterations (2^8 = 256x precision) |
| `LS_EXPANSION_STEPS` | 6 | Max expansion steps (4^6 = 4096x range) |
| `LS_EXPANSION_FACTOR` | 4.0 | Exponential growth factor per expansion step |
| `AMAX` | 1e4 | Maximum allowed alpha |

### Implementation order

1. Extract `_compute_grad_hess` function — test (should match current behavior exactly)
2. Add bisection after Newton correction — test convergence improvement
3. Add expansion for Branch A and grad < -gtol cases — test
4. Measure FPS impact at each step

### Expected impact

- **FPS**: Minimal change. All new logic runs in thread 0's serial block. The constraint loop is ~0.1 us per env. Bisection adds 8 × 0.1 = 0.8 us for envs that need it (~10%). Average: ~0.08 us per env → ~0.3 us per step. Negligible vs 7700 us step time.
- **Convergence**: The 10-15% of envs that currently have 10-60% alpha error should converge to |grad| < gtol, matching the iterative LS's convergence criterion for all envs.
- **Active-env-iters**: Should drop from 7161 toward main's 7298 or potentially below (the gradient criterion is stricter than cost-only).
