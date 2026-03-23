"""
Compare parallel LS alpha vs iterative (sequential Newton) LS alpha,
as a function of K and N_REFINE.

Gold standard: the full iterative Newton linesearch (3-phase bracketing),
implemented in Python on the captured constraint state.
"""
import numpy as np
import torch
import quadrants as qd
import genesis as gs

torch.manual_seed(42)
gs.init(backend=gs.cuda, precision="32", logging_level="warning")

scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        dt=0.005, iterations=10, tolerance=1e-5, ls_iterations=20,
        constraint_solver=gs.constraint_solver.Newton,
    ),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
from tests.test_rigid_benchmarks import get_hf_dataset, get_file_morph_options
asset_path = get_hf_dataset(pattern="unitree_g1/*")
robot = scene.add_entity(
    gs.morphs.MJCF(**get_file_morph_options(
        file=f"{asset_path}/unitree_g1/g1_29dof_rev_1_0.xml", pos=(0, 0, 1.0),
    )), vis_mode="collision",
)
n_envs = 4096
scene.build(n_envs=n_envs)
init_qpos = torch.zeros((robot.n_qs,), dtype=gs.tc_float, device=gs.device)
init_qpos[2] = 1.0; init_qpos[3] = 1.0
robot.set_qpos(init_qpos)
random_forces = torch.zeros((n_envs, robot.n_dofs), dtype=gs.tc_float, device=gs.device)

print("Warming up 275 steps...")
for i in range(275):
    random_forces.uniform_(-50, 50)
    robot.control_dofs_force(random_forces)
    scene.step()

# Capture state after p0 kernel
from genesis.engine.solvers.rigid.constraint import solver_breakdown
captured = {}
_orig_eval = solver_breakdown._kernel_parallel_linesearch_eval
_call_count = [0]
def _capture_eval(*args, **kwargs):
    _call_count[0] += 1
    if _call_count[0] == 1:
        qd.sync()
        cs = args[0]
        captured['quad_gauss'] = cs.quad_gauss.to_numpy().copy()
        captured['eq_sum'] = cs.eq_sum.to_numpy().copy()
        captured['candidates'] = cs.candidates.to_numpy().copy()
        captured['Jaref'] = cs.Jaref.to_numpy().copy()
        captured['jv'] = cs.jv.to_numpy().copy()
        captured['efc_D'] = cs.efc_D.to_numpy().copy()
        captured['n_constraints'] = cs.n_constraints.to_numpy().copy()
        captured['n_constraints_equality'] = cs.n_constraints_equality.to_numpy().copy()
        captured['n_constraints_frictionloss'] = cs.n_constraints_frictionloss.to_numpy().copy()
        captured['efc_frictionloss'] = cs.efc_frictionloss.to_numpy().copy()
        captured['diag'] = cs.diag.to_numpy().copy()
        captured['gauss'] = cs.gauss.to_numpy().copy()
        # Read tolerance info
        captured['meaninertia'] = cs.meaninertia.to_numpy().copy() if hasattr(cs, 'meaninertia') else None
    _orig_eval(*args, **kwargs)
solver_breakdown._kernel_parallel_linesearch_eval = _capture_eval

random_forces.uniform_(-50, 50)
robot.control_dofs_force(random_forces)
scene.step()
qd.sync()
solver_breakdown._kernel_parallel_linesearch_eval = _orig_eval

has_con = np.where(captured['n_constraints'] > 0)[0]
print(f"Captured state for {len(has_con)} envs with constraints")


# === CPU cost function evaluation (scalar and vector) ===
def eval_point(alpha, env_i):
    """Evaluate cost, gradient, hessian at a single alpha for one env.
    Matches func_ls_point_fn_opt from solver.py."""
    qg = captured['quad_gauss'][:, env_i]
    eq = captured['eq_sum'][:, env_i]
    ne = int(captured['n_constraints_equality'][env_i])
    nef = ne + int(captured['n_constraints_frictionloss'][env_i])
    n_con = int(captured['n_constraints'][env_i])

    # Start from gauss + equality
    t0 = qg[0] + eq[0]
    t1 = qg[1] + eq[1]
    t2 = qg[2] + eq[2]

    # Friction constraints
    for ic in range(ne, nef):
        Ja = captured['Jaref'][ic, env_i]
        jv = captured['jv'][ic, env_i]
        D = captured['efc_D'][ic, env_i]
        f_val = captured['efc_frictionloss'][ic, env_i]
        r = captured['diag'][ic, env_i]
        qf0 = D * 0.5 * Ja * Ja
        qf1 = D * jv * Ja
        qf2 = D * 0.5 * jv * jv
        x = Ja + alpha * jv
        rf = r * f_val
        if x <= -rf or x >= rf:
            if x <= -rf:
                qf0 = f_val * (-0.5*rf - Ja)
                qf1 = -f_val * jv
            else:
                qf0 = f_val * (-0.5*rf + Ja)
                qf1 = f_val * jv
            qf2 = 0.0
        t0 += qf0
        t1 += qf1
        t2 += qf2

    # Contact constraints
    for ic in range(nef, n_con):
        Ja = captured['Jaref'][ic, env_i]
        jv = captured['jv'][ic, env_i]
        D = captured['efc_D'][ic, env_i]
        x = Ja + alpha * jv
        active = 1.0 if x < 0 else 0.0
        t0 += D * 0.5 * Ja * Ja * active
        t1 += D * jv * Ja * active
        t2 += D * 0.5 * jv * jv * active

    cost = alpha*alpha*t2 + alpha*t1 + t0
    grad = 2*alpha*t2 + t1
    hess = 2*t2
    if hess <= 0:
        hess = 1e-30
    return cost, grad, hess


def eval_cost_batch(alphas, env_i):
    """Evaluate cost at array of alphas for one env."""
    qg = captured['quad_gauss'][:, env_i]
    eq = captured['eq_sum'][:, env_i]
    ne = int(captured['n_constraints_equality'][env_i])
    nef = ne + int(captured['n_constraints_frictionloss'][env_i])
    n_con = int(captured['n_constraints'][env_i])
    a = np.asarray(alphas, dtype=np.float64)
    costs = a*a*qg[2] + a*qg[1] + qg[0] + a*a*eq[2] + a*eq[1] + eq[0]
    for ic in range(ne, nef):
        Ja = captured['Jaref'][ic, env_i]
        jv = captured['jv'][ic, env_i]
        D = captured['efc_D'][ic, env_i]
        f_val = captured['efc_frictionloss'][ic, env_i]
        r = captured['diag'][ic, env_i]
        x = Ja + a * jv
        rf = r * f_val
        ln = x <= -rf; lp = x >= rf; mid = ~(ln | lp)
        costs += ln*f_val*(-0.5*rf - Ja - a*jv) + lp*f_val*(-0.5*rf + Ja + a*jv) + mid*D*0.5*x**2
    for ic in range(nef, n_con):
        Ja = captured['Jaref'][ic, env_i]
        jv = captured['jv'][ic, env_i]
        D = captured['efc_D'][ic, env_i]
        x = Ja + a * jv
        costs += (x < 0) * D * 0.5 * x**2
    return costs


# === Iterative Newton linesearch (3-phase, matching solver.py) ===
def iterative_newton_ls(env_i, tolerance=1e-5, ls_tolerance=0.2, ls_iterations=20):
    """Full 3-phase Newton linesearch matching func_linesearch_batch."""
    # p0 at alpha=0
    p0_cost, p0_grad, p0_hess = eval_point(0.0, env_i)

    # Adaptive tolerance
    # snorm is already checked in p0 kernel; we just use a reasonable gtol
    # For simplicity, use a small absolute tolerance on gradient
    n_dofs = captured['quad_gauss'].shape[1]  # not exactly right but doesn't matter
    gtol = tolerance * ls_tolerance * 1.0  # simplified

    # Phase 1: Newton step from p0
    p1_alpha = -p0_grad / p0_hess
    p1_cost, p1_grad, p1_hess = eval_point(p1_alpha, env_i)

    if p0_cost < p1_cost:
        p1_alpha, p1_cost, p1_grad, p1_hess = 0.0, p0_cost, p0_grad, p0_hess

    if abs(p1_grad) < gtol:
        return p1_alpha

    # Phase 2: Bracketing
    direction = 1 if p1_grad < 0 else -1
    p2_alpha, p2_cost, p2_grad, p2_hess = p1_alpha, p1_cost, p1_grad, p1_hess
    p2_updated = False
    it = 0

    while p1_grad * direction <= -gtol and it < ls_iterations:
        p2_alpha, p2_cost, p2_grad, p2_hess = p1_alpha, p1_cost, p1_grad, p1_hess
        p2_updated = True
        new_alpha = p1_alpha - p1_grad / p1_hess
        p1_alpha = new_alpha
        p1_cost, p1_grad, p1_hess = eval_point(p1_alpha, env_i)
        it += 1
        if abs(p1_grad) < gtol:
            return p1_alpha

    if it >= ls_iterations or not p2_updated:
        return p1_alpha

    # Phase 3: Refinement with 3-alpha evaluation
    for _ in range(ls_iterations - it):
        alpha_0 = p1_alpha - p1_grad / p1_hess  # Newton from p1
        alpha_1 = p1_alpha
        alpha_2 = (p1_alpha + p2_alpha) * 0.5  # midpoint

        c0, g0, h0 = eval_point(alpha_0, env_i)
        c1, g1, h1 = eval_point(alpha_1, env_i)
        c2, g2, h2 = eval_point(alpha_2, env_i)

        # Check convergence
        candidates = [(alpha_0, c0, g0, h0), (alpha_1, c1, g1, h1), (alpha_2, c2, g2, h2)]
        best = None
        for a, c, g, h in candidates:
            if abs(g) < gtol:
                if best is None or c < best[1]:
                    best = (a, c, g, h)
        if best is not None:
            return best[0]

        # Update brackets
        for a, c, g, h in candidates:
            if g * p1_grad < 0 and abs(g) < abs(p1_grad):
                p1_alpha, p1_cost, p1_grad, p1_hess = a, c, g, h
            if g * p2_grad < 0 and abs(g) < abs(p2_grad):
                p2_alpha, p2_cost, p2_grad, p2_hess = a, c, g, h

    # Return best of p1, p2
    if p1_cost <= p2_cost and p1_cost < p0_cost:
        return p1_alpha
    elif p2_cost < p0_cost:
        return p2_alpha
    return 0.0


def parallel_ls_cpu(env_i, K, N_REFINE, use_newton_init=True):
    """Simulate parallel grid search on CPU."""
    cands = captured['candidates'][:, env_i]
    p0_cost = cands[1]
    lo = cands[2]
    hi = cands[3]
    if use_newton_init:
        best_alpha = cands[0]  # Newton initial
        best_cost = cands[4]
    else:
        best_alpha = 0.0
        best_cost = 1e30
    for _r in range(N_REFINE):
        log_lo = np.log(max(lo, 1e-30))
        log_hi = np.log(max(hi, 1e-30))
        step = (log_hi - log_lo) / max(1, K - 1)
        alphas = np.exp(log_lo + np.arange(K) * step)
        costs = eval_cost_batch(alphas, env_i)
        best_idx = int(np.argmin(costs))
        if costs[best_idx] < p0_cost and costs[best_idx] < best_cost:
            best_alpha = alphas[best_idx]
            best_cost = costs[best_idx]
            lo_idx = max(0, best_idx - 1)
            hi_idx = min(K - 1, best_idx + 1)
            lo = np.exp(log_lo + lo_idx * step)
            hi = np.exp(log_lo + hi_idx * step)
    return best_alpha


# === Run experiments ===
n_sample = 500
sample_envs = has_con[np.linspace(0, len(has_con)-1, n_sample, dtype=int)]

# 1. Compute iterative Newton LS alphas (gold standard)
print("Computing iterative Newton LS alphas (gold standard)...")
newton_alphas = np.zeros(n_sample)
for idx, env_i in enumerate(sample_envs):
    newton_alphas[idx] = iterative_newton_ls(env_i)

# Also compute exhaustive search to check if Newton is optimal
print("Computing exhaustive search (100K points)...")
exhaust_alphas = np.zeros(n_sample)
for idx, env_i in enumerate(sample_envs):
    cands = captured['candidates'][:, env_i]
    lo, hi = cands[2], cands[3]
    test_alphas = np.logspace(np.log10(max(lo*0.1, 1e-10)), np.log10(hi*10), 100000)
    costs = eval_cost_batch(test_alphas, env_i)
    exhaust_alphas[idx] = test_alphas[np.argmin(costs)]

# Compare Newton vs exhaustive
newton_vs_exhaust = np.abs(newton_alphas - exhaust_alphas) / np.maximum(np.abs(exhaust_alphas), 1e-30)
valid = exhaust_alphas > 1e-12
print(f"\nIterative Newton vs exhaustive search:")
print(f"  median error: {np.median(newton_vs_exhaust[valid]):.4e}")
print(f"  mean error:   {np.mean(newton_vs_exhaust[valid]):.4e}")
print(f"  p90 error:    {np.percentile(newton_vs_exhaust[valid], 90):.4e}")
print(f"  exact (<1e-6): {(newton_vs_exhaust[valid] < 1e-6).mean()*100:.1f}%")

# 2. Compute parallel LS errors vs iterative Newton
print(f"\n{'='*80}")
print("Parallel LS error vs ITERATIVE NEWTON LS (gold standard)")
print(f"{'='*80}\n")

print("WITH Newton initial best:")
print(f"{'K':>4s} {'N_REF':>5s} | {'median':>10s} {'mean':>10s} {'p90':>10s} {'p99':>10s} {'exact%':>7s}")
print("-" * 62)

for K in [4, 8, 16, 32, 64]:
    for N in [1, 3, 5, 8]:
        errors = []
        for idx, env_i in enumerate(sample_envs):
            alpha_par = parallel_ls_cpu(env_i, K, N, use_newton_init=True)
            alpha_ref = newton_alphas[idx]
            if abs(alpha_ref) > 1e-12:
                errors.append(abs(alpha_par - alpha_ref) / abs(alpha_ref))
            elif abs(alpha_par) < 1e-12:
                errors.append(0.0)
            else:
                errors.append(abs(alpha_par))
        errors = np.array(errors)
        exact = (errors < 1e-6).mean() * 100
        print(f"{K:4d} {N:5d} | {np.median(errors):10.4e} {np.mean(errors):10.4e} {np.percentile(errors, 90):10.4e} {np.percentile(errors, 99):10.4e} {exact:6.1f}%")
    print()

print("\nWITHOUT Newton initial (grid only):")
print(f"{'K':>4s} {'N_REF':>5s} | {'median':>10s} {'mean':>10s} {'p90':>10s} {'p99':>10s} {'exact%':>7s}")
print("-" * 62)

for K in [8, 16, 32, 64]:
    for N in [1, 3, 5, 8]:
        errors = []
        for idx, env_i in enumerate(sample_envs):
            alpha_par = parallel_ls_cpu(env_i, K, N, use_newton_init=False)
            alpha_ref = newton_alphas[idx]
            if abs(alpha_ref) > 1e-12:
                errors.append(abs(alpha_par - alpha_ref) / abs(alpha_ref))
            elif abs(alpha_par) < 1e-12:
                errors.append(0.0)
            else:
                errors.append(abs(alpha_par))
        errors = np.array(errors)
        exact = (errors < 1e-6).mean() * 100
        print(f"{K:4d} {N:5d} | {np.median(errors):10.4e} {np.mean(errors):10.4e} {np.percentile(errors, 90):10.4e} {np.percentile(errors, 99):10.4e} {exact:6.1f}%")
    print()

# Newton-step-only (from p0, no grid)
newton_step_errors = []
for idx, env_i in enumerate(sample_envs):
    cands = captured['candidates'][:, env_i]
    alpha_step = cands[0]  # Newton step from p0
    alpha_ref = newton_alphas[idx]
    if abs(alpha_ref) > 1e-12:
        newton_step_errors.append(abs(alpha_step - alpha_ref) / abs(alpha_ref))
    elif abs(alpha_step) < 1e-12:
        newton_step_errors.append(0.0)
    else:
        newton_step_errors.append(abs(alpha_step))
newton_step_errors = np.array(newton_step_errors)
print(f"Newton step only (p0, no grid): median={np.median(newton_step_errors):.4e} mean={np.mean(newton_step_errors):.4e} p90={np.percentile(newton_step_errors, 90):.4e} exact={((newton_step_errors < 1e-6).mean()*100):.1f}%")
