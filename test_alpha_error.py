"""
Measure parallel linesearch alpha error vs gold standard (fine-grid optimum),
as a function of N_REFINE and K (fan-out).

For a single solver iteration at step 276, captures the constraint state
after mv/jv/p0 are computed, then runs the parallel grid search on CPU
with various (K, N_REFINE) settings. Compares against a 100,000-point
exhaustive search as the gold standard.
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

# Patch solver to capture state after p0 (before eval) on the first iteration only
from genesis.engine.solvers.rigid.constraint import solver_breakdown

captured = {}
_orig_eval = solver_breakdown._kernel_parallel_linesearch_eval
_call_count = [0]

def _capture_eval(*args, **kwargs):
    _call_count[0] += 1
    if _call_count[0] == 1:
        # First eval call of first iteration — capture state
        qd.sync()
        cs = args[0]  # constraint_state
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
    _orig_eval(*args, **kwargs)

solver_breakdown._kernel_parallel_linesearch_eval = _capture_eval

# Run one step to capture
random_forces.uniform_(-50, 50)
robot.control_dofs_force(random_forces)
scene.step()
qd.sync()

# Also capture the actual alpha chosen by the solver
cs = scene.sim.rigid_solver.constraint_solver.constraint_state
actual_alpha = cs.candidates.to_numpy()[0, :]

solver_breakdown._kernel_parallel_linesearch_eval = _orig_eval

print(f"Captured state for {(captured['n_constraints'] > 0).sum()} envs with constraints")


# === CPU-side cost evaluation ===
def eval_cost_batch(alphas, env_idx):
    """Evaluate cost at array of alphas for one env. Returns array of costs."""
    qg = captured['quad_gauss'][:, env_idx]
    eq = captured['eq_sum'][:, env_idx]
    ne = int(captured['n_constraints_equality'][env_idx])
    nef = ne + int(captured['n_constraints_frictionloss'][env_idx])
    n_con = int(captured['n_constraints'][env_idx])

    a = np.asarray(alphas, dtype=np.float64)
    # Gauss + equality (quadratic in alpha)
    costs = a*a*qg[2] + a*qg[1] + qg[0] + a*a*eq[2] + a*eq[1] + eq[0]

    # Friction constraints
    for ic in range(ne, nef):
        Ja = captured['Jaref'][ic, env_idx]
        jv = captured['jv'][ic, env_idx]
        D = captured['efc_D'][ic, env_idx]
        f = captured['efc_frictionloss'][ic, env_idx]
        r = captured['diag'][ic, env_idx]
        x = Ja + a * jv
        rf = r * f
        ln = x <= -rf
        lp = x >= rf
        mid = ~(ln | lp)
        costs += ln * f * (-0.5*rf - Ja - a*jv) + lp * f * (-0.5*rf + Ja + a*jv) + mid * D * 0.5 * x**2

    # Contact constraints
    for ic in range(nef, n_con):
        Ja = captured['Jaref'][ic, env_idx]
        jv = captured['jv'][ic, env_idx]
        D = captured['efc_D'][ic, env_idx]
        x = Ja + a * jv
        active = x < 0
        costs += active * D * 0.5 * x**2

    return costs


def parallel_ls_cpu(env_idx, K, N_REFINE, use_newton_init=True):
    """Simulate parallel grid search on CPU. Returns best alpha."""
    cands = captured['candidates'][:, env_idx]
    p0_cost = cands[1]
    lo = cands[2]
    hi = cands[3]

    if use_newton_init:
        best_alpha = cands[0]  # Newton initial
        best_cost = cands[4]   # Newton quadratic cost
    else:
        best_alpha = 0.0
        best_cost = 1e30

    for _r in range(N_REFINE):
        log_lo = np.log(max(lo, 1e-30))
        log_hi = np.log(max(hi, 1e-30))
        step = (log_hi - log_lo) / max(1, K - 1)
        alphas = np.exp(log_lo + np.arange(K) * step)
        costs = eval_cost_batch(alphas, env_idx)

        best_idx = np.argmin(costs)
        if costs[best_idx] < p0_cost and costs[best_idx] < best_cost:
            best_alpha = alphas[best_idx]
            best_cost = costs[best_idx]
            lo_idx = max(0, best_idx - 1)
            hi_idx = min(K - 1, best_idx + 1)
            lo = np.exp(log_lo + lo_idx * step)
            hi = np.exp(log_lo + hi_idx * step)

    return best_alpha


# === Find gold standard: exhaustive search with 100K points ===
print("Computing gold standard (100K-point exhaustive search)...")
n_sample = 500  # sample envs for speed
has_con = np.where(captured['n_constraints'] > 0)[0]
sample_envs = has_con[np.linspace(0, len(has_con)-1, n_sample, dtype=int)]

gold_alphas = np.zeros(n_sample)
for idx, env_i in enumerate(sample_envs):
    cands = captured['candidates'][:, env_i]
    lo, hi = cands[2], cands[3]
    # Search over wider range than the grid to find true global min
    test_alphas = np.logspace(np.log10(max(lo*0.1, 1e-10)), np.log10(hi*10), 100000)
    costs = eval_cost_batch(test_alphas, env_i)
    gold_alphas[idx] = test_alphas[np.argmin(costs)]

print(f"Gold standard computed for {n_sample} envs")


# === Sweep K and N_REFINE ===
print("\n=== Alpha error vs gold standard ===\n")

K_values = [4, 8, 16, 32, 64]
N_values = [1, 2, 3, 5, 8]

# With Newton initial
print("WITH Newton initial best:")
print(f"{'K':>4s} {'N_REF':>5s} | {'median_err':>10s} {'mean_err':>10s} {'p90_err':>10s} {'p99_err':>10s} {'exact%':>7s}")
print("-" * 65)

for K in K_values:
    for N in N_values:
        errors = []
        for idx, env_i in enumerate(sample_envs):
            alpha_par = parallel_ls_cpu(env_i, K, N, use_newton_init=True)
            alpha_gold = gold_alphas[idx]
            if abs(alpha_gold) > 1e-12:
                rel_err = abs(alpha_par - alpha_gold) / abs(alpha_gold)
            else:
                rel_err = abs(alpha_par)
            errors.append(rel_err)
        errors = np.array(errors)
        exact = (errors < 1e-6).mean() * 100
        print(f"{K:4d} {N:5d} | {np.median(errors):10.4e} {np.mean(errors):10.4e} {np.percentile(errors, 90):10.4e} {np.percentile(errors, 99):10.4e} {exact:6.1f}%")
    print()

# Without Newton initial
print("\nWITHOUT Newton initial (grid only):")
print(f"{'K':>4s} {'N_REF':>5s} | {'median_err':>10s} {'mean_err':>10s} {'p90_err':>10s} {'p99_err':>10s} {'exact%':>7s}")
print("-" * 65)

for K in [8, 16, 32, 64]:
    for N in [1, 3, 5, 8]:
        errors = []
        for idx, env_i in enumerate(sample_envs):
            alpha_par = parallel_ls_cpu(env_i, K, N, use_newton_init=False)
            alpha_gold = gold_alphas[idx]
            if abs(alpha_gold) > 1e-12:
                rel_err = abs(alpha_par - alpha_gold) / abs(alpha_gold)
            else:
                rel_err = abs(alpha_par)
            errors.append(rel_err)
        errors = np.array(errors)
        exact = (errors < 1e-6).mean() * 100
        print(f"{K:4d} {N:5d} | {np.median(errors):10.4e} {np.mean(errors):10.4e} {np.percentile(errors, 90):10.4e} {np.percentile(errors, 99):10.4e} {exact:6.1f}%")
    print()

# Newton-only (no grid at all)
errors_newton = []
for idx, env_i in enumerate(sample_envs):
    cands = captured['candidates'][:, env_i]
    alpha_newton = cands[0]  # Newton initial
    alpha_gold = gold_alphas[idx]
    if abs(alpha_gold) > 1e-12:
        rel_err = abs(alpha_newton - alpha_gold) / abs(alpha_gold)
    else:
        rel_err = abs(alpha_newton)
    errors_newton.append(rel_err)
errors_newton = np.array(errors_newton)
print(f"\nNewton-only (no grid): median={np.median(errors_newton):.4e} mean={np.mean(errors_newton):.4e} p90={np.percentile(errors_newton, 90):.4e} exact={((errors_newton < 1e-6).mean()*100):.1f}%")
