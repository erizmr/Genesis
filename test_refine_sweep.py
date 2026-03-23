"""Sweep N_REFINE and measure active-env-iters to find when precision matches main."""
import time, torch, quadrants as qd, genesis as gs

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
max_force = 50.0

# We need to re-compile for each N_REFINE value, but the constant is baked at import.
# Instead, dynamically patch the dispatch loop to call eval N times.
from genesis.engine.solvers.rigid.constraint import solver_breakdown

# Save originals
_orig_solve = solver_breakdown.func_solve_decomposed.__wrapped__
_orig_search = solver_breakdown._kernel_update_search_direction

# We'll patch func_solve_decomposed to use variable N_REFINE
# But the eval kernel is already compiled with block_dim=K, so we just call it more times.

print("Warming up 275 steps with N_REFINE=3...")
for i in range(275):
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()

# Now test different N_REFINE values by patching the dispatch loop
print(f"\n{'N_REFINE':>8s}  {'active_iters':>12s}  {'con_iters':>10s}  {'iter0%':>7s}  {'iter3%':>7s}  {'iter5%':>7s}  {'FPS':>8s}")
print("-" * 75)

# Reference: main has 7298 active-env-iters

for n_refine in [3, 4, 5, 6, 8, 10]:
    # Patch the constant
    solver_breakdown.LS_PARALLEL_N_REFINE = n_refine

    # Since the dispatch loop reads LS_PARALLEL_N_REFINE at call time (Python-level for loop),
    # and the eval kernel is already compiled, we can just change the constant.
    # BUT: the for loop `for _refine in range(LS_PARALLEL_N_REFINE)` is in func_solve_decomposed
    # which is a Python function (not compiled). Let me check...
    # Actually func_solve_decomposed IS a Python function decorated with @register,
    # and the for loop runs at Python level. So changing the constant works!

    # Instrument one step
    iter_data = []
    def _patched_search(*args, **kwargs):
        _orig_search(*args, **kwargs)
        qd.sync()
        cs = args[0]
        improved = cs.improved.to_numpy()
        n_con = cs.n_constraints.to_numpy()
        has_con = n_con > 0
        active = improved & has_con
        iter_data.append((int(has_con.sum()), int(active.sum()), int(n_con[active].sum()) if active.any() else 0))

    solver_breakdown._kernel_update_search_direction = _patched_search

    # Average over 3 steps
    all_active = []
    all_con = []
    all_iter0 = []
    all_iter3 = []
    all_iter5 = []

    for _ in range(3):
        iter_data.clear()
        random_forces.uniform_(-max_force, max_force)
        robot.control_dofs_force(random_forces)
        scene.step()
        qd.sync()

        total_con = iter_data[0][0] if iter_data else 0
        active_sum = sum(d[1] for d in iter_data)
        con_sum = sum(d[2] for d in iter_data)
        iter0_pct = (total_con - iter_data[0][1]) / max(1, total_con) * 100 if iter_data else 0
        iter3_pct = (total_con - iter_data[3][1]) / max(1, total_con) * 100 if len(iter_data) > 3 else 0
        iter5_pct = (total_con - iter_data[5][1]) / max(1, total_con) * 100 if len(iter_data) > 5 else 0
        all_active.append(active_sum)
        all_con.append(con_sum)
        all_iter0.append(iter0_pct)
        all_iter3.append(iter3_pct)
        all_iter5.append(iter5_pct)

    solver_breakdown._kernel_update_search_direction = _orig_search

    # Quick FPS
    qd.sync()
    t0 = time.time(); n = 0
    while time.time() - t0 < 3.0:
        random_forces.uniform_(-max_force, max_force)
        robot.control_dofs_force(random_forces)
        scene.step(); n += 1
    qd.sync()
    fps = n * n_envs / (time.time() - t0)

    avg_active = sum(all_active) / len(all_active)
    avg_con = sum(all_con) / len(all_con)
    avg_i0 = sum(all_iter0) / len(all_iter0)
    avg_i3 = sum(all_iter3) / len(all_iter3)
    avg_i5 = sum(all_iter5) / len(all_iter5)

    print(f"{n_refine:8d}  {avg_active:12.0f}  {avg_con:10.0f}  {avg_i0:6.1f}%  {avg_i3:6.1f}%  {avg_i5:6.1f}%  {fps:8.0f}")

# Reset
solver_breakdown.LS_PARALLEL_N_REFINE = 3
print(f"\nMain reference: active_iters=7298, iter0=15.3%, iter3=88.7%, iter5=99.4%")
