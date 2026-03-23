"""Per-iteration convergence instrumentation.

Monkey-patches the decomposed solver loop to count converged envs after each
solver iteration, for a single simulation step.
"""
import time
import numpy as np
import torch
import quadrants as qd
import genesis as gs

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
    gs.morphs.MJCF(
        **get_file_morph_options(
            file=f"{asset_path}/unitree_g1/g1_29dof_rev_1_0.xml",
            pos=(0, 0, 1.0),
        )
    ),
    vis_mode="collision",
)

n_envs = 4096
scene.build(n_envs=n_envs)

init_qpos = torch.zeros((robot.n_qs,), dtype=gs.tc_float, device=gs.device)
init_qpos[2] = 1.0
init_qpos[3] = 1.0
robot.set_qpos(init_qpos)

random_forces = torch.zeros((n_envs, robot.n_dofs), dtype=gs.tc_float, device=gs.device)
max_force = 50.0

print("Warming up 275 steps...")
for i in range(275):
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()

# Monkey-patch the solver to log per-iteration convergence
from genesis.engine.solvers.rigid.constraint import solver_breakdown

iter_convergence = []  # list of (n_with_constraints, n_still_active) per iteration

_original_search_dir_fn = solver_breakdown._kernel_update_search_direction

def _patched_search_dir(*args, **kwargs):
    _original_search_dir_fn(*args, **kwargs)
    qd.sync()
    cs = args[0]  # constraint_state is first arg
    improved = cs.improved.to_numpy()
    n_con = cs.n_constraints.to_numpy()
    has_con = int((n_con > 0).sum())
    active = int((improved & (n_con > 0)).sum())
    iter_convergence.append((has_con, active))

solver_breakdown._kernel_update_search_direction = _patched_search_dir

# Run a few steps with instrumentation
print("\nPer-iteration convergence for steps 276-280:")
for step_idx in range(5):
    iter_convergence.clear()
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()
    qd.sync()

    print(f"\n  Step {276 + step_idx}: {iter_convergence[0][0] if iter_convergence else '?'} envs with constraints")
    print(f"  {'iter':>4s}  {'active':>7s}  {'converged_this_iter':>20s}  {'cum_converged':>14s}  {'rate':>6s}")
    prev_active = iter_convergence[0][0] if iter_convergence else 0
    total_con = prev_active
    for i, (has_con, active) in enumerate(iter_convergence):
        newly_converged = prev_active - active
        cum_converged = total_con - active
        rate = cum_converged / max(1, total_con) * 100
        print(f"  {i:4d}  {active:7d}  {newly_converged:20d}  {cum_converged:14d}  {rate:5.1f}%")
        prev_active = active

# Restore and measure FPS
solver_breakdown._kernel_update_search_direction = _original_search_dir_fn

print("\nFPS measurement (5 seconds)...")
qd.sync()
t0 = time.time()
n_steps = 0
while time.time() - t0 < 5.0:
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()
    n_steps += 1
qd.sync()
elapsed = time.time() - t0
fps = n_steps * n_envs / elapsed
print(f"FPS: {fps:.0f}")
