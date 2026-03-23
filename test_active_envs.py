"""Count active envs AND n_constraints at each solver iteration for a single step.

This tells us the actual workload the hessian/gradient kernels see.
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

# Instrument
from genesis.engine.solvers.rigid.constraint import solver_breakdown

iter_data = []
_orig = solver_breakdown._kernel_update_search_direction

def _patched(*args, **kwargs):
    _orig(*args, **kwargs)
    qd.sync()
    cs = args[0]
    improved = cs.improved.to_numpy()
    n_con = cs.n_constraints.to_numpy()
    # Count active envs and total constraints for active envs
    has_con = n_con > 0
    active = improved & has_con
    n_active = int(active.sum())
    # Total constraints across active envs (= hessian workload)
    total_constraints_active = int(n_con[active].sum()) if n_active > 0 else 0
    # Total constraints across all constrained envs
    total_constraints_all = int(n_con[has_con].sum())
    n_constrained = int(has_con.sum())
    iter_data.append({
        'n_constrained': n_constrained,
        'n_active': n_active,
        'total_con_active': total_constraints_active,
        'total_con_all': total_constraints_all,
    })

solver_breakdown._kernel_update_search_direction = _patched

# Run 5 steps, average the per-iteration active env counts
all_steps = []
for step_idx in range(5):
    iter_data.clear()
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()
    qd.sync()
    all_steps.append(list(iter_data))

# Print average across 5 steps
print(f"\nAverage per-iteration stats across 5 steps (276-280):")
print(f"{'iter':>4s}  {'active':>7s}  {'%active':>8s}  {'con/active':>11s}  {'total_con':>10s}")
print(f"{'-'*4}  {'-'*7}  {'-'*8}  {'-'*11}  {'-'*10}")

total_active_iters = 0
total_con_iters = 0

for i in range(10):
    n_active = np.mean([s[i]['n_active'] for s in all_steps])
    n_constrained = np.mean([s[i]['n_constrained'] for s in all_steps])
    con_active = np.mean([s[i]['total_con_active'] for s in all_steps])
    pct = n_active / max(1, n_constrained) * 100
    avg_con = con_active / max(1, n_active)
    total_active_iters += n_active
    total_con_iters += con_active
    print(f"{i:4d}  {n_active:7.0f}  {pct:7.1f}%  {avg_con:11.1f}  {con_active:10.0f}")

print(f"\nTotal active-env-iterations: {total_active_iters:.0f}")
print(f"Total constraint-iterations: {total_con_iters:.0f}")

solver_breakdown._kernel_update_search_direction = _orig
