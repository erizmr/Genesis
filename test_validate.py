"""Quick validation: per-iteration convergence (1 step) + FPS."""
import time, torch, quadrants as qd, genesis as gs

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
max_force = 50.0

print("Warming up 275 steps...")
for i in range(275):
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()

# Instrument 1 step
from genesis.engine.solvers.rigid.constraint import solver_breakdown
iter_data = []
_orig = solver_breakdown._kernel_update_search_direction
def _patched(*args, **kwargs):
    _orig(*args, **kwargs); qd.sync()
    cs = args[0]
    improved = cs.improved.to_numpy(); n_con = cs.n_constraints.to_numpy()
    has_con = n_con > 0; active = improved & has_con
    iter_data.append((int(has_con.sum()), int(active.sum()), int(n_con[active].sum()) if active.any() else 0))
solver_breakdown._kernel_update_search_direction = _patched

random_forces.uniform_(-max_force, max_force)
robot.control_dofs_force(random_forces)
scene.step(); qd.sync()

print(f"\nStep 276: {iter_data[0][0]} envs with constraints")
print(f"{'iter':>4s}  {'active':>7s}  {'cum%':>6s}  {'con_iters':>10s}")
total_con = iter_data[0][0]; total_active = 0; total_con_iters = 0
for i, (has_con, active, con_iters) in enumerate(iter_data):
    cum = (total_con - active) / max(1, total_con) * 100
    total_active += active; total_con_iters += con_iters
    print(f"{i:4d}  {active:7d}  {cum:5.1f}%  {con_iters:10d}")
print(f"Total active-env-iters: {total_active}, constraint-iters: {total_con_iters}")

solver_breakdown._kernel_update_search_direction = _orig

# FPS
print("\nFPS (5s)...")
qd.sync(); t0 = time.time(); n = 0
while time.time() - t0 < 5.0:
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step(); n += 1
qd.sync()
print(f"FPS: {n * n_envs / (time.time() - t0):.0f}")
