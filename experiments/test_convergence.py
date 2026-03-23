"""Measure per-iteration convergence of the constraint solver.

Instruments the decomposed solver loop to count how many envs remain active
after each solver iteration, for a single simulation step.
"""
import time
import torch
import quadrants as qd
import genesis as gs

gs.init(backend=gs.cuda, precision="32", logging_level="warning")

# Match g1_fall benchmark setup exactly
scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        dt=0.005,
        iterations=10,
        tolerance=1e-5,
        ls_iterations=20,
        constraint_solver=gs.constraint_solver.Newton,
    ),
    show_viewer=False,
    show_FPS=False,
)

scene.add_entity(gs.morphs.Plane())

# Use the same asset as the benchmark
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

# Set initial position
init_qpos = torch.zeros((robot.n_qs,), dtype=gs.tc_float, device=gs.device)
init_qpos[2] = 1.0
init_qpos[3] = 1.0
robot.set_qpos(init_qpos)

random_forces = torch.zeros((n_envs, robot.n_dofs), dtype=gs.tc_float, device=gs.device)
max_force = 50.0

# Warm up for 275 steps (matching profiler wait)
print("Warming up 275 steps...")
for i in range(275):
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()

# Now monkey-patch the solver to instrument convergence
from genesis.engine.solvers.rigid.constraint import solver_breakdown, solver

original_func = solver_breakdown.func_solve_decomposed.__wrapped__ if hasattr(solver_breakdown.func_solve_decomposed, '__wrapped__') else None

convergence_log = []

# Patch the search direction kernel to log convergence after each iteration
original_search_dir = solver_breakdown._kernel_update_search_direction

def instrumented_solve():
    """Run one step and read improved flags after the solver."""
    random_forces.uniform_(-max_force, max_force)
    robot.control_dofs_force(random_forces)
    scene.step()
    qd.sync()

    # Read post-solver state
    cs = scene.sim.rigid_solver.constraint_solver.constraint_state
    improved = cs.improved.to_numpy()
    n_con = cs.n_constraints.to_numpy()

    has_constraints = (n_con > 0).sum().item()
    still_active = (improved & (n_con > 0)).sum().item()
    converged = has_constraints - still_active

    return has_constraints, converged, still_active

# Run several steps and collect convergence stats
print("\nCollecting convergence stats over 20 steps (steps 276-295)...")
print(f"{'step':>5s}  {'w/constraints':>13s}  {'converged':>10s}  {'active':>7s}  {'rate':>6s}")
print("-" * 50)

total_constrained = 0
total_converged = 0

for step in range(20):
    has_con, converged, active = instrumented_solve()
    rate = converged / max(1, has_con) * 100
    print(f"{276+step:5d}  {has_con:13d}  {converged:10d}  {active:7d}  {rate:5.1f}%")
    total_constrained += has_con
    total_converged += converged

avg_rate = total_converged / max(1, total_constrained) * 100
print("-" * 50)
print(f"Average convergence rate: {avg_rate:.1f}%")

# Also run a quick FPS measurement
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
