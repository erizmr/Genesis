"""Check: does the quadratic Newton cost underestimate the actual cost?"""
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
    ), show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
from tests.test_rigid_benchmarks import get_hf_dataset, get_file_morph_options
asset_path = get_hf_dataset(pattern="unitree_g1/*")
robot = scene.add_entity(gs.morphs.MJCF(**get_file_morph_options(
    file=f"{asset_path}/unitree_g1/g1_29dof_rev_1_0.xml", pos=(0,0,1.0),
)), vis_mode="collision")
scene.build(n_envs=4096)
init_qpos = torch.zeros((robot.n_qs,), dtype=gs.tc_float, device=gs.device)
init_qpos[2] = 1.0; init_qpos[3] = 1.0
robot.set_qpos(init_qpos)
rf = torch.zeros((4096, robot.n_dofs), dtype=gs.tc_float, device=gs.device)
for i in range(275):
    rf.uniform_(-50, 50); robot.control_dofs_force(rf); scene.step()

# Capture after p0
from genesis.engine.solvers.rigid.constraint import solver_breakdown
cap = {}
_orig = solver_breakdown._kernel_parallel_linesearch_eval
_c = [0]
def _hook(*a, **k):
    _c[0] += 1
    if _c[0] == 1:
        qd.sync(); cs = a[0]
        cap['cands'] = cs.candidates.to_numpy().copy()
        cap['qg'] = cs.quad_gauss.to_numpy().copy()
        cap['eq'] = cs.eq_sum.to_numpy().copy()
        cap['Ja'] = cs.Jaref.to_numpy().copy()
        cap['jv'] = cs.jv.to_numpy().copy()
        cap['D'] = cs.efc_D.to_numpy().copy()
        cap['nc'] = cs.n_constraints.to_numpy().copy()
        cap['ne'] = cs.n_constraints_equality.to_numpy().copy()
        cap['nf'] = cs.n_constraints_frictionloss.to_numpy().copy()
        cap['fl'] = cs.efc_frictionloss.to_numpy().copy()
        cap['dg'] = cs.diag.to_numpy().copy()
    _orig(*a, **k)
solver_breakdown._kernel_parallel_linesearch_eval = _hook
rf.uniform_(-50, 50); robot.control_dofs_force(rf); scene.step(); qd.sync()
solver_breakdown._kernel_parallel_linesearch_eval = _orig

def actual_cost(alpha, ei):
    qg = cap['qg'][:, ei]; eq = cap['eq'][:, ei]
    ne = int(cap['ne'][ei]); nef = ne + int(cap['nf'][ei]); nc = int(cap['nc'][ei])
    c = alpha**2*qg[2] + alpha*qg[1] + qg[0] + alpha**2*eq[2] + alpha*eq[1] + eq[0]
    for ic in range(ne, nef):
        Ja = cap['Ja'][ic,ei]; jv = cap['jv'][ic,ei]; D = cap['D'][ic,ei]
        f = cap['fl'][ic,ei]; r = cap['dg'][ic,ei]; x = Ja+alpha*jv; rf_ = r*f
        if x <= -rf_: c += f*(-0.5*rf_-Ja-alpha*jv)
        elif x >= rf_: c += f*(-0.5*rf_+Ja+alpha*jv)
        else: c += D*0.5*x*x
    for ic in range(nef, nc):
        Ja = cap['Ja'][ic,ei]; jv = cap['jv'][ic,ei]; D = cap['D'][ic,ei]
        x = Ja+alpha*jv
        if x < 0: c += D*0.5*x*x
    return c

has_con = np.where(cap['nc'] > 0)[0]
under = 0; over = 0; total = 0; diffs = []
for ei in has_con:
    alpha_n = cap['cands'][0, ei]
    quad_cost = cap['cands'][4, ei]
    if alpha_n > 1e-12:
        ac = actual_cost(alpha_n, ei)
        total += 1
        diff = quad_cost - ac
        diffs.append(diff)
        if quad_cost < ac - 1e-10: under += 1
        elif quad_cost > ac + 1e-10: over += 1

diffs = np.array(diffs)
print(f"Envs checked: {total}")
print(f"Quadratic cost < actual (underestimate): {under} ({under/total*100:.1f}%)")
print(f"Quadratic cost > actual (overestimate):  {over} ({over/total*100:.1f}%)")
print(f"Cost difference (quad - actual): median={np.median(diffs):.4e}, mean={np.mean(diffs):.4e}")
print(f"  min={np.min(diffs):.4e}, max={np.max(diffs):.4e}")
