import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.constraint_solver as constraint_solver


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_solve_body_decomposed(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Single kernel containing all solver steps as separate top-level loops.

    This reduces Python→C++ boundary crossing overhead (1 call per iteration instead of 6)
    while still allowing Taichi to launch each step as a separate GPU kernel internally.
    """
    _B = constraint_state.grad.shape[1]

    # Step 1: Linesearch and update qacc, Ma, Jaref
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 0
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_linesearch_top_level(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        else:
            constraint_state.improved[i_b] = False

    # Step 2: Save prev_grad and prev_Mgrad (CG only)
    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        # Index: 1 if CG
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                constraint_solver.func_save_prev_grad(i_b, constraint_state=constraint_state)

    # Step 3: Update constraints
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 1 if Newton else 2
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_constraint(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

    # Step 4: Newton Hessian update (Newton only)
    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        # Index: 2 if Newton
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                constraint_solver.func_nt_hessian_incremental(
                    i_b,
                    entities_info=entities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

    # Step 5: Update gradient
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 3
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_gradient(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

    # Step 6: Check convergence and update search direction
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 4
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_search_direction(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


def func_solve_decomposed_microkernels(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Uses a single kernel with multiple top-level for loops per iteration, reducing
    Python→C++ boundary crossing overhead from 6× to 1× per iteration.
    """
    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        _kernel_solve_body_decomposed(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_linesearch_top_level(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            print("I am using line search")
            constraint_solver.func_linesearch_top_level(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        else:
            constraint_state.improved[i_b] = False


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_cg_only_save_prev_grad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Save prev_grad and prev_Mgrad (CG only)"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_save_prev_grad(i_b, constraint_state=constraint_state)


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_constraint(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_newton_only_nt_hessian_incremental(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Step 4: Newton Hessian update (Newton only)"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_nt_hessian_incremental(
                i_b,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_gradient(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Step 5: Update gradient"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_gradient(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_search_direction(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Step 6: Check convergence and update search direction"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_search_direction(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


def func_solve_decomposed_macrokernels(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Uses separate kernels for each solver step per iteration.

    This maximizes kernel granularity, potentially allowing better GPU scheduling
    and more flexibility in execution, at the cost of more Python→C++ boundary crossings
    (6× per iteration instead of 1×).
    """
    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        _kernel_linesearch_top_level(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if static_rigid_sim_config.solver_type == gs.constraint_solver.CG:
            _kernel_cg_only_save_prev_grad(
                constraint_state,
                static_rigid_sim_config,
            )
        _kernel_update_constraint(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
            _kernel_newton_only_nt_hessian_incremental(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
        _kernel_update_gradient(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        _kernel_update_search_direction(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )


# =============================================================================
# Strategy B: Shared Memory Line Search Implementation
# =============================================================================
# This implementation caches constraint data (Jaref, jv, quad, etc.) in shared
# memory to reduce global memory bandwidth during line search iterations.
# NOTE: Shared memory (ti.simt.block.SharedArray) is only supported on CUDA GPUs.
# On other backends (CPU, Metal, Vulkan), this will fall back to the macro kernel.
# =============================================================================


def _is_shared_mem_supported():
    """Check if shared memory is supported on the current backend."""
    # Shared memory (ti.simt.block.SharedArray) is only supported on CUDA GPUs
    # gs.backend values: cpu=0, gpu=1, cuda=2, metal=4
    from genesis.constants import backend as gs_backend

    return gs.backend == gs_backend.cuda


@ti.func
def _func_ls_point_cached(
    alpha: gs.ti_float,
    n_constraints: ti.i32,
    ne: ti.i32,
    nef: ti.i32,
    quad_gauss_0: gs.ti_float,
    quad_gauss_1: gs.ti_float,
    quad_gauss_2: gs.ti_float,
    # Shared memory arrays (passed as template parameters)
    cached_Jaref: ti.template(),
    cached_jv: ti.template(),
    cached_quad_0: ti.template(),
    cached_quad_1: ti.template(),
    cached_quad_2: ti.template(),
    cached_frictionloss: ti.template(),
    cached_diag: ti.template(),
    EPS: gs.ti_float,
):
    """
    Evaluate line search cost function using cached constraint data in shared memory.
    This avoids repeated global memory accesses during line search iterations.
    """
    tmp_quad_total_0 = quad_gauss_0
    tmp_quad_total_1 = quad_gauss_1
    tmp_quad_total_2 = quad_gauss_2

    for i_c in range(n_constraints):
        x = cached_Jaref[i_c] + alpha * cached_jv[i_c]
        qf_0 = cached_quad_0[i_c]
        qf_1 = cached_quad_1[i_c]
        qf_2 = cached_quad_2[i_c]

        active = gs.ti_bool(True)  # Equality constraints are always active
        if ne <= i_c and i_c < nef:  # Friction constraints
            f = cached_frictionloss[i_c]
            r = cached_diag[i_c]
            rf = r * f
            linear_neg = x <= -rf
            linear_pos = x >= rf

            if linear_neg or linear_pos:
                qf_0 = linear_neg * f * (-0.5 * rf - cached_Jaref[i_c]) + linear_pos * f * (
                    -0.5 * rf + cached_Jaref[i_c]
                )
                qf_1 = linear_neg * (-f * cached_jv[i_c]) + linear_pos * (f * cached_jv[i_c])
                qf_2 = 0.0
        elif nef <= i_c:  # Contact constraints
            active = x < 0

        tmp_quad_total_0 = tmp_quad_total_0 + qf_0 * active
        tmp_quad_total_1 = tmp_quad_total_1 + qf_1 * active
        tmp_quad_total_2 = tmp_quad_total_2 + qf_2 * active

    cost = alpha * alpha * tmp_quad_total_2 + alpha * tmp_quad_total_1 + tmp_quad_total_0

    deriv_0 = 2 * alpha * tmp_quad_total_2 + tmp_quad_total_1
    deriv_1 = 2 * tmp_quad_total_2
    if deriv_1 <= 0.0:
        deriv_1 = EPS

    return cost, deriv_0, deriv_1


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_linesearch_shared_mem(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Line search kernel with shared memory caching for constraint data.

    Strategy B optimization: Caches Jaref, jv, quad, frictionloss, and diag
    in shared memory to reduce global memory bandwidth during iterative
    line search evaluations.

    Note: MAX_TILE_SIZE is set to 64 to balance shared memory usage and
    occupancy. Adjust based on GPU shared memory limits (typically 48KB-96KB).
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.search.shape[0]

    # Maximum constraints that can be cached in shared memory per tile
    # Set to 64 to balance shared memory usage and GPU occupancy
    MAX_TILE_SIZE = ti.static(64)

    # Use block_dim matching constraint tile size for efficient loading
    BLOCK_DIM = ti.static(64)

    ti.loop_config(block_dim=BLOCK_DIM)
    for idx in range(_B * BLOCK_DIM):
        tid = idx % BLOCK_DIM
        i_b = idx // BLOCK_DIM

        if i_b >= _B:
            continue

        n_constraints = constraint_state.n_constraints[i_b]

        # Early exit conditions
        if n_constraints == 0 or not constraint_state.improved[i_b]:
            if tid == 0:
                constraint_state.improved[i_b] = False
            continue

        # Allocate shared memory for constraint data caching
        cached_Jaref = ti.simt.block.SharedArray((MAX_TILE_SIZE,), gs.ti_float)
        cached_jv = ti.simt.block.SharedArray((MAX_TILE_SIZE,), gs.ti_float)
        cached_quad_0 = ti.simt.block.SharedArray((MAX_TILE_SIZE,), gs.ti_float)
        cached_quad_1 = ti.simt.block.SharedArray((MAX_TILE_SIZE,), gs.ti_float)
        cached_quad_2 = ti.simt.block.SharedArray((MAX_TILE_SIZE,), gs.ti_float)
        cached_frictionloss = ti.simt.block.SharedArray((MAX_TILE_SIZE,), gs.ti_float)
        cached_diag = ti.simt.block.SharedArray((MAX_TILE_SIZE,), gs.ti_float)

        # Thread 0 performs the line search initialization and main computation
        # Other threads help with data loading

        # ===== Step 1: Compute search norm and tolerance (thread 0) =====
        snorm = gs.ti_float(0.0)
        gtol = gs.ti_float(0.0)
        if tid == 0:
            for jd in range(n_dofs):
                snorm = snorm + constraint_state.search[jd, i_b] ** 2
            snorm = ti.sqrt(snorm)
            scale = rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs)
            gtol = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
            constraint_state.gtol[i_b] = gtol
            constraint_state.ls_it[i_b] = 0
            constraint_state.ls_result[i_b] = 0

        ti.simt.block.sync()

        # Early exit if search direction is negligible
        # (Note: snorm is computed by thread 0, we need to broadcast or recompute)
        # For simplicity, thread 0 will set a flag
        should_exit = ti.simt.block.SharedArray((1,), gs.ti_int)
        res_alpha_shared = ti.simt.block.SharedArray((1,), gs.ti_float)

        if tid == 0:
            # Recompute snorm for thread 0's use
            snorm_local = gs.ti_float(0.0)
            for jd in range(n_dofs):
                snorm_local = snorm_local + constraint_state.search[jd, i_b] ** 2
            snorm_local = ti.sqrt(snorm_local)

            if snorm_local < rigid_global_info.EPS[None]:
                constraint_state.ls_result[i_b] = 1
                res_alpha_shared[0] = 0.0
                should_exit[0] = 1
            else:
                should_exit[0] = 0

        ti.simt.block.sync()

        if should_exit[0] == 1:
            if tid == 0:
                alpha = res_alpha_shared[0]
                if ti.abs(alpha) < rigid_global_info.EPS[None]:
                    constraint_state.improved[i_b] = False
            continue

        # ===== Step 2: Initialize line search (func_ls_init equivalent) =====
        # Thread 0 computes mv, jv, quad_gauss, and quad arrays
        if tid == 0:
            constraint_solver.func_ls_init(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )

        ti.simt.block.sync()

        # ===== Step 3: Load constraint data into shared memory (parallel) =====
        # Each thread loads a subset of constraints
        i_c = tid
        while i_c < n_constraints and i_c < MAX_TILE_SIZE:
            cached_Jaref[i_c] = constraint_state.Jaref[i_c, i_b]
            cached_jv[i_c] = constraint_state.jv[i_c, i_b]
            cached_quad_0[i_c] = constraint_state.quad[i_c, 0, i_b]
            cached_quad_1[i_c] = constraint_state.quad[i_c, 1, i_b]
            cached_quad_2[i_c] = constraint_state.quad[i_c, 2, i_b]
            cached_frictionloss[i_c] = constraint_state.efc_frictionloss[i_c, i_b]
            cached_diag[i_c] = constraint_state.diag[i_c, i_b]
            i_c = i_c + BLOCK_DIM

        ti.simt.block.sync()

        # ===== Step 4: Perform line search using cached data (thread 0) =====
        if tid == 0:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_constr_clamped = ti.min(n_constraints, MAX_TILE_SIZE)

            quad_gauss_0 = constraint_state.quad_gauss[0, i_b]
            quad_gauss_1 = constraint_state.quad_gauss[1, i_b]
            quad_gauss_2 = constraint_state.quad_gauss[2, i_b]

            EPS = rigid_global_info.EPS[None]
            ls_iterations = rigid_global_info.ls_iterations[None]

            # Recompute gtol locally
            snorm_local = gs.ti_float(0.0)
            for jd in range(n_dofs):
                snorm_local = snorm_local + constraint_state.search[jd, i_b] ** 2
            snorm_local = ti.sqrt(snorm_local)
            scale = rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs)
            gtol_local = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm_local * scale

            ls_it_count = 0

            # Initial point evaluation at alpha = 0
            p0_cost, p0_deriv_0, p0_deriv_1 = _func_ls_point_cached(
                gs.ti_float(0.0),
                n_constr_clamped,
                ne,
                nef,
                quad_gauss_0,
                quad_gauss_1,
                quad_gauss_2,
                cached_Jaref,
                cached_jv,
                cached_quad_0,
                cached_quad_1,
                cached_quad_2,
                cached_frictionloss,
                cached_diag,
                EPS,
            )
            p0_alpha = gs.ti_float(0.0)
            ls_it_count += 1

            # Newton step from p0
            p1_alpha = p0_alpha - p0_deriv_0 / p0_deriv_1
            p1_cost, p1_deriv_0, p1_deriv_1 = _func_ls_point_cached(
                p1_alpha,
                n_constr_clamped,
                ne,
                nef,
                quad_gauss_0,
                quad_gauss_1,
                quad_gauss_2,
                cached_Jaref,
                cached_jv,
                cached_quad_0,
                cached_quad_1,
                cached_quad_2,
                cached_frictionloss,
                cached_diag,
                EPS,
            )
            ls_it_count += 1

            # If p0 is better, reset to p0
            if p0_cost < p1_cost:
                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1

            res_alpha = gs.ti_float(0.0)
            done = False

            if ti.abs(p1_deriv_0) < gtol_local:
                if ti.abs(p1_alpha) < EPS:
                    constraint_state.ls_result[i_b] = 2
                else:
                    constraint_state.ls_result[i_b] = 0
                res_alpha = p1_alpha
                done = True
            else:
                direction = (p1_deriv_0 < 0) * 2 - 1
                p2update = 0
                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1

                # Bracketing phase
                while p1_deriv_0 * direction <= -gtol_local and ls_it_count < ls_iterations:
                    p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                    p2update = 1

                    new_alpha = p1_alpha - p1_deriv_0 / p1_deriv_1
                    p1_cost, p1_deriv_0, p1_deriv_1 = _func_ls_point_cached(
                        new_alpha,
                        n_constr_clamped,
                        ne,
                        nef,
                        quad_gauss_0,
                        quad_gauss_1,
                        quad_gauss_2,
                        cached_Jaref,
                        cached_jv,
                        cached_quad_0,
                        cached_quad_1,
                        cached_quad_2,
                        cached_frictionloss,
                        cached_diag,
                        EPS,
                    )
                    p1_alpha = new_alpha
                    ls_it_count += 1

                    if ti.abs(p1_deriv_0) < gtol_local:
                        res_alpha = p1_alpha
                        done = True
                        break

                if not done:
                    if ls_it_count >= ls_iterations:
                        constraint_state.ls_result[i_b] = 3
                        res_alpha = p1_alpha
                        done = True

                    if not p2update and not done:
                        constraint_state.ls_result[i_b] = 6
                        res_alpha = p1_alpha
                        done = True

                    # Refinement phase with bisection
                    if not done:
                        p2_next_alpha = p1_alpha
                        p2_next_cost, p2_next_deriv_0, p2_next_deriv_1 = p1_cost, p1_deriv_0, p1_deriv_1

                        p1_next_alpha = p1_alpha - p1_deriv_0 / p1_deriv_1
                        p1_next_cost, p1_next_deriv_0, p1_next_deriv_1 = _func_ls_point_cached(
                            p1_next_alpha,
                            n_constr_clamped,
                            ne,
                            nef,
                            quad_gauss_0,
                            quad_gauss_1,
                            quad_gauss_2,
                            cached_Jaref,
                            cached_jv,
                            cached_quad_0,
                            cached_quad_1,
                            cached_quad_2,
                            cached_frictionloss,
                            cached_diag,
                            EPS,
                        )
                        ls_it_count += 1

                        while ls_it_count < ls_iterations and not done:
                            # Midpoint
                            pmid_alpha = (p1_alpha + p2_alpha) * 0.5
                            pmid_cost, pmid_deriv_0, pmid_deriv_1 = _func_ls_point_cached(
                                pmid_alpha,
                                n_constr_clamped,
                                ne,
                                nef,
                                quad_gauss_0,
                                quad_gauss_1,
                                quad_gauss_2,
                                cached_Jaref,
                                cached_jv,
                                cached_quad_0,
                                cached_quad_1,
                                cached_quad_2,
                                cached_frictionloss,
                                cached_diag,
                                EPS,
                            )
                            ls_it_count += 1

                            # Store candidates in local variables instead of shared array
                            cand_alpha_0, cand_cost_0, cand_deriv_0 = p1_next_alpha, p1_next_cost, p1_next_deriv_0
                            cand_alpha_1, cand_cost_1, cand_deriv_1 = p2_next_alpha, p2_next_cost, p2_next_deriv_0
                            cand_alpha_2, cand_cost_2, cand_deriv_2 = pmid_alpha, pmid_cost, pmid_deriv_0

                            # Find best candidate that meets tolerance
                            best_alpha = gs.ti_float(0.0)
                            best_cost = gs.ti_float(1e30)
                            found_best = False

                            if ti.abs(cand_deriv_0) < gtol_local and cand_cost_0 < best_cost:
                                best_cost = cand_cost_0
                                best_alpha = cand_alpha_0
                                found_best = True
                            if ti.abs(cand_deriv_1) < gtol_local and cand_cost_1 < best_cost:
                                best_cost = cand_cost_1
                                best_alpha = cand_alpha_1
                                found_best = True
                            if ti.abs(cand_deriv_2) < gtol_local and cand_cost_2 < best_cost:
                                best_cost = cand_cost_2
                                best_alpha = cand_alpha_2
                                found_best = True

                            if found_best:
                                res_alpha = best_alpha
                                done = True
                            else:
                                # Update brackets
                                # Update p1 bracket
                                if p1_deriv_0 < 0:
                                    if cand_deriv_0 < 0 and p1_deriv_0 < cand_deriv_0:
                                        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = (
                                            cand_alpha_0,
                                            cand_cost_0,
                                            cand_deriv_0,
                                            p1_deriv_1,
                                        )
                                    elif cand_deriv_1 < 0 and p1_deriv_0 < cand_deriv_1:
                                        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = (
                                            cand_alpha_1,
                                            cand_cost_1,
                                            cand_deriv_1,
                                            p1_deriv_1,
                                        )
                                    elif cand_deriv_2 < 0 and p1_deriv_0 < cand_deriv_2:
                                        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = (
                                            cand_alpha_2,
                                            cand_cost_2,
                                            cand_deriv_2,
                                            p1_deriv_1,
                                        )
                                elif p1_deriv_0 > 0:
                                    if cand_deriv_0 > 0 and p1_deriv_0 > cand_deriv_0:
                                        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = (
                                            cand_alpha_0,
                                            cand_cost_0,
                                            cand_deriv_0,
                                            p1_deriv_1,
                                        )
                                    elif cand_deriv_1 > 0 and p1_deriv_0 > cand_deriv_1:
                                        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = (
                                            cand_alpha_1,
                                            cand_cost_1,
                                            cand_deriv_1,
                                            p1_deriv_1,
                                        )
                                    elif cand_deriv_2 > 0 and p1_deriv_0 > cand_deriv_2:
                                        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = (
                                            cand_alpha_2,
                                            cand_cost_2,
                                            cand_deriv_2,
                                            p1_deriv_1,
                                        )

                                # Similar update for p2 bracket
                                if p2_deriv_0 < 0:
                                    if cand_deriv_0 < 0 and p2_deriv_0 < cand_deriv_0:
                                        p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = (
                                            cand_alpha_0,
                                            cand_cost_0,
                                            cand_deriv_0,
                                            p2_deriv_1,
                                        )
                                    elif cand_deriv_1 < 0 and p2_deriv_0 < cand_deriv_1:
                                        p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = (
                                            cand_alpha_1,
                                            cand_cost_1,
                                            cand_deriv_1,
                                            p2_deriv_1,
                                        )
                                    elif cand_deriv_2 < 0 and p2_deriv_0 < cand_deriv_2:
                                        p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = (
                                            cand_alpha_2,
                                            cand_cost_2,
                                            cand_deriv_2,
                                            p2_deriv_1,
                                        )
                                elif p2_deriv_0 > 0:
                                    if cand_deriv_0 > 0 and p2_deriv_0 > cand_deriv_0:
                                        p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = (
                                            cand_alpha_0,
                                            cand_cost_0,
                                            cand_deriv_0,
                                            p2_deriv_1,
                                        )
                                    elif cand_deriv_1 > 0 and p2_deriv_0 > cand_deriv_1:
                                        p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = (
                                            cand_alpha_1,
                                            cand_cost_1,
                                            cand_deriv_1,
                                            p2_deriv_1,
                                        )
                                    elif cand_deriv_2 > 0 and p2_deriv_0 > cand_deriv_2:
                                        p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = (
                                            cand_alpha_2,
                                            cand_cost_2,
                                            cand_deriv_2,
                                            p2_deriv_1,
                                        )

                                # Recompute next points
                                p1_next_alpha = p1_alpha - p1_deriv_0 / p1_deriv_1
                                p1_next_cost, p1_next_deriv_0, p1_next_deriv_1 = _func_ls_point_cached(
                                    p1_next_alpha,
                                    n_constr_clamped,
                                    ne,
                                    nef,
                                    quad_gauss_0,
                                    quad_gauss_1,
                                    quad_gauss_2,
                                    cached_Jaref,
                                    cached_jv,
                                    cached_quad_0,
                                    cached_quad_1,
                                    cached_quad_2,
                                    cached_frictionloss,
                                    cached_diag,
                                    EPS,
                                )
                                ls_it_count += 1

                                p2_next_alpha = p2_alpha - p2_deriv_0 / p2_deriv_1
                                p2_next_cost, p2_next_deriv_0, p2_next_deriv_1 = _func_ls_point_cached(
                                    p2_next_alpha,
                                    n_constr_clamped,
                                    ne,
                                    nef,
                                    quad_gauss_0,
                                    quad_gauss_1,
                                    quad_gauss_2,
                                    cached_Jaref,
                                    cached_jv,
                                    cached_quad_0,
                                    cached_quad_1,
                                    cached_quad_2,
                                    cached_frictionloss,
                                    cached_diag,
                                    EPS,
                                )
                                ls_it_count += 1

                                # Check for convergence failure
                                if p1_alpha == p2_alpha:
                                    if pmid_cost < p0_cost:
                                        constraint_state.ls_result[i_b] = 0
                                    else:
                                        constraint_state.ls_result[i_b] = 7
                                    res_alpha = pmid_alpha
                                    done = True

                        if not done:
                            # Max iterations reached, pick best
                            if p1_cost <= p2_cost and p1_cost < p0_cost:
                                constraint_state.ls_result[i_b] = 4
                                res_alpha = p1_alpha
                            elif p2_cost <= p1_cost and p2_cost < p0_cost:
                                constraint_state.ls_result[i_b] = 4
                                res_alpha = p2_alpha
                            else:
                                constraint_state.ls_result[i_b] = 5
                                res_alpha = 0.0

            constraint_state.ls_it[i_b] = ls_it_count
            res_alpha_shared[0] = res_alpha

        ti.simt.block.sync()

        # ===== Step 5: Update qacc, Ma, Jaref using the computed alpha =====
        alpha = res_alpha_shared[0]

        if ti.abs(alpha) < rigid_global_info.EPS[None]:
            if tid == 0:
                constraint_state.improved[i_b] = False
        else:
            # Parallel update of qacc and Ma
            i_d = tid
            while i_d < n_dofs:
                constraint_state.qacc[i_d, i_b] = (
                    constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                )
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha
                i_d = i_d + BLOCK_DIM

            # Parallel update of Jaref
            i_c = tid
            while i_c < n_constraints:
                constraint_state.Jaref[i_c, i_b] = (
                    constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha
                )
                i_c = i_c + BLOCK_DIM

        ti.simt.block.sync()


def func_solve_shared_mem(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Solver using shared memory optimized line search (Strategy B).

    Uses _kernel_linesearch_shared_mem for line search, then the standard
    kernels for the remaining solver steps.

    NOTE: Falls back to macro kernels if shared memory is not supported
    (non-CUDA backends like CPU, Metal, Vulkan).
    """
    # Check if shared memory is supported
    if not _is_shared_mem_supported():
        # Fall back to macro kernels on non-CUDA backends
        func_solve_decomposed_macrokernels(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        return

    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        # Use shared memory optimized line search
        _kernel_linesearch_shared_mem(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )

        # CG: save previous gradient
        if static_rigid_sim_config.solver_type == gs.constraint_solver.CG:
            _kernel_cg_only_save_prev_grad(
                constraint_state,
                static_rigid_sim_config,
            )

        # Update constraints
        _kernel_update_constraint(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )

        # Newton: Hessian update
        if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
            _kernel_newton_only_nt_hessian_incremental(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )

        # Update gradient
        _kernel_update_gradient(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )

        # Update search direction
        _kernel_update_search_direction(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
