"""
Allegro Hand — Interactive Tuning Demo
=======================================

Live interactive viewer with keyboard-driven parameter tuning.
The thermal model parameters can be adjusted in real time using
keyboard controls while the simulation runs.

Controls:
    TAB         — Cycle selected parameter
    UP / DOWN   — Increase / decrease selected parameter
    SHIFT+UP/DN — Large increase / decrease
    0           — Reset all parameters to defaults
    T           — Reset all temperatures to ambient
    ESC         — Quit

Plus all default viewer controls (press 'i' to see them).

Usage:
    python allegro_tuning_demo.py --cpu
    python allegro_tuning_demo.py          # GPU
"""

import argparse
import math
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.ext.pyrender import TextAlign
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, KeyMod, Keybind

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLEGRO_PATH = os.path.join(SCRIPT_DIR, "right_hand.xml")

# ── Simulation ──────────────────────────────────────────────────────────
DT = 5e-3
SUBSTEPS = 10
CYCLE_PERIOD = 1.5

# ── Motor thermal model ────────────────────────────────────────────────
ALPHA = 10.0
BETA = 1.0
JOINT_BASE_POWER = 0.5
JOINT_MAX_POWER = 50.0
C_THERMAL = 2.0
R_COOLING = 30.0
MAX_TEMP = 80.0
AMBIENT_TEMP = 22.0

# ── Visualization ──────────────────────────────────────────────────────
VIS_TEMP_MIN = 22.0
VIS_TEMP_MAX = 25.0
VOXEL_RESOLUTION = 0.005

VOXELIZED_LINK_NAMES = [
    "palm",
    "ff_base",
    "ff_proximal",
    "ff_medial",
    "ff_distal",
    "mf_base",
    "mf_proximal",
    "mf_medial",
    "mf_distal",
    "rf_base",
    "rf_proximal",
    "rf_medial",
    "rf_distal",
    "th_base",
    "th_proximal",
    "th_medial",
    "th_distal",
]

DIFFUSION_ALPHA = 0.15
CONDUCTION_RATE = 500.0
HEAT_INJECT_RADIUS = 0.015
N_EDGE_VOXELS = 8

# ── Joint limits ───────────────────────────────────────────────────────
JOINT_LO = np.array(
    [
        -0.47,
        -0.47,
        -0.47,
        0.263,
        -0.196,
        -0.196,
        -0.196,
        -0.105,
        -0.174,
        -0.174,
        -0.174,
        -0.189,
        -0.227,
        -0.227,
        -0.227,
        -0.162,
    ],
    dtype=np.float32,
)
JOINT_HI = np.array(
    [
        0.47,
        0.47,
        0.47,
        1.396,
        1.61,
        1.61,
        1.61,
        1.163,
        1.709,
        1.709,
        1.709,
        1.644,
        1.618,
        1.618,
        1.618,
        1.719,
    ],
    dtype=np.float32,
)
JOINT_MID = (JOINT_LO + JOINT_HI) / 2
JOINT_AMP = (JOINT_HI - JOINT_LO) / 2

FINGER_PHASE = np.zeros(16, dtype=np.float32)
for _fi, _phase in enumerate([0.0, 0.25, 0.50, 0.75]):
    for _ji in range(4):
        FINGER_PHASE[_ji * 4 + _fi] = _phase * (2 * math.pi)

# Colormap: blue -> purple -> red
_STOP_T = np.array([0.0, 0.5, 1.0])
_STOP_R = np.array([0.1, 0.5, 1.0])
_STOP_G = np.array([0.1, 0.0, 0.0])
_STOP_B = np.array([0.9, 0.5, 0.1])


def temps_to_colors(temps, t_min, t_max):
    norm = np.clip((temps - t_min) / (t_max - t_min + 1e-12), 0.0, 1.0)
    r = np.interp(norm, _STOP_T, _STOP_R)
    g = np.interp(norm, _STOP_T, _STOP_G)
    b = np.interp(norm, _STOP_T, _STOP_B)
    a = np.full_like(r, 0.85)
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)


# ── Tunable parameter definitions ─────────────────────────────────────
# (name, default, small_step, large_step, min, max, format)
PARAM_DEFS = [
    ("alpha", ALPHA, 1.0, 5.0, 0.0, 100.0, ".1f"),
    ("beta", BETA, 0.1, 0.5, 0.0, 10.0, ".2f"),
    ("base_power", JOINT_BASE_POWER, 0.1, 0.5, 0.0, 10.0, ".2f"),
    ("max_power", JOINT_MAX_POWER, 5.0, 20.0, 1.0, 200.0, ".0f"),
    ("r_cooling", R_COOLING, 2.0, 10.0, 1.0, 200.0, ".0f"),
    ("diffusion_alpha", DIFFUSION_ALPHA, 0.01, 0.05, 0.0, 0.16, ".3f"),
    ("conduction_rate", CONDUCTION_RATE, 50.0, 200.0, 0.0, 5000.0, ".0f"),
    ("vis_temp_min", VIS_TEMP_MIN, 0.5, 2.0, 0.0, 80.0, ".1f"),
    ("vis_temp_max", VIS_TEMP_MAX, 0.5, 2.0, 1.0, 80.0, ".1f"),
]
PARAM_NAMES = [p[0] for p in PARAM_DEFS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.4, 0.4, 0.4)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.12, 0.22),
            camera_lookat=(0, 0, 0.14),
            max_FPS=60,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane(visualization=False))
    allegro = scene.add_entity(gs.morphs.MJCF(file=ALLEGRO_PATH, pos=(0, 0, 0.15), euler=(0, -90, 0)))
    link_start = allegro.link_start

    voxelized_links = {}
    for name in VOXELIZED_LINK_NAMES:
        try:
            link = allegro.get_link(name)
            voxelized_links[link.idx - link_start] = name
        except Exception:
            pass
    voxelized_link_set = set(voxelized_links.keys())

    TP = gs.sensors.TemperatureProperties
    temp_props = {
        -1: TP(base_temperature=AMBIENT_TEMP, conductivity=0.0, density=1000.0, specific_heat=100.0, emissivity=0.0)
    }

    sensors = []
    sensor_by_local_idx = {}
    first_sensor = True
    for link in allegro.links:
        if link.n_geoms == 0:
            continue
        local_idx = link.idx - link_start
        is_voxelized = local_idx in voxelized_link_set
        sensor = scene.add_sensor(
            gs.sensors.TemperatureGrid(
                entity_idx=allegro.idx,
                link_idx_local=local_idx,
                voxel_resolution=VOXEL_RESOLUTION if is_voxelized else None,
                grid_size=(1, 1, 1),
                properties_dict=temp_props,
                simulate_all_link_temperatures=first_sensor,
                kinematic_conduction=False,
                ambient_temperature=AMBIENT_TEMP,
                convection_coefficient=0.0,
                contact_depth_weight=0.0,
                debug_temperature_range=(VIS_TEMP_MIN, VIS_TEMP_MAX),
            )
        )
        sensors.append(sensor)
        if is_voxelized:
            sensor_by_local_idx[local_idx] = sensor
        first_sensor = False

    scene.build()

    # Remove default visual meshes
    vis_context = scene._visualizer.context
    for vgeom in allegro.vgeoms:
        if vgeom.uid in vis_context.rigid_nodes:
            node = vis_context.rigid_nodes.pop(vgeom.uid)
            vis_context.remove_node(node)

    allegro.set_dofs_kp([5.0] * 16)
    allegro.set_dofs_kv([0.5] * 16)

    joints = allegro.joints
    dof_to_link_idx = [j.link.idx for j in joints]
    n_dofs = len(dof_to_link_idx)
    all_links = allegro.links
    n_links = len(all_links)

    adjacency_edges = []
    for link in all_links:
        if link.parent_idx >= 0:
            adjacency_edges.append((link.parent_idx - link_start, link.idx - link_start))

    link_temps = sensors[0].link_temperatures
    dtype = sensors[0]._get_cache_dtype()
    gt_cache = sensors[0]._manager._ground_truth_cache[dtype]
    cache_slice = sensors[0]._manager._cache_slices_by_type[type(sensors[0])]

    # Build voxel cache info
    voxel_cache_info = {}
    for local_idx, sensor in sorted(sensor_by_local_idx.items()):
        sm = sensor._shared_metadata
        i_s = sensor._idx
        start = sm.sensor_cache_start[i_s].item()
        size = sm.cache_sizes[i_s]
        occ = sm.occupancy_mask[i_s]
        occ_flat = occ.reshape(-1) if occ is not None else None
        global_idx = local_idx + link_start

        cell_positions = sensor._debug_cell_local_positions
        solid = occ_flat.cpu().numpy() > 0.5 if occ_flat is not None else np.ones(len(cell_positions), dtype=bool)
        dist_to_origin = np.linalg.norm(cell_positions, axis=1)
        near_joint = (dist_to_origin < HEAT_INJECT_RADIUS) & solid
        n_near = max(int(near_joint.sum()), 1)

        n_solid = int((occ_flat > 0.5).sum()) if occ_flat is not None else size
        grid_dims = tuple(sm.grid_size[i_s].tolist())
        voxel_cache_info[global_idx] = {
            "cache_start": start + cache_slice.start,
            "cache_size": size,
            "occ_flat": occ_flat,
            "n_solid": n_solid,
            "grid_dims": grid_dims,
            "near_joint_mask": torch.tensor(near_joint, dtype=torch.bool, device=gt_cache.device),
            "n_near_voxels": n_near,
        }

    # Build inter-link conduction edges
    voxel_conduction_edges = []
    for parent_local, child_local in adjacency_edges:
        pg = parent_local + link_start
        cg = child_local + link_start
        if pg not in voxel_cache_info or cg not in voxel_cache_info:
            continue
        p_sensor = sensor_by_local_idx[parent_local]
        c_sensor = sensor_by_local_idx[child_local]
        p_cells = p_sensor._debug_cell_local_positions
        c_cells = c_sensor._debug_cell_local_positions
        p_solid_np = (
            voxel_cache_info[pg]["occ_flat"].cpu().numpy() > 0.5
            if voxel_cache_info[pg]["occ_flat"] is not None
            else np.ones(len(p_cells), dtype=bool)
        )
        c_solid_np = (
            voxel_cache_info[cg]["occ_flat"].cpu().numpy() > 0.5
            if voxel_cache_info[cg]["occ_flat"] is not None
            else np.ones(len(c_cells), dtype=bool)
        )

        parent_link = allegro.links[parent_local]
        child_link = allegro.links[child_local]
        p_pos = tensor_to_array(parent_link.get_pos(0)).reshape(3).astype(np.float64)
        p_quat = tensor_to_array(parent_link.get_quat(0)).reshape(4).astype(np.float64)
        c_pos = tensor_to_array(child_link.get_pos(0)).reshape(3).astype(np.float64)
        p_T = gu.trans_quat_to_T(p_pos, p_quat)
        p_T_inv = np.linalg.inv(p_T)
        child_in_parent = (p_T_inv[:3, :3] @ c_pos) + p_T_inv[:3, 3]

        dist_to_child_joint = np.linalg.norm(p_cells - child_in_parent, axis=1)
        dist_to_child_joint[~p_solid_np] = 1e6
        sorted_idx = np.argsort(dist_to_child_joint)
        parent_edge = np.zeros(len(p_cells), dtype=bool)
        parent_edge[sorted_idx[:N_EDGE_VOXELS]] = True
        parent_edge &= p_solid_np
        n_parent = max(int(parent_edge.sum()), 1)

        c_dist = np.linalg.norm(c_cells, axis=1)
        c_dist[~c_solid_np] = 1e6
        sorted_idx_c = np.argsort(c_dist)
        child_edge_np = np.zeros(len(c_cells), dtype=bool)
        child_edge_np[sorted_idx_c[:N_EDGE_VOXELS]] = True
        child_edge_np &= c_solid_np
        n_child = max(int(child_edge_np.sum()), 1)

        voxel_conduction_edges.append(
            (
                pg,
                cg,
                torch.tensor(parent_edge, dtype=torch.bool, device=gt_cache.device),
                torch.tensor(child_edge_np, dtype=torch.bool, device=gt_cache.device),
                n_parent,
                n_child,
            )
        )

    # Mesh draw info
    mesh_draw_info = {}
    for local_idx, sensor in sensor_by_local_idx.items():
        link = allegro.links[local_idx]
        cell_positions = sensor._debug_cell_local_positions
        global_idx = local_idx + link_start
        occ_info = voxel_cache_info.get(global_idx)
        solid_np = (
            occ_info["occ_flat"].cpu().numpy() > 0.5
            if occ_info and occ_info["occ_flat"] is not None
            else np.ones(len(cell_positions), dtype=bool)
        )

        all_verts_list, all_faces_list = [], []
        vert_offset = 0
        for vgeom in link.vgeoms:
            link_verts = gu.transform_by_trans_quat(vgeom.init_vverts, vgeom._init_pos, vgeom._init_quat)
            all_verts_list.append(link_verts)
            all_faces_list.append(vgeom.init_vfaces + vert_offset)
            vert_offset += len(link_verts)
        if not all_verts_list:
            continue

        combined_verts = np.concatenate(all_verts_list)
        combined_faces = np.concatenate(all_faces_list)
        base_mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=False)

        solid_indices = np.where(solid_np)[0]
        if len(solid_indices) == 0:
            tree = cKDTree(cell_positions)
            _, vertex_to_voxel = tree.query(combined_verts)
        else:
            tree = cKDTree(cell_positions[solid_indices])
            _, nearest_solid = tree.query(combined_verts)
            vertex_to_voxel = solid_indices[nearest_solid]

        mesh_draw_info[local_idx] = {
            "sensor": sensor,
            "base_trimesh": base_mesh,
            "vertex_to_voxel": vertex_to_voxel,
            "link": link,
        }

    # Thermal capacitance
    props = temp_props.get(-1)
    voxel_vol = VOXEL_RESOLUTION**3
    link_capacitance = np.full(n_links, C_THERMAL, dtype=np.float64)
    for link in all_links:
        local_idx = link.idx - link_start
        gi = local_idx + link_start
        if gi in voxel_cache_info:
            n_solid = voxel_cache_info[gi]["n_solid"]
            solid_volume = n_solid * voxel_vol
            if solid_volume > 0:
                link_capacitance[local_idx] = props.density * props.specific_heat * solid_volume

    # Neighbor counts for diffusion
    nbr_counts = {}
    for gi, info in voxel_cache_info.items():
        nx, ny, nz = info["grid_dims"]
        n_nbr = torch.zeros(nx, ny, nz, dtype=gt_cache.dtype, device=gt_cache.device)
        n_nbr[1:, :, :] += 1
        n_nbr[:-1, :, :] += 1
        n_nbr[:, 1:, :] += 1
        n_nbr[:, :-1, :] += 1
        n_nbr[:, :, 1:] += 1
        n_nbr[:, :, :-1] += 1
        n_nbr.clamp_(min=1)
        nbr_counts[gi] = n_nbr

    # ── Mutable parameters ─────────────────────────────────────────────
    params = {
        "alpha": ALPHA,
        "beta": BETA,
        "base_power": JOINT_BASE_POWER,
        "max_power": JOINT_MAX_POWER,
        "r_cooling": R_COOLING,
        "diffusion_alpha": DIFFUSION_ALPHA,
        "conduction_rate": CONDUCTION_RATE,
        "vis_temp_min": VIS_TEMP_MIN,
        "vis_temp_max": VIS_TEMP_MAX,
    }
    defaults = dict(params)

    # ── Interactive tuning state ───────────────────────────────────────
    selected_idx = [0]  # mutable container for closure

    def cycle_param():
        selected_idx[0] = (selected_idx[0] + 1) % len(PARAM_DEFS)
        _update_caption()
        name = PARAM_DEFS[selected_idx[0]][0]
        scene.viewer._pyrender_viewer.set_message_text(f"Selected: {name}")

    def adjust_param(direction, large=False):
        pdef = PARAM_DEFS[selected_idx[0]]
        name, _, small_step, large_step, lo, hi, _ = pdef
        step = large_step if large else small_step
        params[name] = max(lo, min(hi, params[name] + direction * step))
        _update_caption()
        fmt = pdef[6]
        scene.viewer._pyrender_viewer.set_message_text(f"{name} = {params[name]:{fmt}}")

    def reset_params():
        params.update(defaults)
        _update_caption()
        scene.viewer._pyrender_viewer.set_message_text("Parameters reset to defaults")

    def reset_temps():
        for gi, info in voxel_cache_info.items():
            s = info["cache_start"]
            gt_cache[0, s : s + info["cache_size"]] = AMBIENT_TEMP
        link_temps[0, :] = AMBIENT_TEMP
        scene.viewer._pyrender_viewer.set_message_text("Temperatures reset to ambient")

    is_running = [True]

    def stop():
        is_running[0] = False

    def _build_caption_text():
        lines = ["  Thermal Tuning  [TAB] cycle  [UP/DN] adjust  [0] reset  [T] cool"]
        for i, pdef in enumerate(PARAM_DEFS):
            name, _, _, _, _, _, fmt = pdef
            marker = "> " if i == selected_idx[0] else "  "
            lines.append(f"{marker}{name:>16s} = {params[name]:{fmt}}")
        return "\n".join(lines)

    def _update_caption():
        caption_text = _build_caption_text()
        scene.viewer._pyrender_viewer.viewer_flags["caption"] = [
            {
                "text": caption_text,
                "location": TextAlign.TOP_LEFT,
                "font_name": "OpenSans-Regular",
                "font_pt": 14,
                "color": np.array([1.0, 1.0, 0.6, 0.95]),
                "scale": 1.0,
            }
        ]

    # Register keybindings
    scene.viewer.register_keybinds(
        Keybind("tune_cycle", Key.TAB, KeyAction.PRESS, callback=cycle_param),
        Keybind("tune_up", Key.UP, KeyAction.HOLD, callback=adjust_param, args=(1,)),
        Keybind("tune_down", Key.DOWN, KeyAction.HOLD, callback=adjust_param, args=(-1,)),
        Keybind(
            "tune_up_large",
            Key.UP,
            KeyAction.HOLD,
            key_mods=(KeyMod.SHIFT,),
            callback=adjust_param,
            args=(1,),
            kwargs={"large": True},
        ),
        Keybind(
            "tune_down_large",
            Key.DOWN,
            KeyAction.HOLD,
            key_mods=(KeyMod.SHIFT,),
            callback=adjust_param,
            args=(-1,),
            kwargs={"large": True},
        ),
        Keybind("tune_reset_params", Key._0, KeyAction.PRESS, callback=reset_params),
        Keybind("tune_reset_temps", Key.T, KeyAction.PRESS, callback=reset_temps),
        Keybind("tune_quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
        overwrite=True,
    )

    # Set initial caption
    _update_caption()

    debug_objs = []
    step = 0

    print("\n=== Allegro Interactive Tuning Demo ===")
    print("TAB: cycle parameter | UP/DOWN: adjust | SHIFT+UP/DOWN: large adjust")
    print("0: reset params | T: reset temperatures | ESC: quit\n")

    try:
        while is_running[0] and scene.viewer.is_alive():
            t = step * DT

            # ── Finger motion ──────────────────────────────────────────────
            omega = 2 * math.pi / CYCLE_PERIOD
            phase = omega * t + FINGER_PHASE
            target = JOINT_MID + 0.8 * JOINT_AMP * np.sin(phase)
            allegro.control_dofs_position(target)
            scene.step()

            # ── 1. Motor heat ──────────────────────────────────────────────
            dof_vel = tensor_to_array(allegro.get_dofs_velocity()).reshape(-1)
            dof_force = tensor_to_array(allegro.get_dofs_force()).reshape(-1)
            for dof_i in range(n_dofs):
                tau = float(dof_force[dof_i])
                omega_j = float(dof_vel[dof_i])
                heat_power = params["base_power"] + params["alpha"] * tau * tau + params["beta"] * omega_j * omega_j
                heat_power = min(heat_power, params["max_power"])
                gi = dof_to_link_idx[dof_i]
                if gi not in voxel_cache_info:
                    continue
                info = voxel_cache_info[gi]
                local_idx = gi - link_start
                C_i = link_capacitance[local_idx]
                delta_T = DT * heat_power / C_i
                link_temps[0, gi] += delta_T
                start = info["cache_start"]
                gt_cache[0, start : start + info["cache_size"]][info["near_joint_mask"]] += delta_T
                gt_cache[0, start : start + info["cache_size"]].clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)

            # ── 2. Newton cooling ──────────────────────────────────────────
            for i_link in range(n_links):
                global_idx = i_link + link_start
                T_current = float(link_temps[0, global_idx])
                C_i = link_capacitance[i_link]
                delta_cool = DT * (T_current - AMBIENT_TEMP) / (params["r_cooling"] * C_i)
                link_temps[0, global_idx] = max(AMBIENT_TEMP, min(T_current - delta_cool, MAX_TEMP))

            for gi, info in voxel_cache_info.items():
                start = info["cache_start"]
                size = info["cache_size"]
                data = gt_cache[0, start : start + size]
                solid_mask = info["occ_flat"] > 0.5 if info["occ_flat"] is not None else None
                local_idx = gi - link_start
                C_i = link_capacitance[local_idx]
                n_solid = info["n_solid"]
                if n_solid > 0 and C_i > 0:
                    cool_rate = DT / (params["r_cooling"] * C_i) * n_solid
                    if solid_mask is not None:
                        data[solid_mask] -= cool_rate * (data[solid_mask] - AMBIENT_TEMP)
                    else:
                        data -= cool_rate * (data - AMBIENT_TEMP)
                    data.clamp_(min=AMBIENT_TEMP)

            # ── 3. Inter-link conduction ───────────────────────────────────
            for pg, cg, parent_edge_mask, child_edge_mask, n_parent, n_child in voxel_conduction_edges:
                p_info = voxel_cache_info[pg]
                c_info = voxel_cache_info[cg]
                p_data = gt_cache[0, p_info["cache_start"] : p_info["cache_start"] + p_info["cache_size"]]
                c_data = gt_cache[0, c_info["cache_start"] : c_info["cache_start"] + c_info["cache_size"]]
                T_p = p_data[parent_edge_mask].mean().item()
                T_c = c_data[child_edge_mask].mean().item()
                flux = params["conduction_rate"] * (T_p - T_c) * DT
                p_solid = p_info["occ_flat"]
                c_solid = c_info["occ_flat"]
                p_mask = (
                    p_solid > 0.5
                    if p_solid is not None
                    else torch.ones(p_info["cache_size"], dtype=torch.bool, device=gt_cache.device)
                )
                c_mask = (
                    c_solid > 0.5
                    if c_solid is not None
                    else torch.ones(c_info["cache_size"], dtype=torch.bool, device=gt_cache.device)
                )
                p_data[p_mask] -= flux / max(int(p_mask.sum()), 1)
                c_data[c_mask] += flux / max(int(c_mask.sum()), 1)
                p_data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)
                c_data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)

            # ── 4. Intra-link diffusion ────────────────────────────────────
            alpha_d = params["diffusion_alpha"]
            if alpha_d > 0:
                for gi, info in voxel_cache_info.items():
                    start = info["cache_start"]
                    nx, ny, nz = info["grid_dims"]
                    data = gt_cache[0, start : start + info["cache_size"]].view(nx, ny, nz)
                    lap = torch.zeros_like(data)
                    lap[1:] += data[:-1]
                    lap[:-1] += data[1:]
                    lap[:, 1:] += data[:, :-1]
                    lap[:, :-1] += data[:, 1:]
                    lap[:, :, 1:] += data[:, :, :-1]
                    lap[:, :, :-1] += data[:, :, 1:]
                    data += alpha_d * (lap / nbr_counts[gi] - data)
                    data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)

            # ── Draw thermal meshes ────────────────────────────────────────
            for obj in debug_objs:
                scene.clear_debug_object(obj)
            debug_objs.clear()

            vis_min = params["vis_temp_min"]
            vis_max = params["vis_temp_max"]

            all_world_verts, all_world_faces, all_face_colors = [], [], []
            vert_offset = 0
            for local_idx, minfo in mesh_draw_info.items():
                link = minfo["link"]
                base_mesh = minfo["base_trimesh"]
                v2v = minfo["vertex_to_voxel"]
                link_pos = tensor_to_array(link.get_pos(0)).reshape(3)
                link_quat = tensor_to_array(link.get_quat(0)).reshape(4)
                link_T = gu.trans_quat_to_T(link_pos, link_quat)
                world_verts = (link_T[:3, :3] @ base_mesh.vertices.T).T + link_T[:3, 3]

                temps = tensor_to_array(minfo["sensor"].read_ground_truth(None)).reshape(-1)
                vertex_colors = temps_to_colors(temps[v2v], vis_min, vis_max)
                face_colors = vertex_colors[base_mesh.faces].mean(axis=1).astype(np.uint8)

                all_world_verts.append(world_verts)
                all_world_faces.append(base_mesh.faces + vert_offset)
                all_face_colors.append(face_colors)
                vert_offset += len(world_verts)

            if all_world_verts:
                combined = trimesh.Trimesh(
                    vertices=np.concatenate(all_world_verts),
                    faces=np.concatenate(all_world_faces),
                    process=False,
                )
                combined.visual.face_colors = np.concatenate(all_face_colors)
                debug_objs.append(scene.draw_debug_mesh(combined))

            # Update caption each frame to reflect latest param values
            _update_caption()

            if step % 200 == 0:
                summary = []
                for name in ["palm", "ff_proximal", "mf_proximal", "th_distal"]:
                    try:
                        lk = allegro.get_link(name)
                        gi = lk.idx
                        if gi in voxel_cache_info:
                            info = voxel_cache_info[gi]
                            data = gt_cache[0, info["cache_start"] : info["cache_start"] + info["cache_size"]]
                            sm = (
                                info["occ_flat"] > 0.5
                                if info["occ_flat"] is not None
                                else torch.ones(info["cache_size"], dtype=torch.bool, device=data.device)
                            )
                            summary.append(f"{name}={data[sm].mean().item():.1f}")
                    except Exception:
                        pass
                print(f"  t={t:.1f}s: {' | '.join(summary)}")

            step += 1

            if "PYTEST_VERSION" in os.environ:
                break

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
