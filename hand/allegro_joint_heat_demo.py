"""
Allegro Hand — Voxelized Joint Heat Demo
=========================================

Simulates motor heat generation in the Allegro hand's joints during cyclic
finger movements, with heat conducting between adjacent links through the
kinematic chain.

Thermal model:
  - Heat source: P = alpha * tau^2 + beta * omega^2  (I^2R + viscous loss)
  - Cooling: Newton's law  dT/dt = -(T - T_amb) / (R * C)
  - Intra-link diffusion: explicit 6-neighbor Laplacian
  - Inter-link conduction: edge-voxel-driven flux at joint boundaries
  - Thermal capacitance: rho * c_p * solid_voxel_volume (not AABB)

Usage:
    python allegro_joint_heat_demo.py --vis         # interactive viewer with keybinds
    python allegro_joint_heat_demo.py              # record video (no viewer)
    python allegro_joint_heat_demo.py --debug-palm  # single palm heat source
    python allegro_joint_heat_demo.py --cpu         # CPU backend
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
from genesis.utils.misc import tensor_to_array

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLEGRO_PATH = os.path.join(SCRIPT_DIR, "right_hand.xml")

# ── Simulation ──────────────────────────────────────────────────────────
DT = 5e-3
SUBSTEPS = 10
TOTAL_SECONDS = 15.0
CYCLE_PERIOD = 1.5  # seconds per open/close cycle

# ── Motor thermal model parameters ─────────────────────────────────────
# Physical motor heat: P = alpha * tau^2 + beta * omega^2
# alpha ~ R_winding / K_t^2  (I^2R loss: higher torque = more current = more heat)
# beta  ~ viscous damping     (bearing/friction loss proportional to speed^2)
ALPHA = 10.0      # W/Nm^2 — resistive loss coefficient
BETA = 1.0        # W/(rad/s)^2 — viscous/friction loss coefficient
JOINT_BASE_POWER = 0.5    # W — idle electronics heat per joint
JOINT_MAX_POWER = 50.0    # W — clamp total heat to filter collision spikes

C_THERMAL = 2.0   # J/K — fallback thermal capacitance
R_COOLING = 30.0  # K/W — thermal resistance to ambient
MAX_TEMP = 80.0   # degC — clamp ceiling
AMBIENT_TEMP = 22.0  # degC

# ── Visualization ───────────────────────────────────────────────────────
VIS_TEMP_MIN = 22.0
VIS_TEMP_MAX = 25.0

# Voxelized links: palm + all finger chains
VOXEL_RESOLUTION = 0.005  # 5mm voxels
VOXELIZED_LINK_NAMES = [
    "palm",
    "ff_base", "ff_proximal", "ff_medial", "ff_distal",
    "mf_base", "mf_proximal", "mf_medial", "mf_distal",
    "rf_base", "rf_proximal", "rf_medial", "rf_distal",
    "th_base", "th_proximal", "th_medial", "th_distal",
]
PALM_HEAT_POWER = 200.0  # W — constant heat source at palm center (debug mode)

# ── Diffusion / conduction ────────────────────────────────────────────
DIFFUSION_ALPHA = 0.15    # explicit diffusion coefficient (max stable ~0.16 for 3D)
CONDUCTION_RATE = 500.0   # inter-link conduction strength
HEAT_INJECT_RADIUS = 0.015  # meters — inject heat within this distance of joint origin

# ── Joint limits from MJCF ──────────────────────────────────────────────
JOINT_LO = np.array([
    -0.47, -0.47, -0.47,  0.263,
    -0.196, -0.196, -0.196, -0.105,
    -0.174, -0.174, -0.174, -0.189,
    -0.227, -0.227, -0.227, -0.162,
], dtype=np.float32)
JOINT_HI = np.array([
     0.47,  0.47,  0.47,  1.396,
     1.61,  1.61,  1.61,  1.163,
     1.709, 1.709, 1.709, 1.644,
     1.618, 1.618, 1.618, 1.719,
], dtype=np.float32)
JOINT_MID = (JOINT_LO + JOINT_HI) / 2
JOINT_AMP = (JOINT_HI - JOINT_LO) / 2

FINGER_PHASE = np.zeros(16, dtype=np.float32)
for _fi, _phase in enumerate([0.0, 0.25, 0.50, 0.75]):
    for _ji in range(4):
        FINGER_PHASE[_ji * 4 + _fi] = _phase * (2 * math.pi)

FINGER_NAMES = ["FF", "MF", "RF", "TH"]


# Colormap stops: blue -> purple -> red
_STOP_T = np.array([0.0, 0.5, 1.0])
_STOP_R = np.array([0.1, 0.5, 1.0])
_STOP_G = np.array([0.1, 0.0, 0.0])
_STOP_B = np.array([0.9, 0.5, 0.1])


def temps_to_colors(temps, t_min, t_max):
    """Vectorized temperature -> RGBA uint8 color mapping."""
    norm = np.clip((temps - t_min) / (t_max - t_min + 1e-12), 0.0, 1.0)
    r = np.interp(norm, _STOP_T, _STOP_R)
    g = np.interp(norm, _STOP_T, _STOP_G)
    b = np.interp(norm, _STOP_T, _STOP_B)
    a = np.full_like(r, 0.85)
    colors = np.stack([r, g, b, a], axis=-1)
    return (colors * 255).astype(np.uint8)


class ThermalParams:
    """Mutable thermal parameters for interactive tuning."""
    def __init__(self):
        self.alpha = ALPHA
        self.beta = BETA
        self.base_power = JOINT_BASE_POWER
        self.max_power = JOINT_MAX_POWER
        self.r_cooling = R_COOLING
        self.diffusion_alpha = DIFFUSION_ALPHA
        self.conduction_rate = CONDUCTION_RATE
        self.vis_temp_min = VIS_TEMP_MIN
        self.vis_temp_max = VIS_TEMP_MAX

        # UI state
        self._param_names = [
            "alpha", "beta", "base_power", "r_cooling",
            "diffusion_alpha", "conduction_rate", "vis_temp_min", "vis_temp_max",
        ]
        self._param_steps = {
            "alpha": 1.0,
            "beta": 0.2,
            "base_power": 0.1,
            "r_cooling": 5.0,
            "diffusion_alpha": 0.01,
            "conduction_rate": 50.0,
            "vis_temp_min": 0.5,
            "vis_temp_max": 0.5,
        }
        self._param_ranges = {
            "alpha": (0.0, 100.0),
            "beta": (0.0, 20.0),
            "base_power": (0.0, 5.0),
            "r_cooling": (1.0, 500.0),
            "diffusion_alpha": (0.0, 0.16),
            "conduction_rate": (0.0, 2000.0),
            "vis_temp_min": (0.0, 50.0),
            "vis_temp_max": (1.0, 80.0),
        }
        self._selected_idx = 0

    @property
    def selected_name(self):
        return self._param_names[self._selected_idx]

    def select_next(self):
        self._selected_idx = (self._selected_idx + 1) % len(self._param_names)

    def select_prev(self):
        self._selected_idx = (self._selected_idx - 1) % len(self._param_names)

    def increase(self):
        name = self.selected_name
        step = self._param_steps[name]
        lo, hi = self._param_ranges[name]
        val = min(getattr(self, name) + step, hi)
        setattr(self, name, val)

    def decrease(self):
        name = self.selected_name
        step = self._param_steps[name]
        lo, hi = self._param_ranges[name]
        val = max(getattr(self, name) - step, lo)
        setattr(self, name, val)

    def status_text(self):
        lines = []
        for i, name in enumerate(self._param_names):
            marker = ">" if i == self._selected_idx else " "
            val = getattr(self, name)
            lines.append(f"{marker} {name}: {val:.3g}")
        return "  ".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Allegro voxelized joint heat demo")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--seconds", "-t", type=float, default=TOTAL_SECONDS)
    parser.add_argument("--debug-palm", action="store_true", help="Use single palm heat source (debug mode)")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            ambient_light=(0.4, 0.4, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.12, 0.22),
            camera_lookat=(0, 0, 0.14),
            max_FPS=60,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane(visualization=False))

    allegro = scene.add_entity(gs.morphs.MJCF(
        file=ALLEGRO_PATH,
        pos=(0, 0, 0.15),
        euler=(0, -90, 0),
    ))

    link_start = allegro.link_start

    # Resolve which links to voxelize
    voxelized_links = {}  # local_idx -> link_name
    for name in VOXELIZED_LINK_NAMES:
        try:
            link = allegro.get_link(name)
            voxelized_links[link.idx - link_start] = name
        except Exception:
            pass
    voxelized_link_set = set(voxelized_links.keys())

    # ── Add sensors: voxelized on selected links, 1x1x1 for others ──
    # conductivity=0 disables sensor's internal contact heat transfer (which is
    # numerically unstable for this demo). We run our own diffusion instead.
    TP = gs.sensors.TemperatureProperties
    temp_props = {-1: TP(base_temperature=AMBIENT_TEMP, conductivity=0.0, density=1000.0, specific_heat=100.0, emissivity=0.0)}

    sensors = []
    sensor_by_local_idx = {}  # local_link_idx -> sensor
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

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.25, -0.12, 0.22),
        lookat=(0, 0, 0.14),
        fov=45,
        GUI=False,
    )

    scene.build()

    # Remove the entity's default visual meshes — we draw our own temperature-colored meshes
    vis_context = scene._visualizer.context
    for vgeom in allegro.vgeoms:
        if vgeom.uid in vis_context.rigid_nodes:
            node = vis_context.rigid_nodes.pop(vgeom.uid)
            vis_context.remove_node(node)

    allegro.set_dofs_kp([5.0] * 16)
    allegro.set_dofs_kv([0.5] * 16)

    # Build DOF -> link mapping
    joints = allegro.joints
    dof_to_link_idx = [j.link.idx for j in joints]
    n_dofs = len(dof_to_link_idx)

    all_links = allegro.links
    n_links = len(all_links)

    # Adjacency edges
    adjacency_edges = []
    for link in all_links:
        if link.parent_idx >= 0:
            parent_local = link.parent_idx - link_start
            child_local = link.idx - link_start
            adjacency_edges.append((parent_local, child_local))

    link_temps = sensors[0].link_temperatures

    # Access the ground truth cache for writing heat into voxel grids
    dtype = sensors[0]._get_cache_dtype()
    gt_cache = sensors[0]._manager._ground_truth_cache[dtype]
    cache_slice = sensors[0]._manager._cache_slices_by_type[type(sensors[0])]

    # Build mapping: global_link_idx -> cache info + masks
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

        max_dist = dist_to_origin[solid].max() if solid.any() else 0.01
        far_end = (dist_to_origin > max_dist - HEAT_INJECT_RADIUS) & solid
        n_far = max(int(far_end.sum()), 1)

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
            "far_end_mask": torch.tensor(far_end, dtype=torch.bool, device=gt_cache.device),
            "n_far_voxels": n_far,
        }

    # Build inter-link conduction edges
    N_EDGE_VOXELS = 8
    voxel_conduction_edges = []
    link_name_by_local = {}
    for name in VOXELIZED_LINK_NAMES:
        try:
            link = allegro.get_link(name)
            link_name_by_local[link.idx - link_start] = name
        except Exception:
            pass

    for parent_local, child_local in adjacency_edges:
        pg = parent_local + link_start
        cg = child_local + link_start
        if pg not in voxel_cache_info or cg not in voxel_cache_info:
            continue
        p_sensor = sensor_by_local_idx[parent_local]
        c_sensor = sensor_by_local_idx[child_local]
        p_cells = p_sensor._debug_cell_local_positions
        c_cells = c_sensor._debug_cell_local_positions
        p_solid = voxel_cache_info[pg]["occ_flat"]
        p_solid_np = p_solid.cpu().numpy() > 0.5 if p_solid is not None else np.ones(len(p_cells), dtype=bool)
        c_solid_np = voxel_cache_info[cg]["occ_flat"].cpu().numpy() > 0.5 if voxel_cache_info[cg]["occ_flat"] is not None else np.ones(len(c_cells), dtype=bool)

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

        parent_edge_t = torch.tensor(parent_edge, dtype=torch.bool, device=gt_cache.device)
        child_edge_t = torch.tensor(child_edge_np, dtype=torch.bool, device=gt_cache.device)
        voxel_conduction_edges.append((pg, cg, parent_edge_t, child_edge_t, n_parent, n_child))

    # Precompute palm heat source mask (debug mode only)
    palm_global_idx = None
    if args.debug_palm:
        palm_link = allegro.get_link("palm")
        palm_global_idx = palm_link.idx
        palm_sensor = sensor_by_local_idx.get(palm_global_idx - link_start)
        if palm_sensor is not None:
            palm_cells = palm_sensor._debug_cell_local_positions
            palm_occ = voxel_cache_info[palm_global_idx]["occ_flat"]
            solid = palm_occ.cpu().numpy() > 0.5 if palm_occ is not None else np.ones(len(palm_cells), dtype=bool)
            centroid = palm_cells[solid].mean(axis=0)
            dist_to_center = np.linalg.norm(palm_cells - centroid, axis=1)
            palm_center_mask = (dist_to_center < HEAT_INJECT_RADIUS) & solid
            if palm_center_mask.sum() == 0:
                nearest = np.argsort(dist_to_center)[:10]
                palm_center_mask = np.zeros(len(palm_cells), dtype=bool)
                palm_center_mask[nearest] = True
            palm_center_mask_t = torch.tensor(palm_center_mask, dtype=torch.bool, device=gt_cache.device)
            n_palm_center = max(int(palm_center_mask.sum()), 1)

    # Precompute per-link mesh info for thermal visualization
    mesh_draw_info = {}
    for local_idx, sensor in sensor_by_local_idx.items():
        link = allegro.links[local_idx]
        cell_positions = sensor._debug_cell_local_positions
        global_idx = local_idx + link_start

        occ_info = voxel_cache_info.get(global_idx)
        if occ_info is not None and occ_info["occ_flat"] is not None:
            solid_np = occ_info["occ_flat"].cpu().numpy() > 0.5
        else:
            solid_np = np.ones(len(cell_positions), dtype=bool)

        all_verts_list = []
        all_faces_list = []
        vert_offset = 0
        for vgeom in link.vgeoms:
            vverts = vgeom.init_vverts
            vfaces = vgeom.init_vfaces
            link_verts = gu.transform_by_trans_quat(vverts, vgeom._init_pos, vgeom._init_quat)
            all_verts_list.append(link_verts)
            all_faces_list.append(vfaces + vert_offset)
            vert_offset += len(link_verts)

        if not all_verts_list:
            continue

        combined_verts = np.concatenate(all_verts_list, axis=0)
        combined_faces = np.concatenate(all_faces_list, axis=0)
        base_mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=False)

        solid_indices = np.where(solid_np)[0]
        if len(solid_indices) == 0:
            tree = cKDTree(cell_positions)
            _, vertex_to_voxel = tree.query(combined_verts)
        else:
            solid_positions = cell_positions[solid_indices]
            tree = cKDTree(solid_positions)
            _, nearest_solid = tree.query(combined_verts)
            vertex_to_voxel = solid_indices[nearest_solid]

        mesh_draw_info[local_idx] = {
            "sensor": sensor,
            "base_trimesh": base_mesh,
            "vertex_to_voxel": vertex_to_voxel,
            "link": link,
        }

    total_draw_verts = sum(len(v["base_trimesh"].vertices) for v in mesh_draw_info.values())

    # Per-link thermal capacitance from solid voxel volume (more accurate than AABB)
    props = temp_props.get(-1)
    voxel_vol = VOXEL_RESOLUTION ** 3
    link_capacitance = np.full(n_links, C_THERMAL, dtype=np.float64)
    for link in all_links:
        local_idx = link.idx - link_start
        gi = local_idx + link_start
        if gi in voxel_cache_info:
            n_solid = voxel_cache_info[gi]["n_solid"]
            solid_volume = n_solid * voxel_vol
            if solid_volume > 0:
                link_capacitance[local_idx] = props.density * props.specific_heat * solid_volume
        elif link.n_geoms > 0:
            aabb = link.get_AABB()
            if aabb.ndim == 3:
                aabb = aabb[0]
            extents = (aabb[1] - aabb[0]).cpu().numpy()
            volume = float(abs(extents[0] * extents[1] * extents[2]))
            if volume > 0:
                link_capacitance[local_idx] = props.density * props.specific_heat * volume

    # Precompute neighbor count tensor for diffusion (doesn't change per frame)
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

    # ── Interactive parameter tuning (--vis mode) ────────────────────────
    params = ThermalParams()
    is_running = True

    if args.vis:
        from genesis.vis.keybindings import Key, KeyAction, Keybind

        def stop():
            nonlocal is_running
            is_running = False

        def show_params():
            if scene.viewer._pyrender_viewer is not None:
                scene.viewer._pyrender_viewer.set_message_text(params.status_text())

        def on_next_param():
            params.select_next()
            show_params()

        def on_prev_param():
            params.select_prev()
            show_params()

        def on_increase():
            params.increase()
            show_params()

        def on_decrease():
            params.decrease()
            show_params()

        def on_reset_temp():
            # Reset all voxel temperatures to ambient
            for gi, info in voxel_cache_info.items():
                start = info["cache_start"]
                size = info["cache_size"]
                gt_cache[0, start : start + size] = AMBIENT_TEMP
            link_temps[0, :] = AMBIENT_TEMP
            if scene.viewer._pyrender_viewer is not None:
                scene.viewer._pyrender_viewer.set_message_text("Temperatures reset to ambient")

        scene.viewer.register_keybinds(
            Keybind("param_next", Key.RIGHT, KeyAction.PRESS, callback=on_next_param),
            Keybind("param_prev", Key.LEFT, KeyAction.PRESS, callback=on_prev_param),
            Keybind("param_up", Key.UP, KeyAction.PRESS, callback=on_increase),
            Keybind("param_up_hold", Key.UP, KeyAction.HOLD, callback=on_increase),
            Keybind("param_down", Key.DOWN, KeyAction.PRESS, callback=on_decrease),
            Keybind("param_down_hold", Key.DOWN, KeyAction.HOLD, callback=on_decrease),
            Keybind("show_params", Key.T, KeyAction.PRESS, callback=show_params),
            Keybind("reset_temp", Key.N, KeyAction.PRESS, callback=on_reset_temp),
            Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
        )

    cam.start_recording()

    total_steps = int(args.seconds / DT)
    print(f"\n=== Allegro Voxelized Joint Heat Demo ===")
    heat_mode = "palm center (debug)" if args.debug_palm else f"motor model (alpha={ALPHA}, beta={BETA})"
    print(f"Heat source: {heat_mode}")
    print(f"Voxel resolution: {VOXEL_RESOLUTION*1000:.0f}mm on {len(voxelized_links)} links")
    print(f"Drawing {total_draw_verts} mesh vertices per frame")
    print(f"Simulating {args.seconds}s ({total_steps} steps)")
    print(f"Temperature visualization: blue={VIS_TEMP_MIN}degC -> red={VIS_TEMP_MAX}degC")
    if args.vis:
        print(f"\nInteractive controls:")
        print(f"  Left/Right arrows : select parameter")
        print(f"  Up/Down arrows    : adjust value (hold for continuous)")
        print(f"  T                 : show current parameters")
        print(f"  N                 : reset temperatures to ambient")
        print(f"  ESC               : quit")
    print()

    debug_objs = []

    step = 0
    while is_running and step < total_steps:
        t = step * DT
        omega = 2 * math.pi / CYCLE_PERIOD
        phase = omega * t + FINGER_PHASE
        target = JOINT_MID + 0.8 * JOINT_AMP * np.sin(phase)

        allegro.control_dofs_position(target)
        scene.step()

        # ── 1. Heat sources ───────────────────────────────────────────────
        if args.debug_palm:
            if palm_global_idx in voxel_cache_info:
                palm_info = voxel_cache_info[palm_global_idx]
                palm_local = palm_global_idx - link_start
                C_palm = link_capacitance[palm_local]
                delta_T = DT * PALM_HEAT_POWER / C_palm
                link_temps[0, palm_global_idx] += delta_T
                start = palm_info["cache_start"]
                size = palm_info["cache_size"]
                cache_data = gt_cache[0, start : start + size]
                cache_data[palm_center_mask_t] += delta_T
                cache_data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)
        else:
            # Motor heat model: P = alpha * tau^2 + beta * omega^2 + base
            # This correctly models I^2R loss (high torque = high current = more heat,
            # even at stall) and viscous loss (bearing friction at high speed).
            dof_vel = tensor_to_array(allegro.get_dofs_velocity()).reshape(-1)
            dof_force = tensor_to_array(allegro.get_dofs_force()).reshape(-1)
            for dof_i in range(n_dofs):
                tau = float(dof_force[dof_i])
                omega_j = float(dof_vel[dof_i])
                heat_power = params.base_power + params.alpha * tau * tau + params.beta * omega_j * omega_j
                heat_power = min(heat_power, params.max_power)
                gi = dof_to_link_idx[dof_i]
                if gi not in voxel_cache_info:
                    continue
                info = voxel_cache_info[gi]
                local_idx = gi - link_start
                C_i = link_capacitance[local_idx]
                delta_T = DT * heat_power / C_i
                link_temps[0, gi] += delta_T
                start = info["cache_start"]
                size = info["cache_size"]
                near_mask = info["near_joint_mask"]
                cache_data = gt_cache[0, start : start + size]
                cache_data[near_mask] += delta_T
                cache_data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)

        # ── 2. Newton cooling ──────────────────────────────────────────
        for i_link in range(n_links):
            global_idx = i_link + link_start
            T_current = float(link_temps[0, global_idx])
            C_i = link_capacitance[i_link]
            delta_T_cool = DT * (T_current - AMBIENT_TEMP) / (params.r_cooling * C_i)
            new_T = max(AMBIENT_TEMP, min(T_current - delta_T_cool, MAX_TEMP))
            link_temps[0, global_idx] = new_T

        for gi, info in voxel_cache_info.items():
            start = info["cache_start"]
            size = info["cache_size"]
            data = gt_cache[0, start : start + size]
            solid_mask = info["occ_flat"] > 0.5 if info["occ_flat"] is not None else None
            local_idx = gi - link_start
            C_i = link_capacitance[local_idx]
            n_solid = info["n_solid"]
            if n_solid > 0 and C_i > 0:
                cool_rate = DT / (params.r_cooling * C_i) * n_solid
                if solid_mask is not None:
                    data[solid_mask] -= cool_rate * (data[solid_mask] - AMBIENT_TEMP)
                else:
                    data -= cool_rate * (data - AMBIENT_TEMP)
                data.clamp_(min=AMBIENT_TEMP)

        # ── 3. Inter-link conduction at joints ──────────────────────────
        for ei, (pg, cg, parent_edge_mask, child_edge_mask, n_parent, n_child) in enumerate(voxel_conduction_edges):
            p_info = voxel_cache_info[pg]
            c_info = voxel_cache_info[cg]
            p_data = gt_cache[0, p_info["cache_start"] : p_info["cache_start"] + p_info["cache_size"]]
            c_data = gt_cache[0, c_info["cache_start"] : c_info["cache_start"] + c_info["cache_size"]]
            T_parent_end = p_data[parent_edge_mask].mean().item()
            T_child_start = c_data[child_edge_mask].mean().item()
            delta_T = T_parent_end - T_child_start
            total_flux = params.conduction_rate * delta_T * DT
            p_solid = p_info["occ_flat"]
            c_solid = c_info["occ_flat"]
            p_solid_mask = p_solid > 0.5 if p_solid is not None else torch.ones(p_info["cache_size"], dtype=torch.bool, device=gt_cache.device)
            c_solid_mask = c_solid > 0.5 if c_solid is not None else torch.ones(c_info["cache_size"], dtype=torch.bool, device=gt_cache.device)
            n_p_solid = max(int(p_solid_mask.sum()), 1)
            n_c_solid = max(int(c_solid_mask.sum()), 1)
            p_data[p_solid_mask] -= total_flux / n_p_solid
            c_data[c_solid_mask] += total_flux / n_c_solid
            p_data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)
            c_data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)

        # ── 4. Intra-link diffusion ──────────────────────────────────────
        alpha_d = params.diffusion_alpha
        if alpha_d > 0:
            for gi, info in voxel_cache_info.items():
                start = info["cache_start"]
                size = info["cache_size"]
                nx, ny, nz = info["grid_dims"]
                data = gt_cache[0, start : start + size].view(nx, ny, nz)
                lap = torch.zeros_like(data)
                lap[1:, :, :] += data[:-1, :, :]
                lap[:-1, :, :] += data[1:, :, :]
                lap[:, 1:, :] += data[:, :-1, :]
                lap[:, :-1, :] += data[:, 1:, :]
                lap[:, :, 1:] += data[:, :, :-1]
                lap[:, :, :-1] += data[:, :, 1:]
                data += alpha_d * (lap / nbr_counts[gi] - data)
                data.clamp_(min=AMBIENT_TEMP, max=MAX_TEMP)

        # ── Draw ────────────────────────────────────────────────────────
        for obj in debug_objs:
            scene.clear_debug_object(obj)
        debug_objs.clear()

        vis_min = params.vis_temp_min
        vis_max = params.vis_temp_max

        all_world_verts = []
        all_world_faces = []
        all_face_colors = []
        vert_offset = 0
        for local_idx, info in mesh_draw_info.items():
            sensor = info["sensor"]
            link = info["link"]
            base_mesh = info["base_trimesh"]
            vertex_to_voxel = info["vertex_to_voxel"]

            link_pos = tensor_to_array(link.get_pos(0)).reshape(3)
            link_quat = tensor_to_array(link.get_quat(0)).reshape(4)
            link_T = gu.trans_quat_to_T(link_pos, link_quat)
            world_verts = (link_T[:3, :3] @ base_mesh.vertices.T).T + link_T[:3, 3]

            temps = tensor_to_array(sensor.read_ground_truth(None)).reshape(-1)
            vertex_temps = temps[vertex_to_voxel]
            vertex_colors = temps_to_colors(vertex_temps, vis_min, vis_max)
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

        # Orbit camera (recording mode only)
        if not args.vis:
            cam_angle = 2 * math.pi * t / args.seconds
            cam_radius = 0.30
            cam_height = 0.22
            cam_x = cam_radius * math.cos(cam_angle)
            cam_y = cam_radius * math.sin(cam_angle)
            cam.set_pose(pos=(cam_x, cam_y, cam_height), lookat=(0, 0, 0.14))

        cam.render()

        if step % 200 == 0:
            summary = []
            for name in ["palm", "ff_base", "ff_proximal", "mf_proximal", "rf_proximal", "th_distal"]:
                try:
                    lk = allegro.get_link(name)
                    gi = lk.idx
                    if gi in voxel_cache_info:
                        info = voxel_cache_info[gi]
                        data = gt_cache[0, info["cache_start"] : info["cache_start"] + info["cache_size"]]
                        solid_m = info["occ_flat"] > 0.5 if info["occ_flat"] is not None else torch.ones(info["cache_size"], dtype=torch.bool, device=data.device)
                        summary.append(f"{name}={data[solid_m].mean().item():.1f}")
                except Exception:
                    pass
            print(f"  t={t:.1f}s: {' | '.join(summary)}")

        if "PYTEST_VERSION" in os.environ:
            break

        step += 1

    out_path = os.path.join(SCRIPT_DIR, "allegro_joint_heat.mp4")
    cam.stop_recording(save_to_filename=out_path, fps=60)
    print(f"\nSaved video: {out_path}")


if __name__ == "__main__":
    main()
