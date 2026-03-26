"""Replay a recorded locomotion episode with thermal simulation and visualization.

Computes joint heat from motor torque/velocity during replay, applies thermal
coloring to robot links, and composites a HUD overlay showing all 29 joint
temperatures.

Usage:
    python replay_heat.py -f dataset1.npz --rec -o replay1_heat.mp4
    python replay_heat.py -f dataset1.npz --vis
"""

import os
import shutil
import subprocess
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

import genesis as gs
from genesis.utils import geom as gu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
G1_MJCF = os.path.join(SCRIPT_DIR, "assets", "g1_29dof_rev_1_0.xml")

# Eden DOF ordering (from dataset)
EDEN_DOF_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Joint group labels for HUD (eden order)
JOINT_GROUPS = {
    "L.Hip": [0, 1, 2],
    "L.Knee": [3],
    "L.Ankle": [4, 5],
    "R.Hip": [6, 7, 8],
    "R.Knee": [9],
    "R.Ankle": [10, 11],
    "Waist": [12, 13, 14],
    "L.Shldr": [15, 16, 17],
    "L.Elbow": [18],
    "L.Wrist": [19, 20, 21],
    "R.Shldr": [22, 23, 24],
    "R.Elbow": [25],
    "R.Wrist": [26, 27, 28],
}

# Short joint labels (eden order)
JOINT_SHORT = [
    "LHP", "LHR", "LHY", "LKn", "LAP", "LAR",
    "RHP", "RHR", "RHY", "RKn", "RAP", "RAR",
    "WY", "WR", "WP",
    "LSP", "LSR", "LSY", "LEl", "LWR", "LWP", "LWY",
    "RSP", "RSR", "RSY", "REl", "RWR", "RWP", "RWY",
]

# ---------------------------------------------------------------------------
# Thermal constants (per joint in eden order)
# Columns: R (winding resistance), C (thermal capacitance), k (cooling coeff)
# Grouped by actuator size: legs > waist > shoulders > elbows > wrists
# Columns: R (winding resistance), C (thermal capacitance), k (cooling coeff)
# ---------------------------------------------------------------------------
# fmt: off
JOINT_THERMAL = np.array([
    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    [2.5, 15.0, 1.2],  [2.0, 12.0, 1.0],  [1.8, 12.0, 1.0],
    [3.0, 18.0, 1.5],  [1.5, 10.0, 0.8],  [1.2, 8.0,  0.7],
    # Right leg: same pattern
    [2.5, 15.0, 1.2],  [2.0, 12.0, 1.0],  [1.8, 12.0, 1.0],
    [3.0, 18.0, 1.5],  [1.5, 10.0, 0.8],  [1.2, 8.0,  0.7],
    # Waist: yaw, roll, pitch
    [1.5, 20.0, 1.0],  [1.5, 20.0, 1.0],  [1.8, 20.0, 1.0],
    # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    [1.5, 10.0, 0.9],  [1.5, 10.0, 0.9],  [1.2, 8.0,  0.8],
    [1.8, 8.0,  1.0],
    # Left wrist: roll, pitch, yaw
    [0.8, 6.0,  0.8],  [0.8, 6.0,  0.8],  [0.8, 6.0,  0.8],
    # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    [1.5, 10.0, 0.9],  [1.5, 10.0, 0.9],  [1.2, 8.0,  0.8],
    [1.8, 8.0,  1.0],
    # Right wrist: roll, pitch, yaw
    [0.8, 6.0,  0.8],  [0.8, 6.0,  0.8],  [0.8, 6.0,  0.8],
], dtype=np.float32)
# fmt: on

T_AMBIENT = 25.0
WARN_TEMP = 55.0
CRIT_TEMP = 75.0
MAX_TEMP = 100.0

# Heat generation scale: maps (effort^2) to watts per joint group.
# effort = |action - current_pos| (radians) + velocity_contribution
# Tuned so locomotion reaches ~60-80C over 14s.
HEAT_SCALE = np.array([
    # Left leg: hip(p,r,y), knee, ankle(p,r)
    12.0, 8.0, 6.0, 18.0, 8.0, 5.0,
    # Right leg
    12.0, 8.0, 6.0, 18.0, 8.0, 5.0,
    # Waist
    5.0, 5.0, 6.0,
    # Left arm: shoulder(p,r,y), elbow, wrist(r,p,y)
    4.0, 4.0, 3.0, 5.0, 0.5, 0.5, 0.5,
    # Right arm
    4.0, 4.0, 3.0, 5.0, 0.5, 0.5, 0.5,
], dtype=np.float32)

VEL_HEAT_SCALE = 0.3  # contribution of |velocity| to heating


# ---------------------------------------------------------------------------
# Thermal model
# ---------------------------------------------------------------------------
class ThermalModel:
    """Standalone motor thermal model for 29 DOFs.

    Uses effort-based heating: effort = |action - current_pos| + vel_scale * |vel|.
    P_heat = HEAT_SCALE * effort^2
    dT = (P_heat - k * (T - T_amb)) / C * dt
    """

    def __init__(self, t_ambient=T_AMBIENT, cooling=1.0):
        self.t_amb = t_ambient
        self.temps = np.full(29, t_ambient, dtype=np.float32)
        self.C = JOINT_THERMAL[:, 1]       # thermal capacitance
        self.k = JOINT_THERMAL[:, 2] * cooling  # cooling coefficient

    def step(self, action, qpos_dofs, velocities, dt):
        """Update temperatures given action targets, current positions, and velocities."""
        effort = np.abs(action - qpos_dofs) + VEL_HEAT_SCALE * np.abs(velocities)
        P_heat = HEAT_SCALE * effort**2
        dT = (P_heat - self.k * (self.temps - self.t_amb)) / self.C * dt
        self.temps = np.clip(self.temps + dT, self.t_amb, MAX_TEMP)
        return self.temps.copy()


# ---------------------------------------------------------------------------
# DOF remapping (eden <-> genesis)
# ---------------------------------------------------------------------------
def build_dof_remap(robot):
    """Build index map: remap[i] = j means genesis DOF i gets value from eden DOF j."""
    genesis_dof_order = []
    for joint in robot.joints:
        if joint.n_dofs == 1:
            genesis_dof_order.append(joint.name)
    return [EDEN_DOF_ORDER.index(name) for name in genesis_dof_order]


def build_eden_remap(robot):
    """Build inverse: eden_remap[i] = j means eden DOF i gets value from genesis DOF j."""
    genesis_dof_order = []
    for joint in robot.joints:
        if joint.n_dofs == 1:
            genesis_dof_order.append(joint.name)
    return [genesis_dof_order.index(name) for name in EDEN_DOF_ORDER]


def remap_qpos(qpos_eden, remap):
    """Remap qpos from eden to genesis order. qpos: [pos(3), quat(4), dofs(29)]."""
    root = qpos_eden[..., :7]
    dofs_eden = qpos_eden[..., 7:]
    dofs_genesis = dofs_eden[..., remap]
    return torch.cat([root, dofs_genesis], dim=-1)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------
def temp_to_rgb(t, t_min=T_AMBIENT, t_max=95.0):
    """Temperature -> (R, G, B) floats in [0,1]. Blue -> Yellow -> Red."""
    n = float(np.clip((t - t_min) / (t_max - t_min), 0, 1))
    if n < 0.5:
        f = n / 0.5
        return (0.1 + f * 0.9, 0.4 + f * 0.2, 1.0 - f * 0.7)
    f = (n - 0.5) / 0.5
    return (1.0, 0.6 - f * 0.6, 0.3 - f * 0.3)


def temp_to_bgr255(t, t_min=T_AMBIENT, t_max=95.0):
    r, g, b = temp_to_rgb(t, t_min, t_max)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Per-vertex thermal coloring via debug mesh overlay
# ---------------------------------------------------------------------------
def temps_to_face_colors(vertex_temps, t_min=T_AMBIENT, t_max=95.0):
    """Convert per-vertex temperatures to RGBA uint8 array. Blue -> Yellow -> Red."""
    n = np.clip((vertex_temps - t_min) / (t_max - t_min), 0, 1)
    rgba = np.zeros((len(n), 4), dtype=np.uint8)
    rgba[:, 3] = 255
    lo = n < 0.5
    hi = ~lo
    f_lo = n[lo] / 0.5
    f_hi = (n[hi] - 0.5) / 0.5
    rgba[lo, 0] = (255 * (0.1 + f_lo * 0.9)).astype(np.uint8)
    rgba[lo, 1] = (255 * (0.4 + f_lo * 0.2)).astype(np.uint8)
    rgba[lo, 2] = (255 * (1.0 - f_lo * 0.7)).astype(np.uint8)
    rgba[hi, 0] = 255
    rgba[hi, 1] = (255 * (0.6 - f_hi * 0.6)).astype(np.uint8)
    rgba[hi, 2] = (255 * (0.3 - f_hi * 0.3)).astype(np.uint8)
    return rgba


def build_thermal_mesh_info(robot):
    """Precompute per-link mesh data and vertex-to-joint temperature weights.

    For each link driven by a 1-DOF joint, we collect the mesh vertices in link-local
    frame, then compute a blend weight per vertex: vertices near the joint origin
    get the parent's temperature, vertices far from the origin get this joint's
    temperature. This creates a smooth gradient across each link.

    Returns list of dicts with keys:
        eden_idx, link, local_verts, faces, blend_weights, parent_eden_idx
    """
    # Build joint -> eden index and parent chain
    joint_to_eden = {}
    for i, name in enumerate(EDEN_DOF_ORDER):
        joint_to_eden[name] = i

    # Build link_idx -> eden_idx for all actuated links
    link_idx_to_eden = {}
    for joint in robot.joints:
        if joint.n_dofs == 1 and joint.name in joint_to_eden:
            link_idx_to_eden[joint.link.idx] = joint_to_eden[joint.name]

    # Build parent eden idx for each link (walk up kinematic chain)
    link_parent_eden = {}
    for joint in robot.joints:
        if joint.n_dofs == 1 and joint.name in joint_to_eden:
            eden_idx = joint_to_eden[joint.name]
            # Find parent: walk up the kinematic chain
            parent_link = joint.link
            parent_eden = None
            if parent_link.parent_idx >= 0:
                # Check if parent link has an eden mapping
                pid = parent_link.parent_idx
                if pid in link_idx_to_eden:
                    parent_eden = link_idx_to_eden[pid]
                else:
                    # Walk further up (e.g., pelvis -> hip)
                    for lk in robot.links:
                        if lk.idx == pid and lk.parent_idx >= 0:
                            if lk.parent_idx in link_idx_to_eden:
                                parent_eden = link_idx_to_eden[lk.parent_idx]
            link_parent_eden[eden_idx] = parent_eden

    mesh_infos = []
    for joint in robot.joints:
        if joint.n_dofs != 1 or joint.name not in joint_to_eden:
            continue

        link = joint.link
        eden_idx = joint_to_eden[joint.name]
        parent_eden = link_parent_eden.get(eden_idx)

        # Collect all visual geometry for this link in link-local frame
        all_verts = []
        all_faces = []
        vert_offset = 0
        for vg in link.vgeoms:
            vverts = vg._init_vverts
            vfaces = vg._init_vfaces
            local_verts = gu.transform_by_trans_quat(vverts, vg._init_pos, vg._init_quat)
            all_verts.append(local_verts)
            all_faces.append(vfaces + vert_offset)
            vert_offset += len(local_verts)

        if not all_verts:
            continue

        combined_verts = np.concatenate(all_verts, axis=0)
        combined_faces = np.concatenate(all_faces, axis=0)

        # Compute blend weights: 0 = joint origin (parent temp), 1 = far end (own temp)
        # Distance from origin along the link's principal axis
        dists = np.linalg.norm(combined_verts, axis=1)
        max_dist = max(dists.max(), 1e-6)
        blend = np.clip(dists / max_dist, 0, 1)

        mesh_infos.append({
            "eden_idx": eden_idx,
            "parent_eden_idx": parent_eden,
            "link": link,
            "local_verts": combined_verts.astype(np.float64),
            "faces": combined_faces,
            "blend_weights": blend.astype(np.float32),
        })

    total_verts = sum(len(m["local_verts"]) for m in mesh_infos)
    print(f"Thermal mesh: {len(mesh_infos)} links, {total_verts} total vertices")
    return mesh_infos


def build_thermal_debug_mesh(mesh_infos, temps, robot):
    """Build a single combined trimesh with per-face thermal colors.

    Each vertex temperature is interpolated between the parent joint temp and
    this joint's temp using precomputed blend weights, creating a smooth gradient.
    """
    all_world_verts = []
    all_faces = []
    all_face_colors = []
    vert_offset = 0

    for info in mesh_infos:
        link = info["link"]
        eden_idx = info["eden_idx"]
        parent_eden = info["parent_eden_idx"]
        blend = info["blend_weights"]

        # Get link world transform
        link_pos = link.get_pos().cpu().numpy().reshape(3).astype(np.float64)
        link_quat = link.get_quat().cpu().numpy().reshape(4).astype(np.float64)
        link_T = gu.trans_quat_to_T(link_pos, link_quat)

        # Transform vertices to world frame
        local_verts = info["local_verts"]
        world_verts = (link_T[:3, :3] @ local_verts.T).T + link_T[:3, 3]

        # Interpolate temperature per vertex
        # Near joint origin (blend~0): parent joint temperature
        # Far end (blend~1): this joint's own temperature
        own_temp = temps[eden_idx]
        parent_temp = temps[parent_eden] if parent_eden is not None else T_AMBIENT
        vertex_temps = parent_temp * (1 - blend) + own_temp * blend

        # Per-face color (average of 3 vertex colors)
        vertex_rgba = temps_to_face_colors(vertex_temps)
        face_colors = vertex_rgba[info["faces"]].mean(axis=1).astype(np.uint8)

        all_world_verts.append(world_verts)
        all_faces.append(info["faces"] + vert_offset)
        all_face_colors.append(face_colors)
        vert_offset += len(world_verts)

    if not all_world_verts:
        return None

    combined = trimesh.Trimesh(
        vertices=np.concatenate(all_world_verts),
        faces=np.concatenate(all_faces),
        process=False,
    )
    combined.visual.face_colors = np.concatenate(all_face_colors)
    return combined


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------
def draw_hud(frame, temps, sim_time):
    """Composite thermal HUD onto RGB frame. Returns BGR image for video."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    # Semi-transparent panel (left side)
    panel_w, panel_h = 260, 360
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Title
    cv2.putText(img, "Joint Temperatures (C)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Layout: body part groups in rows
    y = 44
    groups_layout = [
        ("LEFT LEG", [0, 1, 2, 3, 4, 5]),
        ("RIGHT LEG", [6, 7, 8, 9, 10, 11]),
        ("WAIST", [12, 13, 14]),
        ("LEFT ARM", [15, 16, 17, 18, 19, 20, 21]),
        ("RIGHT ARM", [22, 23, 24, 25, 26, 27, 28]),
    ]

    for group_name, indices in groups_layout:
        cv2.putText(img, group_name, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
        y += 16

        x = 14
        for idx in indices:
            t = temps[idx]
            color = temp_to_bgr255(t)
            # Cell
            cw, ch = 36, 32
            cv2.rectangle(img, (x, y), (x + cw - 2, y + ch - 2), color, -1)
            cv2.rectangle(img, (x, y), (x + cw - 2, y + ch - 2), (50, 50, 50), 1)
            # Temperature
            txt_col = (20, 20, 20) if t < 60 else (255, 255, 255)
            cv2.putText(img, f"{t:.0f}", (x + 3, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, txt_col, 1, cv2.LINE_AA)
            # Joint label
            short = JOINT_SHORT[idx]
            cv2.putText(img, short[-2:], (x + 3, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.24, txt_col, 1, cv2.LINE_AA)
            # Warning markers
            if t >= CRIT_TEMP:
                cv2.putText(img, "!", (x + cw - 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1, cv2.LINE_AA)
            elif t >= WARN_TEMP:
                cv2.putText(img, "!", (x + cw - 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 165, 255), 1, cv2.LINE_AA)
            x += cw

        y += ch + 6

    # Color bar (bottom of panel)
    bar_x, bar_y, bar_w, bar_h = 14, y + 4, panel_w - 12, 12
    for px in range(bar_w):
        t_px = T_AMBIENT + (px / bar_w) * (95 - T_AMBIENT)
        col = temp_to_bgr255(t_px)
        cv2.line(img, (bar_x + px, bar_y), (bar_x + px, bar_y + bar_h), col, 1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    # Threshold markers
    for thresh, col in [(WARN_TEMP, (0, 165, 255)), (CRIT_TEMP, (0, 0, 220))]:
        tx = bar_x + int((thresh - T_AMBIENT) / (95 - T_AMBIENT) * bar_w)
        cv2.line(img, (tx, bar_y - 3), (tx, bar_y + bar_h + 3), col, 2)
    cv2.putText(img, f"{T_AMBIENT:.0f}C", (bar_x, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, "95C", (bar_x + bar_w - 24, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)

    # Sim time (top-right)
    ts = f"t = {sim_time:.2f}s"
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(img, ts, (W - tw - 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Hottest joint (top-right below time)
    hot_i = int(np.argmax(temps))
    hot_t = temps[hot_i]
    hot_lbl = f"Peak: {JOINT_SHORT[hot_i]} {hot_t:.1f}C"
    col = (0, 0, 220) if hot_t >= CRIT_TEMP else (0, 140, 255) if hot_t >= WARN_TEMP else (80, 200, 80)
    cv2.putText(img, hot_lbl, (W - 180, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Replay locomotion with thermal visualization")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to NPZ dataset file")
    parser.add_argument("--episode", "-e", type=str, default="demo_0", help="Episode name")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("--vis", "-v", action="store_true", help="Show interactive viewer")
    parser.add_argument("--rec", "-r", action="store_true", help="Record video")
    parser.add_argument("--output", "-o", type=str, default="replay_heat.mp4", help="Output video file")
    parser.add_argument("--fps", type=int, default=50, help="Output video FPS (default: 50 = dataset rate)")
    parser.add_argument("--res", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
    parser.add_argument("--cooling", type=float, default=1.0, help="Cooling multiplier (1=default, 2=more cooling)")
    args = parser.parse_args()

    # Load dataset
    data = np.load(args.file, allow_pickle=True)
    ep = args.episode

    if f"{ep}/action" not in data:
        episodes = sorted(set(k.split("/")[0] for k in data.keys() if "/" in k and not k.startswith("__")))
        print(f"Episode '{ep}' not found. Available: {episodes}")
        return

    state_qpos_np = data[f"{ep}/state/qpos"]       # (T, 36) eden order
    action_np = data[f"{ep}/action"]                 # (T, 29) eden order
    dofs_vel_np = data[f"{ep}/state/dofs_vel"]       # (T, 29) eden order
    num_frames = state_qpos_np.shape[0]
    sim_dt = 0.02  # 50 Hz

    state_qpos = torch.from_numpy(state_qpos_np)

    # Init Genesis
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt),
        show_viewer=args.vis,
        viewer_options=gs.options.ViewerOptions(max_FPS=60),
        renderer=gs.renderers.Rasterizer(),
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=G1_MJCF))

    W, H = args.res
    cam = scene.add_camera(res=(W, H), pos=(2.0, -1.0, 1.2), lookat=(0.0, 0.0, 0.6), fov=50, GUI=False)

    scene.build()

    # Build DOF remapping
    remap = build_dof_remap(robot)  # eden -> genesis for qpos
    device = gs.device
    state_qpos = remap_qpos(state_qpos, remap).to(device)

    vis_context = scene._visualizer.context

    # Remove default visual meshes for actuated links — we draw our own colored meshes
    rigid_nodes = vis_context.rigid_nodes
    joint_link_idxs = set()
    for joint in robot.joints:
        if joint.n_dofs == 1:
            joint_link_idxs.add(joint.link.idx)

    for vgeom in robot.vgeoms:
        if vgeom._link.idx in joint_link_idxs:
            if vgeom.uid in rigid_nodes:
                node = rigid_nodes.pop(vgeom.uid)
                vis_context.remove_node(node)

    # Build per-vertex thermal mesh info (precompute blend weights)
    mesh_infos = build_thermal_mesh_info(robot)

    # Thermal model (operates in eden order — no remapping needed)
    thermal = ThermalModel(cooling=args.cooling)

    cam_offset = torch.tensor([2.0, -1.0, 0.4], device=device)

    if args.rec:
        frame_dir = Path(tempfile.mkdtemp(prefix="genesis_heat_"))
        print(f"Rendering {num_frames} frames @ {args.fps}fps ({num_frames * sim_dt:.1f}s)")

    frame_idx = 0
    debug_obj = None
    for i in tqdm(range(num_frames), desc="Replaying with thermal sim"):
        # Kinematic replay
        robot.set_qpos(state_qpos[i])
        scene.visualizer.update()

        # Thermal step (all in eden order — dataset native order)
        qpos_dofs_eden = state_qpos_np[i, 7:]
        action_eden = action_np[i]
        vel_eden = dofs_vel_np[i]
        temps = thermal.step(action_eden, qpos_dofs_eden, vel_eden, sim_dt)

        # Build and draw thermal-colored debug mesh
        if debug_obj is not None:
            scene.clear_debug_object(debug_obj)
        thermal_mesh = build_thermal_debug_mesh(mesh_infos, temps, robot)
        if thermal_mesh is not None:
            debug_obj = scene.draw_debug_mesh(thermal_mesh)

        # Render / record
        if args.rec:
            root_pos = state_qpos[i, :3]
            cam.set_pose(
                pos=root_pos + cam_offset,
                lookat=root_pos + torch.tensor([0.0, 0.0, 0.2], device=device),
            )
            rgb, _, _, _ = cam.render(rgb=True)
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            rgb = np.ascontiguousarray(rgb)

            bgr = draw_hud(rgb, temps, i * sim_dt)
            frame_path = frame_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), bgr)
            frame_idx += 1

            if (i + 1) % 100 == 0:
                hot_i = int(np.argmax(temps))
                print(f"  Frame {i+1}/{num_frames}  peak={temps[hot_i]:.1f}C ({JOINT_SHORT[hot_i]})")

        if args.vis:
            time.sleep(sim_dt)

    # Encode video
    if args.rec:
        out = args.output
        print(f"\nEncoding {frame_idx} frames -> {out}")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(frame_dir / "frame_%06d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            out,
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("ffmpeg error:", result.stderr)
        else:
            size_mb = Path(out).stat().st_size / 1e6
            print(f"Video saved: {out} ({size_mb:.1f} MB)")
        shutil.rmtree(frame_dir)


if __name__ == "__main__":
    main()
