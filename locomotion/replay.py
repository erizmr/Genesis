"""Standalone replay of a recorded locomotion episode from an NPZ dataset.

No eden dependency — uses genesis directly.

Usage:
    python replay.py -f dataset.npz --vis
    python replay.py -f dataset.npz --rec -o replay.mp4
    python replay.py -f dataset.npz --episode demo_0 --vis
"""

import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

import genesis as gs

# Path to the G1 29-DOF MJCF asset (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
G1_MJCF = os.path.join(SCRIPT_DIR, "assets", "g1_29dof_rev_1_0.xml")

# Eden stores DOFs in this order (from G1_29Dof.dofs_name):
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


def build_dof_remap(robot):
    """Build index map to remap DOFs from eden order to genesis (BFS) order.

    Returns a list where remap[i] = j means genesis DOF i gets value from eden DOF j.
    """
    genesis_dof_order = []
    for joint in robot.joints:
        if joint.n_dofs == 1:
            genesis_dof_order.append(joint.name)
    remap = [EDEN_DOF_ORDER.index(name) for name in genesis_dof_order]
    return remap


def remap_qpos(qpos_eden, remap):
    """Remap qpos from eden order to genesis order.

    qpos layout: [pos(3), quat(4), dofs(29)] = 36 total.
    Only the 29 DOF columns need remapping.
    """
    root = qpos_eden[..., :7]
    dofs_eden = qpos_eden[..., 7:]
    dofs_genesis = dofs_eden[..., remap]
    return torch.cat([root, dofs_genesis], dim=-1)


def main():
    parser = ArgumentParser(description="Replay a recorded locomotion episode (standalone)")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to NPZ dataset file")
    parser.add_argument("--episode", "-e", type=str, default="demo_0", help="Episode name to replay")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("--vis", "-v", action="store_true", help="Show viewer")
    parser.add_argument("--rec", "-r", action="store_true", help="Record video")
    parser.add_argument("--output", "-o", type=str, default="replay.mp4", help="Output video filename")
    args = parser.parse_args()

    # Load episode data
    data = np.load(args.file, allow_pickle=True)
    ep = args.episode

    action_key = f"{ep}/action"
    if action_key not in data:
        episodes = sorted(set(k.split("/")[0] for k in data.keys() if "/" in k and not k.startswith("__")))
        print(f"Episode '{ep}' not found. Available: {episodes}")
        return

    state_qpos = torch.from_numpy(data[f"{ep}/state/qpos"])  # (T, 36)
    num_frames = state_qpos.shape[0]

    # Init genesis
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02),
        show_viewer=args.vis,
        viewer_options=gs.options.ViewerOptions(max_FPS=60),
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Robot
    robot = scene.add_entity(
        gs.morphs.MJCF(file=G1_MJCF),
    )

    # Camera for recording (offset relative to robot root)
    cam_offset = torch.tensor([2.0, -1.0, 0.4])  # relative to root pos
    if args.rec:
        cam = scene.add_camera(
            res=(960, 720),
            pos=(2.0, -1.0, 1.2),
            lookat=(0.0, 0.0, 0.6),
            fov=50,
        )

    scene.build()

    device = gs.device

    # Build DOF remapping from eden order -> genesis BFS order
    remap = build_dof_remap(robot)
    state_qpos = remap_qpos(state_qpos, remap).to(device)
    cam_offset = cam_offset.to(device)

    if args.rec:
        cam.start_recording()

    for i in tqdm(range(num_frames), desc="Replaying"):
        robot.set_qpos(state_qpos[i])
        scene.visualizer.update()

        if args.rec:
            root_pos = state_qpos[i, :3]
            cam.set_pose(
                pos=root_pos + cam_offset,
                lookat=root_pos + torch.tensor([0.0, 0.0, 0.2], device=device),
            )
            cam.render()

        if args.vis:
            time.sleep(0.02)

    if args.rec:
        cam.stop_recording(args.output)
        print(f"Saved recording to {args.output}")


if __name__ == "__main__":
    main()
