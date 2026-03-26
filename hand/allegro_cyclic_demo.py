"""
Allegro Hand — Cyclic Finger Movement
======================================

Loads the Wonik Allegro right hand and drives all 16 finger joints through
smooth sinusoidal open/close cycles.  Each finger is phase-shifted so the
motion looks like a wave rippling across the hand.

Asset: mujoco_menagerie/wonik_allegro/right_hand.xml
  16 DOFs: 4 fingers x 4 joints (base, proximal, medial, distal)
    FFJ0-3  (first finger)
    MFJ0-3  (middle finger)
    RFJ0-3  (ring finger)
    THJ0-3  (thumb)

Usage:
    python allegro_cyclic_demo.py
"""

import math
import os

import numpy as np

import genesis as gs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLEGRO_PATH = os.path.join(SCRIPT_DIR, "right_hand.xml")

# Joint limits per DOF class (from the MJCF)
# Order: ff0 ff1 ff2 ff3  mf0 mf1 mf2 mf3  rf0 rf1 rf2 rf3  th0 th1 th2 th3
JOINT_LO = np.array([
    -0.47, -0.196, -0.174, -0.227,   # first finger
    -0.47, -0.196, -0.174, -0.227,   # middle finger
    -0.47, -0.196, -0.174, -0.227,   # ring finger
     0.263, -0.105, -0.189, -0.162,  # thumb
], dtype=np.float32)

JOINT_HI = np.array([
     0.47,  1.61,  1.709,  1.618,
     0.47,  1.61,  1.709,  1.618,
     0.47,  1.61,  1.709,  1.618,
     1.396, 1.163, 1.644,  1.719,
], dtype=np.float32)

JOINT_MID = (JOINT_LO + JOINT_HI) / 2
JOINT_AMP = (JOINT_HI - JOINT_LO) / 2

DT = 5e-3
TOTAL_STEPS = 600
CYCLE_PERIOD = 1.5  # seconds per open/close cycle


def main():
    gs.init(backend=gs.gpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=10),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            ambient_light=(0.3, 0.3, 0.3),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    allegro = scene.add_entity(gs.morphs.MJCF(
        file=ALLEGRO_PATH,
        pos=(0, 0, 0.15),
        euler=(0, -90, 0),  # fingers pointing right, palm facing camera
    ))

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.25, -0.12, 0.18),
        lookat=(0, 0, 0.14),
        fov=45,
        GUI=False,
    )

    scene.build()

    allegro.set_dofs_kp([5.0] * 16)
    allegro.set_dofs_kv([0.5] * 16)

    cam.start_recording()

    # Phase offsets: each finger is shifted by 1/4 cycle
    # Finger groups: FF(0-3), MF(4-7), RF(8-11), TH(12-15)
    finger_phase = np.zeros(16, dtype=np.float32)
    finger_phase[0:4]   = 0.0                        # first finger
    finger_phase[4:8]   = 0.25 * (2 * math.pi)       # middle finger
    finger_phase[8:12]  = 0.50 * (2 * math.pi)       # ring finger
    finger_phase[12:16] = 0.75 * (2 * math.pi)       # thumb

    for step in range(TOTAL_STEPS):
        t = step * DT
        omega = 2 * math.pi / CYCLE_PERIOD

        # Sinusoidal targets — each joint oscillates within its limits
        phase = omega * t + finger_phase
        # Use 0.8 amplitude to stay safely within limits
        target = JOINT_MID + 0.8 * JOINT_AMP * np.sin(phase)

        allegro.control_dofs_position(target)
        scene.step()
        cam.render()

        if step % 100 == 0:
            print(f"  step {step}/{TOTAL_STEPS}  t={t:.2f}s")

    out_path = os.path.join(SCRIPT_DIR, "allegro_cyclic.mp4")
    cam.stop_recording(save_to_filename=out_path, fps=60)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
