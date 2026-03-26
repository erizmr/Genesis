"""
Genesis + Allegro Right Hand — Headless Thermal Simulation + Video
==================================================================
Runs without a display. Renders frames via Genesis camera, composites
a thermal HUD overlay, then encodes to MP4 with ffmpeg.

Install:
    pip install genesis-world numpy opencv-python-headless
    apt install ffmpeg   (or brew install ffmpeg on mac)

Run:
    python allegro_thermal_video.py --xml right_hand.xml
    python allegro_thermal_video.py --xml right_hand.xml --duration 20 --fps 30 --res 1280 720
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import genesis as gs


# ─────────────────────────────────────────────────────────────────────
# Thermal constants  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────
K_TORQUE = 0.1
FRICTION  = 0.02
WARN, CRIT = 65.0, 80.0

JOINT_THERMAL = np.array([
    [1.5, 8.0,  0.8], [2.0, 10.0, 1.0], [1.8, 7.0, 0.9], [1.5, 5.0, 0.7],
    [1.5, 8.0,  0.8], [2.0, 10.0, 1.0], [1.8, 7.0, 0.9], [1.5, 5.0, 0.7],
    [1.5, 8.0,  0.8], [2.0, 10.0, 1.0], [1.8, 7.0, 0.9], [1.5, 5.0, 0.7],
    [2.0, 12.0, 1.0], [2.2, 10.0, 1.1], [2.0, 8.0,  1.0], [1.8, 5.0, 0.8],
], dtype=np.float32)

JOINT_NAMES = [
    "ffj0","ffj1","ffj2","ffj3",
    "mfj0","mfj1","mfj2","mfj3",
    "rfj0","rfj1","rfj2","rfj3",
    "thj0","thj1","thj2","thj3",
]
JOINT_TO_LINK = [
    "ff_base","ff_proximal","ff_medial","ff_distal",
    "mf_base","mf_proximal","mf_medial","mf_distal",
    "rf_base","rf_proximal","rf_medial","rf_distal",
    "th_base","th_proximal","th_medial","th_distal",
]
FINGER_LABELS = ["FF","FF","FF","FF","MF","MF","MF","MF",
                 "RF","RF","RF","RF","TH","TH","TH","TH"]
JOINT_LABELS  = ["Base","Prox","Med","Dist"] * 4

CTRL_RANGES = np.array([
    [-0.47,0.47],[-0.196,1.61],[-0.174,1.709],[-0.227,1.618],
    [-0.47,0.47],[-0.196,1.61],[-0.174,1.709],[-0.227,1.618],
    [-0.47,0.47],[-0.196,1.61],[-0.174,1.709],[-0.227,1.618],
    [0.263,1.396],[-0.105,1.163],[-0.189,1.644],[-0.162,1.719],
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# Thermal model
# ─────────────────────────────────────────────────────────────────────
class ThermalModel:
    def __init__(self, t_ambient=25.0, cooling=1.0):
        self.t_amb = t_ambient
        self.temps = np.full(16, t_ambient, dtype=np.float32)
        self.R = JOINT_THERMAL[:, 0]
        self.C = JOINT_THERMAL[:, 1]
        self.k = JOINT_THERMAL[:, 2] * cooling

    def step(self, torques, velocities, dt):
        I      = np.abs(torques) / K_TORQUE
        P_heat = I**2 * self.R + np.abs(torques) * np.abs(velocities) * FRICTION
        dT     = (P_heat - self.k * (self.temps - self.t_amb)) / self.C * dt
        self.temps = np.clip(self.temps + dT, self.t_amb, 110.0)
        return self.temps.copy()


# ─────────────────────────────────────────────────────────────────────
# Color utils
# ─────────────────────────────────────────────────────────────────────
def temp_to_rgb_float(t: float):
    """Single temperature → (R,G,B) floats 0–1."""
    n = float(np.clip((t - 20.0) / 75.0, 0, 1))
    if n < 0.5:
        f = n / 0.5
        return (0.1 + f*0.9, 0.55 - f*0.15, 1.0 - f*0.85)
    f = (n - 0.5) / 0.5
    return (1.0, 0.4 - f*0.4, 0.15 - f*0.15)

def temp_to_bgr255(t: float):
    """Temperature → BGR uint8 for OpenCV."""
    r, g, b = temp_to_rgb_float(t)
    return (int(b*255), int(g*255), int(r*255))

def temp_to_rgba_array(temps: np.ndarray) -> np.ndarray:
    """(16,) → (16,4) RGBA float32 for Genesis geom coloring."""
    n = np.clip((temps - 20.0) / 75.0, 0, 1)
    rgba = np.zeros((16, 4), dtype=np.float32)
    rgba[:, 3] = 1.0
    lo, hi = n < 0.5, n >= 0.5
    f_lo = n[lo] / 0.5
    f_hi = (n[hi] - 0.5) / 0.5
    rgba[lo, 0] = 0.1 + f_lo*0.9;  rgba[lo, 1] = 0.55 - f_lo*0.15; rgba[lo, 2] = 1.0 - f_lo*0.85
    rgba[hi, 0] = 1.0;              rgba[hi, 1] = 0.4  - f_hi*0.4;  rgba[hi, 2] = 0.15 - f_hi*0.15
    return rgba


# ─────────────────────────────────────────────────────────────────────
# Genesis geom color overlay
# ─────────────────────────────────────────────────────────────────────
def apply_heat_colors(robot, temps: np.ndarray):
    rgba = temp_to_rgba_array(temps)
    for i, link_name in enumerate(JOINT_TO_LINK):
        try:
            link = robot.get_link(link_name)
            for geom in link.geoms:
                geom.set_rgba(rgba[i])
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# HUD overlay (drawn on top of rendered frame with OpenCV)
# ─────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, temps: np.ndarray, sim_time: float) -> np.ndarray:
    """
    Composites a thermal HUD onto the rendered RGB frame.
    frame: H×W×3 uint8 (RGB from Genesis camera)
    Returns: H×W×3 uint8 BGR (ready for VideoWriter)
    """
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    # ── semi-transparent dark panel (top-left) ──
    panel_w, panel_h = 230, 230
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8+panel_w, 8+panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # ── title ──
    cv2.putText(img, "Joint temperatures", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    # ── 16-joint grid (4 fingers × 4 joints) ──
    cell_w, cell_h = 52, 46
    ox, oy = 12, 36
    finger_cols = [(92,160,220),(80,200,140),(80,160,240),(120,100,220)]  # BGR per finger

    for fi in range(4):
        for ji in range(4):
            idx   = fi*4 + ji
            t     = temps[idx]
            cx    = ox + fi * cell_w
            cy    = oy + ji * cell_h
            color = temp_to_bgr255(t)
            # cell background
            cv2.rectangle(img, (cx, cy), (cx+cell_w-3, cy+cell_h-4), color, -1)
            cv2.rectangle(img, (cx, cy), (cx+cell_w-3, cy+cell_h-4), (50,50,50), 1)
            # temperature text
            txt_col = (20,20,20) if t < 60 else (255,255,255)
            cv2.putText(img, f"{t:.0f}deg", (cx+4, cy+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, txt_col, 1, cv2.LINE_AA)
            # joint label
            lbl = JOINT_LABELS[ji]
            cv2.putText(img, lbl, (cx+4, cy+32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, txt_col, 1, cv2.LINE_AA)
            # warning / critical icon
            if t >= CRIT:
                cv2.putText(img, "!!", (cx+cell_w-16, cy+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,255), 1, cv2.LINE_AA)
            elif t >= WARN:
                cv2.putText(img, "!", (cx+cell_w-10, cy+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,165,255), 1, cv2.LINE_AA)

    # finger column headers (below grid)
    for fi, lbl in enumerate(["FF","MF","RF","TH"]):
        cx = ox + fi * cell_w + cell_w//2
        cv2.putText(img, lbl, (cx-8, oy + 4*cell_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1, cv2.LINE_AA)

    # ── color bar (bottom) ──
    bar_x, bar_y, bar_w, bar_h = 8, H-28, W-16, 12
    for px in range(bar_w):
        t_px = 20 + (px / bar_w) * 75
        col  = temp_to_bgr255(t_px)
        cv2.line(img, (bar_x+px, bar_y), (bar_x+px, bar_y+bar_h), col, 1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80,80,80), 1)
    # threshold markers on bar
    for thresh, col in [(WARN, (0,165,255)), (CRIT, (0,0,220))]:
        tx = bar_x + int((thresh-20)/75 * bar_w)
        cv2.line(img, (tx, bar_y-4), (tx, bar_y+bar_h+4), col, 2)
    cv2.putText(img, "20C", (bar_x, H-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(img, "95C", (bar_x+bar_w-28, H-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,150,150), 1, cv2.LINE_AA)

    # ── sim time (top-right) ──
    ts = f"t = {sim_time:.2f}s"
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(img, ts, (W-tw-10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

    # ── hottest joint label (top-right) ──
    hot_i   = int(np.argmax(temps))
    hot_lbl = f"Hot: {FINGER_LABELS[hot_i]} {JOINT_LABELS[hot_i%4]} {temps[hot_i]:.1f}C"
    col     = (0,0,220) if temps[hot_i] >= CRIT else (0,140,255) if temps[hot_i] >= WARN else (80,200,80)
    cv2.putText(img, hot_lbl, (W-200, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────────────────────────────
# Motion controller
# ─────────────────────────────────────────────────────────────────────
def get_ctrl(t: float, speed: float = 1.0) -> np.ndarray:
    mids   = (CTRL_RANGES[:, 0] + CTRL_RANGES[:, 1]) / 2
    amps   = (CTRL_RANGES[:, 1] - CTRL_RANGES[:, 0]) * 0.35
    phases = np.arange(16) * np.pi / 8
    return mids + amps * np.sin(2 * np.pi * 0.5 * speed * t + phases)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def run(xml_path: str, out_video: str, fps: int, res: tuple,
        duration: float, t_ambient: float, cooling: float, speed: float):

    W, H = res
    sim_dt   = 0.002              # 500 Hz physics
    frame_dt = 1.0 / fps          # seconds between captured frames

    # ── Genesis (headless — no show_viewer) ──
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=False,        # <-- headless
        sim_options=gs.options.SimOptions(dt=sim_dt),
        rigid_options=gs.options.RigidOptions(gravity=(0, 0, -9.81)),
        renderer=gs.renderers.Rasterizer(),   # fast rasterizer for video
    )

    robot = scene.add_entity(gs.morphs.MJCF(file=xml_path))

    # Attach a camera — orbit angle chosen to show the hand well
    cam = scene.add_camera(
        res=(W, H),
        pos=(0.25, -0.35, 0.20),
        lookat=(0.0, 0.0, 0.0),
        fov=42,
        GUI=False,
    )

    scene.build()

    thermal = ThermalModel(t_ambient=t_ambient, cooling=cooling)

    # ── Temp frame dir ──
    frame_dir = Path(tempfile.mkdtemp(prefix="genesis_thermal_"))
    print(f"Writing frames to {frame_dir}")

    sim_time   = 0.0
    frame_idx  = 0
    next_frame = 0.0

    total_frames = int(duration * fps)
    print(f"Rendering {total_frames} frames @ {fps} fps  ({duration}s)")

    while sim_time < duration:

        # Control
        ctrl = get_ctrl(sim_time, speed)
        dof_idxs = [robot.get_joint(n).dof_idx_local for n in JOINT_NAMES]
        robot.set_dofs_position(ctrl, dof_idxs)

        # Physics step
        scene.step()

        # Thermal step
        torques    = robot.get_dofs_force().cpu().numpy()
        velocities = robot.get_dofs_vel().cpu().numpy()
        temps      = thermal.step(torques, velocities, sim_dt)

        # Apply heat colors to geoms
        apply_heat_colors(robot, temps)

        # Capture frame at desired fps
        if sim_time >= next_frame:
            cam.set_pose(
                pos=(
                    0.25 * np.cos(sim_time * 0.3),
                    -0.35 + 0.05 * np.sin(sim_time * 0.2),
                    0.20,
                ),
                lookat=(0.0, 0.0, 0.0),
            )

            rgb = cam.render(rgb=True)        # returns H×W×3 uint8

            # Composite HUD
            bgr = draw_hud(rgb, temps, sim_time)

            # Save frame
            frame_path = frame_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), bgr)

            frame_idx  += 1
            next_frame += frame_dt

            if frame_idx % fps == 0:
                print(f"  {sim_time:.1f}s / {duration:.1f}s  "
                      f"({frame_idx}/{total_frames} frames)  "
                      f"hottest={np.max(temps):.1f}C")

        sim_time += sim_dt

    print(f"\nEncoding {frame_idx} frames → {out_video}")

    # ── ffmpeg encode ──
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",                 # quality: 0=lossless, 23=default, 51=worst
        "-pix_fmt", "yuv420p",        # required for broad player compatibility
        "-vf", f"scale={W}:{H}",
        out_video,
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ffmpeg error:", result.stderr)
    else:
        size_mb = Path(out_video).stat().st_size / 1e6
        print(f"Video saved: {out_video}  ({size_mb:.1f} MB)")

    shutil.rmtree(frame_dir)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml",      default="right_hand.xml",
                   help="Path to Allegro MJCF file")
    p.add_argument("--out",      default="allegro_thermal.mp4",
                   help="Output video path")
    p.add_argument("--fps",      type=int,   default=30)
    p.add_argument("--res",      type=int,   nargs=2, default=[1280, 720],
                   metavar=("W","H"))
    p.add_argument("--duration", type=float, default=20.0,
                   help="Simulation duration (seconds)")
    p.add_argument("--ambient",  type=float, default=25.0)
    p.add_argument("--cooling",  type=float, default=1.0,
                   help="1=none  2=natural  4=forced  8=liquid")
    p.add_argument("--speed",    type=float, default=1.0,
                   help="Motion speed multiplier")
    p.add_argument("--cpu",      action="store_true",
                   help="Use CPU backend (slow but no GPU needed)")
    args = p.parse_args()

    # swap gs.gpu → gs.cpu if requested (patch before gs.init)
    if args.cpu:
        import genesis as gs_patch
        gs_patch.gpu = gs_patch.cpu

    run(
        xml_path  = args.xml,
        out_video = args.out,
        fps       = args.fps,
        res       = tuple(args.res),
        duration  = args.duration,
        t_ambient = args.ambient,
        cooling   = args.cooling,
        speed     = args.speed,
    )