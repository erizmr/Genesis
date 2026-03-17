"""
Allegro Hand — Temperature Sensing Demo
========================================

The Allegro hand falls onto a hot platform and curls its fingers.
A TemperatureGrid sensor on the platform shows heat flow as the
hand's cooler links contact the hot surface (blue = cool, red = hot).

Renders a video to hand/allegro_temperature.mp4.

Usage:
    python allegro_temperature_demo.py              # headless, renders video
    python allegro_temperature_demo.py --vis         # interactive viewer
    python allegro_temperature_demo.py --cpu         # CPU backend
"""

import argparse
import os

import numpy as np

import genesis as gs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLEGRO_PATH = os.path.join(SCRIPT_DIR, "right_hand.xml")

# ── Simulation ──────────────────────────────────────────────────────────
DT = 5e-3
SUBSTEPS = 10
TOTAL_SECONDS = 4.0

# ── Geometry ────────────────────────────────────────────────────────────
PLATFORM_SIZE = 0.3
PLATFORM_HEIGHT = 0.3

# ── Temperature ─────────────────────────────────────────────────────────
GRID_SIZE = (10, 10, 1)
BASE_TEMP = 22.0  # °C (room temperature, for the hand)
HOT_TEMP = 122.0  # °C (platform)

# ── Joint limits from MJCF ──────────────────────────────────────────────
JOINT_LO = np.array(
    [
        -0.47,
        -0.196,
        -0.174,
        -0.227,
        -0.47,
        -0.196,
        -0.174,
        -0.227,
        -0.47,
        -0.196,
        -0.174,
        -0.227,
        0.263,
        -0.105,
        -0.189,
        -0.162,
    ],
    dtype=np.float32,
)
JOINT_HI = np.array(
    [
        0.47,
        1.61,
        1.709,
        1.618,
        0.47,
        1.61,
        1.709,
        1.618,
        0.47,
        1.61,
        1.709,
        1.618,
        1.396,
        1.163,
        1.644,
        1.719,
    ],
    dtype=np.float32,
)
JOINT_MID = (JOINT_LO + JOINT_HI) / 2
JOINT_AMP = (JOINT_HI - JOINT_LO) / 2


def main():
    parser = argparse.ArgumentParser(description="Allegro hand temperature sensing demo")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--seconds", "-t", type=float, default=TOTAL_SECONDS, help="Seconds to simulate")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.3, 0.55),
            camera_lookat=(0, 0, PLATFORM_HEIGHT),
            max_FPS=60,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=args.vis,
    )

    # ── Entities ────────────────────────────────────────────────────────
    scene.add_entity(gs.morphs.Plane())

    hot_platform = scene.add_entity(
        gs.morphs.Box(
            size=(PLATFORM_SIZE, PLATFORM_SIZE, PLATFORM_HEIGHT),
            pos=(0.0, 0.0, PLATFORM_HEIGHT / 2),
            fixed=True,
            visualization=False,  # sensor debug_draw will visualize it
        ),
    )

    allegro = scene.add_entity(
        gs.morphs.MJCF(
            file=ALLEGRO_PATH,
            pos=(0, 0, PLATFORM_HEIGHT + 0.06),
        )
    )

    palm_link = allegro.get_link("palm")

    # ── Temperature properties ──────────────────────────────────────────
    TP = gs.sensors.TemperatureProperties
    properties_dict = {
        -1: TP(base_temperature=BASE_TEMP, conductivity=150.0, density=8000.0, specific_heat=1.0, emissivity=0.2),
        hot_platform.base_link_idx: TP(
            base_temperature=HOT_TEMP, conductivity=400.0, density=2000.0, specific_heat=1.0, emissivity=0.95
        ),
        palm_link.idx: TP(
            base_temperature=BASE_TEMP, conductivity=200.0, density=3000.0, specific_heat=1.0, emissivity=0.1
        ),
    }

    sensor = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=hot_platform.idx,
            link_idx_local=0,
            grid_size=GRID_SIZE,
            properties_dict=properties_dict,
            simulate_all_link_temperatures=True,
            ambient_temperature=BASE_TEMP,
            convection_coefficient=0.0,
            sensor_time_constant=0.0,
            contact_depth_weight=1.0,
            draw_debug=True,
            debug_temperature_range=(BASE_TEMP, HOT_TEMP),
        )
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.5, -0.3, 0.55),
        lookat=(0, 0, PLATFORM_HEIGHT),
        fov=45,
        GUI=False,
    )

    scene.build()

    allegro.set_dofs_kp([5.0] * 16)
    allegro.set_dofs_kv([0.5] * 16)

    cam.start_recording()

    total_steps = int(args.seconds / DT)
    print("\n=== Allegro Hand Temperature Demo ===")
    print(f"Platform at {HOT_TEMP}°C, hand at {BASE_TEMP}°C")
    print(f"Simulating {args.seconds}s ({total_steps} steps) ...")
    print()

    for step in range(total_steps):
        t = step * DT

        # Close fingers gradually over 2s
        alpha = min(t / 2.0, 1.0)
        target = JOINT_MID + 0.7 * alpha * JOINT_AMP
        allegro.control_dofs_position(target.astype(np.float32))

        scene.step()
        cam.render()

        if step % 100 == 0:
            data = sensor.read()
            t_min, t_max = float(data.min()), float(data.max())
            print(f"  t={t:.2f}s: platform temp range [{t_min:.1f}, {t_max:.1f}] °C")

        if "PYTEST_VERSION" in os.environ:
            break

    out_path = os.path.join(SCRIPT_DIR, "allegro_temperature.mp4")
    cam.stop_recording(save_to_filename=out_path, fps=60)
    print(f"\nSaved video: {out_path}")


if __name__ == "__main__":
    main()
