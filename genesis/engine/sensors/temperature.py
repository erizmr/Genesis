import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Type

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import TemperatureGrid as TemperatureGridOptions
from genesis.options.sensors import TemperatureProperties
from genesis.utils import mesh as mu
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import (
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager

import trimesh

STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m²·K⁴)
KELVIN_OFFSET = 273.15
PI = math.pi
TWO_PI = 2.0 * PI
MAX_TEMP = 1000.0  # °C


class _PropIdx(IntEnum):
    BASE_TEMP = 0
    CONDUCTIVITY = 1
    EMISSIVITY = 2
    RHO_CP = 3


class _ScratchIdx(IntEnum):
    OTHER_LINK = 0
    CONTACT_IDX = 1
    DEPTH = 2
    POS_X = 3
    POS_Y = 4
    POS_Z = 5
    NORMAL_X = 6
    NORMAL_Y = 7
    NORMAL_Z = 8
    GROUP_CONTACT_IDX = 9
    GROUP_POS_X = 10
    GROUP_POS_Y = 11
    GROUP_POS_Z = 12
    GROUP_NORMAL_X = 13
    GROUP_NORMAL_Y = 14
    GROUP_NORMAL_Z = 15
    GROUP_DEPTH = 16
    GROUP_POS2_X = 17
    GROUP_POS2_Y = 18


def _link_aabb_extent_from_geoms(link: "RigidLink") -> np.ndarray:
    """Compute AABB extent of a link from its collision geom vertices in link-local frame.

    Works before scene.build() since it only uses geom init data (vertices, pos, quat).
    """
    all_verts = []
    for geom in link.geoms:
        verts = geom.init_verts  # (N, 3) in geom mesh frame
        # Transform to link-local frame: p_link = R(quat) @ p_mesh + pos
        verts_local = gu.transform_by_trans_quat(verts, geom.init_pos, geom.init_quat)
        all_verts.append(verts_local)
    if not all_verts:
        return np.array([0.01, 0.01, 0.01])
    all_verts = np.concatenate(all_verts, axis=0)
    extent = all_verts.max(axis=0) - all_verts.min(axis=0)
    return np.maximum(extent, 1e-6)


def _build_occupancy_mask(
    link: "RigidLink", grid_size: tuple[int, int, int], aabb_min: np.ndarray, voxel_size: np.ndarray
) -> torch.Tensor:
    """Build binary occupancy mask (1=solid, 0=air) by voxelizing link collision geoms.

    For each geom, creates a trimesh, transforms to link-local frame, and checks which
    voxel centers fall inside the mesh (negative signed distance).

    Returns shape (nx, ny, nz) float tensor on gs.device.
    """
    nx, ny, nz = grid_size
    # Voxel center positions in link-local frame
    xs = (np.arange(nx) + 0.5) * voxel_size[0] + aabb_min[0]
    ys = (np.arange(ny) + 0.5) * voxel_size[1] + aabb_min[1]
    zs = (np.arange(nz) + 0.5) * voxel_size[2] + aabb_min[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    query_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (N, 3)

    occupancy = np.zeros(nx * ny * nz, dtype=np.float32)

    for geom in link.geoms:
        verts = geom.init_verts.copy()
        faces = geom.init_faces.copy()
        if len(verts) == 0 or len(faces) == 0:
            continue
        # Transform geom vertices to link-local frame
        verts_local = gu.transform_by_trans_quat(verts, geom.init_pos, geom.init_quat)
        mesh = trimesh.Trimesh(vertices=verts_local, faces=faces, process=False)
        if not mesh.is_watertight:
            # For non-watertight meshes, fill the convex hull instead
            try:
                mesh = mesh.convex_hull
            except Exception:
                # Fallback: mark all voxels as solid
                occupancy[:] = 1.0
                continue
        # Check which query points are inside this geom
        inside = mesh.contains(query_points)
        occupancy[inside] = 1.0

    # If no geom produced any solid voxels, fall back to all-solid
    if occupancy.sum() == 0:
        occupancy[:] = 1.0

    return torch.tensor(occupancy.reshape(nx, ny, nz), dtype=gs.tc_float, device=gs.device)


def _compute_surface_mask_from_occupancy(occupancy: torch.Tensor) -> torch.Tensor:
    """Surface = solid voxel with at least one air face-neighbor (6-connectivity).

    Args:
        occupancy: shape (nx, ny, nz), float 0/1
    Returns:
        surface mask shape (nx, ny, nz), float 0/1
    """
    # Pad with air (0) on all faces
    padded = torch.nn.functional.pad(occupancy.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), value=0.0)
    padded = padded.squeeze(0).squeeze(0)  # (nx+2, ny+2, nz+2)
    has_air_neighbor = (
        (padded[:-2, 1:-1, 1:-1] < 0.5)
        | (padded[2:, 1:-1, 1:-1] < 0.5)
        | (padded[1:-1, :-2, 1:-1] < 0.5)
        | (padded[1:-1, 2:, 1:-1] < 0.5)
        | (padded[1:-1, 1:-1, :-2] < 0.5)
        | (padded[1:-1, 1:-1, 2:] < 0.5)
    )
    return ((occupancy > 0.5) & has_air_neighbor).to(gs.tc_float)


def _compute_K2_rfft3(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float) -> torch.Tensor:
    """Squared wave numbers for 3D real FFT: K2[i,j,k] = (2*pi*kx)^2 + (2*pi*ky)^2 + (2*pi*kz)^2 with rfft layout."""
    kx = torch.fft.fftfreq(nx, d=dx, device=gs.device).to(gs.tc_float)
    ky = torch.fft.fftfreq(ny, d=dy, device=gs.device).to(gs.tc_float)
    kz = torch.fft.rfftfreq(nz, d=dz, device=gs.device).to(gs.tc_float)
    K2 = (TWO_PI * kx).reshape(-1, 1, 1) ** 2
    K2 = K2 + (TWO_PI * ky).reshape(1, -1, 1) ** 2
    K2 = K2 + (TWO_PI * kz).reshape(1, 1, -1) ** 2
    K2[0, 0, 0] = max(K2[0, 0, 0].item(), gs.EPS)
    return K2


def _compute_surface_mask(nx: int, ny: int, nz: int) -> torch.Tensor:
    """Boolean mask of boundary voxels (at least one face on grid boundary). Shape (nx, ny, nz)."""
    ix, iy, iz = torch.meshgrid(
        torch.arange(nx, device=gs.device),
        torch.arange(ny, device=gs.device),
        torch.arange(nz, device=gs.device),
        indexing="ij",
    )
    return ((ix == 0) | (ix == nx - 1) | (iy == 0) | (iy == ny - 1) | (iz == 0) | (iz == nz - 1)).to(gs.tc_float)


def _apply_diffusion_and_heat_generation(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    grid_size: torch.Tensor,
    heat_generation: list[torch.Tensor | None],
    occupancy_mask: list[torch.Tensor | None],
    voxel_size: torch.Tensor,
    links_idx: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_rho_cp: torch.Tensor,
    link_conductivity: torch.Tensor,
    link_base_temperature: torch.Tensor,
    K2_spectral: list[torch.Tensor],
    ambient_temperature: float,
    dt: float,
    output: torch.Tensor,
) -> None:
    """Batched FFT semi-implicit diffusion with mirror padding (Neumann BC, no wrap-around)."""
    n_sensors = sensor_cache_start.shape[0]
    n_batches = output.shape[0]
    for i_s in range(n_sensors):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        nx, ny, nz = grid_size[i_s].tolist()
        mat_idx = link_to_material_idx[links_idx[i_s].item()].item()
        rcp = link_rho_cp[mat_idx].item()
        k = link_conductivity[mat_idx].item()
        alpha = k / rcp
        T = output[:, start : start + size].reshape(-1, nx, ny, nz)
        # Mirror-pad to (2*nx, 2*ny, 2*nz) for zero-flux (Neumann) boundaries; avoids FFT wrap-around.
        T_x = torch.cat([T, torch.flip(T, dims=(1,))], dim=1)
        T_xy = torch.cat([T_x, torch.flip(T_x, dims=(2,))], dim=2)
        T_pad = torch.cat([T_xy, torch.flip(T_xy, dims=(3,))], dim=3)
        T_hat = torch.fft.rfftn(T_pad, dim=(-3, -2, -1))
        T_hat = T_hat / (1.0 + dt * alpha * K2_spectral[i_s])
        T_pad = torch.fft.irfftn(T_hat, s=(2 * nx, 2 * ny, 2 * nz), dim=(-3, -2, -1)).real
        T = T_pad[:, :nx, :ny, :nz]

        # Project: zero air voxels but recover leaked heat to conserve energy.
        # FFT diffusion spreads heat into air voxels. We recover that excess energy
        # (above ambient) and redistribute it to solid voxels before zeroing air.
        occ = occupancy_mask[i_s]
        if occ is not None:
            air = occ < 0.5  # (nx, ny, nz)
            solid = ~air
            n_solid = solid.sum().item()
            n_air = air.sum().item()
            if n_solid > 0 and n_air > 0:
                # Heat that leaked to air = sum of (T_air - ambient) across air voxels
                leaked = (T[:, air] - ambient_temperature).sum(dim=-1)  # (B,)
                # Return leaked heat to solid voxels
                T[:, solid] += (leaked / n_solid).unsqueeze(-1)
            T[:, air] = ambient_temperature

        output[:, start : start + size] = T.reshape(n_batches, -1)

        # Add internal heat generation (W/m² -> Q_vol = Q_surface / dz).
        q = heat_generation[i_s]
        if q is not None:
            dz = max(voxel_size[i_s, 2].item(), gs.EPS)
            Q_vol = q.reshape(-1) / dz
            delta_T = dt * Q_vol / rcp
            output[:, start : start + size] += delta_T.unsqueeze(0).expand(n_batches, -1)


@qd.func
def _qd_polygon_area_from_points_3d(
    n: gs.qd_int,
    scratch: qd.types.ndarray(),
    i_b: gs.qd_int,
    eps: gs.qd_float,
) -> gs.qd_float:
    """Area of polygon from scratch buffer."""
    area = gs.qd_float(0.0)
    if n >= 3:
        cx = gs.qd_float(0.0)
        cy = gs.qd_float(0.0)
        cz = gs.qd_float(0.0)
        nx = gs.qd_float(0.0)
        ny = gs.qd_float(0.0)
        nz = gs.qd_float(0.0)
        for i in range(n):
            cx = cx + scratch[i_b, i, _ScratchIdx.GROUP_POS_X]
            cy = cy + scratch[i_b, i, _ScratchIdx.GROUP_POS_Y]
            cz = cz + scratch[i_b, i, _ScratchIdx.GROUP_POS_Z]
            nx = nx + scratch[i_b, i, _ScratchIdx.GROUP_NORMAL_X]
            ny = ny + scratch[i_b, i, _ScratchIdx.GROUP_NORMAL_Y]
            nz = nz + scratch[i_b, i, _ScratchIdx.GROUP_NORMAL_Z]
        n_inv = gs.qd_float(1.0) / gs.qd_float(n)
        cx, cy, cz = cx * n_inv, cy * n_inv, cz * n_inv
        nx, ny, nz = nx * n_inv, ny * n_inv, nz * n_inv
        n_norm = qd.sqrt(nx * nx + ny * ny + nz * nz) + eps
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm
        ax = 0 if qd.abs(nx) < gs.qd_float(0.9) else 1
        ux = gs.qd_float(0.0)
        uy = gs.qd_float(0.0)
        uz = gs.qd_float(0.0)
        if ax == 0:
            ux = gs.qd_float(1.0)
        else:
            uy = gs.qd_float(1.0)
        dot = ux * nx + uy * ny + uz * nz
        ux, uy, uz = ux - dot * nx, uy - dot * ny, uz - dot * nz
        u_norm = qd.sqrt(ux * ux + uy * uy + uz * uz) + eps
        ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux
        v_norm = qd.sqrt(vx * vx + vy * vy + vz * vz) + eps
        vx, vy, vz = vx / v_norm, vy / v_norm, vz / v_norm
        for i in range(n):
            rx = scratch[i_b, i, 10] - cx
            ry = scratch[i_b, i, 11] - cy
            rz = scratch[i_b, i, 12] - cz
            scratch[i_b, i, _ScratchIdx.GROUP_POS2_X] = rx * ux + ry * uy + rz * uz
            scratch[i_b, i, _ScratchIdx.GROUP_POS2_Y] = rx * vx + ry * vy + rz * vz
        for i in range(1, n):
            key_x = scratch[i_b, i, _ScratchIdx.GROUP_POS2_X]
            key_y = scratch[i_b, i, _ScratchIdx.GROUP_POS2_Y]
            j = i - 1
            key_angle = qd.atan2(key_y, key_x)
            while (
                j >= 0
                and qd.atan2(scratch[i_b, j, _ScratchIdx.GROUP_POS2_Y], scratch[i_b, j, _ScratchIdx.GROUP_POS2_X])
                > key_angle
            ):
                scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_X] = scratch[i_b, j, _ScratchIdx.GROUP_POS2_X]
                scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_Y] = scratch[i_b, j, _ScratchIdx.GROUP_POS2_Y]
                j = j - 1
            scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_X] = key_x
            scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_Y] = key_y
        for i in range(n):
            i_next = (i + 1) % n
            area = (
                area
                + scratch[i_b, i, _ScratchIdx.GROUP_POS2_X] * scratch[i_b, i_next, _ScratchIdx.GROUP_POS2_Y]
                - scratch[i_b, i_next, _ScratchIdx.GROUP_POS2_X] * scratch[i_b, i, _ScratchIdx.GROUP_POS2_Y]
            )
        area = qd.abs(area) * gs.qd_float(0.5)

    return area


@qd.kernel
def _kernel_compute_contact_areas(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    contact_area: qd.types.ndarray(),
    scratch: qd.types.ndarray(),
    eps: gs.qd_float,
):
    # contact_area shape (n_c_max, n_batches). scratch (n_batches, n_c_max, len(_ScratchIdx)).
    n_batches = contact_area.shape[1]
    for i_b in range(n_batches):
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            la = collider_state.contact_data.link_a[i_c, i_b]
            lb = collider_state.contact_data.link_b[i_c, i_b]
            scratch[i_b, i_c, _ScratchIdx.OTHER_LINK] = gs.qd_float(lb)
            scratch[i_b, i_c, _ScratchIdx.CONTACT_IDX] = gs.qd_float(i_c)
            scratch[i_b, i_c, _ScratchIdx.DEPTH] = collider_state.contact_data.penetration[i_c, i_b]
            p_world = collider_state.contact_data.pos[i_c, i_b]
            link_pos = links_state.pos[la, i_b]
            link_quat = links_state.quat[la, i_b]
            p_local = gu.qd_inv_transform_by_trans_quat(p_world, link_pos, link_quat)
            scratch[i_b, i_c, _ScratchIdx.POS_X] = p_local.x
            scratch[i_b, i_c, _ScratchIdx.POS_Y] = p_local.y
            scratch[i_b, i_c, _ScratchIdx.POS_Z] = p_local.z
            n_w = collider_state.contact_data.normal[i_c, i_b]
            scratch[i_b, i_c, _ScratchIdx.NORMAL_X] = n_w.x
            scratch[i_b, i_c, _ScratchIdx.NORMAL_Y] = n_w.y
            scratch[i_b, i_c, _ScratchIdx.NORMAL_Z] = n_w.z

        for i_c in range(n_c):
            la = collider_state.contact_data.link_a[i_c, i_b]
            lb = collider_state.contact_data.link_b[i_c, i_b]
            is_first = True
            for k in range(i_c):
                la_k = collider_state.contact_data.link_a[k, i_b]
                lb_k = collider_state.contact_data.link_b[k, i_b]
                if la_k == la and lb_k == lb:
                    is_first = False
            if not is_first:
                continue

            count = 0
            for j in range(n_c):
                la_j = collider_state.contact_data.link_a[j, i_b]
                lb_j = collider_state.contact_data.link_b[j, i_b]
                if la_j == la and lb_j == lb:
                    scratch[i_b, count, _ScratchIdx.GROUP_CONTACT_IDX] = scratch[i_b, j, _ScratchIdx.CONTACT_IDX]
                    scratch[i_b, count, _ScratchIdx.GROUP_POS_X] = scratch[i_b, j, _ScratchIdx.POS_X]
                    scratch[i_b, count, _ScratchIdx.GROUP_POS_Y] = scratch[i_b, j, _ScratchIdx.POS_Y]
                    scratch[i_b, count, _ScratchIdx.GROUP_POS_Z] = scratch[i_b, j, _ScratchIdx.POS_Z]
                    scratch[i_b, count, _ScratchIdx.GROUP_NORMAL_X] = scratch[i_b, j, _ScratchIdx.NORMAL_X]
                    scratch[i_b, count, _ScratchIdx.GROUP_NORMAL_Y] = scratch[i_b, j, _ScratchIdx.NORMAL_Y]
                    scratch[i_b, count, _ScratchIdx.GROUP_NORMAL_Z] = scratch[i_b, j, _ScratchIdx.NORMAL_Z]
                    scratch[i_b, count, _ScratchIdx.GROUP_DEPTH] = scratch[i_b, j, _ScratchIdx.DEPTH]
                    count = count + 1

            group_area = eps
            if count >= 3:
                group_area = _qd_polygon_area_from_points_3d(count, scratch, i_b, eps)
            else:
                for k in range(count):
                    d = scratch[i_b, k, _ScratchIdx.GROUP_DEPTH]
                    group_area = group_area + d * PI

            area_per_contact = group_area / (gs.qd_float(count) + eps)
            for k in range(count):
                contact_idx = gs.qd_int(scratch[i_b, k, _ScratchIdx.GROUP_CONTACT_IDX])
                contact_area[contact_idx, i_b] = area_per_contact


@qd.func
def _qd_k_eff(k_a: gs.qd_float, k_b: gs.qd_float, eps: gs.qd_float) -> gs.qd_float:
    """Effective conductivity for series thermal resistance: 2*k_a*k_b/(k_a+k_b+eps)."""
    return 2.0 * k_a * k_b / (k_a + k_b + eps)


@qd.kernel
def _kernel_contact_heat(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    links_idx: qd.types.ndarray(),
    aabb_min: qd.types.ndarray(),
    grid_size: qd.types.ndarray(),
    voxel_size: qd.types.ndarray(),
    voxel_volume: qd.types.ndarray(),
    depth_weight: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    link_temps: qd.types.ndarray(),
    link_volume: qd.types.ndarray(),
    link_to_material_idx: qd.types.ndarray(),
    link_base_temperature: qd.types.ndarray(),
    link_conductivity: qd.types.ndarray(),
    link_rho_cp: qd.types.ndarray(),
    contact_area: qd.types.ndarray(),
    dt: gs.qd_float,
    eps: gs.qd_float,
    output: qd.types.ndarray(),
):
    # contact_area shape (n_c_max, n_batches)
    n_batches = output.shape[0]
    n_sensors = links_idx.shape[0]
    use_link_temps = link_temps.shape[0] > 0

    # Grid update: only for contacts that involve a sensorized link; use contact_area[i_c, i_b]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        sensor_link_idx = links_idx[i_s]
        dw = depth_weight[i_s]
        start = sensor_cache_start[i_s]
        nx = grid_size[i_s, 0]
        ny = grid_size[i_s, 1]
        nz = grid_size[i_s, 2]
        vol = voxel_volume[i_s] + eps
        mat_idx_sensor = link_to_material_idx[sensor_link_idx]
        if mat_idx_sensor < 0:
            continue
        rcp = link_rho_cp[mat_idx_sensor] + eps

        k_sensor = link_conductivity[mat_idx_sensor]
        amin = qd.math.vec3(aabb_min[i_s, 0], aabb_min[i_s, 1], aabb_min[i_s, 2])
        vs = qd.math.vec3(
            voxel_size[i_s, 0] + eps,
            voxel_size[i_s, 1] + eps,
            voxel_size[i_s, 2] + eps,
        )
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            la = collider_state.contact_data.link_a[i_c, i_b]
            lb = collider_state.contact_data.link_b[i_c, i_b]
            if la != sensor_link_idx and lb != sensor_link_idx:
                continue
            other_link = lb if la == sensor_link_idx else la
            mat_other = link_to_material_idx[other_link]
            if mat_other >= 0:
                T_other = qd.select(use_link_temps, link_temps[i_b, other_link], link_base_temperature[mat_other])
                k_other = link_conductivity[mat_other]
                k_eff = _qd_k_eff(k_sensor, k_other, eps)
                p_world = collider_state.contact_data.pos[i_c, i_b]
                link_pos = links_state.pos[sensor_link_idx, i_b]
                link_quat = links_state.quat[sensor_link_idx, i_b]
                p_local = gu.qd_inv_transform_by_trans_quat(p_world, link_pos, link_quat)
                u_x = (p_local.x - amin.x) / vs.x
                u_y = (p_local.y - amin.y) / vs.y
                u_z = (p_local.z - amin.z) / vs.z
                ix = min(max(0, int(u_x)), nx - 1)
                iy = min(max(0, int(u_y)), ny - 1)
                iz = min(max(0, int(u_z)), nz - 1)
                cell_idx = ix * (ny * nz) + iy * nz + iz
                T_cell = output[i_b, start + cell_idx]
                area_base = contact_area[i_c, i_b] + eps
                area = qd.max(area_base, PI * dw * collider_state.contact_data.penetration[i_c, i_b])
                flux = k_eff * (T_other - T_cell) / (vol / area + eps)
                Q_vol = flux * area / vol
                delta_T = dt * Q_vol / rcp
                output[i_b, start + cell_idx] = T_cell + delta_T

    # Link temps update for all contacts (both links) when use_link_temps
    if use_link_temps:
        for i_b in range(n_batches):
            n_c = collider_state.n_contacts[i_b]
            for i_c in range(n_c):
                la = collider_state.contact_data.link_a[i_c, i_b]
                lb = collider_state.contact_data.link_b[i_c, i_b]
                mat_la = link_to_material_idx[la]
                mat_lb = link_to_material_idx[lb]
                if mat_la < 0 or mat_lb < 0:
                    continue
                T_la = link_temps[i_b, la]
                T_lb = link_temps[i_b, lb]
                k_la = link_conductivity[mat_la] + eps
                k_lb = link_conductivity[mat_lb] + eps
                k_eff = _qd_k_eff(k_la, k_lb, eps)
                area = contact_area[i_c, i_b] + eps
                vol_la = link_volume[la] + eps
                vol_lb = link_volume[lb] + eps
                length_scale = (vol_la + vol_lb) / (gs.qd_float(2.0) * area)
                flux = k_eff * (T_la - T_lb) / length_scale
                power = flux * area
                rcp_vol_la = link_rho_cp[mat_la] * vol_la + eps
                rcp_vol_lb = link_rho_cp[mat_lb] * vol_lb + eps
                delta_T_la = gs.qd_float(-1.0) * dt * power / rcp_vol_la
                delta_T_lb = dt * power / rcp_vol_lb
                link_temps[i_b, la] = link_temps[i_b, la] + delta_T_la
                link_temps[i_b, lb] = link_temps[i_b, lb] + delta_T_lb


def _radiation_convection_delta_T(
    T: torch.Tensor,
    emissivity: torch.Tensor | float,
    convection_coeff: float,
    ambient_temp: float,
    rho_cp_vol: torch.Tensor | float,
    dt: float,
) -> torch.Tensor:
    """Temperature change (to subtract) from radiation + convection: -dt * (q_rad + q_conv) / (rho_cp * vol)."""
    T_K = T + KELVIN_OFFSET
    T_amb_K = ambient_temp + KELVIN_OFFSET
    q_rad = emissivity * STEFAN_BOLTZMANN * (T_K**4 - T_amb_K**4)
    q_conv = convection_coeff * (T - ambient_temp)
    return dt * (q_rad + q_conv) / (rho_cp_vol + gs.EPS)


def _apply_radiation_convection(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    sensor_surface_mask: list[torch.Tensor],
    voxel_volume: torch.Tensor,
    links_idx: torch.Tensor,
    link_temps: torch.Tensor,
    link_volume: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_emissivity: torch.Tensor,
    link_rho_cp: torch.Tensor,
    ambient_temp: float,
    convection_coeff: float,
    dt: float,
    output: torch.Tensor,
) -> None:
    """Radiation + convection on surface voxels and (when allocated) on link temperatures.

    For link_temps, links with link_to_material_idx == -1 are treated as material index 0
    (default properties) for emissivity/rho_cp; only links with valid material are updated.
    """
    for i_s in range(sensor_cache_start.shape[0]):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        mask = sensor_surface_mask[i_s].reshape(-1)
        vol = max(voxel_volume[i_s].item(), gs.EPS)
        mat_idx = link_to_material_idx[links_idx[i_s]]
        emiss = link_emissivity[mat_idx].item()
        rcp = link_rho_cp[mat_idx].item()
        denom = rcp * vol
        T_flat = output[:, start : start + size]
        delta = _radiation_convection_delta_T(T_flat, emiss, convection_coeff, ambient_temp, denom, dt)
        output[:, start : start + size] -= delta * mask

    if link_temps.numel() > 0:
        valid = link_to_material_idx >= 0  # (n_links,)
        mat_idx = link_to_material_idx.clamp(min=0)  # -1 -> 0 (default material) for indexing
        rcp_vol = link_rho_cp[mat_idx] * link_volume  # (n_links,)
        delta = _radiation_convection_delta_T(
            link_temps, link_emissivity[mat_idx], convection_coeff, ambient_temp, rcp_vol.unsqueeze(0), dt
        )
        link_temps.sub_(delta * valid.unsqueeze(0).to(gs.tc_float))


def _apply_kinematic_conduction(
    kinematic_edges: torch.Tensor,
    kinematic_joint_area: torch.Tensor,
    link_temps: torch.Tensor,
    link_volume: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_conductivity: torch.Tensor,
    link_rho_cp: torch.Tensor,
    dt: float,
) -> None:
    """Heat conduction between kinematically adjacent (parent-child) link pairs.

    Uses the same effective-conductivity model as _kernel_contact_heat (lines 374-400):
        k_eff = 2*k_a*k_b / (k_a + k_b)
        length_scale = (vol_a + vol_b) / (2 * joint_area)
        flux = k_eff * (T_a - T_b) / length_scale
        power = flux * joint_area
        dT_a = -dt * power / (rho_cp_a * vol_a)
        dT_b =  dt * power / (rho_cp_b * vol_b)
    """
    if kinematic_edges.shape[0] == 0 or link_temps.numel() == 0:
        return

    parent_idx = kinematic_edges[:, 0]  # (E,)
    child_idx = kinematic_edges[:, 1]  # (E,)
    joint_area = kinematic_joint_area  # (E,)

    mat_p = link_to_material_idx[parent_idx]  # (E,)
    mat_c = link_to_material_idx[child_idx]  # (E,)

    # Only conduct between pairs where both links have valid materials
    valid = (mat_p >= 0) & (mat_c >= 0)
    if not valid.any():
        return

    k_p = link_conductivity[mat_p]  # (E,)
    k_c = link_conductivity[mat_c]  # (E,)
    k_eff = 2.0 * k_p * k_c / (k_p + k_c + gs.EPS)  # (E,)

    vol_p = link_volume[parent_idx] + gs.EPS  # (E,)
    vol_c = link_volume[child_idx] + gs.EPS  # (E,)

    length_scale = (vol_p + vol_c) / (2.0 * joint_area + gs.EPS)  # (E,)
    rcp_vol_p = link_rho_cp[mat_p] * vol_p + gs.EPS  # (E,)
    rcp_vol_c = link_rho_cp[mat_c] * vol_c + gs.EPS  # (E,)

    # link_temps shape: (n_batches, n_links)
    T_p = link_temps[:, parent_idx]  # (B, E)
    T_c = link_temps[:, child_idx]  # (B, E)

    flux = k_eff.unsqueeze(0) * (T_p - T_c) / length_scale.unsqueeze(0)  # (B, E)
    power = flux * joint_area.unsqueeze(0)  # (B, E)

    delta_T_p = -dt * power / rcp_vol_p.unsqueeze(0)  # (B, E)
    delta_T_c = dt * power / rcp_vol_c.unsqueeze(0)  # (B, E)

    # Mask invalid edges
    mask = valid.unsqueeze(0).to(gs.tc_float)  # (1, E)
    delta_T_p = delta_T_p * mask
    delta_T_c = delta_T_c * mask

    # Scatter-add deltas back to link_temps
    link_temps.scatter_add_(1, parent_idx.unsqueeze(0).expand_as(delta_T_p), delta_T_p)
    link_temps.scatter_add_(1, child_idx.unsqueeze(0).expand_as(delta_T_c), delta_T_c)


def _apply_T_measured_filter(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    sensor_time_const: torch.Tensor,
    dt: float,
    T_actual: torch.Tensor,
    T_measured: torch.Tensor,
) -> None:
    """T_measured += (dt/tau)*(T - T_measured); if tau<=0 then T_measured = T. Batched over envs."""
    for i_s in range(sensor_cache_start.shape[0]):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        tau = sensor_time_const[i_s].item()
        T_slice = T_actual[:, start : start + size]
        T_meas_slice = T_measured[:, start : start + size]
        if tau > 0:
            alpha = dt / tau
            T_measured[:, start : start + size] = T_meas_slice + alpha * (T_slice - T_meas_slice)
        else:
            T_measured[:, start : start + size] = T_slice


@dataclass
class TemperatureGridSensorMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all temperature grid sensors."""

    ambient_temperature: float = 21.0
    convection_coeff: float = 1.0
    link_to_material_idx: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    link_material_properties: torch.Tensor = make_tensor_field((0, len(_PropIdx)), dtype=gs.tc_float)
    properties_dict: dict[int, TemperatureProperties] = field(default_factory=dict)
    simulate_all_link_temps: bool = False
    link_temps: torch.Tensor = make_tensor_field((0, 0))
    link_volume: torch.Tensor = make_tensor_field((0,))

    kinematic_conduction: bool = False
    kinematic_edges: torch.Tensor = make_tensor_field((0, 2), dtype=gs.tc_int)
    kinematic_joint_area: torch.Tensor = make_tensor_field((0,))

    aabb_min: torch.Tensor = make_tensor_field((0, 3))
    aabb_extent: torch.Tensor = make_tensor_field((0, 3))
    grid_size: torch.Tensor = make_tensor_field((0, 3), dtype=gs.tc_int)
    voxel_size: torch.Tensor = make_tensor_field((0, 3))
    voxel_volume: torch.Tensor = make_tensor_field((0,))
    sensor_time_const: torch.Tensor = make_tensor_field((0,))
    contact_depth_weight: torch.Tensor = make_tensor_field((0,))
    K2_spectral: list[torch.Tensor] = field(default_factory=list)
    sensor_surface_mask: list[torch.Tensor] = field(default_factory=list)
    occupancy_mask: list[torch.Tensor | None] = field(default_factory=list)
    heat_generation: list[torch.Tensor | None] = field(default_factory=list)
    contact_area_scratch: torch.Tensor = make_tensor_field((0, len(_ScratchIdx)))
    contact_area_buffer: torch.Tensor = make_tensor_field((0, 0))

    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)


@register_sensor(TemperatureGridOptions, TemperatureGridSensorMetadata, tuple)
class TemperatureGridSensor(
    RigidSensorMixin[TemperatureGridSensorMetadata],
    NoisySensorMixin[TemperatureGridSensorMetadata],
    Sensor[TemperatureGridSensorMetadata],
):
    def __init__(
        self,
        sensor_options: TemperatureGridOptions,
        sensor_idx: int,
        data_cls: Type[tuple],
        sensor_manager: "SensorManager",
    ):
        # If voxel_resolution is set, compute grid_size from link geometry before super().__init__
        # (which calls _get_return_format to determine cache size).
        self._effective_grid_size: tuple[int, int, int] | None = None
        if sensor_options.voxel_resolution is not None:
            entity_idx = sensor_options.entity_idx
            if entity_idx is not None and entity_idx >= 0:
                entity = sensor_manager._sim._scene.entities[entity_idx]
                link = entity.links[sensor_options.link_idx_local]
                extent = _link_aabb_extent_from_geoms(link)
                res = sensor_options.voxel_resolution
                self._effective_grid_size = tuple(max(1, int(math.ceil(e / res))) for e in extent)

        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

        self._link: "RigidLink | None" = None
        self._debug_objects: list = []
        self._debug_t_min: float = self._options.debug_temperature_range[0]
        self._debug_t_range: float = self._options.debug_temperature_range[1] - self._debug_t_min
        self._debug_cell_local_positions: np.ndarray = np.array([])  # set in build

    def build(self):
        super().build()

        solver = self._shared_metadata.solver

        # Same for all sensors
        if self._options.ambient_temperature is not None:
            self._shared_metadata.ambient_temperature = self._options.ambient_temperature
        if self._options.convection_coefficient is not None:
            self._shared_metadata.convection_coeff = self._options.convection_coefficient

        if self._shared_metadata.link_to_material_idx.shape[0] == 0:
            self._shared_metadata.link_to_material_idx = torch.full(
                (solver.n_links,), -1, dtype=gs.tc_int, device=gs.device
            )
        self._shared_metadata.properties_dict.update(self._options.properties_dict)
        if len(self._shared_metadata.properties_dict) > len(self._shared_metadata.link_material_properties):
            self._shared_metadata.link_material_properties = torch.empty(
                (len(_PropIdx), len(self._shared_metadata.properties_dict)), dtype=gs.tc_float, device=gs.device
            )
            # -1 in link_to_material_idx means invalid, 0 uses the default properties
            self._shared_metadata.link_to_material_idx[:] = 0 if -1 in self._shared_metadata.properties_dict else -1
            # sort properties_dict by link index to ensure default properties are at index 0
            for i, (prop_idx, props) in enumerate(
                sorted(self._shared_metadata.properties_dict.items(), key=lambda x: x[0])
            ):
                self._shared_metadata.link_material_properties[:, i] = torch.tensor(
                    # order should match _PropIdx
                    [props.base_temperature, props.conductivity, props.emissivity, props.density * props.specific_heat],
                    dtype=gs.tc_float,
                    device=gs.device,
                )
                if prop_idx >= 0:
                    self._shared_metadata.link_to_material_idx[prop_idx] = i
        assert self._link.idx in self._shared_metadata.properties_dict or -1 in self._shared_metadata.properties_dict, (
            f"Temperature properties for the attached link index {self._link.idx} should be provided"
            " in properties_dict, or use key -1 for default properties for all links."
        )
        if self._options.simulate_all_link_temperatures:
            self._shared_metadata.simulate_all_link_temps = True
            if len(self._shared_metadata.link_temps) == 0:
                self._shared_metadata.link_temps = torch.empty(
                    (solver._B, solver.n_links), dtype=gs.tc_float, device=gs.device
                )
                self._shared_metadata.link_volume = torch.empty(solver.n_links, dtype=gs.tc_float, device=gs.device)

                link_volume = self._shared_metadata.link_volume
                for entity in solver._entities:
                    for link in entity.links:
                        li = link.idx
                        if link.n_geoms > 0:
                            aabb = link.get_AABB()
                            if aabb.ndim == 3:
                                aabb = aabb[0]
                            vol = (aabb[1] - aabb[0]).prod().clamp(min=gs.EPS)
                            link_volume[li] = vol
                ambient_T = self._shared_metadata.ambient_temperature
                link_base_T = self._shared_metadata.link_material_properties[_PropIdx.BASE_TEMP]
                link_to_mat = self._shared_metadata.link_to_material_idx
                base_T_per_link = torch.where(
                    link_to_mat >= 0,
                    link_base_T[link_to_mat],
                    torch.tensor(ambient_T, dtype=gs.tc_float, device=gs.device),
                )
                n_batches = solver._B
                self._shared_metadata.link_temps.copy_(base_T_per_link.unsqueeze(0).expand(n_batches, -1))

        if self._options.kinematic_conduction:
            self._shared_metadata.kinematic_conduction = True
            if self._shared_metadata.kinematic_edges.shape[0] == 0 and self._shared_metadata.simulate_all_link_temps:
                edges = []
                for entity in solver._entities:
                    for link in entity.links:
                        if link.parent_idx >= 0:
                            edges.append((link.parent_idx, link.idx))
                if edges:
                    self._shared_metadata.kinematic_edges = torch.tensor(edges, dtype=gs.tc_int, device=gs.device)
                    # Joint cross-section area: min face area of child link's AABB
                    joint_areas = []
                    for _, child_idx in edges:
                        child_link = solver.links[child_idx]
                        if child_link.n_geoms > 0:
                            aabb = child_link.get_AABB()
                            if aabb.ndim == 3:
                                aabb = aabb[0]
                            ext = (aabb[1] - aabb[0]).clamp(min=gs.EPS)
                            dx, dy, dz = ext[0].item(), ext[1].item(), ext[2].item()
                            area = min(dx * dy, dy * dz, dx * dz)
                        else:
                            area = gs.EPS
                        joint_areas.append(area)
                    self._shared_metadata.kinematic_joint_area = torch.tensor(
                        joint_areas, dtype=gs.tc_float, device=gs.device
                    )

        # Per-sensor properties
        aabb_world = self._link.get_AABB()
        if aabb_world.ndim == 2:
            aabb_world = aabb_world.unsqueeze(0)  # (1, 2, 3)
        aabb_min_w = aabb_world[0, 0]  # (3,)
        aabb_max_w = aabb_world[0, 1]  # (3,)
        link_pos = self._link.get_pos(0)
        link_quat = self._link.get_quat(0)
        aabb_min_local = gu.inv_transform_by_trans_quat(aabb_min_w, link_pos, link_quat)
        aabb_max_local = gu.inv_transform_by_trans_quat(aabb_max_w, link_pos, link_quat)
        aabb_extent = (aabb_max_local - aabb_min_local).reshape(3)
        self._shared_metadata.aabb_min = concat_with_tensor(
            self._shared_metadata.aabb_min, aabb_min_local, expand=(1, 3), dim=0
        )
        self._shared_metadata.aabb_extent = concat_with_tensor(
            self._shared_metadata.aabb_extent, aabb_extent, expand=(1, 3), dim=0
        )
        actual_grid_size = (
            self._effective_grid_size if self._effective_grid_size is not None else self._options.grid_size
        )
        grid_size_tensor = torch.tensor(actual_grid_size, dtype=gs.tc_int, device=gs.device)
        self._shared_metadata.grid_size = concat_with_tensor(
            self._shared_metadata.grid_size, grid_size_tensor, expand=(1, 3), dim=0
        )
        voxel_size = aabb_extent / grid_size_tensor
        self._shared_metadata.voxel_size = concat_with_tensor(
            self._shared_metadata.voxel_size, voxel_size, expand=(1, 3), dim=0
        )
        self._shared_metadata.voxel_volume = concat_with_tensor(
            self._shared_metadata.voxel_volume, voxel_size.prod(), expand=(1,), dim=0
        )
        self._shared_metadata.sensor_time_const = concat_with_tensor(
            self._shared_metadata.sensor_time_const, self._options.sensor_time_constant, expand=(1,), dim=0
        )
        self._shared_metadata.contact_depth_weight = concat_with_tensor(
            self._shared_metadata.contact_depth_weight, self._options.contact_depth_weight, expand=(1,), dim=0
        )

        dx, dy, dz = voxel_size.tolist()
        nx, ny, nz = grid_size_tensor.tolist()

        xs = torch.arange(nx, device=gs.device, dtype=gs.tc_float) + 0.5
        ys = torch.arange(ny, device=gs.device, dtype=gs.tc_float) + 0.5
        zs = torch.arange(nz, device=gs.device, dtype=gs.tc_float) + 0.5
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)
        self._debug_cell_local_positions = (aabb_min_local.unsqueeze(0) + grid * voxel_size.unsqueeze(0)).cpu().numpy()

        K2_padded = _compute_K2_rfft3(nx * 2, ny * 2, nz * 2, dx, dy, dz)
        self._shared_metadata.K2_spectral.append(K2_padded)

        # Build occupancy mask from link collision geometry (if voxel_resolution is set)
        if self._options.voxel_resolution is not None and self._link.n_geoms > 0:
            aabb_min_np = aabb_min_local.detach().cpu().numpy().ravel()
            voxel_size_np = voxel_size.detach().cpu().numpy().ravel()
            occ = _build_occupancy_mask(self._link, (nx, ny, nz), aabb_min_np, voxel_size_np)
            self._shared_metadata.occupancy_mask.append(occ)
            surface_mask = _compute_surface_mask_from_occupancy(occ)
            n_solid = int(occ.sum().item())
            n_surface = int(surface_mask.sum().item())
            gs.logger.info(
                f"TemperatureGrid on link {self._link.idx}: grid ({nx},{ny},{nz}), "
                f"{n_solid}/{nx * ny * nz} solid voxels, {n_surface} surface voxels"
            )
        else:
            self._shared_metadata.occupancy_mask.append(None)
            surface_mask = _compute_surface_mask(nx, ny, nz)
        self._shared_metadata.sensor_surface_mask.append(surface_mask)

        if self._options.heat_generation is not None:
            q = torch.tensor(self._options.heat_generation, dtype=gs.tc_float, device=gs.device)
            if q.shape != (nx, ny, nz):
                raise ValueError(f"heat_generation shape {tuple(q.shape)} does not match grid_size ({nx}, {ny}, {nz})")
            self._shared_metadata.heat_generation.append(q)
        else:
            self._shared_metadata.heat_generation.append(None)

        current_cache_start = sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start, current_cache_start, expand=(1,), dim=0
        )

        # Contact area buffers
        n_c_max = int(solver.collider._collider_info.max_contact_pairs[None])
        self._shared_metadata.contact_area_buffer = torch.zeros(
            (n_c_max, solver._B), device=gs.device, dtype=gs.tc_float
        )
        self._shared_metadata.contact_area_scratch = torch.empty(
            (solver._B, n_c_max, len(_ScratchIdx)), device=gs.device, dtype=gs.tc_float
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        if self._effective_grid_size is not None:
            return (self._effective_grid_size,)
        return (self._options.grid_size,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: TemperatureGridSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        for i_s in range(shared_metadata.sensor_cache_start.shape[0]):
            link_idx = shared_metadata.links_idx[i_s].item()
            mat_idx = shared_metadata.link_to_material_idx[link_idx].item()
            base_T = shared_metadata.link_material_properties[_PropIdx.BASE_TEMP][mat_idx].item()
            start = shared_metadata.sensor_cache_start[i_s].item()
            shared_ground_truth_cache[envs_idx, start : start + shared_metadata.cache_sizes[i_s]] = base_T
        if shared_metadata.link_temps.numel() > 0:
            ambient_T = shared_metadata.ambient_temperature
            link_base_T = shared_metadata.link_material_properties[_PropIdx.BASE_TEMP]
            link_to_mat = shared_metadata.link_to_material_idx
            base_T_per_link = torch.where(
                link_to_mat >= 0,
                link_base_T[link_to_mat],
                torch.tensor(ambient_T, dtype=gs.tc_float, device=shared_metadata.link_temps.device),
            )
            n_envs = envs_idx.shape[0]
            shared_metadata.link_temps[envs_idx, :] = base_T_per_link.unsqueeze(0).expand(n_envs, -1)

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: TemperatureGridSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        solver = shared_metadata.solver
        dt = solver._sim.dt
        props = shared_metadata.link_material_properties
        link_conductivity = props[_PropIdx.CONDUCTIVITY]
        link_base_temperature = props[_PropIdx.BASE_TEMP]
        link_emissivity = props[_PropIdx.EMISSIVITY]
        link_rho_cp = props[_PropIdx.RHO_CP]

        # 1) Batched FFT semi-implicit diffusion + 2) Heat generation
        _apply_diffusion_and_heat_generation(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.grid_size,
            shared_metadata.heat_generation,
            shared_metadata.occupancy_mask,
            shared_metadata.voxel_size,
            shared_metadata.links_idx,
            shared_metadata.link_to_material_idx,
            link_rho_cp,
            link_conductivity,
            link_base_temperature,
            shared_metadata.K2_spectral,
            shared_metadata.ambient_temperature,
            dt,
            shared_ground_truth_cache,
        )
        # 3) Contact heat transfer
        collider_state = solver.collider._collider_state
        shared_metadata.contact_area_buffer.zero_()
        _kernel_compute_contact_areas(
            solver.links_state,
            collider_state,
            shared_metadata.contact_area_buffer,
            shared_metadata.contact_area_scratch,
            gs.EPS,
        )
        output = shared_ground_truth_cache.contiguous()
        _kernel_contact_heat(
            solver.links_state,
            collider_state,
            shared_metadata.links_idx,
            shared_metadata.aabb_min,
            shared_metadata.grid_size,
            shared_metadata.voxel_size,
            shared_metadata.voxel_volume,
            shared_metadata.contact_depth_weight,
            shared_metadata.sensor_cache_start,
            shared_metadata.link_temps,
            shared_metadata.link_volume,
            shared_metadata.link_to_material_idx,
            link_base_temperature,
            link_conductivity,
            link_rho_cp,
            shared_metadata.contact_area_buffer,
            dt,
            gs.EPS,
            output,
        )
        output.clamp_(-MAX_TEMP, MAX_TEMP)
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache.copy_(output)
        # 3b) Kinematic conduction between parent-child link pairs
        if shared_metadata.kinematic_conduction and shared_metadata.kinematic_edges.shape[0] > 0:
            _apply_kinematic_conduction(
                shared_metadata.kinematic_edges,
                shared_metadata.kinematic_joint_area,
                shared_metadata.link_temps,
                shared_metadata.link_volume,
                shared_metadata.link_to_material_idx,
                link_conductivity,
                link_rho_cp,
                dt,
            )
        # 4) Radiation and convection
        _apply_radiation_convection(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.sensor_surface_mask,
            shared_metadata.voxel_volume,
            shared_metadata.links_idx,
            shared_metadata.link_temps,
            shared_metadata.link_volume,
            shared_metadata.link_to_material_idx,
            link_emissivity,
            link_rho_cp,
            shared_metadata.ambient_temperature,
            shared_metadata.convection_coeff,
            dt,
            shared_ground_truth_cache,
        )

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: TemperatureGridSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        dt = shared_metadata.solver._sim.dt
        buffered_data.set(shared_ground_truth_cache)
        _apply_T_measured_filter(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.sensor_time_const,
            dt,
            shared_ground_truth_cache,
            buffered_data.at(0),
        )
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw each grid cell as a box colored by temperature (cool=blue, hot=red).
        Only draws for the first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        if self._link is None:
            return

        for obj in self._debug_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self._debug_objects = []

        link_pos = self._link.get_pos(env_idx)
        link_quat = self._link.get_quat(env_idx)
        link_pos = tensor_to_array(link_pos).reshape(3)
        link_quat = tensor_to_array(link_quat).reshape(4)
        link_T = gu.trans_quat_to_T(link_pos, link_quat)

        i_s = self._idx
        voxel_size = tensor_to_array(self._shared_metadata.voxel_size[i_s]).reshape(3)

        # World poses: same rotation as link, translation = link_T @ local_pos
        world_trans = (link_T[:3, :3] @ self._debug_cell_local_positions.T).T + link_T[:3, 3]
        poses = np.tile(link_T[np.newaxis], (len(world_trans), 1, 1)).astype(np.float64)
        poses[:, :3, 3] = world_trans

        # Per-cell color from temperature (blue=cool, red=hot)
        temps = self.read_ground_truth(env_idx)
        temps = tensor_to_array(temps).reshape(-1)
        t_min, t_range = self._debug_t_min, self._debug_t_range
        if t_range <= 0:
            t_range = 1.0
        norm = np.clip((temps - t_min) / t_range, 0.0, 1.0)
        # Blue (0,0,1) -> Red (1,0,0)
        colors = np.column_stack((norm, np.zeros_like(norm), 1.0 - norm, np.full_like(norm, 0.5)))
        for i, pose in enumerate(poses):
            mesh = mu.create_box(extents=voxel_size, color=tuple(float(c) for c in colors[i]))
            self._debug_objects.append(context.draw_debug_mesh(mesh, T=pose))

    @property
    def link_temperatures(self) -> torch.Tensor:
        return self._shared_metadata.link_temps
