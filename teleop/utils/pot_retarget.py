from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return np.asarray(fallback, dtype=np.float64).reshape(-1).copy()
    return vec / n


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return _safe_normalize(q, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([qq[1], qq[2], qq[3], qq[0]], dtype=np.float64)


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_xyzw(_rotmat_to_quat_wxyz(R))


def build_level_pot_rotation(side_axis: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = _safe_normalize(side_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _safe_normalize(y_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis = _safe_normalize(np.cross(y_axis, z_axis), np.array([1.0, 0.0, 0.0], dtype=np.float64))
    return np.column_stack([x_axis, y_axis, z_axis])


def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"ema_smooth expects 2D array, got shape={arr.shape}")
    out = np.zeros_like(arr, dtype=np.float64)
    if arr.shape[0] == 0:
        return out
    a = float(np.clip(alpha, 0.0, 1.0))
    out[0] = arr[0]
    for i in range(1, arr.shape[0]):
        out[i] = out[i - 1] + a * (arr[i] - out[i - 1])
    return out


def estimate_grasp_offsets_local(
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    pot_center: np.ndarray,
    pot_rot: np.ndarray,
    left_closed: np.ndarray,
    right_closed: np.ndarray,
    min_samples: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    l = np.asarray(left_pos, dtype=np.float64)
    r = np.asarray(right_pos, dtype=np.float64)
    c = np.asarray(pot_center, dtype=np.float64)
    rot = np.asarray(pot_rot, dtype=np.float64)
    lc = np.asarray(left_closed, dtype=np.float64).reshape(-1)
    rc = np.asarray(right_closed, dtype=np.float64).reshape(-1)
    n = l.shape[0]
    mask = (lc > 0.5) & (rc > 0.5)
    if int(np.sum(mask)) < int(min_samples):
        mask = np.ones(n, dtype=bool)

    left_local = []
    right_local = []
    for i in range(n):
        if not bool(mask[i]):
            continue
        Ri = rot[i]
        left_local.append(Ri.T @ (l[i] - c[i]))
        right_local.append(Ri.T @ (r[i] - c[i]))
    if not left_local or not right_local:
        raise ValueError("No valid samples to estimate grasp offsets")

    left_grasp = np.median(np.asarray(left_local, dtype=np.float64), axis=0)
    right_grasp = np.median(np.asarray(right_local, dtype=np.float64), axis=0)
    return left_grasp, right_grasp


@dataclass(frozen=True)
class PotFromHandsOutput:
    pot_center: np.ndarray
    pot_quat_wxyz: np.ndarray
    pot_rot: np.ndarray
    left_grasp_local: np.ndarray
    right_grasp_local: np.ndarray


@dataclass(frozen=True)
class HandTargetsOutput:
    left_target_pos: np.ndarray
    right_target_pos: np.ndarray
    left_target_quat_xyzw: np.ndarray
    right_target_quat_xyzw: np.ndarray


def hands_to_pot_trajectory(
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    left_closed: np.ndarray,
    right_closed: np.ndarray,
    center_offset_local: np.ndarray,
    side_alpha: float = 0.3,
) -> PotFromHandsOutput:
    """Stage 1: dual-hand positions -> pot center/orientation + grasp offsets."""
    l = np.asarray(left_pos, dtype=np.float64)
    r = np.asarray(right_pos, dtype=np.float64)
    offset_local = np.asarray(center_offset_local, dtype=np.float64).reshape(3)
    if l.shape != r.shape or l.ndim != 2 or l.shape[1] != 3:
        raise ValueError(f"left/right positions must be (N,3), got {l.shape} and {r.shape}")

    n = l.shape[0]
    mid = 0.5 * (l + r)
    side_smooth = ema_smooth(r - l, alpha=side_alpha)

    pot_rot = np.zeros((n, 3, 3), dtype=np.float64)
    pot_center = np.zeros((n, 3), dtype=np.float64)
    pot_quat_wxyz = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        Ri = build_level_pot_rotation(side_smooth[i])
        pot_rot[i] = Ri
        pot_center[i] = mid[i] + Ri @ offset_local
        pot_quat_wxyz[i] = _rotmat_to_quat_wxyz(Ri)

    left_grasp, right_grasp = estimate_grasp_offsets_local(
        left_pos=l,
        right_pos=r,
        pot_center=pot_center,
        pot_rot=pot_rot,
        left_closed=left_closed,
        right_closed=right_closed,
    )
    return PotFromHandsOutput(
        pot_center=pot_center,
        pot_quat_wxyz=pot_quat_wxyz,
        pot_rot=pot_rot,
        left_grasp_local=left_grasp,
        right_grasp_local=right_grasp,
    )


# Fixed dual-hand offset from pot center in world frame (±Y lateral, +Z up).
FIXED_GRASP_LATERAL_OFFSET_M = 0.04
FIXED_GRASP_UP_OFFSET_M =  0.0


def grasp_locals_from_fixed_world_offsets(
    pot_pos_world: np.ndarray,
    pot_quat_wxyz: np.ndarray,
    lateral_offset_m: float = FIXED_GRASP_LATERAL_OFFSET_M,
    up_offset_m: float = FIXED_GRASP_UP_OFFSET_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed world lateral/up offsets from the pot center, expressed pot-local.

    Robot base is rotated 180° around world Z:
      - Robot left hand sits at world Y - lateral_offset
      - Robot right hand sits at world Y + lateral_offset
    """
    pot_pos = np.asarray(pot_pos_world, dtype=np.float64).reshape(3)
    q = np.asarray(pot_quat_wxyz, dtype=np.float64).reshape(4)
    q = q / (float(np.linalg.norm(q)) or 1.0)
    w, x, y, z = q[0], q[1], q[2], q[3]
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    lateral = float(lateral_offset_m)
    up = float(up_offset_m)
    left_world = pot_pos + np.array([0.0, -lateral, up], dtype=np.float64)
    # right_world = pot_pos + np.array([0.0, lateral, up], dtype=np.float64)
    right_world = pot_pos + np.array([0.0, lateral, up], dtype=np.float64)
    left_local = rot.T @ (left_world - pot_pos)
    right_local = rot.T @ (right_world - pot_pos)
    return left_local, right_local


def pot_trajectory_to_hand_targets(
    pot_center: np.ndarray,
    pot_quat_wxyz: np.ndarray,
    left_grasp_local: np.ndarray,
    right_grasp_local: np.ndarray,
) -> HandTargetsOutput:
    """Stage 2: pot trajectory + fixed grasp offsets -> dual-hand IK targets."""
    c = np.asarray(pot_center, dtype=np.float64)
    left_grasp = np.asarray(left_grasp_local, dtype=np.float64).reshape(3)
    right_grasp = np.asarray(right_grasp_local, dtype=np.float64).reshape(3)
    n = c.shape[0]

    left_target = np.zeros((n, 3), dtype=np.float64)
    right_target = np.zeros((n, 3), dtype=np.float64)
    left_quat = np.zeros((n, 4), dtype=np.float64)
    right_quat = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        q = np.asarray(pot_quat_wxyz[i], dtype=np.float64).reshape(4)
        w, x, y, z = q / (float(np.linalg.norm(q)) or 1.0)
        Ri = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        left_target[i] = c[i] + Ri @ left_grasp
        right_target[i] = c[i] + Ri @ right_grasp
        q_xyzw = rotmat_to_quat_xyzw(Ri)
        left_quat[i] = q_xyzw
        right_quat[i] = q_xyzw

    return HandTargetsOutput(
        left_target_pos=left_target,
        right_target_pos=right_target,
        left_target_quat_xyzw=left_quat,
        right_target_quat_xyzw=right_quat,
    )
