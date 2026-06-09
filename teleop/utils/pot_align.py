"""Align replay trajectory start pose to the pot prim pose read from simulation."""

from __future__ import annotations

import time

import numpy as np


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm([w, x, y, z])) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        s = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s if s > 1e-12 else 1.0
        x = 0.25 * s if s > 1e-12 else 0.0
        y = (m[0, 1] + m[1, 0]) / s if s > 1e-12 else 0.0
        z = (m[0, 2] + m[2, 0]) / s if s > 1e-12 else 0.0
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (float(np.linalg.norm(q)) or 1.0)


def pose_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rotmat(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return m[:3, 3].copy(), rotmat_to_quat_wxyz(m[:3, :3])


def compose_pose(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    local_pos: np.ndarray,
    local_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = pose_to_matrix(base_pos, base_quat_wxyz) @ pose_to_matrix(local_pos, local_quat_wxyz)
    return matrix_to_pose(T)


def compute_start_align_delta(
    scene_pos: np.ndarray,
    scene_quat_wxyz: np.ndarray,
    traj_pos: np.ndarray,
    traj_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Return T_delta so that T_delta @ T_traj_start == T_scene_start."""
    T_scene = pose_to_matrix(scene_pos, scene_quat_wxyz)
    T_traj = pose_to_matrix(traj_pos, traj_quat_wxyz)
    return T_scene @ np.linalg.inv(T_traj)


def apply_align_delta(
    T_delta: np.ndarray,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = T_delta @ pose_to_matrix(pos, quat_wxyz)
    return matrix_to_pose(T)


def _read_pose_block(payload: dict, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    scene = payload.get(key)
    if not isinstance(scene, dict):
        return None
    pos = scene.get("pos", scene.get("position"))
    quat = scene.get("quat_wxyz", scene.get("quat"))
    if pos is None or quat is None:
        return None
    pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
    quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)
    if not (np.all(np.isfinite(pos_arr)) and np.all(np.isfinite(quat_arr))):
        return None
    return pos_arr, quat_arr


def wait_scene_pot_pose(
    shm,
    *,
    timeout_sec: float = 60.0,
    poll_sec: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    while time.monotonic() < deadline:
        payload = shm.read_data() if hasattr(shm, "read_data") else {}
        if not isinstance(payload, dict):
            payload = {}
        pose = _read_pose_block(payload, "prim_pose_world")
        if pose is not None:
            return pose
        time.sleep(float(poll_sec))
    raise TimeoutError("timed out waiting for prim_pose_world from pot_kinematic_replay")


def wait_scene_target_pose(
    shm,
    *,
    timeout_sec: float = 60.0,
    poll_sec: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    while time.monotonic() < deadline:
        payload = shm.read_data() if hasattr(shm, "read_data") else {}
        if not isinstance(payload, dict):
            payload = {}
        pose = _read_pose_block(payload, "target_prim_pose_world")
        if pose is not None:
            return pose
        time.sleep(float(poll_sec))
    raise TimeoutError("timed out waiting for target_prim_pose_world from pot_kinematic_replay")


def compute_endpoint_pos_target(trivet_pos: np.ndarray, z_offset_m: float = 0.18) -> np.ndarray:
    pos = np.asarray(trivet_pos, dtype=np.float64).reshape(3)
    return np.array([pos[0], pos[1], pos[2] + float(z_offset_m)], dtype=np.float64)


def compute_arclength_alpha(pos: np.ndarray) -> np.ndarray:
    """Normalized arc-length parameter in [0, 1] for N trajectory points."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = int(pos.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    total = float(s[-1])
    if total < 1e-12:
        return np.linspace(0.0, 1.0, n, dtype=np.float64)
    return s / total


def apply_endpoint_linear_correction(
    pos: np.ndarray,
    end_target: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Linearly distribute endpoint position error along the trajectory."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    end_target = np.asarray(end_target, dtype=np.float64).reshape(3)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if pos.shape[0] != alpha.shape[0]:
        raise ValueError(f"pos rows={pos.shape[0]} != alpha len={alpha.shape[0]}")
    delta = end_target - pos[-1]
    return pos + alpha[:, None] * delta


def scene_start_pos_with_z_offset(scene_pos: np.ndarray, z_offset_m: float) -> np.ndarray:
    """Return scene pot position with optional +Z offset for start alignment."""
    out = np.asarray(scene_pos, dtype=np.float64).reshape(3).copy()
    out[2] += float(z_offset_m)
    return out


def clamp_z_to_start(pos: np.ndarray, z_floor: float | None = None) -> tuple[np.ndarray, int]:
    """Clamp trajectory Z so no point falls below the initial frame height."""
    out = np.asarray(pos, dtype=np.float64).reshape(-1, 3).copy()
    if out.shape[0] == 0:
        return out, 0
    floor_z = float(out[0, 2] if z_floor is None else z_floor)
    below = out[:, 2] < floor_z
    n_clamped = int(np.sum(below))
    if n_clamped:
        out[below, 2] = floor_z
    return out, n_clamped
