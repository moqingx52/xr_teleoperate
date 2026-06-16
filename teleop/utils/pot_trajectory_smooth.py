"""Temporal smoothing for pot trajectory positions and orientations."""

from __future__ import annotations

import numpy as np

from teleop.utils.pot_retarget import ema_smooth


def _default_dt(times: np.ndarray | None, n: int) -> float:
    if times is not None and len(times) >= 2:
        dt = float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
        if dt > 1e-6:
            return dt
    return 1.0 / 30.0


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64).reshape(4)
    b = np.asarray(q1, dtype=np.float64).reshape(4)
    a = a / (float(np.linalg.norm(a)) or 1.0)
    b = b / (float(np.linalg.norm(b)) or 1.0)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + float(t) * (b - a)
        return out / (float(np.linalg.norm(out)) or 1.0)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * float(t)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    out = s0 * a + s1 * b
    return out / (float(np.linalg.norm(out)) or 1.0)


def slerp_ema_quaternions(quat: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average on wxyz quaternions via slerp."""
    arr = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    if arr.shape[0] == 0:
        return arr.copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    out = np.zeros_like(arr)
    out[0] = arr[0] / (float(np.linalg.norm(arr[0])) or 1.0)
    for i in range(1, arr.shape[0]):
        out[i] = _slerp(out[i - 1], arr[i], a)
    return out


def kalman_smooth_positions(
    pos: np.ndarray,
    times: np.ndarray | None = None,
    *,
    process_noise_pos: float = 1e-5,
    process_noise_vel: float = 5e-3,
    measurement_noise: float = 2.5e-5,
) -> np.ndarray:
    """Forward constant-velocity Kalman filter on Nx3 positions."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = int(pos.shape[0])
    if n == 0:
        return pos.copy()
    if n == 1:
        return pos.copy()

    dt = _default_dt(times, n)
    # State: [px, py, pz, vx, vy, vz]
    x = np.zeros(6, dtype=np.float64)
    x[:3] = pos[0]
    P = np.eye(6, dtype=np.float64)
    P[:3, :3] *= float(measurement_noise)
    P[3:, 3:] *= float(process_noise_vel)

    F = np.eye(6, dtype=np.float64)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    H = np.zeros((3, 6), dtype=np.float64)
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.eye(3, dtype=np.float64) * float(measurement_noise)
    q_pos = float(process_noise_pos) * dt * dt
    q_vel = float(process_noise_vel) * dt
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

    out = np.zeros_like(pos)
    out[0] = pos[0]
    I6 = np.eye(6, dtype=np.float64)
    for i in range(1, n):
        if times is not None and i < len(times):
            step_dt = float(times[i] - times[i - 1])
            if step_dt > 1e-6:
                dt = step_dt
                F[0, 3] = F[1, 4] = F[2, 5] = dt
                q_pos = float(process_noise_pos) * dt * dt
                q_vel = float(process_noise_vel) * dt
                Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        x = F @ x
        P = F @ P @ F.T + Q
        z = pos[i]
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I6 - K @ H) @ P
        out[i] = x[:3]
    return out


def smooth_pot_trajectory(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    times: np.ndarray | None = None,
    *,
    method: str = "kalman",
    pos_alpha: float = 0.25,
    quat_alpha: float = 0.3,
    process_noise_pos: float = 1e-5,
    process_noise_vel: float = 5e-3,
    measurement_noise: float = 2.5e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth pot center and wxyz quaternion trajectories."""
    pos_in = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    quat_in = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    if pos_in.shape[0] != quat_in.shape[0]:
        raise ValueError("pos and quat must have the same number of frames")

    mode = str(method).strip().lower()
    if mode in ("none", "off", ""):
        return pos_in.copy(), quat_in.copy()
    if mode == "ema":
        pos_out = ema_smooth(pos_in, alpha=float(pos_alpha))
    elif mode == "kalman":
        pos_out = kalman_smooth_positions(
            pos_in,
            times,
            process_noise_pos=float(process_noise_pos),
            process_noise_vel=float(process_noise_vel),
            measurement_noise=float(measurement_noise),
        )
    else:
        raise ValueError(f"Unsupported smooth method: {method}")

    quat_out = slerp_ema_quaternions(quat_in, alpha=float(quat_alpha))
    return pos_out, quat_out
