"""HaMeR 输入平滑与步进限幅（纯函数，便于单测与调参）。"""
import numpy as np
import pinocchio as pin


def smooth_vec(prev: np.ndarray, cur: np.ndarray, alpha: float) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float64).reshape(-1)
    cur = np.asarray(cur, dtype=np.float64).reshape(-1)
    return alpha * cur + (1.0 - alpha) * prev


def hold_last_valid(cur: np.ndarray, prev: np.ndarray, valid: bool) -> np.ndarray:
    return np.asarray(cur, dtype=np.float64) if valid else np.asarray(prev, dtype=np.float64)


def clamp_translation_step(prev: np.ndarray, cur: np.ndarray, max_step_m: float) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float64).reshape(3)
    cur = np.asarray(cur, dtype=np.float64).reshape(3)
    delta = cur - prev
    n = np.linalg.norm(delta)
    if n <= max_step_m or n < 1e-12:
        return cur
    return prev + delta * (max_step_m / n)


def clamp_rotation_step(prev_R: np.ndarray, cur_R: np.ndarray, max_step_rad: float) -> np.ndarray:
    """对相对旋转进行角度限幅：R_rel = prev_R^T @ cur_R，按轴角缩放。"""
    prev_R = np.asarray(prev_R, dtype=np.float64).reshape(3, 3)
    cur_R = np.asarray(cur_R, dtype=np.float64).reshape(3, 3)
    R_rel = prev_R.T @ cur_R
    rotvec = pin.log3(R_rel)
    angle = float(np.linalg.norm(rotvec))
    if angle <= max_step_rad or angle < 1e-8:
        return cur_R
    rotvec_scaled = rotvec * (max_step_rad / angle)
    R_scaled = pin.exp3(rotvec_scaled)
    return prev_R @ R_scaled
