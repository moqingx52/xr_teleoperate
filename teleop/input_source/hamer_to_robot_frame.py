"""HaMeR 腕部系到机器人末端目标系的固定外参补偿。"""
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class WristToEEConfig:
    t_left: np.ndarray  # 腕系下末端相对平移
    R_left: np.ndarray  # 腕系下末端相对旋转 3x3
    t_right: np.ndarray
    R_right: np.ndarray

    @staticmethod
    def identity():
        return WristToEEConfig(
            t_left=np.zeros(3),
            R_left=np.eye(3),
            t_right=np.zeros(3),
            R_right=np.eye(3),
        )


def wrist_to_ee_target(side: str, p_wrist_base: np.ndarray, R_wrist_base: np.ndarray, calib: WristToEEConfig):
    """
    p_ee = p_wrist + R_wrist @ t_wrist_to_ee
    R_ee = R_wrist @ R_wrist_to_ee
    """
    p = np.asarray(p_wrist_base, dtype=np.float64).reshape(3)
    R = np.asarray(R_wrist_base, dtype=np.float64).reshape(3, 3)
    if side == "left":
        t = np.asarray(calib.t_left, dtype=np.float64).reshape(3)
        R_off = np.asarray(calib.R_left, dtype=np.float64).reshape(3, 3)
    else:
        t = np.asarray(calib.t_right, dtype=np.float64).reshape(3)
        R_off = np.asarray(calib.R_right, dtype=np.float64).reshape(3, 3)
    p_ee = p + R @ t
    R_ee = R @ R_off
    return p_ee, R_ee


def convert(side: str, p_wrist_base: np.ndarray, R_wrist_base: np.ndarray, cfg: WristToEEConfig):
    return wrist_to_ee_target(side, p_wrist_base, R_wrist_base, cfg)


def _rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy_deg_to_R_xyz(rpy_deg) -> np.ndarray:
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    return _rot_x(r) @ _rot_y(p) @ _rot_z(y)


def _as_vec3_or_default(obj: Dict[str, Any], key: str, default: np.ndarray) -> np.ndarray:
    if key not in obj:
        return default.copy()
    return np.asarray(obj[key], dtype=np.float64).reshape(3)


def _parse_rotation(block: Dict[str, Any], default_R: np.ndarray) -> np.ndarray:
    if "R" in block:
        return np.asarray(block["R"], dtype=np.float64).reshape(3, 3)
    if "rpy_deg" in block:
        return _rpy_deg_to_R_xyz(block["rpy_deg"])
    return default_R.copy()


def _parse_side_block(root: Dict[str, Any], side: str, default_t: np.ndarray, default_R: np.ndarray):
    side_block = root.get(side, {})
    t = _as_vec3_or_default(side_block, "t", default_t)
    R = _parse_rotation(side_block, default_R)
    return t, R


def load_wrist_to_ee_config_from_json(
    json_path: str,
    default_cfg: Optional[WristToEEConfig] = None,
) -> WristToEEConfig:
    """
    读取 wrist->EE 固定外参。
    JSON 支持:
      1) side 分组:
         {"left":{"t":[...], "R":[[...],[...],[...]]}, "right":{...}}
         或 {"left":{"rpy_deg":[rx,py,yz]}, "right":{"rpy_deg":[...]}}
      2) 扁平键:
         {"t_left":[...], "R_left":[...], "t_right":[...], "R_right":[...]}
         或 {"rpy_left_deg":[...], "rpy_right_deg":[...]}
    """
    base = default_cfg if default_cfg is not None else WristToEEConfig.identity()
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    t_left = np.asarray(base.t_left, dtype=np.float64).reshape(3)
    t_right = np.asarray(base.t_right, dtype=np.float64).reshape(3)
    R_left = np.asarray(base.R_left, dtype=np.float64).reshape(3, 3)
    R_right = np.asarray(base.R_right, dtype=np.float64).reshape(3, 3)

    if isinstance(root.get("left"), dict) or isinstance(root.get("right"), dict):
        t_left, R_left = _parse_side_block(root, "left", t_left, R_left)
        t_right, R_right = _parse_side_block(root, "right", t_right, R_right)
    else:
        if "t_left" in root:
            t_left = np.asarray(root["t_left"], dtype=np.float64).reshape(3)
        if "t_right" in root:
            t_right = np.asarray(root["t_right"], dtype=np.float64).reshape(3)
        if "R_left" in root:
            R_left = np.asarray(root["R_left"], dtype=np.float64).reshape(3, 3)
        elif "rpy_left_deg" in root:
            R_left = _rpy_deg_to_R_xyz(root["rpy_left_deg"])
        if "R_right" in root:
            R_right = np.asarray(root["R_right"], dtype=np.float64).reshape(3, 3)
        elif "rpy_right_deg" in root:
            R_right = _rpy_deg_to_R_xyz(root["rpy_right_deg"])

    return WristToEEConfig(t_left=t_left, R_left=R_left, t_right=t_right, R_right=R_right)
