"""HaMeR 腕部系到机器人末端目标系的固定外参补偿。"""
from dataclasses import dataclass

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
