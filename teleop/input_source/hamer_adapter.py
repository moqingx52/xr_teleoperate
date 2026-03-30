"""HaMeR 帧数据 -> 双臂末端目标（齐次矩阵由主循环或 IK 打包）。"""
from typing import Any, Dict, Optional

import numpy as np

from teleop.input_source import hamer_filters as hf
from teleop.input_source.hamer_to_robot_frame import WristToEEConfig, wrist_to_ee_target


class HamerAdapter:
    def __init__(
        self,
        wrist_calib: WristToEEConfig,
        smooth_alpha: float = 0.2,
        max_gap_frames: int = 5,
        max_step_m: float = 0.03,
        max_step_rad: float = 0.2,
    ):
        self.wrist_calib = wrist_calib
        self.smooth_alpha = float(smooth_alpha)
        self.max_gap_frames = int(max_gap_frames)
        self.max_step_m = float(max_step_m)
        self.max_step_rad = float(max_step_rad)
        self._gap_left = 0
        self._gap_right = 0
        self._have_left = False
        self._have_right = False
        self._p_l = np.zeros(3)
        self._R_l = np.eye(3)
        self._p_r = np.zeros(3)
        self._R_r = np.eye(3)

    def _side_valid(self, side_data: Dict[str, Any]) -> bool:
        return isinstance(side_data, dict) and side_data.get("valid", False)

    def step(self, frame_data: Optional[Dict[str, Any]], current_q: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        current_q 预留与 XR 对齐；当前未使用。
        返回 left_arm / right_arm: valid, p_ee_target, R_ee_target
        """
        _ = current_q
        out_left = {"valid": False, "p_ee_target": self._p_l.copy(), "R_ee_target": self._R_l.copy()}
        out_right = {"valid": False, "p_ee_target": self._p_r.copy(), "R_ee_target": self._R_r.copy()}
        if not frame_data:
            self._gap_left += 1
            self._gap_right += 1
            if self._gap_left > self.max_gap_frames:
                self._have_left = False
            if self._gap_right > self.max_gap_frames:
                self._have_right = False
            if self._have_left and self._gap_left <= self.max_gap_frames:
                out_left["valid"] = True
                out_left["p_ee_target"] = self._p_l.copy()
                out_left["R_ee_target"] = self._R_l.copy()
            if self._have_right and self._gap_right <= self.max_gap_frames:
                out_right["valid"] = True
                out_right["p_ee_target"] = self._p_r.copy()
                out_right["R_ee_target"] = self._R_r.copy()
            return {"left_arm": out_left, "right_arm": out_right}

        for side_key, gap_attr, have_attr, p_attr, R_attr, ee_side in (
            ("left", "_gap_left", "_have_left", "_p_l", "_R_l", "left"),
            ("right", "_gap_right", "_have_right", "_p_r", "_R_r", "right"),
        ):
            sd = frame_data.get(side_key, {})
            valid = self._side_valid(sd)
            if valid:
                setattr(self, gap_attr, 0)
                p_w = sd["p_wrist_base"]
                R_w = sd["R_wrist_base"]
                p_ee, R_ee = wrist_to_ee_target(ee_side, p_w, R_w, self.wrist_calib)
                prev_p = getattr(self, p_attr).copy()
                prev_R = getattr(self, R_attr).copy()
                if getattr(self, have_attr):
                    p_ee = hf.smooth_vec(prev_p, p_ee, self.smooth_alpha)
                    # 旋转按列平滑：先 slerp 近似用 log/exp 在切空间 EMA
                    r_prev = prev_R
                    # 对位置、旋转做步进限幅
                    p_ee = hf.clamp_translation_step(prev_p, p_ee, self.max_step_m)
                    R_ee = hf.clamp_rotation_step(r_prev, R_ee, self.max_step_rad)
                else:
                    p_ee = np.asarray(p_ee, dtype=np.float64).reshape(3)
                    R_ee = np.asarray(R_ee, dtype=np.float64).reshape(3, 3)
                setattr(self, p_attr, p_ee)
                setattr(self, R_attr, R_ee)
                setattr(self, have_attr, True)
                if side_key == "left":
                    out_left["valid"] = True
                    out_left["p_ee_target"] = p_ee.copy()
                    out_left["R_ee_target"] = R_ee.copy()
                else:
                    out_right["valid"] = True
                    out_right["p_ee_target"] = p_ee.copy()
                    out_right["R_ee_target"] = R_ee.copy()
            else:
                setattr(self, gap_attr, getattr(self, gap_attr) + 1)
                if getattr(self, gap_attr) > self.max_gap_frames:
                    setattr(self, have_attr, False)
                if getattr(self, have_attr):
                    if side_key == "left":
                        out_left["valid"] = True
                        out_left["p_ee_target"] = getattr(self, p_attr).copy()
                        out_left["R_ee_target"] = getattr(self, R_attr).copy()
                    else:
                        out_right["valid"] = True
                        out_right["p_ee_target"] = getattr(self, p_attr).copy()
                        out_right["R_ee_target"] = getattr(self, R_attr).copy()

        return {"left_arm": out_left, "right_arm": out_right}

    def build_arm_target(self, side: str, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._side_valid(hand_data):
            return {"valid": False}
        p_ee, R_ee = wrist_to_ee_target(side, hand_data["p_wrist_base"], hand_data["R_wrist_base"], self.wrist_calib)
        return {"valid": True, "p_ee_target": p_ee, "R_ee_target": R_ee}
