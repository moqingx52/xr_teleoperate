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
        relative_position_mode: bool = False,
        left_home: Optional[np.ndarray] = None,
        right_home: Optional[np.ndarray] = None,
        mirror_lr_across_xz: bool = False,
        relative_compress: bool = False,
        relative_scale: float = 0.02,
        relative_clip_xyz: Optional[np.ndarray] = None,
        debug_freeze_wrist_rotation: bool = False,
    ):
        # relative 模式：位置为 home + scale*(p_ee - p_anchor)；旋转为相对首帧锚点角增量乘同一 scale。
        # relative_compress：仅对缩放后的位置增量做 per-axis clip。
        self.wrist_calib = wrist_calib
        self.smooth_alpha = float(smooth_alpha)
        self.max_gap_frames = int(max_gap_frames)
        self.max_step_m = float(max_step_m)
        self.max_step_rad = float(max_step_rad)
        self.relative_position_mode = bool(relative_position_mode)
        self.left_home = np.asarray(left_home if left_home is not None else [0.25, 0.25, 0.1], dtype=np.float64).reshape(3)
        self.right_home = np.asarray(right_home if right_home is not None else [0.25, -0.25, 0.1], dtype=np.float64).reshape(3)
        self.mirror_lr_across_xz = bool(mirror_lr_across_xz)
        self.relative_compress = bool(relative_compress)
        self.relative_scale = float(relative_scale)
        if relative_clip_xyz is None:
            self.relative_clip_xyz = np.asarray([0.12, 0.12, 0.10], dtype=np.float64)
        else:
            self.relative_clip_xyz = np.asarray(relative_clip_xyz, dtype=np.float64).reshape(3)
        self.debug_freeze_wrist_rotation = bool(debug_freeze_wrist_rotation)
        self._gap_left = 0
        self._gap_right = 0
        self._have_left = False
        self._have_right = False
        self._p_l = np.zeros(3)
        self._R_l = np.eye(3)
        self._p_r = np.zeros(3)
        self._R_r = np.eye(3)
        self._anchor_l = None
        self._anchor_r = None
        self._anchor_R_l = None
        self._anchor_R_r = None
        self._M_xz = np.diag(np.array([1.0, -1.0, 1.0], dtype=np.float64))

    def _side_valid(self, side_data: Dict[str, Any]) -> bool:
        return isinstance(side_data, dict) and side_data.get("valid", False)

    def _mirror_across_xz(self, p: np.ndarray, R: np.ndarray):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        M = self._M_xz
        p_m = M @ p
        R_m = M @ R @ M
        return p_m, R_m

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
            src_side_key = side_key
            if self.mirror_lr_across_xz:
                src_side_key = "right" if side_key == "left" else "left"
            sd = frame_data.get(src_side_key, {})
            valid = self._side_valid(sd)
            if valid:
                setattr(self, gap_attr, 0)
                p_w = sd["p_wrist_base"]
                R_w = sd["R_wrist_base"]
                if self.mirror_lr_across_xz:
                    # Correct order for symmetric remap:
                    # 1) swap side source, 2) mirror wrist pose across robot xz plane, 3) apply target-side wrist->ee calibration.
                    p_w, R_w = self._mirror_across_xz(p_w, R_w)
                p_ee, R_ee = wrist_to_ee_target(ee_side, p_w, R_w, self.wrist_calib)
                if self.relative_position_mode:
                    if side_key == "left":
                        if self._anchor_l is None:
                            self._anchor_l = np.asarray(p_ee, dtype=np.float64).reshape(3).copy()
                            self._anchor_R_l = np.asarray(R_ee, dtype=np.float64).reshape(3, 3).copy()
                        p_rel = np.asarray(p_ee, dtype=np.float64).reshape(3) - self._anchor_l
                        p_rel = p_rel * self.relative_scale
                        if self.relative_compress:
                            p_rel = np.clip(p_rel, -self.relative_clip_xyz, self.relative_clip_xyz)
                        p_ee = self.left_home + p_rel
                        if self.debug_freeze_wrist_rotation:
                            R_ee = self._anchor_R_l.copy()
                        else:
                            R_ee = hf.scale_rotation_about_anchor(self._anchor_R_l, R_ee, self.relative_scale)
                    else:
                        if self._anchor_r is None:
                            self._anchor_r = np.asarray(p_ee, dtype=np.float64).reshape(3).copy()
                            self._anchor_R_r = np.asarray(R_ee, dtype=np.float64).reshape(3, 3).copy()
                        p_rel = np.asarray(p_ee, dtype=np.float64).reshape(3) - self._anchor_r
                        p_rel = p_rel * self.relative_scale
                        if self.relative_compress:
                            p_rel = np.clip(p_rel, -self.relative_clip_xyz, self.relative_clip_xyz)
                        p_ee = self.right_home + p_rel
                        if self.debug_freeze_wrist_rotation:
                            R_ee = self._anchor_R_r.copy()
                        else:
                            R_ee = hf.scale_rotation_about_anchor(self._anchor_R_r, R_ee, self.relative_scale)
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
