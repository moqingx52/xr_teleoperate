# teleop/input_source/egodex_input.py
# -*- coding: utf-8 -*-

"""
Read EgoDex HDF5 and expose the same per-frame schema as HamerInputSource.

Output frame format:
{
    "frame_idx": int,
    "timestamp_sec": float,
    "left": {
        "valid": bool,
        "p_wrist_base": np.ndarray(3,),
        "R_wrist_base": np.ndarray(3,3),
        "keypoints_3d_local": np.ndarray(21,3),   # optional, for future hand bridge
        "score": float,
    },
    "right": {...}
}

Important:
- EgoDex transforms are in ARKit origin frame (world-like frame for this episode),
  not robot base frame.
- For robot replay, we use a root frame (recommended: hip), then rely on the
  existing relative-position mode in HamerAdapter to map motion onto robot home poses.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import logging_mp

logger_mp = logging_mp.getLogger(__name__)


def _import_h5py():
    try:
        import h5py
        return h5py
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "EgoDex HDF5 回放需要安装 h5py，请在当前环境执行: pip install h5py"
        ) from e


def _ensure_tf44(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (4, 4):
        raise ValueError(f"Expected (4,4) transform, got {x.shape}")
    return x


def _make_valid_side_false() -> Dict[str, Any]:
    return {
        "valid": False,
        "p_wrist_base": np.zeros(3, dtype=np.float64),
        "R_wrist_base": np.eye(3, dtype=np.float64),
        "keypoints_3d_local": np.zeros((21, 3), dtype=np.float64),
        "score": 0.0,
    }


class EgoDexInputSource:
    """
    Minimal reader for EgoDex HDF5.
    Recommended usage:
        root_frame='hip'
        + --hamer-relative-pos
        + --hamer-relative-scale ~ 0.8 ~ 1.2
    """

    # 21-keypoint order chosen to match the OpenPose-style 21 used by the existing Hamer bridge:
    # wrist,
    # thumb: 4 joints
    # index/middle/ring/little: 4 joints each
    #
    # We intentionally ignore the ARKit metacarpal joints here, because the current
    # Hamer bridge expects 21 points rather than 25 hand joints.
    LEFT_KP21_NAMES = [
        "leftHand",
        "leftThumbKnuckle",
        "leftThumbIntermediateBase",
        "leftThumbIntermediateTip",
        "leftThumbTip",
        "leftIndexFingerKnuckle",
        "leftIndexFingerIntermediateBase",
        "leftIndexFingerIntermediateTip",
        "leftIndexFingerTip",
        "leftMiddleFingerKnuckle",
        "leftMiddleFingerIntermediateBase",
        "leftMiddleFingerIntermediateTip",
        "leftMiddleFingerTip",
        "leftRingFingerKnuckle",
        "leftRingFingerIntermediateBase",
        "leftRingFingerIntermediateTip",
        "leftRingFingerTip",
        "leftLittleFingerKnuckle",
        "leftLittleFingerIntermediateBase",
        "leftLittleFingerIntermediateTip",
        "leftLittleFingerTip",
    ]

    RIGHT_KP21_NAMES = [
        "rightHand",
        "rightThumbKnuckle",
        "rightThumbIntermediateBase",
        "rightThumbIntermediateTip",
        "rightThumbTip",
        "rightIndexFingerKnuckle",
        "rightIndexFingerIntermediateBase",
        "rightIndexFingerIntermediateTip",
        "rightIndexFingerTip",
        "rightMiddleFingerKnuckle",
        "rightMiddleFingerIntermediateBase",
        "rightMiddleFingerIntermediateTip",
        "rightMiddleFingerTip",
        "rightRingFingerKnuckle",
        "rightRingFingerIntermediateBase",
        "rightRingFingerIntermediateTip",
        "rightRingFingerTip",
        "rightLittleFingerKnuckle",
        "rightLittleFingerIntermediateBase",
        "rightLittleFingerIntermediateTip",
        "rightLittleFingerTip",
    ]

    def __init__(
        self,
        hdf5_path: str,
        loop: bool = False,
        score_thresh: float = 0.2,
        root_frame: str = "hip",   # one of: world, hip, camera
        fps: float = 30.0,
    ):
        self.hdf5_path = os.path.abspath(os.path.expanduser(hdf5_path))
        self.loop = bool(loop)
        self.score_thresh = float(score_thresh)
        self.root_frame = str(root_frame).strip().lower()
        self.fps = float(fps)
        self._index = 0

        if self.root_frame not in ("world", "hip", "camera"):
            raise ValueError("root_frame must be one of: world, hip, camera")

        if not os.path.isfile(self.hdf5_path):
            raise FileNotFoundError(f"EgoDex HDF5 not found: {self.hdf5_path}")

        h5py = _import_h5py()
        self._f = h5py.File(self.hdf5_path, "r")

        if "transforms" not in self._f:
            raise ValueError(f"{self.hdf5_path} has no /transforms group")

        self._tf_group = self._f["transforms"]
        self._conf_group = self._f["confidences"] if "confidences" in self._f else None

        required = ["leftHand", "rightHand", "camera"]
        for k in required:
            if k not in self._tf_group:
                raise ValueError(f"Missing /transforms/{k} in {self.hdf5_path}")

        if self.root_frame == "hip" and "hip" not in self._tf_group:
            raise ValueError("root_frame='hip' requires /transforms/hip")

        self._num_frames = int(self._tf_group["camera"].shape[0])
        if self._num_frames <= 0:
            raise ValueError(f"No frames found in {self.hdf5_path}")

        logger_mp.info(
            f"[EgoDexInputSource] loaded {self._num_frames} frames from {self.hdf5_path}, "
            f"root_frame={self.root_frame}, fps={self.fps}, score_thresh={self.score_thresh}"
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if hasattr(self, "_f") and self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def reset(self):
        self._index = 0

    def _get_root_tf(self, frame_idx: int) -> np.ndarray:
        if self.root_frame == "world":
            return np.eye(4, dtype=np.float64)
        if self.root_frame == "hip":
            return _ensure_tf44(self._tf_group["hip"][frame_idx])
        if self.root_frame == "camera":
            return _ensure_tf44(self._tf_group["camera"][frame_idx])
        raise RuntimeError("Unreachable root_frame")

    def _get_score(self, side: str, frame_idx: int) -> float:
        if self._conf_group is None:
            return 1.0
        name = "leftHand" if side == "left" else "rightHand"
        if name not in self._conf_group:
            return 1.0
        return float(self._conf_group[name][frame_idx])

    def _get_rel_tf(self, tf_name: str, frame_idx: int, T_root_inv: np.ndarray) -> np.ndarray:
        T_world_joint = _ensure_tf44(self._tf_group[tf_name][frame_idx])
        if self.root_frame == "world":
            return T_world_joint
        return T_root_inv @ T_world_joint

    def _get_keypoints_21_local(
        self,
        side: str,
        frame_idx: int,
        T_root_inv: np.ndarray,
    ) -> np.ndarray:
        names = self.LEFT_KP21_NAMES if side == "left" else self.RIGHT_KP21_NAMES
        pts = []
        for name in names:
            if name not in self._tf_group:
                # If any joint is missing, fall back to zeros.
                return np.zeros((21, 3), dtype=np.float64)
            T_rel = self._get_rel_tf(name, frame_idx, T_root_inv)
            pts.append(T_rel[:3, 3].copy())

        pts = np.asarray(pts, dtype=np.float64).reshape(21, 3)
        wrist = pts[0:1, :]
        pts_local = pts - wrist
        return pts_local

    def _make_side(
        self,
        side: str,
        frame_idx: int,
        T_root_inv: np.ndarray,
    ) -> Dict[str, Any]:
        wrist_name = "leftHand" if side == "left" else "rightHand"
        score = self._get_score(side, frame_idx)

        if score < self.score_thresh:
            return _make_valid_side_false()

        T_rel = self._get_rel_tf(wrist_name, frame_idx, T_root_inv)
        p = T_rel[:3, 3].astype(np.float64).copy()
        R = T_rel[:3, :3].astype(np.float64).copy()

        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(R)):
            return _make_valid_side_false()

        kp21_local = self._get_keypoints_21_local(side, frame_idx, T_root_inv)

        return {
            "valid": True,
            "p_wrist_base": p,
            "R_wrist_base": R,
            "keypoints_3d_local": kp21_local,
            "score": score,
        }

    def get_frame(self) -> Optional[Dict[str, Any]]:
        if self._num_frames <= 0:
            return None

        if self._index >= self._num_frames:
            if self.loop:
                self._index = 0
            else:
                return None

        i = self._index
        self._index += 1

        T_root = self._get_root_tf(i)
        T_root_inv = np.linalg.inv(T_root)

        left = self._make_side("left", i, T_root_inv)
        right = self._make_side("right", i, T_root_inv)

        return {
            "frame_idx": int(i),
            "timestamp_sec": float(i) / self.fps,
            "left": left,
            "right": right,
        }
