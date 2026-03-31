"""从 HaMeR 结构化 JSON 按帧读取左右腕位姿。

与 ``hamer/demo.py`` 约定：默认写入 ``<out_folder>/<structured_file>``，
其中 ``--structured_file`` 默认为 ``hamer_structured_results.json``。
遥操作侧可用 ``--hamer-json`` 指向该文件，或用 ``--hamer-out-dir`` 指向 ``out_folder``。
"""
import json
import logging_mp
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger_mp = logging_mp.getLogger(__name__)


def _as_vec3(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size < 3:
        raise ValueError(f"expected length>=3 for 3-vector, got {a.size}")
    return a[:3].copy()


def _as_mat33(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 9:
        return a.reshape(3, 3).copy()
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return np.asarray(x, dtype=np.float64).reshape(3, 3).copy()
    raise ValueError("R_wrist_base must be 3x3 or length-9")


def _as_R_wrist_cam(x) -> np.ndarray:
    """HaMeR `R_wrist`: 3×3 旋转矩阵，或展平为 9，或轴角 3 维（与 MANO global_orient 首关节一致）。"""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 9:
        return a.reshape(3, 3).copy()
    if a.size == 3:
        return _rotvec_to_mat33(a)
    raise ValueError("R_wrist must be 3x3, length-9, or axis-angle length-3")


def _rotvec_to_mat33(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = v / theta
    x, y, z = k[0], k[1], k[2]
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def load_T_cam2base_from_json(path: str) -> np.ndarray:
    """与 HaMeR ``demo.py --cam2base_json`` 相同：JSON 内含 4×4 ``T_cam2base``（相机→基座）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "T_cam2base" not in raw:
        raise ValueError("cam2base JSON must be an object with key 'T_cam2base' (4x4 matrix)")
    T = np.asarray(raw["T_cam2base"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam2base must have shape (4, 4), got {T.shape}")
    return T


def wrist_pose_cam_to_base(
    p_cam: np.ndarray, R_cam: np.ndarray, T_cam2base: np.ndarray
) -> tuple:
    """与 ``hamer/demo.py`` 中 ``_wrist_pose_cam_to_base`` 一致。"""
    R_cb = T_cam2base[:3, :3]
    t_cb = T_cam2base[:3, 3]
    p_cam = np.asarray(p_cam, dtype=np.float64).reshape(3)
    R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
    p_base = R_cb @ p_cam + t_cb
    R_base = R_cb @ R_cam
    return p_base, R_base


def _record_timestamp_sec(rec: Dict[str, Any]) -> float:
    if "timestamp_sec" in rec:
        try:
            return float(rec["timestamp_sec"])
        except (TypeError, ValueError):
            pass
    if rec.get("t") is not None:
        try:
            return float(rec["t"])
        except (TypeError, ValueError):
            pass
    ts = rec.get("timestamp")
    if isinstance(ts, dict) and "timestamp_sec" in ts:
        try:
            return float(ts["timestamp_sec"])
        except (TypeError, ValueError):
            pass
    return 0.0


def _norm_side(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("l", "left"):
        return "left"
    if s in ("r", "right"):
        return "right"
    raise ValueError(f"unknown hand_side: {s}")


class HamerJsonReader:
    """读取 hamer_structured_results.json，按 frame_idx 聚合左右手。"""

    def __init__(
        self,
        json_path: str,
        loop: bool = False,
        score_thresh: float = 0.5,
        frame_offset_map: Optional[Dict[int, int]] = None,
        global_frame_offset: int = 0,
        cam2base_json: Optional[str] = None,
    ):
        self.json_path = os.path.abspath(json_path)
        self.loop = loop
        self.score_thresh = float(score_thresh)
        self.frame_offset_map = frame_offset_map or {}
        self.global_frame_offset = int(global_frame_offset)
        self._T_cam2base: Optional[np.ndarray] = None
        if cam2base_json:
            self._T_cam2base = load_T_cam2base_from_json(os.path.abspath(os.path.expanduser(cam2base_json)))
        self._frames: List[Dict[str, Any]] = []
        self._index = 0
        self._load()

    def _apply_frame_offset(self, frame_idx: int) -> int:
        off = self.frame_offset_map.get(int(frame_idx), 0) + self.global_frame_offset
        return int(frame_idx) + int(off)

    def _load(self):
        if not os.path.isfile(self.json_path):
            raise FileNotFoundError(f"Hamer JSON not found: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "frames" in raw:
            records = raw["frames"]
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError("JSON must be a list of records or {\"frames\": [...]}")

        by_f: Dict[int, Dict[str, Any]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            fi = rec.get("frame_idx", rec.get("frame", rec.get("idx")))
            if fi is None:
                logger_mp.warning("skip record without frame_idx")
                continue
            fi = self._apply_frame_offset(int(fi))
            ts = _record_timestamp_sec(rec)
            side = _norm_side(rec["hand_side"])
            score = float(rec.get("score", 1.0))
            if score < self.score_thresh:
                continue
            p: Optional[np.ndarray] = None
            R: Optional[np.ndarray] = None
            if "p_wrist_base" in rec and "R_wrist_base" in rec:
                p = _as_vec3(rec["p_wrist_base"])
                R = _as_mat33(rec["R_wrist_base"])
            elif "p_wrist" in rec and "R_wrist" in rec and self._T_cam2base is not None:
                p_cam = _as_vec3(rec["p_wrist"])
                R_cam = _as_R_wrist_cam(rec["R_wrist"])
                p, R = wrist_pose_cam_to_base(p_cam, R_cam, self._T_cam2base)
            elif "p_wrist" in rec and "R_wrist" in rec and self._T_cam2base is None:
                logger_mp.warning(
                    "skip record: has p_wrist/R_wrist but no p_wrist_base; "
                    "pass cam2base_json on xr_teleoperate (or regenerate HaMeR with --cam2base_json)"
                )
                continue
            else:
                logger_mp.warning(
                    "skip record: need (p_wrist_base, R_wrist_base) or (p_wrist, R_wrist) with --hamer-cam2base-json"
                )
                continue
            assert p is not None and R is not None
            if fi not in by_f:
                by_f[fi] = {"frame_idx": fi, "timestamp_sec": ts, "left": None, "right": None}
            by_f[fi][side] = {
                "valid": True,
                "p_wrist_base": p,
                "R_wrist_base": R,
                "hand_pose": np.asarray(rec.get("hand_pose", []), dtype=np.float64).reshape(-1),
                "global_orient": np.asarray(rec.get("global_orient", []), dtype=np.float64).reshape(-1),
                "keypoints_3d_local": np.asarray(rec.get("keypoints_3d_local", []), dtype=np.float64).reshape(-1, 3),
                "score": score,
            }

        sorted_keys = sorted(by_f.keys())
        self._frames = []
        for k in sorted_keys:
            fr = by_f[k]
            for side in ("left", "right"):
                if fr[side] is None:
                    fr[side] = {"valid": False}
            self._frames.append(fr)
        if not self._frames:
            raise ValueError(
                f"No playable frames in {self.json_path}. "
                "Either export p_wrist_base/R_wrist_base from HaMeR (--cam2base_json), "
                "or run xr_teleoperate with --hamer-cam2base-json <same json> when only p_wrist/R_wrist exist."
            )
        logger_mp.info(f"[HamerJsonReader] loaded {len(self._frames)} frames from {self.json_path}")

    def get_frame(self) -> Optional[Dict[str, Any]]:
        """返回一帧标准化结构；读完且非 loop 时返回 None。"""
        if not self._frames:
            return None
        if self._index >= len(self._frames):
            if self.loop:
                self._index = 0
            else:
                return None
        out = self._frames[self._index]
        self._index += 1
        return {
            "frame_idx": out["frame_idx"],
            "timestamp_sec": out["timestamp_sec"],
            "left": out["left"],
            "right": out["right"],
        }

    def reset(self):
        self._index = 0


class HamerInputSource:
    """与 TeleVuer 并列的输入源，供主循环 `get_frame()` 调用。"""

    def __init__(
        self,
        json_path: str,
        score_thresh: float = 0.5,
        loop: bool = False,
        frame_offset_json: Optional[str] = None,
        cam2base_json: Optional[str] = None,
    ):
        frame_offset_map: Dict[int, int] = {}
        global_off = 0
        if frame_offset_json and os.path.isfile(frame_offset_json):
            with open(frame_offset_json, "r", encoding="utf-8") as f:
                off_raw = json.load(f)
            if isinstance(off_raw, dict):
                if "global_offset" in off_raw:
                    global_off = int(off_raw["global_offset"])
                for k, v in off_raw.items():
                    if k == "global_offset":
                        continue
                    try:
                        frame_offset_map[int(k)] = int(v)
                    except (TypeError, ValueError):
                        pass
        self._reader = HamerJsonReader(
            json_path,
            loop=loop,
            score_thresh=score_thresh,
            frame_offset_map=frame_offset_map,
            global_frame_offset=global_off,
            cam2base_json=cam2base_json,
        )

    def get_frame(self) -> Optional[Dict[str, Any]]:
        return self._reader.get_frame()

    def reset(self):
        self._reader.reset()
