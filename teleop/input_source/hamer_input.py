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
    ):
        self.json_path = os.path.abspath(json_path)
        self.loop = loop
        self.score_thresh = float(score_thresh)
        self.frame_offset_map = frame_offset_map or {}
        self.global_frame_offset = int(global_frame_offset)
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
            ts = rec.get("timestamp_sec", rec.get("t", 0.0))
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                ts = 0.0
            side = _norm_side(rec["hand_side"])
            score = float(rec.get("score", 1.0))
            if score < self.score_thresh:
                continue
            p = _as_vec3(rec["p_wrist_base"])
            R = _as_mat33(rec["R_wrist_base"])
            if fi not in by_f:
                by_f[fi] = {"frame_idx": fi, "timestamp_sec": ts, "left": None, "right": None}
            by_f[fi][side] = {
                "valid": True,
                "p_wrist_base": p,
                "R_wrist_base": R,
                "hand_pose": np.asarray(rec.get("hand_pose", []), dtype=np.float64).reshape(-1),
                "global_orient": np.asarray(rec.get("global_orient", []), dtype=np.float64).reshape(-1),
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
        )

    def get_frame(self) -> Optional[Dict[str, Any]]:
        return self._reader.get_frame()

    def reset(self):
        self._reader.reset()
