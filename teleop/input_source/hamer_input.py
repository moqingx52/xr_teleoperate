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


def _normalize_vec(v: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return (v / n).astype(np.float64)


def _orthonormalize_cols(x: np.ndarray, z: np.ndarray) -> Optional[np.ndarray]:
    x = _normalize_vec(x)
    z = _normalize_vec(z)
    if x is None or z is None:
        return None
    y = _normalize_vec(np.cross(z, x))
    if y is None:
        return None
    z2 = _normalize_vec(np.cross(x, y))
    if z2 is None:
        return None
    return np.stack([x, y, z2], axis=1)


def _wrist_basis_from_openpose21(kp21: np.ndarray, side: str) -> np.ndarray:
    kp = np.asarray(kp21, dtype=np.float64).reshape(21, 3)
    wrist = kp[0]
    index_mcp = kp[5]
    little_mcp = kp[17]
    middle_like = 0.5 * (index_mcp + little_mcp)
    x = _normalize_vec(middle_like - wrist)
    if x is None:
        return np.eye(3, dtype=np.float64)
    if side == "left":
        y_hint = _normalize_vec(little_mcp - index_mcp)
    else:
        y_hint = _normalize_vec(index_mcp - little_mcp)
    if y_hint is None:
        return np.eye(3, dtype=np.float64)
    z = _normalize_vec(np.cross(x, y_hint))
    if z is None:
        return np.eye(3, dtype=np.float64)
    R = _orthonormalize_cols(x, z)
    return np.eye(3, dtype=np.float64) if R is None else R


def _rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy_rad_to_R_xyz(rpy_rad: np.ndarray) -> np.ndarray:
    r, p, y = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
    return _rot_x(r) @ _rot_y(p) @ _rot_z(y)


def _import_pandas():
    try:
        import pandas as pd

        return pd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Parquet replay requires pandas in current Python env. Install with: pip install pandas pyarrow"
        ) from e


def _choose_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    for c in candidates:
        for k in columns:
            if k.endswith(c):
                return k
    return None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _infer_meta_dir_from_parquet(parquet_path: str) -> Optional[str]:
    p = os.path.abspath(parquet_path)
    cur = os.path.dirname(p)
    while True:
        name = os.path.basename(cur)
        if name == "data":
            meta_dir = os.path.join(os.path.dirname(cur), "meta")
            if os.path.isdir(meta_dir):
                return meta_dir
            return None
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _to_int_or_none(x: Any) -> Optional[int]:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return None
        x = arr[0]
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


class _OmnipickerTaskGripperPolicy:
    def __init__(self, meta_dir: str, open_value: float = 1.0, close_value: float = 0.0):
        self.meta_dir = os.path.abspath(meta_dir)
        self.open_value = float(open_value)
        self.close_value = float(close_value)
        self._subtask_text: Dict[int, str] = {}
        self._load_subtask_text()

    def _load_subtask_text(self) -> None:
        subtasks_path = os.path.join(self.meta_dir, "subtasks.jsonl")
        items = _load_jsonl(subtasks_path)
        for item in items:
            idx = _to_int_or_none(item.get("subtask_index"))
            text = item.get("task")
            if idx is None or not isinstance(text, str):
                continue
            self._subtask_text[idx] = text.strip()
        if not self._subtask_text:
            raise ValueError(
                f"[OmnipickerTaskGripperPolicy] failed to load subtask text from {subtasks_path}. "
                "Expected JSONL rows with fields: subtask_index, task."
            )
        logger_mp.info(
            f"[OmnipickerTaskGripperPolicy] loaded {len(self._subtask_text)} subtasks from {subtasks_path}"
        )

    @staticmethod
    def _infer_side(task_text: str) -> str:
        s = task_text.lower()
        has_left = ("left arm" in s) or ("left hand" in s) or ("左臂" in task_text) or ("左手" in task_text)
        has_right = ("right arm" in s) or ("right hand" in s) or ("右臂" in task_text) or ("右手" in task_text)
        if has_left and not has_right:
            return "left"
        if has_right and not has_left:
            return "right"
        return "both"

    @staticmethod
    def _infer_action(task_text: str) -> Optional[str]:
        s = task_text.lower()
        close_keywords = (
            "pick up",
            "pickup",
            "pick",
            "grasp",
            "grab",
            "take",
            "close",
            "抓",
            "拿",
            "拾取",
            "夹住",
        )
        open_keywords = (
            "place",
            "put",
            "release",
            "open",
            "放",
            "置于",
            "放入",
            "松开",
        )
        if any(k in s for k in close_keywords) or any(k in task_text for k in close_keywords):
            return "close"
        if any(k in s for k in open_keywords) or any(k in task_text for k in open_keywords):
            return "open"
        return None

    def command(self, subtask_index: Optional[int], prev_left: float, prev_right: float) -> tuple:
        left, right = float(prev_left), float(prev_right)
        if subtask_index is None:
            return left, right
        text = self._subtask_text.get(int(subtask_index))
        if not text:
            return left, right
        action = self._infer_action(text)
        if action is None:
            return left, right
        side = self._infer_side(text)
        cmd = self.open_value if action == "open" else self.close_value
        if side in ("left", "both"):
            left = cmd
        if side in ("right", "both"):
            right = cmd
        return left, right


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


class HamerParquetReader:
    """读取 dataset parquet（left_kp3d/right_kp3d）并输出与 HamerJsonReader 相同帧结构。"""

    def __init__(
        self,
        parquet_path: str,
        loop: bool = False,
        score_thresh: float = 0.5,
        cam2base_json: Optional[str] = None,
        gripper_action_key: str = "action",
        gripper_left_idx: int = 22,
        gripper_right_idx: int = 29,
        action_fallback_mode: str = "ee_base",
        gripper_source: str = "action",
        task_meta_dir: Optional[str] = None,
        joint_action_key: str = "action",
        joint_left_start: int = 0,
        joint_left_end: int = 7,
        joint_right_start: int = 8,
        joint_right_end: int = 15,
    ):
        self.parquet_path = os.path.abspath(parquet_path)
        self.loop = bool(loop)
        self.score_thresh = float(score_thresh)
        self._T_cam2base: Optional[np.ndarray] = None
        self.gripper_action_key = str(gripper_action_key)
        self.gripper_left_idx = int(gripper_left_idx)
        self.gripper_right_idx = int(gripper_right_idx)
        self.action_fallback_mode = str(action_fallback_mode)
        self.gripper_source = str(gripper_source)
        self.joint_action_key = str(joint_action_key)
        self.joint_left_start = int(joint_left_start)
        self.joint_left_end = int(joint_left_end)
        self.joint_right_start = int(joint_right_start)
        self.joint_right_end = int(joint_right_end)
        if self.gripper_source not in ("action", "task"):
            raise ValueError(
                f"Unknown gripper_source: {self.gripper_source}. Expected one of: action, task."
            )
        self.task_meta_dir = (
            os.path.abspath(os.path.expanduser(task_meta_dir))
            if task_meta_dir
            else _infer_meta_dir_from_parquet(self.parquet_path)
        )
        self._task_gripper_policy: Optional[_OmnipickerTaskGripperPolicy] = None
        if self.gripper_source == "task":
            if not self.task_meta_dir:
                raise ValueError(
                    "[HamerParquetReader] gripper_source=task requires dataset meta dir "
                    "(pass --omnipicker-task-meta-dir, or ensure parquet path has sibling ./meta)."
                )
            self._task_gripper_policy = _OmnipickerTaskGripperPolicy(self.task_meta_dir)
        if self.action_fallback_mode not in ("ee_base", "wrist_cam"):
            raise ValueError(
                f"Unknown action_fallback_mode: {self.action_fallback_mode}. "
                "Expected one of: ee_base, wrist_cam."
            )
        if cam2base_json:
            self._T_cam2base = load_T_cam2base_from_json(os.path.abspath(os.path.expanduser(cam2base_json)))
        self._frames: List[Dict[str, Any]] = []
        self._index = 0
        self._load()

    def _load(self):
        if not os.path.isfile(self.parquet_path):
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")
        pd = _import_pandas()
        df = pd.read_parquet(self.parquet_path)
        columns = list(df.columns)

        left_col = _choose_col(columns, ["left_kp3d", "observation.left_kp3d"])
        right_col = _choose_col(columns, ["right_kp3d", "observation.right_kp3d"])
        frame_col = _choose_col(columns, ["frame_index", "episode_frame_index"])
        ts_col = _choose_col(columns, ["timestamp"])
        gripper_col = _choose_col(columns, [self.gripper_action_key])
        joint_col = _choose_col(columns, [self.joint_action_key])
        subtask_col = _choose_col(columns, ["subtask_index", "annotation.human.subtask_description"])
        arm_pose_col = _choose_col(
            columns,
            [
                "action",
                "observation.state",
                "original_action",
                "original_state",
            ],
        )

        has_kp3d = (left_col is not None) and (right_col is not None)
        if (not has_kp3d) and (arm_pose_col is None):
            raise ValueError(
                "Parquet must include either left/right kp3d columns or one vector pose column "
                f"(action/observation.state/original_action/original_state), got columns like: {columns[:20]}"
            )
        if not has_kp3d:
            logger_mp.warning(
                f"[HamerParquetReader] left/right kp3d not found, fallback to arm pose vector '{arm_pose_col}'. "
                "Assume [x,y,z,roll,pitch,yaw] in action slices left[16:22], right[23:29]."
            )
            if self.action_fallback_mode == "wrist_cam":
                if self._T_cam2base is None:
                    raise ValueError(
                        "[HamerParquetReader] action fallback mode 'wrist_cam' requires --hamer-cam2base-json, "
                        "because action slices are interpreted as camera-frame wrist pose."
                    )
                logger_mp.info(
                    "[HamerParquetReader] action fallback mode: wrist_cam "
                    "(action->wrist_cam -> cam2base -> wrist_to_ee)."
                )
            elif self._T_cam2base is not None:
                logger_mp.warning(
                    "[HamerParquetReader] --hamer-cam2base-json is ignored in arm-pose fallback mode "
                    "(no left/right kp3d columns, mode=ee_base)."
                )
        if gripper_col is None:
            if self.gripper_source == "action":
                logger_mp.warning(
                    f"[HamerParquetReader] gripper action key '{self.gripper_action_key}' not found; "
                    "offline gripper input will keep previous value."
                )
        if joint_col is None:
            logger_mp.warning(
                f"[HamerParquetReader] joint action key '{self.joint_action_key}' not found; "
                "offline direct-joint mode will not have joint targets."
            )
        if self.gripper_source == "task" and subtask_col is None:
            raise ValueError(
                "[HamerParquetReader] gripper_source=task requires parquet column `subtask_index`."
            )

        frames: List[Dict[str, Any]] = []
        last_left_gripper = 1.0
        last_right_gripper = 1.0
        for i, row in df.iterrows():
            fi = int(row[frame_col]) if frame_col is not None else int(i)
            ts = float(row[ts_col]) if ts_col is not None else 0.0

            frame = {
                "frame_idx": fi,
                "timestamp_sec": ts,
                "left": {"valid": False},
                "right": {"valid": False},
                "joint_target": {"left": None, "right": None},
            }

            if joint_col is not None:
                joint_arr = np.asarray(row[joint_col], dtype=np.float64).reshape(-1)
                if joint_arr.size >= max(self.joint_left_end, self.joint_right_end):
                    l_joint = joint_arr[self.joint_left_start:self.joint_left_end]
                    r_joint = joint_arr[self.joint_right_start:self.joint_right_end]
                    if (
                        l_joint.size == (self.joint_left_end - self.joint_left_start)
                        and r_joint.size == (self.joint_right_end - self.joint_right_start)
                        and np.all(np.isfinite(l_joint))
                        and np.all(np.isfinite(r_joint))
                    ):
                        frame["joint_target"]["left"] = np.asarray(l_joint, dtype=np.float64).copy()
                        frame["joint_target"]["right"] = np.asarray(r_joint, dtype=np.float64).copy()

            left_gripper = None
            right_gripper = None
            if self.gripper_source == "task":
                subtask_idx = _to_int_or_none(row[subtask_col]) if subtask_col is not None else None
                left_gripper, right_gripper = self._task_gripper_policy.command(
                    subtask_idx, last_left_gripper, last_right_gripper
                )
            elif gripper_col is not None:
                action_arr = np.asarray(row[gripper_col], dtype=np.float64).reshape(-1)
                if action_arr.size > max(self.gripper_left_idx, self.gripper_right_idx):
                    lv = float(action_arr[self.gripper_left_idx])
                    rv = float(action_arr[self.gripper_right_idx])
                    if np.isfinite(lv):
                        left_gripper = lv
                    if np.isfinite(rv):
                        right_gripper = rv
            if left_gripper is not None:
                last_left_gripper = float(left_gripper)
            if right_gripper is not None:
                last_right_gripper = float(right_gripper)

            if has_kp3d:
                for side, col in (("left", left_col), ("right", right_col)):
                    arr = np.asarray(row[col], dtype=np.float64).reshape(-1)
                    if arr.size < 63:
                        continue
                    kp21 = arr[:63].reshape(21, 3)
                    if not np.all(np.isfinite(kp21)):
                        continue
                    p_wrist = kp21[0].copy()
                    R_wrist = _wrist_basis_from_openpose21(kp21, side=side)
                    if self._T_cam2base is not None:
                        p_wrist, R_wrist = wrist_pose_cam_to_base(p_wrist, R_wrist, self._T_cam2base)

                    frame[side] = {
                        "valid": True,
                        "p_wrist_base": p_wrist,
                        "R_wrist_base": R_wrist,
                        "hand_pose": np.asarray([], dtype=np.float64),
                        "global_orient": np.asarray([], dtype=np.float64),
                        "keypoints_3d_local": (kp21 - kp21[0:1, :]),
                        "score": 1.0,
                        "gripper_input": left_gripper if side == "left" else right_gripper,
                    }
            else:
                pose_arr = np.asarray(row[arm_pose_col], dtype=np.float64).reshape(-1)
                # dataset modality:
                # left ee pose  -> [16:22], right ee pose -> [23:29], each as [x,y,z,roll,pitch,yaw]
                for side, s0, s1 in (("left", 16, 22), ("right", 23, 29)):
                    if pose_arr.size < s1:
                        continue
                    p6 = pose_arr[s0:s1]
                    if p6.size != 6 or (not np.all(np.isfinite(p6))):
                        continue
                    p_fallback = np.asarray(p6[:3], dtype=np.float64).reshape(3)
                    R_fallback = _rpy_rad_to_R_xyz(p6[3:6])
                    if self.action_fallback_mode == "wrist_cam":
                        # Omnipicker replay: interpret fallback slices as camera-frame wrist pose.
                        # Then convert to base frame here, and let HamerAdapter apply wrist->EE calibration.
                        p_wrist_base, R_wrist_base = wrist_pose_cam_to_base(p_fallback, R_fallback, self._T_cam2base)
                        frame[side] = {
                            "valid": True,
                            "p_wrist_base": p_wrist_base,
                            "R_wrist_base": R_wrist_base,
                            "hand_pose": np.asarray([], dtype=np.float64),
                            "global_orient": np.asarray([], dtype=np.float64),
                            "keypoints_3d_local": np.zeros((0, 3), dtype=np.float64),
                            "score": 1.0,
                            "gripper_input": left_gripper if side == "left" else right_gripper,
                        }
                    else:
                        p_ee = p_fallback
                        R_ee = R_fallback
                        frame[side] = {
                            "valid": True,
                            "p_ee_base": p_ee,
                            "R_ee_base": R_ee,
                            # Keep wrist fields for backwards compatibility.
                            "p_wrist_base": p_ee.copy(),
                            "R_wrist_base": R_ee.copy(),
                            "hand_pose": np.asarray([], dtype=np.float64),
                            "global_orient": np.asarray([], dtype=np.float64),
                            "keypoints_3d_local": np.zeros((0, 3), dtype=np.float64),
                            "score": 1.0,
                            "gripper_input": left_gripper if side == "left" else right_gripper,
                        }
            if frame["left"]["valid"] or frame["right"]["valid"]:
                frames.append(frame)

        if not frames:
            raise ValueError(f"No playable frames in parquet: {self.parquet_path}")
        self._frames = frames
        logger_mp.info(f"[HamerParquetReader] loaded {len(self._frames)} frames from {self.parquet_path}")

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
            "joint_target": out.get("joint_target", {"left": None, "right": None}),
        }

    def reset(self):
        self._index = 0


class HamerInputSource:
    """与 TeleVuer 并列的输入源，供主循环 `get_frame()` 调用。"""

    def __init__(
        self,
        json_path: Optional[str] = None,
        parquet_path: Optional[str] = None,
        score_thresh: float = 0.5,
        loop: bool = False,
        frame_offset_json: Optional[str] = None,
        cam2base_json: Optional[str] = None,
        gripper_action_key: str = "action",
        gripper_left_idx: int = 22,
        gripper_right_idx: int = 29,
        gripper_source: str = "action",
        omnipicker_task_meta_dir: Optional[str] = None,
        parquet_action_fallback_mode: str = "ee_base",
        joint_action_key: str = "action",
        joint_left_start: int = 0,
        joint_left_end: int = 7,
        joint_right_start: int = 8,
        joint_right_end: int = 15,
    ):
        if bool(json_path) == bool(parquet_path):
            raise ValueError("Provide exactly one of json_path or parquet_path")
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
        if parquet_path:
            self._reader = HamerParquetReader(
                parquet_path=parquet_path,
                loop=loop,
                score_thresh=score_thresh,
                cam2base_json=cam2base_json,
                gripper_action_key=gripper_action_key,
                gripper_left_idx=gripper_left_idx,
                gripper_right_idx=gripper_right_idx,
                gripper_source=gripper_source,
                task_meta_dir=omnipicker_task_meta_dir,
                action_fallback_mode=parquet_action_fallback_mode,
                joint_action_key=joint_action_key,
                joint_left_start=joint_left_start,
                joint_left_end=joint_left_end,
                joint_right_start=joint_right_start,
                joint_right_end=joint_right_end,
            )
        else:
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
