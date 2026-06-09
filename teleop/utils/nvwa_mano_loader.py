"""Load nvwa MANO init pickle and export dual-hand wrist poses."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np

from teleop.utils.pot_retarget import rotmat_to_quat_xyzw


def load_T_cam2base_from_json(path: str | Path) -> np.ndarray:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "T_cam2base" not in raw:
        raise ValueError("cam2base JSON must be an object with key 'T_cam2base' (4x4 matrix)")
    T = np.asarray(raw["T_cam2base"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam2base must have shape (4, 4), got {T.shape}")
    return T


def wrist_pose_cam_to_base(
    p_cam: np.ndarray, R_cam: np.ndarray, T_cam2base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    R_cb = T_cam2base[:3, :3]
    t_cb = T_cam2base[:3, 3]
    p_cam = np.asarray(p_cam, dtype=np.float64).reshape(3)
    R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
    p_base = R_cb @ p_cam + t_cb
    R_base = R_cb @ R_cam
    return p_base, R_base

JointsCamera = Literal["left_cam", "right_cam", "avg"]


def _as_rotmat3(value: object, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (1, 3, 3):
        arr = arr.reshape(3, 3)
    if arr.shape != (3, 3):
        raise ValueError(f"{label} must be (3, 3) or (1, 3, 3), got {arr.shape}")
    return arr


def _wrist_position_cam(frame: dict, hand_idx: int, joints_camera: JointsCamera) -> np.ndarray:
    if joints_camera == "left_cam":
        joints = frame["joints_left_cam"][hand_idx]
    elif joints_camera == "right_cam":
        joints = frame["joints_right_cam"][hand_idx]
    else:
        left = np.asarray(frame["joints_left_cam"][hand_idx][0], dtype=np.float64).reshape(3)
        right = np.asarray(frame["joints_right_cam"][hand_idx][0], dtype=np.float64).reshape(3)
        return 0.5 * (left + right)
    return np.asarray(joints[0], dtype=np.float64).reshape(3)


def build_T_cam2base_from_camera_params(
    camera_params_json: str | Path,
    episode_key: str,
    camera_name: str,
) -> np.ndarray:
    raw = json.loads(Path(camera_params_json).read_text(encoding="utf-8"))
    if episode_key not in raw:
        raise KeyError(f"episode {episode_key!r} not found in {camera_params_json}")
    episode = raw[episode_key]
    if camera_name not in episode:
        raise KeyError(f"camera {camera_name!r} not found under episode {episode_key}")
    extrinsic = episode[camera_name]["extrinsic"]
    R = np.asarray(extrinsic["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extrinsic["translation_vector"], dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_nvwa_manoinit_pkl(
    pkl_path: str | Path,
    *,
    cam2base_json: str | Path | None = None,
    camera_params_json: str | Path | None = None,
    camera_params_episode: str = "episode_000000",
    camera_name: str = "head",
    joints_camera: JointsCamera = "left_cam",
    assume_gripper_closed: bool = True,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> dict[str, np.ndarray]:
    """Load nvwa ``*_manoinit*.pkl`` and return robot-base wrist trajectories.

    Each pickle frame stores MANO params plus 21 keypoints in camera frame.
    Wrist position uses keypoint index 0; wrist rotation uses ``global_orient``.
    Hands are ordered by ``is_right`` (0=left, 1=right).
    """
    path = Path(pkl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    if cam2base_json is not None:
        T_cam2base = load_T_cam2base_from_json(str(cam2base_json))
    elif camera_params_json is not None:
        T_cam2base = build_T_cam2base_from_camera_params(
            camera_params_json,
            episode_key=camera_params_episode,
            camera_name=camera_name,
        )
    else:
        raise ValueError("Provide cam2base_json or camera_params_json for camera-to-base transform")

    with path.open("rb") as f:
        frames = pickle.load(f)
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} does not contain a non-empty list of frames")

    start = max(0, int(frame_start))
    end = len(frames) if frame_end is None else min(len(frames), int(frame_end))
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")

    left_pos = []
    right_pos = []
    left_quat = []
    right_quat = []
    frame_ids = []
    for item in frames[start:end]:
        if not isinstance(item, dict):
            continue
        mano_params = item.get("mano_params")
        is_right = np.asarray(item.get("is_right", [0, 1]), dtype=np.int64).reshape(-1)
        if not isinstance(mano_params, list) or len(mano_params) < 2:
            continue

        side_to_idx: dict[str, int] = {}
        for idx, flag in enumerate(is_right.tolist()):
            side_to_idx["right" if int(flag) == 1 else "left"] = int(idx)
        if "left" not in side_to_idx or "right" not in side_to_idx:
            side_to_idx = {"left": 0, "right": 1}

        frame_out: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            hand_idx = side_to_idx[side]
            p_cam = _wrist_position_cam(item, hand_idx, joints_camera)
            R_cam = _as_rotmat3(mano_params[hand_idx]["global_orient"], f"{side}.global_orient")
            p_base, R_base = wrist_pose_cam_to_base(p_cam, R_cam, T_cam2base)
            frame_out[side] = (p_base, R_base)

        left_pos.append(frame_out["left"][0])
        right_pos.append(frame_out["right"][0])
        left_quat.append(rotmat_to_quat_xyzw(frame_out["left"][1]))
        right_quat.append(rotmat_to_quat_xyzw(frame_out["right"][1]))
        frame_ids.append(int(item.get("frame", len(frame_ids))))

    if not left_pos:
        raise ValueError(f"No valid MANO frames parsed from {path}")

    closed = np.ones(len(left_pos), dtype=np.float64) if assume_gripper_closed else np.zeros(len(left_pos))
    return {
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "left_pos": np.asarray(left_pos, dtype=np.float64),
        "right_pos": np.asarray(right_pos, dtype=np.float64),
        "left_quat_xyzw": np.asarray(left_quat, dtype=np.float64),
        "right_quat_xyzw": np.asarray(right_quat, dtype=np.float64),
        "left_closed": closed,
        "right_closed": closed.copy(),
        "command_t_ns": np.arange(len(left_pos), dtype=np.int64) * int(1e9 / 30),
    }
