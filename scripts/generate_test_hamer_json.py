#!/usr/bin/env python3
"""Generate offline replay JSON for xr_teleoperate.

Outputs a structured JSON compatible with:
    python teleop/teleop_hamer_and_arm.py --input-source hamer --hamer-json <file>

The generated file contains per-frame left/right hand wrist poses in robot base frame
and synthetic 21-point local hand keypoints so that arm IK and Inspire hand retargeting
can both be exercised without a live XR device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


Vec3 = np.ndarray
Mat3 = np.ndarray


@dataclass
class MotionSample:
    left_pos: Vec3
    right_pos: Vec3
    left_rot: Mat3
    right_rot: Mat3
    left_curl: float
    right_curl: float


def rot_x(rad: float) -> Mat3:
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def rot_y(rad: float) -> Mat3:
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def rot_z(rad: float) -> Mat3:
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def smoothstep01(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def mirrored_right_rotation(left_rot: Mat3) -> Mat3:
    mirror = np.diag(np.array([1.0, -1.0, 1.0], dtype=np.float64))
    return mirror @ left_rot @ mirror


def make_openpose21_hand(side: str, curl: float, splay: float = 0.0) -> np.ndarray:
    """Build a simple 21-point hand skeleton in wrist-local coordinates.

    OpenPose layout:
        0 wrist
        1..4 thumb
        5..8 index
        9..12 middle
        13..16 ring
        17..20 pinky
    """
    curl = clamp01(curl)
    side_sign = -1.0 if side == "left" else 1.0
    wrist = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    fingers = [
        {"base_y": -0.040, "lengths": [0.035, 0.028, 0.022, 0.018], "base_x": 0.010},  # thumb
        {"base_y": -0.018, "lengths": [0.045, 0.028, 0.022, 0.018], "base_x": 0.035},  # index
        {"base_y": 0.000, "lengths": [0.050, 0.030, 0.024, 0.020], "base_x": 0.040},   # middle
        {"base_y": 0.018, "lengths": [0.045, 0.028, 0.022, 0.018], "base_x": 0.036},   # ring
        {"base_y": 0.034, "lengths": [0.038, 0.024, 0.020, 0.016], "base_x": 0.032},   # pinky
    ]

    points: List[np.ndarray] = [wrist]
    for finger_idx, finger in enumerate(fingers):
        base = np.array(
            [finger["base_x"], side_sign * (finger["base_y"] + splay * finger_idx * 0.004), 0.0],
            dtype=np.float64,
        )

        if finger_idx == 0:
            bend = (20.0 + 45.0 * curl) * math.pi / 180.0
            spread = side_sign * (-0.65 + 0.35 * (1.0 - curl))
            direction = rot_z(spread) @ rot_y(-0.35) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            bend = (5.0 + 75.0 * curl) * math.pi / 180.0
            spread = side_sign * (0.12 * (finger_idx - 2))
            direction = rot_z(spread) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

        joint = base.copy()
        seg_dir = direction.copy()
        for seg_idx, seg_len in enumerate(finger["lengths"]):
            if finger_idx == 0:
                local_bend = bend * (0.35 + 0.18 * seg_idx)
                seg_dir = rot_y(-local_bend) @ seg_dir
            else:
                local_bend = bend * (0.45 + 0.15 * seg_idx)
                seg_dir = rot_z(side_sign * -0.03 * seg_idx) @ rot_y(-local_bend) @ seg_dir
            joint = joint + seg_len * seg_dir
            points.append(joint.copy())

    hand = np.asarray(points, dtype=np.float64)
    if hand.shape != (21, 3):
        raise RuntimeError(f"Unexpected hand shape: {hand.shape}")
    return hand


def phase_segment(t: float, start: float, end: float) -> float:
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    return smoothstep01((t - start) / (end - start))


def motion_home(t: float) -> MotionSample:
    left_pos = np.array([0.25, 0.22, 0.10], dtype=np.float64)
    right_pos = np.array([0.25, -0.22, 0.10], dtype=np.float64)
    left_rot = rot_y(math.radians(8.0)) @ rot_z(math.radians(8.0))
    right_rot = mirrored_right_rotation(left_rot)
    return MotionSample(left_pos, right_pos, left_rot, right_rot, 0.05, 0.05)


def motion_hand_open_close(t: float) -> MotionSample:
    left_pos = np.array([0.25, 0.22, 0.11], dtype=np.float64)
    right_pos = np.array([0.25, -0.22, 0.11], dtype=np.float64)
    left_rot = rot_y(math.radians(10.0)) @ rot_z(math.radians(12.0))
    right_rot = mirrored_right_rotation(left_rot)
    curl = 0.5 - 0.45 * math.cos(2.0 * math.pi * t)
    return MotionSample(left_pos, right_pos, left_rot, right_rot, curl, curl)


def motion_arm_lift(t: float) -> MotionSample:
    z = 0.10 + 0.12 * smoothstep01(0.5 - 0.5 * math.cos(2.0 * math.pi * t))
    x = 0.24 + 0.03 * math.sin(2.0 * math.pi * t)
    y = 0.24 + 0.02 * math.sin(2.0 * math.pi * t)
    left_pos = np.array([x, y, z], dtype=np.float64)
    right_pos = np.array([x, -y, z], dtype=np.float64)
    left_rot = rot_y(math.radians(20.0)) @ rot_z(math.radians(18.0)) @ rot_x(math.radians(-10.0 * math.sin(2.0 * math.pi * t)))
    right_rot = mirrored_right_rotation(left_rot)
    return MotionSample(left_pos, right_pos, left_rot, right_rot, 0.15, 0.15)


def motion_arm_sweep(t: float) -> MotionSample:
    y = 0.18 + 0.12 * math.sin(2.0 * math.pi * t)
    x = 0.26 + 0.04 * math.sin(4.0 * math.pi * t)
    z = 0.12 + 0.04 * math.sin(2.0 * math.pi * t + math.pi / 2.0)
    left_pos = np.array([x, y, z], dtype=np.float64)
    right_pos = np.array([x, -y, z], dtype=np.float64)
    left_rot = rot_y(math.radians(12.0)) @ rot_z(math.radians(30.0 * math.sin(2.0 * math.pi * t)))
    right_rot = mirrored_right_rotation(left_rot)
    return MotionSample(left_pos, right_pos, left_rot, right_rot, 0.10, 0.10)


def motion_combined(t: float) -> MotionSample:
    reach = phase_segment(t, 0.05, 0.30) - phase_segment(t, 0.70, 0.95)
    spread = 0.22 + 0.08 * math.sin(2.0 * math.pi * t)
    z = 0.10 + 0.08 * phase_segment(t, 0.20, 0.48) + 0.04 * math.sin(4.0 * math.pi * t)
    x = 0.23 + 0.10 * reach + 0.02 * math.sin(2.0 * math.pi * t)
    left_pos = np.array([x, spread, z], dtype=np.float64)
    right_pos = np.array([x, -spread, z], dtype=np.float64)

    yaw = math.radians(22.0 * math.sin(2.0 * math.pi * t))
    pitch = math.radians(18.0 + 12.0 * math.sin(4.0 * math.pi * t))
    roll = math.radians(-10.0 * math.sin(2.0 * math.pi * t))
    left_rot = rot_x(roll) @ rot_y(pitch) @ rot_z(yaw)
    right_rot = mirrored_right_rotation(left_rot)

    curl_left = 0.15 + 0.75 * phase_segment(t, 0.32, 0.55) - 0.60 * phase_segment(t, 0.55, 0.82)
    curl_right = 0.15 + 0.75 * phase_segment(t, 0.36, 0.60) - 0.60 * phase_segment(t, 0.60, 0.85)
    return MotionSample(left_pos, right_pos, left_rot, right_rot, clamp01(curl_left), clamp01(curl_right))


MOTIONS: Dict[str, Callable[[float], MotionSample]] = {
    "home": motion_home,
    "hand_open_close": motion_hand_open_close,
    "arm_lift": motion_arm_lift,
    "arm_sweep": motion_arm_sweep,
    "combined": motion_combined,
}


def build_record(frame_idx: int, timestamp_sec: float, side: str, pos: Vec3, rot: Mat3, curl: float) -> Dict:
    return {
        "frame_idx": int(frame_idx),
        "timestamp_sec": float(timestamp_sec),
        "hand_side": side,
        "score": 1.0,
        "p_wrist_base": [float(v) for v in np.asarray(pos, dtype=np.float64).reshape(3)],
        "R_wrist_base": np.asarray(rot, dtype=np.float64).reshape(3, 3).tolist(),
        "keypoints_3d_local": make_openpose21_hand(side=side, curl=curl).tolist(),
        "hand_pose": [],
        "global_orient": [],
    }


def generate_frames(motion_name: str, fps: float, duration: float) -> List[Dict]:
    motion_fn = MOTIONS[motion_name]
    frame_count = max(1, int(round(duration * fps)))
    records: List[Dict] = []
    for frame_idx in range(frame_count):
        t = 0.0 if frame_count == 1 else frame_idx / (frame_count - 1)
        timestamp_sec = frame_idx / fps
        sample = motion_fn(t)
        records.append(build_record(frame_idx, timestamp_sec, "left", sample.left_pos, sample.left_rot, sample.left_curl))
        records.append(build_record(frame_idx, timestamp_sec, "right", sample.right_pos, sample.right_rot, sample.right_curl))
    return records


def build_output(args: argparse.Namespace) -> Dict:
    frames = generate_frames(args.motion, args.fps, args.duration)
    return {
        "meta": {
            "generator": "xr_teleoperate/scripts/generate_test_hamer_json.py",
            "format": "hamer_structured_results.json compatible",
            "motion": args.motion,
            "fps": args.fps,
            "duration_sec": args.duration,
            "frame_count": len(frames) // 2,
            "notes": [
                "Contains p_wrist_base/R_wrist_base, so no cam2base JSON is required for replay.",
                "Contains 21-point keypoints_3d_local, so Inspire/Dex/BrainCo hand retargeting can be exercised.",
                "This is a synthetic debugging trajectory for teleop pipeline validation, not human capture data.",
            ],
        },
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    default_output = os.path.join(os.path.dirname(__file__), "..", "generated", "hamer_structured_test_combined.json")
    parser = argparse.ArgumentParser(description="Generate synthetic offline replay JSON for xr_teleoperate")
    parser.add_argument("--motion", choices=sorted(MOTIONS.keys()), default="combined", help="Test motion preset")
    parser.add_argument("--fps", type=float, default=30.0, help="Replay FPS")
    parser.add_argument("--duration", type=float, default=8.0, help="Duration in seconds")
    parser.add_argument("--output", type=str, default=default_output, help="Output JSON path")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = build_output(args)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=args.indent)

    print(f"Wrote {output_path}")
    print(f"Motion: {args.motion}, fps={args.fps}, duration={args.duration}s, frames={payload['meta']['frame_count']}")


if __name__ == "__main__":
    main()
