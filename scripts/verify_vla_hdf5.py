#!/usr/bin/env python3
"""Verify ALOHA/ACT-style VLA HDF5 episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_IMAGES = {
    "head": (800, 1280, 3),
    "left_wrist": (480, 848, 3),
    "right_wrist": (480, 848, 3),
}
OPTIONAL_EE_POSE_KEY = "observations/ee_pose_base"


def _iter_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p.resolve() for p in path.glob(pattern) if p.is_file())
    return files


def _sample_black_ratio(dataset: h5py.Dataset, max_samples: int) -> float:
    frames = int(dataset.shape[0])
    if frames == 0:
        return 1.0
    count = min(max_samples, frames)
    indices = np.linspace(0, frames - 1, count, dtype=np.int64)
    black = 0
    for idx in indices:
        frame = dataset[int(idx)]
        if float(np.mean(frame)) <= 1.0:
            black += 1
    return float(black) / float(count)


def verify_file(path: Path, max_black_samples: int) -> tuple[bool, str]:
    errors: list[str] = []
    warnings: list[str] = []
    with h5py.File(path, "r") as h5:
        for key in ("observations/images", "observations/qpos", "action", "timestamps"):
            if key not in h5:
                errors.append(f"missing /{key}")

        if errors:
            return False, "; ".join(errors)

        qpos = h5["observations/qpos"]
        action = h5["action"]
        timestamps = h5["timestamps"]
        image_group = h5["observations/images"]
        lengths = [int(qpos.shape[0]), int(action.shape[0]), int(timestamps.shape[0])]
        ee_pose_status = "absent"

        if qpos.shape[1:] != (16,):
            errors.append(f"qpos shape={qpos.shape}, expected [T,16]")
        if action.shape[1:] != (16,):
            errors.append(f"action shape={action.shape}, expected [T,16]")
        if OPTIONAL_EE_POSE_KEY in h5:
            ee_pose = h5[OPTIONAL_EE_POSE_KEY]
            lengths.append(int(ee_pose.shape[0]))
            ee_pose_status = "present"
            if ee_pose.shape[1:] != (7,):
                errors.append(f"ee_pose_base shape={ee_pose.shape}, expected [T,7]")
            if ee_pose.dtype != np.float32:
                errors.append(f"ee_pose_base dtype={ee_pose.dtype}, expected float32")
            ee_pose_arr = ee_pose[:]
            if not np.all(np.isfinite(ee_pose_arr)):
                errors.append("ee_pose_base contains non-finite values")
            if ee_pose_arr.shape[0] > 0 and ee_pose_arr.shape[1] == 7:
                quat_norm = np.linalg.norm(ee_pose_arr[:, 3:7], axis=1)
                if np.any(np.abs(quat_norm - 1.0) > 5e-2):
                    warnings.append("ee_pose_base quaternion norm deviates > 0.05")
        else:
            warnings.append("missing optional /observations/ee_pose_base")

        black_parts = []
        for camera_name, image_shape in EXPECTED_IMAGES.items():
            if camera_name not in image_group:
                errors.append(f"missing image camera={camera_name}")
                continue
            ds = image_group[camera_name]
            lengths.append(int(ds.shape[0]))
            if ds.shape[1:] != image_shape:
                errors.append(f"{camera_name} shape={ds.shape}, expected [T,{image_shape}]")
            if ds.dtype != np.uint8:
                errors.append(f"{camera_name} dtype={ds.dtype}, expected uint8")
            black_parts.append(f"{camera_name}={_sample_black_ratio(ds, max_black_samples):.3f}")

        if len(set(lengths)) != 1:
            errors.append(f"inconsistent lengths={lengths}")
        if lengths and lengths[0] <= 0:
            errors.append("episode has zero frames")

        if timestamps.shape[0] > 1:
            dt = np.diff(timestamps[:])
            if np.any(dt < -1e-9):
                errors.append("timestamps are not monotonic")

        action_arr = action[:]
        action_min = float(np.min(action_arr)) if action_arr.size else 0.0
        action_max = float(np.max(action_arr)) if action_arr.size else 0.0
        frames = lengths[0] if lengths else 0
        source = h5.attrs.get("source_parquet", "")

    ok = len(errors) == 0
    summary = (
        f"frames={frames} action_range=[{action_min:.4f},{action_max:.4f}] "
        f"black_ratio({', '.join(black_parts)}) ee_pose_base={ee_pose_status} source={source}"
    )
    if not ok:
        summary += " errors=" + "; ".join(errors)
    if warnings:
        summary += " warnings=" + "; ".join(warnings)
    return ok, summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify VLA HDF5 episodes under ./record/hdf5.")
    parser.add_argument("--path", default=str(REPO_ROOT / "record" / "hdf5"), help="HDF5 file or directory.")
    parser.add_argument("--glob", default="episode_*.hdf5", help="Glob used when --path is a directory.")
    parser.add_argument("--max-black-samples", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.path).expanduser().resolve()
    files = _iter_files(path, pattern=str(args.glob))
    if not files:
        print(f"[verify-vla-hdf5] no files found: {path} pattern={args.glob!r}", flush=True)
        return 1

    failed = 0
    print(f"[verify-vla-hdf5] episodes={len(files)}", flush=True)
    for file_path in files:
        ok, summary = verify_file(file_path, max_black_samples=max(1, int(args.max_black_samples)))
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {file_path}: {summary}", flush=True)
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
