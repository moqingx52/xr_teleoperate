import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from teleop.input_source.hamer_input import (  # noqa: E402
    _as_R_wrist_cam,
    _as_vec3,
    load_T_cam2base_from_json,
    wrist_pose_cam_to_base,
)


def _load_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "frames" in raw:
        records = raw["frames"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("JSON must be a list or an object with key 'frames'")
    return [r for r in records if isinstance(r, dict)]


def _side_name(rec: Dict) -> str:
    s = str(rec.get("hand_side", "")).strip().lower()
    if s in ("l", "left"):
        return "left"
    if s in ("r", "right"):
        return "right"
    return "unknown"


def _fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v.tolist()) + "]"


def _print_stats(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"{name}: no data")
        return
    print(f"{name}:")
    print(f"  min  = {_fmt_vec(np.min(arr, axis=0))}")
    print(f"  max  = {_fmt_vec(np.max(arr, axis=0))}")
    print(f"  mean = {_fmt_vec(np.mean(arr, axis=0))}")


def _linear_slope(frame_idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
    if frame_idx.size < 2:
        return np.zeros(3, dtype=np.float64)
    x = frame_idx.astype(np.float64)
    x_mean = float(np.mean(x))
    den = float(np.sum((x - x_mean) ** 2))
    if den < 1e-12:
        return np.zeros(3, dtype=np.float64)
    slopes = []
    for i in range(3):
        y = vals[:, i].astype(np.float64)
        y_mean = float(np.mean(y))
        num = float(np.sum((x - x_mean) * (y - y_mean)))
        slopes.append(num / den)
    return np.asarray(slopes, dtype=np.float64)


def _print_segment_stats(name: str, arr: np.ndarray) -> None:
    n = arr.shape[0]
    if n < 3:
        print(f"{name}: not enough data for segment stats")
        return
    a = arr[: max(1, n // 3)]
    b = arr[n // 3 : max(n // 3 + 1, 2 * n // 3)]
    c = arr[2 * n // 3 :]
    print(f"{name} (first/middle/last mean):")
    print(f"  first  = {_fmt_vec(np.mean(a, axis=0))}")
    print(f"  middle = {_fmt_vec(np.mean(b, axis=0))}")
    print(f"  last   = {_fmt_vec(np.mean(c, axis=0))}")


def _print_drift_diagnostics(
    side: str,
    frame_idx: np.ndarray,
    base_arr: np.ndarray,
    home: np.ndarray,
    relative_compress: bool = False,
    relative_scale: float = 0.02,
    relative_clip_xyz: Optional[np.ndarray] = None,
) -> None:
    if base_arr.shape[0] == 0:
        print("drift diagnostics: no data")
        return

    anchor = base_arr[0].copy()
    rel = (base_arr - anchor[None, :]) * float(relative_scale)
    if relative_compress:
        if relative_clip_xyz is None:
            relative_clip_xyz = np.asarray([0.12, 0.12, 0.10], dtype=np.float64)
        else:
            relative_clip_xyz = np.asarray(relative_clip_xyz, dtype=np.float64).reshape(3)
        rel = np.clip(rel, -relative_clip_xyz, relative_clip_xyz)
    target = home[None, :] + rel
    step = np.diff(target, axis=0) if target.shape[0] > 1 else np.empty((0, 3), dtype=np.float64)

    slope_base = _linear_slope(frame_idx, base_arr)
    slope_rel = _linear_slope(frame_idx, rel)
    slope_target = _linear_slope(frame_idx, target)

    print("drift diagnostics:")
    print(f"  relative_compress (clip scaled delta) = {relative_compress}")
    print(f"  anchor(base first frame) = {_fmt_vec(anchor)}")
    print(f"  home(target center)      = {_fmt_vec(home)}")
    print(f"  slope p_base   (m/frame) = {_fmt_vec(slope_base)}")
    print(f"  slope rel      (m/frame) = {_fmt_vec(slope_rel)}")
    print(f"  slope p_target (m/frame) = {_fmt_vec(slope_target)}")
    rel_label = "scaled rel = scale * (p_base - anchor)" + (" then clipped" if relative_compress else "")
    tgt_label = "p_target = home + rel"
    _print_stats(rel_label, rel)
    _print_stats(tgt_label, target)
    _print_segment_stats("p_target", target)
    if step.shape[0] > 0:
        _print_stats("frame-to-frame step of p_target", step)
        pos_z_ratio = float(np.mean(step[:, 2] > 0.0))
        print(f"  ratio(step.z > 0) = {pos_z_ratio:.3f}")
    else:
        print("frame-to-frame step of p_target: not enough frames")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug HaMeR p_wrist/R_wrist -> p_base transform using T_cam2base."
    )
    parser.add_argument("--hamer-json", required=True, help="Path to hamer structured JSON")
    parser.add_argument("--hamer-cam2base-json", required=True, help="Path to cam2base JSON with 4x4 T_cam2base")
    parser.add_argument("--show-n", type=int, default=5, help="Show first N converted samples per side")
    parser.add_argument(
        "--left-home",
        type=float,
        nargs=3,
        default=[0.25, 0.25, 0.1],
        metavar=("X", "Y", "Z"),
        help="Left home position used to simulate --hamer-relative-pos",
    )
    parser.add_argument(
        "--right-home",
        type=float,
        nargs=3,
        default=[0.25, -0.25, 0.1],
        metavar=("X", "Y", "Z"),
        help="Right home position used to simulate --hamer-relative-pos",
    )
    parser.add_argument(
        "--relative-compress",
        action="store_true",
        help="Simulate --hamer-relative-compress (clip scaled position delta) like teleop_hand_and_arm",
    )
    parser.add_argument("--relative-scale", type=float, default=0.02, help="Scale (p_base - anchor) before optional clip")
    parser.add_argument(
        "--relative-clip",
        type=float,
        nargs=3,
        default=[0.12, 0.12, 0.10],
        metavar=("DX", "DY", "DZ"),
        help="With --relative-compress",
    )
    args = parser.parse_args()

    hamer_json = os.path.abspath(os.path.expanduser(args.hamer_json))
    cam2base_json = os.path.abspath(os.path.expanduser(args.hamer_cam2base_json))

    records = _load_records(hamer_json)
    T = load_T_cam2base_from_json(cam2base_json)
    R_cb = T[:3, :3]
    t_cb = T[:3, 3]

    print("=== Inputs ===")
    print(f"hamer_json:    {hamer_json}")
    print(f"cam2base_json: {cam2base_json}")
    print(f"records:       {len(records)}")
    print("")
    print("=== T_cam2base ===")
    print(np.array2string(T, precision=5, suppress_small=True))
    print("")
    print("=== Decomposed ===")
    print("R_cb:")
    print(np.array2string(R_cb, precision=5, suppress_small=True))
    print(f"t_cb: {_fmt_vec(t_cb)}")
    print("")

    cam_pts = {"left": [], "right": []}
    base_pts = {"left": [], "right": []}
    frame_idx_pts = {"left": [], "right": []}
    samples = {"left": [], "right": []}
    skipped = 0

    for rec in records:
        side = _side_name(rec)
        if side not in ("left", "right"):
            skipped += 1
            continue
        if "p_wrist" not in rec or "R_wrist" not in rec:
            skipped += 1
            continue
        try:
            p_cam = _as_vec3(rec["p_wrist"])
            R_cam = _as_R_wrist_cam(rec["R_wrist"])
            p_base, _ = wrist_pose_cam_to_base(p_cam, R_cam, T)
        except Exception:
            skipped += 1
            continue

        cam_pts[side].append(p_cam)
        base_pts[side].append(p_base)
        frame_idx_pts[side].append(int(rec.get("frame_idx", -1)))
        if len(samples[side]) < max(0, int(args.show_n)):
            samples[side].append(
                (
                    int(rec.get("frame_idx", -1)),
                    p_cam.copy(),
                    p_base.copy(),
                )
            )

    print("=== Counts ===")
    print(f"left usable:   {len(cam_pts['left'])}")
    print(f"right usable:  {len(cam_pts['right'])}")
    print(f"skipped:       {skipped}")
    print("")

    left_home = np.asarray(args.left_home, dtype=np.float64).reshape(3)
    right_home = np.asarray(args.right_home, dtype=np.float64).reshape(3)
    print("=== Relative mode simulation ===")
    print(f"left_home:  {_fmt_vec(left_home)}")
    print(f"right_home: {_fmt_vec(right_home)}")
    print(f"relative_compress: {args.relative_compress}")
    print(f"relative_scale: {args.relative_scale}, relative_compress: {args.relative_compress}")
    if args.relative_compress:
        print(f"relative_clip: {_fmt_vec(np.asarray(args.relative_clip, dtype=np.float64))}")
    print("")

    for side in ("left", "right"):
        print(f"=== {side.upper()} position stats ===")
        cam_arr = np.asarray(cam_pts[side], dtype=np.float64).reshape(-1, 3) if cam_pts[side] else np.empty((0, 3))
        base_arr = np.asarray(base_pts[side], dtype=np.float64).reshape(-1, 3) if base_pts[side] else np.empty((0, 3))
        frame_arr = (
            np.asarray(frame_idx_pts[side], dtype=np.int64).reshape(-1)
            if frame_idx_pts[side]
            else np.empty((0,), dtype=np.int64)
        )
        if frame_arr.size > 0:
            order = np.argsort(frame_arr)
            frame_arr = frame_arr[order]
            cam_arr = cam_arr[order]
            base_arr = base_arr[order]
        _print_stats("p_wrist(cam)", cam_arr)
        _print_stats("p_base", base_arr)
        if frame_arr.size > 0:
            print(f"frame range: [{int(frame_arr[0])}, {int(frame_arr[-1])}] ({frame_arr.size} samples)")
        home = left_home if side == "left" else right_home
        _print_drift_diagnostics(
            side,
            frame_arr.astype(np.float64),
            base_arr,
            home,
            relative_compress=args.relative_compress,
            relative_scale=float(args.relative_scale),
            relative_clip_xyz=np.asarray(args.relative_clip, dtype=np.float64),
        )
        print("samples (frame_idx, p_cam -> p_base):")
        if not samples[side]:
            print("  no samples")
        else:
            for fi, p_cam, p_base in samples[side]:
                print(f"  {fi:>6d}: {_fmt_vec(p_cam)} -> {_fmt_vec(p_base)}")
        print("")


if __name__ == "__main__":
    main()
