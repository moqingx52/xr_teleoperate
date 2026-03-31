import argparse
import json
import os
import sys
from typing import Dict, List

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug HaMeR p_wrist/R_wrist -> p_base transform using T_cam2base."
    )
    parser.add_argument("--hamer-json", required=True, help="Path to hamer structured JSON")
    parser.add_argument("--hamer-cam2base-json", required=True, help="Path to cam2base JSON with 4x4 T_cam2base")
    parser.add_argument("--show-n", type=int, default=5, help="Show first N converted samples per side")
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

    for side in ("left", "right"):
        print(f"=== {side.upper()} position stats ===")
        cam_arr = np.asarray(cam_pts[side], dtype=np.float64).reshape(-1, 3) if cam_pts[side] else np.empty((0, 3))
        base_arr = np.asarray(base_pts[side], dtype=np.float64).reshape(-1, 3) if base_pts[side] else np.empty((0, 3))
        _print_stats("p_wrist(cam)", cam_arr)
        _print_stats("p_base", base_arr)
        print("samples (frame_idx, p_cam -> p_base):")
        if not samples[side]:
            print("  no samples")
        else:
            for fi, p_cam, p_base in samples[side]:
                print(f"  {fi:>6d}: {_fmt_vec(p_cam)} -> {_fmt_vec(p_base)}")
        print("")


if __name__ == "__main__":
    main()
