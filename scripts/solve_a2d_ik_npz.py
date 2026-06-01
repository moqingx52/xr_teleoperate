#!/usr/bin/env python3
"""Solve A2D Omnipicker arm IK from numpy arrays.

This helper intentionally has no pandas/pyarrow dependency so it can run in
environments that contain CasADi + Pinocchio but not the parquet stack.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
XR_ROOT = REPO_ROOT / "xr_teleoperate"
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))


def _pose_to_tf(eepose_row: np.ndarray, arm: str) -> np.ndarray:
    start = 0 if arm == "left" else 7
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = Rotation.from_quat(eepose_row[start + 3 : start + 7]).as_matrix()
    tf[:3, 3] = eepose_row[start : start + 3]
    return tf


def solve(input_npz: Path, output_npz: Path) -> None:
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    data = np.load(input_npz, allow_pickle=False)
    eepose = np.asarray(data["eepose"], dtype=np.float64)
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    generalized_mask = np.asarray(data["generalized_mask"], dtype=bool)
    protected_mask = np.asarray(data["protected_mask"], dtype=bool)

    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="fastsim_a2d_ik_cache_") as tmp_dir:
            os.chdir(tmp_dir)
            arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    finally:
        os.chdir(cwd)

    arm_ik.enable_joint_smoothing = False
    out = joint_pos.copy()
    status = np.array(["protected" if protected_mask[i] else "original" for i in range(eepose.shape[0])], dtype=object)
    current_q = joint_pos[0].copy()
    current_dq = np.zeros(14, dtype=np.float64)

    for i in range(eepose.shape[0]):
        if protected_mask[i]:
            current_q = joint_pos[i].copy()
            continue
        if not generalized_mask[i]:
            continue
        left_tf = _pose_to_tf(eepose[i], "left")
        right_tf = _pose_to_tf(eepose[i], "right")
        try:
            sol_q, _ = arm_ik.solve_ik(left_tf, right_tf, current_q, current_dq)
            sol_q = np.asarray(sol_q, dtype=np.float64).reshape(-1)
            if sol_q.size != 14 or not np.all(np.isfinite(sol_q)):
                raise ValueError("IK returned invalid 14-vector")
            out[i] = sol_q
            current_q = sol_q
            status[i] = "pinocchio_ok"
        except Exception as exc:  # noqa: BLE001
            out[i] = current_q
            status[i] = f"pinocchio_failed:{type(exc).__name__}"
        if (i + 1) % 100 == 0:
            print(f"solved_through_frame={i + 1}/{eepose.shape[0]}", flush=True)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, joint_pos=out, status=status.astype(str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()
    solve(args.input_npz.resolve(), args.output_npz.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
