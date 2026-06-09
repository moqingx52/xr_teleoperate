#!/usr/bin/env python3
"""Replay pot trajectory (stage1) or pot + dual-arm IK (stage2)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Lock, Value
from pathlib import Path

# Import CasADi before pandas.  Importing pandas first can cause libcasadi to
# resolve against the system libstdc++ instead of conda's (CXXABI_1.3.15 error).
try:
    import casadi as _casadi  # noqa: F401
except Exception as _casadi_exc:
    _casadi = None
    _casadi_import_error = _casadi_exc
else:
    _casadi_import_error = None

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
XR_ROOT = REPO_ROOT / "xr_teleoperate"
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))

from teleop.utils.isaac_shm import try_open_shm  # noqa: E402
from teleop.utils.pot_align import (  # noqa: E402
    apply_align_delta,
    apply_endpoint_linear_correction,
    clamp_z_to_start,
    compose_pose,
    compute_arclength_alpha,
    compute_endpoint_pos_target,
    compute_start_align_delta,
    quat_wxyz_to_rotmat,
    scene_start_pos_with_z_offset,
    wait_scene_pot_pose,
    wait_scene_target_pose,
)
from teleop.utils.pot_retarget import (  # noqa: E402
    FIXED_GRASP_LATERAL_OFFSET_M,
    FIXED_GRASP_UP_OFFSET_M,
    grasp_locals_from_fixed_world_offsets,
)
from teleop.utils.pot_trajectory_clean import clean_pot_trajectory  # noqa: E402


def _as_arr(v: object, n: int, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} has len={arr.size}, expected={n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _pose_wxyz_to_homo(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rotmat(_as_arr(quat_wxyz, 4, "quat_wxyz"))
    T[:3, 3] = _as_arr(pos, 3, "pos")
    return T


def _world_pose_to_base_homo(
    pos_world: np.ndarray,
    quat_wxyz_world: np.ndarray,
    *,
    base_pos_world: np.ndarray,
    base_quat_world_wxyz: np.ndarray,
) -> np.ndarray:
    """Convert an Isaac world-frame target to the xr_teleop IK base frame."""
    T_world_base = _pose_wxyz_to_homo(base_pos_world, base_quat_world_wxyz)
    T_world_target = _pose_wxyz_to_homo(pos_world, quat_wxyz_world)
    return np.linalg.inv(T_world_base) @ T_world_target


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _as_arr(q, 4, "quat_xyzw")
    return np.array([w, x, y, z], dtype=np.float64)


def _load(path: Path, mode: str, *, require_hand_columns: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "entry_type" in df.columns:
        df = df.loc[(df["entry_type"].astype(str) == "command")].copy()
    if "command.t_ns" in df.columns:
        df = df.sort_values("command.t_ns", kind="mergesort")
    if df.empty:
        raise ValueError(f"No playable rows in {path}")
    required = ["retarget.pot_center", "retarget.pot_quat_wxyz"]
    if mode == "pot_ik" and require_hand_columns:
        required.extend(
            [
                "retarget.left_target_pos",
                "retarget.right_target_pos",
                "retarget.left_target_quat_xyzw",
                "retarget.right_target_quat_xyzw",
            ]
        )
    for k in required:
        if k not in df.columns:
            raise ValueError(f"Missing required column for mode={mode}: {k}")
    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay pot-driven trajectory parquet.")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--mode",
        choices=("pot_only", "pot_ik"),
        default="pot_only",
        help="pot_only: kinematic pot only (robot static); pot_ik: pot + dual-arm IK.",
    )
    parser.add_argument("--hz", type=float, default=0.0, help="0 uses command.t_ns pacing.")
    parser.add_argument("--pot-shm-name", default="isaac_pot_pose_ctl")
    parser.add_argument("--pot-shm-size", type=int, default=4096)
    parser.add_argument("--robot-base-pos-world", nargs=3, type=float, default=(0.0, 0.4, 0.0))
    parser.add_argument("--robot-base-quat-world-wxyz", nargs=4, type=float, default=(0.0, 0.0, 0.0, 1.0))
    parser.add_argument("--pot-frame", choices=("robot_base", "world"), default="robot_base")
    parser.add_argument("--align-to-scene-pot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-to-scene-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-z-offset-m", type=float, default=0.18)
    parser.add_argument(
        "--start-z-offset-m",
        type=float,
        default=0.05,
        help="Extra +Z offset applied to scene pot start alignment (m).",
    )
    parser.add_argument(
        "--clamp-z-to-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp trajectory Z so no frame goes below the initial Z.",
    )
    parser.add_argument("--align-timeout-sec", type=float, default=60.0)
    parser.add_argument("--clean-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jump-thresh-m", type=float, default=0.05)
    parser.add_argument("--tail-speed-thresh-m", type=float, default=0.03)
    parser.add_argument("--gripper-open-input", type=float, default=1.0)
    parser.add_argument("--gripper-close-input", type=float, default=0.0)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument(
        "--fixed-grasp-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pot_ik: derive hand targets from hard-coded fixed pot offsets.",
    )
    parser.add_argument(
        "--grasp-from-scene-bbox",
        action=argparse.BooleanOptionalAction,
        dest="fixed_grasp_offsets",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--grasp-z-offset-m",
        type=float,
        default=0.0,
        help=f"pot_ik: extra +Z offset added to the fixed +{FIXED_GRASP_UP_OFFSET_M}m gripper height.",
    )
    parser.add_argument(
        "--grasp-y-half-span-scale",
        type=float,
        default=1.0,
        help=f"pot_ik: scale factor on fixed {FIXED_GRASP_LATERAL_OFFSET_M}m lateral gripper offset.",
    )
    return parser.parse_args()


def _init_pot_ik_stack():
    """Initialize xr_teleoperate G1_29 arm IK + sim controller (stable CasADi path)."""
    if _casadi is None:
        raise RuntimeError(
            "CasADi is required for pot_ik but failed to import. "
            f"reason={type(_casadi_import_error).__name__}: {_casadi_import_error}"
        ) from _casadi_import_error

    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    arm_ctrl = G1_29_ArmController(simulation_mode=True)
    cwd = os.getcwd()
    try:
        # Use cached omnipicker model under xr_teleoperate/ when available.
        os.chdir(str(XR_ROOT))
        arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    finally:
        os.chdir(cwd)
    arm_ik.enable_joint_smoothing = False
    return arm_ctrl, arm_ik


def _make_gripper_controller(sim: bool):
    from teleop.robot_control.robot_hand_inspire import Inspire_Gripper_Controller

    left_value = Value("d", 1.0, lock=True)
    right_value = Value("d", 1.0, lock=True)
    data_lock = Lock()
    state_array = np.zeros(2, dtype=np.float64)
    action_array = np.zeros(2, dtype=np.float64)
    Inspire_Gripper_Controller(
        left_value,
        right_value,
        data_lock,
        state_array,
        action_array,
        simulation_mode=bool(sim),
        active_sides=("left", "right"),
    )
    return left_value, right_value


def _traj_pose_world(
    pos_local: np.ndarray,
    quat_wxyz_local: np.ndarray,
    *,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    pot_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    if pot_frame == "robot_base":
        return compose_pose(base_pos, base_quat_wxyz, pos_local, quat_wxyz_local)
    return np.asarray(pos_local, dtype=np.float64), np.asarray(quat_wxyz_local, dtype=np.float64)


def _build_pot_world_arrays(
    df: pd.DataFrame,
    *,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    pot_frame: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n = len(df)
    pos_world = np.zeros((n, 3), dtype=np.float64)
    quat_world = np.zeros((n, 4), dtype=np.float64)
    times = None
    if "command.t_ns" in df.columns:
        t_ns = df["command.t_ns"].to_numpy(dtype=np.float64)
        times = (t_ns - t_ns[0]) / 1e9

    for i in range(n):
        row = df.iloc[i]
        pos_local = _as_arr(row["retarget.pot_center"], 3, "pot_center")
        quat_local = _as_arr(row["retarget.pot_quat_wxyz"], 4, "pot_quat")
        pos_w, quat_w = _traj_pose_world(
            pos_local,
            quat_local,
            base_pos=base_pos,
            base_quat_wxyz=base_quat_wxyz,
            pot_frame=pot_frame,
        )
        pos_world[i] = pos_w
        quat_world[i] = quat_w
    return pos_world, quat_world, times


def _build_hand_world_arrays(
    df: pd.DataFrame,
    *,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    pot_frame: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    left_pos = np.zeros((n, 3), dtype=np.float64)
    left_quat = np.zeros((n, 4), dtype=np.float64)
    right_pos = np.zeros((n, 3), dtype=np.float64)
    right_quat = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        row = df.iloc[i]
        lp, lq = _traj_pose_world(
            _as_arr(row["retarget.left_target_pos"], 3, "left_target_pos"),
            _xyzw_to_wxyz(_as_arr(row["retarget.left_target_quat_xyzw"], 4, "left_target_quat")),
            base_pos=base_pos,
            base_quat_wxyz=base_quat_wxyz,
            pot_frame=pot_frame,
        )
        rp, rq = _traj_pose_world(
            _as_arr(row["retarget.right_target_pos"], 3, "right_target_pos"),
            _xyzw_to_wxyz(_as_arr(row["retarget.right_target_quat_xyzw"], 4, "right_target_quat")),
            base_pos=base_pos,
            base_quat_wxyz=base_quat_wxyz,
            pot_frame=pot_frame,
        )
        left_pos[i], left_quat[i] = lp, lq
        right_pos[i], right_quat[i] = rp, rq
    return left_pos, left_quat, right_pos, right_quat


def _apply_align_to_positions(
    pos_world: np.ndarray,
    quat_world: np.ndarray,
    T_align: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(pos_world.shape[0])
    pos_out = np.zeros_like(pos_world)
    quat_out = np.zeros_like(quat_world)
    for i in range(n):
        pos_out[i], quat_out[i] = apply_align_delta(T_align, pos_world[i], quat_world[i])
    return pos_out, quat_out


def _apply_endpoint_correction_batch(
    pos_aligned: np.ndarray,
    end_target: np.ndarray,
) -> np.ndarray:
    alpha = compute_arclength_alpha(pos_aligned)
    return apply_endpoint_linear_correction(pos_aligned, end_target, alpha)


def _hands_from_pot_trajectory(
    pot_pos: np.ndarray,
    pot_quat_wxyz: np.ndarray,
    left_grasp_local: np.ndarray,
    right_grasp_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly attach left/right EE poses to the pot trajectory."""
    n = int(pot_pos.shape[0])
    left_pos = np.zeros((n, 3), dtype=np.float64)
    right_pos = np.zeros((n, 3), dtype=np.float64)
    left_quat = np.zeros((n, 4), dtype=np.float64)
    right_quat = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        rot = quat_wxyz_to_rotmat(pot_quat_wxyz[i])
        left_pos[i] = pot_pos[i] + rot @ left_grasp_local
        right_pos[i] = pot_pos[i] + rot @ right_grasp_local
        left_quat[i] = pot_quat_wxyz[i]
        right_quat[i] = pot_quat_wxyz[i]
    return left_pos, left_quat, right_pos, right_quat


def main() -> int:
    args = parse_args()
    mode = str(args.mode)
    fixed_grasp_offsets = bool(args.fixed_grasp_offsets) and mode == "pot_ik"
    path = Path(args.input).expanduser().resolve()
    df = _load(path, mode=mode, require_hand_columns=not fixed_grasp_offsets)

    pot_shm = try_open_shm(name=str(args.pot_shm_name), size=int(args.pot_shm_size))
    if pot_shm is None:
        raise RuntimeError(f"Failed to open pot shm: {args.pot_shm_name}")

    base_pos = np.asarray(args.robot_base_pos_world, dtype=np.float64)
    base_quat_wxyz = np.asarray(args.robot_base_quat_world_wxyz, dtype=np.float64)
    pot_frame = str(args.pot_frame)

    pos_world, quat_world, times = _build_pot_world_arrays(
        df,
        base_pos=base_pos,
        base_quat_wxyz=base_quat_wxyz,
        pot_frame=pot_frame,
    )

    if bool(args.clean_trajectory):
        keep, summary = clean_pot_trajectory(
            pos_world,
            quat_world,
            times,
            jump_thresh_floor=float(args.jump_thresh_m),
            tail_speed_thresh=float(args.tail_speed_thresh_m),
        )
        df = df.iloc[keep].reset_index(drop=True)
        pos_world = pos_world[keep]
        quat_world = quat_world[keep]
        print(
            f"[clean] frames={summary['frames_in']}->{summary['frames_out']} "
            f"removed_outlier={summary['removed_outlier']} "
            f"removed_tail={summary['removed_tail']} "
            f"tail_cut_index={summary['tail_cut_index']}",
            flush=True,
        )

    left_pos_w = left_quat_w = right_pos_w = right_quat_w = None
    if mode == "pot_ik" and not fixed_grasp_offsets:
        left_pos_w, left_quat_w, right_pos_w, right_quat_w = _build_hand_world_arrays(
            df,
            base_pos=base_pos,
            base_quat_wxyz=base_quat_wxyz,
            pot_frame=pot_frame,
        )

    T_align = np.eye(4, dtype=np.float64)
    end_target = None
    scene_pos_w = None
    scene_quat_w = None
    scene_start_pos = None
    if bool(args.align_to_scene_pot):
        scene_pos_w, scene_quat_w = wait_scene_pot_pose(
            pot_shm,
            timeout_sec=float(args.align_timeout_sec),
        )
        scene_start_pos = scene_start_pos_with_z_offset(scene_pos_w, float(args.start_z_offset_m))
        T_align = compute_start_align_delta(
            scene_start_pos,
            scene_quat_w,
            pos_world[0],
            quat_world[0],
        )
        print(
            f"[align] scene_pot={scene_pos_w.round(4).tolist()} "
            f"start_z_offset={float(args.start_z_offset_m):.4f} "
            f"align_start={scene_start_pos.round(4).tolist()} "
            f"traj0={pos_world[0].round(4).tolist()}",
            flush=True,
        )

    left_grasp_local = right_grasp_local = None
    if fixed_grasp_offsets:
        if scene_start_pos is None or scene_quat_w is None:
            raise RuntimeError("fixed pot grasp offsets require --align-to-scene-pot")
        lateral_offset = FIXED_GRASP_LATERAL_OFFSET_M * float(args.grasp_y_half_span_scale)
        up_offset = FIXED_GRASP_UP_OFFSET_M + float(args.grasp_z_offset_m)
        left_grasp_local, right_grasp_local = grasp_locals_from_fixed_world_offsets(
            scene_start_pos,
            scene_quat_w,
            lateral_offset_m=lateral_offset,
            up_offset_m=up_offset,
        )
        print(
            f"[grasp] fixed lateral={lateral_offset:.4f} up={up_offset:.4f} "
            f"left_local={left_grasp_local.round(4).tolist()} "
            f"right_local={right_grasp_local.round(4).tolist()}",
            flush=True,
        )

    if bool(args.align_to_scene_target):
        trivet_pos_w, _ = wait_scene_target_pose(
            pot_shm,
            timeout_sec=float(args.align_timeout_sec),
        )
        end_target = compute_endpoint_pos_target(trivet_pos_w, z_offset_m=float(args.target_z_offset_m))
        print(
            f"[align] end_target={end_target.round(4).tolist()} "
            f"trivet={trivet_pos_w.round(4).tolist()} z_offset={args.target_z_offset_m}",
            flush=True,
        )

    pot_pos_aligned, pot_quat_aligned = _apply_align_to_positions(pos_world, quat_world, T_align)
    pot_pos_final = pot_pos_aligned
    if end_target is not None:
        pot_pos_final = _apply_endpoint_correction_batch(pot_pos_aligned, end_target)

    z_floor = float(pot_pos_final[0, 2])
    if bool(args.clamp_z_to_start):
        pot_pos_final, n_clamped = clamp_z_to_start(pot_pos_final, z_floor=z_floor)
        if n_clamped:
            print(
                f"[align] clamp_z_to_start floor={z_floor:.4f} clamped_frames={n_clamped}",
                flush=True,
            )

    left_pos_final = right_pos_final = None
    left_quat_final = right_quat_final = None
    if mode == "pot_ik" and fixed_grasp_offsets:
        assert left_grasp_local is not None and right_grasp_local is not None
        left_pos_final, left_quat_final, right_pos_final, right_quat_final = _hands_from_pot_trajectory(
            pot_pos_final,
            pot_quat_aligned,
            left_grasp_local,
            right_grasp_local,
        )
    elif mode == "pot_ik" and left_pos_w is not None:
        assert left_quat_w is not None and right_pos_w is not None and right_quat_w is not None
        left_pos_aligned, left_quat_final = _apply_align_to_positions(left_pos_w, left_quat_w, T_align)
        right_pos_aligned, right_quat_final = _apply_align_to_positions(right_pos_w, right_quat_w, T_align)
        left_pos_final = left_pos_aligned
        right_pos_final = right_pos_aligned
        if end_target is not None:
            pot_delta = end_target - pot_pos_aligned[-1]
            alpha = compute_arclength_alpha(pot_pos_aligned)
            left_pos_final = left_pos_aligned + alpha[:, None] * pot_delta
            right_pos_final = right_pos_aligned + alpha[:, None] * pot_delta

    arm_ctrl = None
    arm_ik = None
    left_gripper = None
    right_gripper = None
    if mode == "pot_ik":
        arm_ctrl, arm_ik = _init_pot_ik_stack()
        left_gripper, right_gripper = _make_gripper_controller(sim=True)

    fixed_period = None if args.hz <= 0 else (1.0 / float(args.hz))
    first_t_ns = float(df.iloc[0]["command.t_ns"]) if "command.t_ns" in df.columns else 0.0
    t0_wall = time.monotonic()
    last_print = 0.0
    printed_ik_frame = False
    n_frames = len(df)

    for frame_idx in range(n_frames):
        row = df.iloc[frame_idx]
        frame_start = time.monotonic()
        if fixed_period is None and "command.t_ns" in df.columns:
            target_elapsed = (float(row["command.t_ns"]) - first_t_ns) / 1e9
            sleep_t = t0_wall + target_elapsed - frame_start
            if sleep_t > 0:
                time.sleep(sleep_t)

        if mode == "pot_ik":
            assert (
                arm_ctrl is not None
                and arm_ik is not None
                and left_pos_final is not None
                and left_quat_final is not None
                and right_pos_final is not None
                and right_quat_final is not None
            )
            left_pos_w = left_pos_final[frame_idx]
            left_quat_w = left_quat_final[frame_idx]
            right_pos_w = right_pos_final[frame_idx]
            right_quat_w = right_quat_final[frame_idx]

            left_tf = _world_pose_to_base_homo(
                left_pos_w,
                left_quat_w,
                base_pos_world=base_pos,
                base_quat_world_wxyz=base_quat_wxyz,
            )
            right_tf = _world_pose_to_base_homo(
                right_pos_w,
                right_quat_w,
                base_pos_world=base_pos,
                base_quat_world_wxyz=base_quat_wxyz,
            )
            if not printed_ik_frame:
                print(
                    "[ik] first target "
                    f"left_world={left_pos_w.round(4).tolist()} "
                    f"left_base={left_tf[:3, 3].round(4).tolist()} "
                    f"right_world={right_pos_w.round(4).tolist()} "
                    f"right_base={right_tf[:3, 3].round(4).tolist()}",
                    flush=True,
                )
                printed_ik_frame = True
            current_q = arm_ctrl.get_current_dual_arm_q()
            current_dq = np.zeros(14, dtype=np.float64)
            sol_q, sol_tauff = arm_ik.solve_ik(left_tf, right_tf, current_q, current_dq)
            arm_ctrl.ctrl_dual_arm(np.asarray(sol_q, dtype=np.float64), np.asarray(sol_tauff, dtype=np.float64))

            l_closed = float(np.clip(row.get("aug.left_gripper_closed_smooth", 1.0), 0.0, 1.0))
            r_closed = float(np.clip(row.get("aug.right_gripper_closed_smooth", 1.0), 0.0, 1.0))
            assert left_gripper is not None and right_gripper is not None
            with left_gripper.get_lock():
                left_gripper.value = float(
                    args.gripper_open_input + l_closed * (args.gripper_close_input - args.gripper_open_input)
                )
            with right_gripper.get_lock():
                right_gripper.value = float(
                    args.gripper_open_input + r_closed * (args.gripper_close_input - args.gripper_open_input)
                )

        pot_pos_w = pot_pos_final[frame_idx]
        pot_quat_w = pot_quat_aligned[frame_idx]

        pot_shm.write_data(
            {
                "frame_index": int(frame_idx),
                "pot_pos_world": pot_pos_w.astype(float).tolist(),
                "pot_quat_wxyz": pot_quat_w.astype(float).tolist(),
                "updated_at_sec": float(time.time()),
            }
        )

        now = time.monotonic()
        if now - last_print >= float(args.print_period):
            print(f"mode={mode} frame={frame_idx + 1}/{n_frames}", flush=True)
            last_print = now

        if fixed_period is not None:
            elapsed = time.monotonic() - frame_start
            time.sleep(max(0.0, fixed_period - elapsed))

    print(f"done mode={mode} input={path} frames={n_frames}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
