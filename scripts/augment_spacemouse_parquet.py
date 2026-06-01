#!/usr/bin/env python3
"""Create augmented SpaceMouse EE-command parquet episodes.

The input format is produced by teleop/teleop_spacemouse_ee_and_arm.py.  Raw
SpaceMouse button edges are used to reconstruct binary gripper state, then
out-and-back EE pose perturbations are inserted near gripper events.  Scale
inserted segment length with --perturb-duration-pct-min/max (%% of command
frames), plus --perturb-trans-*-m and --perturb-rot-*-deg.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Import CasADi before pandas/scipy.  In this environment, importing pandas first
# can cause libcasadi to resolve against the system libstdc++ instead of conda's.
try:
    import casadi as _casadi  # noqa: F401
except Exception:
    _casadi = None

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
XR_ROOT = REPO_ROOT / "xr_teleoperate"
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))

BTN_1 = 4096
BTN_2 = 8192
BTN_3 = 16384
BTN_4 = 32768

EVENT_NONE = "none"
EVENT_OPEN_TO_CLOSE = "open_to_close"
EVENT_CLOSE_TO_OPEN = "close_to_open"
EVENT_HOLD_CLOSED_START = "hold_closed_start"
EVENT_HOLD_CLOSED_END = "hold_closed_end"


@dataclass(frozen=True)
class CommandData:
    df: pd.DataFrame
    command_indices: np.ndarray
    command_t_ns: np.ndarray
    eepose: np.ndarray
    joint_pos: np.ndarray


@dataclass(frozen=True)
class GripperSeries:
    left_raw: np.ndarray
    right_raw: np.ndarray
    left_smooth: np.ndarray
    right_smooth: np.ndarray
    event_type: list[str]
    event_arm: list[str]
    event_indices: list[int]


@dataclass(frozen=True)
class PerturbationSpec:
    anchor_index: int
    arm: str
    xyz: np.ndarray
    rpy_deg: np.ndarray
    duration_frac: float
    insert_count: int
    wall_duration_sec: float


@dataclass(frozen=True)
class SegmentBridgeSpec:
    event_indices: tuple[int, int]
    protected_ranges: tuple[tuple[int, int, int], tuple[int, int, int]]
    bridge_segments: tuple[tuple[int, int, int], ...]


def _list_array(value: object, expected: int, column: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected:
        raise ValueError(f"{column} has length {arr.size}, expected {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{column} contains non-finite values")
    return arr


def _load_command_data(df: pd.DataFrame) -> CommandData:
    if "entry_type" not in df.columns:
        raise ValueError("Missing `entry_type` column")
    cmd_mask = df["entry_type"].astype(str) == "command"
    cmd_df = df.loc[cmd_mask & df["command.t_ns"].notna()].copy()
    if cmd_df.empty:
        raise ValueError("No command rows found")
    cmd_df = cmd_df.sort_values("command.t_ns", kind="mergesort")
    command_indices = cmd_df.index.to_numpy()
    t_ns = cmd_df["command.t_ns"].to_numpy(dtype=np.float64)
    eepose = np.vstack([_list_array(v, 14, "command.eepose") for v in cmd_df["command.eepose"]])
    joint_pos = np.vstack([_list_array(v, 14, "command.ik_joint_pos") for v in cmd_df["command.ik_joint_pos"]])
    return CommandData(df=cmd_df, command_indices=command_indices, command_t_ns=t_ns, eepose=eepose, joint_pos=joint_pos)


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = float(values[0])
    a = float(np.clip(alpha, 0.0, 1.0))
    for i in range(1, values.size):
        out[i] = out[i - 1] + a * (float(values[i]) - out[i - 1])
    return out


def _estimate_hz(t_ns: np.ndarray) -> float:
    if t_ns.size < 2:
        return 30.0
    dt = np.diff(t_ns) / 1e9
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if dt.size == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def _reconstruct_gripper(
    df: pd.DataFrame,
    command_t_ns: np.ndarray,
    smooth_alpha: float,
    closed_high: float,
    open_low: float,
    hold_closed_sec: float,
) -> GripperSeries:
    raw = df.loc[df["entry_type"].astype(str) == "raw"].copy()
    raw = raw[raw["raw.t_ns"].notna()].sort_values("raw.t_ns", kind="mergesort")

    left = np.zeros(command_t_ns.size, dtype=np.float64)
    right = np.zeros(command_t_ns.size, dtype=np.float64)
    selected = "left"
    left_state = 0.0
    right_state = 0.0
    raw_pos = 0
    raw_t = raw["raw.t_ns"].to_numpy(dtype=np.float64) if not raw.empty else np.zeros(0)
    raw_edges = raw["raw.edges_mask"].fillna(0).to_numpy(dtype=np.int64) if not raw.empty else np.zeros(0, dtype=np.int64)

    for i, t_ns in enumerate(command_t_ns):
        while raw_pos < raw_t.size and raw_t[raw_pos] <= t_ns:
            edges = int(raw_edges[raw_pos])
            if edges & BTN_1:
                selected = "left"
            if edges & BTN_2:
                selected = "right"
            if edges & BTN_3:
                if selected == "left":
                    left_state = 0.0
                else:
                    right_state = 0.0
            if edges & BTN_4:
                if selected == "left":
                    left_state = 1.0
                else:
                    right_state = 1.0
            raw_pos += 1
        left[i] = left_state
        right[i] = right_state

    left_smooth = _ema(left, smooth_alpha)
    right_smooth = _ema(right, smooth_alpha)
    event_type, event_arm, event_indices = _detect_events(
        command_t_ns=command_t_ns,
        left_smooth=left_smooth,
        right_smooth=right_smooth,
        closed_high=closed_high,
        open_low=open_low,
        hold_closed_sec=hold_closed_sec,
    )
    return GripperSeries(
        left_raw=left,
        right_raw=right,
        left_smooth=left_smooth,
        right_smooth=right_smooth,
        event_type=event_type,
        event_arm=event_arm,
        event_indices=event_indices,
    )


def _hysteresis_closed(values: np.ndarray, closed_high: float, open_low: float) -> np.ndarray:
    out = np.zeros(values.size, dtype=bool)
    state = bool(values[0] >= closed_high) if values.size else False
    for i, value in enumerate(values):
        if value >= closed_high:
            state = True
        elif value <= open_low:
            state = False
        out[i] = state
    return out


def _detect_events(
    command_t_ns: np.ndarray,
    left_smooth: np.ndarray,
    right_smooth: np.ndarray,
    closed_high: float,
    open_low: float,
    hold_closed_sec: float,
) -> tuple[list[str], list[str], list[int]]:
    n = command_t_ns.size
    event_type = [EVENT_NONE] * n
    event_arm = [""] * n
    event_indices: list[int] = []
    hz = _estimate_hz(command_t_ns)
    hold_frames = max(1, int(round(hold_closed_sec * hz)))

    for arm, smooth in (("left", left_smooth), ("right", right_smooth)):
        closed = _hysteresis_closed(smooth, closed_high=closed_high, open_low=open_low)
        prev = False
        for i, cur in enumerate(closed):
            if cur and not prev:
                event_type[i] = EVENT_OPEN_TO_CLOSE
                event_arm[i] = arm
                event_indices.append(i)
            elif prev and not cur:
                event_type[i] = EVENT_CLOSE_TO_OPEN
                event_arm[i] = arm
                event_indices.append(i)
            prev = bool(cur)

        i = 0
        while i < n:
            if not closed[i]:
                i += 1
                continue
            j = i
            while j < n and closed[j]:
                j += 1
            if j - i >= hold_frames:
                if event_type[i] == EVENT_NONE:
                    event_type[i] = EVENT_HOLD_CLOSED_START
                    event_arm[i] = arm
                    event_indices.append(i)
                end_idx = j - 1
                if event_type[end_idx] == EVENT_NONE:
                    event_type[end_idx] = EVENT_HOLD_CLOSED_END
                    event_arm[end_idx] = arm
                    event_indices.append(end_idx)
                mid = i + (j - i) // 2
                event_indices.append(mid)
            i = j

    event_indices = sorted(set(int(i) for i in event_indices if 0 <= i < n))
    if not event_indices and n:
        event_indices = [n // 2]
    return event_type, event_arm, event_indices


def _pose_to_tf(eepose: np.ndarray, arm: str) -> np.ndarray:
    start = 0 if arm == "left" else 7
    xyz = eepose[start : start + 3]
    quat_xyzw = eepose[start + 3 : start + 7]
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    tf[:3, 3] = xyz
    return tf


def _tf_to_pose_vec(left_tf: np.ndarray, right_tf: np.ndarray) -> list[float]:
    left_quat = Rotation.from_matrix(left_tf[:3, :3]).as_quat()
    right_quat = Rotation.from_matrix(right_tf[:3, :3]).as_quat()
    return (
        left_tf[:3, 3].astype(float).tolist()
        + left_quat.astype(float).tolist()
        + right_tf[:3, 3].astype(float).tolist()
        + right_quat.astype(float).tolist()
    )


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _sample_perturbation(
    command_data: CommandData,
    gripper: GripperSeries,
    rng: np.random.Generator,
    protected_mask: np.ndarray,
    duration_pct_min: float,
    duration_pct_max: float,
    trans_min_m: float,
    trans_max_m: float,
    rot_min_deg: float,
    rot_max_deg: float,
) -> PerturbationSpec:
    lo, hi = float(trans_min_m), float(trans_max_m)
    if lo > hi:
        lo, hi = hi, lo
    if lo < 0.0 or hi < 0.0:
        raise ValueError("trans_min_m and trans_max_m must be non-negative")
    rlo, rhi = float(rot_min_deg), float(rot_max_deg)
    if rlo > rhi:
        rlo, rhi = rhi, rlo
    if rlo < 0.0:
        raise ValueError("rot_min_deg must be non-negative")

    flo = float(duration_pct_min) / 100.0
    fhi = float(duration_pct_max) / 100.0
    if flo > fhi:
        flo, fhi = fhi, flo
    if flo <= 0.0 or fhi <= 0.0:
        raise ValueError("perturb duration pct min/max must be > 0")
    if flo > 1.0 or fhi > 1.0:
        raise ValueError("perturb duration pct min/max must be <= 100")

    n = int(command_data.command_t_ns.size)
    if protected_mask.shape != (n,):
        raise ValueError("protected_mask shape mismatch")
    candidate = np.where(~protected_mask)[0]
    candidate = candidate[candidate < (n - 1)]
    if candidate.size == 0:
        raise ValueError("No candidate anchor outside protected segments")

    anchor = int(rng.choice(candidate))
    if gripper.event_arm[anchor] in ("left", "right"):
        arm = gripper.event_arm[anchor]
    else:
        arm = "left" if float(rng.random()) < 0.5 else "right"
    radius = float(rng.uniform(lo, hi))
    xyz = _random_unit_vector(rng) * radius
    angle_mag = float(rng.uniform(rlo, rhi))
    rpy_deg = _random_unit_vector(rng) * angle_mag
    anchor = int(np.clip(anchor, 0, command_data.command_t_ns.size - 2))

    duration_frac = float(rng.uniform(flo, fhi))
    insert_count = max(2, int(round(duration_frac * max(n, 1))))
    hz = _estimate_hz(command_data.command_t_ns)
    wall_duration_sec = float(insert_count) / max(hz, 1e-6)
    return PerturbationSpec(
        anchor_index=anchor,
        arm=arm,
        xyz=xyz,
        rpy_deg=rpy_deg,
        duration_frac=duration_frac,
        insert_count=insert_count,
        wall_duration_sec=wall_duration_sec,
    )


def _left_toggle_event_indices(gripper: GripperSeries) -> list[int]:
    idx: list[int] = []
    for i, (event_type, event_arm) in enumerate(zip(gripper.event_type, gripper.event_arm)):
        if event_arm != "left":
            continue
        if event_type in (EVENT_OPEN_TO_CLOSE, EVENT_CLOSE_TO_OPEN):
            idx.append(i)
    return idx


def _build_protected_mask(
    command_count: int,
    gripper: GripperSeries,
    protect_window_frames: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    if command_count <= 0:
        return np.zeros(0, dtype=bool), []

    n = max(0, int(protect_window_frames))
    left_toggle = _left_toggle_event_indices(gripper)
    protected = np.zeros(command_count, dtype=bool)
    protected_ranges: list[tuple[int, int, int]] = []
    for event_idx in left_toggle[:2]:
        start = max(0, int(event_idx) - n)
        end = min(command_count - 1, int(event_idx) + n)
        protected[start : end + 1] = True
        protected_ranges.append((start, end, int(event_idx)))
    return protected, protected_ranges


def _require_segment_bridge_spec(
    command_count: int,
    gripper: GripperSeries,
    protect_window_frames: int,
) -> SegmentBridgeSpec:
    if command_count <= 0:
        raise ValueError("No command frames available")
    left_toggle = _left_toggle_event_indices(gripper)
    if len(left_toggle) < 2:
        raise ValueError(
            "segment_bridge requires at least two left gripper toggle events "
            f"(open_to_close/close_to_open), got {len(left_toggle)}"
        )

    window = max(0, int(protect_window_frames))
    event_indices = (int(left_toggle[0]), int(left_toggle[1]))
    protected_ranges_raw = []
    for event_idx in event_indices:
        start = max(0, event_idx - window)
        end = min(command_count - 1, event_idx + window)
        protected_ranges_raw.append((start, end, event_idx))
    protected_ranges_raw.sort(key=lambda item: item[0])

    first, second = protected_ranges_raw
    if first[1] >= second[0]:
        raise ValueError(
            "segment_bridge protected windows overlap; reduce --protect-window-frames "
            f"or use a longer episode. windows={protected_ranges_raw}"
        )

    bridge_segments: list[tuple[int, int, int]] = []
    candidates = (
        (0, first[0] - 1, 0),
        (first[1] + 1, second[0] - 1, 1),
        (second[1] + 1, command_count - 1, 2),
    )
    for start, end, segment_id in candidates:
        if start <= end:
            bridge_segments.append((start, end, segment_id))

    return SegmentBridgeSpec(
        event_indices=event_indices,
        protected_ranges=(first, second),
        bridge_segments=tuple(bridge_segments),
    )


def _protected_mask_from_ranges(command_count: int, protected_ranges: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    protected = np.zeros(command_count, dtype=bool)
    for start, end, _ in protected_ranges:
        protected[int(start) : int(end) + 1] = True
    return protected


def _bezier(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    controls = np.asarray(points, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    degree = controls.shape[0] - 1
    out = np.zeros((tt.size, controls.shape[1]), dtype=np.float64)
    for k in range(degree + 1):
        coeff = math.comb(degree, k) * ((1.0 - tt) ** (degree - k)) * (tt**k)
        out += coeff[:, None] * controls[k]
    return out


def _sample_bridge_xyz(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    frame_count: int,
    control_points: int,
    noise_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if frame_count == 1:
        return np.asarray(start_xyz, dtype=np.float64).reshape(1, 3)

    interior_count = max(0, int(control_points))
    p0 = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    p1 = np.asarray(end_xyz, dtype=np.float64).reshape(3)
    controls = [p0]
    for k in range(1, interior_count + 1):
        u = k / float(interior_count + 1)
        base = (1.0 - u) * p0 + u * p1
        offset = rng.normal(size=3)
        norm = float(np.linalg.norm(offset))
        if norm > 1e-9:
            offset = offset / norm
        else:
            offset = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        amp = float(max(0.0, noise_m)) * math.sin(math.pi * u) * float(rng.uniform(0.25, 1.0))
        controls.append(base + amp * offset)
    controls.append(p1)
    t = np.linspace(0.0, 1.0, frame_count)
    return _bezier(np.vstack(controls), t)


def _sample_bridge_rotations(
    start_quat: np.ndarray,
    end_quat: np.ndarray,
    frame_count: int,
    noise_deg: float,
    rng: np.random.Generator,
) -> Rotation:
    start_rot = Rotation.from_quat(np.asarray(start_quat, dtype=np.float64).reshape(4))
    end_rot = Rotation.from_quat(np.asarray(end_quat, dtype=np.float64).reshape(4))
    if frame_count <= 1:
        return Rotation.from_quat(np.repeat(start_rot.as_quat()[None, :], max(frame_count, 1), axis=0))

    t = np.linspace(0.0, 1.0, frame_count)
    base = Slerp([0.0, 1.0], Rotation.concatenate([start_rot, end_rot]))(t)
    max_rad = math.radians(max(0.0, float(noise_deg)))
    if max_rad <= 0.0:
        return base

    axis = _random_unit_vector(rng)
    signed_mag = float(rng.uniform(-max_rad, max_rad))
    perturb = Rotation.from_rotvec((np.sin(math.pi * t) * signed_mag)[:, None] * axis[None, :])
    return perturb * base


def _bridge_pose_segment(
    command_data: CommandData,
    start: int,
    end: int,
    start_pose_idx: int,
    end_pose_idx: int,
    rng: np.random.Generator,
    control_points: int,
    trans_noise_m: float,
    rot_noise_deg: float,
) -> np.ndarray:
    frame_count = int(end) - int(start) + 1
    out = np.zeros((frame_count, 14), dtype=np.float64)
    start_pose = command_data.eepose[int(start_pose_idx)]
    end_pose = command_data.eepose[int(end_pose_idx)]
    for arm_start in (0, 7):
        xyz = _sample_bridge_xyz(
            start_xyz=start_pose[arm_start : arm_start + 3],
            end_xyz=end_pose[arm_start : arm_start + 3],
            frame_count=frame_count,
            control_points=control_points,
            noise_m=trans_noise_m,
            rng=rng,
        )
        rots = _sample_bridge_rotations(
            start_quat=start_pose[arm_start + 3 : arm_start + 7],
            end_quat=end_pose[arm_start + 3 : arm_start + 7],
            frame_count=frame_count,
            noise_deg=rot_noise_deg,
            rng=rng,
        )
        out[:, arm_start : arm_start + 3] = xyz
        out[:, arm_start + 3 : arm_start + 7] = rots.as_quat()
    return out


def _solve_inserted_ik(rows: list[dict], use_ik: bool) -> None:
    if not use_ik:
        for row in rows:
            row["aug.ik_status"] = "copied"
        return

    try:
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    except Exception as exc:  # noqa: BLE001 - IK is optional for producing inspectable variants.
        print(
            "warning: cannot import G1_29_ArmIK; inserted frames will copy anchor joints. "
            f"reason={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        for row in rows:
            row["aug.ik_status"] = f"ik_unavailable:{type(exc).__name__}"
        return

    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="fastsim_ik_cache_") as tmp_dir:
            os.chdir(tmp_dir)
            arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    except Exception as exc:  # noqa: BLE001 - corrupted local cache should not block augmentation.
        os.chdir(cwd)
        print(
            "warning: cannot initialize G1_29_ArmIK; inserted frames will copy anchor joints. "
            f"reason={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        for row in rows:
            row["aug.ik_status"] = f"ik_init_failed:{type(exc).__name__}"
        return
    finally:
        os.chdir(cwd)
    arm_ik.enable_joint_smoothing = False
    current_q: np.ndarray | None = None
    current_dq = np.zeros(14, dtype=np.float64)

    for row in rows:
        if current_q is None:
            current_q = _list_array(row["command.ik_joint_pos"], 14, "command.ik_joint_pos")
        left_tf = _pose_to_tf(np.asarray(row["command.eepose"], dtype=np.float64), "left")
        right_tf = _pose_to_tf(np.asarray(row["command.eepose"], dtype=np.float64), "right")
        try:
            sol_q, _ = arm_ik.solve_ik(left_tf, right_tf, current_q, current_dq)
            sol_q = np.asarray(sol_q, dtype=np.float64).reshape(-1)
            if sol_q.size != 14 or not np.all(np.isfinite(sol_q)):
                raise ValueError("IK returned invalid joint vector")
            row["command.ik_joint_pos"] = sol_q.tolist()
            row["aug.ik_status"] = "ok"
            current_q = sol_q
        except Exception as exc:  # noqa: BLE001 - keep batch augmentation moving.
            row["aug.ik_status"] = f"failed:{type(exc).__name__}"


def _build_variant_rows(
    source_path: str,
    command_data: CommandData,
    gripper: GripperSeries,
    variant_id: int,
    rng: np.random.Generator,
    protected_mask: np.ndarray,
    perturb_duration_pct_min: float,
    perturb_duration_pct_max: float,
    use_ik: bool,
    trans_min_m: float,
    trans_max_m: float,
    rot_min_deg: float,
    rot_max_deg: float,
) -> tuple[list[dict], PerturbationSpec]:
    spec = _sample_perturbation(
        command_data,
        gripper,
        rng,
        protected_mask=protected_mask,
        duration_pct_min=perturb_duration_pct_min,
        duration_pct_max=perturb_duration_pct_max,
        trans_min_m=trans_min_m,
        trans_max_m=trans_max_m,
        rot_min_deg=rot_min_deg,
        rot_max_deg=rot_max_deg,
    )
    hz = _estimate_hz(command_data.command_t_ns)
    dt_ns = int(round(1e9 / hz))
    insert_count = int(spec.insert_count)
    rows: list[dict] = []
    inserted_rows: list[dict] = []
    time_offset = 0.0

    for i, (_, src_row) in enumerate(command_data.df.iterrows()):
        row = src_row.to_dict()
        row["entry_type"] = "command"
        row["command.t_ns"] = float(command_data.command_t_ns[i] + time_offset)
        _fill_aug_fields(row, source_path, variant_id, gripper, i, inserted=False, spec=None)
        rows.append(row)

        if i != spec.anchor_index:
            continue

        anchor_pose = command_data.eepose[i]
        anchor_left_tf = _pose_to_tf(anchor_pose, "left")
        anchor_right_tf = _pose_to_tf(anchor_pose, "right")
        rot_delta = Rotation.from_euler("xyz", spec.rpy_deg, degrees=True).as_matrix()
        for k in range(1, insert_count + 1):
            amp = math.sin(math.pi * k / (insert_count + 1))
            left_tf = anchor_left_tf.copy()
            right_tf = anchor_right_tf.copy()
            target_tf = left_tf if spec.arm == "left" else right_tf
            target_tf[:3, 3] = target_tf[:3, 3] + amp * spec.xyz
            target_tf[:3, :3] = Rotation.from_rotvec(amp * Rotation.from_matrix(rot_delta).as_rotvec()).as_matrix() @ target_tf[:3, :3]

            new_row = src_row.to_dict()
            new_row["entry_type"] = "command"
            new_row["command.t_ns"] = float(command_data.command_t_ns[i] + time_offset + k * dt_ns)
            new_row["command.eepose"] = _tf_to_pose_vec(left_tf, right_tf)
            _fill_aug_fields(new_row, source_path, variant_id, gripper, i, inserted=True, spec=spec)
            rows.append(new_row)
            inserted_rows.append(new_row)
        time_offset += insert_count * dt_ns

    _solve_inserted_ik(inserted_rows, use_ik=use_ik)
    return rows, spec


def _segment_boundary_pose_indices(
    command_count: int,
    spec: SegmentBridgeSpec,
    start: int,
    end: int,
    segment_id: int,
) -> tuple[int, int]:
    first, second = spec.protected_ranges
    if segment_id == 0:
        return 0, first[0]
    if segment_id == 1:
        return first[1], second[0]
    if segment_id == 2:
        return second[1], command_count - 1
    return int(start), int(end)


def _build_segment_bridge_rows(
    source_path: str,
    command_data: CommandData,
    gripper: GripperSeries,
    variant_id: int,
    rng: np.random.Generator,
    spec: SegmentBridgeSpec,
    control_points: int,
    trans_noise_m: float,
    rot_noise_deg: float,
) -> list[dict]:
    command_count = int(command_data.command_t_ns.size)
    protected_mask = _protected_mask_from_ranges(command_count, spec.protected_ranges)
    generated_pose = np.asarray(command_data.eepose, dtype=np.float64).copy()
    segment_ids = np.full(command_count, -1, dtype=np.int64)

    for start, end, segment_id in spec.bridge_segments:
        start_pose_idx, end_pose_idx = _segment_boundary_pose_indices(command_count, spec, start, end, segment_id)
        generated_pose[start : end + 1] = _bridge_pose_segment(
            command_data=command_data,
            start=start,
            end=end,
            start_pose_idx=start_pose_idx,
            end_pose_idx=end_pose_idx,
            rng=rng,
            control_points=control_points,
            trans_noise_m=trans_noise_m,
            rot_noise_deg=rot_noise_deg,
        )
        segment_ids[start : end + 1] = int(segment_id)

    rows: list[dict] = []
    for i, (_, src_row) in enumerate(command_data.df.iterrows()):
        row = src_row.to_dict()
        row["entry_type"] = "command"
        row["command.t_ns"] = float(command_data.command_t_ns[i])
        row["command.eepose"] = generated_pose[i].astype(float).tolist()
        row["command.ik_joint_pos"] = command_data.joint_pos[i].astype(float).tolist()
        _fill_aug_fields(
            row,
            source_path=source_path,
            variant_id=variant_id,
            gripper=gripper,
            command_i=i,
            inserted=False,
            spec=None,
            mode="segment_bridge",
            segment_id=int(segment_ids[i]),
            protected=bool(protected_mask[i]),
        )
        row["aug.ik_status"] = "copied"
        rows.append(row)
    return rows


def _validate_segment_bridge_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No command rows generated")
    t_ns = np.asarray([float(row["command.t_ns"]) for row in rows], dtype=np.float64)
    if np.any(np.diff(t_ns) <= 0):
        raise ValueError("Generated command.t_ns is not strictly increasing")
    for idx, row in enumerate(rows):
        pose = _list_array(row["command.eepose"], 14, f"row[{idx}].command.eepose")
        for arm_start in (0, 7):
            quat = pose[arm_start + 3 : arm_start + 7]
            norm = float(np.linalg.norm(quat))
            if not np.isfinite(norm) or abs(norm - 1.0) > 1e-5:
                raise ValueError(f"row[{idx}] quaternion norm is {norm}, expected 1")


def _fill_aug_fields(
    row: dict,
    source_path: str,
    variant_id: int,
    gripper: GripperSeries,
    command_i: int,
    inserted: bool,
    spec: PerturbationSpec | None,
    mode: str = "pulse",
    segment_id: int = -1,
    protected: bool = False,
) -> None:
    i = int(command_i)
    row["aug.left_gripper_closed_raw"] = float(gripper.left_raw[i])
    row["aug.right_gripper_closed_raw"] = float(gripper.right_raw[i])
    row["aug.left_gripper_closed_smooth"] = float(gripper.left_smooth[i])
    row["aug.right_gripper_closed_smooth"] = float(gripper.right_smooth[i])
    row["aug.event_type"] = EVENT_NONE if inserted else gripper.event_type[i]
    row["aug.event_arm"] = "" if inserted else gripper.event_arm[i]
    row["aug.variant_id"] = int(variant_id)
    row["aug.source_path"] = source_path
    row["aug.mode"] = str(mode)
    row["aug.segment_id"] = int(segment_id)
    row["aug.is_protected_frame"] = bool(protected)
    row["aug.is_inserted_frame"] = bool(inserted)
    row["aug.anchor_command_index"] = int(i if inserted else -1)
    row["aug.perturbation_xyz"] = None if spec is None else np.asarray(spec.xyz, dtype=np.float64).tolist()
    row["aug.perturbation_rpy_deg"] = None if spec is None else np.asarray(spec.rpy_deg, dtype=np.float64).tolist()
    row["aug.perturb_duration_frac"] = np.nan if spec is None else float(spec.duration_frac)
    row["aug.perturb_insert_frames"] = np.nan if spec is None else int(spec.insert_count)
    row["aug.perturb_duration_sec"] = np.nan if spec is None else float(spec.wall_duration_sec)
    row.setdefault("aug.ik_status", "original")


def _add_aug_columns_to_raw(raw_rows: pd.DataFrame, source_path: str, variant_id: int) -> pd.DataFrame:
    out = raw_rows.copy()
    out["aug.left_gripper_closed_raw"] = np.nan
    out["aug.right_gripper_closed_raw"] = np.nan
    out["aug.left_gripper_closed_smooth"] = np.nan
    out["aug.right_gripper_closed_smooth"] = np.nan
    out["aug.event_type"] = EVENT_NONE
    out["aug.event_arm"] = ""
    out["aug.variant_id"] = int(variant_id)
    out["aug.source_path"] = source_path
    out["aug.mode"] = ""
    out["aug.segment_id"] = -1
    out["aug.is_protected_frame"] = False
    out["aug.is_inserted_frame"] = False
    out["aug.anchor_command_index"] = -1
    out["aug.perturbation_xyz"] = None
    out["aug.perturbation_rpy_deg"] = None
    out["aug.perturb_duration_frac"] = np.nan
    out["aug.perturb_insert_frames"] = np.nan
    out["aug.perturb_duration_sec"] = np.nan
    out["aug.ik_status"] = ""
    return out


def _effective_t_ns(row: pd.Series) -> float:
    if row.get("entry_type") == "raw":
        return float(row.get("raw.t_ns", 0.0))
    return float(row.get("command.t_ns", 0.0))


def _write_variant(
    original_df: pd.DataFrame,
    command_rows: list[dict],
    source_path: str,
    variant_id: int,
    output_dir: Path,
    input_path: Path,
) -> Path:
    raw_rows = original_df.loc[original_df["entry_type"].astype(str) == "raw"]
    raw_aug = _add_aug_columns_to_raw(raw_rows, source_path=source_path, variant_id=variant_id)
    out_df = pd.concat([raw_aug, pd.DataFrame(command_rows)], ignore_index=True, sort=False)
    out_df["_sort_t_ns"] = out_df.apply(_effective_t_ns, axis=1)
    out_df["_sort_kind"] = np.where(out_df["entry_type"].astype(str) == "raw", 0, 1)
    out_df = out_df.sort_values(["_sort_t_ns", "_sort_kind"], kind="mergesort").drop(columns=["_sort_t_ns", "_sort_kind"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}_aug_{variant_id:03d}.parquet"
    out_df.to_parquet(out_path, index=False)
    return out_path


def _print_summary(command_data: CommandData, gripper: GripperSeries) -> None:
    hz = _estimate_hz(command_data.command_t_ns)
    counts: dict[str, int] = {}
    for event in gripper.event_type:
        counts[event] = counts.get(event, 0) + 1
    print(f"command_frames={command_data.command_t_ns.size} hz_median={hz:.3f}")
    print(
        "left_gripper "
        f"raw_closed_frames={int(np.sum(gripper.left_raw > 0.5))} "
        f"smooth_range=[{float(np.min(gripper.left_smooth)):.3f},{float(np.max(gripper.left_smooth)):.3f}]"
    )
    print(
        "right_gripper "
        f"raw_closed_frames={int(np.sum(gripper.right_raw > 0.5))} "
        f"smooth_range=[{float(np.min(gripper.right_smooth)):.3f},{float(np.max(gripper.right_smooth)):.3f}]"
    )
    print("events=" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()) if k != EVENT_NONE))
    print(f"candidate_anchor_indices={gripper.event_indices[:20]}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment SpaceMouse command parquet episodes.")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output-dir", default="record_augmented", help="Output directory")
    parser.add_argument("--num-variants", type=int, default=1, help="Number of augmented parquet files")
    parser.add_argument(
        "--mode",
        choices=("segment_bridge", "pulse"),
        default="segment_bridge",
        help="segment_bridge preserves gripper-event windows and bridges EE pose between them; pulse keeps the old local perturbation mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA alpha for gripper closed state")
    parser.add_argument("--closed-high", type=float, default=0.65, help="Closed threshold after smoothing")
    parser.add_argument("--open-low", type=float, default=0.35, help="Open threshold after smoothing")
    parser.add_argument("--hold-closed-sec", type=float, default=0.8, help="Closed duration threshold for hold events")
    parser.add_argument(
        "--perturb-duration-pct-min",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Minimum inserted perturbation length as %% of total command frames (each variant samples uniformly in [min,max] %% )",
    )
    parser.add_argument(
        "--perturb-duration-pct-max",
        type=float,
        default=25.0,
        metavar="PCT",
        help="Maximum inserted perturbation length as %% of total command frames",
    )
    parser.add_argument(
        "--protect-window-frames",
        type=int,
        default=50,
        help="Protect +/-N frames around first two left gripper toggle events",
    )
    parser.add_argument(
        "--bridge-trans-noise-m",
        type=float,
        default=0.06,
        metavar="M",
        help="segment_bridge maximum random translation control-point offset in meters",
    )
    parser.add_argument(
        "--bridge-rot-noise-deg",
        type=float,
        default=10.0,
        metavar="DEG",
        help="segment_bridge maximum smooth orientation perturbation in degrees",
    )
    parser.add_argument(
        "--bridge-control-points",
        type=int,
        default=3,
        help="segment_bridge number of interior Bezier control points per free segment",
    )
    parser.add_argument(
        "--perturb-trans-min-m",
        type=float,
        default=0.02,
        metavar="M",
        help="Minimum EE translation pulse magnitude in meters (uniform along random direction)",
    )
    parser.add_argument(
        "--perturb-trans-max-m",
        type=float,
        default=0.05,
        metavar="M",
        help="Maximum EE translation pulse magnitude in meters",
    )
    parser.add_argument(
        "--perturb-rot-min-deg",
        type=float,
        default=0.0,
        metavar="DEG",
        help="Minimum rotation pulse magnitude in degrees (axis-angle style, scaled random axis)",
    )
    parser.add_argument(
        "--perturb-rot-max-deg",
        type=float,
        default=8.0,
        metavar="DEG",
        help="Maximum rotation pulse magnitude in degrees",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze events without writing output")
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        default=True,
        help="Copy existing joint angles instead of solving IK offline (default).",
    )
    parser.add_argument(
        "--solve-ik",
        action="store_false",
        dest="skip_ik",
        help="Old pulse mode only: solve IK for inserted frames.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    df = pd.read_parquet(input_path)
    command_data = _load_command_data(df)
    gripper = _reconstruct_gripper(
        df=df,
        command_t_ns=command_data.command_t_ns,
        smooth_alpha=float(args.smooth_alpha),
        closed_high=float(args.closed_high),
        open_low=float(args.open_low),
        hold_closed_sec=float(args.hold_closed_sec),
    )
    _print_summary(command_data, gripper)

    bridge_spec: SegmentBridgeSpec | None = None
    protected_mask: np.ndarray | None = None
    if args.mode == "segment_bridge":
        bridge_spec = _require_segment_bridge_spec(
            command_count=command_data.command_t_ns.size,
            gripper=gripper,
            protect_window_frames=int(args.protect_window_frames),
        )
        protected_mask = _protected_mask_from_ranges(command_data.command_t_ns.size, bridge_spec.protected_ranges)
        desc = ", ".join([f"evt@{evt}=>[{start},{end}]" for start, end, evt in bridge_spec.protected_ranges])
        segments = ", ".join([f"seg{sid}=[{start},{end}]" for start, end, sid in bridge_spec.bridge_segments])
        print(f"segment_bridge_left_toggle_indices={list(bridge_spec.event_indices)}")
        print(f"protected_segments(left toggles): {desc}")
        print(f"bridge_segments: {segments if segments else 'none'}")
    else:
        protected_mask, protected_ranges = _build_protected_mask(
            command_count=command_data.command_t_ns.size,
            gripper=gripper,
            protect_window_frames=int(args.protect_window_frames),
        )
        if protected_ranges:
            desc = ", ".join([f"evt@{evt}=>[{start},{end}]" for start, end, evt in protected_ranges])
            print(f"protected_segments(left toggles): {desc}")
        else:
            print("protected_segments(left toggles): none (left toggle events < 1)")
    if args.dry_run:
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    rng = np.random.default_rng(int(args.seed))
    pct_min = float(args.perturb_duration_pct_min)
    pct_max = float(args.perturb_duration_pct_max)
    for variant_id in range(int(args.num_variants)):
        if args.mode == "segment_bridge":
            assert bridge_spec is not None
            rows = _build_segment_bridge_rows(
                source_path=str(input_path),
                command_data=command_data,
                gripper=gripper,
                variant_id=variant_id,
                rng=rng,
                spec=bridge_spec,
                control_points=int(args.bridge_control_points),
                trans_noise_m=float(args.bridge_trans_noise_m),
                rot_noise_deg=float(args.bridge_rot_noise_deg),
            )
            _validate_segment_bridge_rows(rows)
            spec = None
        else:
            assert protected_mask is not None
            rows, spec = _build_variant_rows(
                source_path=str(input_path),
                command_data=command_data,
                gripper=gripper,
                variant_id=variant_id,
                rng=rng,
                protected_mask=protected_mask,
                perturb_duration_pct_min=pct_min,
                perturb_duration_pct_max=pct_max,
                use_ik=not bool(args.skip_ik),
                trans_min_m=float(args.perturb_trans_min_m),
                trans_max_m=float(args.perturb_trans_max_m),
                rot_min_deg=float(args.perturb_rot_min_deg),
                rot_max_deg=float(args.perturb_rot_max_deg),
            )
        out_path = _write_variant(
            original_df=df,
            command_rows=rows,
            source_path=str(input_path),
            variant_id=variant_id,
            output_dir=output_dir,
            input_path=input_path,
        )
        if args.mode == "segment_bridge":
            print(
                f"wrote {out_path} mode=segment_bridge variants={variant_id} "
                f"protected={list(bridge_spec.protected_ranges)} "
                f"bridge_noise_m={float(args.bridge_trans_noise_m):.4f} "
                f"bridge_noise_deg={float(args.bridge_rot_noise_deg):.3f}"
            )
        else:
            assert spec is not None
            print(
                f"wrote {out_path} anchor={spec.anchor_index} arm={spec.arm} "
                f"pct={100.0 * spec.duration_frac:.2f}% insert_frames={spec.insert_count} "
                f"approx_wall={spec.wall_duration_sec:.3f}s "
                f"xyz={np.array2string(spec.xyz, precision=4)} rpy_deg={np.array2string(spec.rpy_deg, precision=3)}"
            )


if __name__ == "__main__":
    main()
