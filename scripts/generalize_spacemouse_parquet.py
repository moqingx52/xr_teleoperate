#!/usr/bin/env python3
"""Generalize SpaceMouse EE command trajectories outside gripper events.

Frame indices in this script refer to command rows only.  The script preserves
left-gripper open/close windows exactly, rebuilds left EE poses for all other
command frames, and optionally resolves the 14 arm joints through the existing
Pinocchio/CasADi A2D IK implementation.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
XR_ROOT = REPO_ROOT / "xr_teleoperate"
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))
IKFLOW_ROOT = REPO_ROOT / "ikflow"
if str(IKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(IKFLOW_ROOT))


BTN_1 = 4096
BTN_2 = 8192
BTN_3 = 16384
BTN_4 = 32768


@dataclass(frozen=True)
class CommandData:
    df: pd.DataFrame
    command_indices: np.ndarray
    t_ns: np.ndarray
    eepose: np.ndarray
    joint_pos: np.ndarray


@dataclass(frozen=True)
class ProtectionPlan:
    left_event_indices: list[int]
    protected_mask: np.ndarray
    protected_ranges: list[tuple[int, int]]
    generalized_ranges: list[tuple[int, int]]


def _list_array(value: object, expected: int, column: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected:
        raise ValueError(f"{column} has length {arr.size}, expected {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{column} contains non-finite values")
    return arr


def _load_command_data(df: pd.DataFrame) -> CommandData:
    required = {"entry_type", "command.t_ns", "command.eepose", "command.ik_joint_pos"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    cmd = df.loc[(df["entry_type"].astype(str) == "command") & df["command.t_ns"].notna()].copy()
    if cmd.empty:
        raise ValueError("no command rows found")
    cmd = cmd.sort_values("command.t_ns", kind="mergesort")
    command_indices = cmd.index.to_numpy()
    t_ns = cmd["command.t_ns"].to_numpy(dtype=np.float64)
    eepose = np.vstack([_list_array(v, 14, "command.eepose") for v in cmd["command.eepose"]])
    joint_pos = np.vstack([_list_array(v, 14, "command.ik_joint_pos") for v in cmd["command.ik_joint_pos"]])
    return CommandData(df=cmd, command_indices=command_indices, t_ns=t_ns, eepose=eepose, joint_pos=joint_pos)


def _estimate_hz(t_ns: np.ndarray) -> float:
    if t_ns.size < 2:
        return 30.0
    dt = np.diff(t_ns) / 1e9
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if dt.size == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def _nearest_command_index(command_t_ns: np.ndarray, raw_t_ns: float) -> int:
    pos = int(np.searchsorted(command_t_ns, raw_t_ns, side="left"))
    if pos <= 0:
        return 0
    if pos >= command_t_ns.size:
        return int(command_t_ns.size - 1)
    prev_i = pos - 1
    if abs(command_t_ns[prev_i] - raw_t_ns) <= abs(command_t_ns[pos] - raw_t_ns):
        return prev_i
    return pos


def _detect_left_gripper_events(df: pd.DataFrame, command_t_ns: np.ndarray) -> list[int]:
    raw = df.loc[df["entry_type"].astype(str) == "raw"].copy()
    raw = raw[raw["raw.t_ns"].notna()].sort_values("raw.t_ns", kind="mergesort")
    if raw.empty:
        return []

    selected = "left"
    left_state = 0
    events: list[int] = []
    for _, row in raw.iterrows():
        edges = int(row.get("raw.edges_mask", 0) or 0)
        if edges & BTN_1:
            selected = "left"
        if edges & BTN_2:
            selected = "right"
        if selected != "left":
            continue
        next_left_state = left_state
        if edges & BTN_3:
            next_left_state = 0
        if edges & BTN_4:
            next_left_state = 1
        if next_left_state != left_state:
            events.append(_nearest_command_index(command_t_ns, float(row["raw.t_ns"])))
            left_state = next_left_state

    deduped: list[int] = []
    for idx in events:
        if not deduped or idx != deduped[-1]:
            deduped.append(idx)
    return deduped


def _ranges_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, mask.size - 1))
    return ranges


def _build_plan(df: pd.DataFrame, command_t_ns: np.ndarray, window_frames: int) -> ProtectionPlan:
    n = command_t_ns.size
    left_events = _detect_left_gripper_events(df, command_t_ns)
    if len(left_events) < 2:
        raise ValueError(f"need at least 2 left gripper open/close events, found {len(left_events)}: {left_events}")

    selected_events = left_events[:2]
    protected = np.zeros(n, dtype=bool)
    for idx in selected_events:
        start = max(0, idx - window_frames)
        end = min(n - 1, idx + window_frames)
        protected[start : end + 1] = True

    generalized = ~protected
    return ProtectionPlan(
        left_event_indices=selected_events,
        protected_mask=protected,
        protected_ranges=_ranges_from_mask(protected),
        generalized_ranges=_ranges_from_mask(generalized),
    )


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, lift: float, s: np.ndarray) -> np.ndarray:
    mid = 0.5 * (p0 + p1)
    mid[2] += lift
    a = ((1.0 - s) ** 2)[:, None] * p0[None, :]
    b = (2.0 * (1.0 - s) * s)[:, None] * mid[None, :]
    c = (s**2)[:, None] * p1[None, :]
    return a + b + c


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, s: np.ndarray) -> np.ndarray:
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    return Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([q0, q1])))(s).as_quat()


def _generalize_left_eepose(
    eepose: np.ndarray,
    generalized_ranges: list[tuple[int, int]],
    lift_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = eepose.copy()
    generalized_mask = np.zeros(eepose.shape[0], dtype=bool)
    for start, end in generalized_ranges:
        if end <= start:
            continue
        count = end - start + 1
        u = np.linspace(0.0, 1.0, count)
        s = _smoothstep(u)
        p0 = eepose[start, 0:3]
        p1 = eepose[end, 0:3]
        q0 = eepose[start, 3:7]
        q1 = eepose[end, 3:7]
        out[start : end + 1, 0:3] = _quadratic_bezier(p0, p1, lift_height, s)
        out[start : end + 1, 3:7] = _slerp_xyzw(q0, q1, s)
        generalized_mask[start : end + 1] = True
    return out, generalized_mask


def _pose_to_tf(eepose_row: np.ndarray, arm: str) -> np.ndarray:
    start = 0 if arm == "left" else 7
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = Rotation.from_quat(eepose_row[start + 3 : start + 7]).as_matrix()
    tf[:3, 3] = eepose_row[start : start + 3]
    return tf


def _solve_pinocchio_ik(
    eepose: np.ndarray,
    joint_pos: np.ndarray,
    generalized_mask: np.ndarray,
    protected_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    try:
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot import G1_29_ArmIK: {type(exc).__name__}: {exc}") from exc

    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="fastsim_a2d_ik_cache_") as tmp_dir:
            os.chdir(tmp_dir)
            arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    finally:
        os.chdir(cwd)

    arm_ik.enable_joint_smoothing = False
    out = joint_pos.copy()
    status = ["protected" if protected_mask[i] else "original" for i in range(eepose.shape[0])]
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
    return out, status


def _xyzw_to_wxyz_pose(left_pose_xyzw: np.ndarray) -> list[float]:
    x, y, z, qx, qy, qz, qw = left_pose_xyzw.astype(float).tolist()
    return [x, y, z, qw, qx, qy, qz]


def _solve_ikflow_left_arm(
    eepose: np.ndarray,
    joint_pos: np.ndarray,
    generalized_mask: np.ndarray,
    protected_mask: np.ndarray,
    model_name: str,
    samples_per_pose: int,
    seed: int | None,
) -> tuple[np.ndarray, list[str]]:
    try:
        import torch
        from ikflow.config import DEVICE
        from ikflow.model_loading import get_ik_solver
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot import IKFlow: {type(exc).__name__}: {exc}") from exc

    ik_solver, _ = get_ik_solver(model_name)
    out = joint_pos.copy()
    status = ["protected" if protected_mask[i] else "original" for i in range(eepose.shape[0])]
    current_left = joint_pos[0, 0:7].copy()
    device = torch.device(DEVICE)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    generalized_indices = np.flatnonzero(generalized_mask & ~protected_mask)
    candidate_map: dict[int, np.ndarray] = {}
    if generalized_indices.size:
        targets = torch.tensor(
            [_xyzw_to_wxyz_pose(eepose[i, 0:7]) for i in generalized_indices],
            dtype=torch.float32,
            device=device,
        )
        repeated_targets = targets.repeat_interleave(int(samples_per_pose), dim=0)
        q_candidates = ik_solver.generate_ik_solutions(
            repeated_targets,
            n=None,
            clamp_to_joint_limits=True,
        )
        q_candidates_np = q_candidates.detach().cpu().numpy().reshape(
            generalized_indices.size,
            int(samples_per_pose),
            7,
        )
        if not np.all(np.isfinite(q_candidates_np)):
            raise ValueError("IKFlow returned non-finite values")
        for local_i, frame_i in enumerate(generalized_indices):
            candidate_map[int(frame_i)] = q_candidates_np[local_i]

    for i in range(eepose.shape[0]):
        if protected_mask[i]:
            current_left = joint_pos[i, 0:7].copy()
            continue
        if not generalized_mask[i]:
            continue

        try:
            q_candidates_np = candidate_map[int(i)]
            if q_candidates_np.ndim != 2 or q_candidates_np.shape[1] != 7:
                raise ValueError(f"IKFlow returned shape {q_candidates_np.shape}, expected [n, 7]")
            best = int(np.argmin(np.sum((q_candidates_np - current_left[None, :]) ** 2, axis=1)))
            out[i, 0:7] = q_candidates_np[best]
            current_left = out[i, 0:7].copy()
            status[i] = "ikflow_ok"
        except Exception as exc:  # noqa: BLE001
            out[i, 0:7] = current_left
            status[i] = f"ikflow_failed:{type(exc).__name__}"
    return out, status


def _format_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"{start}:{end}" for start, end in ranges)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_generalized{input_path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input SpaceMouse command parquet")
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path")
    parser.add_argument("--window-frames", type=int, default=100, help="Frames protected before/after each left event")
    parser.add_argument("--lift-height", type=float, default=0.08, help="World-Z lift for generalized Bezier segments, meters")
    parser.add_argument(
        "--ik-mode",
        choices=("copy", "pinocchio", "ikflow"),
        default="copy",
        help="copy leaves joints unchanged; pinocchio recomputes both arms; ikflow recomputes the left arm only",
    )
    parser.add_argument("--ikflow-model-name", default="a2d_left_arm__local", help="IKFlow model name for --ik-mode ikflow")
    parser.add_argument("--ikflow-samples-per-pose", type=int, default=64, help="IKFlow candidates per generalized pose")
    parser.add_argument("--ikflow-seed", type=int, default=None, help="Optional torch seed for reproducible IKFlow samples")
    parser.add_argument(
        "--export-ik-npz",
        type=Path,
        default=None,
        help="Write generalized EE/joint/mask arrays for solving IK in another environment",
    )
    parser.add_argument(
        "--import-ik-npz",
        type=Path,
        default=None,
        help="Read joint_pos/status arrays produced by solve_a2d_ik_npz.py",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (args.output or _default_output_path(input_path)).resolve()
    if args.window_frames < 0:
        raise ValueError("--window-frames must be non-negative")

    df = pd.read_parquet(input_path)
    command_data = _load_command_data(df)
    plan = _build_plan(df, command_data.t_ns, window_frames=int(args.window_frames))
    new_eepose, generalized_mask = _generalize_left_eepose(
        command_data.eepose,
        plan.generalized_ranges,
        lift_height=float(args.lift_height),
    )

    if args.export_ik_npz is not None:
        export_path = args.export_ik_npz.resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            export_path,
            eepose=new_eepose,
            joint_pos=command_data.joint_pos,
            generalized_mask=generalized_mask,
            protected_mask=plan.protected_mask,
        )

    if args.import_ik_npz is not None:
        imported = np.load(args.import_ik_npz.resolve(), allow_pickle=False)
        new_joint_pos = np.asarray(imported["joint_pos"], dtype=np.float64)
        if new_joint_pos.shape != command_data.joint_pos.shape:
            raise ValueError(f"imported joint_pos shape {new_joint_pos.shape}, expected {command_data.joint_pos.shape}")
        if "status" in imported.files:
            ik_status = [str(v) for v in imported["status"].astype(str).tolist()]
        else:
            ik_status = ["imported" if generalized_mask[i] else "protected" for i in range(command_data.t_ns.size)]
        if len(ik_status) != command_data.t_ns.size:
            raise ValueError(f"imported status length {len(ik_status)}, expected {command_data.t_ns.size}")
        effective_ik_mode = "imported"
    elif args.ik_mode == "pinocchio":
        new_joint_pos, ik_status = _solve_pinocchio_ik(
            new_eepose,
            command_data.joint_pos,
            generalized_mask=generalized_mask,
            protected_mask=plan.protected_mask,
        )
        effective_ik_mode = args.ik_mode
    elif args.ik_mode == "ikflow":
        if args.ikflow_samples_per_pose <= 0:
            raise ValueError("--ikflow-samples-per-pose must be positive")
        new_joint_pos, ik_status = _solve_ikflow_left_arm(
            new_eepose,
            command_data.joint_pos,
            generalized_mask=generalized_mask,
            protected_mask=plan.protected_mask,
            model_name=str(args.ikflow_model_name),
            samples_per_pose=int(args.ikflow_samples_per_pose),
            seed=args.ikflow_seed,
        )
        effective_ik_mode = args.ik_mode
    else:
        new_joint_pos = command_data.joint_pos.copy()
        ik_status = ["protected" if plan.protected_mask[i] else "copied" for i in range(command_data.t_ns.size)]
        effective_ik_mode = args.ik_mode

    out = df.copy()
    for local_i, row_idx in enumerate(command_data.command_indices):
        out.at[row_idx, "command.eepose"] = new_eepose[local_i].astype(float).tolist()
        out.at[row_idx, "command.ik_joint_pos"] = new_joint_pos[local_i].astype(float).tolist()
        out.at[row_idx, "gen.protected"] = bool(plan.protected_mask[local_i])
        out.at[row_idx, "gen.generalized"] = bool(generalized_mask[local_i])
        out.at[row_idx, "gen.ik_status"] = ik_status[local_i]

    out["gen.source_path"] = str(input_path)
    out["gen.left_event_command_indices"] = ",".join(str(i) for i in plan.left_event_indices)
    out["gen.protected_ranges"] = _format_ranges(plan.protected_ranges)
    out["gen.generalized_ranges"] = _format_ranges(plan.generalized_ranges)
    out["gen.window_frames"] = int(args.window_frames)
    out["gen.lift_height"] = float(args.lift_height)
    out["gen.ik_mode"] = effective_ik_mode

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    hz = _estimate_hz(command_data.t_ns)
    protected_count = int(plan.protected_mask.sum())
    generalized_count = int(generalized_mask.sum())
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"commands={command_data.t_ns.size} hz={hz:.3f}")
    print(f"left_events={plan.left_event_indices}")
    print(f"protected_ranges={_format_ranges(plan.protected_ranges)} protected_frames={protected_count}")
    print(f"generalized_ranges={_format_ranges(plan.generalized_ranges)} generalized_frames={generalized_count}")
    if args.export_ik_npz is not None:
        print(f"export_ik_npz={args.export_ik_npz.resolve()}")
    if args.import_ik_npz is not None:
        print(f"import_ik_npz={args.import_ik_npz.resolve()}")
    print(f"ik_mode={effective_ik_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
