#!/usr/bin/env python3
"""Create visual-only left-hand trajectories with clearly diverged paths.

This tool rewrites only ``command.eepose[0:3]`` (left XYZ) in derived parquet
files.  It preserves the reference start/end and protected left-gripper event
windows, then bends editable spans through random smooth offsets so trajectories
are visually separable.  The output is intended for visualization, not replay.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    print("Missing dependency: pandas. Activate the project environment or install pandas + pyarrow.", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.interpolate import PchipInterpolator
except ModuleNotFoundError:
    PchipInterpolator = None  # type: ignore[assignment]


BTN_1 = 4096
BTN_2 = 8192
BTN_3 = 16384
BTN_4 = 32768

EVENT_OPEN_TO_CLOSE = "open_to_close"
EVENT_CLOSE_TO_OPEN = "close_to_open"


@dataclass(frozen=True)
class ReferenceData:
    path: Path
    command_t_ns: np.ndarray
    left_xyz: np.ndarray
    protected_mask: np.ndarray
    protected_ranges: list[tuple[int, int, int]]


@dataclass(frozen=True)
class OutputTrajectory:
    path: Path
    left_xyz: np.ndarray
    max_mid_offset_m: float


def _as_pose_array(value: object, row_idx: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 14:
        raise ValueError(f"Row {row_idx} command.eepose has length {arr.size}, expected 14")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Row {row_idx} command.eepose contains non-finite values")
    return arr


def _load_command_pose(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"entry_type", "command.t_ns", "command.eepose"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    mask = (df["entry_type"].astype(str) == "command") & df["command.eepose"].notna()
    cmd = df.loc[mask].sort_values("command.t_ns", kind="mergesort")
    if cmd.empty:
        raise ValueError("No command rows with command.eepose found")
    poses = np.vstack([_as_pose_array(v, int(i)) for i, v in cmd["command.eepose"].items()])
    t_ns = pd.to_numeric(cmd["command.t_ns"], errors="raise").to_numpy(dtype=np.float64)
    return cmd.index.to_numpy(), t_ns, poses


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = float(values[0])
    a = float(np.clip(alpha, 0.0, 1.0))
    for i in range(1, values.size):
        out[i] = out[i - 1] + a * (float(values[i]) - out[i - 1])
    return out


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


def _left_toggle_events(
    df: pd.DataFrame,
    command_t_ns: np.ndarray,
    smooth_alpha: float,
    closed_high: float,
    open_low: float,
) -> list[int]:
    raw = df.loc[df["entry_type"].astype(str) == "raw"].copy()
    raw = raw[raw["raw.t_ns"].notna()].sort_values("raw.t_ns", kind="mergesort")

    left = np.zeros(command_t_ns.size, dtype=np.float64)
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

    closed = _hysteresis_closed(_ema(left, smooth_alpha), closed_high=closed_high, open_low=open_low)
    events: list[int] = []
    prev = False
    for i, cur in enumerate(closed):
        if bool(cur) != prev:
            events.append(i)
        prev = bool(cur)
    return events


def _build_reference(
    reference_parquet: Path,
    protect_window_frames: int,
    smooth_alpha: float,
    closed_high: float,
    open_low: float,
) -> ReferenceData:
    df = pd.read_parquet(reference_parquet)
    _, t_ns, poses = _load_command_pose(df)
    left_xyz = poses[:, 0:3].copy()
    protected = np.zeros(left_xyz.shape[0], dtype=bool)
    protected[0] = True
    protected[-1] = True

    events = _left_toggle_events(
        df=df,
        command_t_ns=t_ns,
        smooth_alpha=smooth_alpha,
        closed_high=closed_high,
        open_low=open_low,
    )
    ranges: list[tuple[int, int, int]] = []
    for event_idx in events[:2]:
        start = max(0, int(event_idx) - int(protect_window_frames))
        end = min(left_xyz.shape[0] - 1, int(event_idx) + int(protect_window_frames))
        protected[start : end + 1] = True
        ranges.append((start, end, int(event_idx)))

    return ReferenceData(
        path=reference_parquet,
        command_t_ns=t_ns,
        left_xyz=left_xyz,
        protected_mask=protected,
        protected_ranges=ranges,
    )


def _variant_id_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"(?:aug|variant|diverged|ikflow)_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return int(fallback)


def _noninserted_reference_ordinals(cmd: pd.DataFrame, reference_len: int) -> np.ndarray:
    inserted = np.zeros(len(cmd), dtype=bool)
    if "aug.is_inserted_frame" in cmd.columns:
        inserted = cmd["aug.is_inserted_frame"].where(cmd["aug.is_inserted_frame"].notna(), False).astype(bool).to_numpy()
    ordinals = np.full(len(cmd), -1, dtype=np.int64)
    noninserted_seen = 0
    for i, is_inserted in enumerate(inserted):
        if is_inserted:
            continue
        if noninserted_seen < reference_len:
            ordinals[i] = noninserted_seen
        noninserted_seen += 1
    return ordinals


def _orthonormal_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    direction = pts[-1] - pts[0] if pts.shape[0] >= 2 else np.array([1.0, 0.0, 0.0])
    if float(np.linalg.norm(direction)) < 1e-8:
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0] if vh.size else np.array([1.0, 0.0, 0.0])
    tangent = direction / max(float(np.linalg.norm(direction)), 1e-12)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, helper))) > 0.88:
        helper = np.array([0.0, 1.0, 0.0])
    n1 = np.cross(tangent, helper)
    n1 = n1 / max(float(np.linalg.norm(n1)), 1e-12)
    n2 = np.cross(tangent, n1)
    n2 = n2 / max(float(np.linalg.norm(n2)), 1e-12)
    return n1, n2


def _contiguous_false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if not value and start is None:
            start = i
        elif value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, mask.size - 1))
    return runs


def _interp_controls(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    if PchipInterpolator is None:
        return np.interp(x, xp, fp)
    return PchipInterpolator(xp, fp)(x)


def _generate_offsets(
    base_xyz: np.ndarray,
    protected_mask: np.ndarray,
    variant_id: int,
    seed: int,
    amplitude_min_m: float,
    amplitude_max_m: float,
    min_control_points: int,
    max_control_points: int,
) -> tuple[np.ndarray, float]:
    xyz = np.asarray(base_xyz, dtype=np.float64).reshape(-1, 3)
    offsets = np.zeros_like(xyz)
    n1, n2 = _orthonormal_basis(xyz)
    rng = np.random.default_rng(int(seed) + 1009 * int(variant_id))
    amp_lo, amp_hi = sorted((float(amplitude_min_m), float(amplitude_max_m)))
    amp_hi = max(amp_hi, amp_lo)
    max_amp = 0.0

    for span_idx, (start, end) in enumerate(_contiguous_false_runs(protected_mask)):
        count = end - start + 1
        if count < 3:
            continue
        u = np.linspace(0.0, 1.0, count)
        ctrl_count = int(rng.integers(int(min_control_points), int(max_control_points) + 1))
        interior_count = max(0, ctrl_count - 2)
        if interior_count:
            interior = np.sort(rng.uniform(0.08, 0.92, size=interior_count))
            xp = np.concatenate(([0.0], interior, [1.0]))
        else:
            xp = np.array([0.0, 1.0], dtype=np.float64)

        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        base_angle = variant_id * golden_angle + span_idx * 0.73
        ctrl_a = np.zeros(xp.size, dtype=np.float64)
        ctrl_b = np.zeros(xp.size, dtype=np.float64)
        for j in range(1, xp.size - 1):
            angle = base_angle + rng.normal(scale=0.75) + 0.9 * j
            amp = rng.uniform(amp_lo, amp_hi) * rng.choice([-1.0, 1.0])
            side = rng.uniform(0.55, 1.0)
            ctrl_a[j] = amp * math.cos(angle)
            ctrl_b[j] = amp * math.sin(angle) * side

        a = _interp_controls(u, xp, ctrl_a)
        b = _interp_controls(u, xp, ctrl_b)
        envelope = np.sin(math.pi * u)
        local = (a[:, None] * n1[None, :] + b[:, None] * n2[None, :]) * envelope[:, None]
        local[0] = 0.0
        local[-1] = 0.0
        offsets[start : end + 1] = local
        if local.size:
            max_amp = max(max_amp, float(np.max(np.linalg.norm(local, axis=1))))

    return offsets, max_amp


def _apply_visual_divergence(
    input_path: Path,
    output_dir: Path,
    reference: ReferenceData,
    variant_id: int,
    seed: int,
    amplitude_min_m: float,
    amplitude_max_m: float,
    min_control_points: int,
    max_control_points: int,
) -> OutputTrajectory:
    df = pd.read_parquet(input_path)
    cmd_idx, _, poses = _load_command_pose(df)
    cmd = df.loc[cmd_idx].copy()
    ordinals = _noninserted_reference_ordinals(cmd, reference.left_xyz.shape[0])

    protected = np.zeros(len(cmd_idx), dtype=bool)
    protected[0] = True
    protected[-1] = True
    valid_ord = (ordinals >= 0) & (ordinals < reference.protected_mask.shape[0])
    protected[valid_ord] = reference.protected_mask[ordinals[valid_ord]]

    base_xyz = poses[:, 0:3].copy()
    ref_for_rows = base_xyz.copy()
    ref_for_rows[valid_ord] = reference.left_xyz[ordinals[valid_ord]]
    ref_for_rows[0] = reference.left_xyz[0]
    ref_for_rows[-1] = reference.left_xyz[-1]

    offsets, max_mid_offset = _generate_offsets(
        base_xyz=ref_for_rows,
        protected_mask=protected,
        variant_id=variant_id,
        seed=seed,
        amplitude_min_m=amplitude_min_m,
        amplitude_max_m=amplitude_max_m,
        min_control_points=min_control_points,
        max_control_points=max_control_points,
    )
    new_xyz = ref_for_rows + offsets
    new_xyz[protected] = ref_for_rows[protected]
    new_xyz[0] = reference.left_xyz[0]
    new_xyz[-1] = reference.left_xyz[-1]

    new_poses = poses.copy()
    new_poses[:, 0:3] = new_xyz
    for row_index, pose in zip(cmd_idx, new_poses, strict=True):
        df.at[row_index, "command.eepose"] = pose.astype(float).tolist()

    df["viz.diverged"] = False
    df["viz.source_path"] = str(input_path)
    df["viz.variant_id"] = int(variant_id)
    df["viz.anchor_policy"] = "start_end_first_two_left_toggle_windows"
    df["viz.amplitude_m"] = np.nan
    df["viz.visual_only"] = True
    df.loc[cmd_idx, "viz.diverged"] = True
    df.loc[cmd_idx, "viz.amplitude_m"] = float(max_mid_offset)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}_visual_diverged.parquet"
    df.to_parquet(out_path, index=False)
    return OutputTrajectory(path=out_path, left_xyz=new_xyz, max_mid_offset_m=max_mid_offset)


def _set_equal_3d(ax, points: list[np.ndarray]) -> None:
    valid = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in points if np.asarray(p).size]
    if not valid:
        return
    all_points = np.vstack(valid)
    all_points = all_points[np.all(np.isfinite(all_points), axis=1)]
    if all_points.size == 0:
        return
    p_min = np.min(all_points, axis=0)
    p_max = np.max(all_points, axis=0)
    center = 0.5 * (p_min + p_max)
    span = float(np.max(p_max - p_min))
    if span < 1e-4:
        span = 1.0
    radius = 0.58 * span
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _write_plots(reference: ReferenceData, outputs: list[OutputTrajectory], output_dir: Path, seed: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    colors = [tuple(0.18 + 0.78 * rng.random(3)) for _ in outputs]
    all_xyz = [reference.left_xyz] + [o.left_xyz for o in outputs]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        reference.left_xyz[:, 0],
        reference.left_xyz[:, 1],
        reference.left_xyz[:, 2],
        color="black",
        lw=3.0,
        alpha=0.95,
        label="reference left",
    )
    for output, color in zip(outputs, colors, strict=True):
        xyz = output.left_xyz
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.25, alpha=0.42)
    ax.scatter(*reference.left_xyz[0], color="black", s=42, marker="o", edgecolors="white", linewidths=0.5)
    ax.scatter(*reference.left_xyz[-1], color="black", s=56, marker="X", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Visual-only diverged left trajectories ({len(outputs)} variants)")
    ax.legend(loc="upper left", fontsize=8)
    _set_equal_3d(ax, all_xyz)
    fig.tight_layout()
    path3d = output_dir / "left_visual_diverged_3d.png"
    fig.savefig(path3d, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig2, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    projections = ((0, 1, "X (m)", "Y (m)", "XY"), (0, 2, "X (m)", "Z (m)", "XZ"), (1, 2, "Y (m)", "Z (m)", "YZ"))
    for ax2, (a, b, xlabel, ylabel, title) in zip(axes, projections, strict=True):
        ax2.plot(reference.left_xyz[:, a], reference.left_xyz[:, b], color="black", lw=2.4, alpha=0.95)
        for output, color in zip(outputs, colors, strict=True):
            xyz = output.left_xyz
            ax2.plot(xyz[:, a], xyz[:, b], color=color, lw=1.0, alpha=0.36)
        ax2.scatter(reference.left_xyz[0, a], reference.left_xyz[0, b], color="black", s=30, marker="o")
        ax2.scatter(reference.left_xyz[-1, a], reference.left_xyz[-1, b], color="black", s=42, marker="X")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(title)
        ax2.grid(True, alpha=0.25)
        ax2.set_aspect("equal", adjustable="datalim")
    fig2.suptitle(f"Visual-only diverged left trajectories projections ({len(outputs)} variants)", fontsize=13)
    fig2.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    path_proj = output_dir / "left_visual_diverged_projections.png"
    fig2.savefig(path_proj, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved plot: {path3d}")
    print(f"Saved plot: {path_proj}")


def _validate_outputs(reference: ReferenceData, outputs: list[OutputTrajectory], input_count: int) -> None:
    if len(outputs) != input_count:
        raise RuntimeError(f"Output count mismatch: {len(outputs)} != {input_count}")
    for output in outputs:
        if float(np.linalg.norm(output.left_xyz[0] - reference.left_xyz[0])) > 1e-9:
            raise RuntimeError(f"{output.path} start point differs from reference")
        if float(np.linalg.norm(output.left_xyz[-1] - reference.left_xyz[-1])) > 1e-9:
            raise RuntimeError(f"{output.path} end point differs from reference")
    enough = sum(1 for o in outputs if o.max_mid_offset_m > 0.10)
    ratio = enough / max(len(outputs), 1)
    if ratio < 0.90:
        raise RuntimeError(f"Only {ratio:.1%} outputs have max mid offset > 0.10 m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual-only diverged left-hand trajectory parquet files.")
    parser.add_argument("--reference-parquet", required=True, help="Original reference parquet path.")
    parser.add_argument("--input-dir", required=True, help="Directory containing source parquet files.")
    parser.add_argument("--glob", dest="glob_pattern", default="*.parquet", help="Glob under --input-dir.")
    parser.add_argument("--output-dir", required=True, help="Directory for visual-only derived parquet files and plots.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic visual divergence.")
    parser.add_argument("--protect-window-frames", type=int, default=15, help="Protect +/-N frames around first two left toggle events.")
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA alpha for gripper state reconstruction.")
    parser.add_argument("--closed-high", type=float, default=0.65, help="Closed threshold after smoothing.")
    parser.add_argument("--open-low", type=float, default=0.35, help="Open threshold after smoothing.")
    parser.add_argument("--amplitude-min-m", type=float, default=0.12, help="Minimum random spline offset amplitude.")
    parser.add_argument("--amplitude-max-m", type=float, default=0.35, help="Maximum random spline offset amplitude.")
    parser.add_argument("--min-control-points", type=int, default=4, help="Minimum spline control point count per editable span.")
    parser.add_argument("--max-control-points", type=int, default=7, help="Maximum spline control point count per editable span.")
    parser.add_argument("--no-plots", action="store_true", help="Write parquet files only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference_parquet).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    input_paths = [Path(p) for p in sorted(glob.glob(str(input_dir / args.glob_pattern)))]
    if not input_paths:
        raise FileNotFoundError(f"No parquet files found for pattern: {input_dir / args.glob_pattern}")

    reference = _build_reference(
        reference_parquet=reference_path,
        protect_window_frames=int(args.protect_window_frames),
        smooth_alpha=float(args.smooth_alpha),
        closed_high=float(args.closed_high),
        open_low=float(args.open_low),
    )
    ranges = ", ".join(f"evt@{evt}=>[{start},{end}]" for start, end, evt in reference.protected_ranges)
    print(f"reference_frames={reference.left_xyz.shape[0]} protected_ranges={ranges or 'none'}")
    print(f"input_files={len(input_paths)} output_dir={output_dir}")

    outputs: list[OutputTrajectory] = []
    for fallback, input_path in enumerate(input_paths):
        variant_id = _variant_id_from_path(input_path, fallback)
        out = _apply_visual_divergence(
            input_path=input_path,
            output_dir=output_dir,
            reference=reference,
            variant_id=variant_id,
            seed=int(args.seed),
            amplitude_min_m=float(args.amplitude_min_m),
            amplitude_max_m=float(args.amplitude_max_m),
            min_control_points=max(2, int(args.min_control_points)),
            max_control_points=max(2, int(args.max_control_points)),
        )
        outputs.append(out)
        print(f"wrote {out.path} variant={variant_id} max_mid_offset_m={out.max_mid_offset_m:.4f}")

    _validate_outputs(reference, outputs, input_count=len(input_paths))
    if not args.no_plots:
        _write_plots(reference, outputs, output_dir=output_dir, seed=int(args.seed))
    print(f"done outputs={len(outputs)} min_offset={min(o.max_mid_offset_m for o in outputs):.4f} max_offset={max(o.max_mid_offset_m for o in outputs):.4f}")


if __name__ == "__main__":
    main()
