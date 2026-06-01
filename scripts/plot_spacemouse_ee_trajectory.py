#!/usr/bin/env python3
"""Visualize SpaceMouse EE command trajectories recorded by teleop_spacemouse_ee_and_arm.py."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    print("Missing dependency: pandas. Activate the project environment or install pandas + pyarrow.", file=sys.stderr)
    sys.exit(1)


DEFAULT_RECORD_DIR = "/home/gsy/work/fastsim/record"
DEFAULT_BATCH_GLOB = "*.parquet"


@dataclass
class EeTrajectory:
    parquet_path: str
    t_s: np.ndarray
    left_xyz: np.ndarray
    left_xyzw: np.ndarray
    right_xyz: np.ndarray
    right_xyzw: np.ndarray
    valid_mask: np.ndarray
    sync_dt_ms: np.ndarray
    ik_joint_pos: np.ndarray | None
    generalized_mask: np.ndarray | None
    protected_mask: np.ndarray | None
    inserted_mask: np.ndarray | None


def _latest_parquet(record_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(record_dir, "*.parquet")), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {record_dir}")
    return paths[-1]


def _as_float_array(value, expected_len: int, row_idx: int, column: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected_len:
        raise ValueError(f"Row {row_idx} column `{column}` has length {arr.size}, expected {expected_len}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Row {row_idx} column `{column}` contains non-finite values")
    return arr


def _quat_xyzw_to_rot(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, dtype=np.float64).reshape(4)
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _set_equal_3d(ax, points: list[np.ndarray]) -> None:
    valid = [p.reshape(-1, 3) for p in points if p.size]
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


def load_ee_trajectory(parquet_path: str, valid_only: bool) -> EeTrajectory:
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    required = {"entry_type", "command.t_ns", "command.eepose"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {parquet_path}: {missing}")

    cmd = df[df["entry_type"] == "command"].copy()
    cmd = cmd[cmd["command.eepose"].notna()]
    if "command.valid_mask" in cmd.columns:
        cmd["command.valid_mask"] = cmd["command.valid_mask"].where(cmd["command.valid_mask"].notna(), False).astype(bool)
        if valid_only:
            cmd = cmd[cmd["command.valid_mask"]]
    else:
        cmd["command.valid_mask"] = True
    cmd = cmd.sort_values("command.t_ns")
    if cmd.empty:
        raise ValueError(f"No command rows with command.eepose found in {parquet_path}")

    eepose = np.vstack(
        [
            _as_float_array(value, expected_len=14, row_idx=int(idx), column="command.eepose")
            for idx, value in cmd["command.eepose"].items()
        ]
    )
    t_ns = cmd["command.t_ns"].astype("int64").to_numpy()
    t_s = (t_ns - t_ns[0]).astype(np.float64) * 1e-9

    sync_dt_ms = np.full(len(cmd), np.nan, dtype=np.float64)
    if "command.sync_dt_ns" in cmd.columns:
        sync_dt_ms = pd.to_numeric(cmd["command.sync_dt_ns"], errors="coerce").to_numpy(dtype=np.float64) * 1e-6

    ik_joint_pos = None
    if "command.ik_joint_pos" in cmd.columns and cmd["command.ik_joint_pos"].notna().any():
        ik_rows = []
        for value in cmd["command.ik_joint_pos"]:
            if value is None:
                ik_rows.append(None)
                continue
            ik_rows.append(np.asarray(value, dtype=np.float64).reshape(-1))
        lengths = {row.size for row in ik_rows if row is not None}
        if len(lengths) == 1:
            width = lengths.pop()
            ik_joint_pos = np.full((len(ik_rows), width), np.nan, dtype=np.float64)
            for i, row in enumerate(ik_rows):
                if row is not None:
                    ik_joint_pos[i] = row

    generalized_mask = None
    if "gen.generalized" in cmd.columns:
        generalized_mask = cmd["gen.generalized"].where(cmd["gen.generalized"].notna(), False).to_numpy(dtype=bool)

    protected_mask = None
    if "gen.protected" in cmd.columns:
        protected_mask = cmd["gen.protected"].where(cmd["gen.protected"].notna(), False).to_numpy(dtype=bool)

    inserted_mask = None
    if "aug.is_inserted_frame" in cmd.columns:
        inserted_mask = cmd["aug.is_inserted_frame"].where(cmd["aug.is_inserted_frame"].notna(), False).astype(bool).to_numpy()

    return EeTrajectory(
        parquet_path=parquet_path,
        t_s=t_s,
        left_xyz=eepose[:, 0:3],
        left_xyzw=eepose[:, 3:7],
        right_xyz=eepose[:, 7:10],
        right_xyzw=eepose[:, 10:14],
        valid_mask=cmd["command.valid_mask"].to_numpy(dtype=bool),
        sync_dt_ms=sync_dt_ms,
        ik_joint_pos=ik_joint_pos,
        generalized_mask=generalized_mask,
        protected_mask=protected_mask,
        inserted_mask=inserted_mask,
    )


def _plot_path(ax, xyz: np.ndarray, color: str, label: str) -> None:
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.8, label=label)
    ax.scatter(*xyz[0], color=color, s=50, marker="o", edgecolors="black", linewidths=0.6, label=f"{label} start")
    ax.scatter(*xyz[-1], color=color, s=65, marker="X", edgecolors="black", linewidths=0.6, label=f"{label} end")


def _contiguous_bool_runs(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    flags = np.asarray(flags, dtype=bool).reshape(-1)
    if flags.size == 0:
        return []
    runs: list[tuple[int, int, bool]] = []
    start = 0
    cur = bool(flags[0])
    for i in range(1, flags.size):
        if bool(flags[i]) != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = bool(flags[i])
    runs.append((start, flags.size - 1, cur))
    return runs


def _shade_inserted_time_regions(ax, t_s: np.ndarray, inserted_mask: np.ndarray | None, *, color: str = "tab:orange", alpha: float = 0.11) -> None:
    if inserted_mask is None or len(inserted_mask) == 0 or not np.any(inserted_mask):
        return
    t_s = np.asarray(t_s, dtype=np.float64).reshape(-1)
    ins = np.asarray(inserted_mask, dtype=bool).reshape(-1)
    if t_s.shape[0] != ins.shape[0]:
        return
    for s, e, is_ins in _contiguous_bool_runs(ins):
        if not is_ins:
            continue
        ax.axvspan(float(t_s[s]), float(t_s[e]), color=color, alpha=alpha, lw=0)


def _plot_t_series_highlight_insert(
    ax,
    t_s: np.ndarray,
    y: np.ndarray,
    inserted_mask: np.ndarray | None,
    color,
) -> None:
    t_s = np.asarray(t_s, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if t_s.shape[0] != y.shape[0]:
        return
    if inserted_mask is None or not np.any(inserted_mask):
        ax.plot(t_s, y, color=color, lw=1.1)
        return
    ins = np.asarray(inserted_mask, dtype=bool).reshape(-1)
    if ins.shape[0] != y.shape[0]:
        ax.plot(t_s, y, color=color, lw=1.1)
        return
    for s, e, is_ins in _contiguous_bool_runs(ins):
        tt = t_s[s : e + 1]
        yy = y[s : e + 1]
        ok = np.isfinite(tt) & np.isfinite(yy)
        tt, yy = tt[ok], yy[ok]
        if tt.size == 0:
            continue
        ls = "--" if is_ins else "-"
        ax.plot(tt, yy, color=color, lw=1.05, ls=ls, alpha=0.88)


def _plot_path_highlight_insert(ax, xyz: np.ndarray, inserted_mask: np.ndarray | None, color: str, label: str) -> None:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if inserted_mask is None or not np.any(inserted_mask):
        _plot_path(ax, xyz, color, label)
        return
    ins = np.asarray(inserted_mask, dtype=bool).reshape(-1)
    if ins.shape[0] != xyz.shape[0]:
        _plot_path(ax, xyz, color, label)
        return
    label_used = False
    for s, e, is_ins in _contiguous_bool_runs(ins):
        seg = xyz[s : e + 1]
        ok = np.all(np.isfinite(seg), axis=1)
        seg = seg[ok]
        if seg.shape[0] == 0:
            continue
        lab = label if not label_used else None
        if seg.shape[0] == 1:
            ax.scatter(*seg[0], color=color, s=44, marker="o", edgecolors="black", linewidths=0.45, label=lab)
            if lab is not None:
                label_used = True
            continue
        ls = "--" if is_ins else "-"
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=1.65, ls=ls, alpha=0.88, label=lab)
        if lab is not None:
            label_used = True
    ax.scatter(*xyz[0], color=color, s=50, marker="o", edgecolors="black", linewidths=0.6, label=f"{label} start")
    ax.scatter(*xyz[-1], color=color, s=65, marker="X", edgecolors="black", linewidths=0.6, label=f"{label} end")


def _plot_orientation_axes(ax, xyz: np.ndarray, xyzw: np.ndarray, color_scale: tuple[str, str, str], stride: int) -> None:
    if stride <= 0:
        return
    axis_len = 0.035
    for i in range(0, len(xyz), stride):
        R = _quat_xyzw_to_rot(xyzw[i])
        origin = xyz[i]
        for axis_idx, color in enumerate(color_scale):
            vec = R[:, axis_idx] * axis_len
            ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], color=color, linewidth=0.8)


def plot_trajectory(traj: EeTrajectory, output: str | None, show: bool, orientation_stride: int) -> None:
    if not show:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0), height_ratios=(1.0, 0.8))
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_sync = fig.add_subplot(gs[1, 1], sharex=ax_pos)

    title = os.path.basename(traj.parquet_path)
    valid_count = int(np.sum(traj.valid_mask))
    fig.suptitle(f"SpaceMouse EE trajectory: {title}  |  frames={len(traj.t_s)} valid={valid_count}", fontsize=12)

    _plot_path_highlight_insert(ax3d, traj.left_xyz, traj.inserted_mask, "C0", "left EE")
    _plot_path_highlight_insert(ax3d, traj.right_xyz, traj.inserted_mask, "C1", "right EE")
    _plot_orientation_axes(ax3d, traj.left_xyz, traj.left_xyzw, ("#d62728", "#2ca02c", "#1f77b4"), orientation_stride)
    _plot_orientation_axes(ax3d, traj.right_xyz, traj.right_xyzw, ("#d62728", "#2ca02c", "#1f77b4"), orientation_stride)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="upper right", fontsize=8)
    _set_equal_3d(ax3d, [traj.left_xyz, traj.right_xyz])

    _shade_inserted_time_regions(ax_pos, traj.t_s, traj.inserted_mask)
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        ax_pos.plot(traj.t_s, traj.left_xyz[:, axis_idx], color=f"C{axis_idx}", lw=1.2, label=f"left {axis_name}")
        ax_pos.plot(traj.t_s, traj.right_xyz[:, axis_idx], color=f"C{axis_idx}", lw=1.2, ls="--", label=f"right {axis_name}")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.grid(True, alpha=0.25)
    ax_pos.legend(ncol=3, fontsize=8)

    if np.any(np.isfinite(traj.sync_dt_ms)):
        ax_sync.plot(traj.t_s, traj.sync_dt_ms, color="0.25", lw=1.0, label="sync dt")
        invalid = ~traj.valid_mask
        if np.any(invalid):
            ax_sync.scatter(traj.t_s[invalid], traj.sync_dt_ms[invalid], s=8, color="tab:red", alpha=0.6, label="invalid")
        ax_sync.set_ylabel("Sync dt (ms)")
    else:
        joint_delta = np.linalg.norm(np.diff(traj.left_xyz, axis=0), axis=1)
        ax_sync.plot(traj.t_s[1:], joint_delta, color="0.25", lw=1.0, label="left step")
        ax_sync.set_ylabel("Step dist (m)")
    ax_sync.set_xlabel("Time (s)")
    ax_sync.grid(True, alpha=0.25)
    ax_sync.legend(fontsize=8)

    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=170, bbox_inches="tight")
        print(f"Saved plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_folder_eepose_xyz_grid(
    record_dir: str,
    glob_pattern: str,
    valid_only: bool,
    output: str | None,
    show: bool,
    seed: int,
) -> None:
    if not show:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern = os.path.join(record_dir, glob_pattern)
    parquet_paths = sorted(glob.glob(pattern))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for pattern: {pattern}")

    trajectories: list[EeTrajectory] = []
    for path in parquet_paths:
        traj = load_ee_trajectory(path, valid_only=valid_only)
        trajectories.append(traj)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    left_titles = ("Left X (m)", "Left Y (m)", "Left Z (m)")
    right_titles = ("Right X (m)", "Right Y (m)", "Right Z (m)")
    for i in range(3):
        axes[0, i].set_title(left_titles[i])
        axes[1, i].set_title(right_titles[i])

    rng = np.random.default_rng(seed)
    any_insert = any(traj.inserted_mask is not None and np.any(traj.inserted_mask) for traj in trajectories)
    for traj in trajectories:
        color = tuple(0.2 + 0.75 * rng.random(3))
        for axis_idx in range(3):
            _plot_t_series_highlight_insert(axes[0, axis_idx], traj.t_s, traj.left_xyz[:, axis_idx], traj.inserted_mask, color)
            _plot_t_series_highlight_insert(axes[1, axis_idx], traj.t_s, traj.right_xyz[:, axis_idx], traj.inserted_mask, color)

    for row in range(2):
        for col in range(3):
            axes[row, col].grid(True, alpha=0.25)
            axes[row, col].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Position (m)")
    axes[1, 0].set_ylabel("Position (m)")

    title = f"SpaceMouse EE XYZ trajectories ({len(trajectories)} files)"
    if any_insert:
        title += " | dashed = aug inserted, solid = original"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=170, bbox_inches="tight")
        print(f"Saved plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _shade_mask_regions(ax, t_s: np.ndarray, mask: np.ndarray | None, color: str, alpha: float) -> None:
    if mask is None or len(mask) == 0:
        return
    mask = np.asarray(mask, dtype=bool)
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            ax.axvspan(t_s[start], t_s[max(start, i - 1)], color=color, alpha=alpha, lw=0)
            start = None
    if start is not None:
        ax.axvspan(t_s[start], t_s[-1], color=color, alpha=alpha, lw=0)


def plot_folder_left_joint_grid(
    record_dir: str,
    glob_pattern: str,
    valid_only: bool,
    output: str | None,
    show: bool,
    seed: int,
) -> None:
    if not show:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern = os.path.join(record_dir, glob_pattern)
    parquet_paths = sorted(glob.glob(pattern))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for pattern: {pattern}")

    trajectories: list[EeTrajectory] = []
    for path in parquet_paths:
        traj = load_ee_trajectory(path, valid_only=valid_only)
        if traj.ik_joint_pos is None or traj.ik_joint_pos.shape[1] < 7:
            raise ValueError(f"{path} does not contain 7+ values in command.ik_joint_pos")
        trajectories.append(traj)

    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    axes_flat = axes.reshape(-1)
    rng = np.random.default_rng(seed)
    for traj in trajectories:
        color = tuple(0.15 + 0.8 * rng.random(3))
        for joint_i in range(7):
            axes_flat[joint_i].plot(traj.t_s, traj.ik_joint_pos[:, joint_i], color=color, lw=0.8, alpha=0.45)

    ref = trajectories[0]
    for joint_i in range(7):
        ax = axes_flat[joint_i]
        _shade_mask_regions(ax, ref.t_s, ref.generalized_mask, color="tab:green", alpha=0.08)
        _shade_mask_regions(ax, ref.t_s, ref.protected_mask, color="tab:red", alpha=0.06)
        ax.set_title(f"Left Joint {joint_i + 1}")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.25)
    axes_flat[7].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        f"IKFlow left-arm joint trajectories ({len(trajectories)} files) | green=generalized red=protected",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=170, bbox_inches="tight")
        print(f"Saved plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_folder_left_joint_spread(
    record_dir: str,
    glob_pattern: str,
    valid_only: bool,
    output: str | None,
    show: bool,
) -> None:
    if not show:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern = os.path.join(record_dir, glob_pattern)
    parquet_paths = sorted(glob.glob(pattern))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for pattern: {pattern}")

    trajectories: list[EeTrajectory] = []
    for path in parquet_paths:
        traj = load_ee_trajectory(path, valid_only=valid_only)
        if traj.ik_joint_pos is None or traj.ik_joint_pos.shape[1] < 7:
            raise ValueError(f"{path} does not contain 7+ values in command.ik_joint_pos")
        trajectories.append(traj)

    n_frames = min(len(traj.t_s) for traj in trajectories)
    joints = np.stack([traj.ik_joint_pos[:n_frames, 0:7] for traj in trajectories], axis=0)
    spread = np.std(joints, axis=0)
    t_s = trajectories[0].t_s[:n_frames]

    fig, ax = plt.subplots(figsize=(16, 5))
    image = ax.imshow(
        spread.T,
        aspect="auto",
        origin="lower",
        extent=[float(t_s[0]), float(t_s[-1]), 0.5, 7.5],
        cmap="magma",
    )
    _shade_mask_regions(ax, t_s, trajectories[0].generalized_mask[:n_frames] if trajectories[0].generalized_mask is not None else None, color="cyan", alpha=0.08)
    _shade_mask_regions(ax, t_s, trajectories[0].protected_mask[:n_frames] if trajectories[0].protected_mask is not None else None, color="white", alpha=0.08)
    ax.set_yticks(range(1, 8))
    ax.set_ylabel("Left joint")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"IKFlow left-arm joint standard deviation across {len(trajectories)} files")
    fig.colorbar(image, ax=ax, label="std (rad)")
    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=170, bbox_inches="tight")
        print(f"Saved plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(traj: EeTrajectory) -> None:
    duration = float(traj.t_s[-1] - traj.t_s[0]) if len(traj.t_s) > 1 else 0.0
    left_dist = float(np.sum(np.linalg.norm(np.diff(traj.left_xyz, axis=0), axis=1)))
    right_dist = float(np.sum(np.linalg.norm(np.diff(traj.right_xyz, axis=0), axis=1)))
    print(f"Loaded: {traj.parquet_path}")
    print(f"Command frames: {len(traj.t_s)}")
    print(f"Duration: {duration:.3f} s")
    print(f"Valid sync frames: {int(np.sum(traj.valid_mask))}/{len(traj.valid_mask)}")
    print(f"Left path length: {left_dist:.4f} m")
    print(f"Right path length: {right_dist:.4f} m")
    for name, xyz in (("Left", traj.left_xyz), ("Right", traj.right_xyz)):
        print(
            f"{name} bounds: "
            f"X[{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}] "
            f"Y[{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}] "
            f"Z[{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SpaceMouse EE command trajectory parquet files.")
    parser.add_argument("--parquet", default=None, help="Path to a parquet file. Defaults to latest in --record-dir.")
    parser.add_argument("--batch-dir", default=None, help="Directory of parquet files for 2x3 XYZ overlay plot.")
    parser.add_argument(
        "--batch-plot",
        choices=("eepose-xyz", "left-joints", "left-joint-spread"),
        default="eepose-xyz",
        help="Batch visualization type (default: eepose-xyz).",
    )
    parser.add_argument("--glob", dest="glob_pattern", default=DEFAULT_BATCH_GLOB, help=f"Glob pattern used with --batch-dir (default: {DEFAULT_BATCH_GLOB}).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for batch trajectory colors (default: 0).")
    parser.add_argument("--record-dir", default=DEFAULT_RECORD_DIR, help=f"Record directory (default: {DEFAULT_RECORD_DIR})")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults next to the parquet when --show is absent.")
    parser.add_argument("--show", action="store_true", help="Show an interactive Matplotlib window.")
    parser.add_argument("--valid-only", action="store_true", help="Plot only command rows with command.valid_mask == true.")
    parser.add_argument(
        "--orientation-stride",
        type=int,
        default=120,
        help="Draw EE orientation axes every N frames; <=0 disables axes (default: 120).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if args.batch_dir:
        batch_dir = os.path.abspath(os.path.expanduser(args.batch_dir))
        if output is None and not args.show:
            batch_name = os.path.basename(os.path.normpath(batch_dir)) or "batch"
            output = os.path.join(batch_dir, f"{batch_name}_eepose_xyz_grid.png")
        if output:
            output = os.path.abspath(os.path.expanduser(output))
        if args.batch_plot == "left-joints":
            plot_folder_left_joint_grid(
                record_dir=batch_dir,
                glob_pattern=args.glob_pattern,
                valid_only=args.valid_only,
                output=output,
                show=args.show,
                seed=args.seed,
            )
        elif args.batch_plot == "left-joint-spread":
            plot_folder_left_joint_spread(
                record_dir=batch_dir,
                glob_pattern=args.glob_pattern,
                valid_only=args.valid_only,
                output=output,
                show=args.show,
            )
        else:
            plot_folder_eepose_xyz_grid(
                record_dir=batch_dir,
                glob_pattern=args.glob_pattern,
                valid_only=args.valid_only,
                output=output,
                show=args.show,
                seed=args.seed,
            )
        return

    parquet_path = os.path.abspath(os.path.expanduser(args.parquet or _latest_parquet(args.record_dir)))
    traj = load_ee_trajectory(parquet_path, valid_only=args.valid_only)
    print_summary(traj)

    if output is None and not args.show:
        base = os.path.splitext(os.path.basename(parquet_path))[0]
        output = os.path.join(os.path.dirname(parquet_path), f"{base}_ee_trajectory.png")
    if output:
        output = os.path.abspath(os.path.expanduser(output))
    plot_trajectory(traj, output=output, show=args.show, orientation_stride=args.orientation_stride)


if __name__ == "__main__":
    main()
