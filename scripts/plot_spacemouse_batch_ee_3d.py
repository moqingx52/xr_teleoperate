#!/usr/bin/env python3
"""Overlay 3D end-effector trajectories from all parquet files in a directory.

Each file is drawn in a distinct color. Reuses the same parquet schema as
``plot_spacemouse_ee_trajectory.py`` (command.eepose, optional valid_mask).

Augmented rows (``aug.is_inserted_frame``) are shown as **dashed** polylines;
original command rows as **solid** — same linewidth and alpha (no thick/halo
“炸眼” styling). Paths are optionally **moving-average smoothed** for display
only so short pulse inserts read as gentle bends instead of sharp glyphs.

**Default is left arm only** (``--hands left``); use ``--hands right`` or
``--hands both`` when needed.

Example::

  python plot_spacemouse_batch_ee_3d.py \\
    --batch-dir /home/gsy/work/fastsim/record_augmented \\
    --valid-only --seed 42 \\
    --output /tmp/ee_batch_3d.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Literal

import numpy as np

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Missing dependency: matplotlib.", file=sys.stderr)
    sys.exit(1)

try:
    import plot_spacemouse_ee_trajectory as sm  # type: ignore
except ModuleNotFoundError:
    print("Could not import plot_spacemouse_ee_trajectory (expected next to this script).", file=sys.stderr)
    sys.exit(1)


HandsMode = Literal["left", "right", "both"]


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return xyz
    ok = np.all(np.isfinite(xyz), axis=1)
    return xyz[ok]


def _smooth_xyz_ma(xyz: np.ndarray, window: int) -> np.ndarray:
    """Symmetric moving average per axis (display only). ``window`` <= 1 disables."""
    xyz = np.array(xyz, dtype=np.float64, copy=True)
    n, d = xyz.shape[0], xyz.shape[1]
    if n < 2 or window <= 1 or d != 3:
        return xyz
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, n if (n % 2 == 1) else n - 1)
    if w <= 1:
        return xyz
    pad = w // 2
    out = np.zeros_like(xyz)
    for a in range(3):
        col = xyz[:, a]
        padded = np.pad(col, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float64) / w
        out[:, a] = np.convolve(padded, kernel, mode="valid")
    return out


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


def _set_equal_3d(
    ax,
    points: list[np.ndarray],
    *,
    radius_frac: float = 0.58,
    min_axis_span_m: float | None = None,
) -> None:
    valid = [_finite_xyz(p) for p in points if p.size]
    valid = [p for p in valid if p.shape[0]]
    if not valid:
        return
    all_points = np.vstack(valid)
    p_min = np.min(all_points, axis=0)
    p_max = np.max(all_points, axis=0)
    center = 0.5 * (p_min + p_max)
    span = float(np.max(p_max - p_min))
    if min_axis_span_m is not None and min_axis_span_m > 0:
        span = max(span, float(min_axis_span_m))
    if span < 1e-4:
        span = 1.0
    radius = float(radius_frac) * span
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _plot_one_hand(
    ax,
    xyz: np.ndarray,
    color,
    label: str,
    linestyle: str,
    markers: bool,
    marker_size: float,
) -> None:
    p = _finite_xyz(xyz)
    if p.shape[0] == 0:
        return
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, lw=1.55, ls=linestyle, alpha=0.88, label=label)
    if markers and p.shape[0] >= 1:
        ax.scatter(*p[0], color=color, s=marker_size, marker="o", edgecolors="black", linewidths=0.35, zorder=5)
        if p.shape[0] >= 2:
            ax.scatter(*p[-1], color=color, s=marker_size * 1.1, marker="X", edgecolors="black", linewidths=0.35, zorder=5)


def _plot_one_hand_solid_dash_insert(
    ax,
    xyz: np.ndarray,
    inserted_mask: np.ndarray | None,
    color,
    label: str | None,
    markers: bool,
    marker_size: float,
) -> None:
    """Solid = original command, dashed = aug inserted (same lw/alpha)."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return
    ins = None
    if inserted_mask is not None:
        ins = np.asarray(inserted_mask, dtype=bool).reshape(-1)
        if ins.shape[0] != xyz.shape[0]:
            ins = None
    if ins is None or not np.any(ins):
        _plot_one_hand(ax, xyz, color, label or "", "-", markers, marker_size)
        return

    lw, alpha = 1.55, 0.88
    label_used = False
    for s, e, is_ins in _contiguous_bool_runs(ins):
        seg = _finite_xyz(xyz[s : e + 1])
        if seg.shape[0] == 0:
            continue
        lab = label if (label is not None and not label_used) else None
        ls = "--" if is_ins else "-"
        if seg.shape[0] == 1:
            ax.scatter(
                *seg[0],
                color=color,
                s=marker_size * 0.95,
                marker="o",
                edgecolors="black",
                linewidths=0.35,
                label=lab,
                zorder=6,
            )
            if lab is not None:
                label_used = True
            continue
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            label=lab,
            zorder=6 if is_ins else 4,
        )
        if lab is not None:
            label_used = True
    if markers and xyz.shape[0] >= 1:
        p0 = xyz[0]
        p1 = xyz[-1]
        if np.all(np.isfinite(p0)):
            ax.scatter(*p0, color=color, s=marker_size, marker="o", edgecolors="black", linewidths=0.35, zorder=8)
        if np.all(np.isfinite(p1)) and xyz.shape[0] >= 2:
            ax.scatter(*p1, color=color, s=marker_size * 1.1, marker="X", edgecolors="black", linewidths=0.35, zorder=8)


def _collect_inserted_xyz_for_limits(traj: sm.EeTrajectory, hands: HandsMode) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    if traj.inserted_mask is None or not np.any(traj.inserted_mask):
        return out
    m = traj.inserted_mask
    if hands in ("left", "both"):
        out.append(traj.left_xyz[m])
    if hands in ("right", "both"):
        out.append(traj.right_xyz[m])
    return out


def plot_batch_3d(
    batch_dir: str,
    glob_pattern: str,
    valid_only: bool,
    hands: HandsMode,
    output: str | None,
    show: bool,
    seed: int,
    markers: bool,
    max_legend_items: int,
    highlight_insert: bool,
    zoom_augment: bool,
    smooth_window: int,
) -> None:
    if not show:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")

    pattern = os.path.join(os.path.abspath(os.path.expanduser(batch_dir)), glob_pattern)
    parquet_paths = sorted(glob.glob(pattern))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for pattern: {pattern}")

    trajectories: list[sm.EeTrajectory] = []
    for path in parquet_paths:
        trajectories.append(sm.load_ee_trajectory(path, valid_only=valid_only))

    rng = np.random.default_rng(seed)
    colors = [tuple(0.18 + 0.78 * rng.random(3)) for _ in trajectories]
    any_insert = highlight_insert and any(
        t.inserted_mask is not None and np.any(t.inserted_mask) for t in trajectories
    )

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_points: list[np.ndarray] = []
    for traj, color in zip(trajectories, colors):
        stem = os.path.splitext(os.path.basename(traj.parquet_path))[0]
        if hands in ("left", "both"):
            lbl = f"{stem} left" if hands == "both" else stem
            left_xyz = _smooth_xyz_ma(traj.left_xyz, smooth_window)
            if highlight_insert and traj.inserted_mask is not None and np.any(traj.inserted_mask):
                _plot_one_hand_solid_dash_insert(ax, left_xyz, traj.inserted_mask, color, lbl, markers, 28.0)
            else:
                _plot_one_hand(ax, left_xyz, color, lbl, "-", markers, 28.0)
            all_points.append(left_xyz)
        if hands in ("right", "both"):
            lbl = f"{stem} right" if hands == "both" else stem
            right_xyz = _smooth_xyz_ma(traj.right_xyz, smooth_window)
            if highlight_insert and traj.inserted_mask is not None and np.any(traj.inserted_mask):
                _plot_one_hand_solid_dash_insert(ax, right_xyz, traj.inserted_mask, color, lbl, markers, 28.0)
            else:
                base_ls = "--" if hands == "both" else "-"
                _plot_one_hand(ax, right_xyz, color, lbl, base_ls, markers, 28.0)
            all_points.append(right_xyz)

    zoom_points: list[np.ndarray] = []
    for traj in trajectories:
        zoom_points.extend(_collect_inserted_xyz_for_limits(traj, hands))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    hands_label = {"left": "left EE only", "right": "right EE only", "both": "left + right EE"}[hands]
    extra_parts: list[str] = []
    if smooth_window > 1:
        extra_parts.append(f"MA smooth w={smooth_window}")
    if any_insert and highlight_insert:
        extra_parts.append("solid=original, dashed=aug insert")
    if zoom_augment and zoom_points:
        extra_parts.append("zoom to inserted samples")
    extra = (" | " + "; ".join(extra_parts)) if extra_parts else ""

    fig.suptitle(
        f"Batch 3D EE ({len(trajectories)} files) | {hands_label} | valid_only={valid_only}{extra}",
        fontsize=11,
    )

    n_legend = len(trajectories) * (2 if hands == "both" else 1)
    if n_legend <= max_legend_items:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.9)
    else:
        print(
            f"Legend omitted ({n_legend} entries > --max-legend-items {max_legend_items}). "
            "Colors follow sorted file order; see printed list.",
            file=sys.stderr,
        )
        for i, p in enumerate(parquet_paths):
            print(f"  [{i:4d}] {colors[i]}  {p}", file=sys.stderr)

    if zoom_augment and zoom_points:
        _set_equal_3d(ax, zoom_points, radius_frac=0.58, min_axis_span_m=None)
    else:
        _set_equal_3d(ax, all_points)
    if zoom_augment and not zoom_points:
        print("warning: --zoom-augment ignored (no aug.is_inserted_frame True in this batch)", file=sys.stderr)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.95])

    if output:
        output = os.path.abspath(os.path.expanduser(output))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fig.savefig(output, dpi=170, bbox_inches="tight")
        print(f"Saved plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay 3D gripper EE trajectories from parquet files (default: left arm, MA-smoothed display, solid/dash aug)."
    )
    parser.add_argument("--batch-dir", required=True, help="Directory containing parquet recordings.")
    parser.add_argument("--glob", dest="glob_pattern", default="*.parquet", help="Glob under batch-dir (default: *.parquet).")
    parser.add_argument("--valid-only", action="store_true", help="Use only rows with command.valid_mask == true when present.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for per-file colors (default: 0).")
    parser.add_argument(
        "--hands",
        choices=("left", "right", "both"),
        default="left",
        help="Which arm(s) to plot (default: left).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd-ish moving-average window (frames) applied to displayed XYZ only; use 1 to disable (default: 9).",
    )
    parser.add_argument("--output", default=None, help="Output PNG path. Required unless --show.")
    parser.add_argument("--show", action="store_true", help="Show interactive window (still respects MPL backend).")
    parser.add_argument("--no-markers", action="store_true", help="Do not draw start/end scatter markers per trajectory.")
    parser.add_argument(
        "--max-legend-items",
        type=int,
        default=48,
        help="If legend would exceed this many entries, omit legend and print color map to stderr (default: 48).",
    )
    parser.add_argument(
        "--no-augment-highlight",
        action="store_true",
        help="Do not split original vs aug.is_inserted_frame (single solid style).",
    )
    parser.add_argument(
        "--zoom-augment",
        action="store_true",
        help="Crop 3D axis limits to inserted EE samples only (optional; same padding as full view).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None and not args.show:
        print("Provide --output or --show.", file=sys.stderr)
        sys.exit(2)
    sw = int(args.smooth_window)
    if sw < 1:
        sw = 1
    plot_batch_3d(
        batch_dir=args.batch_dir,
        glob_pattern=args.glob_pattern,
        valid_only=args.valid_only,
        hands=args.hands,  # type: ignore[arg-type]
        output=args.output,
        show=args.show,
        seed=args.seed,
        markers=not args.no_markers,
        max_legend_items=args.max_legend_items,
        highlight_insert=not args.no_augment_highlight,
        zoom_augment=args.zoom_augment,
        smooth_window=sw,
    )


if __name__ == "__main__":
    main()
