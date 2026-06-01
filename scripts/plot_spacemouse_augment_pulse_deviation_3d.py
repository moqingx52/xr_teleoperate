#!/usr/bin/env python3
"""Visualize augmented pulse trajectories relative to a fixed reference parquet.

This script is designed for SpaceMouse augmentation outputs that contain
`aug.source_path`, `aug.is_inserted_frame`, and `aug.anchor_command_index`.
It magnifies only the local pulse neighborhood about each file's anchor point:

    p_vis = anchor + dev_scale * (p - anchor)

So start/end anchor consistency is preserved while pulse path differences become
visually separable.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    print("Missing dependency: pandas.", file=sys.stderr)
    sys.exit(1)

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    import plot_spacemouse_ee_trajectory as sm  # type: ignore
except ModuleNotFoundError:
    print("Could not import plot_spacemouse_ee_trajectory (expected next to this script).", file=sys.stderr)
    sys.exit(1)


HandsMode = Literal["left", "right", "both"]


@dataclass
class AugMeta:
    parquet_path: str
    source_path: str
    anchor_idx: int


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return xyz
    return xyz[np.all(np.isfinite(xyz), axis=1)]


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


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0 or radius <= 0:
        return m
    r = int(radius)
    out = np.zeros_like(m)
    idx = np.flatnonzero(m)
    for i in idx:
        s = max(0, i - r)
        e = min(m.size, i + r + 1)
        out[s:e] = True
    return out


def _magnify_about_anchor(xyz: np.ndarray, anchor_xyz: np.ndarray, dev_scale: float) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(anchor_xyz, dtype=np.float64).reshape(1, 3)
    return a + float(dev_scale) * (pts - a)


def _set_equal_3d(ax, points: list[np.ndarray]) -> None:
    valid = [_finite_xyz(p) for p in points if p.size]
    valid = [p for p in valid if p.shape[0]]
    if not valid:
        return
    all_points = np.vstack(valid)
    p_min = np.min(all_points, axis=0)
    p_max = np.max(all_points, axis=0)
    center = 0.5 * (p_min + p_max)
    span = float(np.max(p_max - p_min))
    if span < 1e-5:
        span = 1e-2
    radius = 0.6 * span
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _read_aug_meta(parquet_path: str, source_match: str | None) -> AugMeta | None:
    cols = ["entry_type", "aug.source_path", "aug.is_inserted_frame", "aug.anchor_command_index"]
    try:
        df = pd.read_parquet(parquet_path, columns=cols)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: skip {parquet_path} (cannot read required aug columns: {exc})", file=sys.stderr)
        return None

    if "entry_type" not in df.columns:
        return None
    cmd = df[df["entry_type"].astype(str) == "command"].copy()
    if cmd.empty:
        return None
    if "aug.is_inserted_frame" not in cmd.columns:
        return None

    source_candidates = []
    if "aug.source_path" in cmd.columns:
        source_candidates = [str(v) for v in cmd["aug.source_path"].dropna().astype(str).unique().tolist() if str(v)]
    source_path = source_candidates[0] if source_candidates else ""

    if source_match:
        hit = False
        for s in source_candidates:
            if source_match in s:
                hit = True
                source_path = s
                break
        if not hit:
            return None

    inserted = cmd["aug.is_inserted_frame"].where(cmd["aug.is_inserted_frame"].notna(), False).astype(bool).to_numpy()
    if not np.any(inserted):
        return None

    anchor_vals = pd.to_numeric(cmd.loc[inserted, "aug.anchor_command_index"], errors="coerce").to_numpy(dtype=np.float64)
    anchor_vals = anchor_vals[np.isfinite(anchor_vals) & (anchor_vals >= 0)]
    if anchor_vals.size == 0:
        print(f"warning: skip {parquet_path} (no valid aug.anchor_command_index on inserted rows)", file=sys.stderr)
        return None

    unique_anchor = np.unique(anchor_vals.astype(np.int64))
    if unique_anchor.size > 1:
        print(
            f"warning: {parquet_path} has multiple anchor indices {unique_anchor.tolist()}, using first",
            file=sys.stderr,
        )
    return AugMeta(parquet_path=parquet_path, source_path=source_path, anchor_idx=int(unique_anchor[0]))


def _plot_magnified_runs(
    ax,
    xyz: np.ndarray,
    mask: np.ndarray,
    color,
    label: str,
) -> list[np.ndarray]:
    used = False
    plotted_pts: list[np.ndarray] = []
    for s, e, on in _contiguous_bool_runs(mask):
        if not on:
            continue
        seg = xyz[s : e + 1]
        seg = _finite_xyz(seg)
        if seg.shape[0] == 0:
            continue
        plotted_pts.append(seg)
        lab = label if not used else None
        used = True
        if seg.shape[0] == 1:
            ax.scatter(*seg[0], color=color, s=24, label=lab)
        else:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=1.8, alpha=0.95, label=lab)
    return plotted_pts


def _draw_reference_context(
    ax,
    ref_xyz: np.ndarray,
    anchor_idx: int,
    dev_scale: float,
    context_frames: int,
    label: str,
) -> np.ndarray | None:
    n = ref_xyz.shape[0]
    if n == 0 or anchor_idx < 0 or anchor_idx >= n:
        return None
    w = max(0, int(context_frames))
    s = max(0, anchor_idx - w)
    e = min(n - 1, anchor_idx + w)
    seg = ref_xyz[s : e + 1]
    if seg.shape[0] == 0:
        return None
    anchor = ref_xyz[anchor_idx]
    seg_vis = _magnify_about_anchor(seg, anchor, dev_scale)
    seg_vis = _finite_xyz(seg_vis)
    if seg_vis.shape[0] == 0:
        return None
    if seg_vis.shape[0] == 1:
        ax.scatter(*seg_vis[0], color="0.3", s=28, marker="x", label=label)
    else:
        ax.plot(seg_vis[:, 0], seg_vis[:, 1], seg_vis[:, 2], color="0.35", lw=1.1, alpha=0.85, label=label)
    return seg_vis


def plot_pulse_deviation(
    reference_parquet: str,
    augment_dir: str,
    glob_pattern: str,
    source_match: str | None,
    valid_only: bool,
    hands: HandsMode,
    dev_scale: float,
    context_frames: int,
    show_ref_tangent: bool,
    ref_context_frames: int,
    seed: int,
    max_legend_items: int,
    output: str | None,
    show: bool,
) -> None:
    if not show:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_parquet = os.path.abspath(os.path.expanduser(reference_parquet))
    augment_dir = os.path.abspath(os.path.expanduser(augment_dir))

    ref_traj = sm.load_ee_trajectory(reference_parquet, valid_only=False)
    ref_left = np.asarray(ref_traj.left_xyz, dtype=np.float64)
    ref_right = np.asarray(ref_traj.right_xyz, dtype=np.float64)

    pattern = os.path.join(augment_dir, glob_pattern)
    aug_paths = sorted(glob.glob(pattern))
    if not aug_paths:
        raise FileNotFoundError(f"No parquet files found for pattern: {pattern}")

    metas: list[AugMeta] = []
    for p in aug_paths:
        meta = _read_aug_meta(p, source_match=source_match)
        if meta is not None:
            metas.append(meta)
    if not metas:
        raise ValueError("No augmented parquet matched source filter and required aug columns.")

    rng = np.random.default_rng(seed)
    colors = [tuple(0.18 + 0.78 * rng.random(3)) for _ in metas]

    fig = plt.figure(figsize=(11.5, 9.0))
    ax = fig.add_subplot(111, projection="3d")

    bbox_points: list[np.ndarray] = []
    plotted_count = 0

    for meta, color in zip(metas, colors):
        traj = sm.load_ee_trajectory(meta.parquet_path, valid_only=valid_only)
        ins = traj.inserted_mask
        if ins is None or not np.any(ins):
            print(f"warning: skip {meta.parquet_path} (no inserted frames after valid_only={valid_only})", file=sys.stderr)
            continue

        n_ref = ref_left.shape[0]
        if meta.anchor_idx < 0 or meta.anchor_idx >= n_ref:
            print(
                f"warning: skip {meta.parquet_path} (anchor idx {meta.anchor_idx} out of reference range 0..{n_ref-1})",
                file=sys.stderr,
            )
            continue

        pulse_mask = _dilate_mask(ins, radius=context_frames)
        stem = os.path.splitext(os.path.basename(meta.parquet_path))[0]

        if hands in ("left", "both"):
            anchor = ref_left[meta.anchor_idx]
            left_vis = _magnify_about_anchor(traj.left_xyz, anchor, dev_scale)
            pts = _plot_magnified_runs(ax, left_vis, pulse_mask, color, f"{stem} left")
            bbox_points.extend(pts)
            if show_ref_tangent:
                seg = _draw_reference_context(
                    ax,
                    ref_xyz=ref_left,
                    anchor_idx=meta.anchor_idx,
                    dev_scale=dev_scale,
                    context_frames=ref_context_frames,
                    label=f"ref left @a{meta.anchor_idx}" if plotted_count == 0 else "_nolegend_",
                )
                if seg is not None:
                    bbox_points.append(seg)

        if hands in ("right", "both"):
            if meta.anchor_idx >= ref_right.shape[0]:
                continue
            anchor = ref_right[meta.anchor_idx]
            right_vis = _magnify_about_anchor(traj.right_xyz, anchor, dev_scale)
            pts = _plot_magnified_runs(ax, right_vis, pulse_mask, color, f"{stem} right")
            bbox_points.extend(pts)
            if show_ref_tangent:
                seg = _draw_reference_context(
                    ax,
                    ref_xyz=ref_right,
                    anchor_idx=meta.anchor_idx,
                    dev_scale=dev_scale,
                    context_frames=ref_context_frames,
                    label=f"ref right @a{meta.anchor_idx}" if plotted_count == 0 else "_nolegend_",
                )
                if seg is not None:
                    bbox_points.append(seg)

        plotted_count += 1

    if not bbox_points:
        raise ValueError("No pulse segments were plotted. Try disabling --valid-only or reducing filters.")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Display-only magnified local pulse deviations around per-file anchors")
    _set_equal_3d(ax, bbox_points)

    reference_name = os.path.basename(reference_parquet)
    hands_text = {"left": "left", "right": "right", "both": "both"}[hands]
    fig.suptitle(
        f"Pulse deviation vs reference ({len(metas)} files, plotted={plotted_count}) | "
        f"ref={reference_name} | hands={hands_text} | scale={dev_scale:.2f} | "
        f"context=+/-{context_frames} frames | valid_only={valid_only}",
        fontsize=10,
    )

    n_legend = plotted_count * (2 if hands == "both" else 1)
    if show_ref_tangent:
        n_legend += 1
    if n_legend <= max_legend_items:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.9)
    else:
        print(
            f"Legend omitted ({n_legend} entries > --max-legend-items {max_legend_items}).",
            file=sys.stderr,
        )
        for i, meta in enumerate(metas):
            print(f"  [{i:4d}] {colors[i]}  {meta.parquet_path}  anchor={meta.anchor_idx}", file=sys.stderr)

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.95])
    if output:
        output = os.path.abspath(os.path.expanduser(output))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Saved plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize magnified local pulse deviations relative to a fixed reference parquet."
    )
    parser.add_argument("--reference-parquet", required=True, help="Original source parquet path used to generate augmentations.")
    parser.add_argument("--augment-dir", required=True, help="Directory containing augmented parquet files.")
    parser.add_argument("--glob", dest="glob_pattern", default="*.parquet", help="Glob under --augment-dir (default: *.parquet).")
    parser.add_argument(
        "--source-match",
        default=None,
        help="Substring matched against aug.source_path to select files. Defaults to reference parquet basename.",
    )
    parser.add_argument("--valid-only", action="store_true", help="Use only rows with command.valid_mask == true for display.")
    parser.add_argument(
        "--hands",
        choices=("left", "right", "both"),
        default="left",
        help="Which hand(s) to visualize (default: left).",
    )
    parser.add_argument("--dev-scale", type=float, default=12.0, help="Display-only magnification around anchor (default: 12.0).")
    parser.add_argument(
        "--context-frames",
        type=int,
        default=4,
        help="Include +/- N non-inserted frames around inserted frames (default: 4).",
    )
    parser.add_argument(
        "--also-show-ref-tangent",
        action="store_true",
        help="Plot reference local segment around anchor (magnified by same --dev-scale).",
    )
    parser.add_argument(
        "--ref-context-frames",
        type=int,
        default=8,
        help="Reference context half-window in frames for --also-show-ref-tangent (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for per-file colors (default: 0).")
    parser.add_argument(
        "--max-legend-items",
        type=int,
        default=52,
        help="If legend would exceed this many entries, omit legend and print mapping to stderr.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path. Required unless --show.")
    parser.add_argument("--show", action="store_true", help="Show interactive window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None and not args.show:
        ref_base = os.path.splitext(os.path.basename(os.path.abspath(os.path.expanduser(args.reference_parquet))))[0]
        args.output = os.path.join(
            os.path.abspath(os.path.expanduser(args.augment_dir)),
            f"{ref_base}_pulse_deviation_3d.png",
        )

    source_match = args.source_match
    if source_match is None:
        source_match = os.path.basename(os.path.abspath(os.path.expanduser(args.reference_parquet)))

    plot_pulse_deviation(
        reference_parquet=args.reference_parquet,
        augment_dir=args.augment_dir,
        glob_pattern=args.glob_pattern,
        source_match=source_match,
        valid_only=bool(args.valid_only),
        hands=args.hands,  # type: ignore[arg-type]
        dev_scale=float(args.dev_scale),
        context_frames=max(0, int(args.context_frames)),
        show_ref_tangent=bool(args.also_show_ref_tangent),
        ref_context_frames=max(0, int(args.ref_context_frames)),
        seed=int(args.seed),
        max_legend_items=int(args.max_legend_items),
        output=args.output,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
