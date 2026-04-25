#!/usr/bin/env python3
"""Visualize dual-hand keypoints from a parquet episode.

Default mode is frame-by-frame animation:
  conda activate tv
  python scripts/plot_dual_hand_trajectory_from_parquet.py \
    --parquet dataset/data/chunk-000/episode_000000.parquet \
    --show
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    print("Missing dependency: pandas. Please install it in your environment.", file=sys.stderr)
    sys.exit(1)


def _set_equal_3d(ax, points: list[np.ndarray]) -> None:
    valid = [p for p in points if p.size and np.any(np.isfinite(p))]
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
    if span < 1e-6:
        span = 1.0
    radius = 0.55 * span
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _load_kp3d_frames(series: pd.Series) -> np.ndarray:
    frames = np.full((len(series), 21, 3), np.nan, dtype=np.float64)
    for idx, value in enumerate(series):
        arr = np.asarray(value, dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] < 21:
            continue
        arr = arr[:21].copy()
        if not np.all(np.isfinite(arr)):
            continue
        if np.all(np.abs(arr) < 1e-9):
            continue
        frames[idx] = arr
    return frames


def _filter_outliers_by_wrist(frames: np.ndarray, sigma: float) -> np.ndarray:
    out = frames.copy()
    if out.shape[0] < 8:
        return out
    wrist = out[:, 0, :]
    valid = np.all(np.isfinite(wrist), axis=1)
    if int(np.sum(valid)) < 8:
        return out
    wrist_valid = wrist[valid]
    center = np.median(wrist_valid, axis=0)
    dist = np.linalg.norm(wrist_valid - center, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        return out
    keep_valid = dist <= (med + sigma * scale)
    valid_idx = np.where(valid)[0]
    reject_idx = valid_idx[~keep_valid]
    out[reject_idx] = np.nan
    return out


def _extract_joint_xyz(frames: np.ndarray, joint_index: int) -> np.ndarray:
    if joint_index < 0 or joint_index >= frames.shape[1]:
        raise ValueError(f"joint-index={joint_index} out of range [0, {frames.shape[1]-1}]")
    xyz = frames[:, joint_index, :]
    valid = np.all(np.isfinite(xyz), axis=1)
    return xyz[valid]


def _valid_points(frames: np.ndarray) -> np.ndarray:
    flat = frames.reshape(-1, 3)
    valid = np.all(np.isfinite(flat), axis=1)
    return flat[valid]


def _valid_frame_count(frames: np.ndarray) -> int:
    return int(np.sum(np.all(np.isfinite(frames[:, 0, :]), axis=1)))


def load_dual_hand_kp3d_frames(parquet_path: str, outlier_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    for col in ("left_kp3d", "right_kp3d"):
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {parquet_path}")

    left_frames = _load_kp3d_frames(df["left_kp3d"])
    right_frames = _load_kp3d_frames(df["right_kp3d"])
    if outlier_sigma > 0:
        left_frames = _filter_outliers_by_wrist(left_frames, sigma=outlier_sigma)
        right_frames = _filter_outliers_by_wrist(right_frames, sigma=outlier_sigma)
    return left_frames, right_frames


def plot_trajectory(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    title: str,
    output: str | None,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(title, fontsize=12)

    if left_xyz.size:
        ax.plot(left_xyz[:, 0], left_xyz[:, 1], left_xyz[:, 2], color="C0", lw=1.5, label="left hand")
        ax.scatter(*left_xyz[0], color="0.5", s=35, marker="o")
        ax.scatter(*left_xyz[-1], color="green", s=40, marker="o")
    if right_xyz.size:
        ax.plot(right_xyz[:, 0], right_xyz[:, 1], right_xyz[:, 2], color="C1", lw=1.5, label="right hand")
        ax.scatter(*right_xyz[0], color="0.5", s=35, marker="s")
        ax.scatter(*right_xyz[-1], color="green", s=40, marker="s")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right")
    _set_equal_3d(ax, [left_xyz, right_xyz])
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _set_scatter_xyz(scatter, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        scatter._offsets3d = ([], [], [])
        return
    scatter._offsets3d = ([xyz[0]], [xyz[1]], [xyz[2]])


def _set_scatter_cloud_xyz(scatter, xyzs: np.ndarray) -> None:
    if xyzs.size == 0:
        scatter._offsets3d = ([], [], [])
        return
    scatter._offsets3d = (xyzs[:, 0], xyzs[:, 1], xyzs[:, 2])


def animate_hands(
    left_frames: np.ndarray,
    right_frames: np.ndarray,
    title: str,
    output: str | None,
    show: bool,
    fps: float,
    max_frames: int,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n_total = min(left_frames.shape[0], right_frames.shape[0])
    n = min(n_total, max_frames) if max_frames > 0 else n_total
    if n <= 0:
        raise ValueError("No frames available for animation.")

    left_frames = left_frames[:n]
    right_frames = right_frames[:n]
    all_pts = np.vstack([_valid_points(left_frames), _valid_points(right_frames)])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(title, fontsize=12)
    _set_equal_3d(ax, [all_pts])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    left_wrist = ax.scatter([], [], [], s=180, c="C0", marker="o", label="left wrist")
    right_wrist = ax.scatter([], [], [], s=180, c="C1", marker="o", label="right wrist")
    left_fingers = ax.scatter([], [], [], s=20, c="C0", alpha=0.8, marker="o", label="left fingers")
    right_fingers = ax.scatter([], [], [], s=20, c="C1", alpha=0.8, marker="o", label="right fingers")
    ax.legend(loc="upper right")

    interval_ms = max(1, int(round(1000.0 / max(fps, 1e-3))))

    def update(frame_idx: int):
        lw = left_frames[frame_idx, 0]
        rw = right_frames[frame_idx, 0]
        lf = left_frames[frame_idx, 1:]
        rf = right_frames[frame_idx, 1:]
        _set_scatter_xyz(left_wrist, lw if np.all(np.isfinite(lw)) else np.zeros((0,)))
        _set_scatter_xyz(right_wrist, rw if np.all(np.isfinite(rw)) else np.zeros((0,)))
        _set_scatter_cloud_xyz(left_fingers, lf[np.all(np.isfinite(lf), axis=1)])
        _set_scatter_cloud_xyz(right_fingers, rf[np.all(np.isfinite(rf), axis=1)])
        ax.set_title(f"Frame {frame_idx + 1}/{n}")
        return left_wrist, right_wrist, left_fingers, right_fingers

    ani = FuncAnimation(fig, update, frames=n, interval=interval_ms, blit=False, repeat=True)

    if output:
        ani.save(output, fps=fps)
        print(f"Saved animation: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dual-hand 3D keypoints from parquet episode.")
    parser.add_argument("--parquet", required=True, help="Path to episode parquet file")
    parser.add_argument(
        "--joint-index",
        type=int,
        default=0,
        help="Joint index for trajectory mode only (default=0 for wrist)",
    )
    parser.add_argument("--output", default=None, help="Output path (.gif/.mp4 for animate, .png for trajectory)")
    parser.add_argument("--show", action="store_true", help="Show interactive figure window")
    parser.add_argument(
        "--mode",
        choices=["animate", "trajectory"],
        default="animate",
        help="animate: frame-by-frame hand keypoints; trajectory: static path of one joint",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS for animation mode")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for animation (0 means all)")
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=6.0,
        help="Robust outlier rejection threshold in MAD-sigma; <=0 disables filter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = os.path.abspath(os.path.expanduser(args.parquet))
    left_frames, right_frames = load_dual_hand_kp3d_frames(parquet_path, outlier_sigma=args.outlier_sigma)
    left_xyz = _extract_joint_xyz(left_frames, joint_index=args.joint_index)
    right_xyz = _extract_joint_xyz(right_frames, joint_index=args.joint_index)

    print(f"Loaded parquet: {parquet_path}")
    print(f"Left valid wrist frames: {_valid_frame_count(left_frames)}")
    print(f"Right valid wrist frames: {_valid_frame_count(right_frames)}")
    if left_xyz.size:
        print(
            f"Left bounds X[{left_xyz[:, 0].min():.3f},{left_xyz[:, 0].max():.3f}] "
            f"Y[{left_xyz[:, 1].min():.3f},{left_xyz[:, 1].max():.3f}] "
            f"Z[{left_xyz[:, 2].min():.3f},{left_xyz[:, 2].max():.3f}]"
        )
    if right_xyz.size:
        print(
            f"Right bounds X[{right_xyz[:, 0].min():.3f},{right_xyz[:, 0].max():.3f}] "
            f"Y[{right_xyz[:, 1].min():.3f},{right_xyz[:, 1].max():.3f}] "
            f"Z[{right_xyz[:, 2].min():.3f},{right_xyz[:, 2].max():.3f}]"
        )

    output = args.output
    if output is None and not args.show:
        base = os.path.splitext(os.path.basename(parquet_path))[0]
        ext = "gif" if args.mode == "animate" else "png"
        output = os.path.join(os.getcwd(), f"{base}_dual_hand_{args.mode}.{ext}")
        print(f"No --output and no --show, writing default file: {output}")

    if args.mode == "trajectory":
        title = f"Dual-hand trajectory from parquet (joint={args.joint_index}, wrist=0)"
        plot_trajectory(left_xyz, right_xyz, title=title, output=output, show=args.show)
    else:
        title = "Dual-hand frame playback (wrist=big sphere, fingers=small spheres)"
        animate_hands(
            left_frames=left_frames,
            right_frames=right_frames,
            title=title,
            output=output,
            show=args.show,
            fps=args.fps,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
