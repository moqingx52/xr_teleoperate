#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load left/right wrist positions from EgoDex-style HDF5 and plot 3D trajectories
(multiple views, including an approximate isometric view).

Standalone script; requires: numpy, h5py, matplotlib

Examples:
  cd xr_teleoperate
  python scripts/plot_egodex_wrist_trajectory.py \\
    --hdf5 /path/to/0.hdf5 --root-frame hip --show

  python scripts/plot_egodex_wrist_trajectory.py --hdf5 0.hdf5 -o wrist_traj.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    print("Install h5py: pip install h5py", file=sys.stderr)
    sys.exit(1)


def _ensure_tf44(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (4, 4):
        raise ValueError(f"Expected (4,4) transform, got {x.shape}")
    return x


def load_wrist_positions(
    hdf5_path: str,
    root_frame: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (left_xyz, right_xyz, n_frames), each (N, 3). root_frame matches teleop EgoDex."""
    root_frame = root_frame.strip().lower()
    if root_frame not in ("world", "hip", "camera"):
        raise ValueError("root_frame must be world, hip, or camera")

    path = os.path.abspath(os.path.expanduser(hdf5_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        if "transforms" not in f:
            raise ValueError(f"No /transforms in {path}")
        g = f["transforms"]
        for k in ("leftHand", "rightHand", "camera"):
            if k not in g:
                raise ValueError(f"Missing /transforms/{k}")
        if root_frame == "hip" and "hip" not in g:
            raise ValueError("root_frame=hip requires /transforms/hip")

        n = int(g["camera"].shape[0])
        left = np.zeros((n, 3), dtype=np.float64)
        right = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            if root_frame == "world":
                T_inv = np.eye(4, dtype=np.float64)
            elif root_frame == "hip":
                T_inv = np.linalg.inv(_ensure_tf44(g["hip"][i]))
            else:
                T_inv = np.linalg.inv(_ensure_tf44(g["camera"][i]))

            L = T_inv @ _ensure_tf44(g["leftHand"][i])
            R = T_inv @ _ensure_tf44(g["rightHand"][i])
            left[i] = L[:3, 3]
            right[i] = R[:3, 3]

    return left, right, n


def _set_equal_3d(ax, pts_list: list[np.ndarray]) -> None:
    """Roughly equal axis scaling so 3D trajectories are not squashed."""
    all_p = np.vstack([p for p in pts_list if p.size])
    if all_p.size == 0:
        return
    cmin = all_p.min(axis=0)
    cmax = all_p.max(axis=0)
    center = 0.5 * (cmin + cmax)
    span = float(np.max(cmax - cmin))
    if span < 1e-9:
        span = 1.0
    r = 0.5 * span * 1.05
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_trajectories(
    left: np.ndarray,
    right: np.ndarray,
    root_frame: str,
    out_path: str | None,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    views = [
        ("Isometric (approx.)", 35.264, 45),
        ("Oblique top", 25, -50),
        ("Top (XY)", 90, -90),
        ("Side (XZ)", 0, 0),
    ]

    fig = plt.figure(figsize=(11, 10))
    fig.suptitle(
        f"Wrist position trajectories (root={root_frame}, N={left.shape[0]})\n"
        "Blue=left  Orange=right  Gray=start  Green=end",
        fontsize=12,
    )

    for idx, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        ax.plot(left[:, 0], left[:, 1], left[:, 2], color="C0", lw=1.2, label="left wrist")
        ax.plot(right[:, 0], right[:, 1], right[:, 2], color="C1", lw=1.2, label="right wrist")
        ax.scatter(*left[0], color="0.5", s=40, marker="o", zorder=5)
        ax.scatter(*right[0], color="0.5", s=40, marker="s", zorder=5)
        ax.scatter(*left[-1], color="green", s=45, marker="o", zorder=5)
        ax.scatter(*right[-1], color="green", s=45, marker="s", zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        _set_equal_3d(ax, [left, right])
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot 3D left/right wrist trajectories from EgoDex HDF5")
    p.add_argument("--hdf5", required=True, help="Path to episode .hdf5")
    p.add_argument(
        "--root-frame",
        choices=["world", "hip", "camera"],
        default="hip",
        help="Same as teleop --egodex-root-frame",
    )
    p.add_argument("-o", "--output", default=None, help="Save PNG path")
    p.add_argument("--show", action="store_true", help="Show interactive window")
    args = p.parse_args()

    left, right, n = load_wrist_positions(args.hdf5, args.root_frame)
    print(f"Loaded {n} frames, root_frame={args.root_frame}")
    print(
        f"Left  bounds X[{left[:, 0].min():.3f},{left[:, 0].max():.3f}] "
        f"Y[{left[:, 1].min():.3f},{left[:, 1].max():.3f}] "
        f"Z[{left[:, 2].min():.3f},{left[:, 2].max():.3f}]"
    )
    print(
        f"Right bounds X[{right[:, 0].min():.3f},{right[:, 0].max():.3f}] "
        f"Y[{right[:, 1].min():.3f},{right[:, 1].max():.3f}] "
        f"Z[{right[:, 2].min():.3f},{right[:, 2].max():.3f}]"
    )

    out = args.output
    if out is None and not args.show:
        base, _ = os.path.splitext(os.path.basename(args.hdf5))
        out = os.path.join(os.getcwd(), f"{base}_wrist_traj_{args.root_frame}.png")
        print(f"No -o and no --show: writing default file: {out}")

    plot_trajectories(left, right, args.root_frame, out, args.show)


if __name__ == "__main__":
    main()
