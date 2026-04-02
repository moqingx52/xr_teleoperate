#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 EgoDex 风格 HDF5 读取左右手腕位置，绘制 3D 轨迹（多视角，含正等轴测近似）。

不依赖 teleop 包；需要: numpy, h5py, matplotlib

示例:
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
    print("需要安装 h5py: pip install h5py", file=sys.stderr)
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
    """
    返回 (left_xyz, right_xyz, n_frames)，形状均为 (N, 3)。
    root_frame: world | hip | camera，与 teleop EgoDex 一致。
    """
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
    """近似等比例坐标轴，避免 3D 轨迹被压扁。"""
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
        ("正等轴测(近似)", 35.264, 45),
        ("斜俯视", 25, -50),
        ("俯视 (XY)", 90, -90),
        ("侧视 (XZ)", 0, 0),
    ]

    fig = plt.figure(figsize=(11, 10))
    fig.suptitle(
        f"手腕位置轨迹 (root={root_frame}, N={left.shape[0]})\n"
        "蓝=左手  橙=右手  灰=起点  绿=终点",
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
        print(f"已保存: {out_path}")

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
        help="与 teleop --egodex-root-frame 一致",
    )
    p.add_argument("-o", "--output", default=None, help="保存 PNG")
    p.add_argument("--show", action="store_true", help="交互显示窗口")
    args = p.parse_args()

    left, right, n = load_wrist_positions(args.hdf5, args.root_frame)
    print(f"读取 {n} 帧, root_frame={args.root_frame}")
    print(
        f"左手 范围 X[{left[:, 0].min():.3f},{left[:, 0].max():.3f}] "
        f"Y[{left[:, 1].min():.3f},{left[:, 1].max():.3f}] "
        f"Z[{left[:, 2].min():.3f},{left[:, 2].max():.3f}]"
    )
    print(
        f"右手 范围 X[{right[:, 0].min():.3f},{right[:, 0].max():.3f}] "
        f"Y[{right[:, 1].min():.3f},{right[:, 1].max():.3f}] "
        f"Z[{right[:, 2].min():.3f},{right[:, 2].max():.3f}]"
    )

    out = args.output
    if out is None and not args.show:
        base, _ = os.path.splitext(os.path.basename(args.hdf5))
        out = os.path.join(os.getcwd(), f"{base}_wrist_traj_{args.root_frame}.png")
        print(f"未指定 -o 且未 --show，将保存到: {out}")

    plot_trajectories(left, right, args.root_frame, out, args.show)


if __name__ == "__main__":
    main()
