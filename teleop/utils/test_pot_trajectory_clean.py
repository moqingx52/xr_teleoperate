"""Tests for pot trajectory cleaning and endpoint alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

XR_ROOT = Path(__file__).resolve().parents[2]
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))

from teleop.utils.pot_align import (  # noqa: E402
    apply_endpoint_linear_correction,
    compute_arclength_alpha,
    compute_endpoint_pos_target,
)
from teleop.utils.pot_trajectory_clean import clean_pot_trajectory  # noqa: E402


def test_remove_single_outlier_spike():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.04, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    keep, summary = clean_pot_trajectory(pos, jump_thresh_floor=0.05)
    assert summary["removed_outlier"] >= 1
    assert not keep[3]
    assert int(np.sum(keep)) == summary["frames_out"]


def test_trim_tail_fast_motion():
    pos = np.zeros((20, 3), dtype=np.float64)
    for i in range(1, 15):
        pos[i] = [0.01 * i, 0.0, 0.0]
    for i in range(15, 20):
        # Above tail_speed_thresh (0.03) but below jump_thresh_floor (0.05).
        pos[i] = pos[i - 1] + np.array([0.04, 0.0, 0.0])
    keep, summary = clean_pot_trajectory(
        pos,
        tail_speed_thresh=0.03,
        tail_run_frames=3,
        min_frames=5,
    )
    assert summary["removed_tail"] >= 3
    assert int(np.sum(keep)) < len(pos)
    last_kept = int(np.flatnonzero(keep)[-1])
    assert last_kept == summary["tail_cut_index"]


def test_endpoint_linear_correction():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    end_target = np.array([3.0, 1.0, 0.5])
    alpha = compute_arclength_alpha(pos)
    corrected = apply_endpoint_linear_correction(pos, end_target, alpha)
    np.testing.assert_allclose(corrected[0], pos[0])
    np.testing.assert_allclose(corrected[-1], end_target, atol=1e-12)
    np.testing.assert_allclose(alpha[0], 0.0)
    np.testing.assert_allclose(alpha[-1], 1.0)


def test_compute_endpoint_pos_target():
    trivet = np.array([1.0, 2.0, 0.3])
    target = compute_endpoint_pos_target(trivet, z_offset_m=0.2)
    np.testing.assert_allclose(target, [1.0, 2.0, 0.5])


def test_scene_start_pos_with_z_offset():
    from teleop.utils.pot_align import clamp_z_to_start, scene_start_pos_with_z_offset

    scene = np.array([1.0, 2.0, 0.5])
    start = scene_start_pos_with_z_offset(scene, 0.05)
    np.testing.assert_allclose(start, [1.0, 2.0, 0.55])

    pos = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
            [0.2, 0.0, 1.1],
        ],
        dtype=np.float64,
    )
    clamped, n = clamp_z_to_start(pos)
    assert n == 1
    np.testing.assert_allclose(clamped[1], [0.1, 0.0, 1.0])
    np.testing.assert_allclose(clamped[0], pos[0])


if __name__ == "__main__":
    test_remove_single_outlier_spike()
    test_trim_tail_fast_motion()
    test_endpoint_linear_correction()
    test_compute_endpoint_pos_target()
    test_scene_start_pos_with_z_offset()
    print("tests ok")
