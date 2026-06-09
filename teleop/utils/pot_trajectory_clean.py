"""Remove outlier jumps and tail motion spikes from pot trajectory positions."""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class CleanSummary(TypedDict):
    frames_in: int
    frames_out: int
    removed_outlier: int
    removed_tail: int
    tail_cut_index: int


def _step_lengths(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(pos, axis=0), axis=1)


def _mad_threshold(step: np.ndarray, *, jump_thresh_floor: float, mad_k: float) -> float:
    if step.size == 0:
        return float(jump_thresh_floor)
    med = float(np.median(step))
    mad = float(np.median(np.abs(step - med)))
    robust = med + float(mad_k) * (1.4826 * mad)
    return max(float(jump_thresh_floor), robust)


def _remove_outlier_frames(
    pos: np.ndarray,
    *,
    jump_thresh_floor: float,
    mad_k: float,
) -> tuple[np.ndarray, int]:
    n = int(pos.shape[0])
    if n < 2:
        return np.ones(n, dtype=bool), 0

    keep = np.ones(n, dtype=bool)
    step = _step_lengths(pos)
    thresh = _mad_threshold(step, jump_thresh_floor=jump_thresh_floor, mad_k=mad_k)

    i = 1
    while i < n:
        if step[i - 1] <= thresh:
            i += 1
            continue
        # Drop consecutive high-step frames until motion returns below threshold.
        start = i
        while i < n and step[i - 1] > thresh:
            keep[i] = False
            i += 1
        removed_run = int(np.sum(~keep[start:i]))
        if removed_run == 0 and start < n:
            keep[start] = False
            i = start + 1

    return keep, int(np.sum(~keep))


def _trim_tail_motion(
    pos: np.ndarray,
    keep: np.ndarray,
    *,
    tail_speed_thresh: float,
    tail_run_frames: int,
    min_frames: int,
) -> tuple[np.ndarray, int, int]:
    n = int(pos.shape[0])
    if n < 2:
        return keep, 0, n - 1

    indices = np.flatnonzero(keep)
    if indices.size < 2:
        return keep, 0, n - 1

    pos_kept = pos[indices]
    step = _step_lengths(pos_kept)
    tail_run = max(1, int(tail_run_frames))
    min_keep = max(1, int(min_frames))

    cut_kept_idx = len(indices) - 1
    fast_run = 0
    for k in range(len(step) - 1, -1, -1):
        if step[k] > float(tail_speed_thresh):
            fast_run += 1
            if fast_run >= tail_run:
                cut_kept_idx = k
        else:
            if fast_run >= tail_run:
                break
            fast_run = 0

    if cut_kept_idx >= len(indices) - 1:
        return keep, 0, int(indices[-1])

    if cut_kept_idx + 1 < min_keep:
        cut_kept_idx = min(len(indices) - 1, min_keep - 1)

    last_keep_global = int(indices[cut_kept_idx])
    removed = int(np.sum(keep) - (cut_kept_idx + 1))
    if removed > 0:
        keep[indices[cut_kept_idx + 1] :] = False

    return keep, removed, last_keep_global


def clean_pot_trajectory(
    pos: np.ndarray,
    quat: np.ndarray | None = None,
    times: np.ndarray | None = None,
    *,
    jump_thresh_floor: float = 0.05,
    mad_k: float = 6.0,
    tail_speed_thresh: float = 0.03,
    tail_run_frames: int = 3,
    min_frames: int = 10,
) -> tuple[np.ndarray, CleanSummary]:
    """
    Return keep_mask (bool, len N) and summary dict.

    pos: [N, 3] world or local positions; quat/times are accepted for API symmetry
    but only pos drives cleaning decisions.
    """
    del quat, times  # reserved for future time-aware rules

    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = int(pos.shape[0])
    if n == 0:
        raise ValueError("pos must contain at least one frame")

    keep = np.ones(n, dtype=bool)
    removed_outlier = 0
    removed_tail = 0
    tail_cut_index = n - 1

    if n >= 2:
        keep, removed_outlier = _remove_outlier_frames(
            pos,
            jump_thresh_floor=float(jump_thresh_floor),
            mad_k=float(mad_k),
        )
        keep, removed_tail, tail_cut_index = _trim_tail_motion(
            pos,
            keep,
            tail_speed_thresh=float(tail_speed_thresh),
            tail_run_frames=int(tail_run_frames),
            min_frames=int(min_frames),
        )

    frames_out = int(np.sum(keep))
    if frames_out < 1:
        raise ValueError("clean_pot_trajectory would remove all frames")

    summary: CleanSummary = {
        "frames_in": n,
        "frames_out": frames_out,
        "removed_outlier": int(removed_outlier),
        "removed_tail": int(removed_tail),
        "tail_cut_index": int(tail_cut_index),
    }
    return keep, summary
