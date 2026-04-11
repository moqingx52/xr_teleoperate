"""
Temporal stabilization for dual-hand (25,3) joint positions.

Mitigates occasional single-frame left/right swaps or gross outliers from
RGB-D / structured JSON at ~30 FPS by:
  1) comparing each wrist position to the median of recent accepted wrists;
  2) comparing to a constant-velocity linear prediction and translating the
     whole hand if deviation is too large.

Only wrist position drives gating; finger joints move rigidly with the wrist
when a prediction correction is applied. History rejection replaces the whole
hand with the last good pose for that side.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np


class DualHandPosTemporalStabilizer:
    def __init__(
        self,
        hist_window: int = 4,
        hist_jump_thresh_m: float = 0.25,
        pred_dev_thresh_m: float = 0.25,
        enable_hist: bool = True,
        enable_pred: bool = True,
        wrist_idx: int = 0,
    ) -> None:
        self.hist_window = max(2, int(hist_window))
        self.hist_jump_thresh_m = float(hist_jump_thresh_m)
        self.pred_dev_thresh_m = float(pred_dev_thresh_m)
        self.enable_hist = bool(enable_hist)
        self.enable_pred = bool(enable_pred)
        self.wrist_idx = int(wrist_idx)
        self.reset()

    def reset(self) -> None:
        self._hist_L: deque[np.ndarray] = deque(maxlen=self.hist_window)
        self._hist_R: deque[np.ndarray] = deque(maxlen=self.hist_window)
        self._last_good_L: Optional[np.ndarray] = None
        self._last_good_R: Optional[np.ndarray] = None

    @staticmethod
    def _is_all_zero_pair(
        left: np.ndarray, right: np.ndarray, atol: float = 1e-9
    ) -> bool:
        return bool(np.allclose(left, 0.0, atol=atol) and np.allclose(right, 0.0, atol=atol))

    def stabilize(
        self, left_pos: np.ndarray, right_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        L = np.asarray(left_pos, dtype=np.float64).reshape(25, 3)
        R = np.asarray(right_pos, dtype=np.float64).reshape(25, 3)
        if self._is_all_zero_pair(L, R):
            return L, R

        wi = self.wrist_idx
        cL_in, cR_in = L[wi].copy(), R[wi].copy()

        if self._last_good_L is None:
            self._last_good_L = L.copy()
            self._last_good_R = R.copy()
            self._hist_L.append(cL_in.copy())
            self._hist_R.append(cR_in.copy())
            return L, R

        Lw, Rw = L.copy(), R.copy()

        if self.enable_hist:
            if len(self._hist_L) >= 2:
                med_l = np.median(np.stack(tuple(self._hist_L), axis=0), axis=0)
                if float(np.linalg.norm(cL_in - med_l)) > self.hist_jump_thresh_m:
                    Lw = self._last_good_L.copy()
            if len(self._hist_R) >= 2:
                med_r = np.median(np.stack(tuple(self._hist_R), axis=0), axis=0)
                if float(np.linalg.norm(cR_in - med_r)) > self.hist_jump_thresh_m:
                    Rw = self._last_good_R.copy()

        if self.enable_pred and len(self._hist_L) >= 2:
            pred_l = 2.0 * self._hist_L[-1] - self._hist_L[-2]
            if float(np.linalg.norm(Lw[wi] - pred_l)) > self.pred_dev_thresh_m:
                Lw = Lw + (pred_l - Lw[wi])
        if self.enable_pred and len(self._hist_R) >= 2:
            pred_r = 2.0 * self._hist_R[-1] - self._hist_R[-2]
            if float(np.linalg.norm(Rw[wi] - pred_r)) > self.pred_dev_thresh_m:
                Rw = Rw + (pred_r - Rw[wi])

        self._last_good_L = Lw.copy()
        self._last_good_R = Rw.copy()
        self._hist_L.append(Lw[wi].copy())
        self._hist_R.append(Rw[wi].copy())
        return Lw, Rw
