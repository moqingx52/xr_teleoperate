import numpy as np


def _safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    return float(max(np.linalg.norm(v), eps))


class HamerHandBridge:
    """Convert HaMeR keypoints_3d_local (21x3) to teleop hand points (25x3)."""

    def __init__(self, target_bone_len: float = 0.10):
        # OpenPose hand: 0 wrist, 9 middle_mcp
        self._middle_mcp_idx = 9
        self._target_bone_len = float(target_bone_len)
        self._left_scale_ema = 1.0
        self._right_scale_ema = 1.0

    def _normalize_wrist_local(self, kp21: np.ndarray, side: str) -> np.ndarray:
        kp = np.asarray(kp21, dtype=np.float64).reshape(21, 3).copy()
        kp = kp - kp[0:1, :]

        # GMH-D / MediaPipe 相机系: x右, y下, z前。
        # dex retargeting 侧需要左右手在“手局部关键点语义”上保持镜像一致；
        # 否则一侧手指张合向量会方向相反，导致几乎不动或张合反了。
        if side == "left":
            kp[:, 0] *= -1.0

        ref = _safe_norm(kp[self._middle_mcp_idx])
        scale_raw = self._target_bone_len / ref
        scale_raw = float(np.clip(scale_raw, 0.3, 3.0))

        if side == "left":
            self._left_scale_ema = 0.8 * self._left_scale_ema + 0.2 * scale_raw
            scale = self._left_scale_ema
        else:
            self._right_scale_ema = 0.8 * self._right_scale_ema + 0.2 * scale_raw
            scale = self._right_scale_ema

        return kp * scale

    @staticmethod
    def _openpose21_to_teleop25(kp21: np.ndarray) -> np.ndarray:
        # teleop 25-index layout uses wrist at 0 and tips at 4/9/14/19/24.
        out = np.zeros((25, 3), dtype=np.float64)
        out[0] = kp21[0]

        # thumb: OpenPose 1..4 -> teleop 1..4
        out[1:5] = kp21[1:5]

        # For other fingers, teleop block is 5 points but OpenPose has 4.
        # Duplicate MCP to occupy the first slot of each 5-point block.
        # index finger: OpenPose 5..8 -> teleop 6..9, teleop[5]=MCP duplicate
        out[5] = kp21[5]
        out[6:10] = kp21[5:9]
        # middle finger: OpenPose 9..12 -> teleop 11..14, teleop[10]=MCP duplicate
        out[10] = kp21[9]
        out[11:15] = kp21[9:13]
        # ring finger: OpenPose 13..16 -> teleop 16..19, teleop[15]=MCP duplicate
        out[15] = kp21[13]
        out[16:20] = kp21[13:17]
        # pinky finger: OpenPose 17..20 -> teleop 21..24, teleop[20]=MCP duplicate
        out[20] = kp21[17]
        out[21:25] = kp21[17:21]

        return out

    def step(self, hamer_frame):
        left25 = np.zeros((25, 3), dtype=np.float64)
        right25 = np.zeros((25, 3), dtype=np.float64)

        if hamer_frame is None:
            return left25, right25

        left = hamer_frame.get("left", {}) or {}
        right = hamer_frame.get("right", {}) or {}

        if left.get("valid", False):
            kp_l = np.asarray(left.get("keypoints_3d_local", []), dtype=np.float64).reshape(-1, 3)
            if kp_l.shape[0] >= 21:
                kp_l = self._normalize_wrist_local(kp_l[:21], side="left")
                left25 = self._openpose21_to_teleop25(kp_l)

        if right.get("valid", False):
            kp_r = np.asarray(right.get("keypoints_3d_local", []), dtype=np.float64).reshape(-1, 3)
            if kp_r.shape[0] >= 21:
                kp_r = self._normalize_wrist_local(kp_r[:21], side="right")
                right25 = self._openpose21_to_teleop25(kp_r)

        return left25, right25
