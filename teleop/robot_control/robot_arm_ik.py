import casadi                                                                       
from dataclasses import dataclass
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys
import pickle
import logging_mp
logger_mp = logging_mp.getLogger(__name__)
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
assets_dir = os.path.join(parent2_dir, "assets")

from teleop.utils.weighted_moving_filter import WeightedMovingFilter


def homogeneous_from_position_rotation(p_ee_target, R_ee_target):
    """由位置与旋转矩阵构造 4x4 齐次变换（与 solve_ik 输入一致）。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R_ee_target, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(p_ee_target, dtype=np.float64).reshape(3)
    return T


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    position_error: float
    rotation_error: float
    iterations: int
    mode: str


class _SingleArmPrecisionIK:
    """Task-priority Pinocchio DLS IK over one 7-DoF arm inside a dual-arm model."""

    def __init__(
        self,
        model: pin.Model,
        frame_id: int,
        active_joint_names: list,
        side: str,
    ):
        self.model = model
        self.data = model.createData()
        self.frame_id = frame_id
        self.side = side
        self.active_joint_names = list(active_joint_names)
        self.active_q_indices = np.array(
            [int(model.idx_qs[model.getJointId(name)]) for name in self.active_joint_names],
            dtype=np.int64,
        )
        self.q_lb = np.asarray(model.lowerPositionLimit, dtype=np.float64)[self.active_q_indices]
        self.q_ub = np.asarray(model.upperPositionLimit, dtype=np.float64)[self.active_q_indices]
        self.q_mid = 0.5 * (self.q_lb + self.q_ub)
        self.q_half_range = np.maximum(0.5 * (self.q_ub - self.q_lb), 1e-6)

        self.position_tolerance = 1e-3
        self.rotation_tolerance = np.deg2rad(0.5)
        self.sigma_position = 1e-3
        self.sigma_rotation = np.deg2rad(0.5)
        self.W_task = np.diag(
            [1.0 / self.sigma_position] * 3 + [1.0 / self.sigma_rotation] * 3
        )
        self.W_joint = np.eye(len(self.active_q_indices))
        self.max_iterations = 100
        self.max_iteration_step = 0.20
        self.rank_tolerance = 1e-6
        self.sigma_safe = 0.05
        self.k_continuity = 0.02
        self.k_joint_limit = 0.01
        self.min_line_search_alpha = 1e-3
        self.previous_valid_q = np.clip(self.q_mid.copy(), self.q_lb, self.q_ub)

    def q_from_full(self, q_full: np.ndarray) -> np.ndarray:
        return np.asarray(q_full, dtype=np.float64)[self.active_q_indices].copy()

    def put_q(self, q_full: np.ndarray, q_arm: np.ndarray) -> np.ndarray:
        out = np.asarray(q_full, dtype=np.float64).copy()
        out[self.active_q_indices] = np.asarray(q_arm, dtype=np.float64)
        return out

    def _compute_pose(self, q_full: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id]

    def _pose_errors(self, current: pin.SE3, target: pin.SE3):
        position_error = float(np.linalg.norm(current.translation - target.translation))
        rotation_error = float(np.linalg.norm(pin.log3(current.rotation.T @ target.rotation)))
        return position_error, rotation_error

    def _normalized_error_norm(self, q_full: np.ndarray, target: pin.SE3) -> float:
        current = self._compute_pose(q_full)
        error6 = pin.log6(current.actInv(target)).vector
        return float(np.linalg.norm(self.W_task @ error6))

    def _compute_damping(self, singular_values: np.ndarray) -> float:
        sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
        if sigma_min > self.sigma_safe:
            return 1e-4
        ratio = np.clip(sigma_min / self.sigma_safe, 0.0, 1.0)
        return float(1e-4 + (1.0 - ratio) ** 2 * 0.1)

    def _joint_limit_gradient(self, q: np.ndarray) -> np.ndarray:
        normalized = (q - self.q_mid) / self.q_half_range
        return normalized / self.q_half_range

    def _backtracking_update(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_full_template: np.ndarray,
        target: pin.SE3,
        current_score: float,
    ):
        alpha = 1.0
        while alpha >= self.min_line_search_alpha:
            candidate = np.clip(q + alpha * dq, self.q_lb, self.q_ub)
            score = self._normalized_error_norm(self.put_q(q_full_template, candidate), target)
            if np.isfinite(score) and score < current_score:
                return candidate
            alpha *= 0.5
        return None

    def make_failure_result(self, q: np.ndarray, q_full_template: np.ndarray, target: pin.SE3, iterations: int) -> IKResult:
        current = self._compute_pose(self.put_q(q_full_template, q))
        position_error, rotation_error = self._pose_errors(current, target)
        return IKResult(False, q, position_error, rotation_error, iterations, "local_failed")

    def solve_local(self, target: pin.SE3, q_full_seed: np.ndarray) -> IKResult:
        q = np.clip(self.q_from_full(q_full_seed), self.q_lb, self.q_ub)
        # Anchor null-space continuity to the measured seed, not the joint mid-range.
        q_anchor = self.q_from_full(q_full_seed)
        q_full_template = np.asarray(q_full_seed, dtype=np.float64).copy()

        for iteration in range(self.max_iterations):
            q_full = self.put_q(q_full_template, q)
            current = self._compute_pose(q_full)
            current_to_target = current.actInv(target)
            error6 = pin.log6(current_to_target).vector
            position_error, rotation_error = self._pose_errors(current, target)
            if position_error < self.position_tolerance and rotation_error < self.rotation_tolerance:
                self.previous_valid_q = q.copy()
                return IKResult(True, q, position_error, rotation_error, iteration, "local")

            J_frame = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.frame_id,
                pin.LOCAL,
            )[:, self.active_q_indices]
            J_error = -pin.Jlog6(current_to_target.inverse()) @ J_frame
            error_normalized = self.W_task @ error6
            J_normalized = self.W_task @ J_error

            try:
                U, singular_values, Vt = np.linalg.svd(J_normalized, full_matrices=True)
            except np.linalg.LinAlgError:
                break
            V = Vt.T
            damping = self._compute_damping(singular_values)
            J_damped_inverse = (
                V[:, : singular_values.size]
                @ np.diag(singular_values / (singular_values**2 + damping**2))
                @ U.T
            )
            dq_task = -J_damped_inverse @ error_normalized

            rank = int(np.count_nonzero(singular_values > self.rank_tolerance))
            Z = V[:, rank:]
            N = Z @ Z.T if Z.size else np.zeros((q.size, q.size), dtype=np.float64)
            grad_continuity = self.W_joint @ (q - q_anchor)
            grad_limit = self._joint_limit_gradient(q)
            dq_secondary = -N @ (self.k_continuity * grad_continuity + self.k_joint_limit * grad_limit)
            dq = np.clip(dq_task + dq_secondary, -self.max_iteration_step, self.max_iteration_step)

            current_score = float(np.linalg.norm(error_normalized))
            accepted = self._backtracking_update(q, dq, q_full_template, target, current_score)
            if accepted is None and np.linalg.norm(dq_secondary) > 0.0:
                dq_task_clipped = np.clip(dq_task, -self.max_iteration_step, self.max_iteration_step)
                accepted = self._backtracking_update(q, dq_task_clipped, q_full_template, target, current_score)
            if accepted is None:
                dq_gradient = -(J_normalized.T @ error_normalized)
                dq_gradient_norm = float(np.linalg.norm(dq_gradient, ord=np.inf))
                if dq_gradient_norm > 1e-12:
                    dq_gradient = dq_gradient / dq_gradient_norm * self.max_iteration_step
                    accepted = self._backtracking_update(q, dq_gradient, q_full_template, target, current_score)
            if accepted is None:
                break
            q = accepted

        return self.make_failure_result(q, q_full_template, target, iteration + 1)


class G1_29_ArmIK:
    A2D_ACTIVE_ARM_JOINT_NAMES = [
        "Joint1_l", "Joint2_l", "Joint3_l", "Joint4_l", "Joint5_l", "Joint6_l", "Joint7_l",
        "Joint1_r", "Joint2_r", "Joint3_r", "Joint4_r", "Joint5_r", "Joint6_r", "Joint7_r",
    ]

    A2D_SIM_TO_URDF_LOCK_JOINT_NAMES = {
        "idx01_body_joint1": "joint_lift_body",
        "idx02_body_joint2": "joint_body_pitch",
        # These are fixed joints in the current A2D URDF, but keep the mapping so
        # future movable head joints can be anchored without another code path.
        "idx11_head_joint1": "joint_head_yaw",
        "idx12_head_joint2": "joint_head_pitch",
    }

    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # fixed cache file path
        self.cache_path = "g1_29_model_cache.pkl"

        default_urdf_path = os.path.join(assets_dir, "g1", "g1_body29_hand14.urdf")
        default_model_dir = os.path.join(assets_dir, "g1")
        a2d_urdf_path = os.path.join(assets_dir, "A2D_Omnipicker", "A2D.urdf")
        a2d_model_dir = os.path.join(assets_dir, "A2D_Omnipicker")

        self._use_a2d_omnipicker_urdf = os.path.exists(a2d_urdf_path)
        if self._use_a2d_omnipicker_urdf:
            self.urdf_path = a2d_urdf_path
            self.model_dir = a2d_model_dir
            self.cache_path = "g1_29_a2d_omnipicker_model_cache.pkl"
            logger_mp.info(f"[G1_29_ArmIK] use omnipicker URDF: {self.urdf_path}")
        else:
            self.urdf_path = default_urdf_path
            self.model_dir = default_model_dir
            logger_mp.warning(
                f"[G1_29_ArmIK] omnipicker URDF not found, fallback to default: {self.urdf_path}"
            )

        # Try loading cache first
        if os.path.exists(self.cache_path) and (not self.Visualization):
            logger_mp.info(f"[G1_29_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            try:
                self.robot, self.reduced_robot = self.load_cache()
                self.mixed_jointsToLockIDs = self._make_locked_joint_names()
                self._setup_solver_from_reduced_model()
            except Exception as exc:
                logger_mp.warning(
                    f"[G1_29_ArmIK] failed to load cache {self.cache_path}: {type(exc).__name__}: {exc}; rebuilding."
                )
                try:
                    os.remove(self.cache_path)
                except OSError:
                    pass
                self._load_urdf_and_build_reduced_model()
        else:
            self._load_urdf_and_build_reduced_model()

    def _load_urdf_and_build_reduced_model(self):
        logger_mp.info("[G1_29_ArmIK] >>> Loading URDF (slow)...")
        self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

        self.mixed_jointsToLockIDs = self._make_locked_joint_names()
        self._locked_reference_configuration = np.array([0.0] * self.robot.model.nq)
        self.reduced_robot = self._build_reduced_robot(self._locked_reference_configuration)
        self._ensure_reduced_model_ee_frames()

        # Save cache (only after everything is built)
        if not os.path.exists(self.cache_path):
            self.save_cache()
            logger_mp.info(f">>> Cache saved to {self.cache_path}")

        self._setup_solver_from_reduced_model()

    def _make_locked_joint_names(self):
        if self._use_a2d_omnipicker_urdf:
            model_joint_names = set(self.robot.model.names)
            missing = [n for n in self.A2D_ACTIVE_ARM_JOINT_NAMES if n not in model_joint_names]
            if missing:
                raise ValueError(
                    f"[G1_29_ArmIK] omnipicker active arm joints missing in URDF: {missing}"
                )
            return [
                n for n in self.robot.model.names
                if n not in self.A2D_ACTIVE_ARM_JOINT_NAMES and n != "universe"
            ]
        return [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

    def _ensure_reduced_model_ee_frames(self):
        if self._use_a2d_omnipicker_urdf:
            frame_names = [f.name for f in self.reduced_robot.model.frames]
            if "left_base_link" in frame_names and "right_base_link" in frame_names:
                return
            if "L_ee" not in frame_names:
                self.reduced_robot.model.addFrame(
                    pin.Frame(
                        "L_ee",
                        self.reduced_robot.model.getJointId("Joint7_l"),
                        pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0]).T),
                        pin.FrameType.OP_FRAME,
                    )
                )
            if "R_ee" not in frame_names:
                self.reduced_robot.model.addFrame(
                    pin.Frame(
                        "R_ee",
                        self.reduced_robot.model.getJointId("Joint7_r"),
                        pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0]).T),
                        pin.FrameType.OP_FRAME,
                    )
                )
            return
        frame_names = [f.name for f in self.reduced_robot.model.frames]
        if "L_ee" not in frame_names:
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "L_ee",
                    self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
        if "R_ee" not in frame_names:
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "R_ee",
                    self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )

    def _build_reduced_robot(self, reference_configuration):
        reference = np.asarray(reference_configuration, dtype=np.float64).reshape(-1)
        joint_ids = [
            int(self.robot.model.getJointId(name))
            for name in self.mixed_jointsToLockIDs
            if name in self.robot.model.names
        ]
        reduced_model = pin.buildReducedModel(self.robot.model, joint_ids, reference)
        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = reduced_model
        reduced_robot.data = reduced_model.createData()
        return reduced_robot

    def _reference_with_named_joint_positions(self, named_joint_positions):
        reference = np.array([0.0] * self.robot.model.nq, dtype=np.float64)
        if not getattr(self, "_use_a2d_omnipicker_urdf", False):
            return reference, {}
        if not isinstance(named_joint_positions, dict):
            return reference, {}

        applied = {}
        model_joint_names = set(self.robot.model.names)
        lower = np.asarray(self.robot.model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(self.robot.model.upperPositionLimit, dtype=np.float64)
        for sim_name, urdf_name in self.A2D_SIM_TO_URDF_LOCK_JOINT_NAMES.items():
            if sim_name not in named_joint_positions or urdf_name not in model_joint_names:
                continue
            joint_id = self.robot.model.getJointId(urdf_name)
            idx_q = int(self.robot.model.idx_qs[joint_id])
            nq = int(self.robot.model.nqs[joint_id])
            if nq != 1:
                continue
            raw_value = float(named_joint_positions[sim_name])
            value = float(np.clip(raw_value, lower[idx_q], upper[idx_q]))
            reference[idx_q] = value
            applied[urdf_name] = value
        return reference, applied

    def align_locked_joints_from_sim_state(self, named_joint_positions):
        """Rebuild the reduced IK model with simulated body joints locked at their current pose."""
        if not getattr(self, "_use_a2d_omnipicker_urdf", False):
            return False
        reference, applied = self._reference_with_named_joint_positions(named_joint_positions)
        if not applied:
            logger_mp.warning(
                "[G1_29_ArmIK] no sim body joint positions available for A2D lock reference; "
                "using zero locked joints."
            )
            return False

        self._locked_reference_configuration = reference
        self.reduced_robot = self._build_reduced_robot(reference)
        self._ensure_reduced_model_ee_frames()
        self._setup_solver_from_reduced_model()
        logger_mp.info(
            "[G1_29_ArmIK] aligned A2D locked joints from sim state: %s",
            ", ".join(f"{name}={value:.4f}" for name, value in sorted(applied.items())),
        )
        return True

    def _setup_solver_from_reduced_model(self):
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        if self._use_a2d_omnipicker_urdf:
            frame_names = [f.name for f in self.reduced_robot.model.frames]
            if "left_base_link" in frame_names:
                self.L_hand_id = self.reduced_robot.model.getFrameId("left_base_link")
                self.R_hand_id = self.reduced_robot.model.getFrameId("right_base_link")
            else:
                self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
                self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        else:
            self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
            self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self._setup_precision_ik()

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )
        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[self.L_hand_id, self.R_hand_id], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def _setup_precision_ik(self):
        self.enable_precision_ik = bool(getattr(self, "enable_precision_ik", True))
        self._precision_ik_ready = False
        self.left_precision_ik = None
        self.right_precision_ik = None
        if not getattr(self, "_use_a2d_omnipicker_urdf", False):
            return
        left_joint_names = [
            "Joint1_l", "Joint2_l", "Joint3_l", "Joint4_l", "Joint5_l", "Joint6_l", "Joint7_l",
        ]
        right_joint_names = [
            "Joint1_r", "Joint2_r", "Joint3_r", "Joint4_r", "Joint5_r", "Joint6_r", "Joint7_r",
        ]
        try:
            self.left_precision_ik = _SingleArmPrecisionIK(
                self.reduced_robot.model,
                self.L_hand_id,
                left_joint_names,
                "left",
            )
            self.right_precision_ik = _SingleArmPrecisionIK(
                self.reduced_robot.model,
                self.R_hand_id,
                right_joint_names,
                "right",
            )
            self._precision_ik_ready = True
            self.enable_joint_smoothing = False
            self.precision_single_arm_position_deadband = 2e-3
            self.precision_single_arm_rotation_deadband = np.deg2rad(0.5)
            self.precision_dual_arm_position_threshold = 3e-2
            self.precision_dual_arm_rotation_threshold = np.deg2rad(5.0)
            self._last_precision_left_target = None
            self._last_precision_right_target = None
            logger_mp.info("[G1_29_ArmIK] precision single-arm Pinocchio IK enabled.")
        except Exception as exc:
            logger_mp.warning(
                f"[G1_29_ArmIK] precision IK unavailable, using IPOPT fallback: {type(exc).__name__}: {exc}"
            )

    # Save both robot.model and reduced_robot.model
    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    # Load both robot.model and reduced_robot.model
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_from_ee_pose(
        self,
        side: str,
        p_ee_target: np.ndarray,
        R_ee_target: np.ndarray,
        current_lr_arm_motor_q,
        current_lr_arm_motor_dq,
        other_wrist_4x4: np.ndarray,
    ):
        """单侧末端位姿 + 对侧齐次矩阵，保持外部接口兼容。"""
        T = homogeneous_from_position_rotation(p_ee_target, R_ee_target)
        if str(side).lower() == "left":
            return self.solve_ik(T, other_wrist_4x4, current_lr_arm_motor_q, current_lr_arm_motor_dq)
        return self.solve_ik(other_wrist_4x4, T, current_lr_arm_motor_q, current_lr_arm_motor_dq)

    def _target_from_matrix(self, wrist: np.ndarray) -> pin.SE3:
        wrist = np.asarray(wrist, dtype=np.float64)
        return pin.SE3(wrist[:3, :3], wrist[:3, 3])

    def _current_frame_pose(self, q: np.ndarray, frame_id: int) -> pin.SE3:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        return self.reduced_robot.data.oMf[frame_id].copy()

    def _target_delta(self, previous: pin.SE3, current: pin.SE3):
        position_delta = float(np.linalg.norm(current.translation - previous.translation))
        rotation_delta = float(np.linalg.norm(pin.log3(previous.rotation.T @ current.rotation)))
        return position_delta, rotation_delta

    def _target_changed(self, position_delta: float, rotation_delta: float) -> bool:
        return (
            position_delta > self.precision_single_arm_position_deadband
            or rotation_delta > self.precision_single_arm_rotation_deadband
        )

    def _target_large_motion(self, position_delta: float, rotation_delta: float) -> bool:
        return (
            position_delta > self.precision_dual_arm_position_threshold
            or rotation_delta > self.precision_dual_arm_rotation_threshold
        )

    def _precision_route(self, seed_q: np.ndarray, left_target: pin.SE3, right_target: pin.SE3):
        if self._last_precision_left_target is None:
            previous_left = self._current_frame_pose(seed_q, self.L_hand_id)
        else:
            previous_left = self._last_precision_left_target
        if self._last_precision_right_target is None:
            previous_right = self._current_frame_pose(seed_q, self.R_hand_id)
        else:
            previous_right = self._last_precision_right_target

        left_pos_delta, left_rot_delta = self._target_delta(previous_left, left_target)
        right_pos_delta, right_rot_delta = self._target_delta(previous_right, right_target)
        left_changed = self._target_changed(left_pos_delta, left_rot_delta)
        right_changed = self._target_changed(right_pos_delta, right_rot_delta)
        left_large = self._target_large_motion(left_pos_delta, left_rot_delta)
        right_large = self._target_large_motion(right_pos_delta, right_rot_delta)

        if left_large and right_large:
            return "both", (left_pos_delta, left_rot_delta), (right_pos_delta, right_rot_delta)
        if left_changed and not right_changed:
            return "left", (left_pos_delta, left_rot_delta), (right_pos_delta, right_rot_delta)
        if right_changed and not left_changed:
            return "right", (left_pos_delta, left_rot_delta), (right_pos_delta, right_rot_delta)
        if left_changed and right_changed:
            left_score = max(
                left_pos_delta / self.precision_dual_arm_position_threshold,
                left_rot_delta / self.precision_dual_arm_rotation_threshold,
            )
            right_score = max(
                right_pos_delta / self.precision_dual_arm_position_threshold,
                right_rot_delta / self.precision_dual_arm_rotation_threshold,
            )
            return ("left" if left_score >= right_score else "right"), (
                left_pos_delta,
                left_rot_delta,
            ), (right_pos_delta, right_rot_delta)
        return "hold", (left_pos_delta, left_rot_delta), (right_pos_delta, right_rot_delta)

    def _solve_ik_precision(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            seed_q = np.asarray(current_lr_arm_motor_q, dtype=np.float64).copy()
        else:
            seed_q = np.asarray(self.init_data, dtype=np.float64).copy()

        left_target = self._target_from_matrix(left_wrist)
        right_target = self._target_from_matrix(right_wrist)
        route, left_delta, right_delta = self._precision_route(seed_q, left_target, right_target)
        forced_route = getattr(self, "teleop_forced_route", None)
        if forced_route in ("left", "right", "both"):
            route = forced_route
        left_result = IKResult(True, self.left_precision_ik.q_from_full(seed_q), 0.0, 0.0, 0, "held")
        right_result = IKResult(True, self.right_precision_ik.q_from_full(seed_q), 0.0, 0.0, 0, "held")
        sol_q = seed_q.copy()

        if route in ("left", "both"):
            left_result = self.left_precision_ik.solve_local(left_target, sol_q)
            sol_q = self.left_precision_ik.put_q(sol_q, left_result.q)
        if route in ("right", "both"):
            right_result = self.right_precision_ik.solve_local(right_target, sol_q)
            sol_q = self.right_precision_ik.put_q(sol_q, right_result.q)

        if (
            route in ("left", "right")
            and not (left_result.success and right_result.success)
            and bool(getattr(self, "disable_single_arm_ipopt_fallback", False))
        ):
            logger_mp.warning(
                "[G1_29_ArmIK] precision single-arm IK did not meet strict tolerance; "
                "using local single-arm solution without IPOPT fallback. "
                f"route={route} "
                f"left_delta=(p={left_delta[0]:.6g}, r={left_delta[1]:.6g}) "
                f"right_delta=(p={right_delta[0]:.6g}, r={right_delta[1]:.6g}) "
                f"left=({left_result.mode}, p={left_result.position_error:.6g}, "
                f"r={left_result.rotation_error:.6g}, it={left_result.iterations}) "
                f"right=({right_result.mode}, p={right_result.position_error:.6g}, "
                f"r={right_result.rotation_error:.6g}, it={right_result.iterations})"
            )
        elif route != "hold" and not (left_result.success and right_result.success):
            logger_mp.warning(
                "[G1_29_ArmIK] precision IK failed, falling back to IPOPT. "
                f"route={route} "
                f"left_delta=(p={left_delta[0]:.6g}, r={left_delta[1]:.6g}) "
                f"right_delta=(p={right_delta[0]:.6g}, r={right_delta[1]:.6g}) "
                f"left=({left_result.mode}, p={left_result.position_error:.6g}, "
                f"r={left_result.rotation_error:.6g}, it={left_result.iterations}) "
                f"right=({right_result.mode}, p={right_result.position_error:.6g}, "
                f"r={right_result.rotation_error:.6g}, it={right_result.iterations})"
            )
            return self._solve_ik_ipopt(left_wrist, right_wrist, current_lr_arm_motor_q, current_lr_arm_motor_dq)

        sol_q = np.clip(
            sol_q,
            np.asarray(self.reduced_robot.model.lowerPositionLimit, dtype=np.float64),
            np.asarray(self.reduced_robot.model.upperPositionLimit, dtype=np.float64),
        )
        if route in ("left", "both"):
            self._last_precision_left_target = left_target.copy()
        if route in ("right", "both"):
            self._last_precision_right_target = right_target.copy()
        if route == "hold":
            if self._last_precision_left_target is None:
                self._last_precision_left_target = left_target.copy()
            if self._last_precision_right_target is None:
                self._last_precision_right_target = right_target.copy()
        if current_lr_arm_motor_dq is not None:
            v = np.asarray(current_lr_arm_motor_dq, dtype=np.float64) * 0.0
        else:
            v = (sol_q - seed_q) * 0.0
        self.init_data = sol_q.copy()
        sol_tauff = pin.rnea(
            self.reduced_robot.model,
            self.reduced_robot.data,
            sol_q,
            v,
            np.zeros(self.reduced_robot.model.nv),
        )
        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)
            self.vis.display(sol_q)
        return sol_q, sol_tauff

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if (
            getattr(self, "enable_precision_ik", True)
            and getattr(self, "_precision_ik_ready", False)
        ):
            return self._solve_ik_precision(
                left_wrist,
                right_wrist,
                current_lr_arm_motor_q,
                current_lr_arm_motor_dq,
            )
        return self._solve_ik_ipopt(left_wrist, right_wrist, current_lr_arm_motor_q, current_lr_arm_motor_dq)

    def _solve_ik_ipopt(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
        
class G1_23_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # fixed cache file path
        self.cache_path = "g1_23_model_cache.pkl"

        self.urdf_path = os.path.join(assets_dir, "g1", "g1_body23.urdf")
        self.model_dir = os.path.join(assets_dir, "g1")

        # Try loading cache first
        if os.path.exists(self.cache_path) and (not self.Visualization):
            logger_mp.info(f"[G1_23_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[G1_23_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            self.mixed_jointsToLockIDs = [
                                            "left_hip_pitch_joint" ,
                                            "left_hip_roll_joint" ,
                                            "left_hip_yaw_joint" ,
                                            "left_knee_joint" ,
                                            "left_ankle_pitch_joint" ,
                                            "left_ankle_roll_joint" ,
                                            "right_hip_pitch_joint" ,
                                            "right_hip_roll_joint" ,
                                            "right_hip_yaw_joint" ,
                                            "right_knee_joint" ,
                                            "right_ankle_pitch_joint" ,
                                            "right_ankle_roll_joint" ,
                                            "waist_yaw_joint" ,
                                        ]

            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            self.reduced_robot.model.addFrame(
                pin.Frame('L_ee',
                        self.reduced_robot.model.getJointId('left_wrist_roll_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0.20,0,0]).T),
                        pin.FrameType.OP_FRAME)
            )
            
            self.reduced_robot.model.addFrame(
                pin.Frame('R_ee',
                        self.reduced_robot.model.getJointId('right_wrist_roll_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0.20,0,0]).T),
                        pin.FrameType.OP_FRAME)
            )

            # Save cache (only after everything is built)
            if not os.path.exists(self.cache_path):
                self.save_cache()
                logger_mp.info(f">>> Cache saved to {self.cache_path}")

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     logger_mp.debug(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + 0.5 * self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            # CasADi-level options
            'expand': True, 
            'detect_simple_bounds': True,
            'calc_lam_p': False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
            'print_time':False,   # print or not
            # IPOPT solver options
            'ipopt.sb': 'yes',    # disable Ipopt's license message
            'ipopt.print_level': 0,
            'ipopt.max_iter': 30, 
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none',
            'ipopt.jacobian_approximation': 'exact',
            # 'ipopt.hessian_approximation': 'limited-memory',
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 10)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[67, 68], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    # Save both robot.model and reduced_robot.model
    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    # Load both robot.model and reduced_robot.model
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_from_ee_pose(
        self,
        side: str,
        p_ee_target: np.ndarray,
        R_ee_target: np.ndarray,
        current_lr_arm_motor_q,
        current_lr_arm_motor_dq,
        other_wrist_4x4: np.ndarray,
    ):
        T = homogeneous_from_position_rotation(p_ee_target, R_ee_target)
        if str(side).lower() == "left":
            return self.solve_ik(T, other_wrist_4x4, current_lr_arm_motor_q, current_lr_arm_motor_dq)
        return self.solve_ik(other_wrist_4x4, T, current_lr_arm_motor_q, current_lr_arm_motor_dq)

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)


class H1_2_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # fixed cache file path
        self.cache_path = "h1_2_model_cache.pkl"

        self.urdf_path = os.path.join(assets_dir, "h1_2", "h1_2.urdf")
        self.model_dir = os.path.join(assets_dir, "h1_2")

        # Try loading cache first
        if os.path.exists(self.cache_path) and (not self.Visualization):
            logger_mp.info(f"[H1_2_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[H1_2_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            self.mixed_jointsToLockIDs = [
                                        "left_hip_yaw_joint",
                                        "left_hip_pitch_joint",
                                        "left_hip_roll_joint",
                                        "left_knee_joint",
                                        "left_ankle_pitch_joint",
                                        "left_ankle_roll_joint",
                                        "right_hip_yaw_joint",
                                        "right_hip_pitch_joint",
                                        "right_hip_roll_joint",
                                        "right_knee_joint",
                                        "right_ankle_pitch_joint",
                                        "right_ankle_roll_joint",
                                        "torso_joint",
                                        "L_index_proximal_joint",
                                        "L_index_intermediate_joint",
                                        "L_middle_proximal_joint",
                                        "L_middle_intermediate_joint",
                                        "L_pinky_proximal_joint",
                                        "L_pinky_intermediate_joint",
                                        "L_ring_proximal_joint",
                                        "L_ring_intermediate_joint",
                                        "L_thumb_proximal_yaw_joint",
                                        "L_thumb_proximal_pitch_joint",
                                        "L_thumb_intermediate_joint",
                                        "L_thumb_distal_joint",
                                        "R_index_proximal_joint",
                                        "R_index_intermediate_joint",
                                        "R_middle_proximal_joint",
                                        "R_middle_intermediate_joint",
                                        "R_pinky_proximal_joint",
                                        "R_pinky_intermediate_joint",
                                        "R_ring_proximal_joint",
                                        "R_ring_intermediate_joint",
                                        "R_thumb_proximal_yaw_joint",
                                        "R_thumb_proximal_pitch_joint",
                                        "R_thumb_intermediate_joint",
                                        "R_thumb_distal_joint"
                                        ]

            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            self.reduced_robot.model.addFrame(
                pin.Frame('L_ee',
                        self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0.05,0,0]).T),
                        pin.FrameType.OP_FRAME)
            )
            
            self.reduced_robot.model.addFrame(
                pin.Frame('R_ee',
                        self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0.05,0,0]).T),
                        pin.FrameType.OP_FRAME)
            )

            # Save cache (only after everything is built)
            if not os.path.exists(self.cache_path):
                self.save_cache()
                logger_mp.info(f">>> Cache saved to {self.cache_path}")

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     logger_mp.debug(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            # CasADi-level options
            'expand': True, 
            'detect_simple_bounds': True,
            'calc_lam_p': False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
            'print_time':False,   # print or not
            # IPOPT solver options
            'ipopt.sb': 'yes',    # disable Ipopt's license message
            'ipopt.print_level': 0,
            'ipopt.max_iter': 30, 
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none',
            'ipopt.jacobian_approximation': 'exact',
            # 'ipopt.hessian_approximation': 'limited-memory',
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[113, 114], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 10
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )
    
    # Save both robot.model and reduced_robot.model
    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    # Load both robot.model and reduced_robot.model
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_from_ee_pose(
        self,
        side: str,
        p_ee_target: np.ndarray,
        R_ee_target: np.ndarray,
        current_lr_arm_motor_q,
        current_lr_arm_motor_dq,
        other_wrist_4x4: np.ndarray,
    ):
        T = homogeneous_from_position_rotation(p_ee_target, R_ee_target)
        if str(side).lower() == "left":
            return self.solve_ik(T, other_wrist_4x4, current_lr_arm_motor_q, current_lr_arm_motor_dq)
        return self.solve_ik(other_wrist_4x4, T, current_lr_arm_motor_q, current_lr_arm_motor_dq)

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

class H1_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # fixed cache file path
        self.cache_path = "h1_model_cache.pkl"

        self.urdf_path = os.path.join(assets_dir, "h1", "h1_with_hand.urdf")
        self.model_dir = os.path.join(assets_dir, "h1")

        # Try loading cache first
        if os.path.exists(self.cache_path) and (not self.Visualization):
            logger_mp.info(f"[H1_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[H1_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            self.mixed_jointsToLockIDs = [
                                            "right_hip_roll_joint",
                                            "right_hip_pitch_joint",
                                            "right_knee_joint",
                                            "left_hip_roll_joint",
                                            "left_hip_pitch_joint",
                                            "left_knee_joint",
                                            "torso_joint",
                                            "left_hip_yaw_joint",
                                            "right_hip_yaw_joint",

                                            "left_ankle_joint",
                                            "right_ankle_joint",

                                            "L_index_proximal_joint",
                                            "L_index_intermediate_joint",
                                            "L_middle_proximal_joint",
                                            "L_middle_intermediate_joint",
                                            "L_ring_proximal_joint",
                                            "L_ring_intermediate_joint",
                                            "L_pinky_proximal_joint",
                                            "L_pinky_intermediate_joint",
                                            "L_thumb_proximal_yaw_joint",
                                            "L_thumb_proximal_pitch_joint",
                                            "L_thumb_intermediate_joint",
                                            "L_thumb_distal_joint",
                                            
                                            "R_index_proximal_joint",
                                            "R_index_intermediate_joint",
                                            "R_middle_proximal_joint",
                                            "R_middle_intermediate_joint",
                                            "R_ring_proximal_joint",
                                            "R_ring_intermediate_joint",
                                            "R_pinky_proximal_joint",
                                            "R_pinky_intermediate_joint",
                                            "R_thumb_proximal_yaw_joint",
                                            "R_thumb_proximal_pitch_joint",
                                            "R_thumb_intermediate_joint",
                                            "R_thumb_distal_joint",

                                            "left_hand_joint",
                                            "right_hand_joint"  
                                        ]

            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            self.reduced_robot.model.addFrame(
                pin.Frame('L_ee',
                        self.reduced_robot.model.getJointId('left_elbow_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0.2605 + 0.05,0,0]).T),
                        pin.FrameType.OP_FRAME)
            )
            
            self.reduced_robot.model.addFrame(
                pin.Frame('R_ee',
                        self.reduced_robot.model.getJointId('right_elbow_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0.2605 + 0.05,0,0]).T),
                        pin.FrameType.OP_FRAME)
            )

            # Save cache (only after everything is built)
            if not os.path.exists(self.cache_path):
                self.save_cache()
                logger_mp.info(f">>> Cache saved to {self.cache_path}")

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     logger_mp.debug(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + 0.5 * self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            # CasADi-level options
            'expand': True, 
            'detect_simple_bounds': True,
            'calc_lam_p': False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
            'print_time':False,   # print or not
            # IPOPT solver options
            'ipopt.sb': 'yes',    # disable Ipopt's license message
            'ipopt.print_level': 0,
            'ipopt.max_iter': 30, 
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none',
            'ipopt.jacobian_approximation': 'exact',
            # 'ipopt.hessian_approximation': 'limited-memory',
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 8)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[105, 106], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1.0, 0.3, 0.3], [1.0, 0.7, 0.7],
                          [0.3, 1.0, 0.5], [0.7, 1.0, 0.8],
                          [0.3, 0.8, 1.0], [0.7, 0.9, 1.0]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 10
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    # Save both robot.model and reduced_robot.model
    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    # Load both robot.model and reduced_robot.model
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_from_ee_pose(
        self,
        side: str,
        p_ee_target: np.ndarray,
        R_ee_target: np.ndarray,
        current_lr_arm_motor_q,
        current_lr_arm_motor_dq,
        other_wrist_4x4: np.ndarray,
    ):
        T = homogeneous_from_position_rotation(p_ee_target, R_ee_target)
        if str(side).lower() == "left":
            return self.solve_ik(T, other_wrist_4x4, current_lr_arm_motor_q, current_lr_arm_motor_dq)
        return self.solve_ik(other_wrist_4x4, T, current_lr_arm_motor_q, current_lr_arm_motor_dq)

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            if getattr(self, "enable_joint_smoothing", True):
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
        
if __name__ == "__main__":
    arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = True)
    # arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = True)
    # arm_ik = G1_23_ArmIK(Unit_Test = True, Visualization = True)
    # arm_ik = H1_ArmIK(Unit_Test = True, Visualization = True)

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005
    noise_amplitude_translation = 0.001
    noise_amplitude_rotation = 0.01

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        step = 0
        while True:
            # Apply rotation noise with bias towards y and z axes
            rotation_noise_L = pin.Quaternion(
                np.cos(np.random.normal(0, noise_amplitude_rotation) / 2),0,np.random.normal(0, noise_amplitude_rotation / 2),0).normalized()  # y bias

            rotation_noise_R = pin.Quaternion(
                np.cos(np.random.normal(0, noise_amplitude_rotation) / 2),0,0,np.random.normal(0, noise_amplitude_rotation / 2)).normalized()  # z bias
            
            if step <= 120:
                angle = rotation_speed * step
                L_tf_target.rotation = (rotation_noise_L * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)).toRotationMatrix()  # y axis
                R_tf_target.rotation = (rotation_noise_R * pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))).toRotationMatrix()  # z axis
                L_tf_target.translation += (np.array([0.001,  0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                R_tf_target.translation += (np.array([0.001, -0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
            else:
                angle = rotation_speed * (240 - step)
                L_tf_target.rotation = (rotation_noise_L * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)).toRotationMatrix()  # y axis
                R_tf_target.rotation = (rotation_noise_R * pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))).toRotationMatrix()  # z axis
                L_tf_target.translation -= (np.array([0.001,  0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                R_tf_target.translation -= (np.array([0.001, -0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))

            arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.1)
