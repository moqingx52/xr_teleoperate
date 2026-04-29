import argparse
import os
import sys
import threading
import time
from multiprocessing import Lock, Value, Array

import numpy as np
import pinocchio as pin
import logging_mp
from sshkeyboard import listen_keyboard, stop_listening


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_23_ArmController,
    H1_2_ArmController,
    H1_ArmController,
)
from teleop.robot_control.robot_arm_ik import (
    G1_29_ArmIK,
    G1_23_ArmIK,
    H1_2_ArmIK,
    H1_ArmIK,
)


try:
    logging_mp.basicConfig(level=logging_mp.INFO)
except RuntimeError as exc:
    # logging_mp raises when initialized elsewhere; keep existing config.
    if "already been started" not in str(exc):
        raise
logger_mp = logging_mp.getLogger(__name__)


def _is_pressed(pressed_keys, *aliases: str) -> bool:
    for key in aliases:
        if str(key).lower() in pressed_keys:
            return True
    return False


def _keyboard_delta_world(ee_tf: np.ndarray, pressed, step: float, move_frame: str = "hybrid") -> np.ndarray:
    """
    move_frame:
      - "hybrid": W/A/S/D follow EE local x/y; Up/Down follow world z (recommended)
      - "local": all movement follows EE local frame
      - "world": all movement follows world frame
    """
    cmd = np.zeros(3, dtype=np.float64)

    if _is_pressed(pressed, "w"):
        cmd[0] += step
    if _is_pressed(pressed, "s"):
        cmd[0] -= step
    if _is_pressed(pressed, "a"):
        cmd[1] += step
    if _is_pressed(pressed, "d"):
        cmd[1] -= step
    if _is_pressed(pressed, "up", "arrow_up", "↑", "8"):
        cmd[2] += step
    if _is_pressed(pressed, "down", "arrow_down", "↓", "2"):
        cmd[2] -= step

    if not np.any(cmd):
        return np.zeros(3, dtype=np.float64)

    R = ee_tf[:3, :3]
    if move_frame == "local":
        return R @ cmd
    if move_frame == "world":
        return cmd

    # hybrid: x/y follow EE local frame; z stays world-vertical.
    dp = R @ np.array([cmd[0], cmd[1], 0.0], dtype=np.float64)
    dp[2] += cmd[2]
    return dp


def _keyboard_pitch_delta(pressed, step_rad: float) -> float:
    delta = 0.0
    if _is_pressed(pressed, "left", "arrow_left", "←", "4"):
        delta += step_rad
    if _is_pressed(pressed, "right", "arrow_right", "→", "6"):
        delta -= step_rad
    return float(delta)


def _rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rpy_deg_to_rot(rpy_deg: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _pose_vec_to_tf(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = pose[:3]
    tf[:3, :3] = _rpy_deg_to_rot(pose[3:])
    return tf


def _rot_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def _format_pose(T: np.ndarray) -> str:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = T[:3, 3]
    rpy = _rot_to_rpy_deg(T[:3, :3])
    return (
        f"xyz=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m, "
        f"rpy=({rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f}) deg"
    )


def _fk_dual_ee_tf(arm_ik, q_lr: np.ndarray):
    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    qv = np.asarray(q_lr, dtype=np.float64).reshape(model.nq)
    pin.forwardKinematics(model, data, qv)
    pin.updateFramePlacements(model, data)

    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    left[:3, :3] = np.asarray(data.oMf[arm_ik.L_hand_id].rotation, dtype=np.float64).reshape(3, 3)
    left[:3, 3] = np.asarray(data.oMf[arm_ik.L_hand_id].translation, dtype=np.float64).reshape(3)
    right[:3, :3] = np.asarray(data.oMf[arm_ik.R_hand_id].rotation, dtype=np.float64).reshape(3, 3)
    right[:3, 3] = np.asarray(data.oMf[arm_ik.R_hand_id].translation, dtype=np.float64).reshape(3)
    return left, right


class KeyboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._pressed = set()
        self.selected_arm = "left"
        self.stop = False

    def on_press(self, key):
        k = str(key).lower()
        with self._lock:
            self._pressed.add(k)
            if k == "l":
                self.selected_arm = "left"
                logger_mp.info("[keyboard] switch arm -> LEFT")
            elif k == "r":
                self.selected_arm = "right"
                logger_mp.info("[keyboard] switch arm -> RIGHT")
            elif k == "x":
                self.stop = True

    def on_release(self, key):
        k = str(key).lower()
        with self._lock:
            if k in self._pressed:
                self._pressed.remove(k)

    def snapshot(self):
        with self._lock:
            return set(self._pressed), str(self.selected_arm), bool(self.stop)


def _build_arm_stack(arm_name: str, simulation_mode: bool):
    if arm_name == "G1_29":
        return G1_29_ArmIK(), G1_29_ArmController(simulation_mode=simulation_mode)
    if arm_name == "G1_23":
        return G1_23_ArmIK(), G1_23_ArmController(simulation_mode=simulation_mode)
    if arm_name == "H1_2":
        return H1_2_ArmIK(), H1_2_ArmController(simulation_mode=simulation_mode)
    return H1_ArmIK(), H1_ArmController(simulation_mode=simulation_mode)


def _build_gripper_controller(args):
    if args.ee == "none":
        return None, None, None, None, None, None

    left_gripper_value = Value("d", float(args.gripper_open_input), lock=True)
    right_gripper_value = Value("d", float(args.gripper_open_input), lock=True)
    dual_gripper_data_lock = Lock()
    dual_gripper_state_array = Array("d", 2, lock=False)
    dual_gripper_action_array = Array("d", 2, lock=False)

    if args.ee == "dex1":
        from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller

        gripper_ctrl = Dex1_1_Gripper_Controller(
            left_gripper_value,
            right_gripper_value,
            dual_gripper_data_lock,
            dual_gripper_state_array,
            dual_gripper_action_array,
            simulation_mode=args.sim,
        )
    else:
        from teleop.robot_control.robot_hand_inspire import Inspire_Gripper_Controller

        gripper_ctrl = Inspire_Gripper_Controller(
            left_gripper_value,
            right_gripper_value,
            dual_gripper_data_lock,
            dual_gripper_state_array,
            dual_gripper_action_array,
            simulation_mode=args.sim,
            input_min=float(args.gripper_input_min),
            input_max=float(args.gripper_input_max),
            open_cmd=float(args.inspire_gripper_open),
            close_cmd=float(args.inspire_gripper_close),
            smooth_alpha=float(args.inspire_gripper_alpha),
            max_speed=float(args.inspire_gripper_max_speed),
        )
    return (
        gripper_ctrl,
        left_gripper_value,
        right_gripper_value,
        dual_gripper_data_lock,
        dual_gripper_state_array,
        dual_gripper_action_array,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Keyboard direct EE-pose teleop: WASD + Up/Down move, Left/Right pitch, Q/E gripper."
    )
    parser.add_argument("--arm", type=str, choices=["G1_29", "G1_23", "H1_2", "H1"], default="G1_29")
    parser.add_argument(
        "--ee",
        type=str,
        choices=["none", "dex1", "inspire_gripper", "omnipicker"],
        default="omnipicker",
    )
    parser.add_argument("--frequency", type=float, default=30.0, help="Main control loop frequency (Hz)")
    parser.add_argument("--pos-step", type=float, default=0.005, help="Relative translation per tick (m)")
    parser.add_argument("--pitch-step-deg", type=float, default=1.0, help="Gripper pitch step per tick (deg)")
    parser.add_argument(
        "--left-ee-pose",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Initial left EE target pose. xyz in meters, rpy in degrees. Defaults to current FK.",
    )
    parser.add_argument(
        "--right-ee-pose",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Initial right EE target pose. xyz in meters, rpy in degrees. Defaults to current FK.",
    )
    parser.add_argument(
        "--move-frame",
        type=str,
        choices=["hybrid", "local", "world"],
        default="hybrid",
        help="hybrid: local x/y + world z; local: all local; world: all world",
    )
    parser.add_argument("--print-period", type=float, default=0.25, help="Status print period (s)")
    parser.add_argument("--sim", action="store_true", help="Use simulation mode (shared memory)")
    parser.add_argument("--real", action="store_true", help="Force real robot mode (DDS)")
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--workspace-limit-x", type=float, nargs=2, default=[-0.20, 0.80], metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-limit-y", type=float, nargs=2, default=[-0.80, 0.80], metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-limit-z", type=float, nargs=2, default=[-0.40, 0.80], metavar=("MIN", "MAX"))
    parser.add_argument("--gripper-input-min", type=float, default=0.0)
    parser.add_argument("--gripper-input-max", type=float, default=1.0)
    parser.add_argument("--gripper-open-input", type=float, default=1.0)
    parser.add_argument("--gripper-close-input", type=float, default=0.0)
    parser.add_argument("--inspire-gripper-open", type=float, default=0.05)
    parser.add_argument("--inspire-gripper-close", type=float, default=0.9)
    parser.add_argument("--inspire-gripper-alpha", type=float, default=0.2)
    parser.add_argument("--inspire-gripper-max-speed", type=float, default=1.5)
    parser.add_argument("--go-home-on-exit", action="store_true", help="Send both arms to home pose when exiting")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.real:
        args.sim = False
    if not args.real and not args.sim:
        args.sim = True
    if args.ee == "omnipicker":
        # Omnipicker gripper is implemented by Inspire_Gripper_Controller.
        args.ee = "inspire_gripper"

    if not args.sim:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    elif args.arm == "H1":
        ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        logger_mp.warning("H1 simulation still requires DDS init, using domain 1.")

    arm_ik = None
    arm_ctrl = None
    gripper_ctrl = None
    kb_listener = None

    try:
        arm_ik, arm_ctrl = _build_arm_stack(args.arm, simulation_mode=args.sim)
        (
            gripper_ctrl,
            left_gripper_value,
            right_gripper_value,
            dual_gripper_data_lock,
            dual_gripper_state_array,
            dual_gripper_action_array,
        ) = _build_gripper_controller(args)

        q_now = arm_ctrl.get_current_dual_arm_q()
        left_actual_tf, right_actual_tf = _fk_dual_ee_tf(arm_ik, q_now)
        left_target_tf = _pose_vec_to_tf(args.left_ee_pose) if args.left_ee_pose is not None else left_actual_tf.copy()
        right_target_tf = _pose_vec_to_tf(args.right_ee_pose) if args.right_ee_pose is not None else right_actual_tf.copy()
        hold_q = q_now.copy()
        hold_tauff = np.zeros_like(hold_q, dtype=np.float64)
        target_dirty = args.left_ee_pose is not None or args.right_ee_pose is not None

        keyboard_state = KeyboardState()
        kb_listener = threading.Thread(
            target=listen_keyboard,
            kwargs={
                "on_press": keyboard_state.on_press,
                "on_release": keyboard_state.on_release,
                "until": None,
                "sequential": False,
            },
            daemon=True,
        )
        kb_listener.start()

        logger_mp.info("-------------------------------------------------------------")
        logger_mp.info("Keyboard direct EE-pose teleop started.")
        logger_mp.info(
            "Move frame=%s | W/S(+/-X), A/D(+/-Y), Up/Down or 8/2(+/-Z)",
            args.move_frame,
        )
        logger_mp.info("Pitch: Left/Right arrows or 4/6 rotate selected gripper around local Y.")
        logger_mp.info("hybrid means XY in EE local frame, Z in world frame.")
        logger_mp.info("Arm select: L=left arm, R=right arm (one arm at a time)")
        logger_mp.info("Gripper: Q=open, E=close (selected arm)")
        logger_mp.info("Exit: X")
        if gripper_ctrl is not None:
            logger_mp.info(
                "Gripper mapping(Omnipicker): input %.2f->open_cmd %.2f, input %.2f->close_cmd %.2f",
                float(args.gripper_open_input),
                float(args.inspire_gripper_open),
                float(args.gripper_close_input),
                float(args.inspire_gripper_close),
            )
        logger_mp.info("-------------------------------------------------------------")

        last_print_t = 0.0
        open_input = float(args.gripper_open_input)
        close_input = float(args.gripper_close_input)

        while True:
            tick_t0 = time.time()
            pressed, selected_arm, should_stop = keyboard_state.snapshot()
            if should_stop:
                break

            current_q = arm_ctrl.get_current_dual_arm_q()
            current_dq = arm_ctrl.get_current_dual_arm_dq()
            left_actual_tf, right_actual_tf = _fk_dual_ee_tf(arm_ik, current_q)

            pitch_delta = _keyboard_pitch_delta(pressed, np.deg2rad(float(args.pitch_step_deg)))

            if selected_arm == "left":
                dp_world = _keyboard_delta_world(
                    left_target_tf,
                    pressed,
                    float(args.pos_step),
                    args.move_frame,
                )
                left_target_tf[:3, 3] += dp_world
                if pitch_delta:
                    left_target_tf[:3, :3] = left_target_tf[:3, :3] @ _rot_y(pitch_delta)
                pose_changed = bool(np.any(dp_world)) or bool(pitch_delta)
            else:
                dp_world = _keyboard_delta_world(
                    right_target_tf,
                    pressed,
                    float(args.pos_step),
                    args.move_frame,
                )
                right_target_tf[:3, 3] += dp_world
                if pitch_delta:
                    right_target_tf[:3, :3] = right_target_tf[:3, :3] @ _rot_y(pitch_delta)
                pose_changed = bool(np.any(dp_world)) or bool(pitch_delta)

            if pose_changed:
                target_dirty = True

            # Apply workspace clipping only for actively changed arm.
            if pose_changed and selected_arm == "left":
                left_target_tf[0, 3] = np.clip(left_target_tf[0, 3], args.workspace_limit_x[0], args.workspace_limit_x[1])
                left_target_tf[1, 3] = np.clip(left_target_tf[1, 3], args.workspace_limit_y[0], args.workspace_limit_y[1])
                left_target_tf[2, 3] = np.clip(left_target_tf[2, 3], args.workspace_limit_z[0], args.workspace_limit_z[1])
            elif pose_changed and selected_arm == "right":
                right_target_tf[0, 3] = np.clip(right_target_tf[0, 3], args.workspace_limit_x[0], args.workspace_limit_x[1])
                right_target_tf[1, 3] = np.clip(right_target_tf[1, 3], args.workspace_limit_y[0], args.workspace_limit_y[1])
                right_target_tf[2, 3] = np.clip(right_target_tf[2, 3], args.workspace_limit_z[0], args.workspace_limit_z[1])

            if gripper_ctrl is not None:
                open_hold = ("q" in pressed) and ("e" not in pressed)
                close_hold = ("e" in pressed) and ("q" not in pressed)
                if selected_arm == "left":
                    if open_hold:
                        with left_gripper_value.get_lock():
                            left_gripper_value.value = open_input
                    elif close_hold:
                        with left_gripper_value.get_lock():
                            left_gripper_value.value = close_input
                else:
                    if open_hold:
                        with right_gripper_value.get_lock():
                            right_gripper_value.value = open_input
                    elif close_hold:
                        with right_gripper_value.get_lock():
                            right_gripper_value.value = close_input

            if target_dirty:
                sol_q, sol_tauff = arm_ik.solve_ik(
                    left_target_tf, right_target_tf, current_q, current_dq
                )
                hold_q = np.asarray(sol_q, dtype=np.float64).copy()
                hold_tauff = np.asarray(sol_tauff, dtype=np.float64).copy()
                target_dirty = False
            else:
                sol_q = hold_q
                sol_tauff = hold_tauff
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            now = time.time()
            if now - last_print_t >= float(args.print_period):
                desired_tf = left_target_tf if selected_arm == "left" else right_target_tf
                actual_tf = left_actual_tf if selected_arm == "left" else right_actual_tf
                msg = f"[{selected_arm.upper()}] desired: {_format_pose(desired_tf)} | actual: {_format_pose(actual_tf)}"
                if gripper_ctrl is not None and dual_gripper_state_array is not None:
                    with dual_gripper_data_lock:
                        if selected_arm == "left":
                            g_state = float(dual_gripper_state_array[0])
                            g_action = float(dual_gripper_action_array[0])
                        else:
                            g_state = float(dual_gripper_state_array[1])
                            g_action = float(dual_gripper_action_array[1])
                    msg += f" | gripper(action/state)=({g_action:.3f}/{g_state:.3f})"
                logger_mp.info(msg)
                last_print_t = now

            elapsed = time.time() - tick_t0
            sleep_t = max(0.0, (1.0 / args.frequency) - elapsed)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt received, exiting.")
    finally:
        try:
            stop_listening()
            if kb_listener is not None:
                kb_listener.join(timeout=1.0)
        except Exception:
            pass
        try:
            if args.go_home_on_exit and arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to send go-home on exit: {e}")
        logger_mp.info("Exit keyboard ee teleop.")


if __name__ == "__main__":
    main()
