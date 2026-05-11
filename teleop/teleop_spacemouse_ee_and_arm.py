"""
SpaceMouse (3Dconnexion-style hidraw) direct EE-pose teleop.

Requires read access to the hidraw node (e.g. plugdev group or udev rules).
Protocol matches read_hidraw_3dmouse.py: rid 1 translation, rid 2 rotation, rid 3 buttons.
"""
import argparse
import glob
import os
import struct
import sys
import threading
import time
from multiprocessing import Array, Lock, Value

import logging_mp
import numpy as np
import pinocchio as pin

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.robot_arm import (
    G1_23_ArmController,
    G1_29_ArmController,
    H1_2_ArmController,
    H1_ArmController,
)
from teleop.robot_control.robot_arm_ik import (
    G1_23_ArmIK,
    G1_29_ArmIK,
    H1_2_ArmIK,
    H1_ArmIK,
)

try:
    logging_mp.basicConfig(level=logging_mp.INFO)
except RuntimeError as exc:
    if "already been started" not in str(exc):
        raise
logger_mp = logging_mp.getLogger(__name__)

# Button masks from SpaceMouse / output.md examples
BTN_1 = 4096
BTN_2 = 8192
BTN_3 = 16384
BTN_4 = 32768
BTN_MODE = 67108864  # 0x04000000


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_kv_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    txt = _read_text(path)
    if not txt:
        return out
    for line in txt.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _find_parent_attr(start_dir: str, attr_name: str, max_hops: int = 8) -> str | None:
    cur = os.path.realpath(start_dir)
    for _ in range(max_hops):
        val = _read_text(os.path.join(cur, attr_name))
        if val:
            return val
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _discover_spacemouse_hidraw(
    vid: str = "046d",
    pid: str = "c62b",
    name_contains: str = "SpaceMouse",
) -> str | None:
    """
    Find /dev/hidrawN by usb vid/pid (preferred) and fallback product name.
    """
    vid = str(vid).lower()
    pid = str(pid).lower()
    name_contains_l = str(name_contains).lower()
    by_name = []

    for hidraw_path in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        node = os.path.basename(hidraw_path)
        dev_node = f"/dev/{node}"
        device_dir = os.path.realpath(os.path.join(hidraw_path, "device"))

        # Primary source: HID_ID/HID_NAME from uevent, independent of parent depth.
        ue = _read_kv_file(os.path.join(device_dir, "uevent"))
        hid_id = ue.get("HID_ID", "")
        hid_name = ue.get("HID_NAME", "")
        if hid_id:
            parts = hid_id.split(":")
            if len(parts) == 3:
                vid_hex = parts[1].lower()
                pid_hex = parts[2].lower()
                if vid_hex == vid and pid_hex == pid:
                    return dev_node

        # Fallback: walk up ancestors and search vendor/product attrs.
        vid_text = _find_parent_attr(device_dir, "idVendor")
        pid_text = _find_parent_attr(device_dir, "idProduct")
        if vid_text and pid_text and vid_text.lower() == vid and pid_text.lower() == pid:
            return dev_node

        product = hid_name or _find_parent_attr(device_dir, "product") or _read_text(
            os.path.join(device_dir, "name")
        )
        if product and name_contains_l in product.lower():
            by_name.append(dev_node)

    if by_name:
        return by_name[0]
    return None


def _rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _apply_keyboard_delta(
    target_tf: np.ndarray,
    dp_base: np.ndarray | None = None,
    dR_local: np.ndarray | None = None,
    dR_base: np.ndarray | None = None,
) -> np.ndarray:
    T = np.asarray(target_tf, dtype=np.float64).copy().reshape(4, 4)
    if dp_base is not None:
        T[:3, 3] += np.asarray(dp_base, dtype=np.float64).reshape(3)
    if dR_local is not None:
        T[:3, :3] = T[:3, :3] @ np.asarray(dR_local, dtype=np.float64).reshape(3, 3)
    if dR_base is not None:
        T[:3, :3] = np.asarray(dR_base, dtype=np.float64).reshape(3, 3) @ T[:3, :3]
    return T


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

    try:
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
    except ModuleNotFoundError as e:
        # Lightweight mode: allow arm teleop even when optional hand stack deps (e.g. torch) are missing.
        logger_mp.warning(
            "Disable gripper controller because optional dependency is missing: %s. "
            "Running with --ee none behavior.",
            e,
        )
        return None, None, None, None, None, None
    return (
        gripper_ctrl,
        left_gripper_value,
        right_gripper_value,
        dual_gripper_data_lock,
        dual_gripper_state_array,
        dual_gripper_action_array,
    )


def _deadzone_vec(v: np.ndarray, dz: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    out = v.copy()
    for i in range(3):
        if abs(out[i]) < dz:
            out[i] = 0.0
    return out


def _spacemouse_dp_base(
    tx: int,
    ty: int,
    tz: int,
    trans_scale: float,
    trans_deadzone: float,
    flip: np.ndarray,
) -> np.ndarray:
    """Map device translation to robot base-frame delta (meters per tick)."""
    raw = np.array([-float(ty), -float(tx), -float(tz)], dtype=np.float64)
    raw = _deadzone_vec(raw, trans_deadzone)
    dp = trans_scale * raw * np.asarray(flip, dtype=np.float64).reshape(3)
    return dp


def _spacemouse_dR_base(
    rx: int,
    ry: int,
    rz: int,
    rot_scale: float,
    rot_axis_scale: np.ndarray,
    rot_deadzone: float,
) -> np.ndarray | None:
    """
    Build a rotation delta expressed in the robot base frame.
    rx -> pitch (rotation about base Y), ry -> roll (base X), rz -> yaw (base Z).
    Composed as Rz(yaw) @ Ry(pitch) @ Rx(roll) (small-angle increments) and applied by
    LEFT-multiplication
    so that rotation axes are always aligned with the fixed base frame, regardless of
    the current EE orientation.
    """
    rv = np.array([float(rx), float(ry), float(rz)], dtype=np.float64)
    rv = _deadzone_vec(rv, rot_deadzone)
    if float(np.linalg.norm(rv)) < 1e-12:
        return None
    k = np.asarray(rot_axis_scale, dtype=np.float64).reshape(3)
    pitch_base_y = rot_scale * k[0] * rv[0]
    roll_base_x = rot_scale * k[1] * rv[1]
    yaw_base_z = rot_scale * k[2] * rv[2]
    return _rot_z(yaw_base_z) @ _rot_y(pitch_base_y) @ _rot_x(roll_base_x)


class SpaceMouseHidState:
    """Thread-safe latest axes + rising-edge button bitmask (consumed in snapshot)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.tx = self.ty = self.tz = 0
        self.rx = self.ry = self.rz = 0
        self.buttons = 0
        self._pending_edges = 0

    def on_translation(self, tx: int, ty: int, tz: int):
        with self._lock:
            self.tx, self.ty, self.tz = tx, ty, tz

    def on_rotation(self, rx: int, ry: int, rz: int):
        with self._lock:
            self.rx, self.ry, self.rz = rx, ry, rz

    def on_buttons(self, new_buttons: int, pressed_edges: int):
        with self._lock:
            self.buttons = new_buttons
            self._pending_edges |= pressed_edges

    def snapshot(self):
        with self._lock:
            snap = (
                self.tx,
                self.ty,
                self.tz,
                self.rx,
                self.ry,
                self.rz,
                int(self.buttons),
                int(self._pending_edges),
            )
            self._pending_edges = 0
            return snap


def _hid_reader_loop(path: str, state: SpaceMouseHidState, stop_evt: threading.Event, exit_on_disconnect: bool):
    buttons_prev = 0
    buttons_initialized = False
    short_button_report_warned = False
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        logger_mp.error("Cannot open %s: %s", path, e)
        stop_evt.set()
        return

    logger_mp.info("[spacemouse] Opened %s", path)
    try:
        while not stop_evt.is_set():
            try:
                data = os.read(fd, 64)
            except OSError as e:
                logger_mp.error("[spacemouse] read failed: %s", e)
                if exit_on_disconnect:
                    stop_evt.set()
                break
            if not data:
                continue
            rid = data[0]
            if rid == 1 and len(data) >= 7:
                tx, ty, tz = struct.unpack_from("<hhh", data, 1)
                state.on_translation(tx, ty, tz)
            elif rid == 2 and len(data) >= 7:
                rx, ry, rz = struct.unpack_from("<hhh", data, 1)
                state.on_rotation(rx, ry, rz)
            elif rid == 3:
                if len(data) < 5:
                    if not short_button_report_warned:
                        logger_mp.warning(
                            "[spacemouse] Ignore short button report len=%d (<5), expected rid + 4-byte mask.",
                            len(data),
                        )
                        short_button_report_warned = True
                    continue
                new_b = int.from_bytes(data[1:5], "little")
                if not buttons_initialized:
                    buttons_prev = new_b
                    buttons_initialized = True
                    state.on_buttons(new_b, 0)
                else:
                    pressed = new_b & ~buttons_prev
                    buttons_prev = new_b
                    state.on_buttons(new_b, pressed)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        logger_mp.info("[spacemouse] Reader thread exit.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="SpaceMouse hidraw EE teleop: base translation / base-frame rotation modes, buttons for arm and gripper."
    )
    parser.add_argument("--arm", type=str, choices=["G1_29", "G1_23", "H1_2", "H1"], default="G1_29")
    parser.add_argument(
        "--ee",
        type=str,
        choices=["none", "dex1", "inspire_gripper", "omnipicker"],
        default="none",
    )
    parser.add_argument("--frequency", type=float, default=30.0, help="Main control loop frequency (Hz)")
    parser.add_argument(
        "--hidraw-device",
        type=str,
        default="auto",
        help="hidraw device path, or 'auto' to discover SpaceMouse by VID/PID",
    )
    parser.add_argument("--hidraw-vid", type=str, default="046d", help="USB vendor id for auto-discovery")
    parser.add_argument("--hidraw-pid", type=str, default="c62b", help="USB product id for auto-discovery")
    parser.add_argument(
        "--hidraw-name-contains",
        type=str,
        default="SpaceMouse",
        help="Fallback product name keyword for auto-discovery",
    )
    parser.add_argument(
        "--trans-scale",
        type=float,
        default=1.5e-5,
        help="Base translation (m) per device unit per tick; tune with cap deflection",
    )
    parser.add_argument(
        "--trans-deadzone",
        type=float,
        default=8.0,
        help="Per-axis deadzone on raw tx,ty,tz (after mapping to dp components)",
    )
    parser.add_argument(
        "--flip-x",
        type=float,
        default=1.0,
        help="Multiply base X delta by this sign (+1 or -1)",
    )
    parser.add_argument("--flip-y", type=float, default=1.0, help="Multiply base Y delta")
    parser.add_argument("--flip-z", type=float, default=1.0, help="Multiply base Z delta")
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=4.0e-5,
        help="Rotation (rad) per device unit per tick for rx,ry,rz (scaled by --rot-axis-scale)",
    )
    parser.add_argument(
        "--rot-axis-scale",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("RX", "RY", "RZ"),
        help="Per-axis multipliers (rx pitch@base-Y, ry roll@base-X, rz yaw@base-Z)",
    )
    parser.add_argument(
        "--rot-deadzone",
        type=float,
        default=8.0,
        help="Per-axis deadzone on raw rx,ry,rz",
    )
    parser.add_argument(
        "--exit-on-disconnect",
        action="store_true",
        help="Stop main loop if hidraw read fails",
    )
    parser.add_argument(
        "--left-ee-pose",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Initial left EE target pose (m, deg). Default: current FK.",
    )
    parser.add_argument(
        "--right-ee-pose",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Initial right EE target pose (m, deg). Default: current FK.",
    )
    parser.add_argument("--print-period", type=float, default=0.25, help="Status print period (s)")
    parser.add_argument(
        "--debug-buttons",
        action="store_true",
        help="Print raw button mask and rising-edge mask periodically",
    )
    parser.add_argument("--sim", action="store_true", help="Simulation mode (shared memory)")
    parser.add_argument("--real", action="store_true", help="Real robot (DDS)")
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--workspace-limit-x", type=float, nargs=2, default=[-0.20, 0.80], metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-limit-y", type=float, nargs=2, default=[-0.80, 0.80], metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-limit-z", type=float, nargs=2, default=[-2.0, 2.0], metavar=("MIN", "MAX"))
    parser.add_argument("--gripper-input-min", type=float, default=0.0)
    parser.add_argument("--gripper-input-max", type=float, default=1.0)
    parser.add_argument("--gripper-open-input", type=float, default=1.0)
    parser.add_argument("--gripper-close-input", type=float, default=0.0)
    # InspireDDS protocol endpoints (matches InspireDDS.isaac_output_range = (0.0, 1.2))
    #   open  = 0.0 -> JoySim mirror map drives joints to +-OMNIPICKER_GRIPPER_OPEN_RAD
    #   close = 1.2 -> JoySim mirror map drives joints to 0 (fingertips meet)
    parser.add_argument("--inspire-gripper-open", type=float, default=0.0)
    parser.add_argument("--inspire-gripper-close", type=float, default=1.2)
    parser.add_argument("--inspire-gripper-alpha", type=float, default=0.2)
    parser.add_argument("--inspire-gripper-max-speed", type=float, default=1.5)
    parser.add_argument("--go-home-on-exit", action="store_true", help="Send both arms home on exit")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.real:
        args.sim = False
    if not args.real and not args.sim:
        args.sim = True
    if args.ee == "omnipicker":
        args.ee = "inspire_gripper"
    if str(args.hidraw_device).lower() == "auto":
        resolved = _discover_spacemouse_hidraw(
            vid=args.hidraw_vid,
            pid=args.hidraw_pid,
            name_contains=args.hidraw_name_contains,
        )
        if resolved is None:
            parser.error(
                "Cannot auto-discover SpaceMouse hidraw device. "
                "Pass --hidraw-device /dev/hidrawN explicitly, or adjust "
                "--hidraw-vid/--hidraw-pid/--hidraw-name-contains."
            )
        args.hidraw_device = resolved

    if not args.sim:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    elif args.arm == "H1":
        ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        logger_mp.warning("H1 simulation still requires DDS init, using domain 1.")

    arm_ik = None
    arm_ctrl = None
    gripper_ctrl = None
    stop_evt = threading.Event()
    hid_thread = None
    sm_state = SpaceMouseHidState()

    flip = np.array([float(args.flip_x), float(args.flip_y), float(args.flip_z)], dtype=np.float64)
    rot_axis_scale = np.array(args.rot_axis_scale, dtype=np.float64)

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

        selected_arm = "left"
        control_mode = "position"  # "position" | "orientation"

        hid_thread = threading.Thread(
            target=_hid_reader_loop,
            args=(args.hidraw_device, sm_state, stop_evt, args.exit_on_disconnect),
            daemon=True,
        )
        hid_thread.start()

        logger_mp.info("-------------------------------------------------------------")
        logger_mp.info("SpaceMouse EE teleop started (default arm: LEFT).")
        logger_mp.info("Device: %s", args.hidraw_device)
        logger_mp.info("Position mode: cap translation -> base XYZ | Mode button toggles orientation mode")
        logger_mp.info(
            "Orientation mode: rx,ry,rz -> base-frame Rz(yaw)*Ry(pitch)*Rx(roll) "
            "(left-multiply, axes fixed to robot base)"
        )
        logger_mp.info("Buttons: 1=left arm, 2=right arm | 3=open gripper, 4=close | mode=toggle pos/ori")
        logger_mp.info("Exit: Ctrl+C")
        if gripper_ctrl is not None:
            logger_mp.info(
                "Gripper: input %.2f->open %.2f, input %.2f->close %.2f",
                float(args.gripper_open_input),
                float(args.inspire_gripper_open),
                float(args.gripper_close_input),
                float(args.inspire_gripper_close),
            )
        logger_mp.info("-------------------------------------------------------------")

        last_print_t = 0.0
        open_input = float(args.gripper_open_input)
        close_input = float(args.gripper_close_input)
        last_raw_buttons = -1

        while not stop_evt.is_set():
            tick_t0 = time.time()
            tx, ty, tz, rx, ry, rz, raw_buttons, edges = sm_state.snapshot()

            if args.debug_buttons and raw_buttons != last_raw_buttons:
                logger_mp.info(
                    "[spacemouse] buttons raw=%d bits(mode,1,2,3,4)=(%d,%d,%d,%d,%d) edges=%d",
                    raw_buttons,
                    1 if (raw_buttons & BTN_MODE) else 0,
                    1 if (raw_buttons & BTN_1) else 0,
                    1 if (raw_buttons & BTN_2) else 0,
                    1 if (raw_buttons & BTN_3) else 0,
                    1 if (raw_buttons & BTN_4) else 0,
                    edges,
                )
                last_raw_buttons = raw_buttons

            if edges & BTN_MODE:
                control_mode = "orientation" if control_mode == "position" else "position"
                logger_mp.info("[spacemouse] MODE pressed, control mode -> %s", control_mode.upper())
            if edges & BTN_1:
                selected_arm = "left"
                logger_mp.info("[spacemouse] BTN1 pressed, arm -> LEFT")
            if edges & BTN_2:
                selected_arm = "right"
                logger_mp.info("[spacemouse] BTN2 pressed, arm -> RIGHT")

            current_q = arm_ctrl.get_current_dual_arm_q()
            current_dq = arm_ctrl.get_current_dual_arm_dq()
            left_actual_tf, right_actual_tf = _fk_dual_ee_tf(arm_ik, current_q)

            pose_changed = False
            if control_mode == "position":
                dp_base = _spacemouse_dp_base(
                    tx, ty, tz, float(args.trans_scale), float(args.trans_deadzone), flip
                )
                dR_base = None
                pose_changed = bool(np.any(np.abs(dp_base) > 0))
            else:
                dp_base = np.zeros(3, dtype=np.float64)
                dR_base = _spacemouse_dR_base(
                    rx,
                    ry,
                    rz,
                    float(args.rot_scale),
                    rot_axis_scale,
                    float(args.rot_deadzone),
                )
                pose_changed = dR_base is not None

            if selected_arm == "left":
                left_target_tf = _apply_keyboard_delta(
                    left_target_tf,
                    dp_base=dp_base if control_mode == "position" else None,
                    dR_base=dR_base if control_mode == "orientation" else None,
                )
            else:
                right_target_tf = _apply_keyboard_delta(
                    right_target_tf,
                    dp_base=dp_base if control_mode == "position" else None,
                    dR_base=dR_base if control_mode == "orientation" else None,
                )

            if pose_changed:
                target_dirty = True

            if pose_changed and selected_arm == "left":
                left_target_tf[0, 3] = np.clip(left_target_tf[0, 3], args.workspace_limit_x[0], args.workspace_limit_x[1])
                left_target_tf[1, 3] = np.clip(left_target_tf[1, 3], args.workspace_limit_y[0], args.workspace_limit_y[1])
                left_target_tf[2, 3] = np.clip(left_target_tf[2, 3], args.workspace_limit_z[0], args.workspace_limit_z[1])
            elif pose_changed and selected_arm == "right":
                right_target_tf[0, 3] = np.clip(right_target_tf[0, 3], args.workspace_limit_x[0], args.workspace_limit_x[1])
                right_target_tf[1, 3] = np.clip(right_target_tf[1, 3], args.workspace_limit_y[0], args.workspace_limit_y[1])
                right_target_tf[2, 3] = np.clip(right_target_tf[2, 3], args.workspace_limit_z[0], args.workspace_limit_z[1])

            if gripper_ctrl is not None:
                # Latching behavior: button press updates target once and keeps it.
                if edges & BTN_3:
                    if selected_arm == "left":
                        with left_gripper_value.get_lock():
                            left_gripper_value.value = open_input
                    else:
                        with right_gripper_value.get_lock():
                            right_gripper_value.value = open_input
                    logger_mp.info("[spacemouse] BTN3 pressed, gripper OPEN (%s)", selected_arm)

                if edges & BTN_4:
                    if selected_arm == "left":
                        with left_gripper_value.get_lock():
                            left_gripper_value.value = close_input
                    else:
                        with right_gripper_value.get_lock():
                            right_gripper_value.value = close_input
                    logger_mp.info("[spacemouse] BTN4 pressed, gripper CLOSE (%s)", selected_arm)
            elif edges & (BTN_3 | BTN_4):
                logger_mp.info(
                    "[spacemouse] BTN3/BTN4 ignored because gripper is disabled (--ee none). "
                    "Use --ee inspire_gripper or --ee dex1."
                )

            if target_dirty:
                sol_q, sol_tauff = arm_ik.solve_ik(left_target_tf, right_target_tf, current_q, current_dq)
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
                msg = (
                    f"[{selected_arm.upper()}|{control_mode}] desired: {_format_pose(desired_tf)} | "
                    f"actual: {_format_pose(actual_tf)} | dev t=({tx},{ty},{tz}) r=({rx},{ry},{rz})"
                )
                if args.debug_buttons:
                    msg += f" | buttons(raw/edges)=({raw_buttons}/{edges})"
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
        stop_evt.set()
        if hid_thread is not None:
            hid_thread.join(timeout=2.0)
        try:
            if args.go_home_on_exit and arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error("Failed to send go-home on exit: %s", e)
        logger_mp.info("Exit SpaceMouse ee teleop.")


if __name__ == "__main__":
    main()
