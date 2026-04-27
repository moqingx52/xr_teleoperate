import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os
import sys
import numpy as np
import pinocchio as pin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import (
    G1_29_ArmIK,
    G1_23_ArmIK,
    H1_2_ArmIK,
    H1_ArmIK,
    homogeneous_from_position_rotation,
)
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation (non-sim uses DDS reset publisher)
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def publish_reset_category(category: int, publisher=None, reset_shm=None):
    """Scene reset: Isaac reads ``isaac_reset_pose_cmd`` (shm) or ``rt/reset_pose/cmd`` (DDS)."""
    if reset_shm is not None:
        reset_shm.write_data({"reset_category": str(category)})
        logger_mp.info(f"published reset category (shm): {category}")
        return
    if publisher is not None:
        msg = String_(data=str(category))
        publisher.Write(msg)
        logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


def _rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]], dtype=np.float64)


def _rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy_deg_to_R_xyz(rpy_deg) -> np.ndarray:
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    return _rot_x(r) @ _rot_y(p) @ _rot_z(y)


def _compute_dual_fk_rotations(arm_ik, q):
    """Return {'left': R_l, 'right': R_r} from IK model FK, or None if unavailable."""
    try:
        model = arm_ik.reduced_robot.model
        data = arm_ik.reduced_robot.data
        qv = np.asarray(q, dtype=np.float64).reshape(model.nq)
        pin.forwardKinematics(model, data, qv)
        pin.updateFramePlacements(model, data)
        R_l = np.asarray(data.oMf[arm_ik.L_hand_id].rotation, dtype=np.float64).reshape(3, 3)
        R_r = np.asarray(data.oMf[arm_ik.R_hand_id].rotation, dtype=np.float64).reshape(3, 3)
        return {"left": R_l, "right": R_r}
    except Exception:
        return None


def _scale_tf_translation(tf: np.ndarray, home_xyz: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(tf, dtype=np.float64).copy().reshape(4, 4)
    if abs(float(scale) - 1.0) < 1e-12:
        return out
    p = out[:3, 3]
    out[:3, 3] = np.asarray(home_xyz, dtype=np.float64).reshape(3) + float(scale) * (
        p - np.asarray(home_xyz, dtype=np.float64).reshape(3)
    )
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type=float, default=30.0, help='control and record\'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'inspire_gripper', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action='store_true', help='Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action='store_true', help='Enable isaac simulation mode')
    parser.add_argument('--ipc', action='store_true', help='Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action='store_true', help='Enable high priority and set CPU affinity mode')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logs for offline wrist->IK chain')
    # record mode and task info
    parser.add_argument('--record', action='store_true', help='Enable data recording mode')
    parser.add_argument('--task-dir', type=str, default='./utils/data/', help='path to save data')
    parser.add_argument('--task-name', type=str, default='pick cube', help='task file name for recording')
    parser.add_argument('--task-goal', type=str, default='pick up cube.', help='task goal for recording at json file')
    parser.add_argument('--task-desc', type=str, default='task description', help='task description for recording at json file')
    parser.add_argument('--task-steps', type=str, default='step1: do this; step2: do that;', help='task steps for recording at json file')
    # HaMeR / offline JSON input (与 hamer/demo.py: --structured_file 默认 hamer_structured_results.json)
    parser.add_argument('--input-source', '--input_source', type=str, default='xr',
                        choices=['xr', 'hamer', 'egodex'],
                        dest='input_source',
                        help='xr: TeleVuer; hamer: structured JSON replay; egodex: HDF5 (EgoDex-style) replay')
    parser.add_argument('--hamer-json', '--hamer_json', type=str, default=None, dest='hamer_json',
                        help='Absolute or relative path to structured JSON (demo: out_folder/hamer_structured_results.json)')
    parser.add_argument('--parquet', '--parquent', type=str, default=None, dest='parquet',
                        help='Absolute or relative path to replay parquet (dataset/data/chunk-xxx/episode_xxxxxx.parquet)')
    parser.add_argument('--hamer-out-dir', '--hamer_out_dir', type=str, default=None, dest='hamer_out_dir',
                        help='HaMeR demo --out_folder; JSON path becomes <dir>/<hamer_structured_file>')
    parser.add_argument('--hamer-structured-file', '--hamer_structured_file', type=str, default='hamer_structured_results.json',
                        dest='hamer_structured_file', help='Same default as hamer demo.py --structured_file')
    parser.add_argument('--replay-fps', '--replay_fps', type=float, default=None, dest='replay_fps',
                        help='If set, overrides --frequency for main-loop timing (offline replay)')
    parser.add_argument('--hamer-loop', '--hamer_loop', action='store_true', dest='hamer_loop', help='Loop JSON when end is reached')
    parser.add_argument('--hamer-score-thresh', '--hamer_score_thresh', type=float, default=0.5, dest='hamer_score_thresh',
                        help='Min detection score per hand row')
    parser.add_argument('--hamer-arm-only', '--hamer_arm_only', action='store_true', dest='hamer_arm_only',
                        help='Do not drive hand from XR/HaMeR (arm only)')
    parser.add_argument('--hamer-relative-pos', '--hamer_relative_pos', action='store_true', dest='hamer_relative_pos',
                        help='Wrist EE pose: anchor at first valid frame; position home+scale*Δp, rotation relative anchor with same scale')
    parser.add_argument('--hamer-mirror-lr-xz', '--hamer_mirror_lr_xz', action='store_true', dest='hamer_mirror_lr_xz',
                        help='Swap left/right HaMeR wrists and mirror across robot xz-plane (y->-y)')
    parser.add_argument('--hamer-relative-compress', '--hamer_relative_compress', action='store_true',
                        dest='hamer_relative_compress',
                        help='With --hamer-relative-pos: after scaling position delta, clip each axis to ±--hamer-relative-clip (rotation not clipped)')
    parser.add_argument('--hamer-left-home', '--hamer_left_home', type=float, nargs=3, default=[0.25, 0.25, 0.1],
                        dest='hamer_left_home', metavar=('X', 'Y', 'Z'),
                        help='Left home position (meters) used by --hamer-relative-pos')
    parser.add_argument('--hamer-right-home', '--hamer_right_home', type=float, nargs=3, default=[0.25, -0.25, 0.1],
                        dest='hamer_right_home', metavar=('X', 'Y', 'Z'),
                        help='Right home position (meters) used by --hamer-relative-pos')
    parser.add_argument('--hamer-relative-scale', '--hamer_relative_scale', type=float, default=0.02,
                        dest='hamer_relative_scale',
                        help='With --hamer-relative-pos: scales position Δ and relative rotation angle from anchor (default 0.02)')
    parser.add_argument('--hamer-relative-clip', '--hamer_relative_clip', type=float, nargs=3, default=[0.12, 0.12, 0.10],
                        dest='hamer_relative_clip', metavar=('DX', 'DY', 'DZ'),
                        help='With --hamer-relative-compress: per-axis clip (m) on scaled position delta only')
    parser.add_argument('--hamer-frame-offset-json', '--hamer_frame_offset_json', type=str, default=None,
                        dest='hamer_frame_offset_json', help='Optional JSON for frame index offsets')
    parser.add_argument('--hamer-cam2base-json', '--hamer_cam2base_json', type=str, default=None,
                        dest='hamer_cam2base_json',
                        help='When HaMeR JSON has p_wrist/R_wrist only: same 4x4 T_cam2base JSON as HaMeR --cam2base_json')
    parser.add_argument(
        '--hamer-hand-target-bone-len',
        '--hamer_hand_target_bone_len',
        type=float,
        default=0.10,
        dest='hamer_hand_target_bone_len',
        help='HamerHandBridge: target middle-finger MCP bone length (m) for scaling keypoints_3d_local to 25pt layout (default 0.10)',
    )
    parser.add_argument(
        '--hamer-wrist-max-step-m',
        '--hamer_wrist_max_step_m',
        type=float,
        default=-1.0,
        dest='hamer_wrist_max_step_m',
        help='HamerAdapter: max wrist/EE translation per control tick (m); <=0 disables this limiter (default -1)',
    )
    parser.add_argument(
        '--hamer-wrist-max-step-rad',
        '--hamer_wrist_max_step_rad',
        type=float,
        default=-1.0,
        dest='hamer_wrist_max_step_rad',
        help='HamerAdapter: max wrist/EE rotation per tick (rad); <=0 disables this limiter (default -1)',
    )
    parser.add_argument(
        '--hamer-wrist-smooth-alpha',
        '--hamer_wrist_smooth_alpha',
        type=float,
        default=1.0,
        dest='hamer_wrist_smooth_alpha',
        help='HamerAdapter: EMA blend toward new wrist target (0..1); 1.0 means no smoothing (default 1.0)',
    )
    parser.add_argument(
        '--hamer-debug-freeze-wrist-rotation',
        '--hamer_debug_freeze_wrist_rotation',
        action='store_true',
        dest='hamer_debug_freeze_wrist_rotation',
        help='Debug: with --hamer-relative-pos, keep wrist orientation at first-frame anchor (isolates rotation vs position bugs)',
    )
    parser.add_argument('--swap-hand-input', '--swap_hand_input', action='store_true', dest='swap_hand_input',
                        help='Exchange left/right hand skeleton and wrist poses before dex retargeting / arm IK (dataset vs robot LR mismatch)')
    parser.add_argument(
        '--disable-hand-hist-stab',
        '--disable_hand_hist_stab',
        action='store_true',
        dest='disable_hand_hist_stab',
        help='Disable wrist median history outlier rejection for dex hand (25,3) positions (default: enabled)',
    )
    parser.add_argument(
        '--disable-hand-pred-stab',
        '--disable_hand_pred_stab',
        action='store_true',
        dest='disable_hand_pred_stab',
        help='Disable wrist linear-prediction outlier correction (whole-hand translation; default: enabled)',
    )
    parser.add_argument(
        '--hand-stab-hist-window',
        '--hand_stab_hist_window',
        type=int,
        default=10,
        dest='hand_stab_hist_window',
        help='Median history length (frames) for hand wrist outlier gate (default 4)',
    )
    parser.add_argument(
        '--hand-stab-hist-jump-m',
        '--hand_stab_hist_jump_m',
        type=float,
        default=0.25,
        dest='hand_stab_hist_jump_m',
        help='Max wrist jump vs median of past frames before rejecting frame (m, default 0.25)',
    )
    parser.add_argument(
        '--hand-stab-pred-dev-m',
        '--hand_stab_pred_dev_m',
        type=float,
        default=0.25,
        dest='hand_stab_pred_dev_m',
        help='Max wrist deviation from linear prediction before translating hand to prediction (m, default 0.25)',
    )
    # EgoDex / offline HDF5 input
    parser.add_argument('--egodex-hdf5', '--egodex_hdf5', type=str, default=None, dest='egodex_hdf5',
                        help='Absolute or relative path to EgoDex episode HDF5 file')
    parser.add_argument('--egodex-root-frame', '--egodex_root_frame', type=str,
                        choices=['world', 'hip', 'camera'], default='hip', dest='egodex_root_frame',
                        help='Frame used as offline replay root. Recommended: hip')
    parser.add_argument('--egodex-loop', '--egodex_loop', action='store_true', dest='egodex_loop',
                        help='Loop EgoDex HDF5 when end is reached')
    parser.add_argument('--egodex-score-thresh', '--egodex_score_thresh', type=float, default=0.2,
                        dest='egodex_score_thresh',
                        help='Minimum confidence to accept leftHand/rightHand in EgoDex')
    parser.add_argument('--egodex-fps', '--egodex_fps', type=float, default=30.0, dest='egodex_fps',
                        help='Replay FPS of EgoDex episode (default 30)')
    parser.add_argument(
        '--egodex-arkit-wrist-rotation',
        '--egodex_arkit_wrist_rotation',
        action='store_true',
        dest='egodex_arkit_wrist_rotation',
        help='Use transforms/*Hand 3x3 rotation as R_wrist_base (old behavior); default rebuilds wrist frame from knuckles+forearm for IK',
    )
    parser.add_argument(
        '--wrist-to-ee-json',
        '--wrist_to_ee_json',
        type=str,
        default=None,
        dest='wrist_to_ee_json',
        help='Optional JSON calibration for constant wrist->EE transform (preferred over raw RPY flags)',
    )
    parser.add_argument(
        '--wrist-left-rpy-deg',
        '--wrist_left_rpy_deg',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        dest='wrist_left_rpy_deg',
        metavar=('ROLL_X_DEG', 'PITCH_Y_DEG', 'YAW_Z_DEG'),
        help='WristToEE left constant rotation offset in degrees (XYZ order, applied as R_x@R_y@R_z)',
    )
    parser.add_argument(
        '--wrist-right-rpy-deg',
        '--wrist_right_rpy_deg',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        dest='wrist_right_rpy_deg',
        metavar=('ROLL_X_DEG', 'PITCH_Y_DEG', 'YAW_Z_DEG'),
        help='WristToEE right constant rotation offset in degrees (XYZ order, applied as R_x@R_y@R_z)',
    )
    parser.add_argument(
        '--disable-ik-joint-smoothing',
        '--disable_ik_joint_smoothing',
        action='store_true',
        dest='disable_ik_joint_smoothing',
        help='Disable post-IK moving-average smoothing in ArmIK (for latency debugging)',
    )
    parser.add_argument(
        '--arm-reach-scale',
        '--arm_reach_scale',
        type=float,
        default=1.0,
        dest='arm_reach_scale',
        help='Scale wrist target translation around per-arm home point for non-default robot arm lengths (1.0 keeps original)',
    )
    parser.add_argument(
        '--arm-left-home',
        '--arm_left_home',
        type=float,
        nargs=3,
        default=[0.25, 0.25, 0.1],
        dest='arm_left_home',
        metavar=('X', 'Y', 'Z'),
        help='Left home point used by --arm-reach-scale',
    )
    parser.add_argument(
        '--arm-right-home',
        '--arm_right_home',
        type=float,
        nargs=3,
        default=[0.25, -0.25, 0.1],
        dest='arm_right_home',
        metavar=('X', 'Y', 'Z'),
        help='Right home point used by --arm-reach-scale',
    )
    parser.add_argument(
        '--gripper-input-min',
        '--gripper_input_min',
        type=float,
        default=5.0,
        dest='gripper_input_min',
        help='Input value mapped to closed gripper for inspire_gripper/dex1 semantic mode',
    )
    parser.add_argument(
        '--gripper-input-max',
        '--gripper_input_max',
        type=float,
        default=7.0,
        dest='gripper_input_max',
        help='Input value mapped to open gripper for inspire_gripper/dex1 semantic mode',
    )
    parser.add_argument(
        '--inspire-gripper-open',
        '--inspire_gripper_open',
        type=float,
        default=0.05,
        dest='inspire_gripper_open',
        help='Omnipicker inspire_gripper command value at open state',
    )
    parser.add_argument(
        '--inspire-gripper-close',
        '--inspire_gripper_close',
        type=float,
        default=0.9,
        dest='inspire_gripper_close',
        help='Omnipicker inspire_gripper command value at close state',
    )
    parser.add_argument(
        '--inspire-gripper-alpha',
        '--inspire_gripper_alpha',
        type=float,
        default=0.2,
        dest='inspire_gripper_alpha',
        help='Low-pass alpha for inspire_gripper command smoothing [0,1]',
    )
    parser.add_argument(
        '--inspire-gripper-max-speed',
        '--inspire_gripper_max_speed',
        type=float,
        default=1.5,
        dest='inspire_gripper_max_speed',
        help='Max inspire_gripper command speed (unit/s)',
    )
    parser.add_argument(
        '--omnipicker-gripper-source',
        '--omnipicker_gripper_source',
        type=str,
        choices=['action', 'task'],
        default='task',
        dest='omnipicker_gripper_source',
        help='Offline dex1/inspire_gripper source: action=read scalar from parquet vector, task=derive open/close from subtask text in dataset meta',
    )
    parser.add_argument(
        '--omnipicker-task-meta-dir',
        '--omnipicker_task_meta_dir',
        type=str,
        default=None,
        dest='omnipicker_task_meta_dir',
        help='Optional dataset meta directory used by --omnipicker-gripper-source=task (defaults to sibling ./meta inferred from parquet path)',
    )
    parser.add_argument(
        '--omnipicker-gripper-action-key',
        '--omnipicker_gripper_action_key',
        type=str,
        default='action',
        dest='omnipicker_gripper_action_key',
        help='Parquet vector key used when --omnipicker-gripper-source=action (e.g. action/original_action/observation.state)',
    )
    parser.add_argument(
        '--omnipicker-gripper-left-idx',
        '--omnipicker_gripper_left_idx',
        type=int,
        default=22,
        dest='omnipicker_gripper_left_idx',
        help='Left gripper scalar index inside --omnipicker-gripper-action-key vector',
    )
    parser.add_argument(
        '--omnipicker-gripper-right-idx',
        '--omnipicker_gripper_right_idx',
        type=int,
        default=29,
        dest='omnipicker_gripper_right_idx',
        help='Right gripper scalar index inside --omnipicker-gripper-action-key vector',
    )
    parser.add_argument(
        '--hamer-arm-control-source',
        '--hamer_arm_control_source',
        type=str,
        choices=['ik', 'joint'],
        default='ik',
        dest='hamer_arm_control_source',
        help='Offline arm source in HaMeR parquet replay: ik=use wrist->IK, joint=directly send joint targets from parquet vector',
    )
    parser.add_argument(
        '--omnipicker-joint-action-key',
        '--omnipicker_joint_action_key',
        type=str,
        default='action',
        dest='omnipicker_joint_action_key',
        help='Parquet vector key used when --hamer-arm-control-source=joint (e.g. action/original_action/observation.state)',
    )
    parser.add_argument(
        '--omnipicker-joint-left-range',
        '--omnipicker_joint_left_range',
        type=int,
        nargs=2,
        default=[0, 7],
        metavar=('START', 'END'),
        dest='omnipicker_joint_left_range',
        help='Left arm joint slice [start,end) in --omnipicker-joint-action-key vector',
    )
    parser.add_argument(
        '--omnipicker-joint-right-range',
        '--omnipicker_joint_right_range',
        type=int,
        nargs=2,
        default=[8, 15],
        metavar=('START', 'END'),
        dest='omnipicker_joint_right_range',
        help='Right arm joint slice [start,end) in --omnipicker-joint-action-key vector',
    )
    parser.add_argument(
        '--hamer-parquet-action-fallback-mode',
        '--hamer_parquet_action_fallback_mode',
        type=str,
        choices=['ee_base', 'wrist_cam'],
        default='ee_base',
        dest='hamer_parquet_action_fallback_mode',
        help=(
            'When parquet has no left/right kp3d: '
            'ee_base=use action slices as base-frame EE pose directly; '
            'wrist_cam=use action slices as camera-frame wrist pose, then apply --hamer-cam2base-json and --wrist-to-ee-json'
        ),
    )
    return parser


_DEFAULT_HAMER_REL_SCALE = 0.02
_DEFAULT_HAMER_REL_CLIP = np.array([0.12, 0.12, 0.10], dtype=np.float64)


def _normalize_hamer_relative_args(args: argparse.Namespace) -> None:
    """
    非默认的 scale 或显式 relative/compress/clip 会启用相对手腕模式；自定义 clip 需 compress 才生效。
    """
    if args.input_source not in ("hamer", "egodex"):
        return
    clip = np.asarray(args.hamer_relative_clip, dtype=np.float64).reshape(3)
    scale = float(args.hamer_relative_scale)
    scale_changed = abs(scale - _DEFAULT_HAMER_REL_SCALE) > 1e-12
    clip_changed = bool(np.any(np.abs(clip - _DEFAULT_HAMER_REL_CLIP) > 1e-12))
    wants_relative = (
        bool(args.hamer_relative_pos)
        or bool(args.hamer_relative_compress)
        or scale_changed
        or clip_changed
    )
    if not wants_relative:
        return
    if not args.hamer_relative_pos:
        logger_mp.warning(
            "[HaMeR] relative wrist options need --hamer-relative-pos (anchor at first valid pose). Enabling it."
        )
        args.hamer_relative_pos = True
    if clip_changed and not args.hamer_relative_compress:
        logger_mp.warning(
            "[HaMeR] non-default --hamer-relative-clip only applies with --hamer-relative-compress; enabling compress."
        )
        args.hamer_relative_compress = True


def main(argv=None):
    global START, STOP, READY, RECORD_RUNNING, RECORD_TOGGLE
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.replay_fps is not None:
        args.frequency = float(args.replay_fps)
    if args.input_source == "hamer":
        if args.hamer_json:
            args.hamer_json = os.path.abspath(os.path.expanduser(args.hamer_json))
        if args.parquet:
            args.parquet = os.path.abspath(os.path.expanduser(args.parquet))
        elif args.hamer_out_dir:
            args.hamer_json = os.path.join(
                os.path.abspath(os.path.expanduser(args.hamer_out_dir)),
                args.hamer_structured_file,
            )
        _normalize_hamer_relative_args(args)
    elif args.input_source == "egodex":
        if args.egodex_hdf5:
            args.egodex_hdf5 = os.path.abspath(os.path.expanduser(args.egodex_hdf5))
        _normalize_hamer_relative_args(args)
        if args.replay_fps is None:
            args.frequency = float(args.egodex_fps)
    logger_mp.info(f"args: {args}")
    if args.input_source == "hamer":
        has_json = bool(args.hamer_json)
        has_parquet = bool(args.parquet)
        if has_json and has_parquet:
            logger_mp.error("HaMeR mode: --hamer-json and --parquet are mutually exclusive (choose one).")
            exit(1)
        if (not has_json) and (not has_parquet):
            logger_mp.error("HaMeR mode requires one of --hamer-json or --parquet (or --hamer-out-dir).")
            exit(1)
    if args.input_source == "hamer" and args.parquet and args.hamer_frame_offset_json:
        logger_mp.warning("--hamer-frame-offset-json is ignored for --parquet input")
    if args.input_source == "hamer" and args.parquet and args.hamer_out_dir:
        logger_mp.warning("--hamer-out-dir is ignored when --parquet is given")
    if (
        args.input_source == "hamer"
        and args.hamer_arm_control_source == "joint"
        and not args.parquet
    ):
        logger_mp.error("--hamer-arm-control-source=joint currently requires --parquet input.")
        exit(1)
    if args.hamer_arm_control_source == "joint":
        ll = [int(v) for v in args.omnipicker_joint_left_range]
        rr = [int(v) for v in args.omnipicker_joint_right_range]
        if ll[0] < 0 or rr[0] < 0 or ll[1] <= ll[0] or rr[1] <= rr[0]:
            logger_mp.error("Invalid joint range: require non-negative [start,end) with end>start.")
            exit(1)
    if args.input_source == "egodex" and not args.egodex_hdf5:
        logger_mp.error("EgoDex mode requires --egodex-hdf5/--egodex_hdf5")
        exit(1)
    if args.input_source == "hamer" and args.hamer_cam2base_json:
        args.hamer_cam2base_json = os.path.abspath(os.path.expanduser(args.hamer_cam2base_json))
    if args.input_source in ("hamer", "egodex") and args.wrist_to_ee_json:
        args.wrist_to_ee_json = os.path.abspath(os.path.expanduser(args.wrist_to_ee_json))
    if args.input_source in ("hamer", "egodex") and args.motion and args.input_mode == "controller":
        logger_mp.warning("Offline replay input does not drive locomotion; motion+controller may be inactive.")
    arm_left_home = np.asarray(args.arm_left_home, dtype=np.float64).reshape(3)
    arm_right_home = np.asarray(args.arm_right_home, dtype=np.float64).reshape(3)

    arm_ctrl = None
    sim_state_subscriber = None
    reset_pose_publisher = None
    reset_pose_shm = None
    ipc_server = None
    listen_keyboard_thread = None
    hamer_input = None
    hand_pos_stab = None
    try:
        # Real robot: DDS domain 0. Simulation: shm for G1_23/G1_29/H1_2 + dex3/dex1; other combos still need DDS.
        if not args.sim:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)
        else:
            sim_needs_dds = args.arm == "H1" or args.ee in ("brainco",)
            if sim_needs_dds:
                ChannelFactoryInitialize(1, networkInterface=args.network_interface)
                logger_mp.warning(
                    "Simulation: this arm/EE still uses DDS in teleop; shm path covers G1_23/G1_29/H1_2 + dex3/dex1/inspire."
                )
            else:
                logger_mp.info("Simulation: teleop↔Isaac via shared memory (no DDS init for this arm/EE).")

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])
        if args.input_source in ("hamer", "egodex"):
            xr_need_local_img = False

        tv_wrapper = None
        hamer_adapter = None
        hamer_hand_bridge = None
        last_hamer_left_tf = np.eye(4, dtype=np.float64)
        last_hamer_left_tf[:3, 3] = np.array([0.25, 0.25, 0.1], dtype=np.float64)
        last_hamer_right_tf = np.eye(4, dtype=np.float64)
        last_hamer_right_tf[:3, 3] = np.array([0.25, -0.25, 0.1], dtype=np.float64)
        last_left_gripper_input = float(args.gripper_input_max)
        last_right_gripper_input = float(args.gripper_input_max)

        if args.input_source == "xr":
            tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand",
                                         binocular=camera_config['head_camera']['binocular'],
                                         img_shape=camera_config['head_camera']['image_shape'],
                                         display_mode=args.display_mode,
                                         zmq=camera_config['head_camera']['enable_zmq'],
                                         webrtc=camera_config['head_camera']['enable_webrtc'],
                                         webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                         )
        elif args.input_source == "hamer":
            from teleop.input_source.hamer_input import HamerInputSource
            from teleop.input_source.hamer_adapter import HamerAdapter
            from teleop.input_source.hamer_bridge import HamerHandBridge
            from teleop.input_source.hamer_to_robot_frame import WristToEEConfig, load_wrist_to_ee_config_from_json

            wrist_cfg_default = WristToEEConfig(
                t_left=np.zeros(3, dtype=np.float64),
                R_left=_rpy_deg_to_R_xyz(args.wrist_left_rpy_deg),
                t_right=np.zeros(3, dtype=np.float64),
                R_right=_rpy_deg_to_R_xyz(args.wrist_right_rpy_deg),
            )
            wrist_cfg = wrist_cfg_default
            if args.wrist_to_ee_json:
                wrist_cfg = load_wrist_to_ee_config_from_json(args.wrist_to_ee_json, wrist_cfg_default)
            hamer_input = HamerInputSource(
                json_path=args.hamer_json,
                parquet_path=args.parquet,
                score_thresh=args.hamer_score_thresh,
                loop=args.hamer_loop,
                frame_offset_json=args.hamer_frame_offset_json,
                cam2base_json=args.hamer_cam2base_json,
                gripper_action_key=args.omnipicker_gripper_action_key,
                gripper_left_idx=int(args.omnipicker_gripper_left_idx),
                gripper_right_idx=int(args.omnipicker_gripper_right_idx),
                gripper_source=args.omnipicker_gripper_source,
                omnipicker_task_meta_dir=args.omnipicker_task_meta_dir,
                parquet_action_fallback_mode=args.hamer_parquet_action_fallback_mode,
                joint_action_key=args.omnipicker_joint_action_key,
                joint_left_start=int(args.omnipicker_joint_left_range[0]),
                joint_left_end=int(args.omnipicker_joint_left_range[1]),
                joint_right_start=int(args.omnipicker_joint_right_range[0]),
                joint_right_end=int(args.omnipicker_joint_right_range[1]),
            )
            hamer_adapter = HamerAdapter(
                wrist_cfg,
                smooth_alpha=float(args.hamer_wrist_smooth_alpha),
                max_step_m=float(args.hamer_wrist_max_step_m),
                max_step_rad=float(args.hamer_wrist_max_step_rad),
                relative_position_mode=args.hamer_relative_pos,
                left_home=np.asarray(args.hamer_left_home, dtype=np.float64),
                right_home=np.asarray(args.hamer_right_home, dtype=np.float64),
                mirror_lr_across_xz=args.hamer_mirror_lr_xz,
                relative_compress=args.hamer_relative_compress,
                relative_scale=float(args.hamer_relative_scale),
                relative_clip_xyz=np.asarray(args.hamer_relative_clip, dtype=np.float64),
                debug_freeze_wrist_rotation=args.hamer_debug_freeze_wrist_rotation,
                debug=bool(args.debug),
            )
            hamer_hand_bridge = HamerHandBridge(target_bone_len=float(args.hamer_hand_target_bone_len))
        else:
            from teleop.input_source.egodex_input import EgoDexInputSource
            from teleop.input_source.hamer_adapter import HamerAdapter
            from teleop.input_source.hamer_bridge import HamerHandBridge
            from teleop.input_source.hamer_to_robot_frame import WristToEEConfig, load_wrist_to_ee_config_from_json

            wrist_cfg_default = WristToEEConfig(
                t_left=np.zeros(3, dtype=np.float64),
                R_left=_rpy_deg_to_R_xyz(args.wrist_left_rpy_deg),
                t_right=np.zeros(3, dtype=np.float64),
                R_right=_rpy_deg_to_R_xyz(args.wrist_right_rpy_deg),
            )
            wrist_cfg = wrist_cfg_default
            if args.wrist_to_ee_json:
                wrist_cfg = load_wrist_to_ee_config_from_json(args.wrist_to_ee_json, wrist_cfg_default)
            hamer_input = EgoDexInputSource(
                hdf5_path=args.egodex_hdf5,
                loop=args.egodex_loop,
                score_thresh=args.egodex_score_thresh,
                root_frame=args.egodex_root_frame,
                fps=args.egodex_fps,
                repo_wrist_basis=not args.egodex_arkit_wrist_rotation,
            )
            hamer_adapter = HamerAdapter(
                wrist_cfg,
                smooth_alpha=float(args.hamer_wrist_smooth_alpha),
                max_step_m=float(args.hamer_wrist_max_step_m),
                max_step_rad=float(args.hamer_wrist_max_step_rad),
                relative_position_mode=args.hamer_relative_pos,
                left_home=np.asarray(args.hamer_left_home, dtype=np.float64),
                right_home=np.asarray(args.hamer_right_home, dtype=np.float64),
                mirror_lr_across_xz=args.hamer_mirror_lr_xz,
                relative_compress=args.hamer_relative_compress,
                relative_scale=float(args.hamer_relative_scale),
                relative_clip_xyz=np.asarray(args.hamer_relative_clip, dtype=np.float64),
                debug_freeze_wrist_rotation=args.hamer_debug_freeze_wrist_rotation,
                debug=bool(args.debug),
            )
            hamer_hand_bridge = HamerHandBridge(target_bone_len=float(args.hamer_hand_target_bone_len))
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller" and args.input_source == "xr":
                loco_wrapper = LocoClientWrapper()
        else:
            # 真机：通过 MotionSwitcher 释放占用以便臂控；Isaac 等仿真无该服务，跳过以免 ClientStub 报错
            if not args.sim:
                motion_switcher = MotionSwitcher()
                status, result = motion_switcher.Enter_Debug_Mode()
                logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
            else:
                logger_mp.info("Simulation mode: skip MotionSwitcher / Enter_Debug_Mode")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        if args.disable_ik_joint_smoothing:
            if hasattr(arm_ik, "set_joint_smoothing_enabled"):
                arm_ik.set_joint_smoothing_enabled(False)
            else:
                arm_ik.enable_joint_smoothing = False

        # end-effector
        gripper_input_min = float(args.gripper_input_min)
        gripper_input_max = float(args.gripper_input_max)
        if (
            args.omnipicker_gripper_source in ("action", "task")
            and abs(gripper_input_min - 5.0) < 1e-12
            and abs(gripper_input_max - 7.0) < 1e-12
        ):
            # Omnipicker gripper sources are normalized in [0,1].
            gripper_input_min = 0.0
            gripper_input_max = 1.0
            logger_mp.info(
                "[omnipicker] gripper source enabled; auto-set gripper input range to [0, 1]."
            )

        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_gripper":
            from teleop.robot_control.robot_hand_inspire import Inspire_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # [output]
            dual_gripper_action_array = Array('d', 2, lock=False)  # [output]
            gripper_ctrl = Inspire_Gripper_Controller(
                left_gripper_value,
                right_gripper_value,
                dual_gripper_data_lock,
                dual_gripper_state_array,
                dual_gripper_action_array,
                simulation_mode=args.sim,
                input_min=gripper_input_min,
                input_max=gripper_input_max,
                open_cmd=float(args.inspire_gripper_open),
                close_cmd=float(args.inspire_gripper_close),
                smooth_alpha=float(args.inspire_gripper_alpha),
                max_speed=float(args.inspire_gripper_max_speed),
            )
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass

        use_hand_ee = args.input_mode == "hand" and args.ee in (
            "dex3", "inspire_dfx", "inspire_ftp", "brainco",
        )
        offline_skip_hand = args.input_source in ("hamer", "egodex") and args.hamer_arm_only
        if use_hand_ee and not offline_skip_hand:
            from teleop.utils.hand_pos_temporal_stab import DualHandPosTemporalStabilizer

            hand_pos_stab = DualHandPosTemporalStabilizer(
                hist_window=args.hand_stab_hist_window,
                hist_jump_thresh_m=args.hand_stab_hist_jump_m,
                pred_dev_thresh_m=args.hand_stab_pred_dev_m,
                enable_hist=not args.disable_hand_hist_stab,
                enable_pred=not args.disable_hand_pred_stab,
            )
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode: reset + sim_state via Isaac SHM (see unitree_sim_isaaclab/dds/*_dds.py)
        if args.sim:
            from teleop.utils.isaac_shm import SHM_RESET_POSE_CMD, SIZE_RESET_POSE, try_open_shm
            from teleop.utils.sim_state_topic import start_sim_state_shm_reader

            reset_pose_shm = try_open_shm(SHM_RESET_POSE_CMD, SIZE_RESET_POSE)
            sim_state_subscriber = start_sim_state_shm_reader()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if tv_wrapper is not None and camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        if hand_pos_stab is not None:
            hand_pos_stab.reset()
        arm_ctrl.speed_gradual_max()
        direct_joint_mode = (
            args.input_source == "hamer"
            and bool(args.parquet)
            and args.hamer_arm_control_source == "joint"
        )
        if direct_joint_mode:
            logger_mp.info(
                "[omnipicker] arm control source: joint (skip IK, direct ctrl_dual_arm from parquet)."
            )
        _joint_target_warned = False
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if tv_wrapper is not None and xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, publisher=reset_pose_publisher, reset_shm=reset_pose_shm)

            # get xr's tele data or HaMeR frame
            tele_data = None
            if args.input_source == "xr":
                tele_data = tv_wrapper.get_tele_data()
            hamer_frame = None
            if args.input_source in ("hamer", "egodex"):
                hamer_frame = hamer_input.get_frame()
                offline_loop = args.hamer_loop if args.input_source == "hamer" else args.egodex_loop
                if hamer_frame is None and not offline_loop:
                    logger_mp.info(f"{args.input_source} replay finished (no loop); stopping.")
                    STOP = True
                    break

            skip_hand_tele = args.hamer_arm_only
            if (
                tele_data is not None
                and (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco")
                and args.input_mode == "hand"
                and not skip_hand_tele
            ):
                l_hp = tele_data.left_hand_pos
                r_hp = tele_data.right_hand_pos
                if hand_pos_stab is not None:
                    l_hp, r_hp = hand_pos_stab.stabilize(l_hp, r_hp)
                if args.swap_hand_input:
                    l_hp, r_hp = r_hp, l_hp
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = np.asarray(l_hp, dtype=np.float64).reshape(75)
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = np.asarray(r_hp, dtype=np.float64).reshape(75)
            elif (
                args.input_source in ("hamer", "egodex")
                and hamer_frame is not None
                and (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco")
                and args.input_mode == "hand"
                and not skip_hand_tele
            ):
                left_hand_pos, right_hand_pos = hamer_hand_bridge.step(hamer_frame)
                if hand_pos_stab is not None:
                    left_hand_pos, right_hand_pos = hand_pos_stab.stabilize(
                        left_hand_pos, right_hand_pos
                    )
                if args.swap_hand_input:
                    left_hand_pos, right_hand_pos = right_hand_pos, left_hand_pos
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = right_hand_pos.flatten()
            elif tele_data is not None and args.ee in ("dex1", "inspire_gripper") and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif tele_data is not None and args.ee in ("dex1", "inspire_gripper") and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            elif (
                args.input_source in ("hamer", "egodex")
                and hamer_frame is not None
                and args.ee in ("dex1", "inspire_gripper")
                and args.input_mode == "hand"
                and not skip_hand_tele
            ):
                left_in = None
                right_in = None
                left_raw = hamer_frame.get("left", {}).get("gripper_input")
                right_raw = hamer_frame.get("right", {}).get("gripper_input")
                try:
                    if left_raw is not None and np.isfinite(float(left_raw)):
                        left_in = float(left_raw)
                except (TypeError, ValueError):
                    left_in = None
                try:
                    if right_raw is not None and np.isfinite(float(right_raw)):
                        right_in = float(right_raw)
                except (TypeError, ValueError):
                    right_in = None
                if left_in is None:
                    left_in = float(last_left_gripper_input)
                if right_in is None:
                    right_in = float(last_right_gripper_input)

                last_left_gripper_input = left_in
                last_right_gripper_input = right_in
                with left_gripper_value.get_lock():
                    left_gripper_value.value = left_in
                with right_gripper_value.get_lock():
                    right_gripper_value.value = right_in
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion and tele_data is not None:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            if args.input_source == "xr":
                lw = tele_data.left_wrist_pose
                rw = tele_data.right_wrist_pose
                if args.swap_hand_input:
                    lw, rw = rw, lw
                lw = _scale_tf_translation(lw, arm_left_home, args.arm_reach_scale)
                rw = _scale_tf_translation(rw, arm_right_home, args.arm_reach_scale)
                sol_q, sol_tauff = arm_ik.solve_ik(lw, rw, current_lr_arm_q, current_lr_arm_dq)
            elif args.input_source in ("hamer", "egodex"):
                if direct_joint_mode:
                    l_joint = None
                    r_joint = None
                    if hamer_frame is not None:
                        jt = hamer_frame.get("joint_target", {})
                        l_joint = jt.get("left")
                        r_joint = jt.get("right")
                    if l_joint is not None and r_joint is not None:
                        l_joint = np.asarray(l_joint, dtype=np.float64).reshape(-1)
                        r_joint = np.asarray(r_joint, dtype=np.float64).reshape(-1)
                        direct_q = np.concatenate([l_joint, r_joint], axis=0)
                        if direct_q.size == current_lr_arm_q.size:
                            sol_q = direct_q
                            sol_tauff = np.zeros_like(current_lr_arm_q, dtype=np.float64)
                        else:
                            if not _joint_target_warned:
                                logger_mp.warning(
                                    "[omnipicker] direct joint target size mismatch: "
                                    f"got {direct_q.size}, expected {current_lr_arm_q.size}. "
                                    "Fallback to keep current arm q."
                                )
                                _joint_target_warned = True
                            sol_q = current_lr_arm_q
                            sol_tauff = np.zeros_like(current_lr_arm_q, dtype=np.float64)
                    else:
                        if not _joint_target_warned:
                            logger_mp.warning(
                                "[omnipicker] direct joint mode has no joint_target in frame; "
                                "fallback to keep current arm q."
                            )
                            _joint_target_warned = True
                        sol_q = current_lr_arm_q
                        sol_tauff = np.zeros_like(current_lr_arm_q, dtype=np.float64)
                else:
                    targets = hamer_adapter.step(hamer_frame, current_lr_arm_q)
                    lt, rt = targets["left_arm"], targets["right_arm"]
                    if lt["valid"]:
                        last_hamer_left_tf = homogeneous_from_position_rotation(lt["p_ee_target"], lt["R_ee_target"])
                    if rt["valid"]:
                        last_hamer_right_tf = homogeneous_from_position_rotation(rt["p_ee_target"], rt["R_ee_target"])
                    ll_tf, rr_tf = last_hamer_left_tf, last_hamer_right_tf
                    if args.swap_hand_input:
                        ll_tf, rr_tf = rr_tf, ll_tf
                    ll_tf = _scale_tf_translation(ll_tf, arm_left_home, args.arm_reach_scale)
                    rr_tf = _scale_tf_translation(rr_tf, arm_right_home, args.arm_reach_scale)
                    sol_q, sol_tauff = arm_ik.solve_ik(ll_tf, rr_tf, current_lr_arm_q, current_lr_arm_dq)
                    if args.debug:
                        fk_rot = _compute_dual_fk_rotations(arm_ik, sol_q)
                        if lt.get("valid", False):
                            logger_mp.info(f"[debug][left] R_wrist_base=\n{lt.get('R_wrist_base', np.eye(3))}")
                            logger_mp.info(f"[debug][left] R_ee_target=\n{lt['R_ee_target']}")
                            if fk_rot is not None:
                                logger_mp.info(f"[debug][left] R_ee_fk=\n{fk_rot['left']}")
                        if rt.get("valid", False):
                            logger_mp.info(f"[debug][right] R_wrist_base=\n{rt.get('R_wrist_base', np.eye(3))}")
                            logger_mp.info(f"[debug][right] R_ee_target=\n{rt['R_ee_target']}")
                            if fk_rot is not None:
                                logger_mp.info(f"[debug][right] R_ee_fk=\n{fk_rot['right']}")
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "inspire_gripper") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "inspire_gripper") and args.input_mode == "controller" and tele_data is not None:
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        STOP = True
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        STOP = True
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        STOP = True
        try:
            if arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc and ipc_server is not None:
                ipc_server.stop()
            elif not args.ipc and listen_keyboard_thread is not None:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if hamer_input is not None and hasattr(hamer_input, "close"):
                hamer_input.close()
        except Exception as e:
            logger_mp.error(f"Failed to close offline input source: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim and sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        try:
            if reset_pose_shm is not None:
                reset_pose_shm.close()
        except Exception as e:
            logger_mp.error(f"Failed to close reset pose shm: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)


if __name__ == "__main__":
    main()