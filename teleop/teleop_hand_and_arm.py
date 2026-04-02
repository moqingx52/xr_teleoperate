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

def build_arg_parser():
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type=float, default=30.0, help='control and record\'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action='store_true', help='Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action='store_true', help='Enable isaac simulation mode')
    parser.add_argument('--ipc', action='store_true', help='Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action='store_true', help='Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action='store_true', help='Enable data recording mode')
    parser.add_argument('--task-dir', type=str, default='./utils/data/', help='path to save data')
    parser.add_argument('--task-name', type=str, default='pick cube', help='task file name for recording')
    parser.add_argument('--task-goal', type=str, default='pick up cube.', help='task goal for recording at json file')
    parser.add_argument('--task-desc', type=str, default='task description', help='task description for recording at json file')
    parser.add_argument('--task-steps', type=str, default='step1: do this; step2: do that;', help='task steps for recording at json file')
    # HaMeR / offline JSON input (与 hamer/demo.py: --structured_file 默认 hamer_structured_results.json)
    parser.add_argument('--input-source', '--input_source', type=str, default='xr', choices=['xr', 'hamer'],
                        dest='input_source', help='xr: TeleVuer; hamer: structured JSON replay')
    parser.add_argument('--hamer-json', '--hamer_json', type=str, default=None, dest='hamer_json',
                        help='Absolute or relative path to structured JSON (demo: out_folder/hamer_structured_results.json)')
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
                        help='Use first valid HaMeR wrist position as anchor; move around robot home positions')
    parser.add_argument('--hamer-mirror-lr-xz', '--hamer_mirror_lr_xz', action='store_true', dest='hamer_mirror_lr_xz',
                        help='Swap left/right HaMeR wrists and mirror across robot xz-plane (y->-y)')
    parser.add_argument('--hamer-relative-compress', '--hamer_relative_compress', action='store_true',
                        dest='hamer_relative_compress',
                        help='With --hamer-relative-pos: scale relative delta (--hamer-relative-scale) then clip; auto-on if scale/clip differ from defaults')
    parser.add_argument('--hamer-left-home', '--hamer_left_home', type=float, nargs=3, default=[0.25, 0.25, 0.1],
                        dest='hamer_left_home', metavar=('X', 'Y', 'Z'),
                        help='Left home position (meters) used by --hamer-relative-pos')
    parser.add_argument('--hamer-right-home', '--hamer_right_home', type=float, nargs=3, default=[0.25, -0.25, 0.1],
                        dest='hamer_right_home', metavar=('X', 'Y', 'Z'),
                        help='Right home position (meters) used by --hamer-relative-pos')
    parser.add_argument('--hamer-relative-scale', '--hamer_relative_scale', type=float, default=0.02,
                        dest='hamer_relative_scale',
                        help='Scale relative wrist delta (after anchor); needs --hamer-relative-pos + --hamer-relative-compress (auto-enabled if non-default)')
    parser.add_argument('--hamer-relative-clip', '--hamer_relative_clip', type=float, nargs=3, default=[0.12, 0.12, 0.10],
                        dest='hamer_relative_clip', metavar=('DX', 'DY', 'DZ'),
                        help='Axis clip (m) on scaled relative delta; same prerequisites as --hamer-relative-scale (auto-enabled if non-default)')
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
        default=0.03,
        dest='hamer_wrist_max_step_m',
        help='HamerAdapter: max wrist/EE translation per control tick (m); smaller = smaller motion range per frame (default 0.03)',
    )
    parser.add_argument(
        '--hamer-wrist-max-step-rad',
        '--hamer_wrist_max_step_rad',
        type=float,
        default=0.2,
        dest='hamer_wrist_max_step_rad',
        help='HamerAdapter: max wrist/EE rotation per tick (rad); smaller = slower orientation changes (default 0.2)',
    )
    parser.add_argument(
        '--hamer-wrist-smooth-alpha',
        '--hamer_wrist_smooth_alpha',
        type=float,
        default=0.2,
        dest='hamer_wrist_smooth_alpha',
        help='HamerAdapter: EMA blend toward new wrist target (0..1); lower = smoother, slower to follow (default 0.2)',
    )
    parser.add_argument('--swap-hand-input', '--swap_hand_input', action='store_true', dest='swap_hand_input',
                        help='Exchange left/right hand skeleton and wrist poses before dex retargeting / arm IK (dataset vs robot LR mismatch)')
    return parser


_DEFAULT_HAMER_REL_SCALE = 0.02
_DEFAULT_HAMER_REL_CLIP = np.array([0.12, 0.12, 0.10], dtype=np.float64)


def _normalize_hamer_relative_args(args: argparse.Namespace) -> None:
    """
    HamerAdapter 只在 relative_position_mode 且 (relative_compress 或非默认 scale/clip 意图) 下才用 scale/clip；
    此前若只传 --hamer-relative-scale/--hamer-relative-clip 会被静默忽略。
    """
    if args.input_source != "hamer":
        return
    clip = np.asarray(args.hamer_relative_clip, dtype=np.float64).reshape(3)
    scale = float(args.hamer_relative_scale)
    scale_changed = abs(scale - _DEFAULT_HAMER_REL_SCALE) > 1e-12
    clip_changed = bool(np.any(np.abs(clip - _DEFAULT_HAMER_REL_CLIP) > 1e-12))
    wants_compress_pipeline = bool(args.hamer_relative_compress) or scale_changed or clip_changed
    if not wants_compress_pipeline:
        return
    if not args.hamer_relative_pos:
        logger_mp.warning(
            "[HaMeR] --hamer-relative-scale / --hamer-relative-clip / --hamer-relative-compress require "
            "--hamer-relative-pos (anchor + offset around home). Enabling --hamer-relative-pos so they take effect."
        )
        args.hamer_relative_pos = True
    if not args.hamer_relative_compress and (scale_changed or clip_changed):
        logger_mp.warning(
            "[HaMeR] --hamer-relative-scale/--hamer-relative-clip differ from defaults but compress was off; "
            "enabling --hamer-relative-compress."
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
        elif args.hamer_out_dir:
            args.hamer_json = os.path.join(
                os.path.abspath(os.path.expanduser(args.hamer_out_dir)),
                args.hamer_structured_file,
            )
        _normalize_hamer_relative_args(args)
    logger_mp.info(f"args: {args}")
    if args.input_source == "hamer" and not args.hamer_json:
        logger_mp.error("HaMeR mode requires --hamer-json/--hamer_json or --hamer-out-dir/--hamer_out_dir")
        exit(1)
    if args.input_source == "hamer" and args.hamer_cam2base_json:
        args.hamer_cam2base_json = os.path.abspath(os.path.expanduser(args.hamer_cam2base_json))
    if args.input_source == "hamer" and args.motion and args.input_mode == "controller":
        logger_mp.warning("HaMeR input does not drive locomotion; motion+controller may be inactive.")

    arm_ctrl = None
    sim_state_subscriber = None
    reset_pose_publisher = None
    reset_pose_shm = None
    ipc_server = None
    listen_keyboard_thread = None
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
        if args.input_source == "hamer":
            xr_need_local_img = False

        tv_wrapper = None
        hamer_input = None
        hamer_adapter = None
        hamer_hand_bridge = None
        last_hamer_left_tf = np.eye(4, dtype=np.float64)
        last_hamer_left_tf[:3, 3] = np.array([0.25, 0.25, 0.1], dtype=np.float64)
        last_hamer_right_tf = np.eye(4, dtype=np.float64)
        last_hamer_right_tf[:3, 3] = np.array([0.25, -0.25, 0.1], dtype=np.float64)

        if args.input_source == "xr":
            tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand",
                                         binocular=camera_config['head_camera']['binocular'],
                                         img_shape=camera_config['head_camera']['image_shape'],
                                         display_mode=args.display_mode,
                                         zmq=camera_config['head_camera']['enable_zmq'],
                                         webrtc=camera_config['head_camera']['enable_webrtc'],
                                         webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                         )
        else:
            from teleop.input_source.hamer_input import HamerInputSource
            from teleop.input_source.hamer_adapter import HamerAdapter
            from teleop.input_source.hamer_bridge import HamerHandBridge
            from teleop.input_source.hamer_to_robot_frame import WristToEEConfig

            hamer_input = HamerInputSource(
                json_path=args.hamer_json,
                score_thresh=args.hamer_score_thresh,
                loop=args.hamer_loop,
                frame_offset_json=args.hamer_frame_offset_json,
                cam2base_json=args.hamer_cam2base_json,
            )
            hamer_adapter = HamerAdapter(
                WristToEEConfig.identity(),
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

        # end-effector
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
        arm_ctrl.speed_gradual_max()
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
            if args.input_source == "hamer":
                hamer_frame = hamer_input.get_frame()
                if hamer_frame is None and not args.hamer_loop:
                    logger_mp.info("HaMeR JSON finished (no loop); stopping.")
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
                if args.swap_hand_input:
                    l_hp, r_hp = r_hp, l_hp
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = np.asarray(l_hp, dtype=np.float64).reshape(75)
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = np.asarray(r_hp, dtype=np.float64).reshape(75)
            elif (
                args.input_source == "hamer"
                and hamer_frame is not None
                and (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco")
                and args.input_mode == "hand"
                and not skip_hand_tele
            ):
                left_hand_pos, right_hand_pos = hamer_hand_bridge.step(hamer_frame)
                if args.swap_hand_input:
                    left_hand_pos, right_hand_pos = right_hand_pos, left_hand_pos
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = right_hand_pos.flatten()
            elif tele_data is not None and args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif tele_data is not None and args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
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
                sol_q, sol_tauff = arm_ik.solve_ik(lw, rw, current_lr_arm_q, current_lr_arm_dq)
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
                sol_q, sol_tauff = arm_ik.solve_ik(ll_tf, rr_tf, current_lr_arm_q, current_lr_arm_dq)
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
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller" and tele_data is not None:
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