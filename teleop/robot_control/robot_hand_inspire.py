from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.isaac_shm import (
    SHM_INSPIRE_CMD,
    SHM_INSPIRE_STATE,
    SIZE_INSPIRE,
    try_open_shm,
)
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array
from multiprocessing import Value
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

Inspire_Num_Motors = 6
kTopicInspireDFXCommand = "rt/inspire/cmd"
kTopicInspireDFXState = "rt/inspire/state"


def _denormalize_inspire(idx: int, norm_val: float) -> float:
    n = float(np.clip(norm_val, 0.0, 1.0))
    if idx <= 3:
        return (1.0 - n) * 1.7
    if idx == 4:
        return (1.0 - n) * 0.5
    return (1.0 - n) * 1.4 - 0.1


class Inspire_Gripper_Controller:
    """
    Drive omnipicker dual two-finger grippers through Inspire channels.

    Input values are expected to behave like XR pinch distance in centimeters:
      - larger value -> more open hand
      - smaller value -> pinched hand
    The controller maps this to Inspire normalized command:
      - open_cmd (typically small, e.g. 0.05)
      - close_cmd (typically larger, e.g. 0.9)
    """

    def __init__(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        dual_gripper_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
        fps: float = 100.0,
        simulation_mode: bool = False,
        input_min: float = 5.0,
        input_max: float = 7.0,
        open_cmd: float = 0.05,
        close_cmd: float = 0.9,
        smooth_alpha: float = 0.2,
        max_speed: float = 1.5,
    ):
        logger_mp.info("Initialize Inspire_Gripper_Controller...")
        self.fps = float(fps)
        self.simulation_mode = bool(simulation_mode)

        self.input_min = float(input_min)
        self.input_max = float(input_max)
        if self.input_max <= self.input_min:
            self.input_max = self.input_min + 1e-3
        self.open_cmd = float(np.clip(open_cmd, 0.0, 1.2))
        self.close_cmd = float(np.clip(close_cmd, 0.0, 1.2))
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self.max_speed = float(max(0.0, max_speed))

        self._state_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)

        self.HandCmb_publisher = None
        self.HandState_subscriber = None
        self.inspire_state_shm = None
        self.inspire_cmd_shm = None
        self._inspire_state_ready = False

        if self.simulation_mode:
            self.inspire_state_shm = try_open_shm(SHM_INSPIRE_STATE, SIZE_INSPIRE)
            self.inspire_cmd_shm = try_open_shm(SHM_INSPIRE_CMD, SIZE_INSPIRE)
            logger_mp.info("[Inspire_Gripper_Controller] simulation mode: use shared memory state/cmd")
        else:
            self.HandCmb_publisher = ChannelPublisher(kTopicInspireDFXCommand, MotorCmds_)
            self.HandCmb_publisher.Init()
            self.HandState_subscriber = ChannelSubscriber(kTopicInspireDFXState, MotorStates_)
            self.HandState_subscriber.Init()

        self.left_gripper_state_value = Value('d', 0.0, lock=True)
        self.right_gripper_state_value = Value('d', 0.0, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state, daemon=True)
        self.subscribe_state_thread.start()

        while not self._inspire_state_ready:
            time.sleep(0.01)
            if self.simulation_mode:
                logger_mp.warning("[Inspire_Gripper_Controller] Waiting to read inspire shared memory...")
            else:
                logger_mp.warning("[Inspire_Gripper_Controller] Waiting to subscribe inspire dds...")

        self.control_thread_handle = threading.Thread(
            target=self.control_thread,
            args=(
                left_gripper_value_in,
                right_gripper_value_in,
                self.left_gripper_state_value,
                self.right_gripper_state_value,
                dual_gripper_data_lock,
                dual_gripper_state_out,
                dual_gripper_action_out,
            ),
            daemon=True,
        )
        self.control_thread_handle.start()
        logger_mp.info("Initialize Inspire_Gripper_Controller OK!")

    def _input_to_cmd(self, value: float) -> float:
        ratio = np.interp(float(value), [self.input_min, self.input_max], [0.0, 1.0])
        cmd = self.close_cmd + ratio * (self.open_cmd - self.close_cmd)
        return float(np.clip(cmd, 0.0, 1.2))

    def _subscribe_gripper_state(self):
        while True:
            if self.simulation_mode:
                if self.inspire_state_shm is None:
                    self.inspire_state_shm = try_open_shm(SHM_INSPIRE_STATE, SIZE_INSPIRE)
                    time.sleep(0.01)
                    continue
                msg = self.inspire_state_shm.read_data()
                if msg is not None:
                    q = msg.get("positions", [])
                    if len(q) >= 12:
                        left_mean = float(np.mean(np.asarray(q[6:12], dtype=np.float64)))
                        right_mean = float(np.mean(np.asarray(q[0:6], dtype=np.float64)))
                        self.left_gripper_state_value.value = left_mean
                        self.right_gripper_state_value.value = right_mean
                        self._inspire_state_ready = True
            else:
                hand_msg = self.HandState_subscriber.Read()
                if hand_msg is not None:
                    left_vals = []
                    right_vals = []
                    for i in range(Inspire_Num_Motors):
                        right_vals.append(float(hand_msg.states[i].q))
                        left_vals.append(float(hand_msg.states[i + Inspire_Num_Motors].q))
                    self.left_gripper_state_value.value = float(np.mean(left_vals))
                    self.right_gripper_state_value.value = float(np.mean(right_vals))
                    self._inspire_state_ready = True
            time.sleep(0.002)

    def _publish_cmd(self, left_cmd: float, right_cmd: float):
        if self.simulation_mode and self.inspire_cmd_shm is not None:
            cmd_data = {
                "positions": [float(right_cmd)] * Inspire_Num_Motors + [float(left_cmd)] * Inspire_Num_Motors,
                "velocities": [0.0] * (Inspire_Num_Motors * 2),
                "torques": [0.0] * (Inspire_Num_Motors * 2),
                "kp": [0.0] * (Inspire_Num_Motors * 2),
                "kd": [0.0] * (Inspire_Num_Motors * 2),
            }
            self.inspire_cmd_shm.write_data(cmd_data)
            return

        hand_msg = MotorCmds_()
        hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(Inspire_Num_Motors * 2)]
        for i in range(Inspire_Num_Motors):
            hand_msg.cmds[i].q = float(right_cmd)
            hand_msg.cmds[i + Inspire_Num_Motors].q = float(left_cmd)
        self.HandCmb_publisher.Write(hand_msg)

    def control_thread(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        left_gripper_state_value,
        right_gripper_state_value,
        dual_gripper_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
    ):
        period = 1.0 / max(self.fps, 1.0)
        max_delta = self.max_speed * period
        left_cmd = self.open_cmd
        right_cmd = self.open_cmd
        while True:
            t0 = time.time()
            with left_gripper_value_in.get_lock():
                left_in = float(left_gripper_value_in.value)
            with right_gripper_value_in.get_lock():
                right_in = float(right_gripper_value_in.value)

            left_target = self._input_to_cmd(left_in)
            right_target = self._input_to_cmd(right_in)

            left_target = left_cmd + self.smooth_alpha * (left_target - left_cmd)
            right_target = right_cmd + self.smooth_alpha * (right_target - right_cmd)

            left_cmd = float(np.clip(left_target, left_cmd - max_delta, left_cmd + max_delta))
            right_cmd = float(np.clip(right_target, right_cmd - max_delta, right_cmd + max_delta))

            self._publish_cmd(left_cmd, right_cmd)

            state = np.array([left_gripper_state_value.value, right_gripper_state_value.value], dtype=np.float64)
            self._state_filter.add_data(state)
            state_filtered = self._state_filter.filtered_data
            action = np.array([left_cmd, right_cmd], dtype=np.float64)

            if dual_gripper_state_out is not None and dual_gripper_action_out is not None and dual_gripper_data_lock is not None:
                with dual_gripper_data_lock:
                    dual_gripper_state_out[:] = state_filtered
                    dual_gripper_action_out[:] = action

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

class Inspire_Controller_DFX:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        logger_mp.info("Initialize Inspire_Controller_DFX...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)


        self.HandCmb_publisher = None
        self.HandState_subscriber = None
        self.inspire_state_shm = None
        self.inspire_cmd_shm = None
        if self.simulation_mode:
            self.inspire_state_shm = try_open_shm(SHM_INSPIRE_STATE, SIZE_INSPIRE)
            self.inspire_cmd_shm = try_open_shm(SHM_INSPIRE_CMD, SIZE_INSPIRE)
            logger_mp.info("[Inspire_Controller_DFX] simulation mode: use shared memory state/cmd")
        else:
            # initialize handcmd publisher and handstate subscriber
            self.HandCmb_publisher = ChannelPublisher(kTopicInspireDFXCommand, MotorCmds_)
            self.HandCmb_publisher.Init()
            self.HandState_subscriber = ChannelSubscriber(kTopicInspireDFXState, MotorStates_)
            self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array): # any(self.left_hand_state_array) and 
                break
            time.sleep(0.01)
            if self.simulation_mode:
                logger_mp.warning("[Inspire_Controller_DFX] Waiting to read inspire shared memory...")
            else:
                logger_mp.warning("[Inspire_Controller_DFX] Waiting to subscribe dds...")
        if self.simulation_mode:
            logger_mp.info("[Inspire_Controller_DFX] Shared memory inspire state ready.")
        else:
            logger_mp.info("[Inspire_Controller_DFX] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_DFX OK!")

    def _subscribe_hand_state(self):
        while True:
            if self.simulation_mode:
                if self.inspire_state_shm is None:
                    self.inspire_state_shm = try_open_shm(SHM_INSPIRE_STATE, SIZE_INSPIRE)
                    time.sleep(0.01)
                    continue
                msg = self.inspire_state_shm.read_data()
                if msg is not None:
                    q = msg.get("positions", [])
                    for i in range(Inspire_Num_Motors):
                        q_l = float(q[i + 6]) if i + 6 < len(q) else 0.0
                        q_r = float(q[i]) if i < len(q) else 0.0
                        # align with DDS normalized state q in [0,1]
                        if i <= 3:
                            self.left_hand_state_array[i] = np.clip((1.7 - q_l) / 1.7, 0.0, 1.0)
                            self.right_hand_state_array[i] = np.clip((1.7 - q_r) / 1.7, 0.0, 1.0)
                        elif i == 4:
                            self.left_hand_state_array[i] = np.clip((0.5 - q_l) / 0.5, 0.0, 1.0)
                            self.right_hand_state_array[i] = np.clip((0.5 - q_r) / 0.5, 0.0, 1.0)
                        else:
                            self.left_hand_state_array[i] = np.clip((1.3 - q_l) / 1.4, 0.0, 1.0)
                            self.right_hand_state_array[i] = np.clip((1.3 - q_r) / 1.4, 0.0, 1.0)
            else:
                hand_msg  = self.HandState_subscriber.Read()
                if hand_msg is not None:
                    for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                        self.left_hand_state_array[idx] = hand_msg.states[id].q
                    for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                        self.right_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):             
            self.hand_msg.cmds[id].q = left_q_target[idx]         
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):             
            self.hand_msg.cmds[id].q = right_q_target[idx] 

        if self.simulation_mode and self.inspire_cmd_shm is not None:
            right_pos = [_denormalize_inspire(i, right_q_target[i]) for i in range(Inspire_Num_Motors)]
            left_pos = [_denormalize_inspire(i, left_q_target[i]) for i in range(Inspire_Num_Motors)]
            cmd_data = {
                "positions": [float(v) for v in (right_pos + left_pos)],
                "velocities": [0.0] * (Inspire_Num_Motors * 2),
                "torques": [0.0] * (Inspire_Num_Motors * 2),
                "kp": [0.0] * (Inspire_Num_Motors * 2),
                "kd": [0.0] * (Inspire_Num_Motors * 2),
            }
            self.inspire_cmd_shm.write_data(cmd_data)
        else:
            self.HandCmb_publisher.Write(self.hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True

        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg  = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))]

        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # In website https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand, you can find
                    #     In the official document, the angles are in the range [0, 1] ==> 0.0: fully closed  1.0: fully open
                    # The q_target now is in radians, ranges:
                    #     - idx 0~3: 0~1.7 (1.7 = closed)
                    #     - idx 4:   0~0.5
                    #     - idx 5:  -0.1~1.3
                    # We normalize them using (max - value) / range
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller_DFX has been closed.")



kTopicInspireFTPLeftCommand   = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand  = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState  = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"

class Inspire_Controller_FTP:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        logger_mp.info("Initialize Inspire_Controller_FTP...")
        inspire_dds = None
        self.inspire_hand_default = None
        if not simulation_mode:
            from inspire_sdkpy import inspire_dds  # lazy import
            import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default
            self.inspire_hand_default = inspire_hand_default
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)


        self.LeftHandCmd_publisher = None
        self.RightHandCmd_publisher = None
        self.LeftHandState_subscriber = None
        self.RightHandState_subscriber = None
        self.inspire_state_shm = None
        self.inspire_cmd_shm = None
        if self.simulation_mode:
            self.inspire_state_shm = try_open_shm(SHM_INSPIRE_STATE, SIZE_INSPIRE)
            self.inspire_cmd_shm = try_open_shm(SHM_INSPIRE_CMD, SIZE_INSPIRE)
            logger_mp.info("[Inspire_Controller_FTP] simulation mode: use shared memory state/cmd")
        else:
            # Initialize hand command publishers
            self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
            self.LeftHandCmd_publisher.Init()
            self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
            self.RightHandCmd_publisher.Init()

            # Initialize hand state subscribers
            self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
            self.LeftHandState_subscriber.Init()
            self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
            self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait for initial DDS messages (optional, but good for ensuring connection)
        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0: # Print every second
                if self.simulation_mode:
                    logger_mp.info(f"[Inspire_Controller_FTP] Waiting to read hand states from SHM (L: {any(self.left_hand_state_array)}, R: {any(self.right_hand_state_array)})...")
                else:
                    logger_mp.info(f"[Inspire_Controller_FTP] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array)}, R: {any(self.right_hand_state_array)})...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500: # Timeout after 5 seconds
                logger_mp.warning("[Inspire_Controller_FTP] Warning: Timeout waiting for initial hand states. Proceeding anyway.")
                break
        logger_mp.info("[Inspire_Controller_FTP] Initial hand states received or timeout.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array, self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_FTP OK!\n")

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller_FTP] Subscribe thread started.")
        while True:
            if self.simulation_mode:
                if self.inspire_state_shm is None:
                    self.inspire_state_shm = try_open_shm(SHM_INSPIRE_STATE, SIZE_INSPIRE)
                    time.sleep(0.01)
                    continue
                msg = self.inspire_state_shm.read_data()
                if msg is not None:
                    q = msg.get("positions", [])
                    with self.left_hand_state_array.get_lock(), self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            if i + 6 < len(q):
                                q_l = float(q[i + 6])
                                if i <= 3:
                                    self.left_hand_state_array[i] = np.clip((1.7 - q_l) / 1.7, 0.0, 1.0)
                                elif i == 4:
                                    self.left_hand_state_array[i] = np.clip((0.5 - q_l) / 0.5, 0.0, 1.0)
                                else:
                                    self.left_hand_state_array[i] = np.clip((1.3 - q_l) / 1.4, 0.0, 1.0)
                            else:
                                self.left_hand_state_array[i] = 0.0

                            if i < len(q):
                                q_r = float(q[i])
                                if i <= 3:
                                    self.right_hand_state_array[i] = np.clip((1.7 - q_r) / 1.7, 0.0, 1.0)
                                elif i == 4:
                                    self.right_hand_state_array[i] = np.clip((0.5 - q_r) / 0.5, 0.0, 1.0)
                                else:
                                    self.right_hand_state_array[i] = np.clip((1.3 - q_r) / 1.4, 0.0, 1.0)
                            else:
                                self.right_hand_state_array[i] = 0.0
            else:
                left_state_msg = self.LeftHandState_subscriber.Read()
                if left_state_msg is not None:
                    if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                        with self.left_hand_state_array.get_lock():
                            for i in range(Inspire_Num_Motors):
                                self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0
                    else:
                        logger_mp.warning(f"[Inspire_Controller_FTP] Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}")
                right_state_msg = self.RightHandState_subscriber.Read()
                if right_state_msg is not None:
                    if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                        with self.right_hand_state_array.get_lock():
                            for i in range(Inspire_Num_Motors):
                                self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
                    else:
                        logger_mp.warning(f"[Inspire_Controller_FTP] Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}")

            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        if self.simulation_mode and self.inspire_cmd_shm is not None:
            left_norm = [float(np.clip(v / 1000.0, 0.0, 1.0)) for v in left_angle_cmd_scaled]
            right_norm = [float(np.clip(v / 1000.0, 0.0, 1.0)) for v in right_angle_cmd_scaled]
            right_pos = [_denormalize_inspire(i, right_norm[i]) for i in range(Inspire_Num_Motors)]
            left_pos = [_denormalize_inspire(i, left_norm[i]) for i in range(Inspire_Num_Motors)]
            cmd_data = {
                "positions": [float(v) for v in right_pos + left_pos],
                "velocities": [0.0] * (Inspire_Num_Motors * 2),
                "torques": [0.0] * (Inspire_Num_Motors * 2),
                "kp": [0.0] * (Inspire_Num_Motors * 2),
                "kd": [0.0] * (Inspire_Num_Motors * 2),
            }
            self.inspire_cmd_shm.write_data(cmd_data)
        else:
            # Left Hand Command
            left_cmd_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
            left_cmd_msg.angle_set = left_angle_cmd_scaled
            left_cmd_msg.mode = 0b0001 # Mode 1: Angle control
            self.LeftHandCmd_publisher.Write(left_cmd_msg)

            # Right Hand Command
            right_cmd_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
            right_cmd_msg.angle_set = right_angle_cmd_scaled
            right_cmd_msg.mode = 0b0001 # Mode 1: Angle control
            self.RightHandCmd_publisher.Write(right_cmd_msg)

        # 临时打开前 N 次的 log
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if self._debug_count < 50:
            logger_mp.info(f"[Inspire_Controller_FTP] Publish cmd L={left_angle_cmd_scaled} R={right_angle_cmd_scaled} ")
            self._debug_count += 1


    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        logger_mp.info("[Inspire_Controller_FTP] Control process started.")
        self.running = True

        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target]

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller_FTP has been closed.")

# Update hand state, according to the official documentation:
# 1. https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# 2. https://support.unitree.com/home/en/G1_developer/inspire_ftp_dexterity_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
