import numpy as np
import sys
import threading
import time
from enum import IntEnum

import unitree_sdk2py

from teleop.utils.isaac_shm import (
    SHM_ROBOT_CMD,
    SHM_ROBOT_STATE,
    SIZE_ROBOT,
    try_open_shm,
)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import ( LowCmd_  as go_LowCmd, LowState_ as go_LowState)  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20

DEFAULT_G1_29_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

OMNIPICKER_G1_29_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

A2D_OMNIPICKER_G1_29_ARM_JOINT_NAMES = [
    "Joint1_l",
    "Joint2_l",
    "Joint3_l",
    "Joint4_l",
    "Joint5_l",
    "Joint6_l",
    "Joint7_l",
    "Joint1_r",
    "Joint2_r",
    "Joint3_r",
    "Joint4_r",
    "Joint5_r",
    "Joint6_r",
    "Joint7_r",
]

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class G1_23_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_23_Num_Motors)]

class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

class H1_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_Num_Motors)]

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_list_maybe(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return None


def _extract_sim_joint_state(msg):
    """Best-effort parser for sim shm payloads with different field conventions."""
    if not isinstance(msg, dict):
        return None, None, None

    candidates = [msg]
    for key in ("robot_state", "state", "lowstate", "data", "payload"):
        nested = msg.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    q_keys = ("joint_positions", "positions", "joint_pos", "joint_position", "q")
    dq_keys = ("joint_velocities", "velocities", "joint_vel", "joint_velocity", "dq")
    name_keys = ("joint_names", "names")

    def _extract_from_motor_state(payload_dict):
        for key in ("motor_state", "motor_states", "motorState", "state"):
            seq = payload_dict.get(key)
            if not isinstance(seq, list) or len(seq) == 0:
                continue
            q_out = []
            dq_out = []
            found_any = False
            for item in seq:
                if isinstance(item, dict):
                    q_val = item.get("q", item.get("position", item.get("pos")))
                    dq_val = item.get("dq", item.get("velocity", item.get("vel")))
                else:
                    q_val = getattr(item, "q", None)
                    dq_val = getattr(item, "dq", None)
                q_out.append(q_val)
                dq_out.append(dq_val)
                if q_val is not None:
                    found_any = True
            if found_any:
                return q_out, dq_out
        return None, None

    for payload in candidates:
        q = None
        dq = None
        joint_names = None

        for key in q_keys:
            q = _as_list_maybe(payload.get(key))
            if q is not None:
                break
        if q is None:
            q, dq = _extract_from_motor_state(payload)
            if q is None:
                continue

        if dq is None:
            for key in dq_keys:
                dq = _as_list_maybe(payload.get(key))
                if dq is not None:
                    break

        for key in name_keys:
            joint_names = _as_list_maybe(payload.get(key))
            if joint_names is not None:
                break

        return q, dq, joint_names

    return None, None, None


def _sanitize_lowcmd_for_crc(msg, controller_name):
    for idx, cmd in enumerate(msg.motor_cmd):
        for field in ("q", "dq", "tau", "kp", "kd"):
            v = getattr(cmd, field, None)
            safe_v = _safe_float(v, 0.0)
            if v is None:
                logger_mp.error(f"[{controller_name}] motor_cmd[{idx}].{field} is None before CRC, force 0.0")
            elif safe_v != v:
                logger_mp.error(f"[{controller_name}] motor_cmd[{idx}].{field} invalid={v!r} before CRC, force 0.0")
            setattr(cmd, field, safe_v)


class G1_29_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        logger_mp.info("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self.lowstate_shm = None
        self.lowcmd_shm = None
        if self.simulation_mode:
            self.lowstate_shm = try_open_shm(SHM_ROBOT_STATE, SIZE_ROBOT)
            self.lowcmd_shm = try_open_shm(SHM_ROBOT_CMD, SIZE_ROBOT)
            print("[G1_29_ArmController] simulation mode: use shared memory state/cmd")
        else:
            if self.motion_mode:
                self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
            else:
                self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
            self.lowcmd_publisher.Init()
            self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
            self.lowstate_subscriber.Init()
            print(
                "[G1_29_ArmController] lowstate subscriber: "
                f"topic={kTopicLowState!r} type={hg_LowState!r} typename={getattr(hg_LowState, '__name__', hg_LowState)}"
            )
            print(f"[G1_29_ArmController] runtime: python={sys.executable}")
            print(f"[G1_29_ArmController] runtime: unitree_sdk2py={unitree_sdk2py.__file__}")
            print(
                "[G1_29_ArmController] runtime: "
                f"LowState_ module={hg_LowState.__module__} LowState_={hg_LowState}"
            )
        self.lowstate_buffer = DataBuffer()
        self._lowstate_read_hit_logged = False
        self._last_q = [0.0] * G1_29_Num_Motors
        self._last_dq = [0.0] * G1_29_Num_Motors
        self._sim_lowstate_none_count = 0
        self._sim_joint_name_to_slot = None
        self._sim_joint_mapping_name = "index"
        self._sim_missing_dq_warned = False
        self._sim_unexpected_payload_warned = False

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        wait_log_counter = 0
        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            wait_log_counter += 1
            if wait_log_counter == 1 or wait_log_counter % 50 == 0:
                waited_s = 0.1 * wait_log_counter
                if self.simulation_mode:
                    logger_mp.warning(
                        "[G1_29_ArmController] Waiting to read lowstate shared memory... (%.1fs)",
                        waited_s,
                    )
                else:
                    logger_mp.warning(
                        "[G1_29_ArmController] Waiting to subscribe dds... (%.1fs)",
                        waited_s,
                    )
        if self.simulation_mode:
            logger_mp.info("[G1_29_ArmController] Shared memory lowstate ready.")
        else:
            logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        self.q_target = self.get_current_dual_arm_q().copy()
        self.tauff_target = np.zeros_like(self.q_target, dtype=np.float64)
        logger_mp.debug(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.debug(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = _safe_float(self.all_motor_q[id], 0.0)
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_ArmController OK!")

    def _build_sim_joint_name_mapping(self, joint_names):
        names = [str(n) for n in (joint_names or [])]
        name_set = set(names)
        arm_slots = [int(m.value) for m in G1_29_JointArmIndex]
        if all(n in name_set for n in A2D_OMNIPICKER_G1_29_ARM_JOINT_NAMES):
            mapping = {
                joint_name: arm_slots[idx]
                for idx, joint_name in enumerate(A2D_OMNIPICKER_G1_29_ARM_JOINT_NAMES)
            }
            self._sim_joint_mapping_name = "a2d-urdf-named"
            logger_mp.info(
                "[G1_29_ArmController] simulation state uses A2D URDF arm joint names."
            )
            return mapping
        if all(n in name_set for n in OMNIPICKER_G1_29_ARM_JOINT_NAMES):
            mapping = {
                joint_name: arm_slots[idx]
                for idx, joint_name in enumerate(OMNIPICKER_G1_29_ARM_JOINT_NAMES)
            }
            self._sim_joint_mapping_name = "omnipicker-named"
            logger_mp.info(
                "[G1_29_ArmController] simulation state uses omnipicker named arm joints."
            )
            return mapping
        if all(n in name_set for n in DEFAULT_G1_29_ARM_JOINT_NAMES):
            mapping = {
                joint_name: arm_slots[idx]
                for idx, joint_name in enumerate(DEFAULT_G1_29_ARM_JOINT_NAMES)
            }
            self._sim_joint_mapping_name = "g1-named"
            logger_mp.info(
                "[G1_29_ArmController] simulation state uses default G1 named arm joints."
            )
            return mapping
        return None

    def _update_sim_state_from_named_joints(self, q_safe, dq_safe, joint_names):
        if not isinstance(joint_names, list) or len(joint_names) == 0:
            return False
        if self._sim_joint_name_to_slot is None:
            self._sim_joint_name_to_slot = self._build_sim_joint_name_mapping(joint_names)
        mapping = self._sim_joint_name_to_slot
        if not mapping:
            return False
        name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
        for joint_name, slot in mapping.items():
            idx = name_to_idx.get(joint_name, None)
            if idx is None:
                continue
            if idx < len(q_safe) and q_safe[idx] is not None:
                self._last_q[slot] = _safe_float(q_safe[idx], self._last_q[slot])
            if idx < len(dq_safe) and dq_safe[idx] is not None:
                self._last_dq[slot] = _safe_float(dq_safe[idx], self._last_dq[slot])
        return True

    def _subscribe_motor_state(self):
        _read_n = 0
        while True:
            _read_n += 1
            if self.simulation_mode:
                if self.lowstate_shm is None:
                    self.lowstate_shm = try_open_shm(SHM_ROBOT_STATE, SIZE_ROBOT)
                    if self.lowstate_shm is None:
                        if _read_n <= 3 or _read_n % 5000 == 0:
                            print(f"[G1_29_ArmController] lowstate shm not ready (poll n={_read_n})")
                        time.sleep(0.01)
                        continue
                msg = self.lowstate_shm.read_data() if self.lowstate_shm else None
                if msg is not None:
                    self._sim_lowstate_none_count = 0
                    try:
                        if not self._lowstate_read_hit_logged:
                            print("[G1_29_ArmController] lowstate shm read hit")
                            self._lowstate_read_hit_logged = True
                        q, dq, joint_names = _extract_sim_joint_state(msg)
                        if q is None:
                            if (not self._sim_unexpected_payload_warned) and isinstance(msg, dict):
                                print(
                                    "[G1_29_ArmController] simulation state unexpected keys:",
                                    sorted(list(msg.keys()))[:16],
                                )
                                logger_mp.warning(
                                    "[G1_29_ArmController] simulation state payload has no joint position field, keys=%s",
                                    sorted(list(msg.keys()))[:16],
                                )
                                self._sim_unexpected_payload_warned = True
                            time.sleep(0.002)
                            continue
                        if dq is None:
                            if not self._sim_missing_dq_warned:
                                logger_mp.warning(
                                    "[G1_29_ArmController] simulation state missing velocity field; fallback to previous/zero dq."
                                )
                                self._sim_missing_dq_warned = True
                            dq = []
                        lowstate = G1_29_LowState()
                        q_safe = list(q)
                        dq_safe = list(dq)
                        used_named_mapping = self._update_sim_state_from_named_joints(
                            q_safe, dq_safe, joint_names
                        )
                        if not used_named_mapping:
                            for id in range(G1_29_Num_Motors):
                                if id < len(q_safe) and q_safe[id] is not None:
                                    self._last_q[id] = _safe_float(q_safe[id], self._last_q[id])
                                if id < len(dq_safe) and dq_safe[id] is not None:
                                    self._last_dq[id] = _safe_float(dq_safe[id], self._last_dq[id])
                        for id in range(G1_29_Num_Motors):
                            lowstate.motor_state[id].q = self._last_q[id]
                            lowstate.motor_state[id].dq = self._last_dq[id]
                        self.lowstate_buffer.SetData(lowstate)
                    except Exception as e:
                        print(f"[G1_29_ArmController] lowstate shm parse failed: {e!r}")
                elif _read_n <= 3 or _read_n % 5000 == 0:
                    print(f"[G1_29_ArmController] lowstate shm Read returned None (poll n={_read_n})")
                    self._sim_lowstate_none_count += 1
                    if self._sim_lowstate_none_count >= 500:
                        # Recover from stale/broken shm attachment without restarting Isaac.
                        print("[G1_29_ArmController] lowstate shm stalled; reconnecting...")
                        try:
                            if self.lowstate_shm is not None:
                                self.lowstate_shm.close()
                        except Exception:
                            pass
                        self.lowstate_shm = None
                        self._sim_lowstate_none_count = 0
                time.sleep(0.002)
                continue

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                try:
                    if not self._lowstate_read_hit_logged:
                        print("[G1_29_ArmController] lowstate callback hit")
                        self._lowstate_read_hit_logged = True
                    lowstate = G1_29_LowState()
                    for id in range(G1_29_Num_Motors):
                        lowstate.motor_state[id].q  = msg.motor_state[id].q
                        lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    self.lowstate_buffer.SetData(lowstate)
                except Exception as e:
                    print(
                        f"[G1_29_ArmController] lowstate callback hit but buffer write failed: {e!r}"
                    )
            elif _read_n <= 3 or _read_n % 5000 == 0:
                print(f"[G1_29_ArmController] lowstate Read returned None (poll n={_read_n})")
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = _safe_float(cliped_arm_q_target[idx], 0.0)
                self.msg.motor_cmd[id].dq = 0.0
                self.msg.motor_cmd[id].tau = _safe_float(arm_tauff_target[idx], 0.0)

            _sanitize_lowcmd_for_crc(self.msg, "G1_29_ArmController")
            self.msg.crc = self.crc.Crc(self.msg)
            if self.simulation_mode and self.lowcmd_shm is not None:
                num_cmd_motors = len(self.msg.motor_cmd)
                positions = [float(self.msg.motor_cmd[i].q) for i in range(num_cmd_motors)]
                velocities = [float(self.msg.motor_cmd[i].dq) for i in range(num_cmd_motors)]
                torques = [float(self.msg.motor_cmd[i].tau) for i in range(num_cmd_motors)]
                kp = [float(self.msg.motor_cmd[i].kp) for i in range(num_cmd_motors)]
                kd = [float(self.msg.motor_cmd[i].kd) for i in range(num_cmd_motors)]
                cmd_data = {
                    "mode_pr": int(self.msg.mode_pr),
                    "mode_machine": int(self.msg.mode_machine),
                    # Keep a top-level alias for consumers that read direct positions.
                    "positions": positions,
                    "motor_cmd": {
                        "positions": positions,
                        "velocities": velocities,
                        "torques": torques,
                        "kp": kp,
                        "kd": kd,
                    },
                }
                self.lowcmd_shm.write_data(cmd_data)
            else:
                if self.simulation_mode and self.lowcmd_shm is None:
                    self.lowcmd_shm = try_open_shm(SHM_ROBOT_CMD, SIZE_ROBOT)
                elif self.lowcmd_publisher is not None:
                    self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        if self.simulation_mode:
            return 0
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[G1_29_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class G1_23_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode

        logger_mp.info("Initialize G1_23_ArmController...")
        self.q_target = np.zeros(10)
        self.tauff_target = np.zeros(10)

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self.lowstate_shm = None
        self.lowcmd_shm = None
        if self.simulation_mode:
            self.lowstate_shm = try_open_shm(SHM_ROBOT_STATE, SIZE_ROBOT)
            self.lowcmd_shm = try_open_shm(SHM_ROBOT_CMD, SIZE_ROBOT)
            print("[G1_23_ArmController] simulation mode: use shared memory state/cmd")
        else:
            if self.motion_mode:
                self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
            else:
                self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
            self.lowcmd_publisher.Init()
            self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
            self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()
        self._lowstate_read_hit_logged = False
        self._sim_missing_dq_warned = False
        self._sim_unexpected_payload_warned = False

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        wait_log_counter = 0
        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            wait_log_counter += 1
            if wait_log_counter == 1 or wait_log_counter % 50 == 0:
                waited_s = 0.1 * wait_log_counter
                if self.simulation_mode:
                    logger_mp.warning(
                        "[G1_23_ArmController] Waiting to read lowstate shared memory... (%.1fs)",
                        waited_s,
                    )
                else:
                    logger_mp.warning(
                        "[G1_23_ArmController] Waiting to subscribe dds... (%.1fs)",
                        waited_s,
                    )
        if self.simulation_mode:
            logger_mp.info("[G1_23_ArmController] Shared memory lowstate ready.")
        else:
            logger_mp.info("[G1_23_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        self.q_target = self.get_current_dual_arm_q().copy()
        self.tauff_target = np.zeros_like(self.q_target, dtype=np.float64)
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_23_JointArmIndex)
        for id in G1_23_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_23_ArmController OK!")

    def _subscribe_motor_state(self):
        _read_n = 0
        while True:
            _read_n += 1
            if self.simulation_mode:
                if self.lowstate_shm is None:
                    self.lowstate_shm = try_open_shm(SHM_ROBOT_STATE, SIZE_ROBOT)
                    if self.lowstate_shm is None:
                        if _read_n <= 3 or _read_n % 5000 == 0:
                            print(f"[G1_23_ArmController] lowstate shm not ready (poll n={_read_n})")
                        time.sleep(0.01)
                        continue
                msg = self.lowstate_shm.read_data()
                if msg is not None:
                    try:
                        q, dq, _ = _extract_sim_joint_state(msg)
                        if q is None:
                            if (not self._sim_unexpected_payload_warned) and isinstance(msg, dict):
                                print(
                                    "[G1_23_ArmController] simulation state unexpected keys:",
                                    sorted(list(msg.keys()))[:16],
                                )
                                logger_mp.warning(
                                    "[G1_23_ArmController] simulation state payload has no joint position field, keys=%s",
                                    sorted(list(msg.keys()))[:16],
                                )
                                self._sim_unexpected_payload_warned = True
                            time.sleep(0.002)
                            continue
                        if dq is None:
                            if not self._sim_missing_dq_warned:
                                logger_mp.warning(
                                    "[G1_23_ArmController] simulation state missing velocity field; fallback to zero dq."
                                )
                                self._sim_missing_dq_warned = True
                            dq = []
                        lowstate = G1_23_LowState()
                        n_q = min(G1_23_Num_Motors, len(q))
                        n_dq = min(G1_23_Num_Motors, len(dq))
                        for id in range(n_q):
                            lowstate.motor_state[id].q = q[id]
                        for id in range(n_dq):
                            lowstate.motor_state[id].dq = dq[id]
                        self.lowstate_buffer.SetData(lowstate)
                    except Exception as e:
                        print(f"[G1_23_ArmController] lowstate shm parse failed: {e!r}")
                time.sleep(0.002)
                continue
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_23_LowState()
                for id in range(G1_23_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(G1_23_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            if self.simulation_mode and self.lowcmd_shm is not None:
                num_cmd_motors = len(self.msg.motor_cmd)
                cmd_data = {
                    "mode_pr": int(self.msg.mode_pr),
                    "mode_machine": int(self.msg.mode_machine),
                    "motor_cmd": {
                        "positions": [float(self.msg.motor_cmd[i].q) for i in range(num_cmd_motors)],
                        "velocities": [float(self.msg.motor_cmd[i].dq) for i in range(num_cmd_motors)],
                        "torques": [float(self.msg.motor_cmd[i].tau) for i in range(num_cmd_motors)],
                        "kp": [float(self.msg.motor_cmd[i].kp) for i in range(num_cmd_motors)],
                        "kd": [float(self.msg.motor_cmd[i].kd) for i in range(num_cmd_motors)],
                    },
                }
                self.lowcmd_shm.write_data(cmd_data)
            else:
                if self.simulation_mode and self.lowcmd_shm is None:
                    self.lowcmd_shm = try_open_shm(SHM_ROBOT_CMD, SIZE_ROBOT)
                elif self.lowcmd_publisher is not None:
                    self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        if self.simulation_mode:
            return 0
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_23_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_23_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(10)
            # self.tauff_target = np.zeros(10)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[G1_23_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_23_JointIndex.kLeftAnklePitch.value,
            G1_23_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_23_JointIndex.kLeftShoulderPitch.value,
            G1_23_JointIndex.kLeftShoulderRoll.value,
            G1_23_JointIndex.kLeftShoulderYaw.value,
            G1_23_JointIndex.kLeftElbow.value,
            # Right arm
            G1_23_JointIndex.kRightShoulderPitch.value,
            G1_23_JointIndex.kRightShoulderRoll.value,
            G1_23_JointIndex.kRightShoulderYaw.value,
            G1_23_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_23_JointIndex.kLeftWristRoll.value,
            G1_23_JointIndex.kRightWristRoll.value,
        ]
        return motor_index.value in wrist_motors

class G1_23_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26

class G1_23_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRollNotUsed = 13
    kWaistPitchNotUsed = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitchNotUsed = 20
    kLeftWristyawNotUsed = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitchNotUsed = 27
    kRightWristYawNotUsed = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class H1_2_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode
        
        logger_mp.info("Initialize H1_2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self.lowstate_shm = None
        self.lowcmd_shm = None
        if self.simulation_mode:
            self.lowstate_shm = try_open_shm(SHM_ROBOT_STATE, SIZE_ROBOT)
            self.lowcmd_shm = try_open_shm(SHM_ROBOT_CMD, SIZE_ROBOT)
            print("[H1_2_ArmController] simulation mode: use shared memory state/cmd")
        else:
            if self.motion_mode:
                self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
            else:
                self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
            self.lowcmd_publisher.Init()
            self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
            self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()
        self._lowstate_read_hit_logged = False
        self._sim_missing_dq_warned = False
        self._sim_unexpected_payload_warned = False

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        wait_log_counter = 0
        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            wait_log_counter += 1
            if wait_log_counter == 1 or wait_log_counter % 50 == 0:
                waited_s = 0.1 * wait_log_counter
                if self.simulation_mode:
                    logger_mp.warning(
                        "[H1_2_ArmController] Waiting to read lowstate shared memory... (%.1fs)",
                        waited_s,
                    )
                else:
                    logger_mp.warning(
                        "[H1_2_ArmController] Waiting to subscribe dds... (%.1fs)",
                        waited_s,
                    )
        if self.simulation_mode:
            logger_mp.info("[H1_2_ArmController] Shared memory lowstate ready.")
        else:
            logger_mp.info("[H1_2_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        self.q_target = self.get_current_dual_arm_q().copy()
        self.tauff_target = np.zeros_like(self.q_target, dtype=np.float64)
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in H1_2_JointArmIndex)
        for id in H1_2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_2_ArmController OK!")

    def _subscribe_motor_state(self):
        _read_n = 0
        while True:
            _read_n += 1
            if self.simulation_mode:
                if self.lowstate_shm is None:
                    self.lowstate_shm = try_open_shm(SHM_ROBOT_STATE, SIZE_ROBOT)
                    if self.lowstate_shm is None:
                        if _read_n <= 3 or _read_n % 5000 == 0:
                            print(f"[H1_2_ArmController] lowstate shm not ready (poll n={_read_n})")
                        time.sleep(0.01)
                        continue
                msg = self.lowstate_shm.read_data()
                if msg is not None:
                    try:
                        q, dq, _ = _extract_sim_joint_state(msg)
                        if q is None:
                            if (not self._sim_unexpected_payload_warned) and isinstance(msg, dict):
                                print(
                                    "[H1_2_ArmController] simulation state unexpected keys:",
                                    sorted(list(msg.keys()))[:16],
                                )
                                logger_mp.warning(
                                    "[H1_2_ArmController] simulation state payload has no joint position field, keys=%s",
                                    sorted(list(msg.keys()))[:16],
                                )
                                self._sim_unexpected_payload_warned = True
                            time.sleep(0.002)
                            continue
                        if dq is None:
                            if not self._sim_missing_dq_warned:
                                logger_mp.warning(
                                    "[H1_2_ArmController] simulation state missing velocity field; fallback to zero dq."
                                )
                                self._sim_missing_dq_warned = True
                            dq = []
                        lowstate = H1_2_LowState()
                        n_q = min(H1_2_Num_Motors, len(q))
                        n_dq = min(H1_2_Num_Motors, len(dq))
                        for id in range(n_q):
                            lowstate.motor_state[id].q = q[id]
                        for id in range(n_dq):
                            lowstate.motor_state[id].dq = dq[id]
                        self.lowstate_buffer.SetData(lowstate)
                    except Exception as e:
                        print(f"[H1_2_ArmController] lowstate shm parse failed: {e!r}")
                time.sleep(0.002)
                continue
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_2_LowState()
                for id in range(H1_2_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(H1_2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            if self.simulation_mode and self.lowcmd_shm is not None:
                num_cmd_motors = len(self.msg.motor_cmd)
                cmd_data = {
                    "mode_pr": int(self.msg.mode_pr),
                    "mode_machine": int(self.msg.mode_machine),
                    "motor_cmd": {
                        "positions": [float(self.msg.motor_cmd[i].q) for i in range(num_cmd_motors)],
                        "velocities": [float(self.msg.motor_cmd[i].dq) for i in range(num_cmd_motors)],
                        "torques": [float(self.msg.motor_cmd[i].tau) for i in range(num_cmd_motors)],
                        "kp": [float(self.msg.motor_cmd[i].kp) for i in range(num_cmd_motors)],
                        "kd": [float(self.msg.motor_cmd[i].kd) for i in range(num_cmd_motors)],
                    },
                }
                self.lowcmd_shm.write_data(cmd_data)
            else:
                if self.simulation_mode and self.lowcmd_shm is None:
                    self.lowcmd_shm = try_open_shm(SHM_ROBOT_CMD, SIZE_ROBOT)
                elif self.lowcmd_publisher is not None:
                    self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        if self.simulation_mode:
            return 0
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_2_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_2_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[H1_2_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_2_JointIndex.kLeftAnkle.value,
            H1_2_JointIndex.kRightAnkle.value,
            # Left arm
            H1_2_JointIndex.kLeftShoulderPitch.value,
            H1_2_JointIndex.kLeftShoulderRoll.value,
            H1_2_JointIndex.kLeftShoulderYaw.value,
            H1_2_JointIndex.kLeftElbowPitch.value,
            # Right arm
            H1_2_JointIndex.kRightShoulderPitch.value,
            H1_2_JointIndex.kRightShoulderRoll.value,
            H1_2_JointIndex.kRightShoulderYaw.value,
            H1_2_JointIndex.kRightElbowPitch.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H1_2_JointIndex.kLeftElbowRoll.value,
            H1_2_JointIndex.kLeftWristPitch.value,
            H1_2_JointIndex.kLeftWristyaw.value,
            H1_2_JointIndex.kRightElbowRoll.value,
            H1_2_JointIndex.kRightWristPitch.value,
            H1_2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
    
class H1_2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

class H1_2_JointIndex(IntEnum):
    # Left leg
    kLeftHipYaw = 0
    kLeftHipRoll = 1
    kLeftHipPitch = 2
    kLeftKnee = 3
    kLeftAnkle = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipYaw = 6
    kRightHipRoll = 7
    kRightHipPitch = 8
    kRightKnee = 9
    kRightAnkle = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12

    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

    kNotUsedJoint0 = 27
    kNotUsedJoint1 = 28
    kNotUsedJoint2 = 29
    kNotUsedJoint3 = 30
    kNotUsedJoint4 = 31
    kNotUsedJoint5 = 32
    kNotUsedJoint6 = 33
    kNotUsedJoint7 = 34

class H1_ArmController:
    def __init__(self, simulation_mode = False):
        self.simulation_mode = simulation_mode
        
        logger_mp.info("Initialize H1_ArmController...")
        self.q_target = np.zeros(8)
        self.tauff_target = np.zeros(8)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, go_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, go_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_ArmController] Subscribe dds ok.")

        # initialize h1's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_go_msg_dds__LowCmd_()
        self.msg.head[0] = 0xFE
        self.msg.head[1] = 0xEF
        self.msg.level_flag = 0xFF
        self.msg.gpio = 0

        self.all_motor_q = self.get_current_motor_q()
        self.q_target = self.get_current_dual_arm_q().copy()
        self.tauff_target = np.zeros_like(self.q_target, dtype=np.float64)
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        for id in H1_JointIndex:
            if self._Is_weak_motor(id):
                self.msg.motor_cmd[id].kp = self.kp_low
                self.msg.motor_cmd[id].kd = self.kd_low
                self.msg.motor_cmd[id].mode = 0x01
            else:
                self.msg.motor_cmd[id].kp = self.kp_high
                self.msg.motor_cmd[id].kd = self.kd_high
                self.msg.motor_cmd[id].mode = 0x0A
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_LowState()
                for id in range(H1_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(H1_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(8)
            # self.tauff_target = np.zeros(8)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[H1_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_JointIndex.kLeftAnkle.value,
            H1_JointIndex.kRightAnkle.value,
            # Left arm
            H1_JointIndex.kLeftShoulderPitch.value,
            H1_JointIndex.kLeftShoulderRoll.value,
            H1_JointIndex.kLeftShoulderYaw.value,
            H1_JointIndex.kLeftElbow.value,
            # Right arm
            H1_JointIndex.kRightShoulderPitch.value,
            H1_JointIndex.kRightShoulderRoll.value,
            H1_JointIndex.kRightShoulderYaw.value,
            H1_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
class H1_JointArmIndex(IntEnum):
    # Unlike G1 and H1_2, the arm order in DDS messages for H1 is right then left. 
    # Therefore, the purpose of switching the order here is to maintain consistency with G1 and H1_2.
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

class H1_JointIndex(IntEnum):
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2
    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5
    kWaistYaw = 6
    kLeftHipYaw = 7
    kRightHipYaw = 8
    kNotUsedJoint = 9
    kLeftAnkle = 10
    kRightAnkle = 11
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19

if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
    import pinocchio as pin

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation

    arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    arm = G1_29_ArmController(simulation_mode=True)
    # arm_ik = G1_23_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_23_ArmController()
    # arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = False)
    # arm = H1_2_ArmController()
    # arm_ik = H1_ArmIK(Unit_Test = True, Visualization = True)
    # arm = H1_ArmController()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q  = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)