# -*- coding: utf-8 -*-
"""Shared-memory JSON segments used by unitree_sim_isaaclab DDS bridges (attach-only from teleop)."""

from __future__ import annotations

import json
import threading
import time
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
from typing import Any, Dict, Optional

# Names/sizes must match unitree_sim_isaaclab/dds/*.py
SHM_ROBOT_STATE = "isaac_robot_state"
SHM_ROBOT_CMD = "dds_robot_cmd"
SIZE_ROBOT = 3072

SHM_DEX3_STATE = "isaac_dex3_state"
SHM_DEX3_CMD = "isaac_dex3_cmd"
SIZE_DEX3 = 1180

SHM_GRIPPER_STATE = "isaac_gripper_state"
SHM_GRIPPER_CMD = "isaac_gripper_cmd"
SIZE_GRIPPER = 512

SHM_INSPIRE_STATE = "isaac_inspire_state"
SHM_INSPIRE_CMD = "isaac_inspire_cmd"
SIZE_INSPIRE = 1024

SHM_SIM_STATE = "isaac_sim_state"
SIZE_SIM_STATE = 4096

SHM_RESET_POSE_CMD = "isaac_reset_pose_cmd"
SIZE_RESET_POSE = 512


class IsaacJsonShm:
    """JSON payload in shared memory (4-byte ts + 4-byte len + data), compatible with sim SharedMemoryManager."""

    def __init__(self, name: str, size: int = 3072):
        self.name = name
        self.size = size
        self.lock = threading.RLock()
        self.shm = shared_memory.SharedMemory(name=name)
        # Attach-only consumer: do not let resource_tracker unlink segments
        # owned by Isaac publisher process at interpreter shutdown.
        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception:
            pass

    def read_data(self) -> Optional[Dict[str, Any]]:
        try:
            with self.lock:
                data_len = int.from_bytes(self.shm.buf[4:8], "little")
                if data_len == 0:
                    return None
                raw = bytes(self.shm.buf[8 : 8 + data_len])
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, dict):
                    data["_timestamp"] = int.from_bytes(self.shm.buf[0:4], "little")
                return data
        except Exception:
            return None

    def write_data(self, data) -> bool:
        try:
            with self.lock:
                raw = json.dumps(data).encode("utf-8")
                if len(raw) > self.size - 8:
                    return False
                ts = int(time.time()) & 0xFFFFFFFF
                self.shm.buf[0:4] = ts.to_bytes(4, "little")
                self.shm.buf[4:8] = len(raw).to_bytes(4, "little")
                self.shm.buf[8 : 8 + len(raw)] = raw
                return True
        except Exception:
            return False

    def close(self) -> None:
        if getattr(self, "shm", None) is not None:
            try:
                self.shm.close()
            except Exception:
                pass
            self.shm = None


def try_open_shm(name: str, size: int) -> Optional[IsaacJsonShm]:
    try:
        return IsaacJsonShm(name, size)
    except Exception:
        return None
