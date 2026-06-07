#!/usr/bin/env python3
"""Replay or inspect SpaceMouse EE-command parquet episodes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Array, Lock, Value
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
XR_ROOT = REPO_ROOT / "xr_teleoperate"
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))

BTN_1 = 4096
BTN_2 = 8192
BTN_3 = 16384
BTN_4 = 32768


@dataclass(frozen=True)
class ReplayData:
    commands: pd.DataFrame
    command_t_ns: np.ndarray
    joint_pos: np.ndarray
    left_closed: np.ndarray
    right_closed: np.ndarray


def _list_array(value: object, expected: int, column: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected:
        raise ValueError(f"{column} has length {arr.size}, expected {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{column} contains non-finite values")
    return arr


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = values[0]
    a = float(np.clip(alpha, 0.0, 1.0))
    for i in range(1, values.size):
        out[i] = out[i - 1] + a * (values[i] - out[i - 1])
    return out


def _estimate_hz(t_ns: np.ndarray) -> float:
    if t_ns.size < 2:
        return 30.0
    dt = np.diff(t_ns) / 1e9
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if dt.size == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def _reconstruct_gripper_from_raw(df: pd.DataFrame, command_t_ns: np.ndarray, smooth_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    raw = df.loc[df["entry_type"].astype(str) == "raw"].copy()
    raw = raw[raw["raw.t_ns"].notna()].sort_values("raw.t_ns", kind="mergesort")
    raw_t = raw["raw.t_ns"].to_numpy(dtype=np.float64) if not raw.empty else np.zeros(0)
    raw_edges = raw["raw.edges_mask"].fillna(0).to_numpy(dtype=np.int64) if not raw.empty else np.zeros(0, dtype=np.int64)

    left = np.zeros(command_t_ns.size, dtype=np.float64)
    right = np.zeros(command_t_ns.size, dtype=np.float64)
    selected = "left"
    left_state = 0.0
    right_state = 0.0
    raw_pos = 0
    for i, t_ns in enumerate(command_t_ns):
        while raw_pos < raw_t.size and raw_t[raw_pos] <= t_ns:
            edges = int(raw_edges[raw_pos])
            if edges & BTN_1:
                selected = "left"
            if edges & BTN_2:
                selected = "right"
            if edges & BTN_3:
                if selected == "left":
                    left_state = 0.0
                else:
                    right_state = 0.0
            if edges & BTN_4:
                if selected == "left":
                    left_state = 1.0
                else:
                    right_state = 1.0
            raw_pos += 1
        left[i] = left_state
        right[i] = right_state
    return _ema(left, smooth_alpha), _ema(right, smooth_alpha)


def load_replay_data(path: Path, smooth_alpha: float) -> ReplayData:
    df = pd.read_parquet(path)
    cmd = df.loc[(df["entry_type"].astype(str) == "command") & df["command.t_ns"].notna()].copy()
    if cmd.empty:
        raise ValueError(f"No command rows found in {path}")
    cmd = cmd.sort_values("command.t_ns", kind="mergesort")
    t_ns = cmd["command.t_ns"].to_numpy(dtype=np.float64)
    joint_pos = np.vstack([_list_array(v, 14, "command.ik_joint_pos") for v in cmd["command.ik_joint_pos"]])

    if "aug.left_gripper_closed_smooth" in cmd.columns and "aug.right_gripper_closed_smooth" in cmd.columns:
        left = cmd["aug.left_gripper_closed_smooth"].ffill().fillna(0.0).to_numpy(dtype=np.float64)
        right = cmd["aug.right_gripper_closed_smooth"].ffill().fillna(0.0).to_numpy(dtype=np.float64)
    elif "command.left_gripper_input" in cmd.columns and "command.right_gripper_input" in cmd.columns:
        left_input = cmd["command.left_gripper_input"].ffill().fillna(1.0).to_numpy(dtype=np.float64)
        right_input = cmd["command.right_gripper_input"].ffill().fillna(1.0).to_numpy(dtype=np.float64)
        left = np.clip(1.0 - left_input, 0.0, 1.0)
        right = np.clip(1.0 - right_input, 0.0, 1.0)
    else:
        left, right = _reconstruct_gripper_from_raw(df, t_ns, smooth_alpha=smooth_alpha)

    return ReplayData(commands=cmd, command_t_ns=t_ns, joint_pos=joint_pos, left_closed=left, right_closed=right)


def _print_summary(path: Path, data: ReplayData) -> None:
    hz = _estimate_hz(data.command_t_ns)
    dt = np.diff(data.command_t_ns) / 1e9 if data.command_t_ns.size > 1 else np.zeros(0)
    print(f"input={path}")
    print(f"command_frames={data.command_t_ns.size} hz_median={hz:.3f}")
    if dt.size:
        print(f"dt_sec min/median/max={float(np.min(dt)):.4f}/{float(np.median(dt)):.4f}/{float(np.max(dt)):.4f}")
    print(
        "left_closed "
        f"range=[{float(np.min(data.left_closed)):.3f},{float(np.max(data.left_closed)):.3f}] "
        f"frames_gt_0.5={int(np.sum(data.left_closed > 0.5))}"
    )
    print(
        "right_closed "
        f"range=[{float(np.min(data.right_closed)):.3f},{float(np.max(data.right_closed)):.3f}] "
        f"frames_gt_0.5={int(np.sum(data.right_closed > 0.5))}"
    )
    if "aug.event_type" in data.commands.columns:
        counts = data.commands["aug.event_type"].fillna("none").value_counts()
        event_items = [(k, int(v)) for k, v in counts.items() if k != "none"]
        print("events=" + (", ".join(f"{k}:{v}" for k, v in event_items) if event_items else "none"))
    if "aug.is_inserted_frame" in data.commands.columns:
        print(f"inserted_frames={int(data.commands['aug.is_inserted_frame'].fillna(False).sum())}")


def _make_arm_controller(sim: bool, network_interface: str | None):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.robot_arm import G1_29_ArmController

    if not sim:
        ChannelFactoryInitialize(0, networkInterface=network_interface)
    return G1_29_ArmController(simulation_mode=sim)


def _read_sim_step_from_arm_state(arm_ctrl) -> int | None:
    shm = getattr(arm_ctrl, "lowstate_shm", None)
    if shm is None:
        return None
    try:
        payload = shm.read_data()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("sim_step", None)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _wait_sim_step_available(arm_ctrl, timeout_sec: float = 30.0) -> int:
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    while time.monotonic() < deadline:
        step = _read_sim_step_from_arm_state(arm_ctrl)
        if step is not None:
            return int(step)
        time.sleep(0.01)
    raise TimeoutError(
        "Timed out waiting for sim_step in isaac_robot_state. "
        "Restart JoySim with a DDSBridge that publishes sim_step."
    )


def _wait_until_sim_step(
    arm_ctrl,
    target_step: int,
    *,
    timeout_sec: float,
) -> int:
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last_step = None
    while time.monotonic() < deadline:
        step = _read_sim_step_from_arm_state(arm_ctrl)
        if step is not None:
            last_step = int(step)
            if last_step >= int(target_step):
                return last_step
        time.sleep(0.002)
    raise TimeoutError(f"Timed out waiting for sim_step>={target_step}; last_step={last_step}")


def _make_gripper_controller(args: argparse.Namespace):
    if args.ee == "none":
        return None, None, None
    from teleop.robot_control.robot_hand_inspire import Inspire_Gripper_Controller

    if args.gripper_sides == "left":
        active_sides = ("left",)
    elif args.gripper_sides == "right":
        active_sides = ("right",)
    else:
        active_sides = ("left", "right")

    left_value = Value("d", float(args.gripper_open_input), lock=True)
    right_value = Value("d", float(args.gripper_open_input), lock=True)
    data_lock = Lock()
    state_array = Array("d", 2, lock=False)
    action_array = Array("d", 2, lock=False)
    controller = Inspire_Gripper_Controller(
        left_value,
        right_value,
        data_lock,
        state_array,
        action_array,
        simulation_mode=bool(args.sim),
        input_min=float(args.gripper_input_min),
        input_max=float(args.gripper_input_max),
        open_cmd=float(args.inspire_gripper_open),
        close_cmd=float(args.inspire_gripper_close),
        smooth_alpha=float(args.inspire_gripper_alpha),
        max_speed=float(args.inspire_gripper_max_speed),
        active_sides=active_sides,
    )
    return controller, left_value, right_value


def _closed_to_input(closed_value: float, open_input: float, close_input: float) -> float:
    closed = float(np.clip(closed_value, 0.0, 1.0))
    return float(open_input + closed * (close_input - open_input))


def replay(data: ReplayData, args: argparse.Namespace) -> None:
    arm_ctrl = _make_arm_controller(sim=bool(args.sim), network_interface=args.network_interface)
    gripper_ctrl, left_gripper_value, right_gripper_value = _make_gripper_controller(args)
    del gripper_ctrl

    sim_step_sync = bool(args.sim_step_sync and args.sim)
    sim_steps_per_command = max(1, int(args.sim_steps_per_command))
    sim_step_timeout_sec = max(1.0, float(args.sim_step_timeout_sec))
    hz = float(args.hz) if args.hz and args.hz > 0 else 0.0
    fixed_period = 1.0 / hz if hz > 0 else None
    start_wall = time.monotonic()
    first_t = float(data.command_t_ns[0])
    last_print = 0.0
    next_target_step: int | None = None
    if sim_step_sync:
        base_step = _wait_sim_step_available(arm_ctrl, timeout_sec=sim_step_timeout_sec)
        next_target_step = base_step

    try:
        for i, q in enumerate(data.joint_pos):
            loop_start = time.monotonic()
            if sim_step_sync:
                pass
            elif fixed_period is None:
                target_elapsed = max(0.0, (float(data.command_t_ns[i]) - first_t) / 1e9)
                sleep_t = start_wall + target_elapsed - loop_start
                if sleep_t > 0:
                    time.sleep(sleep_t)
            elif i > 0:
                sleep_t = fixed_period - (loop_start - last_print if False else 0.0)
                del sleep_t

            if left_gripper_value is not None and right_gripper_value is not None:
                with left_gripper_value.get_lock():
                    left_gripper_value.value = _closed_to_input(
                        data.left_closed[i],
                        open_input=float(args.gripper_open_input),
                        close_input=float(args.gripper_close_input),
                    )
                with right_gripper_value.get_lock():
                    right_gripper_value.value = _closed_to_input(
                        data.right_closed[i],
                        open_input=float(args.gripper_open_input),
                        close_input=float(args.gripper_close_input),
                    )

            arm_ctrl.ctrl_dual_arm(np.asarray(q, dtype=np.float64), np.zeros(14, dtype=np.float64))
            now = time.monotonic()
            if now - last_print >= float(args.print_period):
                suffix = ""
                if sim_step_sync and next_target_step is not None:
                    suffix = f" sim_step_target={next_target_step + sim_steps_per_command}"
                print(
                    f"frame={i + 1}/{len(data.joint_pos)} "
                    f"left_closed={data.left_closed[i]:.3f} right_closed={data.right_closed[i]:.3f}"
                    f"{suffix}"
                )
                last_print = now
            if sim_step_sync:
                assert next_target_step is not None
                next_target_step += sim_steps_per_command
                _wait_until_sim_step(
                    arm_ctrl,
                    next_target_step,
                    timeout_sec=sim_step_timeout_sec,
                )
            elif fixed_period is not None:
                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, fixed_period - elapsed))
    finally:
        if args.go_home_on_exit:
            arm_ctrl.ctrl_dual_arm_go_home()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SpaceMouse command parquet files.")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--dry-run", action="store_true", help="Only print replay statistics")
    parser.add_argument("--hz", type=float, default=0.0, help="Fixed replay frequency; 0 uses command timestamps")
    parser.add_argument("--sim", action="store_true", help="Use simulation shared memory")
    parser.add_argument("--real", action="store_true", help="Use real robot DDS")
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--ee", choices=["none", "inspire_gripper", "omnipicker"], default="inspire_gripper")
    parser.add_argument(
        "--gripper-sides",
        choices=["both", "left", "right"],
        default="both",
        help="Which OmniPicker gripper side should be actively driven during replay.",
    )
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA alpha when reconstructing gripper from raw")
    parser.add_argument("--gripper-input-min", type=float, default=0.0)
    parser.add_argument("--gripper-input-max", type=float, default=1.0)
    parser.add_argument("--gripper-open-input", type=float, default=1.0)
    parser.add_argument("--gripper-close-input", type=float, default=0.0)
    parser.add_argument("--inspire-gripper-open", type=float, default=0.0)
    parser.add_argument("--inspire-gripper-close", type=float, default=0.75)
    parser.add_argument("--inspire-gripper-alpha", type=float, default=0.05)
    parser.add_argument("--inspire-gripper-max-speed", type=float, default=0.20)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument(
        "--sim-step-sync",
        action="store_true",
        help=(
            "In --sim mode, pace replay by isaac_robot_state['sim_step'] instead of wall time. "
            "Use this for offline rendering when Isaac runs slower than real time."
        ),
    )
    parser.add_argument(
        "--sim-steps-per-command",
        type=int,
        default=4,
        help="Number of JoySim 120Hz simulation steps to hold each 30Hz parquet command when --sim-step-sync is enabled.",
    )
    parser.add_argument(
        "--sim-step-timeout-sec",
        type=float,
        default=120.0,
        help="Timeout while waiting for simulation steps in --sim-step-sync mode.",
    )
    parser.add_argument("--go-home-on-exit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.real:
        args.sim = False
    if not args.real and not args.sim:
        args.sim = True
    if args.ee == "omnipicker":
        args.ee = "inspire_gripper"
    path = Path(args.input).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = load_replay_data(path, smooth_alpha=float(args.smooth_alpha))
    _print_summary(path, data)
    if args.dry_run:
        return
    replay(data, args)


if __name__ == "__main__":
    main()
