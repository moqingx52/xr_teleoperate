#!/usr/bin/env python3
"""Batch replay parquet trajectories with episode-level recorder control."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
XR_ROOT = REPO_ROOT / "xr_teleoperate"
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))

from teleop.utils.isaac_shm import (  # noqa: E402
    SHM_REPLAY_EPISODE_CTL,
    SIZE_REPLAY_EPISODE_CTL,
    try_open_shm,
)


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_parquet_paths(directory: Path, pattern: str, sort_mode: str) -> list[Path]:
    files = [p.resolve() for p in directory.glob(pattern) if p.is_file()]
    if sort_mode == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: p.name)
    return files


def _wait_for_control_shm(name: str, size: int):
    while True:
        shm = try_open_shm(name=name, size=int(size))
        if shm is not None:
            return shm
        time.sleep(0.2)


def _send_command(
    shm,
    request_id: int,
    command: str,
    episode_tag: str | None = None,
    source_parquet_path: Path | None = None,
) -> None:
    payload = {
        "request_id": int(request_id),
        "command": str(command),
        "episode_tag": episode_tag if episode_tag is not None else "",
        "source_parquet_path": str(source_parquet_path) if source_parquet_path is not None else "",
        "status": "request",
        "sent_at_sec": float(time.time()),
    }
    ok = shm.write_data(payload)
    if not ok:
        print(f"[batch-replay][WARN] failed to write control command: {payload}", flush=True)


def _wait_for_status(shm, request_id: int, expected_status: str) -> dict:
    expected = str(expected_status)
    while True:
        payload = shm.read_data() or {}
        if not isinstance(payload, dict):
            time.sleep(0.05)
            continue

        handled_request_id = _safe_int(payload.get("handled_request_id"), default=-1)
        if handled_request_id != int(request_id):
            time.sleep(0.05)
            continue

        status = str(payload.get("status", "")).strip().lower()
        if status == expected:
            return payload
        if status == "error":
            print(f"[batch-replay][WARN] control command error: {payload}", flush=True)
            return payload
        time.sleep(0.05)


def _run_one_replay(parquet_path: Path, args: argparse.Namespace) -> int:
    if args.replay_mode == "ee_ik":
        replay_script = (XR_ROOT / "teleop" / "teleop_omnipicker_and_arm.py").resolve()
        cmd = [
            sys.executable,
            str(replay_script),
            "--parquet",
            str(parquet_path),
            "--use-ik",
            "--ee",
            str(args.ee),
            "--hamer-parquet-action-fallback-mode",
            "ee_base",
            "--omnipicker-gripper-source",
            "action",
        ]
        if args.hz > 0:
            cmd.extend(["--replay-fps", str(args.hz)])
        if args.sim:
            cmd.append("--sim")
        else:
            cmd.append("--real")
        if args.network_interface:
            cmd.extend(["--network-interface", str(args.network_interface)])
    else:
        replay_script = (Path(__file__).resolve().parent / "replay_spacemouse_parquet.py").resolve()
        cmd = [
            sys.executable,
            str(replay_script),
            "--input",
            str(parquet_path),
            "--ee",
            str(args.ee),
            "--gripper-sides",
            str(args.gripper_sides),
            "--smooth-alpha",
            str(args.smooth_alpha),
            "--print-period",
            str(args.print_period),
            "--inspire-gripper-open",
            str(args.inspire_gripper_open),
            "--inspire-gripper-close",
            str(args.inspire_gripper_close),
            "--inspire-gripper-alpha",
            str(args.inspire_gripper_alpha),
            "--inspire-gripper-max-speed",
            str(args.inspire_gripper_max_speed),
        ]
        if args.hz > 0:
            cmd.extend(["--hz", str(args.hz)])
        if args.sim:
            cmd.append("--sim")
        else:
            cmd.append("--real")
        if args.network_interface:
            cmd.extend(["--network-interface", str(args.network_interface)])
        if args.go_home_on_exit:
            cmd.append("--go-home-on-exit")

    if args.replay_mode == "ee_ik":
        print(
            f"[batch-replay] replay start: {parquet_path} "
            "replay_drive=ee_pose ik_at_replay=true "
            "(teleop_omnipicker_and_arm.py --use-ik)",
            flush=True,
        )
    else:
        print(
            f"[batch-replay] replay start: {parquet_path} "
            "replay_drive=joint_direct ik_at_replay=false "
            "(replay_spacemouse_parquet.py)",
            flush=True,
        )
    result = subprocess.run(cmd, check=False)
    print(
        f"[batch-replay] replay done: {parquet_path.name} returncode={result.returncode}",
        flush=True,
    )
    return int(result.returncode)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch replay SpaceMouse parquet trajectories with JoySim recorder segmentation."
    )
    parser.add_argument(
        "--dir",
        default=str(REPO_ROOT / "record_augmented"),
        help="Directory containing replay parquet files.",
    )
    parser.add_argument("--glob", default="*.parquet", help="Glob pattern under --dir.")
    parser.add_argument("--sort", choices=["name", "mtime"], default="name")
    parser.add_argument("--ctl-shm-name", default=SHM_REPLAY_EPISODE_CTL)
    parser.add_argument("--ctl-shm-size", type=int, default=SIZE_REPLAY_EPISODE_CTL)
    parser.add_argument("--sim", dest="sim", action="store_true", help="Use simulation shared memory mode.")
    parser.add_argument("--real", dest="sim", action="store_false", help="Use real robot DDS mode.")
    parser.set_defaults(sim=True)
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--ee", choices=["none", "inspire_gripper", "omnipicker"], default="inspire_gripper")
    parser.add_argument(
        "--gripper-sides",
        choices=["both", "left", "right"],
        default="both",
        help="Which OmniPicker gripper side should be actively driven during joint replay.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["joint", "ee_ik"],
        default="joint",
        help="joint uses command.ik_joint_pos directly; ee_ik replays command.eepose through teleop_omnipicker_and_arm.py --use-ik.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=0.0,
        help="Fixed replay rate in Hz. 0 uses command.t_ns (required for HDF5 action/gripper alignment).",
    )
    parser.add_argument("--smooth-alpha", type=float, default=0.25)
    parser.add_argument("--inspire-gripper-open", type=float, default=0.0)
    parser.add_argument("--inspire-gripper-close", type=float, default=1.2)
    parser.add_argument("--inspire-gripper-alpha", type=float, default=1.0)
    parser.add_argument("--inspire-gripper-max-speed", type=float, default=100.0)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.5,
        help="Wait this long after replay returns before finalizing the recorder.",
    )
    parser.add_argument("--go-home-on-exit", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Deprecated; kept for CLI compatibility. Batch replay always runs all episodes.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    replay_dir = Path(args.dir).expanduser().resolve()
    if not replay_dir.is_dir():
        raise NotADirectoryError(replay_dir)

    parquet_paths = _load_parquet_paths(replay_dir, pattern=str(args.glob), sort_mode=str(args.sort))
    if len(parquet_paths) == 0:
        print(f"[batch-replay] no parquet files found in {replay_dir} with pattern={args.glob!r}", flush=True)
        return 0

    shm = _wait_for_control_shm(
        name=str(args.ctl_shm_name),
        size=int(args.ctl_shm_size),
    )

    initial_status = shm.read_data() or {}
    handled_request_id = _safe_int(initial_status.get("handled_request_id"), default=-1)
    request_id = handled_request_id if handled_request_id >= 0 else int(time.time() * 1000)
    for parquet_path in parquet_paths:
        request_id += 1
        _send_command(
            shm,
            request_id=request_id,
            command="prepare",
            episode_tag=parquet_path.stem,
            source_parquet_path=parquet_path,
        )
        prepare_info = _wait_for_status(
            shm,
            request_id=request_id,
            expected_status="ready",
        )
        video_output_path = prepare_info.get("video_output_path", None)
        if video_output_path:
            print(f"[batch-replay] recording to: {video_output_path}", flush=True)
        hdf5_output_path = prepare_info.get("hdf5_output_path", None)
        if hdf5_output_path:
            print(f"[batch-replay] recording hdf5 to: {hdf5_output_path}", flush=True)
        record_output_path = prepare_info.get("record_output_path", None)
        if record_output_path and not video_output_path and not hdf5_output_path:
            print(f"[batch-replay] recording to: {record_output_path}", flush=True)

        try:
            _run_one_replay(parquet_path=parquet_path, args=args)
            settle_sec = max(0.0, float(args.settle_sec))
            if settle_sec > 0.0:
                time.sleep(settle_sec)
        except Exception as exc:
            print(f"[batch-replay][WARN] replay raised exception: {exc}", flush=True)
        finally:
            request_id += 1
            _send_command(shm, request_id=request_id, command="finalize", episode_tag=parquet_path.stem)
            _wait_for_status(
                shm,
                request_id=request_id,
                expected_status="idle",
            )

    print(f"[batch-replay] all done, episodes={len(parquet_paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
