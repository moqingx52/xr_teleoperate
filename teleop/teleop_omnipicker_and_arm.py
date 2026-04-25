#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omnipicker 专用入口。

默认行为：
- arm: G1_29
- ee: inspire_gripper（通过 inspire 通道控制双指夹爪）
- sim: true
- input-source: hamer（离线回放）
- input-mode: hand（把输入手部关键点语义映射到二指夹爪开合）

离线回放（input-source=hamer）支持：
- JSON: --hamer-json
- Parquet: --parquet
- 相机外参 T_cam2base: --hamer-cam2base-json
- 手腕到末端固定变换 T_wrist2ee: --wrist-to-ee-json

可用别名（便于迁移旧命令）：
- --json -> --hamer-json
- --cam2base-json -> --hamer-cam2base-json
- --wrist2ee-json -> --wrist-to-ee-json
- --parquet-action-fallback-mode -> --hamer-parquet-action-fallback-mode

其余参数均透传给 teleop_hand_and_arm.py。
"""

import importlib.util
import os
import sys


_current = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_current)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

_teleop_main_fn = None


def _get_teleop_main():
    global _teleop_main_fn
    if _teleop_main_fn is not None:
        return _teleop_main_fn
    path = os.path.join(_current, "teleop_hand_and_arm.py")
    spec = importlib.util.spec_from_file_location("teleop_hand_and_arm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _teleop_main_fn = mod.main
    return _teleop_main_fn


def _argv_has(argv, flag):
    if flag in argv:
        return True
    prefix = flag + "="
    return any(a.startswith(prefix) for a in argv)


def _argv_get_value(argv, flags):
    for i, a in enumerate(argv):
        for f in flags:
            if a == f and i + 1 < len(argv):
                return argv[i + 1]
            prefix = f + "="
            if a.startswith(prefix):
                return a[len(prefix):]
    return None


def _argv_pop_flag(argv, flags):
    """Remove first matched boolean flag in-place and return True if found."""
    for i, a in enumerate(argv):
        if a in flags:
            argv.pop(i)
            return True
    return False


def _rewrite_aliases(argv):
    alias_map = {
        "--json": "--hamer-json",
        "--json_path": "--hamer-json",
        "--cam2base-json": "--hamer-cam2base-json",
        "--cam2base_json": "--hamer-cam2base-json",
        "--wrist2ee-json": "--wrist-to-ee-json",
        "--wrist2ee_json": "--wrist-to-ee-json",
        "--parquet-action-fallback-mode": "--hamer-parquet-action-fallback-mode",
        "--parquet_action_fallback_mode": "--hamer-parquet-action-fallback-mode",
    }
    rewritten = []
    for a in argv:
        replaced = a
        for old, new in alias_map.items():
            if a == old:
                replaced = new
                break
            prefix = old + "="
            if a.startswith(prefix):
                replaced = new + "=" + a[len(prefix):]
                break
        rewritten.append(replaced)
    return rewritten


def run_omnipicker(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    args = _rewrite_aliases(args)

    injected = []
    force_real = _argv_pop_flag(args, {"--real", "--no-sim", "--no_sim"})

    if not _argv_has(args, "--arm"):
        injected += ["--arm", "G1_29"]
    if not _argv_has(args, "--ee"):
        injected += ["--ee", "inspire_gripper"]
    if not _argv_has(args, "--input-source") and not _argv_has(args, "--input_source"):
        injected += ["--input-source", "hamer"]
    if not _argv_has(args, "--input-mode") and not _argv_has(args, "--input_mode"):
        injected += ["--input-mode", "hand"]
    if (not force_real) and (not _argv_has(args, "--sim")):
        injected += ["--sim"]

    effective = injected + args
    input_source = _argv_get_value(effective, {"--input-source", "--input_source"})
    if input_source == "hamer":
        has_json = _argv_has(effective, "--hamer-json") or _argv_has(effective, "--hamer_json")
        has_parquet = _argv_has(effective, "--parquet") or _argv_has(effective, "--parquent")
        has_out_dir = _argv_has(effective, "--hamer-out-dir") or _argv_has(effective, "--hamer_out_dir")
        if int(has_json) + int(has_parquet) + int(has_out_dir) == 0:
            raise SystemExit(
                "Omnipicker offline replay requires one input: "
                "--hamer-json OR --parquet OR --hamer-out-dir."
            )
        if int(has_json) + int(has_parquet) > 1:
            raise SystemExit(
                "Omnipicker offline replay: --hamer-json and --parquet are mutually exclusive."
            )
        # Omnipicker default: if parquet has no kp3d and falls back to action slices,
        # treat slices as camera-frame wrist pose and still apply cam2base + wrist->EE.
        if has_parquet and (not _argv_has(effective, "--hamer-parquet-action-fallback-mode")):
            effective += ["--hamer-parquet-action-fallback-mode", "wrist_cam"]

    _get_teleop_main()(effective)


if __name__ == "__main__":
    run_omnipicker()
