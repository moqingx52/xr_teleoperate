# -*- coding: utf-8 -*-
"""
HaMeR 离线回放专用入口（与 hamer/demo.py 输出约定一致）。

demo.py 默认将结构化结果写入::
    os.path.join(out_folder, structured_file)
其中 ``structured_file`` 默认为 ``hamer_structured_results.json``（见 demo ``--structured_file``）。

用法示例::

    python teleop_hamer_and_arm.py --arm G1_29 --ee dex3 --sim \\
        --hamer_json /path/to/out_demo/hamer_structured_results.json --replay_fps 30

    # 或只给 demo 的 out_folder::
    python teleop_hamer_and_arm.py --arm G1_29 --ee dex3 --sim \\
        --hamer_out_dir /path/to/out_demo --replay_fps 30

本脚本会自动补上 ``--input-source hamer``（若命令行未指定）。
其余参数与 ``teleop_hand_and_arm.py`` 相同，可直接透传。
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
    """与直接运行 teleop_hand_and_arm.py 相同：把 xr_teleoperate 根目录加入 path 后加载。"""
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


def run_hamer_replay(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not _argv_has(args, "--input-source") and not _argv_has(args, "--input_source"):
        args = ["--input-source", "hamer"] + args
    _get_teleop_main()(args)


if __name__ == "__main__":
    run_hamer_replay()
