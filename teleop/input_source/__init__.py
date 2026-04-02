from teleop.input_source.hamer_input import HamerInputSource, HamerJsonReader
from teleop.input_source.hamer_adapter import HamerAdapter
from teleop.input_source.hamer_to_robot_frame import WristToEEConfig, wrist_to_ee_target

__all__ = [
    "HamerInputSource",
    "HamerJsonReader",
    "HamerAdapter",
    "EgoDexInputSource",
    "WristToEEConfig",
    "wrist_to_ee_target",
]


def __getattr__(name: str):
    if name == "EgoDexInputSource":
        from teleop.input_source.egodex_input import EgoDexInputSource
        return EgoDexInputSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
