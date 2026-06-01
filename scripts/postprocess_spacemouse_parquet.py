#!/usr/bin/env python3
"""Trim and de-duplicate SpaceMouse command parquet recordings.

The input format is produced by teleop/teleop_spacemouse_ee_and_arm.py.  Frame
indices in this script refer to command rows only, not raw SpaceMouse event rows.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_STATIC_RUN_FRAMES = 4
pd = None


def _require_pandas():
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as _pd
    except ModuleNotFoundError:
        print("Missing dependency: pandas. Activate the project environment or install pandas + pyarrow.", file=sys.stderr)
        sys.exit(1)
    pd = _pd
    return pd


def _list_array(value: object, expected: int | None, column: str, row_idx: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected is not None and arr.size != expected:
        raise ValueError(f"Row {row_idx} column `{column}` has length {arr.size}, expected {expected}")
    if arr.size == 0:
        raise ValueError(f"Row {row_idx} column `{column}` is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Row {row_idx} column `{column}` contains non-finite values")
    return arr


def _parse_frame_range(text: str) -> tuple[int, int]:
    if ":" in text:
        start_s, end_s = text.split(":", 1)
    elif "-" in text:
        start_s, end_s = text.split("-", 1)
    else:
        raise argparse.ArgumentTypeError("range must be START:END or START-END, with END exclusive")
    try:
        start = int(start_s)
        end = int(end_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range bounds must be integers") from exc
    if start < 0 or end < 0 or end < start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START <= END")
    return start, end


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_post{input_path.suffix}")


def _load_command_rows(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    required = {"entry_type", "command.t_ns", "command.eepose"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    cmd = df.loc[(df["entry_type"].astype(str) == "command") & df["command.t_ns"].notna()].copy()
    if cmd.empty:
        raise ValueError(f"No command rows found in {input_path}")
    return cmd.sort_values("command.t_ns", kind="mergesort")


def _build_trim_keep_mask(
    command_count: int,
    trim_start: int,
    trim_end: int,
    drop_ranges: list[tuple[int, int]],
) -> np.ndarray:
    keep = np.ones(command_count, dtype=bool)
    if trim_start > 0:
        keep[: min(trim_start, command_count)] = False
    if trim_end > 0:
        keep[max(0, command_count - trim_end) :] = False
    for start, end in drop_ranges:
        keep[min(start, command_count) : min(end, command_count)] = False
    return keep


def _same_rows(prev: np.ndarray, cur: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    return np.all(np.isclose(prev, cur, atol=atol, rtol=rtol, equal_nan=False), axis=1)


def _static_pair_mask(
    cmd: pd.DataFrame,
    source: str,
    eepose_atol: float,
    joint_atol: float,
    rtol: float,
) -> np.ndarray:
    if len(cmd) <= 1:
        return np.zeros(0, dtype=bool)

    masks: list[np.ndarray] = []
    if source in {"any", "all", "eepose"}:
        eepose = np.vstack(
            [
                _list_array(value, expected=14, column="command.eepose", row_idx=int(idx))
                for idx, value in cmd["command.eepose"].items()
            ]
        )
        masks.append(_same_rows(eepose[:-1], eepose[1:], atol=eepose_atol, rtol=rtol))

    if source in {"any", "all", "joint"}:
        if "command.ik_joint_pos" not in cmd.columns:
            if source in {"all", "joint"}:
                raise ValueError("Missing `command.ik_joint_pos` column")
        elif not cmd["command.ik_joint_pos"].notna().all():
            if source in {"all", "joint"}:
                raise ValueError("Some command rows have empty `command.ik_joint_pos`")
        else:
            joint_rows = [
                _list_array(value, expected=None, column="command.ik_joint_pos", row_idx=int(idx))
                for idx, value in cmd["command.ik_joint_pos"].items()
            ]
            widths = {row.size for row in joint_rows}
            if len(widths) != 1:
                raise ValueError(f"`command.ik_joint_pos` has inconsistent lengths: {sorted(widths)}")
            joint = np.vstack(joint_rows)
            masks.append(_same_rows(joint[:-1], joint[1:], atol=joint_atol, rtol=rtol))

    if not masks:
        return np.zeros(len(cmd) - 1, dtype=bool)
    if source == "all":
        out = masks[0].copy()
        for mask in masks[1:]:
            out &= mask
        return out

    out = masks[0].copy()
    for mask in masks[1:]:
        out |= mask
    return out


def _static_keep_mask(cmd: pd.DataFrame, pair_same: np.ndarray, min_run_frames: int) -> np.ndarray:
    keep = np.ones(len(cmd), dtype=bool)
    if len(cmd) <= 2 or min_run_frames <= 2:
        return keep

    run_start = 0
    for i in range(1, len(cmd)):
        if pair_same[i - 1]:
            continue
        _drop_static_run_middle(keep, run_start, i, min_run_frames)
        run_start = i
    _drop_static_run_middle(keep, run_start, len(cmd), min_run_frames)
    return keep


def _drop_static_run_middle(keep: np.ndarray, start: int, end: int, min_run_frames: int) -> None:
    run_len = end - start
    if run_len >= min_run_frames and run_len > 2:
        keep[start + 1 : end - 1] = False


def _filter_raw_rows(df: pd.DataFrame, command_t_ns: np.ndarray, keep_raw: str) -> pd.DataFrame:
    if keep_raw == "all":
        return df.loc[df["entry_type"].astype(str) == "raw"].copy()
    if keep_raw == "drop" or command_t_ns.size == 0:
        return df.iloc[0:0].copy()

    raw = df.loc[df["entry_type"].astype(str) == "raw"].copy()
    if raw.empty or "raw.t_ns" not in raw.columns:
        return raw
    start_t = int(np.min(command_t_ns))
    end_t = int(np.max(command_t_ns))
    raw_t = pd.to_numeric(raw["raw.t_ns"], errors="coerce")
    return raw.loc[raw_t.notna() & (raw_t >= start_t) & (raw_t <= end_t)].copy()


def postprocess(args: argparse.Namespace) -> None:
    _require_pandas()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    df = pd.read_parquet(input_path)
    cmd = _load_command_rows(df, input_path)
    command_count = len(cmd)

    trim_keep = _build_trim_keep_mask(
        command_count=command_count,
        trim_start=max(0, int(args.trim_start)),
        trim_end=max(0, int(args.trim_end)),
        drop_ranges=args.drop_frame_range,
    )
    trimmed_cmd = cmd.iloc[trim_keep].copy()

    static_keep = np.ones(len(trimmed_cmd), dtype=bool)
    if args.remove_static and not trimmed_cmd.empty:
        pair_same = _static_pair_mask(
            trimmed_cmd,
            source=args.static_source,
            eepose_atol=float(args.eepose_atol),
            joint_atol=float(args.joint_atol),
            rtol=float(args.rtol),
        )
        static_keep = _static_keep_mask(trimmed_cmd, pair_same=pair_same, min_run_frames=int(args.min_static_run))

    kept_cmd = trimmed_cmd.iloc[static_keep].copy()
    kept_cmd_t_ns = kept_cmd["command.t_ns"].to_numpy(dtype=np.int64) if not kept_cmd.empty else np.zeros(0, dtype=np.int64)

    raw = _filter_raw_rows(df, command_t_ns=kept_cmd_t_ns, keep_raw=args.keep_raw)
    out_df = pd.concat([raw, kept_cmd], axis=0)
    if not out_df.empty:
        out_df["_sort_t_ns"] = pd.to_numeric(out_df["command.t_ns"], errors="coerce")
        raw_t = pd.to_numeric(out_df["raw.t_ns"], errors="coerce") if "raw.t_ns" in out_df.columns else np.nan
        out_df["_sort_t_ns"] = out_df["_sort_t_ns"].fillna(raw_t)
        out_df["_sort_entry"] = np.where(out_df["entry_type"].astype(str) == "raw", 0, 1)
        out_df = out_df.sort_values(["_sort_t_ns", "_sort_entry"], kind="mergesort").drop(
            columns=["_sort_t_ns", "_sort_entry"]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    trimmed_removed = int(command_count - len(trimmed_cmd))
    static_removed = int(len(trimmed_cmd) - len(kept_cmd))
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"command_frames input={command_count} output={len(kept_cmd)}")
    print(f"removed trim={trimmed_removed} static_middle={static_removed}")
    print(f"raw_rows input={int((df['entry_type'].astype(str) == 'raw').sum())} output={len(raw)} keep_raw={args.keep_raw}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess SpaceMouse parquet recordings: trim command-frame ranges and remove "
            "middle frames from long static command runs."
        )
    )
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", default=None, help="Output parquet path. Defaults to <input_stem>_post.parquet")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")

    parser.add_argument("--trim-start", type=int, default=0, help="Drop this many command frames from the start")
    parser.add_argument("--trim-end", type=int, default=0, help="Drop this many command frames from the end")
    parser.add_argument(
        "--drop-frame-range",
        type=_parse_frame_range,
        action="append",
        default=[],
        metavar="START:END",
        help="Drop command frame range [START, END). Can be passed multiple times. Example: --drop-frame-range 0:4600",
    )

    parser.add_argument("--remove-static", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--static-source",
        choices=("any", "all", "eepose", "joint"),
        default="any",
        help="Which signal marks static frames. any means eepose OR joint_pos.",
    )
    parser.add_argument(
        "--min-static-run",
        type=int,
        default=DEFAULT_STATIC_RUN_FRAMES,
        help="Only compress static runs with at least this many frames. Default keeps runs up to 3 frames unchanged.",
    )
    parser.add_argument(
        "--eepose-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for eepose unchanged detection. Larger removes more near-static frames.",
    )
    parser.add_argument(
        "--joint-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for joint unchanged detection. Larger removes more near-static frames.",
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for unchanged detection")
    parser.add_argument(
        "--keep-raw",
        choices=("trim", "all", "drop"),
        default="trim",
        help="How to handle raw SpaceMouse rows. trim keeps raw rows inside the kept command time span.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        postprocess(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
