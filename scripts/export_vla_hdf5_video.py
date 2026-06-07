#!/usr/bin/env python3
"""Export camera videos from ALOHA/ACT-style VLA HDF5 episodes."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    scale = float(height) / float(frame.shape[0])
    width = int(round(frame.shape[1] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _open_writer(path: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter, Path | None]:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if writer.isOpened():
        return writer, None
    writer.release()

    tmp_file = tempfile.NamedTemporaryFile(prefix=f"{path.stem}_", suffix=path.suffix, dir="/tmp", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        writer.release()
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer, tmp_path


def _write_video(path: Path, frames: Iterable[np.ndarray], fps: float) -> int:
    writer = None
    tmp_path = None
    count = 0
    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected RGB frame [H,W,3], got {frame.shape}")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                height, width = bgr.shape[:2]
                writer, tmp_path = _open_writer(path, fps, width, height)
            writer.write(bgr)
            count += 1
    finally:
        if writer is not None:
            writer.release()
    if tmp_path is not None:
        shutil.move(str(tmp_path), path)
    return count


def _transcode_to_x264(path: Path, *, crf: int, preset: str) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH, cannot transcode to x264")

    tmp_out = path.with_name(f"{path.stem}.x264_tmp{path.suffix}")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp_out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.move(str(tmp_out), path)


def _load_timestamps(h5: h5py.File, frame_count: int) -> np.ndarray | None:
    if "timestamps" not in h5:
        return None
    ts = np.asarray(h5["timestamps"][:], dtype=np.float64).reshape(-1)
    if ts.size < frame_count:
        frame_count = int(ts.size)
    if frame_count <= 0:
        return None
    ts = ts[:frame_count]
    if not np.all(np.isfinite(ts)):
        raise ValueError("timestamps contain non-finite values")
    if np.any(np.diff(ts) < 0):
        raise ValueError("timestamps must be non-decreasing")
    return ts


def _fps_from_timestamps(frame_count: int, ts: np.ndarray | None) -> float | None:
    if ts is None or frame_count <= 1:
        return None
    duration = float(ts[-1] - ts[0])
    if duration <= 0:
        return None
    return float(frame_count) / duration


def _sample_indices(h5: h5py.File, frame_count: int, fps: float, use_timestamps: bool) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.int64)
    if not use_timestamps:
        return np.arange(frame_count, dtype=np.int64)
    ts = _load_timestamps(h5, frame_count)
    if ts is None:
        raise ValueError("Missing /timestamps while --use-timestamps is enabled")
    frame_count = min(frame_count, int(ts.size))

    ts = ts - float(ts[0])
    duration = float(ts[-1])
    out_frames = int(np.round(duration * float(fps))) + 1
    out_frames = max(1, out_frames)
    query = np.arange(out_frames, dtype=np.float64) / float(fps)
    right = np.searchsorted(ts, query, side="right")
    idx = np.clip(right - 1, 0, frame_count - 1)
    return idx.astype(np.int64, copy=False)


def _combined_frames(h5: h5py.File, target_height: int, sampled_indices: np.ndarray):
    images = h5["observations/images"]
    for idx in sampled_indices:
        views = [_resize_to_height(images[name][idx], target_height) for name in CAMERA_NAMES]
        yield np.concatenate(views, axis=1)


def export_episode(
    path: Path,
    output_dir: Path,
    *,
    fps: float | None,
    target_height: int,
    individual: bool,
    use_timestamps: bool,
    transcode_x264: bool,
    x264_crf: int,
    x264_preset: str,
) -> list[Path]:
    written: list[Path] = []
    with h5py.File(path, "r") as h5:
        if "observations/images" not in h5:
            raise ValueError(f"Missing /observations/images in {path}")
        images = h5["observations/images"]
        for camera_name in CAMERA_NAMES:
            if camera_name not in images:
                raise ValueError(f"Missing camera {camera_name} in {path}")

        frame_count = min(int(images[name].shape[0]) for name in CAMERA_NAMES)
        ts = _load_timestamps(h5, frame_count)
        used_fps = float(fps) if fps is not None else (
            _fps_from_timestamps(frame_count, ts) or float(h5.attrs.get("fps", 30))
        )
        sampled_indices = _sample_indices(h5, frame_count, used_fps, use_timestamps=use_timestamps)
        stem = path.stem
        combined_path = output_dir / f"{stem}_three_views.mp4"
        _write_video(combined_path, _combined_frames(h5, target_height, sampled_indices), used_fps)
        written.append(combined_path)

        if individual:
            for camera_name in CAMERA_NAMES:
                camera_path = output_dir / f"{stem}_{camera_name}.mp4"
                _write_video(camera_path, (images[camera_name][idx] for idx in sampled_indices), used_fps)
                written.append(camera_path)
    if transcode_x264:
        for video_path in written:
            _transcode_to_x264(video_path, crf=x264_crf, preset=x264_preset)
    return written


def expected_outputs(path: Path, output_dir: Path, individual: bool) -> list[Path]:
    stem = path.stem
    outputs = [output_dir / f"{stem}_three_views.mp4"]
    if individual:
        outputs.extend(output_dir / f"{stem}_{camera_name}.mp4" for camera_name in CAMERA_NAMES)
    return outputs


def export_episode_worker(task: tuple[str, str, float | None, int, bool, bool, bool, int, str]) -> list[str]:
    path, output_dir, fps, target_height, individual, use_timestamps, transcode_x264, x264_crf, x264_preset = task
    return [
        str(written_path)
        for written_path in export_episode(
            Path(path),
            Path(output_dir),
            fps=fps,
            target_height=target_height,
            individual=individual,
            use_timestamps=use_timestamps,
            transcode_x264=transcode_x264,
            x264_crf=x264_crf,
            x264_preset=x264_preset,
        )
    ]


def iter_input_paths(input_path: Path, pattern: str, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export three-view MP4 previews from VLA HDF5 episodes.")
    parser.add_argument("--input", required=True, help="Input episode_*.hdf5 path or directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <input_dir>/video_preview.")
    parser.add_argument("--glob", default="episode_*.hdf5", help="File pattern used when --input is a directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search --input when it is a directory.")
    parser.add_argument("--limit", type=int, default=None, help="Only export the first N matched episodes.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes whose expected MP4 outputs already exist.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes to use.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output fps. By default, infer fps from /timestamps (scheme A), then fallback to attrs['fps'].",
    )
    parser.add_argument(
        "--use-timestamps",
        action="store_true",
        help="Render using /timestamps as real-time axis (resample at output fps, ZOH hold).",
    )
    parser.add_argument("--target-height", type=int, default=480, help="Height used for each view in the combined video.")
    parser.add_argument("--individual", action="store_true", help="Also export one MP4 per camera.")
    parser.add_argument(
        "--no-x264",
        action="store_true",
        help="Disable final ffmpeg transcode to H.264 (libx264).",
    )
    parser.add_argument("--x264-crf", type=int, default=18, help="CRF used for x264 transcode. Lower is higher quality.")
    parser.add_argument("--x264-preset", default="medium", help="x264 preset, e.g. veryfast/medium/slow.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    input_paths = iter_input_paths(input_path, args.glob, bool(args.recursive))
    if args.limit is not None:
        input_paths = input_paths[: max(0, int(args.limit))]
    if not input_paths:
        raise FileNotFoundError(f"No HDF5 files matched {input_path} with pattern {args.glob!r}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_path if input_path.is_dir() else input_path.parent) / "video_preview"
    )
    if args.skip_existing:
        input_paths = [
            path
            for path in input_paths
            if not all(output.exists() for output in expected_outputs(path, output_dir, bool(args.individual)))
        ]

    total_written = 0
    target_height = max(1, int(args.target_height))
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for index, episode_path in enumerate(input_paths, start=1):
            written = export_episode(
                episode_path,
                output_dir,
                fps=args.fps,
                target_height=target_height,
                individual=bool(args.individual),
                use_timestamps=bool(args.use_timestamps),
                transcode_x264=not bool(args.no_x264),
                x264_crf=int(args.x264_crf),
                x264_preset=str(args.x264_preset),
            )
            total_written += len(written)
            print(f"[{index}/{len(input_paths)}] {episode_path}", flush=True)
            for path in written:
                print(path, flush=True)
    else:
        tasks = [
            (
                str(episode_path),
                str(output_dir),
                args.fps,
                target_height,
                bool(args.individual),
                bool(args.use_timestamps),
                not bool(args.no_x264),
                int(args.x264_crf),
                str(args.x264_preset),
            )
            for episode_path in input_paths
        ]
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(export_episode_worker, task): Path(task[0]) for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                episode_path = futures[future]
                written = [Path(path) for path in future.result()]
                total_written += len(written)
                print(f"[{index}/{len(input_paths)}] {episode_path}", flush=True)
                for path in written:
                    print(path, flush=True)
    print(f"exported_episodes={len(input_paths)} exported_videos={total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
