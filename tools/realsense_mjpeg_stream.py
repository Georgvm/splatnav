#!/usr/bin/env python3
"""Headless RealSense D435i color stream as multipart JPEG.

This intentionally mirrors the working Connect-To-Realsense viewer: use the
pyrealsense2 environment, start one rs.pipeline(), and keep it alive.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


RGBD_CONFIGS_TO_TRY = [
    ("640x480 BGR + depth @ 30fps", 640, 480, 30, rs.format.bgr8),
    ("640x480 RGB + depth @ 30fps", 640, 480, 30, rs.format.rgb8),
    ("848x480 BGR + depth @ 30fps", 848, 480, 30, rs.format.bgr8),
    ("848x480 RGB + depth @ 30fps", 848, 480, 30, rs.format.rgb8),
    ("424x240 BGR + depth @ 30fps", 424, 240, 30, rs.format.bgr8),
    ("424x240 RGB + depth @ 30fps", 424, 240, 30, rs.format.rgb8),
    ("640x480 BGR + depth @ 15fps", 640, 480, 15, rs.format.bgr8),
    ("640x480 RGB + depth @ 15fps", 640, 480, 15, rs.format.rgb8),
]

RGB_ONLY_CONFIGS_TO_TRY = [
    ("640x480 BGR only @ 30fps", 640, 480, 30, rs.format.bgr8),
    ("640x480 RGB only @ 30fps", 640, 480, 30, rs.format.rgb8),
    ("848x480 BGR only @ 30fps", 848, 480, 30, rs.format.bgr8),
    ("848x480 RGB only @ 30fps", 848, 480, 30, rs.format.rgb8),
    ("424x240 BGR only @ 30fps", 424, 240, 30, rs.format.bgr8),
    ("424x240 RGB only @ 30fps", 424, 240, 30, rs.format.rgb8),
]


def wait_for_color_frames(pipeline: rs.pipeline, count: int) -> rs.frame:
    color_frame = None
    for _ in range(max(count, 1)):
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("no color frame returned")
    return color_frame


def start_pipeline():
    last_error = None
    for name, width, height, fps, color_format in RGBD_CONFIGS_TO_TRY:
        print(f"Trying {name} ... ", end="", flush=True, file=sys.stderr)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, color_format, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            profile = pipeline.start(config)
            wait_for_color_frames(pipeline, 4)
            print("OK", file=sys.stderr)
            return pipeline, profile, name, color_format
        except RuntimeError as exc:
            print(f"FAILED ({exc})", file=sys.stderr)
            last_error = exc
            try:
                pipeline.stop()
            except RuntimeError:
                pass

    print("RGB+depth did not deliver frames. Trying RGB-only fallback.", file=sys.stderr)
    for name, width, height, fps, color_format in RGB_ONLY_CONFIGS_TO_TRY:
        print(f"Trying {name} ... ", end="", flush=True, file=sys.stderr)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, color_format, fps)
        try:
            profile = pipeline.start(config)
            wait_for_color_frames(pipeline, 4)
            print("OK", file=sys.stderr)
            return pipeline, profile, name, color_format
        except RuntimeError as exc:
            print(f"FAILED ({exc})", file=sys.stderr)
            last_error = exc
            try:
                pipeline.stop()
            except RuntimeError:
                pass

    raise RuntimeError(f"Could not start any RealSense RGB stream: {last_error}")


def frame_to_bgr(color_frame: rs.frame, color_format) -> np.ndarray:
    image = np.asanyarray(color_frame.get_data())
    if color_format == rs.format.rgb8:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def intrinsics_from_profile(profile: rs.pipeline_profile) -> dict[str, float]:
    stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream_profile.get_intrinsics()
    return {
        "width": float(intr.width),
        "height": float(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-out", type=Path)
    parser.add_argument("--intrinsics-out", type=Path)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--max-fps", type=float, default=30.0)
    args = parser.parse_args()

    pipeline = None
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        pipeline, profile, active_config, color_format = start_pipeline()
        intrinsics = intrinsics_from_profile(profile)
        if args.intrinsics_out:
            args.intrinsics_out.write_text(json.dumps(intrinsics, sort_keys=True) + "\n")
        print(f"Active stream: {active_config}", file=sys.stderr, flush=True)

        min_gap = 1.0 / max(args.max_fps, 1.0)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)]
        while running:
            loop_started = time.perf_counter()
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            bgr = frame_to_bgr(color_frame, color_format)
            ok, encoded = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                continue
            jpg = encoded.tobytes()
            if args.latest_out:
                tmp_path = args.latest_out.with_suffix(args.latest_out.suffix + ".tmp")
                tmp_path.write_bytes(jpg)
                tmp_path.replace(args.latest_out)
            sys.stdout.buffer.write(b"--frame\r\n")
            sys.stdout.buffer.write(b"Content-Type: image/jpeg\r\n")
            sys.stdout.buffer.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
            sys.stdout.buffer.write(jpg)
            sys.stdout.buffer.write(b"\r\n")
            sys.stdout.buffer.flush()

            elapsed = time.perf_counter() - loop_started
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)
        return 0
    except Exception as exc:
        print(f"RealSense MJPEG stream failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
