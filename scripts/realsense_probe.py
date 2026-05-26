from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a RealSense D435i RGB stream and save one frame.")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("/tmp/realsense_rgb_probe.jpg"))
    args = parser.parse_args()

    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install librealsense/pyrealsense2, or use the browser webcam path first.\n"
            f"Import error: {exc}"
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        print("RealSense RGB stream")
        print(f"  resolution: {intr.width}x{intr.height} @ {args.fps} FPS")
        print(f"  fx/fy:      {intr.fx:.4f} / {intr.fy:.4f}")
        print(f"  cx/cy:      {intr.ppx:.4f} / {intr.ppy:.4f}")
        print(f"  model:      {intr.model}")
        print(f"  coeffs:     {[round(value, 6) for value in intr.coeffs]}")

        frames = None
        for _ in range(30):
            frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame() if frames else None
        if not color_frame:
            raise RuntimeError("No RGB frame received from RealSense")

        import numpy as np

        image = np.asanyarray(color_frame.get_data())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.out), image):
            raise RuntimeError(f"Could not write frame to {args.out}")
        print(f"  saved:      {args.out}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
