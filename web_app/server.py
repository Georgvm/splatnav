from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


APP_NAME = "splatnav-h100-eval"
PUBLIC_DIR = Path(__file__).resolve().parent / "public"
ROOT_DIR = PUBLIC_DIR.parent.parent
REALSENSE_HELPER = ROOT_DIR / "build" / "realsense_color_bmp"
REALSENSE_LIVE_FRAME = Path("/tmp/realsense_live_latest.bmp")
REALSENSE_LIVE_JPEG = Path("/tmp/realsense_live_latest.jpg")
REALSENSE_LIVE_INTRINSICS = Path("/tmp/realsense_live_intrinsics.json")
REALSENSE_STREAM_LOG = Path("/tmp/realsense_mjpeg_stream.log")
REALSENSE_PYTHON = Path(
    "/Users/georgv.manstein/Downloads/GitHub-Open Mobile Manipulator Project/.venv-realsense/bin/python"
)
REALSENSE_MJPEG_HELPER = ROOT_DIR / "tools" / "realsense_mjpeg_stream.py"
SCENE_PATH = PUBLIC_DIR / "scene_preview.json"
SCENE_FILES = {
    "old_union": "scene_preview.json",
    "pointcloud": "pointcloud_scene_preview.json",
    "stanford": "stanford_university_scene_preview.json",
}


class LocalizeRequest(BaseModel):
    scene: str = "old_union"
    frame: int = Field(default=0, ge=0)
    pose_matrix: list[list[float]] | None = None
    yaw_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    pitch_deg: float = Field(default=0.0, ge=-90.0, le=90.0)
    roll_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    init_yaw_error_deg: float = Field(default=2.0, ge=-45.0, le=45.0)
    init_translation_error_m: float = Field(default=0.05, ge=0.0, le=2.0)


class FrameLocalizeRequest(BaseModel):
    scene: str = "stanford"
    image_base64: str
    pose_matrix: list[list[float]]
    intrinsics: dict[str, float] | None = None
    init_yaw_error_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    init_translation_error_m: float = Field(default=0.0, ge=0.0, le=2.0)
    debug_images: bool = True


class RealSenseLocalizeRequest(BaseModel):
    scene: str = "stanford"
    pose_matrix: list[list[float]]
    init_yaw_error_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    init_translation_error_m: float = Field(default=0.0, ge=0.0, le=2.0)
    debug_images: bool = True


app = FastAPI(title="SplatNav Old Union Live Demo")
_realsense_state: dict[str, Any] = {"pipeline": None, "last_started": 0.0}
_realsense_capture_lock = threading.Lock()
_realsense_stream_lock = threading.Lock()
_realsense_release_lock = threading.Lock()
_realsense_live_lock = threading.Lock()
_realsense_live_proc: subprocess.Popen[bytes] | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/camera-test")
def camera_test() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "camera-test.html")


@app.get("/api/scene")
def scene() -> FileResponse:
    if not SCENE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="scene_preview.json is missing. Run export_old_union_web_assets and pull it from the Modal volume.",
        )
    return FileResponse(SCENE_PATH)


@app.get("/api/scene/{scene_name}")
def named_scene(scene_name: str) -> FileResponse:
    filename = SCENE_FILES.get(scene_name)
    if filename is None:
        raise HTTPException(status_code=404, detail=f"Unknown scene: {scene_name}")
    path = PUBLIC_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Scene file missing: {filename}")
    return FileResponse(path)


@app.post("/api/localize")
async def localize(payload: LocalizeRequest) -> dict[str, Any]:
    function_name = {
        "old_union": "interactive_old_union_localize",
        "stanford": "interactive_stanford_raw_localize",
    }.get(payload.scene)
    if function_name is None:
        raise HTTPException(status_code=400, detail=f"Localization is not configured for scene: {payload.scene}")
    try:
        function = modal.Function.from_name(APP_NAME, function_name)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load deployed Modal function {APP_NAME}.{function_name}: {exc}",
        ) from exc

    kwargs = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    kwargs.pop("scene", None)
    if payload.scene == "stanford":
        kwargs = {
            key: kwargs[key]
            for key in ("pose_matrix", "init_yaw_error_deg", "init_translation_error_m")
            if key in kwargs
        }
    try:
        return await asyncio.to_thread(function.remote, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/localize-frame")
async def localize_frame(payload: FrameLocalizeRequest) -> dict[str, Any]:
    if payload.scene != "stanford":
        raise HTTPException(status_code=400, detail="Real-frame localization is currently wired for scene=stanford")
    function_name = "interactive_stanford_real_frame_localize"
    try:
        function = modal.Function.from_name(APP_NAME, function_name)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load deployed Modal function {APP_NAME}.{function_name}: {exc}",
        ) from exc

    kwargs = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    kwargs.pop("scene", None)
    try:
        return await asyncio.to_thread(function.remote, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _parse_realsense_intrinsics(stderr: str) -> dict[str, float] | None:
    match = re.search(
        r"color\s+(\d+)x(\d+)\s+fx=([0-9.]+)\s+fy=([0-9.]+)\s+ppx=([0-9.]+)\s+ppy=([0-9.]+)",
        stderr,
    )
    if not match:
        return None
    return {
        "width": float(match.group(1)),
        "height": float(match.group(2)),
        "fx": float(match.group(3)),
        "fy": float(match.group(4)),
        "cx": float(match.group(5)),
        "cy": float(match.group(6)),
    }


def _read_live_realsense_intrinsics() -> dict[str, float] | None:
    try:
        data = json.loads(REALSENSE_LIVE_INTRINSICS.read_text())
    except Exception:
        return None
    keys = ("width", "height", "fx", "fy", "cx", "cy")
    if not all(key in data for key in keys):
        return None
    return {key: float(data[key]) for key in keys}


def _read_live_realsense_image(max_age_seconds: float = 2.5) -> tuple[bytes, str, dict[str, float], list[str]] | None:
    try:
        age = time.time() - REALSENSE_LIVE_JPEG.stat().st_mtime
    except FileNotFoundError:
        age = None
    if age is not None and age <= max_age_seconds:
        image_bytes = REALSENSE_LIVE_JPEG.read_bytes()
        if image_bytes.startswith(b"\xff\xd8"):
            intrinsics = _read_live_realsense_intrinsics() or {
                "width": 640.0,
                "height": 480.0,
                "fx": 576.0,
                "fy": 576.0,
                "cx": 320.0,
                "cy": 240.0,
            }
            return image_bytes, "image/jpeg", intrinsics, [f"Using cached live RGB JPEG frame ({age:.2f}s old)"]

    try:
        age = time.time() - REALSENSE_LIVE_FRAME.stat().st_mtime
    except FileNotFoundError:
        return None
    if age > max_age_seconds:
        return None
    image_bytes = REALSENSE_LIVE_FRAME.read_bytes()
    if not image_bytes.startswith(b"BM"):
        return None
    intrinsics = _read_live_realsense_intrinsics() or {
        "width": 848.0,
        "height": 480.0,
        "fx": 763.2,
        "fy": 763.2,
        "cx": 424.0,
        "cy": 240.0,
    }
    return image_bytes, "image/bmp", intrinsics, [f"Using cached live RGB BMP frame ({age:.2f}s old)"]


def _read_live_realsense_bmp(max_age_seconds: float = 2.5) -> tuple[bytes, dict[str, float], list[str]] | None:
    capture = _read_live_realsense_image(max_age_seconds=max_age_seconds)
    if capture is None:
        return None
    image_bytes, media_type, intrinsics, logs = capture
    if media_type != "image/bmp":
        return None
    return image_bytes, intrinsics, logs


def _release_realsense_owners() -> list[str]:
    logs: list[str] = []
    with _realsense_release_lock:
        subprocess.run(["pkill", "-f", "realsense_mjpeg_stream.py"], capture_output=True, text=True, timeout=2.0)
        for name in ("realsense_color_bmp", "UVCAssistant", "VDCAssistant", "cameracaptured"):
            try:
                proc = subprocess.run(["killall", name], capture_output=True, text=True, timeout=2.0)
            except Exception as exc:
                logs.append(f"killall {name}: {type(exc).__name__}: {exc}")
                continue
            if proc.returncode == 0:
                logs.append(f"stopped {name}")
            elif proc.stderr.strip():
                logs.append(f"killall {name}: {proc.stderr.strip()}")
        time.sleep(0.75)
    return logs


def _capture_realsense_bmp(release: bool = True, fast: bool = False) -> tuple[bytes, dict[str, float], list[str]]:
    if not REALSENSE_HELPER.exists():
        raise HTTPException(status_code=503, detail=f"RealSense helper missing: {REALSENSE_HELPER}")

    output_path = Path("/tmp/realsense_color_frame.bmp")
    logs = _release_realsense_owners() if release else ["Skipped release step for fast polling"]
    attempts = [(848, 480, 30)] if fast else [(848, 480, 30), (640, 480, 30), (640, 360, 30), (424, 240, 30)]
    errors = []
    for width, height, fps in attempts:
        command = [
            str(REALSENSE_HELPER),
            "--out",
            str(output_path),
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            str(fps),
            "--warmup",
            "1",
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=15.0)
        except subprocess.TimeoutExpired:
            errors.append(f"{width}x{height}@{fps}: timed out")
            continue

        output = (proc.stderr or proc.stdout).strip()
        if output:
            logs.append(output)
        if proc.returncode == 0:
            intrinsics = _parse_realsense_intrinsics(proc.stderr or proc.stdout) or {
                "width": float(width),
                "height": float(height),
                "fx": float(max(width, height) * 0.9),
                "fy": float(max(width, height) * 0.9),
                "cx": float(width / 2),
                "cy": float(height / 2),
            }
            return output_path.read_bytes(), intrinsics, logs
        errors.append(f"{width}x{height}@{fps}: {output}")

    raise HTTPException(
        status_code=503,
        detail=(
            "Native RealSense RGB helper could open the RGB camera but did not receive a frame. "
            "Close any browser camera preview, unplug/replug the D435i, then retry Backend frame. "
            f"Attempts: {' | '.join(errors)}"
        ),
    )


def _open_realsense_bmp_stream():
    if not REALSENSE_HELPER.exists():
        raise HTTPException(status_code=503, detail=f"RealSense helper missing: {REALSENSE_HELPER}")
    if not _realsense_stream_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A RealSense live stream is already running")

    _release_realsense_owners()
    REALSENSE_STREAM_LOG.write_text("Starting pyrealsense2 MJPEG helper...\n")
    command = [
        str(REALSENSE_HELPER),
        "--stream",
        "--latest-out",
        str(REALSENSE_LIVE_FRAME),
        "--intrinsics-out",
        str(REALSENSE_LIVE_INTRINSICS),
        "--width",
        "848",
        "--height",
        "480",
        "--fps",
        "30",
        "--warmup",
        "1",
    ]
    stderr_file = REALSENSE_STREAM_LOG.open("ab")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file, bufsize=0)
    time.sleep(1.0)
    if proc.poll() is not None:
        stderr_file.close()
        _realsense_stream_lock.release()
        log_text = REALSENSE_STREAM_LOG.read_text(errors="replace")[-4000:]
        raise HTTPException(status_code=503, detail=f"RealSense pyrealsense2 stream exited at startup:\n{log_text}")

    def stream_chunks():
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            stderr_file.close()
            _realsense_stream_lock.release()

    return stream_chunks()


def _open_realsense_python_mjpeg_stream():
    if not REALSENSE_PYTHON.exists():
        raise HTTPException(status_code=503, detail=f"RealSense Python venv missing: {REALSENSE_PYTHON}")
    if not REALSENSE_MJPEG_HELPER.exists():
        raise HTTPException(status_code=503, detail=f"RealSense MJPEG helper missing: {REALSENSE_MJPEG_HELPER}")
    if not _realsense_stream_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A RealSense live stream is already running")

    _release_realsense_owners()
    command = [
        str(REALSENSE_PYTHON),
        str(REALSENSE_MJPEG_HELPER),
        "--latest-out",
        str(REALSENSE_LIVE_JPEG),
        "--intrinsics-out",
        str(REALSENSE_LIVE_INTRINSICS),
        "--quality",
        "85",
        "--max-fps",
        "30",
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    def stream_chunks():
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            _realsense_stream_lock.release()

    return stream_chunks()


def _stop_realsense_live_process() -> None:
    global _realsense_live_proc
    proc = _realsense_live_proc
    _realsense_live_proc = None
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


def _start_realsense_live_process(restart: bool = False) -> dict[str, Any]:
    global _realsense_live_proc
    if not REALSENSE_PYTHON.exists():
        raise HTTPException(status_code=503, detail=f"RealSense Python venv missing: {REALSENSE_PYTHON}")
    if not REALSENSE_MJPEG_HELPER.exists():
        raise HTTPException(status_code=503, detail=f"RealSense MJPEG helper missing: {REALSENSE_MJPEG_HELPER}")

    with _realsense_live_lock:
        if restart:
            _stop_realsense_live_process()
            _release_realsense_owners()
        proc = _realsense_live_proc
        if proc is not None and proc.poll() is None:
            return {"running": True, "reused": True, "pid": proc.pid}

        REALSENSE_STREAM_LOG.write_text("Starting pyrealsense2 latest-frame helper...\n")
        command = [
            str(REALSENSE_PYTHON),
            str(REALSENSE_MJPEG_HELPER),
            "--latest-out",
            str(REALSENSE_LIVE_JPEG),
            "--intrinsics-out",
            str(REALSENSE_LIVE_INTRINSICS),
            "--quality",
            "85",
            "--max-fps",
            "30",
        ]
        stderr_file = REALSENSE_STREAM_LOG.open("ab")
        _realsense_live_proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            bufsize=0,
        )
        proc = _realsense_live_proc

    started = time.time()
    while time.time() - started < 6.0:
        if proc.poll() is not None:
            log_text = REALSENSE_STREAM_LOG.read_text(errors="replace")[-4000:]
            raise HTTPException(status_code=503, detail=f"RealSense latest-frame helper exited:\n{log_text}")
        try:
            if time.time() - REALSENSE_LIVE_JPEG.stat().st_mtime < 2.0:
                return {"running": True, "reused": False, "pid": proc.pid}
        except FileNotFoundError:
            pass
        time.sleep(0.1)

    log_text = REALSENSE_STREAM_LOG.read_text(errors="replace")[-4000:] if REALSENSE_STREAM_LOG.exists() else ""
    return {"running": proc.poll() is None, "reused": False, "pid": proc.pid, "warning": "No fresh frame yet", "log": log_text}


@app.post("/api/localize-realsense-frame")
async def localize_realsense_frame(payload: RealSenseLocalizeRequest) -> dict[str, Any]:
    if payload.scene != "stanford":
        raise HTTPException(status_code=400, detail="D435i localization is currently wired for scene=stanford")
    if not _realsense_capture_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A RealSense RGB capture is already running")
    try:
        live_capture = _read_live_realsense_image()
        if live_capture is None:
            image_bytes, intrinsics, capture_logs = _capture_realsense_bmp()
            media_type = "image/bmp"
        else:
            image_bytes, media_type, intrinsics, capture_logs = live_capture
    finally:
        _realsense_capture_lock.release()

    function_name = "interactive_stanford_real_frame_localize"
    try:
        function = modal.Function.from_name(APP_NAME, function_name)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load deployed Modal function {APP_NAME}.{function_name}: {exc}",
        ) from exc

    image_base64 = f"data:{media_type};base64," + base64.b64encode(image_bytes).decode("ascii")
    kwargs = {
        "image_base64": image_base64,
        "pose_matrix": payload.pose_matrix,
        "intrinsics": intrinsics,
        "init_yaw_error_deg": payload.init_yaw_error_deg,
        "init_translation_error_m": payload.init_translation_error_m,
        "debug_images": payload.debug_images,
    }
    try:
        result = await asyncio.to_thread(function.remote, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result.setdefault("logs", [])
    result["logs"] = [f"RealSense: {line}" for line in capture_logs] + result["logs"]
    result["realsense_intrinsics"] = intrinsics
    return result


def _run_probe(cmd: list[str], timeout: float = 5.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "seconds": time.perf_counter() - started,
        }
    except FileNotFoundError as exc:
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": str(exc), "seconds": 0}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nTimed out after {timeout}s",
            "seconds": time.perf_counter() - started,
        }


def _usb_realsense_status() -> dict[str, Any]:
    probe = _run_probe(["ioreg", "-p", "IOUSB", "-l", "-w", "0"], timeout=6.0)
    text = f"{probe['stdout']}\n{probe['stderr']}"
    marker = "Intel(R) RealSense(TM) Depth Camera 435i"
    serial = None
    if "043323051340" in text:
        serial = "043323051340"
    return {
        "seen": marker in text,
        "name": marker if marker in text else None,
        "serial": serial,
        "probe": {key: probe[key] for key in ("returncode", "stderr", "seconds")},
    }


@app.get("/api/realsense/status")
def realsense_status() -> dict[str, Any]:
    try:
        import pyrealsense2 as rs  # type: ignore

        devices = []
        for device in rs.context().query_devices():
            devices.append(
                {
                    "name": device.get_info(rs.camera_info.name),
                    "serial": device.get_info(rs.camera_info.serial_number),
                    "firmware": device.get_info(rs.camera_info.firmware_version),
                }
            )
        python_binding = {"available": True, "devices": devices}
    except Exception as exc:
        python_binding = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    librealsense_cli = _run_probe(["rs-enumerate-devices"], timeout=6.0)
    return {
        "usb": _usb_realsense_status(),
        "python_binding": python_binding,
        "librealsense_cli": librealsense_cli,
    }


@app.get("/api/realsense/stream-status")
def realsense_stream_status() -> dict[str, Any]:
    log_text = ""
    if REALSENSE_STREAM_LOG.exists():
        log_text = REALSENSE_STREAM_LOG.read_text(errors="replace")[-8000:]
    latest_age = None
    if REALSENSE_LIVE_JPEG.exists():
        latest_age = time.time() - REALSENSE_LIVE_JPEG.stat().st_mtime
    return {
        "python": str(REALSENSE_PYTHON),
        "helper": str(REALSENSE_MJPEG_HELPER),
        "python_exists": REALSENSE_PYTHON.exists(),
        "helper_exists": REALSENSE_MJPEG_HELPER.exists(),
        "latest_jpeg_exists": REALSENSE_LIVE_JPEG.exists(),
        "latest_jpeg_age_seconds": latest_age,
        "log": log_text,
    }


@app.post("/api/realsense/live/start")
def realsense_live_start(restart: bool = True) -> dict[str, Any]:
    return _start_realsense_live_process(restart=restart)


@app.post("/api/realsense/live/stop")
def realsense_live_stop() -> dict[str, Any]:
    with _realsense_live_lock:
        _stop_realsense_live_process()
    return {"ok": True}


@app.post("/api/realsense/release")
def realsense_release() -> dict[str, Any]:
    logs = _release_realsense_owners()
    return {"ok": True, "logs": logs}


@app.get("/api/realsense/latest.jpg")
def realsense_latest_frame() -> Response:
    capture = _read_live_realsense_image(max_age_seconds=5.0)
    if capture is None:
        raise HTTPException(status_code=404, detail="No fresh RealSense frame is available yet")
    image_bytes, media_type, _, _ = capture
    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/realsense/frame.jpg")
def realsense_frame(release: bool = True, fast: bool = False) -> Response:
    if not _realsense_capture_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A RealSense RGB capture is already running")
    if REALSENSE_HELPER.exists():
        try:
            live_capture = _read_live_realsense_bmp()
            if live_capture is None:
                image_bytes, _, _ = _capture_realsense_bmp(release=release, fast=fast)
            else:
                image_bytes, _, _ = live_capture
            return Response(content=image_bytes, media_type="image/bmp")
        finally:
            _realsense_capture_lock.release()

    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        _realsense_capture_lock.release()
        raise HTTPException(
            status_code=503,
            detail=f"pyrealsense2 is not installed in this Python runtime: {type(exc).__name__}: {exc}",
        ) from exc

    try:
        import cv2  # type: ignore
        import numpy as np

        pipeline = _realsense_state.get("pipeline")
        if pipeline is None:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            _realsense_state["pipeline"] = pipeline
            _realsense_state["last_started"] = time.time()

        frames = pipeline.wait_for_frames(5000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame returned by RealSense")
        image = np.asanyarray(color.get_data())
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("Could not JPEG-encode RealSense frame")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")
    except Exception as exc:
        _realsense_state["pipeline"] = None
        raise HTTPException(status_code=503, detail=f"RealSense frame capture failed: {exc}") from exc
    finally:
        _realsense_capture_lock.release()


@app.get("/api/realsense/stream")
def realsense_stream() -> StreamingResponse:
    return StreamingResponse(
        _open_realsense_python_mjpeg_stream() if REALSENSE_PYTHON.exists() else _open_realsense_bmp_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
