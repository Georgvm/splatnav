from __future__ import annotations

import json
import os
import base64
import shutil
import subprocess
from pathlib import Path

import modal


APP_DIR = Path("/root/splatnav")
VOL_DIR = Path("/vol")
GDRIVE_DATA_URL = "https://drive.google.com/drive/folders/1K0zfpuAti43YIBK5APFd-Yv73CvljgMC?usp=sharing"
SCENE_DRIVE_FOLDERS = {
    "old_union": {
        "data_id": "1KzuHWF85t436ptyORlBraB8-Oa1wg19E",
        "data_path": "data/old_union2",
        "images_id": "1OdIF99R6-jhS4fJqr6K9y1giZ3G9y-Kk",
        "transforms_id": "1K0xBG5RCiTMaqRQZA36kqdP0o6NMUBAl",
        "output_id": "1McdaM76WWjxQ_0A_oOE1gIXqWYGjftKg",
        "output_path": "outputs/old_union2/splatfacto/2024-09-02_151414",
    },
    "flight": {
        "data_id": "1BYuXUAQ1VCcNxti_GS7qFgSWtuXPrlhC",
        "data_path": "data/flight",
        "images_id": "1JUCAVx7Aok5Hbs-OklqV4IfkI77cAuEM",
        "transforms_id": "1YgsFyoSmZCOeEBGtCZdZScKaiEZjrpA5",
        "output_id": "19HPDVxcV-8a9OP2kzmVeF_0oZhtA7Tp-",
        "output_path": "outputs/flight/splatfacto/2024-09-12_172434",
    },
    "statues": {
        "data_id": "1QccMAY5s-lJScuQIqzZkGHxODugoUh-2",
        "data_path": "data/statues",
        "images_id": "1XoBi8DVWHKC-tvY40Ua5-JLceDPRd8ml",
        "transforms_id": "1Cwupg41rBD35zzSSnfAGfiMR9ZC4CNJq",
        "output_id": "1ReWEpSiRuSSntXoR4Beq3Z3wL3-Dxjgc",
        "output_path": "outputs/statues/splatfacto/2024-09-11_095852",
    },
    "stonehenge": {
        "data_id": "1QbaOAGk1UjfxWvjuEP7lDVfRYKrqUyQs",
        "data_path": "data/stonehenge",
        "images_id": "1TYOpcctpDkdy9MemyZq7IJam8zsnAymA",
        "transforms_id": "1dgmFoO_1jIWx53zYAYl2RIJsaPNO2Bof",
        "output_id": "1jlL4_aW0K0pSRU9bYNoiONofRTHNw9Kd",
        "output_path": "outputs/stonehenge/splatfacto/2024-09-11_100724",
    },
}
CUSTOM_SCENES = {
    "parkinglot": {
        "data_path": "data/parkinglot",
        "archive_path": "tmp/parkinglot_dataset.tar.gz",
        "archive_member": "Splatcam_iphone_data_parkinglot",
        "output_path": "outputs/parkinglot/splatfacto",
    }
}

app = modal.App("splatnav-h100-eval")
volume = modal.Volume.from_name("splatnav-eval-data", create_if_missing=True)
_interactive_state = {}

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "clang",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "ninja-build",
        "wget",
    )
    .pip_install_from_requirements("requirements-modal.txt")
    .env(
        {
            "MPLBACKEND": "Agg",
            "NERFSTUDIO_CACHE_DIR": "/vol/cache/nerfstudio",
            "TORCH_HOME": "/vol/cache/torch",
            "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions",
        }
    )
    .workdir(str(APP_DIR))
    .add_local_dir(
        ".",
        str(APP_DIR),
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            "data",
            "outputs",
            "trajs",
            "figures",
            "nerfnav_paths",
        ],
    )
    .add_local_file(
        "/Users/georgv.manstein/Downloads/Stanford University.ply",
        str(APP_DIR / "external_assets" / "stanford_university.ply"),
    )
)


def _run(cmd: list[str], cwd: Path = APP_DIR) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _find_dir(root: Path, dirname: str) -> Path | None:
    direct = root / dirname
    if direct.exists():
        return direct
    matches = [p for p in root.rglob(dirname) if p.is_dir()]
    if not matches:
        return None
    return min(matches, key=lambda p: len(p.parts))


def _link_volume_data() -> None:
    for name in ("data", "outputs"):
        source = _find_dir(VOL_DIR, name)
        target = APP_DIR / name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        if source is not None:
            target.symlink_to(source, target_is_directory=True)


def _latest_config(root: Path) -> Path | None:
    configs = sorted(root.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return configs[0] if configs else None


def _download_folder(gdown, folder_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        print(f"Using existing {output}", flush=True)
        return
    print(f"Downloading Google Drive folder {folder_id} to {output}", flush=True)
    gdown.download_folder(
        id=folder_id,
        output=str(output),
        quiet=False,
        use_cookies=False,
        resume=True,
    )


def _download_file(gdown, file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return
    print(f"Downloading Google Drive file {file_id} to {output}", flush=True)
    gdown.download(
        url=f"https://drive.google.com/uc?id={file_id}",
        output=str(output),
        quiet=False,
        use_cookies=False,
        resume=True,
    )


def _list_drive_folder(gdown, folder_id: str):
    import importlib

    module = importlib.import_module("gdown.download_folder")
    sess, _ = module._get_session(
        proxy=None,
        use_cookies=False,
        user_agent="Mozilla/5.0",
    )
    _, children = module._parse_embedded_folder_view(sess, folder_id, verify=True)
    return children


def _download_dataset(gdown, folders: dict, max_frames: int, full_data: bool = False) -> None:
    data_root = VOL_DIR / folders["data_path"]
    transforms_path = data_root / "transforms.json"
    full_transforms_path = data_root / "transforms.full.json"
    _download_file(gdown, folders["transforms_id"], transforms_path)
    if not full_transforms_path.exists():
        shutil.copyfile(transforms_path, full_transforms_path)

    image_dir = data_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_files = [
        (file_id, name)
        for file_id, name, mime_type in _list_drive_folder(gdown, folders["images_id"])
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
        and not mime_type.endswith("folder")
    ]
    image_files.sort(key=lambda item: item[1])
    selected_images = image_files if full_data else image_files[: max(max_frames, 16)]
    selected_names = set()
    for file_id, name in selected_images:
        try:
            _download_file(gdown, file_id, image_dir / name)
        except Exception as exc:
            if full_data:
                print(f"Skipping {name}: {exc}", flush=True)
                continue
            raise
        if (image_dir / name).exists():
            selected_names.add(name)

    transforms = json.loads(full_transforms_path.read_text())
    frames = transforms.get("frames", [])
    transforms["frames"] = [
        frame
        for frame in frames
        if Path(frame.get("file_path", "")).name in selected_names
    ]
    transforms_path.write_text(json.dumps(transforms, indent=2))
    print(
        f"Wrote {len(transforms['frames'])} frame entries to {transforms_path}",
        flush=True,
    )


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={str(VOL_DIR): volume},
)
def unpack_custom_scene(scene: str = "parkinglot") -> dict:
    import tarfile

    config = CUSTOM_SCENES[scene]
    archive = VOL_DIR / config["archive_path"]
    output = VOL_DIR / config["data_path"]
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive in Modal volume: {archive}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (output / "transforms.json").exists():
        return {"scene": scene, "data_path": str(output), "status": "already_unpacked"}

    staging = VOL_DIR / "tmp" / f"{scene}_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} to {staging}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging)

    extracted = staging / config["archive_member"]
    if not extracted.exists():
        raise FileNotFoundError(f"Archive did not contain {config['archive_member']}")
    if output.exists():
        shutil.rmtree(output)
    shutil.move(str(extracted), str(output))
    shutil.rmtree(staging)
    volume.commit()
    return {"scene": scene, "data_path": str(output), "status": "unpacked"}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={str(VOL_DIR): volume},
)
def run_splatloc_eval(
    scene: str = "old_union",
    run_name: str = "test_runs",
    max_frames: int = 3,
    stride: int = 1,
    detector: str = "sift",
    camera_source: str = "external",
    reset_init_each_frame: bool = False,
    rotation_noise_deg: float = 0.0,
    translation_noise_m: float = 0.0,
    download_data: bool = False,
    full_data: bool = False,
    gdrive_url: str = GDRIVE_DATA_URL,
) -> dict:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    VOL_DIR.mkdir(parents=True, exist_ok=True)
    if download_data:
        import gdown

        if gdrive_url != GDRIVE_DATA_URL:
            if _find_dir(VOL_DIR, "data") is None or _find_dir(VOL_DIR, "outputs") is None:
                print(f"Downloading eval/model data from {gdrive_url}", flush=True)
                gdown.download_folder(
                    gdrive_url,
                    output=str(VOL_DIR),
                    quiet=False,
                    use_cookies=False,
                    resume=True,
                )
        else:
            folders = SCENE_DRIVE_FOLDERS[scene]
            _download_dataset(gdown, folders, max_frames * stride, full_data=full_data)
            _download_folder(gdown, folders["output_id"], VOL_DIR / folders["output_path"])
        volume.commit()

    _link_volume_data()

    result_dir = VOL_DIR / "results" / scene / run_name
    result_path = result_dir / "metrics.json"
    debug_dir = result_dir / "debug"
    if result_path.exists():
        result_path.unlink()

    cmd = [
        "python",
        "scripts/run_splatloc_eval.py",
        "--scene",
        scene,
        "--max-frames",
        str(max_frames),
        "--stride",
        str(stride),
        "--detector",
        detector,
        "--camera-source",
        camera_source,
        "--rotation-noise-deg",
        str(rotation_noise_deg),
        "--translation-noise-m",
        str(translation_noise_m),
        "--debug-dir",
        str(debug_dir),
        "--output",
        str(result_path),
    ]
    if reset_init_each_frame:
        cmd.insert(cmd.index("--rotation-noise-deg"), "--reset-init-each-frame")
    _run(cmd)
    return json.loads(result_path.read_text())


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={str(VOL_DIR): volume},
)
def run_video_stride_ablation(scene: str = "old_union") -> dict:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    _link_volume_data()

    configs = [
        {"run_name": "ablation_track_stride1", "max_frames": 40, "stride": 1, "reset": False},
        {"run_name": "ablation_track_stride5", "max_frames": 20, "stride": 5, "reset": False},
        {"run_name": "ablation_track_stride10", "max_frames": 20, "stride": 10, "reset": False},
        {"run_name": "ablation_independent_stride10", "max_frames": 20, "stride": 10, "reset": True},
    ]
    results = {}
    for config in configs:
        result_dir = VOL_DIR / "results" / scene / config["run_name"]
        result_path = result_dir / "metrics.json"
        debug_dir = result_dir / "debug"
        if result_path.exists():
            result_path.unlink()

        cmd = [
            "python",
            "scripts/run_splatloc_eval.py",
            "--scene",
            scene,
            "--max-frames",
            str(config["max_frames"]),
            "--stride",
            str(config["stride"]),
            "--detector",
            "sift",
            "--camera-source",
            "external",
            "--rotation-noise-deg",
            "0",
            "--translation-noise-m",
            "0",
            "--debug-dir",
            str(debug_dir),
            "--output",
            str(result_path),
        ]
        if config["reset"]:
            cmd.insert(cmd.index("--rotation-noise-deg"), "--reset-init-each-frame")
        _run(cmd)
        metrics = json.loads(result_path.read_text())
        results[config["run_name"]] = metrics
        print(
            f"{config['run_name']}: "
            f"{metrics['num_successful_frames']}/{metrics['num_frames']} succeeded, "
            f"rot_mean={metrics['rotation_deg_mean']}, "
            f"trans_mean={metrics['translation_m_mean']}",
            flush=True,
        )

    summary_path = VOL_DIR / "results" / scene / "video_stride_ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    volume.commit()
    return results


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={str(VOL_DIR): volume},
)
def train_custom_splatfacto(
    scene: str = "parkinglot",
    max_num_iterations: int = 7000,
    steps_per_save: int = 3500,
    vis: str = "viewer",
) -> dict:
    import torch

    config = CUSTOM_SCENES[scene]
    _link_volume_data()
    data_path = VOL_DIR / config["data_path"]
    output_path = VOL_DIR / config["output_path"]
    if not (data_path / "transforms.json").exists():
        raise FileNotFoundError(f"Missing transforms.json at {data_path}; run unpack_custom_scene first")

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-train",
        "splatfacto",
        "--data",
        str(data_path),
        "--output-dir",
        str(output_path.parent.parent),
        "--max-num-iterations",
        str(max_num_iterations),
        "--steps-per-save",
        str(steps_per_save),
        "--vis",
        vis,
    ]
    _run(cmd)
    latest = _latest_config(output_path)
    volume.commit()
    return {
        "scene": scene,
        "data_path": str(data_path),
        "output_root": str(output_path),
        "latest_config": None if latest is None else str(latest),
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(VOL_DIR): volume},
)
def export_old_union_web_assets(max_points: int = 60000) -> dict:
    import numpy as np
    import torch

    from ns_utils.nerfstudio_utils import load_dataset
    from splat.splat_utils import GSplatLoader

    _link_volume_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gsplat = GSplatLoader(Path("outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"), device)

    opacities = gsplat.opacities.detach().flatten()
    count = min(max_points, int(gsplat.means.shape[0]))
    if count < gsplat.means.shape[0]:
        indices = torch.topk(opacities, k=count).indices
    else:
        indices = torch.arange(gsplat.means.shape[0], device=device)

    points = gsplat.means[indices].detach().cpu().numpy().astype(np.float32)
    colors = np.clip(gsplat.colors[indices].detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
    alpha = gsplat.opacities[indices].detach().cpu().numpy().reshape(-1).astype(np.float32)

    dataset = load_dataset(data_path=Path("data/old_union2"), dataset_mode="all")
    camera_positions = []
    for idx in range(len(dataset)):
        pose = torch.eye(4)
        pose[:3] = dataset.cameras.camera_to_worlds[idx]
        if idx % 5 == 0:
            camera_positions.append(
                {
                    "frame": idx,
                    "position": pose[:3, 3].detach().cpu().numpy().astype(float).tolist(),
                    "pose": pose.detach().cpu().numpy().astype(float).tolist(),
                }
            )

    output_dir = VOL_DIR / "results" / "old_union" / "web_app"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scene_preview.json"
    output_path.write_text(
        json.dumps(
            {
                "points": points.tolist(),
                "colors": colors.tolist(),
                "opacities": alpha.tolist(),
                "camera_positions": camera_positions,
            }
        )
    )
    volume.commit()
    return {
        "points": int(points.shape[0]),
        "camera_positions": len(camera_positions),
        "path": str(output_path),
    }


def _camera_delta_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float):
    import math
    import numpy as np

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return ry @ rx @ rz


def _read_gaussian_ply(path: Path):
    import numpy as np

    properties = []
    vertex_count = None
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                _, ply_type, name = line.split()
                if ply_type != "float":
                    raise ValueError(f"Unsupported Stanford PLY property type: {ply_type}")
                properties.append((name, "<f4"))
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError("Missing vertex count in PLY header")
        data = np.fromfile(handle, dtype=np.dtype(properties), count=vertex_count)

    means = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    colors = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)
    colors = np.clip(colors * 0.28209479177387814 + 0.5, 0.0, 1.0)
    opacities = data["opacity"].astype(np.float32)
    scales = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32)
    quats = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float32)
    return means, quats, scales, opacities, colors


def _decode_base64_rgb_image(image_base64: str):
    import base64
    import cv2
    import numpy as np

    payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(payload)
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image_base64 as an RGB image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class RawGaussianSplat:
    def __init__(
        self,
        ply_path: Path,
        device,
        width: int = 720,
        height: int = 960,
        fx: float = 665.6,
        fy: float = 665.6,
        cx: float | None = None,
        cy: float | None = None,
        max_gaussians: int = 350_000,
    ) -> None:
        import numpy as np
        import torch

        self.device = device
        self.configure_camera(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
        means, quats, scales, opacities, colors = _read_gaussian_ply(ply_path)
        opacity_prob = 1.0 / (1.0 + np.exp(-opacities))
        if len(means) > max_gaussians:
            idx = np.argpartition(opacity_prob, -max_gaussians)[-max_gaussians:]
            means, quats, scales, opacities, colors = (
                means[idx],
                quats[idx],
                scales[idx],
                opacities[idx],
                colors[idx],
            )
        self.means = torch.tensor(means, dtype=torch.float32, device=device)
        self.quats = torch.nn.functional.normalize(
            torch.tensor(quats, dtype=torch.float32, device=device), dim=-1
        )
        self.scales = torch.exp(torch.tensor(scales, dtype=torch.float32, device=device))
        self.opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float32, device=device))
        self.colors = torch.tensor(colors, dtype=torch.float32, device=device)

    def configure_camera(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float | None = None,
        cy: float | None = None,
    ) -> None:
        import torch

        self.width = int(width)
        self.height = int(height)
        self.K = torch.tensor(
            [
                [float(fx), 0.0, float(cx if cx is not None else width / 2)],
                [0.0, float(fy), float(cy if cy is not None else height / 2)],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def get_camera_intrinsics(self):
        return self.height, self.width, self.K

    def _viewmat_from_pose(self, pose):
        import torch

        pose_cv = pose.detach().clone()
        pose_cv[:, 1] = -pose_cv[:, 1]
        pose_cv[:, 2] = -pose_cv[:, 2]
        return torch.linalg.inv(pose_cv)

    def render(self, pose, compute_semantics: bool = False, debug_mode: bool = False):
        from gsplat import rasterization
        import torch

        viewmat = self._viewmat_from_pose(pose).unsqueeze(0)
        colors, alphas, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmat,
            Ks=self.K.unsqueeze(0),
            width=self.width,
            height=self.height,
            near_plane=0.01,
            far_plane=5000.0,
            render_mode="RGB+ED",
            packed=True,
        )
        rgb = colors[0, ..., :3].clamp(0.0, 1.0)
        depth = colors[0, ..., 3:4]
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return {"rgb": rgb, "depth": depth}

    def generate_RGBD_point_cloud(
        self,
        pose,
        save_image: bool = False,
        filename: str = "/",
        compute_semantics: bool = False,
        max_depth: float | None = None,
        return_pcd: bool = False,
        positives: str = "",
        negatives: str = "",
    ):
        import torch
        import matplotlib.pyplot as plt

        outputs = self.render(pose, compute_semantics=compute_semantics)
        if save_image:
            fig, axs = plt.subplots(2, 1, figsize=(12, 12))
            axs[0].imshow(outputs["rgb"].detach().cpu().numpy())
            axs[1].imshow(outputs["depth"].squeeze(-1).detach().cpu().numpy())
            for ax in axs:
                ax.set_axis_off()
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filename)
            plt.close(fig)

        cam_depth = outputs["depth"].squeeze(-1)
        cam_rgb = outputs["rgb"]
        if max_depth is None:
            depth_mask = cam_depth > 0
        else:
            depth_mask = (cam_depth > 0) & (cam_depth < max_depth)
        u_coords = torch.arange(self.width, device=self.device)
        v_coords = torch.arange(self.height, device=self.device)
        U_grid, V_grid = torch.meshgrid(u_coords, v_coords, indexing="xy")
        cam_pts_x = (U_grid - self.K[0, 2]) * cam_depth / self.K[0, 0]
        cam_pts_y = (V_grid - self.K[1, 2]) * cam_depth / self.K[1, 1]
        cam_pcd_points = torch.stack((cam_pts_x, cam_pts_y, cam_depth), axis=-1)
        return cam_rgb, cam_pcd_points, None, depth_mask, outputs


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 30,
    volumes={str(VOL_DIR): volume},
)
def interactive_old_union_localize(
    frame: int = 0,
    pose_matrix: list[list[float]] | None = None,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    init_yaw_error_deg: float = 2.0,
    init_translation_error_m: float = 0.05,
) -> dict:
    import time
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from ns_utils.nerfstudio_utils import GaussianSplat, load_dataset
    from pose_estimator.utils import POI_Detector, SE3error, execute_PnP_RANSAC

    request_started = time.perf_counter()
    logs = []

    def log(message: str) -> None:
        elapsed = time.perf_counter() - request_started
        line = f"{elapsed:6.2f}s {message}"
        print(line, flush=True)
        logs.append(line)

    log("Started interactive_old_union_localize")
    _link_volume_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    state_key = "old_union"
    if state_key not in _interactive_state:
        log("Loading old_union GaussianSplat checkpoint and dataset")
        gsplat = GaussianSplat(
            config_path=Path("outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"),
            res_factor=None,
            test_mode="test",
            dataset_mode="val",
            device=device,
        )
        dataset = load_dataset(data_path=Path("data/old_union2"), dataset_mode="all")
        _, _, intrinsics = gsplat.get_camera_intrinsics()
        _interactive_state[state_key] = {
            "gsplat": gsplat,
            "dataset": dataset,
            "intrinsics": intrinsics.to(device),
        }
        log("Model, dataset, and camera intrinsics cached in Modal container")
    else:
        log("Reusing cached old_union model and dataset")

    gsplat = _interactive_state[state_key]["gsplat"]
    dataset = _interactive_state[state_key]["dataset"]
    intrinsics = _interactive_state[state_key]["intrinsics"]

    frame = max(0, min(int(frame), len(dataset) - 1))
    if pose_matrix is not None:
        pose_np = np.array(pose_matrix, dtype=np.float32)
        if pose_np.shape != (4, 4):
            raise ValueError("pose_matrix must be a 4x4 camera-to-world matrix")
        pose = torch.tensor(pose_np, device=device)
        log("Using browser-provided 4x4 camera-to-world pose")
    else:
        pose = torch.eye(4, device=device)
        pose[:3] = dataset.cameras.camera_to_worlds[frame].to(device)

        delta = torch.tensor(_camera_delta_matrix(yaw_deg, pitch_deg, roll_deg), device=device)
        pose[:3, :3] = pose[:3, :3] @ delta
        log(f"Using dataset frame {frame} plus yaw/pitch/roll delta")

    log("Rendering synthetic query snapshot from selected pose")
    with torch.no_grad():
        snapshot_outputs = gsplat.render(pose)
    snapshot_rgb = snapshot_outputs["rgb"].detach()
    log("Snapshot render complete")

    init_pose = pose.detach().clone()
    init_delta = torch.tensor(_camera_delta_matrix(init_yaw_error_deg, 0.0, 0.0), device=device)
    init_pose[:3, :3] = init_pose[:3, :3] @ init_delta
    init_pose[:3, 3] += torch.tensor([init_translation_error_m, 0.0, 0.0], device=device)

    result_dir = VOL_DIR / "results" / "old_union" / "web_app" / "last_run"
    result_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = result_dir / "snapshot.png"
    render_path = result_dir / "render_and_depth.png"
    matches_path = result_dir / "matches.png"
    plt.imsave(snapshot_path, snapshot_rgb.cpu().numpy())
    log(f"Wrote snapshot debug image to {snapshot_path}")

    started = time.perf_counter()
    try:
        log("Running SIFT feature matching and PnP RANSAC")
        est_pose = execute_PnP_RANSAC(
            gsplat,
            init_pose.float(),
            camera_intrinsics_K=intrinsics,
            rgb_input=snapshot_rgb,
            feature_detector=POI_Detector.SIFT,
            save_image=True,
            pnp_figure_filename=str(render_path),
            print_stats=True,
            visualize_PnP_matches=True,
            pnp_matches_figure_filename=str(matches_path),
        )
        success = True
        error = None
        rot_error, trans_error = SE3error(pose.detach().cpu().numpy(), est_pose)
        estimated_pose = est_pose.tolist()
        log(f"PnP succeeded: rotation_error={float(rot_error):.4f}deg translation_error={float(trans_error):.4f}m")
    except RuntimeError as exc:
        success = False
        error = str(exc)
        rot_error = None
        trans_error = None
        estimated_pose = None
        log(f"PnP failed: {error}")

    seconds = time.perf_counter() - started
    log("Encoding debug images for browser response")
    log("Committing Modal volume")
    volume.commit()
    log("Returning localization response")
    response = {
        "success": success,
        "error": error,
        "frame": frame,
        "seconds": seconds,
        "rotation_error_deg": None if rot_error is None else float(rot_error),
        "translation_error_m": None if trans_error is None else float(trans_error),
        "selected_pose": pose.detach().cpu().numpy().tolist(),
        "initial_pose": init_pose.detach().cpu().numpy().tolist(),
        "estimated_pose": estimated_pose,
        "snapshot_png": _encode_file_base64(snapshot_path),
        "render_png": _encode_file_base64(render_path) if render_path.exists() else None,
        "matches_png": _encode_file_base64(matches_path) if matches_path.exists() else None,
        "logs": logs,
    }
    return response


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 30,
    volumes={str(VOL_DIR): volume},
)
def interactive_stanford_raw_localize(
    pose_matrix: list[list[float]],
    init_yaw_error_deg: float = 2.0,
    init_translation_error_m: float = 0.5,
) -> dict:
    import time
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from pose_estimator.utils import POI_Detector, SE3error, execute_PnP_RANSAC

    request_started = time.perf_counter()
    logs = []

    def log(message: str) -> None:
        elapsed = time.perf_counter() - request_started
        line = f"{elapsed:6.2f}s {message}"
        print(line, flush=True)
        logs.append(line)

    log("Started interactive_stanford_raw_localize")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    state_key = "stanford_raw"
    if state_key not in _interactive_state:
        log("Loading raw Stanford Gaussian-splat PLY")
        _interactive_state[state_key] = {
            "gsplat": RawGaussianSplat(
                APP_DIR / "external_assets" / "stanford_university.ply",
                device=device,
                width=720,
                height=960,
                fx=665.6,
                fy=665.6,
                max_gaussians=350_000,
            )
        }
        log("Raw Stanford splat cached in Modal container")
    else:
        log("Reusing cached Stanford raw splat")

    gsplat = _interactive_state[state_key]["gsplat"]
    _, _, intrinsics = gsplat.get_camera_intrinsics()

    pose_np = np.array(pose_matrix, dtype=np.float32)
    if pose_np.shape != (4, 4):
        raise ValueError("pose_matrix must be a 4x4 camera-to-world matrix")
    pose = torch.tensor(pose_np, device=device)
    log("Using browser-provided 4x4 camera-to-world pose")

    log("Rendering synthetic query snapshot from raw Stanford PLY")
    with torch.no_grad():
        snapshot_outputs = gsplat.render(pose)
    snapshot_rgb = snapshot_outputs["rgb"].detach()
    log("Snapshot render complete")

    init_pose = pose.detach().clone()
    init_delta = torch.tensor(_camera_delta_matrix(init_yaw_error_deg, 0.0, 0.0), device=device)
    init_pose[:3, :3] = init_pose[:3, :3] @ init_delta
    init_pose[:3, 3] += torch.tensor([init_translation_error_m, 0.0, 0.0], device=device)

    result_dir = VOL_DIR / "results" / "stanford" / "web_app" / "last_run"
    result_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = result_dir / "snapshot.png"
    render_path = result_dir / "render_and_depth.png"
    matches_path = result_dir / "matches.png"
    plt.imsave(snapshot_path, snapshot_rgb.cpu().numpy())

    started = time.perf_counter()
    try:
        log("Running SIFT feature matching and PnP RANSAC")
        est_pose = execute_PnP_RANSAC(
            gsplat,
            init_pose.float(),
            camera_intrinsics_K=intrinsics,
            rgb_input=snapshot_rgb,
            feature_detector=POI_Detector.SIFT,
            save_image=True,
            pnp_figure_filename=str(render_path),
            print_stats=True,
            visualize_PnP_matches=True,
            pnp_matches_figure_filename=str(matches_path),
        )
        success = True
        error = None
        rot_error, trans_error = SE3error(pose.detach().cpu().numpy(), est_pose)
        estimated_pose = est_pose.tolist()
        log(f"PnP succeeded: rotation_error={float(rot_error):.4f}deg translation_error={float(trans_error):.4f}m")
    except RuntimeError as exc:
        success = False
        error = str(exc)
        rot_error = None
        trans_error = None
        estimated_pose = None
        log(f"PnP failed: {error}")

    seconds = time.perf_counter() - started
    log("Encoding debug images for browser response")
    volume.commit()
    log("Returning Stanford raw localization response")
    return {
        "success": success,
        "error": error,
        "frame": 0,
        "seconds": seconds,
        "rotation_error_deg": None if rot_error is None else float(rot_error),
        "translation_error_m": None if trans_error is None else float(trans_error),
        "selected_pose": pose.detach().cpu().numpy().tolist(),
        "initial_pose": init_pose.detach().cpu().numpy().tolist(),
        "estimated_pose": estimated_pose,
        "snapshot_png": _encode_file_base64(snapshot_path),
        "render_png": _encode_file_base64(render_path) if render_path.exists() else None,
        "matches_png": _encode_file_base64(matches_path) if matches_path.exists() else None,
        "logs": logs,
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 30,
    scaledown_window=60 * 60,
    volumes={str(VOL_DIR): volume},
)
def interactive_stanford_real_frame_localize(
    image_base64: str,
    pose_matrix: list[list[float]],
    intrinsics: dict | None = None,
    init_yaw_error_deg: float = 0.0,
    init_translation_error_m: float = 0.0,
    debug_images: bool = True,
) -> dict:
    import time
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from pose_estimator.utils import POI_Detector, execute_PnP_RANSAC

    request_started = time.perf_counter()
    logs = []

    def log(message: str) -> None:
        elapsed = time.perf_counter() - request_started
        line = f"{elapsed:6.2f}s {message}"
        print(line, flush=True)
        logs.append(line)

    log("Started interactive_stanford_real_frame_localize")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    rgb_np = _decode_base64_rgb_image(image_base64)
    height, width = rgb_np.shape[:2]
    log(f"Decoded real RGB frame: {width}x{height}")

    intrinsics = intrinsics or {}
    source_width = float(intrinsics.get("width", width) or width)
    source_height = float(intrinsics.get("height", height) or height)
    scale_x = width / source_width
    scale_y = height / source_height
    fx = float(intrinsics.get("fx", max(width, height) * 0.9)) * scale_x
    fy = float(intrinsics.get("fy", max(width, height) * 0.9)) * scale_y
    cx = float(intrinsics.get("cx", source_width / 2.0)) * scale_x
    cy = float(intrinsics.get("cy", source_height / 2.0)) * scale_y
    log(f"Using intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    state_key = "stanford_raw"
    if state_key not in _interactive_state:
        log("Loading raw Stanford Gaussian-splat PLY")
        _interactive_state[state_key] = {
            "gsplat": RawGaussianSplat(
                APP_DIR / "external_assets" / "stanford_university.ply",
                device=device,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                max_gaussians=350_000,
            )
        }
        log("Raw Stanford splat cached in Modal container")
    else:
        log("Reusing cached Stanford raw splat")

    gsplat = _interactive_state[state_key]["gsplat"]
    gsplat.configure_camera(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    _, _, camera_intrinsics = gsplat.get_camera_intrinsics()

    pose_np = np.array(pose_matrix, dtype=np.float32)
    if pose_np.shape != (4, 4):
        raise ValueError("pose_matrix must be a 4x4 camera-to-world matrix")
    pose_guess = torch.tensor(pose_np, device=device)
    log("Using browser-provided 4x4 camera-to-world pose as the rough prior")

    init_pose = pose_guess.detach().clone()
    if init_yaw_error_deg or init_translation_error_m:
        init_delta = torch.tensor(_camera_delta_matrix(init_yaw_error_deg, 0.0, 0.0), device=device)
        init_pose[:3, :3] = init_pose[:3, :3] @ init_delta
        init_pose[:3, 3] += torch.tensor([init_translation_error_m, 0.0, 0.0], device=device)
        log("Applied optional synthetic offset to the rough prior")

    rgb_input = torch.tensor(rgb_np, dtype=torch.float32, device=device) / 255.0

    result_dir = VOL_DIR / "results" / "stanford" / "web_app" / "last_real_frame"
    result_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = result_dir / "realsense_frame.png"
    render_path = result_dir / "render_and_depth.png"
    matches_path = result_dir / "matches.png"
    for debug_path in (snapshot_path, render_path, matches_path):
        debug_path.unlink(missing_ok=True)
    if debug_images:
        plt.imsave(snapshot_path, rgb_np)
        log(f"Wrote real-frame debug image to {snapshot_path}")

    started = time.perf_counter()
    try:
        log("Rendering splat from rough prior and running SIFT + PnP")
        est_pose = execute_PnP_RANSAC(
            gsplat,
            init_pose.float(),
            camera_intrinsics_K=camera_intrinsics,
            rgb_input=rgb_input,
            feature_detector=POI_Detector.SIFT,
            save_image=False,
            pnp_figure_filename=str(render_path),
            print_stats=True,
            visualize_PnP_matches=False,
            pnp_matches_figure_filename=str(matches_path),
        )
        success = True
        error = None
        estimated_pose = est_pose.tolist()
        log("PnP succeeded for real RGB frame")
        log("Rendering estimated pose for response debug image")
        with torch.no_grad():
            gsplat.generate_RGBD_point_cloud(
                torch.tensor(est_pose, dtype=torch.float32, device=device),
                save_image=debug_images,
                filename=str(render_path),
                compute_semantics=False,
                return_pcd=False,
            )
    except Exception as exc:
        success = False
        error = str(exc)
        estimated_pose = None
        log(f"PnP failed: {error}")

    seconds = time.perf_counter() - started
    if debug_images:
        log("Committing Modal volume with debug images")
        volume.commit()
    log("Returning Stanford real-frame localization response")
    return {
        "success": success,
        "error": error,
        "frame": 0,
        "seconds": seconds,
        "rotation_error_deg": None,
        "translation_error_m": None,
        "selected_pose": pose_guess.detach().cpu().numpy().tolist(),
        "initial_pose": init_pose.detach().cpu().numpy().tolist(),
        "estimated_pose": estimated_pose,
        "snapshot_png": _encode_file_base64(snapshot_path) if debug_images and snapshot_path.exists() else None,
        "render_png": _encode_file_base64(render_path) if debug_images and render_path.exists() else None,
        "matches_png": _encode_file_base64(matches_path) if debug_images and matches_path.exists() else None,
        "logs": logs,
    }


@app.local_entrypoint()
def main(
    scene: str = "old_union",
    run_name: str = "test_runs",
    max_frames: int = 3,
    stride: int = 1,
    detector: str = "sift",
    camera_source: str = "external",
    reset_init_each_frame: bool = False,
    rotation_noise_deg: float = 0.0,
    translation_noise_m: float = 0.0,
    download_data: bool = False,
    full_data: bool = False,
    gdrive_url: str = GDRIVE_DATA_URL,
    background: bool = False,
) -> None:
    kwargs = {
        "scene": scene,
        "run_name": run_name,
        "max_frames": max_frames,
        "stride": stride,
        "detector": detector,
        "camera_source": camera_source,
        "reset_init_each_frame": reset_init_each_frame,
        "rotation_noise_deg": rotation_noise_deg,
        "translation_noise_m": translation_noise_m,
        "download_data": download_data,
        "full_data": full_data,
        "gdrive_url": gdrive_url,
    }
    if background:
        call = run_splatloc_eval.spawn(**kwargs)
        print(f"Spawned Modal call: {call.object_id}")
        print(f"Metrics will be written to /vol/results/{scene}/test_runs/metrics.json")
        return

    metrics = run_splatloc_eval.remote(**kwargs)
    print(json.dumps(metrics, indent=2))
