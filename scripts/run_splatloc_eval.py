from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from ns_utils.nerfstudio_utils import GaussianSplat, load_dataset
from pose_estimator.utils import POI_Detector, SE3error, execute_PnP_RANSAC, vec_to_rot_matrix


SCENES = {
    "old_union": {
        "config": "outputs/old_union2/splatfacto/2024-09-02_151414/config.yml",
        "dataset": "data/old_union2",
    },
    "stonehenge": {
        "config": "outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml",
        "dataset": "data/stonehenge",
    },
    "statues": {
        "config": "outputs/statues/splatfacto/2024-09-11_095852/config.yml",
        "dataset": "data/statues",
    },
    "flight": {
        "config": "outputs/flight/splatfacto/2024-09-12_172434/config.yml",
        "dataset": "data/flight",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Splat-Loc eval.")
    parser.add_argument("--scene", choices=sorted(SCENES), default="old_union")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--detector", choices=["sift", "orb"], default="sift")
    parser.add_argument("--rotation-noise-deg", type=float, default=5.0)
    parser.add_argument("--translation-noise-m", type=float, default=0.1)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument(
        "--camera-source",
        choices=["external", "pipeline", "trained_transform"],
        default="external",
    )
    parser.add_argument("--reset-init-each-frame", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _check_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _load_trained_transform_records(dataset_path: Path, config_path: Path) -> tuple[list[dict], torch.Tensor]:
    transforms_path = dataset_path / "transforms.full.json"
    if not transforms_path.exists():
        transforms_path = dataset_path / "transforms.json"
    dataparser_transform_path = config_path.parent / "dataparser_transforms.json"
    _check_path(transforms_path, "Full transforms")
    _check_path(dataparser_transform_path, "Checkpoint dataparser transforms")

    transforms = json.loads(transforms_path.read_text())
    dataparser_transform = json.loads(dataparser_transform_path.read_text())
    applied_transform = np.eye(4, dtype=np.float32)
    if "applied_transform" in transforms:
        applied_transform[:3, :] = np.array(transforms["applied_transform"], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :] = np.array(dataparser_transform["transform"], dtype=np.float32)
    scale = float(dataparser_transform["scale"])

    records = []
    for frame in transforms["frames"]:
        image_path = dataset_path / frame["file_path"]
        if not image_path.exists():
            continue
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        pose = applied_transform @ pose
        pose = transform @ pose
        pose[:3, 3] *= scale
        records.append({"image_path": image_path, "pose": pose})

    if not records:
        raise RuntimeError(f"No downloaded frame images found in {dataset_path}")

    intrinsics = torch.tensor(
        [
            [transforms["fl_x"], 0.0, transforms["cx"]],
            [0.0, transforms["fl_y"], transforms["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return records, intrinsics


def _load_image_float32(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene_cfg = SCENES[args.scene]
    config_path = args.config or Path(scene_cfg["config"])
    dataset_path = args.dataset or Path(scene_cfg["dataset"])
    output_path = args.output or Path(f"results/{args.scene}/test_runs/metrics.json")

    _check_path(config_path, "Nerfstudio config")
    _check_path(dataset_path, "Eval dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = POI_Detector.SIFT if args.detector == "sift" else POI_Detector.ORB

    started = time.perf_counter()
    gsplat = GaussianSplat(
        config_path=config_path,
        res_factor=None,
        test_mode="test",
        dataset_mode="val",
        device=device,
    )
    records = None
    if args.camera_source == "pipeline":
        dataset = gsplat.dataset
        _, _, camera_intrinsics = gsplat.get_camera_intrinsics()
    else:
        if args.camera_source == "trained_transform":
            records, camera_intrinsics = _load_trained_transform_records(dataset_path, config_path)
            dataset = records
        else:
            dataset = load_dataset(data_path=dataset_path, dataset_mode="all")
            _, _, camera_intrinsics = gsplat.get_camera_intrinsics()
    camera_intrinsics = camera_intrinsics.to(device)

    frame_ids = list(range(0, len(dataset), args.stride))[: args.max_frames]
    if not frame_ids:
        raise RuntimeError(f"No frames found in {dataset_path}")

    pose_errors: list[dict] = []
    estimated_poses: dict[str, list[list[float]]] = {}
    local_est_pose = None
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    for eval_index, frame_id in enumerate(frame_ids):
        print(f"evaluating frame={frame_id}", flush=True)
        if records is not None:
            cam_rgb = _load_image_float32(records[frame_id]["image_path"]).to(device)
        else:
            cam_rgb = dataset.get_image_float32(frame_id).to(device)
        if args.debug_dir is not None:
            plt.imsave(
                args.debug_dir / f"frame_{frame_id:05d}_input.png",
                cam_rgb.detach().cpu().numpy(),
            )

        gt_pose = torch.eye(4)
        if records is not None:
            gt_pose = torch.from_numpy(records[frame_id]["pose"]).to(device)
        else:
            gt_pose[:3] = dataset.cameras.camera_to_worlds[frame_id].to(device)
        gt_pose_np = gt_pose.cpu().numpy()

        if eval_index == 0 or local_est_pose is None or args.reset_init_each_frame:
            init_guess = gt_pose_np.copy()
            rand_rot_axis = torch.nn.functional.normalize(torch.rand(3, device=device), dim=-1)
            rand_rot = vec_to_rot_matrix(np.deg2rad(args.rotation_noise_deg) * rand_rot_axis)
            init_guess[:3, :3] = rand_rot.cpu().numpy() @ init_guess[:3, :3]
            init_guess[:3, 3] += (
                args.translation_noise_m
                * torch.nn.functional.normalize(torch.rand(3, device=device), dim=-1).cpu().numpy()
            )
        else:
            init_guess = local_est_pose

        frame_started = time.perf_counter()
        try:
            local_est_pose = execute_PnP_RANSAC(
                gsplat,
                torch.tensor(init_guess, device=device).float(),
                camera_intrinsics_K=camera_intrinsics,
                rgb_input=cam_rgb,
                feature_detector=detector,
                save_image=args.debug_dir is not None,
                pnp_figure_filename=(
                    str(args.debug_dir / f"frame_{frame_id:05d}_render.png")
                    if args.debug_dir is not None
                    else "/"
                ),
                print_stats=True,
                visualize_PnP_matches=args.debug_dir is not None,
                pnp_matches_figure_filename=(
                    str(args.debug_dir / f"frame_{frame_id:05d}_matches.png")
                    if args.debug_dir is not None
                    else "/"
                ),
            )
        except RuntimeError as exc:
            elapsed = time.perf_counter() - frame_started
            pose_errors.append(
                {
                    "frame": frame_id,
                    "success": False,
                    "error": str(exc),
                    "seconds": elapsed,
                }
            )
            print(f"frame={frame_id} failed error={exc} seconds={elapsed:.3f}", flush=True)
            continue
        elapsed = time.perf_counter() - frame_started
        rot_error, trans_error = SE3error(gt_pose_np, local_est_pose)

        pose_errors.append(
            {
                "frame": frame_id,
                "success": True,
                "rotation_deg": float(rot_error),
                "translation_m": float(trans_error),
                "seconds": elapsed,
            }
        )
        estimated_poses[f"frame_{frame_id:05d}"] = local_est_pose.tolist()
        print(
            f"frame={frame_id} rotation_deg={rot_error:.4f} "
            f"translation_m={trans_error:.4f} seconds={elapsed:.3f}",
            flush=True,
        )

    successful_frames = [item for item in pose_errors if item["success"]]
    rotations = np.array([item["rotation_deg"] for item in successful_frames], dtype=np.float64)
    translations = np.array([item["translation_m"] for item in successful_frames], dtype=np.float64)
    timings = np.array([item["seconds"] for item in pose_errors], dtype=np.float64)

    metrics = {
        "scene": args.scene,
        "config": str(config_path),
        "dataset": str(dataset_path),
        "camera_source": args.camera_source,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_frames": len(frame_ids),
        "num_successful_frames": len(successful_frames),
        "num_failed_frames": len(pose_errors) - len(successful_frames),
        "rotation_deg_mean": float(rotations.mean()) if len(rotations) else None,
        "rotation_deg_median": float(np.median(rotations)) if len(rotations) else None,
        "translation_m_mean": float(translations.mean()) if len(translations) else None,
        "translation_m_median": float(np.median(translations)) if len(translations) else None,
        "seconds_per_frame_mean": float(timings.mean()),
        "total_seconds": float(time.perf_counter() - started),
        "frames": pose_errors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    (output_path.parent / "est_pose.json").write_text(json.dumps(estimated_poses, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
