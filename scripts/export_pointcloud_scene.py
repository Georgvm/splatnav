from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def read_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY ended before end_header")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        vertex_count = None
        for line in header_lines:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
        if vertex_count is None:
            raise ValueError("PLY header does not contain an element vertex count")

        properties = []
        for line in header_lines:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "property":
                ply_type, name = parts[1], parts[2]
                if ply_type == "float":
                    dtype = "<f4"
                elif ply_type == "uchar":
                    dtype = "u1"
                else:
                    raise ValueError(f"Unsupported PLY property type: {ply_type}")
                properties.append((name, dtype))
        dtype = np.dtype(properties)
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)

    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    names = set(data.dtype.names or [])
    is_gaussian_splat = {"f_dc_0", "f_dc_1", "f_dc_2", "opacity"}.issubset(names)
    if is_gaussian_splat:
        colors = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]])
        colors = np.clip(colors * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
        opacities = sigmoid(data["opacity"].astype(np.float32))
    elif {"red", "green", "blue"}.issubset(names):
        colors = (
            np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.float32)
            / 255.0
        )
        opacities = np.ones(len(points), dtype=np.float32)
    else:
        colors = np.full((len(points), 3), 0.75, dtype=np.float32)
        opacities = np.ones(len(points), dtype=np.float32)
    return points, colors, opacities.astype(np.float32), is_gaussian_splat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    points, colors, opacities, is_gaussian_splat = read_binary_ply(args.input)
    if len(points) > args.max_points:
        if is_gaussian_splat:
            indices = np.argpartition(opacities, -args.max_points)[-args.max_points:]
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        opacities = opacities[indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "name": args.input.stem,
                "localization_enabled": False,
                "points": points.tolist(),
                "colors": colors.tolist(),
                "opacities": opacities.tolist(),
                "source_type": "gaussian_splat_ply" if is_gaussian_splat else "point_cloud_ply",
                "camera_positions": [],
            }
        )
    )
    print(f"Wrote {len(points)} points to {args.output} ({'gaussian splat' if is_gaussian_splat else 'point cloud'})")


if __name__ == "__main__":
    main()
