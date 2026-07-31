"""Estimate RGB surface-normal maps from organized XYZ TIFF files on CPU."""

import argparse
from pathlib import Path

import numpy as np


def convert_tiff_to_normal_image(tiff_path, output_path, radius=0.01, max_nn=30):
    """Convert one organized XYZ TIFF into an RGB normal-map PNG."""
    import cv2
    import open3d as o3d
    import tifffile

    xyz_data = tifffile.imread(tiff_path)
    if xyz_data.ndim != 3 or xyz_data.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 XYZ data, got {xyz_data.shape} from {tiff_path}")

    xyz_data = np.nan_to_num(xyz_data).astype(np.float64, copy=False)
    height, width, _ = xyz_data.shape

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_data.reshape(-1, 3))
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    point_cloud.orient_normals_towards_camera_location(camera_location=np.zeros(3))

    normals = np.asarray(point_cloud.normals).reshape(height, width, 3)
    background = np.all(np.isclose(xyz_data, 0.0), axis=2)
    normals[background] = 0.0

    normal_rgb = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
    normal_rgb = (normal_rgb * 255.0).astype(np.uint8)
    normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), normal_bgr):
        raise RuntimeError(f"Failed to write {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--max_nn", type=int, default=30)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*.tiff"))
    if not files:
        raise SystemExit(f"No .tiff files found under {args.input_dir}")

    for input_path in files:
        relative_path = input_path.relative_to(args.input_dir).with_suffix(".png")
        output_path = args.output_dir / relative_path
        print(f"[{files.index(input_path) + 1}/{len(files)}] {input_path} -> {output_path}")
        convert_tiff_to_normal_image(input_path, output_path, args.radius, args.max_nn)


if __name__ == "__main__":
    main()
