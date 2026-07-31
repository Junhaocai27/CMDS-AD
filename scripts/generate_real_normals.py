"""Prepare real surface-normal images for the training data."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


MVTEC_CLASSES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
]
EYECANDIES_CLASSES = [
    "CandyCane", "ChocolateCookie", "ChocolatePraline", "Confetto",
    "GummyBear", "HazelnutTruffle", "LicoriceSandwich", "Lollipop",
    "Marshmallow", "PeppermintCandy",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec", "eyecandies"], required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train"])
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--max_nn", type=int, default=30)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    return parser.parse_args()


def get_gpu_converter(radius, max_nn):
    """Load the Open3D CUDA normal estimator only when it is needed."""
    import open3d.core as o3c

    if not o3c.cuda.is_available():
        raise RuntimeError("CUDA is required for real-normal generation, but Open3D CUDA is unavailable")

    from processing.pc2sn_o3d_gpu_mvtec3dad import convert_mvtec_tiff_to_normal_o3d_gpu

    device = o3c.Device("CUDA:0")

    def convert(input_path, output_path):
        convert_mvtec_tiff_to_normal_o3d_gpu(
            str(input_path),
            str(output_path),
            radius=radius,
            max_nn=max_nn,
            target_size=(224, 224),
            device=device,
        )

    return convert


def main():
    args = parse_args()
    convert_gpu = get_gpu_converter(args.radius, args.max_nn)
    if args.dataset == "mvtec":
        dataset_root = args.dataset_root or ROOT / "data/derived/mvtec_3d"
        output_root = args.output_root or ROOT / "data/derived/normal_output_train_new_full/real_normals"
        classes = args.classes or MVTEC_CLASSES
        for class_name in classes:
            for split in args.splits:
                split_root = dataset_root / class_name / split
                if not split_root.is_dir():
                    continue
                for defect_root in sorted(p for p in split_root.iterdir() if p.is_dir()):
                    input_dir = defect_root / "xyz"
                    if not input_dir.is_dir():
                        continue
                    output_dir = output_root / class_name / split / defect_root.name
                    for input_path in sorted(input_dir.glob("*.tiff")):
                        output_path = output_dir / f"{input_path.stem}.png"
                        convert_gpu(input_path, output_path)
        return 0

    dataset_root = args.dataset_root or ROOT / "data/derived/eyecandies_mvtec_format"
    output_root = args.output_root or ROOT / "data/derived/normal_output_train_eyecandies/real_normals"
    classes = args.classes or EYECANDIES_CLASSES
    for class_name in classes:
        for split in args.splits:
            source_root = dataset_root / class_name / split
            if not source_root.is_dir():
                continue
            for defect_root in sorted(p for p in source_root.iterdir() if p.is_dir()):
                source_dir = defect_root / "normals"
                target_dir = output_root / class_name / split / defect_root.name
                target_dir.mkdir(parents=True, exist_ok=True)
                if source_dir.is_dir():
                    for source_path in sorted(source_dir.glob("*.png")):
                        shutil.copy2(source_path, target_dir / source_path.name)
                else:
                    # The standard Eyecandies converter writes XYZ only. In that
                    # case generate the same real-normal representation as MVTec.
                    for source_path in sorted((defect_root / "xyz").glob("*.tiff")):
                        convert_gpu(source_path, target_dir / f"{source_path.stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
