"""Run the optional Marigold normal estimator for all selected classes."""

import argparse
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--dataset_root", type=Path, default=None,
                        help="Converted dataset root used for test split traversal")
    parser.add_argument("--rgb_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Marigold checkpoint or model id")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoise_steps", type=int, default=10)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.dataset == "mvtec":
        classes = args.classes or MVTEC_CLASSES
        dataset_root = args.dataset_root or root / "data/derived/mvtec_3d"
        rgb_root = args.rgb_root or root / "data/derived/output_generation_full"
        output_root = args.output_root or root / "data/derived/normal_output_train_new_full/estimated_normals"
    else:
        classes = args.classes or EYECANDIES_CLASSES
        dataset_root = args.dataset_root or root / "data/derived/eyecandies_mvtec_format"
        rgb_root = args.rgb_root or root / "data/derived/output_generation_eyecandies"
        output_root = args.output_root or root / "data/derived/normal_output_train_eyecandies/estimated_normals"

    estimator = root / "third_party/marigold/run_normals.py"
    for class_name in classes:
        jobs = []
        for split in args.splits:
            if split == "train":
                jobs.append((rgb_root / class_name, output_root / class_name / "train/good"))
                continue

            split_root = dataset_root / class_name / split
            if not split_root.is_dir():
                continue
            for defect_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
                input_dir = defect_root / "rgb"
                if not input_dir.is_dir():
                    input_dir = defect_root
                jobs.append((input_dir, output_root / class_name / split / defect_root.name))

        for input_dir, output_dir in jobs:
            command = [
                sys.executable,
                str(estimator),
                "--checkpoint", args.checkpoint,
                "--input_rgb_dir", str(input_dir),
                "--output_dir", str(output_dir),
                "--batch_size", str(args.batch_size),
                "--seed", str(args.seed),
                "--denoise_steps", str(args.denoise_steps),
                "--device", args.device,
            ]
            print("$", " ".join(command))
            if not args.dry_run:
                completed = subprocess.run(command, cwd=root)
                if completed.returncode != 0:
                    return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
