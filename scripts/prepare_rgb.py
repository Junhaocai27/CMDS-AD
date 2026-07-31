"""Copy original training RGB images into the flat layout used by trainers."""

import argparse
import shutil
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
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.dataset == "mvtec":
        dataset_root = args.dataset_root or root / "data/derived/mvtec_3d"
        output_root = args.output_root or root / "data/derived/output_generation_full"
        classes = args.classes or MVTEC_CLASSES
    else:
        dataset_root = args.dataset_root or root / "data/derived/eyecandies_mvtec_format"
        output_root = args.output_root or root / "data/derived/output_generation_eyecandies"
        classes = args.classes or EYECANDIES_CLASSES

    copied = 0
    for class_name in classes:
        source_dir = dataset_root / class_name / "train/good/rgb"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {source_dir}")
        target_dir = output_root / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path in sorted(source_dir.iterdir()):
            if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            shutil.copy2(source_path, target_dir / source_path.name)
            copied += 1

    print(f"Copied {copied} original RGB images to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
