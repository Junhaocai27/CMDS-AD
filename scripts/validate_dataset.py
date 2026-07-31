"""Audit dataset RGB files and test-set anomaly annotations."""

import argparse
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec", "eyecandies"], required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["test"])
    return parser.parse_args()


def image_files(directory):
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def find_ground_truth(class_root, defect_root, image_path):
    stem = image_path.stem
    candidates = [
        defect_root / "gt" / image_path.name,
        defect_root / "gt" / f"{stem}.png",
        defect_root / "gt" / f"{stem}_mask.png",
        class_root / "ground_truth" / defect_root.name / image_path.name,
        class_root / "ground_truth" / defect_root.name / f"{stem}.png",
        class_root / "ground_truth" / defect_root.name / f"{stem}_mask.png",
    ]
    return next((path for path in candidates if path.is_file()), None)


def inspect_split(class_root, split):
    split_root = class_root / split
    if not split_root.is_dir():
        return {"missing_split": 1, "images": 0, "anomalies": 0, "gt": 0, "missing_gt": 0}

    result = {"missing_split": 0, "images": 0, "anomalies": 0, "gt": 0, "missing_gt": 0}
    for defect_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
        rgb_root = defect_root / "rgb"
        if not rgb_root.is_dir():
            rgb_root = defect_root
        images = image_files(rgb_root)
        if not images:
            continue

        for image_path in images:
            result["images"] += 1
            if defect_root.name.lower() == "good":
                continue
            result["anomalies"] += 1
            if find_ground_truth(class_root, defect_root, image_path) is None:
                result["missing_gt"] += 1
            else:
                result["gt"] += 1
    return result


def main():
    args = parse_args()
    classes = args.classes or (MVTEC_CLASSES if args.dataset == "mvtec" else EYECANDIES_CLASSES)

    totals = {"missing_split": 0, "images": 0, "anomalies": 0, "gt": 0, "missing_gt": 0}
    failures = []
    for class_name in classes:
        class_root = args.dataset_root / class_name
        if not class_root.is_dir():
            failures.append(f"missing class directory: {class_root}")
            continue
        for split in args.splits:
            result = inspect_split(class_root, split)
            for key, value in result.items():
                totals[key] += value
            if result["missing_split"]:
                failures.append(f"missing split directory: {class_root / split}")
            elif result["images"] == 0:
                failures.append(f"no RGB images found: {class_root / split}")
            if result["missing_gt"]:
                failures.append(
                    f"{class_name}/{split}: {result['missing_gt']} anomalous images have no ground-truth mask"
                )

    print(
        f"Scanned {totals['images']} images; "
        f"{totals['anomalies']} anomalous images; "
        f"{totals['gt']} anomaly masks found."
    )
    if failures:
        print("Dataset annotation validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Dataset annotation validation passed. Good images may omit an explicit mask and are treated as normal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
