"""Create strict per-shot LoRA training folders from the AD few-shot selection."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance

from fewshot_lora_common import (
    SHOTS,
    default_work_root,
    get_spec,
    lora_augmented_dir,
    lora_originals_dir,
    manifest_path,
    read_manifest,
    require_shots,
    resolve_classes,
    select_source_images,
    sha256sum,
    source_rgb_dir,
    write_json,
)


AUGMENTATIONS = (
    "original",
    "brightness_110",
    "contrast_090",
    "rotate_m3",
    "rotate_p3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec3dad", "eyecandies"], required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--work_root", type=Path, default=default_work_root())
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing strict data only under --work_root.")
    return parser.parse_args()


def save_png(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")


def augmentation_image(image: Image.Image, name: str) -> Image.Image:
    if name == "brightness_110":
        return ImageEnhance.Brightness(image).enhance(1.10)
    if name == "contrast_090":
        return ImageEnhance.Contrast(image).enhance(0.90)
    if name in {"rotate_m3", "rotate_p3"}:
        angle = -3 if name == "rotate_m3" else 3
        fill = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=fill)
    if name == "original":
        return image
    raise ValueError(f"Unknown augmentation: {name}")


def existing_manifest_matches(work_root: Path, dataset: str, shots: int, class_name: str,
                              selected: list[Path]) -> bool:
    try:
        manifest = read_manifest(work_root, dataset, shots, class_name)
    except FileNotFoundError:
        return False
    expected = [(path.stem, sha256sum(path)) for path in selected]
    actual = [(item["stem"], item["sha256"]) for item in manifest.get("selected_images", [])]
    originals = lora_originals_dir(work_root, dataset, shots, class_name)
    augmented = lora_augmented_dir(work_root, dataset, shots, class_name)
    expected_originals = {f"{stem}.png" for stem, _ in expected}
    expected_augmented = {f"{stem}__{name}.png" for stem, _ in expected for name in AUGMENTATIONS}
    actual_originals = {path.name for path in originals.iterdir() if path.is_file()} if originals.is_dir() else set()
    actual_augmented = {path.name for path in augmented.iterdir() if path.is_file()} if augmented.is_dir() else set()
    return actual == expected and actual_originals == expected_originals and actual_augmented == expected_augmented


def prepare_one(work_root: Path, dataset: str, shots: int, class_name: str, overwrite: bool) -> None:
    source_dir = source_rgb_dir(dataset, class_name)
    selected = select_source_images(source_dir, shots)
    if existing_manifest_matches(work_root, dataset, shots, class_name, selected) and not overwrite:
        print(f"[skip] {dataset}/{class_name}/{shots}shot already matches its manifest")
        return

    originals_dir = lora_originals_dir(work_root, dataset, shots, class_name)
    augmented_dir = lora_augmented_dir(work_root, dataset, shots, class_name)
    if overwrite:
        for directory in (originals_dir, augmented_dir):
            if directory.exists():
                shutil.rmtree(directory)
    originals_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for source in selected:
        original_destination = originals_dir / f"{source.stem}.png"
        if source.suffix.lower() == ".png":
            shutil.copy2(source, original_destination)
        else:
            with Image.open(source) as image:
                save_png(image.convert("RGB"), original_destination)

        with Image.open(source) as source_image:
            image = source_image.convert("RGB")
            for augmentation in AUGMENTATIONS:
                destination = augmented_dir / f"{source.stem}__{augmentation}.png"
                if augmentation == "original" and source.suffix.lower() == ".png":
                    shutil.copy2(source, destination)
                else:
                    save_png(augmentation_image(image, augmentation), destination)

        entries.append({
            "stem": source.stem,
            "source_file": source.name,
            "sha256": sha256sum(source),
        })

    payload = {
        "dataset": dataset,
        "class_name": class_name,
        "shots": shots,
        "selection_policy": "lexicographically sorted normal training RGB stems; first N stems",
        "source_rgb_dir": str(source_dir),
        "selected_images": entries,
        "offline_augmentations": list(AUGMENTATIONS),
        "online_augmentations": ["ColorJitter(brightness=0.1, contrast=0.1)", "RandomHorizontalFlip"],
    }
    write_json(manifest_path(work_root, dataset, shots, class_name), payload)
    print(
        f"[ready] {dataset}/{class_name}/{shots}shot: {len(entries)} original images, "
        f"{len(entries) * len(AUGMENTATIONS)} LoRA training images"
    )


def main() -> int:
    args = parse_args()
    get_spec(args.dataset)
    classes = resolve_classes(args.dataset, args.classes)
    shots = require_shots(args.shots)
    work_root = args.work_root.expanduser().resolve()
    for shot in shots:
        for class_name in classes:
            prepare_one(work_root, args.dataset, shot, class_name, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
