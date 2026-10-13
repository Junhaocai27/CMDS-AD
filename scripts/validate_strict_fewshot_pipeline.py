"""Validate that a strict few-shot pipeline stage contains no extra training inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from fewshot_lora_common import (
    SHOTS,
    default_work_root,
    estimated_normal_dir,
    expected_generated_filenames,
    expected_normal_filenames,
    generated_rgb_dir,
    lora_augmented_dir,
    lora_originals_dir,
    lora_weight_path,
    read_manifest,
    require_shots,
    resolve_classes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec3dad", "eyecandies"], required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--work_root", type=Path, default=default_work_root())
    parser.add_argument("--stage", choices=["prepared", "lora", "rgb", "normals"], default="normals")
    return parser.parse_args()


def files(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()


def validate_one(work_root: Path, dataset: str, shots: int, class_name: str, stage: str) -> list[str]:
    manifest = read_manifest(work_root, dataset, shots, class_name)
    selected = [item["stem"] for item in manifest["selected_images"]]
    errors = []
    if len(selected) != shots or selected != sorted(selected):
        errors.append("manifest does not contain exactly the sorted requested number of stems")

    originals = lora_originals_dir(work_root, dataset, shots, class_name)
    expected_originals = {f"{stem}.png" for stem in selected}
    actual_originals = files(originals)
    if actual_originals != expected_originals:
        errors.append(f"original selection mismatch: expected {sorted(expected_originals)}, got {sorted(actual_originals)}")
    if stage == "prepared":
        return errors

    augmented = lora_augmented_dir(work_root, dataset, shots, class_name)
    if len(files(augmented)) != shots * len(manifest["offline_augmentations"]):
        errors.append("LoRA augmentation count does not match the manifest")
    if not lora_weight_path(work_root, dataset, shots, class_name).is_file():
        errors.append("final_lora.safetensors is missing")
    if stage == "lora":
        return errors

    rgb_dir = generated_rgb_dir(work_root, dataset, shots, class_name)
    expected_rgb = set(expected_generated_filenames(manifest))
    actual_rgb = files(rgb_dir)
    if actual_rgb != expected_rgb:
        errors.append(f"i2i RGB mismatch: expected {len(expected_rgb)} files, got {len(actual_rgb)}")
    if stage == "rgb":
        return errors

    normal_dir = estimated_normal_dir(work_root, dataset, shots, class_name) / "normals_vis"
    expected_normals = set(expected_normal_filenames(manifest))
    actual_normals = files(normal_dir)
    if actual_normals != expected_normals:
        errors.append(f"estimated-normal mismatch: expected {len(expected_normals)} files, got {len(actual_normals)}")
    return errors


def main() -> int:
    args = parse_args()
    work_root = args.work_root.expanduser().resolve()
    classes = resolve_classes(args.dataset, args.classes)
    shots = require_shots(args.shots)
    failures = []
    for shot in shots:
        for class_name in classes:
            errors = validate_one(work_root, args.dataset, shot, class_name, args.stage)
            label = f"{args.dataset}/{class_name}/{shot}shot"
            if errors:
                failures.append((label, errors))
                print(f"[failed] {label}: {'; '.join(errors)}")
            else:
                print(f"[ok] {label}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
