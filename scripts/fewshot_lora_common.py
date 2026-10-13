"""Shared paths, selection, and validation for the strict few-shot LoRA workflow."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FIXED_I2I_SEEDS = (42, 1024, 2026, 8888, 12345)
SHOTS = (1, 2, 4)

MVTEC_CLASSES = (
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
)
EYECANDIES_CLASSES = (
    "CandyCane", "ChocolateCookie", "ChocolatePraline", "Confetto",
    "GummyBear", "HazelnutTruffle", "LicoriceSandwich", "Lollipop",
    "Marshmallow", "PeppermintCandy",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    classes: tuple[str, ...]
    source_subdir: str
    real_normal_subdir: str
    mask_subdir: str
    test_real_normal_subdir: str
    test_est_normal_subdir: str
    trainer_dataset: str
    trainer_2dto3d: str
    trainer_3dto2d: str


SPECS = {
    "mvtec3dad": DatasetSpec(
        name="mvtec3dad",
        classes=MVTEC_CLASSES,
        source_subdir="data/derived/mvtec_3d",
        real_normal_subdir="data/derived/normal_output_train_new_full/real_normals",
        mask_subdir="data/derived/mvtec_3d_masks_generated",
        test_real_normal_subdir="data/derived/normal_output_mv_format_infer/real_normals",
        test_est_normal_subdir="data/derived/normal_output_mv_format_infer/estimated_normals",
        trainer_dataset="mvtec",
        trainer_2dto3d="train_2d_to_3d_mvtec3dad.py",
        trainer_3dto2d="train_3d_to_2d_mvtec3dad.py",
    ),
    "eyecandies": DatasetSpec(
        name="eyecandies",
        classes=EYECANDIES_CLASSES,
        source_subdir="data/derived/eyecandies_mvtec_format",
        real_normal_subdir="data/derived/normal_output_train_eyecandies/real_normals",
        mask_subdir="data/derived/eyecandies_masks_generated",
        test_real_normal_subdir="data/derived/normal_output_eyecandies_infer/real_normals",
        test_est_normal_subdir="data/derived/normal_output_eyecandies_infer/estimated_normals",
        trainer_dataset="eyecandies",
        trainer_2dto3d="train_2d_to_3d_eyecandies.py",
        trainer_3dto2d="train_3d_to_2d_eyecandies.py",
    ),
}


def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_work_root() -> Path:
    configured = os.getenv("CMDS_AD_STRICT_FEWSHOT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return release_root() / "results" / "strict_fewshot_lora"


def get_spec(dataset: str) -> DatasetSpec:
    try:
        return SPECS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset}") from error


def resolve_classes(dataset: str, requested: Iterable[str] | None) -> list[str]:
    spec = get_spec(dataset)
    classes = list(requested) if requested else list(spec.classes)
    unknown = sorted(set(classes) - set(spec.classes))
    if unknown:
        raise ValueError(f"Unsupported {dataset} classes: {', '.join(unknown)}")
    return classes


def require_shots(shots: Iterable[int]) -> list[int]:
    values = list(shots)
    invalid = sorted(set(values) - set(SHOTS))
    if invalid:
        raise ValueError(f"Only {SHOTS} are supported, received {invalid}")
    return values


def source_rgb_dir(dataset: str, class_name: str) -> Path:
    spec = get_spec(dataset)
    return release_root() / spec.source_subdir / class_name / "train" / "good" / "rgb"


def manifest_path(work_root: Path, dataset: str, shots: int, class_name: str) -> Path:
    return work_root / "manifests" / dataset / f"{shots}shot" / f"{class_name}.json"


def lora_originals_dir(work_root: Path, dataset: str, shots: int, class_name: str) -> Path:
    return work_root / "lora_training_images" / dataset / f"{shots}shot" / class_name / "originals"


def lora_augmented_dir(work_root: Path, dataset: str, shots: int, class_name: str) -> Path:
    return work_root / "lora_training_images" / dataset / f"{shots}shot" / class_name / "augmented"


def lora_weight_path(work_root: Path, dataset: str, shots: int, class_name: str) -> Path:
    return work_root / "lora_weights" / dataset / f"{shots}shot" / class_name / "final_lora.safetensors"


def generated_rgb_dir(work_root: Path, dataset: str, shots: int, class_name: str) -> Path:
    return work_root / "generated_rgb" / dataset / f"{shots}shot" / class_name


def estimated_normal_dir(work_root: Path, dataset: str, shots: int, class_name: str) -> Path:
    return work_root / "estimated_normals" / dataset / f"{shots}shot" / class_name / "train" / "good"


def checkpoint_dir(work_root: Path, dataset: str, shots: int, class_name: str, direction: str) -> Path:
    return work_root / "checkpoints" / dataset / class_name / f"{shots}shot" / direction


def strict_result_dir(work_root: Path, dataset: str, shots: int) -> Path:
    return work_root / "inference" / dataset / f"{shots}shot"


def list_source_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Training RGB directory not found: {directory}")
    by_stem: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "_seed" in path.stem:
            continue
        if path.stem in by_stem:
            raise ValueError(f"Duplicate training RGB stem '{path.stem}' in {directory}")
        by_stem[path.stem] = path
    return [by_stem[stem] for stem in sorted(by_stem)]


def select_source_images(directory: Path, shots: int) -> list[Path]:
    images = list_source_images(directory)
    if len(images) < shots:
        raise ValueError(f"{directory} contains {len(images)} images, fewer than {shots}-shot")
    return images[:shots]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(work_root: Path, dataset: str, shots: int, class_name: str) -> dict:
    path = manifest_path(work_root, dataset, shots, class_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Few-shot manifest not found: {path}. Run prepare_fewshot_lora_data.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def expected_generated_filenames(manifest: dict) -> list[str]:
    names: list[str] = []
    for item in manifest["selected_images"]:
        stem = item["stem"]
        names.append(f"{stem}.png")
        names.extend(f"{stem}_seed{seed}.png" for seed in FIXED_I2I_SEEDS)
    return names


def expected_normal_filenames(manifest: dict) -> list[str]:
    return [f"{Path(name).stem}_normals.png" for name in expected_generated_filenames(manifest)]
