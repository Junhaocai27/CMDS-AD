"""Estimate normals for the strict per-shot original and generated RGB images."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from fewshot_lora_common import (
    SHOTS,
    default_work_root,
    estimated_normal_dir,
    expected_generated_filenames,
    expected_normal_filenames,
    generated_rgb_dir,
    read_manifest,
    require_shots,
    resolve_classes,
)


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item) for item in value.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("Provide one or more unique GPU ids")
    return gpu_ids


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec3dad", "eyecandies"], required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--work_root", type=Path, default=default_work_root())
    parser.add_argument("--checkpoint", type=Path, default=root / "weights/marigold")
    parser.add_argument("--gpus", type=parse_gpu_ids, default=[0])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoise_steps", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Recompute existing normal maps.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_job(args: argparse.Namespace, class_name: str, shots: int) -> tuple[Path, Path, dict]:
    work_root = args.work_root.expanduser().resolve()
    manifest = read_manifest(work_root, args.dataset, shots, class_name)
    rgb_dir = generated_rgb_dir(work_root, args.dataset, shots, class_name)
    missing_rgb = [rgb_dir / name for name in expected_generated_filenames(manifest) if not (rgb_dir / name).is_file()]
    if missing_rgb:
        raise FileNotFoundError(f"Strict i2i RGB missing: {missing_rgb[0]}")
    output_dir = estimated_normal_dir(work_root, args.dataset, shots, class_name)
    return rgb_dir, output_dir, manifest


def run_worker(gpu_id: int, tasks: queue.Queue, args: argparse.Namespace,
               failures: list[tuple[str, int]], lock: threading.Lock) -> None:
    root = Path(__file__).resolve().parents[1]
    work_root = args.work_root.expanduser().resolve()
    estimator = root / "third_party" / "marigold" / "run_normals.py"
    while True:
        try:
            class_name, shots = tasks.get_nowait()
        except queue.Empty:
            return
        label = f"{args.dataset}/{class_name}/{shots}shot"
        try:
            rgb_dir, output_dir, manifest = validate_job(args, class_name, shots)
            expected = [output_dir / "normals_vis" / name for name in expected_normal_filenames(manifest)]
            if not args.force and all(path.is_file() for path in expected):
                print(f"[skip] {label}: normal maps already complete")
                continue
            command = [
                sys.executable, str(estimator),
                "--checkpoint", str(args.checkpoint),
                "--input_rgb_dir", str(rgb_dir),
                "--output_dir", str(output_dir),
                "--batch_size", str(args.batch_size),
                "--seed", str(args.seed),
                "--denoise_steps", str(args.denoise_steps),
                "--device", "cuda",
            ]
            print(f"[start] GPU {gpu_id}: {label}")
            print("$ CUDA_VISIBLE_DEVICES=" + str(gpu_id), " ".join(command))
            if args.dry_run:
                continue
            log_path = work_root / "logs" / "estimated_normals" / args.dataset / f"{shots}shot" / f"{class_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("$ CUDA_VISIBLE_DEVICES=" + str(gpu_id) + " " + " ".join(command) + "\n\n")
                completed = subprocess.run(command, cwd=root, stdout=log_file, stderr=subprocess.STDOUT,
                                           env=environment, check=False)
            missing = [path for path in expected if not path.is_file()]
            if completed.returncode != 0 or missing:
                with lock:
                    failures.append((label, completed.returncode))
                print(f"[failed] GPU {gpu_id}: {label}; see {log_path}")
            else:
                print(f"[done] GPU {gpu_id}: {label}")
        except Exception as error:
            with lock:
                failures.append((label, 1))
            print(f"[failed] GPU {gpu_id}: {label}: {error}")
        finally:
            tasks.task_done()


def main() -> int:
    args = parse_args()
    args.work_root = args.work_root.expanduser().resolve()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Marigold checkpoint not found: {args.checkpoint}")
    classes = resolve_classes(args.dataset, args.classes)
    shots = require_shots(args.shots)
    tasks: queue.Queue = queue.Queue()
    for shot in shots:
        for class_name in classes:
            validate_job(args, class_name, shot)
            tasks.put((class_name, shot))
    failures: list[tuple[str, int]] = []
    lock = threading.Lock()
    workers = [threading.Thread(target=run_worker, args=(gpu_id, tasks, args, failures, lock))
               for gpu_id in args.gpus]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        print("Failed normal-estimation tasks:")
        for label, return_code in failures:
            print(f"  {label} (exit {return_code})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
