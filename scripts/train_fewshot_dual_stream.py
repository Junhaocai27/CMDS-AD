"""Train CMDS-AD mappings with strict per-shot LoRA-generated RGB and normals."""

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
    expected_normal_filenames,
    generated_rgb_dir,
    get_spec,
    read_manifest,
    release_root,
    require_shots,
    resolve_classes,
)


STEPS_BY_SHOT = {1: 3000, 2: 1500, 4: 750}


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item) for item in value.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("Provide one or more unique GPU ids")
    return gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec3dad", "eyecandies"], required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--directions", choices=["2dto3d", "3dto2d", "both"], default="both")
    parser.add_argument("--work_root", type=Path, default=default_work_root())
    parser.add_argument("--gpus", type=parse_gpu_ids, default=[0])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="0 uses the validated 3000/1500/750-shot schedule.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--force", action="store_true", help="Retrain even when final checkpoint files exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint_root(work_root: Path, dataset: str, shots: int) -> Path:
    return work_root / "checkpoints" / dataset / f"{shots}shot"


def expected_checkpoint_files(work_root: Path, dataset: str, class_name: str, shots: int,
                              direction: str) -> list[Path]:
    root = checkpoint_root(work_root, dataset, shots)
    if dataset == "mvtec3dad":
        prefix = "checkpoints_dual_2dto3d" if direction == "2dto3d" else "checkpoints_dual_3dto2d"
    else:
        prefix = "checkpoints_eyecandies_dual_2dto3d" if direction == "2dto3d" else "checkpoints_eyecandies_dual_3dto2d"
    directory = root / f"{prefix}_{class_name}_{shots}shot"
    suffix = "2d23d" if direction == "2dto3d" else "3d22d"
    return [directory / f"model_real_{suffix}_final.pth", directory / f"model_est_{suffix}_final.pth"]


def validate_job(args: argparse.Namespace, class_name: str, shots: int) -> None:
    work_root = args.work_root.expanduser().resolve()
    manifest = read_manifest(work_root, args.dataset, shots, class_name)
    rgb_dir = generated_rgb_dir(work_root, args.dataset, shots, class_name)
    if not all((rgb_dir / f"{item['stem']}.png").is_file() for item in manifest["selected_images"]):
        raise FileNotFoundError(f"Strict RGB originals are incomplete: {rgb_dir}")
    normal_dir = estimated_normal_dir(work_root, args.dataset, shots, class_name) / "normals_vis"
    missing = [normal_dir / name for name in expected_normal_filenames(manifest) if not (normal_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Strict estimated normal is missing: {missing[0]}")


def build_command(args: argparse.Namespace, class_name: str, shots: int, direction: str) -> list[str]:
    root = release_root()
    work_root = args.work_root.expanduser().resolve()
    spec = get_spec(args.dataset)
    trainer_name = spec.trainer_2dto3d if direction == "2dto3d" else spec.trainer_3dto2d
    trainer = root / trainer_name
    generated_root = work_root / "generated_rgb" / args.dataset / f"{shots}shot"
    estimated_root = work_root / "estimated_normals" / args.dataset / f"{shots}shot"
    max_steps = args.max_steps or STEPS_BY_SHOT[shots]
    return [
        sys.executable, str(trainer),
        "--class_name", class_name,
        "--batch_size", str(args.batch_size),
        "--max_steps", str(max_steps),
        "--shots", str(shots),
        "--lr", str(args.lr),
        "--rgb_root", str(generated_root),
        "--real_normal_root", str(root / spec.real_normal_subdir),
        "--est_normal_root", str(estimated_root),
        "--mask_root", str(root / spec.mask_subdir),
        "--checkpoint_root", str(checkpoint_root(work_root, args.dataset, shots)),
        "--device", "cuda",
        "--num_workers", str(args.num_workers),
        "--wandb_mode", args.wandb_mode,
    ]


def run_worker(gpu_id: int, tasks: queue.Queue, args: argparse.Namespace,
               failures: list[tuple[str, int]], lock: threading.Lock) -> None:
    root = release_root()
    work_root = args.work_root.expanduser().resolve()
    while True:
        try:
            class_name, shots, direction = tasks.get_nowait()
        except queue.Empty:
            return
        label = f"{args.dataset}/{class_name}/{shots}shot/{direction}"
        try:
            validate_job(args, class_name, shots)
            expected = expected_checkpoint_files(work_root, args.dataset, class_name, shots, direction)
            if not args.force and all(path.is_file() and path.stat().st_size > 0 for path in expected):
                print(f"[skip] {label}: final checkpoints already complete")
                continue
            command = build_command(args, class_name, shots, direction)
            print(f"[start] GPU {gpu_id}: {label}")
            print("$ CUDA_VISIBLE_DEVICES=" + str(gpu_id), " ".join(command))
            if args.dry_run:
                continue
            log_path = work_root / "logs" / "dual_stream" / args.dataset / f"{shots}shot" / f"{class_name}_{direction}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("$ CUDA_VISIBLE_DEVICES=" + str(gpu_id) + " " + " ".join(command) + "\n\n")
                completed = subprocess.run(command, cwd=root, stdout=log_file, stderr=subprocess.STDOUT,
                                           env=environment, check=False)
            missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
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
    classes = resolve_classes(args.dataset, args.classes)
    shots = require_shots(args.shots)
    directions = ["2dto3d", "3dto2d"] if args.directions == "both" else [args.directions]
    tasks: queue.Queue = queue.Queue()
    for shot in shots:
        for class_name in classes:
            validate_job(args, class_name, shot)
            for direction in directions:
                tasks.put((class_name, shot, direction))
    failures: list[tuple[str, int]] = []
    lock = threading.Lock()
    workers = [threading.Thread(target=run_worker, args=(gpu_id, tasks, args, failures, lock))
               for gpu_id in args.gpus]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        print("Failed dual-stream tasks:")
        for label, return_code in failures:
            print(f"  {label} (exit {return_code})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
