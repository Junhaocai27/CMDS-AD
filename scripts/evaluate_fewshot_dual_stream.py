"""Evaluate strict few-shot checkpoints after all dual-stream tasks are complete."""

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
    get_spec,
    release_root,
    require_shots,
    resolve_classes,
)
from train_fewshot_dual_stream import expected_checkpoint_files


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
    parser.add_argument("--work_root", type=Path, default=default_work_root())
    parser.add_argument("--gpus", type=parse_gpu_ids, default=[0])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--force", action="store_true", help="Rerun when the shot log already exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_shot(args: argparse.Namespace, classes: list[str], shots: int) -> None:
    work_root = args.work_root.expanduser().resolve()
    missing = []
    for class_name in classes:
        for direction in ("2dto3d", "3dto2d"):
            missing.extend(path for path in expected_checkpoint_files(
                work_root, args.dataset, class_name, shots, direction
            ) if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Strict final checkpoint missing: {missing[0]}")


def build_command(args: argparse.Namespace, classes: list[str], shots: int) -> list[str]:
    root = release_root()
    spec = get_spec(args.dataset)
    script = root / ("test_anomaly_fusion_mvtec3dad.py" if args.dataset == "mvtec3dad"
                     else "test_anomaly_fusion_eyecandies.py")
    checkpoint_root = args.work_root / "checkpoints" / args.dataset / f"{shots}shot"
    prefix_3d2d = "checkpoints_dual_3dto2d" if args.dataset == "mvtec3dad" else "checkpoints_eyecandies_dual_3dto2d"
    prefix_2d3d = "checkpoints_dual_2dto3d" if args.dataset == "mvtec3dad" else "checkpoints_eyecandies_dual_2dto3d"
    command = [
        sys.executable, str(script),
        "--dataset_root", str(root / spec.source_subdir),
        "--real_normal_root", str(root / spec.test_real_normal_subdir),
        "--est_normal_root", str(root / spec.test_est_normal_subdir),
        "--mask_root", str(root / spec.mask_subdir),
        "--ckpt_root_3d2d", str(checkpoint_root / prefix_3d2d),
        "--ckpt_root_2d3d", str(checkpoint_root / prefix_2d3d),
        "--ckpt_suffix", f"_{shots}shot",
        "--classes", *classes,
        "--batch_size", str(args.batch_size),
        "--device", "cuda",
        "--wandb_mode", args.wandb_mode,
    ]
    if args.max_samples:
        command.extend(["--max_samples", str(args.max_samples)])
    return command


def run_worker(gpu_id: int, tasks: queue.Queue, args: argparse.Namespace, classes: list[str],
               failures: list[tuple[str, int]], lock: threading.Lock) -> None:
    root = release_root()
    while True:
        try:
            shots = tasks.get_nowait()
        except queue.Empty:
            return
        label = f"{args.dataset}/{shots}shot"
        try:
            validate_shot(args, classes, shots)
            log_path = args.work_root / "logs" / "inference" / args.dataset / f"{shots}shot.log"
            if log_path.is_file() and not args.force:
                print(f"[skip] {label}: {log_path}")
                continue
            command = build_command(args, classes, shots)
            print(f"[start] GPU {gpu_id}: {label}")
            print("$ CUDA_VISIBLE_DEVICES=" + str(gpu_id), " ".join(command))
            if args.dry_run:
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("$ CUDA_VISIBLE_DEVICES=" + str(gpu_id) + " " + " ".join(command) + "\n\n")
                completed = subprocess.run(command, cwd=root, stdout=log_file, stderr=subprocess.STDOUT,
                                           env=environment, check=False)
            if completed.returncode != 0:
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
    tasks: queue.Queue = queue.Queue()
    for shot in shots:
        validate_shot(args, classes, shot)
        tasks.put(shot)
    failures: list[tuple[str, int]] = []
    lock = threading.Lock()
    workers = [threading.Thread(target=run_worker, args=(gpu_id, tasks, args, classes, failures, lock))
               for gpu_id in args.gpus]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        print("Failed inference tasks:")
        for label, return_code in failures:
            print(f"  {label} (exit {return_code})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
