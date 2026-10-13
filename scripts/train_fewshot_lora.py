"""Train one isolated LoRA per dataset, category, and few-shot setting."""

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
    lora_augmented_dir,
    lora_weight_path,
    read_manifest,
    require_shots,
    resolve_classes,
)


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item) for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise argparse.ArgumentTypeError("At least one GPU id is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("GPU ids must be unique")
    return gpu_ids


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec3dad", "eyecandies"], required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--work_root", type=Path, default=default_work_root())
    parser.add_argument("--sd_model", type=Path, default=root / "weights/stable-diffusion-2-1-base")
    parser.add_argument("--gpus", type=parse_gpu_ids, default=[0])
    parser.add_argument("--max_train_steps_ti", type=int, default=1000)
    parser.add_argument("--max_train_steps_tuning", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Retrain even when final_lora.safetensors exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, class_name: str, shots: int, gpu_id: int) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    work_root = args.work_root.expanduser().resolve()
    instance_dir = lora_augmented_dir(work_root, args.dataset, shots, class_name)
    output_dir = lora_weight_path(work_root, args.dataset, shots, class_name).parent
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Strict LoRA training data missing: {instance_dir}")
    read_manifest(work_root, args.dataset, shots, class_name)
    return [
        str(Path(sys.executable).with_name("lora_pti")),
        f"--pretrained_model_name_or_path={args.sd_model}",
        f"--instance_data_dir={instance_dir}",
        f"--output_dir={output_dir}",
        f"--device=cuda:{gpu_id}",
        f"--placeholder_tokens=<{class_name}>",
        "--train_text_encoder=True",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--gradient_checkpointing=True",
        "--scale_lr=True",
        "--learning_rate_unet=1e-4",
        "--learning_rate_text=1e-5",
        "--learning_rate_ti=5e-4",
        "--color_jitter=True",
        "--lr_scheduler=linear",
        "--lr_warmup_steps=0",
        "--lr_scheduler_lora=linear",
        "--lr_warmup_steps_lora=100",
        "--use_template=object",
        "--save_steps=100",
        f"--max_train_steps_ti={args.max_train_steps_ti}",
        f"--max_train_steps_tuning={args.max_train_steps_tuning}",
        "--perform_inversion=True",
        "--clip_ti_decay=True",
        "--weight_decay_ti=0.000",
        "--weight_decay_lora=0.001",
        "--continue_inversion=True",
        "--continue_inversion_lr=1e-4",
        "--lora_rank=8",
        "--lora_clip_target_modules={'CLIPSdpaAttention'}",
        "--seed=42",
        "--out_name=final_lora",
    ]


def run_worker(gpu_id: int, tasks: queue.Queue, args: argparse.Namespace,
               failures: list[tuple[str, int]], lock: threading.Lock) -> None:
    root = Path(__file__).resolve().parents[1]
    work_root = args.work_root.expanduser().resolve()
    while True:
        try:
            class_name, shots = tasks.get_nowait()
        except queue.Empty:
            return
        label = f"{args.dataset}/{class_name}/{shots}shot"
        target = lora_weight_path(work_root, args.dataset, shots, class_name)
        try:
            if target.is_file() and not args.force:
                print(f"[skip] {label}: {target}")
                continue
            command = build_command(args, class_name, shots, gpu_id)
            print(f"[start] GPU {gpu_id}: {label}")
            print("$", " ".join(command))
            if args.dry_run:
                continue
            log_path = work_root / "logs" / "lora" / args.dataset / f"{shots}shot" / f"{class_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("$ " + " ".join(command) + "\n\n")
                completed = subprocess.run(command, cwd=root, stdout=log_file, stderr=subprocess.STDOUT,
                                           env=environment, check=False)
            if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
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
    classes = resolve_classes(args.dataset, args.classes)
    shots = require_shots(args.shots)
    if not args.sd_model.is_dir():
        raise FileNotFoundError(f"Stable Diffusion model not found: {args.sd_model}")

    tasks: queue.Queue = queue.Queue()
    for shot in shots:
        for class_name in classes:
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
        print("Failed LoRA tasks:")
        for label, return_code in failures:
            print(f"  {label} (exit {return_code})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
