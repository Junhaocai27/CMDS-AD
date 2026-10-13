"""Generate i2i RGB only for the exact images used by each strict LoRA."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import shutil
import traceback
from pathlib import Path

from fewshot_lora_common import (
    FIXED_I2I_SEEDS,
    SHOTS,
    default_work_root,
    expected_generated_filenames,
    generated_rgb_dir,
    lora_originals_dir,
    lora_weight_path,
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
    parser.add_argument("--sd_model", type=Path, default=root / "weights/stable-diffusion-2-1-base")
    parser.add_argument("--gpus", type=parse_gpu_ids, default=[0])
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Regenerate existing RGB files.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expected_paths(output_dir: Path, manifest: dict) -> list[Path]:
    return [output_dir / name for name in expected_generated_filenames(manifest)]


def validate_tasks(args: argparse.Namespace, classes: list[str], shots: list[int]) -> list[tuple[str, int]]:
    work_root = args.work_root.expanduser().resolve()
    tasks = []
    for shot in shots:
        for class_name in classes:
            manifest = read_manifest(work_root, args.dataset, shot, class_name)
            originals = lora_originals_dir(work_root, args.dataset, shot, class_name)
            lora = lora_weight_path(work_root, args.dataset, shot, class_name)
            expected_originals = [originals / f"{item['stem']}.png" for item in manifest["selected_images"]]
            missing = [path for path in expected_originals if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Selected original image missing: {missing[0]}")
            if not lora.is_file():
                raise FileNotFoundError(f"Strict LoRA checkpoint missing: {lora}")
            tasks.append((class_name, shot))
    return tasks


def worker(gpu_id: int, jobs: list[tuple[str, int]], args_dict: dict, event_queue: mp.Queue) -> None:
    try:
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
        from PIL import Image

        root = Path(__file__).resolve().parents[1]
        import sys
        sys.path.insert(0, str(root / "third_party"))
        from lora_diffusion.lora import patch_pipe, tune_lora_scale

        args = argparse.Namespace(**args_dict)
        work_root = Path(args.work_root)
        for class_name, shots in jobs:
            label = f"{args.dataset}/{class_name}/{shots}shot"
            manifest = read_manifest(work_root, args.dataset, shots, class_name)
            originals = lora_originals_dir(work_root, args.dataset, shots, class_name)
            lora = lora_weight_path(work_root, args.dataset, shots, class_name)
            output_dir = generated_rgb_dir(work_root, args.dataset, shots, class_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            expected = expected_paths(output_dir, manifest)
            if not args.force and all(path.is_file() for path in expected):
                event_queue.put(("skip", gpu_id, label))
                continue

            event_queue.put(("start", gpu_id, label))
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                args.sd_model, safety_checker=None, torch_dtype=torch.float16
            ).to(f"cuda:{gpu_id}")
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            patch_pipe(pipe, str(lora), patch_text=True, patch_ti=True, patch_unet=True)
            tune_lora_scale(pipe.unet, 1.0)
            tune_lora_scale(pipe.text_encoder, 1.0)
            prompt = f"a low high light and real photo of <{class_name}>"

            for item in manifest["selected_images"]:
                stem = item["stem"]
                source = originals / f"{stem}.png"
                target_original = output_dir / source.name
                if args.force or not target_original.is_file():
                    shutil.copy2(source, target_original)
                with Image.open(source) as image_file:
                    image = image_file.convert("RGB").resize((512, 512))
                    for seed in FIXED_I2I_SEEDS:
                        target = output_dir / f"{stem}_seed{seed}.png"
                        if target.is_file() and not args.force:
                            continue
                        generator = torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed)
                        result = pipe(
                            prompt=prompt,
                            image=image,
                            strength=args.strength,
                            num_inference_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            generator=generator,
                        ).images[0]
                        result.save(target)

            del pipe
            torch.cuda.empty_cache()
            missing = [path for path in expected if not path.is_file()]
            if missing:
                raise RuntimeError(f"i2i output missing after generation: {missing[0]}")
            event_queue.put(("done", gpu_id, label))
    except Exception as error:
        event_queue.put(("error", gpu_id, str(error), traceback.format_exc()))


def main() -> int:
    args = parse_args()
    if not args.sd_model.is_dir():
        raise FileNotFoundError(f"Stable Diffusion model not found: {args.sd_model}")
    args.work_root = args.work_root.expanduser().resolve()
    classes = resolve_classes(args.dataset, args.classes)
    shots = require_shots(args.shots)
    jobs = validate_tasks(args, classes, shots)
    if args.dry_run:
        for class_name, shot in jobs:
            print(f"[ready] {args.dataset}/{class_name}/{shot}shot")
        return 0

    assignments = [[] for _ in args.gpus]
    for index, job in enumerate(jobs):
        assignments[index % len(assignments)].append(job)
    context = mp.get_context("spawn")
    event_queue = context.Queue()
    args_dict = vars(args).copy()
    args_dict["work_root"] = str(args.work_root)
    args_dict["sd_model"] = str(args.sd_model)
    processes = [
        context.Process(target=worker, args=(gpu_id, assignment, args_dict, event_queue))
        for gpu_id, assignment in zip(args.gpus, assignments) if assignment
    ]
    for process in processes:
        process.start()

    failures = []
    while any(process.is_alive() for process in processes):
        try:
            event = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if event[0] == "error":
            _, gpu_id, error, trace = event
            failures.append((gpu_id, error))
            print(f"[failed] GPU {gpu_id}: {error}\n{trace}")
        else:
            event_type, gpu_id, label = event
            print(f"[{event_type}] GPU {gpu_id}: {label}")
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append(("process", f"worker exit code {process.exitcode}"))
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break
        if event[0] == "error":
            _, gpu_id, error, trace = event
            failures.append((gpu_id, error))
            print(f"[failed] GPU {gpu_id}: {error}\n{trace}")
        else:
            event_type, gpu_id, label = event
            print(f"[{event_type}] GPU {gpu_id}: {label}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
