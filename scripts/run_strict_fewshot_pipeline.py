"""Run the strict few-shot LoRA, i2i, normal-estimation, and mapping stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fewshot_lora_common import SHOTS, default_work_root


STAGES = ("prepare", "lora", "rgb", "normals", "mappings", "evaluation")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec3dad", "eyecandies"], required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--work_root", type=Path, default=default_work_root())
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical CUDA device ids.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--sd_model", type=Path, default=root / "weights/stable-diffusion-2-1-base")
    parser.add_argument("--marigold_checkpoint", type=Path, default=root / "weights/marigold")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--force", action="store_true",
                        help="Recompute completed LoRA, i2i, normal, and mapping outputs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.work_root = args.work_root.expanduser().resolve()
    common = ["--dataset", args.dataset, "--shots", *(str(shot) for shot in args.shots),
              "--work_root", str(args.work_root)]
    if args.classes:
        common.extend(["--classes", *args.classes])
    commands: dict[str, list[str]] = {
        "prepare": [sys.executable, str(root / "scripts/prepare_fewshot_lora_data.py"), *common],
        "lora": [sys.executable, str(root / "scripts/train_fewshot_lora.py"), *common,
                 "--sd_model", str(args.sd_model), "--gpus", args.gpus],
        "rgb": [sys.executable, str(root / "scripts/generate_fewshot_rgb.py"), *common,
                "--sd_model", str(args.sd_model), "--gpus", args.gpus],
        "normals": [sys.executable, str(root / "scripts/generate_fewshot_estimated_normals.py"), *common,
                    "--checkpoint", str(args.marigold_checkpoint), "--gpus", args.gpus,
                    "--batch_size", str(args.batch_size)],
        "mappings": [sys.executable, str(root / "scripts/train_fewshot_dual_stream.py"), *common,
                     "--gpus", args.gpus, "--batch_size", str(args.batch_size),
                     "--max_steps", str(args.max_steps), "--wandb_mode", args.wandb_mode],
        "evaluation": [sys.executable, str(root / "scripts/evaluate_fewshot_dual_stream.py"), *common,
                       "--gpus", args.gpus, "--batch_size", str(args.batch_size),
                       "--wandb_mode", args.wandb_mode],
    }
    selected_commands = [(stage, commands[stage]) for stage in STAGES if stage in args.stages]
    if args.force:
        for stage, command in selected_commands:
            if stage != "prepare":
                command.append("--force")
    if args.dry_run:
        for stage, command in selected_commands:
            print(f"# {stage}")
            print("$", " ".join(command), flush=True)
        return 0
    for stage, command in selected_commands:
        print("$", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            print(f"Stage '{stage}' failed with exit code {completed.returncode}.")
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
