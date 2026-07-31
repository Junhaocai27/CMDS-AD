"""Launch the reproducible CMDS-AD dual-direction training matrix."""

import argparse
import subprocess
import sys
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
STEPS_BY_SHOT = {1: 3000, 2: 1500, 4: 750}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec", "eyecandies"], required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, choices=[1, 2, 4], default=[1, 2, 4])
    parser.add_argument("--directions", choices=["2dto3d", "3dto2d", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per stream; use 1 on a 24 GB GPU")
    parser.add_argument("--max_steps", type=int, default=0, help="0 selects the standard step count for each shot")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--rgb_root", type=Path, default=None)
    parser.add_argument("--real_normal_root", type=Path, default=None)
    parser.add_argument("--est_normal_root", type=Path, default=None)
    parser.add_argument("--mask_root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def dataset_defaults(dataset, root):
    if dataset == "mvtec":
        return {
            "classes": MVTEC_CLASSES,
            "rgb_root": root / "data/derived/output_generation_full",
            "real_normal_root": root / "data/derived/normal_output_train_new_full/real_normals",
            "est_normal_root": root / "data/derived/normal_output_train_new_full/estimated_normals",
            "mask_root": root / "data/derived/mvtec_3d_masks_generated",
            "trainers": {
                "2dto3d": root / "train_2d_to_3d.py",
                "3dto2d": root / "train_3d_to_2d.py",
            },
        }
    return {
        "classes": EYECANDIES_CLASSES,
        "rgb_root": root / "data/derived/output_generation_eyecandies",
        "real_normal_root": root / "data/derived/normal_output_train_eyecandies/real_normals",
        "est_normal_root": root / "data/derived/normal_output_train_eyecandies/estimated_normals",
        "mask_root": root / "data/derived/eyecandies_masks_generated",
        "trainers": {
            "2dto3d": root / "train_2d_to_3d_eyecandies.py",
            "3dto2d": root / "train_3d_to_2d_eyecandies.py",
        },
    }


def build_command(trainer, args, defaults, class_name, shots):
    max_steps = args.max_steps or STEPS_BY_SHOT[shots]
    return [
        sys.executable,
        str(trainer),
        "--class_name", class_name,
        "--batch_size", str(args.batch_size),
        "--max_steps", str(max_steps),
        "--shots", str(shots),
        "--lr", str(args.lr),
        "--rgb_root", str(args.rgb_root or defaults["rgb_root"]),
        "--real_normal_root", str(args.real_normal_root or defaults["real_normal_root"]),
        "--est_normal_root", str(args.est_normal_root or defaults["est_normal_root"]),
        "--mask_root", str(args.mask_root or defaults["mask_root"]),
        "--checkpoint_root", str(args.checkpoint_root),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--wandb_mode", args.wandb_mode,
    ]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    defaults = dataset_defaults(args.dataset, root)
    classes = args.classes or defaults["classes"]
    directions = ["2dto3d", "3dto2d"] if args.directions == "both" else [args.directions]

    for direction in directions:
        for class_name in classes:
            for shots in args.shots:
                command = build_command(defaults["trainers"][direction], args, defaults, class_name, shots)
                print("$", " ".join(command))
                if args.dry_run:
                    continue
                completed = subprocess.run(command, cwd=root)
                if completed.returncode != 0 and args.stop_on_error:
                    return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
