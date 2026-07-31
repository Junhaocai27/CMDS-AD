"""Run CPU-safe structural checks for the source-only release."""

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "processing/preprocess_mvtec.py",
    "processing/preprocess_eyecandies.py",
    "processing/pc2sn_o3d_cpu.py",
    "train_2d_to_3d.py",
    "train_3d_to_2d.py",
    "train_2d_to_3d_eyecandies.py",
    "train_3d_to_2d_eyecandies.py",
    "test_anomaly_fusion.py",
    "test_anomaly_fusion_eyecandies.py",
    "scripts/train.py",
    "scripts/prepare_rgb.py",
    "scripts/validate_dataset.py",
]
FORBIDDEN = [
    "f1d6a732d2453eceb40329334f6e83e503e88428",
    "WANDB_API_KEY =",
]


def main():
    failures = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            failures.append(f"syntax error: {path.relative_to(ROOT)}: {exc}")

        # The validator necessarily contains the forbidden patterns itself.
        if path.resolve() != Path(__file__).resolve():
            text = path.read_text(encoding="utf-8", errors="replace")
            for forbidden in FORBIDDEN:
                if forbidden in text:
                    failures.append(f"forbidden text in {path.relative_to(ROOT)}: {forbidden}")

    oversized = [p for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size > 50 * 1024 * 1024]
    if oversized:
        failures.extend(f"unexpected large file: {p.relative_to(ROOT)}" for p in oversized)

    if failures:
        print("CMDS-AD release validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Validated {len(python_files)} Python files; no forbidden secrets or >50 MB files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
