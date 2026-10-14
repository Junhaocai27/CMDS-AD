import os
import sys
import shutil
import subprocess
import glob
import threading
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.table import Table
from rich import box

from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]




ORIG_DATASET_ROOT = os.getenv("CMDS_AD_MVTEC_ROOT", str(RELEASE_ROOT / "data/derived/mvtec_3d"))
GEN_RGB_ROOT = os.getenv("CMDS_AD_MVTEC_RGB_ROOT", str(RELEASE_ROOT / "data/derived/output_generation_full"))
TRAIN_OUTPUT_ROOT = os.getenv("CMDS_AD_MVTEC_NORMAL_ROOT", str(RELEASE_ROOT / "data/derived/normal_output_train_new_full"))


SCRIPT_REAL_PATH = str(RELEASE_ROOT / "processing/pc2sn_o3d_gpu_mvtec3dad.py")
SCRIPT_EST_PATH = str(RELEASE_ROOT / "third_party/marigold/run_normals.py")


PYTHON_BIN = "python -u"
MARIGOLD_BATCH_SIZE = 1
SEED = 42

# 4. Categories and GPU IDs can be selected with CMDS_AD_CLASSES and CMDS_AD_GPUS.
ALL_CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire"
]
SELECTED_CATEGORIES = os.getenv("CMDS_AD_CLASSES", " ".join(ALL_CATEGORIES)).split()
GPU_IDS = [int(value) for value in os.getenv("CMDS_AD_GPUS", "0,1").split(",") if value.strip()]
GPU_ASSIGNMENT = {
    gpu_id: SELECTED_CATEGORIES[index::len(GPU_IDS)]
    for index, gpu_id in enumerate(GPU_IDS)
    if SELECTED_CATEGORIES[index::len(GPU_IDS)]
}

# ===========================================

console = Console()
task_status = {}

def run_cmd(cmd, gpu_id, description):
    try:

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


        subprocess.run(cmd, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:

        return False

def process_class(gpu_id, class_name):
    global task_status


    task_status[class_name] = {"gpu": str(gpu_id), "status": "Initializing...", "step": "Prep"}


    input_rgb_dir = os.path.join(GEN_RGB_ROOT, class_name)
    orig_xyz_source_dir = os.path.join(ORIG_DATASET_ROOT, class_name, "train", "good", "xyz")


    out_real_dir = os.path.join(TRAIN_OUTPUT_ROOT, "real_normals", class_name, "train", "good")
    out_est_dir = os.path.join(TRAIN_OUTPUT_ROOT, "estimated_normals", class_name, "train", "good")

    if not os.path.exists(input_rgb_dir):
        task_status[class_name]["status"] = "[dim]Skipped (No Data)[/dim]"
        return

    os.makedirs(out_real_dir, exist_ok=True)
    os.makedirs(out_est_dir, exist_ok=True)

    # =========================================================

    # =========================================================
    task_status[class_name]["status"] = "[yellow]Running Real Normals[/yellow]"
    task_status[class_name]["step"] = "Real (XYZ->Normal)"


    temp_xyz_dir = os.path.join(GEN_RGB_ROOT, f"{class_name}_temp_xyz_{gpu_id}")
    if os.path.exists(temp_xyz_dir):
        shutil.rmtree(temp_xyz_dir)
    os.makedirs(temp_xyz_dir)

    try:

        all_imgs = sorted(glob.glob(os.path.join(input_rgb_dir, "*")))
        found_real_count = 0

        for img_path in all_imgs:
            filename = os.path.basename(img_path)
            name_stem, ext = os.path.splitext(filename)


            if "seed" in name_stem or ext.lower() not in ['.png', '.jpg', '.jpeg']:
                continue


            tiff_name = f"{name_stem}.tiff"
            src_tiff_path = os.path.join(orig_xyz_source_dir, tiff_name)

            if os.path.exists(src_tiff_path):
                shutil.copy(src_tiff_path, os.path.join(temp_xyz_dir, tiff_name))
                found_real_count += 1

        if found_real_count > 0:
            cmd_real = (
                f"{PYTHON_BIN} {SCRIPT_REAL_PATH} "
                f"--input_dir {temp_xyz_dir} "
                f"--output_dir {out_real_dir}"
            )
            if not run_cmd(cmd_real, gpu_id, f"Real-{class_name}"):
                task_status[class_name]["status"] = "[red]Failed Real[/red]"
                return
        else:

            pass

    finally:
        if os.path.exists(temp_xyz_dir):
            shutil.rmtree(temp_xyz_dir)

    # =========================================================

    # =========================================================
    task_status[class_name]["status"] = "[cyan]Running Est Normals[/cyan]"
    task_status[class_name]["step"] = "Est (Marigold)"

    cmd_est = (
        f"{PYTHON_BIN} {SCRIPT_EST_PATH} "
        f"--input_rgb_dir {input_rgb_dir} "
        f"--output_dir {out_est_dir} "
        f"--batch_size {MARIGOLD_BATCH_SIZE} "
        f"--seed {SEED}"
    )

    if not run_cmd(cmd_est, gpu_id, f"Est-{class_name}"):
        task_status[class_name]["status"] = "[red]Failed Est[/red]"
    else:
        task_status[class_name]["status"] = "[green]Finished[/green]"
        task_status[class_name]["step"] = "Done"

def worker(gpu_id, categories):
    for cat in categories:
        process_class(gpu_id, cat)

def generate_table():
    table = Table(title="Parallel Training Normal Generation", box=box.ROUNDED)
    table.add_column("GPU", style="magenta", justify="center", width=5)
    table.add_column("Category", style="cyan", width=15)
    table.add_column("Current Task", style="dim", width=20)
    table.add_column("Status", justify="center", width=20)


    for gpu_id in sorted(GPU_ASSIGNMENT.keys()):
        cats = GPU_ASSIGNMENT[gpu_id]
        for cat in cats:
            info = task_status.get(cat, {"status": "Waiting", "step": "-"})
            table.add_row(str(gpu_id), cat, info["step"], info["status"])

    return table

def main():

    if not os.path.exists(SCRIPT_REAL_PATH) or not os.path.exists(SCRIPT_EST_PATH):
        console.print("[bold red]Scripts not found![/bold red]")
        sys.exit(1)


    for gpu_id, cats in GPU_ASSIGNMENT.items():
        for cat in cats:
            task_status[cat] = {"gpu": str(gpu_id), "status": "Waiting", "step": "-"}


    threads = []
    console.print(f"[bold green]Starting parallel generation on 4 GPUs...[/bold green]")

    for gpu_id, cats in GPU_ASSIGNMENT.items():
        t = threading.Thread(target=worker, args=(gpu_id, cats))
        t.start()
        threads.append(t)


    with Live(generate_table(), refresh_per_second=4) as live:
        while any(t.is_alive() for t in threads):
            live.update(generate_table())
            time.sleep(0.5)
        live.update(generate_table())

    console.print("\n[bold green]All tasks completed![/bold green]")

if __name__ == "__main__":
    main()
