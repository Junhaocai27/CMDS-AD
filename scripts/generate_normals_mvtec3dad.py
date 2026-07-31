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

# ================= 配置区域 =================

# 1. 路径配置 (请确保与服务器一致)
ORIG_DATASET_ROOT = os.getenv("CMDS_AD_MVTEC_ROOT", str(RELEASE_ROOT / "data/derived/mvtec_3d"))
GEN_RGB_ROOT = os.getenv("CMDS_AD_MVTEC_RGB_ROOT", str(RELEASE_ROOT / "data/derived/output_generation_full"))
TRAIN_OUTPUT_ROOT = os.getenv("CMDS_AD_MVTEC_NORMAL_ROOT", str(RELEASE_ROOT / "data/derived/normal_output_train_new_full"))

# 2. 脚本路径
SCRIPT_REAL_PATH = str(RELEASE_ROOT / "processing/pc2sn_o3d_gpu_mvtec3dad.py")
SCRIPT_EST_PATH = str(RELEASE_ROOT / "third_party/marigold/run_normals.py")

# 3. 运行参数
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
task_status = {} # 用于存储实时状态

def run_cmd(cmd, gpu_id, description):
    """在指定 GPU 上执行命令"""
    try:
        # 环境变量隔离：子进程只能看到被分配的那张卡
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 使用 DEVNULL 屏蔽子进程输出，避免控制台混乱
        subprocess.run(cmd, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        # 如果出错，可以将 e.stderr 打印到日志文件
        return False

def process_class(gpu_id, class_name):
    """单个类别的处理逻辑"""
    global task_status

    # 更新状态
    task_status[class_name] = {"gpu": str(gpu_id), "status": "Initializing...", "step": "Prep"}

    # 1. 路径定义
    input_rgb_dir = os.path.join(GEN_RGB_ROOT, class_name)
    orig_xyz_source_dir = os.path.join(ORIG_DATASET_ROOT, class_name, "train", "good", "xyz")

    # 按照 MVTec 结构输出: class/train/good
    out_real_dir = os.path.join(TRAIN_OUTPUT_ROOT, "real_normals", class_name, "train", "good")
    out_est_dir = os.path.join(TRAIN_OUTPUT_ROOT, "estimated_normals", class_name, "train", "good")

    if not os.path.exists(input_rgb_dir):
        task_status[class_name]["status"] = "[dim]Skipped (No Data)[/dim]"
        return

    os.makedirs(out_real_dir, exist_ok=True)
    os.makedirs(out_est_dir, exist_ok=True)

    # =========================================================
    # Task 1: Generate Real Normals (筛选非 Seed 图片)
    # =========================================================
    task_status[class_name]["status"] = "[yellow]Running Real Normals[/yellow]"
    task_status[class_name]["step"] = "Real (XYZ->Normal)"

    # 创建临时文件夹
    temp_xyz_dir = os.path.join(GEN_RGB_ROOT, f"{class_name}_temp_xyz_{gpu_id}")
    if os.path.exists(temp_xyz_dir):
        shutil.rmtree(temp_xyz_dir)
    os.makedirs(temp_xyz_dir)

    try:
        # 扫描并复制对应的 TIFF
        all_imgs = sorted(glob.glob(os.path.join(input_rgb_dir, "*")))
        found_real_count = 0

        for img_path in all_imgs:
            filename = os.path.basename(img_path)
            name_stem, ext = os.path.splitext(filename)

            # 只处理原图 (不含 seed)
            if "seed" in name_stem or ext.lower() not in ['.png', '.jpg', '.jpeg']:
                continue

            # 找 TIFF
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
            # 可能是正常的，比如只生成了图片但没放原图？
            pass

    finally:
        if os.path.exists(temp_xyz_dir):
            shutil.rmtree(temp_xyz_dir)

    # =========================================================
    # Task 2: Generate Estimated Normals (全部图片)
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
    """显卡工作线程"""
    for cat in categories:
        process_class(gpu_id, cat)

def generate_table():
    """生成状态表格"""
    table = Table(title="Parallel Training Normal Generation", box=box.ROUNDED)
    table.add_column("GPU", style="magenta", justify="center", width=5)
    table.add_column("Category", style="cyan", width=15)
    table.add_column("Current Task", style="dim", width=20)
    table.add_column("Status", justify="center", width=20)

    # 按 GPU 顺序显示
    for gpu_id in sorted(GPU_ASSIGNMENT.keys()):
        cats = GPU_ASSIGNMENT[gpu_id]
        for cat in cats:
            info = task_status.get(cat, {"status": "Waiting", "step": "-"})
            table.add_row(str(gpu_id), cat, info["step"], info["status"])

    return table

def main():
    # 0. 检查脚本
    if not os.path.exists(SCRIPT_REAL_PATH) or not os.path.exists(SCRIPT_EST_PATH):
        console.print("[bold red]Scripts not found![/bold red]")
        sys.exit(1)

    # 1. 初始化状态
    for gpu_id, cats in GPU_ASSIGNMENT.items():
        for cat in cats:
            task_status[cat] = {"gpu": str(gpu_id), "status": "Waiting", "step": "-"}

    # 2. 启动 4 个线程
    threads = []
    console.print(f"[bold green]Starting parallel generation on 4 GPUs...[/bold green]")

    for gpu_id, cats in GPU_ASSIGNMENT.items():
        t = threading.Thread(target=worker, args=(gpu_id, cats))
        t.start()
        threads.append(t)

    # 3. 监控面板
    with Live(generate_table(), refresh_per_second=4) as live:
        while any(t.is_alive() for t in threads):
            live.update(generate_table())
            time.sleep(0.5)
        live.update(generate_table())

    console.print("\n[bold green]All tasks completed![/bold green]")

if __name__ == "__main__":
    main()
