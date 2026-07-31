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

# ================= 配置区域 (适配 Eyecandies) =================

# 1. 路径配置
# Eyecandies 数据集根目录 (包含现成的 normals)
ORIG_DATASET_ROOT = os.getenv("CMDS_AD_EYECANDIES_ROOT", str(RELEASE_ROOT / "data/derived/eyecandies_mvtec_format"))
# img2img 生成结果的输入目录
GEN_RGB_ROOT = os.getenv("CMDS_AD_EYECANDIES_RGB_ROOT", str(RELEASE_ROOT / "data/derived/output_generation_eyecandies"))
# 最终 Train Normal 输出目录 (建议换个名字防覆盖)
TRAIN_OUTPUT_ROOT = os.getenv("CMDS_AD_EYECANDIES_NORMAL_ROOT", str(RELEASE_ROOT / "data/derived/normal_output_train_eyecandies"))

# 2. 脚本路径
# 删除了 SCRIPT_REAL_PATH，因为直接拷贝图片即可
SCRIPT_EST_PATH = str(RELEASE_ROOT / "third_party/marigold/run_normals.py")

# 3. 运行参数
PYTHON_BIN = "python -u"
MARIGOLD_BATCH_SIZE = 1
SEED = 42

# 4. Categories and GPU IDs can be selected with CMDS_AD_CLASSES and CMDS_AD_GPUS.
ALL_CATEGORIES = [
    "CandyCane", "ChocolateCookie", "ChocolatePraline", "GummyBear",
    "HazelnutTruffle", "LicoriceSandwich", "Lollipop", "Confetto",
    "Marshmallow", "PeppermintCandy"
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
        return False

def process_class(gpu_id, class_name):
    """单个类别的处理逻辑"""
    global task_status

    # 更新状态
    task_status[class_name] = {"gpu": str(gpu_id), "status": "Initializing...", "step": "Prep"}

    # 1. 路径定义
    input_rgb_dir = os.path.join(GEN_RGB_ROOT, class_name)
    # 现成 Real Normal 的源目录 (train/good/normals)
    orig_normal_source_dir = os.path.join(ORIG_DATASET_ROOT, class_name, "train", "good", "normals")

    # 输出目录
    out_real_dir = os.path.join(TRAIN_OUTPUT_ROOT, "real_normals", class_name, "train", "good")
    out_est_dir = os.path.join(TRAIN_OUTPUT_ROOT, "estimated_normals", class_name, "train", "good")

    if not os.path.exists(input_rgb_dir):
        task_status[class_name]["status"] = "[dim]Skipped (No Data)[/dim]"
        return

    os.makedirs(out_real_dir, exist_ok=True)
    os.makedirs(out_est_dir, exist_ok=True)

    # =========================================================
    # Task 1: Copy Real Normals (直接复制现成图片，无需执行脚本)
    # =========================================================
    task_status[class_name]["status"] = "[yellow]Copying Real Normals[/yellow]"
    task_status[class_name]["step"] = "Real (Copying)"

    try:
        # 扫描 input_rgb_dir 里的所有图片
        all_imgs = sorted(glob.glob(os.path.join(input_rgb_dir, "*")))
        found_real_count = 0

        for img_path in all_imgs:
            filename = os.path.basename(img_path)
            name_stem, ext = os.path.splitext(filename)

            # 只处理原图 (不含 seed 的名称，例如 000.png)
            if "seed" in name_stem or ext.lower() not in ['.png', '.jpg', '.jpeg']:
                continue

            # 寻找对应的现成 Normal 图
            src_normal_path = os.path.join(orig_normal_source_dir, f"{name_stem}.png")

            # 如果源 Normal 存在，直接复制到 out_real_dir
            if os.path.exists(src_normal_path):
                shutil.copy(src_normal_path, os.path.join(out_real_dir, f"{name_stem}.png"))
                found_real_count += 1

        # 由于只是复制文件，速度极快，无需调用 subprocess
        if found_real_count == 0:
            task_status[class_name]["status"] = "[yellow]Warn: No Real Normals Found[/yellow]"

    except Exception as e:
        task_status[class_name]["status"] = "[red]Failed Real Copy[/red]"
        return

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
    table = Table(title="Eyecandies Normal Generation Status", box=box.ROUNDED)
    table.add_column("GPU", style="magenta", justify="center", width=5)
    table.add_column("Category", style="cyan", width=18)
    table.add_column("Current Task", style="dim", width=20)
    table.add_column("Status", justify="center", width=25)

    # 按 GPU 顺序显示
    for gpu_id in sorted(GPU_ASSIGNMENT.keys()):
        cats = GPU_ASSIGNMENT[gpu_id]
        for cat in cats:
            info = task_status.get(cat, {"status": "Waiting", "step": "-"})
            table.add_row(str(gpu_id), cat, info["step"], info["status"])

    return table

def main():
    # 0. 检查脚本
    if not os.path.exists(SCRIPT_EST_PATH):
        console.print(f"[bold red]Marigold script not found at {SCRIPT_EST_PATH}![/bold red]")
        sys.exit(1)

    # 1. 初始化状态
    for gpu_id, cats in GPU_ASSIGNMENT.items():
        for cat in cats:
            task_status[cat] = {"gpu": str(gpu_id), "status": "Waiting", "step": "-"}

    # 2. 启动 4 个线程
    threads = []
    console.print(f"[bold green]Starting parallel normal generation on 4 GPUs...[/bold green]")

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
