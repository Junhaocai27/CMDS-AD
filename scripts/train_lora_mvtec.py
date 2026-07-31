import os
import sys
import subprocess
import time
import threading
from datetime import datetime
from queue import Queue
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box

from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]

# ================= 配置区域 =================

# 基础路径配置
MODEL_NAME = os.getenv("CMDS_AD_SD_MODEL", str(RELEASE_ROOT / "weights/stable-diffusion-2-1-base"))
DATA_ROOT = os.getenv("CMDS_AD_MVTEC_ROOT", str(RELEASE_ROOT / "data/derived/mvtec_3d"))
OUTPUT_ROOT = os.getenv("CMDS_AD_MVTEC_LORA_ROOT", str(RELEASE_ROOT / "weights/lora_mvtec"))
LOG_DIR = str(RELEASE_ROOT / "logs/lora_mvtec")

# MVTec 3D-AD categories. Set CMDS_AD_CLASSES to select a subset.
ALL_CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire"
]
CATEGORIES = os.getenv("CMDS_AD_CLASSES", " ".join(ALL_CATEGORIES)).split()

# Comma-separated GPU IDs, for example CMDS_AD_GPUS=0,1.
AVAILABLE_GPUS = [int(value) for value in os.getenv("CMDS_AD_GPUS", "0,1").split(",") if value.strip()]

# LoRA 训练参数 (除了变化的 path 和 token，其他固定参数)
BASE_ARGS = {
    "--train_text_encoder": "",
    "--resolution": "512",
    "--train_batch_size": "1",
    "--gradient_accumulation_steps": "4",
    "--gradient_checkpointing": "",
    "--scale_lr": "",
    "--learning_rate_unet": "1e-4",
    "--learning_rate_text": "1e-5",
    "--learning_rate_ti": "5e-4",
    "--color_jitter": "",
    "--lr_scheduler": "linear",
    "--lr_warmup_steps": "0",
    "--lr_scheduler_lora": "linear",
    "--lr_warmup_steps_lora": "100",
    "--use_template": "object",
    "--save_steps": "100",
    "--max_train_steps_ti": "1000",
    "--max_train_steps_tuning": "1000",
    "--perform_inversion": "True",
    "--clip_ti_decay": "",
    "--weight_decay_ti": "0.000",
    "--weight_decay_lora": "0.001",
    "--continue_inversion": "",
    "--continue_inversion_lr": "1e-4",
    "--lora_rank": "8",
    "--lora_clip_target_modules": "{'CLIPSdpaAttention'}", # 注意这里作为字符串传递
    # "--use_face_segmentation_condition": "" # 如果需要取消注释
}

# ===========================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 状态管理字典
task_status = {cat: {"status": "Waiting", "gpu": "-", "time": "0s"} for cat in CATEGORIES}
gpu_queue = Queue()
for gpu in AVAILABLE_GPUS:
    gpu_queue.put(gpu)

console = Console()

def get_command(category, gpu_id):
    """构建命令行参数列表"""
    instance_dir = os.path.join(DATA_ROOT, category, "train/good/rgb")
    output_dir = os.path.join(OUTPUT_ROOT, category)

    cmd = [
        "lora_pti",
        f"--pretrained_model_name_or_path={MODEL_NAME}",
        f"--instance_data_dir={instance_dir}",
        f"--output_dir={output_dir}",
        f"--device=cuda:{gpu_id}",
        f"--placeholder_tokens=<{category}>",
    ]

    # 添加基础参数
    for key, value in BASE_ARGS.items():
        if value:
            cmd.append(f"{key}={value}")
        else:
            cmd.append(key) # 处理像 --scale_lr 这种没有值的 flag

    return cmd

def run_task(category):
    """单个任务的执行逻辑"""
    global task_status

    # 1. 获取 GPU 资源 (如果队列为空，这里会阻塞等待)
    gpu_id = gpu_queue.get()

    try:
        # 更新状态
        start_time = time.time()
        task_status[category]["status"] = "[yellow]Running[/yellow]"
        task_status[category]["gpu"] = f"CUDA {gpu_id}"

        # 构建命令
        cmd = get_command(category, gpu_id)
        log_file_path = os.path.join(LOG_DIR, f"{category}.log")

        # 执行命令，重定向输出到日志文件
        with open(log_file_path, "w") as log_file:
            # 写入启动信息
            log_file.write(f"Starting training for {category} on GPU {gpu_id}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.flush()

            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy() # 继承当前环境变量
            )

            # 等待进程结束
            while process.poll() is None:
                elapsed = time.time() - start_time
                task_status[category]["time"] = f"{elapsed:.1f}s"
                time.sleep(1) # 减少刷新频率

            return_code = process.returncode

        elapsed = time.time() - start_time
        task_status[category]["time"] = f"{elapsed:.1f}s"

        if return_code == 0:
            task_status[category]["status"] = "[green]Finished[/green]"
        else:
            task_status[category]["status"] = "[red]Failed (See Log)[/red]"

    except Exception as e:
        task_status[category]["status"] = f"[red]Error: {str(e)}[/red]"
    finally:
        # 3. 释放 GPU 资源，归还到队列
        task_status[category]["gpu"] = "-"
        gpu_queue.put(gpu_id)

def generate_table():
    """生成 Rich 表格"""
    table = Table(title="MVTec 3D-AD LoRA Training Status", box=box.ROUNDED)
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("GPU", justify="center", style="magenta")
    table.add_column("Elapsed Time", justify="right")

    for cat in CATEGORIES:
        info = task_status[cat]
        table.add_row(cat, info["status"], info["gpu"], info["time"])

    return table

def main():
    threads = []

    # 创建所有任务线程
    for category in CATEGORIES:
        t = threading.Thread(target=run_task, args=(category,))
        threads.append(t)
        t.start()

    # 使用 Rich Live 显示动态面板
    with Live(generate_table(), refresh_per_second=4) as live:
        while any(t.is_alive() for t in threads):
            live.update(generate_table())
            time.sleep(0.5)
        # 最后更新一次确保状态正确
        live.update(generate_table())

    print(f"\nAll tasks completed. Logs available in {LOG_DIR}")

if __name__ == "__main__":
    main()
