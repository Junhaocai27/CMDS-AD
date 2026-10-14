import os
import sys
import glob
import shutil
import torch
import time
import multiprocessing
from queue import Empty
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "third_party"))

from lora_diffusion.lora import tune_lora_scale, patch_pipe


from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import box
from rich.panel import Panel
from rich.layout import Layout



MODEL_ID = os.getenv("CMDS_AD_SD_MODEL", str(RELEASE_ROOT / "weights/stable-diffusion-2-1-base"))
LORA_PATH_TEMPLATE = os.path.join(os.getenv("CMDS_AD_MVTEC_LORA_ROOT", str(RELEASE_ROOT / "weights/lora_mvtec")), "{cat}", "final_lora.safetensors")
INPUT_DIR_TEMPLATE = os.path.join(os.getenv("CMDS_AD_MVTEC_ROOT", str(RELEASE_ROOT / "data/derived/mvtec_3d")), "{cat}", "train/good/rgb")
OUTPUT_DIR_TEMPLATE = os.path.join(os.getenv("CMDS_AD_MVTEC_RGB_ROOT", str(RELEASE_ROOT / "data/derived/output_generation_full")), "{cat}")

NUM_FEW_SHOT = 9999
PROMPT_TEMPLATE = "a low high light and real photo of <{cat}>"
STRENGTH = 0.2
GUIDANCE_SCALE = 4
STEPS = 50
FIXED_SEEDS = [42, 1024, 2026, 8888, 12345]
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

def worker_process(gpu_id, categories, queue):
    try:

        queue.put(("log", (gpu_id, "Starting Process")))

        for category in categories:

            queue.put(("status", (gpu_id, category, "Loading Model...", 0, 0)))


            lora_path = LORA_PATH_TEMPLATE.format(cat=category)
            input_dir = INPUT_DIR_TEMPLATE.format(cat=category)
            output_dir = OUTPUT_DIR_TEMPLATE.format(cat=category)
            prompt = PROMPT_TEMPLATE.format(cat=category)

            if not os.path.exists(lora_path):
                queue.put(("log", (gpu_id, f"[Skip] No LoRA for {category}")))
                continue

            os.makedirs(output_dir, exist_ok=True)


            try:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    MODEL_ID,
                    safety_checker=None,
                    torch_dtype=torch.float16
                ).to(f"cuda:{gpu_id}")

                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                pipe.set_progress_bar_config(disable=True)

                patch_pipe(pipe, lora_path, patch_text=True, patch_ti=True, patch_unet=True)
                tune_lora_scale(pipe.unet, 1.0)
                tune_lora_scale(pipe.text_encoder, 1.0)
            except Exception as e:
                queue.put(("error", (gpu_id, f"Model Error: {str(e)[:20]}")))
                continue


            image_extensions = ['*.png', '*.jpg', '*.jpeg']
            all_image_paths = []
            for ext in image_extensions:
                all_image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
            all_image_paths = sorted(all_image_paths)

            selected_paths = all_image_paths[:NUM_FEW_SHOT] if len(all_image_paths) > NUM_FEW_SHOT else all_image_paths

            total_tasks = len(selected_paths)



            queue.put(("status", (gpu_id, category, "Generating...", 0, total_tasks)))


            for i, img_path in enumerate(selected_paths):
                file_name = os.path.basename(img_path)
                file_stem = os.path.splitext(file_name)[0]

                try:

                    target_real_path = os.path.join(output_dir, file_name)
                    if not os.path.exists(target_real_path):
                        shutil.copy(img_path, target_real_path)

                    init_image = Image.open(img_path).convert("RGB").resize((512, 512))

                    for seed in FIXED_SEEDS:
                        generator = torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed)

                        result_image = pipe(
                            prompt=prompt,
                            image=init_image,
                            strength=STRENGTH,
                            num_inference_steps=STEPS,
                            guidance_scale=GUIDANCE_SCALE,
                            generator=generator
                        ).images[0]

                        save_name = f"{file_stem}_seed{seed}.png"
                        result_image.save(os.path.join(output_dir, save_name))

                except Exception as e:
                    queue.put(("log", (gpu_id, f"Error {file_name}: {e}")))


                queue.put(("progress", (gpu_id, i + 1)))


            del pipe
            torch.cuda.empty_cache()

            queue.put(("log", (gpu_id, f"Finished {category}")))

        queue.put(("done", gpu_id))

    except Exception as e:
        queue.put(("error", (gpu_id, str(e))))



def make_table(gpu_states):
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("GPU", justify="center", style="magenta", width=6)
    table.add_column("Category", style="cyan", width=14)
    table.add_column("Progress", ratio=1)
    table.add_column("Count", justify="right", width=10)
    table.add_column("Status", style="dim", width=20)

    for gpu_id in sorted(gpu_states.keys()):
        state = gpu_states[gpu_id]


        if state["total"] > 0:
            completed = state["completed"]
            total = state["total"]
            percent = (completed / total) * 100


            bar_len = 20
            filled_len = int(bar_len * (completed / total))
            bar = "█" * filled_len + "░" * (bar_len - filled_len)
            progress_render = f"[{bar}] {percent:.0f}%"
            count_render = f"{completed}/{total}"
        else:
            progress_render = "[--------------------] 0%"
            count_render = "-/-"

        status_style = "green" if state["status"] == "Done" else "yellow" if "Generating" in state["status"] else "white"
        if state["error"]:
            status_style = "bold red"
            state["status"] = "ERROR"

        table.add_row(
            str(gpu_id),
            state["category"],
            progress_render,
            count_render,
            f"[{status_style}]{state['status']}[/{status_style}]"
        )
    return table

def main():
    multiprocessing.set_start_method('spawn', force=True)


    manager = multiprocessing.Manager()
    queue = manager.Queue()

    console = Console()
    console.print("[bold green]Starting Parallel Img2Img Generation...[/bold green]")


    processes = []
    for gpu_id, cats in GPU_ASSIGNMENT.items():
        p = multiprocessing.Process(target=worker_process, args=(gpu_id, cats, queue))
        p.start()
        processes.append(p)


    gpu_states = {
        gid: {
            "category": "-",
            "status": "Waiting",
            "completed": 0,
            "total": 0,
            "error": False
        }
        for gid in GPU_ASSIGNMENT.keys()
    }

    active_gpus = len(GPU_ASSIGNMENT)


    with Live(make_table(gpu_states), refresh_per_second=10) as live:
        while active_gpus > 0:
            try:

                while not queue.empty():
                    msg_type, data = queue.get_nowait()

                    if msg_type == "status":
                        # (gpu_id, category, status_text, current, total)
                        gid, cat, stat, curr, tot = data
                        gpu_states[gid]["category"] = cat
                        gpu_states[gid]["status"] = stat
                        gpu_states[gid]["completed"] = curr
                        gpu_states[gid]["total"] = tot

                    elif msg_type == "progress":
                        # (gpu_id, current)
                        gid, curr = data
                        gpu_states[gid]["completed"] = curr

                    elif msg_type == "log":

                        pass

                    elif msg_type == "done":
                        gid = data
                        gpu_states[gid]["status"] = "Done"
                        gpu_states[gid]["category"] = "All Finished"
                        active_gpus -= 1

                    elif msg_type == "error":
                        gid, err = data
                        gpu_states[gid]["error"] = True
                        gpu_states[gid]["status"] = err

                        active_gpus -= 1

                live.update(make_table(gpu_states))
                time.sleep(0.1)

            except KeyboardInterrupt:
                console.print("[bold red]Interrupted by user![/bold red]")
                for p in processes:
                    p.terminate()
                break

    console.print("[bold green]All tasks finished![/bold green]")

if __name__ == "__main__":
    main()
