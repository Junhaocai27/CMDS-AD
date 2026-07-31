# CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection

[![ARXIV · PAPER](https://img.shields.io/badge/ARXIV-PAPER?style=for-the-badge&labelColor=555555&color=b31b1b&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.20300)
[![PROJECT · PAGE](https://img.shields.io/badge/PROJECT-PAGE?style=for-the-badge&labelColor=555555&color=0077b6&logo=googlechrome&logoColor=white)](https://cmds-ad.github.io/)
[![MODEL · WEIGHTS](https://img.shields.io/badge/MODEL-WEIGHTS?style=for-the-badge&labelColor=555555&color=2e7d32&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1CNmmkdtiumA47AJ1PoZ4klwAyvFbvvYq/view?usp=sharing)

Official code release for CMDS-AD, a cross-modal few-shot anomaly detection
framework for 3D industrial inspection. The repository contains the complete
data preparation, RGB augmentation, real/estimated normal generation,
foreground-mask generation, dual-direction training, and evaluation code.

The repository is source-only. Datasets, foundation-model checkpoints, LoRA
files, and CMDS-AD checkpoints are downloaded or obtained separately.

## 📰 News

- **2026-06-18:** Our paper was accepted to ECCV.
- **2026-06-20:** We uploaded the arXiv version and released the project page.
- **2026-07-31:** We released the complete source code and trained weights.

## 📦 Released weights

After extracting the release archive, use these logical paths:

```text
weights/lora_mvtec/<class>/final_lora.safetensors
weights/lora_eyecandies/<class>/final_lora.safetensors

checkpoints/checkpoints_dual_2dto3d_<class>_<shots>shot/
checkpoints/checkpoints_dual_3dto2d_<class>_<shots>shot/
checkpoints/checkpoints_eyecandies_dual_2dto3d_<class>_<shots>shot/
checkpoints/checkpoints_eyecandies_dual_3dto2d_<class>_<shots>shot/
```

Each CMDS-AD checkpoint directory contains the corresponding
`model_real_*_final.pth` and `model_est_*_final.pth` files. `<shots>` is `1`,
`2`, or `4`; inference uses the suffix `_1shot`, `_2shot`, or `_4shot`.
Older `_new3` checkpoint directory names are also recognized.

The archive does not contain either dataset, Stable Diffusion 2.1, Marigold,
or DINO. See the asset links below.

For evaluation with a released CMDS-AD checkpoint, skip LoRA training, i2i
generation, and CMDS-AD training: prepare the dataset, generate the test
normals and foreground masks, then run [Inference and evaluation](#-inference-and-evaluation).
For a from-scratch reproduction, follow the sections below in order.

## 🧰 Installation and external assets

Linux, CUDA, and Python 3.10 are recommended. Install a PyTorch/torchvision
pair compatible with the NVIDIA driver first.

```bash
git clone https://github.com/Junhaocai27/CMDS-AD.git
cd CMDS-AD
conda create -n cmds-ad python=3.10 -y
conda activate cmds-ad
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p data/raw data/derived weights checkpoints results
```

Install PointNet2 for foreground masks:

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
python -m pip install -e Pointnet2_PyTorch/pointnet2_ops_lib
```

The LoRA stage follows the upstream [LoRA PTI interface](https://github.com/cloneofsimo/lora)
and calls the local vendored package's `lora_pti` executable. Install its
dependencies, install the local package, and verify the entry point before
training:

```bash
python -m pip install -r third_party/lora_requirements.txt
python -m pip install -e third_party --no-deps
command -v lora_pti
lora_pti --help
```

The `--no-deps` flag avoids replacing the already selected PyTorch/CUDA stack;
the LoRA-specific dependencies are installed by the preceding command. The
i2i scripts import the same vendored implementation directly.

### 📊 Dataset and model links

| Resource | Source | Used for |
|---|---|---|
| MVTec 3D-AD | [Dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad) | RGB, XYZ, anomaly GT |
| Eyecandies | [Project page](https://eyecan-ai.github.io/) · [Downloader](https://github.com/eyecan-ai/eyecandies) | RGB, depth, anomaly masks |
| Stable Diffusion 2.1 | [Model card](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) | LoRA and i2i RGB |
| Marigold normals | [Model card](https://huggingface.co/prs-eth/marigold-normals-v1-1) | Estimated normals |
| DINO ViT-B/8 | [Model card](https://huggingface.co/timm/vit_base_patch8_224.dino) | RGB/normal features |

Place MVTec categories directly under `data/raw/mvtec_3d/`:

```text
data/raw/mvtec_3d/{bagel,cable_gland,carrot,cookie,dowel,foam,peach,potato,rope,tire}/
```

Download Eyecandies with its official downloader:

```bash
git clone https://github.com/eyecan-ai/eyecandies.git /path/to/eyecandies
python -m pip install -e "/path/to/eyecandies[torch]"
eyec ec-get +o "$PWD/data/raw/Eyecandies"
```

Download the foundation models with the Hugging Face CLI. Stable Diffusion
requires accepting its license and, when requested, running `hf auth login`.

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download stabilityai/stable-diffusion-2-1-base \
  --local-dir weights/stable-diffusion-2-1-base
hf download prs-eth/marigold-normals-v1-1 \
  --local-dir weights/marigold
hf download timm/vit_base_patch8_224.dino
```

The DINO model is loaded as `vit_base_patch8_224.dino` through `timm` and is
normally cached under `~/.cache/huggingface`. Set `HF_HOME` if using another
cache location.

## 🧪 Dataset preparation and annotations

Keep the raw datasets unchanged: MVTec preprocessing modifies XYZ TIFF files
in place, so it must operate on a copy.

### MVTec 3D-AD

```bash
cp -a data/raw/mvtec_3d data/derived/mvtec_3d
python processing/preprocess_mvtec3dad.py \
  --dataset_path data/derived/mvtec_3d
```

The converted tree contains, for example:

```text
data/derived/mvtec_3d/bagel/train/good/rgb/
data/derived/mvtec_3d/bagel/train/good/xyz/
data/derived/mvtec_3d/bagel/test/<defect>/{rgb,xyz,gt}/
```

The MVTec `gt/` files are test-set anomaly annotations and must be retained.
`test/good` is normal and may use an implicit all-zero anomaly mask.

### Eyecandies

```bash
python processing/preprocess_eyecandies.py \
  --dataset_path data/raw/Eyecandies \
  --target_dir data/derived/eyecandies_mvtec_format
```

The converter creates RGB, XYZ, and test anomaly-mask trees in an MVTec-like
layout. It writes XYZ rather than normal maps; real normals are consequently
computed from XYZ by the CUDA Open3D stage below.

Anomaly ground truth and foreground masks are different. Ground truth is used
only for test metrics. Foreground masks are generated by
`mask_vis_mvtec3dad.py` or `mask_vis_eyecandies.py` and are used by the
training/inference foreground constraint.

Audit the converted annotations before running an experiment:

```bash
python scripts/validate_dataset.py \
  --dataset mvtec \
  --dataset_root data/derived/mvtec_3d \
  --splits test
python scripts/validate_dataset.py \
  --dataset eyecandies \
  --dataset_root data/derived/eyecandies_mvtec_format \
  --splits test
```

Expected complete-dataset totals are 1,197 MVTec test images with 948
anomalous images, and 500 Eyecandies test images with 240 anomalous images.

## 🔄 RGB, normal, and foreground-mask generation

Run the following blocks in one shell. Choose one dataset profile first; the
CLI scripts use `--classes`, while the LoRA/i2i launchers use
`CMDS_AD_CLASSES`. Change all profile variables when switching datasets.

The class names are:

```text
MVTec:      bagel cable_gland carrot cookie dowel foam peach potato rope tire
Eyecandies: CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear
            HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy
```

### Dataset profile: MVTec 3D-AD

```bash
DATASET=mvtec
CLASSES="bagel carrot"
DATASET_ROOT="$PWD/data/derived/mvtec_3d"
RGB_ROOT="$PWD/data/derived/output_generation_full"
TRAIN_REAL_ROOT="$PWD/data/derived/normal_output_train_new_full/real_normals"
TRAIN_EST_ROOT="$PWD/data/derived/normal_output_train_new_full/estimated_normals"
TEST_REAL_ROOT="$PWD/data/derived/normal_output_mv_format_infer/real_normals"
TEST_EST_ROOT="$PWD/data/derived/normal_output_mv_format_infer/estimated_normals"
MASK_ROOT="$PWD/data/derived/mvtec_3d_masks_generated"
GPUS="0,1"       # use "0" on a single-GPU machine
SD_MODEL="$PWD/weights/stable-diffusion-2-1-base"
LORA_ROOT="$PWD/weights/lora_mvtec"
```

### Dataset profile: Eyecandies

```bash
DATASET=eyecandies
CLASSES="CandyCane ChocolateCookie"
DATASET_ROOT="$PWD/data/derived/eyecandies_mvtec_format"
RGB_ROOT="$PWD/data/derived/output_generation_eyecandies"
TRAIN_REAL_ROOT="$PWD/data/derived/normal_output_train_eyecandies/real_normals"
TRAIN_EST_ROOT="$PWD/data/derived/normal_output_train_eyecandies/estimated_normals"
TEST_REAL_ROOT="$PWD/data/derived/normal_output_eyecandies_infer/real_normals"
TEST_EST_ROOT="$PWD/data/derived/normal_output_eyecandies_infer/estimated_normals"
MASK_ROOT="$PWD/data/derived/eyecandies_masks_generated"
GPUS="0,1"
SD_MODEL="$PWD/weights/stable-diffusion-2-1-base"
LORA_ROOT="$PWD/weights/lora_eyecandies"
```

### Common data-generation commands

```bash
python scripts/prepare_rgb.py \
  --dataset "$DATASET" \
  --dataset_root "$DATASET_ROOT" \
  --output_root "$RGB_ROOT" \
  --classes $CLASSES
```

Train the category-specific LoRA and generate five i2i images per original
training RGB. The `CMDS_AD_*` settings are scoped to each launcher command;
they do not need to be set globally. Run only the block for the selected
dataset profile:

#### MVTec 3D-AD

```bash
CMDS_AD_CLASSES="$CLASSES" CMDS_AD_GPUS="$GPUS" \
CMDS_AD_SD_MODEL="$SD_MODEL" CMDS_AD_MVTEC_ROOT="$DATASET_ROOT" \
CMDS_AD_MVTEC_LORA_ROOT="$LORA_ROOT" \
python scripts/train_lora_mvtec3dad.py

CMDS_AD_CLASSES="$CLASSES" CMDS_AD_GPUS="$GPUS" \
CMDS_AD_SD_MODEL="$SD_MODEL" CMDS_AD_MVTEC_ROOT="$DATASET_ROOT" \
CMDS_AD_MVTEC_LORA_ROOT="$LORA_ROOT" CMDS_AD_MVTEC_RGB_ROOT="$RGB_ROOT" \
python scripts/generate_rgb_mvtec3dad.py
```

#### Eyecandies

```bash
CMDS_AD_CLASSES="$CLASSES" CMDS_AD_GPUS="$GPUS" \
CMDS_AD_SD_MODEL="$SD_MODEL" CMDS_AD_EYECANDIES_ROOT="$DATASET_ROOT" \
CMDS_AD_EYECANDIES_LORA_ROOT="$LORA_ROOT" \
python scripts/train_lora_eyecandies.py

CMDS_AD_CLASSES="$CLASSES" CMDS_AD_GPUS="$GPUS" \
CMDS_AD_SD_MODEL="$SD_MODEL" CMDS_AD_EYECANDIES_ROOT="$DATASET_ROOT" \
CMDS_AD_EYECANDIES_LORA_ROOT="$LORA_ROOT" CMDS_AD_EYECANDIES_RGB_ROOT="$RGB_ROOT" \
python scripts/generate_rgb_eyecandies.py
```

The fixed i2i seeds are `42`, `1024`, `2026`, `8888`, and `12345`; defaults
are strength `0.2`, guidance scale `4`, and `50` diffusion steps.
If an output directory was generated with a different seed list, rerun the
i2i command to keep filenames and image contents consistent; do not rename
the existing images manually.

Generate training and test real normals:

```bash
python scripts/generate_real_normals.py \
  --dataset "$DATASET" \
  --dataset_root "$DATASET_ROOT" \
  --output_root "$TRAIN_REAL_ROOT" \
  --classes $CLASSES \
  --splits train \
  --device cuda

python scripts/generate_real_normals.py \
  --dataset "$DATASET" \
  --dataset_root "$DATASET_ROOT" \
  --output_root "$TEST_REAL_ROOT" \
  --classes $CLASSES \
  --splits test \
  --device cuda
```

Generate training and test estimated normals with Marigold:

```bash
python scripts/generate_estimated_normals.py \
  --dataset "$DATASET" \
  --dataset_root "$DATASET_ROOT" \
  --rgb_root "$RGB_ROOT" \
  --output_root "$TRAIN_EST_ROOT" \
  --checkpoint weights/marigold \
  --classes $CLASSES \
  --splits train \
  --batch_size 1 \
  --seed 42 \
  --denoise_steps 10 \
  --device cuda

python scripts/generate_estimated_normals.py \
  --dataset "$DATASET" \
  --dataset_root "$DATASET_ROOT" \
  --output_root "$TEST_EST_ROOT" \
  --checkpoint weights/marigold \
  --classes $CLASSES \
  --splits test \
  --batch_size 1 \
  --seed 42 \
  --denoise_steps 10 \
  --device cuda
```

Marigold therefore uses `denoise_steps=10`, `ensemble_size=10`, `seed=42`,
the checkpoint default resolution, full precision, and batch size 1. Real
normals use Open3D CUDA with `radius=0.01` and `max_nn=30`.

Generate foreground masks. Use `--class_name all` for a complete dataset, or
run the command once per selected class:

```bash
# MVTec
python mask_vis_mvtec3dad.py \
  --dataset_path data/derived/mvtec_3d \
  --save_dir data/derived/mvtec_3d_masks_generated \
  --class_name all \
  --batch_size 1

# Eyecandies
python mask_vis_eyecandies.py \
  --dataset_path data/derived/eyecandies_mvtec_format \
  --save_dir data/derived/eyecandies_masks_generated \
  --class_name all \
  --batch_size 1
```

## 🧠 CMDS-AD training

The real stream uses original RGB, real normal, and foreground mask. The
estimated stream uses original and generated RGB with Marigold normals;
generated stems reuse the corresponding original real-normal and mask files.

| **shots** | **1** | **2** | **4** |
|---|---:|---:|---:|
| **default training steps** | 3000 | 1500 | 750 |

With either dataset profile selected above, run:

```bash
python scripts/train.py \
  --dataset "$DATASET" \
  --classes $CLASSES \
  --shots 1 2 4 \
  --directions both \
  --batch_size 1 \
  --device cuda \
  --checkpoint_root checkpoints \
  --wandb_mode disabled
```

This batch-size-1 command is the validated 24 GB configuration. On the
original 32 GB-class deployment, change only `--batch_size 1` to
`--batch_size 2`. Use `--directions 2dto3d` or `--directions 3dto2d` for one
direction only; use `--max_steps` only when intentionally changing the shot
schedule.

## 🔍 Inference and evaluation

Select the same dataset profile used for data generation. For MVTec, set:

```bash
INFER_SCRIPT=test_anomaly_fusion_mvtec3dad.py
CKPT_ROOT_3D2D=checkpoints/checkpoints_dual_3dto2d
CKPT_ROOT_2D3D=checkpoints/checkpoints_dual_2dto3d
RESULT_FILE=results/mvtec_result.txt
```

For Eyecandies, set instead:

```bash
INFER_SCRIPT=test_anomaly_fusion_eyecandies.py
CKPT_ROOT_3D2D=checkpoints/checkpoints_eyecandies_dual_3dto2d
CKPT_ROOT_2D3D=checkpoints/checkpoints_eyecandies_dual_2dto3d
RESULT_FILE=results/eyecandies_result.txt
```

Run the same evaluator command for either profile:

```bash
python "$INFER_SCRIPT" \
  --dataset_root "$DATASET_ROOT" \
  --real_normal_root "$TEST_REAL_ROOT" \
  --est_normal_root "$TEST_EST_ROOT" \
  --mask_root "$MASK_ROOT" \
  --ckpt_root_3d2d "$CKPT_ROOT_3D2D" \
  --ckpt_root_2d3d "$CKPT_ROOT_2D3D" \
  --ckpt_suffix _1shot \
  --classes $CLASSES \
  --batch_size 1 \
  --wandb_mode disabled \
  2>&1 | tee "$RESULT_FILE"
```

Generated RGB images enlarge the training stream; original test RGB is used
for evaluation together with both normal streams and the foreground mask.
Use `--batch_size 2` on the original high-memory deployment and
`--max_samples 1` for a smoke test. Remove `--max_samples` for reported
metrics. The scripts print I-AUROC, P-AUROC, and AUPRO to stdout; WandB
`offline` or `online` additionally stores metrics and visualizations.

## 📋 Expected data contract

Training data must provide:

```text
<RGB_ROOT>/<class>/
<TRAIN_REAL_ROOT>/<class>/train/good/
<TRAIN_EST_ROOT>/<class>/train/good/normals_vis/
<MASK_ROOT>/<class>/train/good/
```

Test data must provide RGB, XYZ, anomaly GT, test real normals, test estimated
normals, and test foreground masks under the corresponding profile roots.
Estimated normals contain one file per RGB image. Real normals and masks are
indexed by the base stem and reused for generated training variants.

## ✅ Validation and smoke tests

```bash
python scripts/validate_release.py
python -m compileall -q .
python scripts/validate_dataset.py \
  --dataset mvtec --dataset_root data/derived/mvtec_3d --splits test
python scripts/validate_dataset.py \
  --dataset eyecandies \
  --dataset_root data/derived/eyecandies_mvtec_format --splits test
python scripts/train.py \
  --dataset mvtec --classes bagel --shots 1 --directions both --dry-run
python scripts/generate_estimated_normals.py \
  --dataset mvtec --classes bagel --checkpoint weights/marigold --dry-run
```

Before a full experiment, validate one category through LoRA, i2i, real
normal generation, Marigold, foreground-mask generation, and inference with
`--max_samples 1`. CUDA normal maps can differ slightly across GPU models.

## 📚 Citation

```bibtex
@article{cmdsad2026,
  title   = {CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection},
  journal = {arXiv preprint arXiv:2606.20300},
  year    = {2026}
}
```

## 🙏 Acknowledgements and licenses

- [CFM — Crossmodal Feature Mapping](https://github.com/CVLAB-Unibo/crossmodal-feature-mapping)
- [Marigold](https://github.com/prs-eth/Marigold)
- [DINO/timm](https://github.com/huggingface/pytorch-image-models)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [LoRA Diffusion](https://github.com/cloneofsimo/lora)
- [Open3D](https://github.com/isl-org/Open3D)
- [PointNet2](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [Point-MAE](https://github.com/Pang-Yunsheng/Point-MAE)
- [Eyecandies](https://github.com/eyecan-ai/eyecandies)

<details>
<summary>🔗 Full upstream references</summary>

- [CFM project page](https://cvlab-unibo.github.io/CrossmodalFeatureMapping/) · [paper](https://arxiv.org/abs/2312.04521)
- [Marigold normals model](https://huggingface.co/prs-eth/marigold-normals-v1-1) · [project page](https://marigoldmonodepth.github.io/)
- [DINO model](https://huggingface.co/timm/vit_base_patch8_224.dino)
- [Stable Diffusion 2.1 model card](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- [Eyecandies project page](https://eyecan-ai.github.io/eyecandies/)

</details>

Please review all upstream licenses, model cards, and dataset terms before
redistribution or commercial use. See [LICENSE](LICENSE) and the license files
under `third_party/`.
