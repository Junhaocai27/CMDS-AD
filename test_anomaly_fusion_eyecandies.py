import warnings
warnings.filterwarnings("ignore")

import os

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import glob
import gc
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import wandb
from sklearn.metrics import roc_auc_score

# =========================================================================
# === 1. 环境与 WandB 设置 ===
# =========================================================================
os.environ["WANDB_START_METHOD"] = "thread"

try:
    from utils.metrics_utils import calculate_au_pro
except ImportError:
    calculate_au_pro = None
    print("Warning: 'utils.metrics_utils' not found. AUPRO metrics will be skipped.")

# =========================================================================
# 0. 参数解析 (完全适配 Eyecandies)
# =========================================================================
def get_args():
    parser = argparse.ArgumentParser(description="Fusion Inference: Direct Box Filter Integration (Eyecandies)")

    parser.add_argument("--dataset_root", type=str, default="./data/derived/eyecandies_mvtec_format")
    parser.add_argument("--real_normal_root", type=str, default="./data/derived/normal_output_eyecandies_infer/real_normals")
    parser.add_argument("--est_normal_root", type=str, default="./data/derived/normal_output_eyecandies_infer/estimated_normals")
    parser.add_argument("--mask_root", type=str, default="./data/derived/eyecandies_masks_generated",
                        help="Root directory for foreground masks")

    parser.add_argument("--ckpt_root_3d2d", type=str, default="./checkpoints/checkpoints_eyecandies_dual_3dto2d",
                        help="Prefix for 3D->2D ckpt folders")
    parser.add_argument("--ckpt_root_2d3d", type=str, default="./checkpoints/checkpoints_eyecandies_dual_2dto3d",
                        help="Prefix for 2D->3D ckpt folders")
    parser.add_argument("--ckpt_suffix", type=str, default="_4shot",
                        help="Suffix for checkpoint folder, e.g., _1shot, _2shot, or _4shot")

    parser.add_argument("--output_dir", type=str, default="./inference_results_eyecandies_boxfilter")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Classes to evaluate; omit to evaluate all Eyecandies classes")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit samples per class for a smoke test; 0 means all samples")

    parser.add_argument("--wandb_project", type=str, default="CFM_SOTA_Eyecandies")
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default=os.getenv("WANDB_MODE", "disabled"))
    parser.add_argument("--device", choices=["cuda"], default="cuda")

    return parser.parse_args()

# =========================================================================
# 2. Dataset 类
# =========================================================================
class SingleClassDataset(Dataset):
    def __init__(self, dataset_root, real_root, est_root, class_name, mask_root=None, img_size=224):
        self.img_size = img_size
        self.samples = []

        test_root = os.path.join(dataset_root, class_name, 'test')
        if not os.path.exists(test_root):
            print(f"Warning: Dataset root not found: {test_root}. Skipping class.")
            return

        defect_types = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])

        for dtype in defect_types:
            type_dir = os.path.join(test_root, dtype)
            rgb_dir = os.path.join(type_dir, 'rgb')
            if not os.path.isdir(rgb_dir): rgb_dir = type_dir

            img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.[jp][pn]g")))
            if len(img_paths) == 0: continue

            gt_dir_local = os.path.join(type_dir, 'gt')
            gt_dir_global = os.path.join(dataset_root, class_name, 'ground_truth', dtype)

            for img_path in img_paths:
                fname = os.path.basename(img_path)
                name_stem = os.path.splitext(fname)[0]

                gt_path = None
                if dtype != 'good':
                    candidates = [
                        os.path.join(gt_dir_local, fname),
                        os.path.join(gt_dir_local, f"{name_stem}_mask.png"),
                        os.path.join(gt_dir_global, f"{name_stem}_mask.png"),
                        os.path.join(gt_dir_global, fname)
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            gt_path = c
                            break

                fg_mask_path = None
                if mask_root is not None:
                    candidate_mask = os.path.join(mask_root, class_name, 'test', dtype, f"{name_stem}.png")
                    if os.path.exists(candidate_mask):
                        fg_mask_path = candidate_mask

                real_normal_path = os.path.join(real_root, class_name, 'test', dtype, fname)
                est_normal_path = os.path.join(est_root, class_name, 'test', dtype, "normals_vis", f"{name_stem}_normals.png")

                if os.path.exists(real_normal_path) and os.path.exists(est_normal_path):
                    self.samples.append({
                        "rgb_path": img_path,
                        "gt_path": gt_path,
                        "fg_mask_path": fg_mask_path,
                        "real_normal_path": real_normal_path,
                        "est_normal_path": est_normal_path,
                        "is_anomaly": 0 if dtype == 'good' else 1,
                        "name": f"{dtype}_{fname}"
                    })

        self.tf_rgb = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tf_norm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        self.tf_gt = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        rgb_pil = Image.open(s["rgb_path"]).convert('RGB')
        real_normal_pil = Image.open(s["real_normal_path"]).convert('RGB')
        est_normal_pil = Image.open(s["est_normal_path"]).convert('RGB')

        rgb = self.tf_rgb(rgb_pil)
        real_normal = self.tf_norm(real_normal_pil)
        est_normal = self.tf_norm(est_normal_pil)

        if s["gt_path"] is not None:
            gt_img = Image.open(s["gt_path"]).convert('L')
            gt = self.tf_gt(gt_img)
            gt = (gt > 0.5).float()
        else:
            gt = torch.zeros((1, self.img_size, self.img_size))

        if s["fg_mask_path"] is not None:
            fg_mask_img = Image.open(s["fg_mask_path"]).convert('L')
            fg_mask = self.tf_gt(fg_mask_img)
            fg_mask = (fg_mask > 0.5).float()
        else:
            fg_mask = torch.ones((1, self.img_size, self.img_size))

        return (rgb, real_normal, est_normal, gt, s["name"], torch.tensor([s["is_anomaly"]]), fg_mask)

# =========================================================================
# 3. 模型定义
# =========================================================================
class TripleLayerExtractor(nn.Module):
    def __init__(self, device, target_layers=[4, 7, 11], img_size=224):
        super().__init__()
        self.device, self.target_layers, self.img_size = device, target_layers, img_size
        self.backbone = timm.create_model('vit_base_patch8_224.dino', pretrained=True).to(device).eval()
        for p in self.backbone.parameters(): p.requires_grad = False

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x.to(self.device), n=max(self.target_layers)+1)
        layer_outputs = []
        for idx in self.target_layers:
            feat = features[idx]
            B, N, C = feat.shape
            if N > 784: feat = feat[:, -784:, :]
            h_dim = int(np.sqrt(feat.shape[1]))
            feat_map = feat.permute(0, 2, 1).reshape(B, C, h_dim, h_dim)
            layer_outputs.append(feat_map)

        final_outputs = []
        for feat in layer_outputs:
            feat_up = F.interpolate(feat, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            final_outputs.append(feat_up)
        return final_outputs

class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y); y = self.bn1(y); y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class HierarchicalGatedMapper(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proc_l4 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1), nn.GroupNorm(8, embed_dim), nn.GELU())
        self.proc_l7 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1), nn.GroupNorm(8, embed_dim), nn.GELU())
        self.proc_l11 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1), nn.GroupNorm(8, embed_dim), nn.GELU())
        self.coord_att = CoordinateAttention(embed_dim * 3, embed_dim * 3)
        self.spatial_selector = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=1)
        )
        self.fusion_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1)
        )

    def forward(self, features_list):
        x_l4, x_l7, x_l11 = features_list[0], features_list[1], features_list[2]
        f4 = self.proc_l4(x_l4)
        f7 = self.proc_l7(x_l7)
        f11 = self.proc_l11(x_l11)
        cat_feat = torch.cat([f4, f7, f11], dim=1)
        refined_feat = self.coord_att(cat_feat)
        spatial_weights = self.spatial_selector(refined_feat)
        spatial_weights = F.softmax(spatial_weights, dim=1)
        w4 = spatial_weights[:, 0:1, :, :]
        w7 = spatial_weights[:, 1:2, :, :]
        w11 = spatial_weights[:, 2:3, :, :]
        fused = w4 * f4 + w7 * f7 + w11 * f11
        out = self.fusion_head(fused)
        out_flat = out.flatten(2).transpose(1, 2)
        return out_flat, w4, out

# =========================================================================
# 4. 辅助距离计算函数
# =========================================================================
def compute_cosine_distance_torch(feat1, feat2):
    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1)
    sim = F.cosine_similarity(feat1, feat2, dim=-1)
    dist = 1.0 - sim
    B, N = dist.shape
    H = int(np.sqrt(N))
    return dist.view(B, 1, H, H)

# =========================================================================
# 5. 可视化工具
# =========================================================================
def denorm_rgb(tensor):
    m = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(tensor.device)
    s = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(tensor.device)
    return torch.clamp(tensor * s + m, 0, 1).permute(1, 2, 0).cpu().numpy()

def generate_visual_row(rgb, map_3d2d, map_2d3d, map_fused, gt_mask, name):
    plt.ioff()
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    rgb_img = denorm_rgb(rgb)
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"{name}\nInput RGB")
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    im2 = axes[2].imshow(map_3d2d, cmap='jet')
    axes[2].set_title("3D->2D (Real+0.1*Est)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(map_2d3d, cmap='jet')
    axes[3].set_title("2D->3D (Real+0.1*Est)")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    im4 = axes[4].imshow(map_fused, cmap='jet')
    axes[4].set_title("Combined (Box Blurred)")
    axes[4].axis('off')
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    plt.tight_layout()
    return fig

# =========================================================================
# 6. 处理逻辑
# =========================================================================
def resolve_checkpoint_dir(prefix, class_name, suffix):
    """Resolve clean checkpoint names and retain compatibility with older archives."""
    candidate = f"{prefix}_{class_name}{suffix}"
    if os.path.isdir(candidate):
        return candidate
    legacy_prefix = prefix.replace("checkpoints_eyecandies_dual_3dto2d", "checkpoints_eyecandies_dual_3dto2d_new3")
    legacy_prefix = legacy_prefix.replace("checkpoints_eyecandies_dual_2dto3d", "checkpoints_eyecandies_dual_2dto3d_new3")
    legacy_candidate = f"{legacy_prefix}_{class_name}{suffix}"
    if os.path.isdir(legacy_candidate):
        print(f"[{class_name}] Using legacy checkpoint directory: {legacy_candidate}")
        return legacy_candidate
    return candidate


def process_class(class_name, args, device):
    ckpt_dir_3d2d = resolve_checkpoint_dir(args.ckpt_root_3d2d, class_name, args.ckpt_suffix)
    ckpt_dir_2d3d = resolve_checkpoint_dir(args.ckpt_root_2d3d, class_name, args.ckpt_suffix)

    print(f"[{class_name}] Checking path: {ckpt_dir_3d2d}")

    if not os.path.exists(ckpt_dir_3d2d) or not os.path.exists(ckpt_dir_2d3d):
        print(f"[{class_name}] ERR: Checkpoint folder missing at {ckpt_dir_3d2d}. Skipping.")
        return None

    dataset = SingleClassDataset(
        args.dataset_root, args.real_normal_root, args.est_normal_root,
        class_name=class_name, mask_root=args.mask_root
    )
    if len(dataset) == 0:
        print(f"[{class_name}] ERR: Dataset empty. Skipping.")
        return None
    if args.max_samples > 0:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"[{class_name}] Limiting inference to {len(dataset)} samples")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    extractor = TripleLayerExtractor(device).to(device)

    model_real_3d2d = HierarchicalGatedMapper(embed_dim=768).to(device)
    model_est_3d2d = HierarchicalGatedMapper(embed_dim=768).to(device)
    try:
        model_real_3d2d.load_state_dict(torch.load(os.path.join(ckpt_dir_3d2d, "model_real_3d22d_final.pth"), map_location=device))
        model_est_3d2d.load_state_dict(torch.load(os.path.join(ckpt_dir_3d2d, "model_est_3d22d_final.pth"), map_location=device))
    except FileNotFoundError: return None
    model_real_3d2d.eval(); model_est_3d2d.eval()

    model_real_2d3d = HierarchicalGatedMapper(embed_dim=768).to(device)
    model_est_2d3d = HierarchicalGatedMapper(embed_dim=768).to(device)
    try:
        model_real_2d3d.load_state_dict(torch.load(os.path.join(ckpt_dir_2d3d, "model_real_2d23d_final.pth"), map_location=device))
        model_est_2d3d.load_state_dict(torch.load(os.path.join(ckpt_dir_2d3d, "model_est_2d23d_final.pth"), map_location=device))
    except FileNotFoundError: return None
    model_real_2d3d.eval(); model_est_2d3d.eval()

    # --- Box Filters Setup ---
    # 如果发现某些细长类（如 CandyCane, PeppermintCandy）效果不佳，可以把它们加入 small_kernel_classes
    small_kernel_classes = {
        "CandyCane", "ChocolateCookie", "ChocolatePraline",
        "LicoriceSandwich", "Lollipop", "Marshmallow", "PeppermintCandy",
        "GummyBear", "Confetto"
    }

    if class_name in small_kernel_classes:
        w_l, w_u = 3, 5
        pad_l, pad_u = 1, 2
        print(f"[{class_name}] Using Small Kernels: (3, 5)")
    else:
        # HazelnutTruffle 使用大核进行强平滑
        w_l, w_u = 5, 7
        pad_l, pad_u = 2, 3
        print(f"[{class_name}] Using Large Kernels: (5, 7)")

    weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u**2)

    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []
    gts_list, preds_list = [], []

    with torch.no_grad():
        for batch_idx, (rgb, real_normal, est_normal, gt_masks, names, labels, fg_masks) in enumerate(tqdm(dataloader, desc=f"Infer {class_name}")):
            rgb = rgb.to(device)
            real_normal = real_normal.to(device)
            est_normal = est_normal.to(device)
            fg_masks = fg_masks.to(device)

            rgb_feats = extractor(rgb)
            rgb_feats_flat = torch.cat(rgb_feats, dim=1).flatten(2).transpose(1, 2)
            real_norm_feats = extractor(real_normal)
            real_norm_flat = torch.cat(real_norm_feats, dim=1).flatten(2).transpose(1, 2)
            est_norm_feats = extractor(est_normal)

            pred_rgb_from_real, _, _ = model_real_3d2d(real_norm_feats)
            pred_rgb_from_est, _, _ = model_est_3d2d(est_norm_feats)

            pred_norm_from_rgb_real, _, _ = model_real_2d3d(rgb_feats)
            pred_norm_from_rgb_est, _, _ = model_est_2d3d(rgb_feats)

            for b in range(rgb.shape[0]):
                is_anomaly = labels[b].item()

                # --- 1. 计算余弦距离 ---
                dist_real_3d2d = compute_cosine_distance_torch(pred_rgb_from_real[b:b+1], rgb_feats_flat[b:b+1])
                dist_est_3d2d = compute_cosine_distance_torch(pred_rgb_from_est[b:b+1], rgb_feats_flat[b:b+1])

                dist_real_2d3d = compute_cosine_distance_torch(pred_norm_from_rgb_real[b:b+1], real_norm_flat[b:b+1])
                dist_est_2d3d = compute_cosine_distance_torch(pred_norm_from_rgb_est[b:b+1], real_norm_flat[b:b+1])

                # --- 2. 融合 real 和 est ---
                map_3d2d = dist_real_3d2d + 0.1 * dist_est_3d2d
                map_2d3d = dist_real_2d3d + 0.1 * dist_est_2d3d

                # --- 3. Mask 置零逻辑 ---
                bg_mask = (fg_masks[b:b+1] == 0) # 维度 (1, 1, 224, 224)
                map_3d2d[bg_mask] = 0.
                map_2d3d[bg_mask] = 0.

                # --- 4. 结合双向结果 ---
                map_comb = map_3d2d * map_2d3d
                map_comb[bg_mask] = 0.

                # --- 5. Box Filter 模糊 ---
                map_comb = F.conv2d(input=map_comb, padding=pad_l, weight=weight_l)
                map_comb = F.conv2d(input=map_comb, padding=pad_l, weight=weight_l)
                map_comb = F.conv2d(input=map_comb, padding=pad_l, weight=weight_l)
                map_comb = F.conv2d(input=map_comb, padding=pad_l, weight=weight_l)
                map_comb = F.conv2d(input=map_comb, padding=pad_l, weight=weight_l)

                map_comb = F.conv2d(input=map_comb, padding=pad_u, weight=weight_u)
                map_comb = F.conv2d(input=map_comb, padding=pad_u, weight=weight_u)
                map_comb = F.conv2d(input=map_comb, padding=pad_u, weight=weight_u)

                map_comb_eval = map_comb.squeeze() # (224, 224)
                gt_eval = gt_masks[b].squeeze().numpy()

                # --- 6. 收集评价指标所需数据 ---
                gts_list.append(gt_eval)

                # 针对 PRO
                mean_val_pro = map_comb_eval[map_comb_eval!=0].mean() if map_comb_eval[map_comb_eval!=0].numel() > 0 else torch.tensor(1.0, device=device)
                preds_list.append((map_comb_eval / mean_val_pro).cpu().detach().numpy())

                # 针对 AUROC
                image_labels.append(is_anomaly)
                pixel_labels.extend(gt_eval.flatten())

                mean_val_img = torch.sqrt(map_comb_eval[map_comb_eval!=0].mean()) if map_comb_eval[map_comb_eval!=0].numel() > 0 else torch.tensor(1.0, device=device)
                image_preds.append((map_comb_eval / mean_val_img).max().item())

                mean_val_pix = torch.sqrt(map_comb_eval.mean()) if map_comb_eval.numel() > 0 else torch.tensor(1.0, device=device)
                pixel_preds.extend((map_comb_eval / mean_val_pix).flatten().cpu().detach().numpy())

                # WandB Visualization
                global_idx = batch_idx * args.batch_size + b
                if wandb.run is not None and (global_idx % 40 == 0):
                    fig = generate_visual_row(
                        rgb[b],
                        map_3d2d.squeeze().cpu().numpy(),
                        map_2d3d.squeeze().cpu().numpy(),
                        map_comb_eval.cpu().numpy(),
                        gt_eval,
                        names[b]
                    )
                    wandb.log({f"{class_name}_Vis/{names[b]}": wandb.Image(fig)})
                    plt.close(fig)

    # --- 计算 Metrics ---
    try:
        if len(np.unique(image_labels)) < 2: image_auroc = 0.5
        else: image_auroc = roc_auc_score(image_labels, image_preds)
    except: image_auroc = 0.0

    try:
        pixel_auroc = roc_auc_score(pixel_labels, pixel_preds)
    except: pixel_auroc = 0.0

    au_pros = [0.0]*4
    if calculate_au_pro is not None:
        try:
            au_pros, _ = calculate_au_pro(gts_list, preds_list)
            if not isinstance(au_pros, list): au_pros = [au_pros]
            while len(au_pros) < 4: au_pros.append(0.0)
        except Exception as e:
            print(f"Error calculating AUPRO for {class_name}: {e}")

    # 清理显存
    del model_real_3d2d, model_est_3d2d, model_real_2d3d, model_est_2d3d, extractor
    del dataloader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "class": class_name,
        "I-AUROC": image_auroc,
        "P-AUROC": pixel_auroc,
        "AUPRO_30": au_pros[0],
        "AUPRO_10": au_pros[1],
        "AUPRO_05": au_pros[2],
        "AUPRO_01": au_pros[3]
    }

# =========================================================================
# 7. 主流程
# =========================================================================
def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CMDS-AD inference, but it is not available")
    device = "cuda"

    EYECANDIES_CLASSES = [
        "CandyCane", "ChocolateCookie", "ChocolatePraline",
        "GummyBear", "HazelnutTruffle", "LicoriceSandwich",
        "Lollipop", "Confetto", "Marshmallow", "PeppermintCandy"
    ]

    wandb.init(
        project=args.wandb_project,
        name="Direct_BoxFilter_Integration_Eyecandies",
        config=vars(args),
        mode=args.wandb_mode,
    )

    final_results = []

    print("=============================================")
    print("STARTING INFERENCE FOR EYECANDIES")
    print("=============================================")

    classes = args.classes or EYECANDIES_CLASSES
    for cls in classes:
        print(f"\n>>> Processing Class: {cls}")
        metrics = process_class(cls, args, device)

        if metrics:
            final_results.append(metrics)

            print(f"Finished {cls} Detailed Metrics:")
            print(f"  I-AUROC   : {metrics['I-AUROC']:.4f}")
            print(f"  P-AUROC   : {metrics['P-AUROC']:.4f}")
            print(f"  AUPRO 30% : {metrics['AUPRO_30']:.4f}")
            print(f"  AUPRO 10% : {metrics['AUPRO_10']:.4f}")
            print(f"  AUPRO 5%  : {metrics['AUPRO_05']:.4f}")
            print(f"  AUPRO 1%  : {metrics['AUPRO_01']:.4f}")
            print("-" * 40)

            wandb.log({
                f"{cls}/I-AUROC": metrics['I-AUROC'],
                f"{cls}/P-AUROC": metrics['P-AUROC'],
                f"{cls}/AUPRO_30": metrics['AUPRO_30'],
                f"{cls}/AUPRO_10": metrics['AUPRO_10'],
                f"{cls}/AUPRO_05": metrics['AUPRO_05'],
                f"{cls}/AUPRO_01": metrics['AUPRO_01']
            })

    if len(final_results) > 0:
        print("\n\n")
        print("="*105)
        print(f"{'Class':<17} | {'I-AUROC':<10} | {'P-AUROC':<10} | {'PRO 30%':<10} | {'PRO 10%':<10} | {'PRO 5%':<10} | {'PRO 1%':<10}")
        print("-" * 105)

        sums = {k: 0.0 for k in final_results[0].keys() if k != "class"}

        for res in final_results:
            print(f"{res['class']:<17} | {res['I-AUROC']:.4f}     | {res['P-AUROC']:.4f}     | {res['AUPRO_30']:.4f}     | {res['AUPRO_10']:.4f}     | {res['AUPRO_05']:.4f}     | {res['AUPRO_01']:.4f}")
            for k in sums.keys():
                sums[k] += res[k]

        print("-" * 105)

        n = len(final_results)
        print(f"{'MEAN':<17} | {sums['I-AUROC']/n:.4f}     | {sums['P-AUROC']/n:.4f}     | {sums['AUPRO_30']/n:.4f}     | {sums['AUPRO_10']/n:.4f}     | {sums['AUPRO_05']/n:.4f}     | {sums['AUPRO_01']/n:.4f}")
        print("="*105)

        wandb.log({
            "MEAN/I-AUROC": sums['I-AUROC']/n,
            "MEAN/P-AUROC": sums['P-AUROC']/n,
            "MEAN/AUPRO_30": sums['AUPRO_30']/n,
            "MEAN/AUPRO_10": sums['AUPRO_10']/n,
            "MEAN/AUPRO_05": sums['AUPRO_05']/n,
            "MEAN/AUPRO_01": sums['AUPRO_01']/n
        })
    else:
        print("No results obtained.")

    wandb.finish()

if __name__ == "__main__":
    main()
