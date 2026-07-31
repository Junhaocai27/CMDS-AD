import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import itertools
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import wandb
import time
from sklearn.decomposition import PCA

import scipy.ndimage

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

def get_args():
    parser = argparse.ArgumentParser(description="Dual Stream SFF Training (RGB->Normal) Real Masked, Est Global")
    parser.add_argument("--class_name", type=str, default="rope", help="Object category")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per stream (use 1 on a 24 GB GPU)")
    parser.add_argument("--max_steps", type=int, default=3000, help="Fixed number of training steps")
    parser.add_argument("--shots", type=int, default=10, help="Number of real images to select (e.g., 5-shot)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rgb_root", type=str, default="./data/derived/output_generation_full")
    parser.add_argument("--real_normal_root", type=str, default="./data/derived/normal_output_train_new_full/real_normals")
    parser.add_argument("--est_normal_root", type=str, default="./data/derived/normal_output_train_new_full/estimated_normals")
    parser.add_argument("--mask_root", type=str, default="./data/derived/mvtec_3d_masks_generated")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default=os.getenv("WANDB_MODE", "disabled"))
    return parser.parse_args()

args = get_args()
console = Console()

ROOT_RGB = args.rgb_root
ROOT_REAL = args.real_normal_root
ROOT_EST = args.est_normal_root
ROOT_MASK = args.mask_root

CONFIG = {
    "class_name": args.class_name,
    "exp_name": f"DualStream_2Dto3D_{args.class_name}_{args.shots}shot_RealMasked_EstGlobal",
    "rgb_dir": os.path.join(ROOT_RGB, args.class_name),
    "real_normal_dir": os.path.join(ROOT_REAL, args.class_name, "train/good"),
    "est_normal_dir": os.path.join(ROOT_EST, args.class_name, "train/good/normals_vis"),
    "mask_dir": os.path.join(ROOT_MASK, args.class_name, "train/good"),
    "ckpt_dir": os.path.join(args.checkpoint_root, f"checkpoints_dual_2dto3d_{args.class_name}_{args.shots}shot"),
    "batch_size": args.batch_size,
    "max_steps": args.max_steps,
    "shots": args.shots,
    "lr": args.lr,
    "target_layers": [4, 7, 11],
    "wandb_project": "CFM_DualStream_Fixed",
    "wandb_mode": args.wandb_mode,
    "num_workers": args.num_workers,
}

console.print(f"[bold yellow]Starting Training (2D->3D) for Class:[/bold yellow] [green]{CONFIG['class_name']}[/green] | [cyan]{CONFIG['shots']}-shot[/cyan]")

# =========================================================================
# 2. Dataset (Modified: With Mask & Few-Shot Logic)
#    Input: RGB (Normalized)
#    Target: Normal (Real / Est)
# =========================================================================
class SingleStreamDataset(Dataset):
    def __init__(self, rgb_dir, real_dir, est_dir, mask_dir, mode='real', img_size=224, shots=5):
        super().__init__()
        self.mode = mode
        self.real_dir = real_dir
        self.est_dir = est_dir
        self.mask_dir = mask_dir
        self.rgb_root = rgb_dir
        self.img_size = img_size

        all_files = os.listdir(rgb_dir)
        unique_stems = sorted(list(set([
            f.split('.')[0] for f in all_files
            if "_seed" not in f and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])))

        if shots > 0 and shots < len(unique_stems):
            unique_stems = unique_stems[:shots]

        self.file_list = []
        seeds = ["42", "1024", "2023", "8888", "12345"]

        for stem in unique_stems:
            if mode == 'real':
                fname = f"{stem}.png"
                if os.path.exists(os.path.join(rgb_dir, fname)):
                    self.file_list.append(fname)
            elif mode == 'all':
                fname_real = f"{stem}.png"
                if os.path.exists(os.path.join(rgb_dir, fname_real)):
                    self.file_list.append(fname_real)
                for s in seeds:
                    fname_gen = f"{stem}_seed{s}.png"
                    if os.path.exists(os.path.join(rgb_dir, fname_gen)):
                        self.file_list.append(fname_gen)

        # Input (RGB): Normalized
        self.tf_rgb = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Target (Normal) & Mask: Raw [0, 1]
        self.tf_norm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        console.print(f"[{mode.upper()} Dataset] Loaded {len(self.file_list)} images for {shots}-shot training.")

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        name_stem = os.path.splitext(fname)[0]
        base_stem = name_stem.split('_seed')[0]

        # 1. Load RGB (Input)
        rgb_path = os.path.join(self.rgb_root, fname)
        rgb_pil = Image.open(rgb_path).convert('RGB')
        rgb = self.tf_rgb(rgb_pil)

        # 2. Load Mask (Only really used for REAL mode, return dummy for EST)
        mask = torch.ones((1, self.img_size, self.img_size))
        if self.mode == 'real':
            mask_path = os.path.join(self.mask_dir, f"{base_stem}.png")
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path).convert('L')
                mask = transforms.ToTensor()(transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST)(mask_pil))

        # 3. Pre-load Normals (Targets)
        real_gt_path = os.path.join(self.real_dir, f"{base_stem}.png")
        has_real_normal = os.path.exists(real_gt_path)
        real_normal_pil = Image.open(real_gt_path).convert('RGB') if has_real_normal else None

        est_path = os.path.join(self.est_dir, f"{name_stem}_normals.png")
        has_est_normal = os.path.exists(est_path)
        est_normal_pil = Image.open(est_path).convert('RGB') if has_est_normal else None

        target_tensor = None

        # --- A. Real Stream: Input=RGB, Target=Real Normal ---
        if self.mode == 'real':
            if has_real_normal:
                target_tensor = self.tf_norm(real_normal_pil)
            else:
                target_tensor = torch.zeros(3, self.img_size, self.img_size)

        # --- B. Est/All Stream: Input=RGB, Target=Est Normal ---
        else:
            if has_est_normal:
                target_tensor = self.tf_norm(est_normal_pil)
            else:
                target_tensor = torch.zeros(3, self.img_size, self.img_size)

        # 返回: (input=RGB, target=Normal, mask)
        return rgb, target_tensor, mask

# =========================================================================
# 3. Models (Unchanged)
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
            feat_map_up = F.interpolate(feat_map, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            layer_outputs.append(feat_map_up)
        return layer_outputs

class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

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
        return out_flat, w4

def add_input_noise(img_tensor, noise_level=0.01):
    if noise_level <= 0: return img_tensor
    noise = torch.randn_like(img_tensor) * noise_level
    return img_tensor + noise

# =========================================================================
# 4. Loss Function (Modified: Optional Mask)
# =========================================================================
def decoupled_multiscale_loss(pred_flat, target_flat, mask=None):
    """
    If mask is provided (Real Stream), compute loss only in masked area.
    If mask is None (Est Stream), compute global mean loss.
    """
    pred_chunks = torch.chunk(pred_flat, 3, dim=-1)
    target_chunks = torch.chunk(target_flat, 3, dim=-1)

    losses = {}
    total_loss = 0
    weights = [1.2, 1.0, 0.8]
    names = ["L4_HF", "L7_MF", "L11_LF"]

    lambda_l2 = 0.0

    for i in range(3):
        p_norm = F.normalize(pred_chunks[i], p=2, dim=-1)
        t_norm = F.normalize(target_chunks[i], p=2, dim=-1)
        cos_sim = F.cosine_similarity(p_norm, t_norm, dim=-1)
        loss_cos = 1 - cos_sim # (B, N)

        mse = (pred_chunks[i] - target_chunks[i]) ** 2
        loss_l2 = mse.mean(dim=-1) # (B, N)

        combined_pixel_loss = loss_cos + lambda_l2 * loss_l2

        # Mask Applied condition
        if mask is not None:
            mask_flat = mask.flatten(1) # (B, N)
            valid_pixels = mask_flat.sum(dim=1) + 1e-6 # (B,)
            layer_loss = (combined_pixel_loss * mask_flat).sum(dim=1) / valid_pixels
        else:
            layer_loss = combined_pixel_loss.mean(dim=1) # (B,)

        layer_loss = layer_loss.mean() # Batch Mean

        total_loss += layer_loss * weights[i]
        losses[f"{names[i]}"] = layer_loss.item()

    return total_loss, losses

# =========================================================================
# 5. Visuals (Modified: Included Mask)
# =========================================================================
def denorm_rgb(tensor):
    m = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(tensor.device)
    s = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(tensor.device)
    return torch.clamp(tensor * s + m, 0, 1).permute(1, 2, 0).cpu().numpy()

def simple_permute(tensor):
    return torch.clamp(tensor, 0, 1).permute(1, 2, 0).cpu().numpy()

def compute_pca_vis(feat_BNC):
    B, N, C = feat_BNC.shape
    flat = feat_BNC.detach().reshape(-1, C).cpu().numpy()
    pca = PCA(n_components=3)
    if flat.shape[0] > 2000:
        idx = np.random.choice(flat.shape[0], 2000, replace=False)
        pca.fit(flat[idx])
    else:
        pca.fit(flat)
    pca_f = pca.transform(flat)
    _min, _max = pca_f.min(0), pca_f.max(0)
    pca_f = (pca_f - _min) / (_max - _min + 1e-6)
    return pca_f.reshape(B, 224, 224, 3)

def log_dual_visuals(
        real_input_rgb, real_target_norm, real_mask, real_target_feat, real_pred_feat,
        est_input_rgb, est_target_norm, est_mask, est_target_feat, est_pred_feat,
        step
    ):

    idx = 0

    def norm_f(x): return F.normalize(x, p=2, dim=-1)

    feats = torch.cat([
        norm_f(real_target_feat[idx:idx+1]),
        norm_f(real_pred_feat[idx:idx+1]),
        norm_f(est_target_feat[idx:idx+1]),
        norm_f(est_pred_feat[idx:idx+1])
    ], dim=0)

    pca_imgs = compute_pca_vis(feats)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # --- Row 1: Real Stream (Input: RGB, Target: Real Normal) ---
    axes[0,0].imshow(denorm_rgb(real_input_rgb[idx]))
    axes[0,0].set_title("Input: RGB")
    axes[0,1].imshow(simple_permute(real_target_norm[idx]))
    axes[0,1].set_title("Target: Real Normal")
    if real_mask is not None:
        axes[0,2].imshow(real_mask[idx].squeeze().cpu().numpy(), cmap='gray')
        axes[0,2].set_title("Mask (Real)")
    else:
        axes[0,2].axis('off')
    axes[0,3].axis('off')

    # --- Row 2: PCA Real ---
    axes[1,0].imshow(pca_imgs[0])
    axes[1,0].set_title("Target Real Feat")
    axes[1,1].imshow(pca_imgs[1])
    axes[1,1].set_title("Pred Real Model")

    # --- Extra: Est Stream Images ---
    axes[1,2].imshow(denorm_rgb(est_input_rgb[idx]))
    axes[1,2].set_title("Input: RGB")
    axes[1,3].imshow(simple_permute(est_target_norm[idx]))
    axes[1,3].set_title("Target: Est Normal")

    # --- Row 3: PCA Est ---
    axes[2,0].imshow(pca_imgs[2])
    axes[2,0].set_title("Target Est Feat")
    axes[2,1].imshow(pca_imgs[3])
    axes[2,1].set_title("Pred Est Model")

    if est_mask is not None:
        axes[2,2].imshow(est_mask[idx].squeeze().cpu().numpy(), cmap='gray')
        axes[2,2].set_title("Mask (Est)")
    else:
        axes[2,2].axis('off')
        axes[2,2].set_title("No Mask (Global)")

    axes[2,3].axis('off')

    plt.tight_layout()
    wandb.log({f"Visuals/Comparison_MixedMask": wandb.Image(fig)}, step=step)
    plt.close(fig)

# =========================================================================
# 6. Training Loop (Fixed Steps)
# =========================================================================
def cycle_loader(dl):
    while True:
        for batch in dl:
            yield batch

def resolve_device(requested):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CMDS-AD training, but it is not available")
    return "cuda"

def run_training():
    device = resolve_device(args.device)
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)

    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["exp_name"], config=CONFIG, mode=CONFIG["wandb_mode"])

    # 1. Dataset
    ds_real = SingleStreamDataset(
        CONFIG["rgb_dir"], CONFIG["real_normal_dir"], CONFIG["est_normal_dir"], CONFIG["mask_dir"],
        mode='real', shots=CONFIG["shots"]
    )
    ds_est_all = SingleStreamDataset(
        CONFIG["rgb_dir"], CONFIG["real_normal_dir"], CONFIG["est_normal_dir"], CONFIG["mask_dir"],
        mode='all', shots=CONFIG["shots"]
    )

    # 2. Dataloader
    dl_real = DataLoader(ds_real, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], drop_last=(len(ds_real)>CONFIG["batch_size"]))
    dl_est_all = DataLoader(ds_est_all, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], drop_last=True)

    iter_real = iter(cycle_loader(dl_real))
    iter_est = iter(cycle_loader(dl_est_all))

    extractor = TripleLayerExtractor(device, target_layers=CONFIG["target_layers"])
    # model_real: RGB -> Real Normal features
    model_real = HierarchicalGatedMapper(embed_dim=768).to(device)
    # model_est: RGB -> Est Normal features
    model_est = HierarchicalGatedMapper(embed_dim=768).to(device)

    optimizer = torch.optim.AdamW([
        {'params': model_real.parameters()},
        {'params': model_est.parameters()}
    ], lr=CONFIG["lr"], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["max_steps"], eta_min=1e-6)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}"),
        console=console
    )

    with progress:
        task = progress.add_task(f"[green]Training {CONFIG['class_name']} (2D->3D)...", total=CONFIG["max_steps"], info="Init")

        model_real.train()
        model_est.train()

        accum_loss_real = 0
        accum_loss_est = 0

        for step in range(1, CONFIG["max_steps"] + 1):

            # --- Load Data ---
            input_rgb_real, target_norm_real, mask_real = next(iter_real)
            input_rgb_est, target_norm_est, _ = next(iter_est) # 丢弃 est 的 dummy mask

            input_rgb_real = input_rgb_real.to(device)
            target_norm_real = target_norm_real.to(device)
            mask_real = mask_real.to(device)

            input_rgb_est = input_rgb_est.to(device)
            target_norm_est = target_norm_est.to(device)

            input_rgb_real_noisy = add_input_noise(input_rgb_real)
            input_rgb_est_noisy = add_input_noise(input_rgb_est)

            with torch.no_grad():
                # Extract Target (Normal) features
                real_gt_feats = extractor(target_norm_real)
                target_real_flat = torch.cat(real_gt_feats, dim=1).flatten(2).transpose(1, 2)

                est_gt_feats = extractor(target_norm_est)
                target_est_flat = torch.cat(est_gt_feats, dim=1).flatten(2).transpose(1, 2)

                # Extract Input (RGB) features
                rgb_real_feats = extractor(input_rgb_real_noisy)
                rgb_est_feats = extractor(input_rgb_est_noisy)

            # --- Forward ---
            # 1. Real Stream (使用 Mask)
            pred_real, _ = model_real(rgb_real_feats)
            loss_real, dict_real = decoupled_multiscale_loss(pred_real, target_real_flat, mask=mask_real)

            # 2. Est Stream (传入 None，全局无 Mask)
            pred_est, _ = model_est(rgb_est_feats)
            loss_est, dict_est = decoupled_multiscale_loss(pred_est, target_est_flat, mask=None)

            total_loss = loss_real + loss_est

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            accum_loss_real += loss_real.item()
            accum_loss_est += loss_est.item()

            if step % 10 == 0:
                log_dict = {
                    "loss/total": total_loss.item(),
                    "loss/stream_real_masked": loss_real.item(),
                    "loss/stream_est_global": loss_est.item(),
                    "lr": optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict, step=step)

            if step % 200 == 0:
                log_dual_visuals(
                    input_rgb_real, target_norm_real, mask_real, target_real_flat, pred_real,
                    input_rgb_est, target_norm_est, None, target_est_flat, pred_est, # est_mask传入None
                    step
                )

            if step % 10 == 0:
                avg_real = accum_loss_real / 10
                avg_est = accum_loss_est / 10
                progress.update(task, advance=10, info=f"L_R(Mask): {avg_real:.4f} | L_E(Glob): {avg_est:.4f}")
                accum_loss_real = 0
                accum_loss_est = 0

            # 按步数定期保存
            if step % 1000 == 0:
                torch.save(model_real.state_dict(), os.path.join(CONFIG["ckpt_dir"], f"model_real_2d23d_step{step}.pth"))
                torch.save(model_est.state_dict(), os.path.join(CONFIG["ckpt_dir"], f"model_est_2d23d_step{step}.pth"))

    # 最终保存
    torch.save(model_real.state_dict(), os.path.join(CONFIG["ckpt_dir"], "model_real_2d23d_final.pth"))
    torch.save(model_est.state_dict(), os.path.join(CONFIG["ckpt_dir"], "model_est_2d23d_final.pth"))

    wandb.finish()
    console.print(f"[bold green]Training (2D->3D) Completed![/bold green]")

if __name__ == "__main__":
    run_training()
