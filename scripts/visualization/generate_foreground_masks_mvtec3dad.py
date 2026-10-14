"""Generate foreground masks for MVTec 3D-AD RGB/XYZ pairs."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import os
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import tifffile as tiff
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.features import MultimodalFeatures

# ==========================================

# ==========================================
class SquarePadResize:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: Tensor [C, H, W]
        Returns:
            Tensor [C, target_size, target_size]
        """
        c, h, w = img_tensor.shape
        max_dim = max(h, w)


        diff_h = max_dim - h
        diff_w = max_dim - w



        pad_left = diff_w // 2
        pad_right = diff_w - pad_left
        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top

        padding = (pad_left, pad_right, pad_top, pad_bottom)


        padded_img = F.pad(img_tensor, padding, value=0)

        # 2. Resize

        resized_img = F.interpolate(
            padded_img.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return resized_img

# ==========================================

# ==========================================
class SimpleMVTec3DDataset(Dataset):
    def __init__(self, root_dir, class_name, split, img_size=224):
        self.img_size = img_size
        self.split = split
        self.class_name = class_name

        self.base_path = os.path.join(root_dir, class_name, split)
        search_pattern = os.path.join(self.base_path, "**", "rgb", "*.png")
        self.rgb_files = glob.glob(search_pattern, recursive=True)
        self.rgb_files.sort()


        self.square_pad_resize = SquarePadResize(target_size=img_size)


        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        xyz_path = rgb_path.replace("rgb", "xyz").replace(".png", ".tiff")



        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_tensor = transforms.functional.to_tensor(rgb_img) # [3, H, W], 0-1


        rgb_tensor = self.square_pad_resize(rgb_tensor)



        rgb_tensor = self.normalize(rgb_tensor)


        if os.path.exists(xyz_path):
            xyz_np = tiff.imread(xyz_path) # [H, W, 3]
            xyz_tensor = transforms.functional.to_tensor(xyz_np) # [3, H, W]

            xyz_tensor = self.square_pad_resize(xyz_tensor)
        else:
            xyz_tensor = torch.zeros_like(rgb_tensor)


        parts = rgb_path.split(os.sep)
        filename = parts[-1]
        defect_type = parts[-3]

        return rgb_tensor, xyz_tensor, self.split, defect_type, filename

# ==========================================

# ==========================================
def process_single_class(args, class_name, feature_extractor, device):
    splits = ['train', 'test']

    for split in splits:
        dataset = SimpleMVTec3DDataset(root_dir=args.dataset_path,
                                       class_name=class_name,
                                       split=split,
                                       img_size=224)

        if len(dataset) == 0:
            print(f"  [!] Skipping {class_name}/{split} (No data found).")
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        desc = f"Processing {class_name} [{split}]"
        for batch_idx, (rgb, pc, split_name, defect_types, filenames) in tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False):

            rgb = rgb.to(device)
            pc = pc.to(device)

            # -------------------------------------------------

            # -------------------------------------------------
            if rgb.shape[0] == 1:
                _, xyz_patch = feature_extractor.get_features_maps(rgb, pc)


                if xyz_patch.dim() == 2:
                    xyz_patch = xyz_patch.unsqueeze(0)
            else:
                xyz_patches = []
                for i in range(rgb.shape[0]):
                    _, x_p = feature_extractor.get_features_maps(rgb[i].unsqueeze(0), pc[i].unsqueeze(0))

                    if x_p.dim() == 3 and x_p.shape[0] == 1:
                        x_p = x_p.squeeze(0)
                    xyz_patches.append(x_p)
                xyz_patch = torch.stack(xyz_patches, dim=0) # [B, N, C]

            # -------------------------------------------------

            # -------------------------------------------------


            is_foreground = ~(xyz_patch.sum(dim=-1) == 0) # [B, N_patches]

            # -------------------------------------------------

            # -------------------------------------------------
            curr_batch_size = rgb.shape[0]

            for i in range(curr_batch_size):
                curr_mask = is_foreground[i]


                if curr_mask.dim() == 0:
                    print(f"[Error] Mask became scalar for {filenames[i]}. Skipping.")
                    continue

                curr_defect = defect_types[i]
                curr_filename = filenames[i]


                n_patches = curr_mask.shape[0]
                side_dim = int(np.sqrt(n_patches))

                # Reshape: [1, 1, H_feat, W_feat]
                viz_mask = curr_mask.view(side_dim, side_dim).float().unsqueeze(0).unsqueeze(0)



                viz_mask_resized = F.interpolate(viz_mask, size=(224, 224), mode='nearest')
                viz_mask_resized = viz_mask_resized.squeeze().cpu().numpy()


                mask_img_np = (viz_mask_resized * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_img_np)


                save_folder = os.path.join(args.save_dir, class_name, split, curr_defect)
                os.makedirs(save_folder, exist_ok=True)

                save_path = os.path.join(save_folder, curr_filename)
                mask_img.save(save_path)

# ==========================================

# ==========================================
def run_all_classes(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for foreground-mask extraction")
    device = "cuda"
    print(f"[*] Device: {device}")

    mvtec_3d_classes = [
        "bagel", "cable_gland", "carrot", "cookie", "dowel",
        "foam", "peach", "potato", "rope", "tire"
    ]

    target_classes = []
    if args.class_name.lower() == 'all':
        target_classes = mvtec_3d_classes
    else:
        target_classes = [args.class_name]

    print("[*] Loading Feature Extractor...")
    feature_extractor = MultimodalFeatures()

    print(f"[*] Output dir: {os.path.abspath(args.save_dir)}")

    for i, cls in enumerate(target_classes):
        print(f"\n[{i+1}/{len(target_classes)}] Processing class: {cls.upper()}")
        try:
            process_single_class(args, cls, feature_extractor, device)
        except Exception as e:
            print(f"[!] Error processing class {cls}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[*] All tasks finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data/derived/mvtec_3d', type=str)
    parser.add_argument('--save_dir', default='./data/derived/mvtec_3d_masks_generated', type=str)
    parser.add_argument('--class_name', type=str, default='all')
    parser.add_argument('--batch_size', default=8, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
    else:
        run_all_classes(args)
