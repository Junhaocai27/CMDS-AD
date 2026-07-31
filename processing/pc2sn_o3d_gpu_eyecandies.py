import os

# ================= 新增：解决 Intel MKL 多进程冲突 (解决 FATAL ERROR) =================
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# ====================================================================================

# 设置 Open3D 头部渲染模式 (必须在导入 open3d 之前)
os.environ["O3D_HEADLESS_RENDERING"] = "1"

import open3d as o3d
import open3d.core as o3c
from shutil import copyfile
import cv2
import numpy as np
import tifffile
import yaml
import imageio.v3 as iio
import math
import argparse
import sys

FOCAL_LENGTH = 711.11

# =========================================================================
#  Part 1: 基础数据读取 (保持不变)
# =========================================================================

def load_and_convert_depth(depth_img, info_depth):
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]
    dimg = iio.imread(depth_img).astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg

def depth_to_pointcloud_raw(depth_img, info_depth, pose_txt, focal_length):
    depth_mt = load_and_convert_depth(depth_img, info_depth)
    pose = np.loadtxt(pose_txt)
    height, width = depth_mt.shape[:2]

    intrinsics_4x4 = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )
    camera_proj = intrinsics_4x4 @ pose

    camera_vectors = np.zeros((width * height, 4))
    count=0
    for j in range(height):
        for i in range(width):
            camera_vectors[count, :] = np.array([i, j, 1, 1/depth_mt[j, i]])
            count += 1

    hom_3d_pts= np.linalg.inv(camera_proj) @ camera_vectors.T
    pcd = depth_mt.reshape(-1, 1) * hom_3d_pts.T

    pose_4x4 = np.eye(4)
    if pose.shape == (3, 4):
        pose_4x4[:3, :4] = pose
    elif pose.shape == (4, 4):
        pose_4x4 = pose
    camera_center = np.linalg.inv(pose_4x4)[:3, 3]

    return pcd[:, :3], camera_center

# =========================================================================
#  Part 2: 核心修复 - 分流处理函数 (保持不变)
# =========================================================================

def process_point_cloud_split_paths(pc_raw):

    dz = pc_raw[256,1] - pc_raw[-256,1]
    dy = pc_raw[256,2] - pc_raw[-256,2]
    norm = math.sqrt(dz**2 + dy**2)

    start_points = np.array([0, pc_raw[-256,1], pc_raw[-256,2]])

    cos_theta = dy / norm
    sin_theta = dz / norm

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])

    pc_rotated = (rotation_matrix @ (pc_raw - start_points).T).T

    bg_mask = (
        (pc_rotated[:,1] > -0.02) |
        (pc_rotated[:,2] > 1.8) |
        (pc_rotated[:,0] > 1) |
        (pc_rotated[:,0] < -1)
    )

    pc_distorted = pc_rotated.copy()
    pc_distorted[bg_mask] = 0

    pc_distorted = (rotation_matrix.T @ pc_distorted.T).T + start_points
    pc_distorted = pc_distorted[:, [0,2,1]] * [0.1,-0.1,0.1]

    pc_physical_clean = pc_raw.copy()
    pc_physical_clean[bg_mask] = 0

    return pc_distorted, pc_physical_clean

# =========================================================================
#  Part 3: 法向量计算
# =========================================================================

def connected_components_cleaning_cpu(organized_pc):
    H, W, C = organized_pc.shape
    flat_pc = organized_pc.reshape(-1, 3)
    nonzero_indices = np.nonzero(np.all(flat_pc != 0, axis=1))[0]

    if len(nonzero_indices) == 0: return organized_pc
    points_nonzero = flat_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_nonzero))

    labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))

    if len(labels) == 0: return organized_pc
    unique_cluster_ids, cluster_size = np.unique(labels, return_counts=True)
    largest_cluster_id = unique_cluster_ids[np.argmax(cluster_size)]

    flat_pc_clean = flat_pc.copy()
    flat_pc_clean[nonzero_indices[labels != largest_cluster_id]] = 0
    return flat_pc_clean.reshape(H, W, C)

def compute_and_save_normal(pc_physical_clean_flat, camera_center, output_path, device, radius=0.04, max_nn=30, target_size=(224, 224)):
    H, W = 512, 512

    xyz_clean_3d = pc_physical_clean_flat.reshape(H, W, 3)
    xyz_clean_3d = connected_components_cleaning_cpu(xyz_clean_3d)
    points_flat_clean = xyz_clean_3d.reshape(-1, 3).astype(np.float32)

    # 彻底移除了腐蚀操作 (cv2.erode)
    final_bg_mask = np.all(np.isclose(points_flat_clean, 0), axis=1)
    fg_mask = ~final_bg_mask

    fg_indices = np.where(fg_mask)[0]
    fg_points = points_flat_clean[fg_indices]

    if len(fg_points) == 0:
        return

    try:
        pcd_t = o3d.t.geometry.PointCloud(device)
        pcd_t.point.positions = o3c.Tensor(fg_points, device=device)
        pcd_t.estimate_normals(max_nn=max_nn, radius=radius)

        center = o3c.Tensor(camera_center, dtype=o3c.float32, device=device)
        pcd_t.orient_normals_towards_camera_location(camera_location=center)

        fg_normals = pcd_t.point.normals.cpu().numpy()
    except RuntimeError as e:
        print(f"Open3D Error: {e}")
        return

    fg_normals = -fg_normals
    fg_normals = fg_normals[:, [0,2,1]]
    fg_normals[:,1] *= -1

    normals_flat = np.full((H * W, 3), [0.0, 0.0, 1.0], dtype=np.float32)
    normals_flat[fg_indices] = fg_normals

    normals_grid = normals_flat.reshape(H, W, 3)
    rgb_uint8 = (np.clip((normals_grid + 1.0) / 2.0, 0.0, 1.0) * 255).astype(np.uint8)

    H_curr, W_curr = rgb_uint8.shape[:2]
    if H_curr != W_curr:
        max_side = max(H_curr, W_curr)
        top = (max_side - H_curr) // 2
        bottom = max_side - H_curr - top
        left = (max_side - W_curr) // 2
        right = max_side - W_curr - left
        rgb_uint8 = cv2.copyMakeBorder(rgb_uint8, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127, 127, 255))

    if target_size is not None:
        rgb_uint8 = cv2.resize(rgb_uint8, target_size, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(output_path, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

# =========================================================================
#  主调度流程 (根据测试代码修改为完整类别处理)
# =========================================================================
def make_mvtec_dirs(base_path, category, split, condition):
    paths = {
        'rgb': os.path.join(base_path, category, split, condition, 'rgb'),
        'xyz': os.path.join(base_path, category, split, condition, 'xyz'),
        'normals': os.path.join(base_path, category, split, condition, 'normals')
    }
    if split == 'test':
        paths['gt'] = os.path.join(base_path, category, split, condition, 'gt')
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def main():
    parser = argparse.ArgumentParser(description='Eyecandies to MVTec 3D-AD (Full Class Processing)')
    parser.add_argument('--dataset_path', required=False, default="./data/raw/Eyecandies", type=str, help="Eyecandies Dataset Root")
    parser.add_argument('--target_dir', required=False, default="./data/derived/eyecandies_mvtec_format", type=str, help="Output Root Directory")
    parser.add_argument('--category', required=False, default="CandyCane", type=str, help="Category name to process")
    args = parser.parse_args()

    print("="*50)
    print(f"🚀 启动生产模式 - 正在处理类别: {args.category}")
    print("="*50)

    if not o3c.cuda.is_available():
        print("错误: GPU 不可用。")
        sys.exit(1)
    device = o3c.Device("CUDA:0")

    category_dir = args.category
    category_root_path = os.path.join(args.dataset_path, category_dir)
    category_train_path = os.path.join(category_root_path, 'train/data')
    category_test_path = os.path.join(category_root_path, 'test_public/data')

    train_good_dirs = make_mvtec_dirs(args.target_dir, category_dir, 'train', 'good')
    test_good_dirs = make_mvtec_dirs(args.target_dir, category_dir, 'test', 'good')
    test_bad_dirs = make_mvtec_dirs(args.target_dir, category_dir, 'test', 'bad')

    # === 处理训练集 (遍历全集) ===
    # 查找所有的 depth 图像
    train_depth_files = sorted([f for f in os.listdir(category_train_path) if f.endswith('_depth.png')])
    print(f"\n⏳ [1/2] 正在处理 训练集 Train Good (共 {len(train_depth_files)} 张)...")

    for i, file_name in enumerate(train_depth_files):
        prefix_in = file_name.split('_')[0]   # 取出原始的前缀，比如 000
        prefix_out = str(i).zfill(3)          # 转换为标准 MVTec 命名，比如 000, 001

        pc_raw, camera_center = depth_to_pointcloud_raw(
            os.path.join(category_train_path, f'{prefix_in}_depth.png'),
            os.path.join(category_train_path, f'{prefix_in}_info_depth.yaml'),
            os.path.join(category_train_path, f'{prefix_in}_pose.txt'),
            FOCAL_LENGTH
        )

        pc_distorted_flat, pc_physical_clean_flat = process_point_cloud_split_paths(pc_raw)

        tifffile.imwrite(os.path.join(train_good_dirs['xyz'], f'{prefix_out}.tiff'), pc_distorted_flat.reshape(512, 512, 3))
        compute_and_save_normal(pc_physical_clean_flat, camera_center, os.path.join(train_good_dirs['normals'], f'{prefix_out}.png'), device)
        copyfile(os.path.join(category_train_path, f'{prefix_in}_image_4.png'), os.path.join(train_good_dirs['rgb'], f'{prefix_out}.png'))

    # === 处理测试集 (遍历全集) ===
    category_test_files = sorted([f for f in os.listdir(category_test_path) if f.endswith('_mask.png')])
    print(f"\n⏳ [2/2] 正在处理 测试集 Test (共 {len(category_test_files)} 张)...")

    good_count, bad_count = 0, 0

    for mask_filename in category_test_files:
        prefix_in = mask_filename.split('_')[0]
        mask = cv2.imread(os.path.join(category_test_path, mask_filename))
        is_bad = np.any(mask)

        if is_bad:
            target_dirs = test_bad_dirs
            prefix_out = str(bad_count).zfill(3)
            bad_count += 1
        else:
            target_dirs = test_good_dirs
            prefix_out = str(good_count).zfill(3)
            good_count += 1

        pc_raw, camera_center = depth_to_pointcloud_raw(
            os.path.join(category_test_path, f'{prefix_in}_depth.png'),
            os.path.join(category_test_path, f'{prefix_in}_info_depth.yaml'),
            os.path.join(category_test_path, f'{prefix_in}_pose.txt'),
            FOCAL_LENGTH
        )

        pc_distorted_flat, pc_physical_clean_flat = process_point_cloud_split_paths(pc_raw)

        tifffile.imwrite(os.path.join(target_dirs['xyz'], f'{prefix_out}.tiff'), pc_distorted_flat.reshape(512, 512, 3))
        compute_and_save_normal(pc_physical_clean_flat, camera_center, os.path.join(target_dirs['normals'], f'{prefix_out}.png'), device)
        copyfile(os.path.join(category_test_path, f'{prefix_in}_image_4.png'), os.path.join(target_dirs['rgb'], f'{prefix_out}.png'))
        cv2.imwrite(os.path.join(target_dirs['gt'], f'{prefix_out}.png'), mask)

    print(f"\n🎉 类别 {args.category} 处理完毕！(Train: {len(train_depth_files)}, Test Good: {good_count}, Test Bad: {bad_count})")

if __name__ == '__main__':
    main()
