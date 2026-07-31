import numpy as np
import os

# 设置 Open3D 头部渲染模式 (必须在导入 open3d 之前)
os.environ["O3D_HEADLESS_RENDERING"] = "1"

import open3d as o3d
import open3d.core as o3c
import tifffile
import cv2
import argparse
from tqdm import tqdm
import sys

# =========================================================================
#  Part 1: CPU 背景清洗逻辑 (RANSAC + DBSCAN)
# =========================================================================

def get_edges_of_pc(organized_pc):
    """
    提取点云边缘的点，用于估计背景平面。
    """
    H, W, C = organized_pc.shape

    top = organized_pc[0:10, :, :].reshape(-1, 3)
    bottom = organized_pc[-10:, :, :].reshape(-1, 3)
    left = organized_pc[:, 0:10, :].reshape(-1, 3)
    right = organized_pc[:, -10:, :].reshape(-1, 3)

    unorganized_edges_pc = np.concatenate([top, bottom, left, right], axis=0)
    unorganized_edges_pc = unorganized_edges_pc[np.nonzero(np.all(unorganized_edges_pc != 0, axis=1))[0], :]
    return unorganized_edges_pc

def get_plane_eq(unorganized_pc, ransac_n_pts=50):
    """
    使用 RANSAC 拟合平面方程
    """
    if unorganized_pc.shape[0] < ransac_n_pts:
        return None, None

    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
    plane_model, inliers = o3d_pc.segment_plane(distance_threshold=0.004, ransac_n=ransac_n_pts, num_iterations=1000)
    return plane_model

def remove_plane_cpu(organized_pc, distance_threshold=0.005):
    """
    去除背景平面
    """
    H, W, C = organized_pc.shape
    flat_pc = organized_pc.reshape(-1, 3)

    edge_points = get_edges_of_pc(organized_pc)
    plane_model = get_plane_eq(edge_points)

    if plane_model is None:
        return organized_pc

    plane_eq = np.array(plane_model)
    points_homogeneous = np.hstack((flat_pc, np.ones((flat_pc.shape[0], 1))))

    distances = np.abs(np.dot(points_homogeneous, plane_eq))
    plane_indices = np.where(distances < distance_threshold)[0]

    flat_pc_clean = flat_pc.copy()
    flat_pc_clean[plane_indices] = 0

    return flat_pc_clean.reshape(H, W, C)

def connected_components_cleaning_cpu(organized_pc):
    """
    使用 DBSCAN 去除离散的噪点
    """
    H, W, C = organized_pc.shape
    flat_pc = organized_pc.reshape(-1, 3)

    nonzero_indices = np.nonzero(np.all(flat_pc != 0, axis=1))[0]

    if len(nonzero_indices) == 0:
        return organized_pc

    points_nonzero = flat_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_nonzero))

    # DBSCAN
    labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))

    if len(labels) == 0:
        return organized_pc

    unique_cluster_ids, cluster_size = np.unique(labels, return_counts=True)
    largest_cluster_id = unique_cluster_ids[np.argmax(cluster_size)]

    outlier_mask_in_nonzero = (labels != largest_cluster_id)
    outlier_indices_original = nonzero_indices[outlier_mask_in_nonzero]

    flat_pc_clean = flat_pc.copy()
    flat_pc_clean[outlier_indices_original] = 0

    return flat_pc_clean.reshape(H, W, C)

# =========================================================================
#  Part 2: 主处理函数
# =========================================================================

def convert_mvtec_tiff_to_normal_o3d_gpu(tiff_path, output_path, radius=0.01, max_nn=30, target_size=(224, 224), device=None):
    # --- 1. 读取数据 (CPU) ---
    try:
        xyz_data = tifffile.imread(tiff_path)
    except Exception as e:
        print(f"错误: 读取文件失败 {tiff_path}: {e}")
        return

    H, W, C = xyz_data.shape

    # =========================================================
    # 步骤 A: 严格的数据清洗 (CPU)
    # =========================================================
    xyz_data = np.nan_to_num(xyz_data, nan=0.0)

    # 【逻辑修改】根据路径判断是否为 dowel
    # MVTec 路径通常包含类别名，例如 .../dowel/train/...
    # 如果路径中包含 "dowel"，使用严格阈值 0.002
    # 其他类别使用较宽松阈值 0.005 以彻底去背景
    if "dowel" in tiff_path.lower():
        plane_thresh = 0.002
    else:
        plane_thresh = 0.005

    # 去除平面 (RANSAC)
    xyz_no_plane = remove_plane_cpu(xyz_data, distance_threshold=plane_thresh)

    # 连通分量去噪 (DBSCAN)
    xyz_clean = connected_components_cleaning_cpu(xyz_no_plane)

    points_flat = xyz_clean.reshape(-1, 3).astype(np.float32)

    # =========================================================
    # 步骤 B: GPU 法向量估计
    # =========================================================
    try:
        pcd_t = o3d.t.geometry.PointCloud(device)
        pcd_t.point.positions = o3c.Tensor(points_flat, device=device)

        pcd_t.estimate_normals(max_nn=max_nn, radius=radius)

        center = o3c.Tensor([0., 0., 0.], dtype=o3c.float32, device=device)
        pcd_t.orient_normals_towards_camera_location(camera_location=center)

        normals_flat = pcd_t.point.normals.cpu().numpy()

    except RuntimeError as e:
        print(f"Open3D 运行时错误 (可能是显存不足或数据异常): {e}")
        return

    normals_grid = normals_flat.reshape(H, W, 3)

    # =========================================================
    # 步骤 C: 渲染与着色
    # =========================================================

    # 1. 法向量取反
    normals_grid = -normals_grid

    # 2. 强制背景为纯平 (蓝色 [0,0,1])
    mask_zeros = np.all(np.isclose(xyz_clean, 0), axis=2)
    normals_grid[mask_zeros] = [0.0, 0.0, 1.0]

    # 3. 颜色映射 [-1, 1] -> [0, 1]
    rgb_normals_float = (normals_grid + 1.0) / 2.0
    rgb_normals_float = np.clip(rgb_normals_float, 0.0, 1.0)

    # 转为 uint8 (此时是 RGB 顺序)
    rgb_uint8 = (rgb_normals_float * 255).astype(np.uint8)

    # =========================================================
    # 步骤 D: 填充至正方形 (Pad to Square) - 针对 Rope/Tire
    # =========================================================
    H_curr, W_curr = rgb_uint8.shape[:2]

    if H_curr != W_curr:
        max_side = max(H_curr, W_curr)

        # 计算填充量
        top = (max_side - H_curr) // 2
        bottom = max_side - H_curr - top
        left = (max_side - W_curr) // 2
        right = max_side - W_curr - left

        # 目标背景色 (RGB):
        # 法向量 [0,0,1] -> 归一化后 [0.5, 0.5, 1.0] -> [127, 127, 255]
        # 注意：这里 rgb_uint8 还是 RGB 顺序，所以用 (127, 127, 255)
        bg_color_rgb = (127, 127, 255)

        # 使用 OpenCV 填充边界
        rgb_uint8 = cv2.copyMakeBorder(
            rgb_uint8,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=bg_color_rgb
        )

    # =========================================================

    # 4. 高质量缩放
    if target_size is not None:
        rgb_uint8 = cv2.resize(rgb_uint8, target_size, interpolation=cv2.INTER_CUBIC)

    # 5. 保存 (OpenCV 使用 BGR)
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_uint8)

def main():
    parser = argparse.ArgumentParser(description="Batch convert MVTec 3D TIFFs to Surface Normal Maps (Cleaned & Padded).")

    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .tiff files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save .png files")

    parser.add_argument("--radius", type=float, default=0.01, help="Search radius for normal estimation")
    parser.add_argument("--max_nn", type=int, default=30, help="Max neighbors for normal estimation")
    parser.add_argument("--img_size", type=int, default=224, help="Target image size (square)")

    args = parser.parse_args()

    # --- 1. 检查 CUDA ---
    if not o3c.cuda.is_available():
        print("错误: 当前 Open3D 环境不支持 CUDA 或未检测到 GPU。")
        sys.exit(1)

    device = o3c.Device("CUDA:0")
    print(f"Using Device: {device}")

    # --- 2. 准备路径 ---
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 3. 获取文件列表 ---
    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.tiff')])

    if len(files) == 0:
        print(f"Warning: No .tiff files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files. Starting processing...")
    print("Logic: RANSAC -> DBSCAN -> GPU Normal -> Pad to Square -> Resize")

    # --- 4. 批量处理 ---
    for f in tqdm(files, desc="Processing TIFFs"):
        input_path = os.path.join(args.input_dir, f)

        output_name = f.replace('.tiff', '.png')
        output_path = os.path.join(args.output_dir, output_name)

        target_size = (args.img_size, args.img_size) if args.img_size > 0 else None

        convert_mvtec_tiff_to_normal_o3d_gpu(
            input_path,
            output_path,
            radius=args.radius,
            max_nn=args.max_nn,
            target_size=target_size,
            device=device
        )

    print(f"\nProcessing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()