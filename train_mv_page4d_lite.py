# In this one we test the performance on dynamic
import os
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from vggt_t_mv.utils.pose_enc import pose_encoding_to_extri_intri
import numpy as np
from vggt.utils.load_fn import load_and_preprocess_images
import json
from utils.metrics import *
from utils.visual import *
from vggt_t_mv.models.vggt import VGGT as VGGT_MV
from PIL import Image
from torchvision import transforms as TF
import re
import glob
import random
from datetime import datetime

# sudo chown -R kz1024 /PHShome/kz1024


def backproject_depth_to_points_batch(depths, intrinsics, extrinsics_3x4):
    # depths: (B, H, W)
    # intrinsics: (B, 3, 3)
    # extrinsics_3x4: (B, 3, 4)
    # return: (B, H*W, 3)
    B, H, W = depths.shape
    device = depths.device
    # Step 1: Create pixel grid (u,v,1) -> [H, W, 3]
    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing='ij')
    ones = torch.ones_like(x)
    pixels = torch.stack([x, y, ones], dim=-1)  # [H, W, 3]
    pixels = pixels.reshape(1, H*W, 3).expand(B, -1, -1).float()  # [B, H*W, 3]
    # Step 2: Inverse intrinsics and pixel -> cam coordinates
    K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
    cam_coords = torch.bmm(pixels, K_inv.transpose(1, 2))  # [B, H*W, 3]
    # Step 3: Depth * cam_coords
    z = depths.reshape(B, -1, 1)  # [B, H*W, 1]
    cam_points = cam_coords * z  # [B, H*W, 3]
    # Step 4: Convert to homogeneous coordinates
    ones = torch.ones(
        (B, cam_points.shape[1], 1), device=device, dtype=cam_points.dtype)
    cam_points_h = torch.cat([cam_points, ones], dim=-1)  # [B, H*W, 4]
    # Step 5: Convert extrinsic 3x4 to 4x4 and invert
    bottom = torch.tensor([[0, 0, 0, 1]], device=device,
                          dtype=cam_points.dtype).expand(B, 1, 4)  # [B, 1, 4]
    extrinsics_4x4 = torch.cat([extrinsics_3x4, bottom], dim=1)  # [B, 4, 4]
    T_wc = torch.inverse(extrinsics_4x4)  # [B, 4, 4]
    # Step 6: Transform to world coordinates
    world_points_h = torch.bmm(
        cam_points_h, T_wc.transpose(1, 2))  # [B, H*W, 4]
    world_points = world_points_h[:, :, :3]  # [B, H*W, 3]
    return world_points  # [B, H*W, 3]


def depth_to_rgb(depth, dir_name):
    depth_normalized = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    depth_colormap = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    cv2.imwrite(dir_name, depth_colormap)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class MultiViewTemporalDataset(Dataset):
    """
    多视角时序数据Dataset，支持数据增强和SegAnyMo监督：
    1. View Permutation（视角打乱）
    2. 时序随机切片（从T_total中随机选择连续窗口）
    3. 可选的帧内数据增强（颜色抖动、轻微旋转等）
    4. SegAnyMo mask监督（从sam2/mask_frames或sam2/initial_preds加载）
    """
    def __init__(self, images_dir, seganymo_dir=None, target_size=378, mode="crop", 
                 T_window_sizes=[6, 8], enable_view_permutation=True,
                 enable_temporal_slice=True, enable_intra_frame_aug=False,
                 use_seganymo_mask=True, mask_source="sam2/mask_frames",
                 train=True):
        """
        Args:
            images_dir: 图像数据目录路径，例如 "data/images"
            seganymo_dir: SegAnyMo数据目录路径，例如 "data/SegAnyMo"，如果为None则不加载mask
            target_size: 目标图像尺寸，默认 378
            mode: 预处理模式，"crop" 或 "pad"
            T_window_sizes: 时序窗口大小列表，例如 [6, 12, 18, 24]
            enable_view_permutation: 是否启用视角打乱
            enable_temporal_slice: 是否启用时序随机切片
            enable_intra_frame_aug: 是否启用帧内数据增强（颜色抖动、旋转等）
            use_seganymo_mask: 是否加载SegAnyMo mask作为监督
            mask_source: mask来源，"sam2/mask_frames" 或 "sam2/initial_preds"
            train: 是否为训练模式（影响数据增强的随机性）
        """
        self.images_dir = os.path.abspath(images_dir)
        self.seganymo_dir = os.path.abspath(seganymo_dir) if seganymo_dir else None
        self.target_size = target_size
        self.mode = mode
        self.T_window_sizes = T_window_sizes if isinstance(T_window_sizes, list) else [T_window_sizes]
        self.enable_view_permutation = enable_view_permutation
        self.enable_temporal_slice = enable_temporal_slice
        self.enable_intra_frame_aug = enable_intra_frame_aug
        self.use_seganymo_mask = use_seganymo_mask and (seganymo_dir is not None)
        self.mask_source = mask_source
        self.train = train
        
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory does not exist: {self.images_dir}")
        
        # 查找所有 time_* 文件夹
        time_dirs = glob.glob(os.path.join(self.images_dir, "time_*"))
        if len(time_dirs) == 0:
            raise ValueError(f"No time_* directories found in {self.images_dir}")
        
        # 提取时间索引并排序
        def extract_time_index(path):
            match = re.search(r'time_(\d+)', os.path.basename(path))
            return int(match.group(1)) if match else float('inf')
        
        time_dirs.sort(key=extract_time_index)
        
        # 查找所有 view*.png 文件（以第一个 time 文件夹为准）
        view_pattern = re.compile(r'view(\d+)\.png')
        first_time_dir = time_dirs[0]
        view_files = [f for f in os.listdir(first_time_dir) if view_pattern.match(f)]
        
        if len(view_files) == 0:
            raise ValueError(f"No view*.png files found in {first_time_dir}")
        
        # 提取视角索引并排序
        def extract_view_index(filename):
            match = view_pattern.match(filename)
            return int(match.group(1)) if match else float('inf')
        
        view_files.sort(key=extract_view_index)
        
        self.time_dirs = time_dirs
        self.view_files = view_files
        self.T_total = len(time_dirs)  # 总时间帧数
        self.V = len(view_files)  # 视角数
        
        # 加载SegAnyMo mask路径（如果启用）
        self.mask_paths = None
        if self.use_seganymo_mask:
            self.mask_paths = {}
            mask_base = os.path.join(self.seganymo_dir, self.mask_source)
            for v_idx, v_file in enumerate(view_files):
                view_num = extract_view_index(v_file)  # 例如 view0.png -> 0
                view_mask_dir = os.path.join(mask_base, f"view_{view_num}")
                if os.path.exists(view_mask_dir):
                    # 查找所有mask文件 (000.png, 001.png, ..., 023.png)
                    mask_files = sorted([f for f in os.listdir(view_mask_dir) 
                                       if f.endswith('.png')])
                    if len(mask_files) == self.T_total:
                        self.mask_paths[view_num] = [
                            os.path.join(view_mask_dir, f) for f in mask_files
                        ]
                    else:
                        print(f"Warning: Expected {self.T_total} mask files for view_{view_num}, "
                              f"found {len(mask_files)}")
                else:
                    print(f"Warning: Mask directory not found: {view_mask_dir}")
            if len(self.mask_paths) == 0:
                print("Warning: No valid mask paths found, disabling mask supervision")
                self.use_seganymo_mask = False
        
        # 帧内数据增强（可选）
        if enable_intra_frame_aug:
            self.color_jitter = TF.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            self.rotation_aug = TF.RandomRotation(degrees=5)  # 最大5度旋转
        else:
            self.color_jitter = None
            self.rotation_aug = None
        
        self.to_tensor = TF.ToTensor()
        
        print(f"Dataset initialized: {self.T_total} time frames, {self.V} views")
        print(f"  View permutation: {enable_view_permutation}")
        print(f"  Temporal slice: {enable_temporal_slice} (window sizes: {self.T_window_sizes})")
        print(f"  Intra-frame aug: {enable_intra_frame_aug}")
        print(f"  SegAnyMo mask: {self.use_seganymo_mask} (source: {mask_source if self.use_seganymo_mask else 'N/A'})")
    
    def __len__(self):
        # 对于过拟合场景，可以返回一个较大的数字，每次访问都会随机增强
        # 或者返回固定的样本数（例如1000）
        return 1000  # 可以根据需要调整
    
    def load_image(self, img_path):
        """加载并预处理单张图像"""
        img = Image.open(img_path)
        
        # 处理 RGBA
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        
        # 帧内数据增强（可选）
        if self.enable_intra_frame_aug and self.train:
            if self.color_jitter is not None and random.random() > 0.5:
                img = self.color_jitter(img)
            if self.rotation_aug is not None and random.random() > 0.5:
                img = self.rotation_aug(img)
        
        width, height = img.size
        
        # Resize
        if self.mode == "pad":
            if width >= height:
                new_width = self.target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = self.target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = self.target_size
            new_height = round(height * (new_width / width) / 14) * 14
        
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = self.to_tensor(img)  # [C, H, W]
        
        # Center crop (crop mode)
        if self.mode == "crop" and new_height > self.target_size:
            start_y = (new_height - self.target_size) // 2
            img = img[:, start_y : start_y + self.target_size, :]
        
        # Pad (pad mode)
        if self.mode == "pad":
            h_padding = self.target_size - img.shape[1]
            w_padding = self.target_size - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), 
                    mode="constant", value=1.0)
        
        return img
    
    def load_mask(self, mask_path):
        """加载并预处理mask图像"""
        mask_img = Image.open(mask_path)
        # 转换为灰度图（如果已经是）
        if mask_img.mode != 'L':
            mask_img = mask_img.convert('L')
        
        # 转换为numpy array
        mask_np = np.array(mask_img, dtype=np.float32)
        
        # 将mask值转换为二值mask (0=静态, 1=动态)
        # mask值为0是背景(静态), >0是动态区域
        mask_binary = (mask_np > 0).astype(np.float32)
        
        # Resize到target_size
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H_orig, W_orig]
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0), 
            size=(self.target_size, self.target_size), 
            mode='nearest'
        ).squeeze(0).squeeze(0)  # [H, W]
        
        return mask_tensor  # [H, W], 值域[0, 1]
    
    def __getitem__(self, idx):
        """
        返回一个batch的数据：[T, V, C, H, W], [T, V, H, W]
        注意：Dataset返回的是 [T, V, C, H, W] 和 [T, V, H, W]，DataLoader会自动添加batch维度
        """
        # 1. 时序随机切片（如果启用）
        if self.enable_temporal_slice and self.train:
            # 随机选择一个窗口大小
            T_window = random.choice(self.T_window_sizes)
            T_window = min(T_window, self.T_total)  # 不超过总帧数
            
            # 随机选择起始时间（确保窗口不超出范围）
            t_start = random.randint(0, max(0, self.T_total - T_window))
            t_end = t_start + T_window
            selected_time_dirs = self.time_dirs[t_start:t_end]
            selected_time_indices = list(range(t_start, t_end))
            T = len(selected_time_dirs)
        else:
            # 使用所有帧
            selected_time_dirs = self.time_dirs
            selected_time_indices = list(range(self.T_total))
            T = self.T_total
        
        # 2. View Permutation（如果启用）
        if self.enable_view_permutation and self.train:
            # 随机打乱视角顺序
            view_perm = torch.randperm(self.V)
            permuted_view_files = [self.view_files[i] for i in view_perm]
            permuted_view_indices = [int(re.search(r'view(\d+)', f).group(1)) for f in permuted_view_files]
        else:
            view_perm = torch.arange(self.V)  # 保持原顺序
            permuted_view_files = self.view_files
            permuted_view_indices = [int(re.search(r'view(\d+)', f).group(1)) for f in permuted_view_files]
        
        # 3. 加载所有图像
        images_list = []
        masks_list = [] if self.use_seganymo_mask else None
        
        for t_idx, time_dir in zip(selected_time_indices, selected_time_dirs):
            time_images = []
            time_masks = [] if self.use_seganymo_mask else None
            
            for v_idx, v_file in zip(permuted_view_indices, permuted_view_files):
                img_path = os.path.join(time_dir, v_file)
                if not os.path.exists(img_path):
                    raise ValueError(f"Image not found: {img_path}")
                
                img = self.load_image(img_path)  # [C, H, W]
                time_images.append(img)
                
                # 加载对应的mask（如果启用）
                if self.use_seganymo_mask and self.mask_paths is not None:
                    if v_idx in self.mask_paths and t_idx < len(self.mask_paths[v_idx]):
                        mask_path = self.mask_paths[v_idx][t_idx]
                        if os.path.exists(mask_path):
                            mask = self.load_mask(mask_path)  # [H, W]
                            if time_masks is not None:
                                time_masks.append(mask)
                        else:
                            # 如果mask不存在，创建全零mask（全静态）
                            mask = torch.zeros(self.target_size, self.target_size)
                            if time_masks is not None:
                                time_masks.append(mask)
                    else:
                        # 如果view不在mask_paths中，创建全零mask
                        mask = torch.zeros(self.target_size, self.target_size)
                        if time_masks is not None:
                            time_masks.append(mask)
            
            images_list.append(time_images)
            if self.use_seganymo_mask and time_masks is not None:
                masks_list.append(time_masks)
        
        # 转换为张量 [T, V, C, H, W]
        images_tensor = torch.stack([torch.stack(time_imgs) for time_imgs in images_list])
        
        # 确保所有图像尺寸一致，统一padding到 target_size x target_size
        T_actual, V_actual, C, H, W = images_tensor.shape
        if H != self.target_size or W != self.target_size:
            images_padded = torch.ones(T_actual, V_actual, C, self.target_size, self.target_size, dtype=images_tensor.dtype)
            
            for t in range(T_actual):
                for v in range(V_actual):
                    img = images_tensor[t, v]  # [C, H, W]
                    h_padding = self.target_size - H
                    w_padding = self.target_size - W
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    img_padded = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant", value=1.0)
                    images_padded[t, v] = img_padded
            
            images_tensor = images_padded
        
        # 返回图像和mask
        if self.use_seganymo_mask and masks_list is not None:
            masks_tensor = torch.stack([torch.stack(time_masks) for time_masks in masks_list])  # [T, V, H, W]
            return images_tensor, masks_tensor
        else:
            return images_tensor, None


def load_time_view_images(images_dir, target_size=378, mode="crop"):
    """
    向后兼容函数：从 data/images 文件夹加载时序多视角图像，组织成 [T, V, C, H, W] 格式
    
    注意：此函数保留用于推理，训练请使用MultiViewTemporalDataset + DataLoader
    """
    dataset = MultiViewTemporalDataset(
        images_dir=images_dir,
        seganymo_dir=None,
        target_size=target_size,
        mode=mode,
        enable_view_permutation=False,
        enable_temporal_slice=False,
        enable_intra_frame_aug=False,
        use_seganymo_mask=False,
        train=False
    )
    
    # 返回第一个样本（所有帧，无增强）
    images_tensor, _ = dataset[0]
    
    metadata = {
        'T': images_tensor.shape[0],
        'V': images_tensor.shape[1],
        'time_dirs': [os.path.basename(d) for d in dataset.time_dirs],
        'view_files': dataset.view_files
    }
    
    return images_tensor, metadata


def save_tensor_image(tensor_img, save_path):
    """
    Save a PyTorch tensor image (3, H, W) to a file.
    Args:
        tensor_img (torch.Tensor): Image tensor with shape (3, H, W), 
                                   values in [0,1] or [-1,1].
        save_path (str): Full file path including filename, e.g. './output/x.png'
    """
    # ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # move to cpu and clone to avoid modifying original tensor
    img = tensor_img.detach().cpu().clone()
    # handle normalized input [-1,1] → [0,1]
    if img.min() < 0:
        img = (img + 1) / 2.0
    img = img.clamp(0, 1)
    # convert to numpy (H, W, 3)
    img_np = (img.numpy() * 255).astype(np.uint8)
    # convert RGB → BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # write image
    cv2.imwrite(save_path, img_bgr)
    # print(f"✅ Saved image to {save_path}")


def process(model, image_names, device, directory='image', name='vggt'):
    images = load_and_preprocess_images(image_names).to(device)
    images = images.unsqueeze(0)    #[1, 24, 3, 294, 378]
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:])
            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, images, ps_idx)
            point_map, point_conf = model.point_head(
                aggregated_tokens_list, images, ps_idx)
    depth_map = depth_map.squeeze(-1)
    world_points = backproject_depth_to_points_batch(
        depth_map[0], intrinsic[0], extrinsic[0])
    # B, H*W, 3
    point_map = point_map.squeeze(0)
    point_map = point_map.reshape(depth_map.size(1), -1, 3)
    images = images.squeeze(0)
    images = images.permute(0, 2, 3, 1).reshape(depth_map.size(1), -1, 3)
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, name), exist_ok=True)
    ext_Dict = {}
    int_Dict = {}
    for num in range(len(images)):
        num_name = image_names[num].split("/")[-1].split(".")[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            world_points[num][::2].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(images[num][::2].cpu().numpy())
        o3d.io.write_point_cloud(f"{directory}/{name}/{num}_dep.ply", pcd)
        # depth_to_rgb(depth_map[0][num].cpu().numpy(), f"{directory}/{name}/{num_name}_depth.png")
        save_tensor_image(images[num].reshape(depth_map[0][num].size(
            0), depth_map[0][num].size(1), 3), f"{directory}/{name}/{num_name}_images.png")
        ext_Dict[num_name] = extrinsic[0][num].cpu().numpy()
        int_Dict[num_name] = intrinsic[0][num].cpu().numpy()
    ext_Dict_clean = {str(k): np.asarray(v).astype(float).tolist()
                      for k, v in ext_Dict.items()}
    int_Dict_clean = {str(k): np.asarray(v).astype(float).tolist()
                      for k, v in int_Dict.items()}
    with open(f"{directory}/{name}/{num_name}_ext.json", "w") as f:
        json.dump(ext_Dict_clean, f, indent=4)
    with open(f"{directory}/{name}/{num_name}_int.json", "w") as f:
        json.dump(int_Dict_clean, f, indent=4)


def save_predictions_visualization(predictions, output_dir="./output_visualization", prefix="output"):
    """
    保存预测结果的可视化：
    1. depth 保存为 4*6 的大图（24个视图）
    2. world_points 保存为 4*6 的大图（24个视图）
    3. 保存全局点云（所有视图的点云合并）
    
    Args:
        predictions: 模型输出字典，包含 'depth', 'world_points' 等
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据形状
    if "depth" in predictions:
        depth = predictions["depth"]  # [B, T, N, H, W] 或 [B, S, H, W]
        depth_shape = depth.shape
        print(f"Depth shape: {depth_shape}")
        
        # 判断输入格式
        if len(depth_shape) == 5:
            # 多视角格式 [B, T, N, H, W]
            B, T, N, H, W = depth_shape
            total_views = T * N
            depth_flat = depth[0].view(total_views, H, W)  # [T*N, H, W]
        elif len(depth_shape) == 4:
            # 单视角格式 [B, S, H, W]
            B, S, H, W = depth_shape
            total_views = S
            depth_flat = depth[0]  # [S, H, W]
        else:
            print(f"Unknown depth shape: {depth_shape}, skipping depth visualization")
            depth_flat = None
        
        if depth_flat is not None:
            # 创建 4*6 大图
            rows, cols = 4, 6
            if total_views <= rows * cols:
                # 转换为 numpy 并归一化
                depth_np = depth_flat.cpu().numpy()
                
                # 归一化到 [0, 1]（每张图独立归一化）
                depth_normalized = []
                for i in range(min(total_views, rows * cols)):
                    d = depth_np[i]
                    if d.max() > d.min():
                        d_norm = (d - d.min()) / (d.max() - d.min())
                    else:
                        d_norm = d
                    depth_normalized.append(d_norm)
                
                # 应用 colormap (viridis)
                cmap = plt.cm.viridis
                depth_colored = [cmap(d)[:, :, :3] for d in depth_normalized]  # 去掉 alpha
                
                # 创建大图
                fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
                axes = axes.flatten() if rows > 1 else [axes]
                
                for idx in range(rows * cols):
                    ax = axes[idx]
                    if idx < len(depth_colored):
                        ax.imshow(depth_colored[idx])
                        ax.set_title(f"View {idx}", fontsize=8)
                    ax.axis('off')
                
                plt.tight_layout()
                depth_path = os.path.join(output_dir, f"{prefix}_depth_grid.png")
                plt.savefig(depth_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved depth grid to {depth_path}")
    
    if "world_points" in predictions:
        world_points = predictions["world_points"]  # [B, T, N, H, W, 3] 或 [B, S, H, W, 3]
        wp_shape = world_points.shape
        print(f"World points shape: {wp_shape}")
        
        # 判断输入格式
        if len(wp_shape) == 6:
            # 多视角格式 [B, T, N, H, W, 3]
            B, T, N, H, W, _ = wp_shape
            total_views = T * N
            wp_flat = world_points[0].view(total_views, H, W, 3)  # [T*N, H, W, 3]
        elif len(wp_shape) == 5:
            # 单视角格式 [B, S, H, W, 3]
            B, S, H, W, _ = wp_shape
            total_views = S
            wp_flat = world_points[0]  # [S, H, W, 3]
        else:
            print(f"Unknown world_points shape: {wp_shape}, skipping visualization")
            wp_flat = None
        
        if wp_flat is not None:
            # 创建 4*6 大图（可视化每个视图的点云投影）
            rows, cols = 4, 6
            
            # 计算每个视图的点云范围（用于可视化）
            wp_np = wp_flat.cpu().numpy()
            
            # 为每个视图创建一个 2D 投影可视化
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            axes = axes.flatten() if rows > 1 else [axes]
            
            for idx in range(min(total_views, rows * cols)):
                ax = axes[idx]
                if idx < total_views:
                    wp_view = wp_np[idx]  # [H, W, 3]
                    # 计算有效的点（非零或非 NaN）
                    valid_mask = np.isfinite(wp_view).all(axis=-1) & (np.linalg.norm(wp_view, axis=-1) > 1e-6)
                    
                    if valid_mask.sum() > 0:
                        # 使用 Z 坐标作为深度可视化
                        z_coords = wp_view[:, :, 2]
                        z_masked = np.where(valid_mask, z_coords, np.nan)
                        
                        im = ax.imshow(z_masked, cmap='viridis')
                        ax.set_title(f"View {idx} (Z)", fontsize=8)
                    else:
                        ax.text(0.5, 0.5, "No valid points", ha='center', va='center')
                    ax.axis('off')
            
            plt.tight_layout()
            wp_path = os.path.join(output_dir, f"{prefix}_world_points_grid.png")
            plt.savefig(wp_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved world points grid to {wp_path}")
            
            # 保存全局点云（合并所有视图）
            print("Collecting global point cloud...")
            all_points = []
            all_colors = []
            
            # 获取原始图像用于着色（如果有）
            images = predictions.get("images", None)
            
            if len(wp_shape) == 6:
                # 多视角格式
                for t in range(T):
                    for v in range(N):
                        wp_view = wp_np[t * N + v]  # [H, W, 3]
                        valid_mask = np.isfinite(wp_view).all(axis=-1) & (np.linalg.norm(wp_view, axis=-1) > 1e-6)
                        
                        if valid_mask.sum() > 0:
                            points_flat = wp_view.reshape(-1, 3)
                            valid_points = points_flat[valid_mask.flatten()]
                            all_points.append(valid_points)
                            
                            # 如果有图像，提取对应的颜色
                            if images is not None:
                                img_view = images[0, t, v].cpu().numpy()  # [C, H, W]
                                if img_view.max() <= 1.0:
                                    img_view = (img_view * 255).astype(np.uint8)
                                else:
                                    img_view = img_view.astype(np.uint8)
                                img_view = img_view.transpose(1, 2, 0)  # [H, W, C]
                                colors_flat = img_view.reshape(-1, 3)
                                valid_colors = colors_flat[valid_mask.flatten()] / 255.0
                                all_colors.append(valid_colors)
            elif len(wp_shape) == 5:
                # 单视角格式
                for s in range(total_views):
                    wp_view = wp_np[s]  # [H, W, 3]
                    valid_mask = np.isfinite(wp_view).all(axis=-1) & (np.linalg.norm(wp_view, axis=-1) > 1e-6)
                    
                    if valid_mask.sum() > 0:
                        points_flat = wp_view.reshape(-1, 3)
                        valid_points = points_flat[valid_mask.flatten()]
                        all_points.append(valid_points)
                        
                        if images is not None:
                            img_view = images[0, s].cpu().numpy()
                            if img_view.max() <= 1.0:
                                img_view = (img_view * 255).astype(np.uint8)
                            else:
                                img_view = img_view.astype(np.uint8)
                            img_view = img_view.transpose(1, 2, 0)
                            colors_flat = img_view.reshape(-1, 3)
                            valid_colors = colors_flat[valid_mask.flatten()] / 255.0
                            all_colors.append(valid_colors)
            
            if all_points:
                # 合并所有点
                global_points = np.concatenate(all_points, axis=0)
                print(f"Global point cloud: {len(global_points)} points")
                
                # 创建 Open3D 点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(global_points)
                
                # 添加颜色（如果有）
                if all_colors:
                    global_colors = np.concatenate(all_colors, axis=0)
                    pcd.colors = o3d.utility.Vector3dVector(global_colors)
                
                # 保存点云
                pcd_path = os.path.join(output_dir, f"{prefix}_global_pointcloud.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                print(f"Saved global point cloud to {pcd_path}")
            else:
                print("Warning: No valid points found for global point cloud")


def freeze_parameters_stage1(model):
    """
    阶段1：冻结基础特征，只训练新增模块
    
    冻结：
    - patch_embed.*
    - register_token, camera_token
    - time_blocks.{0..7}.*, {16..23}.* (前8层和后8层)
    - view_blocks.{0..7}.*, {16..23}.* (前8层和后8层)
    - camera_head.*, depth_head.*, point_head.*, track_head.* (可选)
    
    训练：
    - 两流架构blocks (pose_time_blocks, pose_view_blocks, geo_time_blocks, geo_view_blocks)
    - dynamic_mask_head.*
    - lambda_pose_logit, lambda_geo_logit, lambda_pose_t_logit, lambda_geo_t_logit
    - spatial_mask_head.*
    - time_blocks.{8..15}.*, view_blocks.{8..15}.* (L_mid层)
    - dim_adapter (如果存在)
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # ========== 解冻需要训练的模块 ==========
    aggregator = model.aggregator
    
    # 1. 两流架构blocks (L_mid层: 6-10，共5层)
    if hasattr(aggregator, 'pose_time_blocks'):
        for block in aggregator.pose_time_blocks:
            for param in block.parameters():
                param.requires_grad = True
    if hasattr(aggregator, 'pose_view_blocks'):
        for block in aggregator.pose_view_blocks:
            for param in block.parameters():
                param.requires_grad = True
    if hasattr(aggregator, 'geo_time_blocks'):
        for block in aggregator.geo_time_blocks:
            for param in block.parameters():
                param.requires_grad = True
    if hasattr(aggregator, 'geo_view_blocks'):
        for block in aggregator.geo_view_blocks:
            for param in block.parameters():
                param.requires_grad = True
    
    # 2. Dynamic Mask Head
    if hasattr(aggregator, 'dynamic_mask_head'):
        for param in aggregator.dynamic_mask_head.parameters():
            param.requires_grad = True
    
    # 3. Lambda参数（控制两流分离）
    if hasattr(aggregator, 'lambda_pose_logit'):
        aggregator.lambda_pose_logit.requires_grad = True
    if hasattr(aggregator, 'lambda_geo_logit'):
        aggregator.lambda_geo_logit.requires_grad = True
    if hasattr(aggregator, 'lambda_pose_t_logit'):
        aggregator.lambda_pose_t_logit.requires_grad = True
    if hasattr(aggregator, 'lambda_geo_t_logit'):
        aggregator.lambda_geo_t_logit.requires_grad = True
    
    # 4. Spatial Mask Head (L_mid层掩码头)
    if hasattr(aggregator, 'spatial_mask_head'):
        for param in aggregator.spatial_mask_head.parameters():
            param.requires_grad = True
    
    # 5. L_mid层 (8-15层，索引0-based: 8-15)
    mid_layer_indices = list(range(8, 16))
    for idx in mid_layer_indices:
        if idx < len(aggregator.time_blocks):
            for param in aggregator.time_blocks[idx].parameters():
                param.requires_grad = True
        if idx < len(aggregator.view_blocks):
            for param in aggregator.view_blocks[idx].parameters():
                param.requires_grad = True
    
    # 6. Dimension Adapters (如果存在)
    if model.camera_head is not None and hasattr(model.camera_head, 'dim_adapter'):
        if model.camera_head.dim_adapter is not None:
            for param in model.camera_head.dim_adapter.parameters():
                param.requires_grad = True
    
    if model.depth_head is not None and hasattr(model.depth_head, 'dim_adapter'):
        if model.depth_head.dim_adapter is not None:
            for param in model.depth_head.dim_adapter.parameters():
                param.requires_grad = True
    
    if model.point_head is not None and hasattr(model.point_head, 'dim_adapter'):
        if model.point_head.dim_adapter is not None:
            for param in model.point_head.dim_adapter.parameters():
                param.requires_grad = True
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 1: Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def compute_mask_supervision_loss(predictions, segmask_gt, images):
    """
    SegAnyMo mask监督损失：使用SegAnyMo的mask作为动态区域的监督信号
    
    Args:
        predictions: 模型输出字典
        segmask_gt: SegAnyMo mask [B, T, N, H, W], 值域[0, 1], 1表示动态区域
        images: 输入图像 [B, T, N, 3, H, W]，用于从模型中提取mask
    
    Returns:
        loss: 标量tensor, loss_dict: 字典
    """
    if segmask_gt is None:
        return torch.tensor(0.0, device=list(predictions.values())[0].device, requires_grad=True), {}
    
    B, T, N, C, H, W = images.shape
    
    # 方法1：如果有dynamic_mask在predictions中，直接使用
    if 'dynamic_mask' in predictions:
        mask_pred = predictions['dynamic_mask']  # [B, T, N, H_patch, W_patch] 或 [B, T, N, H, W]
        # 如果尺寸不匹配，resize
        if mask_pred.shape[-2:] != (H, W):
            mask_pred = torch.nn.functional.interpolate(
                mask_pred.view(B*T*N, 1, mask_pred.shape[-2], mask_pred.shape[-1]),
                size=(H, W), mode='bilinear', align_corners=False
            ).view(B, T, N, H, W)
    else:
        # 方法2：从aggregator的dual_stream_outputs或aggregated_tokens_list中提取mask
        # 这里简化：使用depth和world_points的置信度来近似动态mask
        # 动态区域通常depth变化大，confidence低
        if 'depth_conf' in predictions:
            depth_conf = predictions['depth_conf']  # [B, T, N, 1, H, W]
            # 低置信度区域可能是动态的
            mask_pred = 1.0 - depth_conf.squeeze(-3)  # [B, T, N, H, W]
        elif 'world_points_conf' in predictions:
            point_conf = predictions['world_points_conf']  # [B, T, N, 1, H, W]
            mask_pred = 1.0 - point_conf.squeeze(-3)  # [B, T, N, H, W]
        else:
            # 如果没有相关信息，返回0损失
            mask_loss = torch.tensor(0.0, device=segmask_gt.device, requires_grad=True)
            return mask_loss, {'mask_loss': 0.0}
    
    # 确保mask_pred和segmask_gt形状一致
    if mask_pred.shape != segmask_gt.shape:
        # Resize segmask_gt到mask_pred的尺寸，或者反过来
        if mask_pred.shape[-2:] != segmask_gt.shape[-2:]:
            segmask_gt = torch.nn.functional.interpolate(
                segmask_gt.view(B*T*N, 1, segmask_gt.shape[-2], segmask_gt.shape[-1]),
                size=mask_pred.shape[-2:], mode='nearest'
            ).view(segmask_gt.shape[:3] + mask_pred.shape[-2:])
    
    # 计算BCE损失（二值交叉熵）
    mask_pred_clamped = torch.clamp(mask_pred, min=1e-6, max=1-1e-6)
    bce_loss = - (segmask_gt * torch.log(mask_pred_clamped + 1e-6) + 
                  (1 - segmask_gt) * torch.log(1 - mask_pred_clamped + 1e-6))
    mask_loss = bce_loss.mean()
    
    return mask_loss, {'mask_loss': mask_loss.item()}


def compute_multi_view_consistency_loss(predictions, images, segmask_gt=None):
    """
    多视角几何一致性损失（无GT）+ SegAnyMo mask监督
    
    Args:
        predictions: 模型输出字典
        images: 输入图像 [B, T, N, 3, H, W]
        segmask_gt: SegAnyMo mask [B, T, N, H, W], 可选
    
    Returns:
        loss: 标量tensor
    """
    B, T, N, C, H, W = images.shape
    
    # 1. 深度一致性：world_points的z坐标应该与depth一致
    if 'depth' in predictions and 'world_points' in predictions:
        depth = predictions['depth']  # [B, T, N, 1, H, W]
        world_points = predictions['world_points']  # [B, T, N, 3, H, W]
        
        # 提取z坐标（第3个通道，索引2）
        depth_from_points = world_points[..., 2, :, :].unsqueeze(-3)  # [B, T, N, 1, H, W]
        
        # 只对有效的深度区域计算loss（depth > 0）
        valid_mask = (depth > 0) & (depth_from_points > 0)
        if valid_mask.sum() > 0:
            depth_consistency = torch.abs(depth - depth_from_points) * valid_mask.float()
            depth_loss = depth_consistency.sum() / valid_mask.sum().clamp(min=1)
        else:
            depth_loss = torch.tensor(0.0, device=depth.device, requires_grad=True)
    else:
        depth_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
    
    # 2. 多视角重投影一致性（简化版）
    # 使用depth和world_points，计算同一3D点在不同视角的投影一致性
    if 'depth' in predictions and 'world_points' in predictions:
        # 这里简化实现：计算同一时间帧内不同视角的world_points的一致性
        # 更完整的实现需要相机内参和外参来重投影
        world_points = predictions['world_points']  # [B, T, N, 3, H, W]
        
        # 计算同一时间帧内不同视角的3D点距离（简化：假设对应像素是同一3D点）
        # 对每个时间帧，计算不同视角之间world_points的距离
        # 这里简化：计算相邻视角的world_points差异
        if N > 1:
            # 计算视角间的差异
            wp_view_diff = world_points[:, :, 1:] - world_points[:, :, :-1]  # [B, T, N-1, 3, H, W]
            # 我们希望对应像素的3D点在不同视角应该一致（差异小）
            reproj_loss = torch.mean(torch.norm(wp_view_diff, dim=-3, p=2))  # 对通道维度求norm
        else:
            reproj_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
    else:
        reproj_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
    
    # 3. 几何平滑性：相邻像素的world_points应该连续
    if 'world_points' in predictions:
        world_points = predictions['world_points']  # [B, T, N, 3, H, W]
        
        # 计算水平和垂直方向的梯度
        # 水平方向：[B, T, N, 3, H, W] -> [B, T, N, 3, H, W-1]
        diff_h = world_points[..., :-1] - world_points[..., 1:]
        # 垂直方向：[B, T, N, 3, H, W] -> [B, T, N, 3, H-1, W]
        diff_v = world_points[..., :-1, :] - world_points[..., 1:, :]
        
        # 平滑性损失：梯度应该小（对空间维度求norm）
        smoothness_h = torch.mean(torch.norm(diff_h, dim=-3, p=2))  # 对通道维度(3)求norm
        smoothness_v = torch.mean(torch.norm(diff_v, dim=-3, p=2))
        smoothness_loss = (smoothness_h + smoothness_v) / 2.0
    else:
        smoothness_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
    
    # 4. 置信度正则化：避免置信度过低或过高
    # 注意：conf_activation="expp1" 意味着 confidence = 1 + exp(x)，值域为 [1, +∞)
    # 我们希望 confidence 在一个合理范围内，既不要太低也不要过高
    conf_loss = torch.tensor(0.0, device=images.device)
    if 'depth_conf' in predictions:
        depth_conf = predictions['depth_conf']  # [B, T, N, 1, H, W]
        # 由于 depth_conf 值域是 [1, +∞)，我们需要：
        # 1. 鼓励置信度不要太低（避免过低的置信度）
        # 2. 惩罚过高的置信度（避免过度自信）
        # 使用目标置信度范围，例如 [1, 10] 或 [1, 20]
        target_conf_min = 1.0
        target_conf_max = 10.0  # 可以根据实际情况调整
        
        # 低置信度惩罚：当 conf < target_conf_min 时惩罚
        low_conf_penalty = F.relu(target_conf_min - depth_conf).mean()
        
        # 高置信度惩罚：当 conf > target_conf_max 时惩罚
        high_conf_penalty = F.relu(depth_conf - target_conf_max).mean()
        
        conf_loss = conf_loss + 0.1 * (low_conf_penalty + high_conf_penalty)
    
    if 'world_points_conf' in predictions:
        point_conf = predictions['world_points_conf']  # [B, T, N, 1, H, W]
        target_conf_min = 1.0
        target_conf_max = 10.0
        
        low_conf_penalty = F.relu(target_conf_min - point_conf).mean()
        high_conf_penalty = F.relu(point_conf - target_conf_max).mean()
        
        conf_loss = conf_loss + 0.1 * (low_conf_penalty + high_conf_penalty)
    
    # 5. SegAnyMo mask监督损失（如果提供）
    mask_loss_tensor, mask_loss_dict = compute_mask_supervision_loss(predictions, segmask_gt, images)
    
    # 总损失
    total_loss = (
        1.0 * depth_loss +
        0.5 * reproj_loss +
        0.1 * smoothness_loss +
        0.1 * conf_loss +
        0.5 * mask_loss_tensor  # mask监督权重
    )
    
    loss_dict = {
        'depth_loss': depth_loss.item() if isinstance(depth_loss, torch.Tensor) else depth_loss,
        'reproj_loss': reproj_loss.item() if isinstance(reproj_loss, torch.Tensor) else reproj_loss,
        'smoothness_loss': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else smoothness_loss,
        'conf_loss': conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss,
    }
    loss_dict.update(mask_loss_dict)
    
    return total_loss, loss_dict


def compute_dual_stream_separation_loss(predictions):
    """
    两流分离损失：强制位姿流和几何流特征不同
    """
    if 'dual_stream_outputs' not in predictions:
        return torch.tensor(0.0, device=list(predictions.values())[0].device), {}
    
    dual_stream = predictions['dual_stream_outputs']
    if 'pose' not in dual_stream or 'geo' not in dual_stream:
        return torch.tensor(0.0, device=list(predictions.values())[0].device), {}
    
    pose_features = dual_stream['pose'][-1]  # 最后一层特征 [B, T, N, P, C]
    geo_features = dual_stream['geo'][-1]  # [B, T, N, P, C]
    
    # 计算余弦相似度（我们希望它小）
    pose_flat = pose_features.view(-1, pose_features.shape[-1])  # [*, C]
    geo_flat = geo_features.view(-1, geo_features.shape[-1])  # [*, C]
    
    # 归一化
    pose_norm = torch.nn.functional.normalize(pose_flat, p=2, dim=-1)
    geo_norm = torch.nn.functional.normalize(geo_flat, p=2, dim=-1)
    
    # 余弦相似度
    cosine_sim = (pose_norm * geo_norm).sum(dim=-1).mean()
    
    # 分离损失：我们希望相似度小（加上负号使loss变大）
    separation_loss = cosine_sim
    
    return separation_loss, {'separation_loss': separation_loss.item()}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16
    
    # ========== 训练模式：使用DataLoader ==========
    images_dir = "data/images"
    seganymo_dir = "data/SegAnyMo"
    train_mode = True  # 设置为True使用DataLoader训练，False使用推理模式
    
    if train_mode and os.path.exists(images_dir):
        print(f"\n{'='*60}")
        print("Training Mode: Using DataLoader with Augmentation + SegAnyMo Supervision")
        print(f"{'='*60}")
        
        # 创建训练Dataset（启用数据增强和SegAnyMo监督）
        train_dataset = MultiViewTemporalDataset(
            images_dir=images_dir,
            seganymo_dir=seganymo_dir if os.path.exists(seganymo_dir) else None,
            target_size=378,
            mode="crop",
            T_window_sizes=[2],  # 时序窗口大小列表（最大8，避免OOM）
            enable_view_permutation=True,  # 启用视角打乱（Pi3自由打乱）
            enable_temporal_slice=True,  # 启用时序随机切片
            enable_intra_frame_aug=False,  # 帧内增强（可选，默认关闭）
            use_seganymo_mask=True,  # 启用SegAnyMo mask监督
            mask_source="sam2/mask_frames",  # 或 "sam2/initial_preds"
            train=True
        )
        
        # 创建DataLoader
        # 注意：batch_size=1（因为4视角×24帧已经很大）
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,  # 每个epoch打乱
            num_workers=0,  # 多进程可能导致内存问题，设置为0使用主进程
            pin_memory=True if device == "cuda" else False,
            drop_last=False
        )
        
        print(f"DataLoader created: {len(train_dataset)} samples")
        print(f"  Batch size: 1")
        print(f"  Shuffle: True")
        
        # ========== 初始化模型 ==========
        origin = "checkpoint/checkpoint_150.pt"
        model_mv = VGGT_MV(
            img_size=374,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,
            enable_point=True,
            enable_depth=True,
            enable_track=False  # 阶段1不训练track
        )
        
        # 加载权重
        pi3_path = '/home/star/zzb/Pi3/ckpts/model.safetensors'
        stats = model_mv.load_pretrained_weights(
            checkpoint_path=origin if os.path.exists(origin) else None,
            pi3_path=pi3_path,
            device=device
        )
        print(f"Weight loading stats: {stats}")
        
        model_mv.to(device)
        
        # ========== 阶段1：冻结参数 ==========
        print("\n" + "="*60)
        print("Stage 1: Freezing base features, training new modules")
        print("="*60)
        freeze_parameters_stage1(model_mv)
        
        # ========== 设置优化器 ==========
        # 只优化需要训练的参数
        trainable_params = [p for p in model_mv.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-4,  # 两流blocks和mask heads
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器（可选）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        # ========== TensorBoard 可视化 ==========
        log_dir = "results/train/init"
        os.makedirs(log_dir, exist_ok=True)
        # 添加时间戳以区分不同的训练运行
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_with_timestamp = os.path.join(log_dir, timestamp)
        writer = SummaryWriter(log_dir=log_dir_with_timestamp)
        print(f"TensorBoard logs will be saved to: {log_dir_with_timestamp}")
        print(f"  To view: tensorboard --logdir={log_dir_with_timestamp}")
        
        # ========== 训练循环 ==========
        num_epochs = 50
        print(f"\nStarting training for {num_epochs} epochs...")
        print("-"*60)
        
        model_mv.train()
        global_step = 0  # 全局步数计数器
        for epoch in range(num_epochs):
            epoch_losses = {
                'total': 0.0,
                'depth_loss': 0.0,
                'reproj_loss': 0.0,
                'smoothness_loss': 0.0,
                'conf_loss': 0.0,
                'mask_loss': 0.0,
                'separation_loss': 0.0
            }
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                # DataLoader对Dataset返回tuple的处理：
                # - Dataset返回 (images, masks) -> DataLoader返回 (batch_images, batch_masks)
                # - Dataset返回 (images, None) -> DataLoader可能返回 (batch_images, None) 或 list
                
                # 处理不同的返回格式
                if isinstance(batch_data, (tuple, list)):
                    images_batch = batch_data[0]
                    masks_batch = batch_data[1] if len(batch_data) > 1 else None
                    
                    # 处理None的情况
                    if masks_batch is not None and not isinstance(masks_batch, torch.Tensor):
                        masks_batch = None
                else:
                    # 单个tensor（Dataset返回单个值）
                    images_batch = batch_data
                    masks_batch = None
                
                # 调试信息（仅第一个batch）
                if batch_idx == 0:
                    print(f"Debug batch 0: batch_data type={type(batch_data)}, "
                          f"images_batch type={type(images_batch)}, "
                          f"masks_batch type={type(masks_batch)}")
                    if isinstance(images_batch, torch.Tensor):
                        print(f"  images_batch shape: {images_batch.shape}")
                    elif isinstance(images_batch, (list, tuple)):
                        print(f"  images_batch is {type(images_batch)} with len={len(images_batch)}")
                        if len(images_batch) > 0:
                            print(f"  images_batch[0] type: {type(images_batch[0])}")
                
                # 确保images_batch是tensor
                # 如果仍然是list/tuple，说明DataLoader没有正确batch
                if isinstance(images_batch, (list, tuple)):
                    # 尝试从list/tuple中提取tensor
                    if len(images_batch) > 0 and isinstance(images_batch[0], torch.Tensor):
                        # DataLoader没有batch，需要手动stack
                        images_batch = torch.stack(list(images_batch), dim=0)
                        if masks_batch is not None and isinstance(masks_batch, (list, tuple)):
                            if len(masks_batch) > 0 and isinstance(masks_batch[0], torch.Tensor):
                                masks_batch = torch.stack(list(masks_batch), dim=0)
                            else:
                                masks_batch = None
                    else:
                        raise TypeError(f"Expected torch.Tensor in batch_data[0], "
                                      f"got {type(images_batch[0]) if len(images_batch) > 0 else 'empty'}")
                elif not isinstance(images_batch, torch.Tensor):
                    raise TypeError(f"Expected torch.Tensor, got {type(images_batch)}. "
                                  f"This usually means Dataset.__getitem__ returned incorrect format.")
                
                images_batch = images_batch.to(device)  # [B, T, V, C, H, W]
                if masks_batch is not None and isinstance(masks_batch, torch.Tensor):
                    masks_batch = masks_batch.to(device)  # [B, T, V, H, W]
                else:
                    masks_batch = None
                B, T, V, C, H, W = images_batch.shape
                
                # 前向传播
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(dtype=dtype, enabled=True):
                    predictions = model_mv(images_batch)
                    
                    # 计算损失（多视角一致性 + SegAnyMo mask监督）
                    loss_total, loss_dict = compute_multi_view_consistency_loss(
                        predictions, images_batch, segmask_gt=masks_batch
                    )
                    
                    # 两流分离损失
                    sep_loss, sep_dict = compute_dual_stream_separation_loss(predictions)
                    loss_total = loss_total + 0.1 * sep_loss
                    loss_dict.update(sep_dict)
                
                # 反向传播
                loss_total.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                # 累积损失
                epoch_losses['total'] += loss_total.item()
                for key, value in loss_dict.items():
                    if key in epoch_losses:
                        epoch_losses[key] += value
                num_batches += 1
                
                # TensorBoard: 记录每个batch的损失
                global_step += 1
                writer.add_scalar('Loss/Batch/Total', loss_total.item(), global_step)
                for key, value in loss_dict.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'Loss/Batch/{key}', value, global_step)
                
                # 记录学习率
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Training/LearningRate', current_lr, global_step)
                
                # 每10个batch打印一次
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}: "
                          f"Loss={loss_total.item():.6f}, "
                          f"Depth={loss_dict.get('depth_loss', 0):.6f}, "
                          f"Smooth={loss_dict.get('smoothness_loss', 0):.6f}, "
                          f"Mask={loss_dict.get('mask_loss', 0):.6f}")
                
                # 每个epoch的前几个batch记录图像可视化（可选，节省空间）
                if batch_idx < 1 and epoch % 5 == 0:  # 每5个epoch记录一次
                    # 记录输入图像（第一个视角，第一个时间帧）
                    if images_batch.shape[1] > 0 and images_batch.shape[2] > 0:
                        img_to_log = images_batch[0, 0, 0].cpu()  # [C, H, W]
                        # 反归一化：从归一化后的图像恢复（如果需要）
                        writer.add_image('Input/Image_t0_v0', img_to_log, global_step)
                        
                        # 记录预测的depth（如果可用）
                        if 'depth' in predictions:
                            depth_to_log = predictions['depth'][0, 0, 0, 0].cpu().detach()  # [H, W]
                            # 归一化depth用于可视化
                            if depth_to_log.max() > depth_to_log.min():
                                depth_normalized = (depth_to_log - depth_to_log.min()) / (depth_to_log.max() - depth_to_log.min())
                            else:
                                depth_normalized = depth_to_log
                            writer.add_image('Predictions/Depth_t0_v0', depth_normalized.unsqueeze(0), global_step)
                        
                        # 记录mask（如果有）
                        if masks_batch is not None:
                            mask_to_log = masks_batch[0, 0, 0].cpu()  # [H, W]
                            writer.add_image('GT/Mask_t0_v0', mask_to_log.unsqueeze(0), global_step)
            
            # 计算平均损失
            if num_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # TensorBoard: 记录每个epoch的平均损失
            writer.add_scalar('Loss/Epoch/Total', epoch_losses['total'], epoch + 1)
            writer.add_scalar('Loss/Epoch/Depth', epoch_losses['depth_loss'], epoch + 1)
            writer.add_scalar('Loss/Epoch/Reproj', epoch_losses['reproj_loss'], epoch + 1)
            writer.add_scalar('Loss/Epoch/Smoothness', epoch_losses['smoothness_loss'], epoch + 1)
            writer.add_scalar('Loss/Epoch/Conf', epoch_losses['conf_loss'], epoch + 1)
            writer.add_scalar('Loss/Epoch/Mask', epoch_losses['mask_loss'], epoch + 1)
            writer.add_scalar('Loss/Epoch/Separation', epoch_losses['separation_loss'], epoch + 1)
            writer.add_scalar('Training/LearningRate', current_lr, epoch + 1)
            
            # TensorBoard: 记录模型参数的统计信息（每个epoch一次，可选）
            if (epoch + 1) % 5 == 0:  # 每5个epoch记录一次，避免日志过大
                for name, param in model_mv.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        writer.add_histogram(f'Parameters/{name}', param.data, epoch + 1)
                        writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch + 1)
            
            # 打印epoch总结
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Total Loss: {epoch_losses['total']:.6f}")
            print(f"  Depth Loss: {epoch_losses['depth_loss']:.6f}")
            print(f"  Smoothness Loss: {epoch_losses['smoothness_loss']:.6f}")
            print(f"  Conf Loss: {epoch_losses['conf_loss']:.6f}")
            print(f"  Mask Loss: {epoch_losses['mask_loss']:.6f}")
            print(f"  Separation Loss: {epoch_losses['separation_loss']:.6f}")
            print(f"  Learning Rate: {current_lr:.8f}")
            print("-"*60)
            
            # 每10个epoch保存一次checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoints/stage1_epoch_{epoch+1}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_mv.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': epoch_losses,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # 关闭TensorBoard writer
        writer.close()
        
        print("\n" + "="*60)
        print("Stage 1 training completed!")
        print(f"TensorBoard logs saved to: {log_dir_with_timestamp}")
        print(f"  To view: tensorboard --logdir={log_dir_with_timestamp}")
        print("="*60)
    
    # ========== 推理模式：使用原始函数（向后兼容）==========
    elif os.path.exists(images_dir):
        print(f"\n{'='*60}")
        print("Inference Mode: Using load_time_view_images")
        print(f"{'='*60}")
        
        # 加载时序多视角数据（无增强，用于推理）
        images_tv, metadata = load_time_view_images(images_dir, target_size=378, mode="crop")
        print(f"Loaded images with shape: {images_tv.shape}")
        print(f"Time frames: {metadata['T']}, Views: {metadata['V']}")
        
        # 将 [T, V, C, H, W] 转换为模型需要的 [B, T, V, C, H, W] 格式
        T_total, V, C, H, W = images_tv.shape
        images_batch = images_tv.unsqueeze(0).to(device)  # [1, T_total, V, C, H, W]
        
        # 手动设置T：如果指定了T，只使用前T个时间帧
        T = 6  # 手动设置T=6
        if T < T_total:
            images_batch = images_batch[:, :T, :, :, :, :]  # [1, T, V, C, H, W]
            print(f"Using first {T} time frames out of {T_total} total frames")
        
        print(f"Input shape: {images_batch.shape} (B={1}, T={T}, V={V}, C={C}, H={H}, W={W})")
    
    # 模型初始化和推理代码（保持不变）
    if os.path.exists(images_dir):
        
        # 初始化 vggt_t_mv 模型，使用多视角模式
        origin = "checkpoint/checkpoint_150.pt"
        model_mv = VGGT_MV(
            img_size=378,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,
            enable_point=True,
            enable_depth=True,
            enable_track=True
        )
        
        # 加载权重（使用新的权重加载方法）
        pi3_path ='/home/star/zzb/Pi3/ckpts/model.safetensors'   # 可选：Pi3模型路径，如 "facebook/Pi3" 或本地路径
        # pi3_path = "/path/to/pi3/model"  # 取消注释以加载Pi3权重
        
        stats = model_mv.load_pretrained_weights(
            checkpoint_path=origin if os.path.exists(origin) else None,
            pi3_path=pi3_path,
            device=device
        )
        
        print(f"Weight loading stats: {stats}")
        
        model_mv.to(device)
        model_mv.eval()
        
        # 推理
        print(f"Running inference...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model_mv(images_batch)
        # for key, value in predictions.items():
        #     if type(value) == torch.Tensor:
        #         print(f"{key}: {value.shape}")
        #     else:
        #         print(f"{key}: {len(value)}")
        print(f"Inference completed. Predictions keys: {list(predictions.keys())}")
        
        # 保存结果
        save_predictions_visualization(
            predictions=predictions,
            output_dir="./output_visualization",
            prefix="multi_view"
        )
        print()
        
        # for category in folders:
        #     if data_name.startswith("online_img"):
        #         image_names = [f for f in os.listdir(os.path.join(
        #             directory, category)) if f.endswith('.png') or f.endswith('.jpg')]
        #         try:
        #             image_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
        #         except:
        #             image_names.sort(key=lambda x: int(
        #                 x.split('_')[-1].split('.')[0]))
        #         image_names = [os.path.join(directory, category, f)
        #                        for f in image_names][:24]
        #     process(model, image_names, device,
        #             directory=f'{data_name}/fig1_update_dpg', name=f'dpg_{category}')

