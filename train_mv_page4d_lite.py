# In this one we test the performance on dynamic
import os
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from mv_page4d_lite.utils.pose_enc import pose_encoding_to_extri_intri
import numpy as np
from vggt.utils.load_fn import load_and_preprocess_images
import json
from utils.metrics import *
from utils.visual import *
from mv_page4d_lite.models.vggt import VGGT as VGGT_MV
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


def create_multi_view_time_grid(images, V, T, normalize=True):
    """
    Create a grid image with V rows (views) and T columns (time steps).
    
    Args:
        images: torch.Tensor [B, T, V, C, H, W] or [T, V, C, H, W]
        V: number of views
        T: number of time steps
        normalize: whether to normalize images to [0, 1] range
    
    Returns:
        grid_image: torch.Tensor [C, H*V, W*T] - grid image
    """
    # Handle batch dimension
    if len(images.shape) == 6:
        images = images[0]  # [T, V, C, H, W]
    elif len(images.shape) == 5:
        pass  # Already [T, V, C, H, W]
    else:
        raise ValueError(f"Expected images shape [T, V, C, H, W] or [B, T, V, C, H, W], got {images.shape}")
    
    # Ensure on CPU and detach
    images = images.detach().cpu()
    
    # Get dimensions
    T_actual, V_actual, C, H, W = images.shape
    T = min(T, T_actual)
    V = min(V, V_actual)
    
    # Normalize if needed
    if normalize:
        # Clamp to [0, 1] range
        images = torch.clamp(images, 0, 1)
    
    # Create grid: V rows, T columns
    grid_rows = []
    for v in range(V):
        row_images = []
        for t in range(T):
            img = images[t, v]  # [C, H, W]
            row_images.append(img)
        # Concatenate horizontally (T columns)
        row = torch.cat(row_images, dim=2)  # [C, H, W*T]
        grid_rows.append(row)
    # Concatenate vertically (V rows)
    grid = torch.cat(grid_rows, dim=1)  # [C, H*V, W*T]
    
    return grid


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


def print_model_structure(model, max_depth=3, indent=0):
    """
    打印模型的完整结构
    
    Args:
        model: PyTorch模型
        max_depth: 最大递归深度
        indent: 当前缩进级别
    """
    def _print_module(module, name="", depth=0, max_d=max_depth):
        if depth > max_d:
            return
        
        indent_str = "  " * depth
        module_type = type(module).__name__
        
        # 获取参数信息
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        
        if num_params > 0:
            status = "TRAINABLE" if trainable_params > 0 else "FROZEN"
            print(f"{indent_str}{name} ({module_type}): {num_params:,} params ({trainable_params:,} trainable) [{status}]")
        else:
            print(f"{indent_str}{name} ({module_type})")
        
        # 递归打印子模块
        for child_name, child_module in module.named_children():
            _print_module(child_module, child_name, depth + 1, max_d)
    
    print("\n" + "="*80)
    print("Model Structure:")
    print("="*80)
    _print_module(model, "model", depth=0, max_d=max_depth)
    print("="*80)
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    print("="*80 + "\n")


def load_pretrained_weights(model, checkpoint_path, device="cuda", verbose=True):
    """
    加载预训练权重，处理形状不匹配问题
    
    Args:
        model: PyTorch模型
        checkpoint_path: checkpoint文件路径
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含加载统计信息的字典
    """
    import os
    
    if not os.path.exists(checkpoint_path):
        if verbose:
            print(f"Warning: Checkpoint file {checkpoint_path} not found, using random initialization")
        return {
            'loaded': False,
            'missing_keys': [],
            'unexpected_keys': [],
            'size_mismatch_keys': []
        }
    
    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check checkpoint format
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
    else:
        model_state_dict = checkpoint
    
    # 获取模型当前的state_dict
    model_state = model.state_dict()
    
    # 过滤掉形状不匹配的键
    filtered_state_dict = {}
    size_mismatch_keys = []
    
    for key, value in model_state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                size_mismatch_keys.append({
                    'key': key,
                    'checkpoint_shape': list(value.shape),
                    'model_shape': list(model_state[key].shape)
                })
                if verbose:
                    print(f"  Skipping {key}: shape mismatch "
                          f"(checkpoint: {list(value.shape)}, model: {list(model_state[key].shape)})")
        else:
            # 键不存在于模型中（可能是新增的模块）
            pass
    
    # 加载过滤后的权重
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if verbose:
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Successfully loaded: {len(filtered_state_dict)} keys")
        
        if size_mismatch_keys:
            print(f"  Size mismatch (skipped): {len(size_mismatch_keys)} keys")
            for item in size_mismatch_keys:
                print(f"    - {item['key']}: checkpoint{item['checkpoint_shape']} vs model{item['model_shape']}")

        
        if missing_keys:
            print(f"  Missing keys (will use random init): {len(missing_keys)} keys")
            # 按模块分组显示
            missing_by_module = {}
            for key in missing_keys:
                module = key.split('.')[0] if '.' in key else 'root'
                if module not in missing_by_module:
                    missing_by_module[module] = []
                missing_by_module[module].append(key)
            
            for module, keys in missing_by_module.items():
                print(f"    [{module}]: {len(keys)} keys")
                for k in keys:
                    print(f"      - {k}")

        
        if unexpected_keys:
            print(f"  Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
            else:
                for key in unexpected_keys[:5]:
                    print(f"    - {key}")
                print(f"    ... and {len(unexpected_keys) - 5} more")
    
    return {
        'loaded': True,
        'loaded_keys': len(filtered_state_dict),
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'size_mismatch_keys': size_mismatch_keys
    }


def freeze_parameters_stage1(model):
    """
    阶段1：冻结基础特征，只训练新增模块（根据图片中的训练/冻结策略）
    
    冻结：
    - 编码器 (patch_embed)
    - Stage-1/3 (frame_blocks[0:8] 和 frame_blocks[18:24], global_blocks相同)
    - 解码头 (camera_head, depth_head, point_head权重)
    
    训练：
    - ViewMixer (新增的Stage-0模块)
    - Stage-2 (frame_blocks[8:18], global_blocks[8:18])
    - 路由门控 (camera_mask_gate, LoRA参数)
    - Token的可学习初值 (camera_token, register_token)
    - 位置/元信息嵌入 (time_embed, view_embed, camera_param_embed)
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # ========== 解冻需要训练的模块 ==========
    aggregator = model.aggregator
    
    # 1. ViewMixer (新增，Stage-0)
    if hasattr(aggregator, 'viewmixer'):
        for param in aggregator.viewmixer.parameters():
            param.requires_grad = True
    
    # 2. Stage-2 (10层: 8-17)
    stage2_indices = list(range(8, 18))
    for idx in stage2_indices:
        if idx < len(aggregator.frame_blocks):
            for param in aggregator.frame_blocks[idx].parameters():
                param.requires_grad = True
        if idx < len(aggregator.global_blocks):
            for param in aggregator.global_blocks[idx].parameters():
                param.requires_grad = True
    
    # 3. 路由门控：camera_mask_gate (可学习强度门控γ)
    if hasattr(aggregator, 'camera_mask_gate'):
        aggregator.camera_mask_gate.requires_grad = True
    
    # 4. Token的可学习初值
    if hasattr(aggregator, 'camera_token'):
        aggregator.camera_token.requires_grad = True
    if hasattr(aggregator, 'register_token'):
        aggregator.register_token.requires_grad = True
    
    # 5. 位置/元信息嵌入
    if hasattr(aggregator, 'time_embed'):
        for param in aggregator.time_embed.parameters():
            param.requires_grad = True
    if hasattr(aggregator, 'view_embed'):
        for param in aggregator.view_embed.parameters():
            param.requires_grad = True
    if hasattr(aggregator, 'camera_param_embed'):
        for param in aggregator.camera_param_embed.parameters():
            param.requires_grad = True
    
    # 6. Spatial Mask Head (如果需要)
    if hasattr(aggregator, 'spatial_mask_head'):
        for param in aggregator.spatial_mask_head.parameters():
            param.requires_grad = True
    
    # ViewMixer中的LoRA参数已经被包含在viewmixer中
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 1: Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def apply_freeze_train_strategy_precise(model):
    """
    按照图片中的精确指令进行参数冻结/训练划分（"最小侵入版"默认策略）
    
    冻结哪些参数：
    1. 图像编码器 (DINOv2/ViT-L/14): patch_embed - 全部冻结
    2. Stage-1 & Stage-3 的所有 Transformer 层参数: 
       - frame_blocks[0:8] (Stage-1: 8层)
       - frame_blocks[18:24] (Stage-3: 6层)
       - global_blocks 对应的层 - 全部冻结
    3. 深度头/点图头 (DPT-head): depth_head, point_head - 先冻结
    4. 相机头主体权重 (Camera Head main weights): camera_head.trunk, camera_head.pose_branch 等 - 冻结
    
    训练哪些参数：
    1. Stage-0: ViewMixer (跨视角轻量注意力门控残差/LoRA分支) - 全训
    2. Stage-2 (10层: frame_blocks[8:18], global_blocks[8:18]) - 全训
    3. 掩码门控 γ (camera_mask_gate): per-layer/per-branch可训，初始化接近0 - 训练
    4. 相机/注册 tokens 的可学习初值 (camera_token, register_token) - 可训
    5. CameraHead-shim (极薄映射/包装层, 可选): 将按视角的相机token映射到冻结相机头的接口 - 可训
    6. 动态掩码 head (spatial_mask_head: DWConv + τ/α) - 训练
    7. 时间/视角嵌入表 (time_embed, view_embed, camera_param_embed) - 可训
    
    可选细化：如果Stage-2显存/稳定性吃紧，可将QKV投影采用LoRA
    """
    # 1. 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    aggregator = model.aggregator
    
    # ========== 冻结部分 ==========
    print("\n[Freezing Parameters]")
    
    # 1.1 图像编码器 (DINOv2/ViT-L/14): patch_embed - 全部冻结
    if hasattr(aggregator, 'patch_embed'):
        for param in aggregator.patch_embed.parameters():
            param.requires_grad = False
        print("  ✓ Frozen: patch_embed (DINOv2/ViT encoder)")
    
    # 1.2 Stage-1 (8层: 0-7) & Stage-3 (6层: 18-23) 的所有 Transformer 层参数
    # Stage-1: frame_blocks[0:8], global_blocks[0:8]
    stage1_indices = list(range(0, 8))
    for idx in stage1_indices:
        if idx < len(aggregator.frame_blocks):
            for param in aggregator.frame_blocks[idx].parameters():
                param.requires_grad = False
        if idx < len(aggregator.global_blocks):
            for param in aggregator.global_blocks[idx].parameters():
                param.requires_grad = False
    
    # Stage-3: frame_blocks[18:24], global_blocks[18:24]
    stage3_indices = list(range(18, 24))
    for idx in stage3_indices:
        if idx < len(aggregator.frame_blocks):
            for param in aggregator.frame_blocks[idx].parameters():
                param.requires_grad = False
        if idx < len(aggregator.global_blocks):
            for param in aggregator.global_blocks[idx].parameters():
                param.requires_grad = False
    print(f"  ✓ Frozen: Stage-1 ({len(stage1_indices)} layers) & Stage-3 ({len(stage3_indices)} layers) Transformer blocks")
    
    # 1.3 深度头/点图头 (DPT-head): 先冻结（可在后期解冻做轻微收尾）
    if hasattr(model, 'depth_head') and model.depth_head is not None:
        for param in model.depth_head.parameters():
            param.requires_grad = False
        print("  ✓ Frozen: depth_head (DPT-head, can be unfrozen later for fine-tuning)")
    
    if hasattr(model, 'point_head') and model.point_head is not None:
        for param in model.point_head.parameters():
            param.requires_grad = False
        print("  ✓ Frozen: point_head (DPT-head, can be unfrozen later for fine-tuning)")
    
    # 1.4 相机头主体权重: 冻结主干（trunk, pose_branch等），但允许接口薄层可训
    if hasattr(model, 'camera_head') and model.camera_head is not None:
        camera_head = model.camera_head
        # 冻结相机头的主体权重
        if hasattr(camera_head, 'trunk'):
            for param in camera_head.trunk.parameters():
                param.requires_grad = False
        if hasattr(camera_head, 'pose_branch'):
            for param in camera_head.pose_branch.parameters():
                param.requires_grad = False
        if hasattr(camera_head, 'token_norm'):
            for param in camera_head.token_norm.parameters():
                param.requires_grad = False
        if hasattr(camera_head, 'trunk_norm'):
            for param in camera_head.trunk_norm.parameters():
                param.requires_grad = False
        if hasattr(camera_head, 'poseLN_modulation'):
            for param in camera_head.poseLN_modulation.parameters():
                param.requires_grad = False
        if hasattr(camera_head, 'adaln_norm'):
            for param in camera_head.adaln_norm.parameters():
                param.requires_grad = False
        if hasattr(camera_head, 'embed_pose'):
            for param in camera_head.embed_pose.parameters():
                param.requires_grad = False
        if hasattr(camera_head, 'empty_pose_tokens'):
            camera_head.empty_pose_tokens.requires_grad = False
        print("  ✓ Frozen: camera_head main weights (trunk, pose_branch, etc.)")
        
        # Unfreeze view-specific projection layers (shim) for multi-view differentiation
        if hasattr(camera_head, 'view_proj'):
            for proj in camera_head.view_proj:
                for param in proj.parameters():
                    param.requires_grad = True
            print("  ✓ Training: camera_head.view_proj (view-specific shim layers)")
    
    # ========== 训练部分 ==========
    print("\n[Training Parameters]")
    
    # 2.1 Stage-0: ViewMixer - 全训（跨视角轻量注意力门控残差/LoRA分支）
    if hasattr(aggregator, 'viewmixer'):
        for param in aggregator.viewmixer.parameters():
            param.requires_grad = True
        print("  ✓ Training: ViewMixer (Stage-0, cross-view attention with LoRA)")
    
    # 2.2 Stage-2 (10层: 8-17) - 全训（掩码化 Global + 时序 Frame 的主干）
    stage2_indices = list(range(8, 18))
    for idx in stage2_indices:
        if idx < len(aggregator.frame_blocks):
            for param in aggregator.frame_blocks[idx].parameters():
                param.requires_grad = True
        if idx < len(aggregator.global_blocks):
            for param in aggregator.global_blocks[idx].parameters():
                param.requires_grad = True
    print(f"  ✓ Training: Stage-2 ({len(stage2_indices)} layers: frame_blocks[8:18], global_blocks[8:18])")
    
    # 2.3 掩码门控 γ (camera_mask_gate): per-layer/per-branch可训，初始化接近0
    if hasattr(aggregator, 'camera_mask_gate'):
        aggregator.camera_mask_gate.requires_grad = True
        # 确保初始化接近0（弱抑制）
        with torch.no_grad():
            aggregator.camera_mask_gate.fill_(0.0)
        print("  ✓ Training: camera_mask_gate (γ, initialized to 0 for weak suppression)")
    
    # 2.4 相机/注册 tokens 的可学习初值
    if hasattr(aggregator, 'camera_token'):
        aggregator.camera_token.requires_grad = True
        print("  ✓ Training: camera_token (learnable initial values)")
    if hasattr(aggregator, 'register_token'):
        aggregator.register_token.requires_grad = True
        print("  ✓ Training: register_token (learnable initial values)")
    
    # 2.5 CameraHead-shim (极薄映射/包装层, 可选)
    # 注意：这需要单独实现，用于将按视角的相机token映射到冻结相机头的接口
    # 如果存在，将其设置为可训练
    # 示例：if hasattr(model, 'camera_head_shim'): ...
    # 当前代码中可能没有，可以先跳过或后续添加
    print("  ⚠ Note: CameraHead-shim (thin mapping layer) - can be added if needed")
    
    # 2.6 动态掩码 head (spatial_mask_head: DWConv + τ/α)
    if hasattr(aggregator, 'spatial_mask_head'):
        for param in aggregator.spatial_mask_head.parameters():
            param.requires_grad = True
        print("  ✓ Training: spatial_mask_head (DWConv + τ/α, dynamic masking)")
    
    # 2.7 时间/视角嵌入表
    if hasattr(aggregator, 'time_embed'):
        for param in aggregator.time_embed.parameters():
            param.requires_grad = True
        print("  ✓ Training: time_embed (learnable time encoding)")
    if hasattr(aggregator, 'view_embed'):
        for param in aggregator.view_embed.parameters():
            param.requires_grad = True
        print("  ✓ Training: view_embed (learnable view encoding)")
    if hasattr(aggregator, 'camera_param_embed'):
        for param in aggregator.camera_param_embed.parameters():
            param.requires_grad = True
        print("  ✓ Training: camera_param_embed (learnable camera parameter encoding)")
    
    # 2.8 可选细化：如果Stage-2显存/稳定性吃紧，可将QKV投影采用LoRA
    # 注意：当前实现可能没有LoRA版本的Stage-2，如果需要可以后续添加
    print("  ⚠ Note: Optional refinement - Stage-2 QKV can use LoRA if memory/stability is tight")
    
    # ========== 统计可训练参数 ==========
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
    
    print(f"\n[Summary]")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  Frozen parameters: {total_params - trainable_params:,} ({100 - trainable_percent:.2f}%)")
    
    # 验证关键组件
    print(f"\n[Verification]")
    if hasattr(aggregator, 'viewmixer'):
        viewmixer_trainable = sum(p.numel() for p in aggregator.viewmixer.parameters() if p.requires_grad)
        print(f"  ViewMixer trainable: {viewmixer_trainable:,} params")
    
    stage2_frame_trainable = sum(
        p.numel() for idx in stage2_indices 
        if idx < len(aggregator.frame_blocks)
        for p in aggregator.frame_blocks[idx].parameters() if p.requires_grad
    )
    stage2_global_trainable = sum(
        p.numel() for idx in stage2_indices 
        if idx < len(aggregator.global_blocks)
        for p in aggregator.global_blocks[idx].parameters() if p.requires_grad
    )
    print(f"  Stage-2 trainable: frame={stage2_frame_trainable:,}, global={stage2_global_trainable:,} params")


# ============================================================================
# 损失函数配置与扩展接口
# ============================================================================

class LossConfig:
    """
    损失函数配置类：控制哪些损失被启用，以及它们的权重
    使用最小必要配置作为默认，同时预留扩展接口
    """
    def __init__(self, use_minimal=True):
        """
        Args:
            use_minimal: 如果True，使用最小必要配置；如果False，使用完整配置
        """
        # 默认使用最小必要配置
        if use_minimal:
            self.load_minimal_config()
        else:
            self.load_full_config()
    
    def load_minimal_config(self):
        """最小必要配置：只保留核心必要损失和重要损失"""
        # ========== 核心必要损失（必须） ==========
        self.use_mask_ce = True           # L_mask_ce: 掩码监督（BCEWithLogits）
        self.use_depth_point = True       # L_depth->point^S: 深度一致性（静态区，尺度归一化）
        self.use_cam_const = True         # L_cam^const: 相机跨时恒定（SE(3)李代数差分）
        
        # ========== 重要损失（强烈推荐） ==========
        self.use_smooth_edge = True       # L_smooth^S_edge: 边缘感知几何平滑（静态区）
        self.use_mv_geo = True             # L_mv_geo^S: 跨视角几何重投影（当前占位符，但保留接口）
        
        # ========== 可选损失（不参与训练） ==========
        self.use_mask_tv = False          # L_mask_tv: 掩码边界/时序
        self.use_uncert = False           # L_uncert: 置信度正则（可选）
        self.use_scale = False            # L_scale^S: 尺度一致（可选）
        self.use_sep = False               # L_sep: 双流分离（可选，需要dual_stream_outputs）
        
        # ========== 高级损失（占位符，不参与训练） ==========
        self.use_epi = False               # L_epi^S: 对极约束（占位符）
        self.use_photo = False             # L_photo^S: 光度一致性（占位符）
        
        # ========== 关键点损失（需要标注，不参与训练） ==========
        self.use_kpt = False               # L_kpt^D: 关键点重投影
        self.use_temp_d = False            # L_temp^D: 关键点时间平滑
        
        # 权重配置（按照图片中的初始权重）
        self.weights = {
            'mask_ce': 1.0,
            'mask_tv': 0.2,
            'depth_point': 0.5,
            'scale': 0.1,
            'smooth_edge': 0.2,
            'uncert': 0.1,
            'cam_const': 0.1,
            'sep': 0.1,
            'mv_geo': 1.0,
            'epi': 1.0,
            'photo': 0.2,
            'kpt': 2.0,
            'temp_d': 0.5,
        }
    
    def load_full_config(self):
        """完整配置：启用所有已实现的损失（除了占位符和需要标注的）"""
        # 必要损失
        self.use_mask_ce = True
        self.use_depth_point = True
        self.use_cam_const = True
        
        # 推荐损失
        self.use_smooth_edge = True
        self.use_uncert = True
        
        # 可选损失（启用）
        self.use_mask_tv = True
        self.use_scale = True
        self.use_sep = True
        
        # 高级损失（占位符保持关闭，除非已实现）
        self.use_mv_geo = False
        self.use_epi = False
        self.use_photo = False
        
        # 关键点损失（需要标注）
        self.use_kpt = False
        self.use_temp_d = False
        
        # 权重配置（同上）
        self.weights = {
            'mask_ce': 1.0,
            'mask_tv': 0.2,
            'depth_point': 0.5,
            'scale': 0.1,
            'smooth_edge': 0.2,
            'uncert': 0.1,
            'cam_const': 0.1,
            'sep': 0.1,
            'mv_geo': 1.0,
            'epi': 1.0,
            'photo': 0.2,
            'kpt': 2.0,
            'temp_d': 0.5,
        }
    
    def enable_loss(self, loss_name: str, weight: float = None):
        """启用指定的损失函数"""
        if hasattr(self, f'use_{loss_name}'):
            setattr(self, f'use_{loss_name}', True)
            if weight is not None:
                self.weights[loss_name] = weight
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def disable_loss(self, loss_name: str):
        """禁用指定的损失函数"""
        if hasattr(self, f'use_{loss_name}'):
            setattr(self, f'use_{loss_name}', False)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def set_weight(self, loss_name: str, weight: float):
        """设置损失权重"""
        if loss_name in self.weights:
            self.weights[loss_name] = weight
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def get_active_losses(self):
        """返回当前激活的损失列表"""
        active = []
        for attr_name in dir(self):
            if attr_name.startswith('use_'):
                if getattr(self, attr_name):
                    loss_name = attr_name[4:]  # 去掉'use_'前缀
                    active.append((loss_name, self.weights.get(loss_name, 0.0)))
        return active
    
    def print_config(self):
        """打印当前配置"""
        print("\n" + "="*60)
        print("损失函数配置")
        print("="*60)
        active = self.get_active_losses()
        print(f"激活的损失函数 ({len(active)} 个):")
        for loss_name, weight in active:
            print(f"  ✓ {loss_name}: 权重 {weight}")
        print()
        inactive = []
        for attr_name in dir(self):
            if attr_name.startswith('use_') and not getattr(self, attr_name):
                loss_name = attr_name[4:]
                inactive.append(loss_name)
        if inactive:
            print(f"未激活的损失函数 ({len(inactive)} 个):")
            for loss_name in inactive:
                print(f"  - {loss_name}")
        print("="*60)


# ============================================================================
# 损失函数辅助工具
# ============================================================================

def off_diagonal(x):
    """返回方阵的非对角线元素（展平）"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_param: float = 5e-3) -> torch.Tensor:
    """Barlow Twins去相关损失"""
    # 归一化
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
    
    # 交叉相关矩阵
    c = (z1_norm.T @ z2_norm) / z1_norm.shape[0]
    
    # 损失：对角线接近1，非对角线接近0
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    
    return on_diag + lambda_param * off_diag


def compute_boundary_loss(mask_logits: torch.Tensor) -> torch.Tensor:
    """掩码边界锐利化损失（基于梯度）"""
    # 处理不同形状：统一转换为 [B*T*V, H, W] 或类似格式
    original_shape = mask_logits.shape
    if mask_logits.ndim == 5:  # [B, T, V, H, W]
        mask_logits = mask_logits.view(-1, *mask_logits.shape[-2:])  # [B*T*V, H, W]
    elif mask_logits.ndim == 4:  # [B, T, H, W] 或 [B, H, W, C]
        if mask_logits.shape[-1] == 1:
            mask_logits = mask_logits.squeeze(-1)  # [B, T, H, W, 1] -> [B, T, H, W]
        mask_logits = mask_logits.view(-1, *mask_logits.shape[-2:])  # [B*T, H, W]
    
    # 计算梯度（拉普拉斯算子）
    dx = torch.abs(mask_logits[:, :, 1:] - mask_logits[:, :, :-1])  # [..., H, W-1]
    dy = torch.abs(mask_logits[:, 1:, :] - mask_logits[:, :-1, :])  # [..., H-1, W]
    
    return (dx.mean() + dy.mean()) * 0.1


def compute_temporal_consistency_loss_mask(mask_logits: torch.Tensor, static_mask: torch.Tensor = None) -> torch.Tensor:
    """掩码时间一致性损失（简化版，需要warp函数但这里简化）"""
    # 处理不同形状：mask_logits应该是 [B, T, V, H, W] 或类似
    if mask_logits.ndim == 5:
        B, T, V, H, W = mask_logits.shape
        if T < 2:
            return torch.tensor(0.0, device=mask_logits.device)
        
        loss = torch.tensor(0.0, device=mask_logits.device)
        for i in range(T - 1):
            diff = torch.abs(mask_logits[:, i] - mask_logits[:, i+1])  # [B, V, H, W]
            if static_mask is not None:
                if static_mask.ndim == 5:  # [B, T, V, H, W]
                    diff = diff * static_mask[:, i]
                elif static_mask.ndim == 4:  # [B, V, H, W]
                    diff = diff * static_mask
            loss += diff.mean()
        return loss / (T - 1) * 0.05
    elif mask_logits.ndim == 4:
        # [B, T, H, W]
        B, T, H, W = mask_logits.shape
        if T < 2:
            return torch.tensor(0.0, device=mask_logits.device)
        
        loss = torch.tensor(0.0, device=mask_logits.device)
        for i in range(T - 1):
            diff = torch.abs(mask_logits[:, i] - mask_logits[:, i+1])
            if static_mask is not None:
                if static_mask.ndim == 4 and static_mask.shape[1] == T:
                    diff = diff * static_mask[:, i]
                elif static_mask.ndim == 3:
                    diff = diff * static_mask
            loss += diff.mean()
        return loss / (T - 1) * 0.05
    else:
        return torch.tensor(0.0, device=mask_logits.device)


def compute_entropy_regularization_loss(confidence: torch.Tensor) -> torch.Tensor:
    """异方差不确定性损失：exp(-s) * |e| + s, 其中s=log σ"""
    # 假设confidence已经是log-uncertainty或直接作为uncertainty
    # 这里简化：使用熵正则化
    epsilon = 1e-6
    confidence_clamped = torch.clamp(confidence, epsilon, 1.0 - epsilon)
    entropy = -confidence_clamped * torch.log(confidence_clamped + epsilon) - (1 - confidence_clamped) * torch.log(1 - confidence_clamped + epsilon)
    return -entropy.mean() * 0.01  # 最大化熵（最小化负熵）


def compute_se3_lie_algebra_loss(pose_enc_list: list) -> torch.Tensor:
    """SE(3)李代数差分损失（简化版）"""
    if not pose_enc_list or len(pose_enc_list) < 2:
        return torch.tensor(0.0, device=pose_enc_list[0].device if pose_enc_list else 'cuda')
    
    pose_enc = pose_enc_list[-1]  # [B, V, 9]
    loss = torch.tensor(0.0, device=pose_enc.device)
    
    # 简化：计算相邻迭代的差异作为代理
    for i in range(len(pose_enc_list) - 1):
        diff = torch.mean((pose_enc_list[i] - pose_enc_list[i+1]) ** 2)
        loss += diff
    
    return loss / max(len(pose_enc_list) - 1, 1) * 0.1


def compute_mask_supervision_loss(predictions, segmask_gt, images, loss_config=None):
    """
    L_mask_ce (Focal/BCEWithLogits) + L_mask_tv (掩码边界/时序)
    按照图片指令：使用logits监督，添加边界和时间一致性（支持配置控制）
    
    Args:
        predictions: 模型输出字典，需要包含mask_logits
        segmask_gt: SegAnyMo mask [B, T, V, H, W], 值域[0, 1], 1表示动态区域
        images: 输入图像 [B, T, V, C, H, W]
        loss_config: LossConfig实例，控制哪些损失启用
    
    Returns:
        loss: 标量tensor, loss_dict: 字典
    """
    if loss_config is None:
        loss_config = LossConfig(use_minimal=True)
    
    if segmask_gt is None or not loss_config.use_mask_ce:
        return torch.tensor(0.0, device=list(predictions.values())[0].device, requires_grad=True), {
            'mask_loss': 0.0, 'mask_ce_loss': 0.0, 'mask_boundary_loss': 0.0, 'mask_temporal_loss': 0.0
        }
    
    B, T, V, C, H, W = images.shape
    device = images.device
    
    # 提取mask_logits（如果不存在，尝试从predictions中获取）
    if 'mask_logits' in predictions:
        mask_logits = predictions['mask_logits']
    elif 'dynamic_mask' in predictions:
        mask_logits = predictions['dynamic_mask']
    else:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'mask_loss': 0.0, 'mask_ce_loss': 0.0, 'mask_boundary_loss': 0.0, 'mask_temporal_loss': 0.0
        }
    
    try:
        # 处理mask_logits的不同可能形状，统一转换为 [B, T, V, H, W]
        if mask_logits.ndim == 4:
            # [B, T, H, W] 或 [B, H, W, C] - 需要判断
            if mask_logits.shape[1] == T:
                mask_logits = mask_logits.unsqueeze(2)  # [B, T, H, W] -> [B, T, 1, H, W]
            else:
                # 可能是 [B, H, W, 1]，需要reshape
                mask_logits = mask_logits.unsqueeze(1).unsqueeze(1)  # [B, H, W, 1] -> [B, 1, 1, H, W]
        elif mask_logits.ndim == 5:
            if mask_logits.shape[2] == 1 and mask_logits.shape[1] == T:
                # [B, T, 1, H, W] -> [B, T, V, H, W] (broadcast V)
                mask_logits = mask_logits.expand(-1, -1, V, -1, -1)
            elif mask_logits.shape[1] != T or mask_logits.shape[2] != V:
                # 形状不匹配，尝试推断
                if mask_logits.shape[-1] == 1:
                    mask_logits = mask_logits.squeeze(-1)  # [B, T, V, H, W, 1] -> [B, T, V, H, W]
        elif mask_logits.ndim == 6:
            # [B, T, V, 1, H, W] 或 [B, T, V, H, W, 1]
            if mask_logits.shape[-1] == 1:
                mask_logits = mask_logits.squeeze(-1)  # [B, T, V, H, W, 1] -> [B, T, V, H, W]
            elif mask_logits.shape[-3] == 1:
                mask_logits = mask_logits.squeeze(-3)  # [B, T, V, 1, H, W] -> [B, T, V, H, W]
        
        # 确保最终形状是 [B, T, V, H, W]
        if mask_logits.ndim != 5 or mask_logits.shape[:3] != (B, T, V):
            # 尝试reshape
            total_elements = mask_logits.numel()
            expected_elements = B * T * V * H * W
            if total_elements == expected_elements:
                mask_logits = mask_logits.reshape(B, T, V, H, W)
            else:
                raise ValueError(f"Cannot reshape mask_logits from {mask_logits.shape} to [B={B}, T={T}, V={V}, H={H}, W={W}]")
        
        # Resize到匹配（如果空间维度不一致）
        mask_h, mask_w = mask_logits.shape[-2:]
        if mask_h != H or mask_w != W:
            mask_logits = F.interpolate(
                mask_logits.view(B*T*V, 1, mask_h, mask_w),
                size=(H, W), mode='bilinear', align_corners=False
            ).view(B, T, V, H, W)
        
        # 同样处理segmask_gt
        if segmask_gt.shape[-2:] != (H, W):
            segmask_gt = F.interpolate(
                segmask_gt.view(B*T*V, 1, *segmask_gt.shape[-2:]),
                size=(H, W), mode='nearest'
            ).view(B, T, V, H, W)
    except Exception as e:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'mask_loss': 0.0, 'mask_ce_loss': 0.0, 'mask_boundary_loss': 0.0, 'mask_temporal_loss': 0.0
        }
    
    # L_mask_ce: BCEWithLogits (权重1.0)
    segmask_gt_flat = segmask_gt.view(-1)
    mask_logits_flat = mask_logits.view(-1)
    mask_ce_loss = F.binary_cross_entropy_with_logits(mask_logits_flat, segmask_gt_flat.float())
    
    total_mask_loss = mask_ce_loss * loss_config.weights['mask_ce']
    
    # L_mask_tv: 边界和时间一致性 (权重0.2)
    if loss_config.use_mask_tv:
        static_mask = (1 - segmask_gt).bool()  # [B, T, V, H, W]
        mask_boundary_loss = compute_boundary_loss(mask_logits)
        mask_temporal_loss = compute_temporal_consistency_loss_mask(mask_logits, static_mask)
        total_mask_loss = total_mask_loss + (mask_boundary_loss + mask_temporal_loss) * loss_config.weights['mask_tv']
        mask_boundary_loss_val = mask_boundary_loss.item()
        mask_temporal_loss_val = mask_temporal_loss.item()
    else:
        mask_boundary_loss_val = 0.0
        mask_temporal_loss_val = 0.0
    
    return total_mask_loss, {
        'mask_loss': total_mask_loss.item(),
        'mask_ce_loss': mask_ce_loss.item(),
        'mask_boundary_loss': mask_boundary_loss_val,
        'mask_temporal_loss': mask_temporal_loss_val
    }


def compute_camera_temporal_consistency_loss(predictions, T, V, loss_config=None):
    """
    L_cam^const (相机跨时恒定/方差正则, 权重0.1)
    按照图片指令：使用SE(3)李代数差分替代跨迭代差异（支持配置控制）
    
    Args:
        predictions: 模型输出字典
        T: 时间步数
        V: 视角数
        loss_config: LossConfig实例
    """
    if loss_config is None:
        loss_config = LossConfig(use_minimal=True)
    
    if not loss_config.use_cam_const or 'pose_enc_list' not in predictions:
        device = 'cpu'
        if 'images' in predictions:
            device = predictions['images'].device
        elif 'depth' in predictions:
            device = predictions['depth'].device
        return torch.tensor(0.0, device=device, requires_grad=True), {'L_cam_const': 0.0}
    
    try:
        pose_enc_list = predictions['pose_enc_list']
        
        # 处理pose_enc_list的不同可能形状
        # 可能是 [B, V, 9] 或 list of [B, V, 9]
        if isinstance(pose_enc_list, list):
            if len(pose_enc_list) == 0:
                return torch.tensor(0.0, device=list(predictions.values())[0].device, requires_grad=True), {'L_cam_const': 0.0}
            pose_enc = pose_enc_list[-1]  # 使用最后一个迭代的结果
        else:
            pose_enc = pose_enc_list  # 直接是tensor
        
        # pose_enc应该是 [B, V, 9]（多视角）或 [B, T, 9]（单视角时序）
        if len(pose_enc.shape) == 3:
            if pose_enc.shape[1] == V:
                # [B, V, 9] - 多视角，每个视角的相机参数
                # 计算跨视角的一致性（相机参数应该在所有视角间是物理一致的）
                if V > 1:
                    # DEBUG: Print camera token differences to diagnose why they are identical
                    # Check every batch to track improvements
                    view_diffs = []
                    view_diffs_per_dim = {}
                    for v1 in range(V):
                        for v2 in range(v1 + 1, V):
                            diff_per_dim = (pose_enc[0, v1, :] - pose_enc[0, v2, :]).abs()
                            max_diff = diff_per_dim.max().item()
                            mean_diff = diff_per_dim.mean().item()
                            view_diffs.append(max_diff)
                            view_diffs_per_dim[(v1, v2)] = (max_diff, mean_diff)
                            if max_diff < 1e-5:  # Values are essentially identical
                                print(f"[WARNING] Camera params: view {v1} vs {v2} are nearly identical (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                    
                    if len(view_diffs) > 0:
                        avg_diff = sum(view_diffs) / len(view_diffs)
                        max_diff_all = max(view_diffs)
                        min_diff_all = min(view_diffs)
                        # Print summary every 10 iterations
                        if hasattr(compute_camera_temporal_consistency_loss, '_iter_count'):
                            compute_camera_temporal_consistency_loss._iter_count += 1
                        else:
                            compute_camera_temporal_consistency_loss._iter_count = 1
                        
                        # if compute_camera_temporal_consistency_loss._iter_count % 10 == 0:
                        #     print(f"[DEBUG] Camera param differences: avg={avg_diff:.6f}, max={max_diff_all:.6f}, min={min_diff_all:.6f}")
                        #     # Print per-dimension differences for first view pair
                        #     if (0, 1) in view_diffs_per_dim:
                        #         max_d, mean_d = view_diffs_per_dim[(0, 1)]
                        #         print(f"  View 0 vs 1: max={max_d:.6f}, mean={mean_d:.6f}")
                        #         # Print actual values for debugging
                        #         print(f"  View 0 sample: {pose_enc[0, 0, :3].detach().cpu().numpy()}")
                        #         print(f"  View 1 sample: {pose_enc[0, 1, :3].detach().cpu().numpy()}")
                    
                    # 计算所有视角对之间的差异
                    cam_diff = 0.0
                    count = 0
                    for v1 in range(V):
                        for v2 in range(v1 + 1, V):
                            diff = torch.mean((pose_enc[:, v1, :] - pose_enc[:, v2, :]) ** 2)
                            cam_diff += diff
                            count += 1
                    if count > 0:
                        cam_const_loss = (cam_diff / count) * loss_config.weights['cam_const']
                    else:
                        cam_const_loss = torch.tensor(0.0, device=pose_enc.device, requires_grad=True)
                else:
                    cam_const_loss = torch.tensor(0.0, device=pose_enc.device, requires_grad=True)
            elif pose_enc.shape[1] == T:
                # [B, T, 9] - 单视角时序，计算跨时间一致性
                cam_const_loss = compute_se3_lie_algebra_loss([pose_enc]) * loss_config.weights['cam_const']
            else:
                # 未知形状
                cam_const_loss = torch.tensor(0.0, device=pose_enc.device, requires_grad=True)
        else:
            cam_const_loss = compute_se3_lie_algebra_loss([pose_enc]) * loss_config.weights['cam_const']
        
        return cam_const_loss, {'L_cam_const': cam_const_loss.item() if isinstance(cam_const_loss, torch.Tensor) else cam_const_loss}
    except Exception as e:
        device = 'cpu'
        if 'images' in predictions:
            device = predictions['images'].device
        elif 'depth' in predictions:
            device = predictions['depth'].device
        return torch.tensor(0.0, device=device, requires_grad=True), {'L_cam_const': 0.0}


def compute_multi_view_consistency_loss(predictions, images, segmask_gt=None, camera_params=None, loss_config=None):
    """
    按照图片指令重写的多视角一致性损失（支持配置控制）
    
    Args:
        predictions: 模型输出字典
        images: 输入图像 [B, T, V, C, H, W]
        segmask_gt: SegAnyMo mask [B, T, V, H, W], 1表示动态区域
        camera_params: 相机参数 [B, V, 9] (pose_enc)
        loss_config: LossConfig实例，控制哪些损失启用
    
    Returns:
        total_loss, loss_dict
    """
    if loss_config is None:
        # 默认使用最小必要配置
        loss_config = LossConfig(use_minimal=True)
    
    B, T, V, C, H, W = images.shape
    device = images.device
    
    # 静态区掩码 S = (1 - segmask_gt)
    static_mask = None
    if segmask_gt is not None:
        static_mask = (1 - segmask_gt.float()).bool()  # [B, T, V, H, W]
    
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # L_depth->point^S (深度与点图一致, 静态, 权重0.5)
    if loss_config.use_depth_point and 'depth' in predictions and 'world_points' in predictions:
        try:
            depth = predictions['depth']  # [B, T, V, 1, H, W] 或 [B, T, V, H, W, 1]
            world_points = predictions['world_points']  # [B, T, V, H, W, 3] 或 [B, T, V, 3, H, W]
            
            # 处理world_points的不同可能形状
            if world_points.shape[-1] == 3:
                # world_points是 [B, T, V, H, W, 3]，提取z坐标
                depth_from_points = world_points[..., 2]  # [B, T, V, H, W]
                depth_from_points = depth_from_points.unsqueeze(-3)  # [B, T, V, 1, H, W]
            elif len(world_points.shape) >= 5 and world_points.shape[-3] == 3:
                # world_points是 [B, T, V, 3, H, W]，提取z坐标
                depth_from_points = world_points[..., 2, :, :]  # [B, T, V, H, W]
                depth_from_points = depth_from_points.unsqueeze(-3)  # [B, T, V, 1, H, W]
            else:
                # 未知形状，跳过
                loss_dict['L_depth_point'] = 0.0
                if loss_config.use_depth_point:
                    print(f"Warning: Unknown world_points shape {world_points.shape}, skipping L_depth_point")
                depth_from_points = None
            
            if depth_from_points is not None:
                # 处理depth的不同可能形状
                if len(depth.shape) == 6 and depth.shape[-1] == 1:
                    # depth是 [B, T, V, H, W, 1]，转换为 [B, T, V, 1, H, W]
                    depth = depth.squeeze(-1).unsqueeze(-3)
                elif len(depth.shape) == 5 and depth.shape[-3] != 1:
                    # depth是 [B, T, V, H, W]，添加通道维
                    depth = depth.unsqueeze(-3)  # [B, T, V, 1, H, W]
                
                # 确保depth和depth_from_points形状一致
                if depth.shape != depth_from_points.shape:
                    # 尝试对齐形状
                    if depth.shape[-2:] != depth_from_points.shape[-2:]:
                        # 空间维度不匹配，尝试resize
                        target_h, target_w = depth.shape[-2:]
                        depth_from_points = F.interpolate(
                            depth_from_points.view(B * T * V, 1, *depth_from_points.shape[-2:]),
                            size=(target_h, target_w), mode='bilinear', align_corners=False
                        ).view(B, T, V, 1, target_h, target_w)
                    # 确保通道维一致
                    if depth.shape[-3] != depth_from_points.shape[-3]:
                        if depth.shape[-3] == 1:
                            depth_from_points = depth_from_points.mean(dim=-3, keepdim=True)
                
                # 尺度归一化（简化：使用中位数归一化）
                depth_median = torch.median(depth[depth > 0]) if (depth > 0).sum() > 0 else torch.tensor(1.0, device=device)
                wp_depth_median = torch.median(depth_from_points[depth_from_points > 0]) if (depth_from_points > 0).sum() > 0 else torch.tensor(1.0, device=device)
                
                depth_norm = depth / (depth_median + 1e-6)
                wp_depth_norm = depth_from_points / (wp_depth_median + 1e-6)
                
                valid_mask = (depth > 0) & (depth_from_points > 0)  # [B, T, V, 1, H, W]
                if static_mask is not None:
                    # static_mask是 [B, T, V, H, W]，需要unsqueeze到 [B, T, V, 1, H, W]
                    static_mask_expanded = static_mask.unsqueeze(-3)  # [B, T, V, 1, H, W]
                    # 如果空间维度不匹配，需要resize static_mask
                    if static_mask_expanded.shape[-2:] != valid_mask.shape[-2:]:
                        static_mask_expanded = F.interpolate(
                            static_mask_expanded.float().view(B * T * V, 1, *static_mask_expanded.shape[-2:]),
                            size=valid_mask.shape[-2:], mode='nearest'
                        ).bool().view(B, T, V, 1, *valid_mask.shape[-2:])
                    valid_mask = valid_mask & static_mask_expanded
                
                if valid_mask.sum() > 0:
                    depth_point_diff = torch.abs(depth_norm - wp_depth_norm) * valid_mask.float()
                    L_depth_point = depth_point_diff.sum() / valid_mask.sum().clamp(min=1)
                    loss_dict['L_depth_point'] = L_depth_point.item() if isinstance(L_depth_point, torch.Tensor) else L_depth_point
                    total_loss = total_loss + L_depth_point * loss_config.weights['depth_point']
                else:
                    loss_dict['L_depth_point'] = 0.0
        except Exception as e:
            loss_dict['L_depth_point'] = 0.0
            if loss_config.use_depth_point:
                print(f"Warning: Error computing L_depth_point: {e}, skipping")
    else:
        loss_dict['L_depth_point'] = 0.0
    
    # L_mv_geo^S (跨视角几何重投影, 静态, 权重1.0) - 简化版代理（需要完整重投影实现）
    # 注意：已删除错误的 wp_view_diff 直接相减
    if loss_config.use_mv_geo and 'depth' in predictions and 'world_points' in predictions and V > 1:
        # 简化代理：计算静态区深度在视角间的一致性（需要相机参数做重投影）
        L_mv_geo = torch.tensor(0.0, device=device, requires_grad=True)  # 占位符
        # TODO: 实现完整的3D重投影损失
        loss_dict['L_mv_geo'] = L_mv_geo.item()
        total_loss = total_loss + L_mv_geo * loss_config.weights['mv_geo']
    else:
        loss_dict['L_mv_geo'] = 0.0
    
    # L_epi^S (对极约束, 静态, 权重1.0) - 需要相机参数和对应点
    if loss_config.use_epi and camera_params is not None and static_mask is not None:
        # 简化代理：对极约束需要基础矩阵E和对应点
        L_epi = torch.tensor(0.0, device=device, requires_grad=True)  # 占位符
        # TODO: 实现对极约束损失 |x'T E x|
        loss_dict['L_epi'] = L_epi.item()
        total_loss = total_loss + L_epi * loss_config.weights['epi']
    else:
        loss_dict['L_epi'] = 0.0
    
    # L_smooth^S_edge (边缘感知几何平滑, 静态, 权重0.2)
    if loss_config.use_smooth_edge and 'world_points' in predictions and static_mask is not None:
        try:
            world_points = predictions['world_points']  # 可能是 [B, T, V, H, W, 3] 或 [B, T, V, 3, H, W]
            images_gray = images.mean(dim=-3)  # [B, T, V, H, W]
            
            # 处理world_points的不同可能形状，统一转换为 [B, T, V, H, W, 3]
            if world_points.shape[-1] == 3:
                # world_points是 [B, T, V, H, W, 3]，保持不变
                wp_shape = world_points.shape  # [B, T, V, H, W, 3]
            elif len(world_points.shape) >= 6 and world_points.shape[-3] == 3:
                # world_points是 [B, T, V, 3, H, W]，转换为 [B, T, V, H, W, 3]
                world_points = world_points.permute(0, 1, 2, 4, 5, 3)  # [B, T, V, H, W, 3]
                wp_shape = world_points.shape
            else:
                # 未知形状，跳过
                loss_dict['L_smooth_edge'] = 0.0
                if loss_config.use_smooth_edge:
                    print(f"Warning: Unknown world_points shape {world_points.shape}, skipping L_smooth_edge")
                world_points = None
            
            if world_points is not None:
                # 确保world_points和images_gray的空间维度一致
                wp_h, wp_w = world_points.shape[-3:-1]  # 从 [..., H, W, 3] 提取
                img_h, img_w = images_gray.shape[-2:]  # [B, T, V, H, W]
                
                if wp_h != img_h or wp_w != img_w:
                    # 需要resize world_points或images_gray
                    target_h, target_w = img_h, img_w
                    # Resize world_points到target size
                    wp_flat = world_points.reshape(B * T * V, *world_points.shape[-3:])  # [B*T*V, H, W, 3]
                    wp_permuted = wp_flat.permute(0, 3, 1, 2)  # [B*T*V, 3, H, W] for F.interpolate
                    wp_resized = F.interpolate(
                        wp_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False
                    )
                    world_points = wp_resized.permute(0, 2, 3, 1).reshape(B, T, V, target_h, target_w, 3)
                
                # 图像梯度
                img_grad_x = torch.abs(images_gray[:, :, :, :, 1:] - images_gray[:, :, :, :, :-1])  # [B, T, V, H, W-1]
                img_grad_y = torch.abs(images_gray[:, :, :, 1:, :] - images_gray[:, :, :, :-1, :])  # [B, T, V, H-1, W]
                
                # 几何梯度（world_points现在是 [B, T, V, H, W, 3]）
                wp_grad_x = world_points[:, :, :, :, 1:, :] - world_points[:, :, :, :, :-1, :]  # [B, T, V, H, W-1, 3]
                wp_grad_y = world_points[:, :, :, 1:, :, :] - world_points[:, :, :, :-1, :, :]  # [B, T, V, H-1, W, 3]
                
                # 边缘感知权重：exp(-κ|∂I|)
                kappa = 10.0
                weight_x = torch.exp(-kappa * img_grad_x)  # [B, T, V, H, W-1]
                weight_y = torch.exp(-kappa * img_grad_y)  # [B, T, V, H-1, W]
                
                # 应用静态掩码（需要与梯度维度匹配）
                static_mask_x = static_mask[:, :, :, :, :-1].float()  # [B, T, V, H, W-1]
                static_mask_y = static_mask[:, :, :, :-1, :].float()  # [B, T, V, H-1, W]
                
                # 确保空间维度匹配
                if static_mask_x.shape[-2:] != weight_x.shape[-2:]:
                    static_mask_x = F.interpolate(
                        static_mask_x.view(B * T * V, 1, *static_mask_x.shape[-2:]),
                        size=weight_x.shape[-2:], mode='nearest'
                    ).bool().view(B, T, V, *weight_x.shape[-2:]).float()
                if static_mask_y.shape[-2:] != weight_y.shape[-2:]:
                    static_mask_y = F.interpolate(
                        static_mask_y.view(B * T * V, 1, *static_mask_y.shape[-2:]),
                        size=weight_y.shape[-2:], mode='nearest'
                    ).bool().view(B, T, V, *weight_y.shape[-2:]).float()
                
                # 计算平滑损失
                wp_grad_norm_x = torch.norm(wp_grad_x, dim=-1, p=2)  # [B, T, V, H, W-1]
                wp_grad_norm_y = torch.norm(wp_grad_y, dim=-1, p=2)  # [B, T, V, H-1, W]
                
                smooth_x = (wp_grad_norm_x * weight_x * static_mask_x).mean()
                smooth_y = (wp_grad_norm_y * weight_y * static_mask_y).mean()
                L_smooth_edge = (smooth_x + smooth_y) / 2.0
                loss_dict['L_smooth_edge'] = L_smooth_edge.item() if isinstance(L_smooth_edge, torch.Tensor) else L_smooth_edge
                total_loss = total_loss + L_smooth_edge * loss_config.weights['smooth_edge']
        except Exception as e:
            loss_dict['L_smooth_edge'] = 0.0
            if loss_config.use_smooth_edge:
                print(f"Warning: Error computing L_smooth_edge: {e}, skipping")
    else:
        loss_dict['L_smooth_edge'] = 0.0
    
    # L_uncert (异方差/熵正则, 权重0.1) - 替代硬阈值
    if loss_config.use_uncert:
        try:
            L_uncert = torch.tensor(0.0, device=device, requires_grad=True)
            if 'depth_conf' in predictions:
                depth_conf = predictions['depth_conf']
                # 处理不同可能的形状
                if len(depth_conf.shape) == 6:
                    depth_conf = depth_conf.squeeze(-3) if depth_conf.shape[-3] == 1 else depth_conf.squeeze(-1)
                elif len(depth_conf.shape) == 5 and depth_conf.shape[-3] == 1:
                    depth_conf = depth_conf.squeeze(-3)
                L_uncert = L_uncert + compute_entropy_regularization_loss(depth_conf)
            if 'world_points_conf' in predictions:
                point_conf = predictions['world_points_conf']
                # 处理不同可能的形状
                if len(point_conf.shape) == 6:
                    point_conf = point_conf.squeeze(-3) if point_conf.shape[-3] == 1 else point_conf.squeeze(-1)
                elif len(point_conf.shape) == 5 and point_conf.shape[-3] == 1:
                    point_conf = point_conf.squeeze(-3)
                L_uncert = L_uncert + compute_entropy_regularization_loss(point_conf)
            loss_dict['L_uncert'] = L_uncert.item() if isinstance(L_uncert, torch.Tensor) else L_uncert
            total_loss = total_loss + L_uncert * loss_config.weights['uncert']
        except Exception as e:
            loss_dict['L_uncert'] = 0.0
            if loss_config.use_uncert:
                print(f"Warning: Error computing L_uncert: {e}, skipping")
    else:
        loss_dict['L_uncert'] = 0.0
    
    # L_scale^S (尺度一致, 权重0.1) - 静态区深度中位数对齐
    if loss_config.use_scale and 'depth' in predictions and static_mask is not None:
        try:
            depth = predictions['depth']
            
            # 处理depth的不同可能形状，统一转换为 [B, T, V, H, W]
            if len(depth.shape) == 6 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)  # [B, T, V, 1, H, W] -> [B, T, V, H, W]
            elif len(depth.shape) == 6 and depth.shape[-3] == 1:
                depth = depth.squeeze(-3)  # [B, T, V, H, W, 1] -> [B, T, V, H, W]
            elif len(depth.shape) == 5 and depth.shape[-3] == 1:
                depth = depth.squeeze(-3)  # [B, T, V, 1, H, W] -> [B, T, V, H, W]
            
            # 确保空间维度匹配
            if depth.shape[-2:] != static_mask.shape[-2:]:
                depth_h, depth_w = depth.shape[-2:]
                mask_h, mask_w = static_mask.shape[-2:]
                if depth_h != mask_h or depth_w != mask_w:
                    depth = F.interpolate(
                        depth.view(B*T*V, 1, depth_h, depth_w),
                        size=(mask_h, mask_w), mode='bilinear', align_corners=False
                    ).view(B, T, V, mask_h, mask_w)
            
            static_depth = depth * static_mask.float()  # [B, T, V, H, W]
            depth_medians = []
            for t in range(T):
                for v in range(V):
                    d = static_depth[:, t, v, :, :]
                    valid_d = d[d > 0]
                    if len(valid_d) > 0:
                        depth_medians.append(torch.median(valid_d))
            
            if len(depth_medians) > 1:
                medians_tensor = torch.stack(depth_medians)
                global_median = medians_tensor.median()
                L_scale = torch.mean((medians_tensor - global_median) ** 2)
                loss_dict['L_scale'] = L_scale.item() if isinstance(L_scale, torch.Tensor) else L_scale
                total_loss = total_loss + L_scale * loss_config.weights['scale']
            else:
                loss_dict['L_scale'] = 0.0
        except Exception as e:
            loss_dict['L_scale'] = 0.0
            if loss_config.use_scale:
                print(f"Warning: Error computing L_scale: {e}, skipping")
    else:
        loss_dict['L_scale'] = 0.0
    
    # L_photo^S (Charbonnier/SSIM, 可选, 权重0.2) - 占位符
    if loss_config.use_photo:
        L_photo = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict['L_photo'] = L_photo.item()
        total_loss = total_loss + L_photo * loss_config.weights['photo']
    else:
        loss_dict['L_photo'] = 0.0
    
    return total_loss, loss_dict


def compute_dual_stream_separation_loss(predictions, loss_config=None):
    """【待定，暂时没有加入】
    L_sep (双流去相关/Barlow Twins, 权重0.1)
    按照图片指令：替换余弦相似度为Barlow Twins去相关范式，非对称stop-grad版本（支持配置控制）
    
    Args:
        predictions: 模型输出字典
        loss_config: LossConfig实例
    """
    if loss_config is None:
        loss_config = LossConfig(use_minimal=True)
    
    if not loss_config.use_sep or 'dual_stream_outputs' not in predictions:
        return torch.tensor(0.0, device=list(predictions.values())[0].device), {'L_sep': 0.0}
    
    dual_stream = predictions['dual_stream_outputs']
    if 'pose' not in dual_stream or 'geo' not in dual_stream:
        return torch.tensor(0.0, device=list(predictions.values())[0].device), {'L_sep': 0.0}
    
    pose_features = dual_stream['pose'][-1]  # [B, T, V, P, C]
    geo_features = dual_stream['geo'][-1]  # [B, T, V, P, C]
    
    # 展平为 [B*T*V*P, C]
    pose_flat = pose_features.view(-1, pose_features.shape[-1])
    geo_flat = geo_features.view(-1, geo_features.shape[-1])
    
    # Barlow Twins: 非对称stop-grad版本
    # loss1: pose 去相关 (stop-grad) geo
    loss1 = compute_barlow_twins_loss(pose_flat, geo_flat.detach())
    # loss2: geo 去相关 (stop-grad) pose
    loss2 = compute_barlow_twins_loss(geo_flat, pose_flat.detach())
    
    L_sep = (loss1 + loss2) / 2.0 * loss_config.weights['sep']
    
    return L_sep, {'L_sep': L_sep.item()}


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
        
        # 打印模型结构
        print_model_structure(model_mv, max_depth=4)
        
        # 加载权重
        model_mv.to(device)
        load_stats = load_pretrained_weights(model_mv, origin, device=device, verbose=True) #mismatch (skipped): 1 keys： aggregator.patch_embed.pos_embed: checkpoint[1, 1370, 1024] vs model[1, 677, 1024]
        
        # ========== 阶段1：冻结参数（按照图片中的精确指令） ==========
        print("\n" + "="*60)
        print("Stage 1: Freezing base features, training new modules")
        print("Applying precise freeze/train strategy as per instructions")
        print("="*60)
        apply_freeze_train_strategy_precise(model_mv)
        
        # ========== 设置优化器 ==========
        # 只优化需要训练的参数
        trainable_params = [p for p in model_mv.parameters() if p.requires_grad]
        # Verify camera_token is trainable and will be included in optimizer
        aggregator = model_mv.aggregator
        if hasattr(aggregator, 'camera_token'):
            print(f"\n[DEBUG] Camera token status before optimizer creation:")
            print(f"  camera_token.requires_grad: {aggregator.camera_token.requires_grad}")
            print(f"  camera_token.shape: {aggregator.camera_token.shape}")
            # Check if camera tokens for different views are different
            if aggregator.camera_token.shape[1] >= 4:  # At least 4 views
                view_diffs = []
                for v1 in range(min(4, aggregator.camera_token.shape[1])):
                    for v2 in range(v1 + 1, min(4, aggregator.camera_token.shape[1])):
                        diff = (aggregator.camera_token[0, v1, 0, :] - aggregator.camera_token[0, v2, 0, :]).abs().max().item()
                        view_diffs.append(diff)
                if len(view_diffs) > 0:
                    avg_diff = sum(view_diffs) / len(view_diffs)
                    max_diff = max(view_diffs)
                    print(f"  Camera token differences (first 4 views): avg={avg_diff:.6f}, max={max_diff:.6f}")
                    if max_diff < 1e-5:
                        print(f"  WARNING: Camera tokens are too similar! Consider using larger initialization.")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-4,  # 两流blocks和mask heads
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Verify camera_token is in optimizer
        camera_token_in_optimizer = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param is aggregator.camera_token:
                    camera_token_in_optimizer = True
                    break
        if camera_token_in_optimizer:
            print(f"  ✓ Camera token is included in optimizer")
        else:
            print(f"  ✗ WARNING: Camera token is NOT in optimizer! This may cause it not to be trained.")
        
        # 学习率调度器（可选）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        # ========== 损失函数配置（核心必要+重要损失） ==========
        # 当前配置：只保留核心必要损失和重要损失，其余损失不参与训练
        loss_config = LossConfig(use_minimal=True)
        
        # 当前激活的损失：
        # 【核心必要损失】
        #   - L_mask_ce (1.0): 掩码监督
        #   - L_depth_point (0.5): 深度一致性
        #   - L_cam_const (0.1): 相机跨时恒定
        # 【重要损失】
        #   - L_smooth_edge (0.2): 边缘感知几何平滑
        #   - L_mv_geo (1.0): 跨视角几何重投影（当前占位符）
        #
        # 【不参与训练的损失】
        #   - L_mask_tv, L_uncert, L_scale, L_sep (可选损失)
        #   - L_epi, L_photo (占位符)
        #   - L_kpt, L_temp_d (需要关键点标注)
        
        # 预留扩展接口：如果需要启用可选损失，可以取消下面的注释
        # loss_config.enable_loss('mask_tv')           # 启用掩码边界/时序损失
        # loss_config.enable_loss('uncert')            # 启用置信度正则
        # loss_config.enable_loss('scale')             # 启用尺度一致性损失
        # loss_config.enable_loss('sep')                # 启用双流分离损失（需要dual_stream_outputs）
        # loss_config.set_weight('depth_point', 0.8)    # 调整深度一致性损失权重
        
        loss_config.print_config()
        
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
                # 几何损失
                'L_mv_geo': 0.0,
                'L_epi': 0.0,
                'L_depth_point': 0.0,
                'L_scale': 0.0,
                'L_smooth_edge': 0.0,
                'L_uncert': 0.0,
                # 掩码损失
                'mask_loss': 0.0,
                'mask_ce_loss': 0.0,
                'mask_boundary_loss': 0.0,
                'mask_temporal_loss': 0.0,
                # 相机损失
                'L_cam_const': 0.0,
                # 分离损失
                'L_sep': 0.0
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
                    
                    # 获取相机参数
                    camera_params = predictions.get('pose_enc', None)  # [B, V, 9]
                    
                    # 计算损失（使用配置控制，最小必要配置）
                    # 1. 多视角几何一致性损失（包含：'L_depth_point', 'L_mv_geo', 'L_epi', 'L_smooth_edge', 'L_uncert', 'L_scale', 'L_photo'）
                    mv_loss, mv_loss_dict = compute_multi_view_consistency_loss(
                        predictions, images_batch, segmask_gt=masks_batch, 
                        camera_params=camera_params, loss_config=loss_config
                    )
                    
                    # 2. 掩码监督损失（包含：L_mask_ce, 可选L_mask_tv）
                    mask_loss, mask_loss_dict = compute_mask_supervision_loss(
                        predictions, masks_batch, images_batch, loss_config=loss_config
                    )
                    
                    # 3. 相机跨时恒定损失（L_cam^const）
                    cam_loss, cam_loss_dict = compute_camera_temporal_consistency_loss(
                        predictions, T, V, loss_config=loss_config
                    )
                    
                    # 4. 双流分离损失（L_sep，可选）
                    sep_loss, sep_dict = compute_dual_stream_separation_loss(
                        predictions, loss_config=loss_config
                    )
                    
                    # 总损失（所有损失的权重已在各函数内部根据loss_config应用）
                    loss_total = mv_loss + mask_loss + cam_loss + sep_loss
                    
                    # 合并损失字典
                    loss_dict = {}
                    loss_dict.update(mv_loss_dict)
                    loss_dict.update(mask_loss_dict)
                    loss_dict.update(cam_loss_dict)
                    loss_dict.update(sep_dict)
                    loss_dict['total'] = loss_total.item()
                
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
                    # 记录输入图像（V行T列的大图）
                    if len(images_batch.shape) == 6:  # Multi-view: [B, T, V, C, H, W]
                        B, T, V, C, H, W = images_batch.shape
                        if T > 0 and V > 0:
                            # Create grid: V rows (views) × T columns (time)
                            img_grid = create_multi_view_time_grid(images_batch, V, T, normalize=True)
                            writer.add_image('Input/Image_grid_VxT', img_grid, global_step)
                        
                        # 记录预测的depth（如果可用）- 也使用网格布局
                        if 'depth' in predictions:
                            depth = predictions['depth']  # [B, T, V, H, W, 1] or [B, T, V, H, W]
                            # Handle different depth shapes
                            if len(depth.shape) == 6:
                                depth = depth.squeeze(-1)  # [B, T, V, H, W]
                            elif len(depth.shape) == 5:
                                pass  # Already [B, T, V, H, W]
                            else:
                                depth = None
                            
                            if depth is not None:
                                # Normalize depth for visualization
                                depth_vis = depth[0].detach().cpu()  # [T, V, H, W]
                                # Normalize each frame separately
                                depth_normalized = depth_vis.clone()
                                for t in range(T):
                                    for v in range(V):
                                        d = depth_vis[t, v]
                                        if d.max() > d.min():
                                            depth_normalized[t, v] = (d - d.min()) / (d.max() - d.min())
                                        else:
                                            depth_normalized[t, v] = d
                                # Add channel dimension and create grid
                                depth_grid = create_multi_view_time_grid(
                                    depth_normalized.unsqueeze(2), V, T, normalize=False
                                )  # [1, H*V, W*T]
                                writer.add_image('Predictions/Depth_grid_VxT', depth_grid, global_step)
                        
                        # 记录mask（如果有）- 也使用网格布局
                        if masks_batch is not None:
                            # masks_batch: [B, T, V, H, W]
                            # Ensure correct shape: [B, T, V, H, W] -> [B, T, V, 1, H, W] for grid function
                            if len(masks_batch.shape) == 5:
                                # [B, T, V, H, W] -> [B, T, V, 1, H, W]
                                masks_for_grid = masks_batch.unsqueeze(3)  # [B, T, V, 1, H, W]
                            elif len(masks_batch.shape) == 6:
                                # Already has channel dimension
                                if masks_batch.shape[3] == 1:
                                    masks_for_grid = masks_batch  # [B, T, V, 1, H, W]
                                elif masks_batch.shape[5] == 1:
                                    masks_for_grid = masks_batch.permute(0, 1, 2, 5, 3, 4).squeeze(-1)  # Rearrange to [B, T, V, 1, H, W]
                                    masks_for_grid = masks_for_grid.unsqueeze(3)  # Ensure channel dim
                                else:
                                    masks_for_grid = masks_batch.unsqueeze(3) if masks_batch.shape[3] != 1 else masks_batch
                            else:
                                masks_for_grid = None
                            
                            if masks_for_grid is not None:
                                try:
                                    mask_grid = create_multi_view_time_grid(
                                        masks_for_grid, V, T, normalize=False
                                    )  # [1, H*V, W*T]
                                    writer.add_image('GT/Mask_grid_VxT', mask_grid, global_step)
                                except Exception as e:
                                    print(f"[WARNING] Failed to create mask grid: {e}, skipping mask visualization")
                    
                    elif len(images_batch.shape) == 5:  # Legacy: [B, S, C, H, W]
                        # Single view mode: just log first image
                        if images_batch.shape[1] > 0:
                            img_to_log = images_batch[0, 0].cpu()  # [C, H, W]
                            writer.add_image('Input/Image_t0', img_to_log, global_step)
            
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
                checkpoint_path = f"{log_dir_with_timestamp}/stage1_epoch_{epoch+1}.pt"
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
    
