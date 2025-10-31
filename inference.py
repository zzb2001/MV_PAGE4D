# In this one we test the performance on dynamic
import os
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import torch
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


def load_time_view_images(data_dir, target_size=518, mode="crop"):
    """
    从 data/t 文件夹加载时序多视角图像，组织成 [T, V, C, H, W] 格式
    
    Args:
        data_dir (str): 数据目录路径，例如 "data/t"
        target_size (int): 目标图像尺寸，默认 518
        mode (str): 预处理模式，"crop" 或 "pad"
    
    Returns:
        torch.Tensor: 图像张量，形状为 [T, V, C, H, W]
        dict: 元数据，包含时间帧数和视角数
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # 查找所有 time_* 文件夹
    time_dirs = glob.glob(os.path.join(data_dir, "time_*"))
    if len(time_dirs) == 0:
        raise ValueError(f"No time_* directories found in {data_dir}")
    
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
    
    T = len(time_dirs)  # 时间帧数
    V = len(view_files)  # 视角数
    
    print(f"Found {T} time frames and {V} views")
    
    # 加载所有图像
    images_list = []
    to_tensor = TF.ToTensor()
    
    for t_idx, time_dir in enumerate(time_dirs):
        time_images = []
        for v_file in view_files:
            img_path = os.path.join(time_dir, v_file)
            if not os.path.exists(img_path):
                raise ValueError(f"Image not found: {img_path}")
            
            # 加载图像
            img = Image.open(img_path)
            
            # 处理 RGBA
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            img = img.convert("RGB")
            
            width, height = img.size
            
            # Resize
            if mode == "pad":
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14
            else:  # mode == "crop"
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # [C, H, W]
            
            # Center crop (crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]
            
            # Pad (pad mode)
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), 
                        mode="constant", value=1.0)
            
            time_images.append(img)
        
        images_list.append(time_images)
    
    # 转换为张量 [T, V, C, H, W]
    images_tensor = torch.stack([torch.stack(time_imgs) for time_imgs in images_list])
    
    # 确保所有图像尺寸一致，统一padding到 target_size x target_size
    T, V, C, H, W = images_tensor.shape
    if H != target_size or W != target_size:
        # 创建新的tensor来存储统一尺寸的图像
        images_padded = torch.ones(T, V, C, target_size, target_size, dtype=images_tensor.dtype)
        
        for t in range(T):
            for v in range(V):
                img = images_tensor[t, v]  # [C, H, W]
                # 计算padding
                h_padding = target_size - H
                w_padding = target_size - W
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                # Padding
                img_padded = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant", value=1.0)
                images_padded[t, v] = img_padded
        
        images_tensor = images_padded
    
    metadata = {
        'T': T,
        'V': V,
        'time_dirs': [os.path.basename(d) for d in time_dirs],
        'view_files': view_files
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
    images = images.unsqueeze(0)    #[1, 24, 3, 294, 518]
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16
    
    # 加载时序多视角数据
    data_dir = "data/t"
    if os.path.exists(data_dir):
        print(f"Loading time-view images from {data_dir}")
        images_tv, metadata = load_time_view_images(data_dir, target_size=518, mode="crop")
        # images_tv shape: [T, V, C, H, W]
        print(f"Loaded images with shape: {images_tv.shape}")
        print(f"Time frames: {metadata['T']}, Views: {metadata['V']}")
        
        # 如果需要处理这些数据，可以在这里添加代码
        # 例如，将 [T, V, C, H, W] 转换为模型需要的格式
        # 模型通常需要 [B, S, C, H, W]，其中 S 是序列长度（时间*视角）
        # T, V, C, H, W = images_tv.shape
        # images_batch = images_tv.view(1, T * V, C, H, W).to(device)
    
    # 处理时序多视角数据（vggt_t_mv）
    if os.path.exists(data_dir):
        print(f"\n{'='*60}")
        print("Processing temporal multi-view data with VGGT_MV")
        print(f"{'='*60}")
        
        # 将 [T, V, C, H, W] 转换为模型需要的 [B, T, V, C, H, W] 格式
        T, V, C, H, W = images_tv.shape
        images_batch = images_tv.unsqueeze(0).to(device)  # [1, T, V, C, H, W]
        
        print(f"Input shape: {images_batch.shape} (B={1}, T={T}, V={V}, C={C}, H={H}, W={W})")
        
        # 初始化 vggt_t_mv 模型，使用多视角模式
        origin = "checkpoint/checkpoint_150.pt"
        model_mv = VGGT_MV(
            img_size=518,
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
        
        print(f"Inference completed. Predictions keys: {list(predictions.keys())}")
        # 后续处理可以根据需要添加，例如保存点云、深度图等
        
    # 原有的处理逻辑（保留用于向后兼容）
    for data_name in ["online_img3"]:
        if data_name == "online_img3":
            directory = f"./{data_name}/"

        folders = [f for f in os.listdir(directory) if os.path.isdir(
            os.path.join(directory, f)) and not f.startswith('fig1_')]
        
        # 这里保留原有的 DPG 模型处理逻辑作为备选
        # 如果需要，可以注释掉或删除
        # origin = "checkpoint/checkpoint_150.pt"
        # model = DPG()
        # checkpoint = torch.load(origin, map_location=device)
        # model.load_state_dict(checkpoint['model'], strict=False)
        # model.to(device)

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

