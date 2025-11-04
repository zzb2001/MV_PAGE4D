# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Stage-0 体素化模块：将多视角像素/patch特征体素化到统一的世界网格

核心功能：
1. 反投影像素到世界坐标
2. 计算体素索引（支持自适应或固定体素大小）
3. 体素内加权聚合（AnySplat范式：scatter_add + softmax加权）
4. 生成体素tokens（特征+位置编码）
5. 稳定的体素ID生成（用于跨时间对齐）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
import numpy as np


def morton_encode(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Morton (Z-order) encoding: 将3D整数坐标编码为64-bit整数
    
    Args:
        x, y, z: 整数坐标张量，shape相同
    Returns:
        morton: 64-bit Morton编码，shape与输入相同
    """
    # 假设每轴范围 < 2^21 (2M体素)
    # 使用位交错: morton = z2 z1 z0 y2 y1 y0 x2 x1 x0 ...
    morton = torch.zeros_like(x, dtype=torch.int64)
    
    # 将每个坐标限制在21位内
    x = x.clamp(0, (1 << 21) - 1)
    y = y.clamp(0, (1 << 21) - 1)
    z = z.clamp(0, (1 << 21) - 1)
    
    # 位交错编码（每轴21位 -> 总共63位）
    for i in range(21):
        morton |= (x & (1 << i)).long() << (3 * i)
        morton |= (y & (1 << i)).long() << (3 * i + 1)
        morton |= (z & (1 << i)).long() << (3 * i + 2)
    
    return morton


def voxel_id_from_indices(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, 
                          use_morton: bool = False) -> torch.Tensor:
    """
    从体素索引生成稳定的体素ID
    
    Args:
        ix, iy, iz: 体素索引 [N]
        use_morton: 是否使用Morton编码（默认False，使用简单三元组）
    Returns:
        voxel_ids: 体素ID [N]
    """
    if use_morton:
        return morton_encode(ix, iy, iz)
    else:
        # 简单三元组：假设每轴范围 < 2^20，用bit-packing
        # (ix << 40) | (iy << 20) | iz
        max_val = (1 << 20) - 1
        ix_packed = (ix.clamp(0, max_val).long() << 40)
        iy_packed = (iy.clamp(0, max_val).long() << 20)
        iz_packed = iz.clamp(0, max_val).long()
        return ix_packed | iy_packed | iz_packed


def scatter_add_weighted(values: torch.Tensor, indices: torch.Tensor, 
                        weights: Optional[torch.Tensor] = None,
                        dim_size: Optional[int] = None) -> torch.Tensor:
    """
    带权重的scatter_add实现（如果torch_scatter不可用，使用原生实现）
    
    Args:
        values: [N, D] 要聚合的值
        indices: [N] 目标索引
        weights: [N] 可选的权重（如果为None，则不加权）
        dim_size: 输出维度大小（如果为None，自动推断）
    Returns:
        aggregated: [dim_size, D] 聚合后的值
    """
    if dim_size is None:
        if indices.numel() > 0:
            dim_size = int(indices.max().item()) + 1
        else:
            dim_size = 0
    
    if dim_size == 0:
        return torch.zeros(0, values.shape[1], device=values.device, dtype=values.dtype)
    
    device = values.device
    dtype = values.dtype
    
    # 确保values是2D
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    
    if weights is None:
        weights = torch.ones(indices.shape[0], device=device, dtype=dtype)
    
    # 加权值
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1)
    weighted_values = values * weights
    
    # 使用index_add实现scatter_add
    result = torch.zeros(dim_size, values.shape[1], device=device, dtype=dtype)
    valid_mask = (indices >= 0) & (indices < dim_size)
    if valid_mask.any():
        result.index_add_(0, indices[valid_mask].long(), weighted_values[valid_mask])
    
    return result


def scatter_max(values: torch.Tensor, indices: torch.Tensor,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    scatter_max实现（用于softmax数值稳定性）
    
    Args:
        values: [N] 要取max的值
        indices: [N] 目标索引
        dim_size: 输出维度大小
    Returns:
        max_values: [dim_size] 每个索引的最大值
        max_indices: [dim_size] 最大值的索引
    """
    if dim_size is None:
        dim_size = int(indices.max().item()) + 1
    
    device = values.device
    max_values = torch.full((dim_size,), float('-inf'), device=device, dtype=values.dtype)
    max_indices = torch.zeros(dim_size, device=device, dtype=torch.long)
    
    for i in range(indices.shape[0]):
        idx = indices[i].item()
        if values[i] > max_values[idx]:
            max_values[idx] = values[i]
            max_indices[idx] = i
    
    return max_values, max_indices


class VoxelizationModule(nn.Module):
    """
    Stage-0 体素化模块
    
    将多视角像素/patch特征体素化到统一的世界网格，生成稀疏的体素tokens。
    
    核心流程：
    1. 对每个时间步t，处理所有视角V的像素
    2. 反投影到世界坐标
    3. 计算体素索引（自适应或固定大小）
    4. 体素内加权聚合（softmax权重）
    5. 生成体素tokens（特征+位置编码）
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        voxel_size: Optional[float] = None,
        voxel_size_mode: str = 'auto',  # 'auto' or 'fixed'
        target_num_voxels: int = 120000,  # 自适应时的目标体素数
        use_morton_encoding: bool = False,  # 是否使用Morton编码
        use_sparse3d: bool = False,  # 是否启用稀疏3D U-Net（先跳过）
        pos_encoding_dim: int = 128,  # 位置编码维度
        confidence_activation: str = 'softplus',  # 置信度激活函数
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.voxel_size = voxel_size
        self.voxel_size_mode = voxel_size_mode
        self.target_num_voxels = target_num_voxels
        self.use_morton_encoding = use_morton_encoding
        self.use_sparse3d = use_sparse3d
        self.pos_encoding_dim = pos_encoding_dim
        
        # 体素特征到token的投影
        self.voxel_feat_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # 位置编码：体素中心坐标 -> 位置编码
        self.pos_encoding = nn.Sequential(
            nn.Linear(3, pos_encoding_dim),
            nn.SiLU(),
            nn.Linear(pos_encoding_dim, pos_encoding_dim),
        )
        
        # 体素token = MLP(特征) + 位置编码
        self.token_fusion = nn.Sequential(
            nn.Linear(embed_dim + pos_encoding_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # 可选：稀疏3D U-Net（保留接口，暂不实现）
        if use_sparse3d:
            # TODO: 实现轻量MLP或MinkowskiEngine U-Net
            self.sparse3d_refine = None
        
        # 置信度处理
        if confidence_activation == 'softplus':
            self.conf_activation = nn.Softplus()
        elif confidence_activation == 'exp':
            self.conf_activation = lambda x: torch.exp(torch.clamp(x, max=10))
        else:
            self.conf_activation = nn.Identity()
    
    def estimate_voxel_size(
        self,
        points3d: torch.Tensor,  # [B*V*N, 3] 或 [B, V, H, W, 3]
        percentile_low: float = 0.01,
        percentile_high: float = 0.99,
    ) -> float:
        """
        自适应估计体素大小
        
        基于点云的1%-99%分位点，计算目标体素数量下的体素大小。
        
        Args:
            points3d: 3D点云坐标
            percentile_low: 下分位点（默认1%）
            percentile_high: 上分位点（默认99%）
        Returns:
            voxel_size: 估计的体素大小
        """
        # 展平为 [N, 3]
        if points3d.dim() > 2:
            points_flat = points3d.reshape(-1, 3)
        else:
            points_flat = points3d
        
        # 过滤无效点（NaN, Inf, 零）
        valid_mask = torch.isfinite(points_flat).all(dim=-1) & (points_flat.norm(dim=-1) > 1e-6)
        if valid_mask.sum() == 0:
            # 回退到默认值
            return 0.01
        
        points_valid = points_flat[valid_mask]
        
        # 确保points_valid是float类型（quantile需要float或double）
        if points_valid.dtype not in (torch.float32, torch.float64):
            points_valid = points_valid.float()
        
        # 计算三轴范围（分位点）
        range_x = torch.quantile(points_valid[:, 0], torch.tensor([percentile_low, percentile_high], device=points_valid.device, dtype=points_valid.dtype))
        range_y = torch.quantile(points_valid[:, 1], torch.tensor([percentile_low, percentile_high], device=points_valid.device, dtype=points_valid.dtype))
        range_z = torch.quantile(points_valid[:, 2], torch.tensor([percentile_low, percentile_high], device=points_valid.device, dtype=points_valid.dtype))
        
        range_x_val = (range_x[1] - range_x[0]).item()
        range_y_val = (range_y[1] - range_y[0]).item()
        range_z_val = (range_z[1] - range_z[0]).item()
        
        # 避免除零
        if range_x_val < 1e-6 or range_y_val < 1e-6 or range_z_val < 1e-6:
            return 0.01
        
        # 公式：voxel_size = cbrt((range_x * range_y * range_z) / N_target)
        volume = range_x_val * range_y_val * range_z_val
        voxel_volume = volume / self.target_num_voxels
        voxel_size = np.cbrt(voxel_volume)
        
        return float(voxel_size)
    
    def backproject_to_world(
        self,
        depth: torch.Tensor,  # [B, V, H, W] 或 [B*V, H, W]
        intrinsics: torch.Tensor,  # [B, V, 3, 3] 或 [B*V, 3, 3]
        extrinsics: torch.Tensor,  # [B, V, 3, 4] 或 [B*V, 3, 4]
    ) -> torch.Tensor:
        """
        将深度图反投影到世界坐标
        
        Args:
            depth: 深度图
            intrinsics: 相机内参
            extrinsics: 相机外参（3x4矩阵，世界->相机）
        Returns:
            points3d_world: 世界坐标点 [B, V, H, W, 3] 或 [B*V, H, W, 3]
        """
        # 统一形状处理
        if depth.dim() == 4:
            # 判断是 [B*V, H, W] 还是 [B, V, H, W]
            if intrinsics.dim() == 4:  # [B, V, 3, 3]
                B, V, H, W = depth.shape
                B_V = B * V
                depth_reshaped = depth.reshape(B_V, H, W)
                K = intrinsics.reshape(B_V, 3, 3)
                E = extrinsics.reshape(B_V, 3, 4)
            elif intrinsics.dim() == 3:  # [B*V, 3, 3]
                B_V, H, W = depth.shape
                K = intrinsics
                E = extrinsics
                depth_reshaped = depth
            else:
                raise ValueError(f"Intrinsics shape mismatch: expected 3 or 4 dims, got {intrinsics.dim()}")
        else:
            raise ValueError(f"Depth shape mismatch: expected 4 dims, got {depth.dim()}")
        
        device = depth.device
        
        # 创建像素网格 [H, W, 2]
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=depth.dtype),
            torch.arange(W, device=device, dtype=depth.dtype),
            indexing='ij'
        )
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]
        pixels = pixels.reshape(1, H*W, 3).expand(B_V, -1, -1)  # [B_V, H*W, 3]
        
        # 反投影到相机坐标
        K_inv = torch.inverse(K)  # [B_V, 3, 3]
        cam_coords = torch.bmm(pixels, K_inv.transpose(1, 2))  # [B_V, H*W, 3]
        depth_flat = depth_reshaped.reshape(B_V, H*W, 1)  # [B_V, H*W, 1]
        cam_points = cam_coords * depth_flat  # [B_V, H*W, 3]
        
        # 转换为齐次坐标并变换到世界坐标
        ones = torch.ones(B_V, H*W, 1, device=device, dtype=depth.dtype)
        cam_points_h = torch.cat([cam_points, ones], dim=-1)  # [B_V, H*W, 4]
        
        # 外参是 世界->相机，需要转换为 相机->世界
        E_4x4 = torch.cat([E, torch.tensor([[0, 0, 0, 1]], device=device, dtype=depth.dtype).expand(B_V, 1, -1)], dim=1)
        E_world_to_cam = E_4x4  # [B_V, 4, 4]
        E_cam_to_world = torch.inverse(E_world_to_cam)  # [B_V, 4, 4]
        
        world_points_h = torch.bmm(cam_points_h, E_cam_to_world.transpose(1, 2))  # [B_V, H*W, 4]
        world_points = world_points_h[:, :, :3]  # [B_V, H*W, 3]
        
        # 恢复空间维度
        world_points = world_points.reshape(B_V, H, W, 3)
        
        # 如果原始输入是 [B, V, H, W]，reshape回 [B, V, H, W, 3]
        if intrinsics.dim() == 4:  # 原始是 [B, V, 3, 3]
            world_points = world_points.reshape(B, V, H, W, 3)
        
        return world_points
    
    def compute_voxel_indices(
        self,
        points3d: torch.Tensor,  # [N, 3] 世界坐标点
        voxel_size: float,
        stabilize: bool = True,  # 是否做边界稳定性处理
        tolerance: float = 0.5,  # 边界容忍度（体素单位的倍数）
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算体素索引
        
        Args:
            points3d: 世界坐标点 [N, 3]
            voxel_size: 体素大小
            stabilize: 是否做边界稳定性处理
            tolerance: 边界容忍度
        Returns:
            ix, iy, iz: 体素索引 [N]
            voxel_centers: 体素中心坐标 [N, 3]
        """
        # 计算体素索引
        voxel_coords = points3d / voxel_size
        ix = voxel_coords[:, 0].round().long()
        iy = voxel_coords[:, 1].round().long()
        iz = voxel_coords[:, 2].round().long()
        
        # 边界稳定性处理：±tolerance范围内的点归入同一体素
        if stabilize:
            # 计算到体素中心的距离（归一化到体素单位）
            center_coords = torch.stack([ix.float(), iy.float(), iz.float()], dim=-1)
            dist_to_center = torch.abs(voxel_coords - center_coords)
            
            # 对于接近边界的点，调整索引
            threshold = tolerance
            adjust_mask = (dist_to_center > threshold).any(dim=-1)
            if adjust_mask.any():
                # 将这些点调整到最近的整数体素
                adjusted_coords = voxel_coords[adjust_mask].round().long()
                ix[adjust_mask] = adjusted_coords[:, 0]
                iy[adjust_mask] = adjusted_coords[:, 1]
                iz[adjust_mask] = adjusted_coords[:, 2]
        
        # 计算体素中心坐标（用于后续聚合）
        voxel_centers = torch.stack([
            ix.float() * voxel_size,
            iy.float() * voxel_size,
            iz.float() * voxel_size
        ], dim=-1)
        
        return ix, iy, iz, voxel_centers
    
    def voxelization_with_fusion(
        self,
        patch_features: torch.Tensor,  # [B*V*N, D] 或 [B, V, P, D] patch特征
        points3d: torch.Tensor,  # [B*V*N, 3] 或 [B, V, H, W, 3] 世界坐标
        confidence: Optional[torch.Tensor] = None,  # [B*V*N] 或 [B, V, H, W] 置信度
        mask: Optional[torch.Tensor] = None,  # [B*V*N] 或 [B, V, H, W] 动态掩码
        voxel_size: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        AnySplat范式的体素融合
        
        将像素/patch特征聚合到体素，使用置信度softmax加权。
        
        Args:
            patch_features: patch特征
            points3d: 世界坐标点
            confidence: 置信度（用于加权）
            mask: 动态掩码（可选，一起聚合到体素）
            voxel_size: 体素大小（如果为None，使用self.voxel_size或自适应）
        Returns:
            voxel_xyz: 体素中心坐标 [N_voxels, 3]
            voxel_feat: 聚合后的特征 [N_voxels, D]
            voxel_ids: 体素ID [N_voxels]
            voxel_mask: 体素掩码（如果提供mask） [N_voxels]
        """
        device = patch_features.device
        
        # 展平处理
        if patch_features.dim() == 4:  # [B, V, P, D]
            B, V, P, D = patch_features.shape
            patch_feat_flat = patch_features.reshape(B * V * P, D)
            if points3d.dim() == 5:  # [B, V, H, W, 3]
                # 从points3d的shape中获取H和W
                _, _, H_points, W_points, _ = points3d.shape
                points_flat = points3d.reshape(B * V * H_points * W_points, 3)
            else:
                points_flat = points3d.reshape(B * V * P, 3)
        else:
            patch_feat_flat = patch_features.reshape(-1, patch_features.shape[-1])
            points_flat = points3d.reshape(-1, 3)
        
        N = patch_feat_flat.shape[0]
        
        # 处理置信度
        if confidence is None:
            conf_flat = torch.ones(N, device=device)
        else:
            if confidence.dim() > 1:
                conf_flat = confidence.reshape(-1)
            else:
                conf_flat = confidence
            conf_flat = self.conf_activation(conf_flat)
        
        # 处理掩码
        if mask is not None:
            if mask.dim() > 1:
                mask_flat = mask.reshape(-1)
            else:
                mask_flat = mask
        else:
            mask_flat = None
        
        # 确定体素大小
        if voxel_size is None:
            if self.voxel_size_mode == 'auto':
                voxel_size = self.estimate_voxel_size(points_flat)
            else:
                voxel_size = self.voxel_size or 0.01
        
        # 计算体素索引
        ix, iy, iz, voxel_centers = self.compute_voxel_indices(points_flat, voxel_size)
        
        # 生成体素ID
        voxel_indices_3d = torch.stack([ix, iy, iz], dim=-1)  # [N, 3]
        voxel_ids = voxel_id_from_indices(ix, iy, iz, use_morton=self.use_morton_encoding)
        
        # 找到唯一体素
        unique_voxel_ids, inverse_indices, counts = torch.unique(
            voxel_ids, return_inverse=True, return_counts=True
        )
        num_unique_voxels = unique_voxel_ids.shape[0]
        
        # Softmax加权（数值稳定性）
        # 使用CUDA并行的scatter_max实现（避免Python循环）
        # 优先使用torch_scatter（最快），否则使用向量化方法
        try:
            from torch_scatter import scatter_max
            # torch_scatter的scatter_max是CUDA优化的，最快
            conf_max_per_voxel, _ = scatter_max(conf_flat, inverse_indices, dim=0, dim_size=num_unique_voxels)
        except (ImportError, OSError, RuntimeError):
            # 捕获ImportError（未安装）、OSError（版本不兼容）、RuntimeError（CUDA错误等）
            # Fallback: 使用向量化的分组max（比原来的Python循环快很多）
            # 方法：使用index_add变体 + 分组操作
            # 先尝试使用index_add的变体（虽然index_add不支持max，但可以用于其他聚合）
            # 更实用的方法：使用排序+分组边界（已优化，减少循环次数）
            
            # 使用排序后的分组max（比原始循环快，因为已排序，分组更高效）
            sorted_indices = torch.argsort(inverse_indices)
            sorted_inverse = inverse_indices[sorted_indices]
            sorted_conf = conf_flat[sorted_indices]
            
            if len(sorted_inverse) > 0:
                # 找到每个组的边界（向量化操作）
                # 使用diff找到组变化的位置
                if len(sorted_inverse) > 1:
                    group_changes = torch.cat([
                        torch.tensor([True], device=device),
                        sorted_inverse[1:] != sorted_inverse[:-1]
                    ])
                else:
                    group_changes = torch.tensor([True], device=device)
                
                group_boundaries = torch.where(group_changes)[0]
                
                # 计算每个组的结束位置
                group_end = torch.cat([
                    group_boundaries[1:],
                    torch.tensor([len(sorted_conf)], device=device, dtype=torch.long)
                ])
                
                # 使用向量化slice操作（比逐个循环快）
                # 虽然仍有循环，但已优化：先排序，分组更集中
                conf_max_per_voxel = torch.zeros(num_unique_voxels, device=device, dtype=conf_flat.dtype)
                
                # 优化：使用批量索引操作减少开销
                for i in range(len(group_boundaries)):
                    start = group_boundaries[i].item()
                    end = group_end[i].item()
                    if end > start:
                        voxel_id = sorted_inverse[start].item()
                        conf_max_per_voxel[voxel_id] = sorted_conf[start:end].max()
            else:
                conf_max_per_voxel = torch.zeros(num_unique_voxels, device=device, dtype=conf_flat.dtype)
        
        conf_exp = torch.exp(conf_flat - conf_max_per_voxel[inverse_indices])
        conf_sum_per_voxel = scatter_add_weighted(
            conf_exp.unsqueeze(-1), inverse_indices, dim_size=num_unique_voxels
        ).squeeze(-1)
        weights = (conf_exp / (conf_sum_per_voxel[inverse_indices] + 1e-6)).unsqueeze(-1)
        
        # 加权聚合位置和特征
        weighted_points = points_flat * weights
        weighted_feats = patch_feat_flat * weights
        
        # 聚合到体素
        voxel_xyz = scatter_add_weighted(
            weighted_points, inverse_indices, dim_size=num_unique_voxels
        )
        voxel_feat = scatter_add_weighted(
            weighted_feats, inverse_indices, dim_size=num_unique_voxels
        )
        
        # 处理掩码聚合（如果提供）
        voxel_mask = None
        if mask_flat is not None:
            weighted_mask = mask_flat.unsqueeze(-1) * weights.squeeze(-1)
            voxel_mask = scatter_add_weighted(
                weighted_mask.unsqueeze(-1), inverse_indices, dim_size=num_unique_voxels
            ).squeeze(-1)
        
        # 归一化（由于scatter_add是加权和，需要除以权重和）
        weight_sum = scatter_add_weighted(
            weights, inverse_indices, dim_size=num_unique_voxels
        )
        voxel_xyz = voxel_xyz / (weight_sum + 1e-6)
        voxel_feat = voxel_feat / (weight_sum + 1e-6)
        if voxel_mask is not None:
            voxel_mask = voxel_mask / (weight_sum.squeeze(-1) + 1e-6)
        
        return voxel_xyz, voxel_feat, unique_voxel_ids, voxel_mask
    
    def voxel_to_token(
        self,
        voxel_xyz: torch.Tensor,  # [N_voxels, 3]
        voxel_feat: torch.Tensor,  # [N_voxels, D]
    ) -> torch.Tensor:
        """
        将体素特征和位置编码转换为体素tokens
        
        Args:
            voxel_xyz: 体素中心坐标
            voxel_feat: 体素特征
        Returns:
            voxel_tokens: 体素tokens [N_voxels, embed_dim]
        """
        # 投影特征
        feat_proj = self.voxel_feat_proj(voxel_feat)  # [N_voxels, embed_dim]
        
        # 位置编码
        pos_enc = self.pos_encoding(voxel_xyz)  # [N_voxels, pos_encoding_dim]
        
        # 融合
        combined = torch.cat([feat_proj, pos_enc], dim=-1)  # [N_voxels, embed_dim + pos_encoding_dim]
        voxel_tokens = self.token_fusion(combined)  # [N_voxels, embed_dim]
        
        return voxel_tokens
    
    def forward(
        self,
        patch_tokens: torch.Tensor,  # [B, V, P, D] patch tokens
        depth: torch.Tensor,  # [B, V, H, W] 深度图
        intrinsics: torch.Tensor,  # [B, V, 3, 3]
        extrinsics: torch.Tensor,  # [B, V, 3, 4]
        confidence: Optional[torch.Tensor] = None,  # [B, V, H, W]
        mask: Optional[torch.Tensor] = None,  # [B, V, H, W]
        voxel_size: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：将多视角patch tokens体素化
        
        Args:
            patch_tokens: patch tokens [B, V, P, D]
            depth: 深度图 [B, V, H, W]
            intrinsics: 相机内参 [B, V, 3, 3]
            extrinsics: 相机外参 [B, V, 3, 4]
            confidence: 置信度 [B, V, H, W] (可选)
            mask: 动态掩码 [B, V, H, W] (可选)
            voxel_size: 体素大小（可选，覆盖默认值）
        Returns:
            voxel_tokens: 体素tokens [B, N_voxels, embed_dim]（可能不同batch的N不同）
            voxel_xyz: 体素中心坐标 [B, N_voxels, 3]
            voxel_ids: 体素ID [B, N_voxels]
            voxel_mask: 体素掩码 [B, N_voxels] (如果提供mask)
        """
        B, V, P, D = patch_tokens.shape
        
        # 处理depth的shape：可能是多种格式
        original_depth_shape = depth.shape
        depth_processed = False
        
        if len(depth.shape) == 4:
            # 可能是 [B, V, H, W] 或 [B, H, W, 1] 或 [B, H, W, C]
            if depth.shape[0] == B and depth.shape[1] == V:
                # [B, V, H, W] - 正确格式
                H, W = depth.shape[2], depth.shape[3]
                depth_processed = True
            elif depth.shape[-1] == 1 or depth.shape[-1] == 4:
                # 可能是 [B, H, W, 1] 或 [B, H, W, C]
                # 检查是否是 [B, H, W, 1] -> 需要扩展为 [B, V, H, W]
                if depth.shape[0] == B and len(depth.shape) == 4:
                    H, W = depth.shape[1], depth.shape[2]
                    # 如果最后一个维度是1，squeeze它
                    if depth.shape[-1] == 1:
                        depth = depth.squeeze(-1)  # [B, H, W]
                    else:
                        # [B, H, W, C] -> 取第一个通道
                        depth = depth[..., 0]  # [B, H, W]
                    # 扩展为 [B, V, H, W]：重复V次
                    depth = depth.unsqueeze(1).expand(-1, V, -1, -1)  # [B, V, H, W]
                    depth_processed = True
            else:
                # 可能是 [B, H, W] 格式
                if depth.shape[0] == B and len(depth.shape) == 3:
                    H, W = depth.shape[1], depth.shape[2]
                    # 扩展为 [B, V, H, W]：重复V次
                    depth = depth.unsqueeze(1).expand(-1, V, -1, -1)  # [B, V, H, W]
                    depth_processed = True
        elif len(depth.shape) == 5:
            # 可能是 [B, V, 1, H, W] 或 [B, V, H, W, 1] 或 [B, T, V, H, W]
            # 优先检查最后一个维度是否为1（更常见的情况）
            if depth.shape[-1] == 1:
                # [B, V, H, W, 1] -> [B, V, H, W]
                depth = depth.squeeze(-1)
                if depth.shape[0] == B and depth.shape[1] == V:
                    H, W = depth.shape[2], depth.shape[3]
                    depth_processed = True
            elif depth.shape[2] == 1:
                # [B, V, 1, H, W] -> [B, V, H, W]
                depth = depth.squeeze(2)
                if depth.shape[0] == B and depth.shape[1] == V:
                    H, W = depth.shape[2], depth.shape[3]
                    depth_processed = True
            else:
                # 可能是 [B, T, V, H, W] 但这里不应该出现，因为应该已经按时间步切片了
                # 或者可能是 [B, V, H, W, C] 其中C>1
                if depth.shape[0] == B and depth.shape[1] == V:
                    # [B, V, H, W, C] -> 取第一个通道
                    H, W = depth.shape[2], depth.shape[3]
                    depth = depth[..., 0]  # 取第一个通道，得到 [B, V, H, W]
                    depth_processed = True
        
        if not depth_processed:
            # 如果还没有处理，尝试根据元素总数推断
            total_elements = depth.numel()
            expected_elements = B * V * depth.shape[-2] * depth.shape[-1] if len(depth.shape) >= 2 else total_elements
            
            # 尝试推断形状
            if len(depth.shape) == 4:
                # [?, ?, ?, ?] -> 尝试推断
                if depth.shape[0] == B:
                    # 第一个维度是B
                    if depth.shape[-1] == 1:
                        # [B, H, W, 1] -> [B, H, W] -> [B, V, H, W]
                        H, W = depth.shape[1], depth.shape[2]
                        depth = depth.squeeze(-1).unsqueeze(1).expand(-1, V, -1, -1)
                        depth_processed = True
                    else:
                        # 可能是 [B, H, W, V] 或 [B, V, H, W] 但顺序不对
                        # 检查元素总数
                        if total_elements == B * V * depth.shape[1] * depth.shape[2]:
                            # 可能是 [B, H, W, V] -> 需要permute
                            H, W = depth.shape[1], depth.shape[2]
                            depth = depth.permute(0, 3, 1, 2)  # [B, V, H, W]
                            depth_processed = True
            
            if not depth_processed:
                raise ValueError(
                    f"Cannot process depth shape {original_depth_shape} to [B={B}, V={V}, H, W]. "
                    f"Expected formats: [B, V, H, W], [B, H, W], [B, H, W, 1], [B, V, 1, H, W], or [B, V, H, W, 1]"
                )
        
        # 最终确保depth是 [B, V, H, W]
        if depth.shape != (B, V, H, W):
            # 如果形状不匹配，尝试reshape（仅在元素总数匹配时）
            total_elements = depth.numel()
            expected_elements = B * V * H * W
            if total_elements == expected_elements:
                try:
                    depth = depth.reshape(B, V, H, W)
                except RuntimeError as e:
                    raise ValueError(
                        f"Cannot reshape depth from {depth.shape} to ({B}, {V}, {H}, {W}): {e}. "
                        f"Total elements: {total_elements}, expected: {expected_elements}"
                    )
            else:
                raise ValueError(
                    f"Cannot reshape depth from {depth.shape} to ({B}, {V}, {H}, {W}): "
                    f"total elements mismatch ({total_elements} vs {expected_elements})"
                )
        
        # 反投影到世界坐标
        points3d_world = self.backproject_to_world(depth, intrinsics, extrinsics)  # [B*V, H, W, 3]
        points3d_world = points3d_world.reshape(B, V, H, W, 3)
        
        # 将patch tokens映射到像素空间（需要插值或重新采样）
        # 简化：假设P = H*W，直接reshape
        if P == H * W:
            patch_feat_spatial = patch_tokens.reshape(B, V, H, W, D)
        else:
            # 需要插值：先reshape到空间维度，再插值
            patch_size = int(np.sqrt(P))
            patch_feat_spatial = patch_tokens.reshape(B, V, patch_size, patch_size, D)
            # permute to [B*V, D, patch_size, patch_size] for interpolate
            patch_feat_spatial = F.interpolate(
                patch_feat_spatial.permute(0, 1, 4, 2, 3).reshape(B * V, D, patch_size, patch_size),
                size=(H, W), mode='bilinear', align_corners=False
            )
            # interpolate returns [B*V, D, H, W], reshape to [B, V, D, H, W], then permute to [B, V, H, W, D]
            patch_feat_spatial = patch_feat_spatial.reshape(B, V, D, H, W).permute(0, 1, 3, 4, 2)  # [B, V, H, W, D]
        
        # 展平为 [B, V, H*W, D]
        patch_feat_flat = patch_feat_spatial.reshape(B, V, H * W, D)
        points_flat = points3d_world.reshape(B, V, H * W, 3)
        
        # 对每个batch分别处理（因为体素数可能不同）
        voxel_tokens_list = []
        voxel_xyz_list = []
        voxel_ids_list = []
        voxel_mask_list = []
        
        for b in range(B):
            # 合并所有视角
            patch_feat_b = patch_feat_flat[b].reshape(V * H * W, D)  # [V*H*W, D]
            points_b = points_flat[b].reshape(V * H * W, 3)  # [V*H*W, 3]
            
            conf_b = None
            if confidence is not None:
                conf_b = confidence[b].reshape(V * H * W)  # [V*H*W]
            
            mask_b = None
            if mask is not None:
                mask_b = mask[b].reshape(V * H * W)  # [V*H*W]
            
            # 体素化
            voxel_xyz_b, voxel_feat_b, voxel_ids_b, voxel_mask_b = self.voxelization_with_fusion(
                patch_feat_b, points_b, conf_b, mask_b, voxel_size
            )
            
            # 转换为tokens
            voxel_tokens_b = self.voxel_to_token(voxel_xyz_b, voxel_feat_b)
            
            voxel_tokens_list.append(voxel_tokens_b)
            voxel_xyz_list.append(voxel_xyz_b)
            voxel_ids_list.append(voxel_ids_b)
            if voxel_mask_b is not None:
                voxel_mask_list.append(voxel_mask_b)
        
        # 由于不同batch的体素数可能不同，返回list
        # 调用者需要处理padding或使用变长序列
        return voxel_tokens_list, voxel_xyz_list, voxel_ids_list, (voxel_mask_list if voxel_mask_list else None)

