# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
掩码抬升模块：将SegAnyMo的像素掩码抬升到体素域

核心功能：
1. 多视角加权投票：将像素掩码聚合到体素
2. 体素掩码生成：M_vox_t(idx) = sigmoid(α·mean_v[mask_t,v(p) for p∈idx] / τ)
3. 可学习参数：alpha（初始小值），tau（初始≈2.0）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MaskLiftingModule(nn.Module):
    """
    掩码抬升模块
    
    将像素域的SegAnyMo掩码抬升到体素域，支持多视角加权投票。
    """
    
    def __init__(
        self,
        alpha_init: float = 0.5,  # 初始值较小，避免过度抑制
        tau_init: float = 2.0,  # 温度参数初始值
        learnable: bool = True,  # 参数是否可学习
    ):
        super().__init__()
        
        if learnable:
            # 可学习参数：使用log空间以保证正值
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_init).log())
            self.tau_logit = nn.Parameter(torch.tensor(tau_init).log())
        else:
            self.register_buffer('alpha_logit', torch.tensor(alpha_init).log())
            self.register_buffer('tau_logit', torch.tensor(tau_init).log())
        
        self.learnable = learnable
    
    @property
    def alpha(self) -> torch.Tensor:
        """获取alpha值（softplus保证正值）"""
        return F.softplus(self.alpha_logit) + 1e-6
    
    @property
    def tau(self) -> torch.Tensor:
        """获取tau值（softplus保证正值）"""
        return F.softplus(self.tau_logit) + 1e-6
    
    def lift_mask_to_voxel(
        self,
        pixel_mask: torch.Tensor,  # [B, T, V, H, W] 像素掩码
        voxel_indices: torch.Tensor,  # [B*T*V*N, 3] 或 [N, 3] 体素索引 (ix, iy, iz)
        inverse_indices: torch.Tensor,  # [B*T*V*N] 或 [N] 每个点对应的体素ID
        num_voxels: int,  # 唯一体素数
        points_per_voxel: Optional[torch.Tensor] = None,  # [num_voxels] 每个体素包含的点数（用于归一化）
    ) -> torch.Tensor:
        """
        将像素掩码抬升到体素域
        
        公式：M_vox_t(idx) = sigmoid(α·mean_v[mask_t,v(p) for p∈idx] / τ)
        
        Args:
            pixel_mask: 像素掩码 [B, T, V, H, W]
            voxel_indices: 体素索引，对应每个像素点所属的体素
            inverse_indices: 每个点对应的体素ID（用于scatter）
            num_voxels: 唯一体素数
            points_per_voxel: 每个体素包含的点数（可选，用于归一化）
        Returns:
            voxel_mask: 体素掩码 [B, T, num_voxels]
        """
        device = pixel_mask.device
        B, T, V, H, W = pixel_mask.shape
        
        # 展平像素掩码 [B*T*V*H*W]
        mask_flat = pixel_mask.reshape(-1)  # [B*T*V*H*W]
        
        # 确保inverse_indices和mask_flat长度一致
        if inverse_indices.shape[0] != mask_flat.shape[0]:
            # 如果体素索引只对应有效点，需要扩展掩码
            # 简化处理：假设所有像素都有对应的体素
            if inverse_indices.shape[0] < mask_flat.shape[0]:
                # 扩展到相同长度（填充无效点）
                extended_indices = torch.full((mask_flat.shape[0],), -1, device=device, dtype=inverse_indices.dtype)
                extended_indices[:inverse_indices.shape[0]] = inverse_indices
                inverse_indices = extended_indices
        
        # 计算每个体素内的掩码平均值（多视角加权）
        # 使用scatter_mean或手动实现
        valid_mask = inverse_indices >= 0
        if valid_mask.sum() == 0:
            return torch.zeros(B, T, num_voxels, device=device, dtype=pixel_mask.dtype)
        
        valid_indices = inverse_indices[valid_mask]
        valid_mask_values = mask_flat[valid_mask]
        
        # 按体素聚合：计算每个体素内掩码的平均值
        # 使用index_add实现scatter_mean
        mask_sum = torch.zeros(num_voxels, device=device, dtype=pixel_mask.dtype)
        mask_count = torch.zeros(num_voxels, device=device, dtype=torch.long)
        
        mask_sum.index_add_(0, valid_indices.long(), valid_mask_values)
        mask_count.index_add_(0, valid_indices.long(), torch.ones_like(valid_indices))
        
        # 计算平均值
        mask_mean = mask_sum / (mask_count.float() + 1e-6)  # [num_voxels]
        
        # 应用公式：sigmoid(α·mean / τ)
        alpha = self.alpha
        tau = self.tau
        
        # 缩放：alpha * mean / tau
        scaled_mask = alpha * mask_mean / tau
        
        # Sigmoid激活
        voxel_mask_flat = torch.sigmoid(scaled_mask)  # [num_voxels]
        
        # 重塑为 [B, T, num_voxels]
        # 注意：这里假设每个时间步的体素数是相同的
        # 如果不同，需要单独处理每个batch和时间步
        if voxel_mask_flat.shape[0] == num_voxels:
            # 简单情况：所有时间步共享相同的体素结构
            voxel_mask = voxel_mask_flat.unsqueeze(0).expand(B * T, -1)  # [B*T, num_voxels]
            voxel_mask = voxel_mask.reshape(B, T, num_voxels)
        else:
            # 复杂情况：不同时间步可能有不同的体素数
            # 需要根据实际的体素ID进行映射
            # 简化：假设每个时间步的体素数是num_voxels
            voxel_mask = voxel_mask_flat.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        
        return voxel_mask
    
    def lift_mask_to_voxel_per_timestep(
        self,
        pixel_mask_t: torch.Tensor,  # [B, V, H, W] 时间步t的像素掩码
        voxel_indices_t: torch.Tensor,  # [B*V*N, 3] 或 [N, 3] 体素索引
        inverse_indices_t: torch.Tensor,  # [N] 每个点对应的体素ID
        num_voxels_t: int,  # 时间步t的唯一体素数
    ) -> torch.Tensor:
        """
        对单个时间步进行掩码抬升
        
        Args:
            pixel_mask_t: 时间步t的像素掩码 [B, V, H, W]
            voxel_indices_t: 体素索引 [N, 3]
            inverse_indices_t: 体素ID映射 [N]
            num_voxels_t: 唯一体素数
        Returns:
            voxel_mask_t: 体素掩码 [B, num_voxels_t]
        """
        B, V, H, W = pixel_mask_t.shape
        
        # 展平
        mask_flat = pixel_mask_t.reshape(B, V * H * W)  # [B, V*H*W]
        
        # 处理inverse_indices
        if inverse_indices_t.shape[0] != V * H * W:
            # 需要对齐
            if inverse_indices_t.shape[0] < V * H * W:
                # 填充
                extended_indices = torch.full(
                    (V * H * W,), -1, 
                    device=pixel_mask_t.device, 
                    dtype=inverse_indices_t.dtype
                )
                extended_indices[:inverse_indices_t.shape[0]] = inverse_indices_t
                inverse_indices_t = extended_indices
        
        # 对每个batch分别处理
        voxel_mask_list = []
        for b in range(B):
            mask_b = mask_flat[b]  # [V*H*W]
            
            # 有效点
            valid_mask = inverse_indices_t >= 0
            if valid_mask.sum() == 0:
                voxel_mask_list.append(torch.zeros(num_voxels_t, device=pixel_mask_t.device))
                continue
            
            valid_indices = inverse_indices_t[valid_mask].long()
            valid_mask_values = mask_b[valid_mask]
            
            # 聚合到体素
            mask_sum = torch.zeros(num_voxels_t, device=pixel_mask_t.device, dtype=pixel_mask_t.dtype)
            mask_count = torch.zeros(num_voxels_t, device=pixel_mask_t.device, dtype=torch.long)
            
            mask_sum.index_add_(0, valid_indices, valid_mask_values)
            mask_count.index_add_(0, valid_indices, torch.ones_like(valid_indices))
            
            # 平均值
            mask_mean = mask_sum / (mask_count.float() + 1e-6)  # [num_voxels_t]
            
            # 应用公式
            alpha = self.alpha
            tau = self.tau
            scaled_mask = alpha * mask_mean / tau
            voxel_mask_b = torch.sigmoid(scaled_mask)  # [num_voxels_t]
            
            voxel_mask_list.append(voxel_mask_b)
        
        voxel_mask_t = torch.stack(voxel_mask_list, dim=0)  # [B, num_voxels_t]
        return voxel_mask_t
    
    def forward(
        self,
        pixel_mask: torch.Tensor,  # [B, T, V, H, W] 或 [B, V, H, W]
        voxel_data: Dict[str, torch.Tensor],  # 体素相关数据
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pixel_mask: 像素掩码
            voxel_data: 包含以下键的字典：
                - 'voxel_indices': [N, 3] 体素索引
                - 'inverse_indices': [N] 体素ID映射
                - 'num_voxels': 唯一体素数
                - 'timestep': 时间步（如果是单时间步）
        Returns:
            voxel_mask: 体素掩码
        """
        if pixel_mask.dim() == 4:  # [B, V, H, W] 单时间步
            return self.lift_mask_to_voxel_per_timestep(
                pixel_mask,
                voxel_data['voxel_indices'],
                voxel_data['inverse_indices'],
                voxel_data['num_voxels'],
            )
        else:  # [B, T, V, H, W] 多时间步
            # 需要按时间步分别处理（因为不同时间步的体素可能不同）
            B, T, V, H, W = pixel_mask.shape
            voxel_mask_list = []
            
            for t in range(T):
                mask_t = pixel_mask[:, t]  # [B, V, H, W]
                # 假设voxel_data包含每个时间步的数据
                if 'timestep_data' in voxel_data and t < len(voxel_data['timestep_data']):
                    voxel_data_t = voxel_data['timestep_data'][t]
                else:
                    # 使用共享的体素数据
                    voxel_data_t = voxel_data
                
                voxel_mask_t = self.lift_mask_to_voxel_per_timestep(
                    mask_t,
                    voxel_data_t.get('voxel_indices', voxel_data['voxel_indices']),
                    voxel_data_t.get('inverse_indices', voxel_data['inverse_indices']),
                    voxel_data_t.get('num_voxels', voxel_data['num_voxels']),
                )
                voxel_mask_list.append(voxel_mask_t)
            
            # 堆叠为 [B, T, num_voxels]（注意：不同时间步的num_voxels可能不同）
            # 简化：假设所有时间步的num_voxels相同，或使用padding
            if all(v.shape[1] == voxel_mask_list[0].shape[1] for v in voxel_mask_list):
                voxel_mask = torch.stack(voxel_mask_list, dim=1)  # [B, T, num_voxels]
            else:
                # 不同长度，需要padding
                max_num_voxels = max(v.shape[1] for v in voxel_mask_list)
                padded_list = []
                for v in voxel_mask_list:
                    if v.shape[1] < max_num_voxels:
                        padding = torch.zeros(v.shape[0], max_num_voxels - v.shape[1], 
                                             device=v.device, dtype=v.dtype)
                        v = torch.cat([v, padding], dim=1)
                    padded_list.append(v)
                voxel_mask = torch.stack(padded_list, dim=1)  # [B, T, max_num_voxels]
            
            return voxel_mask


