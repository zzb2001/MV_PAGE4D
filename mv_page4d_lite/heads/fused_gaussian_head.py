# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
FusedGaussianHead: Student端融合高斯头
从体素/聚合特征预测G_t_full（统一世界坐标系的高斯参数）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class FusedGaussianHead(nn.Module):
    """
    Student端融合高斯头：从体素tokens预测统一高斯参数
    
    功能：
    1. 从融合后的体素tokens预测高斯参数
    2. 输出G_t_full（时刻t的统一高斯场）
    3. 支持静态/动态高斯分离
    """
    
    def __init__(
        self,
        dim_in: int = 1024,
        output_dim: int = 83,  # 1(opacity) + 3(scales) + 4(rotations) + 3*25(SH_4) = 83
        sh_degree: int = 4,  # 球谐函数阶数
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.dim_in = dim_in
        self.output_dim = output_dim
        self.sh_degree = sh_degree
        self.sh_channels = 3 * ((sh_degree + 1) ** 2)  # SH系数通道数
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 高斯参数预测头
        # Opacity: 1维
        self.opacity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # opacity in [0, 1]
        )
        
        # Scales: 3维 (sx, sy, sz)
        self.scales_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )  # 输出log scales，后续exp
        
        # Rotations: 4维 (quaternion)
        self.rotations_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )  # 输出四元数，需要归一化
        
        # SH coefficients: 3 * (sh_degree+1)^2 维
        self.sh_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.sh_channels),
        )
        
        # Delta position (optional): 3维偏移
        self.delta_pos_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
    
    def forward(
        self,
        voxel_tokens: torch.Tensor,  # [B, T, N_t, C]
        voxel_xyz: Optional[torch.Tensor] = None,  # [B, T, N_t, 3]
        voxel_size: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：预测融合高斯参数
        
        Args:
            voxel_tokens: 融合后的体素特征 [B, T, N_t, C]
            voxel_xyz: 体素坐标 [B, T, N_t, 3]
            voxel_size: 体素大小（用于初始化scales）
        
        Returns:
            dict: 包含以下键
                - gaussian_params: [B, T, N_t, 83] 完整高斯参数
                - gaussian_xyz: [B, T, N_t, 3] 高斯位置
                - opacity: [B, T, N_t] 不透明度
                - scales: [B, T, N_t, 3] 尺度
                - rotations: [B, T, N_t, 4] 旋转（四元数）
                - sh_coeffs: [B, T, N_t, sh_channels] SH系数
                - delta_x: [B, T, N_t, 3] 位置偏移（可选）
        """
        B, T, N_t, C = voxel_tokens.shape
        device = voxel_tokens.device
        
        # 特征提取
        features = self.feature_extractor(voxel_tokens)  # [B, T, N_t, hidden_dim]
        
        # 预测各个高斯参数
        opacity = self.opacity_head(features).squeeze(-1)  # [B, T, N_t]
        
        # Scales: 使用exp激活确保正值
        scales_log = self.scales_head(features)  # [B, T, N_t, 3]
        scales = torch.exp(torch.clamp(scales_log, min=-10, max=10))
        # 如果提供了voxel_size，可以初始化scales
        if voxel_size is not None:
            # 将scales限制在合理范围（基于voxel_size）
            scales = scales * voxel_size * 0.5  # 默认0.5倍体素大小
        
        # Rotations: 四元数归一化
        rotations_raw = self.rotations_head(features)  # [B, T, N_t, 4]
        rotations = F.normalize(rotations_raw, p=2, dim=-1)  # 归一化四元数
        
        # SH coefficients
        sh_coeffs = self.sh_head(features)  # [B, T, N_t, sh_channels]
        
        # Delta position (可选)
        delta_x = self.delta_pos_head(features)  # [B, T, N_t, 3]
        
        # 高斯位置：体素坐标 + 偏移
        if voxel_xyz is not None:
            gaussian_xyz = voxel_xyz + delta_x  # [B, T, N_t, 3]
        else:
            gaussian_xyz = delta_x  # 如果没有提供体素坐标，只使用偏移
        
        # 组装完整高斯参数 [B, T, N_t, 83]
        # 顺序：opacity(1) + scales(3) + rotations(4) + sh_coeffs(sh_channels)
        gaussian_params = torch.cat([
            opacity.unsqueeze(-1),  # [B, T, N_t, 1]
            scales,  # [B, T, N_t, 3]
            rotations,  # [B, T, N_t, 4]
            sh_coeffs,  # [B, T, N_t, sh_channels]
        ], dim=-1)  # [B, T, N_t, 1+3+4+sh_channels]
        
        # 确保output_dim匹配
        if gaussian_params.shape[-1] != self.output_dim:
            # 如果维度不匹配，进行截断或填充
            if gaussian_params.shape[-1] > self.output_dim:
                gaussian_params = gaussian_params[..., :self.output_dim]
            else:
                padding = torch.zeros(
                    B, T, N_t, self.output_dim - gaussian_params.shape[-1],
                    device=device, dtype=gaussian_params.dtype
                )
                gaussian_params = torch.cat([gaussian_params, padding], dim=-1)
        
        return {
            'gaussian_params': gaussian_params,  # [B, T, N_t, 83]
            'gaussian_xyz': gaussian_xyz,  # [B, T, N_t, 3]
            'opacity': opacity,  # [B, T, N_t]
            'scales': scales,  # [B, T, N_t, 3]
            'rotations': rotations,  # [B, T, N_t, 4]
            'sh_coeffs': sh_coeffs,  # [B, T, N_t, sh_channels]
            'delta_x': delta_x,  # [B, T, N_t, 3]
        }

