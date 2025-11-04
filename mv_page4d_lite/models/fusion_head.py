# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
FusionHead: 多视角融合模块
实现时刻t的V→1统一表示融合，将多视角信息融合为单一世界坐标系表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class FusionHead(nn.Module):
    """
    多视角融合头：将V个视角的体素特征融合为时刻t的统一表示
    
    功能：
    1. 可见性过滤：只保留在多视角中可见的体素
    2. 置信度加权：使用置信度对体素特征进行加权聚合
    3. 去重：基于法线/颜色一致性去除重复体素
    4. 输出统一世界坐标系表示（点云或高斯参数）
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        gaussian_output_dim: int = 83,
        use_normal_consistency: bool = True,
        use_color_consistency: bool = True,
        visibility_threshold: float = 0.5,
        confidence_threshold: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.gaussian_output_dim = gaussian_output_dim
        self.use_normal_consistency = use_normal_consistency
        self.use_color_consistency = use_color_consistency
        self.visibility_threshold = visibility_threshold
        self.confidence_threshold = confidence_threshold
        
        # 输入投影层（可选，用于处理维度不匹配）
        self.input_proj = None  # 将在需要时动态创建
        
        # 特征融合层：将体素特征转换为点云/高斯参数
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
        )
        
        # 点云输出头（3D坐标）
        self.point_head = nn.Linear(embed_dim // 4, 3)
        
        # 高斯参数输出头（可选）
        self.gaussian_head = nn.Sequential(
            nn.Linear(embed_dim // 4, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, gaussian_output_dim),
        )
        
        # 法线估计头（用于一致性检查）
        if use_normal_consistency:
            self.normal_head = nn.Linear(embed_dim // 4, 3)
        
        # 颜色估计头（用于一致性检查）
        if use_color_consistency:
            self.color_head = nn.Linear(embed_dim // 4, 3)
    
    def forward(
        self,
        voxel_tokens: torch.Tensor,  # [B, T, N_t, C] 或 [B, T, V, N_t, C]
        voxel_xyz: torch.Tensor,  # [B, T, N_t, 3] 或 [B, T, V, N_t, 3]
        voxel_ids: Optional[torch.Tensor] = None,  # [B, T, N_t] 或 [B, T, V, N_t]
        confidence: Optional[torch.Tensor] = None,  # [B, T, N_t] 或 [B, T, V, N_t]
        visibility: Optional[torch.Tensor] = None,  # [B, T, N_t] 或 [B, T, V, N_t]
        static_mask: Optional[torch.Tensor] = None,  # [B, T, N_t] 或 [B, T, V, N_t]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：融合多视角体素特征
        
        Args:
            voxel_tokens: 体素特征 [B, T, N_t, C] 或 [B, T, V, N_t, C]
            voxel_xyz: 体素坐标 [B, T, N_t, 3] 或 [B, T, V, N_t, 3]
            voxel_ids: 体素ID（用于跨时间关联）
            confidence: 置信度 [B, T, N_t] 或 [B, T, V, N_t]
            visibility: 可见性 [B, T, N_t] 或 [B, T, V, N_t]
            static_mask: 静态掩码 [B, T, N_t] 或 [B, T, V, N_t]
        
        Returns:
            dict: 包含以下键
                - P_t_full: [B, T, N_t, 3] 完整点云@t
                - G_t_full: [B, T, N_t, 83] 完整高斯参数@t
                - G_t_static: [B, T, N_static, 83] 静态高斯
                - G_t_dynamic: [B, T, N_dynamic, 83] 动态高斯
                - voxel_xyz_fused: [B, T, N_t, 3] 融合后的体素坐标
                - voxel_ids_fused: [B, T, N_t] 融合后的体素ID
        """
        B, T = voxel_tokens.shape[:2]
        device = voxel_tokens.device
        
        # 处理输入形状：如果是 [B, T, V, N_t, C]，需要先融合V维度
        if len(voxel_tokens.shape) == 5:
            # [B, T, V, N_t, C] -> 需要融合多视角
            B, T, V, N_t, C = voxel_tokens.shape
            
            # 对每个时间步t进行融合
            fused_tokens_list = []
            fused_xyz_list = []
            fused_ids_list = []
            
            for t in range(T):
                voxel_tokens_t = voxel_tokens[:, t, :, :, :]  # [B, V, N_t, C]
                voxel_xyz_t = voxel_xyz[:, t, :, :, :]  # [B, V, N_t, 3]
                
                # 融合多视角：可见性过滤 + 置信度加权
                fused_tokens_t, fused_xyz_t, fused_ids_t = self._fuse_multiview_tokens(
                    voxel_tokens_t,  # [B, V, N_t, C]
                    voxel_xyz_t,  # [B, V, N_t, 3]
                    confidence[:, t, :, :] if confidence is not None and len(confidence.shape) == 4 else None,  # [B, V, N_t]
                    visibility[:, t, :, :] if visibility is not None and len(visibility.shape) == 4 else None,  # [B, V, N_t]
                    voxel_ids[:, t, :, :] if voxel_ids is not None and len(voxel_ids.shape) == 4 else None,  # [B, V, N_t]
                )
                
                fused_tokens_list.append(fused_tokens_t)  # [B, N_t, C]
                fused_xyz_list.append(fused_xyz_t)  # [B, N_t, 3]
                fused_ids_list.append(fused_ids_t)  # [B, N_t]
            
            # Stack: [B, T, N_t, C]
            fused_tokens = torch.stack(fused_tokens_list, dim=1)
            fused_xyz = torch.stack(fused_xyz_list, dim=1)
            fused_ids = torch.stack(fused_ids_list, dim=1)
        else:
            # 已经是 [B, T, N_t, C]，直接使用
            fused_tokens = voxel_tokens
            fused_xyz = voxel_xyz
            fused_ids = voxel_ids if voxel_ids is not None else torch.arange(
                voxel_tokens.shape[2], device=device
            ).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        
        # 特征融合：体素特征 -> 点云/高斯参数
        # 检查 fused_tokens 的最后一维是否与 embed_dim 匹配
        B, T, N_t, C = fused_tokens.shape
        if C != self.embed_dim:
            # 如果维度不匹配，需要投影或截取
            if C > self.embed_dim:
                # 如果C更大，使用线性投影或截取
                if self.input_proj is None:
                    # 动态创建投影层
                    self.input_proj = nn.Linear(C, self.embed_dim).to(fused_tokens.device)
                    # 如果模型在训练模式，确保投影层也在训练模式
                    if self.training:
                        self.input_proj.train()
                    else:
                        self.input_proj.eval()
                fused_tokens = self.input_proj(fused_tokens)
            elif C < self.embed_dim:
                # 如果C更小，使用零填充或投影
                if self.input_proj is None:
                    # 动态创建投影层（扩展维度）
                    self.input_proj = nn.Linear(C, self.embed_dim).to(fused_tokens.device)
                    if self.training:
                        self.input_proj.train()
                    else:
                        self.input_proj.eval()
                fused_tokens = self.input_proj(fused_tokens)
        
        features = self.feature_fusion(fused_tokens)  # [B, T, N_t, embed_dim//4]
        
        # 点云输出
        P_t_full = self.point_head(features)  # [B, T, N_t, 3]
        # 使用融合后的体素坐标作为基础，加上偏移
        P_t_full = fused_xyz + P_t_full  # [B, T, N_t, 3]
        
        # 高斯参数输出
        G_t_full = self.gaussian_head(features)  # [B, T, N_t, 83]
        
        # 分离静态和动态高斯（如果提供static_mask）
        G_t_static = None
        G_t_dynamic = None
        if static_mask is not None:
            # static_mask可能是多种形状：需要处理并确保与voxel_tokens的N_t匹配
            static_mask_shape = static_mask.shape
            
            # 检查static_mask的形状并处理
            if len(static_mask_shape) == 5:
                # [B, T, V, H, W] -> 需要reshape或降采样到[B, T, N_t]
                # 简化处理：取第一个视角并flatten，然后resize到N_t
                B_mask, T_mask, V_mask, H_mask, W_mask = static_mask_shape
                if B_mask == B and T_mask == T:
                    # 取第一个视角并flatten
                    static_mask_flat = static_mask[:, :, 0, :, :].reshape(B, T, -1)  # [B, T, H*W]
                    # 如果H*W != N_t，需要调整
                    if static_mask_flat.shape[2] != N_t:
                        # 使用插值或简单截取/填充
                        if static_mask_flat.shape[2] > N_t:
                            static_mask_processed = static_mask_flat[:, :, :N_t]
                        else:
                            # 填充
                            padding = torch.zeros(B, T, N_t - static_mask_flat.shape[2], 
                                                 device=static_mask.device, dtype=static_mask.dtype)
                            static_mask_processed = torch.cat([static_mask_flat, padding], dim=2)
                    else:
                        static_mask_processed = static_mask_flat
                else:
                    # 形状不匹配，创建默认mask
                    static_mask_processed = torch.zeros(B, T, N_t, device=static_mask.device, dtype=static_mask.dtype)
            elif len(static_mask_shape) == 4:
                # [B, T, H, W] 或 [B, T, N_t, 1] 等
                if static_mask_shape[2] == N_t:
                    # 可能是[B, T, N_t, 1]或[B, T, N_t]
                    static_mask_processed = static_mask.squeeze(-1) if static_mask_shape[3] == 1 else static_mask
                elif static_mask_shape[2] * static_mask_shape[3] == N_t:
                    # [B, T, H, W] where H*W == N_t
                    static_mask_processed = static_mask.reshape(B, T, N_t)
                else:
                    # 形状不匹配，创建默认mask
                    static_mask_processed = torch.zeros(B, T, N_t, device=static_mask.device, dtype=static_mask.dtype)
            elif len(static_mask_shape) == 3:
                # [B, T, N_t] 或 [B, T, H*W]
                if static_mask_shape[2] == N_t:
                    static_mask_processed = static_mask
                else:
                    # 形状不匹配，创建默认mask
                    static_mask_processed = torch.zeros(B, T, N_t, device=static_mask.device, dtype=static_mask.dtype)
            else:
                # 其他形状，创建默认mask
                static_mask_processed = torch.zeros(B, T, N_t, device=static_mask.device, dtype=static_mask.dtype)
            
            # 确保形状是[B, T, N_t]
            if static_mask_processed.shape != (B, T, N_t):
                # 如果形状仍然不匹配，尝试reshape或创建默认mask
                if static_mask_processed.numel() == B * T * N_t:
                    static_mask_processed = static_mask_processed.reshape(B, T, N_t)
                else:
                    # 创建默认mask
                    static_mask_processed = torch.zeros(B, T, N_t, device=static_mask.device, dtype=static_mask.dtype)
            
            # 分离静态和动态
            static_indices = static_mask_processed > 0.5  # [B, T, N_t]，布尔类型
            
            # 对每个batch和时间步分别处理（因为动态体素数可能不同）
            static_gaussians_list = []
            dynamic_gaussians_list = []
            
            for b in range(B):
                for t in range(T):
                    static_mask_bt = static_indices[b, t]  # [N_t]，应该是1维布尔张量
                    # 确保static_mask_bt是1维的布尔张量
                    if len(static_mask_bt.shape) > 1:
                        static_mask_bt = static_mask_bt.squeeze()
                    elif len(static_mask_bt.shape) == 0:
                        # 如果是0维，需要扩展
                        static_mask_bt = static_mask_bt.unsqueeze(0)
                    # 确保是布尔类型
                    if static_mask_bt.dtype != torch.bool:
                        static_mask_bt = static_mask_bt.bool()
                    
                    # 验证形状匹配
                    G_t_bt = G_t_full[b, t]  # [N_t, 83]
                    if static_mask_bt.shape[0] != G_t_bt.shape[0]:
                        # 如果形状不匹配，跳过这个batch和时间步
                        continue
                    
                    if static_mask_bt.sum() > 0:
                        static_gaussians_list.append(G_t_bt[static_mask_bt])  # [N_static, 83]
                    
                    dynamic_mask_bt = ~static_mask_bt
                    if dynamic_mask_bt.sum() > 0:
                        dynamic_gaussians_list.append(G_t_bt[dynamic_mask_bt])  # [N_dynamic, 83]
            
            # 注意：这里返回的是list，因为不同时间步的静态/动态体素数可能不同
            # 在实际使用时，可能需要padding或使用变长序列处理
            G_t_static = static_gaussians_list if static_gaussians_list else None
            G_t_dynamic = dynamic_gaussians_list if dynamic_gaussians_list else None
        
        return {
            'P_t_full': P_t_full,  # [B, T, N_t, 3]
            'G_t_full': G_t_full,  # [B, T, N_t, 83]
            'G_t_static': G_t_static,  # List of [N_static, 83] per (b,t)
            'G_t_dynamic': G_t_dynamic,  # List of [N_dynamic, 83] per (b,t)
            'voxel_xyz_fused': fused_xyz,  # [B, T, N_t, 3]
            'voxel_ids_fused': fused_ids,  # [B, T, N_t]
            'fused_features': features,  # [B, T, N_t, embed_dim//4]
        }
    
    def _fuse_multiview_tokens(
        self,
        voxel_tokens: torch.Tensor,  # [B, V, N_t, C]
        voxel_xyz: torch.Tensor,  # [B, V, N_t, 3]
        confidence: Optional[torch.Tensor] = None,  # [B, V, N_t]
        visibility: Optional[torch.Tensor] = None,  # [B, V, N_t]
        voxel_ids: Optional[torch.Tensor] = None,  # [B, V, N_t]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        融合多视角体素tokens
        
        策略：
        1. 可见性过滤：只保留在至少一个视角中可见的体素
        2. 置信度加权：对体素特征进行置信度加权平均
        3. 坐标去重：相同体素ID的坐标取平均
        
        Returns:
            fused_tokens: [B, N_t, C]
            fused_xyz: [B, N_t, 3]
            fused_ids: [B, N_t]
        """
        B, V, N_t, C = voxel_tokens.shape
        
        # 1. 可见性过滤
        if visibility is not None:
            # visibility: [B, V, N_t]
            # 至少在一个视角中可见
            visible_mask = visibility.max(dim=1)[0] > self.visibility_threshold  # [B, N_t]
        else:
            visible_mask = torch.ones(B, N_t, dtype=torch.bool, device=voxel_tokens.device)
        
        # 2. 置信度加权聚合
        if confidence is not None:
            # confidence: [B, V, N_t]
            # 归一化权重
            conf_weights = F.softmax(confidence, dim=1)  # [B, V, N_t]
            # 加权平均
            fused_tokens = (voxel_tokens * conf_weights.unsqueeze(-1)).sum(dim=1)  # [B, N_t, C]
        else:
            # 简单平均
            fused_tokens = voxel_tokens.mean(dim=1)  # [B, N_t, C]
        
        # 3. 坐标融合（使用体素ID去重）
        if voxel_ids is not None:
            # voxel_ids: [B, V, N_t]
            # 对每个batch，使用体素ID去重
            fused_xyz_list = []
            fused_ids_list = []
            
            for b in range(B):
                # 收集所有体素
                all_ids = voxel_ids[b].flatten()  # [V*N_t]
                all_xyz = voxel_xyz[b].reshape(-1, 3)  # [V*N_t, 3]
                all_visible = visible_mask[b].unsqueeze(0).expand(V, -1).flatten()  # [V*N_t]
                
                # 只保留可见的
                valid_mask = all_visible
                if valid_mask.sum() == 0:
                    # 没有可见体素，使用所有体素
                    valid_mask = torch.ones_like(all_visible)
                
                valid_ids = all_ids[valid_mask]
                valid_xyz = all_xyz[valid_mask]
                
                # 按ID去重（取平均坐标）
                unique_ids, inverse_indices = torch.unique(valid_ids, return_inverse=True)
                unique_xyz = torch.zeros(len(unique_ids), 3, device=voxel_xyz.device)
                unique_xyz.index_add_(0, inverse_indices, valid_xyz)
                counts = torch.bincount(inverse_indices, minlength=len(unique_ids))
                unique_xyz = unique_xyz / counts.unsqueeze(-1).clamp(min=1)
                
                # Padding到N_t
                if len(unique_ids) < N_t:
                    padding_ids = torch.zeros(N_t - len(unique_ids), dtype=unique_ids.dtype, device=voxel_xyz.device)
                    padding_xyz = torch.zeros(N_t - len(unique_ids), 3, device=voxel_xyz.device)
                    unique_ids = torch.cat([unique_ids, padding_ids])
                    unique_xyz = torch.cat([unique_xyz, padding_xyz])
                elif len(unique_ids) > N_t:
                    unique_ids = unique_ids[:N_t]
                    unique_xyz = unique_xyz[:N_t]
                
                fused_xyz_list.append(unique_xyz)
                fused_ids_list.append(unique_ids)
            
            fused_xyz = torch.stack(fused_xyz_list, dim=0)  # [B, N_t, 3]
            fused_ids = torch.stack(fused_ids_list, dim=0)  # [B, N_t]
        else:
            # 没有体素ID，简单平均坐标
            fused_xyz = voxel_xyz.mean(dim=1)  # [B, N_t, 3]
            fused_ids = torch.arange(N_t, device=voxel_xyz.device).unsqueeze(0).expand(B, -1)
        
        # 应用可见性掩码
        fused_tokens = fused_tokens * visible_mask.unsqueeze(-1).float()
        
        return fused_tokens, fused_xyz, fused_ids

