"""
极线/几何先验工具函数
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def compute_epipolar_bias(pos_2d_q: torch.Tensor, pos_2d_k: torch.Tensor,
                         K_q: torch.Tensor, K_k: torch.Tensor,
                         T_q_to_k: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """
    计算极线约束偏置（用于 View-SA 的极线先验）
    
    Args:
        pos_2d_q: Query 位置的 2D 坐标 [B, T, N_q, P, 2] (x, y in patch coordinates)
        pos_2d_k: Key 位置的 2D 坐标 [B, T, N_k, P, 2]
        K_q: Query 相机内参 [B, T, N_q, 3, 3]
        K_k: Key 相机内参 [B, T, N_k, 3, 3]
        T_q_to_k: 从 query 到 key 的变换矩阵 [B, T, N_q, N_k, 4, 4]
        patch_size: patch 大小
        
    Returns:
        epipolar_bias: 极线偏置 [B, T, N_q, P, N_k, P] (负值表示不满足极线约束)
    """
    B, T, N_q, P, _ = pos_2d_q.shape
    _, _, N_k, P_k, _ = pos_2d_k.shape
    
    # 转换为像素坐标（从 patch 坐标）
    # patch 坐标 (x, y) -> 像素坐标 (u, v)
    pos_pixel_q = pos_2d_q * patch_size + patch_size / 2  # [B, T, N_q, P, 2]
    pos_pixel_k = pos_2d_k * patch_size + patch_size / 2  # [B, T, N_k, P_k, 2]
    
    # 转换为齐次坐标
    ones_q = torch.ones(B, T, N_q, P, 1, device=pos_2d_q.device)
    ones_k = torch.ones(B, T, N_k, P_k, 1, device=pos_2d_k.device)
    p_homo_q = torch.cat([pos_pixel_q, ones_q], dim=-1)  # [B, T, N_q, P, 3]
    p_homo_k = torch.cat([pos_pixel_k, ones_k], dim=-1)  # [B, T, N_k, P_k, 3]
    
    # 计算极线
    # 对于每个 (query_view, key_view) pair，计算极线约束
    epipolar_bias = torch.zeros(B, T, N_q, P, N_k, P_k, device=pos_2d_q.device)
    
    for b in range(B):
        for t in range(T):
            for v_q in range(N_q):
                for v_k in range(N_k):
                    if v_q == v_k:
                        continue  # 同一视角，跳过
                    
                    # 获取变换矩阵
                    T_mat = T_q_to_k[b, t, v_q, v_k]  # [4, 4]
                    K_q_mat = K_q[b, t, v_q]  # [3, 3]
                    K_k_mat = K_k[b, t, v_k]  # [3, 3]
                    
                    # 提取旋转和平移
                    R = T_mat[:3, :3]  # [3, 3]
                    t = T_mat[:3, 3:4]  # [3, 1]
                    
                    # 计算本质矩阵 E = [t]_x R
                    t_cross = torch.tensor([
                        [0, -t[2, 0], t[1, 0]],
                        [t[2, 0], 0, -t[0, 0]],
                        [-t[1, 0], t[0, 0], 0]
                    ], device=T_mat.device, dtype=T_mat.dtype)
                    E = t_cross @ R  # [3, 3]
                    
                    # 计算基础矩阵 F = K_k^{-T} E K_q^{-1}
                    K_q_inv = torch.inverse(K_q_mat)
                    K_k_inv = torch.inverse(K_k_mat)
                    F = K_k_inv.T @ E @ K_q_inv  # [3, 3]
                    
                    # 对每个 query point，计算到 key view 中对应极线的距离
                    for p_q in range(P):
                        p_q_homo = p_homo_q[b, t, v_q, p_q]  # [3]
                        
                        # 计算极线 l = F @ p_q
                        l = F @ p_q_homo  # [3]
                        l_norm = torch.norm(l[:2])
                        if l_norm > 1e-6:
                            l = l / l_norm
                        
                        # 计算所有 key points 到极线的距离
                        for p_k in range(P_k):
                            p_k_homo = p_homo_k[b, t, v_k, p_k]  # [3]
                            # 点到直线的距离：|ax + by + c| / sqrt(a^2 + b^2)
                            distance = torch.abs((l[0] * p_k_homo[0] + l[1] * p_k_homo[1] + l[2]) / l_norm)
                            
                            # 转换为偏置（距离越大，偏置越小/越负）
                            # 使用负指数：bias = -exp(-distance/threshold)
                            threshold = 2.0  # 阈值（像素）
                            bias = -torch.exp(-distance / threshold)
                            epipolar_bias[b, t, v_q, p_q, v_k, p_k] = bias
    
    return epipolar_bias


def compute_plucker_angle_weight(pos_3d_q: torch.Tensor, pos_3d_k: torch.Tensor) -> torch.Tensor:
    """
    计算 Plücker 光线角度加权（用于几何先验）
    
    Args:
        pos_3d_q: Query 的 3D 位置 [B, T, N_q, P, 3]
        pos_3d_k: Key 的 3D 位置 [B, T, N_k, P, 3]
        
    Returns:
        angle_weights: 角度权重 [B, T, N_q, P, N_k, P] (值域 [0, 1])
    """
    # 归一化到单位向量
    pos_3d_q_norm = F.normalize(pos_3d_q, p=2, dim=-1)  # [B, T, N_q, P, 3]
    pos_3d_k_norm = F.normalize(pos_3d_k, p=2, dim=-1)  # [B, T, N_k, P, 3]
    
    # 计算余弦相似度（内积）
    # Expand for broadcasting: [B, T, N_q, P, 1, 3] @ [B, T, 1, 1, N_k, P, 3]
    pos_q_expanded = pos_3d_q_norm.unsqueeze(4).unsqueeze(4)  # [B, T, N_q, P, 1, 1, 3]
    pos_k_expanded = pos_3d_k_norm.unsqueeze(2).unsqueeze(2)  # [B, T, 1, 1, N_k, P, 3]
    
    cosine_sim = (pos_q_expanded * pos_k_expanded).sum(dim=-1)  # [B, T, N_q, P, N_k, P]
    
    # 转换为角度权重（余弦越大，权重越大）
    # 使用 softmax-like 变换：weight = (cosine + 1) / 2，然后归一化
    angle_weights = (cosine_sim + 1.0) / 2.0  # [0, 1]
    
    return angle_weights



