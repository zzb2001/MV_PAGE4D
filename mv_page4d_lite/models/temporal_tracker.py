# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TemporalTracker: 跨时间步体素/高斯关联与轨迹维护模块
实现跨时间步的体素/高斯匹配，维护轨迹用于时序一致性约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class TemporalTracker(nn.Module):
    """
    时序跟踪器：跨时间步关联体素/高斯，维护轨迹
    
    功能：
    1. 基于体素ID的精确匹配（优先）
    2. 基于kNN + 外观/法线相似度的匹配
    3. 背景：全局SE(3)预对齐
    4. 前景：允许大位移
    5. 维护轨迹信息（速度、加速度等）
    """
    
    def __init__(
        self,
        use_knn_matching: bool = True,
        k: int = 5,  # kNN的k值
        distance_threshold: float = 0.1,  # 距离阈值（世界坐标单位）
        appearance_dim: int = 256,  # 外观特征维度
        use_se3_alignment: bool = True,  # 是否使用SE(3)对齐
    ):
        super().__init__()
        
        self.use_knn_matching = use_knn_matching
        self.k = k
        self.distance_threshold = distance_threshold
        self.use_se3_alignment = use_se3_alignment
        
        # 外观特征提取器（用于相似度匹配）
        if appearance_dim > 0:
            self.appearance_extractor = nn.Sequential(
                nn.Linear(appearance_dim, appearance_dim // 2),
                nn.ReLU(),
                nn.Linear(appearance_dim // 2, 128),
            )
        else:
            self.appearance_extractor = None
    
    def match_across_time(
        self,
        voxel_xyz_t: torch.Tensor,  # [B, N_t, 3]
        voxel_xyz_t1: torch.Tensor,  # [B, N_t1, 3]
        voxel_ids_t: Optional[torch.Tensor] = None,  # [B, N_t]
        voxel_ids_t1: Optional[torch.Tensor] = None,  # [B, N_t1]
        appearance_features_t: Optional[torch.Tensor] = None,  # [B, N_t, D]
        appearance_features_t1: Optional[torch.Tensor] = None,  # [B, N_t1, D]
        static_mask_t: Optional[torch.Tensor] = None,  # [B, N_t]
        static_mask_t1: Optional[torch.Tensor] = None,  # [B, N_t1]
    ) -> Dict[str, torch.Tensor]:
        """
        跨时间步匹配体素/高斯
        
        Args:
            voxel_xyz_t: 时间步t的体素坐标 [B, N_t, 3]
            voxel_xyz_t1: 时间步t+1的体素坐标 [B, N_t1, 3]
            voxel_ids_t: 时间步t的体素ID [B, N_t]
            voxel_ids_t1: 时间步t+1的体素ID [B, N_t1]
            appearance_features_t: 时间步t的外观特征 [B, N_t, D]
            appearance_features_t1: 时间步t+1的外观特征 [B, N_t1, D]
            static_mask_t: 时间步t的静态掩码 [B, N_t]
            static_mask_t1: 时间步t+1的静态掩码 [B, N_t1]
        
        Returns:
            dict: 包含以下键
                - matched_indices_t: [B, M] 时间步t的匹配索引
                - matched_indices_t1: [B, M] 时间步t+1的匹配索引
                - match_confidences: [B, M] 匹配置信度
                - trajectories: [B, M, 2, 3] 轨迹（t和t+1的坐标）
                - velocities: [B, M, 3] 速度（t -> t+1）
        """
        B = voxel_xyz_t.shape[0]
        device = voxel_xyz_t.device
        
        matched_indices_t_list = []
        matched_indices_t1_list = []
        match_confidences_list = []
        
        # 对每个batch分别处理
        for b in range(B):
            xyz_t_b = voxel_xyz_t[b]  # [N_t, 3]
            xyz_t1_b = voxel_xyz_t1[b]  # [N_t1, 3]
            
            # 方法1: 基于体素ID的精确匹配（优先）
            if voxel_ids_t is not None and voxel_ids_t1 is not None:
                ids_t_b = voxel_ids_t[b]  # [N_t]
                ids_t1_b = voxel_ids_t1[b]  # [N_t1]
                
                # 使用精确ID匹配
                matched_idx_t, matched_idx_t1 = self._match_by_ids(
                    ids_t_b, ids_t1_b
                )
                
                if matched_idx_t.numel() > 0:
                    # 使用ID匹配结果
                    matched_indices_t_list.append(matched_idx_t)
                    matched_indices_t1_list.append(matched_idx_t1)
                    # ID匹配的置信度为1.0
                    confidences = torch.ones(
                        matched_idx_t.numel(), device=device
                    )
                    match_confidences_list.append(confidences)
                    continue
            
            # 方法2: 基于kNN + 外观相似度的匹配
            if self.use_knn_matching:
                # SE(3)预对齐（如果是静态区域）
                if self.use_se3_alignment:
                    if static_mask_t is not None and static_mask_t1 is not None:
                        static_mask_t_b = static_mask_t[b]  # [N_t]
                        static_mask_t1_b = static_mask_t1[b]  # [N_t1]
                        
                        # 只对静态区域进行SE(3)对齐
                        static_xyz_t = xyz_t_b[static_mask_t_b]
                        static_xyz_t1 = xyz_t1_b[static_mask_t1_b]
                        
                        if static_xyz_t.shape[0] >= 3 and static_xyz_t1.shape[0] >= 3:
                            # 计算SE(3)变换
                            T = self._compute_se3_alignment(
                                static_xyz_t, static_xyz_t1
                            )
                            # 应用变换到所有点
                            xyz_t1_b_aligned = self._apply_se3_transform(
                                xyz_t1_b, T
                            )
                        else:
                            xyz_t1_b_aligned = xyz_t1_b
                    else:
                        xyz_t1_b_aligned = xyz_t1_b
                else:
                    xyz_t1_b_aligned = xyz_t1_b
                
                # kNN匹配
                matched_idx_t, matched_idx_t1, confidences = self._match_by_knn(
                    xyz_t_b,  # [N_t, 3]
                    xyz_t1_b_aligned,  # [N_t1, 3]
                    appearance_features_t[b] if appearance_features_t is not None else None,
                    appearance_features_t1[b] if appearance_features_t1 is not None else None,
                )
                
                matched_indices_t_list.append(matched_idx_t)
                matched_indices_t1_list.append(matched_idx_t1)
                match_confidences_list.append(confidences)
            else:
                # 回退：简单最近邻匹配
                distances = torch.cdist(xyz_t_b, xyz_t1_b)  # [N_t, N_t1]
                matched_idx_t1 = distances.argmin(dim=1)  # [N_t]
                matched_idx_t = torch.arange(
                    xyz_t_b.shape[0], device=device
                )
                confidences = torch.ones(
                    xyz_t_b.shape[0], device=device
                )
                
                matched_indices_t_list.append(matched_idx_t)
                matched_indices_t1_list.append(matched_idx_t1)
                match_confidences_list.append(confidences)
        
        # 合并所有batch的结果
        # 注意：不同batch的匹配数可能不同，需要padding或使用list
        max_matches = max([m.shape[0] for m in matched_indices_t_list]) if matched_indices_t_list else 0
        
        if max_matches > 0:
            # Padding到统一长度
            matched_indices_t_padded = []
            matched_indices_t1_padded = []
            match_confidences_padded = []
            
            for b in range(B):
                m = matched_indices_t_list[b].shape[0]
                if m < max_matches:
                    # Padding
                    padding_t = torch.zeros(
                        max_matches - m, dtype=matched_indices_t_list[b].dtype, device=device
                    )
                    padding_t1 = torch.zeros(
                        max_matches - m, dtype=matched_indices_t1_list[b].dtype, device=device
                    )
                    padding_conf = torch.zeros(max_matches - m, device=device)
                    
                    matched_indices_t_padded.append(
                        torch.cat([matched_indices_t_list[b], padding_t])
                    )
                    matched_indices_t1_padded.append(
                        torch.cat([matched_indices_t1_list[b], padding_t1])
                    )
                    match_confidences_padded.append(
                        torch.cat([match_confidences_list[b], padding_conf])
                    )
                else:
                    matched_indices_t_padded.append(matched_indices_t_list[b][:max_matches])
                    matched_indices_t1_padded.append(matched_indices_t1_list[b][:max_matches])
                    match_confidences_padded.append(match_confidences_list[b][:max_matches])
            
            matched_indices_t = torch.stack(matched_indices_t_padded, dim=0)  # [B, max_matches]
            matched_indices_t1 = torch.stack(matched_indices_t1_padded, dim=0)  # [B, max_matches]
            match_confidences = torch.stack(match_confidences_padded, dim=0)  # [B, max_matches]
        else:
            # 没有匹配
            matched_indices_t = torch.zeros(B, 0, dtype=torch.long, device=device)
            matched_indices_t1 = torch.zeros(B, 0, dtype=torch.long, device=device)
            match_confidences = torch.zeros(B, 0, device=device)
        
        # 计算轨迹和速度
        trajectories = []
        velocities = []
        
        for b in range(B):
            valid_mask = match_confidences[b] > 0  # [max_matches]
            if valid_mask.sum() > 0:
                valid_idx_t = matched_indices_t[b][valid_mask]  # [M]
                valid_idx_t1 = matched_indices_t1[b][valid_mask]  # [M]
                
                # 获取匹配的坐标
                xyz_matched_t = voxel_xyz_t[b][valid_idx_t]  # [M, 3]
                xyz_matched_t1 = voxel_xyz_t1[b][valid_idx_t1]  # [M, 3]
                
                # 轨迹：[M, 2, 3]
                traj = torch.stack([xyz_matched_t, xyz_matched_t1], dim=1)
                
                # 速度：[M, 3]
                vel = xyz_matched_t1 - xyz_matched_t
                
                # Padding
                if traj.shape[0] < max_matches:
                    padding = torch.zeros(
                        max_matches - traj.shape[0], 2, 3, device=device
                    )
                    traj = torch.cat([traj, padding], dim=0)
                    vel_padding = torch.zeros(
                        max_matches - vel.shape[0], 3, device=device
                    )
                    vel = torch.cat([vel, vel_padding], dim=0)
                else:
                    traj = traj[:max_matches]
                    vel = vel[:max_matches]
                
                trajectories.append(traj)
                velocities.append(vel)
            else:
                # 没有有效匹配
                trajectories.append(torch.zeros(max_matches, 2, 3, device=device))
                velocities.append(torch.zeros(max_matches, 3, device=device))
        
        trajectories = torch.stack(trajectories, dim=0)  # [B, max_matches, 2, 3]
        velocities = torch.stack(velocities, dim=0)  # [B, max_matches, 3]
        
        return {
            'matched_indices_t': matched_indices_t,  # [B, max_matches]
            'matched_indices_t1': matched_indices_t1,  # [B, max_matches]
            'match_confidences': match_confidences,  # [B, max_matches]
            'trajectories': trajectories,  # [B, max_matches, 2, 3]
            'velocities': velocities,  # [B, max_matches, 3]
        }
    
    def _match_by_ids(
        self,
        ids_t: torch.Tensor,  # [N_t]
        ids_t1: torch.Tensor,  # [N_t1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于体素ID的精确匹配
        """
        device = ids_t.device
        
        # 使用searchsorted进行高效匹配
        ids_t_sorted, sort_indices_t = torch.sort(ids_t)
        ids_t1_sorted, sort_indices_t1 = torch.sort(ids_t1)
        
        # 查找匹配
        search_indices = torch.searchsorted(ids_t1_sorted, ids_t_sorted, side='left')
        valid_mask = search_indices < ids_t1_sorted.shape[0]
        match_mask = valid_mask & (ids_t1_sorted[search_indices] == ids_t_sorted)
        
        if match_mask.sum() == 0:
            return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device)
        
        matched_sorted_indices_t = torch.where(match_mask)[0]
        matched_sorted_indices_t1 = search_indices[match_mask]
        
        # 还原到原始索引
        matched_indices_t = sort_indices_t[matched_sorted_indices_t]
        matched_indices_t1 = sort_indices_t1[matched_sorted_indices_t1]
        
        return matched_indices_t, matched_indices_t1
    
    def _match_by_knn(
        self,
        xyz_t: torch.Tensor,  # [N_t, 3]
        xyz_t1: torch.Tensor,  # [N_t1, 3]
        appearance_t: Optional[torch.Tensor] = None,  # [N_t, D]
        appearance_t1: Optional[torch.Tensor] = None,  # [N_t1, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于kNN + 外观相似度的匹配
        """
        device = xyz_t.device
        N_t = xyz_t.shape[0]
        
        if N_t == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], device=device)
            )
        
        # 计算距离
        distances = torch.cdist(xyz_t, xyz_t1)  # [N_t, N_t1]
        
        # kNN
        k = min(self.k, xyz_t1.shape[0])
        if k == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], device=device)
            )
        
        knn_distances, knn_indices = torch.topk(distances, k, dim=1, largest=False)  # [N_t, k]
        
        # 如果有外观特征，使用外观相似度进一步筛选
        if appearance_t is not None and appearance_t1 is not None:
            # 提取外观特征
            if self.appearance_extractor is not None:
                feat_t = self.appearance_extractor(appearance_t)  # [N_t, 128]
                feat_t1 = self.appearance_extractor(appearance_t1)  # [N_t1, 128]
            else:
                feat_t = appearance_t
                feat_t1 = appearance_t1
            
            # 归一化
            feat_t = F.normalize(feat_t, p=2, dim=-1)
            feat_t1 = F.normalize(feat_t1, p=2, dim=-1)
            
            # 计算外观相似度
            appearance_similarities = []
            for i in range(N_t):
                knn_feats = feat_t1[knn_indices[i]]  # [k, 128]
                sim = (feat_t[i:i+1] @ knn_feats.T).squeeze(0)  # [k]
                appearance_similarities.append(sim)
            
            appearance_similarities = torch.stack(appearance_similarities, dim=0)  # [N_t, k]
            
            # 综合距离和外观相似度
            # 距离越小越好，相似度越大越好
            distance_scores = -knn_distances  # [N_t, k] (负距离，越大越好)
            similarity_scores = appearance_similarities  # [N_t, k] (越大越好)
            
            # 归一化并加权
            distance_scores = F.softmax(distance_scores / self.distance_threshold, dim=-1)
            similarity_scores = F.softmax(similarity_scores, dim=-1)
            
            combined_scores = 0.6 * distance_scores + 0.4 * similarity_scores  # [N_t, k]
            
            # 选择最佳匹配
            best_match_idx = combined_scores.argmax(dim=1)  # [N_t]
            matched_indices_t1 = knn_indices[torch.arange(N_t), best_match_idx]  # [N_t]
            match_confidences = combined_scores[torch.arange(N_t), best_match_idx]  # [N_t]
        else:
            # 只使用距离
            matched_indices_t1 = knn_indices[:, 0]  # [N_t] (最近邻)
            match_confidences = torch.exp(-knn_distances[:, 0] / self.distance_threshold)  # [N_t]
        
        matched_indices_t = torch.arange(N_t, device=device)
        
        # 过滤低置信度匹配
        valid_mask = match_confidences > 0.1
        matched_indices_t = matched_indices_t[valid_mask]
        matched_indices_t1 = matched_indices_t1[valid_mask]
        match_confidences = match_confidences[valid_mask]
        
        return matched_indices_t, matched_indices_t1, match_confidences
    
    def _compute_se3_alignment(
        self,
        points_t: torch.Tensor,  # [N, 3]
        points_t1: torch.Tensor,  # [M, 3]
    ) -> torch.Tensor:
        """
        计算SE(3)对齐变换（最小二乘）
        
        Returns:
            T: [4, 4] 齐次变换矩阵
        """
        device = points_t.device
        
        # 计算质心
        centroid_t = points_t.mean(dim=0)  # [3]
        centroid_t1 = points_t1.mean(dim=0)  # [3]
        
        # 去中心化
        points_t_centered = points_t - centroid_t  # [N, 3]
        points_t1_centered = points_t1 - centroid_t1  # [M, 3]
        
        # 使用SVD计算旋转（如果点数足够）
        if points_t_centered.shape[0] >= 3:
            # 计算协方差矩阵
            H = points_t_centered.T @ points_t1_centered  # [3, 3]
            
            # SVD
            U, S, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T  # [3, 3]
            
            # 确保是旋转矩阵（det(R) = 1）
            if torch.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # 平移
            t = centroid_t1 - R @ centroid_t  # [3]
        else:
            # 点数不足，使用单位变换
            R = torch.eye(3, device=device)
            t = torch.zeros(3, device=device)
        
        # 构建4x4变换矩阵
        T = torch.eye(4, device=device)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def _apply_se3_transform(
        self,
        points: torch.Tensor,  # [N, 3]
        T: torch.Tensor,  # [4, 4]
    ) -> torch.Tensor:
        """
        应用SE(3)变换
        """
        # 转换为齐次坐标
        points_homo = torch.cat([
            points,
            torch.ones(points.shape[0], 1, device=points.device)
        ], dim=-1)  # [N, 4]
        
        # 变换
        points_transformed = (points_homo @ T.T)[:, :3]  # [N, 3]
        
        return points_transformed

