"""
动态掩码小头 (Dynamic Mask Head)
生成内生掩码 M_tilde，并与外源掩码 M_ext_seq 融合得到最终掩码 M。

根据架构图2:
- 输入: 中层特征 (例如 feat0 或其后若干层输出)
- 输出: M_tilde [B,T,V,P,1] (与 M_ext_seq 对齐)
- 融合: M = sigmoid(α * M_tilde + β * M_ext_seq) → [B,T,V,P,1]
- α, β 为可学习标量或1×1MLP门控
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMaskHead(nn.Module):
    """
    动态掩码小头：生成内生掩码
    
    参数来源：PAGE-4D (checkpoint/checkpoint_150.pt 优先)，不匹配则新增结构初始化
    
    Args:
        embed_dim (int): 特征维度
        use_gating (bool): 是否使用1×1MLP门控，False则使用可学习标量
    """
    def __init__(self, embed_dim=1024, use_gating=False):
        super().__init__()
        
        # 生成内生掩码的网络
        # 从 [B,T,V,R+P,C] 或 [B,T,V,P,C] 的特征中提取
        self.mask_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),  # depthwise
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 1, 1)  # 输出单通道掩码
        )
        
        self.use_gating = use_gating
        if use_gating:
            # 1×1MLP门控：学习 α, β
            self.gate_alpha = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()  # α ∈ [0, 1]
            )
            self.gate_beta = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()  # β ∈ [0, 1]
            )
        else:
            # 修改3: 可学习标量，初始化α=β=0.5（而不是1.0）
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, feat, patch_start_idx, H_patch, W_patch, M_ext_seq=None):
        """
        生成动态掩码
        
        Args:
            feat: 中层特征 [B, T, V, R+P, C] 或 [B, T, V, P, C]
            patch_start_idx: patch tokens 的起始索引（R+1，包括camera token）
            H_patch: patch 高度
            W_patch: patch 宽度
            M_ext_seq: 外源掩码 [B, T, V, P, 1]，可选
            
        Returns:
            M: 最终掩码 [B, T, V, P, 1]
            M_tilde: 内生掩码 [B, T, V, P, 1]（用于调试/可视化）
        """
        B, T, V, P_total, C = feat.shape
        
        # 提取 patch tokens（去掉 camera token 和 register tokens）
        # 如果 P_total = R+P，则 patch 部分从 patch_start_idx 开始
        if P_total > H_patch * W_patch:
            # 包含特殊 tokens，只取 patch 部分
            feat_patch = feat[:, :, :, patch_start_idx:, :]  # [B, T, V, P, C]
        else:
            # 只有 patch tokens
            feat_patch = feat  # [B, T, V, P, C]
        
        P = feat_patch.shape[3]  # 实际 patch 数量
        
        # Reshape 为卷积格式: [B*T*V, C, H_patch, W_patch]
        feat_patch_2d = feat_patch.view(B * T * V, P, C)
        feat_patch_2d = feat_patch_2d.transpose(1, 2)  # [B*T*V, C, P]
        feat_patch_2d = feat_patch_2d.view(B * T * V, C, H_patch, W_patch)
        
        # 生成内生掩码 M_tilde
        M_tilde_logit = self.mask_head(feat_patch_2d)  # [B*T*V, 1, H_patch, W_patch]
        M_tilde = M_tilde_logit.view(B, T, V, H_patch * W_patch, 1)  # [B, T, V, P, 1]
        
        # 修改3: 归一化M_tilde为零均值单位方差
        M_tilde_mean = M_tilde.mean(dim=3, keepdim=True)  # [B, T, V, 1, 1]
        M_tilde_std = M_tilde.std(dim=3, keepdim=True) + 1e-8
        M_tilde_normalized = (M_tilde - M_tilde_mean) / M_tilde_std
        
        # 如果没有提供外源掩码，直接返回归一化后的内生掩码（经过 sigmoid）
        if M_ext_seq is None:
            M = torch.sigmoid(M_tilde_normalized)
            return M, M_tilde
        
        # 修改3: 归一化M_ext_seq为零均值单位方差
        M_ext_mean = M_ext_seq.mean(dim=3, keepdim=True)  # [B, T, V, 1, 1]
        M_ext_std = M_ext_seq.std(dim=3, keepdim=True) + 1e-8
        M_ext_normalized = (M_ext_seq - M_ext_mean) / M_ext_std
        
        # 融合：M = sigmoid(α * M_tilde_normalized + β * M_ext_normalized)
        # 如果使用门控，根据特征全局池化生成 α, β
        if self.use_gating:
            # 全局平均池化获取上下文
            feat_global = feat_patch.mean(dim=3)  # [B, T, V, C]
            feat_global = feat_global.view(B * T * V, C)
            
            alpha_gate = self.gate_alpha(feat_global).view(B, T, V, 1, 1)  # [B, T, V, 1, 1]
            beta_gate = self.gate_beta(feat_global).view(B, T, V, 1, 1)  # [B, T, V, 1, 1]
            
            # 确保 M_ext_seq 形状匹配
            if M_ext_seq.shape != M_tilde.shape:
                # 尝试广播或调整
                if M_ext_seq.shape[3] != M_tilde.shape[3]:
                    # 如果维度不匹配，插值调整
                    M_ext_seq = F.interpolate(
                        M_ext_seq.view(B * T * V, 1, -1),
                        size=M_tilde.shape[3],
                        mode='linear',
                        align_corners=False
                    ).view(B, T, V, -1, 1)
                
                # 重新归一化调整后的M_ext_seq
                M_ext_seq_mean = M_ext_seq.mean(dim=3, keepdim=True)
                M_ext_seq_std = M_ext_seq.std(dim=3, keepdim=True) + 1e-8
                M_ext_seq_normalized = (M_ext_seq - M_ext_seq_mean) / M_ext_seq_std
            else:
                M_ext_seq_normalized = M_ext_normalized
            
            M = torch.sigmoid(alpha_gate * M_tilde_normalized + beta_gate * M_ext_seq_normalized)
        else:
            # 使用可学习标量
            # 确保 M_ext_seq_normalized 形状匹配
            if M_ext_seq_normalized.shape != M_tilde_normalized.shape:
                if M_ext_seq_normalized.shape[3] != M_tilde_normalized.shape[3]:
                    M_ext_seq_normalized = F.interpolate(
                        M_ext_seq_normalized.view(B * T * V, 1, -1),
                        size=M_tilde_normalized.shape[3],
                        mode='linear',
                        align_corners=False
                    ).view(B, T, V, -1, 1)
                    # 重新归一化
                    M_ext_seq_normalized_mean = M_ext_seq_normalized.mean(dim=3, keepdim=True)
                    M_ext_seq_normalized_std = M_ext_seq_normalized.std(dim=3, keepdim=True) + 1e-8
                    M_ext_seq_normalized = (M_ext_seq_normalized - M_ext_seq_normalized_mean) / M_ext_seq_normalized_std
            
            M = torch.sigmoid(self.alpha * M_tilde_normalized + self.beta * M_ext_seq_normalized)
        
        return M, M_tilde

