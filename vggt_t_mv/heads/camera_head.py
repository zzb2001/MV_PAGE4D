# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose
from typing import Optional, Dict


class CameraHead(nn.Module):
    """
    CameraHead (相机位姿, 相对监督)
    
    根据架构图4.1:
    - 来源: Pi3 (结构兼容)
    - 输入: 从位姿流的 register/camera tokens，或对patches应用GAP
    - 输出: R9D [B,T,V,9] (通过SVD转为正交矩阵), t [B,T,V,3]
    - 损失: 所有(t,v)和(t',v')的成对相对位姿损失
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "R9D_t",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        use_gap: bool = False,  # 是否使用GAP而不是register/camera tokens
    ):
        super().__init__()

        if pose_encoding_type == "R9D_t":
            self.pose_dim = 9  # R9D: 9维旋转表示
            self.trans_dim = 3  # t: 3维平移
        elif pose_encoding_type == "absT_quaR_FoV":
            self.pose_dim = 7  # quat (4) + FOV (2) + T (1 在旧版本中)
            self.trans_dim = 3
            # 向后兼容
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        
        self.pose_encoding_type = pose_encoding_type
        self.use_gap = use_gap
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        
        # GAP projection (如果使用GAP)
        if use_gap:
            self.gap_proj = nn.Linear(dim_in, dim_in)
        
        # 维度适配投影（如果输入的tokens维度与dim_in不匹配）
        self.dim_adapter = None  # 在forward中动态创建

        # Learnable empty camera pose token
        if pose_encoding_type == "R9D_t":
            self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.pose_dim + self.trans_dim))
        else:
            self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, 9))  # 向后兼容
        
        self.embed_pose = nn.Linear(self.pose_dim + self.trans_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        
        # Pose branch: 输出 R9D + t
        self.pose_branch = Mlp(
            in_features=dim_in, 
            hidden_features=dim_in // 2, 
            out_features=self.pose_dim + self.trans_dim, 
            drop=0
        )

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4, 
                dual_stream_outputs: Optional[Dict] = None) -> list:
        """
        Forward pass to predict camera parameters.
        
        根据架构图4.1:
        - 输入: 从位姿流的 register/camera tokens，或对patches应用GAP
        - 输出: R9D [B,T,V,9] (通过SVD转为正交矩阵), t [B,T,V,3]

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            dual_stream_outputs (dict, optional): 两流架构输出，包含'pose'流特征

        Returns:
            list: A list of predicted camera encodings from each iteration.
                Each encoding has shape [B, T, V, 12] where:
                - [:, :, :, :9] = R9D (9维旋转表示，需通过SVD转为3x3正交矩阵)
                - [:, :, :, 9:] = t (3维平移)
        """
        # Use tokens from the last block for camera prediction
        tokens = aggregated_tokens_list[-1]
        
        # 如果使用两流架构，优先使用位姿流的输出
        if dual_stream_outputs is not None and 'pose' in dual_stream_outputs:
            pose_tokens_list = dual_stream_outputs['pose']
            if len(pose_tokens_list) > 0:
                tokens = pose_tokens_list[-1]  # 使用位姿流最后一个中间层
        
        # 检测输入格式：[B, S, P, C] 或 [B, T, V, P, C]
        if len(tokens.shape) == 4:
            # 单视角格式: [B, S, P, C]
            B, S, P, C = tokens.shape
            T, V = S, 1
        elif len(tokens.shape) == 5:
            # 多视角格式: [B, T, V, P, C]
            B, T, V, P, C = tokens.shape
            S = T * V
        else:
            raise ValueError(f"Unexpected token shape: {tokens.shape}")

        if self.use_gap:
            # 对patches应用GAP
            # 假设patch tokens从某个索引开始
            patch_tokens = tokens[..., 1:, :] if len(tokens.shape) == 4 else tokens[:, :, :, 1:, :]
            if len(tokens.shape) == 4:
                pose_tokens = patch_tokens.mean(dim=2)  # [B, S, C]
            else:
                pose_tokens = patch_tokens.mean(dim=3)  # [B, T, V, C]
                pose_tokens = pose_tokens.view(B * T * V, C)  # [B*T*V, C]
            pose_tokens = self.gap_proj(pose_tokens)
        else:
            # 从位姿流提取register/camera tokens
            if len(tokens.shape) == 4:
                # [B, S, P, C] -> 取camera token (index 0) 和 register tokens
                pose_tokens = tokens[:, :, 0]  # [B, S, C]
            else:
                # [B, T, V, P, C] -> 取camera token和register tokens
                pose_tokens = tokens[:, :, :, 0, :]  # [B, T, V, C]
                pose_tokens = pose_tokens.view(B * T * V, C)  # [B*T*V, C]
        
        # 维度适配：如果输入维度不等于dim_in，添加投影层
        if pose_tokens.shape[-1] != self.token_norm.normalized_shape[0]:
            if self.dim_adapter is None:
                # 动态创建维度适配器并注册为子模块
                input_dim = pose_tokens.shape[-1]
                self.dim_adapter = nn.Linear(input_dim, self.token_norm.normalized_shape[0])
                self.dim_adapter = self.dim_adapter.to(pose_tokens.device)
                # 注册为子模块，使其成为模型的一部分
                self.add_module('dim_adapter', self.dim_adapter)
            pose_tokens = self.dim_adapter(pose_tokens)
        
        pose_tokens = self.token_norm(pose_tokens)
        
        if len(tokens.shape) == 5:
            # 需要reshape回[B, T, V, dim_in]进行处理
            # 注意：经过维度适配后，C已经是dim_in了
            dim_in = pose_tokens.shape[-1]
            pose_tokens = pose_tokens.view(B, T, V, dim_in)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens 
                - [B, S, C] for single-view format
                - [B, T, V, C] for multi-view format
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of camera encodings from each iteration.
                Each encoding: [B, T, V, 12] (R9D [9] + t [3])
        """
        # 处理不同的输入格式
        if len(pose_tokens.shape) == 3:
            # [B, S, C] -> 转换为 [B, T, V, C]
            B, S, C = pose_tokens.shape
            # 假设S = T*V，需要知道具体的T和V
            # 这里简化处理，假设V=1
            T, V = S, 1
            pose_tokens = pose_tokens.view(B, T, V, C)
        else:
            B, T, V, C = pose_tokens.shape
        
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B * T * V, 1, -1))
                module_input = module_input.view(B, T, V, C)
            else:
                # Detach the previous prediction to avoid backprop through time
                pred_pose_enc_detached = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc_detached.view(B * T * V, -1))
                module_input = module_input.view(B, T, V, C)

            # Generate modulation parameters
            module_input_flat = module_input.view(B * T * V, C)
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input_flat).chunk(3, dim=-1)
            shift_msa = shift_msa.view(B, T, V, C)
            scale_msa = scale_msa.view(B, T, V, C)
            gate_msa = gate_msa.view(B, T, V, C)

            # Adaptive layer normalization and modulation
            pose_tokens_flat = pose_tokens.view(B * T * V, C)
            pose_tokens_modulated_flat = gate_msa.view(B * T * V, C) * modulate(
                self.adaln_norm(pose_tokens_flat), 
                shift_msa.view(B * T * V, C), 
                scale_msa.view(B * T * V, C)
            )
            pose_tokens_modulated = pose_tokens_modulated_flat.view(B, T, V, C) + pose_tokens

            # Apply trunk
            # trunk期望输入形状为[B, N, C]，其中N是序列长度
            # 对于camera tokens，每个(B, T, V)位置是一个token，所以N=1
            pose_tokens_modulated_flat = pose_tokens_modulated.view(B * T * V, C)
            # 添加序列维度：[B*T*V, C] -> [B*T*V, 1, C]
            pose_tokens_modulated_flat = pose_tokens_modulated_flat.unsqueeze(1)  # [B*T*V, 1, C]
            pose_tokens_modulated_flat = self.trunk(pose_tokens_modulated_flat)  # [B*T*V, 1, C]
            # 移除序列维度：[B*T*V, 1, C] -> [B*T*V, C]
            pose_tokens_modulated_flat = pose_tokens_modulated_flat.squeeze(1)  # [B*T*V, C]
            
            # Compute the delta update for the pose encoding
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated_flat))
            pred_pose_enc_delta = pred_pose_enc_delta.view(B, T, V, -1)  # [B, T, V, 12]

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # 根据编码类型处理输出
            if self.pose_encoding_type == "R9D_t":
                # R9D通过SVD转为正交矩阵（在loss计算时处理）
                # 这里直接返回原始输出
                pred_pose_enc_list.append(pred_pose_enc)
            else:
                # 向后兼容：使用旧的激活函数
                pred_pose_enc_flat = pred_pose_enc.view(B * T * V, -1)
                activated_pose = activate_pose(
                    pred_pose_enc_flat, 
                    trans_act=getattr(self, 'trans_act', 'linear'), 
                    quat_act=getattr(self, 'quat_act', 'linear'), 
                    fl_act=getattr(self, 'fl_act', 'relu')
                )
                pred_pose_enc_list.append(activated_pose.view(B, T, V, -1))

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
