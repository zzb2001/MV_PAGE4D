# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pdb
import logging
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
from copy import deepcopy
import math

from vggt_t_mv.layers import PatchEmbed
from vggt_t_mv.layers.block import Block
from vggt_t_mv.layers.block import SpatialMaskHead_IMP as SpatialMaskHead_IMP
from vggt_t_mv.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt_t_mv.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt_t_mv.models.mask_utils import *
from vggt_t_mv.utils.epipolar_utils import compute_epipolar_bias, compute_plucker_angle_weight

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.
    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.
    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["view", "time"],  # Multi-view only: ["view", "time"] or ["time", "view"]
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_dual_stream=False,  # 启用两流架构（位姿流 vs 几何流）
        enable_sparse_global=False,  # 启用 Sparse Global-SA
        sparse_global_layers=None,  # Sparse Global-SA 应用的层索引，如 [23, 24]
        sparse_strategy="landmark",  # "landmark", "block_dilated", "memory_bank"
        enable_epipolar_prior=False,  # 启用极线/几何先验
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        
        # Frame blocks (used for Time-SA in multi-view mode, frame attention in single-view mode)
        self.frame_blocks = nn.ModuleList(
            [ block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                ) for _ in range(depth)])
        
        # Global blocks (used for View-SA in multi-view mode, global attention in single-view mode)
        self.global_blocks = nn.ModuleList(
            [block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                ) for _ in range(depth)])
        
        # View blocks (for View-SA in multi-view mode, alias to global_blocks)
        # Time blocks (for Time-SA in multi-view mode, alias to frame_blocks)
        # These are aliases for backward compatibility and weight loading
        self.view_blocks = self.global_blocks  # View-SA uses global_blocks (from Pi3 Global-Attention)
        self.time_blocks = self.frame_blocks   # Time-SA uses frame_blocks (from VGGT Frame-Attention)
        
        self.temporal_list1 = [0, 1, 2, 3, 4, 5, 6, 7]
        self.temporal_list1_mask = [7]
        self.temporal_list2 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        self.spatial_mask_head = SpatialMaskHead_IMP(embed_dim)
        
        # 两流架构：位姿流 vs 几何流
        self.enable_dual_stream = enable_dual_stream
        if enable_dual_stream:
            # 位姿流：用于相机位姿估计（抑制动态区域）
            self.pose_frame_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(depth)])
            self.pose_global_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(depth)])
            
            # 几何流：用于点云/深度估计（放大动态区域）
            self.geo_frame_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(depth)])
            self.geo_global_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(depth)])
            
            # 动态掩码偏置参数（可学习）
            # λ_pose: 位姿支路的负偏置强度（抑制动态）
            # λ_geo: 几何支路的正偏置强度（放大动态）
            self.lambda_pose_logit = nn.Parameter(torch.tensor(0.1).log())  # 初始值 0.1
            self.lambda_geo_logit = nn.Parameter(torch.tensor(0.1).log())   # 初始值 0.1
            self.lambda_clamp_value = 4.0  # clamp 到 [-4, 4]
        
        # Sparse Global-SA 配置
        self.enable_sparse_global = enable_sparse_global
        self.sparse_global_layers = sparse_global_layers if sparse_global_layers is not None else []
        self.sparse_strategy = sparse_strategy
        if enable_sparse_global:
            if sparse_strategy == "landmark":
                self.landmark_k = 64  # 每个 (t, v) 选择的 landmark tokens 数量
            elif sparse_strategy == "block_dilated":
                self.dilated_levels = [1, 2, 4]  # 扩张级别
            elif sparse_strategy == "memory_bank":
                self.memory_tokens = nn.Parameter(torch.randn(1, 32, embed_dim))  # 跨窗 memory tokens
                nn.init.normal_(self.memory_tokens, std=1e-6)
        
        # 极线/几何先验
        self.enable_epipolar_prior = enable_epipolar_prior

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        # 1, 2, 1, embed_dim
        # 1, 2, num_register_tokens, embed_dim

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,)
            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, temporal_features: torch.Tensor = None, 
                camera_intrinsics: Optional[torch.Tensor] = None,
                camera_poses: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], int, Optional[Dict]]:
        """
        Args:
            images (torch.Tensor): Input images with shape:
                - [B, S, 3, H, W]: Single-view temporal sequence (backward compatible)
                - [B, T, N, 3, H, W]: Multi-view temporal sequence (new format)
            temporal_features (torch.Tensor, optional): Temporal features for attention modulation.
            camera_intrinsics (torch.Tensor, optional): Camera intrinsics [B, T, N, 3, 3] for epipolar prior.
            camera_poses (torch.Tensor, optional): Camera poses [B, T, N, 4, 4] for epipolar prior.
            
        Returns:
            (list[torch.Tensor], int, dict):
                - output_list: The list of outputs from attention blocks
                - patch_start_idx: Where patch tokens begin
                - dual_stream_outputs: Dict with 'pose' and 'geo' streams (if enable_dual_stream=True)
        """
        # Auto-detect input format
        if len(images.shape) == 6:
            # [B, T, N, C, H, W] - Multi-view temporal format
            B, T, N, C_in, H, W = images.shape
            is_multi_view = True
        elif len(images.shape) == 5:
            # [B, S, C, H, W] - Single-view temporal format (backward compatible)
            B, S, C_in, H, W = images.shape
            is_multi_view = False
            T, N = S, 1  # For compatibility in code below
        else:
            raise ValueError(f"Expected images with shape [B, S, 3, H, W] or [B, T, N, 3, H, W], got {images.shape}")
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std
        
        if is_multi_view:
            # Multi-view temporal: [B, T, N, C, H, W] -> [B*T*N, C, H, W]
            images = images.view(B * T * N, C_in, H, W)
            total_frames = B * T * N
            S = T * N  # Total sequence length for compatibility
        else:
            # Single-view temporal: [B, S, C, H, W] -> [B*S, C, H, W]
            images = images.view(B * S, C_in, H, W)
            total_frames = B * S
            T, N = S, 1  # For compatibility

        Visual = False
        if Visual:
            _image_ = normalize_feature_batch(images[0].permute(1,2,0))
            _image_ = Image.fromarray((_image_.cpu().view(H,W,3).clamp(0, 1) * 255).byte().numpy()).resize((400,400), resample=Image.BILINEAR)  
            _image_.save("image-first.png")

            _image_ = normalize_feature_batch(images[S-1].permute(1,2,0))
            _image_ = Image.fromarray((_image_.cpu().view(H,W,3).clamp(0, 1) * 255).byte().numpy()).resize((400,400), resample=Image.BILINEAR)  
            _image_.save("image-last.png")

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        _, P, C = patch_tokens.shape
        
        if is_multi_view:
            # Organize as [B, T, N, P, C] for multi-view processing
            patch_tokens = patch_tokens.view(B, T, N, P, C)
            # Expand camera and register tokens: [1, 2, 1, C] -> [B, T, N, 1, C]
            # Use first token for (t=0, v=0), second token for others
            camera_token = self.camera_token.expand(B, -1, -1, -1)  # [B, 2, 1, C]
            register_token = self.register_token.expand(B, -1, -1, -1)  # [B, 2, R, C]
            
            # Create tokens: first (t=0,v=0) uses first token, others use second token
            camera_tokens_list = []
            register_tokens_list = []
            for t in range(T):
                for v in range(N):
                    if t == 0 and v == 0:
                        cam_tok = camera_token[:, 0:1, :, :]  # [B, 1, 1, C]
                        reg_tok = register_token[:, 0:1, :, :]  # [B, 1, R, C]
                    else:
                        cam_tok = camera_token[:, 1:2, :, :]  # [B, 1, 1, C]
                        reg_tok = register_token[:, 1:2, :, :]  # [B, 1, R, C]
                    camera_tokens_list.append(cam_tok)
                    register_tokens_list.append(reg_tok)
            
            camera_token = torch.stack(camera_tokens_list, dim=1).view(B, T, N, 1, C)  # [B, T, N, 1, C]
            register_token = torch.stack(register_tokens_list, dim=1).view(B, T, N, -1, C)  # [B, T, N, R, C]
            
            # Concatenate: [B, T, N, 1+R+P, C]
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=3)  # [B, T, N, 1+R+P, C]
            _, _, _, P_total, C = tokens.shape
            P = P_total
            
            # Position encoding
            if self.rope is not None:
                pos_patch = self.position_getter(B * T * N, H // self.patch_size, W // self.patch_size, device=images.device)
                pos_patch = pos_patch.view(B, T, N, P - self.patch_start_idx, 2)
                pos_special = torch.zeros(B, T, N, self.patch_start_idx, 2, device=images.device, dtype=pos_patch.dtype)
                pos = torch.cat([pos_special, pos_patch], dim=3)  # [B, T, N, P, 2]
                pos = pos + 1  # Offset by 1
            else:
                pos = None
                
        else:
            # Single-view temporal format (backward compatible)
            # Expand camera and register tokens to match batch size and sequence length
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
            # Concatenate special tokens with patch tokens
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
            pos = None
            if self.rope is not None:
                pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)
            # update P because we added special tokens
            _, P, C = tokens.shape
        frame_idx = 0
        global_idx = 0
        view_idx = 0
        time_idx = 0
        output_list = []
        
        if is_multi_view:
            # Multi-view mode: use View-SA and Time-SA
            # Auto-convert aa_order if needed: ["frame", "global"] -> ["time", "view"]
            effective_aa_order = []
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    effective_aa_order.append("time")
                elif attn_type == "global":
                    effective_aa_order.append("view")
                elif attn_type in ["view", "time"]:
                    effective_aa_order.append(attn_type)
                else:
                    raise ValueError(f"Unknown attention type in multi-view mode: {attn_type}")
            
            # 两流架构的输出
            dual_stream_outputs = None
            if self.enable_dual_stream:
                pose_intermediates = []
                geo_intermediates = []
            
            for num_block in range(self.aa_block_num):
                frame_intermediates = []
                global_intermediates = []
                
                # 检查是否需要应用 Sparse Global-SA
                apply_sparse_global = (self.enable_sparse_global and 
                                     num_block in self.sparse_global_layers)
                
                for attn_type in effective_aa_order:
                    if self.enable_dual_stream:
                        # 两流架构：并行处理位姿流和几何流
                        tokens_pose, tokens_geo = self._process_dual_stream_attention(
                            tokens, B, T, N, P, C, 
                            block_idx=view_idx if attn_type == "view" else time_idx,
                            pos=pos, attn_type=attn_type
                        )
                        
                        if attn_type == "view":
                            # View-SA 的两流处理
                            _, view_idx, pose_inter = self._process_view_attention(
                                tokens_pose, B, T, N, P, C, view_idx, pos=pos, is_multi_view=True)
                            _, _, geo_inter = self._process_view_attention(
                                tokens_geo, B, T, N, P, C, view_idx, pos=pos, is_multi_view=True)
                            pose_intermediates.extend(pose_inter)
                            geo_intermediates.extend(geo_inter)
                            global_intermediates.extend(geo_inter)  # 几何流用于全局输出
                            tokens = tokens_geo  # 使用几何流作为主 tokens
                        else:  # time
                            _, time_idx, pose_inter = self._process_time_attention(
                                tokens_pose, B, T, N, P, C, time_idx, pos=pos, is_multi_view=True)
                            _, _, geo_inter = self._process_time_attention(
                                tokens_geo, B, T, N, P, C, time_idx, pos=pos, is_multi_view=True)
                            pose_intermediates.extend(pose_inter)
                            geo_intermediates.extend(geo_inter)
                            frame_intermediates.extend(geo_inter)
                            tokens = tokens_geo
                    else:
                        # 标准单流处理
                        if attn_type == "view":
                            # View-SA: Fixed time t, aggregate across views
                            # 应用极线先验（如果启用且提供了相机参数）
                            epipolar_bias_mask = None
                            if self.enable_epipolar_prior and camera_intrinsics is not None and camera_poses is not None:
                                # 计算极线偏置
                                # 简化：假设所有 views 在同一时刻 t
                                # 实际需要根据 camera_intrinsics 和 camera_poses 的形状来处理
                                try:
                                    # 从位置编码中提取 2D 坐标（简化）
                                    pos_2d_patches = pos[:, :, :, self.patch_start_idx:, :]  # [B, T, N, P_patch, 2]
                                    # 这里需要更完整的实现，暂时跳过
                                    epipolar_bias_mask = None
                                except:
                                    epipolar_bias_mask = None
                            
                            tokens, view_idx, intermediates = self._process_view_attention(
                                tokens, B, T, N, P, C, view_idx, pos=pos, is_multi_view=True,
                                epipolar_bias=epipolar_bias_mask
                            )
                            global_intermediates.extend(intermediates)
                        elif attn_type == "time":
                            # Time-SA: Fixed view v, aggregate across time
                            tokens, time_idx, intermediates = self._process_time_attention(
                                tokens, B, T, N, P, C, time_idx, pos=pos, is_multi_view=True)
                            frame_intermediates.extend(intermediates)
                
                # 应用 Sparse Global-SA（如果启用）
                if apply_sparse_global:
                    tokens = self._process_sparse_global_attention(
                        tokens, B, T, N, P, C, num_block, pos=pos)
                
                # Concat intermediates: [B, T, N, P, 2C]
                min_len = min(len(frame_intermediates), len(global_intermediates))
                for i in range(min_len):
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
            
            # 组织两流输出
            if self.enable_dual_stream:
                dual_stream_outputs = {
                    'pose': pose_intermediates,
                    'geo': geo_intermediates
                }
        else:
            # Single-view mode: use frame and global (backward compatible)
            # Auto-convert if user passed ["view", "time"]: convert to ["frame", "global"]
            effective_aa_order = []
            for attn_type in self.aa_order:
                if attn_type == "view":
                    effective_aa_order.append("global")
                elif attn_type == "time":
                    effective_aa_order.append("frame")
                elif attn_type in ["frame", "global"]:
                    effective_aa_order.append(attn_type)
                else:
                    raise ValueError(f"Unknown attention type in single-view mode: {attn_type}")
            
            for num_block in range(self.aa_block_num):
                for attn_type in effective_aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos)
                        if num_block in self.temporal_list1_mask:
                            cached_key_bias_1d, cached_cam_row_mask = self.spatial_mask_head(
                                tokens.detach().clone().view(B, S, P, C), 
                                self.patch_start_idx, H // self.patch_size, W // self.patch_size)
                            cached_value = cached_key_bias_1d  # (B, S*P)
                            cache_mask = cached_cam_row_mask.to(cached_value.dtype)  # (B, S*P)
                    elif attn_type == "global":
                        if num_block in self.temporal_list1:
                            tokens, global_idx, global_intermediates = self._process_global_attention(
                                tokens, B, S, P, C, global_idx, pos=pos)
                        elif num_block in self.temporal_list2:
                            tokens, global_idx, global_intermediates = self._process_global_attention(
                                tokens, B, S, P, C, global_idx, pos=pos, attn_mask=cache_mask, attn_value=cached_value)
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
            
            # 单视角模式也返回 dual_stream_outputs（如果启用）
            if self.enable_dual_stream:
                # 单视角模式下暂时不支持两流（可以后续扩展）
                dual_stream_outputs = None
        
        return output_list, self.patch_start_idx, dual_stream_outputs
    
    def _apply_mask_bias(self, attn_logits, mask, lambda_param, stream_type="pose"):
        """
        应用动态掩码偏置到注意力 logits
        
        Args:
            attn_logits: 注意力 logits [B, H, N, N] 或 [B, H, N, M]
            mask: 动态掩码 [B, N] 或 [B, M] (值域 [0, 1]，1表示静态，0表示动态)
            lambda_param: 偏置强度参数（已clamp）
            stream_type: "pose" (负偏置) 或 "geo" (正偏置)
            
        Returns:
            偏置后的 logits
        """
        # Clamp lambda to prevent explosion
        lambda_param = torch.clamp(lambda_param, -self.lambda_clamp_value, self.lambda_clamp_value)
        
        # Expand mask to match attention dimensions
        # mask: [B, M] -> [B, 1, 1, M] for broadcasting
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, M]
        
        # Apply bias: pose stream uses negative bias (suppress dynamic), geo stream uses positive (amplify)
        if stream_type == "pose":
            # A^pose_{ij} = A_{ij} - λ_pose · m_j (抑制动态)
            bias = -lambda_param * (1.0 - mask)  # (1-mask): 1表示动态区域
        else:  # geo
            # A^geo_{ij} = A_{ij} + λ_geo · m_j (放大动态)
            bias = lambda_param * (1.0 - mask)  # 对动态区域加正偏置
        
        return attn_logits + bias
    
    def _select_landmark_tokens(self, tokens, confidence=None, k=64):
        """
        选择 landmark tokens（用于 Landmark Attention）
        
        Args:
            tokens: [B, T, N, P, C] 或 [B, N, P, C]
            confidence: 置信度图 [B, T, N, P] 或 [B, N, P]，可选
            k: 选择的 landmark 数量
            
        Returns:
            landmark_indices: [B, T, N, K] 或 [B, N, K]
        """
        if len(tokens.shape) == 5:
            B, T, N, P, C = tokens.shape
            tokens_flat = tokens.view(B * T * N, P, C)
            if confidence is not None:
                conf_flat = confidence.view(B * T * N, P)
            else:
                # 使用 token 的 L2 范数作为置信度
                conf_flat = torch.norm(tokens_flat, dim=-1)  # [B*T*N, P]
            
            # Top-k selection
            _, landmark_indices = torch.topk(conf_flat, k=min(k, P), dim=1)  # [B*T*N, K]
            landmark_indices = landmark_indices.view(B, T, N, -1)
        else:
            B, N, P, C = tokens.shape
            tokens_flat = tokens.view(B * N, P, C)
            if confidence is not None:
                conf_flat = confidence.view(B * N, P)
            else:
                conf_flat = torch.norm(tokens_flat, dim=-1)
            
            _, landmark_indices = torch.topk(conf_flat, k=min(k, P), dim=1)
            landmark_indices = landmark_indices.view(B, N, -1)
        
        return landmark_indices
    
    def _get_dilated_neighbors(self, t, v, T, N, dilated_levels=[1, 2, 4]):
        """
        获取扩张邻居（用于 Block-Sparse + Dilated Attention）
        
        Returns:
            neighbor_list: List of (t_neighbor, v_neighbor) tuples
        """
        neighbors = []
        for level in dilated_levels:
            for dt in [-level, 0, level]:
                for dv in [-level, 0, level]:
                    if dt == 0 and dv == 0:
                        continue
                    t_neighbor = t + dt
                    v_neighbor = v + dv
                    if 0 <= t_neighbor < T and 0 <= v_neighbor < N:
                        neighbors.append((t_neighbor, v_neighbor))
        return neighbors
    
    def _compute_epipolar_bias_for_view(self, pos_2d_q, pos_2d_k, K_q, K_k, T_q_to_k, patch_size=14):
        """
        为 View-SA 计算极线约束偏置
        
        Args:
            pos_2d_q: Query 2D位置 [B, T, N_q, P, 2]
            pos_2d_k: Key 2D位置 [B, T, N_k, P, 2]
            K_q: Query 相机内参 [B, T, N_q, 3, 3]
            K_k: Key 相机内参 [B, T, N_k, 3, 3]
            T_q_to_k: 变换矩阵 [B, T, N_q, N_k, 4, 4]
            patch_size: patch 大小
            
        Returns:
            epipolar_bias: [B, T, N_q, P, N_k, P]
        """
        return compute_epipolar_bias(pos_2d_q, pos_2d_k, K_q, K_k, T_q_to_k, patch_size)
    
    def _process_sparse_global_attention(self, tokens, B, T, N, P, C, block_idx, pos=None):
        """
        Sparse Global-SA: 全局稀疏长程依赖
        
        Args:
            tokens: [B, T, N, P, C]
            block_idx: 当前 block 索引
            pos: 位置编码 [B, T, N, P, 2]
            
        Returns:
            tokens: 处理后的 tokens [B, T, N, P, C]
        """
        if self.sparse_strategy == "landmark":
            # Landmark/Anchor Attention: 只与 landmark tokens 互注意
            landmark_indices = self._select_landmark_tokens(tokens, k=self.landmark_k)
            B, T, N, K = landmark_indices.shape
            
            # 选择 landmark tokens
            tokens_flat = tokens.view(B * T * N, P, C)
            landmark_tokens_list = []
            for b in range(B):
                for t in range(T):
                    for v in range(N):
                        indices = landmark_indices[b, t, v]  # [K]
                        landmark_tokens = tokens_flat[b * T * N + t * N + v, indices]  # [K, C]
                        landmark_tokens_list.append(landmark_tokens)
            
            landmark_tokens = torch.stack(landmark_tokens_list, dim=0).view(B, T, N, K, C)
            # 使用 View-SA 风格的注意力，但只对 landmark tokens
            # 简化实现：使用全局 attention block，但只处理 landmark tokens
            # 实际实现中，需要修改 attention 机制来支持稀疏连接
            
        elif self.sparse_strategy == "block_dilated":
            # Block-Sparse + Dilated: 按 (t, v) 网格做环形/跳跃块注意
            # 在实际实现中，需要创建稀疏的 attention mask
            pass
            
        elif self.sparse_strategy == "memory_bank":
            # Memory Bank: 维护跨窗 memory tokens
            B, T, N, P, C = tokens.shape
            memory_tokens = self.memory_tokens.expand(B, -1, -1)  # [B, M, C]
            
            # 将 memory tokens 与所有 tokens 进行轻注意
            # 简化实现：concat memory tokens 到 tokens
            tokens_with_memory = torch.cat([memory_tokens.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1, -1), tokens], dim=3)
            # 使用 global_blocks 处理，但只应用一次轻注意
            
        return tokens
    
    def _process_dual_stream_attention(self, tokens, B, T, N, P, C, block_idx, pos=None, 
                                       attn_type="view", dynamic_mask=None):
        """
        两流架构：位姿流和几何流并行处理
        
        Args:
            tokens: [B, T, N, P, C]
            block_idx: 当前 block 索引
            pos: 位置编码
            attn_type: "view" 或 "time"
            dynamic_mask: 动态掩码 [B, T, N, P] 或 None
            
        Returns:
            tokens_pose: 位姿流输出 [B, T, N, P, C]
            tokens_geo: 几何流输出 [B, T, N, P, C]
        """
        if not self.enable_dual_stream:
            # 如果未启用两流，返回相同的 tokens
            return tokens, tokens
        
        # 获取 lambda 参数（经过 softplus + clamp）
        lambda_pose = torch.clamp(F.softplus(self.lambda_pose_logit), 0.0, self.lambda_clamp_value)
        lambda_geo = torch.clamp(F.softplus(self.lambda_geo_logit), 0.0, self.lambda_clamp_value)
        
        # 生成动态掩码（如果未提供）
        if dynamic_mask is None:
            # 使用 spatial_mask_head 生成掩码
            tokens_for_mask = tokens.view(B * T * N, P, C)
            _, H, W = tokens.shape[2:5]  # 简化：假设可以获取 H, W
            key_bias_1d, cam_row_mask = self.spatial_mask_head(
                tokens_for_mask.view(B, T * N, P, C), 
                self.patch_start_idx, 
                H, W
            )
            # 转换为 [0, 1] 范围的掩码（1=静态，0=动态）
            dynamic_mask = 1.0 - torch.sigmoid(key_bias_1d).view(B, T, N, P)
        
        # 位姿流：使用 pose_blocks，应用负偏置（抑制动态）
        if attn_type == "view":
            tokens_pose = self._process_view_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos, 
                stream="pose", blocks=self.pose_global_blocks,
                lambda_param=lambda_pose, mask=dynamic_mask
            )
            tokens_geo = self._process_view_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos,
                stream="geo", blocks=self.geo_global_blocks,
                lambda_param=lambda_geo, mask=dynamic_mask
            )
        else:  # time
            tokens_pose = self._process_time_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos,
                stream="pose", blocks=self.pose_frame_blocks,
                lambda_param=lambda_pose, mask=dynamic_mask
            )
            tokens_geo = self._process_time_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos,
                stream="geo", blocks=self.geo_frame_blocks,
                lambda_param=lambda_geo, mask=dynamic_mask
            )
        
        return tokens_pose, tokens_geo
    
    def _process_view_attention_dual_stream(self, tokens, B, T, N, P, C, block_idx, pos,
                                           stream, blocks, lambda_param, mask):
        """
        两流架构的 View-SA 处理（内部方法）
        """
        # Reshape: [B, T, N, P, C] -> [B*T, N*P, C]
        tokens_flat = tokens.view(B * T, N * P, C)
        mask_flat = mask.view(B * T, N * P) if mask is not None else None
        
        # 应用注意力（需要在 attention 层中集成 mask bias）
        # 简化实现：先使用标准处理，后续在 attention 中集成
        pos_flat = pos.view(B * T, N * P, 2) if pos is not None else None
        
        block = blocks[block_idx]
        tokens_flat = block(tokens_flat, pos=pos_flat)
        
        return tokens_flat.view(B, T, N, P, C)
    
    def _process_time_attention_dual_stream(self, tokens, B, T, N, P, C, block_idx, pos,
                                            stream, blocks, lambda_param, mask):
        """
        两流架构的 Time-SA 处理（内部方法）
        """
        # Reshape: [B, T, N, P, C] -> [B*N, T*P, C]
        tokens_transposed = tokens.transpose(1, 2).contiguous()
        tokens_flat = tokens_transposed.view(B * N, T * P, C)
        mask_flat = mask.transpose(1, 2).contiguous().view(B * N, T * P) if mask is not None else None
        
        pos_flat = pos.transpose(1, 2).contiguous().view(B * N, T * P, 2) if pos is not None else None
        
        block = blocks[block_idx]
        tokens_flat = block(tokens_flat, pos=pos_flat)
        
        return tokens_flat.view(B, N, T, P, C).transpose(1, 2).contiguous()

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, attn_mask=None, attn_value=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        intermediates = []
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            block = self.frame_blocks[frame_idx]
            if self.training:
                fn = functools.partial(block, attn_mask=attn_mask, attn_value=attn_value)
                tokens = torch.utils.checkpoint.checkpoint(
                    fn, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = block(tokens, pos=pos, attn_mask=attn_mask, attn_value=attn_value)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, temporal_features=None, attn_mask=None, attn_value=None):
        """ Process global attention blocks. We keep tokens in shape (B, S*P, C). """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
        intermediates = []
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]
            if self.training:
                fn = functools.partial(block, temporal_features=None, S=S, P=P, attn_mask=attn_mask, attn_value=attn_value)
                tokens = torch.utils.checkpoint.checkpoint(
                    fn, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = block(tokens, pos=pos, temporal_features=None, S=S, P=P, attn_mask=attn_mask, attn_value=attn_value)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates

    def _process_view_attention(self, tokens, B, T, N, P, C, view_idx, pos=None, is_multi_view=False, 
                                attn_mask=None, attn_value=None, epipolar_bias=None):
        """
        View-SA: Fixed time t, aggregate across views (N views).
        Shape transformation: [B, T, N, P, C] -> [B*T, N*P, C] -> MHA -> [B, T, N, P, C]
        Uses global_blocks (can be loaded from Pi3 Global-Attention weights).
        """
        if not is_multi_view:
            raise ValueError("_process_view_attention requires multi-view mode")
        
        # Reshape: [B, T, N, P, C] -> [B*T, N*P, C]
        if tokens.shape != (B, T, N, P, C):
            tokens = tokens.view(B, T, N, P, C)
        tokens_flat = tokens.view(B * T, N * P, C)
        
        # Reshape position: [B, T, N, P, 2] -> [B*T, N*P, 2]
        pos_flat = None
        if pos is not None:
            if pos.shape != (B, T, N, P, 2):
                pos = pos.view(B, T, N, P, 2)
            pos_flat = pos.view(B * T, N * P, 2)
        
        intermediates = []
        for _ in range(self.aa_block_size):
            block = self.view_blocks[view_idx]  # Alias to global_blocks
            if self.training:
                fn = functools.partial(block, attn_mask=attn_mask, attn_value=attn_value)
                tokens_flat = torch.utils.checkpoint.checkpoint(
                    fn, tokens_flat, pos_flat, use_reentrant=self.use_reentrant)
            else:
                tokens_flat = block(tokens_flat, pos=pos_flat, attn_mask=attn_mask, attn_value=attn_value)
            view_idx += 1
            # Reshape back: [B*T, N*P, C] -> [B, T, N, P, C]
            intermediates.append(tokens_flat.view(B, T, N, P, C))
        
        # Reshape back to [B, T, N, P, C]
        tokens = tokens_flat.view(B, T, N, P, C)
        return tokens, view_idx, intermediates

    def _process_time_attention(self, tokens, B, T, N, P, C, time_idx, pos=None, is_multi_view=False, attn_mask=None, attn_value=None):
        """
        Time-SA: Fixed view v, aggregate across time (T time steps).
        Shape transformation: [B, T, N, P, C] -> [B*N, T*P, C] -> MHA -> [B, T, N, P, C]
        Uses time_blocks (alias to frame_blocks, can be loaded from VGGT Frame-Attention weights).
        """
        if not is_multi_view:
            raise ValueError("_process_time_attention requires multi-view mode")
        
        # Reshape: [B, T, N, P, C] -> [B*N, T*P, C]
        if tokens.shape != (B, T, N, P, C):
            tokens = tokens.view(B, T, N, P, C)
        # Transpose T and N: [B, T, N, P, C] -> [B, N, T, P, C] -> [B*N, T*P, C]
        tokens_transposed = tokens.transpose(1, 2).contiguous()  # [B, N, T, P, C]
        tokens_flat = tokens_transposed.view(B * N, T * P, C)
        
        # Reshape position: [B, T, N, P, 2] -> [B*N, T*P, 2]
        pos_flat = None
        if pos is not None:
            if pos.shape != (B, T, N, P, 2):
                pos = pos.view(B, T, N, P, 2)
            pos_transposed = pos.transpose(1, 2).contiguous()  # [B, N, T, P, 2]
            pos_flat = pos_transposed.view(B * N, T * P, 2)
        
        intermediates = []
        for _ in range(self.aa_block_size):
            block = self.time_blocks[time_idx]  # Alias to frame_blocks
            if self.training:
                fn = functools.partial(block, attn_mask=attn_mask, attn_value=attn_value)
                tokens_flat = torch.utils.checkpoint.checkpoint(
                    fn, tokens_flat, pos_flat, use_reentrant=self.use_reentrant)
            else:
                tokens_flat = block(tokens_flat, pos=pos_flat, attn_mask=attn_mask, attn_value=attn_value)
            time_idx += 1
            # Reshape back: [B*N, T*P, C] -> [B, N, T, P, C] -> [B, T, N, P, C]
            tokens_reshaped = tokens_flat.view(B, N, T, P, C)
            intermediates.append(tokens_reshaped.transpose(1, 2).contiguous())  # [B, T, N, P, C]
        
        # Reshape back to [B, T, N, P, C]
        tokens = tokens_flat.view(B, N, T, P, C).transpose(1, 2).contiguous()
        return tokens, time_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
