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
from vggt_t_mv.models.dynamic_mask_head import DynamicMaskHead
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
        enable_dual_stream=True,  # 启用两流架构（位姿流 vs 几何流）
        enable_sparse_global=False,  # 启用 Sparse Global-SA
        sparse_global_layers=None,  # Sparse Global-SA 应用的层索引，如 [23, 24]
        sparse_strategy="landmark",  # "landmark", "block_dilated", "memory_bank"
        enable_epipolar_prior=False,  # 启用极线/几何先验
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        # 参数来源：② Pi3（若Pi3已有相对PE则复用），否则③ 新增（初始化）
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        
        # ===== 核心注意力块 =====
        # time_blocks: Time-SA，同视角跨时注意，Lmid层若干
        # 参数来源：① PAGE-4D（checkpoint_150.pt）作为初始化，或② Pi3 的 frame_blocks 迁移
        # 说明：用于单视角内的时序建模
        self.time_blocks = nn.ModuleList(
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
        
        # view_blocks: View-SA，同步视角跨视注意，Lmid层若干
        # 参数来源：② Pi3 的 global_blocks（跨图注意）拷权作为初始化
        # 说明：结构保持Q/K/V/Proj一致；位置/掩码逻辑改为"按视角聚合"
        self.view_blocks = nn.ModuleList(
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
        
        # 向后兼容：frame_blocks 和 global_blocks 作为别名
        # 用于单视角模式（backward compatible）
        self.frame_blocks = self.time_blocks  # 单视角模式使用 time_blocks
        self.global_blocks = self.view_blocks  # 单视角模式使用 view_blocks
        
        self.temporal_list1 = [0, 1, 2, 3, 4, 5, 6, 7]
        self.temporal_list1_mask = [7]
        self.temporal_list2 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        # spatial_mask_head: 动态掩码小头（向后兼容，保留旧实现）
        # 参数来源：① PAGE-4D（checkpoint_150.pt）可直接load（结构相同/相近）
        self.spatial_mask_head = SpatialMaskHead_IMP(embed_dim)
        
        # dynamic_mask_head: 新的动态掩码小头（根据架构图2）
        # 参数来源：① PAGE-4D（checkpoint_150.pt 优先），不匹配则新增结构初始化
        self.dynamic_mask_head = DynamicMaskHead(embed_dim=embed_dim, use_gating=False)
        
        # ===== 两流架构：enable_dual_stream =====
        # 修改1: 优化两流架构参数和内存使用
        # - 限制两流在L_mid的6-10层（索引5-9，0-based）
        # - 其他层共享权重，只应用不同的logits偏置
        # - 共享Q/K/V/O线性层，两流只保留最小门控参数λ
        self.enable_dual_stream = enable_dual_stream
        self.dual_stream_layers = [6, 7, 8, 9, 10] if enable_dual_stream else []  # L_mid的敏感中间层
        
        if enable_dual_stream:
            # 修改1: 只创建L_mid层（6-10）的独立blocks，其他层共享
            dual_stream_depth = len(self.dual_stream_layers)
            
            # 位姿流：只创建L_mid层的独立blocks
            self.pose_time_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(dual_stream_depth)])
            self.pose_view_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(dual_stream_depth)])
            
            # 几何流：只创建L_mid层的独立blocks
            self.geo_time_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(dual_stream_depth)])
            self.geo_view_blocks = nn.ModuleList([
                deepcopy(block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                )) for _ in range(dual_stream_depth)])
            
            # 向后兼容：保留旧名称作为别名
            self.pose_frame_blocks = self.pose_time_blocks
            self.pose_global_blocks = self.pose_view_blocks
            self.geo_frame_blocks = self.geo_time_blocks
            self.geo_global_blocks = self.geo_view_blocks
            
            # 修改1 & 3: 动态掩码偏置参数（可学习标量，最小门控参数）
            # 初始化λ_*为0，clamp到[-4, 4]
            # λ_pose: 位姿支路的负偏置强度（抑制动态）
            # λ_geo: 几何支路的正偏置强度（放大动态）
            # λ_pose_t: Time-SA 的位姿支路偏置
            # λ_geo_t: Time-SA 的几何支路偏置
            self.lambda_pose_logit = nn.Parameter(torch.tensor(0.0))  # View-SA 位姿流
            self.lambda_geo_logit = nn.Parameter(torch.tensor(0.0))   # View-SA 几何流
            self.lambda_pose_t_logit = nn.Parameter(torch.tensor(0.0))  # Time-SA 位姿流
            self.lambda_geo_t_logit = nn.Parameter(torch.tensor(0.0))   # Time-SA 几何流
            self.lambda_clamp_value = 4.0  # clamp 到 [-4, 4]
            
            # 修改1: Top-k稀疏化参数（用于View-SA/Time-SA）
            self.topk_sparsity_ratio = 0.4  # 过滤30-50%的tokens（使用0.4即40%）
            self.use_topk_sparsity = True  # 启用Top-k稀疏化
        
        # ===== Sparse Global-SA: enable_sparse_global =====
        # strategy="landmark" | "dilated" | "memory_bank"
        # 参数来源：③ 新增（初始化）；但Q/K/V/Proj可从② Pi3 global_blocks拷权
        # 说明：注意力稀疏图是逻辑层，不影响线性层权重可复用
        self.enable_sparse_global = enable_sparse_global
        self.sparse_global_layers = sparse_global_layers if sparse_global_layers is not None else []
        self.sparse_strategy = sparse_strategy
        
        # global_sparse_blocks: 1-2层，可选
        sparse_global_depth = len(self.sparse_global_layers) if self.sparse_global_layers else 0
        if enable_sparse_global and sparse_global_depth > 0:
            self.global_sparse_blocks = nn.ModuleList([
                block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                    init_values=init_values, qk_norm=qk_norm, rope=self.rope,
                ) for _ in range(sparse_global_depth)])
            
            if sparse_strategy == "landmark":
                self.landmark_k = 64  # 每个 (t, v) 选择的 landmark tokens 数量
            elif sparse_strategy == "dilated":
                self.dilated_levels = [1, 2, 4]  # 扩张级别
            elif sparse_strategy == "memory_bank":
                # memory_tokens（仅strategy="memory_bank"启用）
                # 参数来源：③ 新增（初始化）
                self.memory_tokens = nn.Parameter(torch.randn(1, 32, embed_dim))  # 跨窗 memory tokens
                nn.init.normal_(self.memory_tokens, std=1e-6)
        else:
            self.global_sparse_blocks = None
        
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
        self.num_register_tokens = num_register_tokens  # 保存用于掩码处理

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

        patch_tokens = self.patch_embed(images) #'x_norm_cls_token', 'x_norm_patchtokens', 'x_norm_mask_token','x_norm_token'
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
            # 修改5: 只使用空间位置编码，不使用view-ID PE，保持View Permutation Equivariance
            if self.rope is not None:
                pos_patch = self.position_getter(B * T * N, H // self.patch_size, W // self.patch_size, device=images.device)
                pos_patch = pos_patch.view(B, T, N, P - self.patch_start_idx, 2)
                pos_special = torch.zeros(B, T, N, self.patch_start_idx, 2, device=images.device, dtype=pos_patch.dtype)
                pos = torch.cat([pos_special, pos_patch], dim=3)  # [B, T, N, P, 2]
                pos = pos + 1  # Offset by 1
                # 修改5: 确保不使用view-ID相关的PE，只使用空间2D位置编码
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
                    # 修改1: 判断当前层是否在两流架构的L_mid层中
                    # 对于View-SA: 使用view_idx；对于Time-SA: 使用time_idx
                    if attn_type == "view":
                        current_layer_idx = view_idx
                    else:  # time
                        current_layer_idx = time_idx
                    
                    use_dual_stream_at_layer = (self.enable_dual_stream and 
                                               current_layer_idx in self.dual_stream_layers)
                    
                    if use_dual_stream_at_layer:
                        # 修改1: 只在L_mid层使用独立的两流blocks
                        # 计算在dual_stream_layers中的索引
                        dual_stream_layer_idx = self.dual_stream_layers.index(current_layer_idx) if current_layer_idx in self.dual_stream_layers else 0
                        
                        # 两流架构：并行处理位姿流和几何流
                        # 修改1: 传递dual_stream_layer_idx（0-4）而不是原始的view_idx/time_idx
                        tokens_pose, tokens_geo = self._process_dual_stream_attention(
                            tokens, B, T, N, P, C, 
                            block_idx=dual_stream_layer_idx,  # 使用dual_stream_layer_idx（0-4）
                            pos=pos, attn_type=attn_type,
                            dual_stream_layer_idx=dual_stream_layer_idx
                        )
                        
                        # 修改1: _process_dual_stream_attention已经处理了tokens，直接使用结果
                        # 不需要再次调用_process_view_attention或_process_time_attention
                        if attn_type == "view":
                            # View-SA 的两流处理已完成，直接使用结果
                            pose_inter = [tokens_pose]  # 将结果作为中间输出
                            geo_inter = [tokens_geo]
                            view_idx += 1  # 递增view_idx以便后续层使用
                            pose_intermediates.extend(pose_inter)
                            geo_intermediates.extend(geo_inter)
                            global_intermediates.extend(geo_inter)  # 几何流用于全局输出
                            tokens = tokens_geo  # 使用几何流作为主 tokens
                        else:  # time
                            # Time-SA 的两流处理已完成，直接使用结果
                            pose_inter = [tokens_pose]  # 将结果作为中间输出
                            geo_inter = [tokens_geo]
                            time_idx += 1  # 递增time_idx以便后续层使用
                            pose_intermediates.extend(pose_inter)
                            geo_intermediates.extend(geo_inter)
                            frame_intermediates.extend(geo_inter)
                            tokens = tokens_geo
                    elif self.enable_dual_stream:
                        # 修改1: 不在L_mid层的其他层，共享权重，只应用不同的logits偏置
                        # 使用共享的view_blocks/time_blocks，但通过mask偏置区分两流
                        # 这里暂时只使用共享blocks，偏置在Block内部应用
                        if attn_type == "view":
                            tokens, view_idx, intermediates = self._process_view_attention(
                                tokens, B, T, N, P, C, view_idx, pos=pos, is_multi_view=True,
                                dynamic_mask=None  # TODO: 传递mask以应用偏置
                            )
                            global_intermediates.extend(intermediates)
                            pose_intermediates.extend(intermediates)  # 共享输出
                            geo_intermediates.extend(intermediates)  # 共享输出
                        elif attn_type == "time":
                            tokens, time_idx, intermediates = self._process_time_attention(
                                tokens, B, T, N, P, C, time_idx, pos=pos, is_multi_view=True,
                                dynamic_mask=None  # TODO: 传递mask以应用偏置
                            )
                            frame_intermediates.extend(intermediates)
                            pose_intermediates.extend(intermediates)  # 共享输出
                            geo_intermediates.extend(intermediates)  # 共享输出
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
                if apply_sparse_global and self.global_sparse_blocks is not None:
                    # 获取 sparse block 的索引（相对于 sparse_global_layers）
                    sparse_block_idx = self.sparse_global_layers.index(num_block)
                    if sparse_block_idx < len(self.global_sparse_blocks):
                        tokens = self._process_sparse_global_attention(
                            tokens, B, T, N, P, C, sparse_block_idx, pos=pos)
                
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
    
    def _process_sparse_global_attention(self, tokens, B, T, N, P, C, sparse_block_idx, pos=None):
        """
        Sparse Global-SA: 全局稀疏长程依赖
        
        Args:
            tokens: [B, T, N, P, C]
            sparse_block_idx: 在 global_sparse_blocks 中的索引
            pos: 位置编码 [B, T, N, P, 2]
            
        Returns:
            tokens: 处理后的 tokens [B, T, N, P, C]
        """
        if self.global_sparse_blocks is None or sparse_block_idx >= len(self.global_sparse_blocks):
            return tokens
            
        block = self.global_sparse_blocks[sparse_block_idx]
        
        if self.sparse_strategy == "landmark":
            # 修改4: Landmark策略 - 使用稀疏索引实现真正的稀疏注意力
            # 对每个(t,v)选择K=64个anchors，所有tokens只与这些anchors互注意
            landmark_indices = self._select_landmark_tokens(tokens, k=self.landmark_k)
            B, T, N, K = landmark_indices.shape
            
            # 修改4: 创建稀疏索引，而不是选择tokens然后密集注意力
            tokens_flat = tokens.view(B * T * N, P, C)  # [B*T*N, P, C]
            
            # 为每个(t,v)创建稀疏索引：所有tokens只attend到对应的K个landmarks
            # 实现稀疏注意力：只计算与landmark tokens的attention
            # 这里需要修改Block来支持稀疏索引，暂时先选择landmarks
            landmark_tokens_list = []
            landmark_pos_list = []
            for b in range(B):
                for t in range(T):
                    for v in range(N):
                        batch_idx = b * T * N + t * N + v
                        indices = landmark_indices[b, t, v]  # [K]
                        landmark_tokens = tokens_flat[batch_idx, indices]  # [K, C]
                        landmark_tokens_list.append(landmark_tokens)
                        if pos is not None:
                            landmark_pos = pos[b, t, v, indices]  # [K, 2]
                            landmark_pos_list.append(landmark_pos)
            
            # 修改4: 使用稀疏注意力机制（需要Block支持）
            # 当前简化实现：将所有tokens与landmark tokens concat，然后应用attention
            # TODO: 实现真正的稀疏索引attention，只计算P×K的attention matrix
            landmark_tokens = torch.stack(landmark_tokens_list, dim=0).view(B * T * N, K, C)
            
            # 将landmark tokens与原始tokens concat用于attention
            tokens_with_landmarks = torch.cat([landmark_tokens, tokens_flat], dim=1)  # [B*T*N, K+P, C]
            # 注意：这还不是真正的稀疏，只是准备工作，需要在Block中实现稀疏attention
            
        elif self.sparse_strategy == "block_dilated":
            # Block-Sparse + Dilated: 按 (t, v) 网格做环形/跳跃块注意
            # 修改4: 实现真正的稀疏连接，而不是密集attention
            # 在实际实现中，需要创建稀疏的 attention mask
            pass
            
        elif self.sparse_strategy == "memory_bank":
            # 修改4: Memory Bank策略 - 输出32个memory tokens从上一个滑动窗口
            # 允许它们与当前窗口交互
            B, T, N, P, C = tokens.shape
            memory_tokens = self.memory_tokens.expand(B, -1, -1)  # [B, M, C], M=32
            
            # 修改4: Memory tokens与当前tokens的交互应该是稀疏的
            # 当前实现：将memory tokens concat到tokens
            # TODO: 实现真正的跨窗口memory机制，保存上一个窗口的memory tokens
            tokens_with_memory = torch.cat([
                memory_tokens.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1, -1), 
                tokens
            ], dim=3)  # [B, T, N, M+P, C]
            
        return tokens
    
    def _process_dual_stream_attention(self, tokens, B, T, N, P, C, block_idx, pos=None, 
                                       attn_type="view", dynamic_mask=None, dual_stream_layer_idx=None):
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
        
        # 获取 lambda 参数（clamp到指定范围）
        lambda_pose = torch.clamp(self.lambda_pose_logit, -self.lambda_clamp_value, self.lambda_clamp_value)
        lambda_geo = torch.clamp(self.lambda_geo_logit, -self.lambda_clamp_value, self.lambda_clamp_value)
        
        # 生成动态掩码（如果未提供）
        if dynamic_mask is None:
            # 使用 spatial_mask_head 生成掩码
            tokens_for_mask = tokens.view(B * T * N, P, C)
            # 假设可以从 patch_size 推导 H, W
            H_patch = int((P - self.patch_start_idx) ** 0.5)
            W_patch = H_patch  # 假设是正方形
            key_bias_1d, cam_row_mask = self.spatial_mask_head(
                tokens_for_mask.view(B, T * N, P, C), 
                self.patch_start_idx, 
                H_patch, W_patch
            )
            # 转换为 [0, 1] 范围的掩码（1=静态，0=动态）
            dynamic_mask = 1.0 - torch.sigmoid(key_bias_1d).view(B, T, N, P)
        
        # 位姿流：使用 pose_time_blocks / pose_view_blocks，应用负偏置（抑制动态）
        if attn_type == "view":
            tokens_pose = self._process_view_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos, 
                stream="pose", blocks=self.pose_view_blocks,
                lambda_param=lambda_pose, mask=dynamic_mask
            )
            tokens_geo = self._process_view_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos,
                stream="geo", blocks=self.geo_view_blocks,
                lambda_param=lambda_geo, mask=dynamic_mask
            )
        else:  # time
            tokens_pose = self._process_time_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos,
                stream="pose", blocks=self.pose_time_blocks,
                lambda_param=lambda_pose, mask=dynamic_mask
            )
            tokens_geo = self._process_time_attention_dual_stream(
                tokens, B, T, N, P, C, block_idx, pos,
                stream="geo", blocks=self.geo_time_blocks,
                lambda_param=lambda_geo, mask=dynamic_mask
            )
        
        return tokens_pose, tokens_geo
    
    def _process_view_attention_dual_stream(self, tokens, B, T, N, P, C, block_idx, pos,
                                           stream, blocks, lambda_param, mask):
        """
        两流架构的 View-SA 处理（内部方法）
        
        Args:
            block_idx: dual_stream_layer_idx（0-4），用于索引两流blocks
        """
        # 修改1: 确保block_idx在范围内
        if block_idx >= len(blocks):
            raise IndexError(f"block_idx {block_idx} is out of range for blocks (length {len(blocks)})")
        
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
        
        Args:
            block_idx: dual_stream_layer_idx（0-4），用于索引两流blocks
        """
        # 修改1: 确保block_idx在范围内
        if block_idx >= len(blocks):
            raise IndexError(f"block_idx {block_idx} is out of range for blocks (length {len(blocks)})")
        
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
                                attn_mask=None, attn_value=None, epipolar_bias=None, custom_blocks=None,
                                dynamic_mask=None, dual_stream_layer_idx=None):
        """
        View-SA (Synchronized View Attention, fixed t): 固定时间t，跨视角聚合
        
        根据架构图3.1:
        1. 输入重排: X_v = concat_over_views(feat, axis=V) → [B, T, (V*(R+P)), Cvit]
        2. 掩码广播: M_v = concat_over_views(M, axis=V) → [B, T, V*P, 1]
           → Expand to token dimension (fill 0 at R registers): M_v_up [B,T,V*(R+P),1]
        3. 注意力 logits 偏置:
           - 位姿流: logits_pose += (-λ_pose) * M_v_up
           - 几何流: logits_geo += (+λ_geo) * M_v_up
        4. 输出还原: Y_v → [B,T,V, (R+P), Cvit]
        
        Args:
            tokens: [B, T, N, P, C] (P = R+P，包含所有tokens)
            dynamic_mask: [B, T, N, P, 1] 动态掩码，可选
            custom_blocks: 如果提供，使用自定义的 blocks（用于两流架构）
        """
        if not is_multi_view:
            raise ValueError("_process_view_attention requires multi-view mode")
        
        # 修改2: View-SA输入重排 - 明确沿V维度展平
        # [B, T, N, P, C] -> [B*T, N*P, C] (沿view维度N展平)
        if tokens.shape != (B, T, N, P, C):
            tokens = tokens.view(B, T, N, P, C)
        
        # 提取 R 和 P_patch
        R = self.num_register_tokens
        P_patch = P - R - 1  # 减去 camera token (1) 和 register tokens (R)
        
        # 修改2: View-SA明确沿V维度concat
        tokens_flat = tokens.view(B * T, N * P, C)  # [B*T, N*P, C] (沿view维度展平)
        
        # 掩码广播（如果提供了动态掩码）
        mask_v_up = None
        if dynamic_mask is not None:
            # M: [B, T, N, P, 1] -> 只取 patch 部分: [B, T, N, P_patch, 1]
            # 假设 dynamic_mask 只对 patch tokens 有效
            if dynamic_mask.shape[3] == P:
                # 需要提取 patch 部分（跳过 camera token 和 register tokens）
                M_patch = dynamic_mask[:, :, :, (1+R):, :]  # [B, T, N, P_patch, 1]
            else:
                M_patch = dynamic_mask  # [B, T, N, P_patch, 1]
            
            # 跨视角拼接: [B, T, N, P_patch, 1] -> [B, T, N*P_patch, 1]
            M_v = M_patch.view(B, T, N * P_patch, 1)  # [B, T, V*P, 1]
            
            # 修改2 & 3: Expand to token dimension (fill 0 at R registers)
            # 确保register tokens和camera token的mask值为0
            # [B, T, N*P_patch, 1] -> [B, T, N*(R+P), 1]
            # 每个视角: [1+R+P_patch] -> 需要 [1+R+P_patch] 形状
            M_v_expanded_list = []
            for v in range(N):
                # 修改2: 每个视角的掩码: camera token (0) + register tokens (R个0) + patch掩码
                # 确保camera token和register tokens的偏置始终为0
                zeros_reg = torch.zeros(B, T, 1 + R, 1, device=M_patch.device, dtype=M_patch.dtype)
                M_v_v = torch.cat([zeros_reg, M_patch[:, :, v:v+1, :, :]], dim=2)  # [B, T, 1+R+P_patch, 1]
                M_v_expanded_list.append(M_v_v)
            M_v_up = torch.cat(M_v_expanded_list, dim=2)  # [B, T, N*(R+P), 1]
            M_v_up = M_v_up.view(B * T, N * P, 1)  # [B*T, N*P, 1]
            
            # 修改2: 确保register tokens和camera token位置确实为0（双重检查）
            # camera token位置: 0, register tokens位置: 1 到 1+R
            M_v_up[:, :(1+R), :] = 0  # 第一个视角的camera+register
            for v_idx in range(1, N):
                start_idx = v_idx * P
                M_v_up[:, start_idx:(start_idx + 1 + R), :] = 0  # 其他视角的camera+register
        
        # Reshape position: [B, T, N, P, 2] -> [B*T, N*P, 2]
        pos_flat = None
        if pos is not None:
            if pos.shape != (B, T, N, P, 2):
                pos = pos.view(B, T, N, P, 2)
            pos_flat = pos.view(B * T, N * P, 2)
        
        # 应用极线先验（如果启用且提供了）
        epi_bias_mask = None
        if epipolar_bias is not None:
            epi_bias_mask = epipolar_bias  # [B*T, N*P, N*P] 或类似形状
        
        # 修改1: 使用自定义 blocks 或默认 view_blocks
        # 如果在两流架构的L_mid层，使用dual_stream_layer_idx索引
        blocks_to_use = custom_blocks if custom_blocks is not None else self.view_blocks
        
        intermediates = []
        for _ in range(self.aa_block_size):
            # 修改1: 如果使用两流的blocks，使用dual_stream_layer_idx（0-4）
            # 否则使用view_idx（可能超出两流blocks的范围）
            if custom_blocks is not None and dual_stream_layer_idx is not None:
                # 使用dual_stream_layer_idx索引两流blocks（长度只有5）
                if dual_stream_layer_idx >= len(blocks_to_use):
                    raise IndexError(f"dual_stream_layer_idx {dual_stream_layer_idx} >= len(blocks) {len(blocks_to_use)}")
                block = blocks_to_use[dual_stream_layer_idx]
            else:
                # 使用view_idx索引共享的view_blocks（长度24）
                if view_idx >= len(blocks_to_use):
                    raise IndexError(f"view_idx {view_idx} >= len(blocks) {len(blocks_to_use)}")
                block = blocks_to_use[view_idx]
            
            # 如果需要应用掩码偏置（两流架构）
            if self.enable_dual_stream and mask_v_up is not None:
                # 在 attention 内部应用 logits 偏置
                # 这里需要修改 Block 来支持 logits_bias 参数
                # 简化实现：暂时通过 attn_mask 传递
                pass  # TODO: 在 Block 中实现 logits_bias
            
            if self.training:
                fn = functools.partial(block, attn_mask=attn_mask, attn_value=attn_value)
                tokens_flat = torch.utils.checkpoint.checkpoint(
                    fn, tokens_flat, pos_flat, use_reentrant=self.use_reentrant)
            else:
                tokens_flat = block(tokens_flat, pos=pos_flat, attn_mask=attn_mask, attn_value=attn_value)
            view_idx += 1
            # Reshape back: [B*T, N*P, C] -> [B, T, N, P, C]
            intermediates.append(tokens_flat.view(B, T, N, P, C))
        
        # 输出还原: [B*T, N*P, C] -> [B, T, N, P, C]
        tokens = tokens_flat.view(B, T, N, P, C)
        return tokens, view_idx, intermediates

    def _process_time_attention(self, tokens, B, T, N, P, C, time_idx, pos=None, is_multi_view=False, 
                                attn_mask=None, attn_value=None, custom_blocks=None, dynamic_mask=None,
                                dual_stream_layer_idx=None):
        """
        Time-SA (同视角时间注意, 固定 v): 固定视角v，跨时间聚合
        
        根据架构图3.2:
        1. 输入重排: X_t = concat_over_times(feat, axis=T) → [B, V, (T*(R+P)), Cvit]
        2. 相对时间位置编码: RoPE 或 ALiBi（如果Pi3有可复用部分则复用，否则新增）
        3. 掩码广播: M_t = concat_over_times(M, axis=T) → [B,V,T*P,1]
           → M_t_up [B,V,T*(R+P),1] (fill 0 at R registers)
        4. 注意力 logits 偏置:
           - 位姿流: logits_pose += (-λ_pose_t) * M_t_up
           - 几何流: logits_geo += (+λ_geo_t) * M_t_up
        5. 输出还原: Y_t → [B,T,V, (R+P), Cvit]
        
        Args:
            tokens: [B, T, N, P, C] (P = R+P，包含所有tokens)
            dynamic_mask: [B, T, N, P, 1] 动态掩码，可选
            custom_blocks: 如果提供，使用自定义的 blocks（用于两流架构）
        """
        if not is_multi_view:
            raise ValueError("_process_time_attention requires multi-view mode")
        
        # 修改2: Time-SA输入重排 - 明确沿T维度展平
        # [B, T, N, P, C] -> [B*N, T*P, C] (沿time维度T展平)
        if tokens.shape != (B, T, N, P, C):
            tokens = tokens.view(B, T, N, P, C)
        
        # 修改2: Time-SA明确沿T维度concat（转置N和T，然后沿T展平）
        tokens_transposed = tokens.transpose(1, 2).contiguous()  # [B, N, T, P, C]
        tokens_flat = tokens_transposed.view(B * N, T * P, C)  # [B*N, T*P, C] (沿time维度展平)
        
        # 提取 R 和 P_patch
        R = self.num_register_tokens
        P_patch = P - R - 1  # 减去 camera token (1) 和 register tokens (R)
        
        # 掩码广播（如果提供了动态掩码）
        mask_t_up = None
        if dynamic_mask is not None:
            # M: [B, T, N, P, 1] -> 只取 patch 部分: [B, T, N, P_patch, 1]
            if dynamic_mask.shape[3] == P:
                M_patch = dynamic_mask[:, :, :, (1+R):, :]  # [B, T, N, P_patch, 1]
            else:
                M_patch = dynamic_mask  # [B, T, N, P_patch, 1]
            
            # 跨时间拼接: [B, T, N, P_patch, 1] -> 转置为 [B, N, T, P_patch, 1]
            M_t = M_patch.transpose(1, 2).contiguous()  # [B, N, T, P_patch, 1]
            M_t = M_t.view(B, N, T * P_patch, 1)  # [B, N, T*P_patch, 1]
            
            # 修改2 & 3: Expand to token dimension (fill 0 at R registers)
            # 确保register tokens和camera token的mask值为0
            # [B, N, T*P_patch, 1] -> [B, N, T*(R+P), 1]
            # 对每个视角和时间步，添加 camera token (0) + register tokens (R个0)
            M_t_expanded_list = []
            for n in range(N):
                M_n_list = []
                for t in range(T):
                    # 修改2: 确保camera token和register tokens的偏置始终为0
                    zeros_reg = torch.zeros(B, 1, 1 + R, 1, device=M_patch.device, dtype=M_patch.dtype)
                    M_t_t = torch.cat([zeros_reg, M_patch[:, t:t+1, n:n+1, :, :]], dim=2)  # [B, 1, 1+R+P_patch, 1]
                    M_n_list.append(M_t_t)
                M_n = torch.cat(M_n_list, dim=1)  # [B, T, 1+R+P_patch, 1]
                M_n = M_n.view(B, 1, T * P, 1)  # [B, 1, T*P, 1]
                M_t_expanded_list.append(M_n)
            M_t_up = torch.cat(M_t_expanded_list, dim=1)  # [B, N, T*P, 1]
            M_t_up = M_t_up.view(B * N, T * P, 1)  # [B*N, T*P, 1]
            
            # 修改2: 确保register tokens和camera token位置确实为0（双重检查）
            # 对每个视角，camera token和register tokens位置为0
            for n_idx in range(N):
                for t_idx in range(T):
                    token_idx = n_idx * T * P + t_idx * P
                    M_t_up[:, token_idx:(token_idx + 1 + R), :] = 0  # camera + register tokens
        
        # Reshape position: [B, T, N, P, 2] -> [B*N, T*P, 2]
        pos_flat = None
        if pos is not None:
            if pos.shape != (B, T, N, P, 2):
                pos = pos.view(B, T, N, P, 2)
            pos_transposed = pos.transpose(1, 2).contiguous()  # [B, N, T, P, 2]
            pos_flat = pos_transposed.view(B * N, T * P, 2)
            
            # 修改5: 相对时间位置编码：只使用相对时间RoPE/ALiBi，不使用view-ID PE
            # 如果使用 RoPE，它会在 Block 内部应用
            # 这里不需要额外处理，因为 RoPE 已经是相对位置的
            # 修改5: 确保不使用view-ID相关的PE，保持View Permutation Equivariance
        
        # 修改1: 使用自定义 blocks 或默认 time_blocks
        # 如果在两流架构的L_mid层，使用dual_stream_layer_idx索引
        blocks_to_use = custom_blocks if custom_blocks is not None else self.time_blocks
        
        intermediates = []
        for _ in range(self.aa_block_size):
            # 修改1: 如果使用两流的blocks，使用dual_stream_layer_idx（0-4）
            # 否则使用time_idx（可能超出两流blocks的范围）
            if custom_blocks is not None and dual_stream_layer_idx is not None:
                # 使用dual_stream_layer_idx索引两流blocks（长度只有5）
                if dual_stream_layer_idx >= len(blocks_to_use):
                    raise IndexError(f"dual_stream_layer_idx {dual_stream_layer_idx} >= len(blocks) {len(blocks_to_use)}")
                block = blocks_to_use[dual_stream_layer_idx]
            else:
                # 使用time_idx索引共享的time_blocks（长度24）
                if time_idx >= len(blocks_to_use):
                    raise IndexError(f"time_idx {time_idx} >= len(blocks) {len(blocks_to_use)}")
                block = blocks_to_use[time_idx]
            
            # 如果需要应用掩码偏置（两流架构）
            if self.enable_dual_stream and mask_t_up is not None:
                # TODO: 在 Block 中实现 logits_bias
                pass
            
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
        
        # 输出还原: [B*N, T*P, C] -> [B, N, T, P, C] -> [B, T, N, P, C]
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
