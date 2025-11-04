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

from vggt_t_mask_mlp_fin10.layers import PatchEmbed
from vggt_t_mask_mlp_fin10.layers.block import Block
from vggt_t_mask_mlp_fin10.layers.block import SpatialMaskHead_IMP as SpatialMaskHead_IMP
from vggt_t_mask_mlp_fin10.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt_t_mask_mlp_fin10.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt_t_mask_mlp_fin10.models.mask_utils import *
from .viewmixer import ViewMixer
from .voxelization import VoxelizationModule
from .mask_lifting import MaskLiftingModule

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
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        # Voxelization parameters
        voxel_size=None,
        radius_stage1=2.5,
        radius_stage2=3.5,
        temporal_window=3,
        enable_voxelization=False,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        
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
        
        self.temporal_list1 = [0, 1, 2, 3, 4, 5, 6, 7]  # Stage-1: 8 layers
        self.temporal_list1_mask = [7]
        self.temporal_list2 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Stage-2: 10 layers (8-17), Stage-3: 6 layers (18-23)
        
        # Voxelization parameters for attention routing
        self.voxel_size = voxel_size
        self.radius_stage1 = radius_stage1  # Multiplier for voxel_size
        self.radius_stage2 = radius_stage2  # Multiplier for voxel_size
        self.temporal_window = temporal_window  # Delta for temporal attention
        self.enable_voxelization = enable_voxelization

        self.spatial_mask_head = SpatialMaskHead_IMP(embed_dim)
        
        # ViewMixer (Stage-0): lightweight cross-view attention
        self.viewmixer = ViewMixer(
            embed_dim=embed_dim,
            num_heads=8,  # small number of heads
            use_lora=True,
            lora_rank=16,
            use_geometry_guidance=False,
            dropout=0.0,
        )
        
        # Position/View embeddings
        # Time embedding: E_time[t]
        max_time_steps = 100  # reasonable max for time steps
        self.time_embed = nn.Embedding(max_time_steps, embed_dim)
        
        # View embedding: E_view[v]
        max_views = 32  # reasonable max for views
        self.view_embed = nn.Embedding(max_views, embed_dim)
        
        # Camera parameter embedding: E_cam(K, R, t) - simplified MLP
        # For now, use a learnable MLP that takes view_id as input
        # In practice, this could take actual camera parameters
        self.camera_param_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 4),  # input: view_id (normalized)
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )
        
        # Learnable gate for camera branch mask strength (γ)
        # Initialized to 0 (no masking initially)
        self.camera_mask_gate = nn.Parameter(torch.zeros(1))
        
        # Stage-0: Voxelization module (可选，通过参数控制是否启用)
        self.enable_voxelization = False  # 默认关闭，需要深度/相机参数时再启用
        # 初始化体素化模块（但默认不使用）
        self.voxelization_module = VoxelizationModule(
            embed_dim=embed_dim,
            voxel_size=None,  # 自适应
            voxel_size_mode='auto',
            target_num_voxels=120000,
            use_morton_encoding=False,
            use_sparse3d=False,
        )
        self.mask_lifting = MaskLiftingModule(alpha_init=0.5, tau_init=2.0, learnable=True)

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.embed_dim = embed_dim

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Multi-view token structure:
        # - Camera tokens: [B, V, 1, C] - one per view, shared across time
        # - Register tokens: [B, T, num_register_tokens, C] - one per time step
        max_views_mv = 32
        max_time_steps_mv = 100
        self.camera_token = nn.Parameter(torch.randn(1, max_views_mv, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, max_time_steps_mv, num_register_tokens, embed_dim))
        # 1, V, 1, embed_dim
        # 1, T, num_register_tokens, embed_dim

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize camera tokens with view-specific offsets to ensure diversity
        # Each view should have distinct initial values to enable learning different camera parameters
        # Additionally, use camera prior: if V cameras, assume uniform angular spacing of 360/V degrees
        with torch.no_grad():
            # First initialize with small values (base initialization)
            nn.init.normal_(self.camera_token, std=1e-6)
            
            # Add view-specific offsets based on camera prior knowledge
            # For V cameras, assume uniform angular spacing: each camera is 360/V degrees apart
            # This encodes geometric prior: cameras are arranged in a circle with equal spacing
            # The initialization uses sinusoidal patterns to encode this angular information
            for v in range(max_views_mv):
                # Calculate view angle: 360/V degrees per camera, converted to radians
                # For view v: angle = 2π * v / V (in radians)
                view_angle = 2 * 3.14159 * v / max_views_mv  # Angle in radians
                
                # Create embedding offsets using different frequency components
                # This allows the model to learn different aspects of camera geometry
                embed_indices = torch.arange(embed_dim, dtype=torch.float32)
                
                # Low-frequency components (first 1/4): encode absolute view angle
                # This represents the global camera position in the circular arrangement
                num_low = embed_dim // 4
                low_freq_offset = 0.005 * torch.sin(embed_indices[:num_low] * view_angle)
                
                # Medium-frequency components (next 1/4): encode relative view pattern
                # This helps distinguish between adjacent cameras
                num_mid = embed_dim // 4
                mid_freq_offset = 0.003 * torch.sin(embed_indices[num_low:num_low+num_mid] * (2 * 3.14159 * v / max_views_mv))
                
                # High-frequency components (next 1/4): add view-specific signature
                # This provides unique identifier for each view
                num_high = embed_dim // 4
                high_freq_offset = 0.002 * torch.sin(embed_indices[num_low+num_mid:num_low+num_mid+num_high] * (3.14159 * v))
                
                # Remaining dimensions (last 1/4): encode complementary angle information
                # Uses cosine for orthogonal information
                num_remaining = embed_dim - (num_low + num_mid + num_high)
                remaining_offset = 0.002 * torch.cos(embed_indices[num_low+num_mid+num_high:] * view_angle)
                
                # Concatenate all frequency components
                view_offset = torch.cat([low_freq_offset, mid_freq_offset, high_freq_offset, remaining_offset], dim=0)
                
                # Ensure exact length (defensive programming)
                view_offset = view_offset[:embed_dim]
                if len(view_offset) < embed_dim:
                    padding = torch.zeros(embed_dim - len(view_offset), dtype=torch.float32)
                    view_offset = torch.cat([view_offset, padding], dim=0)
                
                # Reshape to [1, 1, embed_dim] to match camera_token[:, v, :, :] shape
                # camera_token shape: [1, max_views_mv, 1, embed_dim]
                # camera_token[:, v, :, :] shape: [1, 1, embed_dim]
                view_offset = view_offset.view(1, 1, embed_dim)
                
                # Add offset to camera token for this view
                # Note: This is during initialization (with torch.no_grad()), so inplace is safe
                # But we'll use non-inplace for consistency
                self.camera_token.data[:, v, :, :] = self.camera_token.data[:, v, :, :] + view_offset
        
        # Initialize register tokens with small values
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

    def forward(
        self, 
        images: torch.Tensor, 
        temporal_features: torch.Tensor = None,
        depth: Optional[torch.Tensor] = None,  # [B, T, V, H, W] 深度图（可选，用于体素化）
        intrinsics: Optional[torch.Tensor] = None,  # [B, T, V, 3, 3] 相机内参（可选）
        extrinsics: Optional[torch.Tensor] = None,  # [B, T, V, 3, 4] 相机外参（可选）
        confidence: Optional[torch.Tensor] = None,  # [B, T, V, H, W] 置信度（可选）
        pixel_mask: Optional[torch.Tensor] = None,  # [B, T, V, H, W] SegAnyMo像素掩码（可选）
        use_voxelization: Optional[bool] = None,  # 是否启用体素化（覆盖默认设置）
    ) -> Tuple[List[torch.Tensor], int, torch.Tensor, Optional[Dict]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, T, V, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size
                T: time steps (multi-view mode) or S: sequence length (legacy mode)
                V: number of views (multi-view mode only)
                3: RGB channels
                H: height, W: width
            depth: Optional depth maps [B, T, V, H, W] for voxelization
            intrinsics: Optional camera intrinsics [B, T, V, 3, 3] for voxelization
            extrinsics: Optional camera extrinsics [B, T, V, 3, 4] for voxelization
            confidence: Optional confidence maps [B, T, V, H, W] for voxelization
            pixel_mask: Optional SegAnyMo masks [B, T, V, H, W] for mask lifting
            use_voxelization: Override enable_voxelization setting
        Returns:
            (list[torch.Tensor], int, torch.Tensor, Optional[Dict]):
            The list of outputs from the attention blocks,
            the patch_start_idx indicating where patch tokens begin,
            mask_logits [B*S, 1, H, W] or None if mask is not generated,
            voxel_data dict containing voxel information if voxelization is used
        """
        # Detect input format: [B, T, V, C, H, W] (multi-view) or [B, S, C, H, W] (legacy)
        is_multi_view = len(images.shape) == 6
        if is_multi_view:
            B, T, V, C_in, H, W = images.shape
            S = T * V  # Total sequence length for legacy compatibility
            # Reshape to [B, T*V, C, H, W] for unified processing
            images_flat = images.reshape(B, T * V, C_in, H, W)
        else:
            B, S, C_in, H, W = images.shape
            T = S
            V = 1
            images_flat = images
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # Normalize images and reshape for patch embed
        images_flat = (images_flat - self._resnet_mean) / self._resnet_std
        # Reshape to [B*S, C, H, W] for patch embedding
        images_for_embed = images_flat.reshape(B * S, C_in, H, W)

        Visual = False
        if Visual:
            _image_ = normalize_feature_batch(images[0].permute(1,2,0))
            _image_ = Image.fromarray((_image_.cpu().view(H,W,3).clamp(0, 1) * 255).byte().numpy()).resize((400,400), resample=Image.BILINEAR)  
            _image_.save("image-first.png")

            _image_ = normalize_feature_batch(images[S-1].permute(1,2,0))
            _image_ = Image.fromarray((_image_.cpu().view(H,W,3).clamp(0, 1) * 255).byte().numpy()).resize((400,400), resample=Image.BILINEAR)  
            _image_.save("image-last.png")

        patch_tokens = self.patch_embed(images_for_embed) #images[B*S, 3, H, W]
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        _, P, C = patch_tokens.shape  # [B*S, P, C]
        
        # ========== Stage-0: Voxelization (if enabled and depth/camera provided) ==========
        voxel_data = None
        use_voxel = (use_voxelization is not None and use_voxelization) or \
                    (use_voxelization is None and self.enable_voxelization)
        
        if is_multi_view and use_voxel and depth is not None and intrinsics is not None and extrinsics is not None:
            # Reshape to [B, T, V, P=72, C=1024] for voxelization
            patch_tokens_mv = patch_tokens.reshape(B, T, V, P, C)
            
            # Apply ViewMixer first (cross-view attention within each time step)
            patch_tokens_mv = self.viewmixer(patch_tokens_mv)   #都是[1, 6, 4, 729, 1024]
            
            # 检查depth、intrinsics、extrinsics的形状，处理时间维度
            # 如果它们的时间维度是1，需要扩展或重复使用
            depth_time_dim = depth.shape[1] if len(depth.shape) == 5 else 0
            intrinsics_time_dim = intrinsics.shape[1] if len(intrinsics.shape) == 5 else 0
            extrinsics_time_dim = extrinsics.shape[1] if len(extrinsics.shape) == 5 else 0
            
            # 如果时间维度是1，扩展为T
            if depth_time_dim == 1 and T > 1:
                depth = depth.expand(-1, T, -1, -1, -1)  # [B, 1, V, H, W] -> [B, T, V, H, W]
            elif len(depth.shape) == 4:
                # depth可能是 [B, V, H, W]，需要添加时间维度
                depth = depth.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, V, H, W] -> [B, T, V, H, W]
            
            if intrinsics_time_dim == 1 and T > 1:
                intrinsics = intrinsics.expand(-1, T, -1, -1, -1)  # [B, 1, V, 3, 3] -> [B, T, V, 3, 3]
            elif len(intrinsics.shape) == 4:
                # intrinsics可能是 [B, V, 3, 3]，需要添加时间维度
                intrinsics = intrinsics.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, V, 3, 3] -> [B, T, V, 3, 3]
            
            if extrinsics_time_dim == 1 and T > 1:
                extrinsics = extrinsics.expand(-1, T, -1, -1, -1)  # [B, 1, V, 3, 4] -> [B, T, V, 3, 4]
            elif len(extrinsics.shape) == 4:
                # extrinsics可能是 [B, V, 3, 4]，需要添加时间维度
                extrinsics = extrinsics.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, V, 3, 4] -> [B, T, V, 3, 4]
            
            # 处理confidence和pixel_mask
            if confidence is not None:
                confidence_time_dim = confidence.shape[1] if len(confidence.shape) == 5 else 0
                if confidence_time_dim == 1 and T > 1:
                    confidence = confidence.expand(-1, T, -1, -1, -1)  # [B, 1, V, H, W] -> [B, T, V, H, W]
                elif len(confidence.shape) == 4:
                    confidence = confidence.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, V, H, W] -> [B, T, V, H, W]
            
            if pixel_mask is not None:
                pixel_mask_time_dim = pixel_mask.shape[1] if len(pixel_mask.shape) == 5 else 0
                if pixel_mask_time_dim == 1 and T > 1:
                    pixel_mask = pixel_mask.expand(-1, T, -1, -1, -1)  # [B, 1, V, H, W] -> [B, T, V, H, W]
                elif len(pixel_mask.shape) == 4:
                    pixel_mask = pixel_mask.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, V, H, W] -> [B, T, V, H, W]
            
            # 对每个时间步t进行体素化
            voxel_tokens_list = []  # T个 [B, N_t, C]
            voxel_xyz_list = []  # T个 [B, N_t, 3]
            voxel_ids_list = []  # T个 [B, N_t]
            voxel_mask_list = []  # T个 [B, N_t] (如果提供pixel_mask)
            
            for t in range(T):
                # 获取时间步t的所有视角
                patch_tokens_t = patch_tokens_mv[:, t, :, :, :]  # [B, V, P, C]
                depth_t = depth[:, t, :, :, :]  # [B, V, H, W]
                intrinsics_t = intrinsics[:, t, :, :, :]  # [B, V, 3, 3]
                extrinsics_t = extrinsics[:, t, :, :, :]  # [B, V, 3, 4]
                
                confidence_t = None
                if confidence is not None:
                    confidence_t = confidence[:, t, :, :, :]  # [B, V, H, W]
                
                pixel_mask_t = None
                if pixel_mask is not None:
                    pixel_mask_t = pixel_mask[:, t, :, :, :]  # [B, V, H, W]
                
                # 体素化：将V个视角的patch tokens体素化到统一的世界网格
                voxel_tokens_batch, voxel_xyz_batch, voxel_ids_batch, voxel_mask_batch = \
                    self.voxelization_module(
                        patch_tokens_t,  # [B, V, P, C]
                        depth_t,  # [B, V, H, W]
                        intrinsics_t,  # [B, V, 3, 3]
                        extrinsics_t,  # [B, V, 3, 4]
                        confidence_t,  # [B, V, H, W] (可选)
                        pixel_mask_t,  # [B, V, H, W] (可选)
                    )
                
                # voxel_tokens_batch 是list of [N_t, C]，需要stack并pad到统一长度
                # 简化：先保存list，后续处理
                voxel_tokens_list.append(voxel_tokens_batch)  # list of [N_t, C] per batch
                voxel_xyz_list.append(voxel_xyz_batch)  # list of [N_t, 3] per batch
                voxel_ids_list.append(voxel_ids_batch)  # list of [N_t] per batch
                if voxel_mask_batch is not None:
                    voxel_mask_list.append(voxel_mask_batch)
            
            # 将体素tokens连接成序列，替换原来的patch tokens
            # 需要处理不同时间步体素数不同的问题：padding或使用变长序列
            # 简化：先找到最大体素数，然后pad
            max_num_voxels = 0
            for batch_tokens_list in voxel_tokens_list:
                for tokens_b in batch_tokens_list:
                    if tokens_b.shape[0] > max_num_voxels:
                        max_num_voxels = tokens_b.shape[0]
            
            # 如果体素化成功，使用体素tokens；否则继续使用patch tokens
            if max_num_voxels > 0:
                # 构建体素tokens序列 [B, T*max_num_voxels, C]
                # 注意：这里每个时间步的体素数可能不同，需要padding
                voxel_tokens_padded_list = []
                for t in range(T):
                    batch_voxel_tokens = []
                    for b in range(B):
                        if t < len(voxel_tokens_list) and b < len(voxel_tokens_list[t]):
                            tokens_b = voxel_tokens_list[t][b]  # [N_t, C]
                            if tokens_b.shape[0] < max_num_voxels:
                                # Padding
                                padding = torch.zeros(
                                    max_num_voxels - tokens_b.shape[0], C,
                                    device=tokens_b.device, dtype=tokens_b.dtype
                                )
                                tokens_b_padded = torch.cat([tokens_b, padding], dim=0)  # [max_num_voxels, C]
                            else:
                                tokens_b_padded = tokens_b[:max_num_voxels]  # 截断
                        else:
                            tokens_b_padded = torch.zeros(max_num_voxels, C, device=images.device)
                        batch_voxel_tokens.append(tokens_b_padded)
                    voxel_tokens_padded_list.append(torch.stack(batch_voxel_tokens, dim=0))  # [B, max_num_voxels, C]
                
                # Stack所有时间步: [B, T, max_num_voxels, C]
                voxel_tokens_all = torch.stack(voxel_tokens_padded_list, dim=1)  # [B, T, max_num_voxels, C]
                # Flatten: [B, T*max_num_voxels, C]
                patch_tokens = voxel_tokens_all.reshape(B, T * max_num_voxels, C)
                # 更新P和S
                P = max_num_voxels
                S = T * max_num_voxels  # 注意：这里S的含义改变了
                
                # 保存体素数据用于后续attention路由
                voxel_data = {
                    'voxel_tokens_list': voxel_tokens_list,
                    'voxel_xyz_list': voxel_xyz_list,
                    'voxel_ids_list': voxel_ids_list,
                    'voxel_mask_list': voxel_mask_list if voxel_mask_list else None,
                    'max_num_voxels': max_num_voxels,
                    'use_voxel_tokens': True,
                }
            else:
                # 体素化失败，回退到patch tokens
                patch_tokens = patch_tokens_mv.reshape(B * S, P, C)
                voxel_data = {'use_voxel_tokens': False}
        
        # Multi-view token handling (both voxelized and non-voxelized paths)
        if is_multi_view:
            if voxel_data is None or not voxel_data.get('use_voxel_tokens', False):
                # 非体素化路径：正常处理
                # Multi-view token handling
                # Camera tokens: [1, V, 1, C] -> [B, V, 1, C] (shared across time)
                camera_token_mv = self.camera_token[:, :V, :, :].expand(B, -1, -1, -1)  # [B, V, 1, C]
                # Register tokens: [1, T, num_register_tokens, C] -> [B, T, num_register_tokens, C]
                register_token_mv = self.register_token[:, :T, :, :].expand(B, -1, -1, -1)  # [B, T, num_register_tokens, C]
                
                # Expand and interleave: for each (t,v) pair, use corresponding tokens
                # camera_token: [B, V, 1, C] -> [B, T, V, 1, C] -> [B*T*V, 1, C]
                camera_token = camera_token_mv.unsqueeze(1).expand(-1, T, -1, -1, -1).contiguous().reshape(B * T * V, 1, C)
                # register_token: [B, T, num_register_tokens, C] -> [B, T, 1, V, num_register_tokens, C] -> [B*T*V, num_register_tokens, C]
                register_token = register_token_mv.unsqueeze(2).expand(-1, -1, V, -1, -1).contiguous().reshape(B * T * V, self.register_token.shape[2], C)
            else:
                # 体素化路径：为每个时间步的体素tokens添加camera/register tokens
                # 简化：每个时间步使用相同的camera/register tokens
                camera_token_mv = self.camera_token[:, :V, :, :].expand(B, -1, -1, -1)  # [B, V, 1, C]
                register_token_mv = self.register_token[:, :T, :, :].expand(B, -1, -1, -1)  # [B, T, num_register_tokens, C]
                
                # 为每个时间步重复camera/register tokens
                # Camera: [B, V, 1, C] -> [B, T, 1, C] (使用第一个视角的camera token)
                camera_token_per_t = camera_token_mv[:, 0:1, :, :].unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, T, 1, 1, C]
                camera_token = camera_token_per_t.reshape(B * T, 1, C)  # [B*T, 1, C]
                
                # Register: [B, T, num_register_tokens, C] -> [B*T, num_register_tokens, C]
                register_token = register_token_mv.reshape(B * T, self.register_token.shape[2], C)
        else:
            # Legacy mode: use original slice_expand_and_flatten
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        
        # Concatenate special tokens with patch tokens
        # 注意：在体素化路径下，patch_tokens已经是[B*T*max_num_voxels, C]，需要reshape
        if is_multi_view and voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
            # 体素化路径：patch_tokens是[B, T*max_num_voxels, C]，需要reshape为[B*T*max_num_voxels, max_num_voxels, C]?
            # 实际上应该是[B, T*max_num_voxels, C]，但我们需要将其reshape为[B*T, max_num_voxels, C]
            # 然后为每个时间步添加camera/register tokens
            # 简化：直接reshape并拼接
            patch_tokens_reshaped = patch_tokens.reshape(B * T, P, C)  # [B*T, P, C]
            tokens = torch.cat([camera_token, register_token, patch_tokens_reshaped], dim=1)  # [B*T, 1+num_register_tokens+P, C]
            S_voxel = T  # 用于体素化的序列长度
        else:
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)  # [B*S, 1+num_register_tokens+P, C]
            S_voxel = S
        
        pos = None
        if self.rope is not None:
            # 对于体素化路径，位置编码需要特殊处理（体素是3D坐标，不是2D patch）
            if is_multi_view and voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
                # 体素化路径：不使用2D位置编码（体素已经有3D位置编码）
                pos = None
            else:
                pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images_flat.device)
                if self.patch_start_idx > 0:
                    pos = pos + 1
                    pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images_flat.device).to(pos.dtype)
                    pos = torch.cat([pos_special, pos], dim=1)
                # RoPE实现使用embedding索引，要求整型
                if pos is not None and pos.dtype != torch.long:
                    pos = pos.long()
        
        # Add time and view embeddings (for multi-view mode)
        if is_multi_view:
            if voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
                # 体素化路径：每个时间步使用独立的时间embedding
                # Time embedding: [B*T, 1, C]
                time_ids = torch.arange(T, device=images_flat.device).unsqueeze(0).expand(B, -1).reshape(B * T)  # [B*T]
                time_emb = self.time_embed(time_ids).unsqueeze(1)  # [B*T, 1, C]
                
                # View embedding: 体素化后不再需要view embedding（多视角已融合）
                view_emb = torch.zeros(B * T, 1, C, device=images_flat.device)
                
                # Camera embedding: 使用第一个视角的camera token
                cam_emb = torch.zeros(B * T, 1, C, device=images_flat.device)
                
                # Add embeddings to tokens
                camera_tokens_with_emb = tokens[:, 0:1, :] + cam_emb + view_emb
                patch_tokens_slice = tokens[:, self.patch_start_idx:, :]
                patch_tokens_with_emb = patch_tokens_slice + time_emb  # 只加时间embedding
                tokens = torch.cat([camera_tokens_with_emb, tokens[:, 1:self.patch_start_idx, :], patch_tokens_with_emb], dim=1)
            else:
                # 非体素化路径：正常处理
                # Time embedding: E_time[t] for each (t,v) pair
                time_ids = torch.arange(T, device=images_flat.device).repeat_interleave(V).unsqueeze(0).expand(B, -1)  # [B, T*V]
                time_emb_raw = self.time_embed(time_ids)  # [B, T*V, C]
                # Reshape: [B, T*V, C] -> [B*S, 1, C] where S = T*V
                time_emb = time_emb_raw.contiguous().reshape(B * S, 1, C)  # [B*S, 1, C]
                
                # View embedding: E_view[v] for each (t,v) pair
                view_ids = torch.arange(V, device=images_flat.device).repeat(T).unsqueeze(0).expand(B, -1)  # [B, T*V]
                view_emb_raw = self.view_embed(view_ids)  # [B, T*V, C]
                # Reshape: [B, T*V, C] -> [B*S, 1, C]
                view_emb = view_emb_raw.contiguous().reshape(B * S, 1, C)  # [B*S, 1, C]
                
                # Camera parameter embedding (simplified: use view_id)
                view_ids_norm = view_ids.float().unsqueeze(-1) / max(V - 1, 1)  # [B, T*V, 1]
                cam_emb_raw = self.camera_param_embed(view_ids_norm)  # [B, T*V, C]
                # Reshape: [B, T*V, C] -> [B*S, 1, C]
                cam_emb = cam_emb_raw.contiguous().reshape(B * S, 1, C)  # [B*S, 1, C]
                
                # Add embeddings to tokens (additive injection)
                camera_tokens_with_emb = tokens[:, 0:1, :] + cam_emb + view_emb
                patch_tokens_slice = tokens[:, self.patch_start_idx:, :]  # [B*S, P-patch_start_idx, C]
                patch_tokens_with_emb = patch_tokens_slice + view_emb + time_emb
                tokens = torch.cat([camera_tokens_with_emb, tokens[:, 1:self.patch_start_idx, :], patch_tokens_with_emb], dim=1)
        # update P because we added special tokens
        # 统一管理维度：确保B, S, P, C始终与实际tokens形状一致
        B_curr, P_curr, C_curr = tokens.shape
        # 计算实际的S（序列长度）
        if B_curr == B:
            S_curr = tokens.shape[0] // B  # 实际的序列长度
        else:
            # tokens可能是[B*S, P, C]格式，需要推断
            S_curr = tokens.shape[0] // B if tokens.shape[0] % B == 0 else 1
        
        # 更新P和C
        P = P_curr
        C = C_curr
        
        # 根据体素化路径更新S
        if is_multi_view and voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
            # 体素化路径：S应该是T（每个时间步的体素序列）
            S_actual = T
            # 但tokens的实际形状是[B*T, P, C]，所以S_curr = T
            if S_curr != S_actual:
                # 如果形状不匹配，尝试reshape
                if tokens.numel() == B * S_actual * P * C:
                    tokens = tokens.reshape(B * S_actual, P, C)
                    S_curr = S_actual
                else:
                    # 使用实际形状
                    S_actual = S_curr
        else:
            # 非体素化路径：S应该是T*V
            S_actual = S if S > 0 else S_curr
        
        # 统一使用S_curr作为当前的序列长度
        S = S_curr
        
        frame_idx = 0
        global_idx = 0
        output_list = []
        mask_logits = None  # Store mask logits for supervision loss (generated at layer 7)
        
        # Determine stage based on block number
        for num_block in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # Determine routing strategy based on stage
                    if num_block in self.temporal_list1:
                        # Stage-1: Frame attention - 体素化路径下使用「同ID时间窗口」路由
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S_voxel if (voxel_data and voxel_data.get('use_voxel_tokens', False)) else S, 
                            P, C, frame_idx, pos=pos, 
                            is_multi_view=is_multi_view, T=T, V=V, stage="stage1",
                            voxel_data=voxel_data, num_block=num_block)
                    elif num_block in list(range(8, 18)):  # Stage-2: layers 8-17
                        # Stage-2: Frame attention - 同ID时间窗口 + 掩码门控
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S_voxel if (voxel_data and voxel_data.get('use_voxel_tokens', False)) else S, 
                            P, C, frame_idx, pos=pos,
                            is_multi_view=is_multi_view, T=T, V=V, stage="stage2",
                            voxel_data=voxel_data, num_block=num_block)
                    elif num_block in list(range(18, 24)):  # Stage-3: layers 18-23
                        # Stage-3: 交替路由：同ID时间注意力 / 同t空间邻域注意力
                        use_temporal = (num_block - 18) < 3  # first 3 layers: temporal, last 3: spatial
                        stage_strategy = "stage2" if use_temporal else "stage1"
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S_voxel if (voxel_data and voxel_data.get('use_voxel_tokens', False)) else S, 
                            P, C, frame_idx, pos=pos,
                            is_multi_view=is_multi_view, T=T, V=V, stage=stage_strategy,
                            voxel_data=voxel_data, num_block=num_block)
                    else:
                        # Fallback
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos,
                            is_multi_view=is_multi_view, T=T, V=V, stage="legacy")
                    
                    if num_block in self.temporal_list1_mask:
                        # Extract mask logits from spatial_mask_head for supervision loss
                        # 统一维度管理：从实际tokens形状推断S和P
                        B_actual, P_actual, C_actual = tokens.shape
                        S_actual = B_actual // B if B_actual % B == 0 else 1
                        
                        # 更新S和P以匹配实际tokens形状
                        if S_actual != S or P_actual != P:
                            # 如果维度不匹配，更新S和P
                            S = S_actual
                            P = P_actual
                            
                            # 尝试reshape tokens为期望的形状 [B, S, P, C]
                            if tokens.numel() == B * S * P * C:
                                tokens = tokens.reshape(B * S, P, C)
                            else:
                                # 如果大小不匹配，使用实际形状，但确保是[B*S, P, C]格式
                                if tokens.shape != (B * S, P, C):
                                    # 强制reshape为[B*S, P, C]，其中S和P是实际值
                                    tokens = tokens.reshape(B * S, P, C)
                        
                        # 对于mask计算，需要区分体素化路径和非体素化路径
                        if voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
                            # 体素化路径：mask计算不适用（体素不是patch网格）
                            # 跳过mask计算或使用简化版本
                            tokens_for_mask = tokens.detach().clone().reshape(B, S, P, C)
                            # 对于体素化路径，mask_h和mask_w可能不适用
                            # 使用默认值或从voxel_data推断
                            max_num_voxels = voxel_data.get('max_num_voxels', P)
                            # 假设体素网格是近似的正方形
                            mask_h = int((max_num_voxels) ** 0.5)
                            mask_w = mask_h
                            # 确保mask_h * mask_w >= P (patch_start_idx之后的部分)
                            if mask_h * mask_w < P - self.patch_start_idx:
                                mask_w = (P - self.patch_start_idx + mask_h - 1) // mask_h
                        else:
                            # 非体素化路径：正常计算mask
                            tokens_for_mask = tokens.detach().clone().reshape(B, S, P, C)
                            mask_h = H // self.patch_size
                            mask_w = W // self.patch_size
                        
                        # Manually compute mask logits (mimicking SpatialMaskHead_IMP.forward)
                        # This is needed because SpatialMaskHead_IMP doesn't return m_logit
                        # 确保tokens_for_mask的形状正确
                        if tokens_for_mask.shape != (B, S, P, C):
                            # 如果reshape失败，尝试从实际形状推断
                            B_mask, S_mask, P_mask, C_mask = tokens_for_mask.shape
                            if B_mask * S_mask * P_mask * C_mask != tokens_for_mask.numel():
                                # 强制reshape为正确的形状
                                tokens_for_mask = tokens_for_mask.reshape(B, S, P, C)
                        
                        # 提取patch tokens（跳过special tokens）
                        patch_tokens_mask = tokens_for_mask.view(B * S, P, C)[:, self.patch_start_idx:, :]  # (B*S, P-patch_start_idx, C)
                        patch_len = patch_tokens_mask.shape[1]  # 实际的patch token数
                        
                        # 关键修复：确保mask_h * mask_w == patch_len（spatial_mask_head的要求）
                        # 从实际的patch_len计算mask_h和mask_w
                        if patch_len != mask_h * mask_w:
                            # 重新计算mask_h和mask_w以匹配patch_len
                            # 尝试找到最接近的正方形
                            mask_h = int(patch_len ** 0.5)
                            mask_w = (patch_len + mask_h - 1) // mask_h
                            # 如果mask_h * mask_w > patch_len，需要padding
                            if mask_h * mask_w > patch_len:
                                padding_size = mask_h * mask_w - patch_len
                                padding = torch.zeros(B * S, padding_size, C, device=patch_tokens_mask.device, dtype=patch_tokens_mask.dtype)
                                patch_tokens_mask = torch.cat([patch_tokens_mask, padding], dim=1)
                            elif mask_h * mask_w < patch_len:
                                # 如果mask_h * mask_w < patch_len，需要截断
                                patch_tokens_mask = patch_tokens_mask[:, :mask_h * mask_w, :]
                        
                        # 验证：确保mask_h * mask_w == patch_tokens_mask.shape[1]
                        assert mask_h * mask_w == patch_tokens_mask.shape[1], \
                            f"mask_h * mask_w ({mask_h * mask_w}) != patch_tokens_mask.shape[1] ({patch_tokens_mask.shape[1]})"
                        
                        h0 = patch_tokens_mask.transpose(1, 2).reshape(B * S, C, mask_h, mask_w)
                        m_logit = self.spatial_mask_head.head0(h0)  # (B*S, 1, H, W)
                        
                        # Now call the full forward to get the outputs we need
                        # 关键：确保传入的mask_h和mask_w满足 (P - patch_start_idx) == mask_h * mask_w
                        # spatial_mask_head.forward要求：assert (P - patch_start) == (H * W)
                        # 所以我们需要确保tokens_for_mask的P满足这个条件
                        actual_patch_len = P - self.patch_start_idx
                        
                        # 如果actual_patch_len与mask_h * mask_w不匹配，需要调整
                        # mask_h和mask_w已经在上面根据patch_len计算过了，应该等于patch_len
                        # 但为了确保一致性，我们使用mask_h * mask_w作为目标
                        target_patch_len = mask_h * mask_w
                        
                        if actual_patch_len != target_patch_len:
                            # 需要padding或调整
                            if actual_patch_len < target_patch_len:
                                # 需要padding
                                padding_size = target_patch_len - actual_patch_len
                                padding_tokens = torch.zeros(B, S, padding_size, C, 
                                                           device=tokens_for_mask.device, dtype=tokens_for_mask.dtype)
                                tokens_for_mask = torch.cat([
                                    tokens_for_mask[:, :, :self.patch_start_idx, :],  # special tokens
                                    tokens_for_mask[:, :, self.patch_start_idx:, :],  # patch tokens
                                    padding_tokens  # padding
                                ], dim=2)
                                P = tokens_for_mask.shape[2]
                            else:
                                # actual_patch_len > target_patch_len，需要调整mask_h和mask_w
                                # 或者截断tokens（但截断可能丢失信息，所以调整mask_h和mask_w更好）
                                mask_h = int(actual_patch_len ** 0.5)
                                mask_w = (actual_patch_len + mask_h - 1) // mask_h
                                # 如果mask_h * mask_w > actual_patch_len，需要padding
                                if mask_h * mask_w > actual_patch_len:
                                    padding_size = mask_h * mask_w - actual_patch_len
                                    padding_tokens = torch.zeros(B, S, padding_size, C,
                                                                device=tokens_for_mask.device, dtype=tokens_for_mask.dtype)
                                    tokens_for_mask = torch.cat([
                                        tokens_for_mask[:, :, :self.patch_start_idx, :],
                                        tokens_for_mask[:, :, self.patch_start_idx:, :],
                                        padding_tokens
                                    ], dim=2)
                                    P = tokens_for_mask.shape[2]
                        
                        # 最终验证：确保 (P - patch_start_idx) == mask_h * mask_w
                        final_patch_len = P - self.patch_start_idx
                        if final_patch_len != mask_h * mask_w:
                            # 强制调整mask_h和mask_w以匹配final_patch_len
                            mask_h = int(final_patch_len ** 0.5)
                            mask_w = (final_patch_len + mask_h - 1) // mask_h
                            # 如果mask_h * mask_w != final_patch_len，需要padding
                            if mask_h * mask_w != final_patch_len:
                                padding_size = mask_h * mask_w - final_patch_len
                                padding_tokens = torch.zeros(B, S, padding_size, C,
                                                            device=tokens_for_mask.device, dtype=tokens_for_mask.dtype)
                                tokens_for_mask = torch.cat([
                                    tokens_for_mask[:, :, :self.patch_start_idx, :],
                                    tokens_for_mask[:, :, self.patch_start_idx:, :],
                                    padding_tokens
                                ], dim=2)
                                P = tokens_for_mask.shape[2]
                        
                        # 最终验证：确保条件满足
                        final_check = P - self.patch_start_idx
                        if final_check != mask_h * mask_w:
                            # 最后一次尝试：重新计算mask_h和mask_w
                            mask_h = int(final_check ** 0.5)
                            mask_w = (final_check + mask_h - 1) // mask_h
                            if mask_h * mask_w != final_check:
                                padding_size = mask_h * mask_w - final_check
                                padding_tokens = torch.zeros(B, S, padding_size, C,
                                                            device=tokens_for_mask.device, dtype=tokens_for_mask.dtype)
                                tokens_for_mask = torch.cat([
                                    tokens_for_mask[:, :, :self.patch_start_idx, :],
                                    tokens_for_mask[:, :, self.patch_start_idx:, :],
                                    padding_tokens
                                ], dim=2)
                                P = tokens_for_mask.shape[2]
                        
                        # 最终断言验证
                        assert (P - self.patch_start_idx) == (mask_h * mask_w), \
                            f"Final check failed: P - patch_start_idx ({P - self.patch_start_idx}) != mask_h * mask_w ({mask_h * mask_w})"
                        
                        cached_key_bias_1d, cached_cam_row_mask = self.spatial_mask_head(
                            tokens_for_mask, 
                            self.patch_start_idx, mask_h, mask_w)
                        
                        # Store mask_logits for supervision loss (only store once, from layer 7)
                        if mask_logits is None:
                            mask_logits = m_logit  # [B*S, 1, H, W]
                        
                        cached_value = cached_key_bias_1d  # (B, S*P)
                        cache_mask = cached_cam_row_mask.to(cached_value.dtype)  # (B, S*P)
                        
                        # Apply learnable gate for camera branch (weak masking)
                        if is_multi_view:
                            gamma = torch.sigmoid(self.camera_mask_gate)
                            # For camera tokens: apply weak masking
                            # For register tokens: keep strong masking
                            # We need to identify camera token positions vs register token positions
                            # For simplicity, apply gamma to all mask for now
                            # In practice, this should be more targeted
                            cache_mask_camera = gamma * cache_mask
                            cache_mask_register = cache_mask  # strong masking for register
                        else:
                            cache_mask_camera = cache_mask
                            cache_mask_register = cache_mask
                elif attn_type == "global":
                    if num_block in self.temporal_list1:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S_voxel if (voxel_data and voxel_data.get('use_voxel_tokens', False)) else S, 
                            P, C, global_idx, pos=pos, is_multi_view=is_multi_view, T=T, V=V,
                            voxel_data=voxel_data, num_block=num_block)
                    elif num_block in self.temporal_list2:
                        # Apply mask: weak for camera branch, strong for register branch
                        # For simplicity, apply camera gate to mask
                        if is_multi_view and hasattr(self, 'camera_mask_gate'):
                            gamma = torch.sigmoid(self.camera_mask_gate)
                            cache_mask_applied = gamma * cache_mask  # weak masking
                        else:
                            cache_mask_applied = cache_mask
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S_voxel if (voxel_data and voxel_data.get('use_voxel_tokens', False)) else S, 
                            P, C, global_idx, pos=pos, 
                            attn_mask=cache_mask_applied, attn_value=cached_value,
                            is_multi_view=is_multi_view, T=T, V=V,
                            voxel_data=voxel_data, num_block=num_block)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                
                # 在每个attention块后，更新S和P以匹配实际tokens形状
                # 这确保维度始终保持一致
                B_after, P_after, C_after = tokens.shape
                S_after = B_after // B if B_after % B == 0 else S
                if S_after != S or P_after != P:
                    S = S_after
                    P = P_after
                    # 确保tokens是[B*S, P, C]格式
                    if tokens.shape != (B * S, P, C):
                        if tokens.numel() == B * S * P * C:
                            tokens = tokens.reshape(B * S, P, C)
            
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)    #24层[1, 24, 782, 1024*2] or [1, T*V, P, 2048]
        del concat_inter
        del frame_intermediates
        del global_intermediates
        
        # 确保voxel_data已初始化
        if voxel_data is None:
            voxel_data = {'use_voxel_tokens': False}
        
        return output_list, self.patch_start_idx, mask_logits, voxel_data

    def _match_voxel_ids_simple(
        self, ids_t: torch.Tensor, ids_t_delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        简化版体素ID匹配（用于attention路由）
        
        Args:
            ids_t: 时间步t的体素ID [N_t]
            ids_t_delta: 时间步t+delta的体素ID [N_t+delta]
        Returns:
            matched_indices_t: 时间步t的匹配索引 [M]
            matched_indices_t_delta: 时间步t+delta的匹配索引 [M]
        """
        device = ids_t.device
        if ids_t.numel() == 0 or ids_t_delta.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device)
        
        # 使用排序和搜索
        ids_t_delta_sorted, sort_indices_delta = torch.sort(ids_t_delta)
        ids_t_sorted, sort_indices_t = torch.sort(ids_t)
        search_indices = torch.searchsorted(ids_t_delta_sorted, ids_t_sorted, side='left')
        
        # 确保search_indices在有效范围内（避免索引越界）
        max_idx = ids_t_delta_sorted.shape[0] - 1
        search_indices = torch.clamp(search_indices, 0, max_idx)
        
        # 检查是否匹配
        valid_mask = search_indices < ids_t_delta_sorted.shape[0]
        # 安全地访问索引：使用clamp后的索引
        matched_values = ids_t_delta_sorted[search_indices]
        match_mask = valid_mask & (matched_values == ids_t_sorted)
        
        if not match_mask.any():
            return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device)
        
        # 获取匹配的索引（确保索引在有效范围内）
        matched_sorted_indices_t = torch.where(match_mask)[0]
        matched_sorted_indices_delta = search_indices[match_mask]
        
        # 确保索引不越界
        if matched_sorted_indices_delta.numel() > 0:
            matched_sorted_indices_delta = torch.clamp(matched_sorted_indices_delta, 0, sort_indices_delta.shape[0] - 1)
        if matched_sorted_indices_t.numel() > 0:
            matched_sorted_indices_t = torch.clamp(matched_sorted_indices_t, 0, sort_indices_t.shape[0] - 1)
        
        matched_indices_t = sort_indices_t[matched_sorted_indices_t]
        matched_indices_delta = sort_indices_delta[matched_sorted_indices_delta]
        
        return matched_indices_t, matched_indices_delta
    
    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos=None, attn_mask=None, attn_value=None,
        is_multi_view=False, T=None, V=None, stage="legacy",
        voxel_data=None, num_block=None
    ):
        """
        Process frame attention blocks with multi-view routing support.
        
        Args:
            tokens: [B*S, P, C] token tensor
            stage: "stage1" (multi-view routing), "stage2" (temporal routing), "legacy" (original)
        """
        # 统一维度管理：从实际tokens形状推断S和P
        B_actual, N_actual, C_actual = tokens.shape
        if B_actual != B:
            # tokens可能是[B*S, P, C]格式，需要推断
            if B_actual % B == 0:
                S_inferred = B_actual // B
                P_inferred = N_actual
            else:
                # 无法推断，使用传入的参数
                S_inferred = S
                P_inferred = P
        else:
            # tokens是[B, N, C]格式
            S_inferred = 1
            P_inferred = N_actual
        
        # 检查是否可以reshape为期望的形状
        expected_size = B * S * P * C
        actual_size = tokens.numel()
        
        if actual_size != expected_size:
            # 大小不匹配，需要重新推断S和P
            if actual_size == B * S_inferred * P_inferred * C:
                # 使用推断的值
                S = S_inferred
                P = P_inferred
            else:
                # 尝试从传入的参数推断
                if T is not None and V is not None:
                    # 非体素化路径：S = T*V
                    if voxel_data is None or not voxel_data.get('use_voxel_tokens', False):
                        S_expected = T * V
                        P_expected = actual_size // (B * S_expected * C)
                        if B * S_expected * P_expected * C == actual_size:
                            S = S_expected
                            P = P_expected
                        else:
                            # 使用实际值
                            S = S_inferred
                            P = P_inferred
                    else:
                        # 体素化路径：S = T
                        S_expected = T
                        P_expected = actual_size // (B * S_expected * C)
                        if B * S_expected * P_expected * C == actual_size:
                            S = S_expected
                            P = P_expected
                        else:
                            S = S_inferred
                            P = P_inferred
                else:
                    S = S_inferred
                    P = P_inferred
        
        # 确保tokens是[B*S, P, C]格式
        if tokens.shape != (B * S, P, C):
            if tokens.numel() == B * S * P * C:
                tokens = tokens.reshape(B * S, P, C)
            else:
                # 如果大小不匹配，使用实际形状
                tokens = tokens.reshape(B_actual, N_actual, C_actual)
                # 更新S和P
                S = B_actual // B if B_actual % B == 0 else 1
                P = N_actual
        
        # 类似地处理pos
        if pos is not None:
            if pos.shape != (B * S, P, 2):
                if pos.numel() == B * S * P * 2:
                    pos = pos.reshape(B * S, P, 2)
                else:
                    # 使用实际形状
                    B_pos, N_pos, D_pos = pos.shape
                    if B_pos == B * S and D_pos == 2:
                        P_pos = N_pos
                        if P_pos != P:
                            # 如果P不匹配，可能需要调整
                            pass
                    pos = pos.reshape(B_pos, N_pos, D_pos)
        
        # 判断是否使用体素化路由
        use_voxel_routing = (voxel_data is not None) and voxel_data.get('use_voxel_tokens', False) and (num_block is not None)
        
        # 构建同体素ID时间窗口attention mask（如果需要）
        temporal_attn_mask = None
        
        if use_voxel_routing and voxel_data is not None:
            voxel_ids_list = voxel_data.get('voxel_ids_list', None)
            if voxel_ids_list is not None:
                max_num_voxels = voxel_data.get('max_num_voxels', P)
                delta = self.temporal_window if hasattr(self, 'temporal_window') else 3
                
                # 创建时间窗口attention mask
                # Stage-1/2: 同体素ID的时间窗口
                if stage in ["stage1", "stage2"] and num_block in self.temporal_list1 + list(range(8, 18)):
                    temporal_attn_mask = torch.zeros(B, S * max_num_voxels, S * max_num_voxels, 
                                                   device=tokens.device, dtype=torch.bool)
                    
                    for b in range(B):
                        for t in range(S):
                            if t < len(voxel_ids_list):
                                ids_list_t = voxel_ids_list[t]
                                if b < len(ids_list_t) and ids_list_t[b] is not None:
                                    ids_t = ids_list_t[b]  # [N_t]
                                    N_t = ids_t.shape[0]
                                    token_start_t = t * max_num_voxels
                                    
                                    # 在时间窗口内查找同ID的体素
                                    t_start = max(0, t - delta)
                                    t_end = min(S, t + delta + 1)
                                    
                                    for t_other in range(t_start, t_end):
                                        if t_other == t:
                                            # 自身：允许所有体素（包括self-attention）
                                            temporal_attn_mask[b, token_start_t:token_start_t+max_num_voxels,
                                                             token_start_t:token_start_t+max_num_voxels] = True
                                        elif t_other < len(voxel_ids_list):
                                            ids_list_other = voxel_ids_list[t_other]
                                            if b < len(ids_list_other) and ids_list_other[b] is not None:
                                                ids_other = ids_list_other[b]  # [N_other]
                                                token_start_other = t_other * max_num_voxels
                                                
                                                # 匹配同ID的体素
                                                matched_t, matched_other = self._match_voxel_ids_simple(ids_t, ids_other)
                                                
                                                if matched_t.numel() > 0 and matched_other.numel() > 0:
                                                    # 确保索引在有效范围内
                                                    matched_t_clamped = torch.clamp(matched_t, 0, N_t - 1)
                                                    matched_other_clamped = torch.clamp(matched_other, 0, ids_other.shape[0] - 1)
                                                    
                                                    # 创建mask：允许匹配的体素之间的attention（使用向量化操作）
                                                    valid_mask = (matched_t_clamped < N_t) & (matched_other_clamped < ids_other.shape[0])
                                                    if valid_mask.any():
                                                        idx_t_valid = matched_t_clamped[valid_mask]
                                                        idx_other_valid = matched_other_clamped[valid_mask]
                                                        
                                                        # 向量化设置mask（避免Python循环）
                                                        temporal_attn_mask[b, token_start_t + idx_t_valid, 
                                                                         token_start_other + idx_other_valid] = True
                                                        temporal_attn_mask[b, token_start_other + idx_other_valid,
                                                                         token_start_t + idx_t_valid] = True
                
                # Stage-3: 交替使用同ID时间窗口和同t空间邻域
                elif stage in ["stage1", "stage2"] and num_block in list(range(18, 24)):
                    # 判断是时间路由还是空间路由
                    use_temporal = (num_block - 18) < 3  # 前3层用时间路由
                    
                    if use_temporal:
                        # 同ID时间窗口（类似Stage-1/2）
                        temporal_attn_mask = torch.zeros(B, S * max_num_voxels, S * max_num_voxels, 
                                                       device=tokens.device, dtype=torch.bool)
                        delta = self.temporal_window if hasattr(self, 'temporal_window') else 3
                        
                        for b in range(B):
                            for t in range(S):
                                if t < len(voxel_ids_list):
                                    ids_list_t = voxel_ids_list[t]
                                    if b < len(ids_list_t) and ids_list_t[b] is not None:
                                        ids_t = ids_list_t[b]
                                        N_t = ids_t.shape[0]
                                        token_start_t = t * max_num_voxels
                                        
                                        t_start = max(0, t - delta)
                                        t_end = min(S, t + delta + 1)
                                        
                                        for t_other in range(t_start, t_end):
                                            if t_other == t:
                                                temporal_attn_mask[b, token_start_t:token_start_t+max_num_voxels,
                                                                 token_start_t:token_start_t+max_num_voxels] = True
                                            elif t_other < len(voxel_ids_list):
                                                ids_list_other = voxel_ids_list[t_other]
                                                if b < len(ids_list_other) and ids_list_other[b] is not None:
                                                    ids_other = ids_list_other[b]
                                                    token_start_other = t_other * max_num_voxels
                                                    matched_t, matched_other = self._match_voxel_ids_simple(ids_t, ids_other)
                                                    
                                                    if matched_t.numel() > 0:
                                                        for idx_t, idx_other in zip(matched_t, matched_other):
                                                            if idx_t < N_t and idx_other < ids_other.shape[0]:
                                                                temporal_attn_mask[b, token_start_t + idx_t, 
                                                                                 token_start_other + idx_other] = True
                                                                temporal_attn_mask[b, token_start_other + idx_other,
                                                                                 token_start_t + idx_t] = True
                    else:
                        # 同t空间邻域（使用build_radius_graph）
                        temporal_attn_mask = torch.zeros(B, S * max_num_voxels, S * max_num_voxels, 
                                                       device=tokens.device, dtype=torch.bool)
                        voxel_xyz_list = voxel_data.get('voxel_xyz_list', None)
                        if voxel_xyz_list is not None and self.voxel_size is not None:
                            radius = self.radius_stage2 * self.voxel_size
                            
                            for b in range(B):
                                for t in range(S):
                                    if t < len(voxel_xyz_list):
                                        xyz_list_t = voxel_xyz_list[t]
                                        if b < len(xyz_list_t) and xyz_list_t[b] is not None:
                                            voxel_xyz_tb = xyz_list_t[b]
                                            if voxel_xyz_tb.shape[0] > 0:
                                                edge_index, _ = self.build_radius_graph(voxel_xyz_tb, radius)
                                                N_t = voxel_xyz_tb.shape[0]
                                                token_start = t * max_num_voxels
                                                
                                                mask_tb = torch.eye(N_t, device=tokens.device, dtype=torch.bool)
                                                if edge_index.shape[1] > 0:
                                                    src, dst = edge_index[0], edge_index[1]
                                                    mask_tb[src, dst] = True
                                                
                                                full_mask_tb = torch.zeros(max_num_voxels, max_num_voxels, 
                                                                          device=tokens.device, dtype=torch.bool)
                                                full_mask_tb[:N_t, :N_t] = mask_tb
                                                temporal_attn_mask[b, token_start:token_start+max_num_voxels, 
                                                                 token_start:token_start+max_num_voxels] = full_mask_tb
        
        # 合并temporal_attn_mask和attn_mask
        final_attn_mask = attn_mask
        if temporal_attn_mask is not None:
            if attn_mask is not None:
                # 合并：取交集
                final_attn_mask = temporal_attn_mask & attn_mask
            else:
                final_attn_mask = temporal_attn_mask
        
        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            block = self.frame_blocks[frame_idx]
            
            # Apply routing strategy for multi-view mode
            if is_multi_view and stage in ["stage1", "stage2"]:
                # 检查是否使用体素化路由
                use_voxel_routing = (voxel_data is not None) and voxel_data.get('use_voxel_tokens', False)
                
                if use_voxel_routing:
                    # 体素化路径：S = T*max_num_voxels，需要reshape为 [B, T, max_num_voxels, C]
                    max_num_voxels = voxel_data.get('max_num_voxels', P)
                    # 验证reshape是否有效
                    expected_size = B * T * max_num_voxels * C
                    actual_size = tokens.numel()
                    if actual_size != expected_size:
                        # 如果大小不匹配，尝试推断实际的max_num_voxels
                        inferred_P = actual_size // (B * T * C)
                        if inferred_P > 0 and actual_size % (B * T * C) == 0:
                            max_num_voxels = inferred_P
                            P = inferred_P
                        else:
                            # 如果无法被T整除，尝试推断实际的S
                            inferred_S = actual_size // (B * P * C) if P > 0 else actual_size // (B * C)
                            if inferred_S > 0 and actual_size % (B * inferred_S * C) == 0:
                                # 重新计算P
                                inferred_P = actual_size // (B * inferred_S * C)
                                max_num_voxels = inferred_P
                                P = inferred_P
                                S = inferred_S
                            else:
                                raise RuntimeError(
                                    f"Invalid tensor size for reshape: expected {expected_size}, got {actual_size}, "
                                    f"shape {tokens.shape}. B={B}, T={T}, max_num_voxels={max_num_voxels}, C={C}"
                                )
                    
                    # 根据实际元素数计算正确的S和P
                    # 优先尝试直接reshape为 [B, T, max_num_voxels, C]
                    if actual_size == B * T * max_num_voxels * C:
                        # 可以直接reshape为期望的形状
                        tokens_reshaped = tokens.reshape(B, T, max_num_voxels, C)
                    else:
                        # 计算实际的S和P
                        actual_S = actual_size // (B * P * C) if P > 0 else 1
                        if actual_S * B * P * C == actual_size:
                            tokens_reshaped = tokens.reshape(B, actual_S, P, C)
                        else:
                            # 尝试重新计算P
                            actual_P = actual_size // (B * actual_S * C) if actual_S > 0 else actual_size // (B * C)
                            if actual_P > 0 and actual_S * B * actual_P * C == actual_size:
                                tokens_reshaped = tokens.reshape(B, actual_S, actual_P, C)
                                max_num_voxels = actual_P
                                P = actual_P
                            else:
                                raise RuntimeError(
                                    f"Cannot reshape tokens: size {actual_size}, shape {tokens.shape}. "
                                    f"Expected size for [B={B}, T={T}, max_num_voxels={max_num_voxels}, C={C}]: "
                                    f"{B * T * max_num_voxels * C}"
                                )
                    
                    if stage == "stage1":
                        # Stage-1: Frame attention over all views at same time
                        # 体素化路径：需要reshape为 [B, T, max_num_voxels, C]
                        # 如果tokens_reshaped还不是这个形状，需要再次reshape
                        if tokens_reshaped.shape != (B, T, max_num_voxels, C):
                            # 尝试reshape为 [B, T, max_num_voxels, C]
                            if tokens_reshaped.numel() == B * T * max_num_voxels * C:
                                tokens_reshaped = tokens_reshaped.reshape(B, T, max_num_voxels, C)
                            else:
                                # 如果无法reshape，可能需要调整max_num_voxels
                                actual_shape = tokens_reshaped.shape
                                if len(actual_shape) == 4:
                                    _, actual_S_reshaped, actual_P_reshaped, _ = actual_shape
                                    if actual_S_reshaped == T:
                                        # 可以直接使用
                                        max_num_voxels = actual_P_reshaped
                                    else:
                                        # 尝试重新reshape
                                        tokens_reshaped = tokens_reshaped.reshape(B, T, -1, C)
                                        max_num_voxels = tokens_reshaped.shape[2]
                        tokens_for_attn = tokens_reshaped.reshape(B * T, max_num_voxels, C)
                        if pos is not None:
                            # 使用计算出的实际S和P值来reshape pos
                            pos_size = pos.numel()
                            if pos_size == B * T * max_num_voxels * 2:
                                pos_reshaped = pos.reshape(B, T, max_num_voxels, 2)
                            else:
                                # 尝试根据实际元素数推断
                                pos_inferred_S = pos_size // (B * max_num_voxels * 2) if max_num_voxels > 0 else 1
                                if pos_inferred_S * B * max_num_voxels * 2 == pos_size:
                                    pos_reshaped = pos.reshape(B, pos_inferred_S, max_num_voxels, 2)
                                    # 如果S不等于T，需要调整
                                    if pos_inferred_S != T and pos_size == B * T * max_num_voxels * 2:
                                        pos_reshaped = pos.reshape(B, T, max_num_voxels, 2)
                                else:
                                    # 尝试直接reshape为 [B, T, max_num_voxels, 2]
                                    if pos_size % (B * max_num_voxels * 2) == 0:
                                        pos_inferred_T = pos_size // (B * max_num_voxels * 2)
                                        pos_reshaped = pos.reshape(B, pos_inferred_T, max_num_voxels, 2)
                                    else:
                                        raise RuntimeError(
                                            f"Cannot reshape pos: size {pos_size}, shape {pos.shape}. "
                                            f"Expected size for [B={B}, T={T}, max_num_voxels={max_num_voxels}, 2]: "
                                            f"{B * T * max_num_voxels * 2}"
                                        )
                            pos_for_attn = pos_reshaped.reshape(B * T, max_num_voxels, 2)
                        else:
                            pos_for_attn = None
                    elif stage == "stage2":
                        # Stage-2: Frame attention over same view across time
                        # 体素化路径：按时间窗口处理
                        tokens_reshaped = tokens_reshaped.reshape(B, T, max_num_voxels, C)
                        tokens_for_attn = tokens_reshaped.reshape(B * T, max_num_voxels, C)
                        if pos is not None:
                            pos_reshaped = pos.reshape(B, S, P, 2)
                            pos_for_attn = pos_reshaped.reshape(B, T, max_num_voxels, 2).reshape(B * T, max_num_voxels, 2)
                        else:
                            pos_for_attn = None
                    else:
                        # 非stage1/stage2的情况（体素化路径），直接使用原始tokens
                        tokens_for_attn = tokens
                        pos_for_attn = pos
                else:
                    # 非体素化路径：S = T*V，可以reshape为 [B, T, V, P, C]
                    # 首先验证tokens是否可以reshape为[B, S, P, C]
                    if tokens.numel() == B * S * P * C:
                        tokens_reshaped = tokens.reshape(B, S, P, C)
                    else:
                        # 如果大小不匹配，重新推断S和P
                        # 从实际大小推断：actual_size = B * S_actual * P_actual * C
                        S_calc = tokens.shape[0] // B if tokens.shape[0] % B == 0 else S
                        P_calc = tokens.shape[1]
                        if B * S_calc * P_calc * C == tokens.numel():
                            S = S_calc
                            P = P_calc
                            tokens_reshaped = tokens.reshape(B, S, P, C)
                        else:
                            # 如果推断失败，直接使用tokens，不进行stage特定的reshape
                            tokens_for_attn = tokens
                            pos_for_attn = pos if pos is not None else None
                            # 跳过stage特定的reshape
                            if stage == "stage1":
                                # Stage-1: 直接使用tokens，不reshape
                                pass
                            elif stage == "stage2":
                                # Stage-2: 直接使用tokens，不reshape
                                pass
                    
                    # 如果tokens_reshaped已设置，继续stage特定的reshape
                    if 'tokens_reshaped' in locals() and 'tokens_for_attn' not in locals():
                        if stage == "stage1":
                            # Stage-1: Frame attention over all views at same time
                            # Reshape: [B, S, P, C] -> [B, T, V, P, C] -> [B*T, V*P, C]
                            # 验证S == T*V
                            if S == T * V and tokens_reshaped.numel() == B * T * V * P * C:
                                try:
                                    tokens_reshaped = tokens_reshaped.reshape(B, T, V, P, C)
                                    tokens_for_attn = tokens_reshaped.reshape(B * T, V * P, C)
                                except RuntimeError as e:
                                    # 如果reshape失败，使用原始tokens
                                    tokens_for_attn = tokens
                            else:
                                # 如果S != T*V，无法reshape，使用原始tokens
                                tokens_for_attn = tokens
                            # Also reshape positions
                            if pos is not None:
                                try:
                                    pos_reshaped = pos.reshape(B, S, P, 2)
                                    if S == T * V:
                                        pos_for_attn = pos_reshaped.reshape(B, T, V, P, 2).reshape(B * T, V * P, 2)
                                    else:
                                        pos_for_attn = None
                                except RuntimeError:
                                    pos_for_attn = None
                            else:
                                pos_for_attn = None
                        elif stage == "stage2":
                            # Stage-2: Frame attention over same view across time
                            # Reshape: [B, S, P, C] -> [B, T, V, P, C] -> [B*V, T*P, C]
                            if S == T * V and tokens_reshaped.numel() == B * T * V * P * C:
                                try:
                                    tokens_reshaped = tokens_reshaped.reshape(B, T, V, P, C)
                                    tokens_for_attn = tokens_reshaped.permute(0, 2, 1, 3, 4).contiguous().reshape(B * V, T * P, C)
                                except RuntimeError:
                                    tokens_for_attn = tokens
                            else:
                                tokens_for_attn = tokens
                            # Also reshape positions
                            if pos is not None:
                                try:
                                    pos_reshaped = pos.reshape(B, S, P, 2)
                                    if S == T * V:
                                        pos_for_attn = pos_reshaped.reshape(B, T, V, P, 2).permute(0, 2, 1, 3, 4).contiguous().reshape(B * V, T * P, 2)
                                    else:
                                        pos_for_attn = None
                                except RuntimeError:
                                    pos_for_attn = None
                            else:
                                pos_for_attn = None
                        else:
                            # 非stage1/stage2的情况（非体素化路径），直接使用原始tokens
                            tokens_for_attn = tokens
                            pos_for_attn = pos
                
                # 确保pos_for_attn不为None（block需要位置编码）
                if pos_for_attn is None:
                    # 生成默认的位置编码（零位置编码作为占位符），整型以满足embedding需求
                    # tokens_for_attn的形状: [B_attn, P_attn, C]
                    B_attn, P_attn, C_attn = tokens_for_attn.shape
                    pos_for_attn = torch.zeros(B_attn, P_attn, 2, device=tokens_for_attn.device, dtype=torch.long)
                else:
                    # 强制转换为整型以匹配F.embedding索引类型
                    if pos_for_attn.dtype != torch.long:
                        pos_for_attn = pos_for_attn.long()
                
                # 确保当final_attn_mask不为None时，attn_value也不为None
                # attention.py期望attn_mask和attn_value同时存在或同时为None
                # attention.py期望attn_mask是[B, N]形状，但我们的temporal_attn_mask是[B, N, N]
                # 需要将[B, N, N]转换为[B, N]：对于每个token，如果它与任何其他token有attention，则值为True
                B_attn, P_attn, C_attn = tokens_for_attn.shape
                attn_mask_for_block = final_attn_mask
                attn_value_for_block = attn_value
                
                if final_attn_mask is not None:
                    # 如果final_attn_mask是3维的[B, N, N]，需要转换为2维的[B, N]
                    if final_attn_mask.dim() == 3:
                        attn_mask_B, attn_mask_N1, attn_mask_N2 = final_attn_mask.shape
                        # 将[B, N, N]转换为[B, N]：对每个token，如果它与任何其他token有attention，则值为True
                        # 方法：对每一行求和，如果sum > 0则True
                        attn_mask_2d = final_attn_mask.any(dim=-1)  # [B, N]
                        # 确保形状与tokens_for_attn匹配
                        if attn_mask_B == B_attn and attn_mask_N1 == P_attn:
                            attn_mask_for_block = attn_mask_2d
                        else:
                            # 如果形状不匹配，尝试reshape或创建默认mask
                            # 这里可能需要根据实际reshape情况调整，暂时使用默认值
                            attn_mask_for_block = torch.ones(B_attn, P_attn, device=tokens_for_attn.device, dtype=torch.bool)
                    elif final_attn_mask.dim() == 2:
                        # 已经是2维的，但需要确保形状匹配
                        attn_mask_B, attn_mask_N = final_attn_mask.shape
                        if attn_mask_B == B_attn and attn_mask_N == P_attn:
                            attn_mask_for_block = final_attn_mask
                        else:
                            attn_mask_for_block = torch.ones(B_attn, P_attn, device=tokens_for_attn.device, dtype=torch.bool)
                    
                    # 创建或调整attn_value
                    if attn_value_for_block is None:
                        attn_value_for_block = torch.ones(B_attn, P_attn, device=tokens_for_attn.device, dtype=tokens_for_attn.dtype)
                    elif attn_value_for_block.dim() == 2:
                        attn_val_B, attn_val_N = attn_value_for_block.shape
                        if attn_val_B != B_attn or attn_val_N != P_attn:
                            attn_value_for_block = torch.ones(B_attn, P_attn, device=tokens_for_attn.device, dtype=tokens_for_attn.dtype)
                
                # Apply attention block with temporal mask
                if self.training:
                    fn = functools.partial(block, attn_mask=attn_mask_for_block, attn_value=attn_value_for_block)
                    tokens_out = torch.utils.checkpoint.checkpoint(
                        fn, tokens_for_attn, pos_for_attn, use_reentrant=self.use_reentrant)
                else:
                    tokens_out = block(tokens_for_attn, pos=pos_for_attn, attn_mask=attn_mask_for_block, attn_value=attn_value_for_block)
                
                # Reshape back: restore original shape
                # 首先检查tokens_out的实际形状
                B_out, P_out, C_out = tokens_out.shape
                S_out = B_out // B if B_out % B == 0 else 1
                
                if use_voxel_routing:
                    # 体素化路径：恢复为 [B, T, max_num_voxels, C] -> [B, S, P, C]
                    if stage == "stage1" or stage == "stage2":
                        # tokens_out应该是[B*T, max_num_voxels, C]
                        tokens_out_size = tokens_out.numel()
                        expected_size_bt_max = B * T * max_num_voxels * C
                        expected_size_bs_p = B * S * P * C
                        
                        if tokens_out_size == expected_size_bt_max:
                            # 可以reshape为 [B, T, max_num_voxels, C]
                            try:
                                tokens_out = tokens_out.reshape(B, T, max_num_voxels, C)
                                # 然后尝试reshape为 [B, S, P, C]
                                if tokens_out_size == expected_size_bs_p:
                                    tokens_out = tokens_out.reshape(B, S, P, C).reshape(B * S, P, C)
                                else:
                                    # 如果大小不匹配，根据实际大小计算S和P
                                    actual_S = tokens_out_size // (B * P * C) if P > 0 else tokens_out_size // (B * C)
                                    if actual_S * B * P * C == tokens_out_size:
                                        tokens_out = tokens_out.reshape(B, actual_S, P, C).reshape(B * actual_S, P, C)
                                    else:
                                        # 重新计算P
                                        actual_P = tokens_out_size // (B * actual_S * C) if actual_S > 0 else tokens_out_size // (B * C)
                                        if actual_P > 0 and actual_S * B * actual_P * C == tokens_out_size:
                                            tokens_out = tokens_out.reshape(B, actual_S, actual_P, C).reshape(B * actual_S, actual_P, C)
                                        else:
                                            # 直接reshape为 [B*T, max_num_voxels, C] -> [B*S, P, C]
                                            # 计算实际可能的S和P
                                            actual_S_calc = tokens_out_size // (B * P * C) if P > 0 else 1
                                            actual_P_calc = tokens_out_size // (B * actual_S_calc * C) if actual_S_calc > 0 else P
                                            tokens_out = tokens_out.reshape(B * actual_S_calc, actual_P_calc, C)
                            except RuntimeError as e:
                                # 如果reshape失败，根据实际元素数计算正确的S和P
                                actual_S_calc = tokens_out_size // (B * P * C) if P > 0 else tokens_out_size // (B * C)
                                actual_P_calc = tokens_out_size // (B * actual_S_calc * C) if actual_S_calc > 0 else P
                                if actual_S_calc > 0 and actual_P_calc > 0 and actual_S_calc * B * actual_P_calc * C == tokens_out_size:
                                    tokens_out = tokens_out.reshape(B * actual_S_calc, actual_P_calc, C)
                                else:
                                    # 使用tokens_out的当前形状
                                    B_out, P_out, C_out = tokens_out.shape
                                    # 保持当前形状，不进行reshape
                                    pass
                        else:
                            # 如果大小不匹配，根据实际元素数计算正确的S和P
                            actual_S_calc = tokens_out_size // (B * P * C) if P > 0 else tokens_out_size // (B * C)
                            actual_P_calc = tokens_out_size // (B * actual_S_calc * C) if actual_S_calc > 0 else P
                            if actual_S_calc > 0 and actual_P_calc > 0 and actual_S_calc * B * actual_P_calc * C == tokens_out_size:
                                tokens_out = tokens_out.reshape(B * actual_S_calc, actual_P_calc, C)
                            else:
                                # 如果无法匹配，保持当前形状
                                pass
                    else:
                        # 非stage1/stage2的情况，根据实际元素数计算
                        tokens_out_size = tokens_out.numel()
                        actual_S_calc = tokens_out_size // (B * P * C) if P > 0 else tokens_out_size // (B * C)
                        actual_P_calc = tokens_out_size // (B * actual_S_calc * C) if actual_S_calc > 0 else P
                        if actual_S_calc > 0 and actual_P_calc > 0 and actual_S_calc * B * actual_P_calc * C == tokens_out_size:
                            tokens_out = tokens_out.reshape(B * actual_S_calc, actual_P_calc, C)
                        else:
                            # 保持当前形状
                            pass
                else:
                    # 非体素化路径：恢复为 [B, T, V, P, C] -> [B, S, P, C]
                    if stage == "stage1":
                        # tokens_out应该是[B*T, V*P, C]（从tokens_for_attn = [B*T, V*P, C]）
                        # 需要恢复为[B*S, P, C]，其中S = T*V
                        # 首先检查tokens_out的实际形状
                        if B_out == B * T:
                            # tokens_out是[B*T, P_out, C]，其中P_out可能是V*P
                            P_out_actual = P_out
                            # 尝试reshape为[B*S, P, C]
                            if tokens_out.numel() == B * S * P * C:
                                # 大小匹配，直接reshape
                                try:
                                    tokens_out = tokens_out.reshape(B * S, P, C)
                                except RuntimeError:
                                    # 如果失败，尝试通过中间步骤
                                    if P_out_actual % V == 0:
                                        P_calc = P_out_actual // V
                                        if tokens_out.numel() == B * T * V * P_calc * C:
                                            tokens_out = tokens_out.reshape(B, T, V, P_calc, C).reshape(B * S, P_calc, C)
                                            P = P_calc
                                        else:
                                            tokens_out = tokens_out.reshape(B * S, P, C)
                                    else:
                                        tokens_out = tokens_out.reshape(B * S, P, C)
                            elif P_out_actual % V == 0:
                                # P_out可以被V整除，尝试推断P
                                P_calc = P_out_actual // V
                                if tokens_out.numel() == B * S * P_calc * C:
                                    # 通过中间步骤reshape
                                    try:
                                        tokens_out = tokens_out.reshape(B, T, V, P_calc, C).reshape(B * S, P_calc, C)
                                        P = P_calc  # 更新P
                                    except RuntimeError:
                                        tokens_out = tokens_out.reshape(B * S, P_calc, C)
                                        P = P_calc
                                else:
                                    # 直接reshape为[B*S, P_calc, C]
                                    tokens_out = tokens_out.reshape(B * S, P_calc, C)
                                    P = P_calc
                            else:
                                # 如果P_out不能被V整除，直接使用实际形状
                                tokens_out = tokens_out.reshape(B_out, P_out, C_out)
                                S = S_out
                                P = P_out
                        else:
                            # tokens_out的形状不是[B*T, ...]，直接reshape为[B*S, P, C]
                            if tokens_out.numel() == B * S * P * C:
                                tokens_out = tokens_out.reshape(B * S, P, C)
                            else:
                                tokens_out = tokens_out.reshape(B_out, P_out, C_out)
                                S = S_out
                                P = P_out
                    elif stage == "stage2":
                        # tokens_out应该是[B*V, T*P, C]
                        expected_size = B * V * T * P * C
                        if tokens_out.numel() == expected_size:
                            try:
                                tokens_out = tokens_out.reshape(B, V, T * P, C).reshape(B, V, T, P, C).permute(0, 2, 1, 3, 4).contiguous().reshape(B, S, P, C).reshape(B * S, P, C)
                            except RuntimeError:
                                tokens_out = tokens_out.reshape(B * S, P, C)
                        else:
                            # 如果大小不匹配，从实际形状推断
                            # tokens_out是[B*V, P_out, C]，需要reshape为[B*S, P, C]
                            if S == T * V and P_out % T == 0:
                                P_calc = P_out // T
                                if tokens_out.numel() == B * S * P_calc * C:
                                    tokens_out = tokens_out.reshape(B, V, T, P_calc, C).permute(0, 2, 1, 3, 4).contiguous().reshape(B * S, P_calc, C)
                                    P = P_calc  # 更新P
                                else:
                                    tokens_out = tokens_out.reshape(B * S, P, C)
                            else:
                                if tokens_out.numel() == B * S * P * C:
                                    tokens_out = tokens_out.reshape(B * S, P, C)
                                else:
                                    tokens_out = tokens_out.reshape(B_out, P_out, C_out)
                                    S = S_out
                                    P = P_out
                    else:
                        # 非stage1/stage2的情况，直接reshape
                        if tokens_out.numel() == B * S * P * C:
                            tokens_out = tokens_out.reshape(B * S, P, C)
                        else:
                            # 使用实际形状
                            tokens_out = tokens_out.reshape(B_out, P_out, C_out)
                            S = S_out
                            P = P_out
                
                tokens = tokens_out
            else:
                # Legacy mode: no routing change
                # 确保pos不为None（block需要位置编码）
                if pos is None:
                    # 生成默认的位置编码（零位置编码作为占位符），整型以满足embedding需求
                    B_legacy, P_legacy, C_legacy = tokens.shape
                    pos = torch.zeros(B_legacy, P_legacy, 2, device=tokens.device, dtype=torch.long)
                else:
                    if pos.dtype != torch.long:
                        pos = pos.long()

                # 确保当final_attn_mask不为None时，attn_value也不为None
                # attention.py期望attn_mask和attn_value同时存在或同时为None
                # attention.py期望attn_mask是[B, N]形状，但我们的temporal_attn_mask是[B, N, N]
                # 需要将[B, N, N]转换为[B, N]：对于每个token，如果它与任何其他token有attention，则值为True
                B_legacy, P_legacy, C_legacy = tokens.shape
                attn_mask_for_block = final_attn_mask
                attn_value_for_block = attn_value
                
                if final_attn_mask is not None:
                    # 如果final_attn_mask是3维的[B, N, N]，需要转换为2维的[B, N]
                    if final_attn_mask.dim() == 3:
                        attn_mask_B, attn_mask_N1, attn_mask_N2 = final_attn_mask.shape
                        # 将[B, N, N]转换为[B, N]：对每个token，如果它与任何其他token有attention，则值为True
                        attn_mask_2d = final_attn_mask.any(dim=-1)  # [B, N]
                        # 确保形状与tokens匹配
                        if attn_mask_B == B_legacy and attn_mask_N1 == P_legacy:
                            attn_mask_for_block = attn_mask_2d
                        else:
                            # 如果形状不匹配，创建默认mask
                            attn_mask_for_block = torch.ones(B_legacy, P_legacy, device=tokens.device, dtype=torch.bool)
                    elif final_attn_mask.dim() == 2:
                        # 已经是2维的，但需要确保形状匹配
                        attn_mask_B, attn_mask_N = final_attn_mask.shape
                        if attn_mask_B == B_legacy and attn_mask_N == P_legacy:
                            attn_mask_for_block = final_attn_mask
                        else:
                            attn_mask_for_block = torch.ones(B_legacy, P_legacy, device=tokens.device, dtype=torch.bool)
                    
                    # 创建或调整attn_value
                    if attn_value_for_block is None:
                        attn_value_for_block = torch.ones(B_legacy, P_legacy, device=tokens.device, dtype=tokens.dtype)
                    elif attn_value_for_block.dim() == 2:
                        attn_val_B, attn_val_N = attn_value_for_block.shape
                        if attn_val_B != B_legacy or attn_val_N != P_legacy:
                            attn_value_for_block = torch.ones(B_legacy, P_legacy, device=tokens.device, dtype=tokens.dtype)

                if self.training:
                    fn = functools.partial(block, attn_mask=attn_mask_for_block, attn_value=attn_value_for_block)
                    tokens = torch.utils.checkpoint.checkpoint(
                        fn, tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = block(tokens, pos=pos, attn_mask=attn_mask_for_block, attn_value=attn_value_for_block)
            
            frame_idx += 1
            # 保存intermediates时，需要reshape为[B, S, P, C]
            # 但需要检查实际元素数是否匹配
            tokens_size = tokens.numel()
            expected_size = B * S * P * C
            if tokens_size == expected_size:
                intermediates.append(tokens.reshape(B, S, P, C))
            else:
                # 如果大小不匹配，根据实际元素数计算正确的S和P
                # tokens的形状应该是[B*S_actual, P_actual, C]
                B_actual, P_actual, C_actual = tokens.shape
                S_actual = B_actual // B if B_actual % B == 0 else 1
                
                # 检查是否可以reshape为[B, S_actual, P_actual, C]
                if tokens_size == B * S_actual * P_actual * C:
                    intermediates.append(tokens.reshape(B, S_actual, P_actual, C))
                else:
                    # 如果仍然不匹配，尝试重新计算
                    # 重新计算S和P
                    S_calc = tokens_size // (B * P * C) if P > 0 else tokens_size // (B * C)
                    if S_calc > 0 and tokens_size % (B * S_calc * C) == 0:
                        P_calc = tokens_size // (B * S_calc * C)
                        if P_calc > 0:
                            intermediates.append(tokens.reshape(B, S_calc, P_calc, C))
                        else:
                            # 如果无法计算，使用实际形状
                            intermediates.append(tokens.reshape(B_actual, P_actual, C_actual))
                    else:
                        # 如果无法计算，使用实际形状
                        intermediates.append(tokens.reshape(B_actual, P_actual, C_actual))
        
        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self, tokens, B, S, P, C, global_idx, pos=None, temporal_features=None, 
        attn_mask=None, attn_value=None, is_multi_view=False, T=None, V=None,
        voxel_data=None, num_block=None
    ):
        """ 
        Process global attention blocks with optional voxel-aware routing.
        
        Stage-1: 空间邻域（体素KNN/半径r）稀疏注意力
        Stage-2: 空间邻域 + 掩码门控（仅在位姿分支）
        """
        # 检查并修复tokens的形状
        # 如果tokens不是期望的形状，尝试推断正确的形状
        B_actual, N_actual, C_actual = tokens.shape
        expected_size = B * S * P * C
        actual_size = tokens.numel()
        
        if tokens.shape != (B, S * P, C):
            if actual_size == expected_size:
                # 大小匹配，但形状不对，直接reshape
                try:
                    tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)
                except RuntimeError:
                    # reshape失败，使用当前形状并更新S和P
                    tokens = tokens.reshape(B_actual, N_actual, C_actual)
                    # 尝试从voxel_data推断S和P
                    if voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
                        max_num_voxels = voxel_data.get('max_num_voxels', P)
                        if T is not None:
                            # 体素化路径：S = T, P = max_num_voxels
                            inferred_S = T
                            inferred_P = max_num_voxels
                            if inferred_S * inferred_P == N_actual:
                                S = inferred_S
                                P = inferred_P
                            else:
                                # 如果推断失败，使用当前N_actual
                                S = N_actual // P if P > 0 else 1
                                P = N_actual // S if S > 0 else N_actual
                        else:
                            S = N_actual // P if P > 0 else 1
                            P = N_actual // S if S > 0 else N_actual
                    else:
                        # 非体素化路径，尝试推断
                        inferred_P = N_actual // S if S > 0 else P
                        inferred_S = N_actual // inferred_P if inferred_P > 0 else S
                        if inferred_S * inferred_P == N_actual:
                            S = inferred_S
                            P = inferred_P
                        else:
                            S = N_actual // P if P > 0 else 1
                            P = N_actual // S if S > 0 else N_actual
            else:
                # 大小不匹配，需要推断正确的S和P
                if B_actual == B and C_actual == C:
                    # 尝试从voxel_data获取信息
                    if voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
                        max_num_voxels = voxel_data.get('max_num_voxels', P)
                        if T is not None:
                            # 体素化路径：S = T, P = max_num_voxels
                            inferred_S = T
                            inferred_P = max_num_voxels
                            if inferred_S * inferred_P == N_actual:
                                S = inferred_S
                                P = inferred_P
                                tokens = tokens.reshape(B, S * P, C)
                            else:
                                # 如果推断失败，使用当前形状
                                tokens = tokens.reshape(B, N_actual, C)
                                S = N_actual // P if P > 0 else 1
                                P = N_actual // S if S > 0 else N_actual
                        else:
                            tokens = tokens.reshape(B, N_actual, C)
                            S = N_actual // P if P > 0 else 1
                            P = N_actual // S if S > 0 else N_actual
                    else:
                        # 非体素化路径，尝试推断
                        inferred_P = N_actual // S if S > 0 else P
                        inferred_S = N_actual // inferred_P if inferred_P > 0 else S
                        if inferred_S * inferred_P == N_actual:
                            S = inferred_S
                            P = inferred_P
                            tokens = tokens.reshape(B, S * P, C)
                        else:
                            tokens = tokens.reshape(B, N_actual, C)
                            S = N_actual // P if P > 0 else 1
                            P = N_actual // S if S > 0 else N_actual
                else:
                    # B或C不匹配，保持当前形状
                    tokens = tokens.reshape(B_actual, N_actual, C_actual)
                    S = N_actual // P if P > 0 else 1
                    P = N_actual // S if S > 0 else N_actual
        
        # 确保tokens是期望的形状 [B, S*P, C]
        if tokens.shape != (B, S * P, C):
            # 如果大小匹配，强制reshape
            if tokens.numel() == B * S * P * C:
                tokens = tokens.reshape(B, S * P, C)
            else:
                # 使用实际形状
                B_actual, N_actual, C_actual = tokens.shape
                tokens = tokens.reshape(B_actual, N_actual, C_actual)
                # 更新S和P以匹配实际形状
                S = N_actual // P if P > 0 else 1
                P = N_actual // S if S > 0 else N_actual
        
        if pos is not None and pos.shape != (B, S * P, 2):
            # 类似地处理pos
            if pos.numel() == B * S * P * 2:
                pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            else:
                # 如果大小不匹配，尝试推断
                B_pos, N_pos, D_pos = pos.shape
                if B_pos == B and D_pos == 2:
                    if N_pos == S * P:
                        pos = pos.reshape(B, S * P, 2)
                    else:
                        # 使用实际形状
                        pos = pos.reshape(B_pos, N_pos, D_pos)
        
        # 判断是否使用体素化路由
        use_voxel_routing = (voxel_data is not None) and voxel_data.get('use_voxel_tokens', False) and (num_block is not None)
        
        # 构建空间邻域attention mask（如果需要）
        spatial_attn_mask = None
        voxel_mask_bias = None
        
        # Stage-1: 空间邻域（体素KNN/半径r）稀疏注意力
        if use_voxel_routing and num_block in self.temporal_list1:
            # 获取体素坐标和半径
            voxel_xyz_list = voxel_data.get('voxel_xyz_list', None)
            if voxel_xyz_list is not None and self.voxel_size is not None:
                # 计算半径
                radius = self.radius_stage1 * self.voxel_size
                
                # 为每个batch和时间步构建邻域mask
                # tokens: [B, S*P, C] where S = T (for voxelized) and P = max_num_voxels
                max_num_voxels = voxel_data.get('max_num_voxels', P)
                spatial_attn_mask = torch.zeros(B, S * max_num_voxels, S * max_num_voxels, 
                                               device=tokens.device, dtype=torch.bool)
                
                # 对每个batch和时间步分别处理
                for b in range(B):
                    for t in range(S):
                        if t < len(voxel_xyz_list):
                            xyz_list_t = voxel_xyz_list[t]
                            if b < len(xyz_list_t) and xyz_list_t[b] is not None:
                                voxel_xyz_tb = xyz_list_t[b]  # [N_t, 3]
                                if voxel_xyz_tb.shape[0] > 0:
                                    # 构建半径图
                                    edge_index, _ = self.build_radius_graph(voxel_xyz_tb, radius)
                                    
                                    # 创建attention mask：只允许邻域内的attention
                                    N_t = voxel_xyz_tb.shape[0]
                                    token_start = t * max_num_voxels
                                    
                                    # 初始化：允许所有token（包括self）
                                    mask_tb = torch.eye(N_t, device=tokens.device, dtype=torch.bool)
                                    
                                    # 添加邻域边
                                    if edge_index.shape[1] > 0:
                                        src = edge_index[0]
                                        dst = edge_index[1]
                                        mask_tb[src, dst] = True
                                    
                                    # 扩展到完整mask（padding部分设为False）
                                    full_mask_tb = torch.zeros(max_num_voxels, max_num_voxels, 
                                                              device=tokens.device, dtype=torch.bool)
                                    full_mask_tb[:N_t, :N_t] = mask_tb
                                    
                                    spatial_attn_mask[b, token_start:token_start+max_num_voxels, 
                                                     token_start:token_start+max_num_voxels] = full_mask_tb
        
        # Stage-2: 空间邻域 + 掩码门控（仅在位姿分支）
        if use_voxel_routing and num_block in list(range(8, 18)):
            # 空间邻域（类似Stage-1）
            voxel_xyz_list = voxel_data.get('voxel_xyz_list', None)
            if voxel_xyz_list is not None and self.voxel_size is not None:
                radius = self.radius_stage2 * self.voxel_size
                max_num_voxels = voxel_data.get('max_num_voxels', P)
                spatial_attn_mask = torch.zeros(B, S * max_num_voxels, S * max_num_voxels, 
                                               device=tokens.device, dtype=torch.bool)
                
                for b in range(B):
                    for t in range(S):
                        if t < len(voxel_xyz_list):
                            xyz_list_t = voxel_xyz_list[t]
                            if b < len(xyz_list_t) and xyz_list_t[b] is not None:
                                voxel_xyz_tb = xyz_list_t[b]
                                if voxel_xyz_tb.shape[0] > 0:
                                    edge_index, _ = self.build_radius_graph(voxel_xyz_tb, radius)
                                    N_t = voxel_xyz_tb.shape[0]
                                    token_start = t * max_num_voxels
                                    
                                    mask_tb = torch.eye(N_t, device=tokens.device, dtype=torch.bool)
                                    if edge_index.shape[1] > 0:
                                        src, dst = edge_index[0], edge_index[1]
                                        mask_tb[src, dst] = True
                                    
                                    full_mask_tb = torch.zeros(max_num_voxels, max_num_voxels, 
                                                              device=tokens.device, dtype=torch.bool)
                                    full_mask_tb[:N_t, :N_t] = mask_tb
                                    spatial_attn_mask[b, token_start:token_start+max_num_voxels, 
                                                     token_start:token_start+max_num_voxels] = full_mask_tb
            
            # 掩码门控：将体素掩码转换为attention bias（仅位姿分支）
            voxel_mask_list = voxel_data.get('voxel_mask_list', None)
            if voxel_mask_list is not None and hasattr(self, 'camera_mask_gate'):
                gamma = torch.sigmoid(self.camera_mask_gate)  # [1]
                max_num_voxels = voxel_data.get('max_num_voxels', P)
                
                # 创建voxel_mask_bias [B, S*P]
                voxel_mask_bias = torch.zeros(B, S * max_num_voxels, device=tokens.device, dtype=tokens.dtype)
                
                for b in range(B):
                    for t in range(S):
                        if t < len(voxel_mask_list):
                            mask_list_t = voxel_mask_list[t]
                            if b < len(mask_list_t) and mask_list_t[b] is not None:
                                voxel_mask_tb = mask_list_t[b]  # [N_t]
                                N_t = voxel_mask_tb.shape[0]
                                token_start = t * max_num_voxels
                                
                                # 将体素掩码转换为bias（动态体素被抑制）
                                # M~ >= τ0 表示动态，需要抑制
                                # bias = -large_value * gamma * M~ （仅动态体素）
                                large_value = 100.0  # 大的负值用于抑制
                                mask_bias_tb = -large_value * gamma * voxel_mask_tb  # [N_t]
                                
                                # 扩展到完整bias
                                full_bias_tb = torch.zeros(max_num_voxels, device=tokens.device, dtype=tokens.dtype)
                                full_bias_tb[:N_t] = mask_bias_tb
                                voxel_mask_bias[b, token_start:token_start+max_num_voxels] = full_bias_tb
        
        # 合并spatial_attn_mask和attn_mask
        final_attn_mask = attn_mask
        if spatial_attn_mask is not None:
            # spatial_attn_mask是bool类型的，需要转换为与attn_mask兼容的格式
            # 如果原attn_mask存在，需要合并（取交集）
            if attn_mask is not None:
                # attn_mask可能是不同的格式，这里简化处理
                # 假设spatial_attn_mask是主要的限制
                final_attn_mask = spatial_attn_mask
            else:
                final_attn_mask = spatial_attn_mask
        
        # 如果有voxel_mask_bias，需要添加到attention logits
        # 注意：这需要在Attention内部实现，这里先保存到attn_value中
        final_attn_value = attn_value
        if voxel_mask_bias is not None:
            # 将bias扩展到attention logits的维度
            # 这里简化：通过修改attn_value传递bias信息
            # 实际实现需要在Attention类中处理
            pass
        
        intermediates = []
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]
            
            # 确保pos不为None（block需要位置编码）
            if pos is None:
                # 生成默认的位置编码（零位置编码作为占位符），整型以满足embedding需求
                B_tokens, P_tokens, C_tokens = tokens.shape
                pos = torch.zeros(B_tokens, P_tokens, 2, device=tokens.device, dtype=torch.long)
            else:
                # 强制转换为整型以匹配F.embedding索引类型
                if pos.dtype != torch.long:
                    pos = pos.long()
            
            # 确保attn_mask和attn_value的形状与tokens匹配
            # block期望tokens的形状是[B*S, P, C]或类似的，但attn_mask需要是[B, N]，其中N是tokens的序列长度
            # 在attention.py中，tokens会被reshape为[B, N, C]用于计算，所以N = tokens.shape[1]
            B_tokens_actual, N_tokens_seq, C_tokens = tokens.shape
            # N_tokens_seq是tokens的序列长度，attn_mask需要匹配这个长度
            # 但attn_mask的batch维度是B，所以需要确保B_tokens_actual能被B整除
            if B_tokens_actual % B == 0:
                # tokens是[B*S, N_tokens_seq, C]格式
                # 在attention中，tokens会被处理，序列长度是N_tokens_seq
                # 所以attn_mask需要是[B, N_tokens_seq]
                attn_seq_len = N_tokens_seq
            else:
                # 如果B_tokens_actual不能被B整除，使用实际序列长度
                attn_seq_len = N_tokens_seq
            
            attn_mask_for_block = final_attn_mask
            attn_value_for_block = final_attn_value
            
            if final_attn_mask is not None:
                # 确保attn_mask的形状是[B, attn_seq_len]
                if final_attn_mask.dim() == 3:
                    # 如果是3维的[B, N, N]，转换为2维的[B, N]
                    attn_mask_B, attn_mask_N1, attn_mask_N2 = final_attn_mask.shape
                    attn_mask_2d = final_attn_mask.any(dim=-1)  # [B, N]
                    if attn_mask_B == B and attn_mask_N1 == attn_seq_len:
                        attn_mask_for_block = attn_mask_2d
                    else:
                        # 如果形状不匹配，创建默认mask
                        attn_mask_for_block = torch.ones(B, attn_seq_len, device=tokens.device, dtype=torch.bool)
                elif final_attn_mask.dim() == 2:
                    # 已经是2维的，检查形状
                    attn_mask_B, attn_mask_N = final_attn_mask.shape
                    if attn_mask_B == B and attn_mask_N == attn_seq_len:
                        attn_mask_for_block = final_attn_mask
                    else:
                        # 如果形状不匹配，创建默认mask
                        attn_mask_for_block = torch.ones(B, attn_seq_len, device=tokens.device, dtype=torch.bool)
                else:
                    # 其他维度，创建默认mask
                    attn_mask_for_block = torch.ones(B, attn_seq_len, device=tokens.device, dtype=torch.bool)
            else:
                # 如果final_attn_mask为None，设置为None（不使用mask）
                attn_mask_for_block = None
            
            # 确保attn_value的形状是[B, attn_seq_len]
            if attn_mask_for_block is not None:
                # 只有当attn_mask不为None时才需要attn_value
                if attn_value_for_block is None:
                    attn_value_for_block = torch.ones(B, attn_seq_len, device=tokens.device, dtype=tokens.dtype)
                elif attn_value_for_block.dim() == 2:
                    attn_val_B, attn_val_N = attn_value_for_block.shape
                    if attn_val_B != B or attn_val_N != attn_seq_len:
                        attn_value_for_block = torch.ones(B, attn_seq_len, device=tokens.device, dtype=tokens.dtype)
                else:
                    attn_value_for_block = torch.ones(B, attn_seq_len, device=tokens.device, dtype=tokens.dtype)
            else:
                # 如果attn_mask为None，attn_value也应该为None
                attn_value_for_block = None
            
            # 传递spatial_attn_mask和voxel_mask_bias
            # 注意：Attention类需要支持这些参数
            if self.training:
                fn = functools.partial(block, temporal_features=None, S=S, P=P, 
                                     attn_mask=attn_mask_for_block, attn_value=attn_value_for_block)
                tokens = torch.utils.checkpoint.checkpoint(
                    fn, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = block(tokens, pos=pos, temporal_features=None, S=S, P=P, 
                             attn_mask=attn_mask_for_block, attn_value=attn_value_for_block)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))
        return tokens, global_idx, intermediates
    
    def get_temporal_neighbors(
        self, voxel_ids: List[torch.Tensor], t: int, delta: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取同体素ID在时间窗口[t-delta..t+delta]的tokens
        
        Args:
            voxel_ids: 每个时间步的体素ID列表 [T个list，每个list是[B, N_t]]
            t: 当前时间步
            delta: 时间窗口大小
        Returns:
            neighbor_ids: 邻域体素ID [N_neighbors]
            neighbor_mask: 邻域mask [N_neighbors]
        """
        t_start = max(0, t - delta)
        t_end = min(len(voxel_ids), t + delta + 1)
        
        # 获取时间窗口内的所有体素ID
        window_ids = []
        for t_idx in range(t_start, t_end):
            if t_idx < len(voxel_ids):
                window_ids.append(voxel_ids[t_idx])
        
        # 找到与当前时间步t的体素ID相同的那些
        if len(window_ids) == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.bool)
        
        # 简化：返回所有窗口内的ID（实际应该做同ID匹配）
        all_ids = torch.cat(window_ids, dim=0)
        return all_ids, torch.ones(len(all_ids), dtype=torch.bool)
    
    def build_radius_graph(
        self, voxel_xyz: torch.Tensor, radius: float, max_num_neighbors: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build radius graph for spatial neighborhood attention.
        Uses torch_cluster if available, otherwise falls back to efficient distance computation.
        
        Args:
            voxel_xyz: Voxel center coordinates [N, 3]
            radius: Search radius in world coordinates
            max_num_neighbors: Maximum number of neighbors per node (for fixed-size output)
        
        Returns:
            edge_index: Edge indices [2, E] where E is the number of edges
            edge_distances: Edge distances [E] (optional)
        """
        N = voxel_xyz.shape[0]
        device = voxel_xyz.device
        
        if N == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)
        
        # Try to use torch_cluster if available
        try:
            import torch_cluster
            edge_index = torch_cluster.radius_graph(
                voxel_xyz, 
                r=radius, 
                max_num_neighbors=max_num_neighbors if max_num_neighbors else 128,
                flow='source_to_target'
            )
            
            # Compute edge distances
            if edge_index.shape[1] > 0:
                edge_src = edge_index[0]
                edge_dst = edge_index[1]
                edge_distances = torch.norm(voxel_xyz[edge_src] - voxel_xyz[edge_dst], dim=1)
            else:
                edge_distances = torch.empty(0, device=device)
            
            return edge_index, edge_distances
        except ImportError:
            # Fallback: use efficient distance computation
            # For large N, use chunked computation to save memory
            chunk_size = 1000
            edges_list = []
            distances_list = []
            
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                xyz_chunk = voxel_xyz[i:end_i]  # [chunk_size, 3]
                
                # Compute distances from this chunk to all points
                dists = torch.cdist(xyz_chunk, voxel_xyz)  # [chunk_size, N]
                mask = dists <= radius
                
                # Create edge indices for this chunk
                for local_idx, global_idx in enumerate(range(i, end_i)):
                    neighbors = torch.where(mask[local_idx])[0]
                    # Exclude self
                    neighbors = neighbors[neighbors != global_idx]
                    
                    if len(neighbors) > 0:
                        # Limit neighbors if max_num_neighbors is set
                        if max_num_neighbors is not None and len(neighbors) > max_num_neighbors:
                            neighbor_dists = dists[local_idx, neighbors]
                            _, sorted_idx = torch.sort(neighbor_dists)
                            neighbors = neighbors[sorted_idx[:max_num_neighbors]]
                        
                        # Create edges: [global_idx] -> [neighbors]
                        src = torch.full((len(neighbors),), global_idx, dtype=torch.long, device=device)
                        dst = neighbors
                        edges_chunk = torch.stack([src, dst], dim=0)  # [2, len(neighbors)]
                        edges_list.append(edges_chunk)
                        
                        neighbor_dists = dists[local_idx, neighbors]
                        distances_list.append(neighbor_dists)
            
            if len(edges_list) > 0:
                edge_index = torch.cat(edges_list, dim=1)  # [2, E]
                edge_distances = torch.cat(distances_list, dim=0)  # [E]
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_distances = torch.empty(0, device=device)
            
            return edge_index, edge_distances
    
    def get_spatial_neighbors(
        self, voxel_xyz: torch.Tensor, radius: Optional[float] = None, 
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取空间邻域（球半径r或KNN）
        
        Args:
            voxel_xyz: 体素中心坐标 [N, 3]
            radius: 球半径（世界坐标单位，与voxel_size同量纲）
            k: KNN的k值（如果提供，优先使用KNN）
        Returns:
            neighbor_indices: 邻域索引 [N, K] 或 [N, variable]
            neighbor_distances: 邻域距离 [N, K]
        """
        # 简化实现：使用欧氏距离计算
        # TODO: 使用KDTree或HashGrid优化
        N = voxel_xyz.shape[0]
        device = voxel_xyz.device
        
        if N == 0:
            return torch.empty(0, 0, dtype=torch.long, device=device), torch.empty(0, 0, device=device)
        
        if k is not None:
            # KNN
            distances = torch.cdist(voxel_xyz, voxel_xyz)  # [N, N]
            distances.fill_diagonal_(float('inf'))  # 排除自身
            k_actual = min(k, N - 1)
            neighbor_distances, neighbor_indices = torch.topk(distances, k_actual, dim=1, largest=False)
            return neighbor_indices, neighbor_distances
        elif radius is not None:
            # 半径查询
            distances = torch.cdist(voxel_xyz, voxel_xyz)  # [N, N]
            mask = distances <= radius
            mask.fill_diagonal_(False)  # 排除自身
            
            # 返回所有在半径内的邻居（变长，需要padding或返回list）
            neighbor_list = []
            dist_list = []
            for i in range(N):
                neighbors = torch.where(mask[i])[0]
                neighbor_list.append(neighbors)
                dist_list.append(distances[i][neighbors])
            
            # 找到最大邻居数并padding
            max_neighbors = max(len(n) for n in neighbor_list) if neighbor_list else 0
            if max_neighbors == 0:
                return torch.empty(N, 0, dtype=torch.long, device=device), torch.empty(N, 0, device=device)
            
            neighbor_indices = torch.full((N, max_neighbors), -1, dtype=torch.long, device=device)
            neighbor_distances = torch.full((N, max_neighbors), float('inf'), device=device, dtype=distances.dtype)
            
            for i, (neighbors, dists) in enumerate(zip(neighbor_list, dist_list)):
                n = len(neighbors)
                if n > 0:
                    neighbor_indices[i, :n] = neighbors
                    neighbor_distances[i, :n] = dists
            
            return neighbor_indices, neighbor_distances
        else:
            # 默认：返回所有体素
            return torch.arange(N, device=device).unsqueeze(0).expand(N, -1), torch.zeros(N, N, device=device)


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
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined


if __name__ == "__main__":
    """
    测试 _process_global_attention 函数
    模拟第 668-671 行的调用场景
    """
    import torch
    import sys
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建 Aggregator 实例用于测试
    print("Creating aggregator instance...")
    aggregator = Aggregator(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        voxel_size=0.05,
        radius_stage1=2.5,
        radius_stage2=3.5,
        temporal_window=3,
        enable_voxelization=True,
    ).to(device)
    aggregator.eval()
    
    # 模拟输入参数（根据第 668-671 行的调用）
    B = 1
    T = 6  # 时间步数
    V = 4  # 视角数
    S = T * V  # 原始序列长度 = 24
    S_voxel = T  # 体素化路径下的序列长度 = 6
    P = 506  # max_num_voxels 或 patch 数
    C = 1024  # embed_dim
    global_idx = 0
    num_block = 0  # 在 temporal_list1 中
    
    # 创建模拟的 tokens
    # 模拟体素化路径：tokens 形状可能是 [B, S*P, C] 或 [B, N_actual, C]
    # 根据错误信息，实际大小是 3108864 = 1 * 3036 * 1024
    N_actual = 3036  # 实际 token 数
    tokens = torch.randn(B, N_actual, C, device=device)
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Expected shape: [B={B}, S*P={S_voxel*P}, C={C}] = [{B}, {S_voxel*P}, {C}]")
    print(f"Actual size: {tokens.numel()}")
    
    # 创建模拟的 pos
    pos = None  # 可以设置为 None 或创建实际的 pos
    
    # 创建模拟的 voxel_data
    max_num_voxels = P
    voxel_data = {
        'use_voxel_tokens': True,
        'max_num_voxels': max_num_voxels,
        'voxel_xyz_list': [
            [torch.randn(100, 3, device=device) for _ in range(B)]  # 每个时间步的体素坐标
            for _ in range(T)
        ],
        'voxel_ids_list': [
            [torch.randint(0, 1000, (100,), device=device) for _ in range(B)]  # 每个时间步的体素ID
            for _ in range(T)
        ],
    }
    
    # 调用 _process_global_attention（模拟第 668-671 行）
    print("\n" + "="*50)
    print("Testing _process_global_attention")
    print("="*50)
    print(f"Input parameters:")
    print(f"  tokens.shape: {tokens.shape}")
    print(f"  B={B}, S={S_voxel}, P={P}, C={C}")
    print(f"  global_idx={global_idx}, num_block={num_block}")
    print(f"  is_multi_view=True, T={T}, V={V}")
    print(f"  voxel_data['use_voxel_tokens']={voxel_data['use_voxel_tokens']}")
    print(f"  voxel_data['max_num_voxels']={voxel_data['max_num_voxels']}")
    
    tokens_out, global_idx_out, global_intermediates = aggregator._process_global_attention(
        tokens, B, S_voxel, P, C, global_idx, 
        pos=pos, 
        is_multi_view=True, 
        T=T, 
        V=V,
        voxel_data=voxel_data, 
        num_block=num_block
    )
    
    print("\n" + "="*50)
    print("Test PASSED!")
    print("="*50)
    print(f"Output tokens shape: {tokens_out.shape}")
    print(f"Output global_idx: {global_idx_out}")
    print(f"Number of intermediates: {len(global_intermediates)}")
    if len(global_intermediates) > 0:
        print(f"First intermediate shape: {global_intermediates[0].shape}")
        

