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

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

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

    def forward(self, images: torch.Tensor, temporal_features: torch.Tensor = None) -> Tuple[List[torch.Tensor], int, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, T, V, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size
                T: time steps (multi-view mode) or S: sequence length (legacy mode)
                V: number of views (multi-view mode only)
                3: RGB channels
                H: height, W: width
        Returns:
            (list[torch.Tensor], int, torch.Tensor):
            The list of outputs from the attention blocks,
            the patch_start_idx indicating where patch tokens begin,
            and mask_logits [B*S, 1, H, W] or None if mask is not generated.
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
        
        if is_multi_view:
            # Reshape to [B, T, V, P, C]
            # Use reshape to handle potential non-contiguous tensors
            patch_tokens_mv = patch_tokens.reshape(B, T, V, P, C)
            
            # Apply ViewMixer (Stage-0): cross-view attention within each time step
            patch_tokens_mv = self.viewmixer(patch_tokens_mv)
            
            # Flatten back to [B*S, P, C] where S = T*V
            patch_tokens = patch_tokens_mv.reshape(B * S, P, C)
            
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
            # Legacy mode: use original slice_expand_and_flatten
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)  # [B*S, 1+num_register_tokens+P, C]
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images_flat.device)
        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images_flat.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        # Add time and view embeddings (for multi-view mode)
        if is_multi_view:
            # Time embedding: E_time[t] for each (t,v) pair
            time_ids = torch.arange(T, device=images_flat.device).repeat_interleave(V).unsqueeze(0).expand(B, -1)  # [B, T*V]
            time_emb_raw = self.time_embed(time_ids)  # [B, T*V, C]
            # Ensure correct shape before reshape
            assert time_emb_raw.shape == (B, T * V, C), f"time_emb_raw shape mismatch: got {time_emb_raw.shape}, expected ({B}, {T * V}, {C})"
            # Reshape: [B, T*V, C] -> [B*S, 1, C] where S = T*V
            time_emb = time_emb_raw.contiguous().reshape(B * S, 1, C)  # [B*S, 1, C]
            assert time_emb.shape == (B * S, 1, C), f"time_emb shape mismatch: got {time_emb.shape}, expected ({B * S}, 1, {C})"
            
            # View embedding: E_view[v] for each (t,v) pair
            view_ids = torch.arange(V, device=images_flat.device).repeat(T).unsqueeze(0).expand(B, -1)  # [B, T*V]
            view_emb_raw = self.view_embed(view_ids)  # [B, T*V, C]
            # Ensure correct shape before reshape
            assert view_emb_raw.shape == (B, T * V, C), f"view_emb_raw shape mismatch: got {view_emb_raw.shape}, expected ({B}, {T * V}, {C})"
            # Reshape: [B, T*V, C] -> [B*S, 1, C]
            view_emb = view_emb_raw.contiguous().reshape(B * S, 1, C)  # [B*S, 1, C]
            assert view_emb.shape == (B * S, 1, C), f"view_emb shape mismatch: got {view_emb.shape}, expected ({B * S}, 1, {C})"
            
            # Camera parameter embedding (simplified: use view_id)
            view_ids_norm = view_ids.float().unsqueeze(-1) / max(V - 1, 1)  # [B, T*V, 1]
            cam_emb_raw = self.camera_param_embed(view_ids_norm)  # [B, T*V, C]
            assert cam_emb_raw.shape == (B, T * V, C), f"cam_emb_raw shape mismatch: got {cam_emb_raw.shape}, expected ({B}, {T * V}, {C})"
            # Reshape: [B, T*V, C] -> [B*S, 1, C]
            cam_emb = cam_emb_raw.contiguous().reshape(B * S, 1, C)  # [B*S, 1, C]
            assert cam_emb.shape == (B * S, 1, C), f"cam_emb shape mismatch: got {cam_emb.shape}, expected ({B * S}, 1, {C})"
            
            # Add embeddings to tokens (additive injection)
            # IMPORTANT: Use non-inplace operations to avoid gradient computation errors
            # Add to camera token position (first token)
            # IMPORTANT: Add both cam_emb (camera parameter embedding) and view_emb (view embedding)
            # to camera tokens to ensure stronger differentiation between views
            camera_tokens_with_emb = tokens[:, 0:1, :] + cam_emb + view_emb
            # Add to patch tokens (after special tokens)
            # view_emb and time_emb are [B*S, 1, C], they will broadcast along the patch dimension
            # Ensure shapes are correct for broadcasting: tokens[:, patch_start_idx:, :] is [B*S, P-patch_start_idx, C]
            patch_tokens_slice = tokens[:, self.patch_start_idx:, :]  # [B*S, P-patch_start_idx, C]
            # view_emb and time_emb: [B*S, 1, C] will broadcast to [B*S, P-patch_start_idx, C]
            patch_tokens_with_emb = patch_tokens_slice + view_emb + time_emb
            # Reconstruct tokens tensor with embeddings added
            tokens = torch.cat([camera_tokens_with_emb, tokens[:, 1:self.patch_start_idx, :], patch_tokens_with_emb], dim=1)
        # update P because we added special tokens
        _, P, C = tokens.shape
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
                        # Stage-1: Frame attention over all views at same time (multi-view routing)
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos, 
                            is_multi_view=is_multi_view, T=T, V=V, stage="stage1")
                    elif num_block in list(range(8, 18)):  # Stage-2: layers 8-17
                        # Stage-2: Frame attention over same view across time (temporal routing)
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos,
                            is_multi_view=is_multi_view, T=T, V=V, stage="stage2")
                    elif num_block in list(range(18, 24)):  # Stage-3: layers 18-23
                        # Stage-3: Alternate between temporal and multi-view routing
                        # Use temporal for first half, multi-view for second half
                        use_temporal = (num_block - 18) < 3  # first 3 layers: temporal, last 3: multi-view
                        stage_strategy = "stage2" if use_temporal else "stage1"
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos,
                            is_multi_view=is_multi_view, T=T, V=V, stage=stage_strategy)
                    else:
                        # Fallback
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos,
                            is_multi_view=is_multi_view, T=T, V=V, stage="legacy")
                    
                    if num_block in self.temporal_list1_mask:
                        # Extract mask logits from spatial_mask_head for supervision loss
                        # We need to access m_logit which is computed internally
                        # Workaround: call the head and extract logits using a custom forward
                        tokens_for_mask = tokens.detach().clone().reshape(B, S, P, C)
                        mask_h = H // self.patch_size
                        mask_w = W // self.patch_size
                        
                        # Manually compute mask logits (mimicking SpatialMaskHead_IMP.forward)
                        # This is needed because SpatialMaskHead_IMP doesn't return m_logit
                        xs = tokens_for_mask.view(B * S, P, C)[:, self.patch_start_idx:, :]  # (B*S, H*W, C)
                        h0 = xs.transpose(1, 2).reshape(B * S, C, mask_h, mask_w)
                        m_logit = self.spatial_mask_head.head0(h0)  # (B*S, 1, H, W)
                        
                        # Now call the full forward to get the outputs we need
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
                            tokens, B, S, P, C, global_idx, pos=pos, is_multi_view=is_multi_view, T=T, V=V)
                    elif num_block in self.temporal_list2:
                        # Apply mask: weak for camera branch, strong for register branch
                        # For simplicity, apply camera gate to mask
                        if is_multi_view and hasattr(self, 'camera_mask_gate'):
                            gamma = torch.sigmoid(self.camera_mask_gate)
                            cache_mask_applied = gamma * cache_mask  # weak masking
                        else:
                            cache_mask_applied = cache_mask
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, 
                            attn_mask=cache_mask_applied, attn_value=cached_value,
                            is_multi_view=is_multi_view, T=T, V=V)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)    #24层[1, 24, 782, 1024*2] or [1, T*V, P, 2048]
        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx, mask_logits

    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos=None, attn_mask=None, attn_value=None,
        is_multi_view=False, T=None, V=None, stage="legacy"
    ):
        """
        Process frame attention blocks with multi-view routing support.
        
        Args:
            tokens: [B*S, P, C] token tensor
            stage: "stage1" (multi-view routing), "stage2" (temporal routing), "legacy" (original)
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)
        
        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            block = self.frame_blocks[frame_idx]
            
            # Apply routing strategy for multi-view mode
            if is_multi_view and stage in ["stage1", "stage2"]:
                # Reshape tokens: [B*S, P, C] -> [B, S, P, C] -> [B, T, V, P, C]
                tokens_reshaped = tokens.reshape(B, S, P, C)
                if stage == "stage1":
                    # Stage-1: Frame attention over all views at same time
                    # Reshape: [B, T, V, P, C] -> [B*T, V*P, C]
                    tokens_reshaped = tokens_reshaped.reshape(B, T, V, P, C)
                    tokens_for_attn = tokens_reshaped.reshape(B * T, V * P, C)
                    # Also reshape positions
                    if pos is not None:
                        pos_reshaped = pos.reshape(B, S, P, 2)
                        pos_for_attn = pos_reshaped.reshape(B, T, V, P, 2).reshape(B * T, V * P, 2)
                    else:
                        pos_for_attn = None
                elif stage == "stage2":
                    # Stage-2: Frame attention over same view across time
                    # Reshape: [B, T, V, P, C] -> [B*V, T*P, C]
                    tokens_reshaped = tokens_reshaped.reshape(B, T, V, P, C)
                    tokens_for_attn = tokens_reshaped.permute(0, 2, 1, 3, 4).contiguous().reshape(B * V, T * P, C)
                    # Also reshape positions
                    if pos is not None:
                        pos_reshaped = pos.reshape(B, S, P, 2)
                        pos_for_attn = pos_reshaped.reshape(B, T, V, P, 2).permute(0, 2, 1, 3, 4).contiguous().reshape(B * V, T * P, 2)
                    else:
                        pos_for_attn = None
                else:
                    tokens_for_attn = tokens
                    pos_for_attn = pos
                
                # Apply attention block
                if self.training:
                    fn = functools.partial(block, attn_mask=attn_mask, attn_value=attn_value)
                    tokens_out = torch.utils.checkpoint.checkpoint(
                        fn, tokens_for_attn, pos_for_attn, use_reentrant=self.use_reentrant)
                else:
                    tokens_out = block(tokens_for_attn, pos=pos_for_attn, attn_mask=attn_mask, attn_value=attn_value)
                
                # Reshape back: restore original shape
                if stage == "stage1":
                    tokens_out = tokens_out.reshape(B, T, V * P, C).reshape(B, T, V, P, C).reshape(B, S, P, C).reshape(B * S, P, C)
                elif stage == "stage2":
                    tokens_out = tokens_out.reshape(B, V, T * P, C).reshape(B, V, T, P, C).permute(0, 2, 1, 3, 4).contiguous().reshape(B, S, P, C).reshape(B * S, P, C)
                
                tokens = tokens_out
            else:
                # Legacy mode: no routing change
                if self.training:
                    fn = functools.partial(block, attn_mask=attn_mask, attn_value=attn_value)
                    tokens = torch.utils.checkpoint.checkpoint(
                        fn, tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = block(tokens, pos=pos, attn_mask=attn_mask, attn_value=attn_value)
            
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))
        
        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self, tokens, B, S, P, C, global_idx, pos=None, temporal_features=None, 
        attn_mask=None, attn_value=None, is_multi_view=False, T=None, V=None
    ):
        """ Process global attention blocks. We keep tokens in shape (B, S*P, C). """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
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
            intermediates.append(tokens.reshape(B, S, P, C))
        return tokens, global_idx, intermediates


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
