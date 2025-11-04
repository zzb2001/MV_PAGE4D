# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# VGGT DPT GS Head for Gaussian Splatting parameters prediction
# Based on AnySplat implementation, adapted for mv_page4d_lite

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dpt_head import DPTHead


class VGGT_DPT_GS_Head(DPTHead):
    """
    VGGT DPT Head for Gaussian Splatting parameter prediction.
    
    This head predicts raw Gaussian parameters (opacity, scales, rotations, SH coefficients)
    from multi-scale encoder tokens using Dense Prediction Transformer architecture.
    """
    
    def __init__(self, 
            dim_in: int,
            patch_size: int = 14,
            output_dim: int = 83,
            activation: str = "inv_log",
            conf_activation: str = "expp1",
            features: int = 256,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            feature_only: bool = False,
            down_ratio: int = 1,
    ):
        super().__init__(
            dim_in, patch_size, output_dim, activation, conf_activation, 
            features, out_channels, intermediate_layer_idx, pos_embed, feature_only, down_ratio
        )
        
        # Additional layers for GS parameter prediction
        head_features_1 = 128
        head_features_2 = 128 if output_dim > 50 else 32  # sh=0: 32, sh=4: 128
        
        # Image feature merger: directly processes input RGB images
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, head_features_2, 7, 1, 3),
            nn.ReLU(),
        )
        
        # Final output convolution
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, encoder_tokens: List[torch.Tensor], images, patch_start_idx: int = 5, 
                image_size=None, conf=None, frames_chunk_size: int = 8,
                is_multi_view: bool = False, T=None, V=None):
        """
        Forward pass for GS parameter prediction.
        
        Args:
            encoder_tokens: List of tensors from encoder, each [B*S, N, C]
            images: Input images [B, S, 3, H, W] or [B, T, V, 3, H, W]
            patch_start_idx: Starting index of patch tokens (skip special tokens)
            image_size: (H, W) tuple
            conf: Optional confidence map
            frames_chunk_size: Chunk size for memory-efficient processing
            is_multi_view: Whether input is multi-view format
            T: Number of time steps (if multi-view)
            V: Number of views (if multi-view)
        
        Returns:
            torch.Tensor: [B, S, output_dim, H, W] or [B, T, V, output_dim, H, W]
        """
        # Handle multi-view format
        if is_multi_view and T is not None and V is not None:
            # Reshape from [B, T, V, 3, H, W] to [B, T*V, 3, H, W]
            B, T_in, V_in, C, H, W = images.shape
            images = images.view(B, T_in * V_in, C, H, W)
            S = T_in * V_in
        else:
            if len(images.shape) == 4:
                images = images.unsqueeze(0)
            B, S, C, H, W = images.shape
        
        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            output = self._forward_impl(encoder_tokens, images, patch_start_idx, B, S, H, W)
        else:
            # Process frames in chunks to manage memory usage
            assert frames_chunk_size > 0
            all_preds = []
            for frames_start_idx in range(0, S, frames_chunk_size):
                frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
                chunk_output = self._forward_impl(
                    encoder_tokens, images, patch_start_idx, B, S, H, W,
                    frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            output = torch.cat(all_preds, dim=1)
        
        # Reshape back to multi-view format if needed
        if is_multi_view and T is not None and V is not None:
            # output: [B, T*V, output_dim, H, W] -> [B, T, V, output_dim, H, W]
            output = output.view(B, T, V, output.shape[-3], H, W)
        
        return output
    
    def _forward_impl(self, encoder_tokens: List[torch.Tensor], images, patch_start_idx: int,
                     B: int, S: int, H: int, W: int,
                     frames_start_idx: int = None, frames_end_idx: int = None):
        """
        Internal forward implementation.
        
        Args:
            encoder_tokens: List of encoder token tensors
            images: Input images [B, S, 3, H, W]
            patch_start_idx: Starting index of patch tokens
            B, S, H, W: Batch, sequence, height, width dimensions
            frames_start_idx: Optional start index for chunk processing
            frames_end_idx: Optional end index for chunk processing
        
        Returns:
            torch.Tensor: [B, S_chunk, output_dim, H, W]
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx]
            S_chunk = frames_end_idx - frames_start_idx
        else:
            S_chunk = S
        
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            # Extract tokens from the specified layer
            if len(encoder_tokens) > 10:
                # If encoder_tokens is indexed by layer number
                x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                # If encoder_tokens is a list indexed by position
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = encoder_tokens[list_idx][:, :, patch_start_idx:]
            
            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()
            
            # Reshape: [B*S, N, C] -> [B*S, C, patch_h, patch_w]
            x = x.view(B * S_chunk, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            # Project to feature space
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1
        
        # Fuse features from multiple layers using DPT refinement
        out = self.scratch_forward(out)
        
        # Merge with direct image features
        direct_img_feat = self.input_merger(images.flatten(0, 1))  # [B*S_chunk, head_features_2, H, W]
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        out = out + direct_img_feat  # Residual connection
        
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
        
        # Final output convolution
        out = self.scratch.output_conv2(out)
        out = out.view(B, S_chunk, *out.shape[1:])  # [B, S_chunk, output_dim, H, W]
        
        return out

