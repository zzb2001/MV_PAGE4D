# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Voxel-level Gaussian Splatting Head
# Predicts Gaussian parameters directly from voxel tokens (not pixel-level features)

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelGaussianHead(nn.Module):
    """
    Voxel-level Gaussian Splatting parameter prediction head.
    
    This head predicts Gaussian parameters directly from voxel tokens,
    outputting per-voxel Gaussian parameters: {Δx, σ, q, SH, α}
    
    Output format:
    - Δx (3): Offset from voxel center, clamped to [-voxel_size/2, voxel_size/2]
    - σ (3): Scale parameters (3D)
    - q (4): Rotation quaternion
    - SH (48): Spherical harmonics coefficients (16*3 for RGB)
    - α (1): Opacity
    Total: 83 dimensions
    """
    
    def __init__(self, dim_in: int, output_dim: int = 83):
        """
        Args:
            dim_in: Input dimension of voxel tokens (typically 2*embed_dim from aggregator)
            output_dim: Output dimension (default 83: 3+3+4+48+1)
        """
        super().__init__()
        self.dim_in = dim_in
        self.output_dim = output_dim
        
        # MLP for predicting Gaussian parameters from voxel tokens
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_in * 2),
            nn.ReLU(),
            nn.Linear(dim_in * 2, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, output_dim)
        )
        
        # Initialize output layers
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize output layers with small values for stability
        # Opacity should start small (near transparent)
        with torch.no_grad():
            # Find the opacity output layer (last linear layer)
            last_layer = self.mlp[-1]
            if isinstance(last_layer, nn.Linear):
                # Initialize opacity (last dim) to small value
                last_layer.bias[-1].fill_(-2.0)  # sigmoid(-2.0) ≈ 0.12 (low opacity)
    
    def forward(
        self,
        voxel_tokens: torch.Tensor,
        voxel_xyz: Optional[torch.Tensor] = None,
        voxel_size: Optional[float] = None
    ) -> dict:
        """
        Forward pass for voxel-level Gaussian parameter prediction.
        
        Args:
            voxel_tokens: Voxel tokens from aggregator
                - Format 1: [B, T, N_voxels, C] (most common)
                - Format 2: [B*T, N_voxels, C] or [B, N_voxels, C]
            voxel_xyz: Optional voxel center coordinates [B, T, N_voxels, 3] or [B, N_voxels, 3]
                Used for clamping Δx offset
            voxel_size: Optional voxel size for clamping Δx
        
        Returns:
            dict containing:
                - 'gaussian_params': [B, T, N_voxels, 83] or [B, N_voxels, 83]
                - 'gaussian_xyz': [B, T, N_voxels, 3] or [B, N_voxels, 3] (if voxel_xyz provided)
                - 'delta_x': [B, T, N_voxels, 3] (offset from voxel center)
                - 'scales': [B, T, N_voxels, 3] (σ)
                - 'rotations': [B, T, N_voxels, 4] (q, quaternion)
                - 'sh_coeffs': [B, T, N_voxels, 48] (spherical harmonics)
                - 'opacity': [B, T, N_voxels, 1] (α)
        """
        # Handle different input shapes
        original_shape = voxel_tokens.shape
        if len(original_shape) == 4:
            B, T, N, C = original_shape
            voxel_tokens_flat = voxel_tokens.reshape(B * T, N, C)
            needs_reshape = True
        elif len(original_shape) == 3:
            if original_shape[0] == B * T:
                B = original_shape[0] // T if T is not None else original_shape[0]
                T = T if T is not None else 1
            else:
                B, N, C = original_shape
                T = 1
            voxel_tokens_flat = voxel_tokens.reshape(B * T, N, C)
            needs_reshape = True
        else:
            # Assume [B*T*N, C] or [N, C]
            voxel_tokens_flat = voxel_tokens
            needs_reshape = False
        
        # Predict Gaussian parameters
        # voxel_tokens_flat: [B*T, N, C] or [..., C]
        gaussian_params_flat = self.mlp(voxel_tokens_flat)  # [B*T, N, 83] or [..., 83]
        
        # Reshape back to original format if needed
        if needs_reshape and len(original_shape) >= 3:
            if len(original_shape) == 4:
                gaussian_params = gaussian_params_flat.reshape(B, T, N, self.output_dim)
            else:
                gaussian_params = gaussian_params_flat.reshape(B, N, self.output_dim)
        else:
            gaussian_params = gaussian_params_flat
        
        # Parse Gaussian parameters into components
        delta_x = gaussian_params[..., 0:3]  # [..., 3]
        scales = gaussian_params[..., 3:6]  # [..., 3] σ
        rotations = gaussian_params[..., 6:10]  # [..., 4] q (quaternion)
        sh_coeffs = gaussian_params[..., 10:58]  # [..., 48] SH coefficients
        opacity = gaussian_params[..., 58:59]  # [..., 1] α
        
        # Apply activations
        scales = F.softplus(scales) + 1e-6  # σ > 0
        opacity = torch.sigmoid(opacity)  # α ∈ [0, 1]
        
        # Normalize quaternion
        rotations = F.normalize(rotations, p=2, dim=-1)
        
        # Clamp Δx to voxel_size if provided
        if voxel_size is not None:
            delta_x = torch.clamp(delta_x, -voxel_size / 2, voxel_size / 2)
        
        # Compute final Gaussian positions (voxel center + offset)
        gaussian_xyz = None
        if voxel_xyz is not None:
            # Handle different voxel_xyz shapes
            if len(voxel_xyz.shape) == 4:  # [B, T, N, 3]
                if len(gaussian_params.shape) == 4:  # [B, T, N, 83]
                    gaussian_xyz = voxel_xyz + delta_x
                else:
                    # Reshape delta_x to match
                    delta_x_reshaped = delta_x.reshape(voxel_xyz.shape)
                    gaussian_xyz = voxel_xyz + delta_x_reshaped
            elif len(voxel_xyz.shape) == 3:  # [B, N, 3]
                if len(gaussian_params.shape) == 3:  # [B, N, 83]
                    gaussian_xyz = voxel_xyz + delta_x
                else:
                    delta_x_reshaped = delta_x.reshape(voxel_xyz.shape)
                    gaussian_xyz = voxel_xyz + delta_x_reshaped
            else:
                # Try to match shapes
                gaussian_xyz = voxel_xyz + delta_x
        
        return {
            'gaussian_params': gaussian_params,
            'gaussian_xyz': gaussian_xyz,
            'delta_x': delta_x,
            'scales': scales,
            'rotations': rotations,
            'sh_coeffs': sh_coeffs,
            'opacity': opacity,
        }


