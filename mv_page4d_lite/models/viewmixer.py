# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Linear layer.
    W' = W + BA, where A and B are low-rank matrices.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize: A with small random values, B with zeros (so output starts near zero)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features]
        Returns:
            [..., out_features]
        """
        # LoRA: delta = (x @ A^T) @ B^T
        lora_output = (x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return self.alpha * lora_output


class ViewMixer(nn.Module):
    """
    Lightweight cross-view attention module (Stage-0).
    Performs cross-view alignment/aggregation within the same time step t.
    
    Options:
    - Option A: Cross-View Attention with LoRA/gated residual (almost zero-intrusive)
    - Option B: Geometry-guided cross-view aggregation (more memory-efficient)
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        use_lora: bool = True,
        lora_rank: int = 16,
        use_geometry_guidance: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_lora = use_lora
        self.use_geometry_guidance = use_geometry_guidance
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        if use_lora:
            # Option A: LoRA-based QKV projections
            self.q_lora = LoRALinear(embed_dim, embed_dim, rank=lora_rank)
            self.k_lora = LoRALinear(embed_dim, embed_dim, rank=lora_rank)
            self.v_lora = LoRALinear(embed_dim, embed_dim, rank=lora_rank)
            
            # Gated residual connection
            self.gate = nn.Parameter(torch.zeros(embed_dim))
            
            # Output projection (also with LoRA)
            self.proj_lora = LoRALinear(embed_dim, embed_dim, rank=lora_rank)
        else:
            # Standard linear layers (narrow width for memory efficiency)
            narrow_dim = embed_dim // 4  # 256 for embed_dim=1024
            self.q_proj = nn.Linear(embed_dim, narrow_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, narrow_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, narrow_dim, bias=False)
            self.proj = nn.Linear(narrow_dim, embed_dim, bias=False)
            
            # Initialize to near-identity
            nn.init.zeros_(self.q_proj.weight)
            nn.init.zeros_(self.k_proj.weight)
            nn.init.zeros_(self.v_proj.weight)
            nn.init.zeros_(self.proj.weight)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        patch_tokens: torch.Tensor,
        view_ids: Optional[torch.Tensor] = None,
        camera_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-view attention within the same time step.
        
        Args:
            patch_tokens: [B, T, V, P, C] patch tokens
            view_ids: [B, V] or [V] view indices (optional, for view embedding)
            camera_params: Camera parameters for geometry guidance (optional)
        
        Returns:
            [B, T, V, P, C] aggregated patch tokens
        """
        B, T, V, P, C = patch_tokens.shape
        
        if self.use_geometry_guidance and camera_params is not None:
            # Option B: Geometry-guided aggregation (simplified implementation)
            # This would use camera extrinsics to perform ray casting and weighted pooling
            # For now, we fall back to attention-based approach
            return self._geometry_guided_aggregation(patch_tokens, camera_params)
        
        # Option A: Cross-view attention
        # Reshape to group by time: [B*T, V, P, C]
        # Use reshape instead of view to handle non-contiguous tensors
        tokens_t = patch_tokens.reshape(B * T, V, P, C)
        
        # Flatten patches: [B*T, V, P, C] -> [B*T, V*P, C]
        tokens_flat = tokens_t.reshape(B * T, V * P, C)
        
        if self.use_lora:
            # LoRA-based attention
            q = self.q_lora(tokens_flat)  # [B*T, V*P, C]
            k = self.k_lora(tokens_flat)  # [B*T, V*P, C]
            v = self.v_lora(tokens_flat)  # [B*T, V*P, C]
        else:
            # Standard narrow projection
            narrow_dim = self.q_proj.weight.shape[0]  # output dimension
            q = self.q_proj(tokens_flat)  # [B*T, V*P, narrow_dim]
            k = self.k_proj(tokens_flat)  # [B*T, V*P, narrow_dim]
            v = self.v_proj(tokens_flat)  # [B*T, V*P, narrow_dim]
        
        # Reshape for multi-head attention
        if self.use_lora:
            q = q.reshape(B * T, V * P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B*T, H, V*P, head_dim]
            k = k.reshape(B * T, V * P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B * T, V * P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            num_heads_effective = min(self.num_heads, narrow_dim // 4)
            head_dim_effective = narrow_dim // num_heads_effective
            q = q.reshape(B * T, V * P, num_heads_effective, head_dim_effective).permute(0, 2, 1, 3)
            k = k.reshape(B * T, V * P, num_heads_effective, head_dim_effective).permute(0, 2, 1, 3)
            v = v.reshape(B * T, V * P, num_heads_effective, head_dim_effective).permute(0, 2, 1, 3)
        
        # Compute attention: [B*T, H, V*P, V*P]
        if self.use_lora:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        else:
            scale_effective = head_dim_effective ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale_effective
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v)  # [B*T, H, V*P, head_dim]
        
        # Reshape back
        if self.use_lora:
            out = out.permute(0, 2, 1, 3).contiguous().reshape(B * T, V * P, C)
            # Apply output projection
            out = self.proj_lora(out)
        else:
            out = out.permute(0, 2, 1, 3).contiguous().reshape(B * T, V * P, narrow_dim)
            out = self.proj(out)  # [B*T, V*P, C]
        
        # Gated residual (for LoRA)
        if self.use_lora:
            gate_weights = torch.sigmoid(self.gate)
            out = gate_weights * out + (1 - gate_weights) * tokens_flat
        
        # Layer norm
        out = self.norm(out)  # [B*T, V*P, C]
        
        # Reshape back to [B, T, V, P, C]
        output = out.reshape(B, T, V, P, C)
        return output
    
    def _geometry_guided_aggregation(
        self,
        patch_tokens: torch.Tensor,
        camera_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Option B: Geometry-guided cross-view aggregation.
        Simplified version: for now, just return the input (to be implemented with proper ray casting).
        """
        # TODO: Implement geometry-guided aggregation using camera extrinsics
        # This would involve:
        # 1. Projecting rays from reference view to other views
        # 2. Sampling features from small neighborhoods
        # 3. Weighted pooling based on geometry
        return patch_tokens

