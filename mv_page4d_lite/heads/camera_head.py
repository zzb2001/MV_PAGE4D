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


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        
        # Add view-specific projection layer (shim) for multi-view differentiation
        # This allows the model to learn view-specific transformations before the frozen trunk
        self.view_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in, dim_in // 4),  # Compress
                nn.GELU(),
                nn.Linear(dim_in // 4, dim_in)   # Expand back
            ) for _ in range(32)  # Max 32 views, but only uses V views
        ])
        # Initialize view_proj to be near-identity (small weights)
        for proj in self.view_proj:
            for m in proj:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4, is_multi_view: bool = False, T: int = None, V: int = None) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            is_multi_view (bool): Whether input is multi-view format [B, T*V, P, C]
            T (int): Number of time steps (multi-view mode)
            V (int): Number of views (multi-view mode)

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
                For multi-view: [B, V, 9] per iteration
                For legacy: [B, S, 9] per iteration (where S is typically 1)
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]  # [B, S, P, C] or [B, T*V, P, C]

        if is_multi_view and T is not None and V is not None:
            # Multi-view mode: extract camera tokens for each view
            # Camera tokens are at position 0 for each (t,v) pair
            # But we want [B, V] camera tokens (one per view, shared across time)
            # Strategy: extract Zc[v] by taking camera token from first time step for each view
            B = tokens.shape[0]
            # Ensure tokens are in correct format: [B, T*V, P, C]
            # Reshape: [B, T*V, P, C] -> [B, T, V, P, C]
            tokens_mv = tokens.reshape(B, T, V, tokens.shape[2], tokens.shape[3])
            # Extract camera tokens from first time step: [B, V, C]
            pose_tokens_mv = tokens_mv[:, 0, :, 0, :]  # [B, V, C]
            
            # DEBUG: Check camera token differences before processing
            if torch.rand(1).item() < 0.01:  # Print 1% of the time to avoid spam
                view_diffs = []
                for v1 in range(min(V, 4)):  # Check first 4 views
                    for v2 in range(v1 + 1, min(V, 4)):
                        diff = (pose_tokens_mv[0, v1, :] - pose_tokens_mv[0, v2, :]).abs().mean().item()
                        view_diffs.append(diff)

            
            # Apply view-specific projection BEFORE normalization to preserve differences
            # This allows learning view-specific transformations
            # IMPORTANT: Use non-inplace operations to avoid gradient computation errors
            pose_tokens_projected = []
            for v_idx in range(V):
                if v_idx < len(self.view_proj):
                    proj_result = self.view_proj[v_idx](pose_tokens_mv[:, v_idx:v_idx+1, :])
                    pose_tokens_projected.append(proj_result)
                else:
                    pose_tokens_projected.append(pose_tokens_mv[:, v_idx:v_idx+1, :])
            # Concatenate all views back
            pose_tokens_mv = torch.cat(pose_tokens_projected, dim=1)  # [B, V, C]
            
            # token_norm expects [B, N, C] format, so ensure it's 3D
            if len(pose_tokens_mv.shape) == 2:
                pose_tokens_mv = pose_tokens_mv.unsqueeze(1)  # [B, V, C] -> [B, V, C] (already 3D, but ensure)
            pose_tokens_mv = self.token_norm(pose_tokens_mv)  # [B, V, C] -> [B, V, C] after norm
            
            # DEBUG: Check camera token differences after norm
            if torch.rand(1).item() < 0.01:
                view_diffs = []
                for v1 in range(min(V, 4)):
                    for v2 in range(v1 + 1, min(V, 4)):
                        diff = (pose_tokens_mv[0, v1, :] - pose_tokens_mv[0, v2, :]).abs().mean().item()
                        view_diffs.append(diff)
                if len(view_diffs) > 0:
                    avg_diff = sum(view_diffs) / len(view_diffs)
                    print(f"[DEBUG] Camera token differences (after norm): avg={avg_diff:.6f}")
            
            # For each view v, call the frozen camera head
            pred_pose_enc_list_all_views = []
            for v_idx in range(V):
                pose_tokens_v = pose_tokens_mv[:, v_idx:v_idx+1, :]  # [B, 1, C]
                # Ensure shape is exactly [B, 1, C] before passing to trunk_fn
                if len(pose_tokens_v.shape) != 3:
                    raise ValueError(f"Expected pose_tokens_v to be 3D [B, 1, C], got shape {pose_tokens_v.shape}")
                pred_pose_enc_list_v = self.trunk_fn(pose_tokens_v, num_iterations)
                
                # DEBUG: Check output differences
                if torch.rand(1).item() < 0.01 and len(pred_pose_enc_list_v) > 0:
                    if v_idx == 0:
                        print(f"[DEBUG] Camera head output for view {v_idx}: shape={pred_pose_enc_list_v[-1].shape}, sample={pred_pose_enc_list_v[-1][0, 0, :3].detach().cpu().numpy()}")
                
                # Stack results: each item in list is [B, 1, 9], we want [B, V, 9] across all iterations
                # IMPORTANT: Use non-inplace operations to avoid gradient computation errors
                if v_idx == 0:
                    # Initialize the list with all views filled with the first view's predictions
                    pred_pose_enc_list_all_views = []
                    for iter_idx, pred_enc in enumerate(pred_pose_enc_list_v):
                        # Create a tensor that will be filled in subsequent iterations
                        pred_pose_all_views = pred_enc.expand(B, V, -1).clone()  # [B, V, 9]
                        pred_pose_enc_list_all_views.append(pred_pose_all_views)
                else:
                    # Replace the corresponding view in each iteration using torch.cat (fully non-inplace)
                    for iter_idx, pred_enc_v in enumerate(pred_pose_enc_list_v):
                        # Get the current tensor
                        pred_pose_all_views = pred_pose_enc_list_all_views[iter_idx]  # [B, V, 9]
                        # Split into parts: before v_idx, at v_idx, after v_idx
                        if v_idx > 0:
                            before_view = pred_pose_all_views[:, :v_idx, :]  # [B, v_idx, 9]
                        else:
                            before_view = None
                        after_view = pred_pose_all_views[:, v_idx+1:, :]  # [B, V-v_idx-1, 9]
                        # Reconstruct with new view inserted
                        if before_view is not None:
                            pred_pose_all_views_new = torch.cat([before_view, pred_enc_v, after_view], dim=1)
                        else:
                            pred_pose_all_views_new = torch.cat([pred_enc_v, after_view], dim=1)
                        pred_pose_enc_list_all_views[iter_idx] = pred_pose_all_views_new
            
            return pred_pose_enc_list_all_views
        else:
            # Legacy mode: extract camera tokens as before
            pose_tokens = tokens[:, :, 0]  # [B, S, C]
            pose_tokens = self.token_norm(pose_tokens)

            pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
            return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        # Ensure pose_tokens is 3D [B, S, C] where S is typically 1
        if len(pose_tokens.shape) == 2:
            pose_tokens = pose_tokens.unsqueeze(1)  # [B, C] -> [B, 1, C]
        elif len(pose_tokens.shape) > 3:
            # Flatten extra dimensions
            B = pose_tokens.shape[0]
            C = pose_tokens.shape[-1]
            pose_tokens = pose_tokens.reshape(B, -1, C)
        
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
