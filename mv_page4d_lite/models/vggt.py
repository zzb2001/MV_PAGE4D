# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29503 /workspace/code/12_4d/VGGT-4D_T/training/launch.py --config training_29.yaml
import pdb
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from mv_page4d_lite.models.aggregator import Aggregator
from mv_page4d_lite.heads.camera_head import CameraHead
from mv_page4d_lite.heads.dpt_head import DPTHead
from mv_page4d_lite.heads.track_head import TrackHead

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, temporal_features: torch.Tensor = None):
        """
        Forward pass of the VGGT model.
        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W], [B, S, 3, H, W], or [B, T, V, 3, H, W], in range [0, 1].
                B: batch size
                S: sequence length (legacy mode)
                T: time steps (multi-view mode)
                V: number of views (multi-view mode)
                3: RGB channels
                H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (legacy) or [B, V, 9] (multi-view)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1] or [B, T, V, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions
                - world_points (torch.Tensor): 3D world coordinates for each pixel
                - world_points_conf (torch.Tensor): Confidence scores for world points
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # Detect input format
        is_multi_view = len(images.shape) == 6
        
        # If without batch dimension, add it
        if is_multi_view:
            if len(images.shape) == 5:
                images = images.unsqueeze(0)
            B, T, V, C, H, W = images.shape
        else:
            if len(images.shape) == 4:
                images = images.unsqueeze(0)
            B, S, C, H, W = images.shape
            T = S
            V = 1
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # images: [B, S, 3, H, W] or [B, T, V, 3, H, W]
        # temporal_features: [B, S] or [B, T*V]
        aggregated_tokens_list, patch_start_idx, mask_logits = self.aggregator(images=images, temporal_features=temporal_features)
        predictions = {}
        
        # Add mask_logits to predictions for supervision loss (if available)
        if mask_logits is not None:
            # mask_logits shape: [B*S, 1, H, W] where S = T*V for multi-view or S for legacy
            # Convert to multi-view format if needed
            if is_multi_view:
                # Reshape from [B*T*V, 1, H, W] to [B, T, V, H, W]
                B_total, _, H_mask, W_mask = mask_logits.shape
                mask_logits = mask_logits.squeeze(1)  # [B*T*V, H, W]
                mask_logits = mask_logits.view(B, T, V, H_mask, W_mask)  # [B, T, V, H, W]
            else:
                # Legacy format: [B*S, 1, H, W] -> [B, S, H, W]
                B_total, _, H_mask, W_mask = mask_logits.shape
                mask_logits = mask_logits.squeeze(1)  # [B*S, H, W]
                mask_logits = mask_logits.view(B, S, H_mask, W_mask)  # [B, S, H, W]
            
            predictions["mask_logits"] = mask_logits

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(
                    aggregated_tokens_list,
                    is_multi_view=is_multi_view,
                    T=T if is_multi_view else None,
                    V=V if is_multi_view else None
                )
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                    is_multi_view=is_multi_view, T=T if is_multi_view else None, V=V if is_multi_view else None
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                    is_multi_view=is_multi_view, T=T if is_multi_view else None, V=V if is_multi_view else None
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points,
                is_multi_view=is_multi_view, T=T if is_multi_view else None, V=V if is_multi_view else None
            )
            predictions["track"] = track_list[-1]  # track of the last iteration for inference
            predictions["track_list"] = track_list  # all iterations for training loss
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
