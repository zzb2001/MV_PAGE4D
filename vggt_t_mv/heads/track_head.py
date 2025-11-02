# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor
import pdb

class TrackHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=2,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        )

        # Tracker module that predicts point trajectories
        # Takes feature maps and predicts coordinates and visibility
        self.tracker = BaseTrackerPredictor(
            latent_dim=features,  # Match the output_dim of feature extractor
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) or (B, T, N, C, H, W) where:
                                   B = batch size, S = sequence length (or T*N for multi-view).
                                   T = time frames, N = number of views.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                    Shape: (B, S, ...) or (B, T, N, ...) depending on input format.
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
        """
        # 检测输入格式：支持 [B, S, C, H, W] 或 [B, T, N, C, H, W]
        is_multi_view = len(images.shape) == 6
        
        if is_multi_view:
            # 多视角时序格式: [B, T, N, C, H, W]
            B, T, N, C, H, W = images.shape
            S = T * N
        else:
            # 单视角时序格式: [B, S, C, H, W]
            B, S, _, H, W = images.shape

        # Extract features from tokens
        # feature_maps has shape (B, S, C, H//2, W//2) or (B, T, N, C, H//2, W//2) due to down_ratio=2
        # feature_extractor (DPTHead) 已经支持多视角格式，会自动处理
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        # Use default iterations if not specified
        if iters is None:
            iters = self.iters

        # BaseTrackerPredictor 期望 fmaps 格式为 [B, S, C, H, W]
        # 如果是多视角格式，需要转换为单序列格式
        if is_multi_view:
            # feature_maps: [B, T, N, C, H//2, W//2] -> [B, T*N, C, H//2, W//2]
            B_f, T_f, N_f = feature_maps.shape[0], feature_maps.shape[1], feature_maps.shape[2]
            feature_maps = feature_maps.view(B_f, T_f * N_f, *feature_maps.shape[3:])

        # Perform tracking using the extracted features
        coord_preds, vis_scores, conf_scores = self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)

        # 如果是多视角格式，需要将输出还原为 [B, T, N, ...] 格式
        if is_multi_view:
            # 假设coord_preds, vis_scores, conf_scores的形状是 [B, T*N, ...]
            # 还原为 [B, T, N, ...]
            coord_preds = coord_preds.view(B, T, N, *coord_preds.shape[2:])
            vis_scores = vis_scores.view(B, T, N, *vis_scores.shape[2:])
            if conf_scores is not None:
                conf_scores = conf_scores.view(B, T, N, *conf_scores.shape[2:])

        return coord_preds, vis_scores, conf_scores
