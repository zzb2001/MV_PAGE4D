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

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None,
                is_multi_view=False, T=None, V=None):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) (legacy) or (B, T, V, C, H, W) (multi-view).
                                   B = batch size, S = sequence length, T = time steps, V = views.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.
            is_multi_view (bool): Whether input is multi-view format [B, T, V, C, H, W]
            T (int): Number of time steps (multi-view mode)
            V (int): Number of views (multi-view mode)

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                    Shape: [B, S, N, 2] (legacy) or [B, T, V, N, 2] (multi-view)
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                    Shape: [B, S, N] (legacy) or [B, T, V, N] (multi-view)
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
                    Shape: [B, S, N] (legacy) or [B, T, V, N] (multi-view)
        """
        # Detect multi-view input format
        if is_multi_view and len(images.shape) == 6:
            B, T_in, V_in, C, H, W = images.shape
            if T is not None and V is not None:
                assert T_in == T and V_in == V, f"Mismatch: images has (T={T_in}, V={V_in}), but expected (T={T}, V={V})"
            T = T_in
            V = V_in
            S = T * V
            # Reshape to legacy format for processing: [B, T, V, C, H, W] -> [B, T*V, C, H, W]
            images = images.reshape(B, T * V, C, H, W)
        else:
            B, S, _, H, W = images.shape
            T = S
            V = 1

        # Extract features from tokens
        # feature_maps has shape (B, S, C, H//2, W//2) (legacy) or (B, T*V, C, H//2, W//2) (multi-view) due to down_ratio=2
        feature_maps = self.feature_extractor(
            aggregated_tokens_list, images, patch_start_idx,
            is_multi_view=is_multi_view, T=T, V=V
        )

        # Use default iterations if not specified
        if iters is None:
            iters = self.iters

        # Perform tracking using the extracted features
        # Note: tracker expects feature_maps in format [B, S, C, H, W]
        # coord_preds is a list of tensors [B, S, N, 2] for each iteration
        # vis_scores and conf_scores are tensors [B, S, N]
        coord_preds, vis_scores, conf_scores = self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)

        # Reshape back to multi-view format if needed
        if is_multi_view and T is not None and V is not None:
            # coord_preds is a list, reshape each element: [B, T*V, N, 2] -> [B, T, V, N, 2]
            if coord_preds is not None and isinstance(coord_preds, list):
                coord_preds = [pred.reshape(B, T, V, *pred.shape[2:]) for pred in coord_preds]
            # vis_scores: [B, T*V, N] -> [B, T, V, N]
            if vis_scores is not None:
                vis_scores = vis_scores.reshape(B, T, V, *vis_scores.shape[2:])
            # conf_scores: [B, T*V, N] -> [B, T, V, N]
            if conf_scores is not None:
                conf_scores = conf_scores.reshape(B, T, V, *conf_scores.shape[2:])

        return coord_preds, vis_scores, conf_scores
