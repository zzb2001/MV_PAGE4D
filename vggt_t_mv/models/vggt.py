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
import logging
from typing import Optional, Dict, Tuple, List

from vggt_t_mv.models.aggregator import Aggregator
from vggt_t_mv.heads.camera_head import CameraHead
from vggt_t_mv.heads.dpt_head import DPTHead
from vggt_t_mv.heads.track_head import TrackHead
from vggt_t_mv.utils.weight_loading import (
    load_checkpoint_weights, load_pi3_weights, adapt_weights_dimension,
    load_weights_three_stage
)

logger = logging.getLogger(__name__)

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
        Forward pass of the VGGT_MV model (multi-view temporal only).
        Args:
            images (torch.Tensor): Input images with shape [B, T, N, 3, H, W]
                Where:
                    B: batch size
                    T: time window length (temporal frames)
                    N: number of synchronized views
                    3: RGB channels
                    H, W: image height and width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [B, N, 2], where N is the number of query points.
                Default: None
            temporal_features (torch.Tensor, optional): Temporal features for attention modulation.
                Shape: [B, T*N]
        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, T, N, 9]
                - depth (torch.Tensor): Predicted depth maps
                - depth_conf (torch.Tensor): Confidence scores for depth predictions
                - world_points (torch.Tensor): 3D world coordinates for each pixel
                - world_points_conf (torch.Tensor): Confidence scores for world points
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks
                - vis (torch.Tensor): Visibility scores for tracked points
                - conf (torch.Tensor): Confidence scores for tracked points
        """        
        # Validate input shape
        if len(images.shape) != 6:
            raise ValueError(f"Expected images with shape [B, T, N, 3, H, W], got {images.shape}")
        
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Multi-view temporal format: [B, T, N, 3, H, W]
        result = self.aggregator(images=images, temporal_features=temporal_features)
        aggregated_tokens_list, patch_start_idx, dual_stream_outputs = result
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                # 根据架构图4.1: 输入来自位姿流
                pose_enc_list = self.camera_head(
                    aggregated_tokens_list, 
                    dual_stream_outputs=dual_stream_outputs
                )
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                # 根据架构图4.2: 输入来自几何流的patch tokens
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, 
                    images=images, 
                    patch_start_idx=patch_start_idx,
                    dual_stream_outputs=dual_stream_outputs
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                # 根据架构图4.2: 输入来自几何流的patch tokens
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, 
                    images=images, 
                    patch_start_idx=patch_start_idx,
                    dual_stream_outputs=dual_stream_outputs
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration for inference
            predictions["track_list"] = track_list  # all iterations for training loss
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference
        
        # 如果启用两流架构，添加两流输出
        if dual_stream_outputs is not None:
            predictions["dual_stream_outputs"] = dual_stream_outputs

        return predictions
    
    def load_pretrained_weights(self, checkpoint_path: Optional[str] = None, 
                                pi3_path: Optional[str] = None, 
                                device: str = 'cpu',
                                page4d_mid_layers: Optional[List[int]] = None,
                                use_three_stage: bool = True) -> Dict[str, int]:
        """
        加载预训练权重：三阶段加载策略
        
        Args:
            checkpoint_path: PAGE-4D checkpoint_150.pt 文件路径（可选）
            pi3_path: Pi3 模型路径（HuggingFace路径或本地路径），可选
            device: 加载设备
            page4d_mid_layers: PAGE-4D 要覆盖的中段层索引（如 [8, 9, 10, 11, 12, 13, 14, 15]）
                如果为 None，默认覆盖层 8-15
            use_three_stage: 是否使用三阶段加载策略（推荐）
                - True: 三阶段（先Pi3，再PAGE-4D覆盖，最后初始化新增模块）
                - False: 向后兼容的旧方法
        
        Returns:
            dict: 加载统计信息
        """
        if use_three_stage:
            # 使用新的三阶段加载策略
            stats = load_weights_three_stage(
                pi3_path=pi3_path,
                checkpoint_path=checkpoint_path,
                model=self,
                device=device,
                page4d_mid_layers=page4d_mid_layers
            )
            return stats
        else:
            # 向后兼容的旧方法
            return self._load_pretrained_weights_legacy(checkpoint_path, pi3_path, device)
    
    def _load_pretrained_weights_legacy(self, checkpoint_path: Optional[str] = None,
                                       pi3_path: Optional[str] = None,
                                       device: str = 'cpu') -> Dict[str, int]:
        """
        向后兼容的旧权重加载方法
        """
        stats = {'checkpoint_loaded': 0, 'pi3_loaded': 0, 'missing': 0, 'unexpected': 0}
        
        # 1. 从 checkpoint_150.pt 加载权重
        if checkpoint_path:
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}...")
                
                if checkpoint_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as safetensors_load
                    checkpoint_dict = safetensors_load(checkpoint_path, device=device)
                else:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    checkpoint_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                
                model_dict = self.state_dict()
                mapped_dict = {}
                
                for key, param in checkpoint_dict.items():
                    if key in model_dict:
                        if model_dict[key].shape == param.shape:
                            mapped_dict[key] = param
                
                missing_keys, unexpected_keys = self.load_state_dict(mapped_dict, strict=False)
                stats['checkpoint_loaded'] = len(mapped_dict)
                stats['missing'] = len(missing_keys)
                stats['unexpected'] = len(unexpected_keys)
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
        
        # 2. 从 Pi3 加载权重
        if pi3_path:
            try:
                pi3_dict, missing_pi3, unexpected_pi3 = load_pi3_weights(
                    pi3_path, self.aggregator, device=device
                )
                stats['pi3_loaded'] = len(pi3_dict)
            except Exception as e:
                logger.warning(f"Failed to load Pi3 weights: {e}", exc_info=True)
        
        # 3. 初始化两流架构
        if self.aggregator.enable_dual_stream:
            logger.info("Initializing dual-stream blocks...")
            for i in range(len(self.aggregator.frame_blocks)):
                self._copy_block_weights(
                    self.aggregator.frame_blocks[i],
                    self.aggregator.pose_frame_blocks[i]
                )
                self._copy_block_weights(
                    self.aggregator.frame_blocks[i],
                    self.aggregator.geo_frame_blocks[i]
                )
            for i in range(len(self.aggregator.global_blocks)):
                self._copy_block_weights(
                    self.aggregator.global_blocks[i],
                    self.aggregator.pose_global_blocks[i]
                )
                self._copy_block_weights(
                    self.aggregator.global_blocks[i],
                    self.aggregator.geo_global_blocks[i]
                )
        
        return stats
    
    def _copy_block_weights(self, source_block, target_block):
        """
        复制 block 的权重（用于两流架构初始化）
        """
        source_state = source_block.state_dict()
        target_state = target_block.state_dict()
        
        for key in target_state:
            if key in source_state:
                if source_state[key].shape == target_state[key].shape:
                    target_state[key] = source_state[key].clone()
                else:
                    # 维度不匹配，尝试适配
                    try:
                        target_state[key] = adapt_weights_dimension(
                            source_state[key], target_state[key].shape, strategy='truncate'
                        )
                    except:
                        logger.warning(f"Cannot copy {key}: shape mismatch {source_state[key].shape} vs {target_state[key].shape}")
        
        target_block.load_state_dict(target_state)
