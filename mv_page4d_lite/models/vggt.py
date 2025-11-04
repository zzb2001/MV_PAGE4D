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
from mv_page4d_lite.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from mv_page4d_lite.heads.voxel_gaussian_head import VoxelGaussianHead
from mv_page4d_lite.heads.fused_gaussian_head import FusedGaussianHead
from mv_page4d_lite.models.voxelization import VoxelizationModule
from mv_page4d_lite.models.fusion_head import FusionHead
from mv_page4d_lite.models.temporal_tracker import TemporalTracker
from mv_page4d_lite.rendering.gaussian_renderer import GaussianRenderer

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 enable_gaussian=True, gaussian_output_dim=83,
                 enable_voxelization=False, voxel_size=None, voxel_size_mode='auto',
                 # Voxelization attention routing parameters
                 radius_stage1=2.5, radius_stage2=3.5, temporal_window=3,
                 # Fusion and rendering parameters
                 enable_fusion=False, enable_rendering=False):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            # Pass voxelization parameters to aggregator
            voxel_size=voxel_size,
            radius_stage1=radius_stage1,
            radius_stage2=radius_stage2,
            temporal_window=temporal_window,
            enable_voxelization=enable_voxelization,
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        
        # Stage-0: Voxelization module
        self.enable_voxelization = enable_voxelization
        if enable_voxelization:
            self.voxelization = VoxelizationModule(
                embed_dim=embed_dim,
                voxel_size=voxel_size,
                voxel_size_mode=voxel_size_mode,
                target_num_voxels=120000,
                use_morton_encoding=False,  # 先用简单实现
                use_sparse3d=False,  # 先跳过
            )
        else:
            self.voxelization = None
        
        # Fusion head: 多视角融合
        self.enable_fusion = enable_fusion
        if enable_fusion and enable_voxelization:
            self.fusion_head = FusionHead(
                embed_dim=embed_dim,
                gaussian_output_dim=gaussian_output_dim,
            )
        else:
            self.fusion_head = None
        
        # Fused Gaussian Head: Student端融合高斯头
        if enable_gaussian and enable_voxelization and enable_fusion:
            self.fused_gaussian_head = FusedGaussianHead(
                dim_in=embed_dim // 4,  # FusionHead输出的特征维度
                output_dim=gaussian_output_dim,
            )
        else:
            self.fused_gaussian_head = None
        
        # Temporal Tracker: 时序关联
        self.temporal_tracker = TemporalTracker() if enable_voxelization else None
        
        # Gaussian Renderer: 可微渲染（使用gsplat）
        self.enable_rendering = enable_rendering
        if enable_rendering and enable_gaussian:
            self.gaussian_renderer = GaussianRenderer(
                background_color=(0.0, 0.0, 0.0),  # 黑色背景
                near_plane=1e-10,
                far_plane=None,
                radius_clip=0.1,
                rasterize_mode='classic',
            )
        else:
            self.gaussian_renderer = None
        
        # Gaussian Splatting parameter head
        # Use voxel-level head if voxelization is enabled, otherwise use pixel-level head
        if enable_gaussian:
            if enable_voxelization:
                # Voxel-level Gaussian head: predicts parameters directly from voxel tokens
                self.voxel_gaussian_head = VoxelGaussianHead(
                    dim_in=2 * embed_dim,
                    output_dim=gaussian_output_dim,
                )
                self.gaussian_param_head = None  # Not used in voxelization mode
            else:
                # Pixel-level Gaussian head: DPT-based head for pixel-level features
                self.gaussian_param_head = VGGT_DPT_GS_Head(
                    dim_in=2 * embed_dim,
                    patch_size=patch_size,
                    output_dim=gaussian_output_dim,
                    activation="norm_exp",
                    conf_activation="expp1",
                    features=256,
                    intermediate_layer_idx=[4, 11, 17, 23],
                )
                self.voxel_gaussian_head = None  # Not used in non-voxelization mode
        else:
            self.gaussian_param_head = None
            self.voxel_gaussian_head = None

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
        is_multi_view = len(images.shape) 
        
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
        # 支持传入depth/intrinsics/extrinsics用于体素化
        # 如果未提供，先做一次forward预测
        
        # 第一次forward：获取深度和相机参数（如果未提供）
        depth_for_voxel = None
        intrinsics_for_voxel = None
        extrinsics_for_voxel = None
        
        if is_multi_view and self.enable_voxelization:
            # 先做一次轻量forward获取相机参数和深度
            # 使用detach避免梯度问题
            with torch.set_grad_enabled(False):
                temp_tokens, temp_patch_idx, _, _ = self.aggregator(
                    images=images,
                    temporal_features=temporal_features,
                    use_voxelization=True,  # 先不使用体素化
                )
                
                # 获取相机参数
                if self.camera_head is not None:
                    pose_enc_temp = self.camera_head(
                        temp_tokens,
                        is_multi_view=is_multi_view,
                        T=T if is_multi_view else None,
                        V=V if is_multi_view else None
                    )[-1]  # [B, V, 9]
                    
                    # 转换为intrinsics和extrinsics
                    from mv_page4d_lite.utils.pose_enc import pose_encoding_to_extri_intri
                    extrinsics_for_voxel, intrinsics_for_voxel = pose_encoding_to_extri_intri(
                        pose_enc_temp, images.shape[-2:]
                    )  # [B, V, 3, 4], [B, V, 3, 3]
                    
                    # 扩展到时间维度 [B, T, V, 3, 4], [B, T, V, 3, 3]
                    if is_multi_view:
                        extrinsics_for_voxel = extrinsics_for_voxel.unsqueeze(1).expand(-1, T, -1, -1, -1)
                        intrinsics_for_voxel = intrinsics_for_voxel.unsqueeze(1).expand(-1, T, -1, -1, -1)
                
                # 获取深度
                if self.depth_head is not None:
                    depth_temp, _ = self.depth_head(
                        temp_tokens, images=images, patch_start_idx=temp_patch_idx,
                        is_multi_view=is_multi_view, T=T if is_multi_view else None, V=V if is_multi_view else None
                    )
                    # depth_temp: [B, T, V, 1, H, W] 或 [B, T, V, H, W]
                    if len(depth_temp.shape) == 6:
                        depth_for_voxel = depth_temp.squeeze(-3)  # [B, T, V, H, W]
                    elif len(depth_temp.shape) == 5:
                        depth_for_voxel = depth_temp  # [B, T, V, H, W]
        
        # 主forward：使用体素化
        aggregated_tokens_list, patch_start_idx, mask_logits, voxel_data = self.aggregator(
            images=images, 
            temporal_features=temporal_features,
            depth=depth_for_voxel,
            intrinsics=intrinsics_for_voxel,
            extrinsics=extrinsics_for_voxel,
            use_voxelization=self.enable_voxelization,
        )
        predictions = {}
        
        # Add mask_logits to predictions for supervision loss (if available)
        if mask_logits is not None:
            # mask_logits shape: [B*S, 1, H, W] where S = T*V for multi-view or S for legacy
            # Convert to multi-view format if needed
            if is_multi_view:
                # Reshape from [B*T*V, 1, H, W] to [B, T, V, H, W]
                B_total, _, H_mask, W_mask = mask_logits.shape
                mask_logits = mask_logits.squeeze(1)  # [B*T*V, H, W]
                
                # 检查实际元素数是否匹配期望的形状
                mask_logits_size = mask_logits.numel()
                expected_size = B * T * V * H_mask * W_mask
                
                if mask_logits_size == expected_size:
                    # 大小匹配，直接reshape
                    mask_logits = mask_logits.view(B, T, V, H_mask, W_mask)  # [B, T, V, H, W]
                else:
                    # 大小不匹配，根据实际元素数计算正确的S
                    # mask_logits的形状是 [B*S_actual, H, W]
                    S_actual = B_total // B if B_total % B == 0 else B_total
                    actual_size = S_actual * H_mask * W_mask
                    
                    if mask_logits_size == B * actual_size:
                        # 可以reshape为 [B, S_actual, H, W]
                        mask_logits = mask_logits.view(B, S_actual, H_mask, W_mask)
                        
                        # 尝试reshape为 [B, T, V, H, W] 如果可能
                        if S_actual == T * V:
                            mask_logits = mask_logits.view(B, T, V, H_mask, W_mask)
                        elif S_actual % T == 0:
                            # 如果S_actual可以被T整除，计算V_actual
                            V_actual = S_actual // T
                            mask_logits = mask_logits.view(B, T, V_actual, H_mask, W_mask)
                        elif S_actual % V == 0:
                            # 如果S_actual可以被V整除，计算T_actual
                            T_actual = S_actual // V
                            mask_logits = mask_logits.view(B, T_actual, V, H_mask, W_mask)
                        # 否则保持 [B, S_actual, H, W] 形状
                    else:
                        # 如果无法匹配，使用实际形状
                        # 计算实际的H和W
                        actual_elements_per_frame = mask_logits_size // (B * S_actual) if S_actual > 0 else mask_logits_size // B
                        # 尝试推断H和W
                        sqrt_elements = int(actual_elements_per_frame ** 0.5)
                        for h in range(sqrt_elements, 0, -1):
                            if actual_elements_per_frame % h == 0:
                                w = actual_elements_per_frame // h
                                if h * w == actual_elements_per_frame:
                                    mask_logits = mask_logits.view(B, S_actual, h, w)
                                    break
                        # 如果无法推断，保持原样
            else:
                # Legacy format: [B*S, 1, H, W] -> [B, S, H, W]
                B_total, _, H_mask, W_mask = mask_logits.shape
                mask_logits = mask_logits.squeeze(1)  # [B*S, H, W]
                
                # 检查实际元素数是否匹配
                mask_logits_size = mask_logits.numel()
                expected_size = B * S * H_mask * W_mask
                
                if mask_logits_size == expected_size:
                    mask_logits = mask_logits.view(B, S, H_mask, W_mask)  # [B, S, H, W]
                else:
                    # 大小不匹配，根据实际元素数计算正确的S
                    S_actual = B_total // B if B_total % B == 0 else B_total
                    if mask_logits_size == B * S_actual * H_mask * W_mask:
                        mask_logits = mask_logits.view(B, S_actual, H_mask, W_mask)
                    else:
                        # 如果仍然不匹配，尝试推断H和W
                        actual_elements_per_frame = mask_logits_size // (B * S_actual) if S_actual > 0 else mask_logits_size // B
                        sqrt_elements = int(actual_elements_per_frame ** 0.5)
                        for h in range(sqrt_elements, 0, -1):
                            if actual_elements_per_frame % h == 0:
                                w = actual_elements_per_frame // h
                                if h * w == actual_elements_per_frame:
                                    mask_logits = mask_logits.view(B, S_actual, h, w)
                                    break
            
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
            
            # Gaussian parameter prediction and fusion
            if is_multi_view and self.enable_voxelization and voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
                # Extract voxel tokens and coordinates
                last_tokens = aggregated_tokens_list[-1]  # 可能是 [B*S, P, C] 或 [B, S, P, C]
                patch_start = patch_start_idx
                
                # 根据实际维度处理
                if len(last_tokens.shape) == 3:
                    # [B*S, P, C]
                    B_total, P_vox, C = last_tokens.shape
                    # Extract voxel tokens
                    voxel_tokens_raw = last_tokens[:, patch_start:, :]  # [B*S, N_voxels, C]
                    # 尝试reshape为 [B, T, N_voxels, C]
                    N_voxels = P_vox - patch_start
                    S_actual = B_total // B if B_total % B == 0 else B_total
                    if S_actual == T:
                        voxel_tokens = voxel_tokens_raw.reshape(B, T, N_voxels, C)  # [B, T, N_voxels, C]
                    else:
                        # 如果S_actual不等于T，尝试根据实际大小reshape
                        voxel_tokens_size = voxel_tokens_raw.numel()
                        if voxel_tokens_size == B * T * N_voxels * C:
                            voxel_tokens = voxel_tokens_raw.reshape(B, T, N_voxels, C)
                        else:
                            # 使用实际的S_actual
                            voxel_tokens = voxel_tokens_raw.reshape(B, S_actual, N_voxels, C)
                elif len(last_tokens.shape) == 4:
                    # [B, S, P, C]
                    B_actual, S_actual, P_vox, C = last_tokens.shape
                    # Extract voxel tokens
                    voxel_tokens_raw = last_tokens[:, :, patch_start:, :]  # [B, S, N_voxels, C]
                    N_voxels = P_vox - patch_start
                    # 尝试reshape为 [B, T, N_voxels, C]
                    if S_actual == T:
                        voxel_tokens = voxel_tokens_raw.reshape(B, T, N_voxels, C)  # [B, T, N_voxels, C]
                    else:
                        # 如果S_actual不等于T，尝试根据实际大小reshape
                        voxel_tokens_size = voxel_tokens_raw.numel()
                        if voxel_tokens_size == B * T * N_voxels * C:
                            voxel_tokens = voxel_tokens_raw.reshape(B, T, N_voxels, C)
                        else:
                            # 使用实际的S_actual
                            voxel_tokens = voxel_tokens_raw  # [B, S_actual, N_voxels, C]
                else:
                    raise ValueError(
                        f"Unexpected last_tokens shape: {last_tokens.shape}, expected 3D [B*S, P, C] or 4D [B, S, P, C]"
                    )
                
                # Get voxel coordinates
                voxel_xyz = None
                voxel_ids = None
                if 'voxel_xyz_list' in voxel_data and voxel_data['voxel_xyz_list'] is not None:
                    voxel_xyz_list = voxel_data['voxel_xyz_list']
                    voxel_ids_list = voxel_data.get('voxel_ids_list', None)
                    
                    if len(voxel_xyz_list) == T and len(voxel_xyz_list[0]) == B:
                        max_n = 0
                        for t_list in voxel_xyz_list:
                            for xyz_b in t_list:
                                if xyz_b is not None and xyz_b.numel() > 0:
                                    max_n = max(max_n, xyz_b.shape[0])
                        
                        if max_n > 0:
                            voxel_xyz_padded = []
                            voxel_ids_padded = []
                            for t in range(T):
                                batch_xyz = []
                                batch_ids = []
                                for b in range(B):
                                    if t < len(voxel_xyz_list) and b < len(voxel_xyz_list[t]) and voxel_xyz_list[t][b] is not None:
                                        xyz_tb = voxel_xyz_list[t][b]
                                        if xyz_tb.shape[0] < max_n:
                                            padding = torch.zeros(max_n - xyz_tb.shape[0], 3, device=xyz_tb.device, dtype=xyz_tb.dtype)
                                            xyz_tb = torch.cat([xyz_tb, padding], dim=0)
                                            if voxel_ids_list is not None and t < len(voxel_ids_list) and b < len(voxel_ids_list[t]):
                                                ids_tb = voxel_ids_list[t][b]
                                                ids_padding = torch.zeros(max_n - ids_tb.shape[0], dtype=ids_tb.dtype, device=ids_tb.device)
                                                ids_tb = torch.cat([ids_tb, ids_padding], dim=0)
                                            else:
                                                ids_tb = torch.zeros(max_n, dtype=torch.long, device=xyz_tb.device)
                                        else:
                                            xyz_tb = xyz_tb[:max_n]
                                            if voxel_ids_list is not None and t < len(voxel_ids_list) and b < len(voxel_ids_list[t]):
                                                ids_tb = voxel_ids_list[t][b][:max_n]
                                            else:
                                                ids_tb = torch.arange(max_n, dtype=torch.long, device=xyz_tb.device)
                                        batch_xyz.append(xyz_tb)
                                        batch_ids.append(ids_tb)
                                    else:
                                        batch_xyz.append(torch.zeros(max_n, 3, device=last_tokens.device, dtype=last_tokens.dtype))
                                        batch_ids.append(torch.zeros(max_n, dtype=torch.long, device=last_tokens.device))
                                voxel_xyz_padded.append(torch.stack(batch_xyz, dim=0))
                                voxel_ids_padded.append(torch.stack(batch_ids, dim=0))
                            voxel_xyz = torch.stack(voxel_xyz_padded, dim=1)  # [B, T, max_n, 3]
                            voxel_ids = torch.stack(voxel_ids_padded, dim=1)  # [B, T, max_n]
                
                # Fusion: 多视角融合为统一表示
                if self.fusion_head is not None and voxel_xyz is not None:
                    # 获取置信度（从depth_conf或world_points_conf）
                    confidence = None
                    if 'depth_conf' in predictions:
                        depth_conf = predictions['depth_conf']
                        # 简化：从depth_conf提取置信度（需要resize到体素级别）
                        # TODO: 实现从depth到体素的置信度映射
                        pass
                    
                    # 融合多视角
                    fusion_output = self.fusion_head(
                        voxel_tokens=voxel_tokens,  # [B, T, N_t, C]
                        voxel_xyz=voxel_xyz,  # [B, T, N_t, 3]
                        voxel_ids=voxel_ids,
                        confidence=confidence,
                        visibility=None,  # TODO: 从渲染结果获取
                        static_mask=mask_logits if mask_logits is not None else None,
                    )
                    
                    # 提取融合后的特征和坐标
                    fused_features = fusion_output['fused_features']  # [B, T, N_t, embed_dim//4]
                    voxel_xyz_fused = fusion_output['voxel_xyz_fused']  # [B, T, N_t, 3]
                    
                    # Fused Gaussian Head: 从融合特征预测统一高斯参数
                    if self.fused_gaussian_head is not None:
                        voxel_size = None
                        if self.voxelization is not None and hasattr(self.voxelization, 'voxel_size'):
                            voxel_size = self.voxelization.voxel_size
                        
                        fused_gaussian_output = self.fused_gaussian_head(
                            voxel_tokens=fused_features,  # [B, T, N_t, embed_dim//4]
                            voxel_xyz=voxel_xyz_fused,  # [B, T, N_t, 3]
                            voxel_size=voxel_size,
                        )
                        
                        predictions["fused_gaussian_params"] = fused_gaussian_output['gaussian_params']  # [B, T, N_t, 83]
                        predictions["fused_gaussian_xyz"] = fused_gaussian_output['gaussian_xyz']  # [B, T, N_t, 3]
                        predictions["fused_gaussian_opacity"] = fused_gaussian_output['opacity']
                        predictions["fused_gaussian_scales"] = fused_gaussian_output['scales']
                        predictions["fused_gaussian_rotations"] = fused_gaussian_output['rotations']
                        predictions["fused_gaussian_sh_coeffs"] = fused_gaussian_output['sh_coeffs']
                        
                        # 分离静态和动态高斯
                        if fusion_output['G_t_static'] is not None:
                            predictions["fused_gaussian_static"] = fusion_output['G_t_static']
                        if fusion_output['G_t_dynamic'] is not None:
                            predictions["fused_gaussian_dynamic"] = fusion_output['G_t_dynamic']
                    
                    # 保存融合后的点云
                    predictions["fused_point_cloud"] = fusion_output['P_t_full']  # [B, T, N_t, 3]
                
                # Teacher: per-view Gaussian (AnySplat GS head)
                if self.voxel_gaussian_head is not None:
                    voxel_size = None
                    if self.voxelization is not None and hasattr(self.voxelization, 'voxel_size'):
                        voxel_size = self.voxelization.voxel_size
                    
                    teacher_output = self.voxel_gaussian_head(
                        voxel_tokens=voxel_tokens,
                        voxel_xyz=voxel_xyz,
                        voxel_size=voxel_size
                    )
                    predictions["teacher_gaussian_params"] = teacher_output['gaussian_params']
                    predictions["teacher_gaussian_xyz"] = teacher_output.get('gaussian_xyz', voxel_xyz)
                
                # Gaussian Renderer: 渲染fused高斯回各视角
                if self.gaussian_renderer is not None and 'fused_gaussian_params' in predictions:
                    render_output = self.gaussian_renderer.render(
                        gaussian_params=predictions["fused_gaussian_params"],  # [B, T, N_t, 83]
                        gaussian_xyz=predictions["fused_gaussian_xyz"],  # [B, T, N_t, 3]
                        intrinsics=intrinsics_for_voxel if intrinsics_for_voxel is not None else predictions.get('intrinsics'),
                        extrinsics=extrinsics_for_voxel if extrinsics_for_voxel is not None else predictions.get('extrinsics'),
                        image_size=images.shape[-2:],
                    )
                    predictions["rendered_images"] = render_output['rendered_images']  # [B, T, V, 3, H, W]
                    predictions["rendered_depth"] = render_output['rendered_depth']  # [B, T, V, H, W]
                    predictions["rendered_alpha"] = render_output['rendered_alpha']  # [B, T, V, H, W]
                    predictions["gaussian_visibility"] = render_output['visibility']  # [B, T, V, N]
                    
            elif self.gaussian_param_head is not None:
                # Use pixel-level head (non-voxelization mode)
                gaussian_params = self.gaussian_param_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                    is_multi_view=is_multi_view, T=T if is_multi_view else None, V=V if is_multi_view else None
                )
                predictions["gaussian_params"] = gaussian_params

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
        
        # 添加体素数据到predictions（如果使用体素化）
        if voxel_data is not None and voxel_data.get('use_voxel_tokens', False):
            predictions["voxel_data"] = voxel_data

        return predictions
