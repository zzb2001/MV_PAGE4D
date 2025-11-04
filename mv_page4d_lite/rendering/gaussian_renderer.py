# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GaussianRenderer: 可微高斯Splatting渲染器
将G_t_full渲染回各视角，用于损失计算和可视化
使用gsplat.rasterization进行高效渲染
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
from math import sqrt

try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available, rendering will use CPU fallback")


class GaussianRenderer(nn.Module):
    """
    可微高斯Splatting渲染器
    
    功能：
    1. 将高斯参数投影到各视角
    2. 进行alpha blending渲染
    3. 输出RGB图像和深度图
    """
    
    def __init__(
        self,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # 背景颜色
        near_plane: float = 1e-10,  # 近平面
        far_plane: Optional[float] = None,  # 远平面（可选）
        radius_clip: float = 0.1,  # 半径裁剪
        rasterize_mode: str = 'classic',  # 渲染模式
    ):
        super().__init__()
        
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.radius_clip = radius_clip
        self.rasterize_mode = rasterize_mode
        self.use_gsplat = GSPLAT_AVAILABLE
        
        if not self.use_gsplat:
            print("Warning: gsplat not available, using CPU fallback")
    
    def render(
        self,
        gaussian_params: torch.Tensor,  # [B, T, N, 83] 高斯参数
        gaussian_xyz: torch.Tensor,  # [B, T, N, 3] 高斯位置
        intrinsics: torch.Tensor,  # [B, T, V, 3, 3] 相机内参
        extrinsics: torch.Tensor,  # [B, T, V, 3, 4] 相机外参
        image_size: Tuple[int, int],  # (H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        渲染高斯到各视角
        
        Args:
            gaussian_params: 高斯参数 [B, T, N, 83]
            gaussian_xyz: 高斯位置 [B, T, N, 3]
            intrinsics: 相机内参 [B, T, V, 3, 3]
            extrinsics: 相机外参 [B, T, V, 3, 4]
            image_size: 图像尺寸 (H, W)
        
        Returns:
            dict: 包含以下键
                - rendered_images: [B, T, V, 3, H, W] 渲染的RGB图像
                - rendered_depth: [B, T, V, H, W] 渲染的深度图
                - rendered_alpha: [B, T, V, H, W] 渲染的alpha通道
                - visibility: [B, T, V, N] 可见性（每个高斯在视角中的可见性）
        """
        B, T, N, _ = gaussian_params.shape
        _, _, V, _, _ = intrinsics.shape
        H, W = image_size
        device = gaussian_params.device
        
        # 解析高斯参数
        # gaussian_params: [B, T, N, 83] = delta_x(3) + scales(3) + rotations(4) + sh_coeffs(48) + opacity(1)
        # 注意：根据VoxelGaussianHead，实际格式是：delta_x(3), scales(3), rotations(4), sh_coeffs(48), opacity(1)
        delta_x = gaussian_params[..., 0:3]  # [B, T, N, 3]
        scales = gaussian_params[..., 3:6]  # [B, T, N, 3]
        rotations = gaussian_params[..., 6:10]  # [B, T, N, 4] 四元数
        sh_coeffs = gaussian_params[..., 10:58]  # [B, T, N, 48] SH系数（假设sh_degree=3: 16*3=48）
        opacity = gaussian_params[..., 58:59]  # [B, T, N, 1]
        
        # 计算最终高斯位置：voxel_xyz + delta_x
        gaussian_xyz_final = gaussian_xyz + delta_x  # [B, T, N, 3]
        
        # 计算协方差矩阵（从scales和rotations）
        # 注意：如果gsplat支持covars参数，可以传入；否则会从scales和rotations计算
        
        # 渲染每个时间步和视角
        rendered_images_list = []
        rendered_depth_list = []
        rendered_alpha_list = []
        visibility_list = []
        
        for b in range(B):
            batch_images_list = []
            batch_depth_list = []
            batch_alpha_list = []
            batch_visibility_list = []
            
            for t in range(T):
                time_images = []
                time_depth = []
                time_alpha = []
                time_visibility = []
                
                # 获取该batch和时间步的高斯参数
                xyz_t = gaussian_xyz_final[b, t]  # [N, 3]
                scale_t = scales[b, t]  # [N, 3]
                rotation_t = rotations[b, t]  # [N, 4]
                opacity_t = opacity[b, t].squeeze(-1)  # [N]
                sh_coeffs_t = sh_coeffs[b, t]  # [N, 48]
                
                # 计算SH阶数（从特征维度推断）
                # sh_coeffs: [N, 48] = N * (sh_degree+1)^2 * 3
                # (sh_degree+1)^2 * 3 = 48 -> (sh_degree+1)^2 = 16 -> sh_degree = 3
                sh_degree = 3
                # 将SH系数reshape为 [N, 3, (sh_degree+1)^2] = [N, 3, 16]
                feature_t = sh_coeffs_t.reshape(N, 3, (sh_degree + 1) ** 2)  # [N, 3, 16]
                # 转换为gsplat需要的格式：permute to [N, 16, 3] then contiguous
                feature_t = feature_t.permute(0, 2, 1).contiguous()  # [N, 16, 3]
                
                # 计算协方差矩阵（如果需要）
                covariances_t = None
                if self.use_gsplat:
                    # gsplat可以从scales和rotations自动计算协方差，也可以传入covars
                    # 这里先不传入，让gsplat自动计算
                    pass
                
                for v in range(V):
                    # 获取该时间步和视角的相机参数
                    K = intrinsics[b, t, v]  # [3, 3] 内参（可能是像素单位或归一化）
                    E_c2w = extrinsics[b, t, v]  # [3, 4] 相机->世界（OpenCV格式：camera from world）
                    
                    # 转换为4x4矩阵（相机->世界）
                    if E_c2w.shape == (3, 4):
                        E_c2w_4x4 = torch.zeros(4, 4, device=E_c2w.device, dtype=E_c2w.dtype)
                        E_c2w_4x4[:3, :] = E_c2w
                        E_c2w_4x4[3, 3] = 1.0
                    elif E_c2w.shape == (4, 4):
                        E_c2w_4x4 = E_c2w
                    else:
                        raise ValueError(f"Unexpected extrinsics shape: {E_c2w.shape}, expected (3, 4) or (4, 4)")
                    
                    # 转换为世界->相机（inverse，gsplat需要）
                    E_w2c_4x4 = E_c2w_4x4.inverse()  # [4, 4]
                    
                    # 处理内参：检查是否归一化（通过检查fx是否<1）
                    # 从pose_encoding构建的intrinsics已经是像素单位，但可能传入的是归一化的
                    K_denorm = K.clone()
                    if K[0, 0] < 1.0 or K[1, 1] < 1.0:
                        # 归一化内参，需要denormalize
                        K_denorm[0, 0] = K[0, 0] * W  # fx
                        K_denorm[1, 1] = K[1, 1] * H  # fy
                        K_denorm[0, 2] = K[0, 2] * W  # cx
                        K_denorm[1, 2] = K[1, 2] * H  # cy
                    # 否则已经是像素单位，直接使用
                    
                    # 渲染该视角
                    if self.use_gsplat:
                        # 确保background_color在正确的设备上
                        background_color_device = self.background_color.to(device)
                        
                        # 使用gsplat.rasterization
                        rendering, alpha, _ = rasterization(
                            xyz_t.float(),  # [N, 3]
                            rotation_t.float(),  # [N, 4]
                            scale_t.float(),  # [N, 3]
                            opacity_t.float(),  # [N]
                            feature_t.float(),  # [N, 16, 3]
                            E_w2c_4x4.unsqueeze(0).float(),  # [1, 4, 4] 世界->相机
                            K_denorm.unsqueeze(0).float(),  # [1, 3, 3] 内参
                            W, H,
                            sh_degree=sh_degree,
                            render_mode="RGB+D",
                            packed=False,
                            near_plane=self.near_plane,
                            backgrounds=background_color_device.unsqueeze(0),  # [1, 3]
                            radius_clip=self.radius_clip,
                            covars=covariances_t.float() if covariances_t is not None else None,
                            rasterize_mode=self.rasterize_mode,
                        )  # rendering: [1, H, W, 4], alpha: [1, H, W]
                        
                        # 分离RGB和深度
                        rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)  # [1, H, W, 3], [1, H, W, 1]
                        rendering_img = rendering_img.clamp(0.0, 1.0)
                        
                        # 转换为 [1, 3, H, W]
                        rendering_img = rendering_img.permute(0, 3, 1, 2)  # [1, 3, H, W]
                        rendering_depth = rendering_depth.squeeze(-1)  # [1, H, W]
                        alpha = alpha.squeeze(0)  # [H, W]
                        
                        # 可见性：基于alpha值（>0表示可见）
                        visibility = (alpha.sum(dim=(0, 1)) > 0).float()  # [N] 简化：基于alpha总和
                        # 更精确的可见性需要从gsplat获取，这里简化处理
                        visibility = torch.ones(N, device=xyz_t.device, dtype=torch.float32)  # 简化
                        
                    else:
                        # CPU fallback
                        render_result = self._render_single_view_cpu(
                            xyz_t.unsqueeze(0),  # [1, N, 3]
                            opacity_t.unsqueeze(0).unsqueeze(-1),  # [1, N, 1]
                            scale_t.unsqueeze(0),  # [1, N, 3]
                            rotation_t.unsqueeze(0),  # [1, N, 4]
                            sh_coeffs_t.unsqueeze(0),  # [1, N, 48]
                            K_denorm.unsqueeze(0),  # [1, 3, 3]
                            E_c2w.unsqueeze(0),  # [1, 3, 4] 相机->世界
                            (H, W),
                        )
                        rendering_img = render_result['image']  # [1, 3, H, W]
                        rendering_depth = render_result['depth']  # [1, H, W]
                        alpha = render_result['alpha']  # [1, H, W]
                        visibility = render_result['visibility']  # [1, N]
                    
                    time_images.append(rendering_img)  # [1, 3, H, W]
                    time_depth.append(rendering_depth)  # [1, H, W]
                    time_alpha.append(alpha.unsqueeze(0) if alpha.dim() == 2 else alpha)  # [1, H, W]
                    time_visibility.append(visibility.unsqueeze(0) if visibility.dim() == 1 else visibility)  # [1, N]
                
                # Stack视角
                batch_images_list.append(torch.cat(time_images, dim=0))  # [V, 3, H, W]
                batch_depth_list.append(torch.cat(time_depth, dim=0))  # [V, H, W]
                batch_alpha_list.append(torch.cat(time_alpha, dim=0))  # [V, H, W]
                batch_visibility_list.append(torch.cat(time_visibility, dim=0))  # [V, N]
            
            # Stack时间步
            rendered_images_list.append(torch.stack(batch_images_list, dim=0))  # [T, V, 3, H, W]
            rendered_depth_list.append(torch.stack(batch_depth_list, dim=0))  # [T, V, H, W]
            rendered_alpha_list.append(torch.stack(batch_alpha_list, dim=0))  # [T, V, H, W]
            visibility_list.append(torch.stack(batch_visibility_list, dim=0))  # [T, V, N]
        
        # Stack batch
        rendered_images = torch.stack(rendered_images_list, dim=0)  # [B, T, V, 3, H, W]
        rendered_depth = torch.stack(rendered_depth_list, dim=0)  # [B, T, V, H, W]
        rendered_alpha = torch.stack(rendered_alpha_list, dim=0)  # [B, T, V, H, W]
        visibility = torch.stack(visibility_list, dim=0)  # [B, T, V, N]
        
        return {
            'rendered_images': rendered_images,  # [B, T, V, 3, H, W]
            'rendered_depth': rendered_depth,  # [B, T, V, H, W]
            'rendered_alpha': rendered_alpha,  # [B, T, V, H, W]
            'visibility': visibility,  # [B, T, V, N]
        }
    
    def _render_single_view_cpu(
        self,
        gaussian_xyz: torch.Tensor,  # [B, N, 3] 世界坐标
        opacity: torch.Tensor,  # [B, N, 1]
        scales: torch.Tensor,  # [B, N, 3]
        rotations: torch.Tensor,  # [B, N, 4] 四元数
        sh_coeffs: torch.Tensor,  # [B, N, 48] SH系数
        intrinsics: torch.Tensor,  # [B, 3, 3]
        extrinsics: torch.Tensor,  # [B, 3, 4]
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        CPU fallback渲染单个视角（简化版）
        
        Returns:
            dict: image [B, 3, H, W], depth [B, H, W], alpha [B, H, W], visibility [B, N]
        """
        B, N, _ = gaussian_xyz.shape
        H, W = image_size
        device = gaussian_xyz.device
        
        # 投影到相机坐标系
        points_cam, depth_cam, valid_mask = self._project_to_camera(
            gaussian_xyz, intrinsics, extrinsics
        )  # points_cam: [B, N, 2], depth_cam: [B, N], valid_mask: [B, N]
        
        # 计算2D协方差矩阵
        cov2d = self._compute_covariance_2d(
            scales, rotations, points_cam, depth_cam, intrinsics
        )  # [B, N, 2, 2]
        
        # 计算颜色（从SH系数）
        # 简化：只使用SH0（环境光）
        colors = self._eval_sh_basic(sh_coeffs, points_cam)  # [B, N, 3]
        
        # CPU fallback：简化渲染（使用alpha blending）
        image, depth, alpha = self._rasterize_cpu(
            points_cam,  # [B, N, 2]
            depth_cam,  # [B, N]
            colors,  # [B, N, 3]
            opacity.squeeze(-1),  # [B, N]
            cov2d,  # [B, N, 2, 2]
            (H, W),
        )
        
        # 可见性：高斯在图像范围内且深度有效
        visibility = valid_mask.float()  # [B, N]
        
        return {
            'image': image,  # [B, 3, H, W]
            'depth': depth,  # [B, H, W]
            'alpha': alpha,  # [B, H, W]
            'visibility': visibility,  # [B, N]
        }
    
    def _project_to_camera(
        self,
        points_world: torch.Tensor,  # [B, N, 3]
        intrinsics: torch.Tensor,  # [B, 3, 3]
        extrinsics: torch.Tensor,  # [B, 3, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将世界坐标投影到相机坐标系
        
        Returns:
            points_2d: [B, N, 2] 像素坐标
            depth: [B, N] 深度
            valid_mask: [B, N] 有效性掩码
        """
        B, N, _ = points_world.shape
        device = points_world.device
        
        # 转换为齐次坐标
        points_homo = torch.cat([
            points_world,
            torch.ones(B, N, 1, device=device)
        ], dim=-1)  # [B, N, 4]
        
        # 世界 -> 相机
        R = extrinsics[:, :, :3]  # [B, 3, 3]
        t = extrinsics[:, :, 3:4]  # [B, 3, 1]
        
        points_cam = (R @ points_world.transpose(-1, -2)).transpose(-1, -2) + t.transpose(-1, -2)  # [B, N, 3]
        depth = points_cam[:, :, 2]  # [B, N]
        
        # 投影到像素坐标
        points_2d_homo = (intrinsics @ points_cam.transpose(-1, -2)).transpose(-1, -2)  # [B, N, 3]
        points_2d = points_2d_homo[:, :, :2] / (points_2d_homo[:, :, 2:3] + 1e-8)  # [B, N, 2]
        
        # 有效性掩码：深度>0且在图像范围内
        valid_mask = (depth > 0.01) & (
            (points_2d[:, :, 0] >= 0) & (points_2d[:, :, 0] < intrinsics.shape[-1]) &
            (points_2d[:, :, 1] >= 0) & (points_2d[:, :, 1] < intrinsics.shape[-1])
        )
        
        return points_2d, depth, valid_mask
    
    def _compute_covariance_2d(
        self,
        scales: torch.Tensor,  # [B, N, 3]
        rotations: torch.Tensor,  # [B, N, 4] 四元数
        points_2d: torch.Tensor,  # [B, N, 2]
        depth: torch.Tensor,  # [B, N]
        intrinsics: torch.Tensor,  # [B, 3, 3]
    ) -> torch.Tensor:
        """
        计算2D协方差矩阵
        
        Returns:
            cov2d: [B, N, 2, 2] 2D协方差矩阵
        """
        B, N, _ = scales.shape
        device = scales.device
        
        # 简化：使用2D圆形高斯（等向）
        # TODO: 实现完整的3D->2D协方差投影
        sigma_2d = scales.mean(dim=-1, keepdim=True) / (depth.unsqueeze(-1) + 1e-8)  # [B, N, 1]
        sigma_2d = torch.clamp(sigma_2d, min=1e-6, max=100.0)
        
        # 构建2D协方差矩阵（对角）
        cov2d = torch.zeros(B, N, 2, 2, device=device)
        cov2d[:, :, 0, 0] = sigma_2d.squeeze(-1) ** 2
        cov2d[:, :, 1, 1] = sigma_2d.squeeze(-1) ** 2
        
        return cov2d
    
    def _eval_sh_basic(
        self,
        sh_coeffs: torch.Tensor,  # [B, N, 48] (假设sh_degree=3)
        viewdirs: Optional[torch.Tensor] = None,  # [B, N, 3] 视角方向（可选）
    ) -> torch.Tensor:
        """
        计算颜色（从SH系数）
        
        简化：只使用SH0（环境光）
        TODO: 实现完整的SH渲染
        """
        # SH0系数：前3个通道
        sh0 = sh_coeffs[:, :, :3]  # [B, N, 3]
        
        # 使用sigmoid激活到[0, 1]
        colors = torch.sigmoid(sh0)
        
        return colors
    
    def _rasterize_cpu(
        self,
        points_2d: torch.Tensor,  # [B, N, 2]
        depth: torch.Tensor,  # [B, N]
        colors: torch.Tensor,  # [B, N, 3]
        opacity: torch.Tensor,  # [B, N]
        cov2d: torch.Tensor,  # [B, N, 2, 2]
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CPU fallback渲染（简化版alpha blending）
        
        Returns:
            image: [B, 3, H, W]
            depth: [B, H, W]
            alpha: [B, H, W]
        """
        B, N, _ = points_2d.shape
        H, W = image_size
        device = points_2d.device
        
        # 初始化输出
        image = torch.zeros(B, 3, H, W, device=device)
        depth_map = torch.zeros(B, H, W, device=device)
        alpha_map = torch.zeros(B, H, W, device=device)
        weight_map = torch.zeros(B, H, W, device=device)
        
        # 简化：对每个高斯，在像素坐标附近进行alpha blending
        # 注意：这是简化实现，实际应该使用tile-based渲染和alpha blending
        for b in range(B):
            for n in range(N):
                x, y = points_2d[b, n, 0].item(), points_2d[b, n, 1].item()
                
                # 检查是否在图像范围内
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue
                
                # 简化的点渲染（直接赋值）
                ix, iy = int(x), int(y)
                if 0 <= ix < W and 0 <= iy < H:
                    # Alpha blending
                    alpha_val = opacity[b, n].item()
                    color_val = colors[b, n].cpu().numpy()
                    depth_val = depth[b, n].item()
                    
                    # 当前权重
                    current_alpha = alpha_map[b, iy, ix].item()
                    new_alpha = current_alpha + alpha_val * (1 - current_alpha)
                    
                    if new_alpha > 1e-6:
                        # 更新颜色（alpha blending）
                        image[b, :, iy, ix] = (
                            image[b, :, iy, ix] * current_alpha +
                            torch.tensor(color_val, device=device) * alpha_val * (1 - current_alpha)
                        ) / new_alpha
                        
                        # 更新深度（加权平均）
                        if depth_val > 0:
                            current_weight = weight_map[b, iy, ix].item()
                            depth_map[b, iy, ix] = (
                                depth_map[b, iy, ix] * current_weight +
                                depth_val * alpha_val
                            ) / (current_weight + alpha_val + 1e-8)
                            weight_map[b, iy, ix] = current_weight + alpha_val
                        
                        alpha_map[b, iy, ix] = new_alpha
        
        return image, depth_map, alpha_map
    

