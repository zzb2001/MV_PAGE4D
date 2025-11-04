# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
import math
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
        is_multi_view: bool = False,
        T: int = None,
        V: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W] (legacy) or [B, T, V, 3, H, W] (multi-view), in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.
            is_multi_view (bool): Whether input is multi-view format [B, T, V, C, H, W]
            T (int): Number of time steps (multi-view mode)
            V (int): Number of views (multi-view mode)

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W] (legacy) or [B, T, V, C, H, W] (multi-view)
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W] (legacy) or [B, T, V, 1, H, W] (multi-view)
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

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            result = self._forward_impl(aggregated_tokens_list, images, patch_start_idx, is_multi_view=is_multi_view, T=T, V=V)
        else:
            # Otherwise, process frames in chunks to manage memory usage
            assert frames_chunk_size > 0

            # Process frames in batches
            all_preds = []
            all_conf = []

            for frames_start_idx in range(0, S, frames_chunk_size):
                frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

                # Process batch of frames
                if self.feature_only:
                    chunk_output = self._forward_impl(
                        aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx,
                        is_multi_view=is_multi_view, T=T, V=V
                    )
                    all_preds.append(chunk_output)
                else:
                    chunk_preds, chunk_conf = self._forward_impl(
                        aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx,
                        is_multi_view=is_multi_view, T=T, V=V
                    )
                    all_preds.append(chunk_preds)
                    all_conf.append(chunk_conf)

            # Concatenate results along the sequence dimension
            if self.feature_only:
                result = torch.cat(all_preds, dim=1)
            else:
                result = (torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1))
        
        # Reshape back to multi-view format if needed
        if is_multi_view and T is not None and V is not None:
            if self.feature_only:
                # result is [B, S_actual, C, H, W] -> [B, T, V, C, H, W]
                # 检查实际的序列长度是否等于 T*V
                S_actual = result.shape[1]
                if S_actual == T * V:
                    result = result.reshape(B, T, V, *result.shape[2:])
                else:
                    # 如果实际序列长度不等于T*V，可能需要特殊处理
                    # 这种情况下，可能需要根据实际S来推断T和V
                    # 或者保持原样，不进行reshape
                    if S_actual % T == 0:
                        V_actual = S_actual // T
                        result = result.reshape(B, T, V_actual, *result.shape[2:])
                    elif S_actual % V == 0:
                        T_actual = S_actual // V
                        result = result.reshape(B, T_actual, V, *result.shape[2:])
                    # 如果都不满足，保持原样
            else:
                # result is tuple of ([B, S_actual, ...], [B, S_actual, ...]) -> ([B, T, V, ...], [B, T, V, ...])
                preds, conf = result
                S_actual = preds.shape[1]
                
                # 检查实际的序列长度是否等于 T*V
                if S_actual == T * V:
                    # 正常情况：可以直接reshape
                    preds = preds.reshape(B, T, V, *preds.shape[2:])
                    conf = conf.reshape(B, T, V, *conf.shape[2:])
                else:
                    # 非常规情况：实际序列长度不等于T*V
                    # 尝试根据实际S来推断T和V，但必须确保reshape是安全的
                    total_elements_preds = preds.numel()
                    total_elements_conf = conf.numel()
                    
                    # 检查是否能够安全地reshape为 (B, T, V, ...)
                    if total_elements_preds % (B * T * V) == 0 and total_elements_conf % (B * T * V) == 0:
                        # 可以整除，尝试reshape
                        elements_per_btv_preds = total_elements_preds // (B * T * V)
                        elements_per_btv_conf = total_elements_conf // (B * T * V)
                        
                        # 检查剩余维度的乘积是否匹配
                        remaining_shape = preds.shape[2:]
                        remaining_elements = 1
                        for dim in remaining_shape:
                            remaining_elements *= dim
                        
                        if remaining_elements == elements_per_btv_preds:
                            # 可以安全reshape
                            try:
                                preds = preds.reshape(B, T, V, *remaining_shape)
                                conf = conf.reshape(B, T, V, *remaining_shape)
                            except RuntimeError as e:
                                # reshape失败，保持原样
                                import warnings
                                warnings.warn(
                                    f"Cannot reshape preds from shape {preds.shape} to (B={B}, T={T}, V={V}, ...). "
                                    f"Actual S={S_actual}, expected S={T*V}. Error: {e}. Keeping original shape."
                                )
                        else:
                            # 元素数不匹配，不能reshape
                            import warnings
                            warnings.warn(
                                f"Cannot reshape preds: shape {preds.shape} cannot be reshaped to (B={B}, T={T}, V={V}, ...). "
                                f"Actual S={S_actual}, expected S={T*V}. Remaining elements={remaining_elements}, "
                                f"elements_per_btv={elements_per_btv_preds}. Keeping original shape."
                            )
                    else:
                        # 不能整除，无法reshape为 (B, T, V, ...)
                        import warnings
                        warnings.warn(
                            f"Cannot reshape preds: total elements {total_elements_preds} cannot be divided by B*T*V={B*T*V}. "
                            f"Actual S={S_actual}, expected S={T*V}. Keeping original shape."
                        )
                
                result = (preds, conf)
        
        return result

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
        is_multi_view: bool = False,
        T: int = None,
        V: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W] (legacy) or [B, T*V, 3, H, W] (multi-view reshaped).
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.
            is_multi_view (bool): Whether input was originally multi-view format
            T (int): Number of time steps (multi-view mode)
            V (int): Number of views (multi-view mode)

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0
        
        # 在第一层确定实际的 S 值（如果与预期不同）
        actual_S = S

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            # 处理不同的输入形状
            # aggregated_tokens_list 可能是 (B, S, P, C) 或 (B*S, P, C)
            if len(x.shape) == 4:
                # 形状是 (B, S, P, C)，需要先reshape为 (B*S, P, C)
                x = x.reshape(-1, x.shape[2], x.shape[3])
                # 更新 actual_S（只在第一次循环时）
                if dpt_idx == 0:
                    actual_S = x.shape[0] // B
            elif len(x.shape) == 3:
                # 形状是 (B*S, P, C) 或类似，需要确保第一个维度正确
                # 如果第一个维度不匹配 B*S，根据实际元素数计算正确的形状
                total_elements = x.numel()
                expected_first_dim = B * actual_S
                expected_elements = expected_first_dim * x.shape[-1]
                
                if total_elements % expected_elements == 0:
                    # 可以整除，reshape为期望的形状
                    mid_dim = total_elements // expected_elements
                    x = x.reshape(expected_first_dim, mid_dim, x.shape[-1])
                else:
                    # 不能整除，使用实际的第一个维度
                    actual_first_dim = x.shape[0]
                    if actual_first_dim % B == 0:
                        # 可以整除B，说明是有效的批次结构
                        layer_actual_S = actual_first_dim // B
                        mid_dim = total_elements // (actual_first_dim * x.shape[-1])
                        x = x.reshape(actual_first_dim, mid_dim, x.shape[-1])
                        # 更新 actual_S（只在第一次循环时，确保所有层使用相同的S）
                        if dpt_idx == 0:
                            actual_S = layer_actual_S
                    else:
                        # 尝试直接reshape为 (actual_first_dim, -1, C)
                        x = x.reshape(actual_first_dim, -1, x.shape[-1])
                        if dpt_idx == 0:
                            actual_S = actual_first_dim // B if actual_first_dim % B == 0 else actual_S
            else:
                raise ValueError(f"Unexpected shape for aggregated_tokens_list[{layer_idx}]: {x.shape}, expected 3D or 4D")

            x = self.norm(x)

            # 计算实际的patch数量
            # x的形状应该是 (B*S, P, C)，其中P是patch数量
            actual_P = x.shape[1]  # 中间维度是patch数量
            
            # 检查实际的P是否等于patch_h * patch_w
            expected_P = patch_h * patch_w
            if actual_P == expected_P:
                # 常规情况：P = patch_h * patch_w，可以直接reshape
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            else:
                # 非常规情况：P != patch_h * patch_w（例如体素化的情况）
                # 需要根据实际的P值来计算合适的patch_h和patch_w
                # 尝试找到能够整除actual_P的h和w，使其最接近正方形
                sqrt_P = int(math.sqrt(actual_P))
                
                # 从sqrt_P开始向下查找，找到能整除actual_P的最大h
                found_shape = False
                for test_h in range(sqrt_P, 0, -1):
                    if actual_P % test_h == 0:
                        test_w = actual_P // test_h
                        # 使用找到的h和w进行reshape
                        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], test_h, test_w))
                        found_shape = True
                        break
                
                if not found_shape:
                    # 理论上不应该到这里，因为任何数都能被1整除
                    # 但如果真的到这里，使用1作为高度
                    x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], 1, actual_P))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, actual_S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.view(B, actual_S, *preds.shape[1:])
        conf = conf.view(B, actual_S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
