"""
权重加载工具：支持从 checkpoint_150.pt 和 Pi3 加载权重
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging
import os

logger = logging.getLogger(__name__)

# 尝试导入 safetensors 库（如果可用）
try:
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.warning("safetensors library not available. Install with: pip install safetensors")


def load_checkpoint_weights(checkpoint_path: str, model: nn.Module, device='cpu') -> Tuple[Dict, Dict]:
    """
    从 checkpoint_150.pt 加载权重
    
    Args:
        checkpoint_path: checkpoint 文件路径
        model: 模型实例
        device: 加载设备
        
    Returns:
        mapped_state_dict: 映射后的权重字典
        missing_keys: 缺失的键
        unexpected_keys: 意外的键
    """
    # 处理 safetensors 格式
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            logger.error(f"Cannot load safetensors file: safetensors library not installed. Install with: pip install safetensors")
            return {}, [], []
        try:
            checkpoint_state_dict = safetensors_load(checkpoint_path, device=device)
            # safetensors 通常不包含 'model' 键，直接使用
            if isinstance(checkpoint_state_dict, dict) and 'model' in checkpoint_state_dict:
                checkpoint_state_dict = checkpoint_state_dict['model']
        except Exception as e:
            logger.error(f"Failed to load safetensors checkpoint: {e}")
            return {}, [], []
    else:
        # PyTorch checkpoint 格式
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_state_dict = model.state_dict()
    
    # 创建映射后的权重字典
    mapped_state_dict = {}
    unmapped_keys = []
    
    for name, param in checkpoint_state_dict.items():
        original_name = name
        
        # 映射规则：
        # frame_blocks -> time_blocks (Time-SA, 时序建模)
        # global_blocks -> view_blocks (View-SA, 多视角聚合)
        # 但实际上 view_blocks 和 time_blocks 是别名，所以保持原名称
        
        # 检查是否匹配模型中的键
        if name in model_state_dict:
            # 形状匹配检查
            if model_state_dict[name].shape == param.shape:
                mapped_state_dict[name] = param
            else:
                logger.warning(f"Shape mismatch for {name}: checkpoint {param.shape} vs model {model_state_dict[name].shape}, skipping")
                unmapped_keys.append(name)
        else:
            # 尝试名称映射
            new_name = None
            if 'frame_blocks' in name:
                # frame_blocks 可以直接使用（因为 time_blocks 是 frame_blocks 的别名）
                new_name = name
            elif 'global_blocks' in name:
                # global_blocks 可以直接使用（因为 view_blocks 是 global_blocks 的别名）
                new_name = name
            else:
                new_name = name
            
            if new_name and new_name in model_state_dict:
                if model_state_dict[new_name].shape == param.shape:
                    mapped_state_dict[new_name] = param
                else:
                    logger.warning(f"Shape mismatch after mapping {name} -> {new_name}: {param.shape} vs {model_state_dict[new_name].shape}")
                    unmapped_keys.append(name)
            else:
                unmapped_keys.append(name)
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    
    logger.info(f"Loaded {len(mapped_state_dict)} parameters from checkpoint")
    if missing_keys:
        logger.info(f"Missing {len(missing_keys)} keys (using random init)")
    if unexpected_keys:
        logger.info(f"Unexpected {len(unexpected_keys)} keys (ignored)")
    if unmapped_keys:
        logger.info(f"Unmapped {len(unmapped_keys)} keys from checkpoint")
    
    return mapped_state_dict, missing_keys, unexpected_keys


def load_pi3_weights(pi3_model_or_path, model: nn.Module, device='cpu') -> Tuple[Dict, List, List]:
    """
    从 Pi3 模型加载多视角聚合相关权重
    
    Args:
        pi3_model_or_path: Pi3 模型实例或权重文件路径
        model: vggt_t_mv 模型实例
        device: 加载设备
        
    Returns:
        mapped_state_dict: 映射后的权重字典
        missing_keys: 缺失的键
        unexpected_keys: 意外的键
    """
    # 加载 Pi3 模型或权重
    if isinstance(pi3_model_or_path, str):
        file_path = pi3_model_or_path
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"Pi3 weight file not found: {file_path}")
            return {}, [], []
        
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.safetensors'):
            # 使用 safetensors 加载
            if not SAFETENSORS_AVAILABLE:
                logger.error(f"Cannot load safetensors file: safetensors library not installed. Install with: pip install safetensors")
                return {}, [], []
            try:
                logger.info(f"Loading safetensors from {file_path}...")
                pi3_state_dict = safetensors_load(file_path, device=device)
                if isinstance(pi3_state_dict, dict) and 'model' in pi3_state_dict:
                    pi3_state_dict = pi3_state_dict['model']
            except Exception as e:
                logger.error(f"Failed to load safetensors from {file_path}: {e}")
                return {}, [], []
        elif file_path.endswith('.pt') or file_path.endswith('.pth'):
            # 使用 torch.load 加载（需要设置 weights_only=False）
            try:
                logger.info(f"Loading PyTorch checkpoint from {file_path}...")
                pi3_state_dict = torch.load(file_path, map_location=device, weights_only=False)
                if isinstance(pi3_state_dict, dict) and 'model' in pi3_state_dict:
                    pi3_state_dict = pi3_state_dict['model']
            except Exception as e:
                logger.error(f"Failed to load PyTorch checkpoint from {file_path}: {e}")
                return {}, [], []
        else:
            # 可能是 HuggingFace 路径，尝试加载
            try:
                # 尝试从 pi3 包导入
                try:
                    from pi3.models.pi3 import Pi3
                    logger.info(f"Loading Pi3 model from HuggingFace path: {file_path}...")
                    pi3_model = Pi3.from_pretrained(file_path)
                    pi3_state_dict = pi3_model.state_dict()
                except ImportError:
                    # 如果无法导入，尝试直接加载权重文件（假设是 .pt/.pth）
                    logger.warning(f"Cannot import Pi3 module, trying to load weights directly from {file_path}")
                    try:
                        pi3_state_dict = torch.load(file_path, map_location=device, weights_only=False)
                        if isinstance(pi3_state_dict, dict) and 'model' in pi3_state_dict:
                            pi3_state_dict = pi3_state_dict['model']
                    except Exception as e2:
                        logger.error(f"Failed to load weights directly: {e2}")
                        return {}, [], []
            except Exception as e:
                logger.error(f"Failed to load Pi3 from {file_path}: {e}")
                return {}, [], []
    else:
        # 直接使用模型实例
        pi3_state_dict = pi3_model_or_path.state_dict()
    
    model_state_dict = model.state_dict()
    mapped_state_dict = {}
    unmapped_keys = []
    
    # Pi3 权重映射规则：
    # decoder.blocks.{i}.* (i % 2 == 0) -> aggregator.frame_blocks.{i//2}.* (用于 Time-SA)
    # decoder.blocks.{i}.* (i % 2 == 1) -> aggregator.global_blocks.{i//2}.* (用于 View-SA)
    # encoder.register_token -> aggregator.register_token
    # encoder.patch_embed.* -> aggregator.patch_embed.* (如果匹配)
    
    for pi3_name, pi3_param in pi3_state_dict.items():
        target_name = None
        
        # 映射 decoder.{i}.*
        # Pi3 的键名格式是 decoder.{i}.attn.* 或 decoder.{i}.mlp.* 等
        if pi3_name.startswith('decoder.'):
            try:
                parts = pi3_name.split('.')
                # 格式: decoder.{i}.attn.* 或 decoder.{i}.mlp.* 等
                if len(parts) >= 2:
                    try:
                        block_idx = int(parts[1])  # decoder.{i}
                    except ValueError:
                        unmapped_keys.append(pi3_name)
                        continue
                    
                    # Pi3 的交替模式：偶数索引是 frame attention (view内空间), 奇数索引是 global attention (跨view)
                    # 在 VGGT_MV 中：
                    # - frame_blocks 用于 Time-SA (固定视角，跨时间)
                    # - global_blocks 用于 View-SA (固定时间，跨视角)
                    if block_idx % 2 == 0:
                        # Frame attention (view内) -> Time-SA (frame_blocks)
                        target_block_idx = block_idx // 2
                        # 构建目标键名: aggregator.frame_blocks.{target_block_idx}.{rest}
                        rest_parts = parts[2:]  # attn.* 或 mlp.* 等
                        target_parts = ['aggregator', 'frame_blocks', str(target_block_idx)] + rest_parts
                        target_name = '.'.join(target_parts)
                    else:
                        # Global attention (跨view) -> View-SA (global_blocks)
                        target_block_idx = block_idx // 2
                        rest_parts = parts[2:]
                        target_parts = ['aggregator', 'global_blocks', str(target_block_idx)] + rest_parts
                        target_name = '.'.join(target_parts)
            except Exception as e:
                logger.warning(f"Failed to parse decoder from {pi3_name}: {e}")
                unmapped_keys.append(pi3_name)
                continue
        
        # 映射 encoder.register_token
        elif pi3_name == 'register_token' or pi3_name == 'encoder.register_token':
            target_name = 'aggregator.register_token'
        
        # 映射 encoder.patch_embed.* (如果结构匹配)
        elif pi3_name.startswith('encoder.patch_embed.'):
            # 尝试直接映射到 aggregator.patch_embed.*
            target_name = pi3_name.replace('encoder.patch_embed.', 'aggregator.patch_embed.', 1)
        
        # 其他 encoder.* 键暂时跳过（如 encoder.blocks.* 可能对应 DINOv2 编码器，已经在 checkpoint 中）
        elif pi3_name.startswith('encoder.'):
            # encoder 的其他部分（blocks 等）可能已经在 checkpoint 中加载，跳过
            unmapped_keys.append(pi3_name)
            continue
        
        # 跳过 decoder、camera_decoder、point_decoder 等（这些是 Pi3 特有的结构）
        elif pi3_name.startswith(('camera_decoder.', 'point_decoder.', 'conf_decoder.', 
                                   'camera_head.', 'point_head.', 'conf_head.', 
                                   'image_mean', 'image_std')):
            unmapped_keys.append(pi3_name)
            continue
        
        # 检查目标名称是否存在且形状匹配
        if target_name:
            if target_name in model_state_dict:
                if model_state_dict[target_name].shape == pi3_param.shape:
                    mapped_state_dict[target_name] = pi3_param
                    logger.debug(f"Mapped: {pi3_name} -> {target_name}")
                else:
                    logger.warning(f"Shape mismatch: Pi3 {pi3_name} ({pi3_param.shape}) vs model {target_name} ({model_state_dict[target_name].shape})")
                    unmapped_keys.append(pi3_name)
            else:
                logger.debug(f"Target key not found in model: {target_name} (from Pi3 {pi3_name})")
                unmapped_keys.append(pi3_name)
        else:
            # 没有找到映射规则
            if not pi3_name.startswith(('camera_decoder.', 'point_decoder.', 'conf_decoder.', 
                                        'camera_head.', 'point_head.', 'conf_head.', 
                                        'image_mean', 'image_std', 'encoder.')):
                # 只记录未预期的键（排除已知的 Pi3 特有结构）
                unmapped_keys.append(pi3_name)
    
    # 加载权重
    if mapped_state_dict:
        missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = [], []
    
    logger.info(f"Loaded {len(mapped_state_dict)} parameters from Pi3")
    if missing_keys:
        logger.info(f"Missing {len(missing_keys)} keys (using random init)")
    if unexpected_keys:
        logger.info(f"Unexpected {len(unexpected_keys)} keys (ignored)")
    if unmapped_keys:
        logger.debug(f"Unmapped {len(unmapped_keys)} Pi3 keys (not used in VGGT_MV)")
    
    # 打印一些映射示例
    if mapped_state_dict:
        sample_keys = list(mapped_state_dict.keys())[:5]
        logger.info(f"Sample Pi3 mapped keys: {sample_keys}")
    
    return mapped_state_dict, missing_keys, unexpected_keys


def adapt_weights_dimension(source_param: torch.Tensor, target_shape: torch.Size, 
                           strategy: str = 'truncate') -> torch.Tensor:
    """
    适配权重维度（当源权重和目标维度不匹配时）
    
    Args:
        source_param: 源权重参数
        target_shape: 目标形状
        strategy: 适配策略
            - 'truncate': 截断到目标维度（用于维度大于目标的情况）
            - 'pad': 填充到目标维度（用于维度小于目标的情况）
            - 'linear': 线性映射
            
    Returns:
        适配后的权重
    """
    if source_param.shape == target_shape:
        return source_param
    
    if strategy == 'truncate':
        # 截断到目标维度
        slices = [slice(0, s) for s in target_shape]
        return source_param[tuple(slices)]
    elif strategy == 'pad':
        # 填充到目标维度
        pad_sizes = [(0, max(0, t - s)) for s, t in zip(source_param.shape, target_shape)]
        return torch.nn.functional.pad(source_param, [item for sublist in reversed(pad_sizes) for item in sublist])
    elif strategy == 'linear':
        # 线性映射（适用于 1D 或 2D 权重）
        if len(source_param.shape) == 2 and len(target_shape) == 2:
            # 线性层权重：使用线性插值
            source_dim, target_dim = source_param.shape[1], target_shape[1]
            if source_dim > target_dim:
                return source_param[:, :target_dim]
            else:
                # 填充（使用零或复制最后一个值）
                padding = torch.zeros(source_param.shape[0], target_dim - source_dim, 
                                    device=source_param.device, dtype=source_param.dtype)
                return torch.cat([source_param, padding], dim=1)
        else:
            # 降级到截断策略
            slices = [slice(0, s) for s in target_shape]
            return source_param[tuple(slices)]
    else:
        raise ValueError(f"Unknown adaptation strategy: {strategy}")

