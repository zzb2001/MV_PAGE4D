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
        model: vggt_t_mv 模型实例或 Aggregator 实例
        device: 加载设备
        
    Returns:
        mapped_state_dict: 映射后的权重字典
        missing_keys: 缺失的键
        unexpected_keys: 意外的键
    """
    # 如果传入的是整个VGGT模型，提取aggregator；如果传入的是aggregator，直接使用
    if hasattr(model, 'aggregator'):
        # 传入的是VGGT模型
        aggregator = model.aggregator
        is_aggregator_only = False
    else:
        # 传入的就是aggregator
        aggregator = model
        is_aggregator_only = True
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
                    # Pi3 有 36 个 decoder blocks (0-35)
                    # VGGT_MV 有 24 个 frame_blocks 和 24 个 global_blocks
                    # 映射：decoder.{偶数} (0,2,4,...,34) -> frame_blocks.{0..17} (18个)
                    #       decoder.{奇数} (1,3,5,...,35) -> global_blocks.{0..17} (18个)
                    if block_idx % 2 == 0:
                        # Frame attention (view内) -> Time-SA (frame_blocks)
                        target_block_idx = block_idx // 2
                        # 确保不超过 frame_blocks 的范围（应该有 18 个，索引 0-17）
                        if target_block_idx >= len(aggregator.frame_blocks):
                            logger.debug(f"Pi3 decoder.{block_idx} -> frame_blocks.{target_block_idx} out of range (max {len(aggregator.frame_blocks)-1})")
                            unmapped_keys.append(pi3_name)
                            continue
                        # 构建目标键名
                        rest_parts = parts[2:]  # attn.* 或 mlp.* 等
                        if is_aggregator_only:
                            target_parts = ['frame_blocks', str(target_block_idx)] + rest_parts
                        else:
                            target_parts = ['aggregator', 'frame_blocks', str(target_block_idx)] + rest_parts
                        target_name = '.'.join(target_parts)
                    else:
                        # Global attention (跨view) -> View-SA (global_blocks)
                        target_block_idx = block_idx // 2
                        # 确保不超过 global_blocks 的范围（应该有 18 个，索引 0-17）
                        if target_block_idx >= len(aggregator.global_blocks):
                            logger.debug(f"Pi3 decoder.{block_idx} -> global_blocks.{target_block_idx} out of range (max {len(aggregator.global_blocks)-1})")
                            unmapped_keys.append(pi3_name)
                            continue
                        rest_parts = parts[2:]
                        if is_aggregator_only:
                            target_parts = ['global_blocks', str(target_block_idx)] + rest_parts
                        else:
                            target_parts = ['aggregator', 'global_blocks', str(target_block_idx)] + rest_parts
                        target_name = '.'.join(target_parts)
            except Exception as e:
                logger.warning(f"Failed to parse decoder from {pi3_name}: {e}")
                unmapped_keys.append(pi3_name)
                continue
        
        # 映射 encoder.register_token (需要形状适配)
        elif pi3_name == 'register_token' or pi3_name == 'encoder.register_token':
            if is_aggregator_only:
                target_name = 'register_token'
            else:
                target_name = 'aggregator.register_token'
            # 注意：Pi3是 [1,1,5,1024]，Model_MV是 [1,2,4,1024]，需要特殊处理
            # 这将在加载时通过adapt_weights_dimension处理
        
        # 映射 encoder.patch_embed.* (如果结构匹配)
        elif pi3_name.startswith('encoder.patch_embed.'):
            # 尝试直接映射到 aggregator.patch_embed.*
            if is_aggregator_only:
                target_name = pi3_name.replace('encoder.patch_embed.', 'patch_embed.', 1)
            else:
                target_name = pi3_name.replace('encoder.patch_embed.', 'aggregator.patch_embed.', 1)
        
        # 映射 encoder.blocks.* 到 patch_embed.blocks.*
        elif pi3_name.startswith('encoder.blocks.'):
            # encoder.blocks.* 应该映射到 patch_embed.blocks.*
            if is_aggregator_only:
                target_name = pi3_name.replace('encoder.blocks.', 'patch_embed.blocks.', 1)
            else:
                target_name = pi3_name.replace('encoder.blocks.', 'aggregator.patch_embed.blocks.', 1)
        
        # 映射 encoder.norm 到 patch_embed.norm
        elif pi3_name == 'encoder.norm':
            if is_aggregator_only:
                target_name = 'patch_embed.norm'
            else:
                target_name = 'aggregator.patch_embed.norm'
        
        # 其他 encoder.* 键
        elif pi3_name.startswith('encoder.'):
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
                    # 只记录前几个映射示例，避免日志过多
                    if len(mapped_state_dict) <= 5:
                        logger.info(f"Mapped: {pi3_name} -> {target_name}")
                else:
                    # 形状不匹配，跳过
                    logger.debug(f"Shape mismatch: Pi3 {pi3_name} ({pi3_param.shape}) vs model {target_name} ({model_state_dict[target_name].shape})")
                    unmapped_keys.append(pi3_name)
            else:
                # 键名不匹配
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
        logger.info(f"Unmapped {len(unmapped_keys)} Pi3 keys (not used in VGGT_MV)")
        # 显示一些未映射的示例
        if len(unmapped_keys) <= 10:
            logger.info(f"Unmapped keys: {unmapped_keys[:5]}")
    
    # 打印映射统计
    if mapped_state_dict:
        # 统计映射到的模块
        frame_count = len([k for k in mapped_state_dict.keys() if 'frame_blocks' in k])
        global_count = len([k for k in mapped_state_dict.keys() if 'global_blocks' in k])
        logger.info(f"Pi3 mapping: frame_blocks={frame_count}, global_blocks={global_count}")
    
    return mapped_state_dict, missing_keys, unexpected_keys


def adapt_weights_dimension(source_param: torch.Tensor, target_shape: torch.Size, 
                           strategy: str = 'submatrix') -> torch.Tensor:
    """
    适配权重维度（当源权重和目标维度不匹配时）
    
    Args:
        source_param: 源权重参数
        target_shape: 目标形状
        strategy: 适配策略
            - 'submatrix': 子矩阵拷贝（取前 C' 通道），优先保持线性层形状一致
            - 'linear_adapt': 线性适配（1×1映射一次性初始化）
            - 'truncate': 截断到目标维度（向后兼容）
            
    Returns:
        适配后的权重
    """
    if source_param.shape == target_shape:
        return source_param
    
    if strategy == 'submatrix':
        # 子矩阵拷贝策略：优先保持线性层形状一致，取前 C' 通道
        # 适用于：Linear权重、Conv权重、LayerNorm等
        if len(source_param.shape) == 2 and len(target_shape) == 2:
            # Linear权重 [out_features, in_features]
            out_s, in_s = source_param.shape
            out_t, in_t = target_shape
            
            if out_s >= out_t and in_s >= in_t:
                # 源维度更大，取子矩阵
                return source_param[:out_t, :in_t].clone()
            elif out_s >= out_t and in_s < in_t:
                # 输出维度足够，输入维度不足：填充输入维度
                padding = torch.zeros(out_t, in_t - in_s, device=source_param.device, dtype=source_param.dtype)
                return torch.cat([source_param[:out_t, :], padding], dim=1)
            elif out_s < out_t and in_s >= in_t:
                # 输出维度不足，输入维度足够：填充输出维度
                padding = torch.zeros(out_t - out_s, in_t, device=source_param.device, dtype=source_param.dtype)
                return torch.cat([source_param[:, :in_t], padding], dim=0)
            else:
                # 两个维度都不足：填充
                param_padded = torch.cat([
                    torch.cat([source_param, torch.zeros(out_s, in_t - in_s, device=source_param.device, dtype=source_param.dtype)], dim=1),
                    torch.zeros(out_t - out_s, in_t, device=source_param.device, dtype=source_param.dtype)
                ], dim=0)
                return param_padded
        elif len(source_param.shape) == 1 and len(target_shape) == 1:
            # 1D权重（bias, LayerNorm等）
            if source_param.shape[0] >= target_shape[0]:
                return source_param[:target_shape[0]].clone()
            else:
                padding = torch.zeros(target_shape[0] - source_param.shape[0], 
                                    device=source_param.device, dtype=source_param.dtype)
                return torch.cat([source_param, padding], dim=0)
        elif len(source_param.shape) == 4 and len(target_shape) == 4:
            # Conv权重 [out_channels, in_channels, kernel_h, kernel_w]
            out_s, in_s, kh_s, kw_s = source_param.shape
            out_t, in_t, kh_t, kw_t = target_shape
            
            if kh_s != kh_t or kw_s != kw_t:
                logger.warning(f"Kernel size mismatch: source {kh_s}x{kw_s} vs target {kh_t}x{kw_t}")
                # 降级到截断
                slices = [slice(0, s) for s in target_shape]
                return source_param[tuple(slices)]
            
            if out_s >= out_t and in_s >= in_t:
                return source_param[:out_t, :in_t, :, :].clone()
            elif out_s >= out_t:
                padding = torch.zeros(out_t, in_t - in_s, kh_t, kw_t, device=source_param.device, dtype=source_param.dtype)
                return torch.cat([source_param[:out_t, :, :, :], padding], dim=1)
            elif in_s >= in_t:
                padding = torch.zeros(out_t - out_s, in_t, kh_t, kw_t, device=source_param.device, dtype=source_param.dtype)
                return torch.cat([source_param[:, :in_t, :, :], padding], dim=0)
            else:
                param_padded = torch.cat([
                    torch.cat([source_param, torch.zeros(out_s, in_t - in_s, kh_t, kw_t, device=source_param.device, dtype=source_param.dtype)], dim=1),
                    torch.zeros(out_t - out_s, in_t, kh_t, kw_t, device=source_param.device, dtype=source_param.dtype)
                ], dim=0)
                return param_padded
        else:
            # 其他维度：降级到截断
            slices = [slice(0, min(s, t)) for s, t in zip(source_param.shape, target_shape)]
            result = source_param[tuple(slices)]
            # 如果还需要填充
            if any(s < t for s, t in zip(result.shape, target_shape)):
                pad_sizes = [(0, max(0, t - s)) for s, t in zip(result.shape, target_shape)]
                return torch.nn.functional.pad(result, [item for sublist in reversed(pad_sizes) for item in sublist])
            return result
    
    elif strategy == 'linear_adapt':
        # 线性适配：创建1×1映射层进行适配
        # 这需要创建临时映射层，然后提取权重
        if len(source_param.shape) == 2 and len(target_shape) == 2:
            # Linear权重
            out_s, in_s = source_param.shape
            out_t, in_t = target_shape
            
            # 创建适配层：target_dim -> source_dim -> target_dim
            if in_s != in_t:
                # 输入维度适配
                adapt_in = nn.Linear(in_t, in_s, bias=False)
                nn.init.eye_(adapt_in.weight[:min(in_s, in_t), :min(in_s, in_t)])
                source_adapted = torch.matmul(source_param, adapt_in.weight.T)  # [out_s, in_t]
            else:
                source_adapted = source_param
            
            if out_s != out_t:
                # 输出维度适配
                adapt_out = nn.Linear(out_s, out_t, bias=False)
                nn.init.eye_(adapt_out.weight[:min(out_s, out_t), :min(out_s, out_t)])
                result = torch.matmul(adapt_out.weight, source_adapted)  # [out_t, in_t]
            else:
                result = source_adapted
            
            return result.to(source_param.device).to(source_param.dtype)
        else:
            # 降级到submatrix策略
            return adapt_weights_dimension(source_param, target_shape, strategy='submatrix')
    
    elif strategy == 'truncate':
        # 向后兼容：简单截断
        slices = [slice(0, s) for s in target_shape]
        return source_param[tuple(slices)]
    
    else:
        raise ValueError(f"Unknown adaptation strategy: {strategy}")


def load_weights_three_stage(pi3_path: Optional[str], checkpoint_path: Optional[str], 
                            model: nn.Module, device: str = 'cpu',
                            page4d_mid_layers: Optional[List[int]] = None) -> Dict[str, int]:
    """
    三阶段权重加载策略：
    1. 先加载 Pi3（编码器、frame/global基础块、camera/point/conf头、tokens）
    2. 再有选择地覆盖 PAGE-4D（仅中段若干层与掩码头）
    3. 最后初始化新增模块（两流、稀疏全局等）
    
    Args:
        pi3_path: Pi3 模型路径
        checkpoint_path: PAGE-4D checkpoint 路径
        model: VGGT_MV 模型实例
        device: 加载设备
        page4d_mid_layers: PAGE-4D 要覆盖的中段层索引（如 [8, 9, 10, 11, 12, 13, 14, 15]）
            如果为 None，则覆盖所有匹配的层
            
    Returns:
        stats: 加载统计信息
    """
    stats = {
        'pi3_loaded': 0,
        'page4d_loaded': 0,
        'page4d_overwritten': 0,
        'missing': 0,
        'unexpected': 0
    }
    
    model_state_dict = model.state_dict()
    loaded_state_dict = {}  # 累积加载的权重
    
    # ========== 阶段1：加载 Pi3 权重 ==========
    if pi3_path:
        logger.info("=" * 60)
        logger.info("阶段1: 加载 Pi3 权重")
        logger.info("=" * 60)
        
        try:
            # 加载 Pi3 权重（包括编码器、基础块、头、tokens）
            pi3_dict, missing_pi3, unexpected_pi3 = load_pi3_weights_comprehensive(
                pi3_path, model, device=device
            )
            
            # 加载到模型（使用适配策略）
            for key, param in pi3_dict.items():
                if key in model_state_dict:
                    target_shape = model_state_dict[key].shape
                    
                    # 特殊处理 register_token: Pi3是 [1,1,5,1024]，Model_MV是 [1,2,4,1024]
                    if 'register_token' in key and param.shape[2] == 5:
                        # 取前4个tokens，然后扩展
                        param_4 = param[:, :, :4, :]  # [1, 1, 4, 1024]
                        # 扩展到 [1, 2, 4, 1024]: 第一帧和其余帧用相同的token
                        param_expanded = param_4.expand(1, 2, 4, 1024)
                        loaded_state_dict[key] = param_expanded
                        stats['pi3_loaded'] += 1
                        logger.info(f"Adapted register_token: Pi3 {param.shape} -> Model {target_shape}")
                    elif param.shape == target_shape:
                        loaded_state_dict[key] = param
                        stats['pi3_loaded'] += 1
                    else:
                        # 维度不匹配，尝试适配
                        try:
                            adapted_param = adapt_weights_dimension(
                                param, target_shape, strategy='submatrix'
                            )
                            loaded_state_dict[key] = adapted_param
                            stats['pi3_loaded'] += 1
                            logger.debug(f"Adapted Pi3 weight: {key} {param.shape} -> {target_shape}")
                        except Exception as e:
                            logger.warning(f"Failed to adapt Pi3 weight {key}: {e}")
            
            logger.info(f"阶段1完成: 加载 {stats['pi3_loaded']} 个 Pi3 参数")
        except Exception as e:
            logger.error(f"阶段1失败: {e}", exc_info=True)
    
    # ========== 阶段2：有选择地覆盖 PAGE-4D 权重 ==========
    if checkpoint_path:
        logger.info("=" * 60)
        logger.info("阶段2: 有选择地覆盖 PAGE-4D 权重")
        logger.info("=" * 60)
        
        try:
            # 加载 PAGE-4D checkpoint
            if checkpoint_path.endswith('.safetensors'):
                if not SAFETENSORS_AVAILABLE:
                    logger.error("safetensors library not installed")
                else:
                    page4d_dict = safetensors_load(checkpoint_path, device=device)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                page4d_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # 定义要覆盖的模块（中段层 + 掩码头）
            overwrite_keys = set()
            
            # 1. 掩码头（spatial_mask_head）
            for key in page4d_dict.keys():
                if 'spatial_mask_head' in key:
                    overwrite_keys.add(key)
            
            # 2. 中段层（time_blocks 和 view_blocks 的指定层）
            if page4d_mid_layers is not None:
                for layer_idx in page4d_mid_layers:
                    for prefix in ['aggregator.time_blocks', 'aggregator.view_blocks',
                                  'aggregator.frame_blocks', 'aggregator.global_blocks']:
                        for key in page4d_dict.keys():
                            if f'{prefix}.{layer_idx}.' in key:
                                overwrite_keys.add(key)
            else:
                # 如果没有指定层，覆盖所有中段层（排除前后几层）
                # 假设总共24层，中段为 8-15 层
                mid_layers = list(range(8, 16)) if page4d_mid_layers is None else page4d_mid_layers
                for layer_idx in mid_layers:
                    for prefix in ['aggregator.time_blocks', 'aggregator.view_blocks',
                                  'aggregator.frame_blocks', 'aggregator.global_blocks']:
                        for key in page4d_dict.keys():
                            if f'{prefix}.{layer_idx}.' in key:
                                overwrite_keys.add(key)
            
            logger.info(f"准备覆盖 {len(overwrite_keys)} 个 PAGE-4D 权重")
            
            # 覆盖权重
            for key in overwrite_keys:
                if key in page4d_dict:
                    param = page4d_dict[key]
                    if key in model_state_dict:
                        target_shape = model_state_dict[key].shape
                        if param.shape == target_shape:
                            loaded_state_dict[key] = param
                            stats['page4d_overwritten'] += 1
                        else:
                            # 修改7: 对于不匹配的线性层，使用子矩阵复制或1x1线性适配
                            try:
                                # 先尝试子矩阵复制策略
                                try:
                                    adapted_param = adapt_weights_dimension(
                                        param, target_shape, strategy='submatrix'
                                    )
                                    loaded_state_dict[key] = adapted_param
                                    stats['page4d_overwritten'] += 1
                                    logger.debug(f"Adapted PAGE-4D weight (submatrix): {key} {param.shape} -> {target_shape}")
                                except:
                                    # 如果子矩阵策略失败，尝试1x1线性适配（对于卷积或线性层）
                                    if len(param.shape) >= 2 and len(target_shape) >= 2:
                                        # 尝试截断或填充
                                        adapted_param = adapt_weights_dimension(
                                            param, target_shape, strategy='truncate'
                                        )
                                        loaded_state_dict[key] = adapted_param
                                        stats['page4d_overwritten'] += 1
                                        logger.debug(f"Adapted PAGE-4D weight (truncate): {key} {param.shape} -> {target_shape}")
                                    else:
                                        raise ValueError("Cannot adapt dimension")
                            except Exception as e:
                                logger.warning(f"Failed to adapt PAGE-4D weight {key}: {e}, using random init")
            
            # 加载其他 PAGE-4D 权重（未在覆盖列表中的）
            for key, param in page4d_dict.items():
                if key not in overwrite_keys and key not in loaded_state_dict:
                    if key in model_state_dict:
                        target_shape = model_state_dict[key].shape
                        if param.shape == target_shape:
                            loaded_state_dict[key] = param
                            stats['page4d_loaded'] += 1
            
            logger.info(f"阶段2完成: 覆盖 {stats['page4d_overwritten']} 个，加载 {stats['page4d_loaded']} 个 PAGE-4D 参数")
        except Exception as e:
            logger.error(f"阶段2失败: {e}", exc_info=True)
    
    # ========== 阶段3：初始化新增模块 ==========
    logger.info("=" * 60)
    logger.info("阶段3: 初始化新增模块")
    logger.info("=" * 60)
    
    # 加载累积的权重到模型
    if loaded_state_dict:
        missing_keys, unexpected_keys = model.load_state_dict(loaded_state_dict, strict=False)
        stats['missing'] = len(missing_keys)
        stats['unexpected'] = len(unexpected_keys)
        logger.info(f"总加载: {len(loaded_state_dict)} 个参数")
        if missing_keys:
            logger.info(f"缺失键: {len(missing_keys)} (使用随机初始化)")
        if unexpected_keys:
            logger.info(f"意外键: {len(unexpected_keys)} (已忽略)")
    
    # 修改7 & 修改1: 初始化两流架构（如果启用）
    # 修改1: 两流架构只存在于L_mid层（6-10），其他层共享权重
    if hasattr(model, 'aggregator') and model.aggregator.enable_dual_stream:
        logger.info("初始化两流架构权重（仅L_mid层）...")
        # 修改1: 两流blocks只存在于dual_stream_layers，长度应该匹配
        dual_stream_layers = getattr(model.aggregator, 'dual_stream_layers', [6, 7, 8, 9, 10])
        
        # 从主 blocks 复制到两流 blocks（仅L_mid层）
        if hasattr(model.aggregator, 'time_blocks'):
            source_blocks = model.aggregator.time_blocks
            if hasattr(model.aggregator, 'pose_time_blocks'):
                # 修改1: 只复制L_mid层对应的blocks
                for dual_idx, layer_idx in enumerate(dual_stream_layers):
                    if layer_idx < len(source_blocks) and dual_idx < len(model.aggregator.pose_time_blocks):
                        _copy_block_with_adaptation(source_blocks[layer_idx], model.aggregator.pose_time_blocks[dual_idx])
            if hasattr(model.aggregator, 'geo_time_blocks'):
                for dual_idx, layer_idx in enumerate(dual_stream_layers):
                    if layer_idx < len(source_blocks) and dual_idx < len(model.aggregator.geo_time_blocks):
                        _copy_block_with_adaptation(source_blocks[layer_idx], model.aggregator.geo_time_blocks[dual_idx])
        
        if hasattr(model.aggregator, 'view_blocks'):
            source_blocks = model.aggregator.view_blocks
            if hasattr(model.aggregator, 'pose_view_blocks'):
                for dual_idx, layer_idx in enumerate(dual_stream_layers):
                    if layer_idx < len(source_blocks) and dual_idx < len(model.aggregator.pose_view_blocks):
                        _copy_block_with_adaptation(source_blocks[layer_idx], model.aggregator.pose_view_blocks[dual_idx])
            if hasattr(model.aggregator, 'geo_view_blocks'):
                for dual_idx, layer_idx in enumerate(dual_stream_layers):
                    if layer_idx < len(source_blocks) and dual_idx < len(model.aggregator.geo_view_blocks):
                        _copy_block_with_adaptation(source_blocks[layer_idx], model.aggregator.geo_view_blocks[dual_idx])
        logger.info(f"两流架构初始化完成（L_mid层: {dual_stream_layers}）")
    
    # 修改7: 新增组件随机初始化
    # Sparse Global blocks 已在 __init__ 中初始化（使用随机初始化）
    # lambda_pose_logit 和 lambda_geo_logit 已在 __init__ 中初始化为 0.0（修改3）
    # Dynamic Mask Head 如果结构不匹配，使用随机初始化
    # Memory tokens 使用随机初始化
    
    logger.info("=" * 60)
    logger.info("三阶段权重加载完成")
    logger.info("=" * 60)
    
    return stats


def load_pi3_weights_comprehensive(pi3_path: str, model: nn.Module, device: str = 'cpu') -> Tuple[Dict, List, List]:
    """
    从 Pi3 全面加载权重：编码器、frame/global基础块、camera/point/conf头、tokens
    
    Args:
        pi3_path: Pi3 模型路径
        model: VGGT_MV 模型实例（整个模型）
        device: 加载设备
        
    Returns:
        mapped_state_dict, missing_keys, unexpected_keys
    """
    # 加载 Pi3 权重到整个模型
    return load_pi3_weights(pi3_path, model, device=device)


def _copy_block_with_adaptation(source_block: nn.Module, target_block: nn.Module):
    """
    复制 block 权重，支持维度适配
    
    Args:
        source_block: 源 block
        target_block: 目标 block
    """
    source_state = source_block.state_dict()
    target_state = target_block.state_dict()
    
    for key in target_state:
        if key in source_state:
            source_param = source_state[key]
            target_shape = target_state[key].shape
            
            if source_param.shape == target_shape:
                target_state[key] = source_param.clone()
            else:
                try:
                    target_state[key] = adapt_weights_dimension(
                        source_param, target_shape, strategy='submatrix'
                    )
                except Exception as e:
                    logger.warning(f"Cannot copy {key}: {e}")
    
    target_block.load_state_dict(target_state)

