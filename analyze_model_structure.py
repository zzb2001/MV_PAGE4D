"""
分析 model_mv 结构和 Pi3 权重加载问题
"""
import torch
import sys
import os

# 添加路径
sys.path.insert(0, '/home/star/zzb/PAGE4D/page-4d')

from vggt_t_mv.models.vggt import VGGT as VGGT_MV
from vggt_t_mv.models.aggregator import Aggregator

print("="*80)
print("1. 分析 model_mv 结构")
print("="*80)

# 初始化模型
model_mv = VGGT_MV(
    img_size=378,
    patch_size=14,
    embed_dim=1024,
    enable_camera=True,
    enable_point=True,
    enable_depth=True,
    enable_track=True
)

print("\n模型层级结构:")
def print_model_structure(model, prefix="", max_depth=3, current_depth=0):
    """递归打印模型结构"""
    if current_depth >= max_depth:
        return
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # 打印模块信息
        if isinstance(module, torch.nn.ModuleList):
            print(f"{'  ' * current_depth}{full_name}: ModuleList({len(module)} items)")
            if current_depth < max_depth - 1:
                print_model_structure(module[0] if len(module) > 0 else None, 
                                     f"{full_name}[0]", max_depth, current_depth+1)
        elif isinstance(module, torch.nn.Parameter):
            print(f"{'  ' * current_depth}{full_name}: Parameter(shape={module.shape})")
        else:
            # 统计参数量
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{'  ' * current_depth}{full_name}: {type(module).__name__} ({num_params:,} params)")
            
            # 递归打印子模块
            if current_depth < max_depth - 1 and len(list(module.children())) > 0:
                print_model_structure(module, full_name, max_depth, current_depth+1)

print_model_structure(model_mv, max_depth=4)

print("\n" + "="*80)
print("2. 分析 Aggregator 的关键模块")
print("="*80)

aggregator = model_mv.aggregator

print(f"\nAggregator 关键组件:")
print(f"  - patch_embed: {type(aggregator.patch_embed).__name__}")
print(f"  - time_blocks: {len(aggregator.time_blocks)} layers")
print(f"  - view_blocks: {len(aggregator.view_blocks)} layers")
print(f"  - camera_token: {aggregator.camera_token.shape}")
print(f"  - register_token: {aggregator.register_token.shape}")
print(f"  - spatial_mask_head: {type(aggregator.spatial_mask_head).__name__}")
print(f"  - enable_dual_stream: {aggregator.enable_dual_stream}")

# 统计 time_blocks 的参数
if len(aggregator.time_blocks) > 0:
    first_time_block = aggregator.time_blocks[0]
    print(f"\ntime_blocks[0] 结构:")
    for name, param in first_time_block.named_parameters():
        print(f"  {name}: {param.shape}")
    print(f"  总参数量: {sum(p.numel() for p in first_time_block.parameters()):,}")

# 统计 view_blocks 的参数
if len(aggregator.view_blocks) > 0:
    first_view_block = aggregator.view_blocks[0]
    print(f"\nview_blocks[0] 结构:")
    for name, param in first_view_block.named_parameters():
        print(f"  {name}: {param.shape}")
    print(f"  总参数量: {sum(p.numel() for p in first_view_block.parameters()):,}")

print("\n" + "="*80)
print("3. 分析模型总参数量")
print("="*80)

total_params = sum(p.numel() for p in model_mv.parameters())
trainable_params = sum(p.numel() for p in model_mv.parameters() if p.requires_grad)

print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# 按模块统计
print("\n按模块统计参数量:")
for name, module in model_mv.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {params:,} ({params/total_params*100:.1f}%)")

print("\n" + "="*80)
print("4. 分析 Pi3 权重结构")
print("="*80)

pi3_path = '/home/star/zzb/Pi3/ckpts/model.safetensors'

if os.path.exists(pi3_path):
    print(f"\n加载 Pi3 权重文件: {pi3_path}")
    
    try:
        from safetensors.torch import load_file as safetensors_load
        
        pi3_weights = safetensors_load(pi3_path)
        print(f"成功加载 Pi3 权重，共 {len(pi3_weights)} 个键")
        
        # 分析键名结构
        print("\nPi3 权重键名结构分析:")
        
        # 统计键名前缀
        prefixes = {}
        for key in pi3_weights.keys():
            prefix = key.split('.')[0] if '.' in key else key
            if prefix not in prefixes:
                prefixes[prefix] = 0
            prefixes[prefix] += 1
        
        print("\n按前缀分类:")
        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count} keys")
            # 打印前几个示例
            examples = [k for k in pi3_weights.keys() if k.startswith(prefix + '.')][:3]
            for ex in examples:
                print(f"    - {ex}: {pi3_weights[ex].shape}")
        
        # 详细分析 decoder 结构
        if 'decoder' in prefixes:
            print("\ndecoder 结构分析:")
            decoder_keys = [k for k in pi3_weights.keys() if k.startswith('decoder.')]
            
            # 提取 block 索引
            import re
            block_indices = set()
            for key in decoder_keys:
                match = re.match(r'decoder\.(\d+)\.', key)
                if match:
                    block_indices.add(int(match.group(1)))
            
            block_indices = sorted(block_indices)
            print(f"  decoder blocks 索引: {block_indices}")
            print(f"  总共 {len(block_indices)} 个 decoder blocks")
            
            # 分析第一个 block 的结构
            if len(block_indices) > 0:
                first_block_idx = block_indices[0]
                block_keys = [k for k in decoder_keys if k.startswith(f'decoder.{first_block_idx}.')]
                print(f"\n  decoder.{first_block_idx} 结构:")
                for key in sorted(block_keys)[:20]:  # 只显示前20个
                    param_name = key.replace(f'decoder.{first_block_idx}.', '')
                    print(f"    {param_name}: {pi3_weights[key].shape}")
        
        # 分析 encoder 结构
        if 'encoder' in prefixes:
            print("\nencoder 结构分析:")
            encoder_keys = [k for k in pi3_weights.keys() if k.startswith('encoder.')]
            
            # 分析 patch_embed
            patch_embed_keys = [k for k in encoder_keys if 'patch_embed' in k]
            if patch_embed_keys:
                print(f"\n  patch_embed 结构 ({len(patch_embed_keys)} keys):")
                for key in sorted(patch_embed_keys)[:10]:
                    param_name = key.replace('encoder.patch_embed.', '')
                    print(f"    {param_name}: {pi3_weights[key].shape}")
            
            # 检查 register_token
            if 'register_token' in pi3_weights:
                print(f"\n  register_token: {pi3_weights['register_token'].shape}")
            if 'encoder.register_token' in pi3_weights:
                print(f"\n  encoder.register_token: {pi3_weights['encoder.register_token'].shape}")
        
        # 分析 camera_head, point_head 等
        for head_name in ['camera_head', 'point_head', 'conf_head', 'camera_decoder', 'point_decoder', 'conf_decoder']:
            if head_name in prefixes:
                print(f"\n{head_name} 结构 ({prefixes[head_name]} keys):")
                head_keys = [k for k in pi3_weights.keys() if k.startswith(f'{head_name}.')]
                for key in sorted(head_keys)[:5]:
                    param_name = key.replace(f'{head_name}.', '')
                    print(f"    {param_name}: {pi3_weights[key].shape}")
        
    except ImportError:
        print("错误: 无法导入 safetensors 库")
        print("请安装: pip install safetensors")
    except Exception as e:
        print(f"加载 Pi3 权重时出错: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\nPi3 文件不存在: {pi3_path}")

print("\n" + "="*80)
print("5. 尝试加载权重并诊断问题")
print("="*80)

if os.path.exists(pi3_path):
    try:
        from vggt_t_mv.utils.weight_loading import load_pi3_weights
        
        print("\n尝试使用 load_pi3_weights 加载...")
        pi3_dict, missing_pi3, unexpected_pi3 = load_pi3_weights(
            pi3_path, model_mv.aggregator, device='cpu'
        )
        
        print(f"\n加载结果:")
        print(f"  成功映射: {len(pi3_dict)} 个参数")
        print(f"  缺失: {len(missing_pi3)} 个")
        print(f"  意外: {len(unexpected_pi3)} 个")
        
        if pi3_dict:
            print(f"\n成功映射的参数示例:")
            for i, (key, param) in enumerate(list(pi3_dict.items())[:10]):
                print(f"  {key}: {param.shape}")
        
        if missing_pi3:
            print(f"\n缺失的参数示例:")
            for i, key in enumerate(missing_pi3[:20]):
                print(f"  {key}")
        
        if unexpected_pi3:
            print(f"\n意外的参数示例:")
            for i, key in enumerate(unexpected_pi3[:20]):
                print(f"  {key}")
                
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("分析完成")
print("="*80)

