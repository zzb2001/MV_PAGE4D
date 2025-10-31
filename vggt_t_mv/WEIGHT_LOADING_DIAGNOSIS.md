# 权重加载问题诊断

## 问题现象
```
checkpoint_loaded = 0
pi3_loaded = 0
missing = 2082
```

## 根本原因

### 1. 原代码的问题
```python
# 原代码（错误）
load_checkpoint_weights(checkpoint_path, self.aggregator, device=device)
```
- **只加载到 aggregator**，但 checkpoint 包含整个模型的权重
- **缺失的权重**：
  - `camera_head.*` (~400 参数)
  - `point_head.*` (~800 参数)
  - `depth_head.*` (~800 参数)
  - `track_head.*` (~80 参数)
  - 总计约 2082 个参数未加载

### 2. 权重键名不匹配的原因
- Checkpoint 的键名：`aggregator.*`, `camera_head.*`, `point_head.*` 等
- 传入 `self.aggregator.state_dict()` 只包含 `aggregator.*` 的键
- 所以 `camera_head.*` 等在 aggregator 的 state_dict 中找不到
- 导致 `name in model_state_dict` 检查失败，所有权重都被跳过

### 3. Pi3 权重加载失败的可能原因
1. Pi3 safetensors 文件格式问题（已修复）
2. 键名映射逻辑问题：`decoder.*` → `aggregator.*` 的映射可能不完整
3. 形状不匹配：Pi3 的 block 结构与 VGGT_MV 不完全一致

## 修复后的逻辑

### 新代码（正确）
```python
# 修复后
def load_pretrained_weights(self, checkpoint_path, pi3_path=None, device='cpu'):
    # 1. 加载整个模型的 checkpoint 权重
    checkpoint_dict = torch.load(checkpoint_path, ...)['model']
    model_dict = self.state_dict()  # 整个模型的权重
    
    # 匹配并加载所有权重（包括 aggregator 和 heads）
    for key, param in checkpoint_dict.items():
        if key in model_dict and model_dict[key].shape == param.shape:
            mapped_dict[key] = param
    
    self.load_state_dict(mapped_dict, strict=False)
    
    # 2. 只加载 Pi3 权重到 aggregator（因为 Pi3 没有 heads）
    load_pi3_weights(pi3_path, self.aggregator, device)
```

## 预期结果（修复后）

### 成功加载的权重
```
checkpoint_loaded = ~2500-3000 (包括 aggregator + heads)
pi3_loaded = ~500-1000 (aggregator 的部分权重)
missing = ~0-100 (新增的参数，如两流架构、sparse global 等)
```

### 权重分布
- **Aggregator**: ~1500-2000 参数
  - frame_blocks: ~800
  - global_blocks: ~800
  - patch_embed: ~200
  - 其他: ~200
- **Heads**: ~1000 参数
  - camera_head: ~200
  - point_head: ~400
  - depth_head: ~400
  - track_head: ~80

## 验证方法

运行以下代码检查权重加载情况：
```python
# 在 inference.py 中添加
stats = model_mv.load_pretrained_weights(...)
print(f"Stats: {stats}")

# 检查模型参数
total_params = sum(p.numel() for p in model_mv.parameters())
trainable_params = sum(p.numel() for p in model_mv.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 检查特定模块的权重是否加载
aggregator_params = sum(p.numel() for p in model_mv.aggregator.parameters())
camera_params = sum(p.numel() for p in model_mv.camera_head.parameters()) if model_mv.camera_head else 0
print(f"Aggregator parameters: {aggregator_params:,}")
print(f"Camera head parameters: {camera_params:,}")
```



