# 功能实现总结

## 1. 两流架构（位姿流 vs 几何流）

### 功能描述
实现了两流并行架构，用于处理动态场景：
- **位姿流（Pose Stream）**：抑制动态区域，专注于静态特征用于相机位姿估计
- **几何流（Geometry Stream）**：放大动态区域，用于点云和深度估计

### 实现位置
- `vggt_t_mv/models/aggregator.py`:
  - `__init__`: 添加 `enable_dual_stream` 参数和相关模块
  - 创建 `pose_frame_blocks`, `pose_global_blocks`, `geo_frame_blocks`, `geo_global_blocks`
  - 添加可学习的动态掩码偏置参数 `lambda_pose_logit`, `lambda_geo_logit`
  - `_process_dual_stream_attention`: 并行处理两流
  - `_apply_mask_bias`: 应用动态掩码偏置到注意力 logits

### 使用方法
```python
model = VGGT_MV(
    aggregator=Aggregator(
        enable_dual_stream=True,  # 启用两流架构
        ...
    )
)
```

### 输出
- `predictions['dual_stream_outputs']`: 包含 `'pose'` 和 `'geo'` 两个流的中间输出
- 位姿头（camera_head）应使用 pose 流输出
- 点云/深度头应使用 geo 流输出

## 2. Sparse Global-SA

### 功能描述
实现全局稀疏长程依赖机制，支持三种策略：
- **Landmark/Anchor Attention**: 选择高置信度的 landmark tokens 进行全局注意力
- **Block-Sparse + Dilated**: 使用扩张采样在 (t, v) 网格中进行稀疏注意力
- **Memory Bank**: 维护跨窗口的 memory tokens 用于长程依赖

### 实现位置
- `vggt_t_mv/models/aggregator.py`:
  - `__init__`: 添加 `enable_sparse_global`, `sparse_global_layers`, `sparse_strategy` 参数
  - `_process_sparse_global_attention`: 实现三种策略的核心逻辑
  - `_select_landmark_tokens`: 选择 landmark tokens
  - `_get_dilated_neighbors`: 计算扩张邻居

### 使用方法
```python
model = VGGT_MV(
    aggregator=Aggregator(
        enable_sparse_global=True,
        sparse_global_layers=[23, 24],  # 在最后两层应用
        sparse_strategy="landmark",  # 或 "block_dilated" 或 "memory_bank"
        ...
    )
)
```

## 3. 极线/几何先验

### 功能描述
- **极线约束偏置**: 在 View-SA 中应用极线几何约束，引导跨视角特征匹配
- **Plücker 光线角度加权**: 基于 3D 几何先验进行角度加权

### 实现位置
- `vggt_t_mv/utils/epipolar_utils.py`:
  - `compute_epipolar_bias`: 计算极线约束偏置
  - `compute_plucker_angle_weight`: 计算 Plücker 光线角度权重
- `vggt_t_mv/models/aggregator.py`:
  - `__init__`: 添加 `enable_epipolar_prior` 参数
  - `_compute_epipolar_bias_for_view`: 为 View-SA 计算极线偏置
  - `forward`: 在 View-SA 处理中集成极线先验

### 使用方法
```python
# 在 forward 时提供相机参数
result = aggregator(
    images=images,
    camera_intrinsics=K,  # [B, T, N, 3, 3]
    camera_poses=T_cam    # [B, T, N, 4, 4]
)
```

或在模型初始化时启用：
```python
model = VGGT_MV(
    aggregator=Aggregator(
        enable_epipolar_prior=True,
        ...
    )
)
```

## 4. 从 Pi3 加载权重

### 功能描述
完整的权重加载系统，支持：
- 从 `checkpoint_150.pt` 加载 VGGT 预训练权重
- 从 Pi3 模型加载多视角聚合权重
- 智能权重映射和维度适配
- 两流架构的权重初始化

### 实现位置
- `vggt_t_mv/utils/weight_loading.py`:
  - `load_checkpoint_weights`: 从 checkpoint 加载权重
  - `load_pi3_weights`: 从 Pi3 模型加载权重
  - `adapt_weights_dimension`: 权重维度适配
- `vggt_t_mv/models/vggt.py`:
  - `load_pretrained_weights`: 统一的权重加载接口
  - `_copy_block_weights`: 复制 block 权重用于两流初始化

### 使用方法
```python
# 在 inference.py 或训练脚本中
stats = model.load_pretrained_weights(
    checkpoint_path="checkpoint/checkpoint_150.pt",
    pi3_path="facebook/Pi3",  # 或本地路径 "/path/to/pi3/model"
    device="cuda"
)
print(f"Loaded: checkpoint={stats['checkpoint_loaded']}, pi3={stats['pi3_loaded']}")
```

### 权重映射规则
- **Checkpoint → Model**:
  - `frame_blocks` → `time_blocks` (Time-SA, 时序建模)
  - `global_blocks` → `view_blocks` (View-SA, 多视角聚合)
- **Pi3 → Model**:
  - `decoder[i]` (i % 2 == 0) → `frame_blocks[i//2]` (Time-SA)
  - `decoder[i]` (i % 2 == 1) → `global_blocks[i//2]` (View-SA)
  - `register_token` → `register_token`

## 注意事项

1. **两流架构**: 会显著增加模型参数和计算量（约2倍），需要更多显存
2. **Sparse Global-SA**: 需要根据实际任务选择合适的策略和层
3. **极线先验**: 需要准确的相机内参和位姿，否则可能影响性能
4. **权重加载**: Pi3 权重路径需要确保正确，如果无法导入 Pi3 模块会尝试直接加载权重文件

## 示例配置

```python
# 完整的多视图时序模型配置
model = VGGT_MV(
    aggregator=Aggregator(
        # 基础配置
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        
        # 多视图配置
        aa_order=["view", "time"],  # 或 ["time", "view"]
        
        # 两流架构（可选）
        enable_dual_stream=True,
        
        # Sparse Global-SA（可选）
        enable_sparse_global=True,
        sparse_global_layers=[23, 24],
        sparse_strategy="landmark",
        
        # 极线先验（可选）
        enable_epipolar_prior=True,
    )
)

# 加载权重
stats = model.load_pretrained_weights(
    checkpoint_path="checkpoint/checkpoint_150.pt",
    pi3_path=None,  # 可选
    device="cuda"
)
```



