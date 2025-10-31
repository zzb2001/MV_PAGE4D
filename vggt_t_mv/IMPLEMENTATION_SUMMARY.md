# vggt_t_mv 实现总结

## 一、已完成的修改

### 1.1 导入路径迁移
- ✅ 将所有 `vggt_t_mask_mlp_fin10` 的引用改为 `vggt_t_mv`
- ✅ 更新了 `vggt.py` 和 `aggregator.py` 中的所有导入语句

### 1.2 输入格式支持
- ✅ 支持 `[B, S, C, H, W]` 格式（向后兼容单视角时序）
- ✅ 支持 `[B, T, N, C, H, W]` 格式（新的时序多视角格式）
- ✅ 自动检测输入格式并选择相应的处理路径

### 1.3 View-SA 和 Time-SA 实现
- ✅ **View-SA**：固定时刻 t，跨视角聚合
  - 形状变换：`[B, T, N, P, C] → [B*T, N*P, C] → MHA → [B, T, N, P, C]`
  - 使用 `global_blocks`（可加载 Pi3 Global-Attention 权重）
  - 实现方法：`_process_view_attention()`

- ✅ **Time-SA**：固定视角 v，跨时聚合
  - 形状变换：`[B, T, N, P, C] → [B*N, T*P, C] → MHA → [B, T, N, P, C]`
  - 使用 `time_blocks`（别名到 `frame_blocks`，可加载 VGGT Frame-Attention 权重）
  - 实现方法：`_process_time_attention()`

### 1.4 交替聚合机制
- ✅ 支持 `aa_order=["view", "time"]` 或 `["time", "view"]`
- ✅ 向后兼容 `aa_order=["frame", "global"]`（自动转换为 view/time）
- ✅ 实现了交替聚合循环，自动选择 View-SA 或 Time-SA

### 1.5 权重加载支持
- ✅ 支持从 `checkpoint/checkpoint_150.pt` 加载权重
- ✅ 自动映射权重名称（frame_blocks → time_blocks，global_blocks → view_blocks）
- ✅ 使用 `strict=False` 允许新参数随机初始化

### 1.6 inference.py 更新
- ✅ 读取 `data/t/time_*/view*.png` 格式数据
- ✅ 转换为 `[B, T, N, C, H, W]` 格式
- ✅ 初始化 VGGT_MV 模型并加载权重
- ✅ 执行推理并输出结果

## 二、核心架构说明

### 2.1 聚合器结构

```
Aggregator
├── patch_embed (DINOv2 encoder)
├── frame_blocks (用于 Time-SA)
├── global_blocks (用于 View-SA)
├── view_blocks (别名到 global_blocks)
├── time_blocks (别名到 frame_blocks)
├── spatial_mask_head (动态掩码生成)
└── camera_token, register_token (特殊 tokens)
```

### 2.2 前向传播流程（多视角模式）

```
Input: [B, T, N, C, H, W]
  ↓
Patch Embedding: [B*T*N, P, C]
  ↓
Organize: [B, T, N, P, C] + special tokens → [B, T, N, 1+R+P, C]
  ↓
For each block:
  ├─ View-SA: [B, T, N, P, C] → [B*T, N*P, C] → MHA → [B, T, N, P, C]
  └─ Time-SA: [B, T, N, P, C] → [B*N, T*P, C] → MHA → [B, T, N, P, C]
  ↓
Concat intermediates: [B, T, N, P, 2C]
  ↓
Output: List of [B, T, N, P, 2C]
```

## 三、权重复用策略

### 3.1 从 checkpoint_150.pt 加载
- `frame_blocks.*` → 用于 Time-SA（时序建模）
- `global_blocks.*` → 用于 View-SA（多视角聚合）
- `patch_embed.*` → DINOv2 encoder（完全复用）
- `camera_token`, `register_token` → 特殊 tokens
- `spatial_mask_head.*` → 动态掩码生成器

### 3.2 从 Pi3 加载（可选）
- Pi3 的 `Global-Attention` → 用于 View-SA blocks
- Pi3 的 `Frame-Attention` → 作为 Time-SA 的初始化参考
- Pi3 的 `register_token` → 如果维度匹配

### 3.3 新增参数（需训练）
- View-SA 和 Time-SA 中的 mask 偏置参数（如果实现两流架构）
- 动态掩码生成器的某些参数（如果需要）

## 四、使用方式

### 4.1 单视角时序模式（向后兼容）
```python
images = torch.randn(1, 24, 3, 518, 518)  # [B, S, C, H, W]
model = VGGT_MV(aa_order=["frame", "global"])  # 默认
predictions = model(images)
```

### 4.2 时序多视角模式（新功能）
```python
images = torch.randn(1, 24, 4, 3, 518, 518)  # [B, T, N, C, H, W]
model = VGGT_MV(aa_order=["view", "time"])  # 多视角模式
predictions = model(images)
```

### 4.3 加载权重
```python
checkpoint = torch.load("checkpoint/checkpoint_150.pt", map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
```

## 五、待实现的扩展功能

### 5.1 两流架构（位姿流 vs 几何流）
- [ ] 在 L_mid 层复制注意力块为两个流
- [ ] 实现 mask 偏置参数 `λ_pose`, `λ_geo`
- [ ] 实现动态感知聚合器

### 5.2 Sparse Global-SA（可选）
- [ ] Landmark/Anchor Attention
- [ ] Block-Sparse + Dilated
- [ ] Memory Bank

### 5.3 极线/几何先验（可选）
- [ ] 极线约束偏置
- [ ] Plücker 光线角度加权

## 六、注意事项

1. **输入格式**：模型会自动检测输入格式，支持向后兼容
2. **权重加载**：使用 `strict=False`，新参数会随机初始化
3. **计算复杂度**：
   - View-SA: `O(B · T · (NP)²)`
   - Time-SA: `O(B · N · (TP)²)`
4. **内存优化**：已实现 gradient checkpointing，训练时自动启用

## 七、测试建议

1. 测试单视角时序输入（向后兼容）
2. 测试多视角时序输入（新功能）
3. 测试权重加载是否正确
4. 测试前向传播是否正常运行
5. 验证输出形状是否符合预期



