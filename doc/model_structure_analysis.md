# Model_MV 结构与 Pi3 权重加载分析

## 1. Model_MV 结构总结

### 1.1 整体架构
```
VGGT_MV (1.26B params)
├─ aggregator: 909M params (72.4%)
│   ├─ patch_embed: DinoVisionTransformer (304M params) - 24层ViT-L
│   ├─ time_blocks: 24层 × 12.6M = 302M params
│   ├─ view_blocks: 24层 × 12.6M = 302M params
│   ├─ camera_token: [1, 2, 1, 1024]
│   ├─ register_token: [1, 2, 4, 1024]
│   └─ spatial_mask_head: 21K params
├─ camera_head: 216M params (17.2%)
├─ point_head: 32.6M params (2.6%)
├─ depth_head: 32.6M params (2.6%)
└─ track_head: 65.9M params (5.2%)
```

### 1.2 Aggregator 关键模块

**time_blocks / view_blocks (每层):**
- LayerNorm: 2K 参数
- Attention (QKV): 3.1M 参数
  - QKV投影: 3 × 1024 × 1024 = 3.1M
  - Q/K norm: 128 参数
  - 输出投影: 1024 × 1024 = 1M
- LayerScale: 1K 参数
- MLP: 8.4M 参数
  - fc1: 1024 → 4096 (4.2M)
  - fc2: 4096 → 1024 (4.2M)
- **每层总计**: 12.6M 参数

**参数总览:**
- time_blocks: 24层 × 12.6M = **302M params**
- view_blocks: 24层 × 12.6M = **302M params**
- patch_embed (ViT-L): **304M params**

---

## 2. Pi3 权重结构分析

### 2.1 Pi3 总体结构
```
Pi3 权重 (1210个键)
├─ encoder: 343 keys
│   ├─ blocks: 24层 ViT-L (与DINOv2一致)
│   ├─ patch_embed: [1024, 3, 14, 14]
│   └─ register_token: [1, 1, 5, 1024] ⚠️ 形状不匹配
├─ decoder: 648 keys
│   └─ 36个blocks (0-35)
│       ├─ 偶数索引 (0,2,4,...,34): View内空间注意力 → time_blocks
│       └─ 奇数索引 (1,3,5,...,35): 跨View注意力 → view_blocks
├─ camera_decoder: 64 keys (4层)
├─ camera_head: 20 keys
├─ point_decoder: 64 keys (4层)
├─ point_head: 2 keys
├─ conf_decoder: 64 keys (4层)
└─ conf_head: 2 keys
```

### 2.2 Pi3 decoder 结构
- **总blocks**: 36个（0-35）
- **每层参数**: 18个（norm1, attn, ls1, norm2, mlp, ls2）
- **结构**: 与 model_mv 的 time_blocks/view_blocks 完全相同

**关键差异:**
- Pi3: 36个decoder blocks
- Model_MV: 24个time_blocks + 24个view_blocks = **48个blocks**

---

## 3. 映射问题分析

### 3.1 成功映射的部分 ✅
```
Pi3 decoder.偶数 → Model_MV time_blocks[i//2]
Pi3 decoder.奇数 → Model_MV view_blocks[i//2]

映射结果:
- 36个decoder → 前18个time_blocks + 前18个view_blocks
- 成功映射: 648个参数
```

### 3.2 形状不匹配问题 ⚠️

#### 问题1: register_token 形状差异
```python
Pi3:        [1, 1, 5, 1024]  # 5个register tokens
Model_MV:   [1, 2, 4, 1024]  # 4个register tokens，分2类

# 解决方案: 需要适配形状或修改Model_MV
```

#### 问题2: decoder数量不足
```python
Pi3: 36个decoder blocks
Model_MV: 24个time_blocks + 24个view_blocks = 48个blocks

实际映射:
- Pi3 decoder.0..34 → Model_MV time_blocks[0..17] (18个)
- Pi3 decoder.1..35 → Model_MV view_blocks[0..17] (18个)
- Model_MV time_blocks[18..23]: 随机初始化
- Model_MV view_blocks[18..23]: 随机初始化
```

#### 问题3: encoder.blocks 未加载
```python
Pi3 encoder.blocks: 24层 ViT-L (304M params)
Model_MV patch_embed.blocks: 24层 ViT-L (304M params)

原因: 权重加载函数只加载到aggregator，没有加载encoder.blocks
可能的覆盖: 这些权重应该从 PAGE-4D checkpoint 加载
```

---

## 4. 权重加载策略建议

### 4.1 三阶段加载流程

**阶段1: 加载 Pi3 权重**
```python
✅ decoder.0-34 → time_blocks[0-17] (18层)
✅ decoder.1-35 → view_blocks[0-17] (18层)
⚠️ register_token: 形状不匹配，需要适配
⚠️ encoder.patch_embed: 应该加载（用于patch embedding）
❌ encoder.blocks: 跳过（应该从PAGE-4D加载）
❌ camera/point/conf_decoder: 跳过（Pi3特有）
```

**阶段2: 覆盖 PAGE-4D 权重**
```python
✅ spatial_mask_head: 完全加载
✅ time_blocks[18-23]: 加载
✅ view_blocks[18-23]: 加载
✅ patch_embed (完整的DINOv2): 加载
✅ camera/point/depth/track heads: 加载
```

**阶段3: 初始化新增模块**
```python
✅ 两流架构blocks: 从主blocks拷贝
✅ Sparse Global blocks: 随机初始化
✅ lambda参数: 初始化为0.0
```

### 4.2 关键修复建议

#### 修复1: register_token 适配
```python
# Pi3: [1, 1, 5, 1024]
# Model_MV: [1, 2, 4, 1024]

# 方案A: 修改Model_MV使用5个tokens
# 方案B: Pi3只取前4个tokens
register_token_pi3 = pi3_weights['register_token']  # [1, 1, 5, 1024]
register_token_adapted = register_token_pi3[:, :, :4, :]  # [1, 1, 4, 1024]
# 然后扩展到 [1, 2, 4, 1024]
```

#### 修复2: 完整加载encoder
```python
# 应该加载 Pi3 的 encoder.patch_embed 和 encoder.blocks
encoder_keys_to_load = [
    'encoder.patch_embed',
    'encoder.blocks.*'  # 所有blocks
]

# 映射到:
target_keys = [
    'aggregator.patch_embed',
    'aggregator.patch_embed.blocks.*'
]
```

#### 修复3: camera_token 处理
```python
# Model_MV有camera_token: [1, 2, 1, 1024]
# Pi3可能没有，需要随机初始化或从其他地方加载
```

---

## 5. 当前加载状态

### 5.1 成功映射统计
```
✅ decoder blocks映射: 648个参数
   - time_blocks[0-17]: 18层 × 18 params/layer = 324 params
   - view_blocks[0-17]: 18层 × 18 params/layer = 324 params

❌ 未映射:
   - register_token: 形状不匹配
   - camera_token: Pi3中没有
   - patch_embed: 只加载了部分
   - time_blocks[18-23]: 需要从PAGE-4D加载
   - view_blocks[18-23]: 需要从PAGE-4D加载
```

### 5.2 根本原因
**Pi3 decoder结构与Model_MV结构不完全匹配**:
- Pi3: 单流的36个decoder blocks
- Model_MV: 双流的24+24个blocks

**映射策略:**
- Pi3的前36个blocks映射到Model_MV的前18个time+view blocks
- Model_MV的后6个blocks需要从PAGE-4D加载

---

## 6. 完整加载方案

### 方案A: 调整映射策略（推荐）
```python
# Pi3 decoder映射:
# - 偶数 → time_blocks (18层)
# - 奇数 → view_blocks (18层)
# 
# 然后从PAGE-4D加载:
# - time_blocks[18-23] 的前身→后6层time_blocks
# - view_blocks[18-23] 的前身→后6层view_blocks
# - spatial_mask_head: 完全加载
```

### 方案B: 修改Model_MV深度
```python
# 将Model_MV改为18层deep（而不是24层）
# 这样Pi3的36个blocks可以完全映射
depth = 18  # 而不是24
```

### 方案C: 混合初始化
```python
# 0-17层: 从Pi3加载（18层）
# 18-23层: 从PAGE-4D加载（6层）
# 或18-23层: 随机初始化并训练
```

---

## 7. 建议的最终方案

### 优先策略
1. **register_token**: 加载Pi3的前4个tokens，适配形状
2. **time_blocks[0-17]**: 从Pi3加载
3. **view_blocks[0-17]**: 从Pi3加载
4. **patch_embed**: 从PAGE-4D加载（完整的DINOv2）
5. **time_blocks[18-23]**: 从PAGE-4D加载（frame_blocks的后6层）
6. **view_blocks[18-23]**: 从PAGE-4D加载（global_blocks的后6层）
7. **camera_token**: 从PAGE-4D加载
8. **spatial_mask_head**: 从PAGE-4D完全加载

### 实现检查点
- [x] 修复 load_pi3_weights 的 aggregator 访问问题
- [ ] 适配 register_token 形状
- [ ] 加载 Pi3 encoder.patch_embed（如果需要）
- [ ] 完整测试三阶段加载
- [ ] 验证推理输出

