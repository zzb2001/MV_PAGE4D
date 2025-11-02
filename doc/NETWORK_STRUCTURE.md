# PAGE-4D 完整网络结构文档

本文档详细描述了根据架构图1-5实现的完整网络结构。

## 网络架构总览

```
PAGE-4D Network
├── 1. 编码 (DINOv2 ViT-L/14)
│   ├── 归一化
│   ├── Reshape (B*T*V, C, H, W)
│   ├── Patch Embed → Token
│   ├── Register Token
│   ├── 拼接 (reg_tok + patch_tok)
│   └── 还原分组 (B, T, V, R+P, Cvit)
├── 2. 动态掩码小头
│   ├── 生成内生掩码 M_tilde [B,T,V,P,1]
│   └── 融合外源掩码 M = sigmoid(α*M_tilde + β*M_ext_seq)
└── 3. 中间阶段交替聚合 (Core)
    ├── 3.1 View-SA (固定时间t)
    │   ├── 输入重排 (跨视角聚合)
    │   ├── 掩码广播
    │   ├── Logits偏置 (位姿流/几何流)
    │   └── 输出还原
    ├── 3.2 Time-SA (固定视角v)
    │   ├── 输入重排 (跨时间聚合)
    │   ├── 相对时间位置编码 (RoPE/ALiBi)
    │   ├── 掩码广播
    │   ├── Logits偏置 (位姿流/几何流)
    │   └── 输出还原
    └── 3.3 Sparse Global-SA (可选, 1-2层)
        ├── 输入展平
        ├── 稀疏策略 (Landmark/Dilated/Memory Bank)
        └── 输出还原
```

## 详细模块说明

### 1. 编码部分 (DINOv2 ViT-L/14)

**位置**: `Aggregator.__init__()` 和 `forward()`

**步骤**:
1. **归一化**: `(images - mean) / std` (逐通道)
   - 输入: `[B, T, V, C, H, W]`
   - 输出: `[B, T, V, C, H, W]`

2. **Reshape**: `images.reshape(B*T*V, C, H, W)`
   - 输入: `[B, T, V, C, H, W]`
   - 输出: `[B*T*V, C, H, W]`

3. **Patch Embed**: `patch_tokens = ViT(images)`
   - 输入: `[B*T*V, C, H, W]`
   - 输出: `patch_tok [B*T*V, P, Cvit]`
   - 说明: 去掉分类token，仅保留patch tokens
   - **参数来源**: Pi3 (`/home/star/zzb/Pi3/ckpts/model.safetensors`)

4. **Register Token**: 每视角相同向量复制
   - Register tokens: `[1, 1, R, Cvit]` → 扩展为 `[B*T*V, R, Cvit]`
   - **参数来源**: 可从Pi3直接加载（需适配形状）

5. **拼接**: `feat0 = concat(reg_tok, patch_tok, dim=1)`
   - 输入: `reg_tok [B*T*V, R, Cvit]`, `patch_tok [B*T*V, P, Cvit]`
   - 输出: `feat0 [B*T*V, R+P, Cvit]`

6. **还原分组**: `feat0.view(B, T, V, R+P, Cvit)`
   - 输入: `feat0 [B*T*V, R+P, Cvit]`
   - 输出: `feat0 [B, T, V, R+P, Cvit]`

7. **添加Camera Token**: `tokens = concat([camera_token, feat0], dim=3)`
   - 输入: `feat0 [B, T, V, R+P, Cvit]`
   - Camera token: `[B, T, V, 1, Cvit]`
   - 输出: `tokens [B, T, V, 1+R+P, Cvit]`

**关键代码位置**:
- `Aggregator.forward()`: 第330-398行

### 2. 动态掩码小头 (Dynamic Mask Head)

**位置**: `vggt_t_mv/models/dynamic_mask_head.py`

**功能**: 生成内生掩码并与外源掩码融合

**输入**: 中层特征 (例如 feat0 或其后若干层输出)
- `feat [B, T, V, R+P, C]` 或 `[B, T, V, P, C]`

**输出**: 
- `M_tilde [B, T, V, P, 1]`: 内生掩码
- `M [B, T, V, P, 1]`: 最终融合掩码

**融合公式**:
```python
M = sigmoid(α * M_tilde + β * M_ext_seq)
```

**参数**:
- `α, β`: 可学习标量或1×1MLP门控（通过 `use_gating` 参数控制）

**参数来源**: PAGE-4D (`checkpoint/checkpoint_150.pt` 优先)，不匹配则新增结构初始化

**用途**: 提供动态/静态概率，在注意力里"抑制/放大"

**关键代码位置**:
- `DynamicMaskHead.forward()`: 第41-120行

### 3. 中间阶段交替聚合 (Mid-stage Alternating Aggregation)

**位置**: `Aggregator.forward()` 和 `_process_view_attention()`, `_process_time_attention()`

**应用层**: `L_mid` 层，例如层8-12

#### 3.1 View-SA (Synchronized View Attention, fixed t)

**功能**: 固定时间t，跨视角聚合

**实现位置**: `Aggregator._process_view_attention()` (第879-977行)

**步骤**:

1. **输入重排**: `X_v = concat_over_views(feat, axis=V)`
   - 输入: `feat [B, T, V, R+P, Cvit]`
   - 输出: `X_v [B*T, V*(R+P), Cvit]` (内部reshape)

2. **掩码广播**: `M_v = concat_over_views(M, axis=V)`
   - 输入: `M [B, T, V, P, 1]`
   - 中间: `M_v [B, T, V*P, 1]`
   - 输出: `M_v_up [B*T, V*(R+P), 1]` (Expand to token dimension, fill 0 at R registers)

3. **注意力 logits 偏置**:
   - 位姿流: `logits_pose += (-λ_pose) * M_v_up`
   - 几何流: `logits_geo += (+λ_geo) * M_v_up`
   - `λ_pose, λ_geo`: 可学习标量，clamp 到 `[-b, b]` (例如 b=4.0)

4. **输出还原**: `Y_v → [B, T, V, R+P, Cvit]`
   - 输入: `X_v [B*T, V*(R+P), Cvit]`
   - 输出: `Y_v [B, T, V, R+P, Cvit]`

**参数来源**:
- 注意力结构: 从 Pi3 `global_blocks` 拷权（多图像注意力结构）
- 掩码逻辑: 新增
- `λ_pose, λ_geo`: 新增（初始化0）

**可选功能**: 极线先验 (Epipolar Prior)
- 如果提供了粗略位姿/内参，构造 `EpiMask [B, T, V*(R+P), V*(R+P)]`
- 对不满足极线带的 (i,j) token对，减去常数偏置

#### 3.2 Time-SA (同视角时间注意, 固定 v)

**功能**: 固定视角v，跨时间聚合

**实现位置**: `Aggregator._process_time_attention()` (第979-1079行)

**步骤**:

1. **输入重排**: `X_t = concat_over_times(feat, axis=T)`
   - 输入: `feat [B, T, V, R+P, Cvit]`
   - 输出: `X_t [B*V, T*(R+P), Cvit]` (内部reshape)

2. **相对时间位置编码**: RoPE 或 ALiBi
   - 如果Pi3有可复用部分则复用，否则新增
   - 在Block内部应用相对位置编码

3. **掩码广播**: `M_t = concat_over_times(M, axis=T)`
   - 输入: `M [B, T, V, P, 1]`
   - 中间: `M_t [B, V, T*P, 1]`
   - 输出: `M_t_up [B*V, T*(R+P), 1]` (Expand, fill 0 at R registers)

4. **注意力 logits 偏置**:
   - 位姿流: `logits_pose += (-λ_pose_t) * M_t_up`
   - 几何流: `logits_geo += (+λ_geo_t) * M_t_up`
   - `λ_pose_t, λ_geo_t`: Time-SA专用的可学习标量

5. **输出还原**: `Y_t → [B, T, V, R+P, Cvit]`
   - 输入: `X_t [B*V, T*(R+P), Cvit]`
   - 输出: `Y_t [B, T, V, R+P, Cvit]`

**参数来源**:
- 注意力结构: PAGE-4D中段层优先，或从Pi3 `frame_blocks` 拷权
- `λ_pose_t, λ_geo_t`: 新增（初始化0）

**目的**:
- 位姿流: 避免将前景运动当作相机运动
- 几何流: 适度利用动态

#### 3.3 Sparse Global-SA (全局稀疏, 1-2层, 可选)

**功能**: 全局稀疏长程依赖

**实现位置**: `Aggregator._process_sparse_global_attention()` (第674-726行)

**步骤**:

1. **输入展平**: `X_g [B, T*V*(R+P), Cvit]`
   - 输入: `feat [B, T, V, R+P, Cvit]`
   - 输出: `X_g [B, T*V*(R+P), Cvit]`

2. **稀疏策略** (三选一):

   **策略1: Landmark**
   - 基于置信度/关键点，从每(t,v)选K个anchor
   - 全局只与anchor互注意
   - 参数: `landmark_k = 64`

   **策略2: Dilated Grid**
   - 在(t,v)网格做膨胀邻接（远距少量连接）
   - 参数: `dilated_levels = [1, 2, 4]`

   **策略3: Memory Bank**
   - 引入 `memory_tokens [B, M, Cvit]` (M=32)
   - 与当前token互注意

3. **掩码/关键点偏置**:
   - 几何流: `logits_geo += (+λ_kpt) * up(K_ext_seq)`
   - 位姿流: 对关键点可中性或轻抑制（依任务而定）

4. **输出**: `Y_g [B, T, V, R+P, Cvit]`

**参数来源**:
- 注意力结构: 可从 Pi3 `global_blocks` 拷权
- 稀疏连接/记忆token: 新增逻辑

### 4. 两流架构 (Dual Stream)

**启用条件**: `enable_dual_stream=True`

**功能**: 并行处理位姿流和几何流

**位姿流**: 用于相机位姿估计（抑制动态区域）
- `pose_time_blocks`: 从 Pi3 `frame_blocks` 拷权
- `pose_view_blocks`: 从 Pi3 `global_blocks` 拷权
- Logits偏置: `logits_pose += (-λ_pose) * mask` (抑制动态)

**几何流**: 用于点云/深度估计（放大动态区域）
- `geo_time_blocks`: 从 Pi3 `frame_blocks` 拷权（会解冻微调）
- `geo_view_blocks`: 从 Pi3 `global_blocks` 拷权（会解冻微调）
- Logits偏置: `logits_geo += (+λ_geo) * mask` (放大动态)

**参数来源**: 从Pi3的对应blocks拷权

## 参数初始化策略

### 三阶段加载策略

1. **阶段1: 加载Pi3权重**
   - Patch Embed (DINOv2 ViT-L): 从Pi3加载
   - Register Token: 从Pi3加载（需适配形状）
   - Time-SA blocks (前18层): 从Pi3 `frame_blocks` 加载
   - View-SA blocks (前18层): 从Pi3 `global_blocks` 加载

2. **阶段2: 覆盖PAGE-4D权重**
   - Spatial Mask Head: 从PAGE-4D加载
   - Time-SA blocks (后6层): 从PAGE-4D加载
   - View-SA blocks (后6层): 从PAGE-4D加载
   - Camera Head, Point Head, Depth Head: 从PAGE-4D加载

3. **阶段3: 初始化新增模块**
   - Dynamic Mask Head: 随机初始化（如果结构不匹配）
   - Lambda参数 (λ_pose, λ_geo, λ_pose_t, λ_geo_t): 初始化为0.0
   - Sparse Global blocks: 随机初始化（或从Pi3拷权）
   - Memory tokens: 随机初始化（如果使用Memory Bank策略）

## 关键参数说明

### 可学习标量参数

| 参数 | 默认值 | 范围 | 用途 |
|------|--------|------|------|
| `λ_pose` | 0.0 | [-4.0, 4.0] | View-SA位姿流偏置（抑制动态） |
| `λ_geo` | 0.0 | [-4.0, 4.0] | View-SA几何流偏置（放大动态） |
| `λ_pose_t` | 0.0 | [-4.0, 4.0] | Time-SA位姿流偏置 |
| `λ_geo_t` | 0.0 | [-4.0, 4.0] | Time-SA几何流偏置 |
| `α` | 1.0 | - | 动态掩码融合权重（内生掩码） |
| `β` | 1.0 | - | 动态掩码融合权重（外源掩码） |
| `λ_kpt` | 0.0 | - | Sparse Global关键点偏置 |

### 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_dim` | 1024 | ViT embedding维度 |
| `num_register_tokens` | 4 | Register token数量 |
| `depth` | 24 | 总层数（Time-SA + View-SA各24层） |
| `landmark_k` | 64 | Landmark策略选择的anchor数量 |
| `memory_tokens_num` | 32 | Memory Bank策略的memory token数量 |
| `lambda_clamp_value` | 4.0 | Lambda参数的clamp范围 |

## 文件结构

```
vggt_t_mv/
├── models/
│   ├── aggregator.py          # 主聚合器（编码+交替聚合）
│   ├── dynamic_mask_head.py   # 动态掩码小头
│   └── vggt.py                 # 主模型（Aggregator + Heads）
├── layers/
│   ├── block.py                # Block实现（包含SpatialMaskHead_IMP）
│   └── vision_transformer.py   # DINOv2 ViT实现
├── DIMENSION_CHANGES.md        # 维度变化详细文档
└── NETWORK_STRUCTURE.md        # 本文档
```

## 使用示例

```python
from vggt_t_mv.models.vggt import VGGT

# 初始化模型
model = VGGT(
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    enable_camera=True,
    enable_point=True,
    enable_depth=True,
    enable_dual_stream=True,  # 启用两流架构
    enable_sparse_global=True,  # 启用Sparse Global-SA
    sparse_global_layers=[23, 24],  # 在最后2层应用
    sparse_strategy="landmark"  # 使用Landmark策略
)

# 输入: [B, T, V, C, H, W]
images = torch.randn(1, 24, 4, 3, 518, 518)

# 前向传播
predictions = model(images)
```

## 参考

- 架构图1: 编码 (DINOv2 ViT-L/14)
- 架构图2: 动态掩码小头
- 架构图3.1: View-SA (固定时间t)
- 架构图3.2: Time-SA (固定视角v)
- 架构图3.3: Sparse Global-SA (可选)


