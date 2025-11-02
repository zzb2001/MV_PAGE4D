# PAGE-4D 网络结构完整维度变化文档

本文档详细记录了从输入图像到最终输出的所有维度变化，根据架构图1-5描述。

## 输入格式

假设输入图像格式为：
- **images**: `[B, T, V, C, H, W]`
  - `B`: Batch size (批次大小)
  - `T`: Time steps (时间帧数)
  - `V`: Views (视角数)
  - `C`: Channels (通道数，通常为3，RGB)
  - `H, W`: Height, Width (图像高度和宽度，例如 518×518)

## 1. 编码部分 (DINOv2 ViT-L/14)

根据架构图1，编码部分包括以下步骤：

### 1.1 归一化 (Normalization)
```python
images = (images - mean) / std  # 逐通道归一化
```
- **输入**: `[B, T, V, C, H, W]` (例如 `[1, 24, 4, 3, 518, 518]`)
- **输出**: `[B, T, V, C, H, W]` (维度不变，值域变换)
- **参数来源**: 标准 ImageNet 归一化参数 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 1.2 Reshape 便于并行编码
```python
x = images.reshape(B*T*V, C, H, W)
```
- **输入**: `[B, T, V, C, H, W]`
- **输出**: `[B*T*V, C, H, W]` (例如 `[96, 3, 518, 518]`)
- **说明**: 将所有图像展平为一维，便于 ViT 并行处理

### 1.3 Patch Embed → Token
```python
tok = ViT(x)  # 去掉分类token，仅保留patch
```
- **输入**: `[B*T*V, C, H, W]`
- **ViT处理**: DINOv2 ViT-L/14
  - Patch size: 14×14
  - Image size: 518×518 → Patch数量: (518/14) × (518/14) = 37×37 = 1369
- **输出**: `patch_tok [B*T*V, P, Cvit]`
  - `P = 1369` (patch tokens数量)
  - `Cvit = 1024` (ViT embedding维度)
- **示例**: `[96, 1369, 1024]`
- **参数来源**: Pi3 (`/home/star/zzb/Pi3/ckpts/model.safetensors`)

### 1.4 Register Token
```python
reg_tok = register_token  # 每视角相同向量复制
```
- **Register tokens**: `[1, 1, R, Cvit]` (从 Pi3 加载)
  - `R = 4` 或 `5` (register token数量，根据Pi3配置)
- **扩展后**: `[B*T*V, R, Cvit]`
- **示例**: `[96, 4, 1024]`
- **参数来源**: 可从 Pi3 直接加载 (Pi3 有 `[1, 1, 5, 1024]`，需要适配到 `[1, 1, 4, 1024]`)

### 1.5 拼接 (Concatenation)
```python
feat0 = concat(reg_tok, patch_tok, dim=1)
```
- **输入**: 
  - `reg_tok [B*T*V, R, Cvit]`
  - `patch_tok [B*T*V, P, Cvit]`
- **输出**: `feat0 [B*T*V, R+P, Cvit]`
- **示例**: `[96, 4+1369, 1024] = [96, 1373, 1024]`

### 1.6 还原分组 (Restore Grouping)
```python
feat0 = feat0.view(B, T, V, R+P, Cvit)
```
- **输入**: `feat0 [B*T*V, R+P, Cvit]`
- **输出**: `feat0 [B, T, V, R+P, Cvit]`
- **示例**: `[1, 24, 4, 1373, 1024]`

### 1.7 Camera Token 添加
```python
camera_token = [B, T, V, 1, Cvit]  # 每个 (t,v) 一个camera token
tokens = concat([camera_token, feat0], dim=3)
```
- **输入**: `feat0 [B, T, V, R+P, Cvit]`
- **Camera token**: `[B, T, V, 1, Cvit]`
- **输出**: `tokens [B, T, V, 1+R+P, Cvit]`
- **示例**: `[1, 24, 4, 1+4+1369, 1024] = [1, 24, 4, 1374, 1024]`
- **总token数**: `P_total = 1 + R + P = 1 + 4 + 1369 = 1374`

## 2. 动态掩码小头 (Dynamic Mask Head)

根据架构图2，生成内生掩码：

### 2.1 输入
- **输入**: 中层特征 `feat0 [B, T, V, R+P, Cvit]` 或其后若干层输出
- **示例**: `[1, 24, 4, 1373, 1024]` (来自编码部分，不含camera token)

### 2.2 生成内生掩码 M_tilde
```python
M_tilde = DynamicMaskHead(feat0)
```
- **输出**: `M_tilde [B, T, V, P, 1]`
  - 注意：这里 `P` 是 patch tokens 数量，不包括 camera token 和 register tokens
- **示例**: `[1, 24, 4, 1369, 1]`
- **参数来源**: PAGE-4D (`checkpoint/checkpoint_150.pt` 优先)，不匹配则新增结构初始化

### 2.3 融合外源掩码
```python
M = sigmoid(α * M_tilde + β * M_ext_seq)
```
- **输入**: 
  - `M_tilde [B, T, V, P, 1]`
  - `M_ext_seq [B, T, V, P, 1]` (外源掩码，与M_tilde对齐)
- **参数**: 
  - `α, β`: 可学习标量或1×1MLP门控
- **输出**: `M [B, T, V, P, 1]`
- **示例**: `[1, 24, 4, 1369, 1]`
- **用途**: 提供动态/静态概率，在注意力里"抑制/放大"

## 3. 中间阶段交替聚合 (Mid-stage Alternating Aggregation)

根据架构图3-5，在 `L_mid` 层（例如层8-12）进行交替聚合。

### 3.1 View-SA (Synchronized View Attention, fixed t)

根据架构图3.1：

#### 3.1.1 输入重排
```python
X_v = concat_over_views(feat, axis=V)
```
- **输入**: `feat [B, T, V, R+P, Cvit]` (例如 `[1, 24, 4, 1374, 1024]`)
- **输出**: `X_v [B, T, V*(R+P), Cvit]`
- **示例**: `[1, 24, 4*1374, 1024] = [1, 24, 5496, 1024]`
- **参数来源**: 从 Pi3 `global_blocks` 拷权（多图像注意力结构）

#### 3.1.2 掩码广播
```python
M_v = concat_over_views(M, axis=V)  # [B, T, V*P, 1]
M_v_up = expand_to_token_dim(M_v)    # Fill 0 at R registers
```
- **输入**: `M [B, T, V, P, 1]` (例如 `[1, 24, 4, 1369, 1]`)
- **中间**: `M_v [B, T, V*P, 1]` (例如 `[1, 24, 4*1369, 1] = [1, 24, 5476, 1]`)
- **输出**: `M_v_up [B, T, V*(R+P), 1]`
  - Register部分填充0: `[B, T, V*R, 1] = zeros`
  - Patch部分使用原掩码: `[B, T, V*P, 1] = M_v`
- **示例**: `[1, 24, 5496, 1]`

#### 3.1.3 注意力 logits 偏置
```python
# 位姿流（抑制动态）
logits_pose += (-λ_pose) * M_v_up

# 几何流（放大动态）
logits_geo += (+λ_geo) * M_v_up
```
- **λ_pose, λ_geo**: 可学习标量，clamp 到 `[-b, b]` (例如 b=4.0)
- **参数来源**: 新增（初始化0）

#### 3.1.4 输出还原
```python
Y_v = attention(X_v)  # 内部处理
Y_v = restore_to_view_grouping(Y_v)
```
- **输入**: `X_v [B*T, V*(R+P), Cvit]` (内部reshape)
- **输出**: `Y_v [B, T, V, R+P, Cvit]`
- **示例**: `[1, 24, 4, 1374, 1024]`
- **说明**: 维度不变，只添加了 logits 偏置

#### 3.1.5 极线先验（可选）
```python
if coarse_pose/intrinsics_available:
    EpiMask = construct_epipolar_mask()  # [B, T, V*(R+P), V*(R+P)]
    # For (i,j) not satisfying epipolar band, subtract constant bias
    logits += EpiMask
```
- **EpiMask**: `[B, T, V*(R+P), V*(R+P)]` (注意力掩码形状)
- **用途**: 基于极线约束，对不满足极线带的token对减去常数偏置

### 3.2 Time-SA (同视角时间注意, 固定 v)

根据架构图3.2：

#### 3.2.1 输入重排
```python
X_t = concat_over_times(feat, axis=T)
```
- **输入**: `feat [B, T, V, R+P, Cvit]` (例如 `[1, 24, 4, 1374, 1024]`)
- **输出**: `X_t [B, V, T*(R+P), Cvit]`
- **示例**: `[1, 4, 24*1374, 1024] = [1, 4, 32976, 1024]`
- **参数来源**: PAGE-4D 中段层优先，或从 Pi3 `frame_blocks` 拷权

#### 3.2.2 相对时间位置编码
```python
# 使用 RoPE 或 ALiBi
if Pi3_has_rope:
    rope = load_from_Pi3()
else:
    rope = new_RoPE()  # 或 ALiBi
```
- **参数来源**: 如果Pi3有可复用部分则复用，否则新增
- **应用**: 在 Block 内部应用相对位置编码

#### 3.2.3 掩码广播
```python
M_t = concat_over_times(M, axis=T)  # [B, V, T*P, 1]
M_t_up = expand_to_token_dim(M_t)  # [B, V, T*(R+P), 1]
```
- **输入**: `M [B, T, V, P, 1]` (例如 `[1, 24, 4, 1369, 1]`)
- **转置**: `[B, V, T, P, 1]`
- **拼接**: `M_t [B, V, T*P, 1]` (例如 `[1, 4, 24*1369, 1] = [1, 4, 32856, 1]`)
- **扩展**: `M_t_up [B, V, T*(R+P), 1]`
  - Register部分填充0
- **示例**: `[1, 4, 32976, 1]`

#### 3.2.4 注意力 logits 偏置
```python
# 位姿流
logits_pose += (-λ_pose_t) * M_t_up

# 几何流
logits_geo += (+λ_geo_t) * M_t_up
```
- **λ_pose_t, λ_geo_t**: Time-SA 专用的可学习标量
- **参数来源**: 新增（初始化0）

#### 3.2.5 输出还原
```python
Y_t = attention(X_t)  # 内部处理
Y_t = restore_to_time_grouping(Y_t)
```
- **输入**: `X_t [B*V, T*(R+P), Cvit]` (内部reshape)
- **输出**: `Y_t [B, T, V, R+P, Cvit]`
- **示例**: `[1, 24, 4, 1374, 1024]`
- **目的**: 确保同一视角的时间一致性，避免将前景运动当作相机运动（位姿流），适度利用动态（几何流）

### 3.3 Sparse Global-SA (全局稀疏, 1-2层, 可选)

根据架构图3.3：

#### 3.3.1 输入展平
```python
X_g = feat.view(B, T*V*(R+P), Cvit)
```
- **输入**: `feat [B, T, V, R+P, Cvit]` (例如 `[1, 24, 4, 1374, 1024]`)
- **输出**: `X_g [B, T*V*(R+P), Cvit]`
- **示例**: `[1, 24*4*1374, 1024] = [1, 131904, 1024]`

#### 3.3.2 稀疏策略（三选一）

**策略1: Landmark**
```python
# 基于置信度/关键点，从每(t,v)选K个anchor
anchors = select_landmarks(X_g, k=64)  # 每个(t,v)选64个
# 全局只与anchor互注意
```
- **输出**: 稀疏注意力，只与landmark tokens交互
- **示例**: 从 `131904` tokens 中选择 `24*4*64 = 6144` 个landmarks

**策略2: Dilated Grid**
```python
# 在(t,v)网格做膨胀邻接（远距少量连接）
connections = dilated_grid_connect(T, V, levels=[1, 2, 4])
```
- **输出**: 稀疏连接图，按扩张级别连接邻居

**策略3: Memory Bank**
```python
memory_tokens = [B, M, Cvit]  # M=32
# 引入memory_tokens与当前token互注意
```
- **Memory tokens**: `[B, M, Cvit]` (例如 `[1, 32, 1024]`)
- **输出**: 通过memory tokens传递全局信息

#### 3.3.3 掩码/关键点偏置
```python
# 几何流
logits_geo += (+λ_kpt) * up(K_ext_seq)

# 位姿流
# 对关键点可中性或轻抑制（依任务而定）
```
- **K_ext_seq**: 外源关键点序列
- **λ_kpt**: 可学习标量

#### 3.3.4 输出
- **输出**: `Y_g [B, T, V, R+P, Cvit]`
- **示例**: `[1, 24, 4, 1374, 1024]`
- **参数来源**: 注意力结构可从 Pi3 `global_blocks` 拷权；稀疏连接/记忆token为新增逻辑

## 4. 输出汇总

经过所有中间层处理后：

- **最终输出**: `output_list [L, B, T, V, R+P, 2*Cvit]`
  - `L`: 中间层数量（例如 `L_mid` 层）
  - `2*Cvit`: 如果启用两流架构，concatenate pose 和 geo 流，否则为 `Cvit`
- **示例**: 
  - 单流: `[16, 1, 24, 4, 1374, 1024]`
  - 两流: `[16, 1, 24, 4, 1374, 2048]`

## 5. 维度变化总览表

| 步骤 | 输入维度 | 输出维度 | 说明 |
|------|---------|---------|------|
| 原始输入 | `[B, T, V, C, H, W]` | - | 例如 `[1, 24, 4, 3, 518, 518]` |
| 归一化 | `[B, T, V, C, H, W]` | `[B, T, V, C, H, W]` | 值域变换 |
| Reshape | `[B, T, V, C, H, W]` | `[B*T*V, C, H, W]` | 例如 `[96, 3, 518, 518]` |
| Patch Embed | `[B*T*V, C, H, W]` | `[B*T*V, P, Cvit]` | `[96, 1369, 1024]` |
| Register Token | `[B*T*V, R, Cvit]` | - | `[96, 4, 1024]` |
| 拼接 | - | `[B*T*V, R+P, Cvit]` | `[96, 1373, 1024]` |
| 还原分组 | `[B*T*V, R+P, Cvit]` | `[B, T, V, R+P, Cvit]` | `[1, 24, 4, 1373, 1024]` |
| Camera Token | - | `[B, T, V, 1+R+P, Cvit]` | `[1, 24, 4, 1374, 1024]` |
| View-SA 输入重排 | `[B, T, V, R+P, Cvit]` | `[B*T, V*(R+P), Cvit]` | `[24, 5496, 1024]` |
| View-SA 输出还原 | `[B*T, V*(R+P), Cvit]` | `[B, T, V, R+P, Cvit]` | `[1, 24, 4, 1374, 1024]` |
| Time-SA 输入重排 | `[B, T, V, R+P, Cvit]` | `[B*V, T*(R+P), Cvit]` | `[4, 32976, 1024]` |
| Time-SA 输出还原 | `[B*V, T*(R+P), Cvit]` | `[B, T, V, R+P, Cvit]` | `[1, 24, 4, 1374, 1024]` |
| Sparse Global 展平 | `[B, T, V, R+P, Cvit]` | `[B, T*V*(R+P), Cvit]` | `[1, 131904, 1024]` |
| Sparse Global 还原 | `[B, T*V*(R+P), Cvit]` | `[B, T, V, R+P, Cvit]` | `[1, 24, 4, 1374, 1024]` |

## 6. 关键参数说明

### 符号定义
- `B`: Batch size (批次大小)
- `T`: Time steps (时间帧数)
- `V`: Views (视角数)
- `C`: Channels (输入通道数，通常为3)
- `H, W`: Image height, width (例如 518, 518)
- `P`: Patch tokens数量 (例如 1369 = 37×37)
- `R`: Register tokens数量 (例如 4)
- `Cvit`: ViT embedding维度 (例如 1024)
- `P_total`: 总token数 = `1 + R + P` (例如 1374)

### 参数来源
1. **Pi3**: `/home/star/zzb/Pi3/ckpts/model.safetensors`
2. **PAGE-4D**: `checkpoint/checkpoint_150.pt`
3. **新增**: 随机初始化

### 可学习参数
- `λ_pose`, `λ_geo`: View-SA 的位姿流和几何流偏置（clamp 到 `[-4, 4]`）
- `λ_pose_t`, `λ_geo_t`: Time-SA 的位姿流和几何流偏置
- `λ_kpt`: Sparse Global-SA 的关键点偏置
- `α`, `β`: 动态掩码融合的权重（可学习标量或1×1MLP门控）

