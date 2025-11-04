# Aggregator 维度变化追踪文档

## 概述

本文档追踪 `mv_page4d_lite/models/aggregator.py` 中 `forward` 方法（第593-731行）的维度变化，帮助理解数据流和避免维度不匹配错误。

## 关键维度定义

- **B**: Batch size（批次大小）
- **T**: Time steps（时间步数）
- **V**: Views（视角数）
- **S**: Sequence length（序列长度）
  - 非体素化路径：`S = T * V`
  - 体素化路径：`S = T`（每个时间步的体素序列）
- **P**: Patch/Voxel tokens per sequence（每个序列的patch/体素token数）
  - 非体素化路径：`P = H*W // (patch_size^2)` + special tokens
  - 体素化路径：`P = max_num_voxels` + special tokens
- **C**: Embedding dimension（嵌入维度，通常是1024）
- **H, W**: Image height and width（图像高度和宽度）

## 维度变化流程

### 1. 输入阶段（第305-586行）

**输入**:
- `images`: `[B, T, V, 3, H, W]`

**Patch Embedding**:
- 非体素化路径：`patch_tokens`: `[B*T*V, P_patch, C]`，其中 `P_patch = H*W // (patch_size^2)`
- 体素化路径：`patch_tokens`: `[B, T*max_num_voxels, C]`

**添加 Special Tokens**:
- Camera token: `[1, C]` → 扩展为 `[B*S, 1, C]`
- Register tokens: `[num_register_tokens, C]` → 扩展为 `[B*S, num_register_tokens, C]`

**输出 tokens**:
- 非体素化路径：`[B*S, P, C]`，其中 `S = T*V`，`P = 1 + num_register_tokens + P_patch`
- 体素化路径：`[B*T, P, C]`，其中 `S = T`，`P = 1 + num_register_tokens + max_num_voxels`

### 2. Attention 处理阶段（第626-780行）

#### 2.1 Frame Attention（第628-658行）

**输入**: `tokens: [B*S, P, C]`

**处理**:
- `_process_frame_attention` 内部可能reshape tokens
- 体素化路径：可能reshape为 `[B, T, max_num_voxels, C]` 或 `[B*T, max_num_voxels, C]`
- 非体素化路径：可能reshape为 `[B, T, V, P, C]` 或 `[B*T, V*P, C]`

**输出**: `tokens: [B*S_out, P_out, C]`，其中 `S_out` 和 `P_out` 可能因reshape而变化

#### 2.2 Mask 计算（第659-751行）

**输入**: `tokens: [B*S, P, C]`

**处理**:
- Reshape为 `[B, S, P, C]` 用于mask计算
- 提取patch tokens：`[B*S, P-patch_start_idx, C]`
- Reshape为 `[B*S, C, mask_h, mask_w]` 用于spatial mask head

**关键修复**:
- 从实际tokens形状推断 `S` 和 `P`
- 体素化路径使用近似的 `mask_h` 和 `mask_w`
- 非体素化路径使用 `H // patch_size` 和 `W // patch_size`

#### 2.3 Global Attention（第759-778行）

**输入**: `tokens: [B*S, P, C]`

**处理**:
- `_process_global_attention` 内部处理reshape
- 可能reshape为 `[B, S*P, C]` 或保持 `[B*S, P, C]`

**输出**: `tokens: [B*S_out, P_out, C]`

### 3. 维度统一机制（新增）

**位置**：第586-618行（初始化）和第782-792行（每次循环后）

**功能**:
1. 从实际tokens形状推断 `S` 和 `P`
2. 在每个attention块后自动更新 `S` 和 `P`
3. 确保tokens始终是 `[B*S, P, C]` 格式

**实现**:
```python
# 初始化时
B_curr, P_curr, C_curr = tokens.shape
S_curr = tokens.shape[0] // B
S = S_curr
P = P_curr

# 每次循环后
B_after, P_after, C_after = tokens.shape
S_after = B_after // B
if S_after != S or P_after != P:
    S = S_after
    P = P_after
    if tokens.shape != (B * S, P, C):
        tokens = tokens.reshape(B * S, P, C)
```

## 常见维度不匹配问题及解决方案

### 问题1: Reshape错误 `shape '[B, S, P, C]' is invalid for input of size X`

**原因**: `tokens` 的实际形状与期望的 `[B*S, P, C]` 不匹配

**解决方案**:
1. 从实际tokens形状推断 `S` 和 `P`
2. 更新 `S` 和 `P` 变量
3. 使用推断的值进行reshape

### 问题2: Mask计算时维度不匹配

**原因**: `mask_h * mask_w` 与实际的patch token数不匹配

**解决方案**:
1. 从实际patch token数计算 `mask_h` 和 `mask_w`
2. 使用padding或截断处理不匹配的情况

### 问题3: 体素化路径下维度混乱

**原因**: 体素化路径改变了 `S` 和 `P` 的含义

**解决方案**:
1. 明确区分体素化路径和非体素化路径
2. 体素化路径：`S = T`，`P = max_num_voxels + special_tokens`
3. 非体素化路径：`S = T*V`，`P = H*W // (patch_size^2) + special_tokens`

## 最佳实践

1. **始终从实际tokens形状推断维度**：不要假设 `S` 和 `P` 的值
2. **在每个reshape前检查维度**：确保大小匹配
3. **使用统一的维度更新机制**：在关键点更新 `S` 和 `P`
4. **区分体素化路径和非体素化路径**：两种路径的维度语义不同
5. **添加维度验证**：在关键操作前验证维度

## 修复总结

本次修复主要解决了以下问题：

1. **初始化维度统一**（第586-618行）：从实际tokens形状推断并设置 `S` 和 `P`
2. **Mask计算维度修复**（第659-751行）：智能推断维度并处理不匹配情况
3. **循环后维度更新**（第782-792行）：在每个attention块后自动更新维度
4. **体素化路径特殊处理**：区分体素化路径和非体素化路径的维度语义

这些修复确保了维度在整个forward过程中保持一致，避免了维度不匹配错误。


