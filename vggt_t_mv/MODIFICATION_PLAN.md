# vggt_t_mv 完整修改思路与实施方案

## 一、核心设计目标

### 1.1 输入格式转换
- **原格式**：`[B, S, C, H, W]` - 单视角时序序列（S为时序帧数）
- **目标格式**：`[B, T, N, C, H, W]` - 时序多视角序列
  - `B`: Batch size
  - `T`: 时间窗长度（Time window length）
  - `N`: 同步视角数（Synchronized views）
  - `C`: RGB通道数（3）
  - `H, W`: 图像高度和宽度

### 1.2 架构演进方向
从 **单视角时序建模** (`vggt_t_mask_mlp_fin10`) 扩展为 **时序多视角联合建模** (`vggt_t_mv`)

## 二、整体修改策略

### 2.1 权重复用策略（优先级排序）

1. **第一优先级**：复用 `checkpoint/checkpoint_150.pt` 中的权重
   - DINOv2 encoder 权重（完全复用）
   - Frame-Attention blocks → 用于 Time-SA（时序建模部分）
   - Global-Attention blocks → 用于 View-SA（多视角聚合部分）
   - LayerNorm、MLP 权重
   - Register tokens 参数

2. **第二优先级**：从 Pi3 加载多视角聚合相关权重
   - Pi3 的 Global-Attention 权重 → 用于 View-SA
   - Pi3 的 Frame-Attention 权重 → 用于 Time-SA（作为初始化）
   - Pi3 的 register tokens → 如果维度匹配

3. **第三优先级**：新增权重（需要训练）
   - 两流架构的 mask 偏置参数：`λ_pose`, `λ_geo`
   - 动态掩码生成器（SpatialMaskHead）
   - 外源掩码融合门控参数

### 2.2 架构整合方案

#### 阶段1：基础结构迁移
- 将所有对 `vggt_t_mask_mlp_fin10` 的引用改为 `vggt_t_mv`
- 保持原有的 encoder、decoder 结构不变
- 修改 aggregator 的输入处理逻辑

#### 阶段2：聚合机制升级
- **替换策略**：将原有的 `frame/global` 交替 → `View-SA/Time-SA` 交替
  - `frame` attention → `Time-SA`（固定视角v，跨时聚合）
  - `global` attention → `View-SA`（固定时刻t，跨视聚合）
- **形状变换**：
  - Time-SA: `[B*N, T*(R+P), C]` - 每个视角的所有时刻堆叠
  - View-SA: `[B*T, N*(R+P), C]` - 每个时刻的所有视角堆叠

#### 阶段3：两流架构实现
- 在 L_mid 层复制注意力块为两个流：
  - **位姿流（Pose Stream）**：用于相机位姿估计
  - **几何流（Geo Stream）**：用于点云、深度、置信度估计
- 参数共享策略（初始实现）：
  - Q/K/V 投影共享，但分别施加 `pose-mask` 和 `geo-mask` 偏置
  - 后续可改为分支化 Q/K/V（效果更好但参数更多）

#### 阶段4：动态感知聚合器
- 实现动态掩码生成（基于 SpatialMaskHead）
- 位姿支路：`A^pose_{ij} = A_{ij} - λ_pose · m_j`（抑制动态）
- 几何支路：`A^geo_{ij} = A_{ij} + λ_geo · m_j`（放大动态）
- 支持外源掩码融合（SegAnyMo）

## 三、具体实现细节

### 3.1 Aggregator 核心修改

#### 输入处理
```python
# 原输入：[B, S, C, H, W] - S为时序帧数（单视角）
# 新输入：[B, T, N, C, H, W] - T为时间，N为视角
def forward(self, images: torch.Tensor, ...):
    B, T, N, C, H, W = images.shape
    
    # Reshape for patch embedding: [B*T*N, C, H, W]
    images = images.view(B * T * N, C, H, W)
    patch_tokens = self.patch_embed(images)  # [B*T*N, P, C]
    
    # 组织为 [B, T, N, P, C] 用于后续聚合
    patch_tokens = patch_tokens.view(B, T, N, P, C)
```

#### View-SA 实现
```python
def _process_view_attention(self, tokens, B, T, N, P, C, view_idx, pos=None):
    """
    固定时刻t，跨视角聚合
    形状变换：[B, T, N, P, C] → [B*T, N*P, C] → MHA → [B, T, N, P, C]
    """
    # Stack all views at time t: [B*T, N*(R+P), C]
    tokens_view = tokens.view(B, T, N, P, C)
    tokens_flat = tokens_view.view(B*T, N*P, C)
    
    # Apply View-SA (复用 Pi3 Global-Attention 权重)
    tokens_flat = self.view_blocks[view_idx](tokens_flat, pos=pos_view, ...)
    
    # Reshape back: [B, T, N, P, C]
    return tokens_flat.view(B, T, N, P, C)
```

#### Time-SA 实现
```python
def _process_time_attention(self, tokens, B, T, N, P, C, time_idx, pos=None):
    """
    固定视角v，跨时聚合
    形状变换：[B, T, N, P, C] → [B*N, T*P, C] → MHA → [B, T, N, P, C]
    """
    # Stack all times at view v: [B*N, T*(R+P), C]
    tokens_time = tokens.view(B, T, N, P, C)
    tokens_flat = tokens_time.transpose(1, 2).contiguous()  # [B, N, T, P, C]
    tokens_flat = tokens_flat.view(B*N, T*P, C)
    
    # Apply Time-SA (复用 vggt Frame-Attention 权重)
    tokens_flat = self.time_blocks[time_idx](tokens_flat, pos=pos_time, ...)
    
    # Reshape back: [B, T, N, P, C]
    return tokens_flat.view(B, N, T, P, C).transpose(1, 2).contiguous()
```

#### 交替聚合流程
```python
aa_order = ["view", "time"]  # 或 ["time", "view"]
for block_idx in range(aa_block_num):
    for attn_type in aa_order:
        if attn_type == "view":
            tokens, view_idx = self._process_view_attention(...)
        elif attn_type == "time":
            tokens, time_idx = self._process_time_attention(...)
    
    # Concat intermediates for output
    output_list.append(tokens)
```

### 3.2 两流架构实现

#### 初始化
```python
# 位姿流：用于相机位姿估计
self.pose_frame_blocks = nn.ModuleList([Block(...) for _ in range(depth)])
self.pose_view_blocks = nn.ModuleList([Block(...) for _ in range(depth)])

# 几何流：用于几何信息估计
self.geo_frame_blocks = nn.ModuleList([Block(...) for _ in range(depth)])
self.geo_view_blocks = nn.ModuleList([Block(...) for _ in range(depth)])

# 动态掩码参数
self.lambda_pose = nn.Parameter(torch.tensor(0.1))  # 可学习，需clamp
self.lambda_geo = nn.Parameter(torch.tensor(0.1))
```

#### 前向传播
```python
# 共享 tokens，但分别处理
tokens_pose = self._process_with_mask_bias(
    tokens, mask, self.lambda_pose, stream="pose"
)
tokens_geo = self._process_with_mask_bias(
    tokens, mask, self.lambda_geo, stream="geo"
)

# 位姿流输出用于 camera_decoder
# 几何流输出用于 point_decoder, depth_decoder
```

### 3.3 权重加载逻辑

```python
def load_pretrained_weights(self, checkpoint_path, pi3_path=None):
    # 1. 加载 checkpoint_150.pt
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    
    # 2. 映射权重名称
    new_state_dict = {}
    for name, param in state_dict.items():
        if 'frame_blocks' in name:
            # Frame-Attention → Time-SA blocks
            new_name = name.replace('frame_blocks', 'time_blocks')
            new_state_dict[new_name] = param
        elif 'global_blocks' in name:
            # Global-Attention → View-SA blocks
            new_name = name.replace('global_blocks', 'view_blocks')
            new_state_dict[new_name] = param
        else:
            # 其他权重直接复制
            new_state_dict[name] = param
    
    # 3. 如果提供 Pi3 路径，加载多视角相关权重
    if pi3_path:
        pi3_state_dict = torch.load(pi3_path, map_location='cpu')
        # 映射 Pi3 的 decoder blocks → View-SA blocks
        # 映射 Pi3 的 decoder blocks → Time-SA blocks（作为初始化）
    
    # 4. 加载到模型（strict=False，允许新参数存在）
    self.load_state_dict(new_state_dict, strict=False)
```

## 四、实施步骤

### Step 1: 修改导入路径
- 将所有 `vggt_t_mask_mlp_fin10` → `vggt_t_mv`
- 确保所有依赖模块都存在于 `vggt_t_mv` 中

### Step 2: 修改 Aggregator 输入处理
- 支持 `[B, T, N, C, H, W]` 输入
- 实现 View-SA 和 Time-SA 机制

### Step 3: 实现两流架构
- 复制注意力块为两个流
- 实现 mask 偏置机制

### Step 4: 实现权重加载
- 从 checkpoint_150.pt 加载基础权重
- 可选从 Pi3 加载多视角权重

### Step 5: 测试与验证
- 确保输入格式正确
- 确保权重正确加载
- 确保前向传播正常运行

## 五、注意事项

1. **向后兼容**：保留对 `[B, S, C, H, W]` 输入的支持（自动检测输入维度）
2. **权重映射**：需要仔细处理权重名称映射，确保正确加载
3. **新参数初始化**：对于新增的参数（如 `λ_pose`, `λ_geo`），需要合理初始化
4. **计算效率**：View-SA 和 Time-SA 的计算复杂度较高，可能需要：
   - Token 下采样
   - 稀疏注意力
   - 分层应用（仅在特定层使用）



