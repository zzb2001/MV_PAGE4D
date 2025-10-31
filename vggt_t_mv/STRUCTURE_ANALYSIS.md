# VGGT_MV 模型结构完整分析

## 1. 模型结构图

```
VGGT_MV
├── aggregator (Aggregator)
│   ├── patch_embed (PatchEmbed / DINOv2)
│   │   └── register_token (可学习 tokens)
│   ├── camera_token (nn.Parameter) [1, 2, 1, embed_dim]
│   ├── register_token (nn.Parameter) [1, 2, num_register_tokens, embed_dim]
│   ├── frame_blocks (ModuleList[Block]) × depth (24)
│   │   └── 用于 Time-SA（多视角模式）或 Frame-SA（单视角模式）
│   ├── global_blocks (ModuleList[Block]) × depth (24)
│   │   └── 用于 View-SA（多视角模式）或 Global-SA（单视角模式）
│   ├── view_blocks (alias to global_blocks)
│   ├── time_blocks (alias to frame_blocks)
│   │
│   ├── [可选] 两流架构 (enable_dual_stream=True)
│   │   ├── pose_frame_blocks (ModuleList[Block]) × depth
│   │   ├── pose_global_blocks (ModuleList[Block]) × depth
│   │   ├── geo_frame_blocks (ModuleList[Block]) × depth
│   │   ├── geo_global_blocks (ModuleList[Block]) × depth
│   │   ├── lambda_pose_logit (nn.Parameter)
│   │   └── lambda_geo_logit (nn.Parameter)
│   │
│   ├── [可选] Sparse Global-SA (enable_sparse_global=True)
│   │   └── memory_tokens (nn.Parameter) [1, 32, embed_dim]
│   │
│   └── spatial_mask_head (SpatialMaskHead_IMP)
│       └── 生成动态掩码用于两流架构
│
├── camera_head (CameraHead)
│   └── 输入: aggregated_tokens_list (每个 block 的输出，2*embed_dim)
│   └── 输出: pose_enc [B, T, N, 9]
│
├── point_head (DPTHead)
│   └── 输入: aggregated_tokens_list
│   └── 输出: world_points [B, T, N, H, W, 3], confidence
│
├── depth_head (DPTHead)
│   └── 输入: aggregated_tokens_list
│   └── 输出: depth [B, T, N, H, W], confidence
│
└── track_head (TrackHead)
    └── 输入: aggregated_tokens_list, query_points
    └── 输出: tracks, visibility, confidence
```

## 2. 权重键名结构

### Checkpoint 权重键名模式（vggt_t_mask_mlp_fin10）
```
aggregator.patch_embed.*
aggregator.frame_blocks.*
aggregator.global_blocks.*
aggregator.camera_token
aggregator.register_token
aggregator.spatial_mask_head.*
camera_head.*
point_head.*
depth_head.*
track_head.*
```

### VGGT_MV 模型权重键名模式
```
aggregator.patch_embed.*
aggregator.frame_blocks.*
aggregator.global_blocks.*
aggregator.view_blocks.* (alias, 实际指向 global_blocks)
aggregator.time_blocks.* (alias, 实际指向 frame_blocks)
aggregator.camera_token
aggregator.register_token
aggregator.spatial_mask_head.*
camera_head.*
point_head.*
depth_head.*
track_head.*
```

## 3. 问题分析

### 当前权重加载失败的原因

**问题1: 只加载 aggregator，忽略其他模块**
- `load_checkpoint_weights(checkpoint_path, self.aggregator, ...)` 只加载到 aggregator
- Checkpoint 包含 `camera_head.*`, `point_head.*`, `depth_head.*`, `track_head.*`
- 这些权重被完全忽略，导致 `missing = 2082`

**问题2: 权重键名匹配逻辑问题**
- 代码只检查 `name in model_state_dict`，但传入的是 `self.aggregator` 而不是整个模型
- 所以即使 checkpoint 有 `camera_head.*`，aggregator 的 state_dict 中没有，无法匹配

**问题3: Pi3 权重加载可能因为键名不匹配失败**
- Pi3 的键名模式是 `decoder.*`，需要映射到 `aggregator.global_blocks.*` 或 `aggregator.frame_blocks.*`
- 映射逻辑可能存在问题

## 4. 权重键名映射关系

### Checkpoint → VGGT_MV 映射

| Checkpoint 键名 | VGGT_MV 键名 | 状态 |
|----------------|--------------|------|
| `aggregator.frame_blocks.*` | `aggregator.frame_blocks.*` | ✅ 直接匹配 |
| `aggregator.global_blocks.*` | `aggregator.global_blocks.*` | ✅ 直接匹配 |
| `aggregator.patch_embed.*` | `aggregator.patch_embed.*` | ✅ 直接匹配 |
| `aggregator.camera_token` | `aggregator.camera_token` | ✅ 直接匹配 |
| `aggregator.register_token` | `aggregator.register_token` | ✅ 直接匹配 |
| `aggregator.spatial_mask_head.*` | `aggregator.spatial_mask_head.*` | ✅ 直接匹配 |
| `camera_head.*` | `camera_head.*` | ❌ **当前被忽略** |
| `point_head.*` | `point_head.*` | ❌ **当前被忽略** |
| `depth_head.*` | `depth_head.*` | ❌ **当前被忽略** |
| `track_head.*` | `track_head.*` | ❌ **当前被忽略** |

### Pi3 → VGGT_MV 映射

| Pi3 键名 | VGGT_MV 键名 | 映射逻辑 |
|---------|--------------|---------|
| `decoder.{i}.attn.*` (i%2==0) | `aggregator.frame_blocks.{i//2}.attn.*` | Frame → Time-SA |
| `decoder.{i}.attn.*` (i%2==1) | `aggregator.global_blocks.{i//2}.attn.*` | Global → View-SA |
| `encoder.register_token` | `aggregator.register_token` | 直接映射 |

## 5. 修复方案

### 方案1: 加载整个模型权重（推荐）
```python
# 应该加载到整个模型，而不是只加载 aggregator
stats = model_mv.load_pretrained_weights(
    checkpoint_path=origin,
    pi3_path=pi3_path,
    device=device
)

# 在 load_pretrained_weights 中：
def load_pretrained_weights(self, checkpoint_path, pi3_path=None, device='cpu'):
    # 1. 加载完整 checkpoint（包括所有 heads）
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 映射到整个模型
        model_dict = self.state_dict()
        mapped_dict = {}
        for key, param in checkpoint_dict.items():
            if key in model_dict and model_dict[key].shape == param.shape:
                mapped_dict[key] = param
        
        missing, unexpected = self.load_state_dict(mapped_dict, strict=False)
    
    # 2. 加载 Pi3 权重（只加载到 aggregator）
    if pi3_path:
        pi3_dict, _, _ = load_pi3_weights(pi3_path, self.aggregator, device)
```

### 方案2: 分阶段加载
1. 先加载 aggregator 权重（从 checkpoint）
2. 再加载 heads 权重（从 checkpoint）
3. 最后加载 Pi3 权重（只到 aggregator）

## 6. 合理性分析

### 当前设计是否合理？

**部分合理：**
- ✅ Aggregator 的权重应该从 checkpoint 加载（frame_blocks, global_blocks 等）
- ✅ Pi3 权重只应该加载到 aggregator 的 global_blocks（用于 View-SA）
- ✅ 保持 heads 的权重应该从 checkpoint 加载（camera_head, point_head 等）

**不合理的地方：**
- ❌ 当前只加载 aggregator，丢失了 heads 的权重
- ❌ 导致 2082 个参数未加载，可能影响性能
- ❌ 权重加载逻辑应该更灵活，支持选择性加载

### 建议的修复
1. **修改 `load_pretrained_weights`**：加载到整个模型，而不是只加载 aggregator
2. **保持 Pi3 加载逻辑**：Pi3 权重只加载到 aggregator（因为 Pi3 只有 encoder/decoder，没有 heads）
3. **添加日志**：详细记录哪些权重加载成功，哪些被跳过



