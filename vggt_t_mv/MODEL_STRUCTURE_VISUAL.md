# VGGT_MV 模型结构可视化

## 1. 完整模型结构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         VGGT_MV Model                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼─────┐      ┌──────▼──────┐
              │Aggregator │      │    Heads    │
              │           │      │             │
              │ [核心模块] │      │ [输出模块]   │
              └─────┬─────┘      └──────┬──────┘
                    │                   │
        ┌───────────┼───────────┐      │
        │           │           │      │
┌───────▼───┐  ┌─────▼─────┐  ┌──▼────┐ │
│PatchEmbed │  │Attention  │  │Tokens │ │
│(DINOv2)  │  │  Blocks   │  │Output │ │
└───────────┘  └───────────┘  └───────┘ │
                                         │
                        ┌────────────────┼────────────────┐
                        │                │                │
                  ┌─────▼─────┐   ┌─────▼─────┐   ┌──────▼─────┐
                  │CameraHead │   │PointHead  │   │DepthHead   │
                  │           │   │           │   │            │
                  │ pose_enc  │   │world_points│  │  depth     │
                  └───────────┘   └───────────┘   └────────────┘
```

## 2. Aggregator 详细结构

```
Aggregator
│
├── patch_embed (DINOv2 ViT-L/14)
│   ├── blocks[0..23] (24 ViT blocks)
│   ├── cls_token
│   ├── pos_embed
│   ├── register_tokens
│   └── mask_token
│
├── Special Tokens
│   ├── camera_token [1, 2, 1, 1024]
│   │   ├── [0]: first frame token
│   │   └── [1]: rest frames token
│   └── register_token [1, 2, 4, 1024]
│       ├── [0]: first frame tokens
│       └── [1]: rest frames tokens
│
├── Attention Blocks (交替注意力)
│   ├── frame_blocks (ModuleList[Block × 24])
│   │   └── 用途:
│   │       - 单视角: Frame-SA (时序内空间注意力)
│   │       - 多视角: Time-SA (固定视角v，跨时间t聚合)
│   │
│   └── global_blocks (ModuleList[Block × 24])
│       └── 用途:
│           - 单视角: Global-SA (全局空间注意力)
│           - 多视角: View-SA (固定时间t，跨视角N聚合)
│
├── Aliases (为了权重加载)
│   ├── time_blocks = frame_blocks
│   └── view_blocks = global_blocks
│
├── [可选] 两流架构 (enable_dual_stream=True)
│   ├── pose_frame_blocks (ModuleList[Block × 24])
│   ├── pose_global_blocks (ModuleList[Block × 24])
│   ├── geo_frame_blocks (ModuleList[Block × 24])
│   ├── geo_global_blocks (ModuleList[Block × 24])
│   ├── lambda_pose_logit (可学习参数)
│   └── lambda_geo_logit (可学习参数)
│
├── [可选] Sparse Global-SA
│   ├── memory_tokens [1, 32, 1024] (如果 strategy="memory_bank")
│   └── landmark_k = 64 (如果 strategy="landmark")
│
└── spatial_mask_head
    └── 生成动态掩码用于两流架构
```

## 3. 数据流图

### 单视角模式 (Backward Compatible)
```
Input: [B, S, 3, H, W]
  │
  ├─► PatchEmbed → [B*S, P, C] patches
  │
  ├─► Concat: [camera_token, register_token, patches]
  │            → [B*S, 1+R+P, C]
  │
  └─► Alternating Attention:
      │
      ├─► Frame-SA (frame_blocks)
      │   └─► [B*S, 1+R+P, C]  (时序内空间注意力)
      │
      └─► Global-SA (global_blocks)
          └─► [B*S, 1+R+P, C]  (全局空间注意力)
      │
      └─► Output: [B*S, 1+R+P, 2C] (concat intermediates)
          │
          └─► Heads → Predictions
```

### 多视角模式 (新功能)
```
Input: [B, T, N, 3, H, W]
  │
  ├─► PatchEmbed → [B*T*N, P, C] patches
  │
  ├─► Reshape: [B, T, N, P, C]
  │
  ├─► Concat: [camera_token, register_token, patches]
  │            → [B, T, N, 1+R+P, C]
  │
  └─► Alternating Attention:
      │
      ├─► View-SA (global_blocks/view_blocks)
      │   └─► Reshape: [B, T, N, P, C] → [B*T, N*P, C]
      │       └─► Apply global_blocks
      │           └─► 固定时间t，跨视角N聚合
      │
      └─► Time-SA (frame_blocks/time_blocks)
          └─► Reshape: [B, T, N, P, C] → [B*N, T*P, C]
              └─► Apply frame_blocks
                  └─► 固定视角v，跨时间T聚合
      │
      └─► Output: [B, T, N, P, 2C] (concat intermediates)
          │
          └─► Heads → Predictions
```

## 4. 权重键名映射表

### Checkpoint → VGGT_MV 映射

| Checkpoint 键名模式 | VGGT_MV 键名模式 | 匹配状态 | 参数数量 |
|-------------------|----------------|---------|---------|
| `aggregator.patch_embed.*` | `aggregator.patch_embed.*` | ✅ 完全匹配 | ~200 |
| `aggregator.frame_blocks.*` | `aggregator.frame_blocks.*` | ✅ 完全匹配 | ~800 |
| `aggregator.global_blocks.*` | `aggregator.global_blocks.*` | ✅ 完全匹配 | ~800 |
| `aggregator.camera_token` | `aggregator.camera_token` | ✅ 完全匹配 | 2 |
| `aggregator.register_token` | `aggregator.register_token` | ✅ 完全匹配 | 8 |
| `aggregator.spatial_mask_head.*` | `aggregator.spatial_mask_head.*` | ✅ 完全匹配 | ~50 |
| `camera_head.*` | `camera_head.*` | ❌ **修复前被忽略** | ~200 |
| `point_head.*` | `point_head.*` | ❌ **修复前被忽略** | ~400 |
| `depth_head.*` | `depth_head.*` | ❌ **修复前被忽略** | ~400 |
| `track_head.*` | `track_head.*` | ❌ **修复前被忽略** | ~80 |

**总计**: ~1411 个键，~2500-3000 个参数张量

### Pi3 → VGGT_MV 映射 (部分)

| Pi3 键名 | VGGT_MV 键名 | 映射规则 |
|---------|--------------|---------|
| `decoder.0.*` | `aggregator.frame_blocks.0.*` | 偶数索引 → frame_blocks |
| `decoder.1.*` | `aggregator.global_blocks.0.*` | 奇数索引 → global_blocks |
| `decoder.2.*` | `aggregator.frame_blocks.1.*` | 交替映射 |
| `decoder.3.*` | `aggregator.global_blocks.1.*` | ... |
| `encoder.register_token` | `aggregator.register_token` | 直接映射 |

## 5. 问题诊断结果

### 原问题
```
checkpoint_loaded = 0
pi3_loaded = 0  
missing = 2082
```

### 根本原因
1. **只加载 aggregator**: `load_checkpoint_weights(..., self.aggregator)` 只匹配 aggregator 的键
2. **Heads 权重被忽略**: checkpoint 中的 `camera_head.*`, `point_head.*` 等无法在 `aggregator.state_dict()` 中找到
3. **导致**: 所有 1411 个键都无法匹配，`checkpoint_loaded = 0`

### 修复后预期
```
checkpoint_loaded = ~1411 (大部分键匹配)
pi3_loaded = ~200-500 (部分 Pi3 权重匹配)
missing = ~0-100 (新增的参数，合理)
```

## 6. 修复验证

修复后的代码会：
1. ✅ 加载整个模型的 checkpoint 权重（包括 aggregator + heads）
2. ✅ 匹配所有可能的键（aggregator.*, camera_head.*, point_head.* 等）
3. ✅ 只跳过真正不存在的键（如两流架构的新参数）
4. ✅ 提供详细的加载日志

### 预期日志输出
```
Loading checkpoint from checkpoint/checkpoint_150.pt...
Loaded 1411 parameters from checkpoint
  - Missing keys: 0-100 (新增参数)
  - Unexpected keys: 0
  - Skipped keys (not in model): 0
Sample loaded keys: ['aggregator.camera_token', 'aggregator.frame_blocks.0.attn.qkv.weight', ...]
```



