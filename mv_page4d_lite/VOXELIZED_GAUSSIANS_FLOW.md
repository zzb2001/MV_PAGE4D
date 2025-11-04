# AnySplat 中 Voxelized 3D Gaussians 生成流程代码标注

本文档详细标注了 AnySplat 中从输入图像生成 Voxelized 3D Gaussians 的完整流程和对应代码位置。

---

## 📍 主要流程概览

```
输入图像 (image)
    ↓
步骤1: VGGT Encoder 特征提取 (Line 525-529)
    ↓
步骤2: 预测相机姿态和深度图 (Line 532-552)
    ↓
步骤3: 深度图反投影到3D点云 (Line 550-552)
    ↓
步骤4: Gaussian参数预测头 (Line 564-571)
    ↓
步骤5: 【核心】体素化与特征融合 (Line 582-597)
    ↓
步骤6: Opacity和密度提取 (Line 608-621)
    ↓
步骤7: Gaussian Adapter转换 (Line 653-658)
    ↓
输出: Gaussians对象 (包含means, covariances, harmonics, opacities等)
```

---

## 🔍 详细代码位置标注

### **文件**: `src/model/encoder/anysplat.py`

---

### **步骤1: VGGT Encoder 特征聚合**
**位置**: `EncoderAnySplat.forward()` - Line 525-529

```python
# 位置: Line 525-529
with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
    aggregated_tokens_list, patch_start_idx = self.aggregator(
        image.to(torch.bfloat16),
        intermediate_layer_idx=self.cfg.intermediate_layer_idx,
    )
```
**作用**: 使用VGGT聚合器对多视角图像进行特征提取，生成聚合的token列表。

---

### **步骤2: 预测相机姿态和深度/点云**
**位置**: `EncoderAnySplat.forward()` - Line 531-552

```python
# 位置: Line 532-536
pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
last_pred_pose_enc = pred_pose_enc_list[-1]
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    last_pred_pose_enc, image.shape[-2:]
)

# 位置: Line 538-552
if self.cfg.pred_head_type == "point":
    pts_all, pts_conf = self.point_head(
        aggregated_tokens_list,
        images=image,
        patch_start_idx=patch_start_idx,
    )
elif self.cfg.pred_head_type == "depth":
    depth_map, depth_conf = self.depth_head(
        aggregated_tokens_list,
        images=image,
        patch_start_idx=patch_start_idx,
    )
    pts_all = batchify_unproject_depth_map_to_point_map(
        depth_map, extrinsic, intrinsic
    )
```
**作用**: 
- 预测相机外参(extrinsic)和内参(intrinsic)
- 预测深度图或直接预测3D点云
- 将深度图反投影到3D空间得到 `pts_all` (形状: `[B, V, H, W, 3]`)

---

### **步骤3: Gaussian参数预测头**
**位置**: `EncoderAnySplat.forward()` - Line 564-571

```python
# 位置: Line 564-571
out = self.gaussian_param_head(
    aggregated_tokens_list,
    pts_all.flatten(0, 1).permute(0, 3, 1, 2),
    image,
    patch_start_idx=patch_start_idx,
    image_size=(h, w),
)
```
**作用**: 
- 输入: 聚合特征tokens + 3D点云 + 原始图像
- 输出: `out` 包含每个像素的Gaussian参数
  - `out[:, :, :self.raw_gs_dim]` = `anchor_feats` (包含opacity + scales + rotations + SH系数)
  - `out[:, :, self.raw_gs_dim]` = `conf` (置信度)

---

### **步骤4: 【核心】体素化与特征融合**
**位置**: `EncoderAnySplat.forward()` - Line 579-597

```python
# 位置: Line 579
anchor_feats, conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

# 位置: Line 581-597
neural_feats_list, neural_pts_list = [], []
if self.cfg.voxelize:  # 如果启用体素化
    for b_i in range(b):
        # 【关键】调用体素化融合函数
        neural_pts, neural_feats = self.voxelizaton_with_fusion(
            anchor_feats[b_i],
            pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
            self.voxel_size,
            conf=conf[b_i],
        )
        neural_feats_list.append(neural_feats)
        neural_pts_list.append(neural_pts)
else:  # 不使用体素化，直接按置信度mask筛选
    for b_i in range(b):
        neural_feats_list.append(
            anchor_feats[b_i].permute(0, 2, 3, 1)[conf_valid_mask[b_i]]
        )
        neural_pts_list.append(pts_all[b_i][conf_valid_mask[b_i]])
```
**作用**: 
- **如果 `voxelize=True`**: 调用 `voxelizaton_with_fusion()` 将像素级Gaussians合并到体素中
- **如果 `voxelize=False`**: 直接按置信度mask筛选像素级Gaussians

---

### **步骤5: 【核心算法】体素化融合函数详解**
**位置**: `EncoderAnySplat.voxelizaton_with_fusion()` - Line 409-446

```python
# 位置: Line 409-446
def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
    """
    输入:
        img_feat: [B*V, C, H, W] - 图像特征
        pts3d: [B*V, 3, H, W] - 3D点坐标
        voxel_size: 体素大小
        conf: [B*V, H, W] - 置信度
    输出:
        voxel_pts: [num_unique_voxels, 3] - 体素中心坐标
        voxel_feats: [num_unique_voxels, feat_dim] - 融合后的特征
    """
    V, C, H, W = img_feat.shape
    pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, 3]
    
    # 【步骤5.1】计算体素索引 (Line 415)
    voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
    
    # 【步骤5.2】找到唯一体素 (Line 416-418)
    unique_voxels, inverse_indices, counts = torch.unique(
        voxel_indices, dim=0, return_inverse=True, return_counts=True
    )
    
    # 【步骤5.3】展平置信度和特征 (Line 421-422)
    conf_flat = conf.flatten()  # [B*V*N]
    anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, feat_dim]
    
    # 【步骤5.4】基于置信度的Softmax加权融合 (Line 425-432)
    conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
    conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
    voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)
    weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)
    
    # 【步骤5.5】加权平均位置和特征 (Line 434-436)
    weighted_pts = pts3d_flatten * weights
    weighted_feats = anchor_feats_flat.squeeze(1) * weights
    
    # 【步骤5.6】按体素聚合 (Line 438-444)
    voxel_pts = scatter_add(weighted_pts, inverse_indices, dim=0)  # [num_unique_voxels, 3]
    voxel_feats = scatter_add(weighted_feats, inverse_indices, dim=0)  # [num_unique_voxels, feat_dim]
    
    return voxel_pts, voxel_feats
```
**算法说明**:
1. **体素索引计算**: 将3D点坐标除以体素大小并取整，得到每个点所属的体素索引
2. **唯一体素提取**: 使用 `torch.unique` 找出所有唯一的体素
3. **置信度加权**: 使用置信度的softmax作为权重，对落在同一体素中的多个像素点进行加权融合
4. **位置和特征聚合**: 使用 `scatter_add` 将加权后的位置和特征聚合到对应体素

**关键特点**:
- 减少Gaussian数量：从像素级 (H×W×V) 减少到体素级 (num_unique_voxels)
- 保持特征质量：通过置信度加权融合，保留高质量特征信息
- 空间一致性：同一体素内的多个观测被融合，提高空间一致性

---

### **步骤6: Padding和维度统一**
**位置**: `EncoderAnySplat.forward()` - Line 599-606

```python
# 位置: Line 599-606
max_voxels = max(f.shape[0] for f in neural_feats_list)
neural_feats = self.pad_tensor_list(
    neural_feats_list, (max_voxels,), value=-1e10
)
neural_pts = self.pad_tensor_list(
    neural_pts_list, (max_voxels,), -1e4
)  # -1e4 == invalid voxel marker
```
**作用**: 将不同batch的体素数量pad到相同长度，便于batch处理。

---

### **步骤7: Opacity和深度提取**
**位置**: `EncoderAnySplat.forward()` - Line 608-621

```python
# 位置: Line 608-609
depths = neural_pts[..., -1].unsqueeze(-1)  # 从3D点中提取深度
densities = neural_feats[..., 0].sigmoid()   # 第一个特征维度是density

# 位置: Line 614
opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)

# 位置: Line 615-621 (可选，如果启用opacity_conf)
if self.cfg.opacity_conf:
    shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
    opacity = opacity * torch.sigmoid(depth_conf - shift)[
        conf_valid_mask
    ].unsqueeze(0)
```
**作用**: 
- 从体素化后的特征中提取深度和密度
- 使用 `map_pdf_to_opacity()` 将密度转换为opacity
- 可选：使用置信度进一步调整opacity

---

### **步骤8: Gaussian剪枝 (可选)**
**位置**: `EncoderAnySplat.forward()` - Line 626-651

```python
# 位置: Line 626-651
if gs_prune and b == 1:
    opacity_threshold = self.cfg.opacity_threshold
    gaussian_usage = opacity > opacity_threshold  # (B, N)
    
    # 如果保留比例超过阈值，按opacity排序保留前N个
    if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
        num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
        idx_sort = opacity.argsort(dim=1, descending=True)
        keep_idx = idx_sort[:, :num_keep]
        gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
        gaussian_usage.scatter_(1, keep_idx, True)
    
    # 根据usage mask筛选Gaussians
    neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
    depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
    neural_feats = neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
    opacity = opacity[gaussian_usage].view(b, -1).contiguous()
```
**作用**: 根据opacity阈值或保留比例，剪枝掉低质量的Gaussians。

---

### **步骤9: 【最终】转换为Gaussians对象**
**位置**: `EncoderAnySplat.forward()` - Line 653-658

```python
# 位置: Line 653-658
gaussians = self.gaussian_adapter.forward(
    neural_pts,        # [B, N, 3] - 体素化的3D位置
    depths,           # [B, N, 1] - 深度
    opacity,          # [B, N] - 不透明度
    neural_feats[..., 1:].squeeze(2),  # [B, N, d_in] - Gaussian参数特征
)
```

**GaussianAdapter的作用** (见 `src/model/encoder/common/gaussian_adapter.py`):
1. **参数分解** (Line 125): 将 `neural_feats` 分解为 `scales`, `rotations`, `sh` (球谐系数)
2. **尺度映射** (Line 127-128): 使用softplus将尺度特征映射到合理范围
3. **四元数归一化** (Line 131): 归一化旋转四元数
4. **协方差矩阵构建** (Line 136): 从scale和rotation构建协方差矩阵
5. **返回Gaussians对象** (Line 138-146):
   - `means`: 3D位置 (`neural_pts`)
   - `covariances`: 协方差矩阵
   - `harmonics`: 球谐系数
   - `opacities`: 不透明度
   - `scales`: 尺度参数
   - `rotations`: 旋转四元数

---

## 📊 数据流变换总结

| 步骤 | 变量名 | 形状 | 说明 |
|------|--------|------|------|
| 输入 | `image` | `[B, V, 3, H, W]` | 多视角图像 |
| 步骤1 | `aggregated_tokens_list` | `List[Tensor]` | 聚合特征tokens |
| 步骤2 | `pts_all` | `[B, V, H, W, 3]` | 像素级3D点云 |
| 步骤3 | `anchor_feats` | `[B, V, H, W, d_gs]` | 像素级Gaussian参数 |
| **步骤4** | **`voxel_pts`** | **`[num_voxels, 3]`** | **体素化3D位置** |
| **步骤4** | **`voxel_feats`** | **`[num_voxels, d_gs]`** | **体素化特征** |
| 步骤5 | `neural_pts` | `[B, max_voxels, 3]` | Padding后的体素位置 |
| 步骤5 | `neural_feats` | `[B, max_voxels, d_gs]` | Padding后的体素特征 |
| 步骤6 | `depths`, `opacity` | `[B, max_voxels]` | 深度和不透明度 |
| 步骤7 | `gaussians` | `Gaussians` | 最终的Gaussians对象 |

**关键数量变化**:
- **体素化前**: `H × W × V` 个像素级Gaussians
- **体素化后**: `num_unique_voxels` 个体素级Gaussians (通常 << H×W×V)

---

## 🔧 配置参数

相关配置在 `EncoderAnySplatCfg` (Line 194-236):
- `voxelize: bool = False` (Line 236): 是否启用体素化
- `voxel_size: float`: 体素大小 (Line 197, 323)
- `opacity_threshold: float = 0.001`: Opacity剪枝阈值 (Line 217)
- `gs_keep_ratio: float = 1.0`: Gaussian保留比例 (Line 218)

---

## 💡 关键代码文件索引

1. **主流程**: `src/model/encoder/anysplat.py`
   - `EncoderAnySplat.forward()`: Line 448-702
   - `voxelizaton_with_fusion()`: Line 409-446

2. **Gaussian适配器**: `src/model/encoder/common/gaussian_adapter.py`
   - `UnifiedGaussianAdapter.forward()`: Line 114-146

3. **Gaussian类型定义**: `src/model/types.py`
   - `Gaussians` dataclass: Line 7-15

---

## 📝 总结

**Voxelized 3D Gaussians 生成的核心流程**:
1. ✅ 从像素级预测 (H×W×V个Gaussians)
2. ✅ **体素化融合** (`voxelizaton_with_fusion`) - **关键步骤**
3. ✅ 减少到体素级 (num_voxels个Gaussians)
4. ✅ 转换为Gaussians对象

**体素化的优势**:
- 🎯 大幅减少Gaussian数量，提高渲染效率
- 🎯 通过加权融合提高空间一致性和特征质量
- 🎯 自然处理多视角观测的融合

