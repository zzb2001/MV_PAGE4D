# Epoch和Batch数量说明

## 当前配置

### 1. Dataset大小（`MultiViewTemporalDataset.__len__()`）

```python
def __len__(self):
    # 对于过拟合场景，可以返回一个较大的数字，每次访问都会随机增强
    # 或者返回固定的样本数（例如1000）
    return 1000  # 可以根据需要调整
```

**当前值**：`1000`（硬编码）

### 2. DataLoader配置

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=1,           # 批量大小
    shuffle=True,            # 每个epoch打乱
    num_workers=0,          # 多进程数量
    pin_memory=True,
    drop_last=False         # 不丢弃最后一个不完整的batch
)
```

**关键参数**：
- `batch_size = 1`
- `drop_last = False`

## Batch数量计算

### 每个Epoch的Batch数量

```
每个epoch的batch数 = len(dataset) / batch_size
                  = 1000 / 1
                  = 1000 batches
```

**结论**：**1个epoch = 1000个batch**

## 实际数据加载机制

### Dataset的`__getitem__()`行为

虽然`__len__()`返回1000，但每次调用`__getitem__()`时：

1. **时序随机切片**（如果启用`enable_temporal_slice=True`）：
   - 随机选择窗口大小（从`T_window_sizes=[2]`中选择）
   - 随机选择起始时间（从`T_total`中随机选择）
   - 每次返回的数据可能不同

2. **视角打乱**（如果启用`enable_view_permutation=True`）：
   - 随机打乱视角顺序
   - 每次返回的视角顺序可能不同

3. **数据增强**（如果启用`enable_intra_frame_aug=True`）：
   - 颜色抖动
   - 轻微旋转
   - 每次返回的图像可能略有不同

### 数据生成模式

**当前模式**：**无限采样模式（Infinite Sampling）**

- `__len__()`返回固定值（1000）
- 每次`__getitem__()`都会动态生成数据（通过随机切片和打乱）
- 同一个索引可能返回不同的数据
- 适合小数据集或过拟合场景

## 修改建议

### 如果想要基于实际数据量的batch数

如果需要根据实际数据量计算batch数，可以修改`__len__()`：

```python
def __len__(self):
    # 选项1: 基于实际时间帧数计算
    # 如果启用时序切片，可能的组合数
    if self.enable_temporal_slice:
        total_combinations = 0
        for T_window in self.T_window_sizes:
            # 对于每个窗口大小，可能的起始位置数
            possible_starts = max(0, self.T_total - T_window + 1)
            total_combinations += possible_starts
        return total_combinations
    else:
        # 不使用切片时，只有1个样本（所有帧）
        return 1
    
    # 选项2: 固定一个较大的数字（当前做法）
    # return 1000
    
    # 选项3: 返回实际时间帧数
    # return self.T_total
```

### 如果想要更大的batch

如果要增加每个batch的样本数（当前是1），可以：

1. **增加batch_size**：
   ```python
   train_loader = DataLoader(
       train_dataset,
       batch_size=2,  # 或更大的值
       ...
   )
   ```
   **注意**：由于每个样本已经是`[T, V, C, H, W]`（T=2，V=4），batch_size=2意味着`[2, 2, 4, 3, H, W]`，可能内存占用较大

2. **保持batch_size=1，但调整Dataset的时序窗口大小**：
   - 当前：`T_window_sizes=[2]`
   - 可以改为：`T_window_sizes=[4, 6, 8]`（更大的窗口，每个batch包含更多时间帧）

## 当前训练的实际情况

### 每个Batch包含的数据

- **形状**：`[1, T, V, C, H, W]`（batch_size=1）
- **T**：根据`T_window_sizes`随机选择（当前只有`[2]`，所以T=2）
- **V**：从数据目录中读取的视角数（例如4）
- **数据内容**：每次都可能不同（因为随机切片和打乱）

### 每个Epoch

- **Batch数量**：1000个
- **总时间帧处理**：1000 × T_window（例如1000 × 2 = 2000个时间帧-视角对）
- **数据多样性**：由于随机增强，同一个epoch内的1000个batch可能包含不同的数据组合

## 建议的修改

如果想基于实际数据量，建议修改`__len__()`：

```python
def __len__(self):
    """
    返回Dataset的大小。
    
    如果启用时序切片，返回可能的组合数。
    否则返回1（只有一种组合：所有帧）。
    """
    if self.enable_temporal_slice:
        # 计算所有可能的时序窗口组合
        total_samples = 0
        for T_window in self.T_window_sizes:
            T_window = min(T_window, self.T_total)
            # 对于每个窗口大小，可能的起始位置数
            possible_starts = max(1, self.T_total - T_window + 1)
            total_samples += possible_starts
        # 乘以视角打乱的组合数（如果启用）
        if self.enable_view_permutation:
            # 视角打乱有V!种组合，但为了计算方便，可以简化为V
            view_combinations = self.V
            total_samples *= view_combinations
        return total_samples
    else:
        # 不使用切片时，返回1
        return 1
```

或者，如果想保持当前的行为（无限采样），可以保持`return 1000`不变。

## 总结

- **当前配置**：1个epoch = 1000个batch
- **Batch数量由以下决定**：
  1. `Dataset.__len__()`的返回值（当前硬编码为1000）
  2. `DataLoader.batch_size`（当前为1）
  3. `drop_last`参数（当前为False，不影响batch数）
- **每个batch的大小**：由`batch_size`决定（当前为1，意味着每个batch包含1个多视角时序样本）

