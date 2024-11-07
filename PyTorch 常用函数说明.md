# PyTorch 常用函数说明

## torch.roll
```python
torch.roll(input: Tensor, shifts: int | Sequence[int], dims: int | Sequence[int] = None) -> Tensor
```
- **功能**: 在指定维度上循环移动张量元素。
- **参数**:
    - `input`: 被移动的张量。
    - `shifts`: 移动的个数，可以是整数或整数序列。正数向右移动，负数向左移动。
    - `dims`: 指定移动的维度，可以是整数或整数序列。如果未指定，则对所有维度进行移动。
- **返回值**: 移动后的张量。

```python
# 示例
import torch
x = torch.tensor([1, 2, 3, 4])
y = torch.roll(x, shifts=2)
print(y)  # 输出: tensor([3, 4, 1, 2])
```


## torch.cat
```python
torch.cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor
```

- **功能**: 沿指定维度连接多个张量。
- **参数**:
    - tensors: 要连接的张量序列。每个张量的形状在其他维度上必须相同。
    - dim: 指定连接的维度，默认为0。
- **返回值**: 连接后的张量。

```python
# 示例
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6]])
c = torch.cat((a, b), dim=0)
print(c)  # 输出: tensor([[1, 2], [3, 4], [5, 6]])
```


## torch.stack
```python
torch.stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor
```

- **功能**: 在指定维度上堆叠张量，增加一个新的维度。
- **参数**:
    - tensors: 要堆叠的张量序列。所有张量的形状必须相同。
    - dim: 指定新维度插入的位置，默认为0。
- **返回值**: 堆叠后的张量。

```python
# 示例
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.stack((a, b), dim=0)
print(c)  # 输出: tensor([[1, 2, 3], [4, 5, 6]])
```


## torch.reshape
```python
torch.reshape(input: Tensor, shape: Tuple[int, ...]) -> Tensor
```

- **功能**: 重新调整张量的形状，不改变数据。
- **参数**:
    - input: 输入的张量。
    - shape: 新的形状，可以是一个整数元组，某一维度可设为-1自动推断。
- **返回值**: 重新调整形状后的张量。

```python
# 示例
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.reshape(x, (3, 2))
print(y)  # 输出: tensor([[1, 2], [3, 4], [5, 6]])
```


## torch.transpose
```python
torch.transpose(input: Tensor, dim0: int, dim1: int) -> Tensor
```

- **功能**: 交换张量的两个维度。
- **参数**:
    - input: 输入的张量。
    - dim0: 要交换的第一个维度。
    - dim1: 要交换的第二个维度。
- **返回值**: 维度交换后的张量。

```python
# 示例
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.transpose(x, 0, 1)
print(y)  # 输出: tensor([[1, 4], [2, 5], [3, 6]])
```


## torch.matmul
```python
torch.matmul(input: Tensor, other: Tensor) -> Tensor
```

- **功能**: 矩阵乘法或批次矩阵乘法。
- **参数**:
    - input: 第一个矩阵或批次矩阵。
    - other: 第二个矩阵或批次矩阵。
- **返回值**: 矩阵乘法后的结果。

```python
# 示例
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.matmul(a, b)
print(c)  # 输出: tensor([[19, 22], [43, 50]])
```


## torch.nn.functional.relu
```python
torch.nn.functional.relu(input: Tensor, inplace: bool = False) -> Tensor
```

- **功能**: 计算ReLU激活函数。
- **参数**:
    - input: 输入张量。
    - inplace: 是否在原地操作。
- **返回值**: ReLU激活后的张量。

```python
# 示例
import torch.nn.functional as F
x = torch.tensor([-1.0, 0.0, 1.0])
y = F.relu(x)
print(y)  # 输出: tensor([0., 0., 1.])
```


## torch.nn.CrossEntropyLoss
```python
torch.nn.CrossEntropyLoss(weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduction: str = 'mean') -> Tensor
```

- **功能**: 计算交叉熵损失，用于分类问题。
- **参数**:
    - weight: 权重张量，为每个类别赋予不同权重。
    - ignore_index: 忽略特定标签的损失计算。
    - reduction: 定义输出方式（"none"、"mean"、"sum"）。
- **返回值**: 交叉熵损失值。

```python
# 示例
loss_fn = torch.nn.CrossEntropyLoss()
input = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
target = torch.tensor([1, 0])
loss = loss_fn(input, target)
print(loss)  # 输出: 交叉熵损失值
```


## torch.nn.Conv2d
```python
torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int] = 1, padding: int | Tuple[int, int] = 0, dilation: int | Tuple[int, int] = 1, groups: int = 1, bias: bool = True) -> Tensor
```

- **功能**: 2D卷积层，用于图像卷积操作。
- **参数**:
    - in_channels: 输入通道数。
    - out_channels: 输出通道数。
    - kernel_size: 卷积核大小。
    - stride: 步长。
    - padding: 填充大小。
    - dilation: 卷积核膨胀大小。
    - groups: 卷积分组数量。
    - bias: 是否使用偏置。
- **返回值**: 卷积后的张量。

```python
# 示例
conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
input_tensor = torch.randn(1, 1, 5, 5)
output_tensor = conv_layer(input_tensor)
print(output_tensor.shape)  # 输出卷积后的张量形状
```



## torch.zeros
```python
torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

- **功能**: 返回一个指定形状的全零张量。
- **参数**:
    - size: 输出张量的形状。
    - dtype: 数据类型。
    - device: 存储张量的设备。
    - requires_grad: 是否记录梯度。
- **返回值**: 全零张量。

```python
# 示例
x = torch.zeros(2, 3)
print(x)  # 输出: tensor([[0., 0., 0.], [0., 0., 0.]])
```


## torch.ones
```python
torch.ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

- **功能**: 返回一个指定形状的全一张量。
- **参数**:
    - size: 输出张量的形状。
    - dtype: 数据类型。
    - device: 存储张量的设备。
    - requires_grad: 是否记录梯度。
- **返回值**: 全一张量。

```python
# 示例
x = torch.ones(2, 3)
print(x)  # 输出: tensor([[1., 1., 1.], [1., 1., 1.]])
```


## torch.eye
```python
torch.eye(n: int, m: Optional[int] = None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

- **功能**: 返回一个二维单位矩阵（对角元素为1，其余元素为0）。
- **参数**:
    - n: 行数。
    - m: 列数，默认与行数相同。
- **返回值**: 单位矩阵。

```python
# 示例
x = torch.eye(3)
print(x)  # 输出: tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
```


## torch.arange
```python
torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

- **功能**: 返回一个从start到end间隔为step的1维张量。
- **参数**:
    - start: 起始值。
    - end: 结束值（不包含）。
    - step: 间隔值。
- **返回值**: 1维张量。

```python
# 示例
x = torch.arange(0, 5, 1)
print(x)  # 输出: tensor([0, 1, 2, 3, 4])
```


## torch.nn.Linear
```python
torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
```

- **功能**: 全连接层，执行线性变换。
- **参数**:
    - in_features: 输入特征数。
    - out_features: 输出特征数。
    - bias: 是否使用偏置。
- **返回值**: 线性变换后的张量。

```python
# 示例
linear = torch.nn.Linear(3, 2)
input = torch.tensor([[1.0, 2.0, 3.0]])
output = linear(input)
print(output)  # 输出: 线性变换后的张量
```


## torch.mean
```python
torch.mean(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor
```

- **功能**: 计算张量在指定维度上的平均值。
- **参数**:
    - input: 输入张量。
    - dim: 计算平均值的维度。
    - keepdim: 是否保持维度。
- **返回值**: 平均值。

```python
# 示例
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.mean(x, dim=0)
print(y)  # 输出: tensor([2., 3.])
```


## torch.sum
```python
torch.sum(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor
```

- **功能**: 计算张量在指定维度上的元素和。
- **参数**:
    - input: 输入张量。
    - dim: 计算和的维度。
    - keepdim: 是否保持维度。
- **返回值**: 元素和。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = torch.sum(x, dim=0)
print(y)  # 输出: tensor([4, 6])
```


## torch.argmax
```python
torch.argmax(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor
```

- **功能**: 返回张量指定维度上最大值的索引。
- **参数**:
    - input: 输入张量。
    - dim: 指定维度。
    - keepdim: 是否保持维度。
- **返回值**: 最大值的索引。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = torch.argmax(x, dim=0)
print(y)  # 输出: tensor([1, 1])
```


## torch.nn.ReLU
```python
torch.nn.ReLU(inplace: bool = False) -> Tensor
```

- **功能**: ReLU激活层，将负数置为零。
- **参数**:
    - inplace: 是否在原地操作。
- **返回值**: ReLU激活后的张量。

```python
# 示例
relu = torch.nn.ReLU()
x = torch.tensor([-1.0, 0.0, 1.0])
y = relu(x)
print(y)  # 输出: tensor([0., 0., 1.])
```


## torch.bmm
```python
torch.bmm(input: Tensor, mat2: Tensor) -> Tensor
```

- **功能**: 批次矩阵乘法。
- **参数**:
    - input: 第一个批次矩阵。
    - mat2: 第二个批次矩阵。
- **返回值**: 批次矩阵乘法的结果。

```python
# 示例
a = torch.randn(10, 3, 4)
b = torch.randn(10, 4, 5)
c = torch.bmm(a, b)
print(c.shape)  # 输出: torch.Size([10, 3, 5])
```


## torch.clone
```python
torch.clone(input: Tensor) -> Tensor
```

- **功能**: 返回张量的副本，具有相同数据和梯度记录。
- **参数**:
    - input: 输入张量。
- **返回值**: 输入张量的副本。

```python
# 示例
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.clone()
print(y)  # 输出: tensor([1., 2.])
```


## torch.nn.Dropout
```python
torch.nn.Dropout(p: float = 0.5, inplace: bool = False)
```

- **功能**: Dropout层，用于在训练过程中随机将部分单元设为零，防止过拟合。
- **参数**:
    - p: 零化概率。
    - inplace: 是否在原地操作。
- **返回值**: Dropout应用后的张量。

```python
# 示例
dropout = torch.nn.Dropout(p=0.2)
x = torch.tensor([1.0, 2.0, 3.0])
y = dropout(x)
print(y)  # 输出: 随机零化部分元素后的张量
```


## torch.autograd.grad
```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False) -> Tuple[Tensor, ...]
```

- **功能**: 计算输出关于输入的梯度。
- **参数**:
    - outputs: 计算梯度的输出张量。
    - inputs: 梯度的输入张量。
    - retain_graph: 是否保留计算图。
- **返回值**: 梯度张量的元组。

```python
# 示例
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
grad = torch.autograd.grad(y, x)
print(grad)  # 输出: (tensor(2.),)
```


## torch.nn.BatchNorm2d
```python
torch.nn.BatchNorm2d(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True)
```

- **功能**: 2D批标准化层。
- **参数**:
    - num_features: 输入的通道数。
    - eps: 增加数值稳定性的小常数。
    - momentum: 运行平均数的动量。
- **返回值**: 批标准化后的张量。

```python
# 示例
batch_norm = torch.nn.BatchNorm2d(3)
x = torch.randn(2, 3, 4, 4)
y = batch_norm(x)
print(y.shape)  # 输出: torch.Size([2, 3, 4, 4])
```


## torch.nn.MaxPool2d
```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

- **功能**: 2D最大池化层。
- **参数**:
    - kernel_size: 池化窗口大小。
    - stride: 步长。
    - padding: 填充大小。
- **返回值**: 池化后的张量。

```python
# 示例
max_pool = torch.nn.MaxPool2d(2)
x = torch.tensor([[[[1, 2], [3, 4]]]])
y = max_pool(x)
print(y)  # 输出: tensor([[[[4]]]])
```


## torch.flatten
```python
torch.flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor
```

- **功能**: 将张量的指定维度展平。
- **参数**:
    - input: 输入张量。
    - start_dim: 开始展平的维度。
    - end_dim: 结束展平的维度。
- **返回值**: 展平后的张量。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = torch.flatten(x)
print(y)  # 输出: tensor([1, 2, 3, 4])
```


## torch.index_select
```python
torch.index_select(input: Tensor, dim: int, index: Tensor) -> Tensor
```

- **功能**: 在指定维度上选择索引对应的元素。
- **参数**:
    - input: 输入张量。
    - dim: 选择的维度。
    - index: 索引张量。
- **返回值**: 选择的张量。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
indices = torch.tensor([0, 2])
y = torch.index_select(x, 0, indices)
print(y)  # 输出: tensor([[1, 2], [5, 6]])
```


## torch.eq
```python
torch.eq(input: Tensor, other: Tensor) -> Tensor
```

- **功能**: 比较两个张量是否相等。
- **参数**:
    - input: 第一个张量。
    - other: 第二个张量。
- **返回值**: 布尔张量。

```python
# 示例
x = torch.tensor([1, 2, 3])
y = torch.tensor([1, 2, 0])
result = torch.eq(x, y)
print(result)  # 输出: tensor([True, True, False])
```


## torch.nn.Softmax
```python
torch.nn.Softmax(dim: int = None)
```

- **功能**: Softmax激活层，计算概率分布。
- **参数**:
    - dim: 应用softmax的维度。
- **返回值**: 归一化的张量。

```python
# 示例
softmax = torch.nn.Softmax(dim=0)
x = torch.tensor([1.0, 2.0, 3.0])
y = softmax(x)
print(y)  # 输出: tensor([0.0900, 0.2447, 0.6652])
```


## torch.gather
```python
torch.gather(input: Tensor, dim: int, index: Tensor) -> Tensor
```

- **功能**: 在指定维度上根据索引选取值。
- **参数**:
    - input: 输入张量。
    - dim: 选择的维度。
    - index: 索引张量。
- **返回值**: 选择的张量。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
index = torch.tensor([[0, 1], [1, 0]])
y = torch.gather(x, 1, index)
print(y)  # 输出: tensor([[1, 2], [4, 3]])
```


## torch.nn.ConvTranspose2d
```python
torch.nn.ConvTranspose2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, output_padding: Union[int, Tuple[int, int]] = 0, groups: int = 1, bias: bool = True, dilation: Union[int, Tuple[int, int]] = 1)
```

- **功能**: 2D转置卷积层，常用于上采样。
- **参数**:
    - in_channels: 输入通道数。
    - out_channels: 输出通道数。
    - kernel_size: 卷积核的大小。
    - stride: 卷积的步幅。
    - padding: 输入张量的填充大小。
    - output_padding: 输出填充。
    - groups: 卷积的组数。
    - bias: 是否使用偏置。
    - dilation: 卷积的扩展因子。
- **返回值**: 转置卷积操作后的张量。

```python
# 示例
conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=3)
x = torch.randn(1, 1, 3, 3)
y = conv_transpose(x)
print(y.shape)  # 输出: torch.Size([1, 1, 5, 5])
```


## torch.narrow
```python
torch.narrow(input: Tensor, dim: int, start: int, length: int) -> Tensor
```

- **功能**: 在指定维度上对张量进行切片操作。
- **参数**:
    - input: 输入张量。
    - dim: 指定的维度。
    - start: 切片的起始位置。
    - length: 切片的长度。
- **返回值**: 切片后的张量。

```python
# 示例
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.narrow(x, 1, 1, 2)
print(y)  # 输出: tensor([[2, 3], [5, 6]])
```


## torch.index_add
```python
torch.index_add(input: Tensor, dim: int, index: Tensor, tensor: Tensor) -> Tensor
```

- **功能**: 在指定维度上通过索引添加张量的元素。
- **参数**:
    - input: 输入张量。
    - dim: 索引的维度。
    - index: 用于索引的张量。
    - tensor: 要添加的张量。
- **返回值**: 更新后的张量。

```python
# 示例
x = torch.zeros(3, 3)
index = torch.tensor([0, 1])
y = torch.tensor([1.0, 2.0])
x = torch.index_add(x, 0, index, y)
print(x)  # 输出: tensor([[1., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
```


## torch.einsum
```python
torch.einsum(equation: str, *operands) -> Tensor
```

- **功能**: 执行爱因斯坦求和约定，支持高维张量的计算。
- **参数**:
    - equation: 字符串，表示操作的求和约定。
    - operands: 张量的列表或元组。
- **返回值**: 执行计算后的张量。

```python
# 示例
x = torch.randn(2, 3)
y = torch.randn(3, 4)
z = torch.einsum('ij,jk->ik', x, y)
print(z.shape)  # 输出: torch.Size([2, 4])
```


## torch.view
```python
torch.view(*shape) -> Tensor
```

- **功能**: 返回一个具有指定形状的新张量。
- **参数**:
    - shape: 新的形状。
- **返回值**: 具有新形状的张量。

```python
# 示例
x = torch.randn(2, 3)
y = x.view(3, 2)
print(y.shape)  # 输出: torch.Size([3, 2])
```


## torch.norm
```python
torch.norm(input: Tensor, p: float = 2, dim: Optional[int] = None, keepdim: bool = False, out=None) -> Tensor
```

- **功能**: 计算张量的范数（L2范数默认）。
- **参数**:
    - input: 输入张量。
    - p: 范数类型，p=2表示L2范数。
    - dim: 计算范数的维度。
    - keepdim: 是否保持维度。
- **返回值**: 计算得到的范数。

```python
# 示例
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.norm(x)
print(y)  # 输出: tensor(3.7417)
```


## torch.matmul
```python
torch.matmul(input: Tensor, other: Tensor) -> Tensor
```

- **功能**: 执行矩阵乘法。
- **参数**:
    - input: 第一个输入张量。
    - other: 第二个输入张量。
- **返回值**: 矩阵乘法结果。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
z = torch.matmul(x, y)
print(z)  # 输出: tensor([[19, 22], [43, 50]])
```


## torch.mm
```python
torch.mm(input: Tensor, mat2: Tensor) -> Tensor
```

- **功能**: 执行两个二维矩阵的矩阵乘法。
- **参数**:
    - input: 第一个二维张量。
    - mat2: 第二个二维张量。
- **返回值**: 矩阵乘法的结果。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
z = torch.mm(x, y)
print(z)  # 输出: tensor([[19, 22], [43, 50]])
```


## torch.bmm
```python
torch.bmm(batch1: Tensor, batch2: Tensor) -> Tensor
```

- **功能**: 批处理的矩阵乘法，用于一批二维矩阵的批量相乘。
- **参数**:
    - batch1: 第一个三维张量。
    - batch2: 第二个三维张量。
- **返回值**: 批处理后的矩阵乘法结果。

```python
# 示例
x = torch.randn(10, 3, 4)
y = torch.randn(10, 4, 5)
z = torch.bmm(x, y)
print(z.shape)  # 输出: torch.Size([10, 3, 5])
```


## torch.cross
```python
torch.cross(input: Tensor, other: Tensor, dim: int = -1) -> Tensor
```

- **功能**: 计算输入张量的叉积。
- **参数**:
    - input: 第一个输入张量。
    - other: 第二个输入张量。
    - dim: 执行叉积的维度。
- **返回值**: 叉积结果张量。

```python
# 示例
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = torch.cross(x, y)
print(z)  # 输出: tensor([-3.,  6., -3.])
```


## torch.cumsum
```python
torch.cumsum(input: Tensor, dim: int, dtype: Optional[torch.dtype] = None) -> Tensor
```

- **功能**: 返回沿指定维度的累积和。
- **参数**:
    - input: 输入张量。
    - dim: 累加的维度。
    - dtype: 可选，指定输出的数据类型。
- **返回值**: 累积和的张量。

```python
# 示例
x = torch.tensor([1, 2, 3, 4])
y = torch.cumsum(x, dim=0)
print(y)  # 输出: tensor([ 1,  3,  6, 10])
```


## torch.topk
```python
torch.topk(input: Tensor, k: int, dim: int = None, largest: bool = True, sorted: bool = True) -> Tuple[Tensor, Tensor]
```

- **功能**: 返回沿指定维度的前k个元素及其索引。
- **参数**:
    - input: 输入张量。
    - k: 要返回的元素数。
    - dim: 应用操作的维度。
    - largest: 是否返回最大值。
    - sorted: 返回值是否按排序顺序。
- **返回值**: 元素和索引的元组。

```python
# 示例
x = torch.tensor([1, 3, 5, 7, 9])
values, indices = torch.topk(x, 3)
print(values)  # 输出: tensor([9, 7, 5])
print(indices)  # 输出: tensor([4, 3, 2])
```


## torch.kthvalue
```python
torch.kthvalue(input: Tensor, k: int, dim: int = -1, keepdim: bool = False) -> Tuple[Tensor, Tensor]
```

- **功能**: 返回沿指定维度的第k个最小值及其索引。
- **参数**:
    - input: 输入张量。
    - k: 第k个值。
    - dim: 执行操作的维度。
    - keepdim: 保持维度。
- **返回值**: 值和索引的元组。

```python
# 示例
x = torch.tensor([1, 3, 5, 7, 9])
value, index = torch.kthvalue(x, 2)
print(value)  # 输出: tensor(3)
print(index)  # 输出: tensor(1)
```


## torch.sigmoid
```python
torch.sigmoid(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的逐元素Sigmoid函数。
- **参数**:
    - input: 输入张量。
- **返回值**: Sigmoid函数后的张量。

```python
# 示例
x = torch.tensor([1.0, 2.0, -1.0])
y = torch.sigmoid(x)
print(y)  # 输出: tensor([0.7311, 0.8808, 0.2689])
```


## torch.tanh
```python
torch.tanh(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的逐元素Tanh函数。
- **参数**:
    - input: 输入张量。
- **返回值**: Tanh函数后的张量。

```python
# 示例
x = torch.tensor([1.0, 2.0, -1.0])
y = torch.tanh(x)
print(y)  # 输出: tensor([0.7616, 0.9640, -0.7616])
```


## torch.relu
```python
torch.relu(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的逐元素ReLU激活函数。
- **参数**:
    - input: 输入张量。
- **返回值**: ReLU函数后的张量。

```python
# 示例
x = torch.tensor([-1.0, 0.0, 1.0])
y = torch.relu(x)
print(y)  # 输出: tensor([0., 0., 1.])
```


## torch.leaky_relu
```python
torch.leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor
```

- **功能**: 计算输入张量的Leaky ReLU激活函数。
- **参数**:
    - input: 输入张量。
    - negative_slope: 负值部分的斜率。
- **返回值**: Leaky ReLU函数后的张量。

```python
# 示例
x = torch.tensor([-1.0, 0.0, 1.0])
y = torch.leaky_relu(x, negative_slope=0.1)
print(y)  # 输出: tensor([-0.1000,  0.0000,  1.0000])
```


## torch.nn.functional.softmax
```python
torch.nn.functional.softmax(input: Tensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> Tensor
```

- **功能**: 计算输入张量在指定维度上的Softmax。
- **参数**:
    - input: 输入张量。
    - dim: 应用Softmax的维度。
    - dtype: 可选，指定输出的dtype。
- **返回值**: Softmax后的张量。

```python
# 示例
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.nn.functional.softmax(x, dim=0)
print(y)  # 输出: tensor([0.0900, 0.2447, 0.6652])
```


## torch.nn.functional.log_softmax
```python
torch.nn.functional.log_softmax(input: Tensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> Tensor
```

- **功能**: 计算输入张量的对数Softmax。
- **参数**:
    - input: 输入张量。
    - dim: 应用对数Softmax的维度。
    - dtype: 可选，指定输出的dtype。
- **返回值**: 对数Softmax后的张量。

```python
# 示例
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.nn.functional.log_softmax(x, dim=0)
print(y)  # 输出: tensor([-2.4076, -1.4076, -0.4076])
```


## torch.nn.functional.mse_loss
```python
torch.nn.functional.mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor
```

- **功能**: 计算均方误差损失（MSE）。
- **参数**:
    - input: 输入张量。
    - target: 目标张量。
    - reduction: 计算损失的方式（'mean'、'sum'或'none'）。
- **返回值**: 损失值张量。

```python
# 示例
input = torch.tensor([0.0, 1.0, 2.0])
target = torch.tensor([1.0, 2.0, 3.0])
loss = torch.nn.functional.mse_loss(input, target)
print(loss)  # 输出: tensor(1.0000)
```


## torch.nn.functional.cross_entropy
```python
torch.nn.functional.cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, ignore_index: int = -100, reduction: str = 'mean') -> Tensor
```

- **功能**: 计算交叉熵损失，用于多分类问题。
- **参数**:
    - input: 输入张量（预测值）。
    - target: 目标张量（标签）。
    - weight: 各类别的权重。
    - ignore_index: 忽略的标签值。
    - reduction: 计算损失的方式（'mean'、'sum'或'none'）。
- **返回值**: 损失值张量。

```python
# 示例
input = torch.tensor([[1.0, 2.0, 0.1]])
target = torch.tensor([1])
loss = torch.nn.functional.cross_entropy(input, target)
print(loss)  # 输出: 张量，值取决于计算的交叉熵
```


## torch.randn
```python
torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

- **功能**: 生成给定大小的均值为0、标准差为1的正态分布随机数张量。
- **参数**:
    - size: 张量的形状。
    - out: 输出张量。
    - dtype: 数据类型。
    - layout: 张量的布局。
    - device: 所在设备。
    - requires_grad: 是否需要梯度。
- **返回值**: 生成的随机数张量。

```python
# 示例
x = torch.randn(3, 3)
print(x)
```


## torch.bernoulli
```python
torch.bernoulli(input: Tensor, *, generator=None, out=None) -> Tensor
```

- **功能**: 对输入张量的每个元素执行伯努利采样，生成0和1的张量。
- **参数**:
    - input: 输入张量，元素表示成功的概率。
    - generator: 随机数生成器。
    - out: 输出张量。
- **返回值**: 伯努利分布采样生成的0和1张量。

```python
# 示例
prob = torch.tensor([0.2, 0.8, 0.5])
x = torch.bernoulli(prob)
print(x)  # 输出: 张量，如 tensor([0., 1., 1.])，结果因概率变化
```


## torch.distributions.Normal
```python
torch.distributions.Normal(loc: float, scale: float)
```

- **功能**: 正态分布对象，可用于采样和计算概率密度函数等。
- **参数**:
    - loc: 均值。
    - scale: 标准差。
- **返回值**: 正态分布对象。

```python
# 示例
normal_dist = torch.distributions.Normal(0, 1)
sample = normal_dist.sample((3,))
print(sample)
```


## torch.logsumexp
```python
torch.logsumexp(input: Tensor, dim: int, keepdim: bool = False) -> Tensor
```

- **功能**: 计算输入张量沿指定维度的对数求和指数值。
- **参数**:
    - input: 输入张量。
    - dim: 计算的维度。
    - keepdim: 是否保持维度。
- **返回值**: 对数求和指数后的张量。

```python
# 示例
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.logsumexp(x, dim=0)
print(y)  # 输出: tensor([3.3133, 4.3133])
```


## torch.unbind
```python
torch.unbind(input: Tensor, dim: int = 0) -> Tuple[Tensor, ...]
```

- **功能**: 沿指定维度解绑定张量，返回一个张量元组。
- **参数**:
    - input: 输入张量。
    - dim: 解绑定的维度。
- **返回值**: 解绑定的张量元组。

```python
# 示例
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.unbind(x, dim=0)
print(y)  # 输出: (tensor([1, 2, 3]), tensor([4, 5, 6]))
```


## torch.meshgrid
```python
torch.meshgrid(*tensors, indexing: str = 'ij') -> Tuple[Tensor, ...]
```

- **功能**: 创建N维网格。
- **参数**:
    - tensors: 张量序列，用于生成网格。
    - indexing: 索引方式，默认为‘ij’。
- **返回值**: 网格张量的元组。

```python
# 示例
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5])
grid_x, grid_y = torch.meshgrid(x, y)
print(grid_x)  # 输出: tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
print(grid_y)  # 输出: tensor([[4, 5], [4, 5], [4, 5]])
```


## torch.squeeze
```python
torch.squeeze(input: Tensor, dim: Optional[int] = None) -> Tensor
```

- **功能**: 去除指定维度或所有大小为1的维度。
- **参数**:
    - input: 输入张量。
    - dim: 选择去除的维度，如果未指定则去除所有大小为1的维度。
- **返回值**: 去除指定维度后的张量。

```python
# 示例
x = torch.tensor([[[1]], [[2]], [[3]]])
y = torch.squeeze(x)
print(y)  # 输出: tensor([1, 2, 3])
```


## torch.unsqueeze
```python
torch.unsqueeze(input: Tensor, dim: int) -> Tensor
```

- **功能**: 在指定维度插入大小为1的维度。
- **参数**:
    - input: 输入张量。
    - dim: 插入的维度。
- **返回值**: 插入新的维度后的张量。

```python
# 示例
x = torch.tensor([1, 2, 3])
y = torch.unsqueeze(x, 0)
print(y)  # 输出: tensor([[1, 2, 3]])
```


## torch.nn.functional.pad
```python
torch.nn.functional.pad(input: Tensor, pad: Tuple[int, ...], mode: str = 'constant', value: float = 0.0) -> Tensor
```

- **功能**: 为输入张量填充边界。
- **参数**:
    - input: 输入张量。
    - pad: 填充的大小，格式为 (左, 右, 上, 下)。
    - mode: 填充模式 ('constant', 'reflect', 'replicate')。
    - value: 常量填充值，仅在 mode 为 'constant' 时有效。
- **返回值**: 填充后的张量。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
print(y)  # 输出: tensor([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
```


## torch.tril
```python
torch.tril(input: Tensor, diagonal: int = 0) -> Tensor
```

- **功能**: 返回输入张量的下三角部分，其他部分置零。
- **参数**:
    - input: 输入张量。
    - diagonal: 控制对角线位置。
- **返回值**: 下三角张量。

```python
# 示例
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = torch.tril(x)
print(y)  # 输出: tensor([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
```


## torch.triu
```python
torch.triu(input: Tensor, diagonal: int = 0) -> Tensor
```

- **功能**: 返回输入张量的上三角部分，其他部分置零。
- **参数**:
    - input: 输入张量。
    - diagonal: 控制对角线位置。
- **返回值**: 上三角张量。

```python
# 示例
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = torch.triu(x)
print(y)  # 输出: tensor([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
```


## torch.scatter
```python
torch.scatter(input: Tensor, dim: int, index: Tensor, src: Union[Tensor, float], reduce: Optional[str] = None) -> Tensor
```

- **功能**: 根据index中的索引在指定维度上将src值散布到输入张量中。
- **参数**:
    - input: 输入张量。
    - dim: 指定维度。
    - index: 索引张量。
    - src: 要散布的值或张量。
    - reduce: 可选，指定散布的方式。
- **返回值**: 修改后的张量。

```python
# 示例
x = torch.zeros(3, 5)
index = torch.tensor([[0, 1, 2, 0, 1]])
y = x.scatter(0, index, 1.0)
print(y)
```


## torch.repeat_interleave
```python
torch.repeat_interleave(input: Tensor, repeats: Union[int, Tensor], dim: Optional[int] = None, output_size: Optional[int] = None) -> Tensor
```

- **功能**: 沿指定维度重复输入张量的元素。
- **参数**:
    - input: 输入张量。
    - repeats: 每个元素的重复次数。
    - dim: 指定维度。
    - output_size: 可选，指定输出张量的大小。
- **返回值**: 重复后的张量。

```python
# 示例
x = torch.tensor([1, 2, 3])
y = torch.repeat_interleave(x, repeats=2)
print(y)  # 输出: tensor([1, 1, 2, 2, 3, 3])
```


## torch.rot90
```python
torch.rot90(input: Tensor, k: int, dims: Tuple[int, int]) -> Tensor
```

- **功能**: 旋转输入张量90度。
- **参数**:
    - input: 输入张量。
    - k: 旋转次数（每次90度）。
    - dims: 指定旋转的维度。
- **返回值**: 旋转后的张量。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = torch.rot90(x, 1, [0, 1])
print(y)  # 输出: tensor([[2, 4], [1, 3]])
```


## torch.chunk
```python
torch.chunk(input: Tensor, chunks: int, dim: int = 0) -> List[Tensor]
```

- **功能**: 将输入张量分割成指定数量的块。
- **参数**:
    - input: 输入张量。
    - chunks: 分割的块数。
    - dim: 指定分割的维度。
- **返回值**: 分割后的张量列表。

```python
# 示例
x = torch.tensor([1, 2, 3, 4, 5, 6])
y = torch.chunk(x, 3)
print(y)  # 输出: (tensor([1, 2]), tensor([3, 4]), tensor([5, 6]))
```


## torch.tensor_split
```python
torch.tensor_split(input: Tensor, indices_or_sections: Union[int, List[int]], dim: int = 0) -> List[Tensor]
```

- **功能**: 沿指定维度分割张量。
- **参数**:
    - input: 输入张量。
    - indices_or_sections: 分割点列表或段数。
    - dim: 指定分割的维度。
- **返回值**: 分割后的张量列表。

```python
# 示例
x = torch.tensor([1, 2, 3, 4, 5, 6])
y = torch.tensor_split(x, [2, 4])
print(y)  # 输出: (tensor([1, 2]), tensor([3, 4]), tensor([5, 6]))
```


## torch.gather
```python
torch.gather(input: Tensor, dim: int, index: Tensor, *, sparse_grad: bool = False, out=None) -> Tensor
```

- **功能**: 从输入张量中收集指定位置的元素。
- **参数**:
    - input: 输入张量。
    - dim: 指定的维度。
    - index: 索引张量，包含要收集的元素索引。
    - sparse_grad: 是否允许稀疏梯度。
    - out: 可选，输出张量。
- **返回值**: 收集后的张量。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
index = torch.tensor([[0, 0], [1, 0]])
y = torch.gather(x, dim=1, index=index)
print(y)  # 输出: tensor([[1, 1], [4, 3]])
```


## torch.clamp
```python
torch.clamp(input: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor
```

- **功能**: 将张量的值裁剪到指定的最小和最大范围。
- **参数**:
    - input: 输入张量。
    - min: 最小值。
    - max: 最大值。
- **返回值**: 裁剪后的张量。

```python
# 示例
x = torch.tensor([0.5, 1.5, 2.5])
y = torch.clamp(x, min=1.0, max=2.0)
print(y)  # 输出: tensor([1.0000, 1.5000, 2.0000])
```


## torch.cumprod
```python
torch.cumprod(input: Tensor, dim: int, dtype: Optional[torch.dtype] = None) -> Tensor
```

- **功能**: 计算指定维度的累积乘积。
- **参数**:
    - input: 输入张量。
    - dim: 指定的维度。
    - dtype: 数据类型。
- **返回值**: 累积乘积后的张量。

```python
# 示例
x = torch.tensor([1, 2, 3])
y = torch.cumprod(x, dim=0)
print(y)  # 输出: tensor([1, 2, 6])
```


## torch.tanh
```python
torch.tanh(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的元素级别的Tanh函数。
- **参数**:
    - input: 输入张量。
- **返回值**: 经过Tanh变换的张量。

```python
# 示例
x = torch.tensor([1.0, 2.0, -1.0])
y = torch.tanh(x)
print(y)  # 输出: tensor([ 0.7616,  0.9640, -0.7616])
```


## torch.erf
```python
torch.erf(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的误差函数（erf）。
- **参数**:
    - input: 输入张量。
- **返回值**: 经过erf变换的张量。

```python
# 示例
x = torch.tensor([0.0, 0.5, -0.5])
y = torch.erf(x)
print(y)  # 输出: tensor([ 0.0000,  0.5205, -0.5205])
```


## torch.sort
```python
torch.sort(input: Tensor, dim: int = -1, descending: bool = False, out=None) -> Tuple[Tensor, Tensor]
```

- **功能**: 对输入张量指定维度进行排序。
- **参数**:
    - input: 输入张量。
    - dim: 指定的维度。
    - descending: 是否降序排列。
    - out: 输出张量。
- **返回值**: 排序后的张量及其索引。

```python
# 示例
x = torch.tensor([3, 1, 2])
sorted_values, indices = torch.sort(x)
print(sorted_values)  # 输出: tensor([1, 2, 3])
print(indices)  # 输出: tensor([1, 2, 0])
```

## torch.Tensor.clone
```python
torch.Tensor.clone(input: Tensor) -> Tensor
```

- **功能**: 克隆一个张量，返回与输入张量相同但内存独立的新张量。
- **参数**:
    - input: 输入张量。
- **返回值**: 克隆的张量。

```python
# 示例
x = torch.tensor([1, 2, 3])
y = x.clone()
print(y)  # 输出: tensor([1, 2, 3])
```


## torch.full
```python
torch.full(size: torch.Size, fill_value: float, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, requires_grad: bool = False) -> Tensor
```

- **功能**: 创建一个填充了指定值的张量。
- **参数**:
    - size: 输出张量的形状。
    - fill_value: 填充的值。
    - dtype: 数据类型。
    - device: 设备。
    - requires_grad: 是否需要计算梯度。
- **返回值**: 填充后的张量。

```python
# 示例
x = torch.full((2, 3), fill_value=7)
print(x)  # 输出: tensor([[7, 7, 7], [7, 7, 7]])
```


## torch.masked_select
```python
torch.masked_select(input: Tensor, mask: Tensor) -> Tensor
```

- **功能**: 从输入张量中选择满足 mask 条件的元素。
- **参数**:
    - input: 输入张量。
    - mask: 布尔类型的张量，指示要选择的元素。
- **返回值**: 满足条件的张量元素。

```python
# 示例
x = torch.tensor([1, 2, 3, 4])
mask = torch.tensor([True, False, True, False])
y = torch.masked_select(x, mask)
print(y)  # 输出: tensor([1, 3])
```


## torch.nonzero
```python
torch.nonzero(input: Tensor, as_tuple: bool = False) -> Tensor
```

- **功能**: 返回输入张量中非零元素的索引。
- **参数**:
    - input: 输入张量。
    - as_tuple: 如果为True，返回元组形式的索引。
- **返回值**: 非零元素的索引。

```python
# 示例
x = torch.tensor([1, 0, 3, 0])
y = torch.nonzero(x)
print(y)  # 输出: tensor([[0], [2]])
```


## torch.unique
```python
torch.unique(input: Tensor, sorted: bool = False, return_inverse: bool = False, return_counts: bool = False, dim: Optional[int] = None) -> Tensor
```

- **功能**: 返回张量中唯一元素。
- **参数**:
    - input: 输入张量。
    - sorted: 是否对结果进行排序。
    - return_inverse: 是否返回元素的反向索引。
    - return_counts: 是否返回元素的出现次数。
    - dim: 可选，指定操作的维度。
- **返回值**: 唯一元素的张量。

```python
# 示例
x = torch.tensor([1, 2, 2, 3])
y = torch.unique(x)
print(y)  # 输出: tensor([1, 2, 3])
```

## torch.fft.fft
```python
torch.fft.fft(input: Tensor, signal_ndim: int, normalized: bool = False) -> Tensor
```

- **功能**: 计算输入张量的快速傅里叶变换（FFT）。
- **参数**:
    - input: 输入张量。
    - signal_ndim: 信号的维度。
    - normalized: 是否归一化。
- **返回值**: 快速傅里叶变换后的张量。


```python
# 示例
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.fft.fft(x, signal_ndim=1)
print(y)  # 输出: tensor([ 6.0000+0.0000j, -0.5000+0.8660j, -0.5000-0.8660j])
```
## torch.fft.rfft
```python
torch.fft.rfft(input: Tensor, signal_ndim: int, normalized: bool = False) -> Tensor
```

- **功能**: 计算输入张量的实数快速傅里叶变换（RFFT），返回频域表示。
- **参数**:
    - input: 输入张量，必须是实数张量。
    - signal_ndim: 信号的维度。
    - normalized: 是否对结果进行归一化。
- **返回值**: 频域表示的复数张量。


```python
# 示例
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.fft.rfft(x, signal_ndim=1)
print(y)  # 输出: tensor([6.0000+0.0000j, -0.5000+0.8660j])
```

## torch.fft.ifft
```python
torch.fft.ifft(input: Tensor, signal_ndim: int, normalized: bool = False) -> Tensor
```

- **功能**: 计算输入张量的逆快速傅里叶变换（IFFT）。
- **参数**:
    - input: 输入张量。
    - signal_ndim: 信号的维度。
    - normalized: 是否归一化。
- **返回值**: 逆傅里叶变换后的张量。

```python
# 示例
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.fft.ifft(x, signal_ndim=1)
print(y)  # 输出: tensor([ 2.0000+0.0000j, -0.5000-0.8660j, -0.5000+0.8660j])
```


## torch.fft.irfft
```python
torch.fft.irfft(input: Tensor, signal_ndim: int, normalized: bool = False) -> Tensor
```

- **功能**: 计算输入张量的逆实数快速傅里叶变换（IRFFT），返回时域信号。
- **参数**:
    - input: 输入张量，必须是复数张量，且代表的是实数信号的频域表示。
    - signal_ndim: 信号的维度。
    - normalized: 是否对结果进行归一化。
- **返回值**: 时域信号的实数张量。

```python
# 示例
x = torch.tensor([6.0 + 0.0j, -0.5 + 0.8660j])
y = torch.fft.irfft(x, signal_ndim=1)
print(y)  # 输出: tensor([1.0000, 2.0000, 3.0000])
```


## torch.atan
```python
torch.atan(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的反正切（arctan）值。
- **参数**:
    - input: 输入张量。
- **返回值**: 反正切后的张量。

```python
# 示例
x = torch.tensor([0.0, 1.0, -1.0])
y = torch.atan(x)
print(y)  # 输出: tensor([ 0.0000,  0.7854, -0.7854])
```


## torch.cos
```python
torch.cos(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的余弦值。
- **参数**:
    - input: 输入张量。
- **返回值**: 余弦后的张量。

```python
# 示例
x = torch.tensor([0.0, 3.1416, 1.5708])
y = torch.cos(x)
print(y)  # 输出: tensor([1.0000, -1.0000,  0.0000])
```


## torch.sin
```python
torch.sin(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的正弦值。
- **参数**:
    - input: 输入张量。
- **返回值**: 正弦后的张量。

```python
# 示例
x = torch.tensor([0.0, 3.1416, 1.5708])
y = torch.sin(x)
print(y)  # 输出: tensor([ 0.0000,  0.0000,  1.0000])
```


## einops.rearrange
```python
einops.rearrange(tensor, pattern: str, **axes_lengths) -> Tensor
```

- **功能**: 按照给定的模式重新排列张量的维度。
- **参数**:
    - tensor: 输入张量。
    - pattern: 字符串模式，指定如何重新排列维度。
    - **axes_lengths: 可选参数，根据模式来调整维度的大小。
- **返回值**: 重排后的张量。

```python
# 示例
import torch
from einops import rearrange

x = torch.randn(2, 3, 4)
y = rearrange(x, 'b c h -> (b h) c')
print(y.shape)  # 输出: torch.Size([6, 4])
```


## einops.reduce
```python
einops.reduce(tensor, pattern: str, reduction: str, **axes_lengths) -> Tensor
```

- **功能**: 按照给定的模式对张量进行降维操作。
- **参数**:
    - tensor: 输入张量。
    - pattern: 字符串模式，指定如何降维。
    - reduction: 降维方法，可以是 "mean", "sum", "min", "max" 等。
    - **axes_lengths: 可选参数，指定维度的大小。
- **返回值**: 降维后的张量。

```python
# 示例
import torch
from einops import reduce

x = torch.randn(2, 3, 4)
y = reduce(x, 'b c h -> b c', 'mean')
print(y.shape)  # 输出: torch.Size([2, 3])
```


## einops.repeat
```python
einops.repeat(tensor, pattern: str, **axes_lengths) -> Tensor
```

- **功能**: 按照指定模式重复张量的某些维度。
- **参数**:
    - tensor: 输入张量。
    - pattern: 字符串模式，指定如何重复维度。
    - **axes_lengths: 可选参数，指定重复的次数。
- **返回值**: 重复后的张量。

```python
# 示例
import torch
from einops import repeat

x = torch.randn(2, 3, 4)
y = repeat(x, 'b c h -> b c h 2', 2=3)
print(y.shape)  # 输出: torch.Size([2, 3, 4, 2])
```


## einx.batch
```python
einx.batch(tensor: Tensor, batch_dims: int) -> Tensor
```

- **功能**: 处理批次维度的函数，按批次合并或拆分张量。
- **参数**:
    - tensor: 输入张量。
    - batch_dims: 批次维度的数量。
- **返回值**: 按批次维度处理后的张量。

```python
# 示例
import torch
import einx

x = torch.randn(4, 3)
y = einx.batch(x, 2)
print(y.shape)  # 输出: torch.Size([2, 6])
```


## einx.split
```python
einx.split(tensor: Tensor, axis: int, num_or_size_splits: int) -> List[Tensor]
```

- **功能**: 按照指定的维度和分割数量或分割尺寸拆分张量。
- **参数**:
    - tensor: 输入张量。
    - axis: 要拆分的维度。
    - num_or_size_splits: 分割的数量或者每个分割的大小。
- **返回值**: 拆分后的张量列表。

```python
# 示例
import torch
import einx

x = torch.randn(6, 3)
y = einx.split(x, 0, 3)
print([yi.shape for yi in y])  # 输出: [torch.Size([2, 3]), torch.Size([2, 3]), torch.Size([2, 3])]
```


## torch.conj
```python
torch.conj(input: Tensor) -> Tensor
```

- **功能**: 计算输入张量的共轭复数。
- **参数**:
    - input: 输入张量，可以是复数类型的张量。
- **返回值**: 输入张量的共轭复数。

```python
# 示例
x = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])
y = torch.conj(x)
print(y)  # 输出: tensor([1.0 - 2.0j, 3.0 - 4.0j])
```


## torch.tensor.repeat
```python
torch.tensor.repeat(*sizes: int) -> Tensor
```

- **功能**: 根据给定的大小，重复张量的元素。
- **参数**:
    - sizes: 每个维度上要重复的次数，可以是多个整数来表示每个维度的重复次数。
- **返回值**: 重复元素后的张量。

```python
# 示例
x = torch.tensor([[1, 2], [3, 4]])
y = x.repeat(2, 3)
print(y)
# 输出:
# tensor([[1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4],
#         [1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4]])
```


## torch.nn.Dropout
```python
torch.nn.Dropout(p: float = 0.5, inplace: bool = False)
```

- **功能**: 随机丢弃输入张量中的一些元素，以防止过拟合。
- **参数**:
    - p: 丢弃的概率，默认为 0.5。
    - inplace: 是否进行原地操作，默认为 False。
- **返回值**: 丢弃后的张量。

```python
# 示例
dropout = nn.Dropout(p=0.3)
x = torch.randn(2, 5)
y = dropout(x)
print(y)
```


## torch.nn.BatchNorm2d
```python
torch.nn.BatchNorm2d(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True)
```

- **功能**: 批归一化层，用于加速训练并提高网络稳定性。
- **参数**:
    - num_features: 输入通道的数量。
    - eps: 数值稳定性常数，默认 1e-5。
    - momentum: 动量参数，默认 0.1。
    - affine: 是否学习仿射参数（gamma 和 beta）。
    - track_running_stats: 是否跟踪均值和方差的历史。
- **返回值**: 批归一化层对象。

```python
# 示例
batch_norm = nn.BatchNorm2d(16)
x = torch.randn(2, 16, 32, 32)  # 输入张量
y = batch_norm(x)
print(y.shape)  # 输出: torch.Size([2, 16, 32, 32])
```


## torch.nn.LSTM
```python
torch.nn.LSTM(input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0, bidirectional: bool = False)
```

- **功能**: 定义一个 LSTM（长短期记忆）层。
- **参数**:
    - input_size: 输入特征的维度。
    - hidden_size: 隐藏层的维度。
    - num_layers: LSTM 层数，默认为 1。
    - bias: 是否使用偏置项，默认为 True。
    - batch_first: 是否将 batch 维度放在第一个，默认为 False。
    - dropout: 如果 num_layers > 1，则使用丢弃概率。
    - bidirectional: 是否使用双向 LSTM。
- **返回值**: LSTM 层对象。

```python
# 示例
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(5, 3, 10)  # 5 个样本，3 个时间步长，10 个特征
output, (hn, cn) = lstm(x)
print(output.shape)  # 输出: torch.Size([5, 3, 20])
```


## torch.nn.Conv1d
```python
torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')
```

- **功能**: 定义一个一维卷积层，常用于处理时序数据。
- **参数**:
    - in_channels: 输入通道数。
    - out_channels: 输出通道数。
    - kernel_size: 卷积核的大小。
    - stride: 卷积步幅，默认为 1。
    - padding: 填充的大小，默认为 0。
    - dilation: 卷积核的扩张，默认为 1。
    - groups: 分组卷积，默认为 1。
    - bias: 是否使用偏置项，默认为 True。
    - padding_mode: 填充模式，默认为 'zeros'。
- **返回值**: 卷积层对象。

```python
# 示例
conv1d = nn.Conv1d(10, 20, kernel_size=3)
x = torch.randn(2, 10, 5)  # 输入 2 个样本，10 个通道，5 个时间步长
y = conv1d(x)
print(y.shape)  # 输出: torch.Size([2, 20, 3])
```


## torch.nn.Embedding
```python
torch.nn.Embedding(num_embeddings: int, embedding_dim: int, padding_idx: int = -1, max_norm: float = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False)
```

- **功能**: 用于将离散的词索引映射到连续的嵌入向量空间，通常用于词嵌入（Word Embedding）。
- **参数**:
    - num_embeddings: 词汇表大小（词的总数）。
    - embedding_dim: 每个词的嵌入维度。
    - padding_idx: 用于填充的索引，默认是 -1。
    - max_norm: 嵌入向量的最大范数。
    - norm_type: 范数的类型，默认为 2。
    - scale_grad_by_freq: 是否根据词频缩放梯度。
    - sparse: 是否使用稀疏更新。
- **返回值**: 嵌入层对象。

```python
# 示例
embedding = nn.Embedding(10, 3)  # 词汇表大小为 10，每个词的嵌入维度为 3
x = torch.tensor([1, 2, 4, 5])
y = embedding(x)
print(y)
```


## torch.nn.Transformer
```python
torch.nn.Transformer(d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu', custom_encoder: Optional[nn.Module] = None, custom_decoder: Optional[nn.Module] = None)
```

- **功能**: 实现完整的 Transformer 模型，包括编码器和解码器。
- **参数**:
    - d_model: 输入和输出的特征维度。
    - nhead: 多头注意力机制中的头数。
    - num_encoder_layers: 编码器层的数量。
    - num_decoder_layers: 解码器层的数量。
    - dim_feedforward: 前馈神经网络的维度。
    - dropout: Dropout 概率，默认为 0.1。
    - activation: 激活函数，默认为 ReLU。
    - custom_encoder: 可选自定义编码器。
    - custom_decoder: 可选自定义解码器。
- **返回值**: Transformer 模型对象。

```python
# 示例
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
src = torch.randn(10, 32, 512)  # 10 时间步，32 样本，512 特征
tgt = torch.randn(20, 32, 512)  # 20 时间步，32 样本，512 特征
output = transformer(src, tgt)
print(output.shape)  # 输出: torch.Size([20, 32, 512])
```


## torch.nn.Lambda
```python
torch.nn.Lambda(lambda: nn.Module)
```

- **功能**: 定义一个自定义的层，使用用户提供的 lambda 函数。
- **参数**:
    - lambda: 用户提供的 lambda 函数，用于定义层的前向传播。
- **返回值**: 自定义层对象。

```python
# 示例
lambda_layer = nn.Lambda(lambda x: x * 2)
x = torch.tensor([1, 2, 3])
y = lambda_layer(x)
print(y)  # 输出: tensor([2, 4, 6])
```


## torch.nn.Sigmoid
```python
torch.nn.Sigmoid()
```

- **功能**: 计算 Sigmoid 激活函数。
- **参数**: 无
- **返回值**: Sigmoid 激活后的张量。

```python
# 示例
sigmoid = nn.Sigmoid()
x = torch.tensor([-1.0, 0.0, 1.0])
y = sigmoid(x)
print(y)  # 输出: tensor([0.2689, 0.5000, 0.7311])
```


```


```


```
```
```
```
```
```
```
```
```
```
```
```
```
```