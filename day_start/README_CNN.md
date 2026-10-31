# CNN 第一课：跟着张量走一遍

目标：能说清一张图片怎样变成分类分数，能在 GPU 上完成一次参数更新。

## 先运行

在项目根目录 `D:\PythonProject1\PythonProject1` 的 PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn01.py
```

只需要当前已安装的 PyTorch。脚本包含三个小实验：手算卷积、逐层查看形状、查看参数更新。无 CUDA 时自动使用 CPU。

## 1. 图片为什么能交给网络

灰度图可以表示成一个二维数值矩阵，每个数字代表一个像素的亮度。
RGB 彩色图通常有三个通道；本课使用一个通道的灰度图。

PyTorch 中，一批图片常用 `[N, C, H, W]` 排列：

| 字母 | 含义 | 本课输入 |
|---|---|---|
| N | 一次处理的图片数量，即 batch size | 4 |
| C | 每张图片的通道数 | 1 |
| H | 高度 | 28 |
| W | 宽度 | 28 |

所以 `torch.randn(4, 1, 28, 28)` 创建了形状正确的随机张量。它只是模拟输入，并非真实手写数字图片。

## 2. 卷积在做什么

先看 `show_one_convolution()`：一个 3×3 的小窗口在 5×5 矩阵上滑动。
每到一个位置，将窗口里的数字与卷积核的对应元素相乘，再求和。本例偏置为 0。
步长为 1、不补边，所以横向、纵向各有 3 个位置，得到 3×3 输出。

这里固定了卷积核，方便手算。训练网络时，卷积核通常从随机权重开始，逐渐根据数据更新；不同的核可以学习不同的局部模式。
同一个核在不同位置复用，这是卷积的权重共享。

技术上，PyTorch 的 `Conv2d` 使用互相关，即不翻转卷积核；深度学习中通常仍称其为卷积。

第一层代码：

```python
nn.Conv2d(1, 8, kernel_size=3, padding=1)
```

- `1`：输入有 1 个通道。
- `8`：输出有 8 个通道，即 8 张特征图，不是 8 张输入图片。
- `kernel_size=3`：核的高、宽为 3。
- `padding=1`：四周各补一圈零；配合默认步长 1，保持本层的高、宽不变。

在默认 `groups=1` 时，每个输出通道都会汇总所有输入通道的卷积结果。因此第二层 `Conv2d(8, 16, 3)` 的权重形状是 `[16, 8, 3, 3]`。

可选的尺寸计算：当 `dilation=1` 时，单个方向的输出尺寸是
`floor((输入尺寸 + 2*padding - kernel_size) / stride) + 1`。
本层为 `(28 + 2 - 3) / 1 + 1 = 28`。

## 3. 每一层做什么

| 步骤 | 输出形状 | 作用 |
|---|---|---|
| 输入 | `[4, 1, 28, 28]` | 4 张单通道模拟图片 |
| Conv2d → ReLU | `[4, 8, 28, 28]` | 计算 8 张特征图，再引入非线性 |
| MaxPool2d | `[4, 8, 14, 14]` | 每个 2×2 区域取最大值，高宽各减半 |
| Conv2d → ReLU | `[4, 16, 14, 14]` | 组合前一层特征，产生 16 个通道 |
| MaxPool2d | `[4, 16, 7, 7]` | 再缩小空间尺寸 |
| Flatten | `[4, 784]` | 每张图的 16×7×7 个数展开成向量 |
| Linear | `[4, 10]` | 给出每个样本的 10 个分类分数 |

`ReLU(x)=max(0,x)`，不会改变形状。它引入非线性，使网络能表达更复杂的关系。
池化也不会改变本例的通道数。它减少空间位置数量，也会丢失一些细节。
`Flatten(start_dim=1)` 保留第 0 维的批量维度，只展开每个样本内部的维度。

## 4. Python 类和训练代码怎么读

`SmallCNN` 继承 `nn.Module`，让 PyTorch 管理网络层与参数。
`__init__` 定义有哪些层，`forward` 定义输入经过这些层的顺序。
调用 `model(images)` 会执行模型的前向计算。

训练的基本步骤是：

```python
optimizer.zero_grad()          # 清除旧梯度
logits = model(images)         # 前向计算，得到原始分类分数
loss = loss_fn(logits, labels) # 衡量这些分数与标签的差异
loss.backward()               # 根据损失计算参数的梯度
optimizer.step()              # 根据梯度更新参数
```

`CrossEntropyLoss` 在本例中接收 `[N, 10]` 的原始分数，以及 `[N]` 的整数标签（0～9、torch.long）。不需要先对分数做 softmax。
`backward()` 计算梯度，`step()` 才更新权重。梯度会累积，所以每个普通训练步骤开始前清零。
模型、图片、标签都要放到同一个设备；代码中的 `.to(device)` 负责移动它们。

本课只有一次随机数据上的更新，没有训练集/测试集，也不报告识别准确率。下一课再换成 MNIST 手写数字，学习 DataLoader、完整训练循环和独立测试集评估。

## 5. 动手练习

每次只改一个地方，先预测输出形状，再运行确认：

1. 把输入和标签中的批量数 `4` 都改成 `8`。哪些维度变化，哪些不变？
2. 把第一层的输出通道 `8` 改成 `12`。第二个卷积层的输入通道应该怎样改？
3. 将两个卷积层都改为 `kernel_size=5, padding=2`。空间尺寸是否改变？

<details>
<summary>做完后查看提示</summary>

1. 各层的第 0 维变成 8；每张图片内部的通道、高和宽不变。
2. 第二层同步改为 `nn.Conv2d(12, 16, kernel_size=3, padding=1)`；后续尺寸不变。
3. 默认步长为 1，补边抵消了核大小带来的缩小；各层形状保持不变，可学习参数增多。

</details>

## 后续学习顺序

1. 本课：图片张量、卷积、ReLU、池化、分类分数。
2. [第二节：用 MNIST 完成训练与评估](README_CNN02.md)，区分训练集和测试集；代码为 `startcnn02.py`。
3. [第三课：观察学习率、batch size 和 epoch 对训练的影响](README_CNN03.md)，代码为 `startcnn03.py`。
4. [第四课：理解过拟合、验证集、数据增强和正则化](README_CNN04.md)，代码为 `startcnn04.py`。
5. 保存/加载模型，并对新图片进行推理。

## 对照资料

- [PyTorch 神经网络入门](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)：模型定义与训练步骤。
- [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)：通道、核、步长、补边与输出尺寸。
- [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)：池化窗口和步长。
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：分类分数与标签的格式。
