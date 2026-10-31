# CNN 第二节：让网络学习真实的手写数字

本节目标：沿用第一节的 SmallCNN，完成真实数据加载、分批训练和独立测试集评估。

## 运行与阅读顺序

在项目根目录 `D:\PythonProject1\PythonProject1` 的 PowerShell 里运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn02.py
```

也可以在 PyCharm 中直接运行 `startcnn02.py`，解释器选择项目的 `.venv`。
首次运行会下载 MNIST，保存到项目的 `data/MNIST` 目录；已有数据会被复用。
本课需要已安装的 `torch` 和 `torchvision`，不需要额外绘图库。

建议先读 `main()` 看整体流程，再按 `create_loaders()`、`train_one_epoch()`、`evaluate()` 的顺序阅读。
网络由 `from startcnn01 import SmallCNN` 导入。第一节的 main 受到 `if __name__ == "__main__"` 保护，因此导入时不会执行第一节的演示。

## 1. 从随机数据换成有答案的图片

MNIST 包含 0～9 的手写数字，每张图片为 28×28 灰度图。
它的标准训练集有 60000 张，测试集有 10000 张。
训练时用图片和正确标签调整参数；训练结束后，用测试集检查模型能否识别未参与训练的图片。

```python
train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
```

| 参数 | 含义 |
|---|---|
| root | 本地保存数据的位置 |
| train=True | 读取官方训练集；False 读取官方测试集 |
| download=True | 数据缺失时自动下载；已有数据时复用 |
| transform | 每次取出图片时执行的转换 |

`ToTensor()` 对本课 MNIST 图片的作用：将灰度图转成形状 `[1, 28, 28]` 的浮点张量，并把像素值从 0～255 缩放到 0～1。
这一步不改变图片的正确标签，也不是减均值、除标准差的标准化。

## 2. Dataset 和 DataLoader

`Dataset` 提供单个样本。例如 `image, label = train_dataset[0]` 会得到一张图片和它的类别。
`DataLoader` 负责把多个样本拼成一批：

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
```

| 参数 | 含义 |
|---|---|
| train_dataset | 从哪个数据集取样 |
| batch_size=64 | 一次返回最多 64 张图片及其标签 |
| shuffle=True | 每一轮重新打乱样本顺序，图片和标签始终配对 |
| num_workers=0 | 在当前进程加载数据，便于本课在 Windows 上运行和调试 |

一个普通批次的形状：

```text
images: [64, 1, 28, 28]，通常为 torch.float32
labels: [64]，torch.int64（即 torch.long），数值为 0～9
logits: [64, 10]，模型给每张图片的 10 个分类分数
```

最后一批可能不足 64 张。本课保留它：60000 = 937×64 + 32，因此每轮有 938 个 batch，最后一批是 32 张。
代码按实际样本数累加损失，最后除以总样本数，避免把 32 张和 64 张的批次当成同样大小。

## 3. batch、step、epoch

| 名词 | 本课中的含义 |
|---|---|
| batch | 一批图片与标签 |
| step | 本课每个 batch 执行一次 optimizer.step，更新一次参数 |
| epoch | 把完整训练集学习一遍 |

默认设置下：一轮 epoch 包含 938 次参数更新；3 轮共 2814 次更新。
训练中重复看到同一张图片是正常的；每轮 shuffle 会改变顺序，但不会改变图片对应的标签。

## 4. 训练循环

```python
model.train()
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
```

这就是把第一节的一次更新，扩展成“对每个 batch 都更新一次”。
`model.train()` 只切换层的工作模式，实际训练依赖后面的计算和参数更新。
当前网络没有 Dropout 或 BatchNorm，因此 train/eval 切换不会改变这些层的具体运算，但保留规范写法方便后续扩展。
`CrossEntropyLoss` 直接接收原始分数，计算损失前不调用 softmax 或 argmax。

## 5. 怎样计算准确率

```python
predictions = logits.argmax(dim=1)
correct = (predictions == labels).sum().item()
```

`argmax(dim=1)` 在每张图片的 10 个分数中选出最大值的类别编号。
与真实标签比较后统计正确个数，再用“总正确个数 / 总样本数”得到准确率。

loss 与准确率关注的内容不同：loss 还反映模型对正确类别的确信程度；准确率只统计最终选出的类别是否正确。
脚本里的训练准确率是在模型不断更新的过程中累计的，不是最终模型重新评估整个训练集的结果。

## 6. 测试为什么不更新参数

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

- `model.eval()`：切换到评估模式，影响 Dropout、BatchNorm 等特定层。
- `torch.no_grad()`：关闭这段代码里的梯度记录，节省资源；它与 eval 的作用不同。
- 测试循环不执行 `backward()` 和 `optimizer.step()`，因此不会根据测试标签学习。

本课在训练前固定轮数，完成训练后只评估一次完整测试集。
如果后续需要根据准确率反复挑选学习率、轮数或网络结构，应从训练集划分验证集，保留测试集用于最后报告。

## 7. 输出文件

每次完整运行后，会在 `output/cnn_lesson02/` 下新建一个带时间戳的目录：

- `mnist_cnn.pth`：训练后的模型参数；未包含优化器状态，用于后续加载和推理学习。
- `metrics.json`：运行设置、每轮训练指标、完整测试集指标和前 8 张测试图片的预测。

每次启动脚本都会创建新模型，从随机初始化开始训练。模型加载与续训将在后续课程讲解。
固定种子方便比较，但不同设备、软件版本和 GPU 算法仍可能导致结果略有差异。

### 已完成的实测记录（2026-10-26）

在本机 NVIDIA GeForce GTX 1650 上，使用本课默认配置完成了 3 轮训练。

| 训练轮次 | 平均训练 loss | 训练过程累计准确率 |
|---|---|---|
| 1 | 0.3229 | 90.58% |
| 2 | 0.1014 | 96.99% |
| 3 | 0.0743 | 97.75% |

训练后在全部 10000 张测试图片上评估：正确 9797 张，准确率 97.97%，平均 loss 为 0.0616。训练和测试共约 33.9 秒，不含数据下载。

- [本次完整运行记录](../output/cnn_lesson02/20261026-214736-194644/metrics.json)
- [本次保存的模型参数](../output/cnn_lesson02/20261026-214736-194644/mnist_cnn.pth)
- [真实测试图片与预测对照图](../output/cnn_lesson02/20261026-214736-194644/mnist_examples.png)：上排是按测试集顺序最先遇到的 4 个正确预测，下排是最先遇到的 4 个错误预测；`true` 是正确标签，`pred` 是模型预测。图中的 8 张图片只是便于观察的例子，准确率仍以完整测试集的 10000 张图片为准。

已验证保存的模型可以在 CPU 上重新加载，并复现记录中的前 8 张测试图片预测；评估函数不会修改模型参数或产生梯度。另用 64+1 个样本检查了最后一个小批次的损失统计，结果与一次计算全部 65 个样本一致。

## 8. 本节练习

1. 先在不运行代码的情况下计算：batch_size=128 时，一轮训练有几个 batch？最后一批有几张？
2. 阅读 `evaluate()`，找出与 `train_one_epoch()` 的三处核心区别。
3. 查看 `logits.argmax(dim=1)`。为什么这里沿着第 1 维选择最大值？

<details>
<summary>做完后查看提示</summary>

1. 60000 = 468×128 + 96，一轮共 469 批，最后一批 96 张。
2. 使用 eval 模式、用 no_grad 关闭梯度记录、不做反向传播和参数更新。
3. 输出是 `[批量数, 类别数]`，第 1 维对应每张图片的 10 个类别分数。

</details>

下一课：[第三课：学习率、batch size 和 epoch](README_CNN03.md)，通过固定条件的对比实验观察训练变化。

## 资料

- [Torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)：数据集参数。
- [MNIST 官方数据镜像仓库](https://github.com/cvdfoundation/mnist)：数据来源与训练/测试划分。
- [ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html)：本课的图片转换。
- [PyTorch Dataset 与 DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)：按样本与按批次读取。
- [PyTorch 训练与评估](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：训练循环和测试循环。
