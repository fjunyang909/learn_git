"""
Fashion-MNIST 服装分类示例，来源于 PyTorch 官方模型训练教程。

执行顺序：准备数据 → 查看样本 → 定义模型 → 训练与验证 → 记录损失并保存权重。
每张输入图片是 28×28 的单通道灰度图，标签是 0～9 的服装类别编号。
Dataset 负责读取单个样本，DataLoader 负责把多个样本组成一批数据。

本文件在顶层执行这些步骤：直接运行或导入文件都会触发数据准备与训练。
"""

import torch
import torchvision
from torchvision.transforms import v2

# 将训练指标写入日志文件，供 TensorBoard 绘制曲线。
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 按顺序处理每张图片：转为图像张量、缩放像素值，再做归一化。
transform = v2.Compose([
    v2.ToImage(),  # 转为形状为 [通道数, 高, 宽] 的图像张量。
    v2.ToDtype(torch.float32, scale=True),  # 将整数像素值 0～255 缩放到 0～1。
    v2.Normalize((0.5,), (0.5,))  # 按 (像素值 - 0.5) / 0.5 映射到 -1～1。
])

# 数据保存在运行时工作目录下的 data 文件夹，缺少时自动下载。
# 注意：train=False 读取官方测试集；本教程将其当作验证集使用。
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# 每批 4 张图片；训练时打乱样本顺序，验证时保持原顺序。
# 一批图片的形状为 [4, 1, 28, 28]，一批标签的形状为 [4]。
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# 类别名称用元组保存，位置与标签编号一一对应，而不是图片在数据集中的位置。
# 0～4：T 恤/上衣、裤子、套头衫、连衣裙、外套；5～9：凉鞋、衬衫、运动鞋、包、短靴。
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# len(数据集) 是样本数；len(数据加载器) 是每轮的批次数。
print(f'Training set has {len(training_set)} instances')
print(f'Validation set has {len(validation_set)} instances')

# 先查看一批样本，检查图片和标签是否对应。

import matplotlib.pyplot as plt
import numpy as np


# 显示图片的辅助函数，不属于神经网络层，因此单独放在模型类之外。
# one_channel 是可选参数：为 True 时按灰度图显示。
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)  # 沿通道维求平均，得到二维灰度图。
    img = img / 2 + 0.5  # 撤销前面的归一化，将像素值恢复到 0～1。
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        # 绘图库需要 [高, 宽, 通道数]，所以要调整维度顺序。
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# iter 创建迭代器，next 取出第一批；等号左侧将图片和标签分别解包。
dataiter = iter(training_loader)
images, labels = next(dataiter)

# 将多张图片拼成网格；labels[j] 是第 j 张图的类别编号。
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))

# 模型采用类似 LeNet-5 的结构：两组卷积与池化，再接三层全连接层。

import torch.nn as nn
import torch.nn.functional as F


# 继承 nn.Module 后，PyTorch 才能统一管理网络层、参数、训练模式和权重保存。
class GarmentClassifier(nn.Module):
    # 构造函数负责创建各层；self 表示当前模型实例，各层保存在实例属性中。
    def __init__(self):
        super().__init__()  # 先初始化父类，再注册下面的网络层。
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入 1 通道，输出 6 通道，卷积核为 5×5。
        self.pool = nn.MaxPool2d(2, 2)  # 取 2×2 区域的最大值，步长为 2。
        self.conv2 = nn.Conv2d(6, 16, 5)  # 将 6 通道特征转换为 16 通道特征。
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 将展平后的 256 个特征映射为 120 个。
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 输出每个类别的原始分数，共 10 个。

    # 前向传播描述数据经过各层的顺序；调用 model(x) 时由 PyTorch 调用此方法。
    # N 表示批次大小；当前模型的尺寸计算建立在输入图片为 28×28 的前提上。
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [N, 1, 28, 28] → [N, 6, 24, 24] → [N, 6, 12, 12]
        x = self.pool(F.relu(self.conv2(x)))  # → [N, 16, 8, 8] → [N, 16, 4, 4]
        x = x.view(-1, 16 * 4 * 4)  # 展平为 [N, 256]；-1 让 PyTorch 自动推算 N。
        # ReLU 将负数置零，为网络引入非线性。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 输出 [N, 10]，这些分数尚未转换成概率。
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备:{device}")
# 创建一个具体的模型对象；本文件没有迁移设备的代码，按通常默认设置在 CPU 上运行。
model = GarmentClassifier().to(device)

# 交叉熵用于单标签多分类：输入为 [N, 10] 的原始分数和 [N] 的整数标签。
# 它内部包含对数概率计算，因此模型末尾不需要额外添加 Softmax。
loss_fn = torch.nn.CrossEntropyLoss()

# 下面只演示损失函数的用法：随机生成 4 组类别分数，并非模型的真实预测。
dummy_outputs = torch.rand(4, 10)
# 每个整数指定对应样本的正确类别，取值范围为 0～9。
dummy_labels = torch.tensor([1, 5, 3, 7])

print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print(f'Total loss for this batch: {loss.item()}')

# 带动量的随机梯度下降：parameters() 提供模型参数，lr 控制更新步长。
# momentum 让参数更新保留之前梯度方向的影响。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 一轮训练是将训练集遍历一次；每一批都执行清梯度、前向计算、反向传播和参数更新。
# 这是训练流程函数，不是网络结构的一部分，因此可以放在模型类外。
# 它使用外部的 training_loader、model、loss_fn、optimizer；两个形参负责轮次和日志。
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # enumerate 同时给出从 0 开始的批次编号 i，以及该批数据 data。
    for i, data in enumerate(training_loader):
        # 将这一批数据解包为图片张量和标签张量。
        inputs, labels = data
        inputs = inputs.to(device,non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # PyTorch 默认累积梯度；每次独立更新前需要清除上一批的梯度。
        optimizer.zero_grad()

        # 调用模型完成前向传播，得到每张图片的 10 个类别分数。
        outputs = model(inputs)

        # 计算预测与标签的差距；backward 计算梯度，本身不更新参数。
        loss = loss_fn(outputs, labels)
        loss.backward()

        # 优化器依据刚才计算的梯度，实际更新网络权重与偏置。
        optimizer.step()

        # item() 将标量张量转成 Python 数值，便于累计和记录。
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # 最近 1000 个批次的平均损失。
            print(f'  batch {i + 1} loss: {last_loss}')
            # 把轮次和当前批次合成连续的全局步数，作为曲线横轴。
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    # 注意：返回的是最后一次记录的 1000 批均值，不是整轮的平均损失。
    # 不满 1000 批的尾段未计入该值；整轮不足 1000 批时会返回初始值 0。
    return last_loss


# 每轮依次执行训练、验证、记录指标，并在验证损失降低时保存权重。
# 时间戳用于区分不同实验；日志写入当前工作目录下的 runs 文件夹。
# 可在对应工作目录运行 tensorboard --logdir=runs，查看损失曲线。
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
epoch_number = 0

# 训练 5 轮；按标准训练集 60000 张、每批 4 张计算，每轮执行 15000 次参数更新。
EPOCHS = 1

# 下划线只是数字分隔符；此处用一个较大的初始值记录目前最小的验证损失。
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number + 1}:')

    # 切换为训练模式；这不会启动训练，也不负责开启梯度跟踪。
    # 真正的数据遍历和参数更新发生在下一行调用的函数中。
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # 切换为评估模式，影响 Dropout、BatchNorm 等层，但不会自动关闭梯度。
    # 当前网络没有这两类层，仍保留标准的训练/评估模式切换流程。
    model.eval()

    # 验证时不需要反向传播；关闭梯度记录，减少计算图带来的内存开销。
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    # i 从 0 开始，所以 i + 1 是验证批次总数；这里统计整个验证集的批均值。
    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss}')

    # 用字典将训练与验证损失放在同一组曲线中，横轴是轮次。
    # 注意：avg_loss 与 avg_vloss 当前的统计范围不同，比较时需留意。
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()  # 将缓冲中的日志写入磁盘。

    # 验证损失比此前更低时才保存；不是每轮都保存，也不是按准确率选模型。
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'model_{timestamp}_{epoch_number}'
        # state_dict 保存参数和缓冲区，不包含模型类定义、优化器状态或当前轮次。
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# 加载权重时，先创建结构完全一致的模型，再读取保存的参数：
# saved_model = GarmentClassifier()
# saved_model.load_state_dict(torch.load(PATH))
# PATH 需要替换为实际权重文件路径；推理前还应调用 saved_model.eval()。
# 若要完整恢复中断的训练，还需保存和恢复优化器状态、轮次等信息。
