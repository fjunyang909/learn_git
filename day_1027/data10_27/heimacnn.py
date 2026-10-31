"""
案例:
    演示CNN的综合案例, 图像分类.

回顾: 深度学习项目的步骤
    1. 准备数据集.
        这里我们用的时候 计算机视觉模块 torchvision自带的 CIFAR10数据集, 包含6W张 (32,32,3)的图片, 5W张训练集, 1W张测试集, 10个分类, 每个分类6K张图片.
        你需要单独安装一下 torchvision包, 即: pip install torchvision
    2. 搭建(卷积)神经网络
    3. 模型训练.
    4. 模型测试.

卷积层:
    提取图像的局部特征 -> 特征图(Feature Map), 计算方式:  N = (W - F + 2P) // S + 1
    一个卷积核在不同空间位置共享参数, 生成一个输出通道的特征图.

池化层:
    降维, 有最大池化 和 平均池化.
    池化只在HW上做调整, 通道上不改变.

案例的优化思路:
    1. 增加卷积核的输出通道数(大白话: 卷积核的数量)
    2. 增加全连接层的参数量.
    3. 调整学习率
    4. 调整优化方法(optimizer...)
    5. 调整激活函数...
    6. ...
"""


import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader

# 每批次样本数。增大批次可以减少每轮的迭代次数。
BATCH_SIZE = 64
EPOCHS = 20
BASE_DIR = Path(__file__).resolve().parent


#1。准备数据集
def create_dataset():
    data_root = BASE_DIR

    train_dataset = CIFAR10(
        root=str(data_root), train=True,
        transform=ToTensor(), download=False
    )
    test_dataset = CIFAR10(
        root=str(data_root), train=False,
        transform=ToTensor(), download=False
    )
    return train_dataset, test_dataset

#2、搭建神经网络
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,3,1,0)
        self.pool1=nn.MaxPool2d(2,2,0)
        self.conv2=nn.Conv2d(6,16,3,1,0)
        self.pool2=nn.MaxPool2d(2,2,0)
        # 第1个隐藏层(全连接层), 输入: 576, 输出: 120
        self.linear1 = nn.Linear(576, 120)
        # 第2个隐藏层 (全连接层), 输入: 120, 输出: 84
        self.linear2 = nn.Linear(120, 84)
        # 第3个隐藏层 (全连接层) -> 输出层,  输入: 84, 输出: 10
        self.output = nn.Linear(84, 10)

    # 2. 定义前向传播
    def forward(self,x):
        # 第1层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        # 分解版.
        # x = self.conv1(x)   # 卷积层
        # x = torch.relu(x)   # 激励层
        # x = self.pool1(x)   # 池化层
        # 合并版   池化   +  激活函数  +  卷积
        x = self.pool1(torch.relu(self.conv1(x)))

        # 第2层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x = self.pool2(torch.relu(self.conv2(x)))

        # 将每张图片的所有特征合并: (N, 16, 6, 6) -> (N, 576)。
        # Linear 作用于输入的最后一个维度; 这里需要让该维度包含整张图的特征。
        # 参1: 样本数(行数), 参2: 列数(特征数), -1表示自动计算.
        x = x.reshape(x.size(0), -1)  # N行576列，兼容最后一个不足64张的批次。
        # print(f'x.shape: {x.shape}')

        # 第3层: 全连接层(加权求和) + 激励层(激活函数)
        x = torch.relu(self.linear1(x))

        # 第4层: 全连接层(加权求和) + 激励层(激活函数)
        x = torch.relu(self.linear2(x))

        # 第5层: 全连接层(加权求和) -> 输出层
        # 输出未经 softmax 的分数（logits），直接交给 CrossEntropyLoss。
        return self.output(x)



def train(train_dataset, epochs=EPOCHS, model_path=None):
    if epochs < 1:
        raise ValueError("epochs 必须至少为 1")

    #1、选择设备并创建数据加载器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_cuda
    )
    #2、创建模型对象
    if device.type == "cuda":
        print(f"使用设备：{device}（{torch.cuda.get_device_name(device)}）")
    else:
        print(f"使用设备：{device}（未检测到 CUDA，退回 CPU）")
    model = ImageModel().to(device)
    #3.创建损失函数
    criterion=nn.CrossEntropyLoss()
    #4.创建优化器对象
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    #5.循环遍历epoch,开始每轮的训练动作
    #5.1 epochs 默认取 EPOCHS，也可以传入较小的轮数进行调试。
    #5.2 遍历，完成每轮的所有批次训练动作
    for epoch_idx in range(epochs):
        # 5.2.1 定义变量, 记录: 总损失, 总样本数据量, 预测正确样本个数, 训练(开始)时间
        # 指标先在当前设备上累计，每轮结束再取回数值，减少 GPU 同步等待。
        total_loss = torch.zeros((), device=device)
        total_correct = torch.zeros((), dtype=torch.long, device=device)
        total_samples, start = 0, time.perf_counter()
        model.train()
        # 5.2.2 遍历数据加载器, 获取到 每批次的 数据.
        for x, y in dataloader:
            # 5.2.3 模型、图片和标签必须在同一个设备上。
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            # 5.2.4 模型预测.
            y_pred = model(x)
            # 5.2.5 计算损失.
            loss = criterion(y_pred, y)
            # 5.2.6 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # 5.2.7 统计预测正确的样本个数.
            # print(y_pred)     # 批次中, 每张图 每个分类的原始分数（不是概率）。

            # argmax() 返回最大值对应的索引, 充当 -> 该图片的 预测分类.
            # tensor([9, 8, 5, 5, 1, 5, 8, 5])
            # print(torch.argmax(y_pred, dim=-1))     # -1表示最后一维，即10个类别所在的维度。
            # print(y)                                # 真实分类
            # print(torch.argmax(y_pred, dim=-1) == y)    # 是否预测正确
            # print((torch.argmax(y_pred, dim=-1) == y).sum())    # 预测正确的样本个数.
            total_correct += (y_pred.detach().argmax(dim=1) == y).sum()

            # 5.2.8 统计当前批次的总损失.          第1批平均损失 * 第1批样本个数
            total_loss += loss.detach() * len(y)  # detach 防止累计指标保留计算图。
            # 5.2.9  统计当前批次的总样本个数.
            total_samples += len(y)
            # break   每轮只训练1批, 提高训练效率, 减少训练时长, 只有测试会这么写, 实际开发绝不要这样做.

        # 5.2.10 一轮训练完毕，打印该轮的训练信息。
        average_loss = total_loss.item() / total_samples
        accuracy = total_correct.item() / total_samples
        print(
            f'epoch: {epoch_idx + 1}/{epochs}, loss: {average_loss:.5f}, '
            f'acc: {accuracy:.2%}, time: {time.perf_counter() - start:.2f}s',
            flush=True,
        )

    # 6. 保存模型。默认路径固定在脚本目录下，并自动创建父目录。
    model_path = Path(model_path) if model_path is not None else BASE_DIR / 'model' / 'image_model.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存：{model_path.resolve()}')
    return model


@torch.inference_mode()
def evaluate(model, test_dataset):
    # 测试时关闭训练行为和梯度计算，统计整个测试集的表现。
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_cuda
    )
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    total_samples = 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        y_pred = model(x)
        total_loss += criterion(y_pred, y)
        total_correct += (y_pred.argmax(dim=1) == y).sum()
        total_samples += len(y)

    metrics = {
        'loss': total_loss.item() / total_samples,
        'accuracy': total_correct.item() / total_samples,
    }
    print(f"test loss: {metrics['loss']:.5f}, test acc: {metrics['accuracy']:.2%}")
    return metrics


if __name__== '__main__':
    # 1. 获取数据集.
    train_dataset, test_dataset = create_dataset()
    #print(f'训练集：{train_dataset.data.shape}')
    #print(f'测试集：{test_dataset.data.shape}')
    #print(f'数据集类别：{train_dataset.class_to_idx}')
    #import matplotlib.pyplot as plt
    #plt.figure(figsize=(2,2))
    #plt.imshow(train_dataset.data[11])
    #plt.title(train_dataset.targets[11])
    #plt.show()
    # 选择设备：有 CUDA 就用 GPU，否则退回 CPU
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #if device.type == "cuda":
    #    print(f"使用设备：{device}（{torch.cuda.get_device_name(device)}）")
    #else:
    #    print(f"使用设备：{device}（未检测到 CUDA，退回 CPU）")

    # 模型必须先在目标设备上创建/移动，再把同设备的信息告诉 summary
    #from torchsummary import summary
    #model = ImageModel().to(device)

    # torchsummary 不会自动搬模型：它只按 device= 把输入张量放到该设备。
    # 模型在 GPU、输入也在 GPU，才不会报设备不一致。
    #summary(model, (3, 32, 32), batch_size=2, device=device.type)
    model = train(train_dataset)
    evaluate(model, test_dataset)
