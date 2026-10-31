r"""CNN 第一课：观察卷积、张量形状，以及一次参数更新。

在项目根目录运行：
    .\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn01.py

本课使用人工矩阵和随机数据，不下载数据集。
随机数据用于观察计算流程，不能证明模型已经学会识别数字。
"""

import torch
from torch import nn
from torch.nn import functional as F


def show_one_convolution():
    """用固定的小矩阵，理解“对应元素相乘，再求和”。"""
    image = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=torch.float32)
    kernel = torch.tensor([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=torch.float32)

    # 增加“批量”和“通道”两个维度：[5, 5] -> [1, 1, 5, 5]。
    # 这里的卷积核由我们指定；真正训练 CNN 时，核的权重由训练更新。
    output = F.conv2d(image[None, None], kernel[None, None])
    first_patch = image[:3, :3]
    print("\n一、手算一个卷积位置（步长 1、不补边、偏置为 0）")
    print("输入图片：\n", image)
    print("卷积核：\n", kernel)
    print("左上角 3×3 区域与卷积核逐元素相乘：\n", first_patch * kernel)
    print("求和得到左上角输出：", (first_patch * kernel).sum().item())
    print("完整的 3×3 输出：\n", output[0, 0])


class SmallCNN(nn.Module):
    """接收 [N, 1, 28, 28] 灰度图，输出 [N, 10] 个分类分数。"""

    def __init__(self):
        super().__init__()
        # Sequential 按顺序组织各层；所有可学习参数会自动登记到模型。
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # [N, 8, 28, 28]
            nn.ReLU(),                                 # 负数变成 0，形状不变
            nn.MaxPool2d(kernel_size=2, stride=2),       # [N, 8, 14, 14]
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # [N, 16, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # [N, 16, 7, 7]
            nn.Flatten(start_dim=1),                    # [N, 16*7*7]
            nn.Linear(16 * 7 * 7, 10),                  # [N, 10]
        )

    def forward(self, x, show_shapes=False):
        # model(images) 会调用 forward，按下面的顺序进行前向计算。
        if show_shapes:
            print(f"输入：{list(x.shape)}，设备：{x.device}")
        for index, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if show_shapes:
                print(f"第 {index} 层 {layer.__class__.__name__:12s} -> {list(x.shape)}")
        return x


def main():
    torch.manual_seed(42)
    show_one_convolution()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n二、观察 CNN 的逐层形状")
    print("使用设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(device))

    model = SmallCNN().to(device)
    # 优化器在模型移动到目标设备之后创建。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 4 张模拟灰度图，每张大小 28×28；随机标签取 0～9。
    # Tensor.to() 返回目标设备上的张量，因此要保存返回值。
    images = torch.randn(4, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (4,)).to(device)
    weight_before = model.layers[0].weight.detach().clone()

    model.train()
    optimizer.zero_grad()                # 清除上一步的梯度
    logits = model(images, show_shapes=True)
    loss = loss_fn(logits, labels)       # 直接传入分数，不先做 softmax
    loss.backward()                     # 计算梯度，这一步不更新权重

    print("\n三、观察一次参数更新")
    print("模型设备：", next(model.parameters()).device)
    print("标签设备：", labels.device)
    print("输出设备：", logits.device)
    print("损失值：", loss.item())
    print("第一层梯度设备：", model.layers[0].weight.grad.device)
    print("第一层梯度绝对值的平均：", model.layers[0].weight.grad.abs().mean().item())

    optimizer.step()                    # 根据梯度更新可学习参数
    change = (model.layers[0].weight.detach() - weight_before).abs().sum().item()
    print("第一层权重变化的绝对值总和：", change)
    print("变化大于 0，表示这次优化步骤确实更新了第一层权重。")
    print("随机图片和随机标签仅演示计算流程；识别效果要用真实数据与独立测试集评估。")


if __name__ == "__main__":
    main()
