r"""CNN 第二课：用 MNIST 训练，并在独立测试集上评估。

在项目根目录运行：
    .\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn02.py

阅读顺序：main -> create_loaders -> train_one_epoch -> evaluate。
首次运行下载 MNIST，以后复用项目 data10_27 目录里的数据。
"""

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 沿用第一课的网络。导入类不会执行第一课的 main()。
from startcnn01 import SmallCNN


# 先只关注这几个设置。
BATCH_SIZE = 64          # 一次送入网络的图片数量
EPOCHS = 3               # 把整个训练集学习几遍
LEARNING_RATE = 0.001    # 优化器更新参数的步幅系数
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data10_27"


def create_loaders():
    """Dataset 提供单个样本，DataLoader 把样本组成一批。"""
    # MNIST 是 28×28 的灰度图。ToTensor 转成 [1, 28, 28]，
    # 并将这里的 uint8 像素从 0～255 缩放到浮点数 0～1。
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,        # 每轮重新打乱顺序；图片和标签仍一一对应
        num_workers=0,       # 本课在主进程加载数据，便于 Windows 入门调试
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,       # 固定顺序，方便对照预测结果
        num_workers=0,
    )
    # 默认 drop_last=False，所以最后不足 64 张的一批也会被保留。
    return train_loader, test_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """学习一遍训练集；每处理一个 batch，更新一次模型参数。"""
    model.train()            # 切换训练模式；真正地更新发生在 optimizer.step()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        # 这五行与第一课相同，只是现在对真实数据的每个 batch 都执行。
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # argmax(dim=1)：为每张图片选出分数最高的类别。
        predictions = logits.argmax(dim=1)
        batch_count = labels.size(0)
        total_correct += (predictions == labels).sum().item()
        # loss 是当前 batch 的平均值，乘实际样本数后再累计，
        # 避免最后一个小 batch 被错误地赋予同样大的权重。
        total_loss += loss.item() * batch_count
        total_samples += batch_count

        if batch_index == 1:
            print(f"  图片形状：{list(images.shape)}，标签形状：{list(labels.shape)}")
            print(f"  图片设备：{images.device}，输出设备：{logits.device}")
        if batch_index % 250 == 0 or batch_index == len(loader):
            print(f"  已处理 {total_samples}/{len(loader.dataset)} 张，当前批次 loss={loss.item():.4f}")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "correct": total_correct,
        "samples": total_samples,
    }


def evaluate(model, loader, loss_fn, device):
    """在测试集上评分：不调用 backward，也不调用 optimizer.step。"""
    model.eval()             # 切换评估模式；它本身不会关闭梯度记录
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 评估不需要梯度，关闭记录可节省内存和计算。
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_count

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "correct": total_correct,
        "samples": total_samples,
    }


def main():
    if BATCH_SIZE < 1 or EPOCHS < 1:
        raise ValueError("BATCH_SIZE 和 EPOCHS 都必须至少为 1。")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"使用设备：{device}（{device_name}）")
    print(f"数据目录：{DATA_DIR}")
    train_loader, test_loader = create_loaders()
    print(f"训练集：{len(train_loader.dataset)} 张；测试集：{len(test_loader.dataset)} 张")
    print(f"batch_size={BATCH_SIZE}；每轮训练 {len(train_loader)} 个 batch；共 {EPOCHS} 轮")

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    start_time = perf_counter()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        history.append({"epoch": epoch, **metrics})
        print(f"  本轮训练：平均 loss={metrics['loss']:.4f}，准确率={metrics['accuracy']:.2%}")

    # 轮数在训练前已经确定；训练结束后再用独立测试集评估一次。
    # 后续若要根据评估结果调参，应从训练集再划分验证集。
    print("\n开始测试：使用训练时未参与参数更新的 10000 张图片。")
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    elapsed = perf_counter() - start_time
    print(f"测试集：平均 loss={test_metrics['loss']:.4f}，准确率={test_metrics['accuracy']:.2%}")
    print(f"预测正确：{test_metrics['correct']}/{test_metrics['samples']}；训练和评估耗时：{elapsed:.1f} 秒")

    # 查看前 8 张测试图片的预测；仍处于评估模式，不更新参数。
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        predictions = model(images[:8].to(device)).argmax(dim=1).cpu()
    preview = {"labels": labels[:8].tolist(), "predictions": predictions.tolist()}
    print("前 8 张的真实标签：", preview["labels"])
    print("前 8 张的预测结果：", preview["predictions"])

    # 每次运行单独保存，便于以后比较；这里保存模型参数，不包含优化器状态。
    output_dir = PROJECT_ROOT / "output" / "cnn_lesson02" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "mnist_cnn.pth")
    report = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device), "device_name": device_name,
        "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "train_history": history, "test": test_metrics,
        "preview": preview, "elapsed_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"模型参数和训练记录已保存到：{output_dir}")


if __name__ == "__main__":
    main()
