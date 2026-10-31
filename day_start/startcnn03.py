r"""CNN 第三课：比较学习率、batch size 和 epoch 对训练的影响。

在项目根目录运行全部五组实验：
    .\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn03.py

先只运行基准组：
    .\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn03.py --experiment baseline

阅读顺序：EXPERIMENTS -> main -> create_subsets -> run_experiment。
沿用前两课的网络和训练函数；本课从官方训练集划分验证集。
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from startcnn01 import SmallCNN
from startcnn02 import evaluate, train_one_epoch


# 相对于 baseline，每组只改变学习率或 batch size 中的一项。
# 每组都从相同初始权重重新训练；每一轮的记录用来观察 epoch 的影响。
EXPERIMENTS = [
    {"name": "baseline",    "learning_rate": 0.001,  "batch_size": 64},
    {"name": "lr_small",    "learning_rate": 0.0001, "batch_size": 64},
    {"name": "lr_large",    "learning_rate": 0.1,    "batch_size": 64},
    {"name": "batch_small", "learning_rate": 0.001,  "batch_size": 32},
    {"name": "batch_large", "learning_rate": 0.001,  "batch_size": 128},
]
EPOCHS = 3
TRAIN_SAMPLES = 12000
VAL_SAMPLES = 3000
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data10_27"


def create_subsets():
    """只加载官方训练集，随机分成训练、验证和本次不用的三个部分。"""
    dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor(),
    )
    if TRAIN_SAMPLES < 1 or VAL_SAMPLES < 1:
        raise ValueError("训练和验证样本数都必须大于 0。")
    if TRAIN_SAMPLES + VAL_SAMPLES > len(dataset):
        raise ValueError("训练和验证样本数之和不能超过官方训练集大小。")

    train_subset, val_subset, unused_subset = random_split(
        dataset,
        [TRAIN_SAMPLES, VAL_SAMPLES, len(dataset) - TRAIN_SAMPLES - VAL_SAMPLES],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"训练：{len(train_subset)} 张；验证：{len(val_subset)} 张；"
          f"本次未使用的训练图片：{len(unused_subset)} 张")
    print("所有实验复用同一划分；本课不加载官方测试集。")
    return train_subset, val_subset


def run_experiment(config, train_subset, val_subset, device, epochs):
    """固定其他条件，只改变 config 指定的设置。"""
    # 每组重设种子并创建新模型、新优化器，避免接着上一组的权重训练。
    torch.manual_seed(SEED)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(SEED),
    )
    # 验证批量固定；独立的 generator 不消耗训练数据打乱顺序用的随机数。
    val_loader = DataLoader(
        val_subset, batch_size=256, shuffle=False,
        num_workers=0, generator=torch.Generator().manual_seed(SEED),
    )
    steps_per_epoch = len(train_loader)
    print(f"\n实验 {config['name']}：lr={config['learning_rate']}，"
          f"batch_size={config['batch_size']}，每轮更新 {steps_per_epoch} 次")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = perf_counter()
    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n  Epoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        # 第二课的 evaluate 也适用于验证集：它不反向传播、不更新参数。
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        history.append({
            "epoch": epoch,
            "total_steps": epoch * steps_per_epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })
        print(f"  训练过程：loss={train_metrics['loss']:.4f}，"
              f"准确率={train_metrics['accuracy']:.2%}")
        print(f"  轮末验证：loss={val_metrics['loss']:.4f}，"
              f"准确率={val_metrics['accuracy']:.2%}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        **config,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": epochs * steps_per_epoch,
        "elapsed_seconds": perf_counter() - start_time,
        "history": history,
    }


def save_results(results, device, device_name, epochs):
    """保存完整历史与逐轮表格，方便对照，不需要额外安装绘图库。"""
    output_dir = PROJECT_ROOT / "output" / "cnn_lesson03" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device), "device_name": device_name,
        "seed": SEED, "epochs": epochs,
        "train_samples": TRAIN_SAMPLES, "validation_samples": VAL_SAMPLES,
        "data_source": "MNIST official training split only",
        "split_lengths": [TRAIN_SAMPLES, VAL_SAMPLES, 60000 - TRAIN_SAMPLES - VAL_SAMPLES],
        "test_set_used": False,
        "experiments": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    columns = [
        "experiment", "learning_rate", "batch_size", "epoch", "total_steps",
        "train_loss", "train_accuracy", "validation_loss", "validation_accuracy",
    ]
    with (output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for row in result["history"]:
                writer.writerow({
                    "experiment": result["name"],
                    "learning_rate": result["learning_rate"],
                    "batch_size": result["batch_size"],
                    "epoch": row["epoch"], "total_steps": row["total_steps"],
                    "train_loss": row["train"]["loss"],
                    "train_accuracy": row["train"]["accuracy"],
                    "validation_loss": row["validation"]["loss"],
                    "validation_accuracy": row["validation"]["accuracy"],
                })
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="CNN 第三课：一次改变一个设置。")
    parser.add_argument(
        "--experiment", choices=["all"] + [item["name"] for item in EXPERIMENTS],
        default="all", help="默认运行全部五组；baseline 只运行基准组。",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="每组训练的轮数，默认 3。")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须至少为 1。")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"使用设备：{device}（{device_name}）")
    train_subset, val_subset = create_subsets()
    configs = [item for item in EXPERIMENTS if args.experiment in ("all", item["name"])]
    results = []
    for config in configs:
        results.append(run_experiment(config, train_subset, val_subset, device, args.epochs))

    print("\n实验汇总：以下均为最后一轮的验证结果。")
    print(f"{'实验':<14} {'lr':>8} {'batch':>6} {'总更新次数':>8} {'验证 loss':>10} {'验证准确率':>10}")
    for result in results:
        val = result["history"][-1]["validation"]
        print(f"{result['name']:<14} {result['learning_rate']:>8g} "
              f"{result['batch_size']:>6} {result['total_steps']:>12} "
              f"{val['loss']:>12.4f} {val['accuracy']:>12.2%}")
    output_dir = save_results(results, device, device_name, args.epochs)
    print(f"\n逐轮记录已保存到：{output_dir}")
    print("相同 epoch 下，batch size 不同会导致更新次数不同；单次实验不能证明某个设置普遍最好。")


if __name__ == "__main__":
    main()
