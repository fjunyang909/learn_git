r"""CNN 第四课：观察过拟合，比较数据增强、Dropout 和权重衰减。

运行全部四组：
    .\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn04.py
只运行基准组：
    .\.venv\Scripts\python.exe -X utf8 .\day_start\startcnn04.py --experiment baseline

阅读顺序：EXPERIMENTS -> RegularizedCNN -> create_subsets -> run_experiment。
本课只使用官方训练集，测试集留作最终评估。
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from startcnn01 import SmallCNN
from startcnn02 import evaluate, train_one_epoch
from plot_cnn04 import plot_augmentation, plot_learning_curves


EXPERIMENTS = [
    {"name": "baseline",     "augmentation": False, "dropout": 0.0, "weight_decay": 0.0},
    {"name": "augmentation", "augmentation": True,  "dropout": 0.0, "weight_decay": 0.0},
    {"name": "dropout",      "augmentation": False, "dropout": 0.5, "weight_decay": 0.0},
    {"name": "weight_decay", "augmentation": False, "dropout": 0.0, "weight_decay": 0.1},
]
TRAIN_SAMPLES = 1000       # 小训练集便于较快观察训练表现与验证表现的差距
VAL_SAMPLES = 3000
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data10_27"


class RegularizedCNN(SmallCNN):
    """保留第一课的网络，在 Flatten 和最后的 Linear 之间加入 Dropout。"""

    def __init__(self, dropout_p=0.0):
        super().__init__()
        layers = list(self.layers.children())
        layers.insert(-1, nn.Dropout(p=dropout_p))
        self.layers = nn.Sequential(*layers)
        # Dropout 没有可学习权重；p=0 时不丢弃特征，各组可学习参数数量相同。


def create_subsets(train_samples=TRAIN_SAMPLES):
    """增强版和原图版使用不同 Dataset，避免把验证图片也随机变换。"""
    plain_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transforms.ToTensor(),
    )
    augmented_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), fill=0),
            transforms.ToTensor(),
        ]),
    )
    if not 1 <= train_samples <= len(plain_dataset) - VAL_SAMPLES:
        raise ValueError(f"训练样本数必须介于 1 和 {len(plain_dataset) - VAL_SAMPLES} 之间。")

    indices = torch.randperm(
        len(plain_dataset), generator=torch.Generator().manual_seed(SEED),
    ).tolist()
    # 先固定验证索引。以后增加训练样本数时，验证集仍然保持相同。
    val_indices = indices[:VAL_SAMPLES]
    train_indices = indices[VAL_SAMPLES:VAL_SAMPLES + train_samples]
    plain_train = Subset(plain_dataset, train_indices)
    augmented_train = Subset(augmented_dataset, train_indices)
    validation = Subset(plain_dataset, val_indices)
    return plain_train, augmented_train, validation


def make_loaders(config, plain_train, augmented_train, validation):
    train_subset = augmented_train if config["augmentation"] else plain_train
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    # 两个评分 loader 都只读原始图片，配合 evaluate 的 eval 模式关闭 Dropout。
    train_eval_loader = DataLoader(
        plain_train, batch_size=256, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        validation, batch_size=256, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    return train_loader, train_eval_loader, val_loader


def run_experiment(config, subsets, device, epochs, patience, output_dir):
    torch.manual_seed(SEED)
    model = RegularizedCNN(dropout_p=config["dropout"]).to(device)
    # 全部使用 AdamW，并显式指定 weight_decay；只有权重衰减组的值不为 0。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    train_loader, train_eval_loader, val_loader = make_loaders(config, *subsets)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    print(f"\n实验 {config['name']}：增强={config['augmentation']}，"
          f"Dropout={config['dropout']}，weight_decay={config['weight_decay']}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = perf_counter()

    for epoch in range(1, epochs + 1):
        print(f"\n  Epoch {epoch}/{epochs}")
        online_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        # 同一个轮末模型、原始图片、eval 模式：两条曲线才适合直接对照。
        clean_metrics = evaluate(model, train_eval_loader, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        if not all(math.isfinite(item["loss"]) for item in (online_metrics, clean_metrics, val_metrics)):
            raise RuntimeError("出现非有限 loss，请检查学习率及训练设置。")
        history.append({
            "epoch": epoch,
            "total_steps": epoch * len(train_loader),
            "train_online": online_metrics,
            "train_eval": clean_metrics,
            "validation": val_metrics,
        })
        print(f"  轮末原图训练集：loss={clean_metrics['loss']:.4f}，"
              f"准确率={clean_metrics['accuracy']:.2%}")
        print(f"  轮末验证集：    loss={val_metrics['loss']:.4f}，"
              f"准确率={val_metrics['accuracy']:.2%}")

        # 按验证 loss 选择模型，准确率只作为同时报告的另一个指标。
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            # 真正复制权重，否则训练继续时“最佳权重”也可能跟着变化。
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"  连续 {patience} 轮验证 loss 未刷新最低值，提前停止。")
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start_time
    checkpoint_name = f"{config['name']}_best.pth"
    torch.save(best_state, output_dir / checkpoint_name)
    best_row = history[best_epoch - 1]
    return {
        **config,
        "epochs_completed": len(history),
        "stopped_early": len(history) < epochs,
        "steps_per_epoch": len(train_loader),
        "total_steps": len(history) * len(train_loader),
        "best_epoch": best_epoch,
        "best_validation": best_row["validation"],
        "checkpoint": checkpoint_name,
        "elapsed_seconds": elapsed,
        "history": history,
    }


def save_results(output_dir, results, subsets, device, device_name, epochs, patience):
    plain_train, _, validation = subsets
    report = {
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "device": str(device), "device_name": device_name,
        "seed": SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW", "max_epochs": epochs, "patience": patience,
        "selection_metric": "minimum validation cross-entropy loss",
        "train_samples": len(plain_train), "validation_samples": len(validation),
        "split_indices": {"train": plain_train.indices, "validation": validation.indices},
        "augmentation": {"degrees": 10, "translate": [0.08, 0.08], "fill": 0},
        "test_set_used": False, "experiments": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    rows = []
    for result in results:
        for epoch in result["history"]:
            row = {"experiment": result["name"], "epoch": epoch["epoch"], "total_steps": epoch["total_steps"]}
            for split in ("train_online", "train_eval", "validation"):
                row[f"{split}_loss"] = epoch[split]["loss"]
                row[f"{split}_accuracy"] = epoch[split]["accuracy"]
            rows.append(row)
    with (output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description="CNN 第四课：过拟合与正则化。")
    parser.add_argument("--experiment", choices=["all"] + [item["name"] for item in EXPERIMENTS], default="all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--patience", type=int, default=0, help="默认 0：不提前停止，以观察完整曲线。")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 0 or not 1 <= args.train_samples <= 60000 - VAL_SAMPLES:
        parser.error("epochs 至少为 1；patience 不能为负；train-samples 必须在 1～57000 之间。")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"使用设备：{device}（{device_name}）")
    subsets = create_subsets(args.train_samples)
    print(f"训练 {len(subsets[0])} 张，验证 {len(subsets[2])} 张；不读取官方测试集。")
    output_dir = PROJECT_ROOT / "output" / "cnn_lesson04" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = [item for item in EXPERIMENTS if args.experiment in ("all", item["name"])]
    results = []
    for config in configs:
        results.append(run_experiment(config, subsets, device, args.epochs, args.patience, output_dir))
        # 每完成一组就保存记录，避免之后的绘图问题影响已完成的结果。
        report = save_results(output_dir, results, subsets, device, device_name, args.epochs, args.patience)

    plot_learning_curves(report, output_dir)
    plot_augmentation(subsets[0], subsets[1], output_dir, SEED)
    print("\n汇总：最佳轮次按验证 loss 最小选择，并非按准确率最大选择。")
    for result in results:
        final = result["history"][-1]
        print(f"{result['name']:<14} 最佳轮次={result['best_epoch']:>2}，"
              f"最佳轮验证 loss={result['best_validation']['loss']:.4f}，"
              f"对应准确率={result['best_validation']['accuracy']:.2%}；"
              f"最后一轮训练/验证准确率={final['train_eval']['accuracy']:.2%}/{final['validation']['accuracy']:.2%}")
    print(f"\n记录、最佳模型参数、训练曲线和增强示例已保存到：{output_dir}")


if __name__ == "__main__":
    main()
