"""第四课的绘图辅助函数。理解训练流程时可以先跳过这个文件。"""

import matplotlib
matplotlib.use("Agg")  # 保存文件，不弹出额外窗口。
import matplotlib.pyplot as plt
import torch


LABELS = {
    "baseline": "Baseline",
    "augmentation": "Data augmentation",
    "dropout": "Dropout (p = 0.5)",
    "weight_decay": "Weight decay (0.1)",
}


def plot_learning_curves(report, output_dir):
    experiments = report["experiments"]
    columns = 2 if len(experiments) > 1 else 1
    rows = (len(experiments) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12 if columns == 2 else 8, 4 * rows), squeeze=False)
    for axis, result in zip(axes.flat, experiments):
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_eval"]["loss"] for row in history], color="#2563eb", linewidth=2, label="Train: original images, eval mode")
        axis.plot(epochs, [row["validation"]["loss"] for row in history], color="#d97706", linewidth=2, label="Validation: original images, eval mode")
        axis.scatter(result["best_epoch"], result["best_validation"]["loss"], marker="*", s=150, color="#15803d", zorder=3)
        axis.axvline(result["best_epoch"], color="#15803d", alpha=0.4, linestyle="--")
        axis.set_title(f"{LABELS[result['name']]} | selected epoch: {result['best_epoch']}", loc="left", fontsize=11)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss (lower is better)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(experiments):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    figure.suptitle(
        f"Lesson 04 | {report['train_samples']:,} training images, {report['validation_samples']:,} validation images\n"
        "Same evaluation conditions for both curves | Test set not used", fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.11 if rows == 1 else 0.07, 1, 0.91))
    figure.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


def plot_augmentation(plain_train, augmented_train, output_dir, seed):
    # 从训练集按顺序选择三个不同标签；预览只展示变换，不表示模型预测。
    positions = []
    seen = set()
    for position in range(len(plain_train)):
        _, label = plain_train[position]
        if label not in seen:
            positions.append(position)
            seen.add(label)
        if len(positions) == 3:
            break
    figure, axes = plt.subplots(len(positions), 6, figsize=(10, 1.8 * len(positions)), squeeze=False)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 100)
        for row, position in enumerate(positions):
            for column in range(6):
                image, label = plain_train[position] if column == 0 else augmented_train[position]
                axis = axes[row, column]
                axis.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title("Original" if column == 0 else f"Variation {column}", fontsize=10)
                if column == 0:
                    axis.set_ylabel(f"Label: {label}")
    figure.suptitle("Training-only augmentation | rotation +/-10 degrees, translation up to 8%\nLabels are preserved; each sample is drawn afresh.", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_dir / "augmentation_examples.png", dpi=150)
    plt.close(figure)
