#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制类似论文 Fig. 4 的 CNN 结构图。只绘图，不训练 CNN。

安装依赖：python -m pip install numpy matplotlib
直接运行：python draw_cnn_architecture.py
替换输入图：python draw_cnn_architecture.py --input-image "my_cwt.png"

输出默认保存在本脚本旁的 figures 文件夹中，不依赖终端当前目录。
"""

from pathlib import Path
import argparse
import sys

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # 不依赖桌面图形环境；仍可在 VS Code / PyCharm 中运行
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
except ImportError as exc:
    raise SystemExit(
        "缺少绘图依赖，请先执行：\n"
        "  python -m pip install numpy matplotlib\n"
        f"原始错误：{exc}"
    ) from exc


# ================== 常用修改位置 ==================
TITLE = "CNN architecture for stroke classification"
INPUT_SHAPE = (16, 68, 256)  # 尺度 × 时间 × 输入通道
FEATURE_SHAPES = [
    (16, 68, 128),
    (8, 34, 128),
    (8, 34, 256),
    (4, 17, 256),
]
DENSE_UNITS = (512, 128)
COLORS = {
    "feature": "#842dff",
    "feature_edge": "#5d15c6",
    "patch": "#ffe000",
    "dense": "#4584fa",
    "dense_edge": "#2357b2",
    "output": "#a9dd85",
    "output_edge": "#54863a",
    "ink": "#20252b",
    "muted": "#646c77",
    "output_connection": "#e34b52",
}
FIGSIZE = (19.5, 5.3)
DEFAULT_DPI = 240
# ==================================================


def put_text(ax, x, y, text, size=12, weight="normal", color=None,
             ha="center", va="center", **kwargs):
    """统一字体设置，便于修改整张图的样式。"""
    return ax.text(
        x, y, text, fontsize=size, fontweight=weight,
        color=color or COLORS["ink"], ha=ha, va=va,
        linespacing=1.3, zorder=20, **kwargs,
    )


def sample_time_frequency_image():
    """人为合成的非负示意矩阵，不是论文数据，也没有执行 CWT。"""
    scales, time_points, _ = INPUT_SHAPE
    t = np.linspace(0.0, 1.0, time_points)[None, :]
    f = np.linspace(0.0, 1.0, scales)[:, None]
    ridge = np.exp(-((t - (0.12 + 0.45 * (1 - f))) / 0.075) ** 2)
    burst = 0.95 * np.exp(-((t - 0.18) / 0.12) ** 2 - ((f - 0.62) / 0.36) ** 2)
    late = 0.28 * np.exp(-((t - 0.68) / 0.19) ** 2 - ((f - 0.32) / 0.23) ** 2)
    texture = 0.07 * (1 + np.sin(22 * t + 7 * f))
    return 0.10 + ridge * (0.25 + 0.65 * f) + burst + late + texture


def draw_input(ax, image_path=None):
    x, y, w, h = 0.55, 2.50, 3.10, 1.65
    if image_path is None:
        data = sample_time_frequency_image()
        image_options = dict(cmap="viridis", origin="lower")
        note = "Illustrative input (synthetic)"
    else:
        data = plt.imread(image_path)
        image_options = dict(origin="upper")
        note = "User-supplied input image"
    ax.imshow(data, extent=(x, x + w, y, y + h), aspect="auto",
              interpolation="bilinear", zorder=1, **image_options)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                           edgecolor="#a8b2b9", linewidth=0.8, zorder=2))
    put_text(ax, x + w / 2, 5.15,
             "Input CWT coefficients\n16 × 68 × 256", 12.8, "bold")
    put_text(ax, x + w / 2, 2.22, note, 9, color=COLORS["muted"])
    return draw_patch(ax, x + 0.56 * w, y + 0.27 * h, 0.43, 0.32)


def draw_patch(ax, x, y, w=0.28, h=0.23):
    """黄色区域只表示局部连接，不按精确卷积感受野比例绘制。"""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=COLORS["patch"],
                           edgecolor="#a89300", linewidth=0.8, zorder=9))
    return x, y, w, h


def draw_feature_stack(ax, x, y, w, h, sheets, dx=0.14, dy=-0.12):
    """只画少数叠片表示通道，实际通道数见标签。"""
    base_rgb = np.array(mcolors.to_rgb(COLORS["feature"]))
    for k in range(sheets):
        white_fraction = 0.42 - 0.27 * k / max(1, sheets - 1)
        face = base_rgb * (1 - white_fraction) + white_fraction
        ax.add_patch(Rectangle(
            (x + k * dx, y + k * dy), w, h,
            facecolor=face, edgecolor=COLORS["feature_edge"],
            linewidth=0.85, alpha=0.88, zorder=3 + k * 0.1,
        ))
    front_x, front_y = x + (sheets - 1) * dx, y + (sheets - 1) * dy
    patch = draw_patch(ax, front_x + w * 0.55, front_y + h * 0.27,
                       min(0.30, w * 0.27), min(0.26, h * 0.31))
    center_x = (x + front_x + w) / 2
    return patch, center_x


def connect_patches(ax, source, target):
    """用两条细线提示相邻层之间的局部映射。"""
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    for source_y, target_y in ((sy, ty), (sy + sh, ty + th)):
        ax.plot([sx + sw, tx], [source_y, target_y],
                color=COLORS["ink"], linewidth=0.85, alpha=0.85, zorder=10)


def operation_label(ax, center_x, patch, identifier, operation):
    put_text(ax, center_x, 1.53, identifier, 10.5, "bold")
    put_text(ax, center_x, 0.89, operation, 11, "bold")
    px, py, pw, _ = patch
    ax.add_patch(FancyArrowPatch(
        (center_x + 0.15, 1.76), (px + pw / 2, py - 0.10),
        arrowstyle="-|>", mutation_scale=9, linewidth=0.85,
        color=COLORS["ink"], zorder=12,
    ))


def draw_neuron_column(ax, x, y, w, h, count, output=False):
    """count 是用于示意的可见节点数，实际单元数单独标注。"""
    fill = COLORS["output"] if output else COLORS["dense"]
    edge = COLORS["output_edge"] if output else COLORS["dense_edge"]
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor=edge, linewidth=1.1, zorder=5))
    node_y = np.linspace(y + 0.16, y + h - 0.16, count)
    points = [(x + w / 2, float(yi)) for yi in node_y]
    for px, py in points:
        ax.add_patch(Circle((px, py), 0.064 if output else 0.042,
                            facecolor=fill if output else COLORS["ink"],
                            edgecolor=edge if output else COLORS["ink"],
                            linewidth=0.8, zorder=8))
    return points


def dense_connections(ax, sources, targets, color, alpha=0.45):
    for sx, sy in sources:
        for tx, ty in targets:
            ax.plot([sx, tx], [sy, ty], color=color,
                    linewidth=0.65, alpha=alpha, zorder=4)


def draw_figure(input_image=None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",  # SVG 文字保留为可编辑文字
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.015, right=0.99, top=0.97, bottom=0.035)
    ax.set(xlim=(0, 26.8), ylim=(-0.45, 6.1))
    ax.axis("off")
    put_text(ax, 13.4, 5.83, TITLE, 17, "bold")
    previous_patch = draw_input(ax, input_image)

    # x、y、宽、高、示意叠片数、位移x、位移y；修改这些数即可调整排版。
    geometry = [
        (4.65, 3.25, 1.45, 1.15, 6, 0.14, -0.12),
        (7.95, 3.00, 1.18, 0.88, 5, 0.14, -0.12),
        (11.05, 3.10, 1.18, 0.88, 7, 0.13, -0.10),
        (14.35, 3.00, 0.90, 0.68, 6, 0.13, -0.10),
    ]
    operations = [
        ("C1", "Convolution\n3 × 3 · ReLU"),
        ("S2", "Max pooling\n2 × 2"),
        ("C3", "Convolution\n3 × 3 · ReLU"),
        ("S4", "Max pooling\n2 × 2"),
    ]
    for layout, shape, (identifier, operation) in zip(
            geometry, FEATURE_SHAPES, operations):
        patch, center_x = draw_feature_stack(ax, *layout)
        height, width, channels = shape
        put_text(ax, center_x, 5.15,
                 f"{channels} feature maps\n{height} × {width}", 12.8, "bold")
        operation_label(ax, center_x, patch, identifier, operation)
        connect_patches(ax, previous_patch, patch)
        previous_patch = patch

    # Flatten 单独画出；4 × 17 × 256 = 17408，不是 512。
    flattened_size = int(np.prod(FEATURE_SHAPES[-1]))
    fx, fy, fw, fh = 17.25, 2.05, 0.16, 2.85
    ax.add_patch(Rectangle((fx, fy), fw, fh, facecolor="#b1a1ec",
                           edgecolor=COLORS["feature_edge"], linewidth=0.8, zorder=5))
    for yy in np.linspace(fy + 0.12, fy + fh - 0.12, 22):
        ax.plot([fx, fx + fw], [yy, yy], color=COLORS["feature_edge"],
                linewidth=0.55, zorder=6)
    px, py, pw, ph = previous_patch
    ax.plot([px + pw, fx], [py, fy], color=COLORS["ink"], linewidth=0.9, zorder=4)
    ax.plot([px + pw, fx], [py + ph, fy + fh], color=COLORS["ink"],
            linewidth=0.9, zorder=4)
    put_text(ax, fx + fw / 2, 5.15, f"Flatten\n{flattened_size:,}", 11.2, "bold")

    layer_512 = draw_neuron_column(ax, 18.85, 1.95, 0.38, 3.15, 8)
    layer_128 = draw_neuron_column(ax, 21.25, 2.42, 0.38, 2.22, 6)
    outputs = draw_neuron_column(ax, 23.75, 3.00, 0.39, 0.88, 2, output=True)
    flatten_points = [(fx + fw, float(y)) for y in np.linspace(fy + 0.12, fy + fh - 0.12, 7)]
    dense_connections(ax, flatten_points, layer_512, COLORS["ink"], alpha=0.20)
    dense_connections(ax, layer_512, layer_128, COLORS["ink"], alpha=0.55)
    dense_connections(ax, layer_128, outputs, COLORS["output_connection"], alpha=0.83)
    put_text(ax, 19.04, 1.15, f"Dense\n{DENSE_UNITS[0]} units", 12, "bold")
    put_text(ax, 21.44, 1.15, f"Dense\n{DENSE_UNITS[1]} units", 12, "bold")
    put_text(ax, 23.95, 4.25, "Sigmoid", 12, "bold")
    for point, label in zip(reversed(outputs), ["Ischemic", "Haemorrhagic"]):
        ax.plot([24.18, 24.44], [point[1], point[1]],
                color=COLORS["ink"], linewidth=0.85)
        put_text(ax, 24.56, point[1], label, 11.3, ha="left")

    put_text(ax, 13.4, -0.16,
             "Dimensions: scales × time × channels.  "
             "Only representative feature maps, neurons and connections are shown.",
             9.5, color=COLORS["muted"])
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-image", type=Path,
                        help="替换左侧示意图，例如自己的 CWT scalogram PNG/JPG")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figures",
                        help="输出目录，默认是脚本旁的 figures 文件夹")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG 分辨率，默认 240")
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi 必须为正整数")
    if args.input_image is not None and not args.input_image.is_file():
        parser.error(f"找不到输入图片：{args.input_image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = draw_figure(args.input_image)
    for extension in ("png", "svg", "pdf"):
        target = args.output_dir / f"cnn_architecture.{extension}"
        figure.savefig(target, dpi=args.dpi, facecolor="white", bbox_inches="tight",
                       pad_inches=0.16)
        print(f"Saved: {target.resolve()}")
    plt.close(figure)
    print("Done. This script only draws the architecture; it does not train a CNN.")


if __name__ == "__main__":
    main()
