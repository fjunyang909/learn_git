"""读取 data10_27/MNIST/raw 下的 IDX 文件，在终端打印 ASCII 图并生成一张 PNG 拼图。"""
import struct
from pathlib import Path

from PIL import Image, ImageDraw

RAW = Path(__file__).parent / "data10_27" / "MNIST" / "raw"
OUT = Path(__file__).parent / "output" / "mnist_samples.png"


def read_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
    return n, rows, cols, buf


def read_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        buf = f.read(n)
    return buf


def pixel(buf, rows, cols, i):
    start = i * rows * cols
    return buf[start:start + rows * cols], start


def ascii_view(buf, rows, cols, i):
    px, _ = pixel(buf, rows, cols, i)
    ramp = " .:-=+*#%@"
    lines = []
    for r in range(rows):
        line = "".join(
            ramp[min(len(ramp) - 1, px[r * cols + c] * len(ramp) // 256)]
            for c in range(cols)
        )
        lines.append(line)
    return lines


def main():
    n, rows, cols, imgs = read_images(RAW / "train-images-idx3-ubyte")
    labels = read_labels(RAW / "train-labels-idx1-ubyte")
    print(f"训练集: {n} 张, {rows}x{cols}")

    # 每个数字取第一张出现的图
    picks = []
    for d in range(10):
        idx = labels.index(d)
        picks.append((idx, d))

    # 终端 ASCII 展示前 3 张
    for idx, d in picks[:3]:
        print(f"\n第 {idx} 张, 标签 = {d}")
        print("\n".join(ascii_view(imgs, rows, cols, idx)))

    # 生成 PNG 拼图：10 张图一行
    scale, pad, label_h = 4, 8, 22
    cell_w = cols * scale + pad
    cell_h = rows * scale + pad + label_h
    grid = Image.new("L", (cell_w * 5 + pad, cell_h * 2 + pad), 32)
    draw = ImageDraw.Draw(grid)

    for k, (idx, d) in enumerate(picks):
        px, start = pixel(imgs, rows, cols, idx)
        tile = Image.frombytes("L", (cols, rows), px).resize(
            (cols * scale, rows * scale), Image.NEAREST
        )
        x = pad + (k % 5) * cell_w
        y = pad + (k // 5) * cell_h
        grid.paste(tile, (x, y))
        draw.text((x + cols * scale // 2 - 5, y + rows * scale + 4),
                  f"{d}  #{idx}", fill=200)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT)
    print(f"\n已保存拼图: {OUT}")


if __name__ == "__main__":
    main()