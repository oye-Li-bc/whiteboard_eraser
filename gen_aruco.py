#!/usr/bin/env python3
# coding=utf-8
"""生成白板四角用的 ArUco marker,输出可精确打印的 A4 PDF + 单个 PNG。

用法:
  python3 gen_aruco.py                       # 默认 DICT_5X5_100, 60mm, id 0..3
  python3 gen_aruco.py --size-mm 80 --dict DICT_4X4_50

关键:打印时务必"实际大小/100%"(不要"适应页面"),打印后用尺子量一个 marker
边长是否等于设定值(如 6.0cm),否则 solvePnP 出来的深度会整体偏错。
"""
import argparse
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORNER_NAMES = ["左上 TL", "右上 TR", "右下 BR", "左下 BL"]  # id 0,1,2,3 对应角


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="DICT_5X5_100", help="ArUco 字典名")
    ap.add_argument("--size-mm", type=float, default=60.0, help="marker 物理边长(mm)")
    ap.add_argument("--ids", type=int, nargs=4, default=[0, 1, 2, 3], help="四个角的 id")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dict_id = getattr(cv2.aruco, args.dict)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    # 生成每个 marker 的高分辨率位图
    px = 700
    imgs = {}
    for mid in args.ids:
        img = np.zeros((px, px), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, mid, px, img, 1)
        imgs[mid] = img
        png = os.path.join(args.out_dir, f"aruco_{args.dict}_id{mid}.png")
        cv2.imwrite(png, img)

    # 排到 A4 PDF,每个 marker 按精确物理尺寸 (2x2 布局,id 顺序对应四角)
    A4_W, A4_H = 21.0, 29.7  # cm
    s = args.size_mm / 10.0  # cm
    fig = plt.figure(figsize=(A4_W / 2.54, A4_H / 2.54))
    # 2x2 网格中心位置(cm),留出标签空间
    cx = [A4_W * 0.28, A4_W * 0.72]
    cy = [A4_H * 0.70, A4_H * 0.30]
    layout = {args.ids[0]: (cx[0], cy[0]), args.ids[1]: (cx[1], cy[0]),
              args.ids[2]: (cx[1], cy[1]), args.ids[3]: (cx[0], cy[1])}

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W); ax.set_ylim(0, A4_H); ax.axis("off")
    for i, mid in enumerate(args.ids):
        x, y = layout[mid]
        ax.imshow(imgs[mid], cmap="gray", extent=[x - s / 2, x + s / 2, y - s / 2, y + s / 2],
                  interpolation="nearest", zorder=1)
        ax.text(x, y + s / 2 + 0.5, f"id={mid}  {CORNER_NAMES[i]}  ({args.size_mm:.0f}mm)",
                ha="center", va="bottom", fontsize=11)
    # 5cm 标尺供核对打印比例
    rx0 = A4_W * 0.5 - 2.5
    ax.plot([rx0, rx0 + 5.0], [A4_H * 0.5, A4_H * 0.5], "k-", lw=1.5)
    for xx in np.arange(rx0, rx0 + 5.01, 1.0):
        ax.plot([xx, xx], [A4_H * 0.5 - 0.15, A4_H * 0.5 + 0.15], "k-", lw=1.0)
    ax.text(rx0 + 2.5, A4_H * 0.5 - 0.6, "5cm 标尺(打印后量一下核对比例)", ha="center", va="top", fontsize=9)

    pdf = os.path.join(args.out_dir, "aruco_corners_print.pdf")
    fig.savefig(pdf)
    plt.close(fig)

    print(f"字典 {args.dict}, 边长 {args.size_mm}mm, ids {args.ids}")
    print(f"PDF(打印用): {pdf}")
    print(f"单张 PNG: {args.out_dir}/aruco_{args.dict}_id*.png")
    print("四角对应: id0=左上 id1=右上 id2=右下 id3=左下 (顺时针)")


if __name__ == "__main__":
    main()
