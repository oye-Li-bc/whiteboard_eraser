"""生成"白板诗句贴图"及其板面布局标注。

做三件事：
  1) 选一首诗(内置随机 / 指定 index / 外部 lines)，用 make_typo 制造一个错别字；
  2) 用中文字体把整首诗横排渲染成 PNG 纹理(白底黑字)，供 Gazebo 白板贴图；
  3) 记录每个字在板面(y,z)的包围盒 + 错字信息，写 poem_layout.yaml。

坐标约定(与 scene.yaml 一致)：世界系 base_link，x 向前(朝板)，z 向上。
朝板看时 +y 在左手边，故：图像列 px_x 越大 → 板面 y 越小；图像行 px_y 越大 → z 越小。
poem_layout.yaml 里的 correct_lines 是 detect 节点比对用的"标准答案"；chars 的板面
bbox 仅作 ground-truth 校验/可视化，detect 节点定位错字靠 OCR 反投影，不直接读这里的坐标。
"""
import os
import argparse
import yaml
from PIL import Image, ImageDraw, ImageFont

from whiteboard_eraser.poems import load_poem, make_typo

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def pick_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("未找到可用中文字体，请安装 fonts-noto-cjk")


def render(poem_lines, board, out_png, px_per_m=2000, margin_frac=0.08,
           char_pitch_frac=1.10, line_pitch_frac=1.95):
    """把 poem_lines 横排渲染为 PNG，返回每个字的像素/板面 bbox 列表。

    关键：行内字距紧凑(char_pitch≈1.1×字号)、行间距拉大(≈1.95×字号)，使 OCR 检测器
    把每一句聚类成一条水平文本行，而不是按竖列聚类。
    """
    W = int(board["width_y"] * px_per_m)
    H = int(board["height_z"] * px_per_m)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    n_rows = len(poem_lines)
    max_cols = max(len(l) for l in poem_lines)
    usable_w = W * (1 - 2 * margin_frac)
    usable_h = H * (1 - 2 * margin_frac)
    # 字号同时受"最长行宽"与"总行高"约束
    font_by_w = usable_w / (max_cols * char_pitch_frac)
    font_by_h = usable_h / ((n_rows - 1) * line_pitch_frac + 1) if n_rows > 1 else usable_h
    font_size = int(min(font_by_w, font_by_h))
    font = pick_font(font_size)
    char_pitch = font_size * char_pitch_frac
    line_pitch = font_size * line_pitch_frac

    cy, cz = board["center_y"], board["center_z"]
    wy, hz = board["width_y"], board["height_z"]

    total_h = (n_rows - 1) * line_pitch + font_size
    top0 = (H - total_h) / 2.0

    chars = []
    for r, line in enumerate(poem_lines):
        ncol = len(line)
        row_w = (ncol - 1) * char_pitch + font_size
        x0 = (W - row_w) / 2.0
        cell_top = top0 + r * line_pitch
        for c, ch in enumerate(line):
            cell_left = x0 + c * char_pitch
            bbox = font.getbbox(ch)
            gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = cell_left + (font_size - gw) / 2.0 - bbox[0]
            ty = cell_top + (font_size - gh) / 2.0 - bbox[1]
            draw.text((tx, ty), ch, font=font, fill=(0, 0, 0))
            # 字格中心 → 板面 (y,z)
            px_cx = cell_left + font_size / 2.0
            px_cy = cell_top + font_size / 2.0
            y_world = cy + wy / 2.0 - (px_cx / W) * wy
            z_world = cz + hz / 2.0 - (px_cy / H) * hz
            chars.append({
                "line": r, "col": c, "ch": ch,
                "y": round(float(y_world), 4), "z": round(float(z_world), 4),
                "w": round(char_pitch / W * wy, 4), "h": round(font_size / H * hz, 4),
            })
    img.save(out_png)
    return chars


def write_material(scripts_dir, tex_name, mat_name):
    os.makedirs(scripts_dir, exist_ok=True)
    mat = f"""material {mat_name}
{{
  technique
  {{
    pass
    {{
      texture_unit
      {{
        texture {tex_name}
        filtering trilinear
      }}
    }}
  }}
}}
"""
    with open(os.path.join(scripts_dir, "poem.material"), "w") as f:
        f.write(mat)


def generate(scene_path, out_dir, layout_path, index=None, seed=None, lines=None):
    """生成诗句纹理 + material 脚本 + poem_layout.yaml。返回 layout dict。

    out_dir: models/ 的父目录(源目录或 install 的 share 目录)。
    纹理写到 out_dir/models/materials/textures/poem.png。
    """
    with open(scene_path) as f:
        board = yaml.safe_load(f)["board"]

    poem = load_poem(index=index, lines=lines)
    typo = make_typo(poem, seed=seed)
    wrong_lines = typo["wrong_lines"]

    tex_dir = os.path.join(out_dir, "models", "materials", "textures")
    scripts_dir = os.path.join(out_dir, "models", "materials", "scripts")
    os.makedirs(tex_dir, exist_ok=True)
    out_png = os.path.join(tex_dir, "poem.png")
    chars = render(wrong_lines, board, out_png)
    write_material(scripts_dir, "poem.png", "WhiteboardPoem/Poem")

    layout = {
        "title": poem["title"], "author": poem["author"],
        "correct_lines": poem["lines"],
        "wrong_lines": wrong_lines,
        "typo": {"line": typo["line"], "col": typo["col"],
                 "correct": typo["correct"], "wrong": typo["wrong"]},
        "chars": chars,
    }
    with open(layout_path, "w") as f:
        yaml.safe_dump(layout, f, allow_unicode=True, sort_keys=False)

    print(f"[gen_poem] 《{poem['title']}》 错字: "
          f"'{typo['correct']}'->'{typo['wrong']}' @行{typo['line']}列{typo['col']}")
    print(f"[gen_poem] 纹理: {out_png}")
    print(f"[gen_poem] 布局: {layout_path}")
    return layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scene.yaml 路径(取板尺寸)")
    ap.add_argument("--out-dir", required=True, help="包 share 或源目录(models/ 的父目录)")
    ap.add_argument("--layout", required=True, help="poem_layout.yaml 输出路径")
    ap.add_argument("--index", type=int, default=None, help="内置诗序号，默认随机")
    ap.add_argument("--seed", type=int, default=None, help="错字随机种子，默认随机")
    ap.add_argument("--lines", nargs="*", default=None, help="自定义诗句(逐句)，覆盖 index")
    args = ap.parse_args()
    generate(args.scene, args.out_dir, args.layout,
             index=args.index, seed=args.seed, lines=args.lines)


if __name__ == "__main__":
    main()
