# Plot the curve starting at 0 with x spaced by 20, showing tick labels every 20
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('--folder', type=str, default=None, help='The folder containing train.log')
args = parse.parse_args()
if args.folder is not None:
    folder_path = args.folder
else:
    folder_path = '/u/haoboxu/work/verl/checkpoints/qwen3-4b_base_grpo_1e-6_math4_16k_dapo'
path = f'{folder_path}/train.log'


datasets = ['grpo_aime2024', 'grpo_gsm8k', 'grpo_amc23', 'grpo_olympiadbench', 'grpo_math500', 'grpo_minervamath', 'grpo_aime2025', 'early_stop', 'advantages']
with open(path, 'r') as f:
    lines = f.readlines()
for content in ['rewards'] + datasets + ['len', 'clip_ratio', 'actor/pg_clipfrac', 'actor/ppo_kl', 'actor/kl_loss', 'actor/pg_loss']:
    matches = {}
    for line in lines:
        if content == 'rewards':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?critic/rewards/mean:([-+]?\d*\.\d+|\d+)'
            match = re.search(pattern, line)
        elif content == 'len':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?response_length_non_aborted/mean:([-+]?\d*\.\d+|\d+)'
            match = re.search(pattern, line)
        elif content == 'clip_ratio':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?response_length/clip_ratio:([-+]?\d*\.\d+|\d+)'
            match = re.search(pattern, line)
        elif content == 'actor/pg_clipfrac':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?actor/pg_clipfrac:np\.float64\(([-+]?\d*\.\d+|\d+)\)'
            match = re.search(pattern, line)
        elif content == 'actor/ppo_kl':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?actor/ppo_kl:np\.float64\(([-+]?\d*\.\d+|\d+)\)'
            match = re.search(pattern, line)
        elif content == 'actor/kl_loss':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?actor/kl_loss:np\.float64\(([-+]?\d*\.\d+|\d+)\)'
            match = re.search(pattern, line)
        elif content == 'actor/pg_loss':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?actor/pg_loss:np\.float64\(([-+]?\d*\.\d+|\d+)\)'
            match = re.search(pattern, line)
        elif 'early_stop' in content:
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?early_stop/ratio:([-+]?\d*\.\d+|\d+)'
            match = re.search(pattern, line)
        elif 'advantages' in content:
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?critic/advantages/mean:([-+]?\d*\.\d+|\d+)'
            match = re.search(pattern, line)
        else:
            pattern = rf'step:([-+]?\d*\.\d+|\d+).*?val-core/{content}/acc/mean@1:np\.float64\(([-+]?\d*\.\d+|\d+)\)'
            match = re.search(pattern, line)

        if match:
            value = float(match.group(2))
            # assert not (match.group(1) in matches and 'grpo_' in content), f"Duplicate step {match.group(1)} for {content}"
            matches[match.group(1)] = value
    if len(matches) == 0:
        continue
    # y = np.array(matches)

    # if content not in ['rewards', 'len']:
    #     x = np.arange(len(y)) * 20
    # else:
    #     x = np.arange(len(y)) + 1
    y = np.array([v for k, v in sorted(matches.items(), key=lambda item: int(item[0]))])
    x = np.array([int(k) for k, v in sorted(matches.items(), key=lambda item: int(item[0]))])

    plt.figure(figsize=(8, 4.8))
    plt.plot(x, y, marker='o')
    # if content not in ['rewards', 'len']:
    #     plt.xticks(np.arange(0, x[-1] + 20, 20))
    # else:
    #     plt.xticks(np.arange(0, x[-1] + 1, 20))
    plt.xlabel("Steps")
    plt.grid(True, axis='y')
    # title
    if content == 'clip_ratio':
        plt.ylabel("Truncation Ratio over Steps")
    else:
        plt.title(f"{content} over Steps")

    # if content == 'rewards':
    #     out_path = f"{folder_path}/rewards.png"
    # elif content == 'len':
    #     out_path = f"{folder_path}/len.png"
    if content == 'clip_ratio':
        out_path = f"{folder_path}/truncation_ratio.png"
    elif content == 'early_stop':
        out_path = f"{folder_path}/early_stop_ratio.png"
        plt.xlim(0, 500)
    else:
        out_path = f"{folder_path}/{content.split('/')[-1]}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    

import sys
from typing import List, Optional, Tuple

from PIL import Image


def cover_resize_and_crop(im: Image.Image, target_size: Tuple[int,int]) -> Image.Image:
    """等比缩放（cover）并中心裁切到 target_size（width, height）。"""
    tw, th = target_size
    iw, ih = im.size
    if iw == tw and ih == th:
        return im.copy()
    # scale so that resized image fully covers target (可能会裁切)
    scale = max(tw / iw, th / ih)
    new_w = max(1, int(iw * scale + 0.5))
    new_h = max(1, int(ih * scale + 0.5))
    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - tw) // 2
    top  = (new_h - th) // 2
    return im_resized.crop((left, top, left + tw, top + th))

def make_collage(image_paths: List[str], out_path: str,
                 tile_size: Optional[Tuple[int,int]] = None) -> None:
    """
    将 9 张图片合成 3x3 九宫格并保存为 out_path。
    - image_paths: 长度必须是 9 的列表（按从左到右、从上到下的顺序）。
    - tile_size: (w,h)；如果为 None，自动取所有图片宽度/高度的最小值以避免放大。
    """
    if len(image_paths) != 9:
        raise ValueError("需要正好 9 张图片（image_paths 列表长度为 9）")

    imgs = [Image.open(p) for p in image_paths]

    # 决定tile大小
    if tile_size is None:
        widths = [im.width for im in imgs]
        heights = [im.height for im in imgs]
        tile_w = min(widths)
        tile_h = min(heights)
    else:
        tile_w, tile_h = tile_size

    # 如果 tile 大小为 0 或 非正，报错
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("tile_size 必须为正整数")

    # 判断是否需要透明通道（若任意图片有 alpha）
    need_alpha = any(im.mode in ("RGBA", "LA") or ("transparency" in im.info) for im in imgs)
    canvas_mode = "RGBA" if need_alpha else "RGB"

    out_w = tile_w * 3
    out_h = tile_h * 3
    canvas = Image.new(canvas_mode, (out_w, out_h), (255,255,255,0) if need_alpha else (255,255,255))

    # 处理并粘贴每张图片
    for idx, im in enumerate(imgs):
        im_conv = im.convert("RGBA") if need_alpha else im.convert("RGB")
        tile = cover_resize_and_crop(im_conv, (tile_w, tile_h))
        row = idx // 3
        col = idx % 3
        x = col * tile_w
        y = row * tile_h
        canvas.paste(tile, (x, y), tile if need_alpha else None)

    # 保存
    canvas.save(out_path)
    print(f"Saved collage to {out_path} ({out_w}x{out_h})")

# folder_path = '/home/yichen/verl/checkpoints/qwen3-1.7b_grpo_1e-6_math4'
# imgs = ['rewards.png','len.png', 'grpo_aime2024.png', 'grpo_aime2025.png','grpo_amc23.png', 'grpo_gsm8k.png', 'grpo_math500.png', 'grpo_minervamath.png', 'grpo_olympiadbench.png']
imgs = ['rewards.png','len.png', 'early_stop_ratio.png', 'advantages.png', 'kl_loss.png', 'pg_loss.png', 'grpo_math500.png', 'grpo_minervamath.png', 'grpo_olympiadbench.png']
imgs = [f'{folder_path}/{img}' for img in imgs]
out = f'{folder_path}/summary.png'
make_collage(imgs, out)
