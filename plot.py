# Plot the curve starting at 0 with x spaced by 20, showing tick labels every 20
import matplotlib.pyplot as plt
import numpy as np

import re
folder_path = '/home/yichen/verl/checkpoints/qwen2.5-3b_dapo17k_grpo_1e-6_math'
path = f'{folder_path}/train.log'

datasets = ['grpo_aime2024', 'grpo_gsm8k', 'grpo_amc23', 'grpo_olympiadbench', 'grpo_math500', 'grpo_minervamath', 'grpo_aime2025']
with open(path, 'r') as f:
    lines = f.readlines()
for content in ['rewards'] + datasets + ['len', 'clip_ratio', 'actor/pg_clipfrac', 'actor/ppo_kl']:
    matches = {}
    for line in lines:
        if content == 'rewards':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?critic/rewards/mean:([-+]?\d*\.\d+|\d+)'
            match = re.search(pattern, line)
        elif content == 'len':
            pattern = r'step:([-+]?\d*\.\d+|\d+).*?response_length/mean:([-+]?\d*\.\d+|\d+)'
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
        else:
            pattern = rf'step:([-+]?\d*\.\d+|\d+).*?val-core/{content}/acc/mean@1:np\.float64\(([-+]?\d*\.\d+|\d+)\)'
            match = re.search(pattern, line)

        if match:
            value = float(match.group(2))
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
    else:
        out_path = f"{folder_path}/{content.split('/')[-1]}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()