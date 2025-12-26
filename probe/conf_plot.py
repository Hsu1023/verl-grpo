import json
from tqdm import trange
import random

# data = json.load(open('/u/haoboxu/work/verl/conf_results_qwen3-4b_dapo17k_traces10_.json', 'r'))
# print(len(data))

# cor = [d['correctness'][0] for d in data]
# cof = [d['min_confs']['128'] for d in data]
# print(cor[0], cof[0])
# cor, cof = [], []
# for d in data:
#     if len(d['correctness']) == 0:
#         continue
#     cor.append(d['correctness'][-1])
#     cof.append(d['min_confs'])



import pickle
# pickle.dump([cor, cof], open('dapo17k_deepconf.pkl', 'wb'))
cor, coff = pickle.load(open('dapo17k_deepconf.pkl', 'rb'))
cof = [d['1024'] for d in coff]
# for j in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
#     cof = [d[str(j)] for d in coff]
#     pairs = [(a, b) for (a, b) in zip(cor, cof)]
#     random.shuffle(pairs)
#     pairs = list(zip(cor, cof))

#     pos = [p for p in pairs if int(p[0]) == 1]
#     neg = [p for p in pairs if int(p[0]) == 0]

#     k = min(len(pos), len(neg))
#     pairs = pos[:k] + neg[:k]   # 50/50
#     cor = [p[0] for p in pairs]
#     cof = [p[1] for p in pairs]
#     # print(cor, cof)
#     best_ans = 0
#     best_thres = None
#     for i in trange(0, 2500 * 5):
#         thres = float(i) / 100 / 5
#         # print(cof[0], cor[0])
#         ans = [int(c > thres) == int(_) for c, _ in zip(cof, cor)]
#         # print(ans)
#         if sum(ans) / len(ans) > best_ans:
#             best_ans = sum(ans) / len(ans)
#             best_thres = thres
#     print(j, best_ans, best_thres)
    
import numpy as np
import matplotlib.pyplot as plt

# 你的数据
# pos = [...]
# neg = [...]
pos = [cof[i] for i in range(len(cor)) if int(cor[i]) == 1]
neg = [cof[i] for i in range(len(cor)) if int(cor[i]) == 0]

pos = np.asarray(pos)
neg = np.asarray(neg)

# 统一 bins：用两组数据的整体范围来切
bins = 60
x_min = min(pos.min(), neg.min())
x_max = max(pos.max(), neg.max())
bin_edges = np.linspace(x_min, x_max, bins + 1)

pos_mean = pos.mean()
neg_mean = neg.mean()

plt.figure(figsize=(6, 4.5))

# 叠加直方图（frequency 计数）
n1, b1, p1 = plt.hist(pos, bins=bin_edges, alpha=0.6, label="Correct", density=True)      # 颜色用默认即可
n2, b2, p2 = plt.hist(neg, bins=bin_edges, alpha=0.6, label="Incorrect", density=True)

# pos_color = p1[0].get_facecolor()
# neg_color = p2[0].get_facecolor()
pos_color = "tab:blue"
neg_color = "tab:orange"
# 均值竖线（虚线）
plt.axvline(pos_mean, color=pos_color, linestyle="--", linewidth=2, label="Correct Mean")
plt.axvline(neg_mean, color=neg_color, linestyle="--", linewidth=2, label="Incorrect Mean")
# x, y axis fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.title("Mean Confidence")
plt.xlabel("Score", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.grid(True, alpha=0.3)
# plt.legend(fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig("dapo17k_deepconf_1024_hist.png", dpi=300)