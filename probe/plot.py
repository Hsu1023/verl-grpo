import matplotlib.pyplot as plt
import numpy as np

datasets = ["MATH500","DAPO17k"]
trace = [0.1232, 0.5607]
head  = [0.7700, 0.7880]

# x = np.arange(len(datasets))
x = np.array([0.3, 0.7])
w = 0.12  # thinner bars

fig, ax = plt.subplots(figsize=(4, 4))

# blue = "tab:blue"
# orange = "tab:orange"

bars_head  = ax.bar(x + w/2, head,  width=w, label="Head",  alpha=0.6)
bars_trace = ax.bar(x - w/2, trace, width=w, label="Group", alpha=0.6)

# ax.set_title("Spearman Correlation (Spearmanr)")
# ax.set_xlabel("Dataset")
ax.set_ylabel("Spearman Correlation", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=16)
ax.set_ylim(0, 1.0)
ax.grid(True, axis="y", alpha=0.3)
# ax.legend(fontsize=16)
plt.yticks(fontsize=14)

# Clean up spines a bit (similar "simple" look)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Value labels
def label_bars(bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.4f}",
                ha="center", va="bottom", fontsize=10)

# label_bars(bars_trace)
# label_bars(bars_head)

plt.tight_layout()
# plt.show()
plt.savefig("dapo17k_math500_spearmanr_probe_conf.png", dpi=300)
