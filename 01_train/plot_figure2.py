"""Plot Figure 2 (right): reverse accuracy on unseen father/mother relations
as a function of weight decay, for all 27 models (9 weight decays x 3 seeds).

Run collect_results.py first to build ../results/training/aggregated_results.json.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

THRESHOLD_LOW = 0.4    # below: "Reversal Cursed"
THRESHOLD_HIGH = 0.98  # above: "Not Reversal Cursed"

df = pd.read_json("../results/training/aggregated_results.json")

sns.set_theme(style="ticks")
plt.figure(figsize=(4, 4))

groups = [
    ("Reversal Cursed", df["reverse_uni_acc"] < THRESHOLD_LOW, "C0"),
    ("Not Reversal Cursed", df["reverse_uni_acc"] > THRESHOLD_HIGH, "C1"),
    (None, df["reverse_uni_acc"].between(THRESHOLD_LOW, THRESHOLD_HIGH), "gray"),
]
for label, mask, color in groups:
    sub = df[mask]
    if len(sub):
        plt.scatter(sub["weight_decay"], sub["reverse_uni_acc"], color=color, s=50,
                    edgecolors="black", linewidth=0.5, label=label)

plt.xlabel("Weight Decay")
plt.ylabel("Reverse Acc. (mother/father)")
plt.ylim(-0.05, 1.05)
plt.legend(loc="lower right", frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig("fig2_right_reverse_acc_vs_wd.pdf", bbox_inches="tight")
print("Saved fig2_right_reverse_acc_vs_wd.pdf")
