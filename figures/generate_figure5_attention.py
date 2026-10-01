"""
Figure 5: Share of attention on image tokens by layer for the 9B models trained on
Subsets 2 and 3 (OOD, bf16).

Input: evaluation/visual_attention_mass/9B_v2_22.8k.jsonl and 9B_v3_27.5k.jsonl,
written by visual_attention_mass.py (one record per OOD line, per-layer img_mass).
(a) Mean share per attention layer with 95% bootstrap CI (lines resampled).
(b) Paired difference (Subset 3 minus Subset 2) per layer with 95% bootstrap CI.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE) if os.path.basename(HERE) == "figures" else HERE
DATA = os.path.join(REPO, "evaluation", "visual_attention_mass")
FILES = {"Subset 2 (22.8k)": "9B_v2_22.8k.jsonl", "Subset 3 (27.5k)": "9B_v3_27.5k.jsonl"}
B, SEED = 2000, 20260929

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Subset 2 (22.8k)": "#2a78d6", "Subset 3 (27.5k)": "#eb6834"}

def load(path):
    recs = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if "img_mass" in r and "error" not in r:
                    recs[r["id"]] = r
    return recs

data = {k: load(os.path.join(DATA, f)) for k, f in FILES.items()}
k2, k3 = list(FILES)
ids = sorted(set(data[k2]) & set(data[k3]))
layers = data[k2][ids[0]]["layers"]
M = {k: np.array([data[k][i]["img_mass"] for i in ids]) for k in FILES}   # (lines, layers)
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(ids), size=(B, len(ids)))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "svg.fonttype": "path",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300,
                             gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(layers))
labels = [str(l) for l in layers]

for k in FILES:
    mean = M[k].mean(0)
    lo, hi = np.percentile(M[k][idx].mean(1), [2.5, 97.5], axis=0)
    a1.fill_between(x, lo, hi, color=COLORS[k], alpha=0.18, linewidth=0)
    a1.plot(x, mean, color=COLORS[k], linewidth=2, marker="o", markersize=4,
            markeredgecolor="white", markeredgewidth=1, label=k)
a1.set_ylabel("Share of attention on image tokens")
a1.legend(frameon=False, loc="lower center", fontsize=7.5)

D = (M[k3] - M[k2]) * 100          # percentage points
dmean = D.mean(0)
dlo, dhi = np.percentile(D[idx].mean(1), [2.5, 97.5], axis=0)
a2.axhline(0, color=MUTED, linewidth=0.8)
a2.errorbar(x, dmean, yerr=[dmean - dlo, dhi - dmean], fmt="o", color=COLORS[k3],
            markersize=4, markeredgecolor="white", markeredgewidth=1,
            ecolor=COLORS[k3], elinewidth=1.2, capsize=2)
a2.set_ylabel("Subset 3 minus Subset 2 (points)")

for ax, tag in ((a1, "(a) Mean share per layer"), (a2, "(b) Paired difference")):
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Decoder layer with explicit attention")
    ax.set_xlim(-0.4, len(layers) - 0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)

fig.tight_layout(w_pad=2.5)
os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(REPO, "figures", f"figure5_attention_by_layer.{ext}"), bbox_inches="tight")
plt.close(fig)
print(f"Lines: {len(ids)}, layers: {layers}")
print("Mean difference over all layers (points):", round(float(D.mean()), 2))
print("Saved figures/figure5_attention_by_layer.png/.pdf/.svg")
