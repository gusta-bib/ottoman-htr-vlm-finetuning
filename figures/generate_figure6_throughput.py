"""
Figure 6: Inference throughput vs. Normalized CER (in-distribution set).

(a) Qwen3.5-2B, 4B and 9B, High HP, Subset 3 (Table 5 and Table 8).
(b) Qwen3.5-4B on Subset 2: High HP, the three single-factor ablations and the
    Low HP reference (Table 6 and Section 4.5).
Throughput: vLLM, GPTQ W4A16, single NVIDIA GeForce RTX 4070 Ti SUPER (16 GB).
Source files: results/eval/*benchmark_matrix*.csv and *_metrics.json.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# (label, throughput lines/s, ID Normalized CER %)
PANEL_A = [("Qwen3.5-2B", 24.37, 14.67), ("Qwen3.5-4B", 15.86, 13.48), ("Qwen3.5-9B", 6.90, 13.14)]
PANEL_B = [("High HP", 15.82, 13.91), ("Lower image limit", 15.84, 14.57),
           ("Lower rank", 15.86, 14.96), ("Lower learning rate", 15.81, 19.77),
           ("Low HP (reference)", 15.69, 24.86)]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GRAY = "#2a78d6", "#8c8b86"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "svg.fonttype": "path",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=300)

def style(ax, tag):
    ax.grid(axis="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Throughput (lines per second)")
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, ha="left", va="bottom")

# (a) model size
xs = [p[1] for p in PANEL_A]; ys = [p[2] for p in PANEL_A]
a1.plot(xs, ys, color=BLUE, linewidth=1.2, alpha=0.5, zorder=2)
a1.scatter(xs, ys, s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
label_pos = {"Qwen3.5-9B": (6.90, 12.45), "Qwen3.5-4B": (15.86, 12.80), "Qwen3.5-2B": (24.37, 15.05)}
for lab, x, y in PANEL_A:
    tx, ty = label_pos[lab]
    a1.text(tx, ty, f"{lab}\n{y:.2f}%, {x:.2f} l/s", ha="center", va="center", fontsize=7.5, color=MUTED)
a1.set_xlim(0, 28); a1.set_ylim(12.0, 15.8)
a1.set_ylabel("ID Normalized CER (%)")
style(a1, "(a) Model size, Subset 3")

# (b) 4B configurations on Subset 2
label_y = {"High HP": 13.2, "Lower image limit": 15.2, "Lower rank": 17.2, "Lower learning rate": 19.77, "Low HP (reference)": 24.86}
for lab, x, y in PANEL_B:
    c = GRAY if lab.startswith("Low HP") else BLUE
    a2.scatter([x], [y], s=46, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    a2.annotate(f"{lab}: {y:.2f}%, {x:.2f} l/s", (x, y), xytext=(19.0, label_y[lab]),
                textcoords="data", ha="left", va="center", fontsize=7.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
a2.set_xlim(0, 28); a2.set_ylim(12.0, 27.0)
style(a2, "(b) Qwen3.5-4B configurations, Subset 2")

fig.tight_layout(w_pad=2.0)
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"figure6_throughput_vs_cer.{ext}"), bbox_inches="tight")
plt.close(fig)
print("Saved figure6_throughput_vs_cer.png/.pdf/.svg to", OUT)
