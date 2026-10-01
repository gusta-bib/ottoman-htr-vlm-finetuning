"""
Figures 1 to 4 of the IP&M paper, drawn from the values reported in the paper.

Figure 1  Line-level data construction pipeline (Section 3.2).
Figure 2  Data scaling by model size, High HP, 95% bootstrap CIs (Table 6).
Figure 3  Low HP to High HP on the same training data (Table 6).
Figure 4  Single-factor ablations of the High HP configuration and Low HP reference, 4B, Subset 2 (Table 6).
CIs: micro Normalized CER, B = 1,000 line resamples, seed 42
(evaluation/stats/main_results_ci.csv and task1_task2 bootstrap outputs).
Outputs: figures/figure{1..4}_*.png / .pdf / .svg next to this script.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, EDGE, FILL = "#0b0b0b", "#52514e", "#e4e3df", "#8c8b86", "#f4f3f0"
C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "svg.fonttype": "path",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved", name)

def clean(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def tag(ax, text):
    ax.text(0, 1.04, text, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)

# ---------------------------------------------------------------- Figure 1
def figure1():
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(8, 46.5); ax.axis("off")
    def box(x, y, w, h, text, strong=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor="#ffffff" if not strong else "#eaf2fc",
                                    edgecolor=C1 if strong else EDGE, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5, color=INK,
                fontweight="normal", linespacing=1.3)
    def arrow(x1, y1, x2, y2, dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.9,
                                    linestyle="--" if dashed else "-", mutation_scale=8))
    # column 1: images
    box(1, 38, 21, 7, "Register page scans\n(300 DPI)")
    box(1, 24, 21, 7, "Line detection\n(AHDoc)")
    box(1, 10, 21, 7, "Arabic-script drafts\n(Low HP HTR model)")
    arrow(11.5, 38, 11.5, 31.6); arrow(11.5, 24, 11.5, 17.6)
    ax.text(12.5, 20.8, "25,635 lines", ha="left", va="center", fontsize=6.8, color=MUTED)
    # column 2: text and skeletons
    box(27, 38, 22, 7, "Latin page transliterations\n(graduate theses, TTK)")
    box(27, 24, 22, 7, "Unvocalized consonant\nskeletons of both texts")
    arrow(38, 38, 38, 31.6)
    arrow(22.6, 14.5, 31, 23.4)
    # column 3: matching
    box(52.5, 24, 23, 7, "Monotonic sliding-window\nmatching (RapidFuzz)", strong=True)
    arrow(49.6, 27.5, 51.9, 27.5)
    # column 4: outcomes
    box(79, 34, 20, 8, "Score of 70 or higher\n16,829 pairs (65.65%)", strong=True)
    box(79, 14, 20, 8, "Score below 70\nor no match\n8,806 lines (34.35%)")
    arrow(76.1, 29.5, 78.4, 35.5); arrow(76.1, 25.5, 78.4, 20.5, dashed=True)
    save(fig, "figure1_data_pipeline")

# ---------------------------------------------------------------- Figure 2
T4 = {  # model: [(ID norm, ID lo, ID hi, OOD norm, OOD lo, OOD hi) for Subsets 1, 2, 3]
    "Qwen3.5-2B": [(16.63, 15.7, 17.6, 43.92, 42.9, 45.1), (14.94, 14.1, 15.8, 44.43, 42.5, 46.5), (14.67, 13.8, 15.6, 42.92, 41.7, 44.4)],
    "Qwen3.5-4B": [(15.03, 14.1, 16.0, 40.65, 39.7, 41.8), (13.91, 13.0, 14.8, 39.21, 38.4, 40.1), (13.48, 12.6, 14.4, 39.66, 38.8, 40.5)],
    "Qwen3.5-9B": [(14.58, 13.7, 15.5, 40.73, 38.9, 42.7), (13.29, 12.6, 14.1, 37.31, 35.9, 38.9), (13.14, 12.3, 14.0, 40.61, 38.7, 42.6)],
}
COL = {"Qwen3.5-2B": C1, "Qwen3.5-4B": C2, "Qwen3.5-9B": C3}
TIERS = ["Subset 1\n(15.0k)", "Subset 2\n(22.8k)", "Subset 3\n(27.5k)"]

def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=300)
    for ax, off, title, lim in ((axes[0], 0, "(a) In-distribution (815 lines)", (11.5, 18.5)),
                                (axes[1], 3, "(b) Out-of-distribution (1,575 lines)", (34.5, 47.5))):
        for j, (m, rows) in enumerate(T4.items()):
            x = [i + (j - 1) * 0.08 for i in range(3)]
            y = [r[off] for r in rows]
            lo = [r[off] - r[off + 1] for r in rows]; hi = [r[off + 2] - r[off] for r in rows]
            ax.errorbar(x, y, yerr=[lo, hi], color=COL[m], linewidth=1.8, marker="o", markersize=4,
                        markeredgecolor="white", markeredgewidth=1, elinewidth=1, capsize=2, label=m)
        ax.set_xticks(range(3)); ax.set_xticklabels(TIERS)
        ax.set_xlim(-0.35, 2.35); ax.set_ylim(*lim)
        ax.set_ylabel("Normalized CER (%)")
        clean(ax); tag(ax, title)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout(w_pad=2.5)
    save(fig, "figure2_data_scaling")

# ---------------------------------------------------------------- Figure 3
T3 = {  # model (data): (Low ID, High ID, Low OOD, High OOD)
    "Qwen3.5-2B (Subset 2)": (27.56, 14.94, 52.91, 44.43),
    "Qwen3.5-4B (Subset 2)": (24.86, 13.91, 52.72, 39.21),
    "Qwen3.5-9B (Subset 1)": (23.36, 14.58, 50.16, 40.73),
}
COL3 = dict(zip(T3, (C1, C2, C3)))

def figure3():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=300)
    for ax, (i0, i1), title, lim in ((axes[0], (0, 1), "(a) In-distribution", (10, 30)),
                                     (axes[1], (2, 3), "(b) Out-of-distribution", (35, 56))):
        for m, v in T3.items():
            ax.plot([0, 1], [v[i0], v[i1]], color=COL3[m], linewidth=1.8, marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=1, label=m)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Low HP", "High HP"])
        ax.set_xlim(-0.3, 1.3); ax.set_ylim(*lim)
        ax.set_ylabel("Normalized CER (%)")
        clean(ax); tag(ax, title)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout(w_pad=2.5)
    save(fig, "figure3_low_to_high_hp")

# ---------------------------------------------------------------- Figure 4
T5 = [("High HP\nLR 1e-4, r 64\n602,112 px", 13.91, 13.0, 14.8, 39.21, 38.4, 40.1),
      ("Lower learning rate\nLR 3e-5, r 64\n602,112 px", 19.77, 18.75, 20.94, 45.31, 44.12, 46.77),
      ("Lower image limit\nLR 1e-4, r 64\n200,704 px", 14.57, 13.7, 15.5, 44.22, 42.8, 45.8),
      ("Lower rank\nLR 1e-4, r 32\n602,112 px", 14.96, 14.1, 15.9, 41.62, 40.3, 43.0),
      ("Low HP (reference)\nLR 3e-5, r 32\n200,704 px", 24.86, 23.7, 26.1, 52.72, 51.3, 54.2)]

def figure4():
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=300)
    w = 0.34
    for k, (lab, color, off) in enumerate((("In-distribution", C1, 1), ("Out-of-distribution", C2, 4))):
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(len(T5))]
        ys = [r[off] for r in T5]
        err = [[r[off] - r[off + 1] for r in T5], [r[off + 2] - r[off] for r in T5]]
        ax.bar(xs, ys, width=w, color=color, label=lab, zorder=2)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2.6, f"{y:.2f}", ha="center", va="bottom", fontsize=7.3, color=MUTED)
    ax.set_xticks(range(len(T5))); ax.set_xticklabels([r[0] for r in T5])
    ax.set_ylabel("Normalized CER (%)"); ax.set_ylim(0, 62)
    ax.tick_params(axis='x', labelsize=7.2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    clean(ax)
    save(fig, "figure4_ablations")

if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
