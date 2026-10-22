"""Combined multi-panel figures for the restructured manuscript.

Drawn only from archived CSV outputs in results/full; no values are typed by hand.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .presentation import (COLORS, DARK, LABELS, REFERENCE_COLOR, setup_style, style_axis,
                           panel_title, save)

NESTED_ORDER = ["exact_bonferroni", "holm", "hoeffding", "multiplier", "normal_bonferroni"]
MISS_ORDER = ["exact_bonferroni", "holm", "multiplier"]
MECHANISMS = ["MCAR", "MAR", "MNAR", "Structured"]


def pipelines_missingness(root, out):
    """Panel A: nested-refit recall by model; panel B: recall by missingness mechanism;
    panel C: multiplier containment brackets by mechanism."""
    setup_style()
    root = Path(root)
    nest = pd.read_csv(root / "results/full/nested_summary.csv")
    miss = pd.read_csv(root / "results/full/missingness_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), gridspec_kw={"width_ratios": [1.15, 1.15, .8]})

    ax = axes[0]
    x = np.arange(len(NESTED_ORDER)); w = .38
    for j, (model, hatch) in enumerate([("Logistic regression", None), ("HistGradientBoosting", "///")]):
        d = nest[nest.Model == model].set_index("method").loc[NESTED_ORDER]
        ax.bar(x + (j - .5) * w, d.reference_ARR, width=w, color=[COLORS[m] for m in NESTED_ORDER],
               edgecolor=[DARK[m] for m in NESTED_ORDER], hatch=hatch, lw=.8)
        ax.errorbar(x + (j - .5) * w, d.reference_ARR, yerr=1.96 * d.reference_ARR_mcse, fmt="none",
                    ecolor=REFERENCE_COLOR, capsize=2.5, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m].replace(" ", "\n", 1) for m in NESTED_ORDER], fontsize=8.5)
    ax.set_ylim(0, 1); ax.set_ylabel("Reference-based ARR")
    panel_title(ax, "A. Nested refits, by fitted model")
    style_axis(ax)
    ax.legend(handles=[Patch(facecolor="#D5DADF", edgecolor="#6C757D", label="Logistic regression"),
                       Patch(facecolor="#D5DADF", edgecolor="#6C757D", hatch="///", label="Histogram gradient boosting")],
              loc="upper left", frameon=False, fontsize=8.5)

    ax = axes[1]
    x = np.arange(len(MECHANISMS)); w = .26
    for j, m in enumerate(MISS_ORDER):
        d = miss[miss.method == m].set_index("Mechanism").loc[MECHANISMS]
        ax.bar(x + (j - 1) * w, d.reference_ARR, width=w, color=COLORS[m], edgecolor=DARK[m], lw=.8, label=LABELS[m])
        ax.errorbar(x + (j - 1) * w, d.reference_ARR, yerr=1.96 * d.reference_ARR_mcse, fmt="none",
                    ecolor=REFERENCE_COLOR, capsize=2.5, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(MECHANISMS, fontsize=9)
    ax.set_ylim(0, .5); ax.set_ylabel("Reference-based ARR")
    panel_title(ax, "B. Missingness mechanisms (LR)")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    ax = axes[2]
    d = miss[miss.method == "multiplier"].set_index("Mechanism").loc[MECHANISMS]
    ax.vlines(x, d.containment_lower, d.containment_upper, color=DARK["multiplier"], lw=5, alpha=.55)
    ax.plot(x, d.reference_containment, "D", color=DARK["multiplier"], mfc="white", mew=1.4, ms=6)
    ax.set_xticks(x); ax.set_xticklabels(MECHANISMS, fontsize=9)
    ax.set_ylim(.88, 1.01); ax.set_xlim(-.5, 3.5); ax.set_ylabel("Containment indicators (multiplier)")
    panel_title(ax, "C. Reference uncertainty")
    style_axis(ax)
    ax.legend(handles=[Line2D([0], [0], marker="D", ls="", mfc="white", mec=DARK["multiplier"], mew=1.4, ms=6,
                              label="Point-reference containment"),
                       Line2D([0], [0], color=DARK["multiplier"], lw=5, alpha=.55, label="Mean $C^-$ to $C^+$")],
              loc="lower left", frameon=False, fontsize=8.5)
    fig.tight_layout(w_pad=2.0)
    return save(fig, out, "pipelines_missingness")


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    print(pipelines_missingness(root, root / "figures/revision"))
