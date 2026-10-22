"""Revision-5 figures (notebook palette; consistent method order across figures).

Every figure here is drawn from exported CSV files only: the original full run
(results/full), the local MIMIC-IV aggregate outputs (results/mimic_local), and
the revision analyses (results/revision).  No numbers are typed by hand.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from .metrics import wilson_interval
from .revision import (DISPLAY_ORDER, CONTINUOUS_TAU, CONTINUOUS_LIPSCHITZ, continuous_grid,
                       continuous_risk, certification_frequency, exact_certification_probability)

COLORS = {"pointwise": "#ADB5BD", "normal_bonferroni": "#E8B77A", "multiplier": "#79B7B4",
          "exact_bonferroni": "#8CA9CF", "hoeffding": "#B5A0CA", "holm": "#99BE9C"}
DARK = {"pointwise": "#6C757D", "normal_bonferroni": "#B9772A", "multiplier": "#2F7F7B",
        "exact_bonferroni": "#3F6EA8", "hoeffding": "#7A5C9E", "holm": "#4E8A52"}
LABELS = {"pointwise": "Pointwise normal", "normal_bonferroni": "Bonferroni normal",
          "multiplier": "Multiplier", "exact_bonferroni": "Bonferroni exact",
          "hoeffding": "Hoeffding", "holm": "Exact Holm"}
WRAPPED = {"pointwise": "Pointwise\nnormal", "normal_bonferroni": "Bonferroni\nnormal", "multiplier": "Multiplier",
           "exact_bonferroni": "Bonferroni\nexact", "hoeffding": "Hoeffding", "holm": "Exact\nHolm"}
GUARANTEE_SHORT = {"exact_bonferroni": "finite-sample", "holm": "finite-sample,\nfixed $\\tau$",
                   "hoeffding": "finite-sample", "multiplier": "asymptotic",
                   "normal_bonferroni": "asymptotic", "pointwise": "none"}
REGION_COLORS = {"safe": "#66C2A5", "borderline": "#FEE08B", "uncertified": "#F28E8C"}
MODEL_COLORS = {"Logistic regression": "#1F4E79", "Extra Trees": "#C2702A",
                "HistGradientBoosting": "#2E8B57", "Always survive": "#8A8F96"}
TARGET_COLOR, REFERENCE_COLOR, HEADER_COLOR = "#A43138", "#303840", "#1F4E79"


def setup_style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def style_axis(ax):
    ax.grid(axis="y", alpha=.2, linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold")


def save(fig, directory, name):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return directory / f"{name}.pdf"


def _wilson(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    p, lo, hi = wilson_interval(v)
    return float(p), float(lo), float(hi)


def _mc_interval(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(max(0, v.mean() - half)), float(min(1, v.mean() + half))


# ---------------------------------------------------------------------------
def calibration_validity(root, out):
    """Panel A: band coverage and containment; panel B: ARR, same order and colors."""
    rep = pd.read_csv(Path(root) / "results/full/calibration_replications.csv")
    methods = [m for m in DISPLAY_ORDER if m in set(rep.method)]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]; low = .86
    for i, m in enumerate(methods):
        d = rep[rep.method == m]
        if d.simultaneous_coverage.notna().any():
            y, lo, hi = _wilson(d.simultaneous_coverage)
            if y < low:
                ax.plot(i - .12, low + .006, marker="v", color=DARK[m], ms=8)
                ax.text(i - .12, low + .016, f"coverage\n{y:.3f}\n[{lo:.3f}, {hi:.3f}]", ha="center",
                        va="bottom", fontsize=7.5, color=DARK[m])
            else:
                ax.errorbar(i - .12, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=7, color=COLORS[m],
                            mec=REFERENCE_COLOR, mew=.8, ecolor=DARK[m], capsize=3, lw=1.3, zorder=3)
        else:
            ax.text(i - .12, .875, "no band\n(fixed-$\\tau$\ntest)", ha="center", va="bottom", fontsize=7.5,
                    color=DARK[m])
        y, lo, hi = _wilson(d.containment)
        ax.errorbar(i + .14, y, yerr=[[y - lo], [hi - y]], fmt="D", ms=6.5, mfc="white", mec=DARK[m],
                    mew=1.4, ecolor=DARK[m], capsize=3, lw=1.1, zorder=3)
    ax.axhline(.95, color=TARGET_COLOR, ls="--", lw=1.1)
    ax.set_ylim(low, 1.008); ax.set_xlim(-.6, len(methods) - .4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f"{WRAPPED[m]}\n({GUARANTEE_SHORT[m]})" for m in methods], fontsize=8)
    ax.set_ylabel("Probability over 500 replications")
    panel_title(ax, "A. Validity: band coverage and containment")
    style_axis(ax)
    handles = [Line2D([0], [0], marker="o", ls="", color="#9AA4AE", mec=REFERENCE_COLOR, ms=7,
                      label="Simultaneous band coverage (95% Wilson CI)"),
               Line2D([0], [0], marker="D", ls="", mfc="white", mec=REFERENCE_COLOR, mew=1.4, ms=6.5,
                      label="Containment at $\\tau=0.34$ (95% Wilson CI)"),
               Line2D([0], [0], color=TARGET_COLOR, ls="--", lw=1.1, label="Nominal level 0.95")]

    ax = axes[1]
    data = [rep[rep.method == m].ARR.to_numpy() for m in methods]
    parts = ax.violinplot(data, positions=range(len(methods)), widths=.75, showextrema=False)
    for body, m in zip(parts["bodies"], methods):
        body.set_facecolor(COLORS[m]); body.set_edgecolor("none"); body.set_alpha(.45)
    for i, m in enumerate(methods):
        y, lo, hi = _mc_interval(rep[rep.method == m].ARR)
        ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], fmt="o", color=REFERENCE_COLOR, ms=5.5, capsize=4, lw=1.3, zorder=3)
        ax.text(i + .1, y + .035, f"{y:.3f}", fontsize=8, color=REFERENCE_COLOR, ha="left")
    ax.set_ylim(0, 1); ax.set_xlim(-.6, len(methods) - .4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f"{WRAPPED[m]}\n({GUARANTEE_SHORT[m]})" for m in methods], fontsize=8)
    ax.set_ylabel("Assurance-region recall (ARR)")
    panel_title(ax, "B. Informativeness: recall over the same replications")
    style_axis(ax)
    handles += [Patch(facecolor="#C9CED3", label="Replication-level ARR distribution"),
                Line2D([0], [0], marker="o", color=REFERENCE_COLOR, ls="", ms=5.5,
                       label="Mean ARR (95% Monte Carlo interval)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, -.07))
    fig.tight_layout(rect=(0, .06, 1, 1), w_pad=2.5)
    return save(fig, out, "calibration_validity")


# ---------------------------------------------------------------------------
STRESS_LABELS = {"bounded_continuous": "Bounded continuous\n($n=500$, $K=25$)",
                 "rare_binary": "Rare binary\n($n=500$, $K=25$)",
                 "high_dimensional_binary": "Larger binary family\n($n=1000$, $K=125$)"}


def stress_validity(root, out):
    s = pd.read_csv(Path(root) / "results/full/stress_summary.csv")
    scenarios = list(STRESS_LABELS)
    band = ["exact_bonferroni", "hoeffding", "multiplier", "normal_bonferroni"]
    allm = ["exact_bonferroni", "holm", "hoeffding", "multiplier", "normal_bonferroni"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    ax = axes[0]; width = .19
    for i, sc in enumerate(scenarios):
        for j, m in enumerate(band):
            x = i + (j - 1.5) * width
            r = s[(s.scenario == sc) & (s.method == m)]
            if not len(r):
                ax.text(x, .712, "n/a", ha="center", fontsize=7.5, color=DARK[m], rotation=90)
                continue
            r = r.iloc[0]; y = r.simultaneous_coverage
            ax.errorbar(x, y, yerr=[[y - r.simultaneous_coverage_lo], [r.simultaneous_coverage_hi - y]], fmt="o",
                        ms=6.5, color=COLORS[m], mec=REFERENCE_COLOR, mew=.7, ecolor=DARK[m], capsize=3, lw=1.3)
    ax.axhline(.95, color=TARGET_COLOR, ls="--", lw=1.1)
    ax.set_ylim(.70, 1.01); ax.set_xticks(range(3)); ax.set_xticklabels([STRESS_LABELS[k] for k in scenarios], fontsize=8.5)
    ax.set_ylabel("Simultaneous band coverage")
    panel_title(ax, "A. Validity: coverage (95% Wilson CI)"); style_axis(ax)
    ax = axes[1]; width = .16
    for i, sc in enumerate(scenarios):
        for j, m in enumerate(allm):
            x = i + (j - 2) * width
            r = s[(s.scenario == sc) & (s.method == m)]
            if not len(r):
                ax.text(x, .012, "n/a", ha="center", fontsize=7, color=DARK[m], rotation=90)
                continue
            r = r.iloc[0]; y = r.ARR; half = 1.96 * r.ARR_mcse
            ax.bar(x, y, width=width * .92, color=COLORS[m], edgecolor="white")
            if y < .015:
                ax.text(x, y + half + .012, f"{y:.3f}", ha="center", va="bottom", fontsize=6.5, color=DARK[m], rotation=90)
            ax.errorbar(x, y, yerr=half, fmt="none", ecolor=REFERENCE_COLOR, capsize=2.5, lw=1)
    ax.set_ylim(0, .75); ax.set_xticks(range(3)); ax.set_xticklabels([STRESS_LABELS[k] for k in scenarios], fontsize=8.5)
    ax.set_ylabel("Assurance-region recall (ARR)")
    panel_title(ax, "B. Informativeness: mean ARR (95% MC interval)"); style_axis(ax)
    handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in allm]
    handles.append(Line2D([0], [0], color=TARGET_COLOR, ls="--", label="Nominal level 0.95"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.5, -.06))
    fig.tight_layout(rect=(0, .05, 1, 1), w_pad=2.5)
    return save(fig, out, "stress_validity")


# ---------------------------------------------------------------------------
POWER_METHODS = ["exact_bonferroni", "holm", "hoeffding", "multiplier"]
MARKERS = {"exact_bonferroni": "o", "holm": "s", "hoeffding": "^", "multiplier": "D"}


def power_figure(root, rev, out):
    """Rows 1-2: certification probability by K; row 3: band coverage for the same cells."""
    s = pd.read_csv(Path(rev) / "power_summary.csv")
    a = pd.read_csv(Path(rev) / "power_analytic.csv")
    Ks = sorted(s.K.unique()); deltas = sorted(s.delta.unique())
    tau = .10; alpha = .05
    fig, axes = plt.subplots(len(Ks) + 1, len(deltas), figsize=(3.05 * len(deltas), 2.55 * (len(Ks) + 1)),
                             sharex=True)
    ticks = [250, 500, 1000, 2000, 4000, 8000]
    for a_i, K in enumerate(Ks):
        for b, delta in enumerate(deltas):
            ax = axes[a_i, b]
            for m in POWER_METHODS:
                d = s[(s.K == K) & (s.delta == delta) & (s.method == m)].sort_values("n")
                y = d.ARR.to_numpy(); half = 1.96 * d.ARR_mcse.to_numpy()
                ax.errorbar(d.n, y, yerr=half, color=DARK[m], marker=MARKERS[m], ms=4.5, lw=1.2, capsize=2,
                            mfc=COLORS[m], mec=DARK[m], zorder=3)
            v = a[(a.K == K) & (a.delta == delta)].sort_values("n")
            ax.plot(v.n, v.exact_certification_probability, ls="", marker="+", ms=9, mew=1.2, color="black", zorder=4)
            ax.axhline(.8, color=TARGET_COLOR, ls=":", lw=1)
            ax.set_ylim(-.02, 1.02)
            if a_i == 0:
                ax.set_title(f"Margin $\\Delta={delta:.2f}$", fontsize=10.5)
            if b == 0:
                ax.set_ylabel(f"$K={K}$: certification\nprobability", fontsize=9.5)
            else:
                ax.set_yticklabels([])
    row = len(Ks)
    for b, delta in enumerate(deltas):
        ax = axes[row, b]
        for m in ["exact_bonferroni", "hoeffding", "multiplier"]:
            for K, ls in zip(Ks, ["-", "--"]):
                d = s[(s.K == K) & (s.delta == delta) & (s.method == m)].sort_values("n")
                ax.plot(d.n, d.simultaneous_coverage, color=DARK[m], ls=ls, marker=MARKERS[m], ms=4,
                        mfc=COLORS[m] if ls == "-" else "white", mec=DARK[m], lw=1.2)
        ax.axhline(.95, color=TARGET_COLOR, ls="--", lw=1)
        ax.set_ylim(.55, 1.02)
        ax.set_xlabel("Assurance sample size $n$")
        if b == 0:
            ax.set_ylabel("Band coverage\n(both $K$)", fontsize=9.5)
        else:
            ax.set_yticklabels([])
    for ax in axes.flat:
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(["250", "500", "1k", "2k", "4k", "8k"])
        ax.minorticks_off(); style_axis(ax)
    handles = [Line2D([0], [0], color=DARK[m], marker=MARKERS[m], mfc=COLORS[m], mec=DARK[m], label=LABELS[m])
               for m in POWER_METHODS]
    handles += [Line2D([0], [0], ls="", marker="+", ms=9, mew=1.2, color="black", label="Bonferroni exact, analytic"),
                Line2D([0], [0], color=TARGET_COLOR, ls=":", label="0.80 reference"),
                Line2D([0], [0], color=TARGET_COLOR, ls="--", label="Nominal coverage 0.95"),
                Line2D([0], [0], color="#55606B", ls="-", label=f"Coverage, $K={Ks[0]}$"),
                Line2D([0], [0], color="#55606B", ls="--", label=f"Coverage, $K={Ks[-1]}$")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, -.045), fontsize=8.5)
    fig.suptitle(f"Exact-route power and band validity ($\\tau={tau:.2f}$, $\\alpha={alpha:.2f}$, latent $\\rho=0.55$)",
                 fontsize=11, y=.995)
    fig.tight_layout(rect=(0, .05, 1, .975))
    return save(fig, out, "power")


# ---------------------------------------------------------------------------
def tolerance_profiles_figure(rev, out):
    p = pd.read_csv(Path(rev) / "tolerance_profiles.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))
    specs = [("WDBC 0-1 risk", "A. WDBC: 0-1 risk\n(36 cells per model)", [.08, .10], (0, .30),
              {"exact_bonferroni": "-", "multiplier": "--"}),
             ("MIMIC-IV 0-1 risk", "B. MIMIC-IV: 0-1 risk\n(20 eligible conditions per model)", [.35], (0, .40),
              {"exact_bonferroni": "-", "multiplier": "--"}),
             ("MIMIC-IV false-negative risk", "C. MIMIC-IV: false-negative risk\n(20 eligible conditions per model)",
              [.20], (0, 1), {"exact_bonferroni": "-"})]
    for ax, (spec, title, marks, xlim, styles) in zip(axes, specs):
        d = p[p.specification == spec]
        for model in dict.fromkeys(d.Model):
            for method, ls in styles.items():
                v = d[(d.Model == model) & (d.method == method)].sort_values("tau")
                if not len(v):
                    continue
                ax.step(v.tau, v.certified_fraction, where="post", color=MODEL_COLORS[model], ls=ls,
                        lw=1.7 if ls == "-" else 1.3, alpha=1 if ls == "-" else .85)
        for i_t, t in enumerate(marks):
            ax.axvline(t, color=TARGET_COLOR, ls=":", lw=1.1)
            ha = "center" if len(marks) == 1 else ("right" if i_t == 0 else "left")
            ax.text(t + (0 if ha == "center" else (-.003 if ha == "right" else .003)), 1.035, f"$\\tau={t:.2f}$",
                    ha=ha, va="bottom", fontsize=8, color=TARGET_COLOR)
        ax.set_xlim(*xlim); ax.set_ylim(-.02, 1.02)
        if spec == "MIMIC-IV 0-1 risk":
            inset = ax.inset_axes([.28, .1, .52, .55])
            for model in dict.fromkeys(d.Model):
                for method, ls in styles.items():
                    v = d[(d.Model == model) & (d.method == method)].sort_values("tau")
                    inset.step(v.tau, v.certified_fraction, where="post", color=MODEL_COLORS[model], ls=ls,
                               lw=1.5 if ls == "-" else 1.1)
            inset.set_xlim(0, .06); inset.set_ylim(-.03, 1.03)
            inset.set_xticks([0, .02, .04, .06]); inset.tick_params(labelsize=7.5)
            inset.set_title("Zoom: $0\\leq\\tau\\leq0.06$", fontsize=8)
            inset.grid(alpha=.2, lw=.5)
        ax.set_xlabel("Tolerance $\\tau$" if "false" not in spec else "False-negative tolerance $\\tau_{FN}$")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=16)
        style_axis(ax)
    axes[0].set_ylabel("Certified fraction (exact upper bounds)")
    model_handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=1.8, label=m) for m in MODEL_COLORS]
    method_handles = [Line2D([0], [0], color="#55606B", lw=1.7, label="Bonferroni exact band"),
                      Line2D([0], [0], color="#55606B", lw=1.3, ls="--", label="Multiplier band (secondary)"),
                      Line2D([0], [0], color=TARGET_COLOR, ls=":", label="Prespecified tolerance")]
    fig.legend(handles=model_handles + method_handles, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(.5, -.06), fontsize=8.5)
    fig.tight_layout(rect=(0, .05, 1, 1), w_pad=2)
    return save(fig, out, "tolerance_profiles")


# ---------------------------------------------------------------------------
def continuous_exact_figure(rev, out):
    bounds = pd.read_csv(Path(rev) / "continuous_exact_grid_bounds.csv")
    grid = continuous_grid(); tau = CONTINUOUS_TAU; L = CONTINUOUS_LIPSCHITZ
    mg, sg = np.meshgrid(np.linspace(0, .4, 301), np.linspace(0, .6, 301), indexing="xy")
    query = np.column_stack([mg.ravel(), sg.ravel()])
    risk = continuous_risk(query).reshape(mg.shape)
    first = bounds[bounds.rep == 0].sort_values(["Missingness", "Shift"])
    dist = np.linalg.norm(query[:, None, :] - grid[None, :, :], axis=2)
    upper = np.minimum(1, np.min(first.upper.to_numpy()[None, :] + L * dist, axis=1)).reshape(mg.shape)
    category = np.where(upper <= tau, 0, np.where(risk <= tau, 1, 2))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6))
    ax = axes[0]
    cmap = ListedColormap([REGION_COLORS["safe"], REGION_COLORS["borderline"], REGION_COLORS["uncertified"]])
    ax.pcolormesh(mg, sg, category, cmap=cmap, vmin=-.5, vmax=2.5, shading="auto", rasterized=True)
    ax.contour(mg, sg, risk, levels=[tau], colors=REFERENCE_COLOR, linewidths=1.6)
    if (upper <= tau).any():
        ax.contour(mg, sg, upper, levels=[tau], colors=REFERENCE_COLOR, linewidths=1.2, linestyles="--")
    cert = first.grid_certified.to_numpy(bool)
    ax.scatter(grid[cert, 0], grid[cert, 1], s=34, facecolor=REGION_COLORS["safe"], edgecolor=REFERENCE_COLOR,
               linewidth=1.2, zorder=4, clip_on=False)
    ax.scatter(grid[~cert, 0], grid[~cert, 1], s=26, facecolor="white", edgecolor=REFERENCE_COLOR, zorder=4, clip_on=False)
    ax.set(xlabel="Missingness severity $m$", ylabel="Distribution-shift severity $s$", xlim=(0, .4), ylim=(0, .6))
    panel_title(ax, "A. First replication (prespecified), exact band")
    ax.legend(handles=[Patch(facecolor=REGION_COLORS["safe"], label="Certified: $U(e)\\leq\\tau$"),
                       Patch(facecolor=REGION_COLORS["borderline"], label="Truly safe, not certified"),
                       Patch(facecolor=REGION_COLORS["uncertified"], label="True risk exceeds $\\tau$"),
                       Line2D([0], [0], color=REFERENCE_COLOR, lw=1.6, label="True boundary $R(e)=\\tau$"),
                       Line2D([0], [0], color=REFERENCE_COLOR, lw=1.2, ls="--", label="Certified boundary $U(e)=\\tau$"),
                       Line2D([0], [0], marker="o", ls="", mfc=REGION_COLORS["safe"], mec=REFERENCE_COLOR, mew=1.2,
                              label="Grid point certified"),
                       Line2D([0], [0], marker="o", ls="", mfc="white", mec=REFERENCE_COLOR, label="Grid point not certified")],
              loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, fontsize=8, frameon=False)
    ax = axes[1]
    mg2, sg2 = np.meshgrid(np.linspace(0, .4, 161), np.linspace(0, .6, 161), indexing="xy")
    q2 = np.column_stack([mg2.ravel(), sg2.ravel()])
    freq = certification_frequency(bounds, q2, tau, L).reshape(mg2.shape)
    cmap2 = LinearSegmentedColormap.from_list("cert", ["#FFFFFF", "#D7F0E6", REGION_COLORS["safe"], "#2E7D63"])
    im = ax.pcolormesh(mg2, sg2, freq, cmap=cmap2, vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.contour(mg2, sg2, continuous_risk(q2).reshape(mg2.shape), levels=[tau], colors=REFERENCE_COLOR, linewidths=1.6)
    if freq.max() >= .5:
        ax.contour(mg2, sg2, freq, levels=[.5], colors="#2E7D63", linewidths=1.1, linestyles="--")
    ax.scatter(grid[:, 0], grid[:, 1], s=14, color=REFERENCE_COLOR, zorder=4, clip_on=False)
    ax.set(xlabel="Missingness severity $m$", ylabel="Distribution-shift severity $s$", xlim=(0, .4), ylim=(0, .6))
    n_rep = bounds.rep.nunique()
    panel_title(ax, f"B. Certification frequency over {n_rep} replications")
    cb = fig.colorbar(im, ax=ax, shrink=.9); cb.set_label("Fraction of replications certifying the point")
    ax.legend(handles=[Line2D([0], [0], color=REFERENCE_COLOR, lw=1.6, label="True boundary $R(e)=\\tau$"),
                       Line2D([0], [0], color="#2E7D63", lw=1.1, ls="--", label="Frequency 0.5 contour"),
                       Line2D([0], [0], marker="o", ls="", color=REFERENCE_COLOR, ms=4, label="Assurance grid")],
              loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(w_pad=2)
    return save(fig, out, "continuous_exact")


def make_all_figures(root):
    setup_style()
    root = Path(root); rev = root / "results/revision"; out = root / "figures/revision"
    made = [calibration_validity(root, out), stress_validity(root, out)]
    if (rev / "power_summary.csv").is_file():
        made.append(power_figure(root, rev, out))
    if (rev / "tolerance_profiles.csv").is_file():
        made.append(tolerance_profiles_figure(rev, out))
    if (rev / "continuous_exact_grid_bounds.csv").is_file():
        made.append(continuous_exact_figure(rev, out))
    for path in made:
        print("Saved", path)
    return made
