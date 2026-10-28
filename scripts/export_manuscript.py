#!/usr/bin/env python3
"""Export tables and prose macros from completed CSVs; no copied numbers.

Inputs (each verified before use):
  results/full/*.csv            original full run, hashes in results/full/run_manifest.json
  results/revision/*.csv        revision-5 analyses, hashes in results/revision/revision_manifest.json
  results/mimic_local/*.csv     aggregate outputs of the local credentialed MIMIC-IV run
Outputs: manuscript/generated/*.tex and manuscript/generated/table_provenance.json
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciguard.experiments import LABELS
from sciguard.revision import DISPLAY_ORDER, GUARANTEE

MODEL_SHORT = {"Logistic regression": "LR", "Extra Trees": "Extra Trees", "HistGradientBoosting": "HGB",
               "Always survive": "Always survive"}


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(path, headers, rows, align=None, notes=None, header_tex=None):
    """Write a booktabs tabular.  header_tex, if given, replaces the single header row
    (used for two-level headers); headers still fixes the number of columns."""
    align = align or "l" + "r" * (len(headers) - 1)
    head = header_tex if header_tex is not None else " & ".join(headers) + r" \\"
    text = [r"\begin{tabular}{" + align + "}", r"\toprule", head, r"\midrule"]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule"]
    for note in notes or []:
        text += [r"\multicolumn{" + str(len(headers)) + r"}{l}{\footnotesize " + note + r"}\\"]
    text += [r"\end{tabular}"]
    path.write_text("\n".join(text) + "\n")


def number(v, d=3):
    return "--" if pd.isna(v) else f"{v:.{d}f}"


def domain(label):
    """Typeset an approximate admission-interval label such as '2017 - 2021' as 2017--2021."""
    return str(label).replace(" - ", "--")


def interval(row, metric):
    if pd.isna(row[metric]):
        return "--"
    return f"{row[metric]:.3f} [{row[metric+'_lo']:.3f}, {row[metric+'_hi']:.3f}]"


def ordered(frame, column="method"):
    rank = {m: i for i, m in enumerate(DISPLAY_ORDER)}
    return frame.assign(_rank=frame[column].map(rank)).sort_values("_rank", kind="stable").drop(columns="_rank")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder, manifest_name, prefix):
    manifest = json.loads((folder / manifest_name).read_text())
    if not str(manifest.get("status", "")).startswith(prefix):
        raise RuntimeError(f"{folder} run is incomplete: {manifest.get('status')}")
    for name, digest in manifest["output_sha256"].items():
        if sha(folder / name) != digest:
            raise RuntimeError(f"CSV changed after run: {folder.name}/{name}")
    return manifest


def export():
    p = ROOT / "results/full"; rv = ROOT / "results/revision"; mi = ROOT / "results/mimic_local"
    out = ROOT / "manuscript/generated"; out.mkdir(exist_ok=True, parents=True)
    manifest = verify(p, "run_manifest.json", "completed")
    rev_manifest = verify(rv, "revision_manifest.json", "completed")
    macros = {}

    # ---- RQ1: calibration with explicit guarantee type -------------------------------
    s = ordered(pd.read_csv(p / "calibration_summary.csv"))
    table(out / "calibration.tex", ["Method", "Guarantee", "Coverage", "Containment", "ARR", "UCP"],
          [[LABELS[r.method], GUARANTEE[r.method], interval(r, "simultaneous_coverage"), interval(r, "containment"),
            number(r.ARR), number(r.UCP, 4)] for _, r in s.iterrows()], "llrrrr")
    si = s.set_index("method")
    macros.update({"CalMultCoverage": number(si.loc["multiplier", "simultaneous_coverage"]),
                   "CalPointCoverage": number(si.loc["pointwise", "simultaneous_coverage"]),
                   "CalExactCoverage": number(si.loc["exact_bonferroni", "simultaneous_coverage"]),
                   "CalNormalCoverage": number(si.loc["normal_bonferroni", "simultaneous_coverage"]),
                   "CalMultContainment": number(si.loc["multiplier", "containment"]),
                   "CalHoeffARR": number(si.loc["hoeffding", "ARR"]), "CalHolmARR": number(si.loc["holm", "ARR"]),
                   "MainMultARR": number(si.loc["multiplier", "ARR"]), "MainExactARR": number(si.loc["exact_bonferroni", "ARR"])})

    n = pd.read_csv(p / "nested_summary.csv")
    n = pd.concat([ordered(n[n.Model == m]) for m in dict.fromkeys(n.Model)])
    table(out / "nested.tex", ["Model", "Method", r"Reference containment [95\% CI]", "ARR", "Reference bracket"],
          [[MODEL_SHORT[r.Model], LABELS[r.method], interval(r, "reference_containment"), number(r.reference_ARR),
            f"[{r.containment_lower:.3f}, {r.containment_upper:.3f}]"] for _, r in n.iterrows()], "llrrr")
    w = pd.read_csv(p / "wdbc_summary.csv"); wp = w[w.design == "primary"]
    wp = pd.concat([ordered(wp[wp.Model == m]) for m in dict.fromkeys(wp.Model)])
    table(out / "wdbc.tex", ["Model", "Method", "Nominal risk", r"Certified $\tau=.08$", r"Certified $\tau=.10$"],
          [[MODEL_SHORT[r.Model], LABELS[r.method], number(r.nominal_risk), f"{r.certified_008}/36", f"{r.certified_010}/36"]
           for _, r in wp.iterrows()], "llrrr")
    grid = w[(w.Model == "Logistic regression") & (w.design != "primary") & (w.method.isin(["multiplier", "exact_bonferroni"]))]
    table(out / "grid.tex", ["Grid", "Method", r"$K$", r"Certified $\tau=.08$", "Fraction"],
          [[r"$" + r.design.replace("x", r"\times") + r"$", LABELS[r.method], r.K, r.certified_008, number(r.certified_fraction)] for _, r in grid.iterrows()], "llrrr")
    g = pd.read_csv(p / "paired_efficiency_gains.csv")
    table(out / "gains.tex", [r"$n$", r"$\rho$", "Comparator", "Paired ARR gain", "MCSE"],
          [[int(r.n), f"{r.rho:.2f}", LABELS[r.baseline], number(max(r.ARR_gain, 0.0)), number(r.ARR_gain_mcse, 4)] for _, r in g.iterrows()], "rrlrr")
    m = pd.read_csv(p / "missingness_summary.csv"); mm = m[m.method == "multiplier"]
    table(out / "missingness.tex", ["Mechanism", "Reference containment", "Reference bracket", "ARR", "Ambiguous cells"],
          [[r.Mechanism, interval(r, "reference_containment"), f"[{r.containment_lower:.3f}, {r.containment_upper:.3f}]",
            number(r.reference_ARR), number(r.ambiguous_environments, 2)] for _, r in mm.iterrows()])
    st_all = pd.read_csv(p / "stress_summary.csv")
    st = st_all[st_all.method.isin(["normal_bonferroni", "multiplier", "exact_bonferroni", "hoeffding", "holm"])]
    st = pd.concat([ordered(st[st.scenario == k]) for k in dict.fromkeys(st.scenario)])
    names = {"bounded_continuous": "Continuous bounded", "rare_binary": "Rare binary", "high_dimensional_binary": "Larger binary family"}
    table(out / "stress.tex", ["Scenario", "Method", r"Coverage [95\% CI]", "Containment", "ARR"],
          [[names[r.scenario], LABELS[r.method], interval(r, "simultaneous_coverage"), number(r.containment), number(r.ARR)]
           for _, r in st.iterrows()], "llrrr")
    stx = st_all.set_index(["scenario", "method"])
    macros.update({"RareMultCoverage": number(stx.loc[("rare_binary", "multiplier"), "simultaneous_coverage"]),
                   "RareExactCoverage": number(stx.loc[("rare_binary", "exact_bonferroni"), "simultaneous_coverage"]),
                   "LargeMultCoverage": number(stx.loc[("high_dimensional_binary", "multiplier"), "simultaneous_coverage"])})
    rep = pd.read_csv(p / "stress_replications.csv")
    identical = {}
    for sc in ["bounded_continuous", "rare_binary"]:
        piv = rep[rep.scenario == sc].pivot(index="rep", columns="method", values="simultaneous_coverage")
        identical[sc] = int((piv.normal_bonferroni == piv.multiplier).sum())
    macros["StressIdenticalReps"] = str(min(identical.values()))
    macros["StressReps"] = str(int(rep.rep.nunique()))

    c = pd.read_csv(p / "continuous_summary.csv")
    c = ordered(c)
    table(out / "continuous.tex", ["Method", "Grid coverage", "Mesh containment", "Mesh recall", "Mesh UCP"],
          [[LABELS[r.method], number(r.grid_coverage), number(r.containment), number(r.ARR), number(r.UCP, 4)] for _, r in c.iterrows()])
    a = pd.read_csv(p / "atc_prediction_summary.csv")
    table(out / "atc.tex", ["Model", "MAE", "Bias", "RMSE"], [[r.Model, number(r.MAE), number(r.bias), number(r.RMSE)] for _, r in a.iterrows()])
    b = pd.read_csv(p / "bootstrap_runtime_summary.csv")
    table(out / "runtime.tex", [r"$n$", r"$K$", r"$B$", "Critical mean", "Critical MCSE", "Mean seconds"],
          [[int(r.n), int(r.K), int(r.B), number(r.critical), number(r.critical_mcse, 4), number(r.seconds, 4)] for _, r in b.iterrows()])
    plan = pd.read_csv(p / "sample_size_planning.csv")
    table(out / "planning.tex", [r"$K$", "Margin", r"Sufficient $n$"], [[int(r.K), f"{r.margin:.2f}", int(r.hoeffding_n)] for _, r in plan.iterrows()])
    macros.update({"NestedReps": str(manifest["config"]["nested_reps"]), "CalibrationReps": str(manifest["config"]["calibration_reps"]),
                   "ReferenceN": str(manifest["config"]["n_reference"]), "MissingnessReps": str(manifest["config"]["missingness_reps"])})

    # ---- RQ2: power study and exact planning (revision) ---------------------------------
    ps = pd.read_csv(rv / "power_summary.csv"); pa = pd.read_csv(rv / "power_analytic.csv")
    pp = pd.read_csv(rv / "power_planning.csv")
    table(out / "power_planning.tex",
          [r"$K$", r"$\Delta$", r"Hoeffding width $n_{\mathrm H}$", r"Exact $n$, 80\%", r"Exact $n$, 90\%",
           r"Hoeffding $P_{\mathrm{cert}}(n_{\mathrm H})$", r"Exact $P_{\mathrm{cert}}(n_{\mathrm H})$"],
          [[int(r.K), f"{r.delta:.2f}", f"{int(r.hoeffding_width_n):,}", f"{int(r.exact_n_80):,}", f"{int(r.exact_n_90):,}",
            number(r.hoeffding_probability_at_hoeffding_n), number(r.exact_probability_at_hoeffding_n)] for _, r in pp.iterrows()],
          "rrrrrrr",
          header_tex=(r" & & Hoeffding & \multicolumn{2}{c}{Exact $n$ for $P_{\mathrm{cert}}\geq$} & "
                      r"\multicolumn{2}{c}{$P_{\mathrm{cert}}$ at $n_{\mathrm H}$} \\" "\n"
                      r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
                      r"$K$ & $\Delta$ & width $n_{\mathrm H}$ & 0.8 & 0.9 & Hoeffding & Exact \\"))
    ratio = pp.exact_n_80 / pp.hoeffding_width_n
    rows = []
    for method in ["exact_bonferroni", "holm", "hoeffding", "multiplier"]:
        for K in sorted(ps.K.unique()):
            d = ps[(ps.method == method) & (ps.K == K)]
            small = d[d.n == d.n.min()]
            rows.append([LABELS[method], int(K),
                         "--" if d.simultaneous_coverage.isna().all() else f"{d.simultaneous_coverage.min():.3f}",
                         "--" if small.simultaneous_coverage.isna().all() else f"{small.simultaneous_coverage.min():.3f}",
                         f"{d.containment.min():.3f}", f"{d.UCP.max():.4f}"])
    table(out / "power_validity.tex", ["Method", r"$K$", "Coverage", "Coverage n250", "Containment", "UCP"], rows, "lrrrrr",
          header_tex=(r" & & \multicolumn{2}{c}{Band coverage (min.)} & Containment & UCP \\" "\n"
                      r"\cmidrule(lr){3-4}" "\n"
                      r"Method & $K$ & all $n$ & $n=" + str(int(ps.n.min())) + r"$ & (min.) & (max.) \\"))
    exact_sim = ps[ps.method == "exact_bonferroni"].merge(pa, on=["K", "n", "delta"])
    dev = (exact_sim.ARR - exact_sim.exact_certification_probability).abs()
    mult = ps[ps.method == "multiplier"]
    gain = ps.pivot_table(index=["K", "n", "delta"], columns="method", values="ARR")
    macros.update({"PowerReps": str(rev_manifest["config"]["power_reps"]),
                   "PowerCells": str(len(pa)),
                   "PowerAnalyticMaxDev": number(dev.max()),
                   "PowerMultCovMin": number(mult.simultaneous_coverage.min()),
                   "PowerMultCovMax": number(mult.simultaneous_coverage.max()),
                   "PowerMultCovMinSmallK": number(mult[mult.K == mult.K.min()].simultaneous_coverage.min()),
                   "PowerExactCovMin": number(ps[ps.method == "exact_bonferroni"].simultaneous_coverage.min()),
                   "PowerMinContainment": number(ps.containment.min()),
                   "PowerHolmGainMax": number((gain.holm - gain.exact_bonferroni).max()),
                   "PowerMultGainMax": number((gain.multiplier - gain.exact_bonferroni).max()),
                   "PlanRatioMin": number(ratio.min(), 2), "PlanRatioMax": number(ratio.max(), 2),
                   "PlanHoeffProbMin": number(pp.hoeffding_probability_at_hoeffding_n.min(), 2),
                   "PlanHoeffProbMax": number(pp.hoeffding_probability_at_hoeffding_n.max(), 2),
                   "PlanExactSmall": f"{int(pp[(pp.K == 25) & np.isclose(pp.delta, .05)].exact_n_80.iloc[0]):,}",
                   "PlanHoeffSmall": f"{int(pp[(pp.K == 25) & np.isclose(pp.delta, .05)].hoeffding_width_n.iloc[0]):,}"})

    small_n = mult[mult.n == mult.n.min()]; large_n = mult[mult.n == mult.n.max()]
    Ksmall, Klarge = sorted(ps.K.unique())[0], sorted(ps.K.unique())[-1]
    best = (gain.multiplier - gain.exact_bonferroni).idxmax()
    best_cov = mult.set_index(["K", "n", "delta"]).loc[best, "simultaneous_coverage"]
    d5 = pp[np.isclose(pp.delta, .05)]; d1 = pp[np.isclose(pp.delta, .01)]
    macros.update({"PowerSmallN": f"{int(mult.n.min()):,}", "PowerLargeN": f"{int(mult.n.max()):,}",
                   "PowerKsmall": str(int(Ksmall)), "PowerKlarge": str(int(Klarge)),
                   "PowerMultCovSmallKLo": number(small_n[small_n.K == Ksmall].simultaneous_coverage.min()),
                   "PowerMultCovSmallKHi": number(small_n[small_n.K == Ksmall].simultaneous_coverage.max()),
                   "PowerMultCovLargeKLo": number(small_n[small_n.K == Klarge].simultaneous_coverage.min()),
                   "PowerMultCovLargeKHi": number(small_n[small_n.K == Klarge].simultaneous_coverage.max()),
                   "PowerMultCovLargeNLo": number(large_n.simultaneous_coverage.min()),
                   "PowerMultCovLargeNHi": number(large_n.simultaneous_coverage.max()),
                   "PowerMultBestGainCov": number(best_cov),
                   "PlanExactDeltaFiveLo": f"{int(d5.exact_n_80.min()):,}", "PlanExactDeltaFiveHi": f"{int(d5.exact_n_80.max()):,}",
                   "PlanExactDeltaOneLo": f"{int(d1.exact_n_80.min()):,}", "PlanExactDeltaOneHi": f"{int(d1.exact_n_80.max()):,}",
                   "PlanExactProbAtHoeffMin": number(pp.exact_probability_at_hoeffding_n.min())})
    an = pd.read_csv(rv / "calibration_anatomy_summary.csv").set_index("method")
    rows = []
    for method in [m for m in DISPLAY_ORDER if m in an.index]:
        r = an.loc[method]
        if pd.isna(r.coverage_failures):
            rows.append([LABELS[method], f"{int(r.reproduces_original)}/{int(r.replications)}", "--", "--", "--",
                         str(int(r.containment_failures))])
        else:
            rows.append([LABELS[method], f"{int(r.reproduces_original)}/{int(r.replications)}", str(int(r.coverage_failures)),
                         str(int(r.failures_safe_only)), str(int(r.failures_with_unsafe_miss)),
                         str(int(r.containment_failures))])
            assert int(r.failures_with_false_certificate) == int(r.containment_failures)
    table(out / "calibration_anatomy.tex", ["Method", "Reproduced", "Band failures", "Only acceptable", "An unacceptable", "Containment"],
          rows, "lrrrrr",
          header_tex=(r" & & Band & \multicolumn{2}{c}{Failures that missed} & Containment \\" "\n"
                      r"\cmidrule(lr){4-5}" "\n"
                      r"Method & Reproduced & failures & only acceptable & an unacceptable & failures \\"))
    macros.update({"AnatReps": str(int(an.loc["exact_bonferroni", "replications"])),
                   "AnatMultReproduced": str(int(an.loc["multiplier", "reproduces_original"])),
                   "AnatExactFailures": str(int(an.loc["exact_bonferroni", "coverage_failures"])),
                   "AnatExactSafeOnly": str(int(an.loc["exact_bonferroni", "failures_safe_only"])),
                   "AnatExactFalseCert": str(int(an.loc["exact_bonferroni", "failures_with_false_certificate"])),
                   "AnatMultFailures": str(int(an.loc["multiplier", "coverage_failures"])),
                   "AnatMultSafeOnly": str(int(an.loc["multiplier", "failures_safe_only"])),
                   "AnatMultFalseCert": str(int(an.loc["multiplier", "failures_with_false_certificate"])),
                   "AnatMultCoverageRerun": number(1 - an.loc["multiplier", "coverage_failures"] / an.loc["multiplier", "replications"])})
    sc = pd.read_csv(rv / "stress_coincidence.csv")
    for tag, scen in [("Cont", "bounded_continuous"), ("Rare", "rare_binary")]:
        d = sc[sc.scenario == scen]
        macros[f"Stress{tag}MultCrit"] = number(d.multiplier_critical.mean(), 2)
        macros[f"Stress{tag}Agree"] = str(int(d.matches_archive.sum()))
    macros["StressBonfCrit"] = number(sc.bonferroni_critical.iloc[0], 2)
    rare = sc[(sc.scenario == "rare_binary") & sc.min_errors_in_missed.notna()]
    macros["StressRareLowCountShare"] = f"{100 * np.mean(rare.min_errors_in_missed <= 3):.0f}"
    g2 = pd.read_csv(p / "paired_efficiency_gains.csv")
    ge = g2[g2.baseline == "exact_bonferroni"].ARR_gain
    macros.update({"GainExactMin": number(ge.min()), "GainExactMax": number(ge.max())})
    ti = pd.read_csv(rv / "studentized_tail_inflation.csv")
    ns = sorted(ti.n.unique())
    rows = []
    for (K, risk), d in ti.groupby(["K", "risk"], sort=True):
        d = d.set_index("n").inflation
        rows.append([str(int(K)), f"{risk:.2f}"] + [f"{d[n]:.2f}" for n in ns])
    table(out / "tail_inflation.tex", [r"$K$", r"$R_k$"] + [f"$n={int(n):,}$" for n in ns], rows, "rr" + "r" * len(ns))
    def tail(K, n):
        v = ti[(ti.K == K) & (ti.n == n) & ti.risk.isin([.09, .11])].inflation
        return f"{v.min():.1f}", f"{v.max():.1f}"
    for tag, (K, n) in {"LargeKSmallN": (100, 250), "LargeKLargeN": (100, 8000), "SmallKLargeN": (25, 8000),
                        "LargeKHugeN": (100, 32000)}.items():
        lo, hi = tail(K, n); macros[f"Tail{tag}Lo"] = lo; macros[f"Tail{tag}Hi"] = hi
    br = pd.read_csv(rv / "bootstrap_reproduction.csv")
    sd = br.groupby("B").difference.std()
    macros.update({"BootDiffBLow": f"{int(sd.index.min()):,}", "BootDiffBHigh": f"{int(sd.index.max()):,}",
                   "BootDiffSdLow": number(sd.iloc[0]), "BootDiffSdHigh": number(sd.iloc[-1]),
                   "BootSameCount": f"{100 * np.mean(br.archived_certified == br.recomputed_certified):.0f}"})
    om = manifest
    macros.update({"OrigPython": om["python"], "OrigNumpy": om["packages"]["numpy"],
                   "RevPython": rev_manifest["python"], "RevNumpy": rev_manifest["packages"]["numpy"],
                   "RevScipy": rev_manifest["packages"]["scipy"], "RevPandas": rev_manifest["packages"]["pandas"],
                   "RevSklearn": rev_manifest["packages"]["scikit-learn"]})
    boot = pd.read_csv(p / "bootstrap_runtime_replications.csv")
    macros["BootSeeds"] = str(int(boot.rep.nunique()))

    ce = pd.read_csv(rv / "continuous_exact_example.csv"); cb = pd.read_csv(rv / "continuous_exact_grid_bounds.csv")
    orig = pd.read_csv(p / "continuous_example.csv")
    first = cb[cb.rep == 0]
    macros.update({"ContExactFirstRecall": number(ce.certified[ce.risk <= .32].mean()),
                   "ContMultFirstRecall": number(orig.certified[orig.risk <= .32].mean()),
                   "ContExactFirstGrid": str(int(first.grid_certified.sum())),
                   "ContExactFirstSafeGrid": str(int((first.risk <= .32).sum())),
                   "ContExactFirstUnsafe": str(int((ce.certified & (ce.risk > .32)).sum())),
                   "ContReps": str(int(cb.rep.nunique()))})

    # ---- RQ4: MIMIC-IV aggregate tables ---------------------------------------------------
    mr = pd.read_csv(mi / "mimic_results.csv"); md = pd.read_csv(mi / "mimic_diagnostics.csv")
    ex = mr[mr.method == "exact_bonferroni"]
    lr0 = md[(md.Model == "Logistic regression") & np.isclose(md.Missingness, 0)].set_index("TemporalDomain")
    rows = []
    for dom, d in ex.groupby("TemporalDomain", sort=True):
        eligible = bool(d.eligible.iloc[0]); r = lr0.loc[dom]
        rows.append([domain(dom), f"{int(d.n.iloc[0]):,}", number(r.mortality_prevalence), number(r.risk),
                     number(r.AUROC) if eligible else "--", number(r.AUPRC) if eligible else "--",
                     number(d.upper.max()) if eligible else "--",
                     f"{int(d.certified.sum())}/{len(d)}" + ("" if eligible else r"$^\dagger$")])
    table(out / "mimic.tex", ["Interval", "$n$", "Mort.", "LR risk", "AUROC", "AUPRC", "Max $U$", "Cert."], rows, "lrrrrrrc")
    macros.update({"MimicEligible": str(int(ex.eligible.sum())), "MimicConditions": str(len(ex)),
                   "MimicMaxExactUpper": number(ex[ex.eligible].upper.max()),
                   "MimicMaxHoeffUpper": number(mr[(mr.method == "hoeffding") & mr.eligible].upper.max())})

    fn = pd.read_csv(mi / "mimic_class_sensitive_results.csv")
    th = pd.read_csv(mi / "mimic_class_sensitive_thresholds.csv")
    rows = []
    for _, r in th.iterrows():
        e = fn[(fn.Model == r.Model) & (fn.method == "exact_bonferroni") & fn.eligible]
        h = fn[(fn.Model == r.Model) & (fn.method == "holm") & fn.eligible]
        rows.append([r.Model, number(r.FrozenThreshold, 4), number(r.AchievedSourceSensitivity), number(r.SourceSpecificity),
                     f"{int(e.certified.sum())}/{len(e)}", f"{int(h.certified.sum())}/{len(h)}"])
    table(out / "mimic_fn.tex", ["Model", "Threshold", "Source sens.", "Source spec.", "Bonferroni", "Holm"], rows, "lrrrrr")
    ess = pd.read_csv(rv / "mimic_effective_sample_size.csv"); need = pd.read_csv(rv / "mimic_deaths_needed.csv")
    rows = []
    for _, r in ess.iterrows():
        elig = bool(r.eligible)
        def sens(tag):
            if not elig:
                return "--"
            return f"{r[f'{tag}_sensitivity_0']:.3f} [{r[f'{tag}_lower_sensitivity_0']:.3f}]"
        rows.append([domain(r.TemporalDomain), f"{int(r.deaths):,}", number(r.best_lower_sensitivity) if elig else "--",
                     "yes" if r.certifiable_even_with_zero_fn else "no", sens("LR"), sens("HGB"),
                     f"{int(r.LR_exact_certified)}/{int(r.LR_conditions)}" if elig else "--",
                     f"{int(r.HGB_exact_certified)}/{int(r.HGB_conditions)}" if elig else "--"])
    table(out / "mimic_ess.tex", ["Interval", "Deaths", "Best lower", "Possible", "LR", "HGB", "LR", "HGB"], rows, "lrrcrrrr",
          header_tex=(r" & & Best lower & Can & \multicolumn{2}{c}{Sensitivity [lower bound]} & "
                      r"\multicolumn{2}{c}{Certified} \\" "\n"
                      r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
                      r"Interval & Deaths & sens. & certify? & LR & HGB & LR & HGB \\"))
    zero = need[np.isclose(need.true_sensitivity, 1.0)].iloc[0]
    def needed(sens, target=.8):
        v = need[np.isclose(need.true_sensitivity, sens) & np.isclose(need.target_probability, target)].deaths_stable.iloc[0]
        return f"{int(v):,}"
    big = ess[ess.deaths == ess.deaths.max()].iloc[0]
    small = ess[ess.eligible & (ess.deaths < ess.deaths.max())]
    macros.update({"EssK": str(int(ess.K.iloc[0])), "EssZeroDeaths": str(int(zero.deaths_stable)),
                   "EssZeroFNUpperFourteen": number(ess[ess.deaths == 14].zero_fn_upper.iloc[0]),
                   "EssBestLowerFourteen": number(ess[ess.deaths == 14].best_lower_sensitivity.iloc[0]),
                   "EssSmallDeathsMin": str(int(small.deaths.min())), "EssSmallDeathsMax": str(int(small.deaths.max())),
                   "EssLargeDeaths": f"{int(big.deaths):,}",
                   "EssDeathsEightyFive": needed(.85), "EssDeathsNinety": needed(.90), "EssDeathsNinetyFive": needed(.95),
                   "EssDeathsEightyFiveNinety": needed(.85, .9)})
    # profile and planning numbers quoted in RQ4 (all from archived per-condition bounds)
    wd = pd.read_csv(p / "wdbc_detail.csv"); wd = wd[(wd.design == "primary") & (wd.method == "exact_bonferroni")]
    wmax = wd.groupby("Model").upper.max(); wmin = wd.groupby("Model").upper.min()
    macros.update({"WdbcExactFullLR": number(wmax["Logistic regression"]), "WdbcExactFullET": number(wmax["Extra Trees"]),
                   "WdbcExactFullHGB": number(wmax["HistGradientBoosting"]), "WdbcExactFirstLR": number(wmin["Logistic regression"])})
    always = ex[ex.eligible & (ex.Model == "Always survive")]
    macros["MimicAlwaysFull"] = number(always.upper.max())
    fx = fn[(fn.method == "exact_bonferroni") & fn.eligible & (fn.Model != "Always survive")]
    large_dom = fx.groupby("TemporalDomain").positive_n.max().idxmax()
    macros.update({"ProfileFnLargeMin": number(fx[fx.TemporalDomain == large_dom].upper_fn_risk.min()),
                   "ProfileFnLargeMax": number(fx[fx.TemporalDomain == large_dom].upper_fn_risk.max()),
                   "ProfileFnSmallMinUpper": number(fx[fx.TemporalDomain != large_dom].upper_fn_risk.min())})
    prev = md[(md.Model == "Logistic regression") & np.isclose(md.Missingness, 0)].set_index("TemporalDomain").mortality_prevalence
    small_prev = prev[[d for d in small.TemporalDomain]]
    d90 = need[np.isclose(need.true_sensitivity, .90) & np.isclose(need.target_probability, .8)].deaths_stable.iloc[0]
    macros.update({"EssPrevLo": f"{100 * small_prev.min():.1f}", "EssPrevHi": f"{100 * small_prev.max():.1f}",
                   "EssAdmissionsLo": f"{round(d90 / small_prev.max(), -3):,.0f}",
                   "EssAdmissionsHi": f"{round(d90 / small_prev.min(), -3):,.0f}"})
    sens = pd.read_csv(p / "sensitivity_summary.csv").set_index(["n", "rho", "method"])
    macros.update({"SensMultCovHighRho": number(sens.loc[(1000, .90, "multiplier"), "simultaneous_coverage"]),
                   "SensMultCovSmallN": number(sens.loc[(500, .55, "multiplier"), "simultaneous_coverage"])})

    operating = mi / "mimic_class_sensitive_operating.csv"
    if operating.is_file():
        op = pd.read_csv(operating); op = op[np.isclose(op.Missingness, 0)]
        rows = [[domain(r.TemporalDomain), MODEL_SHORT[r.Model], f"{int(r.n):,}", f"{int(r.deaths)}", number(r.sensitivity),
                 number(r.specificity), number(r.alerts_per_1000, 1), number(r.PPV)] for _, r in op.iterrows()]
        table(out / "mimic_operating.tex", ["Interval", "Model", "$n$", "Deaths", "Sens.", "Spec.", "Alerts/1,000", "PPV"], rows, "llrrrrrr")
    else:
        stale = out / "mimic_operating.tex"
        if stale.exists():
            stale.unlink()

    (out / "numbers.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in macros.items()) + "\n")
    audit = {"source_profile": "full", "source_csv_sha256": manifest["output_sha256"],
             "revision_csv_sha256": rev_manifest["output_sha256"],
             "mimic_local_csv_sha256": {q.name: sha(q) for q in sorted(mi.glob("*.csv"))},
             "generated_files": {q.name: sha(q) for q in sorted(out.glob("*.tex"))}}
    (out / "table_provenance.json").write_text(json.dumps(audit, indent=2))
    print(f"Exported {len(audit['generated_files'])} manuscript inputs and {len(macros)} macros")


if __name__ == "__main__":
    export()
