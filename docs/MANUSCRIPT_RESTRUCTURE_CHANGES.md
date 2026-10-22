# Manuscript restructuring (October 2026)

The manuscript in `manuscript/` was restructured to follow a standard statistical and JMLR layout. Every reported number still comes from `manuscript/generated/`, and no results changed.

## Structure

The sections are now:

1. Introduction, with 1.1 Related Work in prose
2. Problem Formulation and Sampling Design
3. Inference for Assurance Regions
4. Simulation Studies
5. Real Data Analyses (WDBC; MIMIC-IV)
6. Discussion
7. Conclusion

These are followed by the declarations, then References, then Appendices A–F.

Other structural changes:
- The research-question section titles, the "Finding." paragraphs, and the bold run-in paragraph headings are removed.
- The main text and references fit in 27–28 pages.

## Tables and figures

- **Converted to prose:** tables that contained no numerical results, namely the research-question roadmap, the inferential-procedure summary, and the provenance table.
- **Merged into paneled tables:**
  - calibration with adverse regimes;
  - nested refits with missingness mechanisms;
  - MIMIC-IV effective sample size with operating characteristics.
- **New combined figure:** the nested-refit and missingness figures are now one three-panel figure, `figures/revision/pipelines_missingness.pdf`. It is drawn from the archived CSVs by `src/sciguard/combined_figures.py`.
- **Combined MIMIC-IV maps:** the two assurance maps now form one figure.
- **Removed or moved:** the workflow figure is removed. The adverse-regime and sensitivity figures and the continuous example are now in the appendices.

## Naming and files

- The procedure is now called MiSCert (missingness-and-shift certification), defined by the `\method` macro. The Python package name `sciguard` is unchanged.
- `sciguard_preamble_for_maintex.tex` and `sciguard_visual_for_maintex.tex` are replaced by `preamble_figures.tex` and `figure_overview.tex`.

## Patient-level dependency

`manuscript/generated/mimic_operating.tex` (Table 6, Panel B) requires the credentialed patient-level data. It is produced locally by the notebook function `mimic_class_sensitive_operating`. Until it is added, the manuscript shows Panel A and an author note.
