# Statistical Assurance Regions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23228565.svg)](https://doi.org/10.5281/zenodo.23228565)

Statistical Assurance Regions (SARs) identify the prespecified deployment environments in
which a frozen predictive pipeline remains below a risk tolerance. **SciGuard** is the
reference Python implementation and reproducibility workflow accompanying the JMLR study.

Open **`SciGuard_JMLR_Reproducible_Analysis_MIMIC_IV.ipynb`** after cloning or downloading the entire repository.
It is the sole current notebook. The manuscript is `manuscript/main_jmlr.tex` and its compiled PDF.

This revision unifies the notebook and package, corrects missingness generation, adds exact
finite-sample comparisons, quantifies reference-risk uncertainty, and replaces the empirical
claims with a fresh full run. The multiplier band is asymptotic and undercovers in some tested
regimes. MIMIC-IV v3.1 has now been executed on credentialed local data; only aggregate benchmark outputs, the frozen protocol, and source-file hashes are included.

## Install and run

```bash
git clone https://github.com/dfmaale/sciguard-sar.git
cd sciguard-sar
python -m pip install -e ".[experiments,notebook]"
python -m unittest discover -s tests -v
python scripts/run_analysis.py --profile smoke
```

The synthetic and WDBC analyses above do not require MIMIC-IV. A complete re-execution of
the canonical notebook, including its MIMIC-IV sections, requires credentialed local data:

```bash
SCIGUARD_RUN_MIMIC=1 MIMIC_IV_PATH=/path/to/mimic-iv python scripts/execute_notebook.py
```

This command executes the full canonical notebook in a fresh in-process IPython shell,
captures its outputs, and exports manuscript tables. It does not need kernel sockets.
For a conventional Jupyter kernel, use `python scripts/execute_notebook.py --backend jupyter`,
or open the notebook in JupyterLab and choose Restart Kernel and Run All Cells. The delivered
execution metadata states which backend was used.

For a quick notebook run, set `SCIGUARD_PROFILE=smoke`. On Windows, a convenient first-cell
setting is `os.environ['SCIGUARD_PROFILE'] = 'smoke'` **before** the PROFILE assignment.
The default is `full`. Full and smoke outputs are separate. Results are recomputed, never
silently read from the old archive. The full run can take several minutes.

For non-notebook reproduction:

```bash
python scripts/run_analysis.py --profile full
python scripts/render_results.py
python scripts/export_manuscript.py
```

Compile the manuscript from its directory with `latexmk -pdf -interaction=nonstopmode main_jmlr.tex`.
The official style file supplied in the original archive is retained. Main tables and numeric
macros are generated from current full-run CSVs; do not hand-edit the generated tables.

## MIMIC-IV

1. Keep your credentialed dataset locally; this archive does not download or contain it.
2. `config/mimic_protocol.json` now contains the frozen v3.1 benchmark settings used for the
   completed run, including the operational 0--1-risk tolerance and prospective rationale.
   Change these settings only for a clearly identified new analysis.
3. Set `MIMIC_IV_PATH` to the directory containing `hosp/patients.csv(.gz)` and
   `hosp/admissions.csv(.gz)` (older `core/` layout is also supported).
4. Set `SCIGUARD_RUN_MIMIC=1` and run the notebook, or use:

```bash
python scripts/run_mimic.py --data /path/to/mimic-iv --config config/mimic_protocol.json
```

The completed credentialed run is stored under `results/mimic_local/` and its decision map
under `figures/mimic_local/`. The temporal protocol aligns each admission to its anchor year,
excludes ambiguous source/deployment overlap, and reports approximate admission intervals.
It includes native missingness, class-imbalance diagnostics, an always-survive baseline, and
one simultaneous family across models/domains/missingness. Patient-level MIMIC-IV data are
not redistributed.

The reviewer-requested secondary specification evaluates false-negative risk with
`tau_fn = 0.20`, equivalent to requiring sensitivity of at least 0.80. Under the primary exact
Clopper-Pearson Bonferroni procedure, 3 of 60 eligible conditions were certified; all three
were histogram-gradient-boosting conditions.

## Current files

| Path | Role |
|---|---|
| `src/sciguard/` | Shared inference, DGP, experiments, clinical protocol and plotting |
| `results/full/` | Full-run simulation/application replications, summaries, config, versions and hashes |
| `results/mimic_local/` | Aggregate credentialed MIMIC-IV v3.1 outputs, frozen config and source hashes |
| `results/smoke/` | Execution checks; not manuscript evidence |
| `figures/full/` | Figures from the current full run |
| `figures/mimic_local/` | Completed MIMIC-IV assurance-map figures |
| `results/revision/` | Supplementary analyses from `scripts/run_revision.py` (power study, exact continuous bounds, failure decomposition, tail calculation, profiles, MIMIC-IV effective sample size) with manifest |
| `figures/revision/` | Figures from the supplementary analyses and the combined pipelines/missingness figure |
| `manuscript/generated/` | CSV-derived tables, numerical macros and provenance |
| `docs/REVISION_RESPONSE.md` | Earlier major-revision response and corrective actions |
| `docs/FINAL_REVIEW_RESPONSE.md` | Response to the second-round minor-revision review |
| `docs/NOTEBOOK_AUDIT.md` | Original-to-revised notebook consistency map |
| `docs/VALIDATION_REPORT.md` | What was executed and what remains unverified |
| `docs/RESULTS_REVISION_NOTES.md` | Notes on the supplementary results revision |
| `docs/MANUSCRIPT_RESTRUCTURE_CHANGES.md` | Summary of the October 2026 manuscript restructuring |
| `SciGuard_Revised_Manuscript.pdf` | Current restructured JMLR-format manuscript (main text and references 27--28 pages; appendices follow the references) |
| `SciGuard_Major_Revision_Response.pdf` | Earlier major-revision response, notebook audit and validation report |
| `SciGuard_Final_Review_Response.pdf` | Point-by-point response to the second-round review |

The standalone notebook download still needs this repository's `src/`, `data/`, and `scripts/`
folders to rerun. Clone or download the complete repository. Restricted clinical data,
patient-level derivatives, and trained clinical models are not included.

To rebuild the response PDFs and audit the final artifacts, install the `reports` extra,
then run `python scripts/build_revision_report.py`, `python scripts/build_final_review_response.py`,
and `python scripts/verify_artifacts.py`.
The response PDF renderer uses DejaVu fonts at `/usr/share/fonts/truetype/dejavu`;
adjust that path on other systems. Manuscript compilation requires LaTeX and latexmk, including the algorithm2e, TikZ/PGFPlots, and Latin Modern packages.

## Citation

Please use the metadata in [`CITATION.cff`](CITATION.cff). The authoritative Revision
release (`v1.0.1`) is archived at [Zenodo](https://doi.org/10.5281/zenodo.23241355).
The [concept DOI](https://doi.org/10.5281/zenodo.23228565) resolves to the latest archived
version.

## License

The software is distributed under the [MIT License](LICENSE). MIMIC-IV remains governed by
its own PhysioNet data-use requirements.
