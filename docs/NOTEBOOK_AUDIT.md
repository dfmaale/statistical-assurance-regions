# Canonical notebook consistency audit

Source audited: `SciGuard_JMLR_Reproducible_Analysis_MIMIC_IV.ipynb` from the uploaded ZIP.
Its original bytes are retained under `archive/pre_revision/`.

## Material findings in the original

1. Multiple top-level notebooks and two main manuscript sources competed for authority.
2. Core functions were copied into the notebook and differed from the package in variance
   normalization, input validation, degenerate-column behavior and output attribute names.
3. Later cells redefined `shift_X`, `missing_mask` and other helpers with different signatures.
4. The original MIMIC execution cell contained a saved `FileNotFoundError`; later package/status
   cells were unexecuted. The ZIP contained no completed MIMIC result CSV.
5. Late simulations silently loaded historical outputs unless an extension flag was set.
6. Section numbering skipped 5, duplicated 9, and declared analysis complete before later analyses.
7. The MAR driver was potentially masked, and per-batch probability rescaling linked rows.
8. Nested reference risks from 1,500 observations were used as if their threshold classifications
   were exact; no reference-risk confidence intervals were supplied.
9. ATC appeared alongside certification methods despite having no matching familywise guarantee.
10. The grouped package routine could certify a zero-error group with a zero uncertainty bound.
11. MIMIC grouping used the patient anchor period without admission-year alignment; low-count
    domains could disappear from the report; its illustrative tolerance lacked clinical justification.
12. Historical figures and repeated manuscript figures did not always have identical bytes.
13. The notebook and manuscript carried independent numerical targets and archived assertions,
    making it possible for figures, tables and computation to drift apart.

## Original-to-current coverage map

| Original content | Current canonical location | Treatment |
|---|---|---|
| Imports, paths and FAST_TEST | Section 1 | Explicit full/smoke configuration; portable root discovery |
| Core inference and repeated helper definitions | Section 2 / shared package | Unified implementation; finite-sample methods added |
| True risk surface and copula calibration | Section 3 | Recomputed; original data retained |
| ARR violin, simultaneous coverage and intervals | Section 3 | Rebuilt from current replication rows |
| Nested refits and model variability | Section 4 | 200 refits, shared model comparisons, reference intervals |
| ATC | Section 4.1 | Separate risk-prediction evaluation |
| Alternative missingness mechanisms | Section 5 | Corrected MAR/row locality; 100 refits and realized rates |
| WDBC single-model and model comparisons | Section 6 | Consolidated three-model study; exact baselines |
| Grid-resolution study | Section 6 | Nonuniform primary versus uniform grids distinguished |
| Sample-size and correlation sensitivity | Section 7 | Six methods; paired efficiency-gain estimates |
| Bootstrap sampler and complexity | Sections 2.1 and 8 | Shared implementation plus timing/seed/B study |
| Continuous SAR | Section 9 | Analytic bound, full rerun, figure and sensitivity exports |
| Practitioner environment-design guidance | Section 10 | Independent-pilot experiment and sample-size planner |
| MIMIC loading and execution | Section 11 | Admission-aligned local protocol; explicit pending status |
| Package check and reproducibility checks | Sections 1 and 12 | Imports must succeed; end-of-run structural/hash checks |
| Session and output information | Section 12 | Manifest with configuration, versions, times and hashes |
| Old external benchmark variants | Historical archive | Preserved; not falsely combined with current MIMIC evidence |

## What “complete” means here

The canonical notebook covers every study previously in the MIMIC-IV notebook and the
review-driven additions listed above, using one package implementation and traceable current
outputs. It is complete as an executable public/synthetic workflow plus a credentialed-data
protocol. It is **not** a completed real-MIMIC empirical study. Keeping that distinction is
part of consistency, not an omission hidden by loading a placeholder table.

The ZIP, not a notebook file alone, is the runnable deliverable: the notebook requires the
bundled source modules, data and export scripts. Standalone notebook viewing is supported.
