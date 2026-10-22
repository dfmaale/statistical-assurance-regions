# Results revision notes (October 20, 2026)

This revision responds to the supervisor's request to study how results are written and
presented in the target journal and to strengthen the results. Three published JMLR papers
served as presentation models:

- Gibbs and Candès (2024), Conformal Inference for Online Prediction with Arbitrary
  Distribution Shifts, JMLR 25(162): https://jmlr.org/papers/v25/22-1218.html
- Franc, Prusa, and Voracek (2023), Optimal Strategies for Reject Option Classifiers,
  JMLR 24(11): https://jmlr.org/papers/v24/21-0048.html
- Wang and Qiao (2023), Set-valued Classification with Out-of-distribution Detection for
  Many Classes, JMLR 24(375): https://www.jmlr.org/papers/v24/23-0712.html

The lessons applied are: organize results
around questions, state each answer before the evidence, report validity and efficiency side by
side, use controlled regimes to explain mechanisms, and add a curve over the tuning quantity
(here, the tolerance) so a count at one setting is not read in isolation.

No archived result was changed. The original full-run outputs in `results/full/` are
hash-verified before they are read and are never rewritten. All new numbers in the manuscript
are generated macros or tables; none are typed by hand.

## What changed in the manuscript

Structure

- New abstract that states the findings, including the multiplier's undercoverage.
- Introduction: four research questions (RQ1 to RQ4), a list of principal findings, and
  Table 1 mapping each question to its evidence and metrics.
- Section 6 (evaluation protocol): metrics, an explicit distinction between band coverage and
  containment, and Table 2 listing every inferential route with its guarantee.
- Sections 7 to 10 are organized by research question. Each opens with a "Finding"
  paragraph, then gives the evidence and its limitation.
  - Section 7, RQ1: validity under known risks (primary calibration, adverse regimes).
  - Section 8, RQ2: what determines how much is certified (dependence and sample size,
    risk-margin power study, exact planning, continuous example).
  - Section 9, RQ3: fitted pipelines and missingness mechanisms.
  - Section 10, RQ4: WDBC, the two MIMIC-IV specifications, and tolerance-certification profiles.
- Section 11 adds "Planning the assurance sample"; Section 12 (Conclusion) is rewritten.
- Appendix D adds "Reproduction across computing environments".

New or redesigned figures

- Figure 1: overview figure with the three bound routes written out and corrected notation.
- Figure 3: validity (band coverage and containment with Wilson intervals) and recall
  (replication-level distribution) side by side, same method order and colors.
- Figure 4: adverse regimes, validity and ARR side by side; non-applicable routes marked "n/a".
- Figure 6: risk-margin power study (certification probability and band coverage against n).
- Figure 7: continuous example with the exact Clopper-Pearson Bonferroni grid band.
- Figure 13: tolerance-certification profiles for WDBC and both MIMIC-IV specifications.

New tables

- Table 1 (roadmap), Table 2 (inferential routes), Table 5 (exact planning),
  Table 13 (MIMIC effective sample size), Table 16 (power-study validity),
  Table 17 (exact studentized-tail inflation), Table 20 (anatomy of band failures).

References added or updated: Brown, Cai, and DasGupta (2001); Hall (1992); Franc, Prusa, and
Voracek (2023); Hammar et al. (2026) updated to its IEEE Transactions on Mobile Computing version.

## New findings, in brief (for the supervisor)

1. Validity. Exact Clopper-Pearson Bonferroni covered 0.960 in the primary calibration and
   0.995 in the rare-event design; its lowest coverage over the 48 power-study cells was 0.948.
   The multiplier covered 0.928, 0.780, and 0.910 in the primary, rare-event, and K = 125
   designs, and only 0.610 at n = 250 with K = 100, even though every environment had at least
   12 expected errors. A minimum expected error count is therefore necessary but not sufficient.
2. Why the Gaussian bands undercover at large n. An exact binomial calculation (no simulation)
   shows that the studentized normal upper bound at level alpha/K fails 8.6 to 9.6 times as often
   as nominal at n = 250 with K = 100, and still 1.5 to 1.6 times at n = 8,000. This is the
   known one-sided skewness error of order n^(-1/2), magnified because simultaneous inference
   works far in the tail. It explains the remaining deficit at n = 8,000 for K = 100.
3. What determines how much is certified. The risk margin and the assurance sample size,
   not the choice among valid routes. For exact Bonferroni the certification probability has a
   closed form (equation 9), and the simulation agrees with it within 0.019 in every cell.
   Certifying an environment with margin 0.05 with probability 0.8 needs 422 to 626
   observations (K = 25 to 500), versus 1,243 to 1,843 under the Hoeffding width rule, and
   12,179 to 18,235 for margin 0.01. Exact Holm adds at most 0.102 in certification probability.
   The multiplier's largest gain, 0.337, occurred where its coverage was 0.610.
4. Containment is not validity. Of the 20 exact-Bonferroni coverage failures in the primary
   calibration, 9 missed only truly safe environments and only 1 produced a false certificate.
   This is why containment stays near one while coverage falls.
5. MIMIC-IV false-negative specification. The binding constraint is deaths per domain: with
   K = 72 and alpha = 0.05, a domain needs at least 33 deaths to certify tau_FN = 0.20 even with
   no false negatives; four of the five eligible domains had 5 to 14 deaths. For planning, a
   domain with true sensitivity 0.90 needs about 230 deaths for an 80% chance of certification,
   roughly 20,000 to 38,000 admissions at the mortality prevalence of the small domains.
6. Tolerance-certification profiles. The prespecified WDBC tolerances lie at the onset or on the
   steep part of each model's profile, which is why counts change so much between 0.08 and 0.10.
   For MIMIC 0-1 risk, every curve (including always-survive) reaches one below tau = 0.05; for
   false-negative risk only the 2019-2022 domain is ever certified at a plausible tolerance.
7. Reproducibility across environments. Exact routes reproduced all 500 primary-calibration
   replications exactly on a different operating system and library stack. The multiplier
   reproduced 466 of 500, because its Gaussian draws pass through an eigendecomposition whose
   output can differ across numerical libraries; the regenerated critical values differ from the
   archived ones with standard deviation 0.123 at B = 200 and 0.021 at B = 5,000. All multiplier
   numbers in the paper are from the archived run.

Why this should be sufficient without more datasets or models: each research question now has a
direct answer supported by a controlled experiment, the main negative result is explained
mechanistically rather than only reported, the efficiency comparison is always read with
validity, and the real-data analysis shows what limits certification and how to plan for it.
Adding further datasets or model classes would increase length without answering a question the
paper currently leaves open.

## Files added or changed

- `src/sciguard/revision.py` (new): analyses for all revision sections and the local MIMIC
  operating-characteristics step. `src/sciguard/presentation.py` (new): revision figures.
- `scripts/run_revision.py` (new): runner; writes `results/revision/`, `figures/revision/`, and
  `results/revision/revision_manifest.json` (configuration, package versions, timings, and
  SHA-256 hashes of inputs, outputs, and sources).
- `scripts/export_manuscript.py`: exports the new tables and macros, two-level headers, and
  `tail_inflation.tex`; records revision and MIMIC CSV hashes in `table_provenance.json`.
- `scripts/verify_artifacts.py`: updated for the current notebook name, the completed MIMIC
  benchmark status, Windows-style manifest paths, and the revision manifest.
- `tests/test_revision.py` (new, 11 tests): exact threshold against brute-force Clopper-Pearson
  enumeration, closed-form certification probability, Hoeffding probability at the width rule,
  the planning search, the zero-false-negative bound (0.405 for 14 deaths) and the 33-death
  minimum, monotone tolerance profiles, and the exact studentized-tail inflation.
- `tests/test_scientific_contract.py`: `test_explicit_tolerance_required` previously assumed
  the shipped MIMIC protocol had no tolerance; the protocol is now frozen at tau = 0.35, so the
  test checks unset copies of the configuration instead. All 29 tests pass
  (`tests/test_results.txt`).
- Notebook `SciGuard_JMLR_Reproducible_Analysis_MIMIC_IV_Clean.ipynb`: new Section 13
  (results revision) with outputs; Section 11 text updated to the completed MIMIC run; MIMIC
  execution is now opt-in (`SCIGUARD_RUN_MIMIC=1`) so the notebook runs without credentialed
  data. Earlier cells keep the outputs of the recorded run.
- Manuscript: `manuscript/main_jmlr.tex`, `sciguard_visual_for_maintex.tex`, `references.bib`,
  and `manuscript/generated/*`.

`src/sciguard/mimic.py` is unchanged in this revision. Its hash already differed from the one
recorded in `results/full/run_manifest.json` in the uploaded package, because it was edited for the
credentialed MIMIC run after the archived public/synthetic run; the verification script reports
this explicitly.

## How to reproduce

From the repository root:

```bash
python scripts/run_revision.py          # about 5 minutes, single-threaded
python scripts/export_manuscript.py     # regenerates manuscript/generated/*
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py      # writes docs/artifact_validation.json
cd manuscript && latexmk -pdf main_jmlr   # or: pdflatex, bibtex, then pdflatex three times
```

The compiled manuscript currently has 41 pages, with no LaTeX errors, overfull boxes, or
undefined references.

## Local MIMIC-IV step (credentialed data required)

Section 10.2 contains a boxed author note where a table of deployment specificity, alerts per
1,000 admissions, and PPV at the frozen class-sensitive thresholds will go. These quantities need
patient-level predictions, so they must be computed where the credentialed data are stored:

```bash
export SCIGUARD_RUN_MIMIC=1
export MIMIC_IV_PATH=/path/to/mimiciv/3.1
python - <<'EOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "src")
from sciguard.revision import mimic_class_sensitive_operating
config = json.loads(Path("config/mimic_protocol.json").read_text())
mimic_class_sensitive_operating(Path(os.environ["MIMIC_IV_PATH"]), config, Path("results/mimic_local"))
EOF
python scripts/export_manuscript.py
```

The same step is available in notebook Section 13.4. The function refits the prespecified models
with the archived seed, recomputes the frozen thresholds, and writes
`results/mimic_local/mimic_class_sensitive_operating.csv` only if it reproduces the archived
positive and false-negative counts for all 72 conditions. The export then writes
`manuscript/generated/mimic_operating.tex`, and recompiling replaces the author note with the
table. Share only the aggregate CSV. If you decide not to report operating characteristics,
delete the `\IfFileExists{generated/mimic_operating.tex}` block in Section 10.2 before submission.

## Remaining author actions

1. Run the local MIMIC step above, or remove the author-note block.
2. Read the new Finding paragraphs and the RQ framing and confirm that every claim matches your
   intent.
3. Complete the acknowledgments (left as a LaTeX comment) and the funding and disclosure details.
