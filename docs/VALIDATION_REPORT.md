# Revision validation report

## Scope and outcome

The public-data and synthetic analysis is executed end to end, and the credentialed
MIMIC-IV v3.1 benchmark has now also been completed locally. The canonical notebook imports
the shared package used by the command-line analyses and contains the recorded MIMIC aggregate
outputs. No patient-level MIMIC data are bundled.

## Scientific checks

The 29-test suite passes: 18 scientific-contract tests and 11 results-revision tests
(`tests/test_revision.py`). It checks exact-binomial coverage and Holm
familywise error by count enumeration, rare/zero-event behavior, shared Gaussian multipliers,
unequal independent samples, overlapping groups, confidence ties, row-local missingness,
reference-interval boundaries, continuous envelopes, alpha spending and the MIMIC loader.
The synthetic MIMIC end-to-end fixture remains software validation only. Separately, the
credentialed v3.1 run produced aggregate cohort, result, diagnostic, missingness and manifest
outputs under the frozen protocol.

The full configuration uses 500 calibration replications, 200 independently fitted nested
replications, 100 mechanism replications, 300 replications per sensitivity setting and
200 continuous-envelope replications. Independent reference samples have 20,000 rows.
Monte Carlo uncertainty and uncertainty in reference-risk classification are retained.

In the archived full run (`results/full/`), the main multiplier coverage estimate is 0.928
(nominal target 0.95); the rare-event stress estimate is 0.780. Exact Bonferroni estimates are
0.960 and 0.995 respectively. (An earlier version of this report quoted 0.932 and 0.775, which
do not match the archived outputs used by the manuscript.) These findings
are reported consistently, including in the abstract, rather than hidden by favorable examples.

## Execution and reproducibility

Both the full command-line workflow and the canonical notebook were run. Notebook execution
uses a fresh Python process with an in-process IPython shell, executing cells in order and
capturing their actual text and image output. Kernel socket binding is unavailable in this
runtime, so standard Jupyter-kernel execution could not be validated here. A standard nbclient
backend is included for environments that permit kernel sockets. No outputs were fabricated.

The final notebook metadata identifies its execution backend, completion state and duration.
The manifest records configuration, package versions, section timing, source hashes and
result-file hashes. Tables and numeric manuscript macros are generated from current CSVs;
table_provenance.json records their input hashes. The manuscript is compiled from LaTeX.
Scientific figures and document pages are rendered and visually checked.

The ZIP contains the source, notebook, tests, data, results, figures, manuscript, response,
protocol and historical archive. Use the ZIP for execution; the standalone notebook requires
its bundled modules and data. Requirements record the tested environment, not a guarantee
that every operating system or future dependency version behaves identically.

## Remaining requirements

The completed MIMIC-IV run should not be interpreted as clinical validation. The operational
0--1-risk tolerance is permissive and the always-survive baseline is also certified, so future
clinical work should prospectively specify a task-specific or class-sensitive loss, decision
threshold and tolerance. The approximate calendar-interval padding remains an explicit analysis
choice requiring sensitivity analysis. Broader external datasets and clinical feature-timing
audits are still needed before any deployment-suitability claim.

The revision clarifies established theory and adds useful evidence; it does not supply a new
minimax result or guarantee that the contribution meets JMLR's novelty standard. Author,
affiliation, funding, disclosure and public-code-release details still require author verification.

## Results revision (October 20, 2026)

The revision analyses are produced by `scripts/run_revision.py` and stored with their own
manifest in `results/revision/`; the archived full-run outputs are hash-verified before use and
are not rewritten. `scripts/verify_artifacts.py` checks the notebook, the archived and revision
manifests (outputs, inputs, and sources), the generated manuscript inputs, and the LaTeX log; its
report is `docs/artifact_validation.json`. The compiled manuscript has no LaTeX errors, overfull
boxes, or undefined references. The notebook's Section 13 cells were executed against the
completed revision outputs; earlier cells keep the outputs of the recorded run. Regenerating
the primary calibration on a second platform reproduced every exact-route replication; the
multiplier reproduced 466 of 500 replications because its Gaussian draws depend on the numerical
library (Appendix D of the manuscript). See `docs/RESULTS_REVISION_NOTES.md` for the change log.

