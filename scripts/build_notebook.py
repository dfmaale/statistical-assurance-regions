#!/usr/bin/env python3
"""Build the sole canonical notebook from the reviewed workflow specification."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# SciGuard: complete reproducible analysis and MIMIC-IV protocol
**Canonical notebook:** `SciGuard_JMLR_Reproducible_Analysis_MIMIC_IV.ipynb`  
**Major revision:** September 27, 2026 | Faustus Domebale Maale and Hui Yi

This notebook is the single current workflow. It contains the inferential calibration,
nested model refits, corrected missingness mechanisms, WDBC stress test, sample-size /
dependence / grid / bootstrap ablations, continuous extension, pilot-design example,
and credentialed MIMIC-IV execution path. All statistical functions are imported from
`src/sciguard`; none are redefined later in the notebook.

**Evidence status:** synthetic and WDBC analyses are reproducible using the bundled data.
MIMIC-IV is **pending actual credentialed-data execution**. A successful synthetic loader
test does not constitute clinical benchmark evidence. The revision is an applied-methods
framework, not a claim of a new Gaussian-bootstrap theorem or clinical safety.
""")
md(r"""
## 1. Run controls and reproducibility
Unzip the entire package and open this notebook from its root. Install with
`python -m pip install -e ".[experiments,notebook]"`. Restart the kernel and run all cells.

`SCIGUARD_PROFILE=full` (default) recomputes every public/synthetic section. Use
`SCIGUARD_PROFILE=smoke` for a short execution check; smoke results have their own directory
and are never exported to manuscript tables. Full runs use 500 calibration replications,
200 nested refits, 100 missingness refits, 20,000 independent reference rows per refit,
2,000 bootstrap draws (5,000 for WDBC), and a fixed seed. Training is single-threaded
for reproducibility. Runtime depends on hardware.

Original notebooks and results are retained under `archive/pre_revision/` and are never
silently loaded by this workflow. The root notebook's displayed outputs are from its
recorded execution; rerunning replaces them with current outputs.
""")
code(r"""
from pathlib import Path
import os, sys, json, hashlib, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / 'src/sciguard/experiments.py').is_file()
             and (p / 'data/simulation_true_risks.csv').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Open the notebook inside the unzipped SciGuard_JMLR_Target folder.')
sys.path.insert(0, str(ROOT / 'src'))
import sciguard
from sciguard.experiments import AnalysisRun, RunConfig, SECTIONS, LABELS
from sciguard import certify, certify_overlapping_groups, continuous_upper_envelope
from sciguard.plotting import (schematic, calibration_figure, sensitivity_figure,
    nested_figure, missingness_figure, wdbc_figure, continuous_figure, stress_figure)

PROFILE = os.environ.get('SCIGUARD_PROFILE', 'full')
if PROFILE not in {'full', 'smoke'}:
    raise ValueError('SCIGUARD_PROFILE must be full or smoke')
CONFIG = RunConfig() if PROFILE == 'full' else RunConfig.smoke()
THREAD_LIMIT = threadpool_limits(limits=1)
run = AnalysisRun(ROOT, CONFIG)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
print('SciGuard', sciguard.__version__, '| profile:', PROFILE)
display(pd.DataFrame([run.manifest['config']]).T.rename(columns={0: 'Value'}))
""")
md(r"""
## 2. Target, assumptions, and guarantees
Condition on the training/design data and a frozen pipeline $g$. For a prespecified
environment $e$, write $R_g(e)=E[L_e]$, $0\leq L_e\leq1$, and
$\mathcal A_\tau=\{e:R_g(e)\leq\tau\}$. A simultaneous band $U_e$ yields the inner estimate
$\widehat{\mathcal A}_\tau=\{e:U_e\leq\tau\}$.

The fundamental event is **simultaneous risk coverage**, not only containment at one
tolerance. If every $R_e\leq U_e$, then containment holds for **every** tolerance at once.
Thus inspecting several tolerances on the same fixed band does not invalidate this event.
Operational tolerances should nevertheless be prospectively justified. Changing the loss,
model, environment family, or certification method after seeing assurance outcomes needs
independent data or multiplicity control. Exact Holm here is a test at a fixed tolerance,
not a confidence band constructed uniformly over tolerances.

| Method | Scope |
|---|---|
| Multiplier | Asymptotic simultaneous band; requires appropriate row independence, nondegenerate population variances, and an adequate Gaussian approximation. Finite bootstrap draws add Monte Carlo error. |
| Normal Bonferroni | Multiplicity-corrected normal approximation; not finite-sample exact. |
| Exact Bonferroni | One-sided Clopper-Pearson bounds at $\alpha/K$ for binary losses; finite-sample familywise control. |
| Hoeffding | $\min(1,\widehat R_k+\sqrt{\log(K/\alpha)/(2n_k)})$; finite-sample control for independent bounded observations. |
| Exact Holm | Binomial lower-tail tests of $H_{0k}:R_k\geq\tau$, adjusted by Holm; finite-sample containment at a fixed tolerance. |
| Pointwise normal | Diagnostic comparator only; no familywise guarantee. |

For the multiplier route, zero empirical variance produces upper bound 1, never a false
zero-risk certificate. Exact binomial bounds can still certify a zero-error column if the
sample is sufficiently large. All moments use `ddof=0` consistently.
""".replace('\\\\','\\'))
code(r"""
fig = schematic(run)
display(fig); plt.close(fig)
toy_losses = np.zeros((100, 5))
display(pd.DataFrame({
    'Multiplier upper': certify(toy_losses, .10, n_boot=200, rng=1).upper,
    'Exact binomial upper': certify(toy_losses, .10, method='exact_bonferroni').upper
}))
""")
md(r"""
### 2.1 Why shared transformations do not violate row independence
Let $Z_i$ be independent source observations and $V_i$ independent row-specific stress
randomness. With the design fixed, each vector
$\mathbf L_i=(h_1(Z_i,V_i),\ldots,h_K(Z_i,V_i))$ is a measurable function of one independent
row. Therefore the vectors are i.i.d.; their coordinates can be arbitrarily dependent,
including identical columns. One Gaussian multiplier is shared across each row's columns.

This argument fails if transformations normalize using the **assurance batch**, if repeated
patients are treated as independent, or if a shared random nuisance is not conditioned on.
The revised MAR/MNAR generators use fixed population-calibrated intercepts and row-local
randomness. The conditional-covariance sampler draws from the same Gaussian law as direct
row multipliers; this computational equivalence does not make the inferential approximation exact.

### 2.2 Evaluation metrics
With known truth: simultaneous coverage, containment, ARR (fraction of safe environments
certified), and UCP (unsafe fraction among certified environments). UCP is zero for an empty
certified set; ARR is undefined (`NaN`) if no environment is truly safe. Always report the
certified count so vacuous containment is visible. Confidence intervals for empirical
coverage/containment use Wilson intervals over independent Monte Carlo replications.
""".replace('\\\\','\\'))
md("""
## 3. Calibration with exactly known risks
The supplied 25-cell risk surface is a calibration-only design. Correlated Bernoulli losses
are generated through a Gaussian copula; the model-fitting study is separate. Latent Gaussian
correlation is not the same as observed binary-loss correlation. No archived numerical targets
are imposed as assertions, and unfavorable results remain in the output.
""")
code(r"""
truth = pd.read_csv(ROOT / 'data/simulation_true_risks.csv')
assert len(truth) == 25 and truth.Risk.between(0, 1).all()
display(truth)
calibration_replications, calibration_summary = run.section('calibration', SECTIONS['calibration'])
display(calibration_summary[['method', 'simultaneous_coverage', 'simultaneous_coverage_lo',
    'simultaneous_coverage_hi', 'containment', 'ARR', 'ARR_mcse', 'UCP', 'n_certified']].round(4))
fig = calibration_figure(run); display(fig); plt.close(fig)
""")
md("""
## 4. Nested fitted-pipeline experiment
Every replication draws fresh training data, fits logistic regression and histogram gradient
boosting, then uses separate source-calibration, assurance, and reference samples. Models share
the same generated rows and stress randomness for a paired comparison. The protected first
feature and all preprocessing are fixed before assurance. This changes the earlier experiment;
the revised numbers must not be compared as if only the random seed changed.

**Reference uncertainty:** 20,000 reference observations improve on the former 1,500, but their
means are still not exact true risks. Simultaneous two-sided exact binomial intervals use total
reference error 0.01 across both models in each refit. Cells whose intervals cross the tolerance
are ambiguous. `containment_lower` certifies containment only when all selected cells are
reference-confirmed safe; `containment_upper` rules it out only if a selected cell is
reference-confirmed unsafe. These are bounds on a containment indicator on the reference
coverage event, not confidence limits for a population coverage probability. Their averages
are reported alongside point-reference metrics and Wilson intervals. Reference coverage is
per refit, not simultaneous over every refit in the experiment.
""")
code(r"""
nested_replications, nested_summary = run.section('nested', SECTIONS['nested'])
display(nested_summary[['Model', 'method', 'reference_containment', 'reference_containment_lo',
    'reference_containment_hi', 'containment_lower', 'containment_upper',
    'reference_ARR', 'reference_UCP', 'reference_n_certified', 'ambiguous_environments']].round(4))
display(pd.read_csv(run.out / 'nested_fit_summary.csv').round(4))
fig = nested_figure(run); display(fig); plt.close(fig)
""")
md("""
### 4.1 ATC is evaluated as a performance predictor
Average Thresholded Confidence with maximum-probability scores is calibrated on independent
labeled source data. It uses target confidence scores to predict target risk. We report MAE,
bias, and RMSE against the independent reference estimates. ATC is not placed in a table of
formal certification guarantees. Tied scores use a consistent strict threshold in calibration
and deployment. This is an ATC-MC implementation, not a newly corrected certifier.
""")
code(r"""
display(pd.read_csv(run.out / 'atc_prediction_summary.csv').round(4))
""")
md("""
## 5. Corrected missingness mechanisms
All mechanisms keep feature 0 observed; severity is the missing fraction among the other
nine eligible features. MAR depends only on observed feature 0. MNAR depends on the feature
being hidden. Fixed logistic intercepts are calibrated under the known simulation distribution,
not renormalized on each assurance batch. Structured failure hides features 1-3 together while
maintaining the same marginal severity for all eligible features. The label equation and model
are unchanged across mechanisms, and masks are nested as severity increases.

The reference interval budget is 0.01 across all four mechanisms within each refit. MNAR and
block-failure scenarios are sensitivity assumptions; the results do not identify a real clinical
missingness mechanism. Realized missingness is exported for every environment.
""")
code(r"""
missingness_replications, missingness_summary = run.section('missingness', SECTIONS['missingness'])
display(missingness_summary[['Mechanism', 'method', 'reference_containment',
    'containment_lower', 'containment_upper', 'reference_ARR', 'reference_UCP',
    'ambiguous_environments']].round(4))
rates = pd.read_csv(run.out / 'missingness_environment_detail.csv')
display(rates.groupby(['Mechanism', 'Missingness']).realized_missingness.mean().unstack().round(4))
fig = missingness_figure(run); display(fig); plt.close(fig)
""")
md("""
## 6. WDBC: three frozen models and grid resolution
The 65/35 stratified split (random state 42), train-only imputation/scaling, and shared feature
noise/missingness are applied to logistic regression, Extra Trees, and histogram gradient
boosting. This remains a small **semisynthetic stress test**, not a natural hospital/temporal
shift study. No ground-truth ARR, UCP, or repeated-sampling coverage is inferred from this
single real-data split.

Each model/design is a separate assurance family. Selecting the best model after comparing
these separate bands is not jointly protected. The nonuniform 36-cell primary grid and the
equally spaced 6-by-6 grid contain different missingness values; equal counts do not make
them identical designs. Grid fractions are not continuous areas.
""")
code(r"""
wdbc_detail, wdbc_summary = run.section('wdbc', SECTIONS['wdbc'])
display(wdbc_summary[wdbc_summary.design == 'primary'][['Model', 'method', 'n',
    'nominal_risk', 'certified_008', 'certified_010']].round(4))
display(wdbc_summary[wdbc_summary.design != 'primary'][['design', 'method', 'K',
    'certified_008', 'certified_fraction', 'critical']].round(4))
fig = wdbc_figure(run); display(fig); plt.close(fig)
""")
md("""
## 7. Sample size, dependence, and the cost of multiplicity
The sensitivity study compares all certification methods on the same losses and reports
**paired** ARR differences, with Monte Carlo standard errors, relative to normal Bonferroni,
exact Bonferroni, and exact Holm. Any efficiency gain must be read alongside coverage and
containment. A wider certified region is not a valid improvement if its guarantee fails.
""")
code(r"""
sensitivity_replications, sensitivity_summary = run.section('sensitivity', SECTIONS['sensitivity'])
display(sensitivity_summary[['n', 'rho', 'method', 'simultaneous_coverage', 'containment',
    'ARR', 'UCP']].round(4))
display(pd.read_csv(run.out / 'paired_efficiency_gains.csv').round(4))
fig = sensitivity_figure(run); display(fig); plt.close(fig)
""")
md("""
## 8. Bootstrap Monte Carlo error, runtime, and adverse regimes
For each fixed loss matrix, repeated seeds at B = 200, 1,000, and 5,000 measure bootstrap
critical-value variability and inference time. This is a timing experiment, not a guarantee
that one bootstrap size suffices in all settings. The direct sampler costs O(B n K); the
covariance route costs O(n K² + K³ + B K²), with different memory tradeoffs.

Additional studies use bounded continuous Beta losses, rare binary errors, and a larger
125-environment family. Exact binomial methods apply only to binary losses. Bounded loss
does not imply that the Gaussian approximation is accurate when events are rare. Adversarial
adaptive shifts, unbounded heavy-tailed losses, and high-dimensional continuous environment
geometry are not claimed to be addressed by these stress tests.
""")
code(r"""
bootstrap_replications, stress_replications = run.section('bootstrap_and_stress', SECTIONS['bootstrap_and_stress'])
display(pd.read_csv(run.out / 'bootstrap_runtime_summary.csv').round(5))
display(pd.read_csv(run.out / 'stress_summary.csv')[['scenario', 'method',
    'simultaneous_coverage', 'simultaneous_coverage_lo', 'simultaneous_coverage_hi',
    'containment', 'ARR']].round(4))
fig = stress_figure(run); display(fig); plt.close(fig)
""")
md(r"""
## 9. Continuous SAR and the Lipschitz requirement
For a known global bound $|R(e)-R(e')|\le L_R\|e-e'\|$, the envelope
$U^{\mathrm{cont}}(e)=\min_k\{U_k+L_R\|e-e_k\|\}\wedge1$ covers all continuous risks
on the grid-coverage event. This is a standard consequence of simultaneous coverage and
smoothness, not a new bootstrap theorem.

The bilinear example $R(m,s)=.24+.12m+.10s+.05ms$ on $[0,.4]\times[0,.6]$ has the analytic
bound $L_R=\sqrt{.15^2+.12^2}$. Dense-mesh containment/recall are numerical diagnostics,
not a numerical proof over every point. The proof supplies continuous coverage when the
bound is valid. Files also record sensitivity to half and twice the analytic bound; the
half-bound scenario is not asserted valid.

**Do not estimate a global Lipschitz upper bound by the maximum observed grid slope.**
Even independent pilot slopes do not control unseen behavior between grid points. Without
external smoothness information, report finite-grid results. If a separately justified random
upper bound fails with probability delta, the combined coverage budget is alpha + delta.
""".replace('\\\\','\\'))
code(r"""
continuous_replications, continuous_summary = run.section('continuous', SECTIONS['continuous'])
display(continuous_summary[['method', 'grid_coverage', 'containment', 'ARR', 'UCP']].round(4))
fig = continuous_figure(run); display(fig); plt.close(fig)
""")
md(r"""
## 10. Environment design and independent pilot refinement
Choose the scientific loss and deployment axes; set ranges from external knowledge;
freeze preprocessing and the candidate family; choose sample size for a meaningful margin;
use a pilot to prioritize a finite family; then evaluate that frozen family on **independent**
assurance rows. The example selects grid points near a pilot-estimated boundary plus coordinate
extremes. Its exact-binomial guarantee is conditional on the selected family, not on the
entire continuous rectangle.

The Hoeffding planning inequality $n\ge\log(K/\alpha)/(2\epsilon^2)$ ensures the additive
bound width is at most epsilon. It does not ensure a desired recall or a desired probability
of certification, which also depends on the true margin and sampling variation.

Repeated deployment checks are not time-uniform by default. A simple, conservative extension
allocates $\alpha_t=\alpha/[t(t+1)]$ to each round with valid conditional finite-sample
inference and fresh assurance data. The union bound then controls any false certification
over all rounds. This is standard alpha spending, not a newly optimal sequential SAR method.
""".replace('\\\\','\\'))
code(r"""
pilot_replications, planning_table = run.section('pilot_design', SECTIONS['pilot_design'])
display(pd.read_csv(run.out / 'pilot_design_summary.csv').round(4))
display(planning_table)
""")
md(r"""
## 11. MIMIC-IV protocol: prerequisites and temporal corrections
The attached archive contains no credentialed MIMIC tables and no successful MIMIC results.
The earlier notebook has a recorded `FileNotFoundError`. This section is a fully implemented
**local execution protocol**, not evidence of completed external validation.

Use the actual dataset version in `config/mimic_protocol.json`, supply a prospectively
justified `tau` and `tau_rationale`, and set `MIMIC_IV_PATH` and `SCIGUARD_RUN_MIMIC=1`.
The example configuration is for v3.1; its tolerance is deliberately unset. No universal
clinical tolerance is inferred from an example value such as 0.35.

**Cohort:** first eligible adult hospitalization per patient; in-hospital mortality outcome;
admission-table demographic/admission variables; train-only preprocessing; 0.5 decision
threshold. Patient identifiers, mortality flags, discharge/death dates, and length of stay
are excluded from predictors. Availability of the admission-table fields at the intended
decision time still requires an author/data audit; this is a limited clinical baseline.

**Temporal design:** `anchor_year_group` describes the patient's anchor year, not every
admission. The loader adds `year(admittime) - anchor_year` to its endpoints, uses configurable
one-year padding for approximate year alignment, and restricts intervals to the declared
dataset coverage. Source intervals end by the cutoff; deployment intervals begin after it;
overlapping intervals are excluded. Deployment labels are approximate admission intervals
and may overlap one another in calendar time. They are not exact admission years or sites.
Top-coded age 91 is retained as a category/provenance indicator rather than interpreted as
exact age or used to exclude the oldest patients.

Official date guidance: https://physionet.org/content/mimiciv/3.1/ (accessed September 27, 2026).
The one-year padding is a conservative analysis choice, not an official exact-date guarantee.
""")
md(r"""
### 11.1 Dependence, class imbalance, and natural missingness
Within each domain, missingness levels reuse the same patients. The common-row influence
vector is $\widehat\psi_{i,d,m}=G_{id}(L_i^{(m)}-\widehat R_{d,m})/\widehat p_d$.
Overlapping groups and unequal counts are supported without truncating or inventing pairings.
The multiplier ratio-estimator route assumes nonvanishing group probabilities as well as
appropriate row sampling. Exact group-conditional binomial and Hoeffding bands are supplied.

All models, missingness levels, and domains belong to **one** simultaneous family. Small
domains remain visible and unvalidated. An always-survive baseline, mortality prevalence,
sensitivity/specificity, balanced accuracy, AUROC, and AUPRC reveal when low 0-1 risk is
uninformative under class imbalance. These diagnostic metrics do not inherit the 0-1-risk
certification guarantee. Native feature missingness is reported separately from added MCAR
and total missingness. Adding MCAR does not establish robustness to clinical MAR/MNAR.

Only aggregate result tables are saved. No row-level cohort, patient identifiers, or fitted
model is exported. Source file hashes and the frozen configuration support local provenance.
One admission per patient removes repeated-patient rows; it does not prove exchangeability
or eliminate hospital/time clustering. Actual-data validation remains required.
""".replace('\\\\','\\'))
code(r"""
from sciguard.mimic import run_mimic
MIMIC_CONFIG_PATH = ROOT / 'config/mimic_protocol.json'
MIMIC_CONFIG = json.loads(MIMIC_CONFIG_PATH.read_text())
RUN_MIMIC = os.environ.get('SCIGUARD_RUN_MIMIC', '0') == '1'
MIMIC_ROOT = Path(os.environ.get('MIMIC_IV_PATH', str(ROOT / 'data/mimic-iv'))).expanduser()

if RUN_MIMIC:
    if not MIMIC_ROOT.is_dir():
        raise FileNotFoundError('Requested MIMIC execution, but MIMIC_IV_PATH is not an existing dataset directory.')
    mimic_tables, mimic_manifest = run_mimic(MIMIC_ROOT, MIMIC_CONFIG, ROOT / 'results/mimic_local')
    run.manifest['external_benchmark'] = 'completed_credentialed_data'
    display(mimic_tables['mimic_cohort_flow'])
    display(mimic_tables['mimic_results'].round(4))
    display(mimic_tables['mimic_diagnostics'].round(4))
    display(mimic_tables['mimic_natural_missingness'].round(4))
    from sciguard.plotting import mimic_decision_map
    mimic_figure = mimic_decision_map(mimic_tables['mimic_results'], ROOT / 'figures/mimic_local')
    display(mimic_figure); plt.close(mimic_figure)
else:
    run.manifest['external_benchmark'] = 'pending_credentialed_data_and_prespecified_tolerance'
    display(Markdown('**MIMIC-IV: NOT RUN.** Public/synthetic analyses continue. '
        'No archived or synthetic fixture results are substituted for this benchmark.'))
run.flush()
""")
md("""
## 12. Integrity checks, manuscript export, and interpretation
Assertions below check structural consistency and data provenance, not agreement with a
preferred numerical answer. Manuscript tables/macros are generated directly from the
completed full-run CSVs. Bootstrap seed changes can change numerical results; versions and
random seeds are recorded. The `archive/` results are historical and do not feed current tables.

The software checks and synthetic MIMIC fixture test are available through
`python -m unittest discover -s tests -v`. The real MIMIC data path cannot be validated by a
synthetic fixture. An absent external benchmark remains an open publication limitation.
""")
code(r"""
expected_sections = set(SECTIONS)
assert set(run.manifest['sections']) == expected_sections
assert all(v['status'] == 'completed' for v in run.manifest['sections'].values())
assert len(calibration_replications) == CONFIG.calibration_reps * 6
assert len(nested_replications) == CONFIG.nested_reps * 2 * 5
assert len(missingness_replications) == CONFIG.missingness_reps * 4 * 3
assert (nested_replications.containment_lower <= nested_replications.reference_containment).all()
assert (nested_replications.reference_containment <= nested_replications.containment_upper).all()
assert wdbc_summary['n'].eq(200).all()
assert wdbc_detail.upper.dropna().between(0, 1).all()
assert set(wdbc_summary.Model) == {'Logistic regression', 'Extra Trees', 'HistGradientBoosting'}
run.finish()
if PROFILE == 'full':
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/export_manuscript.py')],
        cwd=ROOT, text=True, capture_output=True, check=True)
    print(completed.stdout.strip())
else:
    print('Smoke output is separate; manuscript tables were not changed.')
display(pd.DataFrame(run.manifest['sections']).T)
print('External benchmark status:', run.manifest['external_benchmark'])
print('Current-run CSV files:', len(run.manifest['output_sha256']))
print('All public/synthetic sections completed. This is not a declaration of submission readiness.')
""")
md("""
## 13. Source and revision map
- Main theory and exact/asymptotic bounds: `src/sciguard/core.py`, `groups.py`, `baselines.py`.
- Corrected DGP and MAR/MNAR mechanisms: `simulation.py`.
- All study definitions and seeds: `experiments.py`.
- Clinical cohort, interval alignment, and aggregate exports: `mimic.py`.
- Figures: `plotting.py`; manuscript tables: `scripts/export_manuscript.py`.
- Full theoretical argument and interpretation: `manuscript/main_jmlr.tex` / PDF.
- Point-by-point reviewer response and remaining evidence gaps: `docs/REVISION_RESPONSE.md`.

Primary methodological references include CCK (2013, Gaussian multiplier approximation),
Hoeffding (1963), Clopper-Pearson (1934), Holm (1979), Learn then Test (Angelopoulos et al.,
2025), and ATC (Garg et al., 2022). Bibliographic details and verified links appear in the
manuscript and `docs/SOURCE_NOTES.md`.
""")
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"sciguard":{"canonical":True,"revision":"2026-09-27","original_notebook":"archive/pre_revision/SciGuard_JMLR_Reproducible_Analysis_MIMIC_IV.ipynb"}})
nbf.validate(nb)
path=ROOT/"SciGuard_JMLR_Reproducible_Analysis_MIMIC_IV.ipynb"
nbf.write(nb,path)
print(path, "cells:",len(cells))
