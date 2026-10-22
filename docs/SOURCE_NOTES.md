# Primary-source checks for the major revision

Checked September 27, 2026. These sources support methodological positioning and the
clinical protocol; they do not supply SciGuard experiment results.

- **Gaussian multiplier approximation:** Chernozhukov, Chetverikov and Kato (2013),
  https://arxiv.org/abs/1212.6906 . Independent random vectors may have dependent coordinates.
  Shared within-row transformations therefore do not by themselves require coordinate independence.
- **Later approximation rates:** Chernozhukov, Chetverikov, Kato and Koike, version 2 (2022),
  https://arxiv.org/abs/1912.10529 . Improved rates exist. The revision does not claim that
  its grouped implementation inherits the sharpest modern rate without an assumption check.
- **Learn then Test:** Angelopoulos et al., https://arxiv.org/abs/2110.01052 . Multiple testing
  is a general risk-control framework and can index environments. The exact Holm comparison
  is an instance of that framework, not proof that SAR is a new class of inference.
- **ATC:** Garg et al. (2022), https://arxiv.org/abs/2201.04234 and authors' paper description,
  https://research.google/pubs/leveraging-unlabeled-data-to-predict-out-of-distribution-performance/ .
  ATC predicts target performance from confidence; it is evaluated separately by prediction error.
- **Weighted conformal:** Tibshirani et al. (2019),
  https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html .
  Covariate-shift prediction intervals rely on their likelihood-ratio/weighted-exchangeability setup.
- **PAC-Bayes:** Alquier (2024), https://arxiv.org/abs/2110.11216 . Generalization bounds for
  randomized/aggregated predictors are distinguished from independent holdout assurance.
- **Other simultaneous-band literature:** Degras (2011),
  https://www3.stat.sinica.edu.tw/sstest/j21n4/J21N412/J21N412.html . The general idea of
  simultaneous confidence bands is not restricted to high-dimensional CCK inference.
- **System-level safety precedent:** Dixit et al. (2025),
  https://proceedings.mlr.press/v270/dixit25a.html . Perception calibration coupled to a safe
  planner has a different target from a bound on stand-alone predictive risk.
- **Operational design domain:** Torens et al. (2025),
  https://doi.org/10.1007/s13272-025-00883-6 . Runtime domain monitoring is distinguished
  from this offline environment-wise average-risk report.
- **Safe-region learning:** Hammar et al. (2026), https://arxiv.org/abs/2602.05280 . Their
  causal observation/intervention procedure is not implemented by static SciGuard bands.
- **Bounded-sum inequality:** Hoeffding (1963),
  https://doi.org/10.1080/01621459.1963.10500830 . The finite-sample bounded-loss route uses
  this established inequality and a union bound; no novelty is claimed for it.
- **MIMIC-IV v3.1:** https://physionet.org/content/mimiciv/3.1/ . The patient anchor-year
  interval is not every admission's calendar interval. Admission dates must be aligned to
  the anchor year; age 91 is a top code. Version and data coverage must be checked locally.
  The protocol's extra one-year padding is an analysis choice, not an exact-date claim
  from the documentation. This source is not evidence that the benchmark has been run.

The original bibliography is preserved, with verified additions. Only cited entries enter
the revised PDF. Existing authorship and affiliation information was retained from the
supplied manuscript; those administrative details were not independently reverified.
