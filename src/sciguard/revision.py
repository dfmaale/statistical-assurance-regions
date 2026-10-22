"""Revision-5 analyses: exact-route power, tolerance profiles, and MIMIC summaries.

This module adds analyses requested during the results revision without
changing any function used by the original full run.  The original modules
(`core`, `experiments`, `simulation`, ...) and their recorded SHA-256 hashes in
`results/full/run_manifest.json` are therefore unchanged.

Contents
--------
* Analytic properties of the exact one-sided Clopper--Pearson Bonferroni route:
  the largest certifiable error count, per-environment certification
  probability, the stable minimal assurance sample size for a target
  certification probability, and the zero-error bound.
* A targeted simulation of certification probability versus sample size and
  risk margin (exact Bonferroni, exact Holm, Hoeffding, multiplier).
* An exact-Bonferroni version of the continuous example: the first
  (prespecified, not selected) replication and the pointwise certification
  frequency across all replications.
* Tolerance--certification profiles computed from already exported upper
  bounds (valid simultaneously in tau for full bands; Holm is excluded).
* MIMIC-IV effective-sample-size summaries computed from aggregate outputs.
* A local, credentialed-data function that recomputes descriptive deployment
  operating characteristics (specificity, alert burden, PPV) at the frozen
  class-sensitive thresholds and verifies that it reproduces the archived
  false-negative certificates exactly.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib, importlib.metadata, json, platform, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import binom

from .core import certify, exact_binomial_upper
from .baselines import ltt_holm_certify
from .metrics import region_metrics
from .continuous import continuous_upper_envelope
from .simulation import correlated_bernoulli
from .experiments import seeded, summarize

# Display order used by every revised table and figure: finite-sample routes,
# then asymptotic routes, then the diagnostic comparator without a guarantee.
DISPLAY_ORDER = ["exact_bonferroni", "holm", "hoeffding", "multiplier",
                 "normal_bonferroni", "pointwise"]
GUARANTEE = {"exact_bonferroni": "Finite-sample band",
             "holm": "Finite-sample, fixed $\\tau$",
             "hoeffding": "Finite-sample band",
             "multiplier": "Asymptotic band",
             "normal_bonferroni": "Asymptotic band",
             "pointwise": "None (marginal)"}


# ---------------------------------------------------------------------------
# Analytic exact-route quantities
# ---------------------------------------------------------------------------
def exact_count_threshold(n, tau, alpha=.05, K=1):
    """Largest error count s whose one-sided exact upper bound at alpha/K is <= tau.

    Uses the Clopper--Pearson duality U(s,n) <= tau  <=>  P{Bin(n,tau) <= s} <= alpha/K
    (for s < n).  Returns -1 when no count certifies.  Vectorized over n.
    """
    n = np.asarray(n, dtype=float)
    if not 0 < tau < 1 or not 0 < alpha < 1 or K < 1:
        raise ValueError("0<tau<1, 0<alpha<1 and K>=1 required")
    level = alpha / K
    q = binom.ppf(level, n, tau)
    s = np.where(binom.cdf(q, n, tau) <= level, q, q - 1)
    return s.astype(int)


def exact_certification_probability(n, risk, tau, alpha=.05, K=1):
    """P(one environment with true risk `risk` is certified by exact Bonferroni)."""
    s = exact_count_threshold(n, tau, alpha, K)
    return np.where(s >= 0, binom.cdf(np.maximum(s, 0), n, risk), 0.)


def hoeffding_certification_probability(n, risk, tau, alpha=.05, K=1):
    """P(mean of n Bernoulli(risk) losses plus the Hoeffding width is <= tau)."""
    n = np.asarray(n, dtype=float)
    width = np.sqrt(np.log(K / alpha) / (2 * n))
    s = np.floor(n * (tau - width) + 1e-9)
    return np.where(s >= 0, binom.cdf(np.maximum(s, 0), n, risk), 0.)


def minimal_assurance_n(risk, tau, alpha, K, target, n_max=400_000):
    """Smallest n with exact-Bonferroni certification probability >= target.

    Binomial discreteness makes the probability saw-toothed in n.  We return
    both the first n that reaches the target and the smallest n after which it
    never falls below the target again (up to n_max).
    """
    n = np.arange(1, n_max + 1)
    p = exact_certification_probability(n, risk, tau, alpha, K)
    ok = p >= target
    if not ok[-1]:
        return np.nan, np.nan
    bad = np.flatnonzero(~ok)
    stable = int(bad[-1] + 2) if len(bad) else 1
    first = int(np.flatnonzero(ok)[0] + 1)
    return first, stable


def zero_error_upper(n, alpha=.05, K=1):
    """Exact one-sided upper bound when no errors are observed among n losses."""
    return 1 - (alpha / K) ** (1 / np.asarray(n, dtype=float))


def minimal_n_zero_errors(tau, alpha=.05, K=1):
    """Smallest n whose zero-error exact bound is <= tau (best case for certification)."""
    return int(np.ceil(np.log(alpha / K) / np.log(1 - tau) - 1e-12))


# ---------------------------------------------------------------------------
# Revision run bookkeeping (separate from the original full-run manifest)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RevisionConfig:
    profile: str = "full"
    seed: int = 20260927
    alpha: float = .05
    bootstrap_draws: int = 2000
    n_assure: int = 1000
    power_tau: float = .10
    power_rho: float = .55
    power_K: tuple = (25, 100)
    power_n: tuple = (250, 500, 1000, 2000, 4000, 8000)
    power_delta: tuple = (.01, .02, .03, .05)
    power_reps: int = 400
    continuous_reps: int = 200
    planning_K: tuple = (25, 100, 500)
    planning_delta: tuple = (.01, .02, .03, .05)
    planning_targets: tuple = (.8, .9)

    @classmethod
    def smoke(cls):
        return cls(profile="smoke", bootstrap_draws=200, power_K=(10,), power_n=(250, 1000),
                   power_delta=(.03,), power_reps=10, continuous_reps=5, planning_K=(25,),
                   planning_delta=(.05,))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RevisionRun:
    """Writes revision outputs to results/revision[_smoke] with their own manifest."""

    def __init__(self, root, config=RevisionConfig()):
        self.root = Path(root).resolve(); self.cfg = config
        suffix = "" if config.profile == "full" else "_" + config.profile
        self.out = self.root / "results" / ("revision" + suffix)
        self.fig = self.root / "figures" / ("revision" + suffix)
        self.out.mkdir(parents=True, exist_ok=True); self.fig.mkdir(parents=True, exist_ok=True)
        path = self.out / "revision_manifest.json"
        previous = json.loads(path.read_text()) if path.is_file() else {}
        self.manifest = {"schema": 1, "status": "running", "config": asdict(config),
                         "started_utc": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "platform": platform.platform(),
                         "packages": {}, "sections": previous.get("sections", {}),
                         "note": ("Revision-5 additions. Original full-run outputs in results/full are read, "
                                  "hash-verified, and never rewritten.")}
        for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl"]:
            try:
                self.manifest["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.manifest["packages"][package] = "not installed"
        self.flush()

    def flush(self):
        (self.out / "revision_manifest.json").write_text(json.dumps(self.manifest, indent=2, allow_nan=False))

    def save(self, name, frame):
        frame.to_csv(self.out / (name + ".csv"), index=False)
        return frame

    def section(self, name, fn):
        start = time.perf_counter(); print(f"Running revision section {name} ({self.cfg.profile})", flush=True)
        try:
            result = fn(self)
        except Exception as exc:
            self.manifest["sections"][name] = {"status": "failed", "error": str(exc)}
            self.manifest["status"] = "failed"; self.flush(); raise
        self.manifest["sections"][name] = {"status": "completed", "seconds": time.perf_counter() - start}
        self.flush(); print(f"Completed {name}: {time.perf_counter()-start:.1f}s", flush=True)
        return result

    def finish(self):
        self.manifest["status"] = "completed"
        self.manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["output_sha256"] = {p.name: _sha256(p) for p in sorted(self.out.glob("*.csv"))}
        inputs = {}
        for folder in ["data", "results/full", "results/mimic_local"]:
            for p in sorted((self.root / folder).glob("*.csv")):
                inputs[str(p.relative_to(self.root))] = _sha256(p)
        self.manifest["input_sha256"] = inputs
        self.manifest["source_sha256"] = {str(p.relative_to(self.root)): _sha256(p)
                                          for p in sorted((self.root / "src").rglob("*.py"))}
        self.flush()


def _band_decisions(losses, tau, cfg, seed, methods):
    out = {}
    for method in methods:
        if method == "holm":
            cert, _ = ltt_holm_certify(losses.sum(axis=0), len(losses), tau, cfg.alpha)
            out[method] = (cert, None)
        else:
            res = certify(losses, tau, alpha=cfg.alpha, method=method, n_boot=cfg.bootstrap_draws, rng=seed)
            out[method] = (res.certified, res.upper)
    return out


# ---------------------------------------------------------------------------
# Section 1: certification probability versus sample size and risk margin
# ---------------------------------------------------------------------------
POWER_METHODS = ["exact_bonferroni", "holm", "hoeffding", "multiplier"]


def power_study(run):
    """Half the environments lie Delta below tau and half Delta above it.

    Losses are Gaussian-copula Bernoulli variables with latent correlation rho,
    as in the primary calibration.  Seeds use stream identifiers 90-91, which
    are not used by any original section.
    """
    cfg = run.cfg; tau = cfg.power_tau; rows = []; analytic = []
    for iK, K in enumerate(cfg.power_K):
        for i_n, n in enumerate(cfg.power_n):
            for i_d, delta in enumerate(cfg.power_delta):
                half = K // 2
                risks = np.r_[np.full(half, tau - delta), np.full(K - half, tau + delta)]
                safe = risks <= tau
                analytic.append(dict(K=K, n=n, delta=delta, tau=tau,
                                     exact_certification_probability=float(exact_certification_probability(n, tau - delta, tau, cfg.alpha, K)),
                                     exact_false_certification_probability_per_unsafe=float(exact_certification_probability(n, tau + delta, tau, cfg.alpha, K)),
                                     hoeffding_certification_probability=float(hoeffding_certification_probability(n, tau - delta, tau, cfg.alpha, K))))
                for r in range(cfg.power_reps):
                    x = correlated_bernoulli(n, risks, cfg.power_rho, seeded(cfg, 90, iK, i_n, i_d, r))
                    for method, (cert, upper) in _band_decisions(x, tau, cfg, seeded(cfg, 91, iK, i_n, i_d, r), POWER_METHODS).items():
                        mm = region_metrics(cert, safe)
                        mm.update(K=K, n=n, delta=delta, rep=r, method=method,
                                  simultaneous_coverage=float(np.all(risks <= upper)) if upper is not None else np.nan)
                        rows.append(mm)
        print(f"  power study: K={K} complete", flush=True)
    rep = run.save("power_replications", pd.DataFrame(rows))
    summary = run.save("power_summary", summarize(rep, ["K", "n", "delta", "method"],
                                                  ["ARR", "containment", "simultaneous_coverage", "UCP", "n_certified"]))
    run.save("power_analytic", pd.DataFrame(analytic))
    return rep, summary


def planning_table(run):
    """Stable minimal n for target exact-route certification probability."""
    cfg = run.cfg; rows = []
    for tau in [cfg.power_tau]:
        for K in cfg.planning_K:
            for delta in cfg.planning_delta:
                row = dict(tau=tau, K=K, delta=delta, alpha=cfg.alpha,
                           hoeffding_width_n=int(np.ceil(np.log(K / cfg.alpha) / (2 * delta ** 2))))
                for target in cfg.planning_targets:
                    first, stable = minimal_assurance_n(tau - delta, tau, cfg.alpha, K, target)
                    row[f"exact_n_{int(target*100)}_first"] = first
                    row[f"exact_n_{int(target*100)}"] = stable
                row["exact_probability_at_hoeffding_n"] = float(exact_certification_probability(row["hoeffding_width_n"], tau - delta, tau, cfg.alpha, K))
                row["hoeffding_probability_at_hoeffding_n"] = float(hoeffding_certification_probability(row["hoeffding_width_n"], tau - delta, tau, cfg.alpha, K))
                rows.append(row)
    return run.save("power_planning", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Section 2: exact-Bonferroni continuous example
# ---------------------------------------------------------------------------
CONTINUOUS_TAU = .32
CONTINUOUS_LIPSCHITZ = float(np.hypot(.15, .12))


def continuous_grid():
    return np.array([(m, s) for m in np.linspace(0, .4, 5) for s in np.linspace(0, .6, 5)])


def continuous_risk(points):
    points = np.asarray(points, float)
    return .24 + .12 * points[:, 0] + .10 * points[:, 1] + .05 * points[:, 0] * points[:, 1]


def continuous_exact(run):
    """Reconstruct the original continuous losses (streams 70/71) and use the exact band.

    Replication 0 is the first replication of the original study; it was not
    selected by its outcome.  The loss matrices are generated exactly as in
    `experiments.continuous`, so the multiplier results of the original run
    and the exact bounds here refer to the same simulated data.
    """
    cfg = run.cfg; grid = continuous_grid(); risk = continuous_risk(grid); rows = []
    for r in range(cfg.continuous_reps):
        x = correlated_bernoulli(cfg.n_assure, risk, .55, seeded(cfg, 70, r))
        res = certify(x, CONTINUOUS_TAU, method="exact_bonferroni", alpha=cfg.alpha)
        for k, (m, s) in enumerate(grid):
            rows.append(dict(rep=r, Missingness=m, Shift=s, risk=risk[k], observed=res.risks[k],
                             upper=res.upper[k], grid_certified=bool(res.certified[k])))
    bounds = run.save("continuous_exact_grid_bounds", pd.DataFrame(rows))
    query = np.array([(m, s) for m in np.linspace(0, .4, 81) for s in np.linspace(0, .6, 81)])
    first = bounds[bounds.rep == 0]
    u = continuous_upper_envelope(grid, first.upper.to_numpy(), query, CONTINUOUS_LIPSCHITZ)
    run.save("continuous_exact_example", pd.DataFrame(dict(Missingness=query[:, 0], Shift=query[:, 1],
                                                           risk=continuous_risk(query), upper=u,
                                                           certified=u <= CONTINUOUS_TAU)))
    return bounds


def certification_frequency(bounds, query, tau=CONTINUOUS_TAU, lipschitz=CONTINUOUS_LIPSCHITZ):
    """Fraction of replications whose Lipschitz envelope certifies each query point."""
    grid = continuous_grid()
    dist = np.linalg.norm(np.asarray(query)[:, None, :] - grid[None, :, :], axis=2)
    count = np.zeros(len(query))
    reps = sorted(bounds.rep.unique())
    for r in reps:
        u = bounds.loc[bounds.rep == r].sort_values(["Missingness", "Shift"]).upper.to_numpy()
        count += np.minimum(1., np.min(u[None, :] + lipschitz * dist, axis=1)) <= tau
    return count / len(reps)


# ---------------------------------------------------------------------------
# Section 3: tolerance--certification profiles from exported upper bounds
# ---------------------------------------------------------------------------
def tolerance_profile(frame, upper_col, group_cols, taus, eligible_col=None):
    """Number of eligible environments with simultaneous upper bound <= tau.

    For a full simultaneous band the bounds do not depend on tau, so by
    Proposition 2 of the manuscript (prop:finite) every point of the profile
    carries the same containment event.  Fixed-tolerance procedures (exact Holm) are deliberately excluded.
    """
    rows = []
    for key, d in frame.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        upper = pd.to_numeric(d[upper_col], errors="coerce").to_numpy(float)
        eligible = d[eligible_col].astype(bool).to_numpy() if eligible_col else np.isfinite(upper)
        if np.isnan(upper[eligible]).any():
            raise ValueError("eligible environments must have finite upper bounds")
        for tau in taus:
            certified = int(np.sum(eligible & (upper <= tau + 1e-12)))
            rows.append({**dict(zip(group_cols, key)), "tau": float(tau), "certified": certified,
                         "eligible": int(eligible.sum()), "total": int(len(d)),
                         "certified_fraction": certified / eligible.sum() if eligible.sum() else np.nan})
    return pd.DataFrame(rows)


def tolerance_profiles(run):
    root = run.root
    wd = pd.read_csv(root / "results/full/wdbc_detail.csv")
    wd = wd[(wd.design == "primary") & wd.method.isin(["exact_bonferroni", "multiplier"])]
    wd_profile = tolerance_profile(wd, "upper", ["Model", "method"], np.round(np.arange(0, .3001, .0025), 4))
    wd_profile.insert(0, "specification", "WDBC 0-1 risk")
    mr = pd.read_csv(root / "results/mimic_local/mimic_results.csv")
    mr = mr[mr.method.isin(["exact_bonferroni", "multiplier", "hoeffding"])]
    mimic_taus = np.unique(np.round(np.r_[np.arange(0, .1, .0005), np.arange(.1, .4001, .005)], 4))
    mr_profile = tolerance_profile(mr, "upper", ["Model", "method"], mimic_taus, "eligible")
    mr_profile.insert(0, "specification", "MIMIC-IV 0-1 risk")
    fn = pd.read_csv(root / "results/mimic_local/mimic_class_sensitive_results.csv")
    fn = fn[fn.method == "exact_bonferroni"]
    fn_profile = tolerance_profile(fn, "upper_fn_risk", ["Model", "method"], np.round(np.arange(0, 1.0001, .0025), 4), "eligible")
    fn_profile.insert(0, "specification", "MIMIC-IV false-negative risk")
    profile = pd.concat([wd_profile, mr_profile, fn_profile], ignore_index=True)
    return run.save("tolerance_profiles", profile)


# ---------------------------------------------------------------------------
# Section 4: MIMIC-IV effective sample size (aggregate outputs only)
# ---------------------------------------------------------------------------
def mimic_effective_sample_size(run):
    root = run.root
    fn = pd.read_csv(root / "results/mimic_local/mimic_class_sensitive_results.csv")
    protocol = json.loads((root / "results/mimic_local/mimic_class_sensitive_protocol.json").read_text())
    alpha = float(protocol["alpha"]); tau = float(protocol["tau_fn"])
    exact = fn[fn.method == "exact_bonferroni"]; holm = fn[fn.method == "holm"]
    K = len(exact)
    dom = pd.read_csv(root / "results/mimic_local/mimic_results.csv")
    domain_n = dom[(dom.method == "exact_bonferroni")].groupby("TemporalDomain").n.first()
    rows = []
    for domain, d in exact.groupby("TemporalDomain", sort=True):
        positives = int(d.positive_n.iloc[0]); eligible = bool(d.eligible.iloc[0])
        row = dict(TemporalDomain=domain, n=int(domain_n.loc[domain]), deaths=positives, eligible=eligible,
                   K=K, alpha=alpha, tau_fn=tau,
                   zero_fn_upper=float(zero_error_upper(positives, alpha, K)) if positives > 0 else np.nan,
                   best_lower_sensitivity=float(1 - zero_error_upper(positives, alpha, K)) if positives > 0 else np.nan,
                   certifiable_even_with_zero_fn=bool(positives > 0 and zero_error_upper(positives, alpha, K) <= tau))
        for model, tag in [("Logistic regression", "LR"), ("HistGradientBoosting", "HGB"), ("Always survive", "AS")]:
            m = d[d.Model == model].sort_values("Missingness")
            zero = m[np.isclose(m.Missingness, 0)].iloc[0]
            row[f"{tag}_sensitivity_0"] = float(zero.sensitivity)
            row[f"{tag}_lower_sensitivity_0"] = float(1 - zero.upper_fn_risk) if eligible else np.nan
            row[f"{tag}_exact_certified"] = int(m.certified.sum())
            hm = holm[(holm.TemporalDomain == domain) & (holm.Model == model)]
            row[f"{tag}_holm_certified"] = int(hm.certified.sum())
            row[f"{tag}_conditions"] = int(len(m))
        rows.append(row)
    table = run.save("mimic_effective_sample_size", pd.DataFrame(rows))
    # Deaths needed per domain for a target certification probability under the same family.
    needs = []
    for sensitivity in [.85, .90, .95]:
        for target in [.8, .9]:
            first, stable = minimal_assurance_n(1 - sensitivity, tau, alpha, K, target, n_max=20000)
            needs.append(dict(true_sensitivity=sensitivity, target_probability=target, K=K, alpha=alpha,
                              tau_fn=tau, deaths_first=first, deaths_stable=stable))
    needs.append(dict(true_sensitivity=1.0, target_probability=1.0, K=K, alpha=alpha, tau_fn=tau,
                      deaths_first=minimal_n_zero_errors(tau, alpha, K), deaths_stable=minimal_n_zero_errors(tau, alpha, K)))
    run.save("mimic_deaths_needed", pd.DataFrame(needs))
    return table


# ---------------------------------------------------------------------------
# Section 5: anatomy of primary-calibration band failures (reproduces the original run)
# ---------------------------------------------------------------------------
def calibration_anatomy(run):
    """Regenerate the original primary calibration and locate band miscoverage.

    The loss matrices and multiplier seeds are those of `experiments.calibration`
    (streams 10/11 with the original RunConfig), so the per-replication results
    must match results/full/calibration_replications.csv.  The section records
    that check, then classifies each band failure by whether the missed
    coordinates are truly safe (a miss there cannot create a false certificate)
    or truly unsafe (a miss there may or may not create one).
    """
    from .experiments import RunConfig, _decisions
    cfg = RunConfig(); root = run.root
    probs = pd.read_csv(root / "data/simulation_true_risks.csv").Risk.to_numpy()
    safe = probs <= cfg.tau_calibration
    stored = pd.read_csv(root / "results/full/calibration_replications.csv").set_index(["rep", "method"])
    rows = []; coord = []
    for r in range(cfg.calibration_reps):
        x = correlated_bernoulli(cfg.n_assure, probs, .55, seeded(cfg, 10, r))
        for method, (cert, upper, _) in _decisions(x, cfg, seeded(cfg, 11, r)).items():
            mm = region_metrics(cert, safe)
            old = stored.loc[(r, method)]
            coverage = float(np.all(probs <= upper)) if upper is not None else np.nan
            match = bool(mm["n_certified"] == old.n_certified and np.isclose(mm["ARR"], old.ARR)
                         and ((np.isnan(coverage) and np.isnan(old.simultaneous_coverage)) or coverage == old.simultaneous_coverage))
            row = dict(rep=r, method=method, reproduces_original=match, coverage=coverage,
                       containment=mm["containment"], n_certified=mm["n_certified"])
            if upper is not None:
                miss = probs > upper
                row.update(missed_safe=int(np.sum(miss & safe)), missed_unsafe=int(np.sum(miss & ~safe)),
                           false_certificates=int(np.sum(cert & ~safe)))
                for k in np.flatnonzero(miss):
                    coord.append(dict(rep=r, method=method, k=int(k), risk=float(probs[k]), safe=bool(safe[k]),
                                      upper=float(upper[k]), certified=bool(cert[k])))
            rows.append(row)
    detail = run.save("calibration_anatomy_replications", pd.DataFrame(rows))
    run.save("calibration_anatomy_missed_coordinates", pd.DataFrame(coord, columns=["rep", "method", "k", "risk", "safe", "upper", "certified"]))
    band = detail[detail.coverage.notna()]
    fail = band[band.coverage == 0]
    summary = []
    for method, d in band.groupby("method", sort=False):
        f = fail[fail.method == method]
        summary.append(dict(method=method, replications=len(d), reproduces_original=int(d.reproduces_original.sum()),
                            coverage_failures=len(f), failures_safe_only=int(np.sum((f.missed_unsafe == 0))),
                            failures_with_unsafe_miss=int(np.sum(f.missed_unsafe > 0)),
                            failures_with_false_certificate=int(np.sum(f.false_certificates > 0)),
                            containment_failures=int(np.sum(d.containment == 0))))
    holm = detail[detail.method == "holm"]
    summary.append(dict(method="holm", replications=len(holm), reproduces_original=int(holm.reproduces_original.sum()),
                        coverage_failures=np.nan, failures_safe_only=np.nan, failures_with_unsafe_miss=np.nan,
                        failures_with_false_certificate=np.nan, containment_failures=int(np.sum(holm.containment == 0))))
    return run.save("calibration_anatomy_summary", pd.DataFrame(summary))


def stress_coincidence(run):
    """Explain identical Bonferroni-normal and multiplier coverage in two stress designs.

    Regenerates the bounded-continuous and rare-binary replications of
    `experiments.bootstrap_and_stress` (streams 32/33), records each
    replication's multiplier critical value, the Bonferroni-normal value
    z_{1-alpha/K}, and the smallest standardized error among missed columns.
    """
    from scipy.stats import norm
    from .experiments import RunConfig
    cfg = RunConfig(); root = run.root
    base = pd.read_csv(root / "data/simulation_true_risks.csv").Risk.to_numpy()
    archived = pd.read_csv(root / "results/full/stress_replications.csv")
    rows = []
    for j, scenario in enumerate(["bounded_continuous", "rare_binary"]):
        n, k = 500, 25
        probs = np.linspace(.002, .08, k) if j == 1 else np.resize(base, k)
        tau = .035 if j == 1 else cfg.tau_calibration
        for r in range(200 if cfg.profile == "full" else 10):
            rng = np.random.default_rng(seeded(cfg, 32, j, r))
            x = rng.beta(probs * 4, (1 - probs) * 4, size=(n, k)) if j == 0 else correlated_bernoulli(n, probs, .55, rng)
            mult = certify(x, tau, alpha=cfg.alpha, method="multiplier", n_boot=cfg.bootstrap_draws, rng=seeded(cfg, 33, j, r))
            nb = certify(x, tau, alpha=cfg.alpha, method="normal_bonferroni")
            old = archived[(archived.scenario == scenario) & (archived.rep == r)].set_index("method").simultaneous_coverage
            cov_m = float(np.all(probs <= mult.upper)); cov_n = float(np.all(probs <= nb.upper))
            se = np.where(nb.se > 0, nb.se, np.nan)
            z = (probs - nb.risks) / se
            miss = probs > nb.upper
            rows.append(dict(scenario=scenario, rep=r, multiplier_critical=mult.critical_value,
                             bonferroni_critical=float(norm.ppf(1 - cfg.alpha / k)),
                             coverage_multiplier=cov_m, coverage_normal_bonferroni=cov_n,
                             matches_archive=bool(cov_m == old["multiplier"] and cov_n == old["normal_bonferroni"]),
                             min_standardized_error_missed=float(np.nanmin(z[miss])) if miss.any() else np.nan,
                             min_errors_in_missed=int(x[:, miss].sum(axis=0).min()) if miss.any() and j == 1 else np.nan))
    return run.save("stress_coincidence", pd.DataFrame(rows))


def studentized_tail(run):
    """Exact one-sided failure probability of the studentized normal upper bound.

    For one Bernoulli(p) environment with n rows, the normal bound
    R_hat + z_{1-a} sqrt(R_hat(1-R_hat)/n) fails when it falls below p.  The
    probability is computed exactly from the binomial law and reported as a
    multiple of the nominal tail a = alpha/K, the per-environment level that a
    Bonferroni-normal band uses (the multiplier uses a slightly less extreme
    level because it exploits dependence).  No simulation is involved.
    """
    from scipy.stats import norm
    cfg = run.cfg; rows = []
    for K in (25, 100, 500):
        a = cfg.alpha / K; z = float(norm.ppf(1 - a))
        for p in (.05, .09, .11, .15):
            for n in (250, 500, 1000, 2000, 4000, 8000, 32000):
                s_ = np.arange(n + 1); r = s_ / n
                upper = r + z * np.sqrt(r * (1 - r) / n)
                fail = float(binom.pmf(s_[upper < p], n, p).sum())
                rows.append(dict(K=K, risk=p, n=n, nominal_tail=a, z=z, failure_probability=fail,
                                 inflation=fail / a))
    return run.save("studentized_tail_inflation", pd.DataFrame(rows))


def bootstrap_reproduction(run):
    """Recompute archived multiplier critical values with their archived seeds.

    Uses the fixed loss matrices and seeds of the bootstrap timing study
    (streams 30/31).  Differences between the archived and recomputed critical
    values measure how the realized Gaussian draws depend on the computing
    environment; their spread should shrink like Monte Carlo error as B grows.
    """
    from .experiments import RunConfig
    cfg = RunConfig(); root = run.root
    base = pd.read_csv(root / "data/simulation_true_risks.csv").Risk.to_numpy()
    archived = pd.read_csv(root / "results/full/bootstrap_runtime_replications.csv")
    rows = []
    for (n, k), d in archived.groupby(["n", "K"], sort=True):
        x = correlated_bernoulli(int(n), np.resize(base, int(k)), .55, seeded(cfg, 30, int(n), int(k)))
        for _, old in d.iterrows():
            res = certify(x, cfg.tau_calibration, alpha=cfg.alpha, method="multiplier", n_boot=int(old.B),
                          rng=seeded(cfg, 31, int(old.rep)))
            rows.append(dict(n=int(n), K=int(k), B=int(old.B), rep=int(old.rep), archived_critical=float(old.critical),
                             recomputed_critical=res.critical_value, difference=res.critical_value - float(old.critical),
                             archived_certified=int(old.n_certified), recomputed_certified=int(res.certified.sum())))
    return run.save("bootstrap_reproduction", pd.DataFrame(rows))


REVISION_SECTIONS = {"calibration_anatomy": calibration_anatomy, "stress_coincidence": stress_coincidence,
                     "bootstrap_reproduction": bootstrap_reproduction,
                     "studentized_tail": studentized_tail,
                     "power_study": power_study, "power_planning": planning_table,
                     "continuous_exact": continuous_exact, "tolerance_profiles": tolerance_profiles,
                     "mimic_effective_sample_size": mimic_effective_sample_size}


# ---------------------------------------------------------------------------
# Local credentialed MIMIC-IV run: descriptive operating characteristics
# ---------------------------------------------------------------------------
def threshold_for_target_sensitivity(probabilities, labels, target_sensitivity=.80):
    """Largest source-holdout threshold attaining the target sensitivity (as in the notebook)."""
    probabilities = np.asarray(probabilities, float); labels = np.asarray(labels, int)
    positive_scores = probabilities[labels == 1]
    if not len(positive_scores):
        raise ValueError("No positive outcomes in source holdout.")
    candidates = np.unique(positive_scores)
    sensitivities = np.array([np.mean(positive_scores >= t) for t in candidates])
    valid = sensitivities >= target_sensitivity
    if not valid.any():
        raise ValueError("Unable to attain requested source sensitivity.")
    threshold = candidates[valid][-1]
    return float(threshold), float(np.mean(positive_scores >= threshold))


def mimic_class_sensitive_operating(mimic_root, config, results_dir, target_sensitivity=.80):
    """Recompute frozen-threshold predictions and summarize alert burden (descriptive only).

    Requires credentialed MIMIC-IV tables.  The function refits the same
    prespecified models with the same seed, recomputes the source-holdout
    thresholds, and verifies that deployment positive counts and false-negative
    counts reproduce `mimic_class_sensitive_results.csv` exactly before writing
    `mimic_class_sensitive_operating.csv`.  No certification decision changes;
    specificity, alert rates and PPV carry no simultaneous guarantee here.
    """
    from sklearn.model_selection import train_test_split
    from .mimic import load_cohort, build_model, FEATURES
    results_dir = Path(results_dir)
    archived = pd.read_csv(results_dir / "mimic_class_sensitive_results.csv")
    archived = archived[archived.method == "exact_bonferroni"]
    thresholds_archived = pd.read_csv(results_dir / "mimic_class_sensitive_thresholds.csv")
    d, _, _ = load_cohort(mimic_root, config)
    source = d[d.split == "source"].copy(); target = d[d.split == "deployment"].copy()
    train, holdout = train_test_split(source, test_size=.20, stratify=source["mortality"], random_state=config["seed"])
    domains = sorted(target.TemporalDomain.unique())
    y = target["mortality"].to_numpy(int); X = target[FEATURES].reset_index(drop=True)
    domain_of = target["TemporalDomain"].to_numpy()
    uniforms = np.random.default_rng(config["seed"]).random(X.shape)
    uniforms[:, FEATURES.index("age_topcoded")] = 1.
    rows = []; threshold_rows = []
    for name in list(config["models"]) + ["Always survive"]:
        if name == "Always survive":
            model = None; threshold = np.nan
        else:
            model = build_model(name, config["seed"]); model.fit(train[FEATURES], train["mortality"])
            threshold, sens = threshold_for_target_sensitivity(model.predict_proba(holdout[FEATURES])[:, 1],
                                                               holdout["mortality"], target_sensitivity)
            frozen = thresholds_archived.loc[thresholds_archived.Model == name, "FrozenThreshold"].iloc[0]
            if not np.isclose(threshold, frozen, rtol=0, atol=1e-10):
                raise RuntimeError(f"{name}: recomputed threshold {threshold} differs from archived {frozen}")
        threshold_rows.append(dict(Model=name, FrozenThreshold=threshold))
        for m in config["missingness_levels"]:
            if model is None:
                pred = np.zeros(len(y), int)
            else:
                pred = (model.predict_proba(X.mask(uniforms < m))[:, 1] >= threshold).astype(int)
            for domain in domains:
                sel = domain_of == domain; yy = y[sel]; pp = pred[sel]
                tp = int(np.sum((yy == 1) & (pp == 1))); fn = int(np.sum((yy == 1) & (pp == 0)))
                fp = int(np.sum((yy == 0) & (pp == 1))); tn = int(np.sum((yy == 0) & (pp == 0)))
                ref = archived[(archived.TemporalDomain == domain) & (archived.Model == name) & np.isclose(archived.Missingness, m)]
                if len(ref) != 1 or int(ref.positive_n.iloc[0]) != tp + fn or int(ref.false_negatives.iloc[0]) != fn:
                    raise RuntimeError(f"Re-run does not reproduce archived FN counts for {name}, {domain}, m={m}")
                n_sel = int(sel.sum())
                rows.append(dict(TemporalDomain=domain, Model=name, Missingness=float(m), threshold=threshold,
                                 n=n_sel, deaths=tp + fn, TP=tp, FN=fn, FP=fp, TN=tn,
                                 sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                                 specificity=tn / (tn + fp) if tn + fp else np.nan,
                                 predicted_positive_fraction=(tp + fp) / n_sel,
                                 alerts_per_1000=1000 * (tp + fp) / n_sel,
                                 PPV=tp / (tp + fp) if tp + fp else np.nan,
                                 NPV=tn / (tn + fn) if tn + fn else np.nan,
                                 certified_fn=bool(ref.certified.iloc[0])))
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "mimic_class_sensitive_operating.csv", index=False)
    meta = {"analysis": "descriptive_operating_characteristics_at_frozen_class_sensitive_thresholds",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "verified_against": "mimic_class_sensitive_results.csv (positive_n and false_negatives, all 72 conditions)",
            "guarantee": "none; descriptive summaries of the frozen operating points",
            "output_sha256": _sha256(results_dir / "mimic_class_sensitive_operating.csv"),
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]}}
    (results_dir / "mimic_class_sensitive_operating_manifest.json").write_text(json.dumps(meta, indent=2))
    return table
