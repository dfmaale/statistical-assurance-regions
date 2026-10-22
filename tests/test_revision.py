"""Regression tests for the revision analyses (power, planning, profiles, effective sample size).

Each test checks an analytic quantity used in the manuscript against an
independent brute-force computation, so that the closed forms in the paper
(eq:pcert, the zero-false-negative bound, the planning search) are verified
without simulation.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciguard.core import exact_binomial_upper  # noqa: E402
from sciguard.revision import (exact_certification_probability, exact_count_threshold,  # noqa: E402
                               hoeffding_certification_probability, minimal_assurance_n,
                               minimal_n_zero_errors, tolerance_profile, zero_error_upper)


class ExactThreshold(unittest.TestCase):
    def test_threshold_matches_brute_force_clopper_pearson(self):
        # s* from Clopper--Pearson duality must equal the largest count whose
        # exact upper bound at level alpha/K is at most tau.
        for n in (14, 30, 100, 250, 1000):
            for tau in (.10, .20, .34):
                for K in (1, 25, 72):
                    counts = np.arange(n + 1)
                    upper = exact_binomial_upper(counts, n, .05 / K)
                    ok = np.flatnonzero(upper <= tau + 1e-12)
                    brute = int(ok.max()) if len(ok) else -1
                    self.assertEqual(int(exact_count_threshold(n, tau, .05, K)), brute, (n, tau, K))

    def test_certification_probability_matches_enumeration(self):
        n, tau, K = 400, .10, 25
        upper = exact_binomial_upper(np.arange(n + 1), n, .05 / K)
        for risk in (.05, .08, .10, .12):
            brute = float(binom.pmf(np.arange(n + 1), n, risk)[upper <= tau].sum())
            self.assertAlmostEqual(float(exact_certification_probability(n, risk, tau, .05, K)), brute, places=12)

    def test_certification_probability_at_tolerance_is_at_most_level(self):
        # An environment exactly at the tolerance is certified with probability <= alpha/K.
        for n in (100, 1000, 8000):
            for K in (1, 25, 100):
                self.assertLessEqual(float(exact_certification_probability(n, .10, .10, .05, K)), .05 / K + 1e-12)

    def test_hoeffding_probability_is_about_half_at_width_rule(self):
        # When the Hoeffding width equals the margin, the estimate sits on the boundary.
        for K in (25, 100, 500):
            for delta in (.01, .05):
                n_h = int(np.ceil(np.log(K / .05) / (2 * delta ** 2)))
                p = float(hoeffding_certification_probability(n_h, .10 - delta, .10, .05, K))
                self.assertGreater(p, .45); self.assertLess(p, .60)


class Planning(unittest.TestCase):
    def test_minimal_assurance_n_is_stable(self):
        first, stable = minimal_assurance_n(.05, .10, .05, 25, .80, n_max=3000)
        self.assertEqual(stable, 422)
        self.assertLessEqual(first, stable)
        n = np.arange(stable, 3001)
        self.assertTrue(np.all(exact_certification_probability(n, .05, .10, .05, 25) >= .80))
        self.assertLess(float(exact_certification_probability(stable - 1, .05, .10, .05, 25)), .80)

    def test_planning_returns_nan_when_target_unreachable(self):
        first, stable = minimal_assurance_n(.099, .10, .05, 25, .80, n_max=500)
        self.assertTrue(np.isnan(first) and np.isnan(stable))


class EffectiveSampleSize(unittest.TestCase):
    def test_zero_false_negative_bound_for_fourteen_deaths(self):
        # Manuscript: zero false negatives among 14 deaths, K=72, alpha=.05 -> upper bound about 0.405.
        u = float(zero_error_upper(14, .05, 72))
        self.assertAlmostEqual(u, 0.405, places=3)
        self.assertAlmostEqual(u, float(exact_binomial_upper(0, 14, .05 / 72)), places=12)

    def test_minimum_deaths_for_fn_certification(self):
        need = minimal_n_zero_errors(.20, .05, 72)
        self.assertEqual(need, 33)
        self.assertLessEqual(float(zero_error_upper(need, .05, 72)), .20)
        self.assertGreater(float(zero_error_upper(need - 1, .05, 72)), .20)


class Profiles(unittest.TestCase):
    def test_tolerance_profile_monotone_and_bounded(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"model": np.repeat(["a", "b"], 30), "upper": rng.uniform(.02, .5, 60),
                              "eligible": True})
        frame.loc[[3, 40], "eligible"] = False
        taus = np.linspace(0, .6, 121)
        prof = tolerance_profile(frame, "upper", ["model"], taus, eligible_col="eligible")
        for model, d in prof.groupby("model"):
            frac = d.sort_values("tau").certified_fraction.to_numpy()
            self.assertTrue(np.all(np.diff(frac) >= 0))
            self.assertEqual(frac[0], 0.0); self.assertEqual(frac[-1], 1.0)
            self.assertEqual(int(d.eligible.iloc[0]), 29)

    def test_ineligible_conditions_are_never_certified(self):
        frame = pd.DataFrame({"g": ["x"] * 3, "upper": [.01, .02, np.nan], "eligible": [True, True, False]})
        prof = tolerance_profile(frame, "upper", ["g"], [1.0], eligible_col="eligible")
        self.assertEqual(int(prof.certified.iloc[0]), 2)


class StudentizedTail(unittest.TestCase):
    def test_normal_bound_fails_more_often_than_nominal(self):
        # Exact binomial calculation behind the large-n undercoverage explanation.
        for K, n in ((100, 8000), (25, 8000), (100, 250)):
            a = .05 / K; z = norm.ppf(1 - a)
            for p in (.09, .11):
                s = np.arange(n + 1); r = s / n
                fail = binom.pmf(s[r + z * np.sqrt(r * (1 - r) / n) < p], n, p).sum()
                self.assertGreater(fail / a, 1.1, (K, n, p))


if __name__ == "__main__":
    unittest.main()
