"""Regression tests for consequential inferential and cohort failure modes."""
import unittest,tempfile,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom,norm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sciguard import (certify,certify_independent,certify_overlapping_groups,continuous_upper_envelope,
                      ltt_holm_certify,atc_threshold,atc_predicted_risk,reference_intervals)
from sciguard.core import gaussian_max_quantile,exact_binomial_upper
from sciguard.simulation import draw_base,missing_mask
from sciguard.mimic import load_cohort,run_mimic,validate_config
from sciguard.design import alpha_spending


class StatisticalContract(unittest.TestCase):
    def test_exact_zero_event_bound(self):
        result=certify(np.zeros((100,5)),.1,method="exact_bonferroni")
        self.assertTrue(np.allclose(result.upper,1-(.05/5)**(1/100)))
        self.assertTrue(np.all(result.certified))

    def test_all_constant_multiplier_is_conservative(self):
        result=certify(np.zeros((100,4)),.1,n_boot=200)
        self.assertTrue(np.all(result.upper==1))
        self.assertFalse(result.certified.any())

    def test_exact_band_finite_sample_enumeration(self):
        # Exhaustively integrate all binomial counts for several boundary risks.
        n=20; counts=np.arange(n+1); upper=exact_binomial_upper(counts,n,.05)
        for risk in [.001,.03,.1,.5,.97,.999]:
            coverage=binom.pmf(counts,n,risk)[upper>=risk].sum()
            self.assertGreaterEqual(coverage, .95-1e-12)

    def test_holm_finite_sample_enumeration_one_environment(self):
        n=25; tau=.15
        rejected=np.array([ltt_holm_certify([s],n,tau)[0][0] for s in range(n+1)])
        for risk in [tau,.2,.5]:
            self.assertLessEqual(binom.pmf(np.arange(n+1),n,risk)[rejected].sum(),.05+1e-12)

    def test_shared_duplicate_columns_need_no_independence(self):
        x=np.random.default_rng(7).binomial(1,.2,size=(600,1)); xx=np.repeat(x,5,axis=1)
        single=certify(x,.3,n_boot=20000,rng=4); duplicate=certify(xx,.3,n_boot=20000,rng=5)
        self.assertLess(abs(single.critical_value-duplicate.critical_value),.08)
        self.assertLess(abs(duplicate.critical_value-norm.ppf(.95)),.06)

    def test_covariance_and_row_sampler_same_gaussian_law(self):
        rng=np.random.default_rng(9); x=rng.binomial(1,[.1,.2,.3],size=(1000,3))
        a,_=gaussian_max_quantile(x,n_boot=30000,rng=3,sampler="rows")
        b,_=gaussian_max_quantile(x,n_boot=30000,rng=4,sampler="covariance")
        self.assertLess(abs(a-b),.06)

    def test_tau_nesting_and_invalid_loss(self):
        x=np.random.default_rng(1).binomial(1,.1,(500,3))
        a=certify(x,.10,rng=2,n_boot=200); b=certify(x,.20,rng=2,n_boot=200)
        self.assertTrue(np.all(a.certified<=b.certified))
        for bad in [np.full((5,2),np.nan),np.full((5,2),1.1),np.zeros((1,2))]:
            with self.assertRaises(ValueError): certify(bad,.1)
        with self.assertRaises(ValueError): certify(x,1.1)
        with self.assertRaises(ValueError): certify(x,.1,n_boot=0)
        with self.assertRaises(ValueError): certify(x,.1,chunk_size=0)
        with self.assertRaises(ValueError): certify(x/2,.1,method="exact_bonferroni")

    def test_groups_retained_and_degenerate_guard(self):
        x=np.zeros((100,2)); g=np.column_stack([np.ones(100),np.zeros(100),np.arange(100)<5])
        result=certify_overlapping_groups(x,g,.1,B=200,min_group_size=20)
        self.assertEqual(len(result.upper),6)
        self.assertFalse(result.certified.any()); self.assertTrue(np.all(result.upper==1))
        exact=certify_overlapping_groups(x,g,.1,method="exact_bonferroni",min_group_size=20)
        self.assertTrue(exact.certified[:2].all()); self.assertFalse(exact.certified[2:].any())

    def test_single_group_equals_paired(self):
        x=np.random.default_rng(10).binomial(1,[.05,.1,.2],size=(500,3))
        a=certify(x,.15,n_boot=1000,rng=1)
        b=certify_overlapping_groups(x,np.ones((len(x),1)),.15,B=1000,seed=1)
        self.assertTrue(np.allclose(a.upper,b.upper))

    def test_unequal_samples_never_truncated(self):
        x=[np.zeros(100),np.ones(50)]
        a=certify_independent(x,.1,method="exact_bonferroni")
        self.assertEqual(a.sample_sizes.tolist(),[100,50]); self.assertEqual(a.risks.tolist(),[0,1])

    def test_atc_ties_and_strict_threshold(self):
        t=atc_threshold([.6,.6,.9,.9],[1,0,1,0])
        self.assertEqual(t,.6); self.assertEqual(atc_predicted_risk([.6,.6,.9,.9],t),.5)

    def test_mar_driver_protected_and_batch_invariant(self):
        x,_,u,b=draw_base(1000,10)
        for mechanism in ["MCAR","MAR","MNAR","Structured"]:
            mask=missing_mask(x,.2,mechanism,u,b)
            subset=missing_mask(x[:100],.2,mechanism,u[:100],b[:100])
            self.assertFalse(mask[:,0].any()); self.assertTrue(np.array_equal(mask[:100],subset))
        small=missing_mask(x,.1,"MAR",u,b); large=missing_mask(x,.3,"MAR",u,b)
        self.assertTrue(np.all(small<=large))

    def test_reference_intervals_include_boundaries(self):
        x=np.column_stack([np.zeros(100),np.ones(100)])
        lo,hi=reference_intervals(x)
        self.assertEqual(lo[0],0); self.assertEqual(hi[1],1)
        self.assertGreater(hi[0],0); self.assertLess(lo[1],1)

    def test_continuous_envelope_known_linear_risk(self):
        grid=np.array([[0.],[1.]]); upper=np.array([.1,.3]); q=np.linspace(0,1,101)[:,None]
        envelope=continuous_upper_envelope(grid,upper,q,.2)
        self.assertTrue(np.all(envelope>=.1+.2*q[:,0]-1e-12))
        with self.assertRaises(ValueError): continuous_upper_envelope(grid,upper,q,float("nan"))

    def test_spending_budget(self):
        self.assertLess(sum(alpha_spending(i) for i in range(1,1001)),.05)


class MimicContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.config=json.loads((ROOT/"config/mimic_protocol.json").read_text())

    def tearDown(self): self.tmp.cleanup()

    def fixture(self,n=800):
        ids=np.arange(n); rng=np.random.default_rng(14)
        patients=pd.DataFrame(dict(subject_id=ids,gender=np.where(ids%2,"F","M"),anchor_age=np.where(ids%13==0,91,50),anchor_year=2150,anchor_year_group=np.where(ids<n//2,"2008 - 2010","2017 - 2019")))
        admissions=pd.DataFrame(dict(subject_id=ids,hadm_id=ids+1000,admittime=np.where(ids<n//2,"2150-02-01","2152-02-01"),hospital_expire_flag=rng.binomial(1,.1,n),admission_type="EMERGENCY",admission_location="EMERGENCY ROOM",insurance="Other",marital_status=np.where(ids%4==0,None,"SINGLE"),race="UNKNOWN"))
        patients.to_csv(self.root/"patients.csv",index=False); admissions.to_csv(self.root/"admissions.csv",index=False)

    def test_admission_year_alignment_and_topcoding(self):
        self.fixture(); d,flow,files=load_cohort(self.root,self.config)
        early=d[d.subject_id==1].iloc[0]; late=d[d.subject_id==500].iloc[0]
        self.assertEqual(early.admission_year_upper,2011)
        # Admission is two shifted years after anchor; unadjusted 2017-2019 is wrong.
        self.assertEqual(late.admission_year_lower,2018)
        self.assertEqual(late.admission_year_upper,2022)
        self.assertEqual(d.loc[d.subject_id==0,"age"].iloc[0],91)
        self.assertTrue(d.subject_id.is_unique)

    def test_explicit_tolerance_required(self):
        # The shipped protocol is frozen with tau=.35 for the credentialed run, so the
        # guard is tested on explicitly unset copies rather than on the shipped file.
        validate_config(self.config)
        with self.assertRaises(ValueError): validate_config(dict(self.config,tau=None))
        with self.assertRaises(ValueError): validate_config(dict(self.config,tau_rationale="  "))
        incomplete=dict(self.config); incomplete.pop("tau")
        with self.assertRaises(ValueError): validate_config(incomplete)

    def test_end_to_end_synthetic_fixture_not_evidence(self):
        self.fixture()
        self.config.update(tau=.2,tau_rationale="TEST FIXTURE ONLY",bootstrap_draws=200,min_group_size=20)
        tables,manifest=run_mimic(self.root,self.config,self.root/"test_outputs")
        self.assertTrue((self.root/"test_outputs/mimic_manifest.json").exists())
        self.assertEqual(manifest["family_size"],12)
        self.assertEqual(set(tables["mimic_results"].method),{"multiplier","exact_bonferroni","hoeffding","holm"})
        self.assertNotIn("subject_id",tables["mimic_results"].columns)
        self.assertEqual(manifest["source_n"]+manifest["deployment_n"],800)
        from sciguard.plotting import mimic_decision_map
        import matplotlib.pyplot as plt
        fig=mimic_decision_map(tables["mimic_results"],self.root/"test_figures")
        self.assertTrue((self.root/"test_figures/mimic_iv_sar.png").exists())
        plt.close(fig)


if __name__=="__main__": unittest.main(verbosity=2)
