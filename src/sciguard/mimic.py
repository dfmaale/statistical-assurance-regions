"""Local credentialed MIMIC-IV protocol; no data download or patient export.

Temporal labels are admission-aligned approximate calendar intervals, never
unadjusted patient anchor groups. This protocol has no claimed benchmark
results until it is run on actual credentialed tables.
"""
from pathlib import Path
import json,hashlib,time,re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from .groups import certify_overlapping_groups
from .baselines import ltt_holm_certify

FEATURES=["age","age_topcoded","gender","admission_type","admission_location","insurance","marital_status","race"]
NUMERIC=["age","age_topcoded"]
CATEGORICAL=[c for c in FEATURES if c not in NUMERIC]


def find_table(root,name):
    root=Path(root)
    for folder in ["hosp","core",""]:
        for extension in [".csv.gz",".csv"]:
            p=root/folder/(name+extension)
            if p.is_file(): return p
    raise FileNotFoundError(f"Missing {name}.csv(.gz) in hosp/, core/, or dataset root")


def load_cohort(root,config):
    pp,ap=find_table(root,"patients"),find_table(root,"admissions")
    # Read only the required columns; outcomes, identifiers and times are used
    # only for labels, cohort/split construction, and disjointness checks.
    pc={"subject_id","gender","anchor_age","anchor_year","anchor_year_group"}
    ac={"subject_id","hadm_id","admittime","hospital_expire_flag","admission_type","admission_location","insurance","marital_status","race","ethnicity"}
    p=pd.read_csv(pp,usecols=lambda c:c.lower() in pc)
    a=pd.read_csv(ap,usecols=lambda c:c.lower() in ac)
    p.columns=p.columns.str.lower(); a.columns=a.columns.str.lower()
    required_a={"subject_id","hadm_id","admittime","hospital_expire_flag","admission_type"}
    if pc-set(p) or required_a-set(a): raise ValueError("MIMIC tables lack required columns")
    if p.subject_id.duplicated().any() or a.hadm_id.duplicated().any(): raise ValueError("duplicate patient/admission identifiers")
    if "race" not in a and "ethnicity" in a: a=a.rename(columns={"ethnicity":"race"})
    d=a.merge(p,on="subject_id",validate="many_to_one",how="inner")
    flow=[{"stage":"joined_admissions","N":len(d)}]
    d["admittime"]=pd.to_datetime(d.admittime,errors="coerce")
    target=pd.to_numeric(d.hospital_expire_flag,errors="coerce")
    valid=d.admittime.notna()&target.isin([0,1])&d.subject_id.notna()&d.hadm_id.notna()
    d=d.loc[valid].copy(); d["mortality"]=target[valid].astype(int)
    delta=d.admittime.dt.year-pd.to_numeric(d.anchor_year,errors="coerce")
    age_anchor=pd.to_numeric(d.anchor_age,errors="coerce")
    d["age_topcoded"]=(age_anchor==91).astype(int)
    age=age_anchor+delta
    d["age"]=np.where(d.age_topcoded==1,91,np.minimum(age,91))
    d=d.loc[(age>=18)|(d.age_topcoded==1)].copy()
    flow.append({"stage":"valid_adult_admissions","N":len(d)})
    # Select first eligible adult admission before any temporal filtering.
    d=d.sort_values(["subject_id","admittime","hadm_id"]).drop_duplicates("subject_id",keep="first").copy()
    flow.append({"stage":"one_first_adult_admission_per_patient","N":len(d)})
    groups=d.anchor_year_group.astype(str).str.extract(r"^(\d{4})\s*-\s*(\d{4})$")
    if groups.isna().any().any(): raise ValueError("unrecognized anchor_year_group; inspect source data")
    offset=d.admittime.dt.year-pd.to_numeric(d.anchor_year,errors="raise")
    padding=int(config.get("temporal_padding_years",1))
    if padding<0: raise ValueError("temporal_padding_years must be nonnegative")
    d["admission_year_lower"]=(groups[0].astype(int)+offset-padding).clip(lower=config["dataset_min_year"])
    d["admission_year_upper"]=(groups[1].astype(int)+offset+padding).clip(upper=config["dataset_max_year"])
    if (d.admission_year_lower>d.admission_year_upper).any(): raise ValueError("admission interval outside declared dataset coverage")
    cutoff=int(config["source_last_year"])
    d["split"]=np.where(d.admission_year_upper<=cutoff,"source",np.where(d.admission_year_lower>cutoff,"deployment","excluded_overlap"))
    d["TemporalDomain"]=d.admission_year_lower.astype(int).astype(str)+" - "+d.admission_year_upper.astype(int).astype(str)
    for split in ["source","deployment","excluded_overlap"]: flow.append({"stage":split,"N":int((d.split==split).sum())})
    for col in FEATURES:
        if col not in d: d[col]=np.nan
    for col in CATEGORICAL:
        d[col]=d[col].astype(object).where(d[col].notna(),np.nan)
    return d,pd.DataFrame(flow),{"patients":pp,"admissions":ap}


def build_model(name,seed):
    pre=ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("scale",StandardScaler())]),NUMERIC),
        ("categorical",Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="__MISSING__",keep_empty_features=True)),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CATEGORICAL),
    ])
    if name=="Logistic regression": est=LogisticRegression(max_iter=3000,random_state=seed)
    elif name=="HistGradientBoosting": est=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=seed)
    else: raise ValueError("unknown prespecified MIMIC model")
    return Pipeline([("preprocess",pre),("model",est)])


def validate_config(config):
    required=["dataset_version","dataset_min_year","dataset_max_year","source_last_year","tau","tau_rationale","alpha","missingness_levels","min_group_size","models","seed","bootstrap_draws"]
    if set(required)-set(config): raise ValueError("incomplete MIMIC configuration")
    if config["tau"] is None or not str(config["tau_rationale"]).strip():
        raise ValueError("Set a prospective tau and scientific tau_rationale before examining deployment outcomes")
    if not 0<=config["tau"]<=1 or not 0<config["alpha"]<1: raise ValueError("invalid tau or alpha")
    levels=config["missingness_levels"]
    if levels!=sorted(set(levels)) or not levels or levels[0]!=0 or any(not 0<=m<1 for m in levels): raise ValueError("unique sorted missingness levels must start at zero and lie in [0,1)")
    if not config["models"] or len(set(config["models"]))!=len(config["models"]): raise ValueError("nonempty distinct models required")
    if config["dataset_min_year"]>=config["source_last_year"] or config["source_last_year"]>=config["dataset_max_year"]: raise ValueError("invalid temporal split")


def run_mimic(root,config,output_directory):
    validate_config(config)
    start=time.perf_counter(); out=Path(output_directory); out.mkdir(parents=True,exist_ok=True)
    # Freeze the exact requested configuration before loading any outcomes.
    config_text=json.dumps(config,indent=2,sort_keys=True)
    (out/"mimic_frozen_config.json").write_text(config_text)
    d,flow,files=load_cohort(root,config)
    source=d[d.split=="source"].copy(); target=d[d.split=="deployment"].copy()
    if len(source)<100 or len(target)<config["min_group_size"]: raise ValueError("insufficient source/deployment cohort after interval exclusions")
    if source.mortality.nunique()!=2: raise ValueError("source labels must contain both classes")
    train,nominal=train_test_split(source,test_size=.2,stratify=source.mortality,random_state=config["seed"])
    assert not set(train.subject_id)&set(nominal.subject_id)
    assert not set(source.subject_id)&set(target.subject_id)
    assert d.subject_id.is_unique
    domains=sorted(target.TemporalDomain.unique())
    g=np.column_stack([target.TemporalDomain.eq(domain).to_numpy() for domain in domains])
    y=target.mortality.to_numpy(int); x=target[FEATURES].reset_index(drop=True)
    rng=np.random.default_rng(config["seed"]); uniforms=rng.random(x.shape)
    # Topcoding is provenance, not an observed measurement: preserve it.
    uniforms[:,FEATURES.index("age_topcoded")]=1.
    natural_rows=[]
    for domain in domains:
        subset=target[target.TemporalDomain==domain]
        for feature in FEATURES: natural_rows.append(dict(TemporalDomain=domain,Feature=feature,n=len(subset),natural_missing_fraction=float(subset[feature].isna().mean())))
    columns=[]; meta=[]; diagnostic=[]; mask_rows=[]
    names=list(config["models"])+["Always survive"]
    for name in names:
        if name=="Always survive":
            model=None; source_risk=float(nominal.mortality.mean())
        else:
            model=build_model(name,config["seed"])
            model.fit(train[FEATURES],train.mortality)
            source_risk=float(np.mean(model.predict(nominal[FEATURES])!=nominal.mortality))
        for m in config["missingness_levels"]:
            xm=x.mask(uniforms<m)
            score=np.zeros(len(y)) if model is None else model.predict_proba(xm)[:,1]
            pred=(score>=.5).astype(int)
            columns.append((pred!=y).astype(float)); meta.append((name,m))
            for domain in domains:
                sel=target.TemporalDomain.eq(domain).to_numpy(); yy=y[sel]; pp=pred[sel]; ss=score[sel]
                tn,fp,fn,tp=confusion_matrix(yy,pp,labels=[0,1]).ravel()
                diagnostic.append(dict(Model=name,TemporalDomain=domain,Missingness=m,n=len(yy),mortality_prevalence=float(yy.mean()),risk=float(np.mean(pp!=yy)),source_holdout_risk=source_risk,
                                       sensitivity=tp/(tp+fn) if tp+fn else np.nan,specificity=tn/(tn+fp) if tn+fp else np.nan,
                                       balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else np.nan,
                                       AUROC=roc_auc_score(yy,ss) if len(np.unique(yy))==2 else np.nan,
                                       AUPRC=average_precision_score(yy,ss) if yy.sum() else np.nan))
                if name==names[0]:
                    for feature in FEATURES:
                        mask_rows.append(dict(TemporalDomain=domain,Missingness=m,Feature=feature,total_missing_fraction=float(xm.loc[sel,feature].isna().mean())))
    losses=np.column_stack(columns)
    # One family across all domains, models (including the trivial baseline),
    # and missingness levels. No post-hoc unadjusted model selection.
    rows=[]
    for method in ["multiplier","exact_bonferroni","hoeffding","holm"]:
        if method=="holm":
            sizes=np.repeat(g.sum(axis=0),len(meta)); counts=(g.T@losses).ravel()
            eligible=sizes>=config["min_group_size"]
            pvalues=np.ones(len(sizes))
            from scipy.stats import binom
            pvalues[eligible]=binom.cdf(counts[eligible],sizes[eligible],config["tau"])
            from .baselines import holm_reject
            certified=holm_reject(pvalues,config["alpha"])&eligible
            risk=np.divide(counts,sizes); upper=np.full(len(sizes),np.nan); crit=np.nan
        else:
            res=certify_overlapping_groups(losses,g,config["tau"],alpha=config["alpha"],B=config["bootstrap_draws"],seed=config["seed"]+1,min_group_size=config["min_group_size"],method=method)
            risk,upper,certified,sizes,eligible,crit=res.risks,res.upper,res.certified,res.sample_sizes,res.eligible,res.critical_value
        for i,domain in enumerate(domains):
            for j,(name,m) in enumerate(meta):
                k=i*len(meta)+j
                rows.append(dict(TemporalDomain=domain,Model=name,Missingness=m,method=method,n=int(sizes[k]),risk=risk[k],upper=upper[k],certified=bool(certified[k]),eligible=bool(eligible[k]),critical=crit,tau=config["tau"],alpha=config["alpha"]))
    outputs={"mimic_cohort_flow":flow,"mimic_results":pd.DataFrame(rows),"mimic_diagnostics":pd.DataFrame(diagnostic),"mimic_natural_missingness":pd.DataFrame(natural_rows),"mimic_total_missingness":pd.DataFrame(mask_rows)}
    for name,table in outputs.items(): table.to_csv(out/(name+".csv"),index=False)
    manifest={"status":"completed_credentialed_data","dataset_version":config["dataset_version"],"config_sha256":hashlib.sha256(config_text.encode()).hexdigest(),"seconds":time.perf_counter()-start,
              "source_n":len(source),"train_n":len(train),"nominal_n":len(nominal),"deployment_n":len(target),
              "features":FEATURES,"prediction_threshold":.5,"family_size":len(domains)*len(meta),
              "claim_scope":"admission-table prediction; approximate temporal shift; additional missingness synthetic",
              "source_tables":{name:{"filename":path.name,"size_bytes":path.stat().st_size,"sha256":file_sha256(path)} for name,path in files.items()},
              "output_sha256":{name+".csv":file_sha256(out/(name+".csv")) for name in outputs}}
    (out/"mimic_manifest.json").write_text(json.dumps(manifest,indent=2))
    return outputs,manifest


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
