#!/usr/bin/env python3
"""v13 analysis (version 2): pre-extubation propofol-to-dexmedetomidine transition
and extubation failure, anchored on the extubation event (MIMIC-IV 3.1).

Change from version 1: all estimators are now reported on the SAME scale, the
48-hour risk of reintubation or restart of invasive ventilation. Version 1 mixed
weighted cumulative incidence (death as competing event) with crude proportions,
which is not a like-for-like comparison. The competing-risk cumulative incidence
is retained as a sensitivity analysis.

Estimators reported on the common scale:
  * crude
  * overlap-weighted (primary)
  * g-computation (outcome model, standardized to the cohort)
  * inverse-probability-of-treatment weighting, targeting the treated

Run:
    set MIMIC_IV_ROOT=E:\\MIMIC-IV\\data\\mimic-iv-3.1
    python v13_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "outputs"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SEED = 20261021
BOOT = 300
WINDOW = 6
COV_LOOKBACK = 24
HORIZON = 48

PROP = 222168
DEX = (225150, 229420)
EXTUBATION = 227194
UNPLANNED = (225468, 225477)
REINTUBATION = 224385
VENT = 225792
TRACH = (225448, 226237)
CHART_ITEMS = (220045, 220052, 220181, 220277, 220339, 223835, 224688, 224689,
               224690, 224686, 224701, 223849, 229314, 224715, 224716, 224717, 228096)
DELIRIUM_ITEM = 228332
LAB_ITEMS = (50813, 50912, 51265, 50885, 51300, 51301, 51222)
VASO = (221289, 221749, 221906, 222315, 229617, 229630, 229631, 229632)
OPIOID = (221744, 221833, 225154, 225942, 225972)
MIDAZOLAM = (221668,)
KETAMINE = (221712, 227211)
INPUT_ITEMS = tuple(sorted({PROP, *DEX, *VASO, *OPIOID, *MIDAZOLAM, *KETAMINE}))
PROC_ITEMS = tuple(sorted({EXTUBATION, *UNPLANNED, REINTUBATION, VENT, *TRACH}))


def sql_list(values) -> str:
    return ",".join(str(int(v)) for v in values)


def qpath(path: Path) -> str:
    return str(path).replace("'", "''")


def build_caches(con, MIMIC: Path) -> dict[str, Path]:
    files = {
        "input": CACHE / "mimic_input.parquet",
        "proc": CACHE / "mimic_procedure.parquet",
        "chart": CACHE / "mimic_chart.parquet",
        "cam": CACHE / "mimic_cam.parquet",
        "lab": CACHE / "mimic_lab.parquet",
        "stat": CACHE / "mimic_static.parquet",
    }
    if all(p.exists() for p in files.values()):
        return files

    def build(target: Path, query: str) -> None:
        if target.exists():
            return
        print("Building", target.name, flush=True)
        con.execute(f"copy ({query}) to '{qpath(target)}' (format parquet, compression zstd)")

    build(files["input"], f"""
        select cast(subject_id as bigint) subject_id, cast(hadm_id as bigint) hadm_id,
               cast(stay_id as bigint) stay_id, cast(starttime as timestamp) starttime,
               cast(endtime as timestamp) endtime, cast(itemid as bigint) itemid,
               try_cast(rate as double) rate, rateuom, coalesce(statusdescription,'') statusdescription,
               try_cast(patientweight as double) patientweight
        from read_csv_auto('{qpath(MIMIC / 'icu/inputevents.csv.gz')}', all_varchar=true, strict_mode=false)
        where cast(itemid as bigint) in ({sql_list(INPUT_ITEMS)})
    """)
    build(files["proc"], f"""
        select cast(subject_id as bigint) subject_id, cast(hadm_id as bigint) hadm_id,
               cast(stay_id as bigint) stay_id, cast(starttime as timestamp) starttime,
               cast(endtime as timestamp) endtime, cast(itemid as bigint) itemid
        from read_csv_auto('{qpath(MIMIC / 'icu/procedureevents.csv.gz')}', all_varchar=true, strict_mode=false)
        where cast(itemid as bigint) in ({sql_list(PROC_ITEMS)})
    """)
    build(files["chart"], f"""
        select cast(stay_id as bigint) stay_id, cast(charttime as timestamp) charttime,
               cast(itemid as bigint) itemid, try_cast(valuenum as double) valuenum, value
        from read_csv_auto('{qpath(MIMIC / 'icu/chartevents.csv.gz')}', all_varchar=true, strict_mode=false)
        where cast(itemid as bigint) in ({sql_list(CHART_ITEMS)}) and coalesce(value,'') <> 'Error'
    """)
    build(files["cam"], f"""
        select cast(stay_id as bigint) stay_id, cast(charttime as timestamp) charttime,
               cast(itemid as bigint) itemid, value
        from read_csv_auto('{qpath(MIMIC / 'icu/chartevents.csv.gz')}', all_varchar=true, strict_mode=false)
        where cast(itemid as bigint) = {DELIRIUM_ITEM}
    """)
    build(files["lab"], f"""
        select cast(hadm_id as bigint) hadm_id, cast(charttime as timestamp) charttime,
               cast(itemid as bigint) itemid, try_cast(valuenum as double) valuenum
        from read_csv_auto('{qpath(MIMIC / 'hosp/labevents.csv.gz')}', all_varchar=true, strict_mode=false, ignore_errors=true)
        where cast(itemid as bigint) in ({sql_list(LAB_ITEMS)}) and try_cast(valuenum as double) is not null
    """)
    build(files["stat"], f"""
        select cast(i.subject_id as bigint) subject_id, cast(i.hadm_id as bigint) hadm_id,
               cast(i.stay_id as bigint) stay_id, cast(i.intime as timestamp) intime,
               cast(i.outtime as timestamp) outtime, i.first_careunit, p.gender,
               cast(p.anchor_age as double) + greatest(0, year(cast(a.admittime as timestamp)) - cast(p.anchor_year as integer)) age,
               a.admission_type, try_cast(a.deathtime as timestamp) deathtime,
               try_cast(a.dischtime as timestamp) dischtime
        from read_csv_auto('{qpath(MIMIC / 'icu/icustays.csv.gz')}', all_varchar=true, strict_mode=false) i
        join read_csv_auto('{qpath(MIMIC / 'hosp/patients.csv.gz')}', all_varchar=true, strict_mode=false) p
          on cast(p.subject_id as bigint) = cast(i.subject_id as bigint)
        join read_csv_auto('{qpath(MIMIC / 'hosp/admissions.csv.gz')}', all_varchar=true, strict_mode=false) a
          on cast(a.hadm_id as bigint) = cast(i.hadm_id as bigint)
    """)
    return files


def build_cohort(con, files: dict[str, Path]) -> pd.DataFrame:
    for name, key in [("inp", "input"), ("proc", "proc"), ("chart", "chart"),
                      ("cam", "cam"), ("lab", "lab"), ("stat", "stat")]:
        con.execute(f"create or replace view {name} as select * from '{qpath(files[key])}'")

    con.execute(f"""
    create or replace table trials as
    with ext as (
      select e.subject_id, e.hadm_id, e.stay_id, e.starttime t0,
             s.intime, s.outtime, s.first_careunit, s.gender, s.age,
             s.admission_type, s.deathtime, s.dischtime
      from proc e join stat s using (stay_id) where e.itemid = {EXTUBATION}
    ),
    elig as (
      select *, row_number() over (order by subject_id, t0) trial_id
      from ext e
      where age >= 18 and t0 >= intime and t0 < outtime
        and t0 - interval {WINDOW} hour >= intime
        and exists (select 1 from proc v where v.stay_id = e.stay_id and v.itemid = {VENT}
                    and v.starttime <= e.t0 and v.endtime >= e.t0)
        and not exists (select 1 from proc z where z.stay_id = e.stay_id
                        and z.itemid in ({sql_list(TRACH)}) and z.starttime <= e.t0)
        and not exists (select 1 from proc u where u.hadm_id = e.hadm_id
                        and u.itemid in ({sql_list(UNPLANNED)})
                        and u.starttime between e.t0 - interval 1 hour and e.t0 + interval 1 hour)
        and (deathtime is null or deathtime > e.t0)
    )
    select e.*,
      coalesce((select max(case when x.starttime <= e.t0 - interval {WINDOW} hour
                                 and x.endtime > e.t0 - interval {WINDOW} hour then 1 else 0 end)
                from inp x where x.stay_id = e.stay_id and x.itemid = {PROP}), 0) prop_at_window_start,
      least({WINDOW}, coalesce((select sum(date_diff('second',
                greatest(x.starttime, e.t0 - interval {WINDOW} hour),
                least(x.endtime, e.t0))) / 3600.0
                from inp x where x.stay_id = e.stay_id and x.itemid = {PROP}
                and least(x.endtime, e.t0) > greatest(x.starttime, e.t0 - interval {WINDOW} hour)), 0)) prop_hours,
      (select min(x.starttime) from inp x where x.stay_id = e.stay_id
        and x.itemid in ({sql_list(DEX)}) and x.starttime > e.t0 - interval {WINDOW} hour
        and x.starttime <= e.t0) dex_start,
      coalesce((select max(case when x.starttime <= e.t0 - interval {WINDOW} hour
                                 and x.endtime > e.t0 - interval {WINDOW} hour then 1 else 0 end)
                from inp x where x.stay_id = e.stay_id and x.itemid in ({sql_list(DEX)})), 0) dex_at_window_start,
      (select min(x.starttime) from inp x where x.stay_id = e.stay_id
        and x.itemid in ({sql_list(DEX)}) and x.starttime <= e.t0
        and x.endtime > e.t0 - interval {WINDOW} hour) dex_any_window,
      (select min(v.starttime) from proc v where v.stay_id = e.stay_id and v.itemid = {VENT}
        and v.starttime <= e.t0 and v.endtime >= e.t0) vent_start
    from elig e
    """)

    con.execute(f"""
    create or replace table ch as
    select t.trial_id,
      min(case when c.itemid = 228096 then c.valuenum end) rass_min,
      min(case when c.itemid in (220052,220181) and c.valuenum between 20 and 200 then c.valuenum end) map_min,
      max(case when c.itemid = 220045 and c.valuenum between 20 and 250 then c.valuenum end) hr_max,
      max(case when c.itemid in (224690,224689,224688) and c.valuenum between 1 and 80 then c.valuenum end) rr_max,
      min(case when c.itemid = 220277 and c.valuenum between 50 and 100 then c.valuenum end) spo2_min,
      max(case when c.itemid = 220339 and c.valuenum between 0 and 30 then c.valuenum end) peep_max,
      max(case when c.itemid = 223835 and c.valuenum between 21 and 100 then c.valuenum end) fio2_max,
      max(case when c.itemid = 224715 then 1 else 0 end) sbt_any
    from trials t join chart c on c.stay_id = t.stay_id
    where c.charttime >= t.t0 - interval {COV_LOOKBACK} hour and c.charttime < t.t0 - interval {WINDOW} hour
    group by 1
    """)

    con.execute(f"""
    create or replace table dl as
    select t.trial_id, max(case when c.value = 'Positive' then 1 else 0 end) delirium_24h
    from trials t join cam c on c.stay_id = t.stay_id
    where c.charttime >= t.t0 - interval {COV_LOOKBACK} hour and c.charttime < t.t0 - interval {WINDOW} hour
    group by 1
    """)

    con.execute(f"""
    create or replace table lb as
    select t.trial_id,
      max(case when l.itemid = 50813 then l.valuenum end) lactate,
      max(case when l.itemid = 50912 then l.valuenum end) creatinine,
      min(case when l.itemid = 51265 then l.valuenum end) platelets,
      max(case when l.itemid = 50885 then l.valuenum end) bilirubin,
      max(case when l.itemid in (51300,51301) then l.valuenum end) wbc,
      min(case when l.itemid = 51222 then l.valuenum end) hemoglobin
    from trials t join lab l on l.hadm_id = t.hadm_id
    where l.charttime >= t.t0 - interval {COV_LOOKBACK} hour and l.charttime < t.t0 - interval {WINDOW} hour
    group by 1
    """)

    con.execute(f"""
    create or replace table ia as
    select t.trial_id,
      max(case when x.itemid in ({sql_list(VASO)}) then 1 else 0 end) vaso_any,
      max(case when x.itemid in ({sql_list(OPIOID)}) then 1 else 0 end) opioid_any,
      max(case when x.itemid in ({sql_list(MIDAZOLAM)}) then 1 else 0 end) midazolam_any,
      max(case when x.itemid in ({sql_list(KETAMINE)}) then 1 else 0 end) ketamine_any,
      median(case when x.itemid = {PROP} and x.rateuom = 'mcg/kg/min'
                   and x.rate > 0 and x.rate <= 200 then x.rate end) propofol_rate,
      max(case when x.itemid in ({sql_list(DEX)}) then 1 else 0 end) dex_before_window
    from trials t join inp x on x.stay_id = t.stay_id
    where x.starttime < t.t0 - interval {WINDOW} hour and x.endtime > t.t0 - interval {COV_LOOKBACK} hour
    group by 1
    """)

    con.execute(f"""
    create or replace table oc as
    select t.trial_id trial_id_oc,
      (select min(r.starttime) from proc r where r.hadm_id = t.hadm_id
        and r.itemid in ({REINTUBATION},{VENT})
        and r.starttime > t.t0 and r.starttime <= t.t0 + interval 72 hour) reintubation_time,
      (select min(r.starttime) from proc r where r.hadm_id = t.hadm_id
        and r.itemid = {REINTUBATION}
        and r.starttime > t.t0 and r.starttime <= t.t0 + interval 72 hour) reintubation_airway_only
    from trials t
    """)

    con.execute(f"""
    create or replace table analytic as
    select t.*, ch.* exclude (trial_id), dl.* exclude (trial_id), lb.* exclude (trial_id),
           ia.* exclude (trial_id), oc.* exclude (trial_id_oc),
      case when t.prop_at_window_start = 1 and t.dex_at_window_start = 0 and t.dex_start is not null then 'transition'
           when t.prop_at_window_start = 1 and t.dex_at_window_start = 0 and t.prop_hours >= 0.8 * {WINDOW} then 'propofol_only'
           when t.prop_at_window_start = 1 and t.dex_at_window_start = 0 then 'propofol_stopped_early'
           when t.prop_at_window_start = 0 and t.dex_at_window_start = 1 then 'dex_only'
           when t.prop_at_window_start = 1 and t.dex_at_window_start = 1 then 'concurrent_at_window_start'
           when t.prop_at_window_start = 0 and t.dex_start is not null then 'dex_started_in_window'
           else 'no_sedative' end grp,
      case when lower(coalesce(t.first_careunit,'')) like '%cvicu%'
             or lower(coalesce(t.first_careunit,'')) like '%csru%'
             or lower(coalesce(t.first_careunit,'')) like '%cardiac%'
             or lower(coalesce(t.first_careunit,'')) like '%coronary%'
             or lower(coalesce(t.first_careunit,'')) like '%ccu%' then 'cardiac'
           when lower(coalesce(t.first_careunit,'')) like '%neuro%' then 'neuro'
           else 'medical_surgical' end unit_class,
      date_diff('second', t.vent_start, t.t0) / 3600.0 vent_hours,
      case when oc.reintubation_time is not null
                and oc.reintubation_time <= t.t0 + interval {HORIZON} hour
                and (t.deathtime is null or oc.reintubation_time < t.deathtime) then 1
           when t.deathtime is not null and t.deathtime <= t.t0 + interval {HORIZON} hour then 2
           else 0 end event_type,
      greatest(0.01, least({HORIZON}, date_diff('second', t.t0,
        coalesce(oc.reintubation_time, t.deathtime, t.dischtime, t.t0 + interval {HORIZON} hour))/3600.0)) time_h
    from trials t
    left join ch using (trial_id)
    left join dl using (trial_id)
    left join lb using (trial_id)
    left join ia using (trial_id)
    left join oc on oc.trial_id_oc = t.trial_id
    """)

    df = con.sql("select * from analytic").df()
    df["event_48h"] = ((df.event_type == 1) & (df.time_h <= HORIZON)).astype(int)
    df["log_propofol_rate"] = np.log1p(df.propofol_rate.fillna(0))
    return df


PARTS = {
    "numeric": ["age", "vent_hours", "rass_min", "fio2_max", "peep_max",
                "lactate", "creatinine", "log_propofol_rate"],
    "categorical": ["gender", "unit_class", "admission_type"],
    "binary": ["vaso_any", "opioid_any", "sbt_any", "delirium_24h"],
}


def design(df: pd.DataFrame):
    work = df.copy()
    for c in PARTS["binary"]:
        work[c] = work[c].fillna(0).astype(int)
    cols = PARTS["numeric"] + PARTS["categorical"] + PARTS["binary"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         PARTS["numeric"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), PARTS["categorical"]),
    ])
    return work, cols, pre


def fit_ps(df: pd.DataFrame):
    work, cols, pre = design(df)
    model = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=4000, C=0.1))])
    model.fit(work[cols], work.treatment)
    work["ps"] = model.predict_proba(work[cols])[:, 1]
    weight_treated = 1 - work["ps"]
    weight_control = work["ps"]
    work["overlap_weight"] = np.where(work.treatment == 1, weight_treated, weight_control)
    work["att_weight"] = np.where(work.treatment == 1, 1.0, work["ps"] / (1 - work["ps"]))
    work["gcomp_weight"] = 1.0
    work["_X"] = list(range(len(work)))
    return work, model, cols, pre


def weighted_risk(group: pd.DataFrame, column: str = "overlap_weight") -> float:
    if group.empty:
        return float("nan")
    w = group[column].to_numpy(float)
    return float(np.average(group["event_48h"].to_numpy(float), weights=w))


def cif_risk(group: pd.DataFrame) -> float:
    if group.empty:
        return float("nan")
    x = group.sort_values("time_h")
    w = x.overlap_weight.to_numpy(float)
    t = x.time_h.to_numpy(float)
    e = x.event_type.to_numpy(int)
    surv, cum = 1.0, 0.0
    for ti in np.unique(t):
        at = w[t >= ti].sum()
        if at <= 0:
            break
        d1 = w[(t == ti) & (e == 1)].sum()
        d2 = w[(t == ti) & (e == 2)].sum()
        cum += surv * d1 / at
        surv *= 1 - (d1 + d2) / at
    return float(cum)


def summarize(work: pd.DataFrame, column: str) -> dict:
    r1 = weighted_risk(work[work.treatment == 1], column)
    r0 = weighted_risk(work[work.treatment == 0], column)
    return {"risk_treated": r1, "risk_control": r0, "rd": r1 - r0,
            "rr": r1 / r0 if r0 > 0 else float("nan")}


def gcomp(df: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression as LR
    work, _, _ = design(df)
    numeric = PARTS["numeric"] + ["treatment"]
    cat = PARTS["categorical"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    X = work[numeric + cat].copy()
    model = Pipeline([("pre", pre), ("lr", LR(max_iter=4000, C=0.1))])
    model.fit(X, work["event_48h"])
    X1 = X.copy(); X1["treatment"] = 1
    X0 = X.copy(); X0["treatment"] = 0
    r1 = float(model.predict_proba(X1)[:, 1].mean())
    r0 = float(model.predict_proba(X0)[:, 1].mean())
    return {"risk_treated": r1, "risk_control": r0, "rd": r1 - r0,
            "rr": r1 / r0 if r0 > 0 else float("nan")}


def all_estimators(df: pd.DataFrame) -> pd.DataFrame:
    work, *_ = fit_ps(df)
    rows = []
    crude = summarize(work.assign(overlap_weight=1.0), "overlap_weight")
    crude["estimator"] = "Crude"
    rows.append(crude)
    overlap = summarize(work, "overlap_weight")
    overlap["estimator"] = "Overlap weighted"
    rows.append(overlap)
    rows.append({**gcomp(df), "estimator": "g-computation"})
    att = summarize(work, "att_weight")
    att["estimator"] = "IPW (ATT)"
    rows.append(att)
    return pd.DataFrame(rows)


def bootstrap_ci(df: pd.DataFrame, estimator: str, reps: int = BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    subjects = df.subject_id.unique()
    out = []
    for _ in range(reps):
        pick = rng.choice(subjects, len(subjects), replace=True)
        sample = df.set_index("subject_id").loc[pick].reset_index()
        if sample.treatment.nunique() < 2:
            continue
        try:
            if estimator == "g-computation":
                e = gcomp(sample)
            else:
                work, *_ = fit_ps(sample)
                col = {"Crude": "gcomp_weight", "Overlap weighted": "overlap_weight",
                       "IPW (ATT)": "att_weight"}[estimator]
                e = summarize(work, col)
        except Exception:
            continue
        if np.isfinite(e["rr"]):
            out.append((e["rd"], e["rr"]))
    arr = np.array(out)
    return {"rd_low": float(np.percentile(arr[:, 0], 2.5)),
            "rd_high": float(np.percentile(arr[:, 0], 97.5)),
            "rr_low": float(np.percentile(arr[:, 1], 2.5)),
            "rr_high": float(np.percentile(arr[:, 1], 97.5)),
            "reps": int(len(arr))}


def balance_table(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ["age", "vent_hours", "rass_min", "fio2_max",
              "peep_max", "lactate", "creatinine", "log_propofol_rate",
              "vaso_any", "opioid_any", "sbt_any", "delirium_24h"]:
        x = work[c].to_numpy(float)
        ok = ~np.isnan(x)
        xx, tt = x[ok], work.treatment.to_numpy()[ok]

        def smd(w):
            m1 = np.average(xx[tt == 1], weights=w[tt == 1])
            m0 = np.average(xx[tt == 0], weights=w[tt == 0])
            v1 = np.average((xx[tt == 1] - m1) ** 2, weights=w[tt == 1])
            v0 = np.average((xx[tt == 0] - m0) ** 2, weights=w[tt == 0])
            return (m1 - m0) / np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 0.0

        rows.append({"variable": c,
                     "smd_unweighted": smd(np.ones(len(xx))),
                     "smd_weighted": smd(work.overlap_weight.to_numpy()[ok]),
                     "treated_mean": float(np.nanmean(work.loc[work.treatment == 1, c])),
                     "control_mean": float(np.nanmean(work.loc[work.treatment == 0, c]))})
    return pd.DataFrame(rows)


def strata_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit in ["cardiac", "medical_surgical", "neuro"]:
        sub = df[df.unit_class == unit]
        if sub.treatment.sum() < 20 or (sub.treatment == 0).sum() < 20:
            continue
        work, *_ = fit_ps(sub)
        e = summarize(work, "overlap_weight")
        ci = bootstrap_ci(sub, "Overlap weighted", reps=200, seed=SEED + 1)
        rows.append({**e, **ci, "stratum": unit,
                     "n_treated": int((sub.treatment == 1).sum()),
                     "events_treated": int(sub.loc[sub.treatment == 1, "event_48h"].sum()),
                     "n_control": int((sub.treatment == 0).sum()),
                     "events_control": int(sub.loc[sub.treatment == 0, "event_48h"].sum())})
    return pd.DataFrame(rows)


def sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    terminal = (df.event_type == 2) & (df.time_h <= 24)

    airway = df.copy()
    airway["event_48h"] = 0
    mask = airway.reintubation_airway_only.notna()
    hours = (pd.to_datetime(airway.loc[mask, "reintubation_airway_only"])
             - pd.to_datetime(airway.loc[mask, "t0"])).dt.total_seconds() / 3600.0
    airway.loc[mask & (hours > 0) & (hours <= HORIZON), "event_48h"] = 1

    variants = {
        "primary_48h_risk": df,
        "excluding_terminal_extubation": df[~terminal],
        "outcome_airway_reintubation_only": airway,
    }
    for label, sub in variants.items():
        work, *_ = fit_ps(sub)
        e = summarize(work, "overlap_weight")
        rows.append({**e, "variant": label})

    work, *_ = fit_ps(df)
    c1 = cif_risk(work[work.treatment == 1])
    c0 = cif_risk(work[work.treatment == 0])
    rows.append({"risk_treated": c1, "risk_control": c0, "rd": c1 - c0,
                 "rr": c1 / c0, "variant": "competing_risk_cumulative_incidence"})
    return pd.DataFrame(rows)


def main() -> None:
    root = os.environ.get("MIMIC_IV_ROOT")
    if not root:
        sys.exit("Set MIMIC_IV_ROOT to the extracted MIMIC-IV 3.1 directory.")
    MIMIC = Path(root).expanduser().resolve()
    if not MIMIC.is_dir():
        sys.exit(f"MIMIC_IV_ROOT is not a directory: {MIMIC}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    files = build_caches(con, MIMIC)
    df = build_cohort(con, files)

    analytic = df[df.grp.isin(["transition", "propofol_only"])].copy()
    analytic["treatment"] = (analytic.grp == "transition").astype(int)

    estimators = all_estimators(analytic)
    primary = estimators[estimators.estimator == "Overlap weighted"].iloc[0].to_dict()
    primary.update(bootstrap_ci(analytic, "Overlap weighted"))
    for name in ["Crude", "Overlap weighted", "g-computation", "IPW (ATT)"]:
        row = estimators.index[estimators.estimator == name][0]
        ci = bootstrap_ci(analytic, name)
        for k, v in ci.items():
            estimators.loc[row, k] = v

    work, *_ = fit_ps(analytic)
    balance = balance_table(work)
    strata = strata_table(analytic)
    sens = sensitivity_table(analytic)

    primary["n_treated"] = int((analytic.treatment == 1).sum())
    primary["n_control"] = int((analytic.treatment == 0).sum())
    primary["events_treated"] = int(analytic.loc[analytic.treatment == 1, "event_48h"].sum())
    primary["events_control"] = int(analytic.loc[analytic.treatment == 0, "event_48h"].sum())
    primary["e_value"] = _e_value(primary["rr"])
    primary["e_value_ci"] = _e_value(primary["rr_low"])
    primary["auc"] = float(_auc(work))

    estimators.to_csv(OUT / "table2_estimators.csv", index=False)
    balance.to_csv(OUT / "table3_balance.csv", index=False)
    strata.to_csv(OUT / "table4_strata.csv", index=False)
    sens.to_csv(OUT / "table5_sensitivity.csv", index=False)
    pd.DataFrame([primary]).to_csv(OUT / "table2_primary.csv", index=False)
    analytic.to_csv(OUT / "analytic_cohort.csv", index=False)
    with open(OUT / "primary_estimate.json", "w", encoding="utf-8") as fh:
        json.dump(primary, fh, indent=2, default=float)

    lines = ["# v13 primary results (common 48-hour risk scale)\n",
             f"Extubations analysed: {primary['n_treated'] + primary['n_control']:,} "
             f"({primary['n_treated']:,} transition, {primary['n_control']:,} continued propofol)\n",
             f"Primary (overlap weighted): {100*primary['risk_treated']:.1f}% vs "
             f"{100*primary['risk_control']:.1f}%; RD {100*primary['rd']:+.2f} pp "
             f"({100*primary['rd_low']:+.2f} to {100*primary['rd_high']:+.2f}); "
             f"RR {primary['rr']:.2f} ({primary['rr_low']:.2f} to {primary['rr_high']:.2f})\n",
             f"E-value {primary['e_value']:.2f} (CI bound {primary['e_value_ci']:.2f}); "
             f"propensity-score AUC {primary['auc']:.3f}\n",
             "\n## Estimators on the common scale\n", estimators.to_string(index=False),
             "\n## Strata\n", strata.to_string(index=False),
             "\n## Sensitivity\n", sens.to_string(index=False),
             "\n## Balance\n", balance.to_string(index=False)]
    (OUT / "results_report.md").write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    print("Done. Outputs in", OUT)


def _auc(work: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(work.treatment, work.ps)


def _e_value(rr: float) -> float:
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


if __name__ == "__main__":
    main()
