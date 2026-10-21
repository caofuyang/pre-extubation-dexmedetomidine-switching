#!/usr/bin/env python3
"""Count-only precheck for the v13 redesign (t0 = actual extubation).

Purpose
-------
Before any modelling, answer one question: is there enough signal to make the
new question answerable? This script builds the extubation-anchored cohort,
assigns the exposure group (sedative actually running in the 6 h before
extubation), and counts 48 h / 72 h reintubation events per group. It fits no
model, estimates no effect, and writes no propensity score.

It deliberately does NOT read chartevents (3.5 GB on this dataset and the main
source of past runtime crashes). Only inputevents, procedureevents, icustays,
patients and admissions are scanned.

Run (Windows, in cmd or PowerShell):
    set MIMIC_IV_ROOT=E:\\MIMIC-IV\\data\\mimic-iv-3.1
    python precheck_v13_extubation.py

Outputs (./precheck_output/):
    precheck_summary.md   human-readable waterfall + group table
    precheck_base.csv     one row per eligible extubation (for inspection)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb


def get_root() -> Path:
    raw = os.environ.get("MIMIC_IV_ROOT")
    if not raw:
        sys.exit("ERROR: set MIMIC_IV_ROOT to the extracted MIMIC-IV 3.1 directory.")
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        sys.exit(f"ERROR: MIMIC_IV_ROOT is not a directory: {p}")
    return p


MIMIC = get_root()
HERE = Path(__file__).resolve().parent
OUT = HERE / "precheck_output"
OUT.mkdir(exist_ok=True)


def q(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


PROP = 222168
DEX = (225150, 229420)
EXTUBATION = 227194
UNPLANNED_EXTUBATION = (225468, 225477)
REINTUBATION = 224385
INVASIVE_VENT = 225792
TRACHEOSTOMY = (225448, 226237)
WINDOW_HOURS = 6
OUTCOME_HOURS = 48
DEX_SQL = ",".join(str(i) for i in DEX)
UNPLANNED_SQL = ",".join(str(i) for i in UNPLANNED_EXTUBATION)
TRACH_SQL = ",".join(str(i) for i in TRACHEOSTOMY)

con = duckdb.connect()
con.execute("PRAGMA threads=4")

print("Scanning MIMIC-IV tables ...", flush=True)

con.execute(
    f"""
    create or replace view inp as
    select cast(subject_id as bigint) subject_id,
           cast(hadm_id as bigint) hadm_id,
           cast(stay_id as bigint) stay_id,
           cast(starttime as timestamp) starttime,
           cast(endtime as timestamp) endtime,
           cast(itemid as bigint) itemid,
           coalesce(statusdescription, '') statusdescription
    from read_csv_auto('{q(MIMIC / 'icu/inputevents.csv.gz')}', all_varchar=true, strict_mode=false)
    where cast(itemid as bigint) in ({PROP},{DEX_SQL})
      and coalesce(statusdescription, '') <> 'Rewritten'
    """
)

con.execute(
    f"""
    create or replace view proc as
    select cast(subject_id as bigint) subject_id,
           cast(hadm_id as bigint) hadm_id,
           cast(stay_id as bigint) stay_id,
           cast(starttime as timestamp) starttime,
           cast(endtime as timestamp) endtime,
           cast(itemid as bigint) itemid,
           coalesce(statusdescription, '') statusdescription
    from read_csv_auto('{q(MIMIC / 'icu/procedureevents.csv.gz')}', all_varchar=true, strict_mode=false)
    where cast(itemid as bigint) in ({EXTUBATION},{UNPLANNED_SQL},{REINTUBATION},{INVASIVE_VENT},{TRACH_SQL})
    """
)

con.execute(
    f"""
    create or replace view stat as
    select cast(i.subject_id as bigint) subject_id,
           cast(i.hadm_id as bigint) hadm_id,
           cast(i.stay_id as bigint) stay_id,
           cast(i.intime as timestamp) intime,
           cast(i.outtime as timestamp) outtime,
           i.first_careunit,
           cast(p.anchor_age as double)
             + greatest(0, year(cast(a.admittime as timestamp)) - cast(p.anchor_year as integer)) age,
           try_cast(a.deathtime as timestamp) deathtime,
           try_cast(a.dischtime as timestamp) dischtime
    from read_csv_auto('{q(MIMIC / 'icu/icustays.csv.gz')}', all_varchar=true, strict_mode=false) i
    join read_csv_auto('{q(MIMIC / 'hosp/patients.csv.gz')}', all_varchar=true, strict_mode=false) p
      on cast(p.subject_id as bigint) = cast(i.subject_id as bigint)
    join read_csv_auto('{q(MIMIC / 'hosp/admissions.csv.gz')}', all_varchar=true, strict_mode=false) a
      on cast(a.hadm_id as bigint) = cast(i.hadm_id as bigint)
    """
)

# ---------------------------------------------------------------------------
# Step 1: every documented extubation event, one row per event
# ---------------------------------------------------------------------------
con.execute(
    f"""
    create or replace table ext0 as
    select e.subject_id, e.hadm_id, e.stay_id, e.starttime t0,
           e.statusdescription,
           s.intime, s.outtime, s.first_careunit, s.age, s.deathtime, s.dischtime
    from proc e
    join stat s using (stay_id)
    where e.itemid = {EXTUBATION}
    """
)

n0 = con.sql("select count(*) from ext0").fetchone()[0]

# ---------------------------------------------------------------------------
# Step 2: eligibility flags, applied in sequence so the waterfall is auditable
# ---------------------------------------------------------------------------
con.execute(
    f"""
    create or replace table ext1 as
    select *,
      case when age >= 18 then 1 else 0 end ok_age,
      case when t0 >= intime and t0 < outtime then 1 else 0 end ok_instay,
      case when t0 - interval {WINDOW_HOURS} hour >= intime then 1 else 0 end ok_window,
      case when exists(
             select 1 from proc v
             where v.stay_id = e.stay_id and v.itemid = {INVASIVE_VENT}
               and v.starttime <= e.t0 and v.endtime >= e.t0
           ) then 1 else 0 end ok_vent,
      case when not exists(
             select 1 from proc z
             where z.stay_id = e.stay_id and z.itemid in ({TRACH_SQL})
               and z.starttime <= e.t0
           ) then 1 else 0 end ok_no_trach,
      case when not exists(
             select 1 from proc u
             where u.hadm_id = e.hadm_id and u.itemid in ({UNPLANNED_SQL})
               and u.starttime between e.t0 - interval 1 hour and e.t0 + interval 1 hour
           ) then 1 else 0 end ok_planned,
      case when deathtime is null or deathtime > e.t0 then 1 else 0 end ok_alive
    from ext0 e
    """
)

# ---------------------------------------------------------------------------
# Step 3: what sedative was actually running in the 6 h before extubation
# ---------------------------------------------------------------------------
con.execute(
    f"""
    create or replace table ext2 as
    select *,
      -- propofol already running when the window opened
      coalesce((
        select max(case when x.starttime <= e.t0 - interval {WINDOW_HOURS} hour
                          and x.endtime > e.t0 - interval {WINDOW_HOURS} hour
                        then 1 else 0 end)
        from inp x where x.stay_id = e.stay_id and x.itemid = {PROP}
      ), 0) prop_at_window_start,
      -- hours of propofol actually covered inside the window (capped at 6)
      least({WINDOW_HOURS}, coalesce((
        select sum(date_diff('second',
                             greatest(x.starttime, e.t0 - interval {WINDOW_HOURS} hour),
                             least(x.endtime, e.t0))) / 3600.0
        from inp x
        where x.stay_id = e.stay_id and x.itemid = {PROP}
          and least(x.endtime, e.t0) > greatest(x.starttime, e.t0 - interval {WINDOW_HOURS} hour)
      ), 0)) prop_hours_in_window,
      -- dexmedetomidine started inside the window (the transition trigger)
      (select min(x.starttime)
       from inp x
       where x.stay_id = e.stay_id and x.itemid in ({DEX_SQL})
         and x.starttime > e.t0 - interval {WINDOW_HOURS} hour
         and x.starttime <= e.t0) dex_start_in_window,
      -- dexmedetomidine already running when the window opened
      coalesce((
        select max(case when x.starttime <= e.t0 - interval {WINDOW_HOURS} hour
                          and x.endtime > e.t0 - interval {WINDOW_HOURS} hour
                        then 1 else 0 end)
        from inp x where x.stay_id = e.stay_id and x.itemid in ({DEX_SQL})
      ), 0) dex_at_window_start,
      -- dexmedetomidine covered any part of the window
      (select min(x.starttime)
       from inp x
       where x.stay_id = e.stay_id and x.itemid in ({DEX_SQL})
         and x.starttime <= e.t0 and x.endtime > e.t0 - interval {WINDOW_HOURS} hour) dex_any_in_window
    from ext1 e
    """
)

con.execute(
    f"""
    create or replace table ext3 as
    select *,
      case
        when dex_start_in_window is not null and prop_at_window_start = 1 then 'transition'
        when dex_start_in_window is null and dex_at_window_start = 0
             and prop_hours_in_window >= 0.8 * {WINDOW_HOURS} then 'propofol_only'
        when prop_at_window_start = 0 and dex_at_window_start = 1 then 'dex_only'
        when prop_at_window_start = 0 and dex_start_in_window is null
             and dex_any_in_window is not null then 'dex_only'
        else 'other'
      end exposure_group,
      case when exists(
             select 1 from proc r
             where r.hadm_id = e.hadm_id and r.itemid in ({REINTUBATION},{INVASIVE_VENT})
               and r.starttime > e.t0 and r.starttime <= e.t0 + interval {OUTCOME_HOURS} hour
           ) then 1 else 0 end reintubation_48h,
      case when exists(
             select 1 from proc r
             where r.hadm_id = e.hadm_id and r.itemid in ({REINTUBATION},{INVASIVE_VENT})
               and r.starttime > e.t0 and r.starttime <= e.t0 + interval 72 hour
           ) then 1 else 0 end reintubation_72h,
      case when deathtime is not null and deathtime > e.t0
                and deathtime <= e.t0 + interval {OUTCOME_HOURS} hour
                and not exists(
                  select 1 from proc r
                  where r.hadm_id = e.hadm_id and r.itemid in ({REINTUBATION},{INVASIVE_VENT})
                    and r.starttime > e.t0 and r.starttime < e.deathtime
                ) then 1 else 0 end death_before_reintubation_48h,
      case when dischtime is not null and dischtime < e.t0 + interval {OUTCOME_HOURS} hour
                and (deathtime is null or deathtime > dischtime) then 1 else 0 end live_discharge_before_48h,
      case
        when lower(coalesce(first_careunit, '')) like '%cvicu%'
          or lower(coalesce(first_careunit, '')) like '%csru%'
          or lower(coalesce(first_careunit, '')) like '%cardiac%'
          or lower(coalesce(first_careunit, '')) like '%coronary%'
          or lower(coalesce(first_careunit, '')) like '%ccu%' then 'cardiac'
        when lower(coalesce(first_careunit, '')) like '%neuro%' then 'neuro'
        else 'other'
      end unit_class
    from ext2 e
    """
)

# ---------------------------------------------------------------------------
# Step 4: waterfall
# ---------------------------------------------------------------------------
waterfall = [
    ("All documented extubation events (itemid 227194)", "1=1"),
    ("Age >= 18", "ok_age = 1"),
    ("Extubation inside an ICU stay", "ok_age = 1 and ok_instay = 1"),
    ("Full 6 h window inside the ICU stay", "ok_age = 1 and ok_instay = 1 and ok_window = 1"),
    ("Invasive ventilation covering t0", "ok_age = 1 and ok_instay = 1 and ok_window = 1 and ok_vent = 1"),
    ("No tracheostomy before t0", "ok_age = 1 and ok_instay = 1 and ok_window = 1 and ok_vent = 1 and ok_no_trach = 1"),
    ("Planned extubation (not self-extubation)", "ok_age = 1 and ok_instay = 1 and ok_window = 1 and ok_vent = 1 and ok_no_trach = 1 and ok_planned = 1"),
    ("Alive at t0", "ok_age = 1 and ok_instay = 1 and ok_window = 1 and ok_vent = 1 and ok_no_trach = 1 and ok_planned = 1 and ok_alive = 1"),
]

waterfall_rows = []
for label, cond in waterfall:
    n = con.sql(f"select count(*) from ext3 where {cond}").fetchone()[0]
    waterfall_rows.append((label, int(n)))

con.execute(
    """
    create or replace table base as
    select * from ext3
    where ok_age = 1 and ok_instay = 1 and ok_window = 1 and ok_vent = 1
      and ok_no_trach = 1 and ok_planned = 1 and ok_alive = 1
    """
)

# ---------------------------------------------------------------------------
# Step 5: the numbers that decide whether this question is viable
# ---------------------------------------------------------------------------
group_sql = """
select exposure_group,
       count(*) n,
       sum(reintubation_48h) events_48h,
       round(100.0 * sum(reintubation_48h) / count(*), 1) pct_48h,
       sum(reintubation_72h) events_72h,
       round(100.0 * sum(reintubation_72h) / count(*), 1) pct_72h,
       sum(death_before_reintubation_48h) deaths_before_48h,
       sum(live_discharge_before_48h) early_live_discharge
from base group by 1 order by n desc
"""
groups = con.sql(group_sql).df()

unit_sql = """
select unit_class, exposure_group, count(*) n,
       sum(reintubation_48h) events_48h
from base group by 1, 2 order by 1, 3 desc
"""
units = con.sql(unit_sql).df()

main_sql = """
select exposure_group, count(*) n, sum(reintubation_48h) events_48h,
       round(100.0 * sum(reintubation_48h) / count(*), 1) pct_48h
from base
where unit_class = 'other'
group by 1 order by n desc
"""
main_only = con.sql(main_sql).df()

con.sql("select * from base").df().to_csv(OUT / "precheck_base.csv", index=False)

lines = []
lines.append("# v13 precheck (t0 = extubation) — count only, no model\n")
lines.append(f"MIMIC_IV_ROOT = `{MIMIC}`\n")
lines.append("## Cohort waterfall\n")
lines.append("| Step | N |")
lines.append("|---|---:|")
for label, n in waterfall_rows:
    lines.append(f"| {label} | {n:,} |")

lines.append("\n## Exposure group (all eligible extubations)\n")
lines.append("| Group | N | Reintubation 48 h | % 48 h | Reintubation 72 h | % 72 h | Death before reintubation (48 h) | Live discharge < 48 h |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for _, r in groups.iterrows():
    lines.append(
        f"| {r.exposure_group} | {int(r.n):,} | {int(r.events_48h):,} | {r.pct_48h} | "
        f"{int(r.events_72h):,} | {r.pct_72h} | {int(r.deaths_before_48h):,} | {int(r.early_live_discharge):,} |"
    )

lines.append("\n## Main analysis restriction (cardiac and neuro ICUs excluded)\n")
lines.append("| Group | N | Reintubation 48 h | % 48 h |")
lines.append("|---|---:|---:|---:|")
for _, r in main_only.iterrows():
    lines.append(f"| {r.exposure_group} | {int(r.n):,} | {int(r.events_48h):,} | {r.pct_48h} |")

lines.append("\n## By unit class and exposure group\n")
lines.append("| Unit class | Group | N | Reintubation 48 h |")
lines.append("|---|---|---:|---:|")
for _, r in units.iterrows():
    lines.append(f"| {r.unit_class} | {r.exposure_group} | {int(r.n):,} | {int(r.events_48h):,} |")

lines.append(
    "\n## How to read this\n"
    "The redesign is only worth building if the `transition` group has enough 48 h events "
    "(rule of thumb: at least ~50-60, and at least ~10 events per covariate you intend to adjust for). "
    "If `transition` is thin, the fix is to widen the exposure window (12 h), relax `propofol_only` "
    "coverage, or move the outcome to 72 h — not to add covariates.\n"
)

report = "\n".join(lines)
(OUT / "precheck_summary.md").write_text(report, encoding="utf-8")
print(report)
print(f"\nBase table written to {OUT / 'precheck_base.csv'}")

if n0 == 0:
    print("\nWARNING: no extubation events were found. Check MIMIC_IV_ROOT and the itemid 227194 mapping.")
