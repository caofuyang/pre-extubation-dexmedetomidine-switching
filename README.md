# Pre-extubation switching from propofol to dexmedetomidine and extubation failure

Analysis code for the manuscript:

> Switching from propofol to dexmedetomidine before extubation and extubation failure:
> a target trial emulation in 15,970 extubations

The study uses MIMIC-IV version 3.1. The database is not redistributed here; it is
available from PhysioNet to credentialed users (https://physionet.org/content/mimiciv/3.1/).

## What the code does

`v13_analysis.py` builds an extubation-anchored cohort and estimates the association
between initiating dexmedetomidine in the six hours before extubation (in patients
already receiving propofol) and reintubation or restart of invasive ventilation
within 48 hours. Time zero is the extubation event, not readiness for extubation,
which removes the overlap between exposure and outcome that affects
readiness-anchored designs.

The script reports the primary overlap-weighted estimate and, deliberately, two
alternative estimators (g-computation and inverse-probability-of-treatment
weighting) because they do not agree. The disagreement is a finding of the study,
not an error, and the manuscript reports the full range.

`precheck_v13_extubation.py` is the count-only feasibility check that was run before
any modelling. It reads no chart data and completes in a few minutes.

## Requirements

Python 3.10+, `duckdb`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` (see
`requirements.txt`).

## Running

```
set MIMIC_IV_ROOT=E:\MIMIC-IV\data\mimic-iv-3.1
python v13_analysis.py
```

The first run builds filtered parquet caches from the CSV files in `icu/` and `hosp/`
and takes about six minutes, most of it reading `chartevents.csv.gz`. Subsequent runs
reuse `cache/`.

Outputs are written to `outputs/`: `table2_primary.csv`, `table3_balance.csv`,
`table4_strata.csv`, `table5_sensitivity.csv`, `primary_estimate.json`, and
`results_report.md`.

## Key definitions

| Element | Definition |
|---|---|
| Time zero | Each documented planned extubation (procedure itemid 227194) |
| Exposure window | The six hours ending at extubation |
| Transition | Propofol running at window start, no dexmedetomidine at window start, dexmedetomidine started during the window |
| Continued propofol | Propofol at window start, no dexmedetomidine during the window, propofol covering at least 80% of the window |
| Outcome | Reintubation (224385) or restart of invasive ventilation (225792) within 48 h |
| Competing event | Death within 48 h without reintubation |

## Licence

MIT. See `LICENSE`.
