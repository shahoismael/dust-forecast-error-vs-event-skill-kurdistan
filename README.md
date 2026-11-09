# dust-forecast-error-vs-event-skill-kurdistan

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24051854.svg)](https://doi.org/10.5281/zenodo.24051854)

Analysis code for the article *Satellite-Based Error and Observer-Based Event Skill Rank Day-Ahead Dust Hindcasts Differently over the Kurdistan Region and Mesopotamia*.

Day-ahead forecasts of regional aerosol optical depth (AOD) are scored twice on the same days: against the MODIS retrieval, and against blowing dust reported by surface observers. The CAMS reanalysis has the lowest error of every forecast that does not use it as an input, yet the lowest critical success index. Only the meteorology-only random forest beats it on event skill after Holm correction across the three models.

## What is here

Analysis code and download scripts only. No data, results or manuscript. Every input is public and free, so the study can be rebuilt from source.

```
code/      analysis pipeline, stages S0 to S15
datasets/  download scripts and instructions, one folder per source
```

## Data sources

All open, all free, none redistributed here.

| Source | Product | Access | Dataset DOI |
|---|---|---|---|
| MODIS Terra | MOD08_D3 C6.1, AOD 550 nm (Dark Target and Deep Blue combined), area averaged | NASA Giovanni | 10.5067/MODIS/MOD08_D3.061 |
| MODIS Aqua | MYD08_D3 C6.1, same variable | NASA Giovanni | 10.5067/MODIS/MYD08_D3.061 |
| OMI | OMAERUVd v003, UV aerosol index, area averaged | NASA Giovanni | 10.5067/Aura/OMI/DATA3003 |
| ERA5 | pressure levels (550, 750, 850, 950, 1000 hPa) and single levels, 12:00 UTC | Copernicus Climate Data Store | 10.24381/cds.bd0915c6, 10.24381/cds.adbb2d47 |
| CAMS EAC4 | total and dust AOD 550 nm, 12:00 UTC | Copernicus Atmosphere Data Store | 10.24381/d58bbf47 |
| NOAA ISD | station observations with present weather and visibility | NCEI, no account needed | none |
| Natural Earth | 1:10m land, lakes, rivers, coastline, admin 0 and admin 1 (maps only) | naturalearthdata.com, or downloaded by cartopy | none |

Domain 30–40° N, 38–50° E. Period 1 January 2003 to 31 December 2024 (8036 days).

## Pipeline

Run in order from the repository root. Each stage writes to `results/`, `tables/` or `figs/` and reads only what earlier stages wrote.

| Stage | Purpose |
|---|---|
| `s0_inventory.py` | completeness and missingness per source |
| `s1_build_daily.py` | daily table: spatial reduction, Terra and Aqua merge, lags |
| `s2_dust_events.py` | dust events from WMO present-weather codes, denominator conditioned on filing stations |
| `s2b_kurdistan_index.py` | event index restricted to the 12 Kurdistan Region stations |
| `s3_baselines.py` | climatology, persistence and CAMS reference forecasts |
| `s4_models.py` | ridge regression, random forest, gradient boosting; block-bootstrap RMSE intervals |
| `s5_event_eval.py` | event verification with thresholds fixed on validation; block-bootstrap HSS intervals |
| `s6_ablation.py` | grouped permutation importance and seasonal error |
| `s7_robustness.py` | Terra or Aqua alone, OMI check, like-for-like comparison with CAMS |
| `s8_paired_tests.py` | paired moving-block bootstrap; Holm adjustment for the three models against CAMS total and dust AOD |
| `s9_figures.py --tables-only` | Tables 1 to 4 (run without the flag to also draw the legacy figure set) |
| `s10_duhok_panel.py` | descriptive panel for Duhok governorate |
| `s11_compute_cost.py` | processor time, memory and energy of each model |
| `s12_calibration_check.py` | threshold placement against discrimination |
| `s13_label_sensitivity.py` | the headline comparison under 15 event-label definitions |
| `s14_journal_figures.py` | journal figures, PNG only |
| `s15_reporting_numbers.py` | reporting-record numbers quoted in the text; fixed-station panels and network composition |

## Reproducing

Tested on Windows 11 with Python 3.11.9 and the exact versions in `requirements.txt`. Refitting the models under other versions of NumPy, scikit-learn or the operating system can change the gradient boosting and random forest predictions slightly, and the event scores with them.

```
git clone https://github.com/shahoismael/dust-forecast-error-vs-event-skill-kurdistan.git
cd dust-forecast-error-vs-event-skill-kurdistan
pip install -r requirements.txt
```

Paths resolve to the repository root. To keep data elsewhere, set `ICIEAHS_ROOT` (PowerShell: `$env:ICIEAHS_ROOT = "D:\path\to\project"`).

Create the free accounts listed in `datasets/`, run the download scripts, then the pipeline (PowerShell):

```
foreach ($s in "s0_inventory","s1_build_daily","s2_dust_events","s2b_kurdistan_index","s3_baselines","s4_models","s5_event_eval","s6_ablation","s7_robustness","s8_paired_tests","s10_duhok_panel","s11_compute_cost","s12_calibration_check","s13_label_sensitivity","s14_journal_figures","s15_reporting_numbers") { python "code\$s.py" }
python code\s9_figures.py --tables-only
```

Credentials live in your home directory, never in the repository; `.gitignore` excludes them with all data and generated output. Seeds are fixed at 20260809, and every table and figure is generated from the result files at run time.

## Method notes

**Splits are by time.** Train 2003–2017, validate 2018–2020, test 2021–2024. Scalers, the ridge penalty, climatology and every event threshold are fitted on training or validation data only.

**The event denominator counts only filing stations.** The share of reporting stations that file a present-weather group fell from 94.1% in 2005 to 34.2% in 2024. A dust day needs at least three filing stations and at least a quarter of them reporting blowing dust (codes 07, 08, 09, 30 to 35).

**CAMS is a reference forecast, never a predictor in the headline models.** The reanalysis assimilates MODIS AOD, the target itself. A diagnostic variant with CAMS as a predictor is reported separately.

**Uncertainty respects persistence.** All intervals and paired tests use a moving-block bootstrap with 7-day blocks and 5000 replicates.

**OMI is used only where it is valid.** Its yearly correlation with the target is 0.46 to 0.71 in 2004–2012 and falls to between −0.23 and 0.32 afterwards, consistent with the row anomaly, so it enters no model and serves only as a 2004–2012 check.

## Citation

If you use this code, please cite the article and this software: Hassen, S.I. dust-forecast-error-vs-event-skill-kurdistan, version 1.0.2; Zenodo, 2026. https://doi.org/10.5281/zenodo.24092541 (all versions: https://doi.org/10.5281/zenodo.24051854) (see `CITATION.cff`).

## Licence

MIT for the code. The underlying data carry their own terms; cite the providers.
