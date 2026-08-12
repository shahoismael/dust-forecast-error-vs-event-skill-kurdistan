# dust-forecast-verification-Kurdistan

Code for *Error–detection decoupling in data-driven sand and dust storm forecasting
over the Kurdistan Region*.

Day-ahead forecasts of regional aerosol optical depth are scored two ways: against
the satellite measurement, and against blowing-dust days recorded by weather
observers at stations in the Kurdistan Region. The two rankings disagree.

## What is here

Analysis code only. No data and no manuscript. Every input is public and free, and
the download scripts are included so the study can be rebuilt from source.

```
code/    analysis pipeline, S0 to S10
datasets/  download scripts, one folder per source
```

## Data sources

All open, all free, none redistributed here.

| Source | Product | Access |
|---|---|---|
| MODIS Terra and Aqua | Level-3 daily AOD 550 nm, area averaged | NASA Giovanni |
| ERA5 | pressure and single levels, 12:00 UTC | Copernicus Climate Data Store |
| CAMS EAC4 | total and dust AOD 550 nm | Copernicus Atmosphere Data Store |
| OMI | UV aerosol index, area averaged | NASA Giovanni |
| NOAA ISD | hourly station observations, Iraq | NCEI, no account needed |

Domain 30–40° N, 38–50° E. Period 2003 to 2024.

## Pipeline

Run in order. Each stage writes to `results/` and reads only what earlier stages wrote.

| Stage | Purpose |
|---|---|
| `s0_inventory.py` | completeness and missingness per source |
| `s1_build_daily.py` | one daily table; spatial reduction, sensor merge, lags |
| `s2_dust_events.py` | dust events from WMO present-weather codes |
| `s2b_kurdistan_index.py` | restrict the event index to Kurdistan Region stations |
| `s3_baselines.py` | climatology, persistence, physics-based product |
| `s4_models.py` | ridge, random forest, gradient boosting |
| `s5_event_eval.py` | categorical verification, thresholds fixed on validation |
| `s6_ablation.py` | grouped permutation importance, seasonal error |
| `s7_robustness.py` | independent sensor, target-construction sensitivity |
| `s8_paired_tests.py` | paired moving-block bootstrap on skill differences |
| `s9_figures.py` | figures and tables |
| `s10_duhok_panel.py` | descriptive panel, Duhok governorate |
| `s11_compute_cost.py` | wall-clock, CPU, memory and energy cost of each model |
| `s12_calibration_check.py` | is the event ordering discrimination or threshold placement |
| `s13_label_sensitivity.py` | does the result survive redefining the event label |

## Reproducing

```bash
git clone https://github.com/shahoismael/dust-forecast-verification-Kurdistan.git
cd dust-forecast-verification-Kurdistan
pip install -r requirements.txt
```

Paths resolve to the repository root automatically. Override with `ICIEAHS_ROOT` if
your data lives elsewhere:

```bash
export ICIEAHS_ROOT=/path/to/project      # Windows: set ICIEAHS_ROOT=D:\path\to\project
```

Create the free accounts listed in `datasets/`, run the download scripts, then the
pipeline in order:

```bash
for s in s0_inventory s1_build_daily s2_dust_events s2b_kurdistan_index \
         s3_baselines s4_models s5_event_eval s6_ablation s7_robustness \
         s8_paired_tests s9_figures s10_duhok_panel; do
    python code/$s.py
done
```

Credentials live in your home directory, never in the repository. `.gitignore`
excludes them along with all data and generated output.

Seeds are fixed at 20260809. Figures are generated from the result files at run
time, so text and figures cannot drift apart.

## Method notes

Three points matter more than the model choice.

**Splits are temporal, never random.** Train 2003–2017, validate 2018–2020, test
2021–2024. Consecutive blocks, so autocorrelation cannot leak across them. All
scaling and thresholds are fitted on training or validation data only.

**The event denominator is conditioned on reporting.** Manual present-weather
coverage in the station record falls from 99.8 per cent of observations in 2003 to
under 20 per cent after 2019. Using all observations as the denominator manufactures
a false decline in dust frequency. `s2_dust_events.py` divides by the stations that
actually reported present weather that day.

**One sensor is used only where it is valid.** The OMI aerosol index correlates with
the satellite target at r ≈ 0.7 through 2012, then degrades to near zero as the
instrument row anomaly progresses. It is excluded from the model features entirely
and used only over 2004–2012.

## Licence

MIT for the code. The underlying data carry their own terms; cite the providers.
