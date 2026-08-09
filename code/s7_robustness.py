"""
S7 - robustness.

  1. Independent sensor: does the OMI UV aerosol index agree with the MODIS
     target, and does it detect ground dust days better or worse?
  2. Sensitivity to the Terra and Aqua merge: rerun the headline model on
     Terra only and on Aqua only.

Writes results/s7_robustness.csv and results/s7_report.txt.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Project root. Override with the ICIEAHS_ROOT environment variable, otherwise
# the parent of this file's directory is used.
ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(ROOT, "results")

TRAIN = range(2003, 2018)
VAL = range(2018, 2021)
TEST = range(2021, 2025)

SEED = 20260809
DROP = {"aod550", "aod550_terra", "aod550_aqua", "year", "month", "omi_uvai"}

# OMI row anomaly degrades OMAERUVd progressively from about 2013.
OMI_VAL = range(2004, 2010)
OMI_TEST = range(2010, 2013)

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def rmse(y, f):
    return float(np.sqrt(np.mean((f - y) ** 2)))


def skill(obs, pred):
    h = int(np.sum((pred == 1) & (obs == 1)))
    f = int(np.sum((pred == 1) & (obs == 0)))
    m = int(np.sum((pred == 0) & (obs == 1)))
    c = int(np.sum((pred == 0) & (obs == 0)))
    n = h + f + m + c
    expected = ((h + m) * (h + f) + (c + m) * (c + f)) / n if n else np.nan
    hss = (h + c - expected) / (n - expected) if n and (n - expected) else np.nan
    pod = h / (h + m) if (h + m) else np.nan
    far = f / (h + f) if (h + f) else np.nan
    csi = h / (h + f + m) if (h + f + m) else np.nan
    return {"POD": pod, "FAR": far, "CSI": csi, "HSS": hss,
            "hits": h, "false_alarms": f, "misses": m}


def best_threshold(obs, score):
    grid = np.linspace(*np.nanpercentile(score, [1, 99]), 200)
    best, best_hss = grid[0], -np.inf
    for t in grid:
        s = skill(obs, (score >= t).astype(int))
        if s["HSS"] is not None and s["HSS"] > best_hss:
            best_hss, best = s["HSS"], t
    return float(best), float(best_hss)


def main():
    df = pd.read_csv(
        os.path.join(RESULTS, "s1_daily.csv"), parse_dates=["date"]
    ).set_index("date")
    dust = pd.read_csv(
        os.path.join(RESULTS, "s2_dust_daily.csv"), parse_dates=["date"]
    ).set_index("date")

    say("=" * 70)
    say("S7 ROBUSTNESS")
    say("=" * 70)
    say()

    rows = []

    # --- 1. OMI as an independent sensor -----------------------------------
    say("--- independent sensor: OMI UV aerosol index ---")
    both = df[["aod550", "omi_uvai"]].dropna()
    r = both["aod550"].corr(both["omi_uvai"])
    say(f"correlation with MODIS AOD: r = {r:.3f} on {len(both)} shared days")

    say("correlation with MODIS AOD by year:")
    yearly = both.join(df[[]], how="left")
    per_year = df[["aod550", "omi_uvai"]].dropna()
    corr_by_year = per_year.groupby(per_year.index.year).apply(
        lambda g: g["aod550"].corr(g["omi_uvai"])
    )
    for year, value in corr_by_year.items():
        say(f"    {year}  r = {value:+.3f}")
    say()
    say("OMI is used only over 2004-2012, before the row anomaly degrades the")
    say("retrieval. Detection thresholds are set on 2004-2009 and applied to")
    say("2010-2012.")
    say()

    joined = df[["aod550", "omi_uvai"]].join(dust[["blowing_day"]]).dropna()
    val = joined[joined.index.year.isin(OMI_VAL)]
    test = joined[joined.index.year.isin(OMI_TEST)]
    say(f"labelled days: val {len(val)}, test {len(test)}")

    for label, col in [("MODIS AOD", "aod550"), ("OMI UVAI", "omi_uvai")]:
        if len(val) < 50 or len(test) < 50:
            continue
        thr, val_hss = best_threshold(
            val["blowing_day"].to_numpy().astype(int), val[col].to_numpy()
        )
        s = skill(
            test["blowing_day"].to_numpy().astype(int),
            (test[col].to_numpy() >= thr).astype(int),
        )
        say(f"  {label:12} threshold {thr:7.3f}  POD {s['POD']:.3f}  "
            f"FAR {s['FAR']:.3f}  CSI {s['CSI']:.3f}  HSS {s['HSS']:.3f}")
        rows.append({"check": "independent_sensor", "variant": label,
                     "threshold": round(thr, 4), **s})
    say()

    # --- 2. Terra and Aqua merge sensitivity --------------------------------
    say("--- sensitivity to the Terra and Aqua merge ---")
    features = [c for c in df.columns if c not in DROP]
    lag_cols = [f"aod550_lag{i}" for i in range(1, 6)]

    for label, target_col in [
        ("merged", "aod550"),
        ("Terra only", "aod550_terra"),
        ("Aqua only", "aod550_aqua"),
    ]:
        work = df.copy()
        work["y"] = work[target_col]
        for i in range(1, 6):
            work[f"aod550_lag{i}"] = work["y"].shift(i)

        sub = work[features + ["y"]].dropna(subset=["y"] + lag_cols)
        sub[features] = sub[features].ffill(limit=1)
        sub = sub.dropna()

        tr = sub[sub.index.year.isin(TRAIN)]
        te = sub[sub.index.year.isin(TEST)]
        if len(tr) < 500 or len(te) < 100:
            say(f"  {label}: too few rows, skipped")
            continue

        model = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.15, random_state=SEED,
        )
        model.fit(tr[features].to_numpy(), tr["y"].to_numpy())
        pred = model.predict(te[features].to_numpy())
        score = rmse(te["y"].to_numpy(), pred)
        r2 = 1 - np.sum((pred - te["y"]) ** 2) / np.sum(
            (te["y"] - te["y"].mean()) ** 2
        )
        say(f"  {label:12} n_test {len(te):5}  RMSE {score:.4f}  R2 {r2:.3f}")
        rows.append({"check": "merge_sensitivity", "variant": label,
                     "rmse": round(score, 5), "r2": round(float(r2), 4),
                     "n_test": len(te)})
    say()

    # --- 3. does the model beat CAMS on the same rows -----------------------
    say("--- like-for-like against CAMS on shared rows ---")
    preds = pd.read_csv(
        os.path.join(RESULTS, "s4_predictions.csv"), parse_dates=["date"]
    ).set_index("date")
    shared = preds[["truth", "full__gradient_boosting"]].join(
        df[["cams_aod550_mean"]]
    ).dropna()
    shared = shared[shared.index.year.isin(TEST)]
    if len(shared) > 100:
        m_rmse = rmse(shared["truth"].to_numpy(),
                      shared["full__gradient_boosting"].to_numpy())
        c_rmse = rmse(shared["truth"].to_numpy(),
                      shared["cams_aod550_mean"].to_numpy())
        say(f"  rows {len(shared)}")
        say(f"  gradient boosting RMSE {m_rmse:.4f}")
        say(f"  CAMS              RMSE {c_rmse:.4f}")
        say(f"  difference {c_rmse - m_rmse:+.4f}  "
            f"({1 - m_rmse/c_rmse:+.1%} relative)")

        rng = np.random.default_rng(SEED)
        diffs = np.empty(1000)
        y = shared["truth"].to_numpy()
        a = shared["full__gradient_boosting"].to_numpy()
        b = shared["cams_aod550_mean"].to_numpy()
        for i in range(1000):
            idx = rng.integers(0, len(y), len(y))
            diffs[i] = rmse(y[idx], b[idx]) - rmse(y[idx], a[idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        say(f"  bootstrap 95% CI on the RMSE difference: [{lo:+.4f}, {hi:+.4f}]")
        say(f"  significant at 5%: {'yes' if lo > 0 else 'no'}")
        rows.append({"check": "vs_cams", "variant": "gradient_boosting",
                     "rmse": round(m_rmse, 5), "cams_rmse": round(c_rmse, 5),
                     "diff_lo": round(float(lo), 5), "diff_hi": round(float(hi), 5)})

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "s7_robustness.csv"),
                              index=False)

    with open(os.path.join(RESULTS, "s7_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {os.path.join(RESULTS, 's7_robustness.csv')}")


if __name__ == "__main__":
    main()
