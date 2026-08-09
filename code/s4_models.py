"""
S4 - models.

Task: predict area-mean MODIS AOD 550 nm 24 hours ahead.

Predictors: meteorological fields for the target day and the satellite target
lagged one to five days. Perfect-prognosis design: the target-day meteorology comes
from reanalysis, standing in for what a numerical forecast would supply in
operations. Reported skill is therefore an upper bound on operational skill.

Two feature sets are run.

    meteo_only  the headline. Meteorology, lagged target, calendar terms.
    plus_reanalysis  adds the atmospheric composition reanalysis, for diagnosis only.

The reanalysis assimilates the same satellite aerosol retrieval used here as the
target (Inness et al., 2019). Using it as a predictor is therefore partly circular:
the feature contains the answer. It is reported to quantify how much apparent skill
an assimilated predictor buys, and it is not the result.

Models are deliberately plain. They are the instrument, not the contribution.

Writes results/s4_scores.csv, results/s4_predictions.csv, results/s4_report.txt.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
N_BOOT = 1000

# omi_uvai is excluded: the OMI row anomaly corrupts OMAERUVd from about
# 2013 onward, so it must not enter a model whose test period is 2021-2024.
DROP = {"aod550", "aod550_terra", "aod550_aqua", "year", "month", "omi_uvai"}

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def metrics(y, f):
    err = f - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def bootstrap_rmse(y, f, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        out[i] = np.sqrt(np.mean((f[idx] - y[idx]) ** 2))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def build_models():
    return {
        "ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.05,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        ),
    }


def main():
    df = pd.read_csv(
        os.path.join(RESULTS, "s1_daily.csv"), parse_dates=["date"]
    ).set_index("date")

    say("=" * 70)
    say("S4 MODELS")
    say("=" * 70)
    say()

    features_all = [c for c in df.columns if c not in DROP]
    cams_cols = [c for c in features_all if c.startswith("cams_")]
    say(f"candidate predictors: {len(features_all)}  of which CAMS: {len(cams_cols)}")

    # Rows usable for supervised learning: target present, all five lags present.
    lag_cols = [f"aod550_lag{i}" for i in range(1, 6)]
    usable = df.dropna(subset=["aod550"] + lag_cols)
    say(f"rows with target and lags: {len(usable)}")
    say()

    feature_sets = {
        "meteo_only": [c for c in features_all if c not in cams_cols],
        "plus_reanalysis": features_all,
    }

    rows = []
    predictions = pd.DataFrame(index=usable.index)
    predictions["truth"] = usable["aod550"]

    for set_name, cols in feature_sets.items():
        data = usable.dropna(subset=cols) if set_name == "plus_reanalysis" else usable
        sub = data[cols + ["aod550"]].copy()

        # ERA5 has no gaps; OMI does. Fill forward at most one day, then drop.
        sub[cols] = sub[cols].ffill(limit=1)
        sub = sub.dropna()

        tr = sub[sub.index.year.isin(TRAIN)]
        va = sub[sub.index.year.isin(VAL)]
        te = sub[sub.index.year.isin(TEST)]

        say(f"--- feature set: {set_name}  ({len(cols)} predictors) ---")
        say(f"  train {len(tr)}  val {len(va)}  test {len(te)}")

        x_tr, y_tr = tr[cols].to_numpy(), tr["aod550"].to_numpy()
        x_va, y_va = va[cols].to_numpy(), va["aod550"].to_numpy()
        x_te, y_te = te[cols].to_numpy(), te["aod550"].to_numpy()

        for model_name, model in build_models().items():
            t0 = time.time()
            model.fit(x_tr, y_tr)
            elapsed = time.time() - t0

            for split_name, x, y, idx in [
                ("train", x_tr, y_tr, tr.index),
                ("val", x_va, y_va, va.index),
                ("test", x_te, y_te, te.index),
            ]:
                pred = model.predict(x)
                m = metrics(y, pred)
                lo, hi = (bootstrap_rmse(y, pred) if split_name == "test" else (np.nan, np.nan))
                rows.append(
                    {
                        "feature_set": set_name,
                        "model": model_name,
                        "split": split_name,
                        **m,
                        "rmse_lo": lo,
                        "rmse_hi": hi,
                        "fit_seconds": round(elapsed, 1),
                    }
                )
                if split_name in ("val", "test"):
                    predictions.loc[idx, f"{set_name}__{model_name}"] = pred

            test_row = rows[-1]
            say(f"  {model_name:19} test RMSE {test_row['rmse']:.4f} "
                f"[{test_row['rmse_lo']:.4f}, {test_row['rmse_hi']:.4f}]  "
                f"R2 {test_row['r2']:.3f}  fit {elapsed:.1f}s")
        say()

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(RESULTS, "s4_scores.csv"), index=False,
                  float_format="%.5f")

    # attach baselines for the same test rows
    base = pd.read_csv(os.path.join(RESULTS, "s3_baselines.csv"))
    ref = base[(base.split == "test") & (base.baseline == "persistence")]["rmse"].iat[0]
    cams_ref = base[(base.split == "test") & (base.baseline == "cams")]["rmse"].iat[0]

    say("=" * 70)
    say("TEST PERIOD, skill against persistence and against CAMS")
    say("=" * 70)
    say(f"  persistence RMSE {ref:.4f}      CAMS RMSE {cams_ref:.4f}")
    say()
    say(f"  {'feature set':10} {'model':19} {'RMSE':>8} {'vs persist':>11} {'vs CAMS':>9}")
    test = scores[scores.split == "test"].sort_values("rmse")
    for _, r in test.iterrows():
        say(f"  {r['feature_set']:10} {r['model']:19} {r['rmse']:8.4f} "
            f"{1 - r['rmse']/ref:+11.1%} {1 - r['rmse']/cams_ref:+9.1%}")
    say()

    predictions.to_csv(os.path.join(RESULTS, "s4_predictions.csv"),
                       index_label="date", float_format="%.5f")

    with open(os.path.join(RESULTS, "s4_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say(f"wrote {os.path.join(RESULTS, 's4_scores.csv')}")
    say(f"wrote {os.path.join(RESULTS, 's4_predictions.csv')}")


if __name__ == "__main__":
    main()
