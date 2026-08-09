"""
S6 - ablation and attribution.

Three questions:
  1. Which groups of predictors carry the skill, by permutation importance?
  2. Where does the error sit across the seasonal cycle?
  3. What remains once the circular reanalysis predictor is removed?

Writes results/s6_importance.csv, results/s6_seasonal.csv, results/s6_report.txt.
"""

import os
import re

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
N_REPEATS = 10

# omi_uvai is excluded: the OMI row anomaly corrupts OMAERUVd from about
# 2013 onward, so it must not enter a model whose test period is 2021-2024.
# The composition reanalysis is excluded: it assimilates the satellite retrieval
# used here as the target, so as a predictor it partly contains the answer.
# omi_uvai is excluded: the row anomaly corrupts the retrieval from about 2013.
DROP = {"aod550", "aod550_terra", "aod550_aqua", "year", "month", "omi_uvai",
        "cams_aod550_mean", "cams_aod550_sd", "cams_aod550_gradEW",
        "cams_aod550_gradNS", "cams_duaod550_mean", "cams_duaod550_sd",
        "cams_duaod550_gradEW", "cams_duaod550_gradNS"}

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def group_of(name):
    if name.startswith("cams_"):
        return "CAMS"
    if name.startswith("aod550_lag"):
        return "lagged AOD"
    if name == "omi_uvai":
        return "OMI"
    if name in ("sin_doy", "cos_doy"):
        return "calendar"
    if re.match(r"^[uvw]\d+_", name):
        return "wind and vertical motion"
    if re.match(r"^[rt]\d+_", name):
        return "humidity and temperature"
    if name.startswith(("u10_", "v10_")):
        return "surface wind"
    if name.startswith(("t2m_", "sp_")):
        return "surface thermodynamics"
    if name.startswith("tp_"):
        return "precipitation"
    if name.startswith("swvl1_"):
        return "soil moisture"
    if name.startswith(("lai_hv_", "lai_lv_")):
        return "vegetation"
    return "other"


def rmse(y, f):
    return float(np.sqrt(np.mean((f - y) ** 2)))


def main():
    df = pd.read_csv(
        os.path.join(RESULTS, "s1_daily.csv"), parse_dates=["date"]
    ).set_index("date")

    say("=" * 70)
    say("S6 ABLATION")
    say("=" * 70)
    say()

    features = [c for c in df.columns if c not in DROP]
    lag_cols = [f"aod550_lag{i}" for i in range(1, 6)]

    sub = df[features + ["aod550"]].dropna(subset=["aod550"] + lag_cols)
    sub[features] = sub[features].ffill(limit=1)
    sub = sub.dropna()

    tr = sub[sub.index.year.isin(TRAIN)]
    te = sub[sub.index.year.isin(TEST)]
    say(f"train {len(tr)}   test {len(te)}   predictors {len(features)}")
    say()

    x_tr, y_tr = tr[features].to_numpy(), tr["aod550"].to_numpy()
    x_te, y_te = te[features].to_numpy(), te["aod550"].to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.15, random_state=SEED,
    )
    model.fit(x_tr, y_tr)
    base_pred = model.predict(x_te)
    base_rmse = rmse(y_te, base_pred)
    say(f"baseline test RMSE: {base_rmse:.4f}")
    say()

    # ---- grouped permutation importance -----------------------------------
    groups = {}
    for i, name in enumerate(features):
        groups.setdefault(group_of(name), []).append(i)

    rng = np.random.default_rng(SEED)
    rows = []
    for group_name, idx in sorted(groups.items()):
        deltas = []
        for _ in range(N_REPEATS):
            shuffled = x_te.copy()
            order = rng.permutation(len(shuffled))
            shuffled[:, idx] = shuffled[order][:, idx]
            deltas.append(rmse(y_te, model.predict(shuffled)) - base_rmse)
        rows.append(
            {
                "group": group_name,
                "n_features": len(idx),
                "rmse_increase": float(np.mean(deltas)),
                "rmse_increase_sd": float(np.std(deltas)),
                "relative": float(np.mean(deltas) / base_rmse),
            }
        )

    importance = pd.DataFrame(rows).sort_values("rmse_increase", ascending=False)
    importance.to_csv(os.path.join(RESULTS, "s6_importance.csv"), index=False,
                      float_format="%.5f")

    say("permutation importance by feature group, test period")
    say(f"  {'group':28} {'n':>4} {'ΔRMSE':>9} {'relative':>9}")
    for _, r in importance.iterrows():
        say(f"  {r['group']:28} {int(r['n_features']):4} "
            f"{r['rmse_increase']:9.4f} {r['relative']:9.1%}")
    say()

    # ---- seasonal error ----------------------------------------------------
    season = pd.DataFrame(
        {"truth": y_te, "pred": base_pred}, index=te.index
    )
    season["month"] = season.index.month
    season["err"] = season["pred"] - season["truth"]

    monthly = season.groupby("month").apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "mean_aod": g["truth"].mean(),
                "rmse": rmse(g["truth"].to_numpy(), g["pred"].to_numpy()),
                "bias": g["err"].mean(),
                "mae": g["err"].abs().mean(),
            }
        ),
        include_groups=False,
    )
    monthly.to_csv(os.path.join(RESULTS, "s6_seasonal.csv"), float_format="%.5f")

    say("error by month, test period")
    say(f"  {'month':>5} {'n':>5} {'mean AOD':>9} {'RMSE':>8} {'bias':>8}")
    for month, r in monthly.iterrows():
        say(f"  {int(month):5} {int(r['n']):5} {r['mean_aod']:9.4f} "
            f"{r['rmse']:8.4f} {r['bias']:8.4f}")
    say()

    dusty = monthly.loc[[4, 5, 6, 7, 8]]
    quiet = monthly.loc[[11, 12, 1, 2]]
    say(f"dust season (Apr-Aug) RMSE {dusty['rmse'].mean():.4f}, "
        f"mean AOD {dusty['mean_aod'].mean():.4f}")
    say(f"quiet season (Nov-Feb) RMSE {quiet['rmse'].mean():.4f}, "
        f"mean AOD {quiet['mean_aod'].mean():.4f}")
    say(f"ratio of RMSE to mean AOD: dust season "
        f"{dusty['rmse'].mean()/dusty['mean_aod'].mean():.3f}, "
        f"quiet season {quiet['rmse'].mean()/quiet['mean_aod'].mean():.3f}")
    say()

    # ---- error against event magnitude -------------------------------------
    dust = pd.read_csv(
        os.path.join(RESULTS, "s2_dust_daily.csv"), parse_dates=["date"]
    ).set_index("date")
    season["event"] = dust["blowing_day"]
    labelled = season.dropna(subset=["event"])
    if len(labelled) > 100:
        for flag, label in [(1, "ground dust days"), (0, "non-dust days")]:
            chunk = labelled[labelled["event"] == flag]
            say(f"{label:20} n {len(chunk):5}  "
                f"RMSE {rmse(chunk['truth'].to_numpy(), chunk['pred'].to_numpy()):.4f}  "
                f"bias {chunk['err'].mean():+.4f}  "
                f"mean observed AOD {chunk['truth'].mean():.4f}")

    with open(os.path.join(RESULTS, "s6_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {os.path.join(RESULTS, 's6_importance.csv')}")
    say(f"wrote {os.path.join(RESULTS, 's6_seasonal.csv')}")


if __name__ == "__main__":
    main()
