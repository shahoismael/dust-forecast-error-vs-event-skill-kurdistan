"""
S3 - baselines.

Three references, all evaluated on the same test period as the models will be:

    climatology  day-of-year mean AOD, fitted on training years only
    persistence  yesterday's AOD
    CAMS         the operational physics-based product, area averaged

Nothing is reported later without these. Writes results/s3_baselines.csv
and results/s3_report.txt.
"""

import os

import numpy as np
import pandas as pd

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

N_BOOT = 1000
SEED = 20260809

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def metrics(truth, pred):
    mask = truth.notna() & pred.notna()
    y, f = truth[mask].to_numpy(), pred[mask].to_numpy()
    if len(y) < 10:
        return {"n": len(y), "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}
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


def bootstrap_rmse(truth, pred, n_boot=N_BOOT, seed=SEED):
    mask = truth.notna() & pred.notna()
    y, f = truth[mask].to_numpy(), pred[mask].to_numpy()
    if len(y) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = np.sqrt(np.mean((f[idx] - y[idx]) ** 2))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    df = pd.read_csv(
        os.path.join(RESULTS, "s1_daily.csv"), parse_dates=["date"]
    ).set_index("date")

    say("=" * 70)
    say("S3 BASELINES")
    say("=" * 70)
    say()

    train = df[df.index.year.isin(TRAIN)]
    say(f"train {min(TRAIN)}-{max(TRAIN)}: {len(train)} days")
    say(f"val   {min(VAL)}-{max(VAL)}: {len(df[df.index.year.isin(VAL)])} days")
    say(f"test  {min(TEST)}-{max(TEST)}: {len(df[df.index.year.isin(TEST)])} days")
    say()

    # climatology from training years only
    clim = train.groupby(train.index.dayofyear)["aod550"].mean()
    clim = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    df["b_climatology"] = df.index.dayofyear.map(clim)

    # persistence
    df["b_persistence"] = df["aod550"].shift(1)

    # CAMS
    if "cams_aod550_mean" in df.columns:
        df["b_cams"] = df["cams_aod550_mean"]
    if "cams_duaod550_mean" in df.columns:
        df["b_cams_dust"] = df["cams_duaod550_mean"]

    baselines = [c for c in df.columns if c.startswith("b_")]

    rows = []
    for split_name, years in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
        chunk = df[df.index.year.isin(years)]
        for name in baselines:
            m = metrics(chunk["aod550"], chunk[name])
            lo, hi = bootstrap_rmse(chunk["aod550"], chunk[name])
            rows.append(
                {
                    "split": split_name,
                    "baseline": name.replace("b_", ""),
                    **m,
                    "rmse_lo": lo,
                    "rmse_hi": hi,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "s3_baselines.csv"), index=False,
               float_format="%.5f")

    for split_name in ["train", "val", "test"]:
        say(f"--- {split_name} ---")
        say(f"  {'baseline':14} {'n':>5} {'RMSE':>8} {'95% CI':>18} "
            f"{'MAE':>8} {'bias':>8} {'R2':>7}")
        chunk = out[out["split"] == split_name].sort_values("rmse")
        for _, r in chunk.iterrows():
            ci = f"[{r['rmse_lo']:.4f}, {r['rmse_hi']:.4f}]" if pd.notna(r["rmse_lo"]) else ""
            say(f"  {r['baseline']:14} {int(r['n']):5} {r['rmse']:8.4f} {ci:>18} "
                f"{r['mae']:8.4f} {r['bias']:8.4f} {r['r2']:7.3f}")
        say()

    test_rows = out[out["split"] == "test"].set_index("baseline")
    if "persistence" in test_rows.index:
        ref = test_rows.loc["persistence", "rmse"]
        say("skill against persistence on test, positive is better")
        for name, r in test_rows.iterrows():
            if name == "persistence":
                continue
            say(f"  {name:14} {1 - r['rmse'] / ref:+7.1%}")
        say()

    say("target statistics by split")
    for split_name, years in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
        chunk = df[df.index.year.isin(years)]["aod550"]
        say(f"  {split_name:6} mean {chunk.mean():.4f}  sd {chunk.std():.4f}  "
            f"95th {chunk.quantile(0.95):.4f}  max {chunk.max():.4f}")

    with open(os.path.join(RESULTS, "s3_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {os.path.join(RESULTS, 's3_baselines.csv')}")


if __name__ == "__main__":
    main()
