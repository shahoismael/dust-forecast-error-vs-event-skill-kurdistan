"""
S5 - event-based evaluation.

The question the continuous scores cannot answer: does a forecast that wins on
RMSE actually call the days when observers on the ground report blowing dust?

Ground truth: ISD blowing-dust day, codes 07, 08, 09 and 30 to 35, at least a
quarter of the stations reporting present weather that day.

Every forecast is converted to a dust-day call by thresholding predicted AOD.
The threshold is chosen on the validation years by maximising the Heidke skill
score, then applied unchanged to the test years. Choosing it on test would be
the same protocol error this paper is about.

Writes results/s5_event_scores.csv, results/s5_roc.csv, results/s5_report.txt.
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

SEED = 20260809
N_BOOT = 1000

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def contingency(obs, pred):
    hits = int(np.sum((pred == 1) & (obs == 1)))
    false_alarms = int(np.sum((pred == 1) & (obs == 0)))
    misses = int(np.sum((pred == 0) & (obs == 1)))
    correct_neg = int(np.sum((pred == 0) & (obs == 0)))
    return hits, false_alarms, misses, correct_neg


def skill(obs, pred):
    h, f, m, c = contingency(obs, pred)
    n = h + f + m + c
    pod = h / (h + m) if (h + m) else np.nan
    far = f / (h + f) if (h + f) else np.nan
    csi = h / (h + f + m) if (h + f + m) else np.nan
    pofd = f / (f + c) if (f + c) else np.nan
    bias = (h + f) / (h + m) if (h + m) else np.nan
    acc = (h + c) / n if n else np.nan

    expected = ((h + m) * (h + f) + (c + m) * (c + f)) / n if n else np.nan
    hss = (h + c - expected) / (n - expected) if n and (n - expected) else np.nan

    denom = (h + m) * (f + c)
    pss = (h * c - f * m) / denom if denom else np.nan

    return {
        "hits": h, "false_alarms": f, "misses": m, "correct_negatives": c,
        "n": n, "base_rate": (h + m) / n if n else np.nan,
        "POD": pod, "FAR": far, "CSI": csi, "POFD": pofd,
        "frequency_bias": bias, "accuracy": acc, "HSS": hss, "PSS": pss,
    }


def sweep(obs, score, grid):
    out = []
    for t in grid:
        pred = (score >= t).astype(int)
        s = skill(obs, pred)
        s["threshold"] = float(t)
        out.append(s)
    return pd.DataFrame(out)


def bootstrap_hss(obs, pred, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(obs)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals[i] = skill(obs[idx], pred[idx])["HSS"]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def auc_from_roc(pofd, pod):
    order = np.argsort(pofd)
    x, y = np.asarray(pofd)[order], np.asarray(pod)[order]
    x = np.concatenate([[0.0], x, [1.0]])
    y = np.concatenate([[0.0], y, [1.0]])
    return float(np.trapezoid(y, x))


def main():
    daily = pd.read_csv(
        os.path.join(RESULTS, "s1_daily.csv"), parse_dates=["date"]
    ).set_index("date")
    dust = pd.read_csv(
        os.path.join(RESULTS, "s2_dust_daily.csv"), parse_dates=["date"]
    ).set_index("date")
    preds = pd.read_csv(
        os.path.join(RESULTS, "s4_predictions.csv"), parse_dates=["date"]
    ).set_index("date")

    say("=" * 70)
    say("S5 EVENT EVALUATION")
    say("=" * 70)
    say()

    frame = daily[["aod550", "cams_aod550_mean", "cams_duaod550_mean"]].copy()
    frame["persistence"] = daily["aod550"].shift(1)

    clim_src = daily[daily.index.year.isin(TRAIN)]
    clim = clim_src.groupby(clim_src.index.dayofyear)["aod550"].mean()
    clim = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    frame["climatology"] = frame.index.dayofyear.map(clim)

    for col in preds.columns:
        if col != "truth":
            frame[col] = preds[col]

    frame["obs_event"] = dust["blowing_day"]
    frame["obs_index"] = dust["blowing_index"]

    forecasts = [
        ("observed_aod", "aod550"),
        ("climatology", "climatology"),
        ("persistence", "persistence"),
        ("cams", "cams_aod550_mean"),
        ("cams_dust", "cams_duaod550_mean"),
    ] + [(c, c) for c in preds.columns if c != "truth"]

    val = frame[frame.index.year.isin(VAL)]
    test = frame[frame.index.year.isin(TEST)]

    say(f"validation days with a ground label: {int(val['obs_event'].notna().sum())}")
    say(f"test days with a ground label:       {int(test['obs_event'].notna().sum())}")
    say(f"test base rate: {test['obs_event'].mean():.1%}")
    say()

    rows, roc_rows = [], []

    for label, col in forecasts:
        if col not in frame.columns:
            continue

        v = val[[col, "obs_event"]].dropna()
        t = test[[col, "obs_event"]].dropna()
        if len(v) < 50 or len(t) < 50:
            say(f"skipping {label}: too few labelled days "
                f"(val {len(v)}, test {len(t)})")
            continue

        lo, hi = np.nanpercentile(v[col], [1, 99])
        grid = np.linspace(lo, hi, 200)

        v_sweep = sweep(v["obs_event"].to_numpy().astype(int), v[col].to_numpy(), grid)
        best = v_sweep.loc[v_sweep["HSS"].idxmax()]
        threshold = best["threshold"]

        obs_t = t["obs_event"].to_numpy().astype(int)
        score_t = t[col].to_numpy()
        pred_t = (score_t >= threshold).astype(int)

        s = skill(obs_t, pred_t)
        hss_lo, hss_hi = bootstrap_hss(obs_t, pred_t)

        t_sweep = sweep(obs_t, score_t, np.linspace(*np.nanpercentile(score_t, [1, 99]), 200))
        auc = auc_from_roc(t_sweep["POFD"].to_numpy(), t_sweep["POD"].to_numpy())

        rows.append(
            {
                "forecast": label,
                "threshold": round(float(threshold), 4),
                "val_HSS": round(float(best["HSS"]), 4),
                **{k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in s.items()},
                "HSS_lo": round(hss_lo, 4),
                "HSS_hi": round(hss_hi, 4),
                "AUC": round(auc, 4),
            }
        )

        t_sweep["forecast"] = label
        roc_rows.append(t_sweep[["forecast", "threshold", "POD", "POFD", "FAR", "CSI", "HSS"]])

    scores = pd.DataFrame(rows).sort_values("HSS", ascending=False)
    scores.to_csv(os.path.join(RESULTS, "s5_event_scores.csv"), index=False)
    pd.concat(roc_rows).to_csv(os.path.join(RESULTS, "s5_roc.csv"), index=False,
                               float_format="%.5f")

    say("=" * 70)
    say("TEST PERIOD, dust-day detection, threshold fixed on validation")
    say("=" * 70)
    say(f"  {'forecast':26} {'thr':>6} {'POD':>6} {'FAR':>6} {'CSI':>6} "
        f"{'HSS':>6} {'95% CI':>16} {'AUC':>6}")
    for _, r in scores.iterrows():
        say(f"  {r['forecast']:26} {r['threshold']:6.3f} {r['POD']:6.3f} "
            f"{r['FAR']:6.3f} {r['CSI']:6.3f} {r['HSS']:6.3f} "
            f"[{r['HSS_lo']:.3f}, {r['HSS_hi']:.3f}] {r['AUC']:6.3f}")
    say()

    say("hits, false alarms, misses, correct negatives")
    for _, r in scores.iterrows():
        say(f"  {r['forecast']:26} H {r['hits']:4}  F {r['false_alarms']:4}  "
            f"M {r['misses']:4}  C {r['correct_negatives']:4}")
    say()

    ranked_rmse = pd.read_csv(os.path.join(RESULTS, "s4_scores.csv"))
    ranked_rmse = ranked_rmse[ranked_rmse.split == "test"]
    say("cross-check: RMSE rank against event-skill rank")
    say(f"  {'forecast':26} {'RMSE':>8} {'HSS':>8}")
    lookup = dict(zip(scores["forecast"], scores["HSS"]))
    for _, r in ranked_rmse.sort_values("rmse").iterrows():
        key = f"{r['feature_set']}__{r['model']}"
        if key in lookup:
            say(f"  {key:26} {r['rmse']:8.4f} {lookup[key]:8.3f}")
    base = pd.read_csv(os.path.join(RESULTS, "s3_baselines.csv"))
    for _, r in base[base.split == "test"].sort_values("rmse").iterrows():
        if r["baseline"] in lookup:
            say(f"  {r['baseline']:26} {r['rmse']:8.4f} {lookup[r['baseline']]:8.3f}")

    with open(os.path.join(RESULTS, "s5_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {os.path.join(RESULTS, 's5_event_scores.csv')}")
    say(f"wrote {os.path.join(RESULTS, 's5_roc.csv')}")


if __name__ == "__main__":
    main()
