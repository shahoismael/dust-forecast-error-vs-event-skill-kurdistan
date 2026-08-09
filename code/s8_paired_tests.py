"""
S8 - paired significance tests on event-detection skill.

Comparing overlapping confidence intervals is a weak and biased test. Because
every forecast is scored on the same days, the paired difference in skill can be
bootstrapped directly, which removes the shared day-to-day variability and is
far more powerful.

Days are resampled in blocks to respect autocorrelation: dust episodes persist
for several days, so an independent daily bootstrap would understate the
uncertainty.

Writes results/s8_paired.csv and results/s8_report.txt.
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
N_BOOT = 5000
BLOCK = 7  # days, roughly the synoptic timescale of a dust episode

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def hss(obs, pred):
    h = np.sum((pred == 1) & (obs == 1))
    f = np.sum((pred == 1) & (obs == 0))
    m = np.sum((pred == 0) & (obs == 1))
    c = np.sum((pred == 0) & (obs == 0))
    n = h + f + m + c
    if n == 0:
        return np.nan
    expected = ((h + m) * (h + f) + (c + m) * (c + f)) / n
    return (h + c - expected) / (n - expected) if (n - expected) else np.nan


def csi(obs, pred):
    h = np.sum((pred == 1) & (obs == 1))
    f = np.sum((pred == 1) & (obs == 0))
    m = np.sum((pred == 0) & (obs == 1))
    return h / (h + f + m) if (h + f + m) else np.nan


def block_indices(n, rng, block=BLOCK):
    starts = rng.integers(0, max(n - block, 1), int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx[:n]


def paired_test(obs, pred_a, pred_b, statistic, rng):
    """Bootstrap the paired difference statistic(a) - statistic(b)."""
    n = len(obs)
    observed = statistic(obs, pred_a) - statistic(obs, pred_b)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = block_indices(n, rng)
        diffs[i] = statistic(obs[idx], pred_a[idx]) - statistic(obs[idx], pred_b[idx])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    # two-sided p from the bootstrap distribution
    p = 2 * min(
        float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))
    )
    return observed, float(lo), float(hi), min(p, 1.0)


def best_threshold(obs, score):
    grid = np.linspace(*np.nanpercentile(score, [1, 99]), 200)
    best_t, best_v = grid[0], -np.inf
    for t in grid:
        v = hss(obs, (score >= t).astype(int))
        if v is not None and not np.isnan(v) and v > best_v:
            best_v, best_t = v, t
    return float(best_t)


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
    say("S8 PAIRED SIGNIFICANCE TESTS")
    say("=" * 70)
    say(f"bootstrap replicates {N_BOOT}, moving block length {BLOCK} days")
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
    frame["event"] = dust["blowing_day"]

    candidates = [
        ("gradient_boosting", "full__gradient_boosting"),
        ("random_forest", "full__random_forest"),
        ("gb_no_cams", "no_cams__gradient_boosting"),
        ("climatology", "climatology"),
        ("cams", "cams_aod550_mean"),
        ("cams_dust", "cams_duaod550_mean"),
        ("persistence", "persistence"),
        ("observed_aod", "aod550"),
    ]

    # thresholds set on validation, exactly as in S5
    val = frame[frame.index.year.isin(VAL)]
    thresholds = {}
    for label, col in candidates:
        v = val[[col, "event"]].dropna()
        if len(v) >= 50:
            thresholds[label] = best_threshold(
                v["event"].to_numpy().astype(int), v[col].to_numpy()
            )

    keep = [c for label, c in candidates if label in thresholds]
    test = frame[frame.index.year.isin(TEST)][keep + ["event"]].dropna()
    obs = test["event"].to_numpy().astype(int)
    say(f"test days scored on all forecasts simultaneously: {len(test)}")
    say(f"base rate: {obs.mean():.1%}")
    say()

    binary = {}
    for label, col in candidates:
        if label in thresholds:
            binary[label] = (test[col].to_numpy() >= thresholds[label]).astype(int)

    say("point estimates on the common test set")
    say(f"  {'forecast':20} {'HSS':>7} {'CSI':>7}")
    point = {}
    for label, pred in binary.items():
        point[label] = (hss(obs, pred), csi(obs, pred))
        say(f"  {label:20} {point[label][0]:7.3f} {point[label][1]:7.3f}")
    say()

    rng = np.random.default_rng(SEED)
    rows = []

    reference = "cams"
    say(f"paired differences against {reference}, positive means better than CAMS")
    say(f"  {'forecast':20} {'dHSS':>7} {'95% CI':>18} {'p':>8}")
    for label in binary:
        if label == reference:
            continue
        d, lo, hi, p = paired_test(
            obs, binary[label], binary[reference], hss, rng
        )
        flag = "*" if lo > 0 or hi < 0 else " "
        say(f"  {label:20} {d:+7.3f} [{lo:+.3f}, {hi:+.3f}] {p:8.4f} {flag}")
        rows.append({"comparison": f"{label}_vs_{reference}", "metric": "HSS",
                     "difference": round(d, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "p": round(p, 4)})
    say()

    say("the key ordering claim: does a lower-RMSE forecast detect fewer events")
    pairs = [
        ("gradient_boosting", "cams"),
        ("climatology", "cams"),
        ("cams_dust", "cams"),
        ("gradient_boosting", "climatology"),
    ]
    for a, b in pairs:
        if a not in binary or b not in binary:
            continue
        for name, stat in [("HSS", hss), ("CSI", csi)]:
            d, lo, hi, p = paired_test(obs, binary[a], binary[b], stat, rng)
            flag = "significant" if (lo > 0 or hi < 0) else "not significant"
            say(f"  {a:20} vs {b:14} {name}  {d:+.3f} "
                f"[{lo:+.3f}, {hi:+.3f}]  p = {p:.4f}  {flag}")
            rows.append({"comparison": f"{a}_vs_{b}", "metric": name,
                         "difference": round(d, 4), "ci_lo": round(lo, 4),
                         "ci_hi": round(hi, 4), "p": round(p, 4)})
    say()

    say("absolute skill: is any forecast better than chance")
    for label, pred in binary.items():
        d, lo, hi, p = paired_test(obs, pred, np.zeros_like(pred), hss, rng)
        say(f"  {label:20} HSS {point[label][0]:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}]  p = {p:.4f}")
        rows.append({"comparison": f"{label}_vs_none", "metric": "HSS",
                     "difference": round(d, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "p": round(p, 4)})

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "s8_paired.csv"), index=False)

    with open(os.path.join(RESULTS, "s8_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {os.path.join(RESULTS, 's8_paired.csv')}")


if __name__ == "__main__":
    main()
