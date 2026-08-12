"""
S12 - is the event-skill ordering a property of the forecasts, or of where their
thresholds happen to land?

Why this exists. In S5 every forecast is thresholded at the value that maximised
the Heidke skill score on the validation years. Two things then differ between
forecasts at once: how well each separates dust days from quiet days, and how
many positives each ends up issuing. The second is not a property of the
forecast, it is a property of the calibration. Areas under the ROC curve in S5
cluster between 0.663 and 0.712, which says the separations are close; if the
thresholded scores are far apart while the ROC curves are not, the ordering
could be threshold placement rather than information.

A referee will ask this, and the honest answer requires computing it.

Three thresholding rules are compared on the same days and the same labels:

  s5          the validation-optimised threshold, reproduced from S5
  freq_match  the threshold that makes each forecast issue exactly as many
              positives on test as there are observed events, so every forecast
              has a frequency bias of one and CSI differences cannot come from
              one forecast simply calling more days
  val_quant   the threshold at the validation-period quantile corresponding to
              the validation base rate, which achieves the same equalisation
              without looking at the test period at all

freq_match uses the test base rate and is therefore diagnostic, not an
operational protocol. It is reported because it isolates the question. val_quant
is the operationally honest version of the same idea.

If the ordering in S5 survives both, the result is about the forecasts. If it
does not, the paper's claim must be narrowed to a statement about calibration.

Writes results/s12_calibration.csv and results/s12_report.txt.
"""

import os

import numpy as np
import pandas as pd

from s5_event_eval import TEST, TRAIN, VAL, contingency, skill, sweep

ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(ROOT, "results")

SEED = 20260809
N_BOOT = 5000
BLOCK = 7

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def block_bootstrap_delta(obs, pa, pb, stat, seed=SEED):
    """Paired moving-block bootstrap on a skill difference.

    Same machinery as S8: the pairing removes the day-to-day variation common to
    both forecasts, and the seven-day block preserves the autocorrelation of a
    dust episode, which an independent daily resample would destroy and thereby
    understate the interval.
    """
    rng = np.random.default_rng(seed)
    n = len(obs)
    nblocks = int(np.ceil(n / BLOCK))
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, max(1, n - BLOCK), nblocks)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        idx = idx[idx < n]
        o = obs[idx]
        deltas[i] = stat(o, pa[idx]) - stat(o, pb[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(np.mean(deltas)), float(lo), float(hi), float(min(1.0, p))


def csi_of(o, p):
    h, f, m, _ = contingency(o, p)
    return h / (h + f + m) if (h + f + m) else np.nan


def hss_of(o, p):
    return skill(o, p)["HSS"]


def main():
    daily = pd.read_csv(os.path.join(RESULTS, "s1_daily.csv"),
                        parse_dates=["date"]).set_index("date")
    dust = pd.read_csv(os.path.join(RESULTS, "s2_dust_daily.csv"),
                       parse_dates=["date"]).set_index("date")
    preds = pd.read_csv(os.path.join(RESULTS, "s4_predictions.csv"),
                        parse_dates=["date"]).set_index("date")

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

    forecasts = [
        ("observed_aod", "aod550"),
        ("climatology", "climatology"),
        ("persistence", "persistence"),
        ("cams", "cams_aod550_mean"),
        ("cams_dust", "cams_duaod550_mean"),
    ] + [(c, c) for c in preds.columns if c != "truth"]

    fc_cols = [c for _, c in forecasts if c in frame.columns]
    common = frame[fc_cols + ["obs_event"]].dropna()
    val = common[common.index.year.isin(VAL)]
    test = common[common.index.year.isin(TEST)]

    obs_t = test["obs_event"].to_numpy().astype(int)
    obs_v = val["obs_event"].to_numpy().astype(int)
    n_events = int(obs_t.sum())
    base_test = obs_t.mean()
    base_val = obs_v.mean()

    say("=" * 74)
    say("S12 CALIBRATION CHECK")
    say("=" * 74)
    say()
    say(f"common test days {len(test)}, observed events {n_events} "
        f"(base rate {base_test:.1%})")
    say(f"validation base rate {base_val:.1%}")
    say()
    say("Rules compared:")
    say("  s5          threshold maximising validation HSS (the paper's protocol)")
    say("  freq_match  threshold issuing exactly as many positives as there are")
    say("              observed events on test (frequency bias = 1, diagnostic)")
    say("  val_quant   threshold at the validation quantile for the validation")
    say("              base rate (equalises frequency without touching test)")
    say()

    rows = {}
    preds_by_rule = {"s5": {}, "freq_match": {}, "val_quant": {}}

    for label, col in forecasts:
        if col not in test.columns:
            continue
        sv, st = val[col].to_numpy(), test[col].to_numpy()

        # rule 1: reproduce S5
        lo, hi = np.nanpercentile(sv, [1, 99])
        grid = np.linspace(lo, hi, 200)
        v_sweep = sweep(obs_v, sv, grid)
        thr_s5 = float(v_sweep.loc[v_sweep["HSS"].idxmax(), "threshold"])

        # rule 2: equal number of positives on test
        thr_fm = float(np.sort(st)[::-1][n_events - 1])

        # rule 3: validation quantile at the validation base rate
        thr_vq = float(np.quantile(sv, 1.0 - base_val))

        for rule, thr in (("s5", thr_s5), ("freq_match", thr_fm),
                          ("val_quant", thr_vq)):
            p = (st >= thr).astype(int)
            preds_by_rule[rule][label] = p
            s = skill(obs_t, p)
            rows.setdefault(label, {})[rule] = {
                "threshold": thr, "n_pos": int(p.sum()),
                "POD": s["POD"], "FAR": s["FAR"], "CSI": s["CSI"],
                "HSS": s["HSS"], "bias": s["frequency_bias"],
            }

    out = []
    for label, byrule in rows.items():
        for rule, s in byrule.items():
            out.append({"forecast": label, "rule": rule,
                        **{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in s.items()}})
    table = pd.DataFrame(out)
    table.to_csv(os.path.join(RESULTS, "s12_calibration.csv"), index=False)

    for rule in ("s5", "freq_match", "val_quant"):
        sub = table[table.rule == rule].sort_values("CSI", ascending=False)
        say("-" * 74)
        say(f"rule: {rule}")
        say(f"  {'forecast':26} {'thr':>7} {'pos':>5} {'bias':>6} "
            f"{'POD':>6} {'FAR':>6} {'CSI':>6} {'HSS':>6}")
        for _, r in sub.iterrows():
            say(f"  {r['forecast']:26} {r['threshold']:7.3f} {r['n_pos']:5d} "
                f"{r['bias']:6.2f} {r['POD']:6.3f} {r['FAR']:6.3f} "
                f"{r['CSI']:6.3f} {r['HSS']:6.3f}")
        say()

    # The comparison the paper's claim rests on, under every rule.
    say("=" * 74)
    say("HEADLINE COMPARISON UNDER EACH RULE")
    say("random forest (meteorology only) minus reanalysis")
    say("=" * 74)
    a, b = "meteo_only__random_forest", "cams"
    for rule in ("s5", "freq_match", "val_quant"):
        pa = preds_by_rule[rule].get(a)
        pb = preds_by_rule[rule].get(b)
        if pa is None or pb is None:
            continue
        for name, stat in (("CSI", csi_of), ("HSS", hss_of)):
            d, lo, hi, p = block_bootstrap_delta(obs_t, pa, pb, stat)
            verdict = "significant" if (lo > 0 or hi < 0) else "not significant"
            say(f"  {rule:11} Δ{name} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                f"p = {p:.4f}  {verdict}")
    say()
    say("Reading this. If the difference holds under freq_match and val_quant,")
    say("the ordering reported in Section 4.3 is a property of the forecasts.")
    say("If it collapses when both forecasts are made to issue the same number")
    say("of positives, the ordering is a property of the thresholds and the")
    say("claim must be restated as one about calibration.")
    say()

    with open(os.path.join(RESULTS, "s12_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    say(f"wrote {os.path.join(RESULTS, 's12_calibration.csv')}")


if __name__ == "__main__":
    main()
