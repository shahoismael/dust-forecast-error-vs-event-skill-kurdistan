"""
S13 - does the headline result depend on how the event label was defined?

Two numbers define the ground truth in Section 3.3 and neither was chosen by
anything more principled than judgement:

    the fraction of filing stations that must report blowing dust, set at 0.25
    the minimum number of stations that must have filed at all, set at 3

Both are defensible. Neither is derived. A referee who suspects the reported
ordering is an artefact of those two choices currently has nothing to check, and
that is the cheapest open hole in the paper.

This script rebuilds the label across a grid of both parameters and, for each
label, reruns the comparison the paper's claim rests on: the meteorology-only
random forest against the operational composition reanalysis, thresholded on the
validation years exactly as in S5 and scored on the test years.

Reported for every cell: the number of labelled days, the test base rate, the
two CSI values and their difference. If the sign of the difference survives the
grid, the result is a property of the forecasts. If it flips anywhere, the paper
must say where.

Writes results/s13_label_sensitivity.csv and results/s13_report.txt.
"""

import os

import numpy as np
import pandas as pd

from s5_event_eval import TEST, TRAIN, VAL, skill, sweep

ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(ROOT, "results")

FRACTIONS = [0.15, 0.20, 0.25, 0.30, 0.35]
MIN_STATIONS = [2, 3, 5]

MODEL = "meteo_only__random_forest"
BASELINE = "cams"

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def build_frame():
    daily = pd.read_csv(os.path.join(RESULTS, "s1_daily.csv"),
                        parse_dates=["date"]).set_index("date")
    preds = pd.read_csv(os.path.join(RESULTS, "s4_predictions.csv"),
                        parse_dates=["date"]).set_index("date")
    frame = pd.DataFrame(index=daily.index)
    frame[BASELINE] = daily["cams_aod550_mean"]
    frame[MODEL] = preds[MODEL]
    return frame


def label_series(dust, fraction, min_stations):
    """Rebuild equations (1) and (2) at a given pair of parameters.

    n_blowing and n_wx are the raw counts written by S2, so the label can be
    rebuilt here without re-reading a million surface observations. Days on
    which fewer than min_stations filed stay unlabelled rather than being
    assumed quiet, which is the whole point of the construction.
    """
    n_b = dust["n_blowing"]
    n_w = dust["n_wx"]
    ok = n_w >= min_stations
    index = (n_b / n_w).where(ok)
    return (index >= fraction).astype(float).where(ok)


def evaluate(frame, obs, col_a, col_b):
    """Threshold each forecast on validation HSS, score both on test."""
    work = frame[[col_a, col_b]].copy()
    work["obs"] = obs
    work = work.dropna()
    val = work[work.index.year.isin(VAL)]
    test = work[work.index.year.isin(TEST)]
    if len(val) < 100 or len(test) < 100 or test["obs"].sum() < 20:
        return None

    obs_v = val["obs"].to_numpy().astype(int)
    obs_t = test["obs"].to_numpy().astype(int)
    out = {}
    for col in (col_a, col_b):
        sv, st = val[col].to_numpy(), test[col].to_numpy()
        grid = np.linspace(*np.nanpercentile(sv, [1, 99]), 200)
        s_val = sweep(obs_v, sv, grid)
        thr = float(s_val.loc[s_val["HSS"].idxmax(), "threshold"])
        s = skill(obs_t, (st >= thr).astype(int))
        out[col] = {"threshold": thr, "CSI": s["CSI"], "HSS": s["HSS"],
                    "bias": s["frequency_bias"]}
    out["_n_test"] = len(test)
    out["_events"] = int(obs_t.sum())
    out["_base"] = float(obs_t.mean())
    return out


def main():
    dust = pd.read_csv(os.path.join(RESULTS, "s2_dust_daily.csv"),
                       parse_dates=["date"]).set_index("date")
    frame = build_frame()

    say("=" * 78)
    say("S13 LABEL SENSITIVITY")
    say("=" * 78)
    say()
    say(f"comparison: {MODEL} minus {BASELINE}")
    say("thresholds fixed on validation, applied to test, as in S5")
    say(f"paper's setting: fraction 0.25, minimum stations 3")
    say()
    say(f"  {'frac':>5} {'min':>4} {'labelled':>9} {'test n':>7} {'events':>7} "
        f"{'base':>6} {'CSI model':>10} {'CSI base':>9} {'dCSI':>8} {'dHSS':>8}")

    rows = []
    for m in MIN_STATIONS:
        for f in FRACTIONS:
            obs = label_series(dust, f, m)
            n_lab = int(obs.notna().sum())
            res = evaluate(frame, obs, MODEL, BASELINE)
            if res is None:
                say(f"  {f:5.2f} {m:4d} {n_lab:9d}   too few labelled days")
                continue
            a, b = res[MODEL], res[BASELINE]
            d_csi = a["CSI"] - b["CSI"]
            d_hss = a["HSS"] - b["HSS"]
            rows.append({
                "fraction": f, "min_stations": m, "labelled_days": n_lab,
                "test_days": res["_n_test"], "test_events": res["_events"],
                "test_base_rate": round(res["_base"], 4),
                "csi_model": round(a["CSI"], 4), "csi_baseline": round(b["CSI"], 4),
                "hss_model": round(a["HSS"], 4), "hss_baseline": round(b["HSS"], 4),
                "delta_csi": round(d_csi, 4), "delta_hss": round(d_hss, 4),
            })
            say(f"  {f:5.2f} {m:4d} {n_lab:9d} {res['_n_test']:7d} "
                f"{res['_events']:7d} {res['_base']:6.1%} "
                f"{a['CSI']:10.3f} {b['CSI']:9.3f} {d_csi:+8.3f} {d_hss:+8.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "s13_label_sensitivity.csv"), index=False)

    say()
    say("=" * 78)
    n = len(table)
    pos_csi = int((table.delta_csi > 0).sum())
    pos_hss = int((table.delta_hss > 0).sum())
    say(f"cells evaluated: {n}")
    say(f"  delta CSI positive in {pos_csi} of {n}, "
        f"range {table.delta_csi.min():+.3f} to {table.delta_csi.max():+.3f}")
    say(f"  delta HSS positive in {pos_hss} of {n}, "
        f"range {table.delta_hss.min():+.3f} to {table.delta_hss.max():+.3f}")
    if pos_csi == n and pos_hss == n:
        say("  the sign of the result is unchanged everywhere on the grid")
    else:
        say("  WARNING: the sign flips somewhere on the grid. The paper must "
            "state where, and the headline claim must be conditioned on the "
            "label definition.")
    say()

    with open(os.path.join(RESULTS, "s13_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    say(f"wrote {os.path.join(RESULTS, 's13_label_sensitivity.csv')}")


if __name__ == "__main__":
    main()
