"""
S15 - numbers quoted in the text about the reporting record and the label.

Uses the same rules as Figure 3 in s14_journal_figures.py: the conditioned
index divides blowing-dust reports by stations that filed a present-weather
group (at least three), the conventional index divides by all reporting
stations (at least three), and annual statistics keep years with at least
180 labelled days. Also reports the event rate in each chronological split.

Writes results/s15_report.txt.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(ROOT, "results")
MIN_LABELLED_DAYS = 180
SPLITS = {"training": range(2003, 2018), "validation": range(2018, 2021), "test": range(2021, 2025)}

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def main():
    dust = pd.read_csv(os.path.join(RESULTS, "s2_dust_daily.csv"),
                       parse_dates=["date"]).set_index("date")
    d = dust[dust["n_stations"] > 0]
    conditioned = d["n_blowing"] / d["n_wx"].where(d["n_wx"] >= 3)
    conventional = d["n_blowing"] / d["n_stations"].where(d["n_stations"] >= 3)
    year = d.index.year
    annual = pd.DataFrame({"conditioned": conditioned, "conventional": conventional}).groupby(year).mean() * 100
    filed = (d["n_wx"] / d["n_stations"]).groupby(year).mean() * 100
    labelled = conditioned.groupby(year).count()
    keep = labelled[labelled >= MIN_LABELLED_DAYS].index
    dropped = sorted(set(labelled.index) - set(keep))

    say("=" * 70)
    say("S15 REPORTING RECORD AND LABEL NUMBERS")
    say("=" * 70)
    say(f"years kept (>= {MIN_LABELLED_DAYS} labelled days): {keep.min()}-{keep.max()}, dropped: {dropped}")
    say(f"labelled days in dropped years: {int(labelled.reindex(dropped).fillna(0).sum())}")
    say()
    say(f"  {'year':>4} {'filed %':>8} {'conventional %':>15} {'conditioned %':>14} {'labelled':>9}")
    for y in keep:
        say(f"  {y:>4} {filed[y]:8.1f} {annual.loc[y, 'conventional']:15.1f} "
            f"{annual.loc[y, 'conditioned']:14.1f} {int(labelled[y]):9d}")
    say()
    for col in ("conventional", "conditioned"):
        s = annual.loc[keep, col]
        slope = np.polyfit(np.asarray(keep, float), s.to_numpy(), 1)[0]
        say(f"{col:13} {s.iloc[0]:.1f}% in {keep[0]} to {s.iloc[-1]:.1f}% in {keep[-1]}, "
            f"OLS trend {slope:+.2f} percentage points per year")
    gap = (annual.loc[keep, "conditioned"] - annual.loc[keep, "conventional"])
    say(f"largest gap up to 2008: {gap[gap.index <= 2008].max():.1f} percentage points")
    say(f"ratio of 2024 values, conditioned / conventional: "
        f"{annual.loc[keep[-1], 'conditioned'] / annual.loc[keep[-1], 'conventional']:.1f}")
    say()
    event = dust["blowing_day"].dropna()
    for name, yrs in SPLITS.items():
        e = event[event.index.year.isin(yrs)]
        say(f"{name:10} labelled days {len(e):5d}  event rate {e.mean():.1%}")

    with open(os.path.join(RESULTS, "s15_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
