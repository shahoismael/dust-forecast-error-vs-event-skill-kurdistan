"""
S15 - numbers quoted in the text about the reporting record and the label.

Uses the same rules as Figure 3 in s14_journal_figures.py: the conditioned
index divides blowing-dust reports by stations that filed a present-weather
group (at least three), the conventional index divides by all reporting
stations (at least three), and annual statistics keep years with at least
180 labelled days. Also reports the event rate in each chronological split.

Also checks whether the composition of the filing network drifts over time
and recomputes the index over fixed station panels.

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


def fixed_panel(say):
    """Composition check for the conditioned index.

    The stations that file present weather change over time. If the late
    network sits closer to dust sources than the early one, the conditioned
    index partly measures which stations file. Two checks:
      1. mean latitude of filing station-days per year (composition drift);
      2. the conditioned index recomputed over a fixed panel of stations that
         filed on at least MIN_DAYS_PER_YEAR days in at least K of the years
         2005-2024, for several K, with the same three-station daily rule.
    """
    MIN_DAYS_PER_YEAR = 30
    YEARS = range(2005, 2025)
    st = pd.read_csv(os.path.join(RESULTS, "s2_station_daily.csv"),
                     parse_dates=["date"], dtype={"station": str})
    st["station"] = st["station"].str.zfill(11)
    st = st[st["date"].dt.year.isin(YEARS)]
    meta = pd.read_csv(os.path.join(ROOT, "datasets", "5_noaa_isd", "stations_northern_iraq.csv"), dtype=str)
    meta["station"] = meta["USAF"].str.zfill(6) + meta["WBAN"].str.zfill(5)
    lat = dict(zip(meta["station"], pd.to_numeric(meta["LAT"], errors="coerce")))
    st["lat"] = st["station"].map(lat)

    filed = st[st["wx_reported"].astype(str).str.lower() == "true"].copy()
    filed["blowing"] = filed["blowing"].astype(str).str.lower() == "true"
    filed["year"] = filed["date"].dt.year

    say()
    say("-" * 70)
    say("COMPOSITION OF THE FILING NETWORK")
    say("-" * 70)
    say("mean latitude of filing station-days (deg N), by year")
    by_year = filed.groupby("year")["lat"].mean()
    say("  " + "  ".join(f"{y}:{v:.2f}" for y, v in by_year.items()))
    slope = np.polyfit(by_year.index.to_numpy(float), by_year.to_numpy(), 1)[0]
    say(f"  trend {slope * 10:+.2f} deg per decade (negative = network shifting south)")

    days = filed.groupby(["station", "year"]).size().unstack(fill_value=0)
    years_ok = (days >= MIN_DAYS_PER_YEAR).sum(axis=1)
    say()
    say(f"stations filing on >= {MIN_DAYS_PER_YEAR} days in a year, counted over 2005-2024")
    for k in (10, 12, 15):
        panel = years_ok[years_ok >= k].index
        say()
        say(f"fixed panel: filed in >= {k} of 20 years -> {len(panel)} stations")
        if len(panel) < 3:
            say("  fewer than three stations; index not computed")
            continue
        sub = filed[filed["station"].isin(panel)]
        daily = sub.groupby("date").agg(n_wx=("station", "size"), n_blowing=("blowing", "sum"))
        daily = daily[daily["n_wx"] >= 3]
        idx = (daily["n_blowing"] / daily["n_wx"]) * 100
        ann = idx.groupby(idx.index.year).agg(["mean", "size"])
        ann = ann[ann["size"] >= MIN_LABELLED_DAYS]
        if len(ann) < 5:
            say(f"  only {len(ann)} years with >= {MIN_LABELLED_DAYS} labelled days; trend not computed")
            continue
        s = ann["mean"]
        tr = np.polyfit(s.index.to_numpy(float), s.to_numpy(), 1)[0]
        say("  " + "  ".join(f"{y}:{v:.1f}" for y, v in s.items()))
        say(f"  {s.index[0]} {s.iloc[0]:.1f}% to {s.index[-1]} {s.iloc[-1]:.1f}%, "
            f"OLS trend {tr:+.2f} percentage points per year, years used {len(s)}")


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

    fixed_panel(say)

    with open(os.path.join(RESULTS, "s15_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
