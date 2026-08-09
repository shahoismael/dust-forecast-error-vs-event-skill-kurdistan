"""
S2b - restrict the ground dust index to the Kurdistan Region.

Stations are selected by governorate rather than by a latitude-longitude box,
so the study area matches an administrative boundary a reader can check.
Governorates included: Duhok, Erbil, Sulaymaniyah, Halabja.

Writes results/s2_dust_daily_kri.csv.
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

START, END = "2003-01-01", "2024-12-31"
INDEX = pd.date_range(START, END, freq="D")

# ISD station names inside the Kurdistan Region
KRI_STATIONS = {
    "ZAKHO", "DUHOOK",                       # Duhok
    "AQRAH", "SORAN", "SALLAHDEEN",          # Erbil
    "ERBEEL", "ERBIL INTL", "ERBIL", "BASHUR",
    "SULYMANIYAH", "CHEM-CHAMAL",            # Sulaymaniyah
    "HALABCHA",                              # Halabja
}

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def main():
    meta = pd.read_csv(
        os.path.join(ROOT, "datasets", "5_noaa_isd", "stations_northern_iraq.csv"),
        dtype=str,
    )
    meta["STATION"] = meta["USAF"].str.zfill(6) + meta["WBAN"].str.zfill(5)
    meta["NAME"] = meta["STATION NAME"].str.strip()

    keep = meta[meta["NAME"].isin(KRI_STATIONS)]
    say("=" * 70)
    say("S2b KURDISTAN REGION DUST INDEX")
    say("=" * 70)
    say(f"stations selected: {len(keep)}")
    for _, r in keep.iterrows():
        say(f"  {r['NAME']:14} {float(r['LAT']):6.3f} N  {float(r['LON']):6.3f} E")
    say()

    sd = pd.read_csv(os.path.join(RESULTS, "s2_station_daily.csv"),
                     parse_dates=["date"])
    sd["station"] = sd["station"].astype(str).str.zfill(11)
    sub = sd[sd["station"].isin(set(keep["STATION"]))]
    say(f"station-days: {len(sub):,}   stations with data: {sub['station'].nunique()}")

    regional = (
        sub.groupby("date")
        .agg(
            n_stations=("station", "nunique"),
            n_wx=("wx_reported", "sum"),
            n_vis=("vis_reported", "sum"),
            n_dust=("dust", "sum"),
            n_severe=("severe", "sum"),
            n_blowing=("blowing", "sum"),
            vis_min=("vis_min", "min"),
        )
        .reindex(INDEX)
    )
    regional.index.name = "date"
    for col in ["n_stations", "n_wx", "n_vis", "n_dust", "n_severe", "n_blowing"]:
        regional[col] = regional[col].fillna(0).astype(int)

    # A smaller network needs a lower minimum than the 3 used regionally,
    # otherwise most of the record is discarded. Two reporting stations is the
    # floor at which a "fraction of stations" index still means anything.
    wx_denom = regional["n_wx"].where(regional["n_wx"] >= 2)

    regional["dust_index"] = regional["n_dust"] / wx_denom
    regional["severe_index"] = regional["n_severe"] / wx_denom
    regional["blowing_index"] = regional["n_blowing"] / wx_denom
    regional["blowing_day"] = (regional["blowing_index"] >= 0.25).astype("Int64")
    regional.loc[wx_denom.isna(), "blowing_day"] = pd.NA
    regional["dust_day"] = (regional["dust_index"] >= 0.25).astype("Int64")
    regional.loc[wx_denom.isna(), "dust_day"] = pd.NA

    regional.to_csv(os.path.join(RESULTS, "s2_dust_daily_kri.csv"),
                    float_format="%.4f")

    usable = regional["blowing_day"].notna()
    say()
    say(f"usable days: {int(usable.sum())} of {len(regional)}")
    say(f"blowing-dust days: {int(regional.loc[usable, 'blowing_day'].sum())} "
        f"({regional.loc[usable, 'blowing_day'].mean():.1%})")
    say()

    say("by split")
    for name, years in [("train", range(2003, 2018)), ("val", range(2018, 2021)),
                        ("test", range(2021, 2025))]:
        chunk = regional[regional.index.year.isin(years)]
        u = chunk["blowing_day"].notna()
        say(f"  {name:6} labelled {int(u.sum()):5}  events "
            f"{int(chunk.loc[u, 'blowing_day'].sum()):4}  "
            f"base rate {chunk.loc[u, 'blowing_day'].mean():.1%}")
    say()

    daily = pd.read_csv(os.path.join(RESULTS, "s1_daily.csv"),
                        parse_dates=["date"]).set_index("date")
    joined = daily[["aod550"]].join(regional[["blowing_day", "blowing_index"]]).dropna()
    say(f"AOD on dust days {joined.loc[joined.blowing_day == 1, 'aod550'].mean():.3f}  "
        f"non-dust {joined.loc[joined.blowing_day == 0, 'aod550'].mean():.3f}")
    say(f"point-biserial r = "
        f"{joined['aod550'].corr(joined['blowing_day'].astype(float)):.3f}")

    say()
    say("by month")
    monthly = regional[usable].groupby(regional[usable].index.month)["blowing_day"].mean()
    say("  " + "  ".join(f"{m}:{v:.0%}" for m, v in monthly.items()))

    with open(os.path.join(RESULTS, "s2b_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {os.path.join(RESULTS, 's2_dust_daily_kri.csv')}")


if __name__ == "__main__":
    main()
