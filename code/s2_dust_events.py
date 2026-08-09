"""
S2 - ground-observed dust events from NOAA ISD.

WMO present-weather code 4677 dust and sand codes:
    06  widespread dust in suspension, not raised by wind at or near the station
    07  dust or sand raised by wind at or near the station
    08  well developed dust or sand whirls
    09  duststorm or sandstorm within sight or at the station during the past hour
    30-32  slight or moderate duststorm or sandstorm
    33-35  severe duststorm or sandstorm

Severe set is 09 and 30 to 35. Codes 06 and 07 are suspension and local raising.

Outputs
    results/s2_dust_daily.csv      regional daily index
    results/s2_station_daily.csv   station by day, for the map and for QC
    results/s2_report.txt
"""

import glob
import os

import numpy as np
import pandas as pd

# Project root. Override with the ICIEAHS_ROOT environment variable, otherwise
# the parent of this file's directory is used.
ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RAW = os.path.join(ROOT, "datasets", "5_noaa_isd", "raw")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

START, END = "2003-01-01", "2024-12-31"
INDEX = pd.date_range(START, END, freq="D")

DUST_CODES = {"06", "07", "08", "09", "30", "31", "32", "33", "34", "35"}
SEVERE_CODES = {"09", "30", "31", "32", "33", "34", "35"}
BLOWING_CODES = {"07", "08", "09", "30", "31", "32", "33", "34", "35"}

WX_COLUMNS = [f"MW{i}" for i in range(1, 5)] + [f"AW{i}" for i in range(1, 5)]

VIS_MISSING = 999999
VIS_DUST_THRESHOLD = 5000     # metres
RH_MAX_FOR_DUST = 60.0        # per cent, separates dust from fog and haze

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def first_field(series):
    """ISD packs comma separated groups; the code is the first field."""
    return series.astype(str).str.split(",").str[0].str.strip()


def parse_signed(series, scale=10.0):
    """TMP and DEW look like +0137,1 meaning 13.7 C. 9999 is missing."""
    raw = first_field(series)
    value = pd.to_numeric(raw, errors="coerce")
    return value.mask(value.abs() >= 9999) / scale


def relative_humidity(t_c, td_c):
    """Magnus formula."""
    a, b = 17.625, 243.04
    with np.errstate(invalid="ignore", over="ignore"):
        num = np.exp(a * td_c / (b + td_c))
        den = np.exp(a * t_c / (b + t_c))
        return 100.0 * num / den


def read_station_file(path):
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:  # noqa: BLE001
        return None
    if "DATE" not in df.columns or "STATION" not in df.columns:
        return None

    out = pd.DataFrame()
    out["station"] = df["STATION"].astype(str)
    out["date"] = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()

    # present weather
    codes = pd.Series([set() for _ in range(len(df))], index=df.index)
    present = [c for c in WX_COLUMNS if c in df.columns]
    for col in present:
        values = first_field(df[col]).str.zfill(2)
        values = values.where(df[col].notna())
        for i, v in values.items():
            if isinstance(v, str) and v in DUST_CODES:
                codes.at[i].add(v)

    out["dust_obs"] = codes.apply(lambda s: len(s) > 0)
    out["severe_obs"] = codes.apply(lambda s: bool(s & SEVERE_CODES))
    out["blowing_obs"] = codes.apply(lambda s: bool(s & BLOWING_CODES))

    # An observation only counts towards the code-based denominator if it
    # actually carries a present-weather group. Manual reporting collapses after
    # 2012, and using all observations as the denominator manufactures a false
    # downward trend in dust frequency.
    if present:
        reported = df[present].notna().any(axis=1)
    else:
        reported = pd.Series(False, index=df.index)
    out["wx_reported"] = reported.values

    # visibility, metres
    if "VIS" in df.columns:
        vis = pd.to_numeric(first_field(df["VIS"]), errors="coerce")
        out["vis"] = vis.mask(vis >= VIS_MISSING)
    else:
        out["vis"] = np.nan

    # humidity, to separate dust from fog
    if "TMP" in df.columns and "DEW" in df.columns:
        t = parse_signed(df["TMP"])
        td = parse_signed(df["DEW"])
        out["rh"] = relative_humidity(t, td)
    else:
        out["rh"] = np.nan

    return out.dropna(subset=["date"])


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    say("=" * 70)
    say("S2 GROUND DUST EVENTS")
    say("=" * 70)
    say(f"station-year files: {len(files)}")

    frames = []
    skipped = 0
    for i, path in enumerate(files, 1):
        part = read_station_file(path)
        if part is None or part.empty:
            skipped += 1
            continue
        frames.append(part)
        if i % 100 == 0:
            say(f"  read {i}/{len(files)}")

    obs = pd.concat(frames, ignore_index=True)
    say(f"observations: {len(obs):,}   files skipped: {skipped}")
    say(f"observations carrying a present-weather group: "
        f"{int(obs['wx_reported'].sum()):,} ({obs['wx_reported'].mean():.1%})")
    say(f"observations carrying a visibility value: "
        f"{int(obs['vis'].notna().sum()):,} ({obs['vis'].notna().mean():.1%})")
    say()

    # visibility-based dust, only where humidity is low enough to exclude fog
    low_vis = obs["vis"] < VIS_DUST_THRESHOLD
    dry = obs["rh"].isna() | (obs["rh"] < RH_MAX_FOR_DUST)
    obs["vis_dust_obs"] = low_vis & dry

    obs["any_dust_obs"] = obs["dust_obs"] | obs["vis_dust_obs"]

    say(f"observations with a dust code       : {int(obs['dust_obs'].sum()):,}")
    say(f"observations with severe dust code  : {int(obs['severe_obs'].sum()):,}")
    say(f"observations flagged by visibility  : {int(obs['vis_dust_obs'].sum()):,}")
    say(f"observations flagged by either      : {int(obs['any_dust_obs'].sum()):,}")
    say()

    # station by day
    obs["vis_reported"] = obs["vis"].notna()
    station_daily = (
        obs.groupby(["station", "date"])
        .agg(
            n_obs=("dust_obs", "size"),
            wx_reported=("wx_reported", "any"),
            vis_reported=("vis_reported", "any"),
            dust=("dust_obs", "any"),
            severe=("severe_obs", "any"),
            blowing=("blowing_obs", "any"),
            vis_dust=("vis_dust_obs", "any"),
            any_dust=("any_dust_obs", "any"),
            vis_min=("vis", "min"),
            rh_mean=("rh", "mean"),
        )
        .reset_index()
    )
    station_daily.to_csv(
        os.path.join(RESULTS, "s2_station_daily.csv"), index=False, float_format="%.2f"
    )
    say(f"station-days: {len(station_daily):,}")

    # regional daily index, with denominators conditioned on what was reported
    regional = (
        station_daily.groupby("date")
        .agg(
            n_stations=("station", "nunique"),
            n_wx=("wx_reported", "sum"),
            n_vis=("vis_reported", "sum"),
            n_dust=("dust", "sum"),
            n_severe=("severe", "sum"),
            n_blowing=("blowing", "sum"),
            n_vis_dust=("vis_dust", "sum"),
            vis_min=("vis_min", "min"),
        )
        .reindex(INDEX)
    )
    regional.index.name = "date"

    for col in ["n_stations", "n_wx", "n_vis", "n_dust", "n_severe",
                "n_blowing", "n_vis_dust"]:
        regional[col] = regional[col].fillna(0).astype(int)

    wx_denom = regional["n_wx"].where(regional["n_wx"] >= 3)
    vis_denom = regional["n_vis"].where(regional["n_vis"] >= 3)

    regional["dust_index"] = regional["n_dust"] / wx_denom
    regional["severe_index"] = regional["n_severe"] / wx_denom
    regional["vis_dust_index"] = regional["n_vis_dust"] / vis_denom

    # Primary event definition: blowing dust, codes 07, 08, 09 and 30 to 35.
    # Code 06 is dust in suspension not raised by local wind, which in this
    # region is near-permanent background haze and occurs on 68 per cent of
    # days. Including it makes "dust day" meaningless as an event label.
    regional["blowing_index"] = regional["n_blowing"] / wx_denom
    regional["blowing_day"] = (regional["blowing_index"] >= 0.25).astype("Int64")
    regional.loc[wx_denom.isna(), "blowing_day"] = pd.NA

    regional["dust_day"] = (regional["dust_index"] >= 0.25).astype("Int64")
    regional.loc[wx_denom.isna(), "dust_day"] = pd.NA

    regional["vis_dust_day"] = (regional["vis_dust_index"] >= 0.25).astype("Int64")
    regional.loc[vis_denom.isna(), "vis_dust_day"] = pd.NA

    regional.to_csv(os.path.join(RESULTS, "s2_dust_daily.csv"), float_format="%.4f")

    say()
    say("=" * 70)
    say("REGIONAL INDEX, denominators conditioned on reporting")
    say("=" * 70)

    for label, day_col, idx_col, denom in [
        ("code-based", "dust_day", "dust_index", "n_wx"),
        ("visibility-based", "vis_dust_day", "vis_dust_index", "n_vis"),
    ]:
        usable = regional[day_col].notna()
        say(f"{label}: {int(usable.sum())} usable days of {len(regional)}")
        say(f"  dust days: {int(regional.loc[usable, day_col].sum())} "
            f"({regional.loc[usable, day_col].mean():.1%})")
        say(f"  mean index: {regional.loc[usable, idx_col].mean():.3f}")
        say()

    say("by year: reporting stations and dust-day rate under both definitions")
    say(f"  {'year':>6} {'wx_st':>6} {'vis_st':>7} {'code%':>7} {'vis%':>7}")
    for year, chunk in regional.groupby(regional.index.year):
        u1 = chunk["dust_day"].notna()
        u2 = chunk["vis_dust_day"].notna()
        code_rate = chunk.loc[u1, "dust_day"].mean() if u1.any() else np.nan
        vis_rate = chunk.loc[u2, "vis_dust_day"].mean() if u2.any() else np.nan
        say(f"  {year:>6} {chunk['n_wx'].mean():6.1f} {chunk['n_vis'].mean():7.1f} "
            f"{code_rate:7.1%} {vis_rate:7.1%}"
            .replace("nan%", "   n/a"))
    say()

    say("dust-day rate by month, code-based")
    usable = regional["dust_day"].notna()
    monthly = regional[usable].groupby(regional[usable].index.month)["dust_day"].mean()
    for month, rate in monthly.items():
        say(f"  month {month:2d}: {rate:6.1%}")

    both = regional[["dust_index", "vis_dust_index"]].dropna()
    if len(both) > 100:
        say()
        say(f"agreement between the two indices: r = "
            f"{both['dust_index'].corr(both['vis_dust_index']):.3f} "
            f"on {len(both)} shared days")

    with open(os.path.join(RESULTS, "s2_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {os.path.join(RESULTS, 's2_dust_daily.csv')}")
    say(f"wrote {os.path.join(RESULTS, 's2_station_daily.csv')}")


if __name__ == "__main__":
    main()
