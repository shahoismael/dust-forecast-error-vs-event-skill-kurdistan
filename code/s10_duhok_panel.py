"""
S10 - descriptive panel for Duhok governorate.

Two stations, Zakho and Duhok, report present weather here. Blowing dust is
coded on too few days to fit or test a model, so this stage is descriptive
only. It reports what the record contains and nothing more.

Writes figs/fig7_duhok.png and .pdf, results/s10_duhok.csv, results/s10_report.txt.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project root. Override with the ICIEAHS_ROOT environment variable, otherwise
# the parent of this file's directory is used.
ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

DUHOK_STATIONS = {"ZAKHO", "DUHOOK"}

MM = 1 / 25.4
COL = 80 * MM
WIDE = 170 * MM

C = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 0.9,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

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
    keep = meta[meta["NAME"].isin(DUHOK_STATIONS)]

    say("=" * 70)
    say("S10 DUHOK GOVERNORATE, DESCRIPTIVE")
    say("=" * 70)
    for _, r in keep.iterrows():
        say(f"  {r['NAME']:8} {float(r['LAT']):.3f} N  {float(r['LON']):.3f} E  "
            f"record {r['BEGIN'][:4]}-{r['END'][:4]}")
    say()

    station = pd.read_csv(os.path.join(RESULTS, "s2_station_daily.csv"),
                          parse_dates=["date"])
    station["station"] = station["station"].astype(str).str.zfill(11)
    sub = station[station["station"].isin(set(keep["STATION"]))].copy()

    name_of = dict(zip(keep["STATION"], keep["NAME"]))
    sub["name"] = sub["station"].map(name_of)

    daily = pd.read_csv(os.path.join(RESULTS, "s1_daily.csv"),
                        parse_dates=["date"]).set_index("date")

    # per station summary
    say("per station")
    say(f"  {'station':10} {'days':>6} {'wx days':>8} {'dust':>6} "
        f"{'blowing':>8} {'severe':>7} {'min vis m':>10}")
    rows = []
    for name, chunk in sub.groupby("name"):
        wx = int(chunk["wx_reported"].sum())
        rows.append({
            "station": name,
            "days": len(chunk),
            "wx_days": wx,
            "dust_days": int(chunk["dust"].sum()),
            "blowing_days": int(chunk["blowing"].sum()),
            "severe_days": int(chunk["severe"].sum()),
            "vis_min_m": float(chunk["vis_min"].min()),
            "first": chunk["date"].min().date(),
            "last": chunk["date"].max().date(),
        })
        say(f"  {name:10} {len(chunk):6} {wx:8} {int(chunk['dust'].sum()):6} "
            f"{int(chunk['blowing'].sum()):8} {int(chunk['severe'].sum()):7} "
            f"{chunk['vis_min'].min():10.0f}")
    say()

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "s10_duhok.csv"), index=False)

    # daily union across the two stations
    union = (
        sub.groupby("date")
        .agg(n_wx=("wx_reported", "sum"),
             dust=("dust", "max"),
             blowing=("blowing", "max"),
             vis_min=("vis_min", "min"))
    )
    union = union[union["n_wx"] >= 1]

    joined = daily[["aod550"]].join(union, how="inner")
    say(f"days with at least one Duhok-governorate report: {len(joined)}")
    say(f"  suspended or blowing dust reported: {int(joined['dust'].sum())} "
        f"({joined['dust'].mean():.1%})")
    say(f"  blowing dust reported:              {int(joined['blowing'].sum())} "
        f"({joined['blowing'].mean():.1%})")
    say()

    for flag, label in [(1, "dust reported"), (0, "no dust reported")]:
        chunk = joined[joined["dust"] == flag]
        if len(chunk):
            say(f"  regional AOD on days with {label:18}: "
                f"{chunk['aod550'].mean():.3f}  (n = {len(chunk)})")
    say()

    say("monthly frequency of any dust report, Duhok governorate")
    monthly = joined.groupby(joined.index.month)["dust"].mean()
    say("  " + "  ".join(f"{m}:{v:.0%}" for m, v in monthly.items()))
    say()

    say("annual count of days with any dust report")
    yearly = joined.groupby(joined.index.year)["dust"].sum()
    say("  " + "  ".join(f"{y}:{int(v)}" for y, v in yearly.items()))
    say()

    say("annual count of days on which either station reported present weather")
    uptime = joined.groupby(joined.index.year)["dust"].size()
    say("  " + "  ".join(f"{y}:{int(v)}" for y, v in uptime.items()))
    say()
    rate = (yearly / uptime).replace([np.inf, -np.inf], np.nan)
    say("dust days as a fraction of reporting days")
    say("  " + "  ".join(f"{y}:{v:.0%}" for y, v in rate.items()))
    say()
    r = np.corrcoef(uptime.values.astype(float),
                    yearly.values.astype(float))[0, 1]
    say(f"correlation between annual reporting days and annual dust-day count: "
        f"r = {r:.2f}")
    say("WARNING. The raw annual counts above track station reporting uptime,")
    say("not the atmosphere. Only the fraction of reporting days is")
    say("interpretable, and only the seasonal cycle is reported in the paper.")
    say("The raw annual series must not be presented as a dust trend.")

    # ------------------------------------------------------------ figure --
    fig, axes = plt.subplots(2, 1, figsize=(WIDE, 85 * MM), sharex=False,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.45})

    ax = axes[0]
    y = daily["aod550"]
    ax.plot(y.index, y.values, lw=0.3, color="0.6", alpha=0.8,
            label="regional AOD")
    roll = y.rolling(31, center=True, min_periods=10).mean()
    ax.plot(roll.index, roll.values, lw=1.0, color=C["black"],
            label="31-day mean")

    events = joined.index[joined["dust"] == 1]
    ax.scatter(events, np.full(len(events), 1.55), s=1.2, marker="|",
               color=C["vermillion"], label="dust reported at Zakho or Duhok")

    ax.set_ylabel("AOD 550 nm")
    ax.set_ylim(0, 1.65)
    ax.set_xlim(pd.Timestamp("2003-01-01"), pd.Timestamp("2025-01-01"))
    ax.grid(alpha=0.25, lw=0.3)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.set_title("(a) regional aerosol loading and Duhok governorate dust reports",
                 loc="left", fontsize=8)

    ax = axes[1]
    months = range(1, 13)
    vals = [monthly.get(m, 0) * 100 for m in months]
    ax.bar(months, vals, color=C["orange"], edgecolor="k", linewidth=0.3)
    ax.set_xticks(list(months))
    ax.set_xlabel("Month")
    ax.set_ylabel("Days with dust\nreported (%)")
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    ax.set_title("(b) seasonal cycle of reported dust, Zakho and Duhok",
                 loc="left", fontsize=8)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig7_duhok.{ext}"))
    plt.close(fig)

    say()
    say("wrote figs/fig7_duhok.png and .pdf")
    say(f"wrote {os.path.join(RESULTS, 's10_duhok.csv')}")

    with open(os.path.join(RESULTS, "s10_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
