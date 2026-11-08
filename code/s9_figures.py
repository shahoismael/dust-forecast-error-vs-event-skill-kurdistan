"""
S9 - figures and tables.

Everything is generated from the results CSVs at run time. Nothing is
transcribed by hand, so the text and the figures cannot drift apart.

PES single column is 80 mm. Figures are sized to that, or to 170 mm where they
must span both columns. Output is PNG at 300 dpi and vector PDF.

Writes figs/fig1..fig6 and tables/table1..table3 as CSV and markdown.
"""

import os
import sys

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
TABLES = os.path.join(ROOT, "tables")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

MM = 1 / 25.4
COL = 80 * MM
WIDE = 170 * MM

TEST = range(2021, 2025)

# Okabe-Ito, colour-blind safe
C = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)
    say(f"  wrote figs/{name}.png and .pdf")


def load():
    daily = pd.read_csv(os.path.join(RESULTS, "s1_daily.csv"),
                        parse_dates=["date"]).set_index("date")
    dust = pd.read_csv(os.path.join(RESULTS, "s2_dust_daily.csv"),
                       parse_dates=["date"]).set_index("date")
    preds = pd.read_csv(os.path.join(RESULTS, "s4_predictions.csv"),
                        parse_dates=["date"]).set_index("date")
    station = pd.read_csv(os.path.join(RESULTS, "s2_station_daily.csv"))
    return daily, dust, preds, station


# ---------------------------------------------------------------- figures --

def fig1_domain(station):
    """Study domain and the ISD stations that returned usable observations."""
    coords = (
        station.groupby("station")
        .size()
        .rename("n_days")
        .to_frame()
        .reset_index()
    )
    coords["station"] = coords["station"].astype(str).str.zfill(11)

    meta = pd.read_csv(
        os.path.join(ROOT, "datasets", "5_noaa_isd", "stations_northern_iraq.csv"),
        dtype=str,
    )
    meta["STATION"] = meta["USAF"].str.zfill(6) + meta["WBAN"].str.zfill(5)
    meta["LAT"] = pd.to_numeric(meta["LAT"], errors="coerce")
    meta["LON"] = pd.to_numeric(meta["LON"], errors="coerce")

    merged = coords.merge(meta, left_on="station", right_on="STATION", how="inner")
    silent = meta[~meta["STATION"].isin(coords["station"])]

    fig, ax = plt.subplots(figsize=(COL, COL * 0.92))

    ax.add_patch(plt.Rectangle((38.5, 30.5), 11, 9, fill=False,
                               edgecolor=C["black"], lw=0.9, ls="--", zorder=1))

    if len(silent):
        ax.scatter(silent["LON"], silent["LAT"], s=9, facecolor="none",
                   edgecolor="0.55", linewidth=0.5, zorder=2,
                   label=f"no usable data ({len(silent)})")

    sizes = 8 + 26 * merged["n_days"] / merged["n_days"].max()
    sc = ax.scatter(merged["LON"], merged["LAT"], s=sizes, c=merged["n_days"],
                    cmap="YlOrBr", edgecolor=C["black"], linewidth=0.35,
                    zorder=3, label=f"reporting ({len(merged)})")

    labels = {
        "DUHOOK": (5, 4), "ZAKHO": (-16, 3), "MOSUL": (4, -8),
        "ERBEEL": (5, 3), "SULYMANIYAH": (4, 2), "KIRKUK AB": (-22, -8),
        "BAGHDAD INTL AIRPORT": (-6, -9),
    }
    pretty = {
        "DUHOOK": "Duhok", "ZAKHO": "Zakho", "MOSUL": "Mosul",
        "ERBEEL": "Erbil", "SULYMANIYAH": "Sulaymaniyah",
        "KIRKUK AB": "Kirkuk", "BAGHDAD INTL AIRPORT": "Baghdad",
    }
    for _, row in merged.iterrows():
        name = str(row.get("STATION NAME", "")).strip()
        if name in labels:
            ax.annotate(pretty[name], (row["LON"], row["LAT"]),
                        textcoords="offset points", xytext=labels[name],
                        fontsize=6, zorder=5)

    ax.set_xlim(37.5, 50.5)
    ax.set_ylim(29.8, 40.2)
    ax.set_aspect(1 / np.cos(np.deg2rad(35)))
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.grid(alpha=0.25, lw=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=6, handletextpad=0.4,
              columnspacing=1.2, borderpad=0.1)

    cb = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label("station-days", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    save(fig, "fig1_domain")
    say(f"  stations in region list {len(meta)}, reporting {len(merged)}, "
        f"silent {len(silent)}")
    return len(merged)


def fig2_series(daily, dust):
    """AOD series with ground dust days marked."""
    fig, ax = plt.subplots(figsize=(WIDE, 55 * MM))
    y = daily["aod550"]
    ax.plot(y.index, y.values, lw=0.35, color=C["black"], alpha=0.85)

    roll = y.rolling(31, center=True, min_periods=10).mean()
    ax.plot(roll.index, roll.values, lw=1.1, color=C["vermillion"],
            label="31-day mean")

    # 2,646 individual markers on a 22-year axis render as a solid bar and
    # carry no information. Show the monthly fraction of reporting days that
    # were dust days instead, on a secondary axis.
    monthly = (dust["blowing_day"].dropna()
               .resample("MS").agg(["mean", "size"]))
    monthly = monthly[monthly["size"] >= 10]
    ax2 = ax.twinx()
    ax2.fill_between(monthly.index, 0, monthly["mean"].values,
                     color=C["blue"], alpha=0.30, lw=0,
                     label="ground dust days, monthly fraction")
    ax2.set_ylim(0, 3.0)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel("Dust-day fraction")
    ax2.tick_params(axis="y", labelsize=6)

    for boundary, text in [("2018-01-01", "validation"), ("2021-01-01", "test")]:
        ax.axvline(pd.Timestamp(boundary), color=C["green"], lw=0.7, ls=":")
        ax.text(pd.Timestamp(boundary), 1.45, f" {text}", fontsize=6,
                color=C["green"], va="top")

    ax.set_ylabel("AOD 550 nm")
    ax.set_ylim(0, 1.7)
    ax.set_xlabel("Year")
    ax.grid(alpha=0.25, lw=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, ncol=2)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    save(fig, "fig2_series")


def fig3_reporting(dust):
    """The reporting-density artefact, and what conditioning the denominator does.

    Panel (a): the fraction of reporting stations that actually filed a
    present-weather group, which falls by more than half over the record.
    Panel (b): the regional blowing-dust index computed two ways. Dividing by all
    reporting stations charges every silent station as a non-event and drives the
    index down with the reporting rate. Dividing by the stations that filed
    recovers part of the signal.
    """
    d = dust[dust["n_stations"] > 0].copy()
    corrected = d["n_blowing"] / d["n_wx"].where(d["n_wx"] >= 3)
    naive = d["n_blowing"] / d["n_stations"].where(d["n_stations"] >= 3)
    frame = pd.DataFrame({"corrected": corrected, "naive": naive},
                         index=d.index)
    annual = frame.groupby(frame.index.year).mean() * 100
    annual = annual.dropna(how="all")
    filed = (d["n_wx"] / d["n_stations"]).groupby(d.index.year).mean() * 100
    filed = filed[filed > 0]

    fig, axes = plt.subplots(2, 1, figsize=(COL, 78 * MM), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.35],
                                          "hspace": 0.16})

    ax = axes[0]
    ax.plot(filed.index, filed.values, marker="o", ms=2.4,
            color=C["black"], lw=0.9)
    ax.set_ylabel("Stations filing\npresent weather (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, lw=0.3)
    ax.text(0.015, 0.10, "(a)", transform=ax.transAxes, fontsize=8)

    ax = axes[1]
    ax.plot(annual.index, annual["naive"].values, marker="s", ms=2.4,
            color=C["vermillion"], lw=0.9,
            label="all stations in denominator")
    ax.plot(annual.index, annual["corrected"].values, marker="o", ms=2.4,
            color=C["blue"], lw=0.9,
            label="reporting stations only")
    ax.set_ylabel("Blowing-dust index (%)")
    ax.set_xlabel("Year")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25, lw=0.3)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.015, 0.92, "(b)", transform=ax.transAxes, fontsize=8)

    save(fig, "fig3_reporting")

    for name in ("naive", "corrected"):
        col = annual[name].dropna()
        slope = np.polyfit(col.index.values.astype(float), col.values, 1)[0]
        say(f"  {name:10} {col.iloc[0]:.1f}% to {col.iloc[-1]:.1f}%, "
            f"trend {slope:+.2f} pp/yr")
    say(f"  stations filing present weather: {filed.iloc[0]:.1f}% to "
        f"{filed.iloc[-1]:.1f}%")


def fig3_scatter(preds):
    """Observed against predicted, test period."""
    col = "meteo_only__random_forest"
    d = preds[["truth", col]].dropna()
    d = d[d.index.year.isin(TEST)]

    fig, ax = plt.subplots(figsize=(COL, COL))
    ax.scatter(d["truth"], d[col], s=4, alpha=0.4, color=C["blue"],
               edgecolor="none")
    lim = [0, max(d["truth"].max(), d[col].max()) * 1.05]
    ax.plot(lim, lim, color=C["black"], lw=0.7, ls="--")

    slope, intercept = np.polyfit(d["truth"], d[col], 1)
    xs = np.linspace(*lim, 50)
    ax.plot(xs, slope * xs + intercept, color=C["vermillion"], lw=0.9)

    r = d["truth"].corr(d[col])
    rmse = float(np.sqrt(np.mean((d[col] - d["truth"]) ** 2)))
    ax.text(0.04, 0.95,
            f"$n$ = {len(d)}\n$r$ = {r:.3f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, va="top", fontsize=7)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed AOD 550 nm")
    ax.set_ylabel("Predicted AOD 550 nm")
    ax.grid(alpha=0.25, lw=0.3)
    save(fig, "fig3_scatter")


def fig4_roc():
    """ROC for dust-day detection."""
    roc = pd.read_csv(os.path.join(RESULTS, "s5_roc.csv"))
    scores = pd.read_csv(os.path.join(RESULTS, "s5_event_scores.csv"))
    auc = dict(zip(scores["forecast"], scores["AUC"]))

    show = [
        ("meteo_only__random_forest", "Random forest, meteorology", C["blue"], "-"),
        ("climatology", "Climatology", C["orange"], "-"),
        ("cams", "Reanalysis", C["vermillion"], "-"),
        ("cams_dust", "Reanalysis, dust", C["green"], "--"),
        ("persistence", "Persistence", C["purple"], ":"),
    ]

    fig, ax = plt.subplots(figsize=(COL, COL))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
    for key, label, colour, style in show:
        chunk = roc[roc["forecast"] == key].sort_values("POFD")
        if chunk.empty:
            continue
        ax.plot(chunk["POFD"], chunk["POD"], color=colour, ls=style,
                label=f"{label} ({auc.get(key, float('nan')):.2f})")
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Probability of detection")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.3)
    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.9,
                edgecolor="none", title="AUC", title_fontsize=7)
    leg.get_frame().set_linewidth(0)
    save(fig, "fig4_roc")


def fig5_importance():
    """Permutation importance by feature group."""
    imp = pd.read_csv(os.path.join(RESULTS, "s6_importance.csv"))
    imp = imp[imp["rmse_increase"] > 0].sort_values("rmse_increase")

    fig, ax = plt.subplots(figsize=(COL, COL * 0.9))
    colours = [C["vermillion"] if g == "CAMS" else C["blue"] for g in imp["group"]]
    ax.barh(imp["group"], imp["rmse_increase"], color=colours,
            edgecolor="k", linewidth=0.3,
            xerr=imp["rmse_increase_sd"], error_kw={"lw": 0.5, "capsize": 1.5})
    ax.set_xlabel("Increase in RMSE when permuted")
    ax.grid(axis="x", alpha=0.25, lw=0.3)
    save(fig, "fig5_importance")


def fig6_seasonal():
    """Error and bias by month, against the seasonal AOD cycle."""
    m = pd.read_csv(os.path.join(RESULTS, "s6_seasonal.csv"))
    months = m["month"].astype(int)

    fig, ax = plt.subplots(figsize=(COL, COL * 0.75))
    ax.bar(months, m["rmse"], color=C["sky"], edgecolor="k", linewidth=0.3,
           label="RMSE")
    ax.plot(months, m["bias"], color=C["vermillion"], marker="o", ms=2.5,
            label="bias")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("AOD 550 nm")
    ax.set_xticks(range(1, 13))

    ax2 = ax.twinx()
    ax2.plot(months, m["mean_aod"], color=C["black"], ls="--", lw=0.8,
             label="mean AOD")
    ax2.set_ylabel("Mean observed AOD", fontsize=7)
    ax2.tick_params(labelsize=6)

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    save(fig, "fig6_seasonal")


# ----------------------------------------------------------------- tables --

def write_table(df, name, caption):
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    with open(os.path.join(TABLES, f"{name}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"**{caption}**\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")
    say(f"  wrote tables/{name}.csv and .md")


def table1_inventory(daily, dust):
    inv = pd.read_csv(os.path.join(RESULTS, "s0_inventory.csv"))
    rows = []
    for dataset, chunk in inv.groupby("dataset"):
        present = int(chunk["days_present"].sum())
        expected = int(chunk["days_expected"].sum())
        rows.append(
            {
                "Dataset": dataset,
                "Years": f"{int(chunk['year'].min())}-{int(chunk['year'].max())}",
                "Days available": present,
                "Days expected": expected,
                "Completeness": f"{present / expected:.1%}",
            }
        )
    write_table(pd.DataFrame(rows), "table1_inventory",
                "Table 1. Data inventory and completeness.")


def table2_continuous():
    """Continuous skill, every forecast scored on an identical set of days.

    The reanalysis-augmented models are defined on fewer days than the
    meteorology-only models, so scoring each on whatever days it happens to
    cover would compare different samples. All forecasts are therefore
    restricted to the test days on which every forecast is available, which is
    the same common set used for the event tables.
    """
    daily = pd.read_csv(
        os.path.join(RESULTS, "s1_daily.csv"), parse_dates=["date"]
    ).set_index("date")
    preds = pd.read_csv(
        os.path.join(RESULTS, "s4_predictions.csv"), parse_dates=["date"]
    ).set_index("date")

    frame = pd.DataFrame(index=daily.index)
    frame["truth"] = daily["aod550"]
    frame["persistence"] = daily["aod550"].shift(1)
    clim_src = daily[daily.index.year.isin(range(2003, 2018))]
    clim = clim_src.groupby(clim_src.index.dayofyear)["aod550"].mean()
    clim = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    frame["climatology"] = frame.index.dayofyear.map(clim)
    frame["cams"] = daily["cams_aod550_mean"]
    frame["cams_dust"] = daily["cams_duaod550_mean"]
    for col in preds.columns:
        if col != "truth":
            frame[col] = preds[col]

    frame = frame.dropna()
    frame = frame[frame.index.year.isin(TEST)]
    y = frame["truth"].to_numpy()
    n = len(frame)

    # moving-block bootstrap, 7-day blocks and 5000 replicates, as in S4, S5 and S8
    rng = np.random.default_rng(20260809)
    block = 7
    boot_idx = []
    for _ in range(5000):
        starts = rng.integers(0, max(n - block, 1), int(np.ceil(n / block)))
        boot_idx.append(np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n])

    rows = []
    for col in frame.columns:
        if col == "truth":
            continue
        # The dust component of the reanalysis is a dust-only field. Scoring it
        # as a forecast of total aerosol optical depth is not a like-for-like
        # comparison and produces a meaningless negative R2. It is kept in the
        # frame so that the common-day set is identical to the event tables, and
        # dropped from this table only, where it would mislead. It is scored in
        # Table 3, where the target is a dust event and the comparison is fair.
        if col == "cams_dust":
            continue
        f = frame[col].to_numpy()
        err = f - y
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        r2 = 1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
        vals = [float(np.sqrt(np.mean((f[i] - y[i]) ** 2))) for i in boot_idx]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        if "__" in col:
            fset, model = col.split("__")
        else:
            fset, model = "-", col
        rows.append({
            "Forecast": model,
            "Features": fset,
            "n": n,
            "RMSE": f"{rmse:.4f}",
            "95% CI": f"[{lo:.4f}, {hi:.4f}]",
            "MAE": f"{mae:.4f}",
            "R2": f"{r2:.3f}",
        })

    out = pd.DataFrame(rows).sort_values("RMSE")
    write_table(out, "table2_continuous",
                f"Table 2. Continuous forecast skill on the test period, "
                f"2021-2024. All forecasts scored on the same {n} days.")


def table3_events():
    ev = pd.read_csv(os.path.join(RESULTS, "s5_event_scores.csv"))
    pair = pd.read_csv(os.path.join(RESULTS, "s8_paired.csv"))

    versus = pair[pair["comparison"].str.endswith("_vs_cams")
                  & (pair["metric"] == "CSI")]
    csi_p = {r["comparison"].replace("_vs_cams", ""): r["p"]
             for _, r in versus.iterrows()}
    # Holm-adjusted p for the primary family (three meteorology-only models)
    csi_holm = {r["comparison"].replace("_vs_cams", ""): r["p_holm"]
                for _, r in versus.iterrows()
                if "p_holm" in versus.columns and pd.notna(r["p_holm"])}

    # S5 uses column names, S8 uses short labels. Bridge them explicitly so
    # the significance column is populated rather than silently blank.
    to_s8 = {
        "meteo_only__random_forest": "random_forest",
        "meteo_only__gradient_boosting": "gradient_boosting",
        "meteo_only__ridge": "ridge",
        "plus_reanalysis__gradient_boosting": "gb_plus_reanalysis",
        "climatology": "climatology",
        "persistence": "persistence",
        "cams_dust": "cams_dust",
        "observed_aod": "observed_aod",
    }

    rows = []
    for _, r in ev.iterrows():
        name = r["forecast"]
        short = name.replace("meteo_only__", "").replace("plus_reanalysis__", "+reanalysis ")
        rows.append(
            {
                "Forecast": short,
                "Threshold": f"{r['threshold']:.3f}",
                "B": f"{r['frequency_bias']:.2f}",
                "POD": f"{r['POD']:.3f}",
                "FAR": f"{r['FAR']:.3f}",
                "CSI": f"{r['CSI']:.3f}",
                "HSS": f"{r['HSS']:.3f}",
                "HSS 95% CI": f"[{r['HSS_lo']:.3f}, {r['HSS_hi']:.3f}]",
                "AUC": f"{r['AUC']:.3f}",
                "p vs CAMS (CSI)": (
                    f"{csi_p[to_s8[name]]:.4f}"
                    if to_s8.get(name) in csi_p else "-"
                ),
                "Holm p (CSI)": (
                    f"{csi_holm[to_s8[name]]:.4f}"
                    if to_s8.get(name) in csi_holm else "-"
                ),
            }
        )
    write_table(pd.DataFrame(rows), "table3_events",
                "Table 3. Dust-day detection skill on the test period. "
                "Thresholds fixed on validation years.")


def table4_cost():
    """Computational cost, straight from S11.

    Written here rather than typed into the manuscript because a hand-typed
    table is the one that goes stale after a rerun, and the paper claims every
    table is generated from the result files.
    """
    path = os.path.join(RESULTS, "s11_compute_cost.csv")
    if not os.path.exists(path):
        say("  table4 skipped: run s11_compute_cost.py first")
        return
    cost = pd.read_csv(path)
    head = cost[cost.feature_set == "meteo_only"]
    names = {"ridge": "Ridge", "random_forest": "Random forest",
             "gradient_boosting": "Grad. boosting"}
    rows = []
    for _, r in head.iterrows():
        rows.append({
            # Five columns, not seven. The build sets any table wider than five
            # across both columns, which costs a section break and a partial
            # page of column balancing. Wall-clock time and the per-day figure
            # are given in the prose instead, where they cost two clauses.
            "Model": names.get(r["model"], r["model"]),
            "Fit, CPU (s)": f"{r['fit_cpu_s']:.1f}",
            "Forecast series (ms)": f"{r['predict_ms']:.1f}",
            "Peak memory (MB)": f"{r['peak_rss_mb']:.0f}",
            "Stored model (MB)": f"{r['model_mb']:.2f}",
        })
    write_table(pd.DataFrame(rows), "table4_cost",
                "Table 4. Computational cost of the meteorology-only "
                "forecasts. Times are medians of five repetitions.")


def main():
    say("=" * 70)
    say("S9 FIGURES AND TABLES")
    say("=" * 70)
    say()

    daily, dust, preds, station = load()

    # The journal figures come from s14_journal_figures.py. Pass --tables-only
    # to rebuild the tables without regenerating the old figure set.
    if "--tables-only" not in sys.argv:
        say("figures")
        n_stations = fig1_domain(station)
        fig2_series(daily, dust)
        fig3_reporting(dust)
        fig3_scatter(preds)
        fig4_roc()
        fig5_importance()
        fig6_seasonal()
        say(f"  stations plotted: {n_stations}")
        say()

    say("tables")
    table1_inventory(daily, dust)
    table2_continuous()
    table3_events()
    table4_cost()

    with open(os.path.join(RESULTS, "s9_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
