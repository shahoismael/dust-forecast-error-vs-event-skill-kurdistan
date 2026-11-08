"""
S14 - journal figures for the Dasinya Journal for Engineering and Informatics.

Redraws every figure of the study from the result files. PNG only, 600 dpi,
sized to the DJ text block (A4, 1.27 cm side margins, so up to 170 mm wide).
Plotting only: result files are read, nothing in results/ is written, and no
number is recomputed differently from S9 and S10.

Main text
  fig1_study_domain_and_stations.png       Figure 1
  fig2_aerosol_series_and_dust_fraction.png Figure 2
  fig3_reporting_artefact.png              Figure 3
  fig4_roc_dust_day_detection.png          Figure 4
  fig5_duhok_governorate.png               Figure 5
Supplementary (repository)
  figS1_observed_vs_predicted.png
  figS2_permutation_importance.png
  figS3_seasonal_error.png

Run from the project root:
  python code\\s14_journal_figures.py

Figure 1 draws borders, rivers and the Kurdistan Region governorates when
cartopy is installed (first run downloads Natural Earth shapes, internet
needed). Without cartopy it draws the same figure on plain axes.
  python -m pip install cartopy
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")
STATIONS_CSV = os.path.join(ROOT, "datasets", "5_noaa_isd", "stations_northern_iraq.csv")
os.makedirs(FIGS, exist_ok=True)

# Analysis box, identical to AREA in 5_code/config.py (N, W, S, E = 40, 38, 30, 50)
LON0, LON1, LAT0, LAT1 = 38.0, 50.0, 30.0, 40.0
TEST = range(2021, 2025)
# A year enters an annual series only if at least this many days carry a label.
# 2003 has 41 labelled days and 2004 none; every year from 2005 has 213 or more.
MIN_LABELLED_DAYS = 180
SPLITS = [("2003-01-01", "2018-01-01", "Training"),
          ("2018-01-01", "2021-01-01", "Validation"),
          ("2021-01-01", "2025-01-01", "Test")]

MM = 1 / 25.4
FULL = 170 * MM
HALF = 85 * MM
DPI = 600

# Okabe-Ito, colour-blind safe
C = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#8C8C8C",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def pick_font():
    """Palatino matches the DJ body text; fall back gracefully."""
    for name in ("Palatino Linotype", "Book Antiqua", "TeX Gyre Pagella",
                 "URW Palladio L", "DejaVu Serif"):
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "serif"


FONT = pick_font()
plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#262626",
    "axes.labelcolor": "#1a1a1a",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 3,
    "xtick.color": "#262626",
    "ytick.color": "#262626",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "grid.color": "#dcdcdc",
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


def panel(ax, letter, dx=-30, dy=6):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9.5, fontweight="bold", ha="left", va="bottom")


def shade_splits(ax, label_y=None):
    """Light bands for validation and test years, with labels on request."""
    for start, end, name in SPLITS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if name != "Training":
            ax.axvspan(s, e, color="#eef2f6" if name == "Validation" else "#e3eaf1",
                       lw=0, zorder=0)
        if label_y is not None:
            ax.text(s + (e - s) / 2, label_y, name, ha="center", va="bottom",
                    fontsize=7, color="#4d4d4d",
                    transform=ax.get_xaxis_transform())


def year_axis(ax):
    ax.set_xlim(pd.Timestamp("2003-01-01"), pd.Timestamp("2025-01-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def save(fig, name):
    path = os.path.join(FIGS, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    # flatten to RGB on white: no alpha channel for Word or the journal system
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    rgb.save(path, dpi=(DPI, DPI))
    print(f"  {name}.png  {rgb.size[0]} x {rgb.size[1]} px, RGB")


# ------------------------------------------------------------------ data --

def load_daily():
    return pd.read_csv(os.path.join(RESULTS, "s1_daily.csv"),
                       usecols=["date", "aod550"],
                       parse_dates=["date"]).set_index("date")


def load_dust():
    return pd.read_csv(os.path.join(RESULTS, "s2_dust_daily.csv"),
                       parse_dates=["date"]).set_index("date")


def load_station_meta():
    meta = pd.read_csv(STATIONS_CSV, dtype=str)
    meta["STATION"] = meta["USAF"].str.zfill(6) + meta["WBAN"].str.zfill(5)
    meta["LAT"] = pd.to_numeric(meta["LAT"], errors="coerce")
    meta["LON"] = pd.to_numeric(meta["LON"], errors="coerce")
    meta["NAME"] = meta["STATION NAME"].astype(str).str.strip()
    return meta


# ------------------------------------------------------------- figure 1 --

NE_DIR = os.path.join(ROOT, "datasets", "natural_earth")
KRI_CODES = {"IQ-DA", "IQ-AR", "IQ-SU"}          # Duhok, Erbil, Sulaymaniyah
ZOOM = (40.6, 46.6, 32.62, 37.83)                  # panel (b) window, lon0 lon1 lat0 lat1
CITY = "#1F3B57"
LAND, SEA = "#f4f1ea", "#d7e6f2"
RIVER, KRI_FILL, KRI_EDGE = "#6f9fc8", "#efdcae", "#a8844a"


def ne_layers():
    """Natural Earth 1:10m layers from datasets/natural_earth, else download."""
    import cartopy.io.shapereader as shp

    def path(category, name):
        local = os.path.join(NE_DIR, f"ne_10m_{name}.shp")
        return local if os.path.exists(local) else shp.natural_earth("10m", category, name)

    return shp, {
        "land": path("physical", "land"),
        "lakes": path("physical", "lakes"),
        "rivers": path("physical", "rivers_lake_centerlines"),
        "coast": path("physical", "coastline"),
        "borders": path("cultural", "admin_0_boundary_lines_land"),
        "admin1": path("cultural", "admin_1_states_provinces"),
    }


def draw_base(ax, shp, layers, pc, iraq_lines=False, scale=1.0):
    ax.set_facecolor(SEA)
    ax.add_geometries(shp.Reader(layers["land"]).geometries(), pc,
                      facecolor=LAND, edgecolor="none", zorder=0)
    admin1 = list(shp.Reader(layers["admin1"]).records())
    kri = [r.geometry for r in admin1
           if r.attributes.get("iso_3166_2") in KRI_CODES]
    ax.add_geometries(kri, pc, facecolor=KRI_FILL, edgecolor="none", zorder=1)
    if iraq_lines:
        iraq = [r.geometry for r in admin1 if r.attributes.get("admin") == "Iraq"]
        ax.add_geometries(iraq, pc, facecolor="none", edgecolor="#c9bfae",
                          linewidth=0.35 * scale, zorder=1.2)
    ax.add_geometries(kri, pc, facecolor="none", edgecolor=KRI_EDGE,
                      linewidth=0.6 * scale, zorder=1.4)
    ax.add_geometries(shp.Reader(layers["lakes"]).geometries(), pc,
                      facecolor=SEA, edgecolor="#8fb3d1", linewidth=0.3, zorder=1.5)
    ax.add_geometries(shp.Reader(layers["rivers"]).geometries(), pc,
                      facecolor="none", edgecolor=RIVER, linewidth=0.5 * scale,
                      zorder=1.6)
    ax.add_geometries(shp.Reader(layers["coast"]).geometries(), pc,
                      facecolor="none", edgecolor="#5f7f99", linewidth=0.4, zorder=1.7)
    ax.add_geometries(shp.Reader(layers["borders"]).geometries(), pc,
                      facecolor="none", edgecolor="#4d4d4d",
                      linewidth=0.6 * scale, linestyle=(0, (4, 1.5)), zorder=1.8)
    return kri


def gridlines(ax, pc, xs, ys):
    gl = ax.gridlines(crs=pc, draw_labels=True, linewidth=0.3, color="#9a9a9a",
                      alpha=0.6, linestyle=":", xlocs=xs, ylocs=ys)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.8}
    gl.xpadding = gl.ypadding = 3
    for side in ax.spines.values():
        side.set_linewidth(0.6)


def fig1_domain():
    try:
        import cartopy.crs as ccrs
        shp, layers = ne_layers()
    except Exception as exc:
        print(f"  Figure 1 needs cartopy and Natural Earth ({exc}); skipped")
        return

    station = pd.read_csv(os.path.join(RESULTS, "s2_station_daily.csv"),
                          usecols=["station"])
    counts = (station["station"].astype(str).str.zfill(11)
              .value_counts().rename("n_days").rename_axis("station")
              .reset_index())
    meta = load_station_meta()
    used = counts.merge(meta, left_on="station", right_on="STATION", how="inner")
    silent = meta[~meta["STATION"].isin(counts["station"])]
    cmap = LinearSegmentedColormap.from_list(
        "station_days", plt.get_cmap("YlOrBr")(np.linspace(0.2, 1.0, 256)))

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(FULL, 90 * MM))
    ax = fig.add_axes([0.035, 0.205, 0.43, 0.75], projection=pc)
    bx = fig.add_axes([0.525, 0.205, 0.43, 0.75], projection=pc)

    # ---- (a) regional setting ------------------------------------------
    ax.set_extent([34.5, 53.5, 26.5, 43.0], crs=pc)
    draw_base(ax, shp, layers, pc, scale=0.85)
    ax.add_patch(Rectangle((LON0, LAT0), LON1 - LON0, LAT1 - LAT0, fill=False,
                           edgecolor=C["black"], lw=1.0, ls=(0, (5, 2)),
                           zorder=4, transform=pc))
    z0, z1, z2, z3 = ZOOM
    ax.add_patch(Rectangle((z0, z2), z1 - z0, z3 - z2, fill=False,
                           edgecolor=CITY, lw=0.9, zorder=4.5, transform=pc))
    ax.text(z0 + 0.1, z3 + 0.15, "(b)", fontsize=6.6, color=CITY,
            fontweight="bold", va="bottom", transform=pc, zorder=5)
    ax.scatter(meta["LON"], meta["LAT"], s=2.2, color="#3a3a3a", lw=0,
               zorder=4.6, transform=pc)
    countries = {"TURKEY": (40.6, 39.35), "SYRIA": (39.3, 35.6),
                 "IRAQ": (42.6, 31.4), "IRAN": (51.3, 33.0),
                 "SAUDI ARABIA": (41.5, 27.6), "KUWAIT": (47.6, 29.2),
                 "JORDAN": (36.6, 31.0)}
    for name, (lon, lat) in countries.items():
        ax.text(lon, lat, name, ha="center", va="center", fontsize=5.8,
                color="#6e6e6e", zorder=3, path_effects=HALO, transform=pc)
    ax.text(LON1 - 0.2, LAT1 - 0.25, "Study domain", ha="right", va="top",
            fontsize=6.4, style="italic", zorder=6, path_effects=HALO,
            transform=pc)
    gridlines(ax, pc, range(34, 56, 4), range(26, 46, 4))
    panel(ax, "a", dx=-6, dy=4)

    # ---- (b) station network --------------------------------------------
    bx.set_extent([z0, z1, z2, z3], crs=pc)
    draw_base(bx, shp, layers, pc, iraq_lines=True, scale=1.0)
    bx.scatter(silent["LON"], silent["LAT"], s=15, facecolor="white",
               edgecolor="#6b6b6b", linewidth=0.6, zorder=5, transform=pc)
    sc = bx.scatter(used["LON"], used["LAT"], s=22, c=used["n_days"], cmap=cmap,
                    vmin=0, edgecolor="#1a1a1a", linewidth=0.45, zorder=6,
                    transform=pc)
    cities = {"ZAKHO": "Zakho", "DUHOOK": "Duhok", "MOSUL": "Mosul",
              "ERBEEL": "Erbil", "SULYMANIYAH": "Sulaymaniyah",
              "KIRKUK AB": "Kirkuk", "BAGHDAD INTL AIRPORT": "Baghdad"}
    lab = meta[meta["NAME"].isin(cities)]
    bx.scatter(lab["LON"], lab["LAT"], s=44, facecolor="none", edgecolor=CITY,
               linewidth=0.75, zorder=6.5, transform=pc)
    # Fixed label positions (degrees). adjustText gave different layouts on
    # different machines, so the published layout is set by hand instead.
    spots = {
        "ZAKHO": (42.15, 37.45, "center"), "DUHOOK": (43.60, 37.22, "center"),
        "MOSUL": (42.95, 36.62, "right"), "ERBEEL": (44.45, 36.03, "left"),
        "KIRKUK AB": (44.35, 35.13, "center"),
        "SULYMANIYAH": (45.80, 35.95, "center"),
        "BAGHDAD INTL AIRPORT": (44.62, 33.27, "left"),
    }
    for _, row in lab.iterrows():
        tx, ty, ha = spots[row["NAME"]]
        ann = bx.annotate(
            cities[row["NAME"]], xy=(row["LON"], row["LAT"]), xytext=(tx, ty),
            ha=ha, va="center", fontsize=6.4, color=CITY, zorder=7,
            path_effects=HALO,
            arrowprops={"arrowstyle": "-", "color": CITY, "lw": 0.55,
                        "shrinkA": 2, "shrinkB": 3.6})
        ann.arrow_patch.set_zorder(6.6)
    bx.text(45.05, 32.80, "Tigris", fontsize=5.8, color=RIVER, style="italic",
            rotation=-30, zorder=3, path_effects=HALO, transform=pc)
    for name, (lon, lat) in {"TURKEY": (41.35, 37.62), "SYRIA": (40.95, 35.75),
                             "IRAN": (46.25, 34.45)}.items():
        bx.text(lon, lat, name, ha="center", va="center", fontsize=5.8,
                color="#6e6e6e", zorder=3, path_effects=HALO, transform=pc)
    bx.text(41.9, 34.62, "Euphrates", fontsize=5.8, color=RIVER, style="italic",
            rotation=-12, zorder=3, path_effects=HALO, transform=pc)
    gridlines(bx, pc, range(40, 48, 1), range(32, 39, 1))
    panel(bx, "b", dx=-6, dy=4)

    # ---- shared colour bar and legend ------------------------------------
    cax = fig.add_axes([0.585, 0.075, 0.31, 0.024])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Days with observations per station", fontsize=7)
    cb.ax.tick_params(labelsize=6.3, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    handles = [
        Line2D([], [], marker="o", ls="", mfc=cmap(0.55), mec="#1a1a1a",
               mew=0.45, ms=4.6, label=f"Station with observations ({len(used)})"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="#6b6b6b", mew=0.6,
               ms=4.2, label=f"Listed, no usable data ({len(silent)})"),
        Patch(facecolor=KRI_FILL, edgecolor=KRI_EDGE, lw=0.6,
              label="Kurdistan Region of Iraq"),
        Line2D([], [], color=C["black"], lw=1.0, ls=(0, (5, 2)),
               label="Study domain (30–40°N, 38–50°E)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.04, 0.0),
               ncol=1, fontsize=6.6, handletextpad=0.5, labelspacing=0.35,
               frameon=False)
    save(fig, "fig1_study_domain_and_stations")


# ------------------------------------------------------------- figure 2 --

def fig2_series(daily, dust):
    y = daily["aod550"]
    roll = y.rolling(31, center=True, min_periods=10).mean()
    monthly = dust["blowing_day"].dropna().resample("MS").agg(["mean", "size"])
    monthly = monthly[monthly["size"] >= 10]

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(FULL, 92 * MM), sharex=True,
                                 gridspec_kw={"height_ratios": [2.1, 1],
                                              "hspace": 0.22})
    shade_splits(ax, label_y=1.01)
    ax.plot(y.index, y.values, lw=0.3, color="#9c9c9c", label="Daily")
    ax.plot(roll.index, roll.values, lw=1.15, color=C["black"],
            label="31-day running mean")
    ax.set_ylabel("AOD at 550 nm")
    ax.set_ylim(0, 1.75)
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 0.99))
    panel(ax, "a")

    shade_splits(bx)
    bx.bar(monthly.index, monthly["mean"] * 100, width=24, align="edge",
           color=C["blue"], lw=0)
    bx.set_ylabel("Dust days (%\nof labelled days)")
    bx.set_ylim(0, max(60, np.ceil(monthly["mean"].max() * 10) * 10))
    bx.grid(axis="y")
    bx.set_xlabel("Year")
    year_axis(bx)
    panel(bx, "b")
    fig.align_ylabels([ax, bx])
    save(fig, "fig2_aerosol_series_and_dust_fraction")


# ------------------------------------------------------------- figure 3 --

def fig3_reporting(dust):
    d = dust[dust["n_stations"] > 0].copy()
    corrected = d["n_blowing"] / d["n_wx"].where(d["n_wx"] >= 3)
    naive = d["n_blowing"] / d["n_stations"].where(d["n_stations"] >= 3)
    annual = (pd.DataFrame({"corrected": corrected, "naive": naive}, index=d.index)
              .groupby(d.index.year).mean() * 100)
    filed = (d["n_wx"] / d["n_stations"]).groupby(d.index.year).mean() * 100
    # keep only years with enough labelled days; reindex so a missing year
    # breaks the line instead of being bridged
    labelled = corrected.groupby(d.index.year).count()
    years = labelled[labelled >= MIN_LABELLED_DAYS].index
    full = range(int(years.min()), int(years.max()) + 1)
    annual = annual.reindex(years).reindex(full)
    filed = filed.reindex(years).reindex(full)
    print(f"  Figure 3 years {years.min()}-{years.max()} "
          f"(dropped: {sorted(set(labelled.index) - set(years))})")

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(HALF, 100 * MM), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.3],
                                              "hspace": 0.18})
    for a in (ax, bx):
        a.axvspan(2008.5, 2013.5, color="#f2f2f2", lw=0, zorder=0)
    ax.plot(filed.index, filed.values, color=C["black"], marker="o", ms=3,
            mfc="white", mew=0.8, lw=1.0, zorder=3)
    ax.set_ylabel("Stations filing\npresent weather (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y")
    ax.text(2011.0, 97, "Reporting decline", ha="center", va="top", fontsize=6.6,
            color="#5a5a5a", style="italic")
    first, last_f = filed.first_valid_index(), filed.last_valid_index()
    for yr, off, ha in ((first, (0, 6), "center"), (last_f, (0, -11), "center")):
        ax.annotate(f"{filed[yr]:.1f}", (yr, filed[yr]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=6.8, path_effects=HALO)
    panel(ax, "a")

    bx.plot(annual.index, annual["naive"], color=C["vermillion"], marker="s",
            ms=3, mew=0, lw=1.0, label="All reporting stations", zorder=3)
    bx.plot(annual.index, annual["corrected"], color=C["blue"], marker="o",
            ms=3, mew=0, lw=1.0, label="Stations filing present weather", zorder=3)
    last = annual["corrected"].last_valid_index()
    for col, colour in (("naive", C["vermillion"]), ("corrected", C["blue"])):
        bx.annotate(f"{annual.loc[last, col]:.1f}", (last, annual.loc[last, col]),
                    textcoords="offset points", xytext=(5, 0), va="center",
                    fontsize=6.8, color=colour, path_effects=HALO)
    bx.set_ylabel("Blowing-dust index (%)")
    bx.set_xlabel("Year")
    bx.set_ylim(0, None)
    bx.grid(axis="y")
    bx.set_xlim(annual.index[0] - 0.8, annual.index[-1] + 1.6)
    bx.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    bx.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    bx.legend(loc="upper right", title="Denominator", title_fontsize=7.2,
              alignment="left")
    panel(bx, "b")
    fig.align_ylabels([ax, bx])
    save(fig, "fig3_reporting_artefact")


# ------------------------------------------------------------- figure 4 --

def fig4_roc():
    roc = pd.read_csv(os.path.join(RESULTS, "s5_roc.csv"))
    scores = pd.read_csv(os.path.join(RESULTS, "s5_event_scores.csv")).set_index("forecast")
    show = [
        ("meteo_only__random_forest", "Random forest, meteorology", C["blue"], "-", 1.5),
        ("climatology", "Climatology", C["orange"], "-", 1.0),
        ("cams", "CAMS reanalysis, total", C["vermillion"], "-", 1.0),
        ("cams_dust", "CAMS reanalysis, dust", C["green"], (0, (4, 1.6)), 1.0),
        ("persistence", "Persistence", C["purple"], (0, (1, 1.3)), 1.1),
    ]
    fig, ax = plt.subplots(figsize=(HALF, HALF))
    ax.plot([0, 1], [0, 1], color="#a6a6a6", lw=0.6, ls=(0, (3, 2)), zorder=1)
    handles = []
    for key, label, colour, style, lw in show:
        chunk = roc[roc["forecast"] == key].sort_values(["POFD", "POD"])
        if chunk.empty:
            continue
        xs = np.r_[0.0, chunk["POFD"].to_numpy(), 1.0]
        ys = np.r_[0.0, chunk["POD"].to_numpy(), 1.0]
        ax.plot(xs, ys, color=colour, ls=style, lw=lw, zorder=3)
        if key in scores.index:
            ax.plot(scores.loc[key, "POFD"], scores.loc[key, "POD"], marker="o",
                    ms=4.6, mfc=colour, mec="white", mew=0.8, ls="", zorder=5)
        auc = scores["AUC"].get(key, np.nan)
        handles.append(Line2D([], [], color=colour, ls=style, lw=lw,
                              label=f"{label} ({auc:.3f})"))
    handles.append(Line2D([], [], marker="o", ls="", mfc="#7f7f7f", mec="white",
                          mew=0.8, ms=4.6, label="Validation-fixed threshold"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True)
    ax.set_xlabel("Probability of false detection")
    ax.set_ylabel("Probability of detection")
    ax.legend(handles=handles, loc="lower right", title="Forecast (AUC)", title_fontsize=7.2,
              alignment="right", fontsize=6.8, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92, borderpad=0.4)
    save(fig, "fig4_roc_dust_day_detection")


# ------------------------------------------------------------- figure 5 --

def fig5_duhok(daily):
    meta = load_station_meta()
    keep = meta[meta["NAME"].isin({"ZAKHO", "DUHOOK"})]
    station = pd.read_csv(os.path.join(RESULTS, "s2_station_daily.csv"),
                          usecols=["station", "date", "wx_reported", "dust"],
                          parse_dates=["date"])
    station["station"] = station["station"].astype(str).str.zfill(11)
    sub = station[station["station"].isin(set(keep["STATION"]))]
    union = sub.groupby("date").agg(n_wx=("wx_reported", "sum"),
                                    dust=("dust", "max"))
    union = union[union["n_wx"] >= 1]
    joined = daily[["aod550"]].join(union, how="inner")
    monthly = joined.groupby(joined.index.month)["dust"].mean()

    y = daily["aod550"]
    roll = y.rolling(31, center=True, min_periods=10).mean()
    events = joined.index[joined["dust"] == 1]

    fig, (sx, ax, bx) = plt.subplots(
        3, 1, figsize=(FULL, 100 * MM),
        gridspec_kw={"height_ratios": [0.16, 1.9, 1], "hspace": 0.08})
    # event strip: one tick per day with dust reported at Zakho or Duhok
    sx.vlines(events, 0, 1, color=C["vermillion"], lw=0.35)
    sx.set_ylim(0, 1)
    sx.set_yticks([])
    sx.spines["left"].set_visible(False)
    sx.spines["bottom"].set_visible(False)
    sx.tick_params(axis="x", bottom=False, labelbottom=False)
    year_axis(sx)
    sx.set_ylabel("Dust\nreported", rotation=0, ha="right", va="center",
                  fontsize=7, labelpad=4)
    panel(sx, "a", dy=4)

    ax.plot(y.index, y.values, lw=0.3, color="#a9a9a9", label="Daily")
    ax.plot(roll.index, roll.values, lw=1.1, color=C["black"],
            label="31-day running mean")
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("AOD at 550 nm")
    year_axis(ax)
    ax.grid(axis="y")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color=C["vermillion"], lw=1.2,
               label=f"Dust reported at Zakho or Duhok ({len(events)} days)")],
        loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.02), fontsize=7)

    # extra gap before the seasonal panel
    pos = bx.get_position()
    bx.set_position([pos.x0, pos.y0 - 0.06, pos.width, pos.height])
    vals = np.array([monthly.get(m, np.nan) * 100 for m in range(1, 13)])
    bars = bx.bar(range(1, 13), vals, width=0.68, color=C["orange"],
                  edgecolor="#7a5200", linewidth=0.4)
    peak = int(np.nanargmax(vals))
    bars[peak].set_color(C["vermillion"])
    bars[peak].set_edgecolor("#6e2d00")
    for i, v in enumerate(vals, start=1):
        if np.isfinite(v):
            bx.text(i, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6)
    bx.set_xticks(range(1, 13))
    bx.set_xticklabels(MONTHS)
    bx.set_xlim(0.4, 12.6)
    bx.set_ylim(0, max(100, np.nanmax(vals) + 12))
    bx.set_ylabel("Dust days (% of\ndays with a report)")
    bx.grid(axis="y")
    panel(bx, "b")
    fig.align_ylabels([ax, bx])
    save(fig, "fig5_duhok_governorate")


# ------------------------------------------------------- supplementary --

def figS1_scatter():
    preds = pd.read_csv(os.path.join(RESULTS, "s4_predictions.csv"),
                        parse_dates=["date"]).set_index("date")
    # Same common-day set as Table 2: test days on which every forecast,
    # including both reanalysis fields, is defined.
    daily = pd.read_csv(os.path.join(RESULTS, "s1_daily.csv"),
                        usecols=["date", "aod550", "cams_aod550_mean",
                                 "cams_duaod550_mean"],
                        parse_dates=["date"]).set_index("date")
    frame = pd.DataFrame(index=daily.index)
    frame["truth"] = daily["aod550"]
    frame["persistence"] = daily["aod550"].shift(1)
    clim_src = daily[daily.index.year.isin(range(2003, 2018))]
    clim = clim_src.groupby(clim_src.index.dayofyear)["aod550"].mean()
    clim = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    frame["climatology"] = frame.index.dayofyear.map(clim)
    frame["cams"] = daily["cams_aod550_mean"]
    frame["cams_dust"] = daily["cams_duaod550_mean"]
    for c in preds.columns:
        if c != "truth":
            frame[c] = preds[c]
    frame = frame.dropna()
    frame = frame[frame.index.year.isin(TEST)]
    col = "meteo_only__random_forest"
    d = frame[["truth", col]]
    print(f"  Figure S1 common test days: {len(d)}")
    lim = [0, float(np.ceil(max(d["truth"].max(), d[col].max()) * 10) / 10)]

    fig, ax = plt.subplots(figsize=(HALF, HALF))
    ax.scatter(d["truth"], d[col], s=6, alpha=0.35, color=C["blue"],
               edgecolor="none", rasterized=True)
    ax.plot(lim, lim, color="#4d4d4d", lw=0.7, ls=(0, (4, 2)), label="1:1")
    slope, intercept = np.polyfit(d["truth"], d[col], 1)
    xs = np.linspace(*lim, 50)
    ax.plot(xs, slope * xs + intercept, color=C["vermillion"], lw=1.1,
            label=f"Least-squares fit (slope {slope:.2f})")
    r = d["truth"].corr(d[col])
    rmse = float(np.sqrt(np.mean((d[col] - d["truth"]) ** 2)))
    ax.text(0.04, 0.96, f"$n$ = {len(d)}\n$r$ = {r:.3f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, va="top", fontsize=7.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("Observed AOD at 550 nm")
    ax.set_ylabel("Random forest forecast, AOD at 550 nm")
    ax.legend(loc="lower right")
    save(fig, "figS1_observed_vs_predicted")


def figS2_importance():
    imp = pd.read_csv(os.path.join(RESULTS, "s6_importance.csv"))
    imp = imp[imp["rmse_increase"] > 0].sort_values("rmse_increase")
    names = [g[0].upper() + g[1:] for g in imp["group"]]
    fig, ax = plt.subplots(figsize=(HALF, 64 * MM))
    ax.barh(names, imp["rmse_increase"], height=0.62, color=C["blue"],
            xerr=imp["rmse_increase_sd"],
            error_kw={"lw": 0.6, "capsize": 1.8, "capthick": 0.6, "ecolor": "#333333"})
    if "relative" in imp:
        for yv, (v, rel) in enumerate(zip(imp["rmse_increase"], imp["relative"])):
            sd = imp["rmse_increase_sd"].iloc[yv]
            ax.text(v + sd + imp["rmse_increase"].max() * 0.03, yv,
                    f"{rel * 100:.1f}%", va="center", fontsize=6.6)
    ax.set_xlim(0, imp["rmse_increase"].max() * 1.25)
    ax.set_xlabel("Increase in test RMSE when permuted")
    ax.text(0.98, 0.04, "Gradient boosting, meteorology only\ntest period 2021\u20132024",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color="#4d4d4d", linespacing=1.2)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    save(fig, "figS2_permutation_importance")


def figS3_seasonal():
    m = pd.read_csv(os.path.join(RESULTS, "s6_seasonal.csv"))
    months = m["month"].astype(int)
    fig, ax = plt.subplots(figsize=(HALF, 64 * MM))
    ax.bar(months, m["rmse"], width=0.66, color=C["sky"], edgecolor="#2c6e91",
           linewidth=0.4, label="RMSE")
    ax.plot(months, m["bias"], color=C["vermillion"], marker="o", ms=3.2,
            mew=0, lw=1.0, label="Bias")
    ax.axhline(0, color="#7f7f7f", lw=0.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([s[0] for s in MONTHS])
    ax.set_ylabel("Error, AOD at 550 nm")
    ax.grid(axis="y")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(months, m["mean_aod"], color=C["black"], ls=(0, (4, 2)), lw=0.9,
             label="Mean observed AOD")
    ax2.set_ylabel("Mean observed AOD")
    ax2.set_ylim(0, max(0.6, m["mean_aod"].max() * 1.25))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    order = [l1.index("RMSE"), l1.index("Bias")]
    ax.legend([h1[i] for i in order] + h2, [l1[i] for i in order] + l2,
              loc="upper right", ncol=1)
    ax.set_title("Gradient boosting, meteorology only, test period 2021\u20132024",
                 loc="left", fontsize=7, color="#4d4d4d", pad=6)
    ax.set_xlabel("Month")
    save(fig, "figS3_seasonal_error")


def main():
    print(f"S14 journal figures, font: {FONT}, output: {FIGS}")
    daily = load_daily()
    dust = load_dust()
    fig1_domain()
    fig2_series(daily, dust)
    fig3_reporting(dust)
    fig4_roc()
    fig5_duhok(daily)
    figS1_scatter()
    figS2_importance()
    figS3_seasonal()
    print("done")


if __name__ == "__main__":
    main()
