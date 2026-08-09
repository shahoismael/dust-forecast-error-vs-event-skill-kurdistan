"""
S1 - harmonise every source onto one daily table.

Target   : merged Terra and Aqua MODIS AOD 550 nm, area mean over the box
Predictors: ERA5 pressure and single levels, reduced to area mean, spatial
            standard deviation and west-to-east gradient; CAMS total and dust AOD;
            OMI UV aerosol index; calendar terms

Writes results/s1_daily.csv and results/s1_report.txt.
"""

import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

# Project root. Override with the ICIEAHS_ROOT environment variable, otherwise
# the parent of this file's directory is used.
ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA = os.path.join(ROOT, "datasets")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

START, END = "2003-01-01", "2024-12-31"
INDEX = pd.date_range(START, END, freq="D")

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------- Giovanni CSV --

def read_giovanni(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    fill, start = None, None
    for i, line in enumerate(lines):
        if line.lower().startswith("fill value"):
            m = re.search(r",\s*(-?[\d.eE+]+)\s*$", line.strip())
            if m:
                fill = float(m.group(1))
        if line.strip().lower().startswith("time,"):
            start = i
            break
    if start is None:
        raise ValueError(f"no data header in {path}")

    df = pd.read_csv(path, skiprows=start)
    df.columns = ["time", "value"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    s = df.dropna(subset=["time"]).set_index("time")["value"]
    s = pd.to_numeric(s, errors="coerce")
    if fill is not None:
        s = s.mask(np.isclose(s, fill, rtol=1e-6, atol=abs(fill) * 1e-6))
    return s.mask(s < -1e10).sort_index()


# -------------------------------------------------------------- NetCDF prep --

def members(path, kind):
    """Return readable NetCDF paths, extracting zipped downloads if needed."""
    if not zipfile.is_zipfile(path):
        return [path]

    extract_dir = os.path.join(os.path.dirname(path), "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    year = re.search(r"(\d{4})", os.path.basename(path)).group(1)

    out = []
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.endswith(".nc"):
                continue
            tag = "accum" if "accum" in member else "instant"
            dest = os.path.join(extract_dir, f"{kind}_{year}_{tag}.nc")
            if not os.path.exists(dest):
                with z.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            out.append(dest)
    return out


def time_name(ds):
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    raise KeyError("no time coordinate")


def lon_name(ds):
    for candidate in ("longitude", "lon"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("no longitude coordinate")


def lat_name(ds):
    for candidate in ("latitude", "lat"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("no latitude coordinate")


def reduce_field(da, tname, latn, lonn, prefix, with_gradient=True):
    """Area mean, spatial standard deviation, and west-to-east gradient."""
    out = {}
    out[f"{prefix}_mean"] = da.mean(dim=[latn, lonn], skipna=True)
    out[f"{prefix}_sd"] = da.std(dim=[latn, lonn], skipna=True)

    if with_gradient:
        lons = da[lonn].values
        mid = np.median(lons)
        west = da.where(da[lonn] <= mid, drop=True).mean(dim=[latn, lonn], skipna=True)
        east = da.where(da[lonn] > mid, drop=True).mean(dim=[latn, lonn], skipna=True)
        out[f"{prefix}_gradEW"] = east - west

        lats = da[latn].values
        midlat = np.median(lats)
        south = da.where(da[latn] <= midlat, drop=True).mean(dim=[latn, lonn], skipna=True)
        north = da.where(da[latn] > midlat, drop=True).mean(dim=[latn, lonn], skipna=True)
        out[f"{prefix}_gradNS"] = north - south

    frames = {}
    for name, series in out.items():
        frames[name] = pd.Series(
            series.values.astype("float64"),
            index=pd.DatetimeIndex(series[tname].values).normalize(),
        )
    return pd.DataFrame(frames)


# -------------------------------------------------------------------- ERA5 --

def build_era5_pressure():
    files = sorted(glob.glob(os.path.join(DATA, "2_era5", "pressure", "*.nc")))
    say(f"ERA5 pressure: {len(files)} files")
    chunks = []
    for path in files:
        for member in members(path, "era5_pressure"):
            with xr.open_dataset(member) as ds:
                tname, latn, lonn = time_name(ds), lat_name(ds), lon_name(ds)
                levels = ds["pressure_level"].values
                per_year = []
                for var in ds.data_vars:
                    for level in levels:
                        da = ds[var].sel(pressure_level=level)
                        per_year.append(
                            reduce_field(da, tname, latn, lonn, f"{var}{int(level)}")
                        )
                chunks.append(pd.concat(per_year, axis=1))
    df = pd.concat(chunks).sort_index()
    say(f"  -> {df.shape[1]} columns, {df.shape[0]} days")
    return df


def build_era5_single():
    files = sorted(glob.glob(os.path.join(DATA, "2_era5", "single", "*.nc")))
    say(f"ERA5 single: {len(files)} files")
    chunks = []
    for path in files:
        parts = []
        for member in members(path, "era5_single"):
            with xr.open_dataset(member) as ds:
                tname, latn, lonn = time_name(ds), lat_name(ds), lon_name(ds)
                for var in ds.data_vars:
                    parts.append(reduce_field(ds[var], tname, latn, lonn, var))
        chunks.append(pd.concat(parts, axis=1))
    df = pd.concat(chunks).sort_index()
    say(f"  -> {df.shape[1]} columns, {df.shape[0]} days")
    return df


# -------------------------------------------------------------------- CAMS --

def build_cams():
    files = sorted(glob.glob(os.path.join(DATA, "3_cams", "*.zip")))
    say(f"CAMS: {len(files)} archives")
    chunks = []
    for path in files:
        for member in members(path, "cams"):
            with xr.open_dataset(member) as ds:
                tname, latn, lonn = time_name(ds), lat_name(ds), lon_name(ds)
                parts = [
                    reduce_field(ds[var], tname, latn, lonn, f"cams_{var}")
                    for var in ds.data_vars
                ]
                chunks.append(pd.concat(parts, axis=1))
    df = pd.concat(chunks).sort_index()
    say(f"  -> {df.shape[1]} columns, {df.shape[0]} days")
    return df


# -------------------------------------------------------------------- main --

def main():
    say("=" * 70)
    say("S1 DAILY TABLE")
    say("=" * 70)
    say()

    terra = read_giovanni(glob.glob(os.path.join(DATA, "1_modis_aod", "*MOD08*.csv"))[0])
    aqua = read_giovanni(glob.glob(os.path.join(DATA, "1_modis_aod", "*MYD08*.csv"))[0])
    terra = terra.reindex(INDEX)
    aqua = aqua.reindex(INDEX)

    target = pd.concat([terra, aqua], axis=1).mean(axis=1, skipna=True)
    target.name = "aod550"
    say(f"target: {int(target.notna().sum())} valid days of {len(INDEX)}")
    say(f"  mean {target.mean():.4f}  sd {target.std():.4f}  "
        f"min {target.min():.4f}  max {target.max():.4f}")
    say(f"  95th percentile {target.quantile(0.95):.4f}  "
        f"99th {target.quantile(0.99):.4f}")
    say()

    frame = pd.DataFrame(index=INDEX)
    frame["aod550"] = target
    frame["aod550_terra"] = terra
    frame["aod550_aqua"] = aqua

    omi_files = glob.glob(os.path.join(DATA, "4_OMI_UV", "*.csv"))
    if omi_files:
        omi = read_giovanni(omi_files[0]).reindex(INDEX)
        frame["omi_uvai"] = omi
        say(f"OMI: {int(omi.notna().sum())} valid days")
        say()

    for builder in (build_era5_pressure, build_era5_single, build_cams):
        part = builder()
        part = part[~part.index.duplicated(keep="first")].reindex(INDEX)
        frame = frame.join(part, how="left")
    say()

    # calendar terms
    doy = frame.index.dayofyear
    frame["sin_doy"] = np.sin(2 * np.pi * doy / 365.2425)
    frame["cos_doy"] = np.cos(2 * np.pi * doy / 365.2425)
    frame["month"] = frame.index.month
    frame["year"] = frame.index.year

    # lagged target, 1 to 5 days, matching the DustNet lag choice
    for lag in range(1, 6):
        frame[f"aod550_lag{lag}"] = frame["aod550"].shift(lag)

    say("=" * 70)
    say(f"final table: {frame.shape[0]} rows, {frame.shape[1]} columns")
    say("=" * 70)

    complete = frame.dropna(subset=["aod550"] + [f"aod550_lag{i}" for i in range(1, 6)])
    say(f"rows with target and all five lags present: {len(complete)}")

    missing = frame.isna().mean().sort_values(ascending=False)
    worst = missing[missing > 0].head(12)
    if len(worst):
        say()
        say("columns with missing values:")
        for name, frac in worst.items():
            say(f"  {name:28} {frac:6.2%}")

    out = os.path.join(RESULTS, "s1_daily.csv")
    frame.to_csv(out, index_label="date", float_format="%.6f")
    say()
    say(f"wrote {out}")

    with open(os.path.join(RESULTS, "s1_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
