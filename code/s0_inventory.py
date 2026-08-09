"""
S0 - inventory and quality control.

Verifies completeness of all five datasets and quantifies missingness per year.
Writes results/s0_inventory.csv and results/s0_report.txt. Reads only, changes nothing.

    python -m pip install pandas numpy xarray netCDF4
"""

import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd

# Project root. Override with the ICIEAHS_ROOT environment variable, otherwise
# the parent of this file's directory is used.
ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA = os.path.join(ROOT, "datasets")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

START = "2003-01-01"
END = "2024-12-31"
FULL_INDEX = pd.date_range(START, END, freq="D")

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


# ---------------------------------------------------------------- Giovanni --

def read_giovanni(path):
    """Giovanni area-averaged CSV: metadata header, blank line, then time,value."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    fill = None
    start = None
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
    df = df.dropna(subset=["time"]).set_index("time")["value"]
    df = pd.to_numeric(df, errors="coerce")

    if fill is not None:
        df = df.mask(np.isclose(df, fill, rtol=1e-6, atol=abs(fill) * 1e-6))
    df = df.mask(df < -1e10)
    return df.sort_index()


def audit_series(name, series, rows):
    aligned = series.reindex(FULL_INDEX)
    for year, chunk in aligned.groupby(aligned.index.year):
        rows.append(
            {
                "dataset": name,
                "year": year,
                "days_expected": len(chunk),
                "days_present": int(chunk.notna().sum()),
                "missing_fraction": round(1 - chunk.notna().mean(), 4),
                "mean": round(float(chunk.mean()), 5) if chunk.notna().any() else np.nan,
                "min": round(float(chunk.min()), 5) if chunk.notna().any() else np.nan,
                "max": round(float(chunk.max()), 5) if chunk.notna().any() else np.nan,
            }
        )
    return aligned


# -------------------------------------------------------------------- ERA5 --

def era5_members(path, kind):
    """
    ERA5 downloads are sometimes plain NetCDF and sometimes a zip holding one
    file per step type. Return a list of readable NetCDF paths, extracting if needed.
    """
    if not zipfile.is_zipfile(path):
        return [path]

    extract_dir = os.path.join(DATA, "2_era5", kind, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    year = re.search(r"(\d{4})\.nc$", path).group(1)

    out = []
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.endswith(".nc"):
                continue
            tag = "accum" if "accum" in member else "instant"
            dest = os.path.join(extract_dir, f"era5_{kind}_{year}_{tag}.nc")
            if not os.path.exists(dest):
                with z.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            out.append(dest)
    return out


def audit_era5(kind, rows):
    import xarray as xr

    files = sorted(glob.glob(os.path.join(DATA, "2_era5", kind, "*.nc")))
    say(f"ERA5 {kind}: {len(files)} files")
    if not files:
        return

    variables, dims = [], {}
    for member in era5_members(files[0], kind):
        with xr.open_dataset(member) as ds:
            variables += list(ds.data_vars)
            dims.update({d: int(s) for d, s in ds.sizes.items()})
    say(f"  variables: {variables}")
    say(f"  dims: {dims}")

    for path in files:
        year = int(re.search(r"(\d{4})\.nc$", path).group(1))
        n_times = 0
        for member in era5_members(path, kind):
            with xr.open_dataset(member) as ds:
                tname = "valid_time" if "valid_time" in ds.coords else "time"
                n_times = max(n_times, int(ds.sizes.get(tname, 0)))
        expected = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365
        rows.append(
            {
                "dataset": f"era5_{kind}",
                "year": year,
                "days_expected": expected,
                "days_present": n_times,
                "missing_fraction": round(1 - n_times / expected, 4),
                "mean": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        )


# -------------------------------------------------------------------- CAMS --

def audit_cams(rows):
    import xarray as xr

    files = sorted(glob.glob(os.path.join(DATA, "3_cams", "*.zip")))
    say(f"CAMS: {len(files)} archives")
    if not files:
        return

    extract_dir = os.path.join(DATA, "3_cams", "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    first = True
    for path in files:
        year = int(re.search(r"(\d{4})\.zip$", path).group(1))
        with zipfile.ZipFile(path) as z:
            member = z.namelist()[0]
            out = os.path.join(extract_dir, f"cams_{year}.nc")
            if not os.path.exists(out):
                with z.open(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())

        with xr.open_dataset(out) as ds:
            if first:
                say(f"  variables: {list(ds.data_vars)}")
                say(f"  dims: { {d: int(s) for d, s in ds.sizes.items()} }")
                first = False
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            n_times = int(ds.sizes.get(tname, 0))

        expected = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365
        rows.append(
            {
                "dataset": "cams",
                "year": year,
                "days_expected": expected,
                "days_present": n_times,
                "missing_fraction": round(1 - n_times / expected, 4),
                "mean": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        )


# --------------------------------------------------------------------- ISD --

def audit_isd(rows):
    files = sorted(glob.glob(os.path.join(DATA, "5_noaa_isd", "raw", "*.csv")))
    say(f"ISD: {len(files)} station-year files")
    if not files:
        return

    per_year = {}
    stations_per_year = {}
    for path in files:
        m = re.search(r"(\d+)_(\d{4})\.csv$", os.path.basename(path))
        if not m:
            continue
        station, year = m.group(1), int(m.group(2))
        try:
            n = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
        except Exception:  # noqa: BLE001
            n = 0
        per_year[year] = per_year.get(year, 0) + max(n, 0)
        stations_per_year.setdefault(year, set()).add(station)

    for year in sorted(per_year):
        expected = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365
        rows.append(
            {
                "dataset": "isd",
                "year": year,
                "days_expected": expected,
                "days_present": len(stations_per_year[year]),
                "missing_fraction": np.nan,
                "mean": per_year[year],
                "min": np.nan,
                "max": np.nan,
            }
        )
    say(f"  observation rows total: {sum(per_year.values()):,}")
    say(f"  stations reporting, by year: "
        f"{ {y: len(s) for y, s in sorted(stations_per_year.items())} }")


# -------------------------------------------------------------------- main --

def main():
    rows = []

    say("=" * 70)
    say("S0 INVENTORY")
    say("=" * 70)
    say()

    # MODIS
    terra_path = glob.glob(os.path.join(DATA, "1_modis_aod", "*MOD08*.csv"))[0]
    aqua_path = glob.glob(os.path.join(DATA, "1_modis_aod", "*MYD08*.csv"))[0]
    terra = read_giovanni(terra_path)
    aqua = read_giovanni(aqua_path)
    say(f"MODIS Terra: {terra.notna().sum()} valid days, "
        f"{terra.index.min().date()} to {terra.index.max().date()}")
    say(f"MODIS Aqua : {aqua.notna().sum()} valid days, "
        f"{aqua.index.min().date()} to {aqua.index.max().date()}")

    terra_a = audit_series("modis_terra", terra, rows)
    aqua_a = audit_series("modis_aqua", aqua, rows)

    both = terra_a.notna() & aqua_a.notna()
    only_t = terra_a.notna() & ~aqua_a.notna()
    only_a = ~terra_a.notna() & aqua_a.notna()
    neither = ~terra_a.notna() & ~aqua_a.notna()
    say(f"  both sensors : {int(both.sum())} days")
    say(f"  Terra only   : {int(only_t.sum())} days")
    say(f"  Aqua only    : {int(only_a.sum())} days")
    say(f"  neither      : {int(neither.sum())} days")
    corr = terra_a[both].corr(aqua_a[both])
    say(f"  Terra-Aqua correlation on shared days: r = {corr:.4f}")
    say()

    # OMI
    omi_files = glob.glob(os.path.join(DATA, "4_OMI_UV", "*.csv"))
    if omi_files:
        omi = read_giovanni(omi_files[0])
        say(f"OMI UV Aerosol Index: {omi.notna().sum()} valid days, "
            f"{omi.index.min().date()} to {omi.index.max().date()}")
        audit_series("omi_uvai", omi, rows)
        say()

    audit_era5("pressure", rows)
    say()
    audit_era5("single", rows)
    say()
    audit_cams(rows)
    say()
    audit_isd(rows)
    say()

    df = pd.DataFrame(rows).sort_values(["dataset", "year"])
    out_csv = os.path.join(RESULTS, "s0_inventory.csv")
    df.to_csv(out_csv, index=False)

    say("=" * 70)
    say("MISSINGNESS BY DATASET, whole period")
    say("=" * 70)
    for name, chunk in df.groupby("dataset"):
        present = chunk["days_present"].sum()
        expected = chunk["days_expected"].sum()
        say(f"{name:16} {present:6} / {expected:6} = {present/expected:6.1%}")
    say()

    say("MODIS seasonal missingness, target only")
    seasonal = pd.DataFrame({"terra": terra_a, "aqua": aqua_a})
    seasonal["month"] = seasonal.index.month
    merged = seasonal[["terra", "aqua"]].mean(axis=1)
    by_month = merged.groupby(seasonal["month"]).apply(lambda s: 1 - s.notna().mean())
    for month, frac in by_month.items():
        say(f"  month {month:2d}: {frac:6.1%} missing")

    with open(os.path.join(RESULTS, "s0_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")

    say()
    say(f"wrote {out_csv}")
    say(f"wrote {os.path.join(RESULTS, 's0_report.txt')}")


if __name__ == "__main__":
    main()
