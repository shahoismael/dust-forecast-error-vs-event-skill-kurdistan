"""
NOAA Integrated Surface Database — Iraqi station observations.
Visibility and present-weather codes give ground-observed dust and sand events.

No account required. Data served over HTTPS from NCEI.

    python -m pip install pandas requests

Step 1 lists Iraqi stations and their record spans, so coverage can be checked
before committing to this dataset.
"""

import io
import os
import requests
import pandas as pd

OUT_DIR = os.path.join(
    os.environ.get(
        "ICIEAHS_ROOT",
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ),
    "datasets", "5_noaa_isd",
)
os.makedirs(OUT_DIR, exist_ok=True)

HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ACCESS = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"

START_YEAR = 2003
END_YEAR = 2024

# Kurdistan Region and northern Iraq
LAT_MIN, LAT_MAX = 33.0, 38.0
LON_MIN, LON_MAX = 41.0, 47.0


def load_stations():
    resp = requests.get(HISTORY, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str)
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    iraq = df[df["CTRY"] == "IZ"].copy()
    region = iraq[
        iraq["LAT"].between(LAT_MIN, LAT_MAX)
        & iraq["LON"].between(LON_MIN, LON_MAX)
    ].copy()
    region["STATION"] = region["USAF"].str.zfill(6) + region["WBAN"].str.zfill(5)
    return iraq, region


def main():
    iraq, region = load_stations()

    iraq.to_csv(os.path.join(OUT_DIR, "stations_iraq_all.csv"), index=False)
    region.to_csv(os.path.join(OUT_DIR, "stations_northern_iraq.csv"), index=False)

    print(f"Iraqi stations in ISD history: {len(iraq)}")
    print(f"Within study box: {len(region)}")
    print(region[["STATION", "STATION NAME", "LAT", "LON", "BEGIN", "END"]].to_string(index=False))

    if region.empty:
        print("\nNo stations in box. This dataset is not usable, stop here.")
        return

    raw = os.path.join(OUT_DIR, "raw")
    os.makedirs(raw, exist_ok=True)

    for station in region["STATION"]:
        for year in range(START_YEAR, END_YEAR + 1):
            target = os.path.join(raw, f"{station}_{year}.csv")
            if os.path.exists(target):
                continue
            url = ACCESS.format(year=year, station=station)
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {station} {year}: {exc}")
                continue
            with open(target, "wb") as fh:
                fh.write(r.content)
            print(f"saved {station} {year}")

    print(f"done -> {raw}")


if __name__ == "__main__":
    main()
