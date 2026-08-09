"""
CAMS global reanalysis EAC4 — dust and total aerosol optical depth at 550 nm.

Credentials file: %USERPROFILE%\\.adsapirc
    url: https://ads.atmosphere.copernicus.eu/api
    key: <your ADS personal access token>

Requested one year at a time; a single 21-year request is rejected as too large.
"""

import os
import sys
import cdsapi

OUT_DIR = os.path.join(
    os.environ.get(
        "ICIEAHS_ROOT",
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ),
    "datasets", "3_cams",
)
os.makedirs(OUT_DIR, exist_ok=True)

ADS_RC = os.path.join(os.path.expanduser("~"), ".adsapirc")

DATASET = "cams-global-reanalysis-eac4"
YEAR_START = 2003
YEAR_END = 2023

VARIABLES = [
    "dust_aerosol_optical_depth_550nm",
    "total_aerosol_optical_depth_550nm",
]

AREA = [40, 38, 30, 50]


def read_credentials(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}\n"
            "Create it with two lines:\n"
            "  url: https://ads.atmosphere.copernicus.eu/api\n"
            "  key: <your ADS token>"
        )
    values = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if ":" in line:
                name, _, value = line.partition(":")
                values[name.strip()] = value.strip()
    if not values.get("key"):
        raise SystemExit(f"No key found in {path}")
    return values.get("url", "https://ads.atmosphere.copernicus.eu/api"), values["key"]


def request_for(year):
    return {
        "variable": VARIABLES,
        "date": [f"{year}-01-01/{year}-12-31"],
        "time": ["12:00"],
        "data_format": "netcdf_zip",
        "area": AREA,
    }


def main():
    url, key = read_credentials(ADS_RC)
    client = cdsapi.Client(url=url, key=key)

    for year in range(YEAR_START, YEAR_END + 1):
        target = os.path.join(OUT_DIR, f"cams_aod_{year}.zip")
        if os.path.exists(target):
            print(f"skip {year}")
            continue
        print(f"requesting {year}")
        try:
            client.retrieve(DATASET, request_for(year)).download(target)
            print(f"saved {year}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {year}: {exc}", file=sys.stderr)

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
