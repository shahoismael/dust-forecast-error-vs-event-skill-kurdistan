import os
import sys
import cdsapi

OUT_DIR = os.path.join(
    os.environ.get(
        "ICIEAHS_ROOT",
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ),
    "datasets", "2_era5", "pressure",
)
os.makedirs(OUT_DIR, exist_ok=True)

DATASET = "reanalysis-era5-pressure-levels"

VARIABLES = [
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

LEVELS = ["550", "750", "850", "950", "1000"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
AREA = [40, 38, 30, 50]

YEAR_START = 2003
YEAR_END = 2024


def request_for(year):
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": ["12:00"],
        "pressure_level": LEVELS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }


def main():
    client = cdsapi.Client()
    for year in range(YEAR_START, YEAR_END + 1):
        target = os.path.join(OUT_DIR, f"era5_pressure_{year}.nc")
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
