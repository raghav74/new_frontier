from validation_util import *

OUTPUTS_DIR = "outputs"
STATIONS_PATH = "old_aadt.csv"

MAX_STATION_MILES = 1.0
TX_LAT_MIN, TX_LAT_MAX = 25.8, 36.5
TX_LON_MIN, TX_LON_MAX = -106.6, -93.5

REQUIRED_COLS = [
    "Crash_ID",
    "Latitude",
    "Longitude",
    "ZIP_Code",
    "Adt_Curnt_Amt",
    "Distance_Miles",
    "Station_Year",
    "year_gap",
    "aadt_match_type",
    "VMT_Multiplier",
]

TX_VMT_MILLIONS = {
    2015: 258300,
    2016: 270700,
    2017: 273200,
    2018: 282200,
    2019: 288400,
    2020: 260000,
    2021: 285200,
    2022: 291100,
    2023: 301500,
    2024: 307800,
    2025: 307800,
}

ATOL_VMT = 1e-9

results = []

year_files = discover_yearly_files(OUTPUTS_DIR)

if not year_files:
    raise FileNotFoundError(f"No yearly files found in {OUTPUTS_DIR}")

print(f"{len(year_files)} yearly files:")
for year, path in year_files:
    print(f"  {year}: {path}")

total_rows = 0
total_filled = 0

for year, path in year_files:
    validate_schema(results, year, path, REQUIRED_COLS)
    validate_crash_year(results, year, path)
    validate_bounds(results, year, path, TX_LAT_MIN, TX_LAT_MAX, TX_LON_MIN, TX_LON_MAX)
    validate_distance_cap(results, year, path, MAX_STATION_MILES)
    validate_zip(results, year, path)
    validate_year_gap(results, year, path)
    validate_vmt(results, year, path, TX_VMT_MILLIONS, ATOL_VMT)
    validate_aadt(results, year, path)
    validate_match_type(results, year, path)

    rows, filled = validate_fill_rate(results, year, path)
    total_rows += rows
    total_filled += filled

overall = 100.0 * total_filled / total_rows if total_rows else 0.0

print(f"\nAll years: {total_filled:,}/{total_rows:,} = {overall:.2f}%")

summary = save_validation_results(results, OUTPUTS_DIR)