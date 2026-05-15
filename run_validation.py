import argparse
import os

import config
from validation_util import *

ATOL_VMT = 1e-9


ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True, choices=sorted(config.STATE_CONFIGS.keys()))
args = ap.parse_args()

state_cfg = config.STATE_CONFIGS[args.state]
OUTPUTS_DIR = os.path.join(config.OUT_DIR_ROOT, args.state)
lat_min, lat_max, lon_min, lon_max = state_cfg["bbox"]
VMT = state_cfg["vmt"]


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
    validate_schema(results, year, path, config.REQUIRED_COLS)
    validate_crash_year(results, year, path)
    validate_bounds(results, year, path, lat_min, lat_max, lon_min, lon_max)
    validate_distance_cap(results, year, path, config.MAX_STATION_MILES)
    validate_zip(results, year, path)
    validate_year_gap(results, year, path)
    validate_vmt(results, year, path, VMT, ATOL_VMT)
    validate_aadt(results, year, path)
    validate_match_type(results, year, path)

    rows, filled = validate_fill_rate(results, year, path)
    total_rows += rows
    total_filled += filled

overall = 100.0 * total_filled / total_rows if total_rows else 0.0

print(f"\nAll years: {total_filled:,}/{total_rows:,} = {overall:.2f}%")

summary = save_validation_results(results, OUTPUTS_DIR)
