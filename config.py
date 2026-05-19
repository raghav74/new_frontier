"""Configuration for the AADT-fill pipeline.

State-specific settings live in STATE_CONFIGS. To add a new state:
  1. Run discover_mapping.py against the new state's master and stations CSVs
  2. Review the proposed mapping and paste it into STATE_CONFIGS below
  3. Run: python aadt_fill_main.py --state <CODE>
"""

# ----------------------------------------------------------------------------
# Universal parameters (apply to all states)
# ----------------------------------------------------------------------------

CRASH_YEARS = list(range(2015, 2025))   # years the loop attempts to process

MAX_STATION_MILES  = 1.0
EARTH_RADIUS_MILES = 3958.8

# Output directory base. Per-state outputs go in OUT_DIR_ROOT/<state>/.
OUT_DIR_ROOT = 'outputs/'

# Required canonical output columns; every state's output CSV must contain these.
REQUIRED_COLS = [
    'Crash_ID', 'Latitude', 'Longitude', 'ZIP_Code',
    'Adt_Curnt_Amt', 'Distance_Miles', 'Station_Year',
    'year_gap', 'aadt_match_type', 'VMT_Multiplier',
]


# ----------------------------------------------------------------------------
# Per-state configuration
# ----------------------------------------------------------------------------
#
# Each state has:
#   - File paths (stations, master crash CSV, ZIP polygon GeoJSON)
#   - Master CSV format quirks (skip_rows, date_format)
#   - master_cols: canonical -> source-column-name mapping. The pipeline
#     renames source -> canonical at load time so the rest of the code can
#     refer to columns by their canonical names. Use None for any canonical
#     column the source doesn't have (it will simply not be in the output).
#   - station_cols: which columns in the stations file represent which roles
#     (id, lat, lon, count, year).
#   - zip_attr_col: the ZIP-code attribute name in the GeoJSON properties.
#   - bbox: (lat_min, lat_max, lon_min, lon_max) for in-state coord validation.
#   - vmt: per-year statewide VMT (millions of miles) for the normalization
#     multiplier. Must cover every Crash_Year and every Station_Year that may
#     appear in the data; otherwise those rows get VMT_Multiplier=NULL and
#     fall into the "year not in VMT dict" diagnostic bucket.
#
# ----------------------------------------------------------------------------

from pathlib import Path

DATA_DIR = Path("master_data")
OUT_DIR_ROOT = Path("outputs")

STATE_CONFIGS = {
    'TX': {
    'stations_path':    DATA_DIR / 'old_aadt.csv',
    'master_path':      DATA_DIR / 'master_cleaned_dataset_2015-2024.csv',
    'zip_geojson_path': DATA_DIR / 'tx_texas_zip_codes_geo.min.json',
    'zip_attr_col':     'ZCTA5CE10',

    'master_skip_rows': 0,
    'date_format':      ['%m/%d/%Y', '%m/%d/%y'],

    'master_cols': {
        'Crash_ID':   'Crash_ID',
        'Crash_Date': 'Crash_Date',
        'Latitude':   'Latitude',
        'Longitude':  'Longitude',
        'ZIP_Code':   'ZIP_Code',
    },

    'station_cols': {
        'id':    'TRFC_STATN_ID',
        'lat':   'LATITUDE',
        'lon':   'LONGITUDE',
        'count': 'LATEST_AADT_QTY',
        'year':  'LATEST_AADT_YR',
    },

    'bbox': (25.8, 36.5, -106.6, -93.5),

    'vmt': {
        2015: 258300, 2016: 270700, 2017: 273200, 2018: 282200, 2019: 288400,
        2020: 260000, 2021: 285200, 2022: 291100, 2023: 301500, 2024: 307800,
        2025: 307800,
    },
},

    # Add new states below by following the same template.
    # 'AZ': {
    #     'stations_path':    'az_traffic_station_with_zip.csv',
    #     'master_path':      'az_master.csv',
    #     ...
    # },
}
