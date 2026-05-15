"""Propose a column mapping for a new state's master or stations CSV.

Run this once per new state, review the output, paste the resulting dict
into config.STATE_CONFIGS.

    python discover_mapping.py path/to/master.csv --kind master --skip-rows 4
    python discover_mapping.py path/to/stations.csv --kind stations
"""

import argparse
import polars as pl
from thefuzz import process as fuzz


MASTER_CANDIDATES = {
    'Crash_ID':   ['Crash_ID', 'IncidentID', 'CrashID', 'Case_ID',
                   'CaseNumber', 'ReportNumber', 'Accident_ID'],
    'Crash_Date': ['Crash_Date', 'IncidentDate', 'CrashDate',
                   'Accident_Date', 'Date', 'IncidentDateTime'],
    'Latitude':   ['Latitude', 'LATITUDE', 'LAT', 'Lat', 'Y'],
    'Longitude':  ['Longitude', 'LONGITUDE', 'LON', 'LONG', 'Lng', 'X'],
    'ZIP_Code':   ['ZIP_Code', 'ZIP', 'Zip', 'ZipCode', 'PostalCode',
                   'ZCTA5CE10', 'ZipCd'],
}

STATION_CANDIDATES = {
    'id':    ['TRFC_STATN_ID', 'StationID', 'Station_ID', 'LocationId',
              'Location_ID', 'STATION_ID', 'LocID'],
    'lat':   ['LATITUDE', 'Latitude', 'LAT', 'Lat', 'Y'],
    'lon':   ['LONGITUDE', 'Longitude', 'LON', 'Lng', 'LONG', 'X'],
    'count': ['LATEST_AADT_QTY', 'AADT_RPT_QTY', 'AADT', 'LastCount_Daily',
              'Count_Daily', 'AADT_Count'],
    'year':  ['LATEST_AADT_YR', 'AADT_RPT_YEAR', 'AADT_YEAR',
              'LastCountYear', 'Count_Year'],
}


def propose(actual_cols, candidates):
    out = []
    for canon, variants in candidates.items():
        matches = [c for c in actual_cols if c in variants]
        if matches:
            out.append((canon, matches[0], 100))  # Takes first match from all. So for example, if for some reason matches is ["latitude", "LAT"] for exact and both there in acutal CSV cols, take first item 
            continue
        best, best_score = None, 0
        for v in variants:
            m = fuzz.extractOne(v, actual_cols)
            if m and (m[1] > best_score):
                best, best_score = m[0], m[1]  # m[0] is the match string, and m[1] is the score 
        out.append((canon, best, best_score))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path')
    ap.add_argument('--kind', choices=['master', 'stations'], default='master')
    ap.add_argument('--skip-rows', type=int, default=0)
    args = ap.parse_args()

    cols = pl.scan_csv(args.csv_path, skip_rows=args.skip_rows, n_rows=0).collect_schema().names()
    print(f'{args.csv_path}: {len(cols)} columns')

    candidates = MASTER_CANDIDATES if args.kind == 'master' else STATION_CANDIDATES
    block = 'master_cols' if args.kind == 'master' else 'station_cols'

    print(f"\n'{block}': {{")
    for canon, actual, score in propose(cols, candidates):
        if actual is None:
            print(f"    '{canon}': None,")
        elif score == 100:
            print(f"    '{canon}': '{actual}',")
        else:
            tag = 'check' if score < 95 else 'fuzzy'
            print(f"    '{canon}': '{actual}',  # {tag}, score={score}")
    print('},')

    print('\nAll columns in file:')
    for c in cols:
        print(f'  {c}')


if __name__ == '__main__':
    main()
