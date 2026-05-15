"""AADT-fill pipeline entry point.

Usage:
    cd new_code/
    python aadt_fill_main.py --state TX
    python aadt_fill_main.py --state AZ

Run inside the `mlops_project` conda env.

Outputs land in OUT_DIR_ROOT/<state>/:
    Cleaned_Data_<YEAR>.csv      (one per year that has data)
    plot1_filled_vs_missing.png
    plot2_year_gap.png
    plot3_distance_hist.png
    plot4_aadt_dist.png
    plot5_distance_bins.png
"""

import argparse
import gc
import os
import time

import polars as pl

import config
from aadt_fill_utils import (
    build_station_index,
    check_required_cols,
    generate_qa_plots,
    load_master,
    load_stations,
    load_zip_polygons,
    pop_year_partition,
    print_timings,
    process_year,
    summarize_missing_aadt,
    summarize_per_year,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='AADT-fill pipeline for a state crash dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--state',
        required=True,
        choices=sorted(config.STATE_CONFIGS.keys()),
        help='State code to process (must be defined in config.STATE_CONFIGS).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    state = args.state
    state_cfg = config.STATE_CONFIGS[state]

    out_dir = os.path.join(config.OUT_DIR_ROOT, state)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n== Pipeline run: state={state} ==')
    print(f'   out_dir={out_dir}\n')

    # Stations + cKDTree
    print('Loading stations...')
    stations = load_stations(state_cfg)
    print(f'  stations kept: {stations.height:,}')

    print('Building cKDTree...')
    station_idx = build_station_index(stations)
    print(f'  cKDTree built on {station_idx["n"]:,} stations')
    del stations    # not needed once the index is built

    # Master crash table
    print('Loading master crash file...')
    master_df = load_master(state_cfg)
    print(f'  rows after dedupe + bounds: {master_df.height:,}')

    # ZIP polygons
    print('Loading ZIP polygons...')
    zip_gdf = load_zip_polygons(state_cfg)
    print(f'  zip polygons: {len(zip_gdf):,}')

    # Partition by year + free the master
    print('Partitioning master_df by Crash_Year...')
    year_partitions = master_df.partition_by('Crash_Year', as_dict=True)
    print(f'  partitions: {len(year_partitions)}')
    del master_df
    gc.collect()

    # Per-year loop
    all_outputs = []
    timings = []

    for crash_year in config.CRASH_YEARS:
        t_year_start = time.perf_counter()
        print(f'\n== {crash_year} ==')

        df = pop_year_partition(year_partitions, crash_year)
        if df is None:
            print(f'  no rows for {crash_year}, skipping')
            continue

        df = process_year(df, crash_year, station_idx, zip_gdf, out_dir, state_cfg)

        elapsed = time.perf_counter() - t_year_start
        timings.append((crash_year, df.height, elapsed))
        print(f'  elapsed               : {elapsed:.1f}s')

        all_outputs.append(df)
        gc.collect()

    print(f'\nTotal years processed: {len(all_outputs)}')

    # Combined summaries
    if all_outputs:
        df_all = pl.concat(all_outputs, how='vertical_relaxed')
        print(f'\nAll years combined: {df_all.height:,} rows')

        print('\n=== Missing AADT breakdown ===')
        summarize_missing_aadt(df_all)

        print('\n=== Per-year fill rate ===')
        summarize_per_year(df_all)

        print('\n=== Generating QA plots ===')
        generate_qa_plots(df_all, out_dir)
        print(f'  plots saved to {out_dir}')

    print('\n=== Required column check ===')
    check_required_cols(all_outputs)

    print('\n=== Timings ===')
    print_timings(timings)


if __name__ == '__main__':
    main()
