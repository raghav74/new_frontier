"""Helper functions for the AADT-fill pipeline.

Every loader and processor takes a `state_cfg` dict (one entry from
config.STATE_CONFIGS). Source-specific column names are read from that
config and renamed to canonical names at load time so the per-year
processing code can refer to columns by canonical name only.
"""

import os
import time

import numpy as np
import polars as pl
import geopandas as gpd
from shapely import points as shapely_points
from scipy.spatial import cKDTree

import config


# ----------------------------------------------------------------------------
# Coordinate helpers
# ----------------------------------------------------------------------------

def latlon_to_unit_xyz(lat_deg, lon_deg):
    """Map (lat°, lon°) to (x, y, z) on the unit sphere.

    Used so we can run a Euclidean kd-tree query whose nearest neighbour is
    also the true great-circle nearest neighbour (chord distance is monotonic
    with arc distance on a sphere).
    """
    lat = np.radians(lat_deg.astype(np.float64, copy=False))
    lon = np.radians(lon_deg.astype(np.float64, copy=False))
    cos_lat = np.cos(lat)
    return np.column_stack([
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ])


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_stations(state_cfg):
    """Load and clean the AADT stations table.

    Renames the state-specific column names to canonical internal names:
        <id_col>    -> station_id
        <lat_col>   -> station_lat
        <lon_col>   -> station_lon
        <count_col> -> Station_Count
        <year_col>  -> Station_Year
    """
    s_cols = state_cfg['station_cols']

    stations = pl.read_csv(state_cfg['stations_path'], infer_schema_length=10_000)
    stations = (
        stations
        .rename({c: c.strip() for c in stations.columns})
        .with_columns([
            pl.col(s_cols['lat']).cast(pl.Float64, strict=False),
            pl.col(s_cols['lon']).cast(pl.Float64, strict=False),
            pl.col(s_cols['count']).cast(pl.Float64, strict=False),
            pl.col(s_cols['year']).cast(pl.Int64, strict=False),
            pl.col(s_cols['id']).cast(pl.Utf8).str.strip_chars(),
        ])
        .filter(
            pl.col(s_cols['lat']).is_not_null()
            & pl.col(s_cols['lon']).is_not_null()
            & pl.col(s_cols['count']).is_not_null()
            & (pl.col(s_cols['count']) > 0)
        )
        # Latest year per station id (no-op for current TX file shape, future-proof)
        .sort(s_cols['year'])
        .unique(subset=[s_cols['id']], keep='last')
        .rename({
            s_cols['id']:    'station_id',
            s_cols['lat']:   'station_lat',
            s_cols['lon']:   'station_lon',
            s_cols['count']: 'Station_Count',
            s_cols['year']:  'Station_Year',
        })
    )
    return stations


def build_station_index(stations):
    """Build a cKDTree on stations' unit-sphere XYZ coordinates + the lookup
    arrays needed to assemble matched station columns back onto crash rows.

    Assumes the stations DataFrame has been renamed to canonical names by
    load_stations.
    """
    station_xyz = latlon_to_unit_xyz(
        stations['station_lat'].to_numpy(),
        stations['station_lon'].to_numpy(),
    )
    tree = cKDTree(station_xyz, leafsize=64)

    # Chord-distance equivalent of MAX_STATION_MILES; used as cKDTree's
    # distance_upper_bound so far-away points return inf immediately.
    max_angle = config.MAX_STATION_MILES / config.EARTH_RADIUS_MILES
    max_chord = 2.0 * np.sin(max_angle / 2.0)

    return {
        'tree':      tree,
        'lat':       stations['station_lat'].to_numpy(),
        'lon':       stations['station_lon'].to_numpy(),
        'id':        stations['station_id'].to_numpy(),
        'count':     stations['Station_Count'].to_numpy(),
        'year':      stations['Station_Year'].to_numpy(),
        'n':         len(station_xyz),
        'max_chord': max_chord,
    }


def load_master(state_cfg):
    """Lazy-scan the master crash CSV, rename source columns to canonical,
    dedupe by Crash_ID, derive Crash_Year, cast lat/lon, apply bounding box,
    then collect via streaming."""
    m_cols = state_cfg['master_cols']
    bbox = state_cfg['bbox']
    lat_min, lat_max, lon_min, lon_max = bbox

    # Build source -> canonical rename map. Skip canonicals the state's source
    # doesn't have (mapped to None), and skip no-op renames where source == canonical.
    rename_map = {
        src: canon
        for canon, src in m_cols.items()
        if src is not None and src != canon
    }

    # date_format may be a single format string or a list of formats to try.
    # Coalesce returns the first non-null parse, so mixed formats in the same
    # column (e.g. '06/07/2023' and '06/07/23') both resolve correctly.
    date_fmts = state_cfg['date_format']
    if isinstance(date_fmts, str):
        date_fmts = [date_fmts]
    crash_year_expr = pl.coalesce([
        pl.col('Crash_Date').str.to_date(fmt, strict=False)
        for fmt in date_fmts
    ]).dt.year().alias('Crash_Year')

    master_lf = (
        pl.scan_csv(
            state_cfg['master_path'],
            skip_rows=state_cfg['master_skip_rows'],
            ignore_errors=True,
            low_memory=True,
        ).select([
            "Crash_ID",
            "Crash_Date",
            "Latitude",
            "Longitude",
            "ZIP_Code",
        ])
        .rename(rename_map, strict=False)
        .unique(subset=['Crash_ID'])
        .with_columns(
            crash_year_expr,
            pl.col('Latitude').cast(pl.Float64, strict=False),
            pl.col('Longitude').cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col('Latitude').is_not_null()
            & pl.col('Longitude').is_not_null()
            & pl.col('Latitude').is_between(lat_min, lat_max)
            & pl.col('Longitude').is_between(lon_min, lon_max)
        )
    )
    return master_lf.collect()


def load_zip_polygons(state_cfg):
    """Load the state's ZIP polygons as a geopandas GeoDataFrame (EPSG:4326)."""
    zip_attr = state_cfg['zip_attr_col']
    zip_gdf = gpd.read_file(state_cfg['zip_geojson_path'])[[zip_attr, 'geometry']]
    zip_gdf = zip_gdf.set_crs('EPSG:4326', allow_override=True)
    return zip_gdf


# ----------------------------------------------------------------------------
# Per-year processing
# ----------------------------------------------------------------------------

def pop_year_partition(year_partitions, crash_year):
    """Pop a year's slice from the partition dict.

    polars 1.x keys partition_by(as_dict=True) by tuples; older versions by
    scalars. Handle both for cross-version compatibility.
    """
    key_tuple = (crash_year,)
    if key_tuple in year_partitions:
        return year_partitions.pop(key_tuple)
    if crash_year in year_partitions:
        return year_partitions.pop(crash_year)
    return None


def process_year(df, crash_year, station_idx, zip_gdf, out_dir, state_cfg):
    """Full per-year pipeline:
        1. Drop the source ZIP_Code (will be replaced by the sjoin result).
        2. Nearest-station match via cKDTree + chord-to-arc conversion.
        3. Compute year_gap, VMT_Crash, VMT_Station, VMT_Multiplier, Adt_Curnt_Amt.
        4. ZIP spatial join (polars -> pandas -> geopandas -> polars hop).
        5. Drop crashes with no ZIP match, write the year's CSV.

    Returns the resulting DataFrame.
    """
    zip_attr = state_cfg['zip_attr_col']
    vmt = state_cfg['vmt']

    print(f'  rows after bounds filter     : {df.height:,}')

    if 'ZIP_Code' in df.columns:
        df = df.drop('ZIP_Code')

    # --- (2) Nearest station via cKDTree on unit-sphere XYZ ---
    t = time.perf_counter()

    crash_xyz = latlon_to_unit_xyz(
        df['Latitude'].to_numpy(),
        df['Longitude'].to_numpy(),
    )

    chord_dist, idx = station_idx['tree'].query(
        crash_xyz,
        k=1,
        distance_upper_bound=station_idx['max_chord'],
        workers=-1,
    )

    valid = np.isfinite(chord_dist) & (idx < station_idx['n'])
    safe_idx = np.where(valid, idx, 0)   # avoid out-of-bounds indexing for unmatched rows

    distances_miles = np.full(df.height, np.nan, dtype=np.float64)
    distances_miles[valid] = (
        2.0 * np.arcsin(np.minimum(chord_dist[valid] / 2.0, 1.0))
        * config.EARTH_RADIUS_MILES
    )

    df = df.with_columns([
        pl.Series('__valid_station',           valid),
        pl.Series('Nearest_Station_LocationId', station_idx['id'][safe_idx]),
        pl.Series('Station_Lat',                station_idx['lat'][safe_idx]),
        pl.Series('Station_Lon',                station_idx['lon'][safe_idx]),
        pl.Series('Station_Count',              station_idx['count'][safe_idx]).cast(pl.Float64),
        pl.Series('Station_Year',               station_idx['year'][safe_idx]).cast(pl.Int64),
        pl.Series('Distance_Miles',             distances_miles),
    ])

    valid_station = pl.col('__valid_station')
    df = (
        df.with_columns([
            pl.when(valid_station).then(pl.col('Nearest_Station_LocationId')).otherwise(None).alias('Nearest_Station_LocationId'),
            pl.when(valid_station).then(pl.col('Station_Lat')).otherwise(None).alias('Station_Lat'),
            pl.when(valid_station).then(pl.col('Station_Lon')).otherwise(None).alias('Station_Lon'),
            pl.when(valid_station).then(pl.col('Station_Count')).otherwise(None).alias('Station_Count'),
            pl.when(valid_station).then(pl.col('Station_Year')).otherwise(None).alias('Station_Year'),
            pl.when(valid_station).then(pl.col('Distance_Miles')).otherwise(None).alias('Distance_Miles'),
        ])
        .drop('__valid_station')
    )

    del crash_xyz, chord_dist, idx, valid, safe_idx, distances_miles
    print(f'  cKDTree nearest-station block: {time.perf_counter() - t:.2f}s')

    # --- (3) year_gap + VMT + AADT ---
    t = time.perf_counter()
    df = df.with_columns(
        (pl.col('Crash_Year') - pl.col('Station_Year')).abs().alias('year_gap')
    )
    df = df.with_columns([
        pl.col('Crash_Year').replace_strict(vmt, default=None, return_dtype=pl.Float64).alias('VMT_Crash'),
        pl.col('Station_Year').replace_strict(vmt, default=None, return_dtype=pl.Float64).alias('VMT_Station'),
    ])
    df = df.with_columns(
        (pl.col('VMT_Crash') / pl.col('VMT_Station')).alias('VMT_Multiplier')
    )
    df = df.with_columns([
        (pl.col('Station_Count') * pl.col('VMT_Multiplier')).alias('Adt_Curnt_Amt'),
        pl.lit(crash_year).cast(pl.Int64).alias('Adt_Curnt_Year'),
        pl.when(pl.col('Station_Count').is_not_null() & pl.col('VMT_Multiplier').is_not_null())
          .then(pl.lit('NEAREST_STATION_VMT_NORM'))
          .otherwise(pl.lit('MISSING'))
          .alias('aadt_match_type'),
    ])
    print(f'  VMT + AADT block: {time.perf_counter() - t:.2f}s')

    # --- (4) ZIP spatial join via shapely + geopandas ---
    t = time.perf_counter()
    pdf = df.to_pandas()
    geoms = shapely_points(pdf['Longitude'].to_numpy(), pdf['Latitude'].to_numpy())
    gdf_crash = gpd.GeoDataFrame(pdf, geometry=geoms, crs='EPSG:4326')
    joined = gpd.sjoin(gdf_crash, zip_gdf, how='left', predicate='within')
    joined = joined.drop(columns=['geometry', 'index_right'])
    df = pl.from_pandas(joined)
    del pdf, geoms, gdf_crash, joined
    print(f'  ZIP join block: {time.perf_counter() - t:.2f}s')

    # --- (5) Drop rows missing ZIP, rename, write ---
    before_zip = df.height
    df = df.filter(pl.col(zip_attr).is_not_null())
    df = df.rename({zip_attr: 'ZIP_Code'})
    df = df.with_columns(pl.col('ZIP_Code').cast(pl.Int64, strict=False))
    print(f'  dropped missing ZIP   : {before_zip - df.height:,}')

    t = time.perf_counter()
    out_path = os.path.join(out_dir, f'Cleaned_Data_{crash_year}.csv')
    df.write_csv(out_path)
    filled = df.filter(pl.col('Adt_Curnt_Amt').is_not_null()).height
    pct = 100.0 * filled / df.height if df.height else 0.0
    print(f'  filled / total        : {filled:,}/{df.height:,} ({pct:.2f}%)')
    print(f'  CSV write block: {time.perf_counter() - t:.2f}s')

    return df


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def summarize_missing_aadt(df_all):
    """Print the 4-bucket missing-AADT breakdown across all years."""
    if not df_all.height:
        print('No data to summarize')
        return

    total = df_all.height
    missing = df_all.filter(pl.col('Adt_Curnt_Amt').is_null()).height

    no_coords = df_all.filter(
        pl.col('Adt_Curnt_Amt').is_null()
        & (pl.col('Latitude').is_null() | pl.col('Longitude').is_null())
    ).height
    no_station = df_all.filter(
        pl.col('Adt_Curnt_Amt').is_null()
        & pl.col('Nearest_Station_LocationId').is_null()
        & pl.col('Latitude').is_not_null()
        & pl.col('Longitude').is_not_null()
    ).height
    no_station_year = df_all.filter(
        pl.col('Adt_Curnt_Amt').is_null()
        & pl.col('Nearest_Station_LocationId').is_not_null()
        & pl.col('Station_Year').is_null()
    ).height
    no_vmt = df_all.filter(
        pl.col('Adt_Curnt_Amt').is_null()
        & pl.col('Nearest_Station_LocationId').is_not_null()
        & pl.col('Station_Year').is_not_null()
        & pl.col('VMT_Multiplier').is_null()
    ).height

    print(f'Total rows  : {total:,}')
    print(f'Missing AADT: {missing:,} ({missing / total * 100:.2f}%)')
    print(f'  no coords             : {no_coords:,}')
    print(f'  no station within 1mi : {no_station:,}')
    print(f'  station year missing  : {no_station_year:,}')
    print(f'  year not in VMT dict  : {no_vmt:,}')


def summarize_per_year(df_all):
    """Print per-year (rows, filled, unique_stations, pct) summary."""
    if not df_all.height:
        return
    summary = (
        df_all
        .group_by('Crash_Year')
        .agg([
            pl.len().alias('rows'),
            pl.col('Adt_Curnt_Amt').is_not_null().sum().alias('filled'),
            pl.col('Nearest_Station_LocationId').n_unique().alias('unique_stations'),
        ])
        .with_columns((pl.col('filled') / pl.col('rows') * 100).round(2).alias('pct'))
        .sort('Crash_Year')
    )
    print(summary)


def generate_qa_plots(df_all, out_dir):
    """Five QA plots, saved as PNGs in out_dir."""
    import matplotlib.pyplot as plt

    if not df_all.height:
        print('No data to plot')
        return

    filled  = df_all.filter(pl.col('Adt_Curnt_Amt').is_not_null()).height
    missing = df_all.height - filled
    total   = df_all.height

    # Plot 1: filled vs missing
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['FILLED', 'MISSING'], [filled, missing], color=['steelblue', 'salmon'])
    plt.title('AADT Coverage')
    plt.ylabel('Crash rows')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        v = int(b.get_height())
        pct = v / total * 100
        plt.text(b.get_x() + b.get_width() / 2, v, f'{v:,}\n({pct:.1f}%)', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot1_filled_vs_missing.png'), dpi=150)
    plt.close()

    # Plot 2: year_gap histogram
    yg = df_all.filter(pl.col('year_gap').is_not_null())['year_gap'].to_numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(yg, bins=range(0, int(yg.max()) + 2), color='steelblue', edgecolor='white')
    plt.title('Year Gap Distribution')
    plt.xlabel('year_gap')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot2_year_gap.png'), dpi=150)
    plt.close()

    # Plot 3: distance histogram
    dist = df_all.filter(pl.col('Distance_Miles').is_not_null())['Distance_Miles'].to_numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(dist, bins=50, color='steelblue', edgecolor='white')
    plt.title('Nearest Station Distance (miles)')
    plt.xlabel('Distance_Miles')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot3_distance_hist.png'), dpi=150)
    plt.close()

    # Plot 4: AADT distribution
    aadt = df_all.filter(pl.col('Adt_Curnt_Amt').is_not_null())['Adt_Curnt_Amt'].to_numpy()
    plt.figure(figsize=(10, 4))
    plt.hist(aadt, bins=60, color='steelblue', edgecolor='white')
    plt.title('Normalized AADT Distribution')
    plt.xlabel('Adt_Curnt_Amt')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot4_aadt_dist.png'), dpi=150)
    plt.close()

    # Plot 5: distance bin chart (0.1 mile bins, within 1 mile)
    df_1mi = df_all.filter(
        pl.col('Adt_Curnt_Amt').is_not_null()
        & (pl.col('Distance_Miles') <= 1.0)
        & (pl.col('Distance_Miles') >= 0.0)
    )
    bin_edges = np.round(np.arange(0.0, 1.0 + 0.1, 0.1), 2)
    dist_1mi = df_1mi['Distance_Miles'].to_numpy()
    counts, _ = np.histogram(dist_1mi, bins=bin_edges)
    labels = [f'({bin_edges[i]:.1f},{bin_edges[i+1]:.1f}]' for i in range(len(bin_edges) - 1)]
    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, counts, color='steelblue', edgecolor='white')
    plt.title('Filled rows by distance bin (0.1mi, <=1mi)')
    plt.xlabel('Distance bin (miles)')
    plt.ylabel('Filled crash count')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    for b in bars:
        v = int(b.get_height())
        plt.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot5_distance_bins.png'), dpi=150)
    plt.close()


def check_required_cols(all_outputs):
    """Verify every year's DataFrame has the canonical required output columns."""
    for df in all_outputs:
        yr = df['Crash_Year'][0]
        cols = df.columns
        missing = [c for c in config.REQUIRED_COLS if c not in cols]
        status = 'OK' if not missing else f'MISSING {missing}'
        print(f'{yr}: {status}  ({df.height:,} rows, {len(cols)} cols)')


def print_timings(timings):
    """Print a per-year timings table and total wall time."""
    if not timings:
        return
    tdf = pl.DataFrame(timings, schema=['year', 'rows', 'seconds'], orient='row')
    tdf = tdf.with_columns((pl.col('rows') / pl.col('seconds')).round(0).alias('rows_per_sec'))
    print(tdf)
    total = sum(t for _, _, t in timings)
    print(f'Total wall time: {total:.1f}s')
