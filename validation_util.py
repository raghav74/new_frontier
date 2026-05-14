import os
import time
import numpy as np
import polars as pl
from sklearn.neighbors import BallTree

# V01 — Schema Validation
# V02 — Crash Year Validation
# V03 — Latitude/Longitude Bounds
# V04 — Distance_Miles Cap
# V05 — ZIP_Code Nulls
# V06 — year_gap Consistency
# V07 — VMT_Multiplier Consistency
# V08 — Adt_Curnt_Amt Consistency
# V09 — aadt_match_type Consistency
# V14 — Fill Rate


def record(results, name, passed, msg):
    results.append({
        "validation": name,
        "passed": bool(passed),
        "msg": msg
    })

    status = "pass" if passed else "FAIL"
    print(f"[{status}] {name}: {msg}")


def discover_yearly_files(outputs_dir):
    year_files = []

    for f in sorted(os.listdir(outputs_dir)):
        if f.startswith("Cleaned_Data_") and f.endswith(".csv") and "_Final_" not in f:
            try:
                year = int(f.replace("Cleaned_Data_", "").replace(".csv", ""))
                year_files.append((year, os.path.join(outputs_dir, f)))
            except ValueError:
                pass

    year_files.sort()
    return year_files


def validate_schema(results, year, path, required_cols):
    header = pl.read_csv(path, n_rows=0).columns
    missing = [c for c in required_cols if c not in header]

    record(
        results,
        f"V01 schema {year}",
        len(missing) == 0,
        f"missing {missing}" if missing else f"{len(required_cols)}/{len(required_cols)} present, {len(header)} cols total"
    )


def validate_crash_year(results, year, path):
    df = pl.read_csv(path, columns=["Crash_Year"])
    years = df.select(pl.col("Crash_Year").drop_nulls().unique()).to_series().to_list()

    ok = len(years) == 1 and int(years[0]) == year

    record(
        results,
        f"V02 Crash_Year {year}",
        ok,
        f"distinct values = {sorted(years)}"
    )


def validate_bounds(results, year, path, tx_lat_min, tx_lat_max, tx_lon_min, tx_lon_max):
    df = pl.read_csv(path, columns=["Latitude", "Longitude"])

    output = df.select([
        (
            pl.col("Latitude").is_null() |
            pl.col("Longitude").is_null()
        ).sum().alias("null_count"),

        (
            (pl.col("Latitude") < tx_lat_min) |
            (pl.col("Latitude") > tx_lat_max) |
            (pl.col("Longitude") < tx_lon_min) |
            (pl.col("Longitude") > tx_lon_max)
        ).sum().alias("out_of_bounds")
    ])

    null_count = output["null_count"][0]
    out_of_bounds = output["out_of_bounds"][0]

    record(
        results,
        f"V03 bounds {year}",
        null_count == 0 and out_of_bounds == 0,
        f"{null_count} null, {out_of_bounds} out-of-TX"
    )


def validate_distance_cap(results, year, path, max_station_miles):
    df = pl.read_csv(path, columns=["Distance_Miles", "Adt_Curnt_Amt"])

    output = df.select([
        (
            (pl.col("Distance_Miles") > max_station_miles) &
            pl.col("Adt_Curnt_Amt").is_not_null()
        ).sum().alias("violations"),

        pl.col("Distance_Miles")
        .filter(pl.col("Adt_Curnt_Amt").is_not_null())
        .max()
        .alias("max_filled")
    ])

    violations = output["violations"][0]
    max_filled = output["max_filled"][0]

    record(
        results,
        f"V04 dist cap {year}",
        violations == 0,
        f"{violations} violations; max Distance_Miles among filled = {max_filled:.4f}"
    )


def validate_zip(results, year, path):
    df = pl.read_csv(path, columns=["ZIP_Code"])
    nulls = df.select(pl.col("ZIP_Code").is_null().sum()).item()

    record(
        results,
        f"V05 ZIP {year}",
        nulls == 0,
        f"{nulls} null ZIP rows"
    )


def validate_year_gap(results, year, path):
    df = pl.read_csv(path, columns=["Crash_Year", "Station_Year", "year_gap"])
    df = df.drop_nulls(["Station_Year", "year_gap"])

    mismatches = df.select(
        (
            (
                (pl.col("Crash_Year").cast(pl.Int64) - pl.col("Station_Year").cast(pl.Int64)).abs()
                != pl.col("year_gap").cast(pl.Int64)
            )
        ).sum()
    ).item()

    record(
        results,
        f"V06 year_gap {year}",
        mismatches == 0,
        f"{mismatches} mismatches"
    )


def validate_vmt(results, year, path, tx_vmt_millions, atol_vmt):
    df = pl.read_csv(path, columns=["Crash_Year", "Station_Year", "VMT_Multiplier"])
    df = df.drop_nulls(["VMT_Multiplier", "Station_Year"])

    df = df.with_columns([
        pl.col("Crash_Year").cast(pl.Int64).replace(tx_vmt_millions).alias("VMT_Crash"),
        pl.col("Station_Year").cast(pl.Int64).replace(tx_vmt_millions).alias("VMT_Station"),
    ])

    df = df.with_columns(
        (pl.col("VMT_Crash") / pl.col("VMT_Station")).alias("expected")
    )

    mismatches = df.select(
        (
            ((pl.col("expected") - pl.col("VMT_Multiplier")).abs() / pl.col("expected").abs())
            > atol_vmt
        ).sum()
    ).item()

    record(
        results,
        f"V07 VMT {year}",
        mismatches == 0,
        f"{mismatches} mismatches (rtol > {atol_vmt:.0e})"
    )


def validate_aadt(results, year, path):
    df = pl.read_csv(path, columns=["Station_Count", "VMT_Multiplier", "Adt_Curnt_Amt"])
    df = df.drop_nulls(["Adt_Curnt_Amt"])

    df = df.with_columns(
        (pl.col("Station_Count") * pl.col("VMT_Multiplier")).alias("expected")
    )

    mismatches = df.select(
        (
            ((pl.col("expected") - pl.col("Adt_Curnt_Amt")).abs() / pl.col("Adt_Curnt_Amt").abs())
            > 1e-6
        ).sum()
    ).item()

    record(
        results,
        f"V08 AADT {year}",
        mismatches == 0,
        f"{mismatches} mismatches (rtol > 1e-6)"
    )


def validate_match_type(results, year, path):
    df = pl.read_csv(path, columns=["Adt_Curnt_Amt", "aadt_match_type"])

    output = df.select([
        (
            pl.col("Adt_Curnt_Amt").is_not_null() &
            (pl.col("aadt_match_type") != "NEAREST_STATION_VMT_NORM")
        ).sum().alias("bad_filled"),

        (
            pl.col("Adt_Curnt_Amt").is_null() &
            (pl.col("aadt_match_type") != "MISSING")
        ).sum().alias("bad_missing")
    ])

    bad_filled = output["bad_filled"][0]
    bad_missing = output["bad_missing"][0]

    record(
        results,
        f"V09 match_type {year}",
        bad_filled == 0 and bad_missing == 0,
        f"{bad_filled} filled+wrong-label, {bad_missing} missing+wrong-label"
    )


def validate_fill_rate(results, year, path):
    df = pl.read_csv(path, columns=["Adt_Curnt_Amt"])

    total = df.height
    filled = df.select(pl.col("Adt_Curnt_Amt").is_not_null().sum()).item()
    pct = 100.0 * filled / total if total else 0.0

    record(
        results,
        f"V14 fill rate {year}",
        True,
        f"{filled:,}/{total:,} ({pct:.2f}%)"
    )

    return total, filled


def save_validation_results(results, outputs_dir):
    summary = pl.DataFrame(results).with_columns(
        pl.when(pl.col("passed"))
        .then(pl.lit("pass"))
        .otherwise(pl.lit("FAIL"))
        .alias("status")
    ).select([
        "status",
        "validation",
        "passed",
        "msg"
    ])

    out_path = os.path.join(outputs_dir, "validation_results.csv")
    summary.write_csv(out_path)

    n_pass = summary.select(pl.col("passed").sum()).item()
    n_fail = summary.height - n_pass

    print(f"{n_pass} pass, {n_fail} fail (of {summary.height})")
    print(f"saved {out_path}")

    return summary