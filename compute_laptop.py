#!/usr/bin/env python
"""compute_laptop.py

Compute JJA (Jun-Aug) marine heatwave threshold and SST climatology
for the South China Sea (5N-25N, 105E-125E) over 1991-2020 baseline.

Outputs:
  ERA5/Climatology/0601.nc ... 0831.nc (92 daily files)
    Each file contains:
      - threshold  (lat, lon) : 90th percentile of +-5-day window SST
      - climatology (lat, lon) : Mean of +-5-day window SST
"""

import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import time

# --- Configuration ---
BASE_DIR = Path("/home/amazcuter/code/detech_waveheat")
DIR_91_20 = BASE_DIR / "OISST_91_20"

LAT_SLICE = slice(380, 460)  # 80 points: 5.125N to 24.875N
LON_SLICE = slice(420, 500)  # 80 points: 105.125E to 124.875E

START_YEAR = 1991
END_YEAR = 2020  # inclusive, 30 years
N_YEARS = END_YEAR - START_YEAR + 1

WINDOW_HALF = 5
WINDOW_SIZE = 2 * WINDOW_HALF + 1  # 11
N_OUTPUT_DAYS = 92  # Jun 1 to Aug 31


def build_date_filepath_map():
    """Glob all .nc files in the data directory, return dict: YYYYMMDD -> filepath."""
    date_to_path = {}
    for fp in sorted(DIR_91_20.glob("*.nc")):
        # Filenames: oisst-avhrr-v02r01.YYYYMMDD.nc or avhrr-only-v2.YYYYMMDD.nc
        parts = fp.name.rsplit(".", 2)
        if len(parts) == 3:
            date_str = parts[1]
            if len(date_str) == 8 and date_str.isdigit():
                date_to_path[date_str] = str(fp)
    return date_to_path


def get_date_strings(year):
    """Return list of YYYYMMDD strings for May 27 through Sep 5 of the given year."""
    start = datetime(year, 5, 27)
    end = datetime(year, 9, 5)
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def main():
    print("Building file index ...")
    date_to_path = build_date_filepath_map()
    print(f"  Indexed {len(date_to_path)} files")

    # Pre-compute the date strings we need
    all_date_strs = []
    for year in range(START_YEAR, END_YEAR + 1):
        all_date_strs.extend(get_date_strings(year))

    n_days_per_year = len(get_date_strings(START_YEAR))  # 102
    print(f"  {N_YEARS} years x {n_days_per_year} days/year = {len(all_date_strs)} files needed")

    # Verify all files exist
    missing = [d for d in all_date_strs if d not in date_to_path]
    if missing:
        print(f"WARNING: {len(missing)} missing dates, first 10: {missing[:10]}")
    else:
        print("  All files present")

    # Allocate: (year, day, lat, lon)
    shape = (N_YEARS, n_days_per_year, 80, 80)
    size_gb = N_YEARS * n_days_per_year * 80 * 80 * 8 / 1024**3
    print(f"Allocating array {shape} float64 ({size_gb:.2f} GB) ...")
    data = np.full(shape, np.nan, dtype=np.float64)

    # Read data year by year
    t_start = time.time()
    for yr_idx, year in enumerate(range(START_YEAR, END_YEAR + 1)):
        date_strs = get_date_strings(year)
        for day_idx, date_str in enumerate(date_strs):
            fp = date_to_path.get(date_str)
            if fp is None:
                continue
            with xr.open_dataset(fp) as ds:
                sst = (
                    ds["sst"]
                    .values[0, 0, LAT_SLICE, LON_SLICE]
                    .astype(np.float64)
                )
            data[yr_idx, day_idx] = sst
        elapsed = time.time() - t_start
        print(f"  Year {year} done ({yr_idx+1}/{N_YEARS}, {elapsed:.1f}s elapsed)")

    # Extract lat/lon from a sample file
    sample_fp = date_to_path.get("19910601") or next(iter(date_to_path.values()))
    with xr.open_dataset(sample_fp) as ds:
        lats = ds["lat"].values[LAT_SLICE].copy()
        lons = ds["lon"].values[LON_SLICE].copy()

    # Compute threshold (90th percentile) and climatology (mean) for each output day
    # Output days: Jun 1 (day_idx=5) to Aug 31 (day_idx=96)
    print("Computing threshold and climatology ...")
    threshold = np.full((N_OUTPUT_DAYS, 80, 80), np.nan, dtype=np.float64)
    climatology = np.full((N_OUTPUT_DAYS, 80, 80), np.nan, dtype=np.float64)

    for out_idx, center_idx in enumerate(range(5, 97)):
        # day 5=Jun1, day 96=Aug31
        w0 = center_idx - WINDOW_HALF
        w1 = center_idx + WINDOW_HALF + 1
        window = data[:, w0:w1, :, :]  # (30, 11, 80, 80)
        flat = window.reshape(N_YEARS * WINDOW_SIZE, 80, 80)  # (330, 80, 80)

        threshold[out_idx] = np.nanpercentile(flat, 90, axis=0).astype(np.float64)
        climatology[out_idx] = np.nanmean(flat, axis=0).astype(np.float64)
        if (out_idx + 1) % 10 == 0:
            print(f"  Day {out_idx+1}/{N_OUTPUT_DAYS} done")

    # Create output directory: ERA5/Climatology/
    out_dir = BASE_DIR / "ERA5" / "Climatology"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save 92 daily files: 0601.nc, 0602.nc, ..., 0831.nc
    output_dates = [datetime(2000, 6, 1) + timedelta(days=d) for d in range(N_OUTPUT_DAYS)]
    for out_idx, dt in enumerate(output_dates):
        fname = dt.strftime("%m%d.nc")  # e.g. "0601.nc"
        ds = xr.Dataset(
            {
                "threshold": (["lat", "lon"], threshold[out_idx]),
                "climatology": (["lat", "lon"], climatology[out_idx]),
            },
            coords={"lat": lats, "lon": lons},
        )
        ds["threshold"].attrs["units"] = "degC"
        ds["threshold"].attrs["long_name"] = "SST 90th percentile (marine heatwave threshold)"
        ds["climatology"].attrs["units"] = "degC"
        ds["climatology"].attrs["long_name"] = "Mean SST climatology"
        ds.attrs["baseline_period"] = "1991-2020"
        ds.to_netcdf(out_dir / fname)
    print(f"Saved {N_OUTPUT_DAYS} daily files to {out_dir}/")

    # Quick validation
    print(f"\nValidation:")
    print(f"  threshold shape: {threshold.shape}")
    print(f"  climatology shape: {climatology.shape}")
    print(f"  threshold range: [{np.nanmin(threshold):.2f}, {np.nanmax(threshold):.2f}] C")
    print(f"  climatology range: [{np.nanmin(climatology):.2f}, {np.nanmax(climatology):.2f}] C")
    print(f"  NaN fraction (threshold): {np.isnan(threshold).mean()*100:.1f}%")
    print(f"  threshold > climatology (mean): {(np.nanmean(threshold) > np.nanmean(climatology))}")

    print("Done!")


if __name__ == "__main__":
    main()
