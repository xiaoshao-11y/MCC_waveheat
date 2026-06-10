#!/usr/bin/env python
"""compute_server.py — ProcessPool I/O + PyTorch ROCm multi-GPU compute (backup).

Same algorithm as compute_server_mpi.py but uses ProcessPoolExecutor for I/O
instead of MPI. Single-node, 4×DCU Z200, float16 throughout.

Usage:
  python compute_server.py
"""

import numpy as np
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict
from netCDF4 import Dataset
import time
import gc

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# ── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = Path("/public/home/achwjznh4b/Newdata/")
OUT_DIR = Path(__file__).resolve().parent / "ERA5" / "Climatology"

START_YEAR, END_YEAR = 1991, 2020
N_YEARS = END_YEAR - START_YEAR + 1                     # 30

WINDOW_HALF = int(os.environ.get("WINDOW_HALF", "5"))
WINDOW_SIZE = 2 * WINDOW_HALF + 1                       # 11

N_LAT, N_LON = 721, 1440
N_DAYS_PER_YEAR = 102                                   # May 27 – Sep 5
N_OUTPUT_DAYS = N_DAYS_PER_YEAR - WINDOW_SIZE + 1       # 92 (Jun 1 – Aug 31)

DATE_START = (5, 27)
DATE_END = (9, 5)

MAX_READ_WORKERS = int(os.environ.get("MAX_READ_WORKERS", "16"))
N_BANDS = int(os.environ.get("N_BANDS", "12"))
SUB_BATCH_DAYS = int(os.environ.get("SUB_BATCH_DAYS", "92"))
IO_THREADS = int(os.environ.get("IO_THREADS", "4"))


# ── Calendar helpers ───────────────────────────────────────────────────────

def get_date_strings_for_year(year):
    """Return YYYYMMDD strings for May 27 – Sep 5 of a given year."""
    start = datetime(year, *DATE_START)
    end = datetime(year, *DATE_END)
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def build_date_filepath_map():
    """Glob DATA_DIR once → {YYYYMMDD: path} dict."""
    return {fp.name: str(fp) for fp in DATA_DIR.iterdir()
            if len(fp.name) == 8 and fp.name.isdigit()}


# ── Raw byte I/O ───────────────────────────────────────────────────────────

def _probe_raw_read(date_to_path):
    """Check if source files support raw seek+read (contiguous, uncompressed).

    Returns dict(offset, dtype, shape, nbytes) or None.
    """
    if not _HAS_H5PY:
        print("    Raw I/O:         DISABLED (h5py not available)")
        return None

    test_paths = [date_to_path[ds] for ds in sorted(date_to_path)[:3]
                  if date_to_path.get(ds)]
    offsets, dtype, shape = [], None, None
    for fp in test_paths:
        try:
            with h5py.File(fp, "r") as f:
                ds = f["data"]
                if ds.chunks is not None:
                    print(f"    Raw I/O:         DISABLED (chunked: {ds.chunks})")
                    return None
                if ds.compression:
                    print(f"    Raw I/O:         DISABLED (compressed)")
                    return None
                off = ds.id.get_offset()
                if off is None:
                    print("    Raw I/O:         DISABLED (no byte offset)")
                    return None
                offsets.append(off)
                dtype, shape = ds.dtype, ds.shape
        except Exception as e:
            print(f"    Raw I/O:         DISABLED (probe error: {e})")
            return None

    if len(set(offsets)) != 1:
        print(f"    Raw I/O:         DISABLED (inconsistent offsets)")
        return None

    nbytes = int(np.prod(shape)) * dtype.itemsize

    try:
        with h5py.File(test_paths[0], "r") as f:
            ref = f["data"][:]
        with open(test_paths[0], "rb") as f:
            f.seek(offsets[0])
            raw = f.read(nbytes)
        for dt in (dtype, dtype.newbyteorder()):
            test = np.frombuffer(raw, dtype=dt).reshape(shape)
            if np.array_equal(ref, test, equal_nan=True):
                print(f"    Raw I/O:         ENABLED (offset={offsets[0]}, "
                      f"{nbytes} bytes, direct seek+read)")
                return {"offset": offsets[0], "dtype": dt, "shape": shape, "nbytes": nbytes}
        print("    Raw I/O:         DISABLED (validation mismatch)")
    except Exception as e:
        print(f"    Raw I/O:         DISABLED (validation error: {e})")
    return None


def raw_read_sst(fp, raw_info):
    """Read one SST file via raw seek+read."""
    with open(fp, "rb") as f:
        f.seek(raw_info["offset"])
        raw = f.read(raw_info["nbytes"])
    return np.frombuffer(raw, dtype=raw_info["dtype"]).reshape(raw_info["shape"])


# ── Source format detection ────────────────────────────────────────────────

def detect_source_format(date_to_path):
    """Probe first available file for SST var name, lat/lon, and raw I/O info."""
    sample = next((date_to_path[ds] for ds in sorted(date_to_path)
                   if date_to_path.get(ds)), None)
    if sample is None:
        print("  WARNING: No files to probe")
        return {"sst_var": "data", "lats": None, "lons": None, "raw_info": None}

    with Dataset(sample, "r") as nc:
        sst_var = next((n for n in ["data", "sst"] if n in nc.variables), None)
        if sst_var is None:
            print("  WARNING: Cannot find SST variable")
            return {"sst_var": "data", "lats": None, "lons": None, "raw_info": None}
        dtype = str(nc.variables[sst_var].dtype)
        shape = nc.variables[sst_var].shape
        lats = nc.variables["lat"][:].copy()
        lons = nc.variables["lon"][:].copy()

    print(f"\n  Source format probe ({sample}):")
    print(f"    SST variable:   {sst_var}")
    print(f"    Storage dtype:  {dtype}")
    print(f"    Shape:          {shape}")
    print(f"    Lat: {len(lats)}  [{lats[0]:.4f} .. {lats[-1]:.4f}]")
    print(f"    Lon: {len(lons)}  [{lons[0]:.4f} .. {lons[-1]:.4f}]")

    raw_info = _probe_raw_read(date_to_path)
    return {"sst_var": sst_var, "lats": lats, "lons": lons, "raw_info": raw_info}


# ── Backend detection ──────────────────────────────────────────────────────

def _detect_backend():
    """Detect PyTorch ROCm/DCU backend. Returns num_gpus."""
    import torch
    print("  Probing accelerator backends ...\n")
    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
    if not is_hip or not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm/DCU backend not available")

    props = torch.cuda.get_device_properties(torch.device("cuda"))
    num_gpus = torch.cuda.device_count()
    mem_gb = props.total_memory / 1024**3
    print(f"  [PyTorch] ROCm/DCU backend  |  {props.name}  |  "
          f"{mem_gb:.1f} GB  |  Compute {props.major}.{props.minor}")
    print(f"    HIP version: {torch.version.hip}")
    print(f"    GPUs: {num_gpus}")
    return num_gpus




# ── GPU compute ────────────────────────────────────────────────────────────

def _compute_torch(data_np, bands):
    """Multi-GPU threaded compute (also handles single-GPU).

    Transfers compact bands to GPU → unfold/permute/reshape → topk P90 + nanmean.
    """
    import torch
    import threading

    num_gpus = torch.cuda.device_count()

    band_groups = [[] for _ in range(num_gpus)]
    for i, band in enumerate(bands):
        band_groups[i % num_gpus].append(band)

    print(f"  GPUs: {num_gpus}, Bands: {bands}")
    for gid, group in enumerate(band_groups):
        if group:
            print(f"    GPU {gid}: {len(group)} bands {group}")

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    # ── Detect P90 backend (quantile vs topk) ────────────────────────────
    _USE_QUANTILE = False
    try:
        torch.quantile(torch.zeros(10, dtype=torch.float16, device="cuda"), 0.9)
        _USE_QUANTILE = True
    except RuntimeError:
        pass
    print(f"  P90 backend: {'quantile' if _USE_QUANTILE else 'topk (quantile not supported for f16)'}")

    def _p90(sub):
        """Compute 90th percentile along dim=1. sub modified in-place (NaN→inf)."""
        sub.nan_to_num_(nan=float("inf"))
        if _USE_QUANTILE:
            p90 = torch.quantile(sub, 0.90, dim=1)
        else:
            topvals = torch.topk(sub, k=34, dim=1, largest=True, sorted=True).values
            p90 = topvals[:, -1, :, :] + (topvals[:, -2, :, :] - topvals[:, -1, :, :]) * 0.1
        return torch.where(torch.isinf(p90), float("nan"), p90)

    def _worker(gpu_id, group):
        if not group:
            return
        torch.cuda.set_device(gpu_id)

        for r0, r1 in group:
            t0 = time.time()

            # .copy() ensures contiguous CPU memory for fast GPU DMA transfer
            band_t = torch.from_numpy(
                data_np[:, :, r0:r1, :].copy()).to(gpu_id)
            windows = band_t.unfold(dimension=1, size=WINDOW_SIZE, step=1)

            with torch.inference_mode():
                for d0 in range(0, N_OUTPUT_DAYS, SUB_BATCH_DAYS):
                    d1 = min(d0 + SUB_BATCH_DAYS, N_OUTPUT_DAYS)
                    n_days = d1 - d0

                    sub = windows[:, d0:d1].permute(1, 0, 4, 2, 3).reshape(
                        n_days, N_YEARS * WINDOW_SIZE, r1 - r0, N_LON)

                    # nanmean first — small output, frees sub for P90
                    climatology[d0:d1, r0:r1, :] = (
                        torch.nanmean(sub, dim=1).cpu().numpy().astype(np.float32))

                    # P90: NaN→inf in-place, then quantile or topk
                    p90 = _p90(sub)

                    threshold[d0:d1, r0:r1, :] = (
                        p90.cpu().numpy().astype(np.float32))

                    del sub, p90

            del band_t, windows
            torch.cuda.empty_cache()
            print(f"  [GPU {gpu_id}] Band [{r0}:{r1}] done "
                  f"({time.time() - t0:.1f}s)")

    threads = []
    for gid, group in enumerate(band_groups):
        t = threading.Thread(target=_worker, args=(gid, group))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return threshold, climatology


# ── Latitude bands ─────────────────────────────────────────────────────────

def build_bands():
    base, rem = divmod(N_LAT, N_BANDS)
    bands, r0 = [], 0
    for i in range(N_BANDS):
        sz = base + (1 if i < rem else 0)
        bands.append((r0, r0 + sz))
        r0 += sz
    return bands


# ── ProcessPool I/O worker ─────────────────────────────────────────────────

def _read_one_year(args):
    """Read one year's files → list of (yr_idx, day_idx, block)."""
    items, sst_var, raw_info = args
    results = []
    for idx_slot, fp in items:
        block = (raw_read_sst(fp, raw_info) if raw_info
                 else Dataset(fp, "r").variables[sst_var][:])
        block = np.squeeze(block)
        if block.shape != (N_LAT, N_LON):
            block = block.T
        yr_idx = idx_slot // N_DAYS_PER_YEAR
        day_idx = idx_slot % N_DAYS_PER_YEAR
        results.append((yr_idx, day_idx, np.asarray(block, dtype=np.float16)))
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t_prep = time.time()

    # ── Prep: file index & format probe ──
    print("Building file index ...")
    date_to_path = build_date_filepath_map()
    print(f"  Indexed {len(date_to_path)} files")

    fmt = detect_source_format(date_to_path)
    sst_var = fmt["sst_var"]
    lats, lons = fmt["lats"], fmt["lons"]
    raw_info = fmt["raw_info"]

    shape = (N_YEARS, N_DAYS_PER_YEAR, N_LAT, N_LON)
    size_gb = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * 2 / 1024**3
    print(f"  Data array {shape} float16 ({size_gb:.2f} GB)")

    # ── Build year tasks ──
    needed_dates = []
    for y in range(START_YEAR, END_YEAR + 1):
        needed_dates.extend(get_date_strings_for_year(y))
    by_year = defaultdict(list)
    for j, ds in enumerate(needed_dates):
        fp = date_to_path.get(ds)
        if fp is not None:
            by_year[ds[:4]].append((j, fp))

    year_tasks = [(items, sst_var, raw_info)
                  for _yr, items in sorted(by_year.items())]
    n_years = len(year_tasks)

    # ── Backend detection ──
    print("\nDetecting accelerator backend ...")
    t_setup = time.time()
    num_gpus = _detect_backend()
    setup_time = time.time() - t_setup

    # ── ProcessPool I/O ──
    raw_label = "raw I/O" if raw_info else "netCDF4"
    print(f"\nReading {n_years} years ({len(needed_dates)} files) "
          f"workers={MAX_READ_WORKERS} (float16, {raw_label}) ...")
    t_io = time.time()

    data = np.full(shape, np.nan, dtype=np.float16)
    executor = ProcessPoolExecutor(max_workers=MAX_READ_WORKERS)
    chunksize = int(os.environ.get("MAP_CHUNKSIZE", "1"))
    done = 0
    for results in executor.map(_read_one_year, year_tasks, chunksize=chunksize):
        for yr_idx, day_idx, blk in results:
            data[yr_idx, day_idx] = blk
        done += len(results)
        if done % 510 == 0 or done >= len(needed_dates):
            print(f"  ... {done}/{len(needed_dates)} files "
                  f"({time.time() - t_io:.1f}s)")
    executor.shutdown()
    io_elapsed = time.time() - t_io
    print(f"  I/O: {io_elapsed:.1f}s")

    # ── Compute ──
    print(f"\nComputing threshold & climatology "
          f"({N_OUTPUT_DAYS} days x 30-year sliding window) ...")
    bands = build_bands()
    t_compute = time.time()
    threshold, climatology = _compute_torch(data, bands)
    compute_elapsed = time.time() - t_compute
    print(f"  Compute: {compute_elapsed:.1f}s")

    del data; gc.collect()

    # ── Save ──
    print("\nSaving ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d0_2019 = datetime(2019, 1, 1)  # 2019 is not a leap year (= 365 days)
    for out_idx in range(N_OUTPUT_DAYS):
        doy = 152 + out_idx                       # Jun 1 = DOY 152
        dt = d0_2019 + timedelta(days=doy - 1)
        fname = dt.strftime("%m%d.nc")

        with Dataset(OUT_DIR / fname, "w", format="NETCDF4") as nc:
            nc.createDimension("Lat", N_LAT)
            nc.createDimension("Lon", N_LON)
            nc.createDimension("Day", 1)

            v_lat = nc.createVariable("Lat", "f4", ("Lat",))
            v_lat[:] = lats
            v_lat.long_name = "Latitude"

            v_lon = nc.createVariable("Lon", "f4", ("Lon",))
            v_lon[:] = lons
            v_lon.long_name = "Longitude"

            v_clim = nc.createVariable("Climmean", "f4", ("Lat", "Lon"))
            v_clim[:] = np.ascontiguousarray(climatology[out_idx])
            v_clim.long_name = "OSTIA SST climatology 1991-2020"

            v_p90 = nc.createVariable("P90_sst", "f4", ("Lat", "Lon"))
            v_p90[:] = np.ascontiguousarray(threshold[out_idx])
            v_p90.long_name = "90th percentile of precipitation"

            v_doy = nc.createVariable("dayofyear", "i4", ("Day",))
            v_doy[:] = doy
            v_doy.long_name = "Day of year (1-365, no 29Feb)"

            nc.baseline_period = "1991-2020"

    t_end = time.time()
    save_elapsed = t_end - t_compute - compute_elapsed
    print(f"  Saved {N_OUTPUT_DAYS} files to {OUT_DIR}/")

    # ── Timing breakdown ──
    prep_elapsed = t_io - t_prep - setup_time
    total_elapsed = io_elapsed + compute_elapsed + save_elapsed

    print(f"\n{'='*60}")
    print(f"  TIMING BREAKDOWN (float16, ProcessPool)")
    print(f"{'='*60}")
    print(f"  Prep:       {prep_elapsed:8.1f}s   (file index + format probe)")
    print(f"  Setup:      {setup_time:8.1f}s   (backend detection, not counted)")
    print(f"  I/O:        {io_elapsed:8.1f}s   (ProcessPool {MAX_READ_WORKERS} workers)")
    print(f"  Compute:    {compute_elapsed:8.1f}s")
    print(f"  Save:       {save_elapsed:8.1f}s")
    print(f"  {'─'*50}")
    print(f"  Total:      {total_elapsed:8.3f}s  ({total_elapsed/60:.2f} min)")
    print(f"{'='*60}")
    print(f"Done!")

    # ── Validation ──
    print(f"\nValidation:")
    print(f"  P90_sst range:       [{np.nanmin(threshold):.2f}, "
          f"{np.nanmax(threshold):.2f}]")
    print(f"  Climmean range:      [{np.nanmin(climatology):.2f}, "
          f"{np.nanmax(climatology):.2f}]")
    print(f"  NaN fraction (P90):  {np.isnan(threshold).mean()*100:.1f}%")
    print(f"  P90 > Climmean (mean): "
          f"{np.nanmean(threshold) > np.nanmean(climatology)}")


if __name__ == "__main__":
    main()
