#!/usr/bin/env python
"""compute_server.py — 海光 DCU / ROCm with banded GPU processing.

Computes 90th-percentile SST threshold (P90_sst) and mean climatology
(Climmean) for June–August over 1991–2020 baseline, using a 365-day
calendar (Feb 29 removed) matching the MATLAB reference.

Output format matches clim_verification.m:
  - Global grid 721×1440
  - Variable names: Climmean, P90_sst
  - Dimension order: [Lon, Lat] (transposed from Python)
  - Dimension names: Lat, Lon, Day
  - 92 daily files: 0601.nc … 0831.nc

Memory strategy (global data ≈ 24 GB CPU; GPU has 32 GB):
  1. Load all data into CPU RAM (fits in 128 GB)
  2. Process on GPU in 6 latitude bands (~120 rows each ≈ 4.2 GB/band)
  3. Per band: sliding 11-day windows → sort → percentile + mean
"""

import numpy as np
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # 关闭 HDF5 文件锁，Lustre 只读场景有效
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from netCDF4 import Dataset
import time
import gc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("/public/home/achwjznh4b/Newdata/")       # files named YYYYMMDD
OUT_DIR = Path(__file__).resolve().parent / "ERA5" / "Climatology"

START_YEAR = 1991
END_YEAR = 2020
N_YEARS = END_YEAR - START_YEAR + 1                       # 30

WINDOW_HALF = 5
WINDOW_SIZE = 2 * WINDOW_HALF + 1                         # 11

N_LAT = 721
N_LON = 1440
N_OUTPUT_DAYS = 92                                         # Jun 1 – Aug 31

# Date range covering all windows: DOY 147–248 in 365-day calendar
# → May 27 to Sep 5 in natural calendar (102 days)
DATE_START_MMDD = (5, 27)
DATE_END_MMDD = (9, 5)
N_DAYS_PER_YEAR = 102

MAX_READ_WORKERS = int(os.environ.get("MAX_READ_WORKERS", "16"))
N_BANDS = 15

VAR_SST = "data"
VAR_LAT = "lat"
VAR_LON = "lon"


# ---------------------------------------------------------------------------
# 365-day calendar helpers (match MATLAB exactly)
# ---------------------------------------------------------------------------

def is_leap_year(yr):
    return (yr % 4 == 0 and yr % 100 != 0) or (yr % 400 == 0)


def build_365_calendar(year):
    """Return list of datetime.date for a 365-day year (Feb 29 removed).

    Matches MATLAB:
      base_dates = datetime(yr,1,1):datetime(yr,12,31);
      if is_leap: base_dates = base_dates(day(base_dates,'dayofyear')~=60);
    """
    d0 = datetime(year, 1, 1)
    if is_leap_year(year):
        # Build all 366 days, then remove Feb 29 (day-of-year 60 in leap year)
        dates = []
        for offset in range(366):
            d = d0 + timedelta(days=offset)
            if not (d.month == 2 and d.day == 29):
                dates.append(d)
    else:
        dates = [d0 + timedelta(days=i) for i in range(365)]
    return dates


def get_date_strings_for_year(year):
    """Return YYYYMMDD strings for May 27 – Sep 5 (natural calendar)."""
    start = datetime(year, *DATE_START_MMDD)
    end = datetime(year, *DATE_END_MMDD)
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def build_date_filepath_map():
    """Glob all data files → {YYYYMMDD: filepath}."""
    date_to_path = {}
    for fp in DATA_DIR.iterdir():
        name = fp.name
        if len(name) == 8 and name.isdigit():
            date_to_path[name] = str(fp)
    return date_to_path


def detect_source_format(date_to_path):
    """Detect source data format from a sample file.

    Returns dict with keys: packed (bool), scale_factor, add_offset,
    fill_value, sst_var (str), dtype (str). If packed=True, the source
    uses int16 storage with scale_factor/add_offset and we can halve
    I/O by reading raw packed data.
    """
    # Find a sample file
    sample_path = None
    for date_str in sorted(date_to_path.keys()):
        fp = date_to_path[date_str]
        if fp:
            sample_path = fp
            break

    _default = {"packed": False, "scale_factor": None, "add_offset": 0.0,
                 "fill_value": None, "sst_var": VAR_SST, "dtype": "float32",
                 "lats": None, "lons": None, "sample_path": None}

    if sample_path is None:
        print("  WARNING: No files to probe — assuming float32")
        return _default

    # Try SST variable names
    sst_var = None
    with Dataset(sample_path, "r") as nc:
        for name in [VAR_SST, "sst"]:
            if name in nc.variables:
                sst_var = name
                break

    if sst_var is None:
        print("  WARNING: Cannot find SST variable — assuming float32")
        return _default

    with Dataset(sample_path, "r") as nc:
        v = nc.variables[sst_var]
        dtype = str(v.dtype)
        scale_factor = getattr(v, "scale_factor", None)
        add_offset = getattr(v, "add_offset", None)
        fill_value = getattr(v, "_FillValue", None)
        shape = v.shape

        # Read lat/lon coords while file is open (avoids separate xarray open)
        lats = nc.variables[VAR_LAT][:].copy()
        lons = nc.variables[VAR_LON][:].copy()

    print(f"\n  Source format probe ({sample_path}):")
    print(f"    SST variable:   {sst_var}")
    print(f"    Storage dtype:  {dtype}")
    print(f"    Shape:          {shape}")
    print(f"    Scale factor:   {scale_factor}")
    print(f"    Add offset:     {add_offset}")
    print(f"    Fill value:     {fill_value}")
    print(f"    Lat: {len(lats)}  [{lats[0]:.4f} .. {lats[-1]:.4f}]")
    print(f"    Lon: {len(lons)}  [{lons[0]:.4f} .. {lons[-1]:.4f}]")

    is_int16 = dtype.startswith("int16")
    packed = is_int16 and scale_factor is not None

    if packed:
        float32_size = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * 4 / 1024**3
        int16_size = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * 2 / 1024**3
        print(f"    PACKED READ ENABLED: I/O {float32_size:.1f} GB → {int16_size:.1f} GB "
              f"(-{float32_size - int16_size:.1f} GB)")
    else:
        print(f"    Standard float32 read (no packed benefit)")

    return {
        "packed": packed,
        "scale_factor": scale_factor,
        "add_offset": add_offset if add_offset is not None else 0.0,
        "fill_value": fill_value,
        "sst_var": sst_var,
        "dtype": dtype,
        "lats": lats,
        "lons": lons,
        "sample_path": sample_path,
    }


# ---------------------------------------------------------------------------
# Parallel I/O workers
# ---------------------------------------------------------------------------

def read_one_year(args):
    """Read daily SST files for a single year — no stat calls at all.

    Receives pre-built (idx_slot, filepath) pairs from date_to_path map.
    No os.path.isfile(), no os.path.join(), no linear scan.
    """
    items, var_sst, np_dtype = args
    out_idx = []
    out_blk = []
    for idx_slot, fp in items:
        with Dataset(fp, "r") as ds:
            block = ds.variables[var_sst][:]
        block = np.squeeze(block)
        if block.shape != (N_LAT, N_LON):
            block = block.T
        out_idx.append(idx_slot)
        out_blk.append(np.asarray(block, dtype=np_dtype))
    return out_idx, out_blk


# ---------------------------------------------------------------------------
# Accelerator backends
# ---------------------------------------------------------------------------

def _detect_backend():
    """Multi-path accelerator detection with diagnostic output.

    Priority:
      1. PyTorch ROCm/DCU (torch.version.hip + torch.cuda.is_available())
      2. PyTorch generic CUDA (torch.cuda.is_available() + AMD GPU check)
      3. HIP native (pyhip / hip-python)
      4. CuPy ROCm backend
      5. CPU fallback
    """
    print("  Probing accelerator backends ...\n")

    # ── Path 1: PyTorch ROCm/DCU version ──
    try:
        import torch
        is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
        if is_hip:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                props = torch.cuda.get_device_properties(device)
                mem_gb = props.total_memory / 1024**3
                print(f"  [PyTorch] ROCm/DCU backend  |  {props.name}  |  "
                      f"{mem_gb:.1f} GB  |  Compute {props.major}.{props.minor}")
                print(f"    HIP version: {torch.version.hip}")
                return "torch", device
            else:
                print("  [PyTorch] ROCm/HIP compiled but torch.cuda.is_available()=False")
                print("    Check ROCm runtime, device permissions, or HIP_VISIBLE_DEVICES")
        else:
            # PyTorch installed but NOT compiled with ROCm/HIP
            if torch.cuda.is_available():
                device = torch.device("cuda")
                props = torch.cuda.get_device_properties(device)
                name = props.name.lower()
                if "amd" in name or "gfx" in name or "radeon" in name:
                    mem_gb = props.total_memory / 1024**3
                    print(f"  [PyTorch] Generic CUDA (AMD GPU)  |  {props.name}  |  "
                          f"{mem_gb:.1f} GB")
                    print(f"    WARNING: Using CUDA path on AMD hardware — "
                          f"recommend DCU PyTorch build")
                    return "torch", device
                else:
                    mem_gb = props.total_memory / 1024**3
                    print(f"  [PyTorch] Generic CUDA (NVIDIA)  |  {props.name}  |  "
                          f"{mem_gb:.1f} GB  |  Compute {props.major}.{props.minor}")
                    return "torch", device
            else:
                print("  [PyTorch] 已安装但未编译 ROCm/HIP 支持，"
                      "请安装 DCU 版 PyTorch")
                print("    e.g. pip install torch==2.0.1+rocm5.4.1 "
                      "--index-url https://download.pytorch.org/whl/rocm5.4.1")
    except ImportError:
        print("  [PyTorch] not installed")
    except Exception as e:
        print(f"  [PyTorch] error during detection: {e}")

    # ── Path 2: HIP native interface ──
    try:
        import hip
        n_dev = hip.getDeviceCount()
        if n_dev > 0:
            props = hip.getDeviceProperties(0)
            mem_gb = props.totalGlobalMem / 1024**3
            print(f"  [HIP] {n_dev} device(s)  |  device 0: {props.name}  ({mem_gb:.1f} GB)")
            print("    INFO: HIP detected but no Python compute backend — "
                  "will use CuPy if available")
        else:
            print("  [HIP] No devices found via hip-python")
    except ImportError:
        print("  [HIP] pyhip / hip-python not installed (optional)")
    except Exception as e:
        print(f"  [HIP] error during detection: {e}")

    # ── Path 3: CuPy ROCm backend ──
    try:
        import cupy as cp
        n_dev = cp.cuda.runtime.getDeviceCount()
        if n_dev > 0:
            cp.cuda.Device(0).use()
            mem_gb = cp.cuda.Device(0).mem_info[1] / 1024**3
            print(f"  [CuPy] {n_dev} device(s)  |  device 0: {mem_gb:.1f} GB")
            return "cupy", 0
        else:
            print("  [CuPy] installed but no devices visible")
    except ImportError:
        print("  [CuPy] not installed")
    except Exception as e:
        print(f"  [CuPy] error during detection: {e}")

    # ── Path 4: CPU fallback ──
    print("\n  [CPU] 未检测到任何加速器，将使用 NumPy (CPU)")
    return "cpu", None


# ---------------------------------------------------------------------------
# PyTorch backend — banded processing
# ---------------------------------------------------------------------------

def _p90_fast_torch(t, dim=1):
    """Fast 90th percentile via topk — only sorts top ~10% instead of full sort.

    Replaces torch.nanquantile(t, 0.90, dim=dim).  Algorithm:
      - Replace NaN with +inf so they sort to the top (largest).
      - topk(largest=True, k=topk_n) gets the largest ~10% values.
      - Linear interpolation between the two boundary values for exact p90.
    """
    import torch

    n_time = t.shape[dim]
    p90_pos = 0.90 * (n_time - 1)
    k0_idx = int(p90_pos)
    k1_idx = min(k0_idx + 1, n_time - 1)
    w_frac = p90_pos - k0_idx
    topk_n = n_time - k0_idx

    inf = torch.tensor(float("inf"), device=t.device, dtype=t.dtype)
    filled = torch.where(torch.isnan(t), inf, t)
    topvals, _ = torch.topk(filled, k=topk_n, dim=dim, largest=True, sorted=True)

    # Index the last element along dim (= smallest among top values = v0)
    idx = [slice(None)] * t.ndim
    idx[dim] = -1
    v0 = topvals[tuple(idx)]

    if k1_idx == k0_idx:
        p90 = v0
    else:
        idx[dim] = -2
        v1 = topvals[tuple(idx)]
        p90 = v0 + (v1 - v0) * w_frac

    p90 = torch.where(torch.isinf(p90), float("nan"), p90)
    return p90


def _compute_torch_band(data_np, device, bands, packed=False,
                        scale_factor=None, add_offset=None, fill_value=None):
    """Banded GPU computation with PyTorch — sub-batch days to fit VRAM.

    When packed=True, data_np is int16; we cast→float32, mask fill values
    to NaN, and apply scale_factor/add_offset on the GPU.
    """
    import torch

    SUB_BATCH_DAYS = 23  # process 23 output days at a time (92/4=23)

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    for band_idx, (r0, r1) in enumerate(bands):
        t0 = time.time()
        band_rows = r1 - r0

        # 1. Copy band to device & unfold sliding windows
        band_data = data_np[:, :, r0:r1, :].copy()  # (30, 102, band_rows, 1440)
        band_t = torch.from_numpy(band_data).to(device)

        # If packed int16, cast to float32, mask fill→NaN, apply scale
        if packed:
            band_t = band_t.to(torch.float32)
            mask = band_t == fill_value
            band_t.mul_(scale_factor).add_(add_offset)
            band_t[mask] = float('nan')

        # (30, 102, band_rows, 1440) -> (30, 92, band_rows, 1440, 11)
        windows_view = band_t.unfold(dimension=1, size=WINDOW_SIZE, step=1)

        # 2. Process output days in sub-batches
        with torch.inference_mode():
            for d0 in range(0, N_OUTPUT_DAYS, SUB_BATCH_DAYS):
                d1 = min(d0 + SUB_BATCH_DAYS, N_OUTPUT_DAYS)
                n_days = d1 - d0

                # Slice sub-batch: (30, n_days, band_rows, 1440, 11)
                sub = windows_view[:, d0:d1, :, :, :]
                # Reshape: (n_days, 330, band_rows, 1440)
                sub = sub.permute(1, 0, 4, 2, 3).reshape(
                    n_days, N_YEARS * WINDOW_SIZE, band_rows, N_LON)

                # Compute
                sub_thresh = _p90_fast_torch(sub, dim=1)
                sub_clim = torch.nanmean(sub, dim=1)

                # Copy back
                threshold[d0:d1, r0:r1, :] = sub_thresh.cpu().numpy().astype(np.float32)
                climatology[d0:d1, r0:r1, :] = sub_clim.cpu().numpy().astype(np.float32)

                del sub, sub_thresh, sub_clim

        del band_t, windows_view
        torch.cuda.empty_cache()
        gc.collect()

        elapsed = time.time() - t0
        print(f"  Band {band_idx+1}/{len(bands)} rows [{r0}:{r1}] done ({elapsed:.1f}s)")

    return threshold, climatology


# ---------------------------------------------------------------------------
# Multi-GPU threaded computation
# ---------------------------------------------------------------------------

def _compute_torch_multi_gpu(data_np, num_gpus, bands, packed=False,
                             scale_factor=None, add_offset=None, fill_value=None):
    """Multi-GPU banded computation using threads (shared memory, zero-copy).

    When packed=True, data_np is int16; each GPU casts→float32, masks fill
    values to NaN, and applies scale_factor/add_offset.
    """
    import torch
    import threading

    # Distribute bands round-robin across GPUs
    band_groups = [[] for _ in range(num_gpus)]
    for i, band in enumerate(bands):
        band_groups[i % num_gpus].append(band)

    print(f"  Multi-GPU ({num_gpus} GPUs):")
    for gpu_id, group in enumerate(band_groups):
        print(f"    GPU {gpu_id}: {len(group)} bands {group}")

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    SUB_BATCH_DAYS = 23

    def _thread_worker(gpu_id, group):
        torch.cuda.set_device(gpu_id)
        for r0, r1 in group:
            t0 = time.time()
            band_rows = r1 - r0

            band_data = data_np[:, :, r0:r1, :].copy()
            band_t = torch.from_numpy(band_data).to(gpu_id)

            # If packed int16, cast to float32, mask fill→NaN, apply scale
            if packed:
                band_t = band_t.to(torch.float32)
                mask = band_t == fill_value
                band_t.mul_(scale_factor).add_(add_offset)
                band_t[mask] = float('nan')

            windows_view = band_t.unfold(dimension=1, size=WINDOW_SIZE, step=1)

            with torch.inference_mode():
                for d0 in range(0, N_OUTPUT_DAYS, SUB_BATCH_DAYS):
                    d1 = min(d0 + SUB_BATCH_DAYS, N_OUTPUT_DAYS)
                    n_days = d1 - d0

                    sub = windows_view[:, d0:d1, :, :, :]
                    sub = sub.permute(1, 0, 4, 2, 3).reshape(
                        n_days, N_YEARS * WINDOW_SIZE, band_rows, N_LON)

                    sub_thresh = _p90_fast_torch(sub, dim=1)
                    sub_clim = torch.nanmean(sub, dim=1)

                    threshold[d0:d1, r0:r1, :] = sub_thresh.cpu().numpy().astype(np.float32)
                    climatology[d0:d1, r0:r1, :] = sub_clim.cpu().numpy().astype(np.float32)

                    del sub, sub_thresh, sub_clim

            del band_t, windows_view
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            print(f"  [GPU {gpu_id}] Band rows [{r0}:{r1}] done ({elapsed:.1f}s)")

    threads = []
    for gpu_id, group in enumerate(band_groups):
        if not group:
            continue
        t = threading.Thread(target=_thread_worker, args=(gpu_id, group))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return threshold, climatology


# ---------------------------------------------------------------------------
# CuPy backend — banded processing
# ---------------------------------------------------------------------------

def _compute_cupy_band(data_np, device_id, bands, packed=False,
                       scale_factor=None, add_offset=None, fill_value=None):
    """Banded GPU computation with CuPy.

    When packed=True, data_np is int16; we cast→float32, mask fill values
    to NaN, and apply scale_factor/add_offset on the GPU.
    """
    import cupy as cp

    cp.cuda.Device(device_id).use()

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    for band_idx, (r0, r1) in enumerate(bands):
        t0 = time.time()
        band_rows = r1 - r0

        # 1. Copy to GPU
        band_data = data_np[:, :, r0:r1, :].copy()
        data_g = cp.asarray(band_data, dtype=cp.float32)  # (30, 102, band_rows, 1440)

        # If packed int16, cast to float32, mask fill→NaN, apply scale
        if packed:
            data_g = data_g.astype(cp.float32)
            mask = data_g == fill_value
            data_g *= scale_factor
            data_g += add_offset
            data_g[mask] = cp.nan

        # 2. Build sliding windows manually
        windows_g = cp.empty((N_OUTPUT_DAYS, N_YEARS * WINDOW_SIZE, band_rows, N_LON),
                             dtype=cp.float32)
        for i in range(N_OUTPUT_DAYS):
            w = data_g[:, i:i + WINDOW_SIZE, :, :]
            windows_g[i] = w.reshape(N_YEARS * WINDOW_SIZE, band_rows, N_LON)

        del data_g
        cp.get_default_memory_pool().free_all_blocks()

        # 3. Sort in-place for percentile
        windows_g.sort(axis=1)

        # Valid count
        valid_count = (~cp.isnan(windows_g)).sum(axis=1)  # (92, band_rows, 1440)

        # 90th percentile with linear interpolation
        v_idx = 0.90 * (valid_count - 1)
        v_idx = cp.clip(v_idx, 0, None)
        lo_idx = cp.floor(v_idx).astype(cp.int32)
        hi_idx = cp.minimum(lo_idx + 1, valid_count - 1).astype(cp.int32)
        frac = (v_idx - lo_idx).astype(cp.float32)

        batch_idx = cp.arange(N_OUTPUT_DAYS)[:, None, None]
        row_idx = cp.arange(band_rows)[None, :, None]
        col_idx = cp.arange(N_LON)[None, None, :]

        lo_vals = windows_g[batch_idx, lo_idx, row_idx, col_idx]
        hi_vals = windows_g[batch_idx, hi_idx, row_idx, col_idx]
        band_thresh = lo_vals + frac * (hi_vals - lo_vals)

        # Mean
        nan_mask = cp.isnan(windows_g)
        windows_clean = cp.where(nan_mask, 0.0, windows_g)
        band_clim = windows_clean.sum(axis=1) / cp.maximum(valid_count, 1)

        # 4. Copy back
        threshold[:, r0:r1, :] = cp.asnumpy(band_thresh.astype(cp.float32))
        climatology[:, r0:r1, :] = cp.asnumpy(band_clim.astype(cp.float32))

        del windows_g, band_thresh, band_clim
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        elapsed = time.time() - t0
        print(f"  Band {band_idx+1}/{len(bands)} rows [{r0}:{r1}] done ({elapsed:.1f}s)")

    return threshold, climatology


# ---------------------------------------------------------------------------
# CPU fallback (vectorized NumPy — full array at once since we have 128 GB RAM)
# ---------------------------------------------------------------------------

def _compute_cpu(data_np, packed=False, scale_factor=None,
                 add_offset=None, fill_value=None):
    """Memory-efficient CPU — process one DOY at a time to avoid ~235 GB alloc.

    When packed=True, data_np is int16; we cast→float32, mask fill→NaN,
    and apply scale on the fly for each window.
    """
    t0 = time.time()

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    for i in range(N_OUTPUT_DAYS):
        window = data_np[:, i:i + WINDOW_SIZE, :, :]           # (30, 11, 721, 1440)

        if packed:
            window = window.astype(np.float32)
            mask = window == fill_value
            window *= scale_factor
            window += add_offset
            window[mask] = np.nan

        flat = window.reshape(N_YEARS * WINDOW_SIZE, N_LAT, N_LON)  # (330, 721, 1440)

        # Fast percentile via np.partition (O(n)) instead of full sort (O(n log n))
        n_time = flat.shape[0]
        p90_pos = 0.90 * (n_time - 1)
        k0_idx = int(p90_pos)
        k1_idx = min(k0_idx + 1, n_time - 1)
        w_frac = p90_pos - k0_idx

        filled = np.where(np.isnan(flat), np.inf, flat)
        part = np.partition(filled, kth=(k0_idx, k1_idx), axis=0)
        v0 = part[k0_idx]
        if k1_idx == k0_idx:
            p90 = v0
        else:
            v1 = part[k1_idx]
            p90 = v0 + (v1 - v0) * w_frac
        threshold[i] = np.where(np.isinf(p90), np.nan, p90).astype(np.float32)
        climatology[i] = np.nanmean(flat, axis=0).astype(np.float32)
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_OUTPUT_DAYS - i - 1) / rate if rate > 0 else 0
            print(f"  Day {i+1}/{N_OUTPUT_DAYS}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  CPU compute: {elapsed:.1f}s")
    return threshold, climatology


# ---------------------------------------------------------------------------
# Build latitude bands
# ---------------------------------------------------------------------------

def build_bands():
    """Split 721 latitude rows into N_BANDS roughly equal bands."""
    base = N_LAT // N_BANDS
    rem = N_LAT % N_BANDS
    bands = []
    r0 = 0
    for i in range(N_BANDS):
        size = base + (1 if i < rem else 0)
        bands.append((r0, r0 + size))
        r0 += size
    return bands


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    prep_start = time.time()

    # ---- File index ----
    print("Building file index ...")
    date_to_path = build_date_filepath_map()
    print(f"  Indexed {len(date_to_path)} files")

    # ---- Detect source format (int16 packed vs float32) ----
    fmt = detect_source_format(date_to_path)
    packed = fmt["packed"]
    scale_factor = fmt["scale_factor"]
    add_offset = fmt["add_offset"]
    fill_value = fmt["fill_value"]
    sst_var = fmt["sst_var"]
    if sst_var != VAR_SST:
        print(f"  NOTE: Using SST variable '{sst_var}' (not '{VAR_SST}')")

    # Safety: ensure fill_value is set when packed (required for NaN masking)
    if packed and fill_value is None:
        fill_value = -32768  # int16 sentinel
        print(f"  NOTE: No _FillValue found — using {fill_value} as fill sentinel")

    # Data array dtype and size — float16 halves memory vs float32 (teammate approach)
    if packed:
        data_dtype = np.int16
        elem_bytes = 2
        nan_fill = fill_value if fill_value is not None else -32768
        packed_tag = "int16-packed"
        print(f"  Scale: raw_int16 * {scale_factor} + {add_offset}")
        print(f"  Fill value masking: raw == {fill_value} → NaN")
    else:
        data_dtype = np.float16
        elem_bytes = 2
        nan_fill = np.nan
        packed_tag = "float16"

    shape = (N_YEARS, N_DAYS_PER_YEAR, N_LAT, N_LON)
    size_gb = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * elem_bytes / 1024**3
    print(f"  Data array {shape} {packed_tag} ({size_gb:.2f} GB)")

    # ---- Year-grain I/O (netCDF4 + ProcessPoolExecutor) ----
    # Build year tasks with pre-resolved filepaths (no stat/isfile in workers)
    needed_dates = []
    for y in range(START_YEAR, END_YEAR + 1):
        needed_dates.extend(get_date_strings_for_year(y))
    by_year = defaultdict(list)
    for j, ds in enumerate(needed_dates):
        fp = date_to_path.get(ds)
        if fp is not None:
            by_year[ds[:4]].append((j, fp))

    year_tasks = [(items, sst_var, data_dtype)
                  for _yr, items in sorted(by_year.items())]
    n_years = len(year_tasks)

    print(f"\nReading {n_years} years ({len(needed_dates)} files) "
          f"grain=year workers={MAX_READ_WORKERS} ({packed_tag}) ...")
    total_start = time.time()

    data = np.full(shape, nan_fill, dtype=data_dtype)

    executor = ProcessPoolExecutor(max_workers=MAX_READ_WORKERS)
    results = executor.map(read_one_year, year_tasks, chunksize=1)
    done_files = 0
    for idx_list, blk_list in results:
        for j, blk in zip(idx_list, blk_list):
            yr_idx = j // N_DAYS_PER_YEAR
            day_idx = j % N_DAYS_PER_YEAR
            data[yr_idx, day_idx] = blk
        done_files += len(idx_list)
        if done_files % 510 == 0 or done_files >= len(needed_dates):
            print(f"  ... {done_files}/{len(needed_dates)} files "
                  f"({time.time() - total_start:.1f}s)")
    executor.shutdown()

    io_elapsed = time.time() - total_start
    print(f"  I/O: {io_elapsed:.1f}s")

    # ---- Backend detection ----
    print(f"\nDetecting accelerator backend ...")
    setup_start = time.time()
    backend, device = _detect_backend()
    setup_time = time.time() - setup_start

    # ---- Lat/lon coords (already read during format detection) ----
    lats = fmt["lats"]
    lons = fmt["lons"]

    io_end = time.time()

    # ---- Compute ----
    print(f"\nComputing threshold & climatology "
          f"({N_OUTPUT_DAYS} days x 30-year sliding window) ...")

    bands = build_bands()

    if backend == "torch":
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"  Detected {num_gpus} GPUs — using multi-GPU parallel mode")
            print(f"  Bands: {bands}")
            threshold, climatology = _compute_torch_multi_gpu(
                data, num_gpus, bands,
                packed=packed, scale_factor=scale_factor,
                add_offset=add_offset, fill_value=fill_value)
        else:
            print(f"  Single GPU mode")
            print(f"  Bands: {bands}")
            threshold, climatology = _compute_torch_band(
                data, device, bands,
                packed=packed, scale_factor=scale_factor,
                add_offset=add_offset, fill_value=fill_value)
    elif backend == "cupy":
        threshold, climatology = _compute_cupy_band(
            data, device, bands,
            packed=packed, scale_factor=scale_factor,
            add_offset=add_offset, fill_value=fill_value)
    else:
        threshold, climatology = _compute_cpu(
            data, packed=packed, scale_factor=scale_factor,
            add_offset=add_offset, fill_value=fill_value)

    # Free CPU data after compute
    del data
    gc.collect()

    compute_end = time.time()
    compute_elapsed = compute_end - io_end
    print(f"  Compute: {compute_elapsed:.1f}s")

    # ---- Save (netCDF4 direct write — faster than xarray on Lustre) ----
    print("\nSaving ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_dates = build_365_calendar(2020)  # 2020 is leap; used as 365-day reference
    for out_idx in range(N_OUTPUT_DAYS):
        doy = 152 + out_idx
        dt = base_dates[doy - 1]
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

    print(f"  Saved {N_OUTPUT_DAYS} files to {OUT_DIR}/")

    save_elapsed = time.time() - compute_end
    print(f"  Save: {save_elapsed:.1f}s")

    # ---- Validation ----
    print(f"\nValidation:")
    print(f"  P90_sst range:       [{np.nanmin(threshold):.2f}, {np.nanmax(threshold):.2f}]")
    print(f"  Climmean range:      [{np.nanmin(climatology):.2f}, {np.nanmax(climatology):.2f}]")
    print(f"  NaN fraction (P90):  {np.isnan(threshold).mean()*100:.1f}%")
    print(f"  P90 > Climmean (mean): {np.nanmean(threshold) > np.nanmean(climatology)}")

    prep_elapsed = total_start - prep_start
    total_elapsed = time.time() - total_start
    accounted = io_elapsed + setup_time + compute_elapsed + save_elapsed
    other = total_elapsed - accounted

    print(f"\n{'='*60}")
    print(f"  TIMING BREAKDOWN ({packed_tag})")
    print(f"{'='*60}")
    print(f"  Prep:       {prep_elapsed:8.1f}s          (file index + format probe)")
    print(f"  I/O:        {io_elapsed:8.1f}s  ({io_elapsed/total_elapsed*100:5.1f}%)")
    print(f"  Setup:      {setup_time:8.1f}s  ({setup_time/total_elapsed*100:5.1f}%)")
    print(f"  Compute:    {compute_elapsed:8.1f}s  ({compute_elapsed/total_elapsed*100:5.1f}%)")
    print(f"  Save:       {save_elapsed:8.1f}s  ({save_elapsed/total_elapsed*100:5.1f}%)")
    if other > 0.5:
        print(f"  Other:      {other:8.1f}s  ({other/total_elapsed*100:5.1f}%)")
    print(f"  {'─'*50}")
    print(f"  Total:      {total_elapsed:8.1f}s  ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    real_no_setup = total_elapsed - setup_time
    print(f"real {int(real_no_setup//60)}m{real_no_setup%60:.3f}s  (I/O+Compute+Save, no setup)")
    print("Done!")


if __name__ == "__main__":
    _wall_start = time.time()
    main()
