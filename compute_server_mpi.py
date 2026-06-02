#!/usr/bin/env python
"""compute_server_mpi.py — MPI distributed I/O version.

Replaces ProcessPoolExecutor with mpi4py + Intel MPI:
  - 30 MPI ranks, each reads 1 year (102 files) independently
  - No pickle serialization — data stays as numpy arrays
  - Isend/Irecv to rank 0 (shared memory on single node)
  - Rank 0 executes GPU computation (same as compute_server.py)

Usage:
  mpirun -np 30 python compute_server_mpi.py
"""

import numpy as np
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from netCDF4 import Dataset
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("/public/home/achwjznh4b/Newdata/")
OUT_DIR = Path(__file__).resolve().parent / "ERA5" / "Climatology"

START_YEAR = 1991
END_YEAR = 2020
N_YEARS = END_YEAR - START_YEAR + 1                       # 30

WINDOW_HALF = int(os.environ.get("WINDOW_HALF", "5"))
WINDOW_SIZE = 2 * WINDOW_HALF + 1

N_LAT = 721
N_LON = 1440
N_DAYS_PER_YEAR = 102
N_OUTPUT_DAYS = N_DAYS_PER_YEAR - WINDOW_SIZE + 1           # e.g. 102-11+1=92 (WH=5)

DATE_START_MMDD = (5, 27)
DATE_END_MMDD = (9, 5)

N_BANDS = int(os.environ.get("N_BANDS", "24"))
SUB_BATCH_DAYS = int(os.environ.get("SUB_BATCH_DAYS", "23"))
IO_THREADS = int(os.environ.get("IO_THREADS", "4"))

VAR_SST = "data"
VAR_LAT = "lat"
VAR_LON = "lon"


# ---------------------------------------------------------------------------
# 365-day calendar helpers (matches compute_server.py exactly)
# ---------------------------------------------------------------------------

def is_leap_year(yr):
    return (yr % 4 == 0 and yr % 100 != 0) or (yr % 400 == 0)


def build_365_calendar(year):
    d0 = datetime(year, 1, 1)
    if is_leap_year(year):
        dates = []
        for offset in range(366):
            d = d0 + timedelta(days=offset)
            if not (d.month == 2 and d.day == 29):
                dates.append(d)
    else:
        dates = [d0 + timedelta(days=i) for i in range(365)]
    return dates


def get_date_strings_for_year(year):
    start = datetime(year, *DATE_START_MMDD)
    end = datetime(year, *DATE_END_MMDD)
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def build_date_filepath_map():
    date_to_path = {}
    for fp in DATA_DIR.iterdir():
        name = fp.name
        if len(name) == 8 and name.isdigit():
            date_to_path[name] = str(fp)
    return date_to_path


# ---------------------------------------------------------------------------
# Raw byte I/O: bypass HDF5 parsing, read float64 bytes directly from disk
# ---------------------------------------------------------------------------

def _probe_raw_read(sample_path, sst_var, date_to_path):
    """Check if source NetCDF4/HDF5 files support raw byte reading.

    Returns dict(offset, dtype, shape, nbytes) or None.
    Raw read works only when the dataset is contiguous and uncompressed
    with a consistent byte offset across all files.
    """
    if not _HAS_H5PY:
        if rank == 0:
            print("    Raw I/O:         DISABLED (h5py not available)")
        return None

    # Pick 3 files from different months to verify consistency
    dates = sorted(date_to_path.keys())
    test_paths = []
    for ds in dates:
        fp = date_to_path[ds]
        if fp:
            test_paths.append(fp)
        if len(test_paths) >= 3:
            break

    offsets = []
    _dtype = None
    _shape = None
    for fp in test_paths:
        try:
            with h5py.File(fp, "r") as f:
                ds_obj = f[sst_var]
                if ds_obj.chunks is not None:
                    if rank == 0:
                        print(f"    Raw I/O:         DISABLED (chunked dataset: {ds_obj.chunks})")
                    return None
                if ds_obj.compression:
                    if rank == 0:
                        print(f"    Raw I/O:         DISABLED (compressed: {ds_obj.compression})")
                    return None
                off = ds_obj.id.get_offset()
                if off is None:
                    if rank == 0:
                        print("    Raw I/O:         DISABLED (no byte offset available)")
                    return None
                offsets.append(off)
                _dtype = ds_obj.dtype
                _shape = ds_obj.shape
        except Exception as e:
            if rank == 0:
                print(f"    Raw I/O:         DISABLED (h5py probe error: {e})")
            return None

    if len(set(offsets)) != 1:
        if rank == 0:
            print(f"    Raw I/O:         DISABLED (inconsistent offsets: {set(offsets)})")
        return None

    nbytes = int(np.prod(_shape)) * _dtype.itemsize

    try:
        with h5py.File(test_paths[0], "r") as f:
            ref = f[sst_var][:]
        with open(test_paths[0], "rb") as f:
            f.seek(offsets[0])
            raw = f.read(nbytes)
        # Try both native and byteswapped byte orders
        test_native = np.frombuffer(raw, dtype=_dtype).reshape(_shape)
        if np.array_equal(ref, test_native, equal_nan=True):
            return {"offset": offsets[0], "dtype": _dtype, "shape": _shape, "nbytes": nbytes}

        test_swapped = np.frombuffer(raw, dtype=_dtype.newbyteorder()).reshape(_shape)
        if np.array_equal(ref, test_swapped, equal_nan=True):
            if rank == 0:
                print(f"    Raw I/O:         ENABLED (offset={offsets[0]}, "
                      f"{nbytes} bytes, byteswapped read)")
            return {"offset": offsets[0], "dtype": _dtype.newbyteorder(),
                    "shape": _shape, "nbytes": nbytes}

        if rank == 0:
            max_diff = np.nanmax(np.abs(ref.astype(float) - test_native.astype(float)))
            print(f"    Raw I/O:         DISABLED (mismatch: max_diff={max_diff:.6f}, "
                  f"ref[:2,:2]={ref[:2,:2].tolist()}, "
                  f"raw[:2,:2]={test_native[:2,:2].tolist()})")
            return None
    except Exception as e:
        if rank == 0:
            print(f"    Raw I/O:         DISABLED (validation error: {e})")
        return None

    return {"offset": offsets[0], "dtype": _dtype, "shape": _shape, "nbytes": nbytes}


def raw_read_sst(fp, raw_info):
    """Read SST data using raw seek+read — bypasses HDF5 parsing entirely."""
    with open(fp, "rb") as f:
        f.seek(raw_info["offset"])
        raw = f.read(raw_info["nbytes"])
    return np.frombuffer(raw, dtype=raw_info["dtype"]).reshape(raw_info["shape"])


# ---------------------------------------------------------------------------
# Source format detection (rank 0 only → broadcast results)
# ---------------------------------------------------------------------------

def detect_source_format(date_to_path):
    sample_path = None
    for date_str in sorted(date_to_path.keys()):
        fp = date_to_path[date_str]
        if fp:
            sample_path = fp
            break

    _default = {"packed": False, "scale_factor": 0.0, "add_offset": 0.0,
                 "fill_value": None, "sst_var": VAR_SST, "dtype": "float32",
                 "lats": None, "lons": None, "sample_path": None}

    if sample_path is None:
        if rank == 0:
            print("  WARNING: No files to probe — assuming float32")
        return _default

    sst_var = None
    with Dataset(sample_path, "r") as nc:
        for name in [VAR_SST, "sst"]:
            if name in nc.variables:
                sst_var = name
                break

    if sst_var is None:
        if rank == 0:
            print("  WARNING: Cannot find SST variable — assuming float32")
        return _default

    with Dataset(sample_path, "r") as nc:
        v = nc.variables[sst_var]
        dtype = str(v.dtype)
        scale_factor = getattr(v, "scale_factor", None)
        add_offset = getattr(v, "add_offset", None)
        fill_value = getattr(v, "_FillValue", None)
        shape = v.shape
        lats = nc.variables[VAR_LAT][:].copy()
        lons = nc.variables[VAR_LON][:].copy()

    is_int16 = dtype.startswith("int16")
    packed = is_int16 and scale_factor is not None

    if rank == 0:
        print(f"\n  Source format probe ({sample_path}):")
        print(f"    SST variable:   {sst_var}")
        print(f"    Storage dtype:  {dtype}")
        print(f"    Shape:          {shape}")
        print(f"    Scale factor:   {scale_factor}")
        print(f"    Add offset:     {add_offset}")
        print(f"    Fill value:     {fill_value}")
        print(f"    Lat: {len(lats)}  [{lats[0]:.4f} .. {lats[-1]:.4f}]")
        print(f"    Lon: {len(lons)}  [{lons[0]:.4f} .. {lons[-1]:.4f}]")
        if packed:
            float32_size = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * 4 / 1024**3
            int16_size = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * 2 / 1024**3
            print(f"    PACKED READ ENABLED: I/O {float32_size:.1f} GB → {int16_size:.1f} GB "
                  f"(-{float32_size - int16_size:.1f} GB)")
        else:
            print(f"    Standard float32 read (no packed benefit)")

    # Probe raw byte read capability
    raw_info = _probe_raw_read(sample_path, sst_var, date_to_path)
    if raw_info and rank == 0:
        print(f"    Raw I/O:         ENABLED (offset={raw_info['offset']}, "
              f"{raw_info['nbytes']} bytes, direct seek+read)")

    return {
        "packed": packed,
        "scale_factor": scale_factor if scale_factor is not None else 0.0,
        "add_offset": add_offset if add_offset is not None else 0.0,
        "fill_value": fill_value,
        "sst_var": sst_var if sst_var is not None else VAR_SST,
        "dtype": dtype,
        "lats": lats,
        "lons": lons,
        "sample_path": sample_path,
        "raw_info": raw_info,
    }


# ---------------------------------------------------------------------------
# Backend detection (rank 0 only)
# ---------------------------------------------------------------------------

def _detect_backend():
    print("  Probing accelerator backends ...\n")

    # ── Path 1: PyTorch ROCm/DCU ──
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
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                props = torch.cuda.get_device_properties(device)
                name = props.name.lower()
                if "amd" in name or "gfx" in name or "radeon" in name:
                    mem_gb = props.total_memory / 1024**3
                    print(f"  [PyTorch] Generic CUDA (AMD GPU)  |  {props.name}  |  "
                          f"{mem_gb:.1f} GB")
                    return "torch", device
                else:
                    mem_gb = props.total_memory / 1024**3
                    print(f"  [PyTorch] Generic CUDA (NVIDIA)  |  {props.name}  |  "
                          f"{mem_gb:.1f} GB")
                    return "torch", device
            else:
                print("  [PyTorch] installed but no CUDA/ROCm support detected")
    except ImportError:
        print("  [PyTorch] not installed")
    except Exception as e:
        print(f"  [PyTorch] error during detection: {e}")

    # ── Path 2: HIP native ──
    try:
        import hip
        n_dev = hip.getDeviceCount()
        if n_dev > 0:
            props = hip.getDeviceProperties(0)
            mem_gb = props.totalGlobalMem / 1024**3
            print(f"  [HIP] {n_dev} device(s)  |  device 0: {props.name}  ({mem_gb:.1f} GB)")
    except ImportError:
        print("  [HIP] pyhip / hip-python not installed (optional)")
    except Exception as e:
        print(f"  [HIP] error during detection: {e}")

    # ── Path 3: CuPy ROCm ──
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

    print("\n  [CPU] No accelerator detected, will use NumPy (CPU)")
    return "cpu", None


# ---------------------------------------------------------------------------
# PyTorch compute kernels (identical to compute_server.py)
# ---------------------------------------------------------------------------

def _p90_fast_torch(t, dim=1):
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
    import torch

    # Warmup float16 — matches actual compute path
    warmup = torch.randn(10, 300, 10, 100, device=device, dtype=torch.float16)
    _p90_fast_torch(warmup, dim=1)
    torch.nanmean(warmup, dim=1)
    del warmup

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    for band_idx, (r0, r1) in enumerate(bands):
        t0 = time.time()
        band_rows = r1 - r0

        band_t = torch.from_numpy(data_np[:, :, r0:r1, :]).to(device)

        if packed:
            band_t = band_t.to(torch.float32)
            mask = band_t == fill_value
            band_t.mul_(scale_factor).add_(add_offset)
            band_t[mask] = float('nan')

        # unfold → permute → reshape (single contiguous copy, same approach as ls/new)
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

        elapsed = time.time() - t0
        print(f"  Band {band_idx+1}/{len(bands)} rows [{r0}:{r1}] done ({elapsed:.1f}s)")

    return threshold, climatology


def _compute_torch_multi_gpu(data_np, num_gpus, bands, packed=False,
                             scale_factor=None, add_offset=None, fill_value=None):
    import torch
    import threading

    band_groups = [[] for _ in range(num_gpus)]
    for i, band in enumerate(bands):
        band_groups[i % num_gpus].append(band)

    print(f"  Multi-GPU ({num_gpus} GPUs):")
    for gpu_id, group in enumerate(band_groups):
        print(f"    GPU {gpu_id}: {len(group)} bands {group}")

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    def _thread_worker(gpu_id, group):
        torch.cuda.set_device(gpu_id)

        # Warmup float16 — matches actual compute path
        warmup = torch.randn(10, 300, 10, 100, device=gpu_id, dtype=torch.float16)
        _p90_fast_torch(warmup, dim=1)
        torch.nanmean(warmup, dim=1)
        del warmup

        for r0, r1 in group:
            t0 = time.time()
            band_rows = r1 - r0

            # No .copy() — slice is contiguous, torch.from_numpy zero-copies
            band_t = torch.from_numpy(data_np[:, :, r0:r1, :]).to(gpu_id)

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

            elapsed = time.time() - t0
            print(f"  [GPU {gpu_id}] Band rows [{r0}:{r1}] done ({elapsed:.1f}s)")

        if gpu_id == 0:
            print(f"  [GPU 0] All bands done")

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
# CuPy backend (identical to compute_server.py)
# ---------------------------------------------------------------------------

def _compute_cupy_band(data_np, device_id, bands, packed=False,
                       scale_factor=None, add_offset=None, fill_value=None):
    import cupy as cp

    cp.cuda.Device(device_id).use()

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    for band_idx, (r0, r1) in enumerate(bands):
        t0 = time.time()
        band_rows = r1 - r0

        band_data = data_np[:, :, r0:r1, :].copy()
        data_g = cp.asarray(band_data, dtype=cp.float32)

        if packed:
            data_g = data_g.astype(cp.float32)
            mask = data_g == fill_value
            data_g *= scale_factor
            data_g += add_offset
            data_g[mask] = cp.nan

        windows_g = cp.empty((N_OUTPUT_DAYS, N_YEARS * WINDOW_SIZE, band_rows, N_LON),
                             dtype=cp.float32)
        for i in range(N_OUTPUT_DAYS):
            w = data_g[:, i:i + WINDOW_SIZE, :, :]
            windows_g[i] = w.reshape(N_YEARS * WINDOW_SIZE, band_rows, N_LON)

        del data_g
        cp.get_default_memory_pool().free_all_blocks()

        windows_g.sort(axis=1)

        valid_count = (~cp.isnan(windows_g)).sum(axis=1)

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

        nan_mask = cp.isnan(windows_g)
        windows_clean = cp.where(nan_mask, 0.0, windows_g)
        band_clim = windows_clean.sum(axis=1) / cp.maximum(valid_count, 1)

        threshold[:, r0:r1, :] = cp.asnumpy(band_thresh.astype(cp.float32))
        climatology[:, r0:r1, :] = cp.asnumpy(band_clim.astype(cp.float32))

        del windows_g, band_thresh, band_clim
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        elapsed = time.time() - t0
        print(f"  Band {band_idx+1}/{len(bands)} rows [{r0}:{r1}] done ({elapsed:.1f}s)")

    return threshold, climatology


# ---------------------------------------------------------------------------
# CPU fallback (identical to compute_server.py)
# ---------------------------------------------------------------------------

def _compute_cpu(data_np, packed=False, scale_factor=None,
                 add_offset=None, fill_value=None):
    t0 = time.time()

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    for i in range(N_OUTPUT_DAYS):
        window = data_np[:, i:i + WINDOW_SIZE, :, :]

        if packed:
            window = window.astype(np.float32)
            mask = window == fill_value
            window *= scale_factor
            window += add_offset
            window[mask] = np.nan

        flat = window.reshape(N_YEARS * WINDOW_SIZE, N_LAT, N_LON)

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
# Latitude bands
# ---------------------------------------------------------------------------

def build_bands():
    base = N_LAT // N_BANDS
    rem = N_LAT % N_BANDS
    bands = []
    r0 = 0
    for i in range(N_BANDS):
        size_band = base + (1 if i < rem else 0)
        bands.append((r0, r0 + size_band))
        r0 += size_band
    return bands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bcast_str(s):
    """Broadcast a Python string from rank 0."""
    # Pack string as fixed-length array
    if rank == 0:
        encoded = s.encode("utf-8")
        n = np.array(len(encoded), dtype=np.int32)
    else:
        n = np.empty(1, dtype=np.int32)
    comm.Bcast(n, root=0)
    if rank == 0:
        buf = np.frombuffer(encoded, dtype=np.uint8)
    else:
        buf = np.empty(n[0], dtype=np.uint8)
    comm.Bcast(buf, root=0)
    return buf.tobytes().decode("utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    prep_start = time.time()

    # ---- Rank 0: Prep & format detection ----
    if rank == 0:
        print(f"MPI mode: {size} ranks")
        print("Building file index ...")
    date_to_path = build_date_filepath_map()
    if rank == 0:
        print(f"  Indexed {len(date_to_path)} files")

    # Detect source format (all ranks probe independently — fast, 1 file)
    fmt = detect_source_format(date_to_path)
    packed = fmt["packed"]
    scale_factor = fmt["scale_factor"]
    add_offset = fmt["add_offset"]
    fill_value = fmt["fill_value"]
    sst_var = fmt["sst_var"]
    raw_info = fmt.get("raw_info")

    if packed and fill_value is None:
        fill_value = -32768
        if rank == 0:
            print(f"  NOTE: No _FillValue found — using {fill_value} as fill sentinel")

    # Data array dtype
    if packed:
        data_dtype = np.int16
        elem_bytes = 2
        nan_fill = fill_value if fill_value is not None else -32768
        packed_tag = "int16-packed"
    else:
        data_dtype = np.float16
        elem_bytes = 2
        nan_fill = np.nan
        packed_tag = "float16"

    shape = (N_YEARS, N_DAYS_PER_YEAR, N_LAT, N_LON)
    size_gb = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * elem_bytes / 1024**3
    if rank == 0:
        if packed:
            print(f"  Scale: raw_int16 * {scale_factor} + {add_offset}")
            print(f"  Fill value masking: raw == {fill_value} → NaN")
        print(f"  Data array {shape} {packed_tag} ({size_gb:.2f} GB)")

    # ---- Build year tasks (all ranks) ----
    needed_dates = []
    for y in range(START_YEAR, END_YEAR + 1):
        needed_dates.extend(get_date_strings_for_year(y))
    by_year = defaultdict(list)
    for j, ds in enumerate(needed_dates):
        fp = date_to_path.get(ds)
        if fp is not None:
            by_year[ds[:4]].append((j, fp))

    # ---- Each rank computes its assigned years ----
    my_years = sorted([yr for yr in by_year.keys()
                       if (int(yr) - START_YEAR) % size == rank])
    n_my_years = len(my_years)

    if rank == 0:
        print(f"\n  Year distribution: {size} ranks for {len(by_year)} years")
        for r in range(size):
            r_years = sorted([yr for yr in by_year.keys()
                             if (int(yr) - START_YEAR) % size == r])
            if r_years:
                print(f"    rank {r:2d}: years {r_years[0]}..{r_years[-1]} ({len(r_years)} years)")

    # ---- Backend detection (rank 0 only, before I/O) ----
    if rank == 0:
        print(f"\nDetecting accelerator backend ...")
        setup_start = time.time()
        backend, device = _detect_backend()
        setup_time = time.time() - setup_start
    else:
        backend, device, setup_time = None, None, 0.0
    # Broadcast backend choice
    backend = comm.bcast(backend, root=0)
    # device is either torch.device or int; string is fine to bcast
    if backend == "torch":
        if rank == 0:
            device_str = str(device)  # "cuda" or "cuda:0"
        else:
            device_str = None
        device_str = comm.bcast(device_str, root=0)
        if rank != 0:
            import torch
            device = torch.device(device_str)
    elif backend == "cupy":
        device = comm.bcast(device, root=0)
    setup_time = comm.bcast(setup_time, root=0)

    # ---- Lat/lon coords (from format detection) ----
    lats = fmt["lats"]
    lons = fmt["lons"]

    # ---- MPI Distributed I/O ----
    if rank == 0:
        io_label = f"{IO_THREADS} threads" if IO_THREADS > 1 else "1 thread"
        print(f"\nReading {n_my_years} years per rank "
              f"(MPI distributed, {packed_tag}"
              f"{', raw I/O' if raw_info else ', netCDF4'}, {io_label}) ...")
    comm.Barrier()
    io_start = time.time()

    # Each rank reads its own years — no pickle, no inter-process communication
    local_data = np.full((n_my_years, N_DAYS_PER_YEAR, N_LAT, N_LON),
                         nan_fill, dtype=data_dtype)

    # Flatten all (year_idx, idx_slot, fp) for this rank
    all_items = []
    for i, year in enumerate(my_years):
        for idx_slot, fp in by_year[year]:
            all_items.append((i, idx_slot, fp))

    def _read_chunk(items_chunk):
        """Read a chunk of files and write directly into local_data."""
        for yr_i, idx_slot, fp in items_chunk:
            if raw_info is not None:
                block = raw_read_sst(fp, raw_info)
            else:
                with Dataset(fp, "r") as ds:
                    block = ds.variables[sst_var][:]
            block = np.squeeze(block)
            if block.shape != (N_LAT, N_LON):
                block = block.T
            day_idx = idx_slot % N_DAYS_PER_YEAR
            local_data[yr_i, day_idx] = np.asarray(block, dtype=data_dtype)

    n_items = len(all_items)
    if IO_THREADS > 1 and n_items > IO_THREADS:
        chunk_size = (n_items + IO_THREADS - 1) // IO_THREADS
        with ThreadPoolExecutor(max_workers=IO_THREADS) as pool:
            futures = []
            for t in range(IO_THREADS):
                start = t * chunk_size
                end = min(start + chunk_size, n_items)
                if start < end:
                    futures.append(pool.submit(_read_chunk, all_items[start:end]))
            for f in futures:
                f.result()
    else:
        _read_chunk(all_items)

    if rank == 0:
        print(f"  Rank 0 local read done ({time.time() - io_start:.1f}s)")

    comm.Barrier()
    io_read_done = time.time()
    if rank == 0:
        print(f"  All ranks read done ({io_read_done - io_start:.1f}s)")

    # ---- Gather data to rank 0 via MPI shared memory (zero-copy) ----
    # On a single node, MPI_Win_allocate_shared creates RAM shared by all ranks.
    # Each rank writes directly into rank 0's section — no message passing.
    total_bytes = shape[0] * shape[1] * shape[2] * shape[3] * elem_bytes
    win_size = total_bytes if rank == 0 else 0
    win = MPI.Win.Allocate_shared(win_size, 1, comm=comm)
    shm_mem, _ = win.Shared_query(0)
    data = np.ndarray(shape, dtype=data_dtype, buffer=shm_mem)

    # Each rank writes its years directly into the shared data array
    for i, year in enumerate(my_years):
        yr_idx = int(year) - START_YEAR
        data[yr_idx] = local_data[i]

    del local_data
    gc.collect()

    # Barrier ensures all writes complete before rank 0 reads
    comm.Barrier()
    gather_elapsed = time.time() - io_read_done

    io_elapsed = time.time() - io_start
    if rank == 0:
        print(f"  MPI I/O total: {io_elapsed:.1f}s "
              f"(read {io_read_done - io_start:.1f}s + gather {gather_elapsed:.1f}s)")

    # ---- Only rank 0 does compute and save ----
    if rank != 0:
        return

    # Copy data from shared memory to local heap memory.
    # Shared memory cross-rank access can trigger NUMA misses / page faults
    # that slow down GPU compute by 3-4x. A one-time copy (~1s) saves 10-15s
    # during the multiday compute loop.
    t_copy = time.time()
    data = data.copy()
    del win
    gc.collect()
    if rank == 0:
        print(f"  Local copy: {time.time() - t_copy:.1f}s")

    # ---- Compute (rank 0 only, same as compute_server.py) ----
    print(f"\nComputing threshold & climatology "
          f"({N_OUTPUT_DAYS} days x 30-year sliding window) ...")

    compute_start = time.time()
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

    del data
    gc.collect()

    compute_elapsed = time.time() - compute_start
    print(f"  Compute: {compute_elapsed:.1f}s")

    # ---- Save ----
    print("\nSaving ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_dates = build_365_calendar(2020)
    for out_idx in range(N_OUTPUT_DAYS):
        doy = 147 + WINDOW_HALF + out_idx
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

    compute_end = time.time()
    save_elapsed = compute_end - (compute_start + compute_elapsed)

    prep_end = io_start  # prep runs until I/O starts
    prep_elapsed = prep_end - prep_start - setup_time
    total_elapsed = io_elapsed + compute_elapsed + save_elapsed

    print(f"\n{'='*60}")
    print(f"  TIMING BREAKDOWN (MPI, {packed_tag})")
    print(f"{'='*60}")
    print(f"  Prep:       {prep_elapsed:8.1f}s          (file index + format probe)")
    print(f"  Setup:      {setup_time:8.1f}s          (before I/O, not counted)")
    print(f"  I/O:        {io_elapsed:8.1f}s  (MPI distributed)")
    print(f"  Compute:    {compute_elapsed:8.1f}s")
    print(f"  Save:       {save_elapsed:8.1f}s")
    print(f"  {'─'*50}")
    print(f"  Total:      {total_elapsed:8.3f}s  ({total_elapsed/60:.2f} min)")
    print(f"{'='*60}")

    print(f"real {int(total_elapsed//60)}m{total_elapsed%60:.3f}s  (I/O+Compute+Save)")
    print("Done!")

    # ---- Validation ----
    print(f"\nValidation:")
    print(f"  P90_sst range:       [{np.nanmin(threshold):.2f}, {np.nanmax(threshold):.2f}]")
    print(f"  Climmean range:      [{np.nanmin(climatology):.2f}, {np.nanmax(climatology):.2f}]")
    print(f"  NaN fraction (P90):  {np.isnan(threshold).mean()*100:.1f}%")
    print(f"  P90 > Climmean (mean): {np.nanmean(threshold) > np.nanmean(climatology)}")


if __name__ == "__main__":
    _wall_start = time.time()
    main()
