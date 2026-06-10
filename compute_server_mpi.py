#!/usr/bin/env python
"""compute_server_mpi.py — MPI distributed I/O + PyTorch ROCm multi-GPU compute.

MPI (30 ranks): raw byte I/O → direct-to-shm → rank 0 GPU compute.
4×DCU Z200 parallel: unfold + topk P90 + nanmean, float16 throughout.

Usage:
  N_BANDS=12 SUB_BATCH_DAYS=46 USE_MPI=1 bash get_climatology.sh
"""

import numpy as np
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from netCDF4 import Dataset
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    """Glob DATA_DIR once → {YYYYMMDD: path} dict. Zero stat calls in I/O loop."""
    return {fp.name: str(fp) for fp in DATA_DIR.iterdir()
            if len(fp.name) == 8 and fp.name.isdigit()}


# ── Raw byte I/O ───────────────────────────────────────────────────────────

def _probe_raw_read(date_to_path):
    """Check if source files support raw seek+read (contiguous, uncompressed).

    Returns dict(offset, dtype, shape, nbytes) or None.
    """
    if not _HAS_H5PY:
        if rank == 0:
            print("    Raw I/O:         DISABLED (h5py not available)")
        return None

    # Verify consistency across 3 sample files
    test_paths = [date_to_path[ds] for ds in sorted(date_to_path)[:3]
                  if date_to_path.get(ds)]
    offsets, dtype, shape = [], None, None
    for fp in test_paths:
        try:
            with h5py.File(fp, "r") as f:
                ds = f["data"]
                if ds.chunks is not None:
                    if rank == 0:
                        print(f"    Raw I/O:         DISABLED (chunked: {ds.chunks})")
                    return None
                if ds.compression:
                    if rank == 0:
                        print(f"    Raw I/O:         DISABLED (compressed)")
                    return None
                off = ds.id.get_offset()
                if off is None:
                    if rank == 0:
                        print("    Raw I/O:         DISABLED (no byte offset)")
                    return None
                offsets.append(off)
                dtype, shape = ds.dtype, ds.shape
        except Exception as e:
            if rank == 0:
                print(f"    Raw I/O:         DISABLED (probe error: {e})")
            return None

    if len(set(offsets)) != 1:
        if rank == 0:
            print(f"    Raw I/O:         DISABLED (inconsistent offsets)")
        return None

    nbytes = int(np.prod(shape)) * dtype.itemsize

    # Validate: raw read must match h5py read
    try:
        with h5py.File(test_paths[0], "r") as f:
            ref = f["data"][:]
        with open(test_paths[0], "rb") as f:
            f.seek(offsets[0])
            raw = f.read(nbytes)
        for dt in (dtype, dtype.newbyteorder()):
            test = np.frombuffer(raw, dtype=dt).reshape(shape)
            if np.array_equal(ref, test, equal_nan=True):
                if rank == 0:
                    print(f"    Raw I/O:         ENABLED (offset={offsets[0]}, "
                          f"{nbytes} bytes, direct seek+read)")
                return {"offset": offsets[0], "dtype": dt, "shape": shape, "nbytes": nbytes}
        if rank == 0:
            print("    Raw I/O:         DISABLED (validation mismatch)")
    except Exception as e:
        if rank == 0:
            print(f"    Raw I/O:         DISABLED (validation error: {e})")
    return None


def raw_read_sst(fp, raw_info):
    """Read one SST file via raw seek+read — bypasses HDF5/netCDF4 entirely."""
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
        if rank == 0:
            print("  WARNING: No files to probe")
        return {"sst_var": "data", "lats": None, "lons": None, "raw_info": None}

    # Find SST variable name
    with Dataset(sample, "r") as nc:
        sst_var = next((n for n in ["data", "sst"] if n in nc.variables), None)
        if sst_var is None:
            if rank == 0:
                print("  WARNING: Cannot find SST variable")
            return {"sst_var": "data", "lats": None, "lons": None, "raw_info": None}
        dtype = str(nc.variables[sst_var].dtype)
        shape = nc.variables[sst_var].shape
        lats = nc.variables["lat"][:].copy()
        lons = nc.variables["lon"][:].copy()

    if rank == 0:
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

# ── HIP fused kernel compilation (called once, before timing) ──────────

def _load_fused_kernel():
    """Compile or load cached HIP kernel (nanmean + P90 in one pass).
    Returns callable(_fused_fn) or None on failure.
    Must be called on rank 0 before compute timing starts.
    """
    import torch

    _kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".fused_kernel")
    try:
        hip_src = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

extern "C" __global__ void fused_stats_kernel(
    const __half* __restrict__ sub,   // (n_days, 330, rows, n_lon) f16
    float* __restrict__ p90_out,      // (n_days, rows, n_lon) f32
    float* __restrict__ mean_out,     // (n_days, rows, n_lon) f32
    long long n_days, long long n_total, long long rows, long long n_lon
) {
    long long lon = blockIdx.x * blockDim.x + threadIdx.x;
    long long day_lat = blockIdx.y;
    long long day = day_lat / rows;
    long long lat = day_lat % rows;

    if (lon >= n_lon) return;

    // Use 64-bit to avoid int32 overflow: day * n_total * rows * n_lon > 2^31
    const __half* base = sub + day * (n_total * rows * n_lon) + lat * n_lon + lon;
    long long stride = rows * n_lon;

    // top-34 in descending order, registers only
    __half top[34];
    #pragma unroll
    for (int i = 0; i < 34; i++) top[i] = __float2half(-__builtin_huge_valf());

    float sum = 0.0f;
    long long count = 0;

    for (long long s = 0; s < n_total; s++) {
        __half v = base[s * stride];
        if (__hisnan(v)) {
            v = __float2half(__builtin_huge_valf());
        } else {
            sum += __half2float(v);
            count++;
        }
        if (__hgt(v, top[33])) {
            int pos = 33;
            while (pos > 0 && __hgt(v, top[pos - 1])) {
                top[pos] = top[pos - 1];
                pos--;
            }
            top[pos] = v;
        }
    }

    long long out_idx = day * (rows * n_lon) + lat * n_lon + lon;
    mean_out[out_idx] = (count > 0) ? sum / (float)count : __builtin_nanf("");

    float v0 = __half2float(top[33]);
    float v1 = __half2float(top[32]);
    float p90 = v0 + (v1 - v0) * 0.1f;
    p90_out[out_idx] = __isinff(v0) ? __builtin_nanf("") : p90;
}

void launch_fused_stats(
    torch::Tensor sub,
    torch::Tensor p90_out,
    torch::Tensor mean_out)
{
    long long n_days = sub.size(0);
    long long n_total = sub.size(1);
    long long rows = sub.size(2);
    long long n_lon = sub.size(3);

    constexpr int BLOCK_LON = 256;
    dim3 block(BLOCK_LON);
    dim3 grid((n_lon + BLOCK_LON - 1) / BLOCK_LON, n_days * rows);

    fused_stats_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __half*>(sub.data_ptr<at::Half>()),
        p90_out.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        n_days, n_total, rows, n_lon);
}
"""

        cpp_src = r"""
#include <torch/extension.h>
void launch_fused_stats(torch::Tensor sub, torch::Tensor p90_out, torch::Tensor mean_out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_stats", &launch_fused_stats);
}
"""

        os.makedirs(_kernel_dir, exist_ok=True)
        _hip_path = os.path.join(_kernel_dir, "fused_stats.hip")
        _cpp_path = os.path.join(_kernel_dir, "fused_stats.cpp")
        _hash_path = os.path.join(_kernel_dir, ".source_hash")

        # Only rewrite sources if kernel code changed (avoid ninja recompile)
        import hashlib
        _new_hash = hashlib.sha256((hip_src + cpp_src).encode()).hexdigest()
        _old_hash = None
        if os.path.exists(_hash_path):
            with open(_hash_path) as f:
                _old_hash = f.read().strip()

        if _new_hash != _old_hash:
            # Kernel changed — blow away cache and rewrite sources
            import shutil
            for _p in [_hip_path, _cpp_path]:
                if os.path.exists(_p):
                    os.remove(_p)
            with open(_hip_path, "w") as f:
                f.write(hip_src)
            with open(_cpp_path, "w") as f:
                f.write(cpp_src)
            with open(_hash_path, "w") as f:
                f.write(_new_hash)

        from torch.utils.cpp_extension import load
        _mod = load(name="fused_stats", sources=[_cpp_path, _hip_path],
                    build_directory=_kernel_dir, verbose=False)
        print("  Fused kernel:   ENABLED (HIP: nanmean + P90 in one pass)")
        return _mod.fused_stats
    except Exception as e:
        print(f"  Fused kernel:   FALLBACK to topk ({e})")
        return None


def _compute_torch(data_np, bands, fused_fn):
    """Multi-GPU threaded compute with HIP-fused P90+mean kernel."""
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

    # ── Fallback P90 ────────────────────────────────────────────────────
    def _p90_fallback(sub):
        sub.nan_to_num_(nan=float("inf"))
        topvals = torch.topk(sub, k=34, dim=1, largest=True, sorted=True).values
        p90 = topvals[:, -1, :, :] + (topvals[:, -2, :, :] - topvals[:, -1, :, :]) * 0.1
        return torch.where(torch.isinf(p90), float("nan"), p90)

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    def _worker(gpu_id, group):
        if not group:
            return
        torch.cuda.set_device(gpu_id)

        for r0, r1 in group:
            t0 = time.time()

            band_t = torch.from_numpy(
                data_np[:, :, r0:r1, :].copy()).to(gpu_id)
            windows = band_t.unfold(dimension=1, size=WINDOW_SIZE, step=1)

            with torch.inference_mode():
                for d0 in range(0, N_OUTPUT_DAYS, SUB_BATCH_DAYS):
                    d1 = min(d0 + SUB_BATCH_DAYS, N_OUTPUT_DAYS)
                    n_days = d1 - d0
                    rows = r1 - r0

                    sub = windows[:, d0:d1].permute(1, 0, 4, 2, 3).reshape(
                        n_days, N_YEARS * WINDOW_SIZE, rows, N_LON)

                    if fused_fn is not None:
                        p90_gpu = torch.empty(n_days, rows, N_LON,
                                              dtype=torch.float32, device=gpu_id)
                        mean_gpu = torch.empty(n_days, rows, N_LON,
                                               dtype=torch.float32, device=gpu_id)
                        fused_fn(sub, p90_gpu, mean_gpu)
                        threshold[d0:d1, r0:r1, :] = p90_gpu.cpu().numpy()
                        climatology[d0:d1, r0:r1, :] = mean_gpu.cpu().numpy()
                        del p90_gpu, mean_gpu
                    else:
                        climatology[d0:d1, r0:r1, :] = (
                            torch.nanmean(sub, dim=1).cpu().numpy().astype(np.float32))
                        threshold[d0:d1, r0:r1, :] = (
                            _p90_fallback(sub).cpu().numpy().astype(np.float32))

                    del sub

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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t_prep = time.time()

    # ── Prep: file index & format probe ──
    if rank == 0:
        print(f"MPI mode: {size} ranks")
        print("Building file index ...")
    date_to_path = build_date_filepath_map()
    if rank == 0:
        print(f"  Indexed {len(date_to_path)} files")

    fmt = detect_source_format(date_to_path)
    sst_var = fmt["sst_var"]
    lats, lons = fmt["lats"], fmt["lons"]
    raw_info = fmt["raw_info"]

    shape = (N_YEARS, N_DAYS_PER_YEAR, N_LAT, N_LON)
    size_gb = N_YEARS * N_DAYS_PER_YEAR * N_LAT * N_LON * 2 / 1024**3  # float16
    if rank == 0:
        print(f"  Data array {shape} float16 ({size_gb:.2f} GB)")

    # ── Build year→files mapping ──
    needed_dates = []
    for y in range(START_YEAR, END_YEAR + 1):
        needed_dates.extend(get_date_strings_for_year(y))
    by_year = defaultdict(list)
    for j, ds in enumerate(needed_dates):
        fp = date_to_path.get(ds)
        if fp is not None:
            by_year[ds[:4]].append((j, fp))

    # ── Assign years to ranks ──
    my_years = sorted(yr for yr in by_year
                      if (int(yr) - START_YEAR) % size == rank)

    if rank == 0:
        print(f"\n  Year distribution: {size} ranks for {len(by_year)} years")
        for r in range(size):
            r_years = sorted(yr for yr in by_year
                            if (int(yr) - START_YEAR) % size == r)
            if r_years:
                print(f"    rank {r:2d}: {r_years[0]}..{r_years[-1]} "
                      f"({len(r_years)} year(s))")

    # ── Backend detection + kernel compile (rank 0, before I/O timer) ──
    if rank == 0:
        print("\nDetecting accelerator backend ...")
        t_setup = time.time()
        num_gpus = _detect_backend()
        fused_fn = _load_fused_kernel()
        setup_time = time.time() - t_setup
    else:
        num_gpus, setup_time = 0, 0.0
        fused_fn = None
    num_gpus = comm.bcast(num_gpus, root=0)
    setup_time = comm.bcast(setup_time, root=0)

    # ── MPI Distributed I/O (direct-to-shm, zero gather) ──
    total_bytes = shape[0] * shape[1] * shape[2] * shape[3] * 2  # float16 = 2 bytes
    win_size = total_bytes if rank == 0 else 0
    win = MPI.Win.Allocate_shared(win_size, 1, comm=comm)
    shm_mem, _ = win.Shared_query(0)
    data = np.ndarray(shape, dtype=np.float16, buffer=shm_mem)

    if rank == 0:
        io_label = f"{IO_THREADS} threads" if IO_THREADS > 1 else "1 thread"
        raw_label = "raw I/O" if raw_info else "netCDF4"
        print(f"\nReading {len(my_years)} years per rank "
              f"(MPI distributed, float16, {raw_label}, {io_label}, "
              f"direct-to-shm) ...")
    comm.Barrier()
    t_io = time.time()

    # Flatten assigned work
    items = [(int(yr) - START_YEAR, slot, fp)
             for yr in my_years
             for slot, fp in by_year[yr]]

    def _read_chunk(chunk):
        for yr_idx, idx_slot, fp in chunk:
            block = (raw_read_sst(fp, raw_info) if raw_info
                     else Dataset(fp, "r").variables[sst_var][:])
            block = np.squeeze(block)
            if block.shape != (N_LAT, N_LON):
                block = block.T
            data[yr_idx, idx_slot % N_DAYS_PER_YEAR] = \
                np.asarray(block, dtype=np.float16)

    if IO_THREADS > 1 and len(items) > IO_THREADS:
        n_per = (len(items) + IO_THREADS - 1) // IO_THREADS
        with ThreadPoolExecutor(max_workers=IO_THREADS) as pool:
            futures = [pool.submit(_read_chunk, items[t * n_per:(t + 1) * n_per])
                       for t in range(IO_THREADS) if t * n_per < len(items)]
            for f in futures:
                f.result()
    else:
        _read_chunk(items)

    comm.Barrier()
    io_elapsed = time.time() - t_io
    if rank == 0:
        print(f"  MPI I/O total: {io_elapsed:.1f}s (no gather — direct to shm)")

    # ── Rank > 0: done ──
    if rank != 0:
        return

    # ── Skip full shm→heap copy; band-level .copy() handles contiguous DMA ──
    # Keep MPI window alive so data buffer stays valid during compute
    gc.collect()

    # ── Compute ──
    print(f"\nComputing threshold & climatology "
          f"({N_OUTPUT_DAYS} days x 30-year sliding window) ...")
    bands = build_bands()
    t_compute = time.time()
    threshold, climatology = _compute_torch(data, bands, fused_fn)
    compute_elapsed = time.time() - t_compute
    print(f"  Compute: {compute_elapsed:.1f}s")

    del data; gc.collect()

    # ── Save ──
    print("\nSaving ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # DOY 152 = Jun 1 in a non-leap year (equivalent to 365-day calendar)
    d0_2019 = datetime(2019, 1, 1)  # 2019 is not a leap year
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
    total_elapsed = t_end - t_io

    print(f"  Saved {N_OUTPUT_DAYS} files to {OUT_DIR}/")

    # ── Timing breakdown ──
    prep_elapsed = t_io - t_prep - setup_time

    print(f"\n{'='*60}")
    print(f"  TIMING BREAKDOWN (MPI, float16)")
    print(f"{'='*60}")
    print(f"  Prep:       {prep_elapsed:8.1f}s   (file index + format probe)")
    print(f"  Setup:      {setup_time:8.1f}s   (backend + kernel compile)")
    print(f"  I/O:        {io_elapsed:8.1f}s   (MPI distributed)")
    print(f"  Compute:    {compute_elapsed:8.1f}s")
    print(f"  Save:       {t_end - t_compute - compute_elapsed:8.1f}s")
    print(f"  {'─'*50}")
    print(f"  Total:      {total_elapsed:8.3f}s  ({total_elapsed/60:.2f} min)")
    print(f"    (I/O start → save end, wall clock)")
    grand_total = setup_time + total_elapsed
    print(f"  Grand:      {grand_total:8.3f}s  ({grand_total/60:.2f} min)")
    print(f"    (setup + I/O + compute + save)")
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
