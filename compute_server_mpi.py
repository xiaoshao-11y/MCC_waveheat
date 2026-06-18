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
import time
import torch  # environment loading (excluded from competition timing per rules)
torch.cuda.init()  # HIP runtime init (environment loading, excluded)
_t0 = time.time()
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
IO_THREADS = int(os.environ.get("IO_THREADS", "2"))

FLAGS_PAD = 64
MAX_RANKS = 64
FLAGS_COUNT = N_DAYS_PER_YEAR * MAX_RANKS  # 102 days × 64 max ranks — per-rank-per-day ready flags


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
    """Construct {YYYYMMDD: path} dict — zero filesystem ops (no iterdir)."""
    data_dir = str(DATA_DIR)
    result = {}
    for y in range(START_YEAR, END_YEAR + 1):
        for ds in get_date_strings_for_year(y):
            result[ds] = os.path.join(data_dir, ds)
    return result


# ── Raw byte I/O ───────────────────────────────────────────────────────────

def _probe_raw_read(test_paths):
    """Check if source files support raw seek+read (contiguous, uncompressed).

    Returns dict(offset, dtype, shape, nbytes) or None.
    """
    if not _HAS_H5PY:
        if rank == 0:
            print("    Raw I/O:         DISABLED (h5py not available)")
        return None

    # Verify consistency across sample files
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


def _read_netcdf_fallback(fp, sst_var):
    """Fallback: read SST via netCDF4 (only if raw I/O unavailable)."""
    from netCDF4 import Dataset
    return Dataset(fp, "r").variables[sst_var][:]


# ── Source format detection ────────────────────────────────────────────────

def detect_source_format(date_to_path):
    """Probe first available file for SST var name, lat/lon, and raw I/O info."""
    from netCDF4 import Dataset
    sample = sorted(date_to_path.values())[0]
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

    raw_info = _probe_raw_read(sorted(date_to_path.values())[:3])

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
    """Compile or load cached HIP kernels (fused_stats + streaming accumulate/finalize).
    Returns (fused_fn, accumulate_fn, finalize_fn) or (None, None, None) on failure.
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

// ── Streaming: accumulate one year into persistent top-34 state ──────────

extern "C" __global__ void accumulate_year_kernel(
    const __half* __restrict__ year_data,  // (102, rows, n_lon) f16 — single year
    __half* __restrict__ top34,            // (92, rows, n_lon, 34) f16 — persistent
    float* __restrict__ partial_sum,       // (92, rows, n_lon) f32
    int* __restrict__ partial_count,       // (92, rows, n_lon) i32
    long long n_days, long long rows, long long n_lon
) {
    long long lon = blockIdx.x * blockDim.x + threadIdx.x;
    long long day_lat = blockIdx.y;
    long long day = day_lat / rows;
    long long lat = day_lat % rows;

    if (lon >= n_lon || day >= n_days) return;

    long long yr_stride = rows * n_lon;
    long long out_stride = rows * n_lon;
    long long idx = day * out_stride + lat * n_lon + lon;
    long long top_base = idx * 34;

    // Load persistent state into registers (one-time global read)
    __half top[34];
    #pragma unroll
    for (int i = 0; i < 34; i++) top[i] = top34[top_base + i];
    float sum = partial_sum[idx];
    int cnt = partial_count[idx];

    // 11-day sliding window — all insertion in registers
    const int WINDOW = 11;
    #pragma unroll
    for (int w = 0; w < WINDOW; w++) {
        __half v = year_data[(day + w) * yr_stride + lat * n_lon + lon];
        if (__hisnan(v)) {
            v = __float2half(__builtin_huge_valf());
        } else {
            sum += __half2float(v);
            cnt++;
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

    // Write back persistent state (one-time global write)
    #pragma unroll
    for (int i = 0; i < 34; i++) top34[top_base + i] = top[i];
    partial_sum[idx] = sum;
    partial_count[idx] = cnt;
}

// ── Streaming: finalize accumulated state → P90 + mean ───────────────────

extern "C" __global__ void finalize_kernel(
    const __half* __restrict__ top34,      // (92, rows, n_lon, 34) f16
    const float* __restrict__ partial_sum, // (92, rows, n_lon) f32
    const int* __restrict__ partial_count, // (92, rows, n_lon) i32
    float* __restrict__ p90_out,           // (92, rows, n_lon) f32
    float* __restrict__ mean_out,          // (92, rows, n_lon) f32
    long long n_days, long long rows, long long n_lon
) {
    long long lon = blockIdx.x * blockDim.x + threadIdx.x;
    long long day_lat = blockIdx.y;
    long long day = day_lat / rows;
    long long lat = day_lat % rows;

    if (lon >= n_lon || day >= n_days) return;

    long long out_stride = rows * n_lon;
    long long idx = day * out_stride + lat * n_lon + lon;

    int cnt = partial_count[idx];
    mean_out[idx] = (cnt > 0) ? partial_sum[idx] / (float)cnt : __builtin_nanf("");

    long long top_base = idx * 34;
    float v0 = __half2float(top34[top_base + 33]);
    float v1 = __half2float(top34[top_base + 32]);
    float p90 = v0 + (v1 - v0) * 0.1f;
    p90_out[idx] = __isinff(v0) ? __builtin_nanf("") : p90;
}

// ── Host launch wrappers ──────────────────────────────────────────────────

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

void launch_accumulate_year(
    torch::Tensor year_data,
    torch::Tensor top34,
    torch::Tensor partial_sum,
    torch::Tensor partial_count)
{
    long long n_days = top34.size(0);
    long long rows = top34.size(1);
    long long n_lon = top34.size(2);

    constexpr int BLOCK_LON = 256;
    dim3 block(BLOCK_LON);
    dim3 grid((n_lon + BLOCK_LON - 1) / BLOCK_LON, n_days * rows);

    accumulate_year_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __half*>(year_data.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(top34.data_ptr<at::Half>()),
        partial_sum.data_ptr<float>(),
        partial_count.data_ptr<int>(),
        n_days, rows, n_lon);
}

void launch_finalize(
    torch::Tensor top34,
    torch::Tensor partial_sum,
    torch::Tensor partial_count,
    torch::Tensor p90_out,
    torch::Tensor mean_out)
{
    long long n_days = top34.size(0);
    long long rows = top34.size(1);
    long long n_lon = top34.size(2);

    constexpr int BLOCK_LON = 256;
    dim3 block(BLOCK_LON);
    dim3 grid((n_lon + BLOCK_LON - 1) / BLOCK_LON, n_days * rows);

    finalize_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __half*>(top34.data_ptr<at::Half>()),
        partial_sum.data_ptr<float>(),
        partial_count.data_ptr<int>(),
        p90_out.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        n_days, rows, n_lon);
}

// ── Streaming: batched multi-year accumulate ─────────────────────────────

extern "C" __global__ void accumulate_batch_kernel(
    const __half* __restrict__ year_data,  // (N, 102, rows, n_lon) f16
    __half* __restrict__ top34,            // (92, rows, n_lon, 34) f16
    float* __restrict__ partial_sum,       // (92, rows, n_lon) f32
    int* __restrict__ partial_count,       // (92, rows, n_lon) i32
    long long n_years, long long n_days, long long rows, long long n_lon
) {
    long long lon = blockIdx.x * blockDim.x + threadIdx.x;
    long long day_lat = blockIdx.y;
    long long day = day_lat / rows;
    long long lat = day_lat % rows;

    if (lon >= n_lon || day >= n_days) return;

    long long yr_batch_stride = 102 * rows * n_lon;
    long long yr_stride = rows * n_lon;
    long long out_stride = rows * n_lon;
    long long idx = day * out_stride + lat * n_lon + lon;
    long long top_base = idx * 34;

    // Load persistent state once
    __half top[34];
    #pragma unroll
    for (int i = 0; i < 34; i++) top[i] = top34[top_base + i];
    float sum = partial_sum[idx];
    int cnt = partial_count[idx];

    // Process all N years in one pass
    const int WINDOW = 11;
    for (int y = 0; y < n_years; y++) {
        const __half* yr_ptr = year_data + y * yr_batch_stride;
        #pragma unroll
        for (int w = 0; w < WINDOW; w++) {
            __half v = yr_ptr[(day + w) * yr_stride + lat * n_lon + lon];
            if (__hisnan(v)) {
                v = __float2half(__builtin_huge_valf());
            } else {
                sum += __half2float(v);
                cnt++;
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
    }

    // Write back once
    #pragma unroll
    for (int i = 0; i < 34; i++) top34[top_base + i] = top[i];
    partial_sum[idx] = sum;
    partial_count[idx] = cnt;
}

void launch_accumulate_batch(
    torch::Tensor year_data,
    torch::Tensor top34,
    torch::Tensor partial_sum,
    torch::Tensor partial_count)
{
    long long n_years = year_data.size(0);
    long long n_days = top34.size(0);
    long long rows = top34.size(1);
    long long n_lon = top34.size(2);

    constexpr int BLOCK_LON = 256;
    dim3 block(BLOCK_LON);
    dim3 grid((n_lon + BLOCK_LON - 1) / BLOCK_LON, n_days * rows);

    accumulate_batch_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __half*>(year_data.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(top34.data_ptr<at::Half>()),
        partial_sum.data_ptr<float>(),
        partial_count.data_ptr<int>(),
        n_years, n_days, rows, n_lon);
}
"""

        cpp_src = r"""
#include <torch/extension.h>
void launch_fused_stats(torch::Tensor sub, torch::Tensor p90_out, torch::Tensor mean_out);
void launch_accumulate_year(torch::Tensor year_data, torch::Tensor top34, torch::Tensor partial_sum, torch::Tensor partial_count);
void launch_finalize(torch::Tensor top34, torch::Tensor partial_sum, torch::Tensor partial_count, torch::Tensor p90_out, torch::Tensor mean_out);
void launch_accumulate_batch(torch::Tensor year_data, torch::Tensor top34, torch::Tensor partial_sum, torch::Tensor partial_count);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_stats", &launch_fused_stats);
    m.def("accumulate_year", &launch_accumulate_year);
    m.def("finalize", &launch_finalize);
    m.def("accumulate_batch", &launch_accumulate_batch);
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
        print("  Fused kernel:   ENABLED (HIP: fused + streaming accumulate/finalize + batch)")
        return _mod.fused_stats, _mod.accumulate_year, _mod.finalize, _mod.accumulate_batch
    except Exception as e:
        print(f"  Fused kernel:   FALLBACK to topk ({e})")
        return None, None, None, None


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


# ── Streaming compute (I/O–Compute overlap) ────────────────────────────────

def _compute_streaming(data, ready_flags, bands, accumulate_fn, finalize_fn,
                       num_gpus):
    """Streaming GPU compute: processes years as they become ready during I/O.

    Returns (threads, threshold, climatology). Caller must join threads.
    Each GPU thread polls ready_flags, copies year slices to GPU, accumulates
    into persistent top-34/sum/count, then finalizes P90+mean per band.
    """
    import torch
    import threading
    import time as _time

    band_groups = [[] for _ in range(num_gpus)]
    for i, band in enumerate(bands):
        band_groups[i % num_gpus].append(band)

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    def _worker(gpu_id, group):
        if not group:
            return
        torch.cuda.set_device(gpu_id)

        for r0, r1 in group:
            t0 = _time.time()
            rows = r1 - r0

            with torch.cuda.device(gpu_id):
                top34 = torch.full((N_OUTPUT_DAYS, rows, N_LON, 34),
                                   float('-inf'), dtype=torch.float16,
                                   device=gpu_id)
                partial_sum = torch.zeros(N_OUTPUT_DAYS, rows, N_LON,
                                          dtype=torch.float32, device=gpu_id)
                partial_count = torch.zeros(N_OUTPUT_DAYS, rows, N_LON,
                                            dtype=torch.int32, device=gpu_id)

                # Pre-allocate GPU buffer for year data transfers (avoid
                # repeated CUDA allocations across 30 years × 3 bands)
                year_gpu = torch.empty(102, rows, N_LON,
                                       dtype=torch.float16, device=gpu_id)

                processed = [False] * N_YEARS
                remaining = N_YEARS

                while remaining > 0:
                    for yr in range(N_YEARS):
                        if not processed[yr] and ready_flags[yr]:
                            year_cpu = torch.from_numpy(
                                np.ascontiguousarray(data[yr, :, r0:r1, :]))
                            year_gpu.copy_(year_cpu)
                            accumulate_fn(year_gpu, top34, partial_sum,
                                          partial_count)
                            processed[yr] = True
                            remaining -= 1
                    if remaining > 0:
                        _time.sleep(0.001)  # 1ms poll interval

                # Finalize: compute P90 + mean from accumulated state
                p90_gpu = torch.empty(N_OUTPUT_DAYS, rows, N_LON,
                                      dtype=torch.float32, device=gpu_id)
                mean_gpu = torch.empty(N_OUTPUT_DAYS, rows, N_LON,
                                       dtype=torch.float32, device=gpu_id)
                finalize_fn(top34, partial_sum, partial_count,
                            p90_gpu, mean_gpu)

                threshold[:, r0:r1, :] = p90_gpu.cpu().numpy()
                climatology[:, r0:r1, :] = mean_gpu.cpu().numpy()

                del top34, partial_sum, partial_count, p90_gpu, mean_gpu
                del year_gpu

            torch.cuda.empty_cache()
            print(f"  [GPU {gpu_id}] Band [{r0}:{r1}] done "
                  f"({_time.time() - t0:.1f}s)")

    threads = []
    for gid, group in enumerate(band_groups):
        t = threading.Thread(target=_worker, args=(gid, group))
        t.start()
        threads.append(t)

    return threads, threshold, climatology


# ── Per-day streaming compute (I/O–Compute overlap at day granularity) ─────

def _compute_streaming_by_day(data, ready_flags, bands, fused_fn, num_gpus):
    """Per-day streaming GPU compute: one output day at a time via fused_stats_kernel.

    Returns (threads, threshold, climatology). Caller must join threads.
    Each GPU thread pre-allocates the full 102-day window, copies day slices
    as they become ready, and computes output days as their 11-day windows
    complete using the existing fused_stats_kernel (n_days=1, n_total=330).

    ready_flags layout: ready_flags[day * MAX_RANKS + rank] int8.
    A day is ready when all N_YEARS rank-slots for that day are set to 1.
    """
    import torch
    import threading
    import time as _time

    band_groups = [[] for _ in range(num_gpus)]
    for i, band in enumerate(bands):
        band_groups[i % num_gpus].append(band)

    threshold = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)
    climatology = np.full((N_OUTPUT_DAYS, N_LAT, N_LON), np.nan, dtype=np.float32)

    def _worker(gpu_id, group):
        if not group:
            return
        torch.cuda.set_device(gpu_id)
        rows_total = sum(r1 - r0 for (r0, r1) in group)

        with torch.cuda.device(gpu_id):
            # Pre-allocate full 102-day window on GPU
            gpu_data = torch.empty(N_YEARS, N_DAYS_PER_YEAR, rows_total, N_LON,
                                   dtype=torch.float16, device=gpu_id)
            # Per-day I/O: poll per-rank flags for this day
            n_readers = N_YEARS  # number of ranks producing data (= years)
            for day_idx in range(N_DAYS_PER_YEAR):
                flag_base = day_idx * MAX_RANKS
                while not np.all(ready_flags[flag_base:flag_base + n_readers]):
                    _time.sleep(0.0005)  # 0.5ms poll interval

                # Copy this day's data for all my bands
                col_offset = 0
                for r0, r1 in group:
                    rows = r1 - r0
                    day_cpu = torch.from_numpy(
                        np.ascontiguousarray(data[:, day_idx, r0:r1, :]))
                    gpu_data[:, day_idx, col_offset:col_offset + rows, :].copy_(day_cpu)
                    col_offset += rows

                # Check if any output day's window just completed
                if day_idx >= WINDOW_HALF * 2:
                    od = day_idx - WINDOW_HALF * 2  # = day_idx - 10

                    col_offset = 0
                    for r0, r1 in group:
                        rows = r1 - r0
                        # Extract window: data[:, od:od+11, rows, lon]
                        window = gpu_data[:, od:od + WINDOW_SIZE,
                                          col_offset:col_offset + rows, :]
                        # Reorder to (1, 330, rows, n_lon) for fused_stats_kernel
                        sub = window.permute(1, 0, 2, 3).reshape(
                            1, N_YEARS * WINDOW_SIZE, rows, N_LON).contiguous()

                        p90_gpu = torch.empty(1, rows, N_LON,
                                              dtype=torch.float32, device=gpu_id)
                        mean_gpu = torch.empty(1, rows, N_LON,
                                               dtype=torch.float32, device=gpu_id)
                        fused_fn(sub, p90_gpu, mean_gpu)

                        threshold[od, r0:r1, :] = p90_gpu[0].cpu().numpy()
                        climatology[od, r0:r1, :] = mean_gpu[0].cpu().numpy()
                        col_offset += rows

            del gpu_data
        torch.cuda.empty_cache()

    threads = []
    for gid, group in enumerate(band_groups):
        t = threading.Thread(target=_worker, args=(gid, group))
        t.start()
        threads.append(t)

    return threads, threshold, climatology


# ── Latitude bands ─────────────────────────────────────────────────────────

def build_bands():
    base, rem = divmod(N_LAT, N_BANDS)
    bands, r0 = [], 0
    for i in range(N_BANDS):
        sz = base + (1 if i < rem else 0)
        bands.append((r0, r0 + sz))
        r0 += sz
    return bands


# ── Parallel save (multiprocessing, fork COW) ──────────────────────────────

_save_threshold = None
_save_climatology = None
_save_lats = None
_save_lons = None
_save_out_dir = None
_save_d0 = None


def _save_chunk(args):
    """Write a subset of output files. Called via multiprocessing fork.

    Accesses module-level globals (_save_*) set by main() before fork.
    Each child inherits parent memory via COW — no data copy overhead.
    """
    start_idx, end_idx = args
    from netCDF4 import Dataset as NCDataset
    from datetime import timedelta
    import numpy as np

    for out_idx in range(start_idx, end_idx):
        doy = 152 + out_idx
        dt = _save_d0 + timedelta(days=doy - 1)
        fname = dt.strftime("%m%d.nc")

        with NCDataset(_save_out_dir / fname, "w", format="NETCDF4") as nc:
            nc.createDimension("Lat", N_LAT)
            nc.createDimension("Lon", N_LON)
            nc.createDimension("Day", 1)

            v_lat = nc.createVariable("Lat", "f4", ("Lat",))
            v_lat[:] = _save_lats
            v_lat.long_name = "Latitude"

            v_lon = nc.createVariable("Lon", "f4", ("Lon",))
            v_lon[:] = _save_lons
            v_lon.long_name = "Longitude"

            v_clim = nc.createVariable("Climmean", "f4", ("Lat", "Lon"))
            v_clim[:] = np.ascontiguousarray(_save_climatology[out_idx])
            v_clim.long_name = "OSTIA SST climatology 1991-2020"

            v_p90 = nc.createVariable("P90_sst", "f4", ("Lat", "Lon"))
            v_p90[:] = np.ascontiguousarray(_save_threshold[out_idx])
            v_p90.long_name = "90th percentile of precipitation"

            v_doy = nc.createVariable("dayofyear", "i4", ("Day",))
            v_doy[:] = doy
            v_doy.long_name = "Day of year (1-365, no 29Feb)"

            nc.baseline_period = "1991-2020"


# ── Main ───────────────────────────────────────────────────────────────────

def main():

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
        by_year[ds[:4]].append((j, date_to_path[ds]))

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

        fused_fn, accumulate_fn, finalize_fn, accumulate_batch_fn = _load_fused_kernel()
        streaming_ok = accumulate_fn is not None

        # Micro-warmup: warm only the kernel path actually used.
        if streaming_ok:
            # Per-day streaming uses fused_stats_kernel (n_days=1, n_total=330)
            for gpu_id in range(num_gpus):
                with torch.cuda.device(gpu_id):
                    d = torch.zeros(1, N_YEARS * WINDOW_SIZE, 1, N_LON,
                                    dtype=torch.float16, device=gpu_id)
                    p = torch.empty(1, 1, N_LON,
                                    dtype=torch.float32, device=gpu_id)
                    m = torch.empty(1, 1, N_LON,
                                    dtype=torch.float32, device=gpu_id)
                    fused_fn(d, p, m)
            torch.cuda.synchronize()
        elif fused_fn is not None:
            for gpu_id in range(num_gpus):
                with torch.cuda.device(gpu_id):
                    d = torch.zeros(1, N_YEARS * WINDOW_SIZE, 1, N_LON,
                                    dtype=torch.float16, device=gpu_id)
                    p = torch.empty(1, 1, N_LON,
                                    dtype=torch.float32, device=gpu_id)
                    m = torch.empty(1, 1, N_LON,
                                    dtype=torch.float32, device=gpu_id)
                    fused_fn(d, p, m)
            torch.cuda.synchronize()

        setup_time = time.time() - t_setup
        print(f"  Setup:          {setup_time:.1f}s")
    else:
        num_gpus, setup_time = 0, 0.0
        fused_fn, accumulate_fn, finalize_fn, accumulate_batch_fn = None, None, None, None
        streaming_ok = False
    num_gpus = comm.bcast(num_gpus, root=0)
    setup_time = comm.bcast(setup_time, root=0)
    streaming_ok = comm.bcast(streaming_ok, root=0)

    # ── MPI Distributed I/O (direct-to-shm, zero gather) ──
    total_bytes = shape[0] * shape[1] * shape[2] * shape[3] * 2  # float16 = 2 bytes
    flags_offset = total_bytes + FLAGS_PAD
    win_size = total_bytes + FLAGS_PAD + FLAGS_COUNT if rank == 0 else 0
    win = MPI.Win.Allocate_shared(win_size, 1, comm=comm)
    shm_mem, _ = win.Shared_query(0)
    data = np.ndarray(shape, dtype=np.float16, buffer=shm_mem)
    ready_flags = np.ndarray(FLAGS_COUNT, dtype=np.int8,
                             buffer=shm_mem, offset=flags_offset)
    if rank == 0:
        ready_flags[:] = 0

    # ── Build per-day file mapping for per-day streaming ──
    # my_day_files[day_idx] = [(yr_idx, fp), ...] for this rank
    my_day_files = [[] for _ in range(N_DAYS_PER_YEAR)]
    for yr in my_years:
        yr_idx = int(yr) - START_YEAR
        for day_idx, (slot, fp) in enumerate(by_year[yr]):
            my_day_files[day_idx].append((yr_idx, fp))

    day_count_target = N_DAYS_PER_YEAR  # total days to read (all ranks)
    if rank == 0 and streaming_ok:
        print(f"\n  Streaming mode: ENABLED "
              f"(per-day I/O–Compute overlap, fused_stats_kernel, 4 GPU workers)")

    if rank == 0:
        raw_label = "raw I/O" if raw_info else "netCDF4"
        print(f"\nReading {day_count_target} days per rank "
              f"(MPI distributed per-day, float16, {raw_label}, "
              f"direct-to-shm) ...")
    comm.Barrier()
    t_io = time.time()

    # ── Launch GPU streaming threads BEFORE I/O (rank 0 only) ──
    gpu_threads = None
    t_gpu = None
    if rank == 0 and streaming_ok:
        bands = build_bands()
        t_gpu = time.time()
        gpu_threads, threshold, climatology = _compute_streaming_by_day(
            data, ready_flags, bands, fused_fn, num_gpus)

    # ── I/O: per-day, per-rank flags (no Barrier → no tail-latency amplification) ──
    def _read_single_file(yr_idx, day_slot, fp):
        block = (raw_read_sst(fp, raw_info) if raw_info
                 else _read_netcdf_fallback(fp, sst_var))
        block = np.squeeze(block)
        if block.shape != (N_LAT, N_LON):
            block = block.T
        data[yr_idx, day_slot] = np.asarray(block, dtype=np.float16)

    for day_idx in range(N_DAYS_PER_YEAR):
        for yr_idx, fp in my_day_files[day_idx]:
            _read_single_file(yr_idx, day_idx, fp)
        # Per-rank flag: each rank owns its slot, no Barrier
        ready_flags[day_idx * MAX_RANKS + rank] = 1

    comm.Barrier()
    io_elapsed = time.time() - t_io
    if rank == 0:
        print(f"  MPI I/O total: {io_elapsed:.1f}s (no gather — direct to shm)")

    # ── Rank > 0: done ──
    if rank != 0:
        return

    # ── Post-I/O: join GPU threads (streaming) or run batch compute ──
    gc.collect()

    if streaming_ok:
        t_join = time.time()
        for t in gpu_threads:
            t.join()
        compute_tail = time.time() - t_join
        compute_total = time.time() - t_gpu
        t_compute = t_join  # for save timing reference
        compute_elapsed = compute_tail
        print(f"  Compute:      {compute_total:8.1f}s  "
              f"({io_elapsed:.1f}s I/O overlap + {compute_tail:.1f}s tail)")
    else:
        print(f"\nComputing threshold & climatology "
              f"({N_OUTPUT_DAYS} days x 30-year sliding window) ...")
        bands = build_bands()
        t_compute = time.time()
        threshold, climatology = _compute_torch(data, bands, fused_fn)
        compute_elapsed = time.time() - t_compute
        print(f"  Compute: {compute_elapsed:.1f}s")

    del data; gc.collect()

    # ── Save (parallel via multiprocessing fork) ──
    print("\nSaving ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp

    N_SAVE_WORKERS = int(os.environ.get("SAVE_WORKERS", "4"))
    N_SAVE_WORKERS = min(N_SAVE_WORKERS, N_OUTPUT_DAYS)

    # Populate module-level globals for worker processes (COW, no copy)
    global _save_threshold, _save_climatology, _save_lats, _save_lons
    global _save_out_dir, _save_d0
    _save_threshold = threshold
    _save_climatology = climatology
    _save_lats = lats
    _save_lons = lons
    _save_out_dir = OUT_DIR
    _save_d0 = datetime(2019, 1, 1)  # DOY 152 = Jun 1 (non-leap year)

    # Build evenly-sized chunks
    base, rem = divmod(N_OUTPUT_DAYS, N_SAVE_WORKERS)
    chunks, idx = [], 0
    for w in range(N_SAVE_WORKERS):
        sz = base + (1 if w < rem else 0)
        if sz > 0:
            chunks.append((idx, idx + sz))
            idx += sz

    if N_SAVE_WORKERS > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=N_SAVE_WORKERS) as pool:
            pool.map(_save_chunk, chunks)
    else:
        _save_chunk((0, N_OUTPUT_DAYS))

    t_end = time.time()
    total_elapsed = t_end - t_io

    print(f"  Saved {N_OUTPUT_DAYS} files to {OUT_DIR}/")

    # ── Timing breakdown ──
    prep_elapsed = t_io - _t0 - setup_time
    mode_label = "per-day streaming" if streaming_ok else "batch"

    print(f"\n{'='*60}")
    print(f"  TIMING BREAKDOWN (MPI, float16, {mode_label})")
    print(f"{'='*60}")
    print(f"  Prep:       {prep_elapsed:8.1f}s   (imports + file index + format probe)")
    print(f"  Setup:      {setup_time:8.1f}s   (kernel compile + micro-warmup)")
    if streaming_ok:
        print(f"  I/O:        {io_elapsed:8.1f}s   (MPI distributed, 30 ranks)")
        print(f"  Compute:    {compute_total:8.1f}s  "
              f"({compute_total - compute_tail:.1f}s in I/O phase + {compute_tail:.1f}s tail)")
    else:
        print(f"  I/O:        {io_elapsed:8.1f}s   (MPI distributed)")
        print(f"  Compute:    {compute_elapsed:8.1f}s")
    print(f"  Save:       {t_end - t_compute - compute_elapsed:8.1f}s")
    print(f"  {'─'*50}")
    print(f"  I/O→save:   {total_elapsed:8.3f}s  ({total_elapsed/60:.2f} min)")
    competition_total = t_end - _t0
    print(f"  Competition:{competition_total:8.3f}s  ({competition_total/60:.2f} min)")
    print(f"    (program start → result output, per competition rules)")
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
