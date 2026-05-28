#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCC 2026 climatology / P90 — DCU optimized with profiling.
- Global date cache (read once per task block)
- Year-grain parallel I/O (30 workers vs thousands of file tasks)
- Pipeline: setup||io_read, write||compute (prefetch=1, default)
- End-of-run timing breakdown + overlap saved
"""
from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

if sys.version_info[0] < 3:
    sys.exit("Use: python3 run_climatology.py")

try:
    import netCDF4 as nc
except ImportError as e:
    sys.exit("netCDF4 required: conda install -c conda-forge netcdf4\n" + str(e))

from mcc_calendar import doy_to_mmdd, window_date_strings_for_doy
from gpu_backend import resolve_device, resolve_compute_dtype, compute_clim_p90


class ProfileTimer(object):
    """Accumulate wall time by phase; print summary at exit."""

    def __init__(self):
        self.t0 = time.time()
        self.phases = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, phase, sec, n=1):
        self.phases[phase] += sec
        self.counts[phase] += n

    class _Block(object):
        def __init__(self, timer, phase):
            self.timer = timer
            self.phase = phase
            self.t0 = None

        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.timer.add(self.phase, time.time() - self.t0, 1)
            return False

    def block(self, phase):
        return self._Block(self, phase)

    def summary(self):
        total = time.time() - self.t0
        print("\n========== TIMING SUMMARY ==========", flush=True)
        order = ("setup", "io_plan", "io_read", "io_gather", "compute", "write", "mp_spawn", "other")
        accounted = 0.0
        phase_sum = 0.0
        for ph in order:
            sec = self.phases.get(ph, 0.0)
            phase_sum += sec
            if sec <= 0:
                continue
            accounted += sec
            cnt = self.counts.get(ph, 0)
            pct = 100.0 * sec / total if total > 0 else 0.0
            extra = " (n={})".format(cnt) if cnt > 1 else ""
            print("  {:10s} {:8.1f}s  {:5.1f}%{}{}".format(
                ph + ":", sec, pct, extra,
                "  [{:.2f}s avg]".format(sec / cnt) if cnt > 1 else ""), flush=True)
        other = max(0.0, total - accounted)
        if other > 0.5:
            print("  {:10s} {:8.1f}s  {:5.1f}%".format(
                "untracked:", other, 100.0 * other / total), flush=True)
        overlap = max(0.0, phase_sum - total)
        if overlap > 0.5:
            print("  {:10s} {:8.1f}s  (setup/io/write overlapped, not additive)".format(
                "overlap:", overlap), flush=True)
        print("  {:10s} {:8.1f}s  {:5.1f}min".format("TOTAL:", total, total / 60.0), flush=True)
        print("====================================", flush=True)


PROFILE = ProfileTimer()


def parse_args():
    p = argparse.ArgumentParser(description="MCC2026 climatology / P90 (DCU optimized)")
    p.add_argument("--nc-path", default=os.environ.get("MCC_NC_PATH", "/public/home/achwjznh4b/Newdata/"))
    p.add_argument("--save-path", default=os.environ.get("MCC_SAVE_PATH", "./ERA5/Climatology/"))
    p.add_argument("--start-yr", type=int, default=1991)
    p.add_argument("--end-yr", type=int, default=2020)
    p.add_argument("--start-doy", type=int, default=152)
    p.add_argument("--end-doy", type=int, default=243)
    p.add_argument("--delta-day", type=int, default=5)
    p.add_argument("--var-sst", default="data")
    p.add_argument("--var-lon", default="lon")
    p.add_argument("--var-lat", default="lat")
    p.add_argument("--row-total", type=int, default=721)
    p.add_argument("--col-total", type=int, default=1440)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", default=os.environ.get("MCC_DEVICE", "auto"),
                   choices=["auto", "cpu", "gpu", "cuda", "dcu"])
    p.add_argument("--batch-rows", type=int, default=721)
    p.add_argument("--io-workers", type=int, default=4)
    p.add_argument("--io-backend", default=os.environ.get("MCC_IO_BACKEND", "process"),
                   choices=["process", "thread", "serial"])
    p.add_argument("--io-grain", default=os.environ.get("MCC_IO_GRAIN", "year"),
                   choices=["year", "file"],
                   help="year: 30 parallel year readers (fast); file: one task per date")
    p.add_argument("--gpu-lat-chunk", type=int, default=0)
    p.add_argument("--prefetch", type=int, default=int(os.environ.get("MCC_PREFETCH", "1")),
                   help="1: overlap setup||io_read and write||compute (lower wall time)")
    p.add_argument("--compute-dtype", default=os.environ.get("MCC_COMPUTE_DTYPE", "float16"),
                   choices=["float16", "float32", "float64", "fp16", "fp32", "fp64", "half"])
    p.add_argument("--cache-all-needed", type=int, default=int(os.environ.get("MCC_CACHE_ALL_NEEDED", "1")))
    p.add_argument("--num-gpu", type=int, default=int(os.environ.get("MCC_NUM_GPU", "1")),
                   help="Split DOYs across N DCU devices in one job (e.g. 4)")
    p.add_argument("--gpu-warmup", type=int, default=int(os.environ.get("MCC_GPU_WARMUP", "0")),
                   help="1: warmup each GPU worker before compute; 0: skip to reduce startup")
    p.add_argument("--sub-batch-days", type=int, default=int(os.environ.get("MCC_SUB_BATCH_DAYS", "2")),
                   help="Compute multiple DOYs together per GPU worker (speed optimization)")
    p.add_argument("--gpu-band-rows", type=int, default=int(os.environ.get("MCC_GPU_BAND_ROWS", "120")),
                   help="Latitude rows per GPU band. 0 means full latitude at once.")
    p.add_argument("--worker-prefetch", type=int,
                   default=int(os.environ.get("MCC_WORKER_PREFETCH", "0")),
                   help="1: prefetch next gather in a thread (overlap CPU/GPU); 0: safer sequential")
    p.add_argument("--worker-write-threads", type=int,
                   default=int(os.environ.get("MCC_WORKER_WRITE_THREADS", "1")),
                   help="Background write threads per GPU worker (1 = safe)")
    return p.parse_args()


def nc_data_path(nc_path, yyyymmdd):
    return os.path.join(nc_path, yyyymmdd)


def _read_one_file(args):
    j, nc_path, date_str, var_sst, row_start, n_rows, ncol, np_dtype = args
    fpath = nc_data_path(nc_path, date_str)
    if not os.path.isfile(fpath):
        return j, None
    with nc.Dataset(fpath, "r") as ds:
        block = ds.variables[var_sst][row_start:row_start + n_rows, 0:ncol]
    return j, np.asarray(block, dtype=np_dtype)


def _read_one_year(args):
    """Read many dates for one year inside a single worker (less pool overhead)."""
    year, nc_path, items, var_sst, row_start, n_rows, ncol, np_dtype = args
    out_idx = []
    out_blk = []
    for j, date_str in items:
        fpath = nc_data_path(nc_path, date_str)
        if not os.path.isfile(fpath):
            continue
        with nc.Dataset(fpath, "r") as ds:
            block = ds.variables[var_sst][row_start:row_start + n_rows, 0:ncol]
        out_idx.append(j)
        out_blk.append(np.asarray(block, dtype=np_dtype))
    return out_idx, out_blk


def load_time_stack(nc_path, date_strs, var_sst, row_start, n_rows, ncol, io_workers,
                    io_backend, np_dtype, io_grain="year", pool=None):
    n_time = len(date_strs)
    stack = np.full((n_time, n_rows, ncol), np.nan, dtype=np_dtype)
    workers = max(1, min(io_workers, n_time)) if io_backend != "serial" else 1
    t0 = time.time()
    print("  reading {} files grain={} backend={} workers={} ...".format(
        n_time, io_grain, io_backend, workers), flush=True)

    if io_grain == "year" and io_backend == "process":
        by_year = defaultdict(list)
        for j, ds in enumerate(date_strs):
            by_year[ds[:4]].append((j, ds))
        year_tasks = [
            (yr, nc_path, items, var_sst, row_start, n_rows, ncol, np_dtype)
            for yr, items in sorted(by_year.items())
        ]
        n_years = len(year_tasks)
        w = max(1, min(io_workers, n_years))
        print("  year-grain: {} years, {} workers".format(n_years, w), flush=True)
        if pool is not None:
            results = pool.map(_read_one_year, year_tasks, chunksize=1)
        else:
            with ProcessPoolExecutor(max_workers=w) as ex:
                results = ex.map(_read_one_year, year_tasks, chunksize=1)
        done_files = 0
        for idx_list, blk_list in results:
            for j, blk in zip(idx_list, blk_list):
                stack[j, :, :] = blk
            done_files += len(idx_list)
            print("  ... {}/{} files ({:.1f}s)".format(
                done_files, n_time, time.time() - t0), flush=True)
    else:
        tasks = [(j, nc_path, ds, var_sst, row_start, n_rows, ncol, np_dtype)
                 for j, ds in enumerate(date_strs)]
        report_every = max(1, n_time // 10)
        done = 0
        chunksize = max(1, n_time // max(workers * 8, 1))
        if io_backend == "serial" or workers == 1:
            for task in tasks:
                j, blk = _read_one_file(task)
                if blk is not None:
                    stack[j, :, :] = blk
                done += 1
                if done == 1 or done % report_every == 0 or done == n_time:
                    print("  ... {}/{} ({:.1f}s)".format(done, n_time, time.time() - t0), flush=True)
        elif io_backend == "thread":
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_read_one_file, t) for t in tasks]
                for fut in as_completed(futs):
                    j, blk = fut.result()
                    if blk is not None:
                        stack[j, :, :] = blk
                    done += 1
                    if done == 1 or done % report_every == 0 or done == n_time:
                        print("  ... {}/{} ({:.1f}s)".format(done, n_time, time.time() - t0), flush=True)
        else:
            if pool is not None:
                it = pool.map(_read_one_file, tasks, chunksize=chunksize)
            else:
                ex_ctx = ProcessPoolExecutor(max_workers=workers)
                it = ex_ctx.map(_read_one_file, tasks, chunksize=chunksize)
            for j, blk in it:
                if blk is not None:
                    stack[j, :, :] = blk
                done += 1
                if done == 1 or done % report_every == 0 or done == n_time:
                    print("  ... {}/{} ({:.1f}s)".format(done, n_time, time.time() - t0), flush=True)
            if pool is None:
                ex_ctx.shutdown(wait=True)

    sec = time.time() - t0
    print("  read done {:.1f}s ({:.3f}s/file)".format(sec, sec / max(n_time, 1)), flush=True)
    return stack, sec


def write_output_nc(out_path, doy, lat, lon, clim_row_major, p90_row_major):
    if os.path.exists(out_path):
        os.remove(out_path)
    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("Lat", clim_row_major.shape[0])
        ds.createDimension("Lon", clim_row_major.shape[1])
        ds.createDimension("Day", 1)
        v_time = ds.createVariable("dayofyear", "f8", ("Day",))
        v_time.long_name = "Day of year (1-365, no 29Feb)"
        v_time[:] = doy
        v_lat = ds.createVariable("Lat", "f8", ("Lat",))
        v_lon = ds.createVariable("Lon", "f8", ("Lon",))
        v_lat[:] = lat
        v_lon[:] = lon
        v_clim = ds.createVariable("Climmean", "f8", ("Lat", "Lon"))
        v_clim.long_name = "OSTIA SST climatology 1991-2020"
        v_p90 = ds.createVariable("P90_sst", "f8", ("Lat", "Lon"))
        v_p90.long_name = "90th percentile of SST"
        v_clim[:, :] = clim_row_major
        v_p90[:, :] = p90_row_major


def _build_needed_dates(doys, args):
    per_doy = {}
    needed = set()
    for doy in doys:
        ds = window_date_strings_for_doy(doy, args.start_yr, args.end_yr, args.delta_day)
        per_doy[doy] = ds
        needed.update(ds)
    needed_sorted = sorted(needed)
    idx_map = {d: i for i, d in enumerate(needed_sorted)}
    return needed_sorted, per_doy, idx_map


def _read_lat_lon(args):
    demo = nc_data_path(args.nc_path, "19910101")
    if not os.path.isfile(demo):
        sys.exit("Demo file missing: {}".format(demo))
    with nc.Dataset(demo, "r") as ds:
        lat = np.asarray(ds.variables[args.var_lat][:], dtype=np.float64)
        lon = np.asarray(ds.variables[args.var_lon][:], dtype=np.float64)
    return lat, lon


def _probe_cuda_worker(_unused):
    import torch
    t0 = time.time()
    if not torch.cuda.is_available():
        return 0, time.time() - t0, []
    nd = torch.cuda.device_count()
    names = []
    for i in range(nd):
        names.append(torch.cuda.get_device_name(i))
        torch.cuda.set_device(i)
        _ = torch.randn(1, device="cuda:{}".format(i))
    torch.cuda.synchronize()
    return nd, time.time() - t0, names


def _probe_cuda_subprocess():
    """Probe DCU in a child process so parent stays torch-free (safe to fork cache)."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(_probe_cuda_worker, (None,))


def _resolve_setup(args, warmup_gpu=True, probe_all_gpus=False):
    """lat/lon + optional GPU warmup; multi-GPU probe uses cheap Slurm hint."""
    device_tag, torch_mod = None, None
    cuda_count = 1
    t_setup = time.time()

    lat, lon = _read_lat_lon(args)

    if probe_all_gpus:
        # Keep parent process torch-free for fork-based cache sharing.
        cuda_count = _slurm_visible_gpus(max(1, args.num_gpu))
        print("[setup] cuda probe(skip torch) -> slurm_visible={}".format(cuda_count), flush=True)
        if cuda_count < 1:
            sys.exit("ERROR: no DCU visible in Slurm env")
    elif warmup_gpu:
        device_tag, torch_mod = resolve_device(args.device)
        if device_tag != "cpu" and torch_mod is not None:
            _ = torch_mod.randn(1, device=device_tag)
            torch_mod.cuda.synchronize()
            cuda_count = torch_mod.cuda.device_count()

    setup_sec = time.time() - t_setup
    return device_tag, torch_mod, lat, lon, cuda_count, setup_sec


def _split_doys_for_gpus(doys, num_gpu):
    n = max(1, min(num_gpu, len(doys)))
    if n == 1:
        return [doys]
    chunk = (len(doys) + n - 1) // n
    return [doys[i:i + chunk] for i in range(0, len(doys), chunk)]


def _slurm_visible_gpus(requested):
    """Hint from Slurm; real count comes from _probe_cuda_subprocess."""
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        val = os.environ.get(key)
        if val:
            try:
                return max(1, min(requested, int(val)))
            except ValueError:
                pass
    return requested


_MP_SHARED = {}


def _multi_gpu_worker(task):
    """Fork worker per DCU (stable on this cluster).

    Pipeline: a single prefetch thread issues the next np.take while GPU
    crunches the current sub-batch (CPU/GPU overlap). Output writes also
    happen in a background thread pool so they don't block compute.
    """
    gpu_id, doy_chunk = task
    args = _MP_SHARED["args"]
    cache = _MP_SHARED["cache"]
    lat = _MP_SHARED["lat"]
    lon = _MP_SHARED["lon"]
    per_doy = _MP_SHARED["per_doy"]
    idx_map = _MP_SHARED["idx_map"]

    import torch
    init_t0 = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("DCU {}: torch.cuda.is_available() is False".format(gpu_id))
    nd = torch.cuda.device_count()
    if gpu_id >= nd:
        raise RuntimeError(
            "DCU {} not available (device_count={}). Unset HIP_VISIBLE_DEVICES if set to 0.".format(
                gpu_id, nd))

    device_tag = "cuda:{}".format(gpu_id)
    torch.cuda.set_device(gpu_id)
    name = torch.cuda.get_device_name(gpu_id)
    if int(getattr(args, "gpu_warmup", 0)) == 1:
        _ = torch.randn(1, device=device_tag)
        torch.cuda.synchronize()
    init_sec = time.time() - init_t0
    print("[gpu{}] {} ({}) init {:.1f}s".format(gpu_id, device_tag, name, init_sec), flush=True)

    n_row = cache.shape[1]
    n_col = cache.shape[2]
    lat_slice = lat[:n_row]
    n_time = len(next(iter(per_doy.values())))
    sub_batch_days = max(1, int(getattr(args, "sub_batch_days", 6)))
    band_rows = int(getattr(args, "gpu_band_rows", 120))
    if band_rows <= 0:
        bands = [(0, n_row)]
    else:
        bands = [(r0, min(r0 + band_rows, n_row)) for r0 in range(0, n_row, band_rows)]
    gather_sec = 0.0
    comp_sec = 0.0
    write_sec = 0.0
    use_fast = os.environ.get("MCC_FAST_P90", "1") != "0"
    doy_ids = np.stack([np.asarray([idx_map[x] for x in per_doy[d]], dtype=np.int32) for d in doy_chunk], axis=0)
    n_doy = len(doy_chunk)
    clim_all = np.empty((n_doy, n_row, n_col), dtype=np.float64)
    p90_all = np.empty((n_doy, n_row, n_col), dtype=np.float64)

    use_prefetch = int(getattr(args, "worker_prefetch", 0)) == 1
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if use_prefetch else None

    def _gather_block(cache_band, ids_mat, n_time_local, rows, n_col_local):
        flat = np.take(cache_band, ids_mat.reshape(-1), axis=0)
        return flat.reshape(ids_mat.shape[0], n_time_local, rows, n_col_local)

    p90_pos = 0.90 * (n_time - 1)
    k0_idx = int(p90_pos)
    k1_idx = min(k0_idx + 1, n_time - 1)
    w_frac = p90_pos - k0_idx
    # We only need the (k0_idx+1)-th and (k1_idx+1)-th smallest along T.
    # Equivalent to taking the smallest among the top (n_time - k0_idx) largest.
    topk_n = n_time - k0_idx  # e.g. 330 - 297 = 33 (≈10% of T)

    def _compute_one_batch(stack_batch, cur_d0, cur_bsz, r0, r1):
        nonlocal comp_sec
        t0 = time.time()
        with torch.inference_mode():
            t = torch.as_tensor(stack_batch, device=device_tag, dtype=torch.float16)
            clim_t = torch.nanmean(t, dim=1)
            if use_fast:
                # topk(largest=True, sorted=True) returns the largest topk_n values
                # along T, ordered DESC. So index -1 = smallest of those = (n-topk_n+1)-th
                # smallest in ASC = (k0_idx+1)-th smallest = v0.
                inf = torch.tensor(float("inf"), device=t.device, dtype=t.dtype)
                filled = torch.where(torch.isnan(t), inf, t)
                topvals, _ = torch.topk(filled, k=topk_n, dim=1, largest=True, sorted=True)
                v0 = topvals[:, -1]
                if k1_idx == k0_idx:
                    p90_t = v0
                else:
                    v1 = topvals[:, -2]  # next one, equals (k1_idx+1)-th smallest in ASC
                    p90_t = v0 + (v1 - v0) * w_frac
                p90_t = torch.where(torch.isinf(p90_t), torch.nan, p90_t)
                del filled, topvals
            else:
                p90_t = torch.nanquantile(t, 0.90, dim=1)
            clim_np = clim_t.float().cpu().numpy().astype(np.float64)
            p90_np = p90_t.float().cpu().numpy().astype(np.float64)
        comp_sec += time.time() - t0
        clim_all[cur_d0:cur_d0 + cur_bsz, r0:r1, :] = clim_np
        p90_all[cur_d0:cur_d0 + cur_bsz, r0:r1, :] = p90_np

    try:
        for (r0, r1) in bands:
            cache_band = cache[:, r0:r1, :]
            d0 = 0
            target = sub_batch_days
            rows = r1 - r0
            pending = None
            pending_meta = None
            if use_prefetch:
                nxt = min(target, n_doy - d0)
                pending = prefetch_pool.submit(
                    _gather_block, cache_band, doy_ids[d0:d0 + nxt], n_time, rows, n_col)
                pending_meta = (d0, nxt)

            while d0 < n_doy:
                cur_target = min(target, n_doy - d0)
                if use_prefetch and pending is not None:
                    t0 = time.time()
                    stack_batch = pending.result()
                    gather_sec += time.time() - t0
                    cur_d0, cur_bsz = pending_meta
                    d_after = cur_d0 + cur_bsz
                    if d_after < n_doy:
                        nxt = min(target, n_doy - d_after)
                        pending = prefetch_pool.submit(
                            _gather_block, cache_band, doy_ids[d_after:d_after + nxt],
                            n_time, rows, n_col)
                        pending_meta = (d_after, nxt)
                    else:
                        pending = None
                else:
                    t0 = time.time()
                    stack_batch = _gather_block(
                        cache_band, doy_ids[d0:d0 + cur_target], n_time, rows, n_col)
                    gather_sec += time.time() - t0
                    cur_d0, cur_bsz = d0, cur_target

                try:
                    _compute_one_batch(stack_batch, cur_d0, cur_bsz, r0, r1)
                    d0 = cur_d0 + cur_bsz
                    del stack_batch
                except torch.cuda.OutOfMemoryError:
                    del stack_batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    if cur_bsz == 1:
                        raise
                    target = max(1, target // 2)
                    print("[gpu{}] OOM band[{}:{}] -> sub_batch_days {}".format(
                        gpu_id, r0, r1, target), flush=True)
                    if use_prefetch:
                        # Drop any in-flight prefetch and re-issue with new target.
                        if pending is not None:
                            try:
                                pending.result()
                            except Exception:
                                pass
                        nxt = min(target, n_doy - cur_d0)
                        pending = prefetch_pool.submit(
                            _gather_block, cache_band, doy_ids[cur_d0:cur_d0 + nxt],
                            n_time, rows, n_col)
                        pending_meta = (cur_d0, nxt)
                    # Loop will re-run cur_d0 with smaller target.
                    d0 = cur_d0

            torch.cuda.empty_cache()
            gc.collect()
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True)

    w_threads = max(1, int(getattr(args, "worker_write_threads", 1)))
    write_t0 = time.time()
    if w_threads <= 1:
        for i, doy in enumerate(doy_chunk):
            out_file = os.path.join(args.save_path, doy_to_mmdd(doy) + ".nc")
            write_output_nc(out_file, doy, lat_slice, lon, clim_all[i], p90_all[i])
    else:
        write_pool = ThreadPoolExecutor(max_workers=w_threads)
        futs = []
        for i, doy in enumerate(doy_chunk):
            out_file = os.path.join(args.save_path, doy_to_mmdd(doy) + ".nc")
            futs.append(write_pool.submit(
                write_output_nc, out_file, doy, lat_slice, lon, clim_all[i], p90_all[i]))
        for f in futs:
            f.result()
        write_pool.shutdown(wait=True)
    write_sec = time.time() - write_t0

    print("[gpu{}] done {} DOYs gather={:.1f}s compute={:.1f}s write={:.1f}s".format(
        gpu_id, len(doy_chunk), gather_sec, comp_sec, write_sec), flush=True)
    return gpu_id, len(doy_chunk), gather_sec, comp_sec, write_sec, init_sec


def _run_multi_gpu_cache(doys, args, cache, lat, lon, per_doy, idx_map, num_gpu):
    num_gpu = min(num_gpu, _slurm_visible_gpus(num_gpu))
    chunks = _split_doys_for_gpus(doys, num_gpu)
    n_workers = len(chunks)
    print("[multi-gpu] {} workers, DOY chunks: {} (fork+cuda:0..{})".format(
        n_workers, [len(c) for c in chunks], n_workers - 1), flush=True)

    _MP_SHARED.clear()
    _MP_SHARED.update({
        "args": args, "cache": cache, "lat": lat, "lon": lon,
        "per_doy": per_doy, "idx_map": idx_map,
    })
    tasks = [(i, chunk) for i, chunk in enumerate(chunks)]

    spawn_t0 = time.time()
    ctx = mp.get_context("fork")
    pool_t0 = time.time()
    with ctx.Pool(processes=n_workers) as pool:
        spawn_sec = pool_t0 - spawn_t0
        results = pool.map(_multi_gpu_worker, tasks)
    pool_wall = time.time() - pool_t0

    PROFILE.add("mp_spawn", spawn_sec, 1)
    max_gather = max(r[2] for r in results)
    max_comp = max(r[3] for r in results)
    max_write = max(r[4] for r in results)
    max_init = max(r[5] for r in results)
    for gpu_id, nd, gsec, csec, wsec, isec in results:
        print("[multi-gpu] gpu{} {} DOYs init={:.1f}s gather={:.1f}s compute={:.1f}s write={:.1f}s".format(
            gpu_id, nd, isec, gsec, csec, wsec), flush=True)
    PROFILE.add("setup", max_init, 1)
    PROFILE.add("io_gather", max_gather, len(doys))
    PROFILE.add("compute", pool_wall, len(doys))
    PROFILE.add("write", max_write, len(doys))
    print("[multi-gpu] wall {:.1f}s (parallel max~{:.1f}s compute per gpu)".format(
        pool_wall, max_comp), flush=True)


def _write_output_nc_profiled(out_path, doy, lat, lon, clim_row_major, p90_row_major):
    t0 = time.time()
    write_output_nc(out_path, doy, lat, lon, clim_row_major, p90_row_major)
    PROFILE.add("write", time.time() - t0, 1)


def _run_with_global_cache(doys, args, lat, lon, device_tag, torch_mod, pool, prefetch,
                           cache=None, read_sec=None, per_doy=None, idx_map=None):
    n_row = args.max_rows if args.max_rows is not None else args.row_total
    _, np_dt = resolve_compute_dtype(args.compute_dtype)

    if cache is None:
        with PROFILE.block("io_plan"):
            needed_dates, per_doy, idx_map = _build_needed_dates(doys, args)
        repeat_reads = len(doys) * len(next(iter(per_doy.values())))
        print("[cache] unique dates: {} (avoid {} repeated per-DOY reads)".format(
            len(needed_dates), repeat_reads), flush=True)
        with PROFILE.block("io_read"):
            cache, read_sec = load_time_stack(
                args.nc_path, needed_dates, args.var_sst, 0, n_row, args.col_total,
                args.io_workers, args.io_backend, np_dt, args.io_grain, pool=pool,
            )
    else:
        print("[cache] using preloaded cache (setup||io_read overlapped)", flush=True)

    print("[cache] shape={} read={:.1f}s".format(cache.shape, read_sec), flush=True)

    write_pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
    pending_write = None
    try:
        for doy in doys:
            print("\n=== DOY {} ({}) ===".format(doy, doy_to_mmdd(doy)), flush=True)
            ids = [idx_map[d] for d in per_doy[doy]]
            with PROFILE.block("io_gather"):
                t0 = time.time()
                stack = np.take(cache, ids, axis=0)
                print("  cache gather {:.2f}s".format(time.time() - t0), flush=True)
            if prefetch and pending_write is not None:
                pending_write.result()
            pending_write = _compute_and_maybe_async_write(
                doy, stack, args, lat, lon, device_tag, torch_mod, write_pool,
            )
            del stack
        if prefetch and pending_write is not None:
            pending_write.result()
    finally:
        if write_pool is not None:
            write_pool.shutdown(wait=True)
    del cache


def _compute_and_maybe_async_write(doy, stack, args, lat, lon, device_tag, torch_mod, write_pool):
    with PROFILE.block("compute"):
        clim, p90 = compute_clim_p90(
            stack, device_tag, torch_mod,
            gpu_lat_chunk=args.gpu_lat_chunk,
            compute_dtype=args.compute_dtype,
        )
    mmdd = doy_to_mmdd(doy)
    out_file = os.path.join(args.save_path, mmdd + ".nc")
    lat_slice = lat[:stack.shape[1]]
    if write_pool is not None:
        fut = write_pool.submit(
            _write_output_nc_profiled, out_file, doy, lat_slice, lon, clim, p90)
        print("  -> {} (write queued)".format(out_file), flush=True)
        return fut
    with PROFILE.block("write"):
        write_output_nc(out_file, doy, lat_slice, lon, clim, p90)
    print("  -> {}".format(out_file), flush=True)
    return None


def compute_one_doy(doy, nc_path, args, lat, lon, device_tag, torch_mod, pool):
    n_row = args.max_rows if args.max_rows is not None else args.row_total
    batch = max(1, args.batch_rows)
    clim = np.full((n_row, args.col_total), np.nan, dtype=np.float64)
    p90 = np.full((n_row, args.col_total), np.nan, dtype=np.float64)
    date_strs = window_date_strings_for_doy(
        doy, args.start_yr, args.end_yr, args.delta_day
    )
    _, np_dt = resolve_compute_dtype(args.compute_dtype)
    row0 = 0
    while row0 < n_row:
        n_blk = min(batch, n_row - row0)
        with PROFILE.block("io_read"):
            stack, _ = load_time_stack(
                nc_path, date_strs, args.var_sst, row0, n_blk, args.col_total,
                args.io_workers, args.io_backend, np_dt, args.io_grain, pool=pool,
            )
        with PROFILE.block("compute"):
            c_blk, p_blk = compute_clim_p90(
                stack, device_tag, torch_mod, gpu_lat_chunk=args.gpu_lat_chunk,
                compute_dtype=args.compute_dtype,
            )
        valid = np.sum(np.isfinite(stack), axis=0) > 0
        clim[row0:row0 + n_blk, :][valid] = c_blk[valid]
        p90[row0:row0 + n_blk, :][valid] = p_blk[valid]
        row0 += n_blk
    mmdd = doy_to_mmdd(doy)
    out_file = os.path.join(args.save_path, mmdd + ".nc")
    with PROFILE.block("write"):
        write_output_nc(out_file, doy, lat[:n_row], lon, clim, p90)
    print("  -> {}".format(out_file), flush=True)


def _load_cache_prefetch(doys, args, np_dt, pool_workers, num_gpu):
    """io_plan + io_read || setup(lat/lon + cuda probe). Returns cache bundle."""
    pool_workers = min(pool_workers, args.end_yr - args.start_yr + 1)
    with ProcessPoolExecutor(max_workers=pool_workers) as pool:
        with PROFILE.block("io_plan"):
            needed_dates, per_doy, idx_map = _build_needed_dates(doys, args)
        repeat_reads = len(doys) * len(next(iter(per_doy.values())))
        print("[cache] unique dates: {} (avoid {} repeated per-DOY reads)".format(
            len(needed_dates), repeat_reads), flush=True)
        print("[pipeline] io_read || setup in parallel", flush=True)

        io_box = {"cache": None, "read_sec": 0.0, "err": None}
        setup_box = {"lat": None, "lon": None, "cuda_count": 1, "setup_sec": 0.0, "err": None}
        n_row = args.row_total

        def _io_bg():
            try:
                with PROFILE.block("io_read"):
                    c, sec = load_time_stack(
                        args.nc_path, needed_dates, args.var_sst, 0, n_row, args.col_total,
                        args.io_workers, args.io_backend, np_dt, args.io_grain, pool=pool,
                    )
                io_box["cache"] = c
                io_box["read_sec"] = sec
            except Exception as exc:
                io_box["err"] = exc

        def _setup_bg():
            try:
                t0 = time.time()
                # Overlap parent torch import with io_read (COW saves child fork cost).
                if num_gpu > 1 and int(os.environ.get("MCC_PRE_IMPORT_TORCH", "1")) == 1:
                    ti = time.time()
                    try:
                        import torch  # noqa: F401
                        print("[setup] parent pre-imported torch {:.1f}s (parallel to io_read)".format(
                            time.time() - ti), flush=True)
                    except Exception as exc:
                        print("[setup] parent torch import failed: {}".format(exc), flush=True)
                probe = num_gpu > 1
                warm = num_gpu <= 1
                _, _, lat, lon, cuda_count, _ = _resolve_setup(
                    args, warmup_gpu=warm, probe_all_gpus=probe)
                setup_box["lat"] = lat
                setup_box["lon"] = lon
                setup_box["cuda_count"] = cuda_count
                setup_box["setup_sec"] = time.time() - t0
            except Exception as exc:
                setup_box["err"] = exc

        io_thread = threading.Thread(target=_io_bg, name="mcc-io-read")
        setup_thread = threading.Thread(target=_setup_bg, name="mcc-setup")
        io_thread.start()
        setup_thread.start()
        io_thread.join()
        setup_thread.join()
        if io_box["err"] is not None:
            raise io_box["err"]
        if setup_box["err"] is not None:
            raise setup_box["err"]

    PROFILE.add("setup", setup_box["setup_sec"], 1)
    print("[cache] shape={} read={:.1f}s".format(
        io_box["cache"].shape, io_box["read_sec"]), flush=True)
    return {
        "cache": io_box["cache"],
        "read_sec": io_box["read_sec"],
        "lat": setup_box["lat"],
        "lon": setup_box["lon"],
        "cuda_count": setup_box["cuda_count"],
        "per_doy": per_doy,
        "idx_map": idx_map,
    }


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    full_grid = args.max_rows is None
    doys = list(range(args.start_doy, args.end_doy + 1))
    use_cache = full_grid and args.cache_all_needed and len(doys) > 1
    prefetch = bool(args.prefetch)
    num_gpu = max(1, args.num_gpu)

    # NOTE: pre-import of torch happens inside _load_cache_prefetch's setup_bg
    # thread, so it overlaps with io_read (don't import here serially).

    dtype_name, np_dt = resolve_compute_dtype(args.compute_dtype)
    pool_workers = args.io_workers
    if args.io_grain == "year" and args.io_backend == "process":
        pool_workers = min(args.io_workers, args.end_yr - args.start_yr + 1)

    if use_cache and prefetch:
        bundle = _load_cache_prefetch(doys, args, np_dt, pool_workers, num_gpu)
        lat, lon = bundle["lat"], bundle["lon"]
        num_gpu = min(num_gpu, bundle["cuda_count"])
        if bundle["cuda_count"] < args.num_gpu:
            print("[WARN] requested {} DCU but probe sees {}; using {}".format(
                args.num_gpu, bundle["cuda_count"], num_gpu), flush=True)
        print("nc_path={}".format(args.nc_path))
        print("save_path={}".format(args.save_path))
        print("device={} compute_dtype={} num_gpu={} cuda_count={} io={}({}) grain={}".format(
            "multi" if num_gpu > 1 else "cuda:0",
            dtype_name, num_gpu, bundle["cuda_count"],
            args.io_workers, args.io_backend, args.io_grain))
        print("DOY {}..{}".format(args.start_doy, args.end_doy))

        if num_gpu > 1:
            _run_multi_gpu_cache(
                doys, args, bundle["cache"], lat, lon,
                bundle["per_doy"], bundle["idx_map"], num_gpu)
        else:
            device_tag, torch_mod = resolve_device(args.device)
            if device_tag != "cpu" and torch_mod is not None:
                _ = torch_mod.randn(1, device=device_tag)
                torch_mod.cuda.synchronize()
            with ProcessPoolExecutor(max_workers=pool_workers) as pool:
                _run_with_global_cache(
                    doys, args, lat, lon, device_tag, torch_mod, pool, prefetch=True,
                    cache=bundle["cache"], read_sec=bundle["read_sec"],
                    per_doy=bundle["per_doy"], idx_map=bundle["idx_map"],
                )
        del bundle["cache"]
    else:
        with ProcessPoolExecutor(max_workers=pool_workers) as pool:
            t0 = time.time()
            device_tag, torch_mod, lat, lon, cuda_count, _ = _resolve_setup(
                args, warmup_gpu=True, probe_all_gpus=(num_gpu > 1))
            PROFILE.add("setup", time.time() - t0, 1)
            num_gpu = min(num_gpu, cuda_count)

            print("nc_path={}".format(args.nc_path))
            print("save_path={}".format(args.save_path))
            print("device={} compute_dtype={} io={}({}) grain={} cache={} prefetch={}".format(
                device_tag, dtype_name, args.io_workers, args.io_backend,
                args.io_grain, args.cache_all_needed, int(prefetch)))
            print("DOY {}..{}".format(args.start_doy, args.end_doy))

            if num_gpu > 1 and use_cache:
                with PROFILE.block("io_plan"):
                    needed_dates, per_doy, idx_map = _build_needed_dates(doys, args)
                with PROFILE.block("io_read"):
                    cache, _ = load_time_stack(
                        args.nc_path, needed_dates, args.var_sst, 0, args.row_total,
                        args.col_total, args.io_workers, args.io_backend, np_dt,
                        args.io_grain, pool=pool,
                    )
                _run_multi_gpu_cache(
                    doys, args, cache, lat, lon, per_doy, idx_map, num_gpu)
                del cache
            elif not full_grid:
                for doy in doys:
                    print("\n=== DOY {} ({}) ===".format(doy, doy_to_mmdd(doy)), flush=True)
                    compute_one_doy(doy, args.nc_path, args, lat, lon, device_tag, torch_mod, pool)
            elif use_cache:
                _run_with_global_cache(
                    doys, args, lat, lon, device_tag, torch_mod, pool, prefetch,
                )
            else:
                write_pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
                pending_write = None
                try:
                    for doy in doys:
                        print("\n=== DOY {} ({}) ===".format(doy, doy_to_mmdd(doy)), flush=True)
                        date_strs = window_date_strings_for_doy(
                            doy, args.start_yr, args.end_yr, args.delta_day)
                        n_row = args.row_total
                        with PROFILE.block("io_read"):
                            stack, _ = load_time_stack(
                                args.nc_path, date_strs, args.var_sst, 0, n_row, args.col_total,
                                args.io_workers, args.io_backend, np_dt, args.io_grain, pool=pool,
                            )
                        if prefetch and pending_write is not None:
                            pending_write.result()
                        pending_write = _compute_and_maybe_async_write(
                            doy, stack, args, lat, lon, device_tag, torch_mod, write_pool,
                        )
                        del stack
                    if prefetch and pending_write is not None:
                        pending_write.result()
                finally:
                    if write_pool is not None:
                        write_pool.shutdown(wait=True)

    PROFILE.summary()


if __name__ == "__main__":
    main()
