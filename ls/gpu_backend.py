# -*- coding: utf-8 -*-
"""GPU/CPU backend — supports float16 / float32 compute; output promoted to float64."""
from __future__ import print_function

import os

import numpy as np


def resolve_compute_dtype(name):
    """float16 | float32 | float64 (aliases: fp16, half, fp32, fp64)."""
    key = (name or os.environ.get("MCC_COMPUTE_DTYPE", "float16")).lower()
    aliases = {
        "fp16": "float16", "half": "float16", "f16": "float16",
        "fp32": "float32", "f32": "float32",
        "fp64": "float64", "f64": "float64", "double": "float64",
    }
    key = aliases.get(key, key)
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    if key not in mapping:
        raise ValueError("Unknown compute dtype: {}".format(name))
    return key, mapping[key]


def torch_dtype_from_np(np_dtype, torch_mod):
    if np_dtype == np.float16:
        return torch_mod.float16
    if np_dtype == np.float32:
        return torch_mod.float32
    return torch_mod.float64


def resolve_device(requested):
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu", None

    try:
        import torch
    except ImportError:
        if req in ("auto", "cpu"):
            print("[gpu_backend] PyTorch not found, using NumPy CPU")
            return "cpu", None
        raise RuntimeError("PyTorch required for --device={}".format(requested))

    if req in ("auto", "gpu", "cuda", "dcu"):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print("[gpu_backend] PyTorch device: cuda:0 ({})".format(name))
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            return "cuda", torch
        if req != "auto":
            raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
        print("[gpu_backend] No CUDA/DCU visible, using NumPy CPU")
        return "cpu", None

    raise ValueError("Unknown --device: {}".format(requested))


def _p90_along_time_gpu(t, torch_mod, pctile=90.0):
    inf = torch_mod.tensor(float("inf"), device=t.device, dtype=t.dtype)
    filled = torch_mod.where(torch_mod.isnan(t), inf, t)
    n = t.shape[0]
    if n <= 1:
        return torch_mod.squeeze(filled, dim=0)
    k0 = int((pctile / 100.0) * (n - 1))
    k1 = min(k0 + 1, n - 1)
    w = (pctile / 100.0) * (n - 1) - k0
    v0, _ = torch_mod.kthvalue(filled, k0 + 1, dim=0)
    if k1 == k0:
        out = v0
    else:
        v1, _ = torch_mod.kthvalue(filled, k1 + 1, dim=0)
        out = v0 + (v1 - v0) * w
    return torch_mod.where(torch_mod.isinf(out), torch_mod.nan, out)


def compute_clim_p90(stack, device_tag, torch_mod, pctile=90.0, gpu_lat_chunk=0,
                     compute_dtype="float16"):
    """
    stack: (n_time, n_lat, n_lon). Internal math uses compute_dtype (default float16).
    Returns float64 arrays for NetCDF output.
    """
    _, np_dt = resolve_compute_dtype(compute_dtype)
    stack = np.ascontiguousarray(stack, dtype=np_dt)

    if device_tag == "cpu" or torch_mod is None:
        clim = np.nanmean(stack, axis=0).astype(np.float64)
        p90 = np.nanpercentile(stack, pctile, axis=0).astype(np.float64)
        return clim, p90

    use_fast = os.environ.get("MCC_FAST_P90", "1") != "0"
    t_dtype = torch_dtype_from_np(np_dt, torch_mod)
    n_lat = stack.shape[1]

    with torch_mod.inference_mode():
        if gpu_lat_chunk <= 0 or n_lat <= gpu_lat_chunk:
            t = torch_mod.as_tensor(stack, device=device_tag, dtype=t_dtype)
            clim = torch_mod.nanmean(t, dim=0)
            if use_fast:
                p90 = _p90_along_time_gpu(t, torch_mod, pctile)
            else:
                p90 = torch_mod.nanquantile(t, pctile / 100.0, dim=0)
            return clim.float().cpu().numpy().astype(np.float64), \
                p90.float().cpu().numpy().astype(np.float64)

        clim_out = np.empty((n_lat, stack.shape[2]), dtype=np.float64)
        p90_out = np.empty((n_lat, stack.shape[2]), dtype=np.float64)
        for r0 in range(0, n_lat, gpu_lat_chunk):
            r1 = min(r0 + gpu_lat_chunk, n_lat)
            slab = stack[:, r0:r1, :]
            t = torch_mod.as_tensor(slab, device=device_tag, dtype=t_dtype)
            c = torch_mod.nanmean(t, dim=0)
            p = _p90_along_time_gpu(t, torch_mod, pctile) if use_fast else \
                torch_mod.nanquantile(t, pctile / 100.0, dim=0)
            clim_out[r0:r1, :] = c.float().cpu().numpy()
            p90_out[r0:r1, :] = p.float().cpu().numpy()
        return clim_out, p90_out
