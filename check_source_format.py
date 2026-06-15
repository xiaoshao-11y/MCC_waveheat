#!/usr/bin/env python
"""Check source NetCDF data format to determine the optimal I/O strategy.

Reads a sample .nc file and inspects the SST variable's dtype,
scale_factor, add_offset, and chunking layout. Determines whether
int16 packed-read is viable (halving I/O volume from 12 GB to ~6 GB).

Usage:
    python check_source_format.py [data_directory]
"""

import sys
from pathlib import Path
from netCDF4 import Dataset
import numpy as np

# Default DATA_DIR matching compute_server_mpi.py
DEFAULT_DATA_DIR = "/public/home/achwjznh4b/Newdata/"

# SST variable names to try (in order of preference)
SST_VAR_NAMES = ["data", "sst"]


def check_file(filepath, sst_var_name):
    """Inspect a single NetCDF file and return format info."""
    info = {"filepath": str(filepath), "sst_var": sst_var_name}

    with Dataset(filepath, "r") as nc:
        nc.set_auto_scale(True)
        nc.set_auto_mask(True)

        if sst_var_name not in nc.variables:
            info["error"] = f"variable '{sst_var_name}' not found"
            return info

        v = nc.variables[sst_var_name]

        # Basic dtype info
        info["dtype"] = str(v.dtype)
        info["dtype_kind"] = v.dtype.kind  # 'i' = int, 'f' = float
        info["shape"] = v.shape
        info["dimensions"] = v.dimensions

        # Scaling attributes
        info["scale_factor"] = getattr(v, "scale_factor", None)
        info["add_offset"] = getattr(v, "add_offset", None)
        info["_FillValue"] = getattr(v, "_FillValue", None)
        info["missing_value"] = getattr(v, "missing_value", None)
        info["valid_min"] = getattr(v, "valid_min", None)
        info["valid_max"] = getattr(v, "valid_max", None)

        # Chunking
        try:
            info["chunking"] = v.chunking()
        except Exception:
            info["chunking"] = "unknown (contiguous?)"

        # Sample raw values (without auto_scale)
        nc.set_auto_scale(False)
        nc.set_auto_mask(False)
        raw = v[:]
        info["raw_dtype"] = str(raw.dtype)
        if raw.dtype.kind == "i":
            # int storage: fill value is an integer like -999
            info["raw_min"] = int(raw.min())
            info["raw_max"] = int(raw.max())
            fv = info.get("_FillValue")
            if fv is not None:
                info["raw_fill_count"] = int((raw == fv).sum())
            else:
                info["raw_fill_count"] = 0
        else:
            # float storage: fill values are NaN
            info["raw_min"] = float(np.nanmin(raw))
            info["raw_max"] = float(np.nanmax(raw))
            info["raw_fill_count"] = int(np.isnan(raw).sum())

    return info


def find_sst_variable(filepath):
    """Determine which SST variable name the file uses."""
    with Dataset(filepath, "r") as nc:
        for name in SST_VAR_NAMES:
            if name in nc.variables:
                return name
    return None


def format_bytes(n_bytes):
    """Convert bytes to human-readable."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def check_all_consistency(data_dir, sst_var, n_files):
    """Check consistency of dtype, scale_factor, add_offset across N files."""
    import os

    files = sorted(
        [f for f in os.listdir(data_dir) if len(f) == 8 and f.isdigit()],
        key=lambda x: x,
    )
    if not files:
        # Try with .nc extension or avhrr pattern
        files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".nc")],
            key=lambda x: x,
        )

    if not files:
        print("  WARNING: No .nc files found in directory")
        return

    sample = min(n_files, len(files))
    print(f"\nChecking consistency across {sample}/{len(files)} files ...")

    dtypes = set()
    scale_factors = set()
    add_offsets = set()

    for fname in files[:sample]:
        fp = os.path.join(data_dir, fname)
        try:
            with Dataset(fp, "r") as nc:
                v = nc.variables[sst_var]
                dtypes.add(str(v.dtype))
                sf = getattr(v, "scale_factor", None)
                ao = getattr(v, "add_offset", None)
                scale_factors.add(sf)
                add_offsets.add(ao)
        except Exception:
            continue

    print(f"  dtypes found:        {dtypes}")
    print(f"  scale_factors found:  {scale_factors}")
    print(f"  add_offsets found:    {add_offsets}")

    if len(dtypes) > 1:
        print("  WARNING: Multiple dtypes detected — packed read may be unreliable")
    if len(scale_factors) > 1:
        print("  WARNING: Multiple scale_factors — need per-file tracking")
    if len(add_offsets) > 1:
        print("  WARNING: Multiple add_offsets — need per-file tracking")

    all_consistent = len(dtypes) == 1 and len(scale_factors) == 1 and len(add_offsets) == 1
    if all_consistent:
        print("  All files consistent — packed read is safe")


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_DATA_DIR)

    if not data_dir.is_dir():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    print(f"Data directory: {data_dir}")

    # Find a sample file and determine SST variable name
    sample = None
    sst_var = None

    for f in sorted(data_dir.iterdir()):
        name = f.name
        # Support both: YYYYMMDD (no ext) and YYYYMMDD.nc / avhrr-only-v2.YYYYMMDD.nc
        fname_no_ext = name.replace(".nc", "")
        if (fname_no_ext.isdigit() and len(fname_no_ext) == 8) or "avhrr" in fname_no_ext:
            sst_var = find_sst_variable(str(f))
            if sst_var:
                sample = str(f)
                break

    if sample is None or sst_var is None:
        print("ERROR: No readable .nc files with SST data found")
        sys.exit(1)

    print(f"Sample file: {sample}")
    print(f"SST variable: '{sst_var}'")

    # Inspect sample
    info = check_file(sample, sst_var)

    print(f"\n{'='*60}")
    print(f"  SOURCE DATA FORMAT")
    print(f"{'='*60}")
    print(f"  Storage dtype:       {info.get('dtype', 'N/A')}")
    print(f"  Raw dtype:           {info.get('raw_dtype', 'N/A')}")
    print(f"  Shape:               {info.get('shape', 'N/A')}")
    print(f"  Dimensions:          {info.get('dimensions', 'N/A')}")
    print(f"  Scale factor:        {info.get('scale_factor', 'N/A')}")
    print(f"  Add offset:          {info.get('add_offset', 'N/A')}")
    print(f"  Fill value:          {info.get('_FillValue', 'N/A')}")
    print(f"  Valid range:         [{info.get('valid_min', '?')}, "
          f"{info.get('valid_max', '?')}]")
    print(f"  Chunking:            {info.get('chunking', 'N/A')}")
    print(f"  Raw value range:     [{info.get('raw_min', '?')}, "
          f"{info.get('raw_max', '?')}]")
    print(f"  Raw fill count:      {info.get('raw_fill_count', '?')}")

    # Analysis
    is_int16 = info.get("dtype_kind") == "i"
    scale = info.get("scale_factor")
    offset = info.get("add_offset")
    fill = info.get("_FillValue")

    if is_int16:
        n_lat = info["shape"][-2]
        n_lon = info["shape"][-1]
        n_files = 3060
        float32_size = n_files * n_lat * n_lon * 4
        int16_size = n_files * n_lat * n_lon * 2

        print(f"\n{'='*60}")
        print(f"  I/O SIZE ESTIMATE (3060 files, {n_lat}x{n_lon})")
        print(f"{'='*60}")
        print(f"  float32 read:  {format_bytes(float32_size)}")
        print(f"  int16 packed:  {format_bytes(int16_size)}")
        print(f"  I/O savings:   {format_bytes(float32_size - int16_size)} "
              f"({(1 - int16_size/float32_size)*100:.0f}%)")

    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION")
    print(f"{'='*60}")

    if is_int16 and scale is not None:
        print(f"  PACKED READ VIABLE")
        print(f"  Read raw int16 with set_auto_scale(False) + set_auto_mask(False)")
        print(f"  Scale on GPU: data_f32 = raw_i16 * {scale} + {offset}")
        print(f"  Expected I/O time reduction: ~50%")
    elif is_int16 and scale is None:
        print(f"  int16 but no scale_factor — packed read still viable (read raw, no scaling needed)")
    else:
        print(f"  Source is float32 — continue current path (no packed read benefit)")
        print(f"  Inode sorting + reduced workers still apply")

    # Consistency check
    check_all_consistency(str(data_dir), sst_var, 30)


if __name__ == "__main__":
    main()
