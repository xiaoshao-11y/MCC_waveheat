#!/usr/bin/env python
"""验证 compute_server.py 的输出与参考标准的一致性（RMSE）。
   替代 clim_verification.m，无需 MATLAB。
"""
import numpy as np
from pathlib import Path

REF_CLIM_PATH   = Path("/public/home/achwjznh4b/ERA5/Climatology/")
OUR_CLIM_PATH   = Path(__file__).resolve().parent / "ERA5" / "Climatology"
SAVE_PATH       = Path(__file__).resolve().parent / "verification"

SAVE_PATH.mkdir(parents=True, exist_ok=True)

import xarray as xr

# Collect all .nc files from output dir, match with reference
our_files = sorted(OUR_CLIM_PATH.glob("*.nc"))
if not our_files:
    print("ERROR: No output files found!")
    exit(1)

n_files = len(our_files)
rmse_clim = np.full(n_files, np.nan)
rmse_P90  = np.full(n_files, np.nan)

for day_index, our_file in enumerate(our_files):
    fname = our_file.name
    ref_file = REF_CLIM_PATH / fname

    if not ref_file.exists():
        print(f"  WARNING: 参考文件不存在: {ref_file}")
        continue
    if not our_file.exists():
        print(f"  WARNING: 输出文件不存在: {our_file}")
        continue

    print(f"  读取: {fname}")

    ref_ds = xr.open_dataset(ref_file)
    our_ds = xr.open_dataset(our_file)

    rmse_clim[day_index] = np.sqrt(
        np.nanmean((ref_ds["Climmean"].values - our_ds["Climmean"].values) ** 2))
    rmse_P90[day_index] = np.sqrt(
        np.nanmean((ref_ds["P90_sst"].values - our_ds["P90_sst"].values) ** 2))

    ref_ds.close()
    our_ds.close()

R_clim   = np.nanmean(rmse_clim)
R_P90    = np.nanmean(rmse_P90)
max_clim = np.nanmax(rmse_clim)
max_P90  = np.nanmax(rmse_P90)

print(f"\n{'='*50}")
print(f"       6~8月夏季 RMSE 统计结果")
print(f"{'='*50}")
print(f"气候态 平均误差 : {R_clim:.4f} °C")
print(f"气候态 最大误差 : {max_clim:.4f} °C")
print(f"{'-'*50}")
print(f"P90分位 平均误差 : {R_P90:.4f} °C")
print(f"P90分位 最大误差 : {max_P90:.4f} °C")
print(f"{'='*50}")

with open(SAVE_PATH / "RMSE_clim.txt", "w") as f:
    f.write(f"{R_clim:.4f}\n")
with open(SAVE_PATH / "RMSE_P90.txt", "w") as f:
    f.write(f"{R_P90:.4f}\n")

print(f"\n结果已保存到 {SAVE_PATH}/")
