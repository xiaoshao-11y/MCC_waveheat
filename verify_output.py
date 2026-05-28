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

rmse_clim = np.full(92, np.nan)
rmse_P90  = np.full(92, np.nan)
day_index = 0

import xarray as xr

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

for month in (6, 7, 8):
    for day in range(1, days_in_month[month - 1] + 1):
        fname = f"{month:02d}{day:02d}.nc"

        ref_file = REF_CLIM_PATH / fname
        our_file = OUR_CLIM_PATH / fname

        if not ref_file.exists():
            print(f"  WARNING: 参考文件不存在: {ref_file}")
            day_index += 1
            continue
        if not our_file.exists():
            print(f"  WARNING: 输出文件不存在: {our_file}")
            day_index += 1
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
        day_index += 1

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
