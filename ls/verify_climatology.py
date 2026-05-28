#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python port of clim_verification.m: RMSE vs reference.

用法:
  python verify_climatology.py \\
    --ref-path /public/home/achwjznh4b/ERA5/Climatology/ \\
    --test-path $HOME/ERA5/Climatology/
"""
import argparse
import os
import sys

if sys.version_info[0] < 3:
    sys.exit("Please use: python3 verify_climatology.py")

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("需要 netCDF4: pip install netCDF4")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-path", required=True, help="参考解目录")
    p.add_argument("--test-path", required=True, help="选手结果目录")
    p.add_argument("--out-path", default="./verification/")
    return p.parse_args()


def align_to_ref(test_arr, ref_arr):
    """参考解为 (721,1440) Lat×Lon；旧输出误为 (1440,721) 时自动转置。"""
    t = np.asarray(test_arr)
    r = np.asarray(ref_arr)
    if t.shape == r.shape:
        return t
    if t.shape == r.shape[::-1]:
        return t.T
    raise ValueError(
        "shape mismatch: test {} vs ref {} (expected (721,1440) or transpose)".format(
            t.shape, r.shape
        )
    )


def rmse(a, b):
    d = a - b
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.mean(d[m] ** 2)))


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    summer = [(6, 31), (7, 31), (8, 31)]
    rmse_clim = []
    rmse_p90 = []

    for month, ndays in summer:
        for day in range(1, ndays + 1):
            fn = f"{month:02d}{day:02d}.nc"
            ref_f = os.path.join(args.ref_path, fn)
            test_f = os.path.join(args.test_path, fn)
            if not os.path.isfile(ref_f):
                print(f"跳过（无参考）: {fn}")
                continue
            if not os.path.isfile(test_f):
                print(f"缺少选手文件: {test_f}")
                sys.exit(1)

            with nc.Dataset(ref_f) as dr, nc.Dataset(test_f) as dt:
                rc = dr.variables["Climmean"][:]
                rp = dr.variables["P90_sst"][:]
                tc = align_to_ref(dt.variables["Climmean"][:], rc)
                tp = align_to_ref(dt.variables["P90_sst"][:], rp)

            rmse_clim.append(rmse(rc, tc))
            rmse_p90.append(rmse(rp, tp))
            print(f"{fn}: RMSE clim={rmse_clim[-1]:.4f}, P90={rmse_p90[-1]:.4f}")

    rmse_clim = np.asarray(rmse_clim)
    rmse_p90 = np.asarray(rmse_p90)
    r_clim = np.nanmean(rmse_clim)
    r_p90 = np.nanmean(rmse_p90)

    print("\n=========================================")
    print("       6~8月夏季 RMSE 统计结果")
    print("=========================================")
    print(f"气候态 平均误差 : {r_clim:.4f} °C  (要求 < 2°C)")
    print(f"气候态 最大日误差 : {np.nanmax(rmse_clim):.4f} °C  (要求 < 1°C)")
    print(f"P90分位 平均误差 : {r_p90:.4f} °C")
    print(f"P90分位 最大日误差 : {np.nanmax(rmse_p90):.4f} °C")
    print("=========================================")

    ok = (r_clim < 2.0 and r_p90 < 2.0 and
          np.nanmax(rmse_clim) < 1.0 and np.nanmax(rmse_p90) < 1.0)
    print("赛题精度: ", "通过" if ok else "未通过")

    with open(os.path.join(args.out_path, "RMSE_clim.txt"), "w") as f:
        f.write(f"{r_clim:.4f}\n")
    with open(os.path.join(args.out_path, "RMSE_P90.txt"), "w") as f:
        f.write(f"{r_p90:.4f}\n")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
