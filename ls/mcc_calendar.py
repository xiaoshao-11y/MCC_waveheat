# -*- coding: utf-8 -*-
"""Calendar / DOY helpers aligned with get_climatology.m (365-day, no Feb 29)."""
from datetime import date, timedelta


def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def year_dates_no_feb29(year):
    """返回某年 365 个 date（闰年去掉 2 月 29 日）。"""
    days = []
    cur = date(year, 1, 1)
    end = date(year, 12, 31)
    while cur <= end:
        if not (is_leap(year) and cur.month == 2 and cur.day == 29):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def matlab_doy(d):
    """日历日对应的 DOY（1–365，无 2 月 29 日）。"""
    for i, dd in enumerate(year_dates_no_feb29(d.year), start=1):
        if dd == d:
            return i
    raise ValueError(f"date not in 365-day calendar: {d}")


def doy_to_mmdd(doy, base_year=2020):
    """DOY -> MMDD 字符串（用于输出文件名）。"""
    d = year_dates_no_feb29(base_year)[doy - 1]
    return d.strftime("%m%d")


def window_dates_for_doy(doy, start_yr=1991, end_yr=2020, delta_day=5):
    """
    对给定 DOY，返回 30×11 个日历日（去重前为列表），
    与 MATLAB date_range 逻辑一致。
    """
    out = []
    for yr in range(start_yr, end_yr + 1):
        center = year_dates_no_feb29(yr)[doy - 1]
        for off in range(-delta_day, delta_day + 1):
            out.append(center + timedelta(days=off))
    return out


def window_date_strings_for_doy(doy, start_yr=1991, end_yr=2020, delta_day=5):
    """返回 YYYYMMDD 字符串列表（330 个）。"""
    return [d.strftime("%Y%m%d") for d in window_dates_for_doy(doy, start_yr, end_yr, delta_day)]
