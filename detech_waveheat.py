import numpy as np
from datetime import date

def calculate_climatology(t, temp,
                          climatologyPeriod=[1991, 2020],
                          pctile=90,
                          windowHalfWidth=5,
                          smoothPercentile=False,      # 赛题要求不进行滑动平均
                          smoothPercentileWidth=31,
                          maxPadLength=False,
                          coldSpells=False,
                          alternateClimatology=False,
                          Ly=False):
    """
    仅计算海洋热浪阈值和气候态海温，不进行事件检测。
    返回一个字典：
        clim['thresh'] : 每日阈值 (长度与输入 t 相同)
        clim['seas']   : 每日气候平均值
        clim['missing']: 原始缺失值标记
    """

    T = len(t)
    year = np.zeros(T, dtype=int)
    month = np.zeros(T, dtype=int)
    day = np.zeros(T, dtype=int)
    doy = np.zeros(T, dtype=int)
    for i in range(T):
        d = date.fromordinal(t[i])
        year[i] = d.year
        month[i] = d.month
        day[i] = d.day

    # 闰年基线（只用于 DOY 计算）
    year_leap = 2012
    t_leap = np.arange(date(year_leap, 1, 1).toordinal(),
                       date(year_leap, 12, 31).toordinal() + 1)
    mon_leap = np.array([date.fromordinal(tt).month for tt in t_leap])
    day_leap = np.array([date.fromordinal(tt).day for tt in t_leap])
    doy_leap = np.array([tt - date(date.fromordinal(tt).year, 1, 1).toordinal() + 1
                         for tt in t_leap])
    for i in range(T):
        doy[i] = doy_leap[(mon_leap == month[i]) & (day_leap == day[i])]

    feb28, feb29 = 59, 60

    # 气候基准期年份
    if climatologyPeriod[0] is None or climatologyPeriod[1] is None:
        climatologyPeriod[0] = year[0]
        climatologyPeriod[1] = year[-1]

    # 准备用于气候态计算的数据
    if alternateClimatology:
        tClim = alternateClimatology[0]
        tempClim = alternateClimatology[1]
        TClim = len(tClim)
        yearClim = np.zeros(TClim, dtype=int)
        monthClim = np.zeros(TClim, dtype=int)
        dayClim = np.zeros(TClim, dtype=int)
        doyClim = np.zeros(TClim, dtype=int)
        for i in range(TClim):
            d = date.fromordinal(tClim[i])
            yearClim[i] = d.year
            monthClim[i] = d.month
            dayClim[i] = d.day
            doyClim[i] = doy_leap[(mon_leap == monthClim[i]) & (day_leap == dayClim[i])]
    else:
        tempClim = temp.copy()
        TClim = T
        yearClim = year.copy()
        monthClim = month.copy()
        dayClim = day.copy()
        doyClim = doy.copy()

    if coldSpells:
        tempClim = -1.0 * tempClim

    if maxPadLength:
        from scipy import ndimage  
        tempClim = pad(tempClim, maxPadLength=maxPadLength)

    # ---------- 计算每日阈值和气候态 ----------
    # 按比赛要求仅计算 6-8 月（DOY 152~243），可缩小循环范围
    start_doy = 1        # 可改为 152
    end_doy = 366        # 可改为 243
    lenClimYear = 366

    clim_start = np.where(yearClim == climatologyPeriod[0])[0][0]
    clim_end   = np.where(yearClim == climatologyPeriod[1])[0][-1]

    thresh_climYear = np.full(lenClimYear, np.nan)
    seas_climYear   = np.full(lenClimYear, np.nan)

    for d in range(start_doy, end_doy + 1):
        if d == feb29:
            continue
        # 寻找气候基准期内该 DOY 的位置
        tt0 = np.where(doyClim[clim_start:clim_end+1] == d)[0]
        if len(tt0) == 0:
            continue
        # 以当天为中心，前后各 windowHalfWidth 天
        tt = np.array([], dtype=int)
        for w in range(-windowHalfWidth, windowHalfWidth + 1):
            tt = np.append(tt, clim_start + tt0 + w)
        tt = tt[(tt >= 0) & (tt < TClim)]
        thresh_climYear[d-1] = np.nanpercentile(tempClim[tt], pctile)
        seas_climYear[d-1]   = np.nanmean(tempClim[tt])

    # 2月29日线性插值
    thresh_climYear[feb29-1] = 0.5 * (thresh_climYear[feb29-2] + thresh_climYear[feb29])
    seas_climYear[feb29-1]   = 0.5 * (seas_climYear[feb29-2]   + seas_climYear[feb29])

    # 31天滑动平均–按赛题要求关闭
    if smoothPercentile:
        if Ly:
            valid = ~np.isnan(thresh_climYear)
            thresh_climYear[valid] = runavg(thresh_climYear[valid], smoothPercentileWidth)
            valid = ~np.isnan(seas_climYear)
            seas_climYear[valid] = runavg(seas_climYear[valid], smoothPercentileWidth)
        else:
            thresh_climYear = runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear   = runavg(seas_climYear, smoothPercentileWidth)

    # 扩展到完整时间序列
    clim = {
        'thresh': thresh_climYear[doy.astype(int) - 1],
        'seas':   seas_climYear[doy.astype(int) - 1],
        'missing': np.isnan(temp)   
    }

    return clim


def runavg(ts, w):
    N = len(ts)
    ts = np.concatenate([ts, ts, ts])
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    return ts_smooth[N:2*N]

def pad(data, maxPadLength=False):
    from scipy import ndimage
    data_padded = data.copy()
    bad = np.isnan(data)
    good = ~bad
    data_padded[bad] = np.interp(bad.nonzero()[0], good.nonzero()[0], data[good])
    if maxPadLength:
        blocks, n_blocks = ndimage.label(np.isnan(data))
        for bl in range(1, n_blocks+1):
            if (blocks == bl).sum() > maxPadLength:
                data_padded[blocks == bl] = np.nan
    return data_padded