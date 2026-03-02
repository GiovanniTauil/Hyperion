import numpy as np

def gregorian_to_doy(year, month, day, hour=0, minute=0, second=0.0):
    """
    Convert Gregorian calendar date to Year and Day of Year (DOY).
    
    Parameters
    ----------
    year : int or array_like
        Year (e.g., 2023)
    month : int or array_like
        Month (1 to 12)
    day : int or array_like
        Day of month (1 to 31)
    hour : int or array_like, optional
        Hour of day (0 to 23), by default 0
    minute : int or array_like, optional
        Minute of hour (0 to 59), by default 0
    second : float or array_like, optional
        Second of minute, by default 0.0
        
    Returns
    -------
    tuple
        (year, doy, hour, minute, second)
    """
    year = np.atleast_1d(year).astype(int)
    month = np.atleast_1d(month).astype(int)
    day = np.atleast_1d(day).astype(int)
    hour = np.atleast_1d(hour)
    minute = np.atleast_1d(minute)
    second = np.atleast_1d(second)
    
    leap_year = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    days_in_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cum_days = np.cumsum(days_in_month)
    
    doy = cum_days[month - 1] + day
    doy[leap_year & (month > 2)] += 1
    
    if year.size == 1:
        return int(year[0]), int(doy[0]), hour[0], minute[0], second[0]
    return year, doy, hour, minute, second

def doy_to_gregorian(year, doy, hour=0, minute=0, second=0.0):
    """
    Convert Year and Day of Year (DOY) to Gregorian calendar date.
    
    Parameters
    ----------
    year : int or array_like
        Year (e.g., 2023)
    doy : int or array_like
        Day of year (1 to 365 or 366 for leap years)
    hour : int or array_like, optional
        Hour of day (0 to 23), by default 0
    minute : int or array_like, optional
        Minute of hour (0 to 59), by default 0
    second : float or array_like, optional
        Second of minute, by default 0.0
        
    Returns
    -------
    tuple
        (year, month, day, hour, minute, second)
    """
    year = np.atleast_1d(year).astype(int)
    doy = np.atleast_1d(doy).astype(int)
    hour = np.atleast_1d(hour)
    minute = np.atleast_1d(minute)
    second = np.atleast_1d(second)
    
    leap_year = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    
    month = np.zeros_like(year)
    day = np.zeros_like(year)
    
    for i in range(year.size):
        days = days_in_month.copy()
        if leap_year[i]:
            days[1] = 29
        cum_days = np.insert(np.cumsum(days), 0, 0)
        m = np.searchsorted(cum_days, doy[i], side='left')
        month[i] = m
        day[i] = doy[i] - cum_days[m - 1]
        
    if year.size == 1:
        return int(year[0]), int(month[0]), int(day[0]), hour[0], minute[0], second[0]
    return year, month, day, hour, minute, second

def gregorian_to_jd(year, month, day, hour=0, minute=0, second=0.0):
    """
    Convert Gregorian calendar date to Julian Date (JD).
    
    Parameters
    ----------
    year : int or array_like
        Year (e.g., 2023)
    month : int or array_like
        Month (1 to 12)
    day : int or array_like
        Day of month (1 to 31)
    hour : int or array_like, optional
        Hour of day (0 to 23), by default 0
    minute : int or array_like, optional
        Minute of hour (0 to 59), by default 0
    second : float or array_like, optional
        Second of minute, by default 0.0
        
    Returns
    -------
    float or array_like
        Julian Date (JD)
    """
    year = np.atleast_1d(year).astype(int)
    month = np.atleast_1d(month).astype(int)
    day = np.atleast_1d(day).astype(float)
    hour = np.atleast_1d(hour).astype(float)
    minute = np.atleast_1d(minute).astype(float)
    second = np.atleast_1d(second).astype(float)
    
    y = year.copy()
    m = month.copy()
    
    mask = m <= 2
    y[mask] -= 1
    m[mask] += 12
    
    A = np.floor(y / 100)
    B = 2 - A + np.floor(A / 4)
    
    jd = np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1)) + day + B - 1524.5
    jd_fraction = (hour + minute / 60.0 + second / 3600.0) / 24.0
    jd += jd_fraction
    
    if jd.size == 1:
        return float(jd[0])
    return jd

def jd_to_gregorian(jd):
    """
    Convert Julian Date (JD) to Gregorian calendar date.
    
    Parameters
    ----------
    jd : float or array_like
        Julian Date (JD)
        
    Returns
    -------
    tuple
        (year, month, day, hour, minute, second)
    """
    jd = np.atleast_1d(jd).astype(float)
    
    jd_adjusted = jd + 0.5
    Z = np.floor(jd_adjusted)
    F = jd_adjusted - Z
    
    A = np.zeros_like(Z)
    alpha = np.zeros_like(Z)
    
    mask = Z < 2299161
    A[mask] = Z[mask]
    
    alpha[~mask] = np.floor((Z[~mask] - 1867216.25) / 36524.25)
    A[~mask] = Z[~mask] + 1 + alpha[~mask] - np.floor(alpha[~mask] / 4)
    
    B = A + 1524
    C = np.floor((B - 122.1) / 365.25)
    D = np.floor(365.25 * C)
    E = np.floor((B - D) / 30.6001)
    
    day_fraction = B - D - np.floor(30.6001 * E) + F
    day = np.floor(day_fraction)
    
    fraction_of_day = day_fraction - day
    hour = np.floor(fraction_of_day * 24.0)
    minute = np.floor((fraction_of_day * 24.0 - hour) * 60.0)
    second = (fraction_of_day * 1440.0 - hour * 60.0 - minute) * 60.0
    
    # Handle floating point inaccuracies
    minute[second >= 59.9999999] += 1
    second[second >= 59.9999999] = 0.0
    hour[minute >= 60] += 1
    minute[minute >= 60] = 0
    
    # We ignore potential rollover for hour to day to keep array-wise operations simple, 
    # but the precision loss usually only affects fractional seconds.
    
    month = np.zeros_like(E)
    month[E < 14] = E[E < 14] - 1
    month[E >= 14] = E[E >= 14] - 13
    
    year = np.zeros_like(C)
    year[month > 2] = C[month > 2] - 4716
    year[month <= 2] = C[month <= 2] - 4715
    
    if year.size == 1:
        return int(year[0]), int(month[0]), int(day[0]), int(hour[0]), int(minute[0]), float(second[0])
    return year.astype(int), month.astype(int), day.astype(int), hour.astype(int), minute.astype(int), second

def gregorian_to_mjd(year, month, day, hour=0, minute=0, second=0.0):
    """
    Convert Gregorian calendar date to Modified Julian Date (MJD).
    MJD = JD - 2400000.5.
    
    Parameters
    ----------
    year : int or array_like
        Year (e.g., 2023)
    month : int or array_like
        Month (1 to 12)
    day : int or array_like
        Day of month (1 to 31)
    hour : int or array_like, optional
        Hour of day (0 to 23), by default 0
    minute : int or array_like, optional
        Minute of hour (0 to 59), by default 0
    second : float or array_like, optional
        Second of minute, by default 0.0
        
    Returns
    -------
    float or array_like
        Modified Julian Date (MJD)
    """
    jd = gregorian_to_jd(year, month, day, hour, minute, second)
    return jd - 2400000.5

def mjd_to_gregorian(mjd):
    """
    Convert Modified Julian Date (MJD) to Gregorian calendar date.
    
    Parameters
    ----------
    mjd : float or array_like
        Modified Julian Date (MJD)
        
    Returns
    -------
    tuple
        (year, month, day, hour, minute, second)
    """
    mjd = np.atleast_1d(mjd).astype(float)
    jd = mjd + 2400000.5
    return jd_to_gregorian(jd)

def gregorian_to_gps_time(year, month, day, hour=0, minute=0, second=0.0):
    """
    Convert Gregorian calendar date to GPS Time.
    GPS Epoch is defined as GMT 1980-01-06 00:00:00.0.
    
    Parameters
    ----------
    year : int or array_like
        Year (e.g., 2023)
    month : int or array_like
        Month (1 to 12)
    day : int or array_like
        Day of month (1 to 31)
    hour : int or array_like, optional
        Hour of day (0 to 23), by default 0
    minute : int or array_like, optional
        Minute of hour (0 to 59), by default 0
    second : float or array_like, optional
        Second of minute, by default 0.0
        
    Returns
    -------
    tuple
        (gps_week, gps_seconds)
    """
    jd_gps_epoch = 2444244.5 
    
    jd = gregorian_to_jd(year, month, day, hour, minute, second)
    jd = np.atleast_1d(jd)
    
    delta_days = jd - jd_gps_epoch
    gps_time_seconds = delta_days * 86400.0
    
    gps_week = np.floor(gps_time_seconds / 604800.0)
    gps_seconds = gps_time_seconds - gps_week * 604800.0
    
    if gps_week.size == 1:
        return int(gps_week[0]), float(gps_seconds[0])
    return gps_week.astype(int), gps_seconds

def gps_time_to_gregorian(gps_week, gps_seconds):
    """
    Convert GPS Time to Gregorian calendar date.
    
    Parameters
    ----------
    gps_week : int or array_like
        GPS Week 
    gps_seconds : float or array_like
        Seconds of GPS Week
        
    Returns
    -------
    tuple
        (year, month, day, hour, minute, second)
    """
    gps_week = np.atleast_1d(gps_week).astype(float)
    gps_seconds = np.atleast_1d(gps_seconds).astype(float)
    
    jd_gps_epoch = 2444244.5
    
    delta_days = (gps_week * 604800.0 + gps_seconds) / 86400.0
    jd = jd_gps_epoch + delta_days
    
    return jd_to_gregorian(jd)
