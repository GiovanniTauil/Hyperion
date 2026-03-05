import pandas as pd
import numpy as np

def _interpolate_linear(t_orig, y_orig, t_new):
    """
    Applies linear interpolation for a 1D series.
    Missing data (NaNs) in the bounding interval will result in NaN.
    """
    y_new = np.full(len(t_new), np.nan)
    n = len(t_orig)
    
    if n < 2:
        return y_new
        
    idxs = np.searchsorted(t_orig, t_new)
    
    for i, t in enumerate(t_new):
        idx = idxs[i]
        
        # Exact match logic
        exact_match = np.where(np.isclose(t_orig, t, rtol=0.0, atol=1e-3))[0]
        if len(exact_match) > 0:
            y_new[i] = y_orig[exact_match[0]]
            continue
            
        if idx == 0 or idx >= n:
            # Out of bounds
            continue
            
        t0, t1 = t_orig[idx-1], t_orig[idx]
        y0, y1 = y_orig[idx-1], y_orig[idx]
        
        if np.isnan(y0) or np.isnan(y1):
            y_new[i] = np.nan
        else:
            y_new[i] = y0 + (t - t0) * (y1 - y0) / (t1 - t0)
            
    return y_new

def _lagrange_interpolate(x, y, x_new):
    """
    Evaluates the Lagrange polynomial defined by (x, y) at point x_new.
    """
    n = len(x)
    y_new = 0.0
    for i in range(n):
        p = 1.0
        for j in range(n):
            if i != j:
                p *= (x_new - x[j]) / (x[i] - x[j])
        y_new += y[i] * p
    return y_new

def _interpolate_series(t_orig, y_orig, t_new, degree=3):
    """
    Applies local Lagrange interpolation for a 1D series.
    Missing data (NaNs) in the local window will result in NaN.
    """
    num_points = degree + 1
    half_window = num_points // 2
    
    y_new = np.full(len(t_new), np.nan)
    n = len(t_orig)
    
    if n < num_points:
        return y_new
        
    idxs = np.searchsorted(t_orig, t_new)
    
    for i, t in enumerate(t_new):
        idx = idxs[i]
        
        start_idx = idx - half_window
        end_idx = start_idx + num_points
        
        if start_idx < 0:
            start_idx = 0
            end_idx = num_points
        elif end_idx >= n:
            end_idx = n
            start_idx = max(0, n - num_points)
            
        window_t = t_orig[start_idx:end_idx]
        window_y = y_orig[start_idx:end_idx]
        
        exact_match = np.where(np.isclose(window_t, t, rtol=0.0, atol=1e-3))[0]
        if len(exact_match) > 0:
            y_new[i] = window_y[exact_match[0]]
        else:
            if np.any(np.isnan(window_y)):
                y_new[i] = np.nan
            else:
                y_new[i] = _lagrange_interpolate(window_t, window_y, t)
                
    return y_new

def interpolation(data, interval, unit, method='polynomial'):
    """
    Interpolates parsed GNSS time-series data using Lagrange polynomials or linear interpolation.
    
    Parameters:
    - data: Pandas DataFrame or string path to an .h5 file containing the parsed GNSS data.
    - interval: float, the desired new time interval.
    - unit: str, the time unit. Accepted values: 'seconds', 'minutes', 'hours'.
    - method: str, the interpolation method. Currently 'polynomial' and 'linear' are supported.
    
    Returns:
    - Interpolated DataFrame, or a string path to the new .h5 file if input was .h5.
    """
    if method not in ['polynomial', 'linear']:
        raise ValueError("Currently only 'polynomial' and 'linear' interpolation are supported.")
        
    if not isinstance(interval, (int, float)) or interval <= 0:
        raise ValueError("Time interval must be a positive number.")
        
    valid_units = ['seconds', 'minutes', 'hours']
    if unit not in valid_units:
        raise ValueError(f"Time unit must be one of {valid_units}.")
        
    input_is_h5 = False
    output_path = None
    if isinstance(data, str) and data.endswith('.h5'):
        input_is_h5 = True
        output_path = data.replace('.h5', '_interpolated.h5')
        df = pd.read_hdf(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input data must be a Pandas DataFrame or a path to .h5 file.")
        
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
        
    time_col = 'Epoch'
    if time_col not in df.columns:
        raise ValueError(f"Required time column '{time_col}' not found.")
        
    # Standardize time
    df[time_col] = pd.to_datetime(df[time_col])
    
    unique_epochs = np.sort(df[time_col].dropna().unique())
    if len(unique_epochs) < 2:
        raise ValueError("Not enough time epochs to determine original interval or perform interpolation.")
        
    diffs_s = np.diff(unique_epochs).astype('timedelta64[s]').astype(float)
    orig_interval_s = float(np.median(diffs_s))
    
    req_interval_s = float(interval)
    if unit == 'minutes':
        req_interval_s *= 60
    elif unit == 'hours':
        req_interval_s *= 3600
        
    # Prevent downsampling or same-interval requests. Adding a small tolerance for "same interval"
    if req_interval_s >= orig_interval_s - 0.001:
        raise ValueError(f"Requested interval ({req_interval_s}s) must be smaller than original interval ({orig_interval_s}s). Downsampling or same-interval not allowed.")
        
    freq_str = f"{int(req_interval_s)}s"
    
    # Identify structure
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    group_cols = [c for c in df.columns if c != time_col and not pd.api.types.is_numeric_dtype(df[c])]
    
    # Generate the global target grid and convert to seconds since epoch
    t_new_global = pd.date_range(start=unique_epochs[0], end=unique_epochs[-1], freq=freq_str)
    t_new_s = t_new_global.to_numpy(dtype='datetime64[ns]').view('int64') / 1e9
    
    new_dfs = []
    
    grouped = df.groupby(group_cols) if group_cols else [(None, df)]
        
    for name, group in grouped:
        # Handle duplicates if they happen
        group = group.drop_duplicates(subset=[time_col]).sort_values(time_col)
        t_orig = group[time_col].values
        
        grp_min_s = t_orig.min().astype('datetime64[ns]').view('int64') / 1e9
        grp_max_s = t_orig.max().astype('datetime64[ns]').view('int64') / 1e9
        
        # Only interpolate points within the group's time bounds
        mask = (t_new_s >= grp_min_s) & (t_new_s <= grp_max_s)
        t_new_group = t_new_global[mask]
        t_new_group_s = t_new_s[mask]
        
        if len(t_new_group) == 0:
            continue
            
        t_orig_s = t_orig.astype('datetime64[ns]').view('int64') / 1e9
        
        res_dict = {time_col: t_new_group}
        
        if group_cols:
            if isinstance(name, tuple):
                for gc, val in zip(group_cols, name):
                    res_dict[gc] = val
            else:
                res_dict[group_cols[0]] = name
                
        for nc in numeric_cols:
            y_orig = group[nc].values
            if method == 'polynomial':
                y_new = _interpolate_series(t_orig_s, y_orig, t_new_group_s, degree=3)
            elif method == 'linear':
                y_new = _interpolate_linear(t_orig_s, y_orig, t_new_group_s)
            
            # If original data was integer type or contained NaNs, cast properly
            # Lagrange output is float. Let's keep it as float.
            res_dict[nc] = y_new
            
        new_dfs.append(pd.DataFrame(res_dict))
        
    if not new_dfs:
        res_df = pd.DataFrame(columns=df.columns)
    else:
        res_df = pd.concat(new_dfs, ignore_index=True)
        # Re-order
        res_df = res_df[df.columns]
        # Sort by grouped columns then Epoch for cleanliness
        sort_order = group_cols + [time_col]
        res_df.sort_values(sort_order, inplace=True, ignore_index=True)
        
    if input_is_h5:
        # Ensure we don't save with pandas fixed format, use table for flexibility
        res_df.to_hdf(output_path, key='data', mode='w')
        return output_path
        
    return res_df
