"""
ERP (Earth Rotation Parameters) Parser.
"""
import os
import typing
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from .hdf5_utils import handle_output

def _mjd_to_datetime(mjd: float) -> datetime:
    """Convert Modified Julian Date to datetime."""
    # MJD 0 is Nov 17, 1858
    mjd_origin = datetime(1858, 11, 17)
    days = int(mjd)
    fractional_day = mjd - days
    return mjd_origin + timedelta(days=days, seconds=int(fractional_day * 86400))

def _parse_erp(file_path: Path) -> pd.DataFrame:
    """Parse IGS ERP version 2 file into a DataFrame."""
    records = []
    
    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        version_line = f.readline().strip()
        if 'VERSION 2' not in version_line.upper():
            pass # Try to parse anyway, usually format is mostly the same
            
        for line in f:
            line = line.strip()
            if not line or line.startswith('MJD') or line.startswith('---'):
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                try:
                    mjd = float(parts[0])
                    xpole = float(parts[1]) * 1e-6 # scale 10^-6 arcsec
                    ypole = float(parts[2]) * 1e-6 # scale 10^-6 arcsec
                    ut1_utc = float(parts[3]) * 1e-7 # scale 10^-7 s
                    lod = float(parts[4]) * 1e-7 # scale 10^-7 s/d
                    
                    xpole_sig = float(parts[5]) * 1e-6 if len(parts) > 5 else None
                    ypole_sig = float(parts[6]) * 1e-6 if len(parts) > 6 else None
                    ut1_utc_sig = float(parts[7]) * 1e-7 if len(parts) > 7 else None
                    lod_sig = float(parts[8]) * 1e-7 if len(parts) > 8 else None
                    
                    records.append({
                        'MJD': mjd,
                        'Date': _mjd_to_datetime(mjd),
                        'Xpole(arcsec)': xpole,
                        'Ypole(arcsec)': ypole,
                        'UT1-UTC(s)': ut1_utc,
                        'LOD(s/d)': lod,
                        'Xpole_sigma(arcsec)': xpole_sig,
                        'Ypole_sigma(arcsec)': ypole_sig,
                        'UT1-UTC_sigma(s)': ut1_utc_sig,
                        'LOD_sigma(s/d)': lod_sig
                    })
                except ValueError:
                    continue

    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    # Reorder columns to have Date first
    cols = ['Date', 'MJD'] + [c for c in df.columns if c not in ['Date', 'MJD']]
    return df[cols]

def read_erp_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    file_path = Path(file_path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"ERP file not found: {file_path}")
    return _parse_erp(file_path)

def load_erp(
    file_path: typing.Union[str, bytes, os.PathLike],
    return_type: str = 'dataframe',
    output_path: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    """
    Load an Earth Rotation Parameters (ERP) file.
    
    Args:
        file_path: Path to the ERP file
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        A pandas DataFrame, or a string pointing to the saved HDF5 file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    df = read_erp_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='erp')
