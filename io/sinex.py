"""
SINEX (Solution INdependent EXchange Format) Parser.
"""
import os
import typing
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from .hdf5_utils import handle_output

def _parse_sinex_date(date_str: str) -> typing.Optional[datetime]:
    """Parse SINEX dates (YY:DOY:SSSSS)."""
    try:
        parts = date_str.split(':')
        if len(parts) != 3:
            return None
        year = int(parts[0])
        doy = int(parts[1])
        sec = int(parts[2])
        if year < 50:
            year += 2000
        else:
            year += 1900
        return datetime(year, 1, 1) + timedelta(days=doy-1, seconds=sec)
    except:
        return None

def _parse_sinex(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse SINEX v2 file, focusing on SITE/ID and SOLUTION/ESTIMATE."""
    sites = []
    estimates = []
    
    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        current_block = None
        
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('*'):
                continue
                
            if line.startswith('+'):
                current_block = line[1:].strip()
                continue
            if line.startswith('-'):
                current_block = None
                continue
                
            if current_block == 'SITE/ID':
                # Site code is 1-4, Point code is 6-7, DOMES is 10-18, Obs structure is 20, Description is 22-
                if len(line) >= 22:
                    site_code = line[1:5].strip()
                    pt_code = line[6:8].strip()
                    domes = line[9:18].strip()
                    description = line[21:].strip()
                    sites.append({
                        'SiteCode': site_code,
                        'PointCode': pt_code,
                        'DOMES': domes,
                        'Description': description
                    })
                    
            elif current_block == 'SOLUTION/ESTIMATE':
                # Format: Index (1-5), Type (7-12), Site (14-17), Pt (19-20), SolId (22-25), Epoch (27-38), 
                # Unit (40-43), Constraint (45), Value (47-67), StdDev (69-79)
                if len(line) >= 67:
                    try:
                        idx = int(line[0:5].strip())
                        param_type = line[6:12].strip()
                        site_code = line[13:17].strip()
                        pt_code = line[18:20].strip()
                        sol_id = line[21:25].strip()
                        epoch_str = line[26:38].strip()
                        unit = line[39:43].strip()
                        value = float(line[46:67].strip())
                        std_dev = float(line[68:79].strip()) if len(line) >= 79 and line[68:79].strip() else None
                        
                        epoch = _parse_sinex_date(epoch_str)
                        
                        estimates.append({
                            'Index': idx,
                            'Parameter': param_type,
                            'SiteCode': site_code,
                            'PointCode': pt_code,
                            'SolID': sol_id,
                            'Epoch': epoch,
                            'Value': value,
                            'Unit': unit,
                            'StdDev': std_dev
                        })
                    except ValueError:
                        continue

    df_sites = pd.DataFrame(sites) if sites else pd.DataFrame()
    df_ests = pd.DataFrame(estimates) if estimates else pd.DataFrame()
    
    return df_sites, df_ests

def load_sinex(
    file_path: typing.Union[str, bytes, os.PathLike],
    return_type: str = 'dataframe',
    output_path: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None,
    **kwargs
) -> typing.Union[typing.Tuple[pd.DataFrame, pd.DataFrame], str]:
    """
    Load a SINEX file containing site coordinates and velocities.
    
    Returns standard parameters from +SITE/ID and +SOLUTION/ESTIMATE blocks.
    
    Args:
        file_path: Path to the SINEX file
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        If return_type='dataframe': A Tuple of (Sites DataFrame, Estimates DataFrame)
        If return_type='hdf5': String path to the HDF5 file containing 'sites' and 'estimates' groups.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    file_path = Path(file_path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"SINEX file not found: {file_path}")
        
    df_sites, df_ests = _parse_sinex(file_path)
    
    if return_type.lower() == 'dataframe':
        return df_sites, df_ests
    elif return_type.lower() == 'hdf5':
        if not output_path:
            raise ValueError("output_path must be provided when return_type is 'hdf5'")
        
        try:
            import tables
        except ImportError:
            raise ImportError("Saving to HDF5 requires 'tables'.")
            
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not df_sites.empty:
            df_sites.to_hdf(str(path), key='sites', mode='w', format='table', data_columns=True)
        if not df_ests.empty:
            mode = 'a' if not df_sites.empty else 'w'
            df_ests.to_hdf(str(path), key='estimates', mode=mode, format='table', data_columns=True)
        return str(path)
    else:
        raise ValueError(f"Unsupported return_type: {return_type}")
