import re
import gzip
import typing
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from .hdf5_utils import handle_output

def _open_file(file_path: Path):
    if file_path.suffix.lower() == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
    return file_path.open('r', encoding='utf-8', errors='ignore')

def _parse_doris(file_path: Path) -> pd.DataFrame:
    records = []
    
    with _open_file(file_path) as f:
        in_header = True
        obs_types = []
        is_v3 = False
        
        for line in f:
            line = line.rstrip('\n')
            
            if in_header:
                if 'RINEX VERSION' in line:
                    try:
                        v = float(line[:9])
                        if v >= 3.0: is_v3 = True
                    except: pass
                elif 'SYS / # / OBS TYPES' in line:
                    if line[0] == 'D':
                        parts = line[6:60].split()
                        obs_types.extend(parts)
                elif 'END OF HEADER' in line:
                    in_header = False
                continue
            
            # Start of epoch block
            if line.startswith('>'):
                parts = line[1:].strip().split()
                if len(parts) >= 6:
                    try:
                        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                        h, min_, s = int(parts[3]), int(parts[4]), float(parts[5])
                        sec = int(s)
                        ms = int((s - sec) * 1e6)
                        current_epoch = datetime(y, m, d, h, min_, sec, ms)
                    except:
                        current_epoch = None
            elif 'current_epoch' in locals() and current_epoch is not None:
                if len(line) < 3: continue
                # Station observation line
                station_id = line[:3].strip()
                
                obs_vals = []
                # First line of observation values
                for i in range(5):
                    idx = 4 + i * 16
                    if idx + 14 <= len(line):
                        val_str = line[idx:idx+14].strip()
                        obs_vals.append(float(val_str) if val_str else float('nan'))
                    else:
                        obs_vals.append(float('nan'))
                        
                # Additional lines if observation types > 5
                remaining = len(obs_types) - 5
                if remaining > 0:
                    next_line = f.readline().rstrip('\n')
                    for i in range(remaining):
                        idx = 4 + i * 16
                        if idx + 14 <= len(next_line):
                            val_str = next_line[idx:idx+14].strip()
                            obs_vals.append(float(val_str) if val_str else float('nan'))
                        else:
                            obs_vals.append(float('nan'))
                
                rec = {'Epoch': current_epoch, 'Station': station_id}
                for i, otype in enumerate(obs_types):
                    rec[otype] = obs_vals[i] if i < len(obs_vals) else float('nan')
                records.append(rec)

    if not records: return pd.DataFrame()
    return pd.DataFrame(records)

def read_rinex_doris_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    file_path = Path(file_path).expanduser()
    if not file_path.exists(): raise FileNotFoundError(f"RINEX DORIS file not found: {file_path}")
    return _parse_doris(file_path)

def load_rinex_doris(file_path, return_type='dataframe', output_path=None, **kwargs):
    df = read_rinex_doris_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='rinex_doris')
