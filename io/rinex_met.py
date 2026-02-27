import re
import os
import gzip
import typing
from pathlib import Path
from datetime import datetime
import pandas as pd

from .hdf5_utils import handle_output

def _open_file(file_path: Path):
    if file_path.suffix.lower() == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
    return file_path.open('r', encoding='utf-8', errors='ignore')

def _parse_met(file_path: Path) -> pd.DataFrame:
    records = []
    obs_types = []
    
    with _open_file(file_path) as f:
        in_header = True
        
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            
            if in_header:
                if '# / TYPES OF OBSERV' in line:
                    parts = line[:60].split()
                    if int(parts[0]) > 0:
                        obs_types.extend(parts[1:])
                elif 'END OF HEADER' in line:
                    if not obs_types:
                        # Fallback common
                        obs_types = ['PR', 'TD', 'HR']
                    in_header = False
                continue
            
            # Data record format usually: YYYY MM DD HH MM SS  OBS1  OBS2  OBS3 ...
            # E.g.  2023 09 11 00 00 00   68.6 1005.8   19.8
            # In older versions: YY MM DD HH MM SS
            if len(line) < 18: continue
            
            try:
                # first elements are time
                time_str = line[:19].strip()
                tparts = time_str.split()
                if len(tparts) != 6: continue
                y, m, d = int(tparts[0]), int(tparts[1]), int(tparts[2])
                if y < 100:
                    y += 1900 if y >= 80 else 2000
                
                h, min_, s = int(tparts[3]), int(tparts[4]), int(float(tparts[5]))
                epoch = datetime(y, m, d, h, min_, s)
                
                obs_part = line[19:].strip()
                vals = []
                for val_str in obs_part.split():
                    try: vals.append(float(val_str))
                    except: vals.append(float('nan'))
                
                record = {'Epoch': epoch}
                for i, otype in enumerate(obs_types):
                    if i < len(vals):
                        record[otype] = vals[i]
                    else:
                        record[otype] = float('nan')
                records.append(record)
            except Exception as e:
                continue

    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    # Reorder columns slightly.
    return df

def read_rinex_met_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    file_path = Path(file_path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"RINEX MET file not found: {file_path}")
    return _parse_met(file_path)

def load_rinex_met(
    file_path: typing.Union[str, bytes, os.PathLike],
    return_type: str = 'dataframe',
    output_path: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    df = read_rinex_met_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='rinex_met')
