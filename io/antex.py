"""
ANTEX (Antenna Exchange Format) Parser.
"""
import os
import typing
import pandas as pd
from datetime import datetime
from pathlib import Path
from .hdf5_utils import handle_output
import gzip

def _open_file(file_path: Path):
    if file_path.suffix.lower() == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
    return file_path.open('r', encoding='utf-8', errors='ignore')

def _parse_antex(file_path: Path) -> pd.DataFrame:
    """Parse ANTEX v1.3 or v1.4 file into a DataFrame."""
    records = []
    
    with _open_file(file_path) as f:
        in_header = True
        in_antenna = False
        current_ant = {}
        
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
                
            # Content is in columns 0-59, Label is in 60-79
            content = line[:60].strip() if len(line) >= 60 else line.strip()
            label = line[60:].strip() if len(line) >= 60 else ""
            
            if in_header:
                if 'END OF HEADER' in label:
                    in_header = False
                continue
                
            if 'START OF ANTENNA' in label:
                in_antenna = True
                current_ant = {
                    'antenna_type': '',
                    'serial': '',
                    'valid_from': None,
                    'valid_until': None,
                    'frequencies': []
                }
                current_freq = None
                continue
                
            if not in_antenna:
                continue
                
            if 'TYPE / SERIAL NO' in label:
                current_ant['antenna_type'] = line[0:20].strip()
                current_ant['serial'] = line[20:40].strip()
                
            elif 'VALID FROM' in label:
                try:
                    parts = list(map(int, content.split()))
                    if len(parts) >= 6:
                        current_ant['valid_from'] = datetime(*parts[:6])
                except:
                    pass
                    
            elif 'VALID UNTIL' in label:
                try:
                    parts = list(map(int, content.split()))
                    if len(parts) >= 6:
                        current_ant['valid_until'] = datetime(*parts[:6])
                except:
                    pass

            elif 'START OF FREQUENCY' in label:
                sys_freq = content.split()
                if sys_freq:
                    current_freq = {
                        'system_freq': sys_freq[0],
                        'pco_n': None,
                        'pco_e': None,
                        'pco_u': None,
                        'pcv_noazi': [],
                        'pcv_grid': []
                    }
                    
            elif 'END OF FREQUENCY' in label:
                if current_freq:
                    current_ant['frequencies'].append(current_freq)
                    current_freq = None
                    
            elif 'NORTH / EAST / UP' in label:
                if current_freq:
                    parts = content.split()
                    if len(parts) >= 3:
                        try:
                            current_freq['pco_n'] = float(parts[0])
                            current_freq['pco_e'] = float(parts[1])
                            current_freq['pco_u'] = float(parts[2])
                        except ValueError:
                            pass
                            
            elif 'END OF ANTENNA' in label:
                # Flatten the data
                for freq in current_ant['frequencies']:
                    records.append({
                        'AntennaType': current_ant['antenna_type'],
                        'Serial': current_ant['serial'],
                        'ValidFrom': current_ant['valid_from'],
                        'ValidUntil': current_ant['valid_until'],
                        'Frequency': freq['system_freq'],
                        'PCO_N(mm)': freq['pco_n'],
                        'PCO_E(mm)': freq['pco_e'],
                        'PCO_U(mm)': freq['pco_u'],
                        'PCV_NOAZI(mm)': str(freq['pcv_noazi']) if freq['pcv_noazi'] else None,
                        'PCV_GRID(mm)': str(freq['pcv_grid']) if freq['pcv_grid'] else None
                    })
                in_antenna = False
                current_ant = {}
                
            else:
                # Inside frequency block: could be NOAZI or azimuth-dependent PCV
                if current_freq is not None:
                    if line.startswith('   NOAZI'):
                        try:
                            parts = line[8:60].split()
                            current_freq['pcv_noazi'] = [float(p) for p in parts]
                        except ValueError:
                            pass
                    elif len(line) >= 8 and line[:8].strip().replace('.', '').isdigit():
                        # Azimuth dependent row
                        try:
                            parts = line[8:60].split()
                            current_freq['pcv_grid'].append([float(p) for p in parts])
                        except ValueError:
                            pass

    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    return df

def read_antex_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    file_path = Path(file_path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"ANTEX file not found: {file_path}")
    return _parse_antex(file_path)

def load_antex(
    file_path: typing.Union[str, bytes, os.PathLike],
    version: typing.Optional[str] = None,
    return_type: str = 'dataframe',
    output_path: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    """
    Load an ANTEX file containing GNSS antenna phase center offsets and variations.
    
    Args:
        file_path: Path to the ANTEX file
        version: Expected ANTEX version (auto-detected if None)
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        A pandas DataFrame, or a string pointing to the saved HDF5 file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    df = read_antex_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='antex')
