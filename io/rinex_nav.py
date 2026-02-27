import os
import re
import logging
import typing
import gzip
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from .hdf5_utils import handle_output

def _open_file(file_path: Path):
    if file_path.suffix.lower() == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
    return file_path.open('r', encoding='utf-8', errors='ignore')

# Helper methods for detection and parsing ported from user's original rinexnav.py
def _detect_rinex_version(file_path: Path) -> tuple[float, str]:
    try:
        with _open_file(file_path) as f:
            first_line = f.readline().strip()
            version_str = first_line[:9].strip()
            try:
                version = float(version_str)
            except ValueError:
                version_match = re.match(r'(\d+\.?\d*)', version_str)
                if version_match: version = float(version_match.group(1))
                else: raise ValueError(f"Cannot parse version from: {version_str}")
            constellation = 'MIXED'
            if version >= 3.0:
                f.seek(0)
                for line_num, line in enumerate(f):
                    if line_num > 20: break
                    if 'GNSS' in line or 'GPS' in line:
                        constellation = 'GPS'
                        break
                    elif 'GLONASS' in line:
                        constellation = 'GLO'
                        break
                    elif 'GALILEO' in line:
                        constellation = 'GAL'
                        break
                    elif 'BEIDOU' in line:
                        constellation = 'BDS'
                        break
                    elif 'QZSS' in line:
                        constellation = 'QZS'
                        break
            return version, constellation
    except Exception as e:
        raise ValueError(f"Error detecting RINEX version: {str(e)}")

def _parse_rinex_2x(file_path: Path) -> pd.DataFrame:
    try:
        nav_columns = [
            'PRN', 'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE',
            'Crs', 'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'TimeEph',
            'Cic', 'OMEGA', 'CIS', 'Io', 'Crc', 'omega', 'OMEGA_DOT', 'IDOT',
            'CodesL2', 'GPSWeek', 'L2Pflag', 'SVacc', 'SVhealth', 'TGD', 'IODC',
            'TransTime', 'FitIntvl'
        ]
        n_lines = 7
        sv_list, epoch_list, data_records = [], [], []
        with _open_file(file_path) as f:
            while True:
                line = f.readline()
                if not line: raise ValueError("End of file reached before 'END OF HEADER'")
                if 'END OF HEADER' in line: break
            while True:
                line = f.readline()
                if not line: break
                line = line.strip()
                if len(line) < 22: continue
                try:
                    prn_str = line[:2].strip()
                    if not prn_str: continue
                    prn = int(prn_str)
                    sv_list.append(prn)
                    parts = line.split()
                    if len(parts) < 7: continue
                    try:
                        year, month, day, hour, minute, second = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), float(parts[6])
                        if 80 <= year <= 99: year += 1900
                        elif year < 80: year += 2000
                        epoch = datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1e6))
                        epoch_list.append(epoch)
                        raw_data = " ".join(parts[7:])
                    except (ValueError, IndexError): continue
                    
                    for _ in range(n_lines):
                        next_line = f.readline()
                        if not next_line: break
                        raw_data += " " + next_line[4:].strip()
                    data_records.append(raw_data)
                except (ValueError, IndexError) as e: continue
        if not data_records: return pd.DataFrame()
        
        processed_data = []
        for i, raw in enumerate(data_records):
            try:
                raw = raw.replace('D', 'E')
                raw = re.sub(r'(\d)(-)', r'\1 \2', raw)
                raw = re.sub(r'\s+', ' ', raw.strip())
                values = [float(val) for val in raw.split() if val]
                while len(values) < 29: values.append(np.nan)
                row_data = [sv_list[i]] + values[:29]
                processed_data.append(row_data)
            except (ValueError, IndexError): continue
        if not processed_data: return pd.DataFrame()
        df = pd.DataFrame(processed_data, columns=nav_columns)
        df['Epoch'] = epoch_list[:len(processed_data)]
        cols = ['Epoch'] + [col for col in df.columns if col != 'Epoch']
        return df[cols]
    except Exception as e:
        raise

def _get_data_lines_count(sat_sys: str, version: float) -> int:
    if sat_sys in ['G', 'E', 'J', 'C', 'I']: return 7
    elif sat_sys in ['R', 'S']: return 3
    return 7

def _add_gps_like_params(record: dict, values: list) -> None:
    param_names = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE',
        'Crs', 'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'TimeEph',
        'Cic', 'OMEGA', 'CIS', 'Io', 'Crc', 'omega', 'OMEGA_DOT', 'IDOT',
        'CodesL2', 'GPSWeek', 'L2Pflag', 'SVacc', 'SVhealth', 'TGD', 'IODC',
        'TransTime', 'FitIntvl'
    ]
    for i, param_name in enumerate(param_names):
        record[param_name] = values[i] if i < len(values) else np.nan

def _add_glonass_params(record: dict, values: list) -> None:
    glonass_params = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Ax', 'Ay', 'Az', 'Health', 'FreqNum', 'AgeOfInfo']
    for i, param_name in enumerate(glonass_params):
        record[param_name] = values[i] if i < len(values) else np.nan

def _add_sbas_params(record: dict, values: list) -> None:
    sbas_params = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Ax', 'Ay', 'Az', 'Health', 'URA', 'IODN']
    for i, param_name in enumerate(sbas_params):
        record[param_name] = values[i] if i < len(values) else np.nan

def _process_rinex3_data(sat_id: str, epoch: datetime, raw_data: str, sat_sys: str) -> dict:
    try:
        raw_data = raw_data.replace('D', 'E')
        raw_data = re.sub(r'(\d)(-)', r'\1 \2', raw_data)
        raw_data = re.sub(r'\s+', ' ', raw_data.strip())
        values = []
        for val_str in raw_data.split():
            try: values.append(float(val_str))
            except ValueError: values.append(np.nan)
        record = {'Epoch': epoch, 'PRN': sat_id, 'SatSystem': sat_sys}
        if len(values) >= 3:
            record['SVclockBias'], record['SVclockDrift'], record['SVclockDriftRate'] = values[0], values[1], values[2]
        if sat_sys in ['G', 'E', 'J', 'C', 'I']: _add_gps_like_params(record, values)
        elif sat_sys == 'R': _add_glonass_params(record, values)
        elif sat_sys == 'S': _add_sbas_params(record, values)
        return record
    except Exception: return None

def _parse_rinex_3x_4x(file_path: Path, version: float, constellation: str) -> pd.DataFrame:
    try:
        records = []
        with _open_file(file_path) as f:
            while True:
                line = f.readline()
                if not line: raise ValueError("End of file reached before 'END OF HEADER'")
                if 'END OF HEADER' in line: break
            while True:
                line = f.readline()
                if not line: break
                line = line.strip()
                if len(line) < 23: continue
                try:
                    if line[0].isalpha():
                        sat_sys = line[0]
                        sat_id = f"{sat_sys}{int(line[1:3]):02d}"
                        time_start = 4
                    else:
                        sat_sys = 'G'
                        sat_id = f"G{int(line[:2]):02d}"
                        time_start = 3
                    year = int(line[time_start:time_start+4])
                    month = int(line[time_start+5:time_start+7])
                    day = int(line[time_start+8:time_start+10])
                    hour = int(line[time_start+11:time_start+13])
                    minute = int(line[time_start+14:time_start+16])
                    second = float(line[time_start+17:time_start+19])
                    epoch = datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1e6))
                    raw_data = line[time_start+20:] if len(line) > time_start+20 else ""
                    n_data_lines = _get_data_lines_count(sat_sys, version)
                    for _ in range(n_data_lines):
                        next_line = f.readline()
                        if not next_line: break
                        raw_data += " " + next_line[4:].strip()
                    record = _process_rinex3_data(sat_id, epoch, raw_data, sat_sys)
                    if record: records.append(record)
                except (ValueError, IndexError): continue
        if not records: return pd.DataFrame()
        return pd.DataFrame(records)
    except Exception: raise

def read_rinex_nav_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    try:
        file_path = Path(file_path).expanduser()
        if not file_path.exists(): raise FileNotFoundError(f"RINEX navigation file not found: {file_path}")
        version, constellation = _detect_rinex_version(file_path)
        if version == 2.11: return _parse_rinex_2x(file_path)
        elif version in [3.01, 3.02, 3.03, 3.04, 3.05, 4.00]: return _parse_rinex_3x_4x(file_path, version, constellation)
        else: raise ValueError(f"Unsupported RINEX version: {version}")
    except Exception as e:
        raise

def load_rinex_nav(
    file_path: typing.Union[str, bytes, os.PathLike],
    version: typing.Optional[str] = None,
    return_type: str = 'dataframe',
    output_path: typing.Optional[str] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    """
    Load a RINEX Navigation file.
    
    Args:
        file_path: Path to the RINEX file
        version: Expected RINEX version (auto-detected if None)
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        A pandas DataFrame, or a string pointing to the saved HDF5 file.
    """
    df = read_rinex_nav_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='rinex.nav')
