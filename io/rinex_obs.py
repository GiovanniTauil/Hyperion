import os
import re
import logging
import typing
import gzip
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .hdf5_utils import handle_output

def _open_file(file_path: Path):
    name = file_path.name.lower()
    if name.endswith('.crx') or name.endswith('.crx.gz') or re.search(r'\.\d{2}d', name) or re.search(r'\.\d{2}d\.gz', name) or re.search(r'\.\d{2}d\.z', name):
        import hatanaka
        import io
        decompressed_bytes = hatanaka.decompress(file_path)
        return io.TextIOWrapper(io.BytesIO(decompressed_bytes), encoding='utf-8', errors='ignore')

    if file_path.suffix.lower() == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
    return file_path.open('r', encoding='utf-8', errors='ignore')

def _detect_obs_version_and_types(file_path: Path) -> tuple[float, dict, list]:
    try:
        with _open_file(file_path) as f:
            first_line = f.readline().strip()
            if first_line.startswith('>'): first_line = f.readline().strip()
            version_str = first_line[:9].strip()
            try: version = float(version_str)
            except ValueError:
                version_match = re.match(r'(\d+\.?\d*)', version_str)
                if version_match: version = float(version_match.group(1))
                else: raise ValueError(f"Cannot parse version from: {version_str}")
            
            f.seek(0)
            obs_types = {}
            sat_systems = []
            for line in f:
                line = line.rstrip('\n')
                if 'END OF HEADER' in line: break
                if version < 3.0:
                    if '# / TYPES OF OBSERV' in line:
                        if 'ALL' not in obs_types:
                            obs_types['ALL'] = []
                        for i in range(9):
                            start_col = 10 + i * 6
                            if start_col + 2 <= len(line):
                                obs_type = line[start_col:start_col+2].strip()
                                if obs_type: obs_types['ALL'].append(obs_type)
                else:
                    if 'SYS / # / OBS TYPES' in line:
                        sat_sys = line[0]
                        n_obs = int(line[3:6].strip())
                        types = []
                        for i in range(min(n_obs, 13)):
                            start_col = 7 + i * 4
                            if start_col < len(line):
                                obs_type = line[start_col:start_col+3].strip()
                                if obs_type: types.append(obs_type)
                        obs_types[sat_sys] = types
                        if sat_sys not in sat_systems: sat_systems.append(sat_sys)
                        if n_obs > 13:
                            remaining = n_obs - 13
                            next_line = f.readline().rstrip('\n')
                            for i in range(remaining):
                                start_col = 7 + i * 4
                                if start_col < len(next_line):
                                    obs_type = next_line[start_col:start_col+3].strip()
                                    if obs_type: types.append(obs_type)
                            obs_types[sat_sys] = types
            return version, obs_types, sat_systems
    except Exception as e:
        raise ValueError(f"Error detecting RINEX observation version and types: {str(e)}")

def _parse_rinex_obs_2x(file_path: Path, version: float, obs_types: dict) -> pd.DataFrame:
    try:
        records = []
        observation_types = obs_types.get('ALL', [])
        with _open_file(file_path) as f:
            while True:
                line = f.readline()
                if not line or 'END OF HEADER' in line: break
            while True:
                line = f.readline()
                if not line: break
                line = line.rstrip('\n')
                if len(line) < 26: continue
                try:
                    year, month, day, hour, minute, second = int(line[1:3]), int(line[4:6]), int(line[7:9]), int(line[10:12]), int(line[13:15]), float(line[16:26])
                    if 80 <= year <= 99: year += 1900
                    elif year < 80: year += 2000
                    epoch = datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1e6))
                    epoch_flag = int(line[28]) if len(line) > 28 else 0
                    num_sats = int(line[29:32]) if len(line) > 32 else 0
                    if epoch_flag != 0:
                        for _ in range((num_sats + 11) // 12): f.readline()
                        continue
                    satellites = []
                    sat_start, remaining_sats, current_pos = 32, num_sats, 32
                    while remaining_sats > 0 and current_pos < len(line):
                        sat_part = line[current_pos:current_pos+3]
                        if sat_part.strip():
                            sat_str = sat_part.strip()
                            if len(sat_str) >= 2 and sat_str[0].isalpha(): satellites.append(f"{sat_str[0]}{int(sat_str[1:]):02d}")
                            else: satellites.append(f"G{int(sat_str):02d}")
                            remaining_sats -= 1
                        current_pos += 3
                    while remaining_sats > 0:
                        cont_line = f.readline().rstrip('\n')
                        if not cont_line: break
                        for i in range(min(12, remaining_sats)):
                            start_col = 32 + i * 3
                            if start_col + 2 < len(cont_line):
                                sat_str = cont_line[start_col:start_col+3].strip()
                                if sat_str:
                                    if sat_str[0].isalpha(): satellites.append(f"{sat_str[0]}{int(sat_str[1:]):02d}")
                                    else: satellites.append(f"G{int(sat_str):02d}")
                                    remaining_sats -= 1
                    for sat in satellites:
                        if len(sat) >= 3 and sat[0].isalpha(): sat_system, prn = sat[0], int(sat[1:])
                        else: sat_system, prn = 'G', int(sat.lstrip('G'))
                        obs_data_lines = []
                        n_obs, obs_per_line = len(observation_types), 5
                        lines_needed = (n_obs + obs_per_line - 1) // obs_per_line
                        for _ in range(lines_needed): obs_data_lines.append(f.readline().rstrip('\n'))
                        obs_index = 0
                        for obs_line in obs_data_lines:
                            for i in range(obs_per_line):
                                if obs_index >= n_obs: break
                                start_col = i * 16
                                obs_str, lli_str, ssi_str = obs_line[start_col:start_col+14].strip(), obs_line[start_col+14:start_col+15].strip(), obs_line[start_col+15:start_col+16].strip()
                                obs_type = observation_types[obs_index]
                                try: obs_value = float(obs_str) if obs_str else np.nan
                                except: obs_value = np.nan
                                try: loss_of_lock = int(lli_str) if lli_str else 0
                                except: loss_of_lock = 0
                                try: signal_strength = int(ssi_str) if ssi_str else 0
                                except: signal_strength = 0
                                records.append({'Epoch': epoch, 'PRN': prn, 'SatSystem': sat_system, 'ObsType': obs_type, 'Value': obs_value, 'LossOfLock': loss_of_lock, 'SignalStrength': signal_strength})
                                obs_index += 1
                except Exception as e: continue
        if records: return pd.DataFrame(records)
        return pd.DataFrame()
    except Exception as e:
                        import traceback
                        raise

def _parse_rinex_obs_3x_4x(file_path: Path, version: float, obs_types: dict, sat_systems: list) -> pd.DataFrame:
    try:
        records = []
        with _open_file(file_path) as f:
            while True:
                line = f.readline()
                if not line or 'END OF HEADER' in line: break
            while True:
                line = f.readline()
                if not line: break
                line = line.rstrip('\n')
                if line.startswith('>'):
                    try:
                        year = int(line[2:6])
                        month, day, hour, minute, second = int(line[7:9]), int(line[10:12]), int(line[13:15]), int(line[16:18]), float(line[19:29])
                        epoch = datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1e6))
                        epoch_flag = int(line[31]) if len(line) > 31 else 0
                        
                        num_sats_part = line[32:35].strip()
                        if not num_sats_part: num_sats_part = line[29:32].strip()
                        try: num_sats = int(num_sats_part) if num_sats_part else 0
                        except:
                            parts = line[1:].strip().split()
                            try: num_sats = int(parts[7])
                            except: num_sats = 0

                        if epoch_flag != 0 and num_sats == 0: continue
                            
                        for _ in range(num_sats):
                            sat_line = f.readline()
                            if not sat_line: break
                            sat_line = sat_line.rstrip('\n')
                            if len(sat_line) < 3: continue
                            sat_sys_prn = sat_line[:3].replace(' ', '0')
                            sat_sys, prn = sat_sys_prn[0], int(sat_sys_prn[1:])
                            if sat_sys not in sat_systems: continue
                            
                            sys_obs_types = obs_types.get(sat_sys, [])
                            n_obs, obs_per_line = len(sys_obs_types), 5
                            lines_needed = (n_obs + obs_per_line - 1) // obs_per_line
                            obs_data_lines, is_inline = [], False
                            
                            if len(sat_line) > 3:
                                obs_data_lines.append(sat_line[3:])
                                is_inline = True
                                lines_needed -= 1
                                
                            for _ in range(max(0, lines_needed)): obs_data_lines.append(f.readline().rstrip('\n'))
                            obs_index = 0
                            for obs_line in obs_data_lines:
                                line_obs_count = obs_per_line if (not is_inline or obs_index > 0) else ((len(sat_line) - 3 + 15) // 16)
                                for i in range(line_obs_count):
                                    if obs_index >= n_obs: break
                                    start_col = i * 16
                                    obs_str, lli_str, ssi_str = obs_line[start_col:start_col+14].strip(), obs_line[start_col+14:start_col+15].strip(), obs_line[start_col+15:start_col+16].strip()
                                    obs_type = sys_obs_types[obs_index]
                                    try: obs_value = float(obs_str) if obs_str else np.nan
                                    except: obs_value = np.nan
                                    try: loss_of_lock = int(lli_str) if lli_str else 0
                                    except: loss_of_lock = 0
                                    try: signal_strength = int(ssi_str) if ssi_str else 0
                                    except: signal_strength = 0
                                    records.append({'Epoch': epoch, 'PRN': prn, 'SatSystem': sat_sys, 'ObsType': obs_type, 'Value': obs_value, 'LossOfLock': loss_of_lock, 'SignalStrength': signal_strength})
                                    obs_index += 1
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print("Error parsing epoch:", e)
                        continue
        if records: return pd.DataFrame(records)
        return pd.DataFrame()
    except Exception as e:
                        import traceback
                        raise

def read_rinex_obs_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    file_path = Path(file_path).expanduser()
    if not file_path.exists(): raise FileNotFoundError(f"RINEX obs file not found: {file_path}")
    version, obs_types, sat_systems = _detect_obs_version_and_types(file_path)
    if int(version) == 2: return _parse_rinex_obs_2x(file_path, version, obs_types)
    elif int(version) in [3, 4]: return _parse_rinex_obs_3x_4x(file_path, version, obs_types, sat_systems)
    else: raise ValueError(f"Unsupported RINEX obs version: {version}")

def load_rinex_obs(
    file_path: typing.Union[str, bytes, os.PathLike],
    version: typing.Optional[str] = None,
    return_type: str = 'dataframe',
    output_path: typing.Optional[str] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    """
    Load a RINEX Observation file.
    
    Args:
        file_path: Path to the RINEX file
        version: Expected RINEX version (auto-detected if None)
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        A pandas DataFrame, or a string pointing to the saved HDF5 file.
    """
    df = read_rinex_obs_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='rinex.obs')
