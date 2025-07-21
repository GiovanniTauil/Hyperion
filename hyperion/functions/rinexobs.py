"""
RINEX Observation File Reader

This module provides functionality to read RINEX observation files 
and convert them to pandas DataFrames. Supports multiple RINEX versions:
- 2.10, 2.11 (GPS and other GNSS observations)
- 3.01, 3.04, 3.05, 4.00 (Multi-constellation observations)

Author: Based on attempted solution and extended for multiple versions
"""

import os
import re
import logging
import typing
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def read_rinex_obs_to_dataframe(
    file_path: typing.Union[str, bytes, os.PathLike]
) -> pd.DataFrame:
    """
    Read RINEX observation file and convert to pandas DataFrame.
    
    Supports RINEX observation versions 2.10, 2.11, 3.01, 3.04, 3.05, and 4.00.
    
    Args:
        file_path: Path to the RINEX observation file
        
    Returns:
        pandas.DataFrame with observation data including:
        - Epoch: Time of observation
        - PRN: Satellite PRN identifier (for v2.x) or SatelliteID (for v3+)
        - SatSystem: Satellite system (G, R, E, C, J, I, S for v3+)
        - ObsType: Observation type (C1, P1, L1, S1, etc.)
        - Value: Observed value
        - LossOfLock: Loss of lock indicator (0-7)
        - SignalStrength: Signal strength indicator (1-9)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    try:
        # Validate file exists
        file_path = Path(file_path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"RINEX observation file not found: {file_path}")
        
        # Detect RINEX version and observation types
        version, obs_types, sat_systems = _detect_obs_version_and_types(file_path)
        
        # Parse based on version
        if version in [2.10, 2.11]:
            return _parse_rinex_obs_2x(file_path, version, obs_types)
        elif version in [3.01, 3.02, 3.03, 3.04, 3.05, 4.00]:
            return _parse_rinex_obs_3x_4x(file_path, version, obs_types, sat_systems)
        else:
            raise ValueError(f"Unsupported RINEX observation version: {version}")
            
    except Exception as e:
        logging.error(f"Error reading RINEX observation file {file_path}: {str(e)}")
        raise


def _detect_obs_version_and_types(file_path: Path) -> tuple[float, dict, list]:
    """
    Detect RINEX version, observation types, and satellite systems from file header.
    
    Args:
        file_path: Path to RINEX observation file
        
    Returns:
        Tuple of (version, obs_types_dict, satellite_systems_list)
    """
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            # Read first line to get version
            first_line = f.readline().strip()
            
            # Handle RINEX 3+ format that might start with '>'
            if first_line.startswith('>'):
                # This is likely a RINEX 3+ file, read the actual header line
                first_line = f.readline().strip()
            
            # Extract version from first line (columns 0-9)
            version_str = first_line[:9].strip()
            
            # Extract file type (column 20) - should be 'O' for observation
            if len(first_line) >= 21:
                file_type = first_line[20]
                if file_type not in ['O', 'V', ' ']:  # Be more lenient
                    logging.warning(f"File type is '{file_type}', expected 'O' for observation file")
            
            # Parse version number
            try:
                version = float(version_str)
            except ValueError:
                # Fallback: try to extract just the first part
                version_match = re.match(r'(\d+\.?\d*)', version_str)
                if version_match:
                    version = float(version_match.group(1))
                else:
                    raise ValueError(f"Cannot parse version from: {version_str}")
            
            # Read header to extract observation types
            f.seek(0)
            obs_types = {}
            sat_systems = []
            
            for line in f:
                line = line.rstrip('\n')
                
                if 'END OF HEADER' in line:
                    break
                
                # Parse observation types based on version
                if version < 3.0:
                    # RINEX 2.x: "# / TYPES OF OBSERV"
                    if '# / TYPES OF OBSERV' in line:
                        # First field is number of observation types
                        n_obs = int(line[:6].strip())
                        # Observation types start at column 10, 6 characters each
                        types = []
                        for i in range(n_obs):
                            start_col = 10 + i * 6
                            if start_col < len(line):
                                obs_type = line[start_col:start_col+2].strip()
                                if obs_type:
                                    types.append(obs_type)
                        obs_types['ALL'] = types  # In v2.x, all systems use same types
                        
                else:
                    # RINEX 3.x+: "SYS / # / OBS TYPES"
                    if 'SYS / # / OBS TYPES' in line:
                        sat_sys = line[0]  # First character is satellite system
                        n_obs = int(line[3:6].strip())
                        
                        # Observation types start at column 7, 4 characters each
                        types = []
                        for i in range(min(n_obs, 13)):  # Max 13 types per line
                            start_col = 7 + i * 4
                            if start_col < len(line):
                                obs_type = line[start_col:start_col+3].strip()
                                if obs_type:
                                    types.append(obs_type)
                        
                        obs_types[sat_sys] = types
                        if sat_sys not in sat_systems:
                            sat_systems.append(sat_sys)
                        
                        # Handle continuation lines if more than 13 observation types
                        if n_obs > 13:
                            remaining = n_obs - 13
                            next_line = f.readline().rstrip('\n')
                            for i in range(remaining):
                                start_col = 7 + i * 4
                                if start_col < len(next_line):
                                    obs_type = next_line[start_col:start_col+3].strip()
                                    if obs_type:
                                        types.append(obs_type)
                            obs_types[sat_sys] = types
            
            return version, obs_types, sat_systems
            
    except Exception as e:
        raise ValueError(f"Error detecting RINEX observation version and types: {str(e)}")


def _parse_rinex_obs_2x(file_path: Path, version: float, obs_types: dict) -> pd.DataFrame:
    """
    Parse RINEX 2.x observation files.
    
    RINEX 2.x format:
    - All satellites use same observation types
    - Epoch header followed by satellite data
    - Each observation has value, loss-of-lock indicator, signal strength
    """
    try:
        records = []
        observation_types = obs_types.get('ALL', [])
        
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            # Skip header until 'END OF HEADER'
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("End of file reached before 'END OF HEADER'")
                if 'END OF HEADER' in line:
                    break
            
            # Read observation data
            while True:
                line = f.readline()
                if not line:
                    break
                
                line = line.rstrip('\n')
                if len(line) < 26:
                    continue
                
                try:
                    # Parse epoch header
                    # Format: YY MM DD HH MM SS.SSSSSSS  EPOCH FLAG  NUM_SAT
                    year = int(line[1:3])
                    month = int(line[4:6])
                    day = int(line[7:9])
                    hour = int(line[10:12])
                    minute = int(line[13:15])
                    second = float(line[16:26])
                    
                    # Handle Y2K problem for 2-digit years
                    if 80 <= year <= 99:
                        year += 1900
                    elif year < 80:
                        year += 2000
                    
                    epoch = datetime(year, month, day, hour, minute, int(second),
                                   int((second - int(second)) * 1e6))
                    
                    # Epoch flag (column 28)
                    epoch_flag = int(line[28]) if len(line) > 28 else 0
                    
                    # Number of satellites (columns 29-32)
                    num_sats = int(line[29:32]) if len(line) > 32 else 0
                    
                    # Skip if special epoch (flag != 0)
                    if epoch_flag != 0:
                        # Skip the required number of lines
                        for _ in range((num_sats + 11) // 12):  # Satellite list lines
                            f.readline()
                        continue
                    
                    # Read satellite PRNs from the same line first (starting at column 32)
                    satellites = []
                    
                    # Parse satellites from current line (max 12 satellites per line)
                    sat_start = 32
                    remaining_sats = num_sats
                    current_pos = sat_start
                    
                    # Parse satellites from the epoch line
                    while remaining_sats > 0 and current_pos < len(line):
                        sat_part = line[current_pos:current_pos+3]
                        if sat_part.strip():
                            sat_str = sat_part.strip()
                            # Handle both formats: "G 1" or " 1" or "01"
                            if len(sat_str) >= 2 and sat_str[0].isalpha():
                                sat_system = sat_str[0]
                                prn = int(sat_str[1:])
                                satellites.append(f"{sat_system}{prn:02d}")
                            else:
                                prn = int(sat_str) 
                                satellites.append(f"G{prn:02d}")  # Default to GPS
                            remaining_sats -= 1
                        current_pos += 3
                    
                    # If more satellites than can fit on one line, read continuation lines
                    while remaining_sats > 0:
                        cont_line = f.readline().rstrip('\n')
                        if not cont_line:
                            break
                        for i in range(min(12, remaining_sats)):
                            start_col = 32 + i * 3
                            if start_col + 2 < len(cont_line):
                                sat_str = cont_line[start_col:start_col+3].strip()
                                if sat_str:
                                    if sat_str[0].isalpha():
                                        sat_system = sat_str[0]
                                        prn = int(sat_str[1:])
                                        satellites.append(f"{sat_system}{prn:02d}")
                                    else:
                                        prn = int(sat_str)
                                        satellites.append(f"G{prn:02d}")  # Default to GPS
                                    remaining_sats -= 1
                    
                    # Read observation data for each satellite
                    for sat in satellites:
                        # Determine satellite system and PRN
                        if len(sat) >= 3 and sat[0].isalpha():
                            sat_system = sat[0]
                            prn = int(sat[1:])
                        else:
                            sat_system = 'G'  # Default to GPS for v2.x
                            prn = int(sat.lstrip('G'))
                        
                        # Read observation line(s) for this satellite
                        obs_data_lines = []
                        n_obs = len(observation_types)
                        obs_per_line = 5  # Maximum 5 observations per line in RINEX 2.x
                        
                        lines_needed = (n_obs + obs_per_line - 1) // obs_per_line
                        for _ in range(lines_needed):
                            obs_line = f.readline().rstrip('\n')
                            obs_data_lines.append(obs_line)
                        
                        # Parse observations
                        obs_index = 0
                        for obs_line in obs_data_lines:
                            for i in range(obs_per_line):
                                if obs_index >= n_obs:
                                    break
                                
                                start_col = i * 16
                                obs_str = obs_line[start_col:start_col+14].strip()
                                lli_str = obs_line[start_col+14:start_col+15].strip()
                                ssi_str = obs_line[start_col+15:start_col+16].strip()
                                
                                obs_type = observation_types[obs_index]
                                
                                # Parse observation value
                                try:
                                    obs_value = float(obs_str) if obs_str else np.nan
                                except ValueError:
                                    obs_value = np.nan
                                
                                # Parse loss of lock indicator
                                try:
                                    loss_of_lock = int(lli_str) if lli_str else 0
                                except ValueError:
                                    loss_of_lock = 0
                                
                                # Parse signal strength indicator
                                try:
                                    signal_strength = int(ssi_str) if ssi_str else 0
                                except ValueError:
                                    signal_strength = 0
                                
                                # Add record
                                records.append({
                                    'Epoch': epoch,
                                    'PRN': sat,
                                    'SatSystem': sat_system,
                                    'ObsType': obs_type,
                                    'Value': obs_value,
                                    'LossOfLock': loss_of_lock,
                                    'SignalStrength': signal_strength
                                })
                                
                                obs_index += 1
                
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error parsing observation epoch: {e}")
                    continue
        
        # Convert to DataFrame
        if records:
            df = pd.DataFrame(records)
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data
            
    except Exception as e:
        logging.error(f"Error parsing RINEX 2.x observation file: {str(e)}")
        raise


def _parse_rinex_obs_3x_4x(file_path: Path, version: float, obs_types: dict, sat_systems: list) -> pd.DataFrame:
    """
    Parse RINEX 3.x and 4.x observation files.
    
    RINEX 3.x+ format:
    - Different observation types per satellite system
    - Epoch header followed by satellite observations
    - Each satellite line contains all observations for that satellite
    """
    try:
        records = []
        
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            # Skip header until 'END OF HEADER'
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("End of file reached before 'END OF HEADER'")
                if 'END OF HEADER' in line:
                    break
            
            # Read observation data
            while True:
                line = f.readline()
                if not line:
                    break
                
                line = line.rstrip('\n')
                if len(line) < 29:
                    continue
                
                # Check if this is an epoch header line (starts with '>')
                if line.startswith('>'):
                    try:
                        # Parse epoch header
                        # Format: > YYYY MM DD HH MM SS.SSSSSSS  EPOCH FLAG  NUM_SAT
                        year = int(line[2:6])
                        month = int(line[7:9])
                        day = int(line[10:12])
                        hour = int(line[13:15])
                        minute = int(line[16:18])
                        second = float(line[19:29])
                        
                        epoch = datetime(year, month, day, hour, minute, int(second),
                                       int((second - int(second)) * 1e6))
                        
                        # Epoch flag (column 31)
                        epoch_flag = int(line[31]) if len(line) > 31 else 0
                        
                        # Number of satellites (columns 32-35)
                        num_sats = int(line[32:35]) if len(line) > 35 else 0
                        
                        # Skip if special epoch (flag != 0)
                        if epoch_flag != 0:
                            # Skip the required number of lines
                            for _ in range(num_sats):
                                f.readline()
                            continue
                        
                        # Read satellite observation data
                        for _ in range(num_sats):
                            sat_line = f.readline().rstrip('\n')
                            if len(sat_line) < 3:
                                continue
                            
                            # Parse satellite identifier
                            sat_system = sat_line[0]
                            try:
                                prn = int(sat_line[1:3])
                                sat_id = f"{sat_system}{prn:02d}"
                            except ValueError:
                                continue
                            
                            # Get observation types for this satellite system
                            system_obs_types = obs_types.get(sat_system, [])
                            
                            # Parse observations for this satellite
                            obs_data = sat_line[3:]  # Observation data starts at column 4
                            
                            for i, obs_type in enumerate(system_obs_types):
                                start_col = i * 16
                                if start_col >= len(obs_data):
                                    break
                                
                                obs_str = obs_data[start_col:start_col+14].strip()
                                lli_str = obs_data[start_col+14:start_col+15].strip()
                                ssi_str = obs_data[start_col+15:start_col+16].strip()
                                
                                # Parse observation value
                                try:
                                    obs_value = float(obs_str) if obs_str else np.nan
                                except ValueError:
                                    obs_value = np.nan
                                
                                # Parse loss of lock indicator
                                try:
                                    loss_of_lock = int(lli_str) if lli_str else 0
                                except ValueError:
                                    loss_of_lock = 0
                                
                                # Parse signal strength indicator
                                try:
                                    signal_strength = int(ssi_str) if ssi_str else 0
                                except ValueError:
                                    signal_strength = 0
                                
                                # Add record
                                records.append({
                                    'Epoch': epoch,
                                    'PRN': sat_id,
                                    'SatSystem': sat_system,
                                    'ObsType': obs_type,
                                    'Value': obs_value,
                                    'LossOfLock': loss_of_lock,
                                    'SignalStrength': signal_strength
                                })
                        
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Error parsing observation epoch: {e}")
                        continue
        
        # Convert to DataFrame
        if records:
            df = pd.DataFrame(records)
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data
            
    except Exception as e:
        logging.error(f"Error parsing RINEX 3.x/4.x observation file: {str(e)}")
        raise