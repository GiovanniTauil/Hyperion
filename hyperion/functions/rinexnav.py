"""
RINEX Navigation File Reader

This module provides functionality to read RINEX navigation files 
and convert them to pandas DataFrames. Supports multiple RINEX versions:
- 2.11 (GPS Navigation)
- 3.01, 3.04, 3.05, 4.00 (Multi-constellation)

Author: Based on attempted solution and extended for multiple versions
"""

import os
import re
import logging
import typing
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def read_rinex_nav_to_dataframe(
    file_path: typing.Union[str, bytes, os.PathLike]
) -> pd.DataFrame:
    """
    Read RINEX navigation file and convert to pandas DataFrame.
    
    Supports RINEX navigation versions 2.11, 3.01, 3.04, 3.05, and 4.00.
    
    Args:
        file_path: Path to the RINEX navigation file
        
    Returns:
        pandas.DataFrame with navigation data including:
        - Epoch: Time of clock correction
        - PRN: Satellite PRN identifier
        - SVclockBias: SV clock bias (seconds)
        - SVclockDrift: SV clock drift (sec/sec)
        - SVclockDriftRate: SV clock drift rate (sec/sec^2)
        - And additional orbital parameters depending on RINEX version
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    try:
        # Validate file exists
        file_path = Path(file_path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"RINEX navigation file not found: {file_path}")
        
        # Detect RINEX version from header
        version, constellation = _detect_rinex_version(file_path)
        
        # Parse based on version
        if version == 2.11:
            return _parse_rinex_2x(file_path)
        elif version in [3.01, 3.02, 3.03, 3.04, 3.05, 4.00]:
            return _parse_rinex_3x_4x(file_path, version, constellation)
        else:
            raise ValueError(f"Unsupported RINEX version: {version}")
            
    except Exception as e:
        logging.error(f"Error reading RINEX navigation file {file_path}: {str(e)}")
        raise


def _detect_rinex_version(file_path: Path) -> tuple[float, str]:
    """
    Detect RINEX version and constellation type from file header.
    
    Args:
        file_path: Path to RINEX file
        
    Returns:
        Tuple of (version, constellation_type)
    """
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            
            # Extract version from first line (columns 0-9)
            version_str = first_line[:9].strip()
            
            # Extract file type (column 20)
            if len(first_line) >= 21:
                file_type = first_line[20]
            else:
                file_type = 'N'  # Default to navigation
            
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
            
            # Determine constellation type for RINEX 3+
            constellation = 'MIXED'
            if version >= 3.0:
                # Check for constellation identifier in header
                f.seek(0)
                for line_num, line in enumerate(f):
                    if line_num > 20:  # Don't search too far
                        break
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
    """
    Parse RINEX 2.x navigation files (primarily GPS).
    
    Based on the attempted solution but converted to return pandas DataFrame.
    """
    try:
        # Navigation parameters for RINEX 2.x (29 parameters total)
        nav_columns = [
            'PRN', 'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE',
            'Crs', 'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'TimeEph',
            'Cic', 'OMEGA', 'CIS', 'Io', 'Crc', 'omega', 'OMEGA_DOT', 'IDOT',
            'CodesL2', 'GPSWeek', 'L2Pflag', 'SVacc', 'SVhealth', 'TGD', 'IODC',
            'TransTime', 'FitIntvl'
        ]
        
        startcol = 3  # Column where numerical data starts
        n_lines = 7   # Number of lines per record
        
        sv_list = []
        epoch_list = []
        data_records = []
        
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            # Skip header until 'END OF HEADER'
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("End of file reached before 'END OF HEADER'")
                if 'END OF HEADER' in line:
                    break
            
            # Read navigation data records
            while True:
                line = f.readline()
                if not line:
                    break
                
                line = line.strip()
                if len(line) < 22:
                    continue
                
                try:
                    # Parse satellite PRN (first 2 characters, handle both formats)
                    prn_str = line[:2].strip()
                    if not prn_str:
                        continue  # Skip empty lines
                    prn = int(prn_str)
                    sv_list.append(prn)
                    
                    # Parse epoch - need to handle flexible spacing
                    # Split the line and parse the date/time fields
                    parts = line.split()
                    if len(parts) < 7:
                        continue  # Not enough parts for epoch
                    
                    # Parts should be: PRN, YY, MM, DD, HH, MM, SS, then data...
                    try:
                        year = int(parts[1])
                        month = int(parts[2])
                        day = int(parts[3])
                        hour = int(parts[4])
                        minute = int(parts[5])
                        second = float(parts[6])
                        
                        # Handle Y2K problem for 2-digit years
                        if 80 <= year <= 99:
                            year += 1900
                        elif year < 80:
                            year += 2000
                        
                        epoch = datetime(year, month, day, hour, minute, int(second),
                                       int((second - int(second)) * 1e6))
                        epoch_list.append(epoch)
                        
                        # Collect raw numerical data (starting from part 7)
                        raw_data = " ".join(parts[7:])
                        
                    except (ValueError, IndexError):
                        logging.warning(f"Error parsing epoch from line: {line[:50]}...")
                        continue
                    
                    # Read additional lines for this satellite
                    for _ in range(n_lines):
                        next_line = f.readline()
                        if not next_line:
                            break
                        # Extract data from columns 4-80 (skip first 4 characters)
                        raw_data += " " + next_line[4:].strip()
                    
                    data_records.append(raw_data)
                    
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping malformed line: {line[:50]}... Error: {e}")
                    continue
        
        if not data_records:
            return pd.DataFrame()  # Return empty DataFrame if no data found
        
        # Process raw data strings
        processed_data = []
        for i, raw in enumerate(data_records):
            try:
                # Replace FORTRAN 'D' exponential notation with 'E'
                raw = raw.replace('D', 'E')
                # Fix negative signs that might be attached to numbers
                raw = re.sub(r'(\d)(-)', r'\1 \2', raw)
                # Clean up whitespace
                raw = re.sub(r'\s+', ' ', raw.strip())
                
                # Split into individual values
                values = [float(val) for val in raw.split() if val]
                
                # Ensure we have exactly 29 parameters (pad with NaN if needed)
                while len(values) < 29:
                    values.append(np.nan)
                
                # Add PRN as first parameter
                row_data = [sv_list[i]] + values[:29]
                processed_data.append(row_data)
                
            except (ValueError, IndexError) as e:
                logging.warning(f"Error processing data for satellite {sv_list[i] if i < len(sv_list) else 'unknown'}: {e}")
                continue
        
        if not processed_data:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(processed_data, columns=nav_columns)
        df['Epoch'] = epoch_list[:len(processed_data)]
        
        # Reorder columns to put Epoch first
        cols = ['Epoch'] + [col for col in df.columns if col != 'Epoch']
        df = df[cols]
        
        return df
        
    except Exception as e:
        logging.error(f"Error parsing RINEX 2.x file: {str(e)}")
        raise


def _parse_rinex_3x_4x(file_path: Path, version: float, constellation: str) -> pd.DataFrame:
    """
    Parse RINEX 3.x and 4.x navigation files (multi-constellation).
    
    RINEX 3+ supports multiple satellite systems with different parameters.
    """
    try:
        records = []
        
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            # Skip header
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("End of file reached before 'END OF HEADER'")
                if 'END OF HEADER' in line:
                    break
            
            # Read navigation records
            while True:
                line = f.readline()
                if not line:
                    break
                
                line = line.strip()
                if len(line) < 23:
                    continue
                
                try:
                    # Parse satellite system and PRN
                    if line[0].isalpha():
                        sat_sys = line[0]  # G, R, E, C, J, I, S
                        prn = int(line[1:3])
                        sat_id = f"{sat_sys}{prn:02d}"
                        time_start = 4
                    else:
                        # Legacy format or space-padded
                        sat_sys = 'G'  # Default to GPS
                        prn = int(line[:2])
                        sat_id = f"G{prn:02d}"
                        time_start = 3
                    
                    # Parse epoch
                    year = int(line[time_start:time_start+4])
                    month = int(line[time_start+5:time_start+7])
                    day = int(line[time_start+8:time_start+10])
                    hour = int(line[time_start+11:time_start+13])
                    minute = int(line[time_start+14:time_start+16])
                    second = float(line[time_start+17:time_start+19])
                    
                    epoch = datetime(year, month, day, hour, minute, int(second),
                                   int((second - int(second)) * 1e6))
                    
                    # Collect data lines
                    raw_data = line[time_start+20:] if len(line) > time_start+20 else ""
                    
                    # Determine number of data lines based on satellite system
                    n_data_lines = _get_data_lines_count(sat_sys, version)
                    
                    for _ in range(n_data_lines):
                        next_line = f.readline()
                        if not next_line:
                            break
                        raw_data += " " + next_line[4:].strip()  # Skip first 4 characters
                    
                    # Process the raw data
                    record = _process_rinex3_data(sat_id, epoch, raw_data, sat_sys)
                    if record:
                        records.append(record)
                        
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping malformed navigation record: {e}")
                    continue
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        logging.error(f"Error parsing RINEX 3.x/4.x file: {str(e)}")
        raise


def _get_data_lines_count(sat_sys: str, version: float) -> int:
    """
    Get the number of data lines for different satellite systems.
    
    Different satellite systems have different numbers of parameters.
    """
    if sat_sys in ['G', 'E', 'J']:  # GPS, Galileo, QZSS
        return 7
    elif sat_sys == 'R':  # GLONASS
        return 3
    elif sat_sys == 'C':  # BeiDou
        return 7
    elif sat_sys == 'I':  # IRNSS
        return 7
    elif sat_sys == 'S':  # SBAS
        return 3
    else:
        return 7  # Default


def _process_rinex3_data(sat_id: str, epoch: datetime, raw_data: str, sat_sys: str) -> dict:
    """
    Process raw navigation data for RINEX 3+ format.
    
    Returns a dictionary with processed navigation parameters.
    """
    try:
        # Clean and parse the raw data
        raw_data = raw_data.replace('D', 'E')  # FORTRAN to Python exponential
        raw_data = re.sub(r'(\d)(-)', r'\1 \2', raw_data)  # Fix negative signs
        raw_data = re.sub(r'\s+', ' ', raw_data.strip())  # Normalize whitespace
        
        values = []
        for val_str in raw_data.split():
            try:
                values.append(float(val_str))
            except ValueError:
                values.append(np.nan)
        
        # Create base record
        record = {
            'Epoch': epoch,
            'PRN': sat_id,
            'SatSystem': sat_sys
        }
        
        # Add parameters common to all systems (clock parameters)
        if len(values) >= 3:
            record['SVclockBias'] = values[0]
            record['SVclockDrift'] = values[1] 
            record['SVclockDriftRate'] = values[2]
        
        # Add system-specific parameters
        if sat_sys in ['G', 'E', 'J', 'C', 'I']:  # GPS-like systems
            _add_gps_like_params(record, values)
        elif sat_sys == 'R':  # GLONASS
            _add_glonass_params(record, values)
        elif sat_sys == 'S':  # SBAS
            _add_sbas_params(record, values)
        
        return record
        
    except Exception as e:
        logging.warning(f"Error processing navigation data for {sat_id}: {e}")
        return None


def _add_gps_like_params(record: dict, values: list) -> None:
    """Add GPS-like orbital parameters to the record."""
    param_names = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE',
        'Crs', 'DeltaN', 'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'TimeEph',
        'Cic', 'OMEGA', 'CIS', 'Io', 'Crc', 'omega', 'OMEGA_DOT', 'IDOT',
        'CodesL2', 'GPSWeek', 'L2Pflag', 'SVacc', 'SVhealth', 'TGD', 'IODC',
        'TransTime', 'FitIntvl'
    ]
    
    for i, param_name in enumerate(param_names):
        if i < len(values):
            record[param_name] = values[i]
        else:
            record[param_name] = np.nan


def _add_glonass_params(record: dict, values: list) -> None:
    """Add GLONASS-specific parameters to the record."""
    glonass_params = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'X', 'Y', 'Z',
        'Vx', 'Vy', 'Vz', 'Ax', 'Ay', 'Az', 'Health', 'FreqNum', 'AgeOfInfo'
    ]
    
    for i, param_name in enumerate(glonass_params):
        if i < len(values):
            record[param_name] = values[i]
        else:
            record[param_name] = np.nan


def _add_sbas_params(record: dict, values: list) -> None:
    """Add SBAS-specific parameters to the record."""
    sbas_params = [
        'SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'X', 'Y', 'Z',
        'Vx', 'Vy', 'Vz', 'Ax', 'Ay', 'Az', 'Health', 'URA', 'IODN'
    ]
    
    for i, param_name in enumerate(sbas_params):
        if i < len(values):
            record[param_name] = values[i]
        else:
            record[param_name] = np.nan