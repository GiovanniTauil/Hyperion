"""
RINEX Clock File Reader

This module provides functionality to read RINEX Clock files (versions 3.0 and 3.04)
and convert them to pandas DataFrames. RINEX Clock files contain satellite and receiver
clock offset data organized by epoch and type.

Author: Based on attempted solution and extended for Hyperion framework
"""

import os
import re
import logging
import typing
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def parse_header(lines: list[str]) -> tuple[dict, int]:
    """
    Parse the header lines of a RINEX Clock file into a dictionary.
    
    Args:
        lines: List of lines from the RINEX Clock file
        
    Returns:
        tuple: (header dictionary, index after header)
    """
    header = {}
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if 'END OF HEADER' in line:
            break
        
        # Extract label from position 60 onwards (RINEX format)
        if len(line) > 60:
            label = line[60:].strip()
            content = line[:60].strip()
        else:
            label = ""
            content = line.strip()
        
        if label:
            if label in header:
                # Handle multiple entries for the same label
                if isinstance(header[label], list):
                    header[label].append(content)
                else:
                    header[label] = [header[label], content]
            else:
                header[label] = content
        i += 1
    
    return header, i + 1  # Return header and index after header


def parse_data(lines: list[str], start_index: int) -> list[dict]:
    """
    Parse the data records from RINEX Clock file into a list of dictionaries.
    
    RINEX Clock format is space-delimited but may have varying spacing.
    This function handles both fixed-width and space-delimited formats.
    
    Args:
        lines: List of lines from the RINEX Clock file
        start_index: Index to start parsing data from
        
    Returns:
        list: List of dictionaries containing parsed clock data
    """
    data = []
    i = start_index
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        i += 1
        
        # Skip blank lines
        if not line.strip():
            continue
            
        # Check if this is the start of a new record (first 2 chars are alphabetic)
        if len(line) >= 2 and line[0:2].isalpha():
            try:
                # Parse record using regex to handle varying spacing
                # RINEX Clock format: TYPE ID YYYY MM DD HH MM SS.SSSSSS N_VAL DATA_VALUES...
                import re
                
                # Match the clock record pattern with flexible spacing
                # More robust pattern that handles station names with numbers
                # Look for the pattern: TYPE STATION YYYY MM DD HH MM SS.SS N_VAL DATA...
                # Use non-greedy matching and specific constraints
                pattern = r'^(\w{2})\s+(\S+?)\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d+\.?\d*)\s+(\d+)(?:\s+(.*))?'
                match = re.match(pattern, line)
                
                if not match:
                    logging.warning(f"Could not parse clock record at line {i}: {line[:50]}...")
                    continue
                
                record_type = match.group(1)  # e.g., 'AS', 'AR'
                record_id = match.group(2)    # e.g., 'G01', 'ALGO'
                year = int(match.group(3))
                month = int(match.group(4))
                day = int(match.group(5))
                hour = int(match.group(6))
                minute = int(match.group(7))
                second = float(match.group(8))
                num_values = int(match.group(9))
                values_str = match.group(10) if match.group(10) else ""  # Remaining values string
                
                # Create datetime object
                epoch = datetime(
                    year, month, day, hour, minute, 
                    int(second), 
                    int((second - int(second)) * 1_000_000)
                )
                
                # Parse data values from the remaining string and continuation lines
                values = []
                
                # Parse values from the current line
                if values_str:
                    value_parts = values_str.split()
                    for part in value_parts:
                        if len(values) < num_values and part.strip():
                            value_str = part.replace('D', 'e').replace('E', 'e')
                            try:
                                values.append(float(value_str))
                            except ValueError:
                                logging.warning(f"Could not parse value '{part}' at line {i}")
                
                # Collect values from continuation lines if needed
                while len(values) < num_values and i < len(lines):
                    next_line = lines[i].rstrip('\n')
                    
                    # Check if this is a continuation line (not starting with alphabetic chars)
                    if next_line.strip() and (len(next_line) < 2 or not next_line[0:2].isalpha()):
                        # Parse values from continuation line
                        cont_parts = next_line.split()
                        for part in cont_parts:
                            if len(values) < num_values and part.strip():
                                value_str = part.replace('D', 'e').replace('E', 'e')
                                try:
                                    values.append(float(value_str))
                                except ValueError:
                                    logging.warning(f"Could not parse continuation value '{part}' at line {i}")
                        i += 1
                    else:
                        break
                
                # Only add record if we have the expected number of values
                if len(values) == num_values:
                    data.append({
                        'Type': record_type,
                        'ID': record_id,
                        'Epoch': epoch,
                        'Values': values
                    })
                else:
                    logging.warning(f"Expected {num_values} values but got {len(values)} for record at line {i}")
                    
            except (ValueError, IndexError) as e:
                # Handle parsing errors gracefully
                logging.warning(f"Error parsing line {i}: {e}")
                continue
    
    return data


def read_rinex_clock_to_dataframe(
    file_path: typing.Union[str, bytes, os.PathLike]
) -> tuple[pd.DataFrame, dict]:
    """
    Read RINEX Clock file and convert to pandas DataFrame.
    
    Supports RINEX Clock versions 3.0 and 3.04. RINEX Clock files contain satellite
    and receiver clock offset data with the following record types:
    - AS: Satellite clock
    - AR: Receiver clock  
    - CR: Receiver clock (alternative)
    - DR: Receiver clock rate
    
    Args:
        file_path: Path to the RINEX Clock file
        
    Returns:
        tuple: (pandas.DataFrame, header dictionary)
        
        DataFrame contains columns:
        - Type: Record type (AS, AR, CR, DR)
        - ID: Satellite PRN or station identifier
        - Epoch: Time of clock measurement (datetime object)
        - Values: List of clock values (bias, rate, acceleration in seconds)
        
        Header dictionary contains metadata from the file header.
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported or parsing fails
    """
    try:
        # Validate file exists
        file_path = Path(file_path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"RINEX Clock file not found: {file_path}")
        
        # Read file contents
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError("File is empty")
        
        # Parse header and get data start index
        header, data_start = parse_header(lines)
        
        # Validate RINEX Clock format
        if 'RINEX VERSION / TYPE' not in header:
            raise ValueError("Invalid RINEX Clock file: missing version/type header")
        
        version_line = header['RINEX VERSION / TYPE']
        if 'CLOCK' not in version_line.upper():
            raise ValueError(f"Not a RINEX Clock file: {version_line}")
        
        # Parse data records
        data_list = parse_data(lines, data_start)
        
        if not data_list:
            logging.warning("No valid clock data records found in file")
            return pd.DataFrame(columns=['Type', 'ID', 'Epoch', 'Values']), header
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Sort by epoch for consistent output
        df = df.sort_values('Epoch').reset_index(drop=True)
        
        return df, header
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading RINEX Clock file: {e}")


def expand_clock_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 'Values' column into separate columns for clock parameters.
    
    This is an optional utility function to expand clock values into separate
    columns when all records have the same number of values.
    
    Args:
        df: DataFrame from read_rinex_clock_to_dataframe
        
    Returns:
        DataFrame with expanded columns:
        - Clock_Bias(s): Clock bias in seconds
        - Clock_Rate(s/s): Clock rate in seconds/second (if available)
        - Clock_Acceleration(s/s²): Clock acceleration (if available)
    """
    if df.empty:
        return df
    
    # Check if all records have the same number of values
    value_lengths = df['Values'].apply(len)
    if value_lengths.nunique() > 1:
        logging.warning("Records have different numbers of values. Cannot expand uniformly.")
        return df
    
    max_values = value_lengths.iloc[0]
    
    # Create expanded DataFrame
    df_expanded = df.copy()
    
    # Define column names based on number of values
    if max_values >= 1:
        df_expanded['Clock_Bias(s)'] = df['Values'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    if max_values >= 2:
        df_expanded['Clock_Rate(s/s)'] = df['Values'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    if max_values >= 3:
        df_expanded['Clock_Acceleration(s/s²)'] = df['Values'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    
    # Add any additional values as generic columns
    for i in range(3, max_values):
        df_expanded[f'Value_{i+1}'] = df['Values'].apply(lambda x: x[i] if len(x) > i else np.nan)
    
    return df_expanded