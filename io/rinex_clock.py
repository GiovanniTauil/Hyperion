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

def parse_header(lines: list[str]) -> tuple[dict, int]:
    header = {}
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if 'END OF HEADER' in line: break
        if len(line) > 60:
            label = line[60:].strip()
            content = line[:60].strip()
        else:
            label = ""
            content = line.strip()
        if label:
            if label in header:
                if isinstance(header[label], list): header[label].append(content)
                else: header[label] = [header[label], content]
            else: header[label] = content
        i += 1
    return header, i + 1

def parse_data(lines: list[str], start_index: int) -> list[dict]:
    data = []
    i = start_index
    while i < len(lines):
        line = lines[i].rstrip('\n')
        i += 1
        if not line.strip(): continue
        if len(line) >= 2 and line[0:2].isalpha():
            try:
                pattern = r'^(\w{2})\s+(\S+?)\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d+\.?\d*)\s+(\d+)(?:\s+(.*))?'
                match = re.match(pattern, line)
                if not match: continue
                record_type, record_id = match.group(1), match.group(2)
                year, month, day, hour, minute = int(match.group(3)), int(match.group(4)), int(match.group(5)), int(match.group(6)), int(match.group(7))
                second, num_values = float(match.group(8)), int(match.group(9))
                values_str = match.group(10) if match.group(10) else ""
                epoch = datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1_000_000))
                values = []
                if values_str:
                    for part in values_str.split():
                        if len(values) < num_values and part.strip():
                            try: values.append(float(part.replace('D', 'e').replace('E', 'e')))
                            except ValueError: pass
                while len(values) < num_values and i < len(lines):
                    next_line = lines[i].rstrip('\n')
                    if next_line.strip() and (len(next_line) < 2 or not next_line[0:2].isalpha()):
                        for part in next_line.split():
                            if len(values) < num_values and part.strip():
                                try: values.append(float(part.replace('D', 'e').replace('E', 'e')))
                                except ValueError: pass
                        i += 1
                    else: break
                if len(values) == num_values:
                    data.append({'Type': record_type, 'ID': record_id, 'Epoch': epoch, 'Values': values})
            except (ValueError, IndexError): continue
    return data

def read_rinex_clock_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> tuple[pd.DataFrame, dict]:
    file_path = Path(file_path).expanduser()
    if not file_path.exists(): raise FileNotFoundError(f"RINEX Clock file not found: {file_path}")
    with _open_file(file_path) as f: lines = f.readlines()
    if not lines: raise ValueError("File is empty")
    header, data_start = parse_header(lines)
    if 'RINEX VERSION / TYPE' not in header: raise ValueError("Invalid RINEX Clock file: missing version/type header")
    version_line = header['RINEX VERSION / TYPE']
    version_parts = re.split(r'\s+', version_line.strip())
    # Support 'C' or 'CLOCK' type designators.
    if len(version_parts) < 2 or ('C' not in version_parts and 'CLOCK' not in version_parts): raise ValueError(f"Not a RINEX Clock file: {version_line}")
    data_list = parse_data(lines, data_start)
    if not data_list: return pd.DataFrame(columns=['Type', 'ID', 'Epoch', 'Values']), header
    df = pd.DataFrame(data_list)
    df = df.sort_values('Epoch').reset_index(drop=True)
    return df, header
    
def load_rinex_clock(
    file_path: typing.Union[str, bytes, os.PathLike],
    version: typing.Optional[str] = None,
    return_type: str = 'dataframe',
    output_path: typing.Optional[str] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    """
    Load a RINEX Clock file.
    
    Args:
        file_path: Path to the RINEX Clock file
        version: Expected RINEX version (auto-detected if None)
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        A pandas DataFrame, or a string pointing to the saved HDF5 file.
    """
    df, _ = read_rinex_clock_to_dataframe(file_path)
    if 'Values' in df.columns and return_type.lower() == 'hdf5':
        df['Values'] = df['Values'].apply(lambda x: str(x) if isinstance(x, list) else x)
    return handle_output(df, return_type, output_path, key='rinex.clock')
