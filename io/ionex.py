"""
IONEX (Ionosphere Map Exchange Format) Parser.
Adapted from the original code previously available.
"""
import os
import typing
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from .hdf5_utils import handle_output

def _open_file(file_path: Path):
    if file_path.suffix.lower() == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
    return file_path.open('r', encoding='utf-8', errors='ignore')

# Import the pre-existing reader from the original ionexdf.py we saw earlier
import sys
# It seems ionexdf.py is in the base directory, let's incorporate standard IONEX parsing logic here
# similar to how we handled SP3, directly embedding the optimized IONEX parser.

import dataclasses
import datetime

@dataclasses.dataclass
class IonexHeader:
    version: typing.Optional[str] = None
    file_type: typing.Optional[str] = None
    program: typing.Optional[str] = None
    run_by: typing.Optional[str] = None
    date: typing.Optional[str] = None
    description: typing.Optional[str] = None
    epoch_first_map: typing.Optional[datetime.datetime] = None
    epoch_last_map: typing.Optional[datetime.datetime] = None
    interval: typing.Optional[float] = None
    num_maps: typing.Optional[int] = None
    mapping_function: typing.Optional[str] = None
    elevation_cutoff: typing.Optional[float] = None
    observables_used: typing.Optional[str] = None
    num_stations: typing.Optional[int] = None
    num_satellites: typing.Optional[int] = None
    base_radius: typing.Optional[float] = None
    map_dimension: typing.Optional[int] = None
    lat1: typing.Optional[float] = None
    lat2: typing.Optional[float] = None
    dlat: typing.Optional[float] = None
    lon1: typing.Optional[float] = None
    lon2: typing.Optional[float] = None
    dlon: typing.Optional[float] = None
    hgt1: typing.Optional[float] = None
    hgt2: typing.Optional[float] = None
    dhgt: typing.Optional[float] = None
    exponent: typing.Optional[int] = None
    comment: typing.List[str] = dataclasses.field(default_factory=list)
    auxiliary_data: typing.Dict[str, typing.List[str]] = dataclasses.field(default_factory=dict)

def parse_header(lines: typing.List[str]) -> typing.Tuple[IonexHeader, int]:
    header = IonexHeader()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if 'END OF HEADER' in line:
            break
        
        label, content = "", ""
        if len(line) >= 59:
            label = line[59:].strip()
            content = line[:59].strip()
            if not label and len(line) >= 60:
                label = line[60:].strip()
                content = line[:60].strip()
        else:
            content = line.strip()
            
        if label:
            try:
                if label == 'IONEX VERSION / TYPE':
                    parts = content.split()
                    if len(parts) >= 1: header.version = parts[0]
                    if len(parts) >= 2: header.file_type = parts[1]
                elif label == 'EPOCH OF FIRST MAP':
                    header.epoch_first_map = datetime.datetime(*map(int, content.split()))
                elif label == 'EPOCH OF LAST MAP':
                    header.epoch_last_map = datetime.datetime(*map(int, content.split()))
                elif label == 'INTERVAL':
                    header.interval = float(content)
                elif label == 'MAP DIMENSION':
                    header.map_dimension = int(content)
                elif 'HGT1 / HGT2 / DHGT' in label:
                    header.hgt1, header.hgt2, header.dhgt = map(float, content.split()[:3])
                elif 'LAT1 / LAT2 / DLAT' in label:
                    header.lat1, header.lat2, header.dlat = map(float, content.split()[:3])
                elif 'LON1 / LON2 / DLON' in label:
                    header.lon1, header.lon2, header.dlon = map(float, content.split()[:3])
                elif label == 'EXPONENT':
                    header.exponent = int(content)
            except:
                pass
        i += 1
    return header, i + 1

def extract_grid_params(header: IonexHeader) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if None in (header.lat1, header.lat2, header.dlat): raise ValueError("Missing lat grid")
    if None in (header.lon1, header.lon2, header.dlon): raise ValueError("Missing lon grid")
    if None in (header.hgt1, header.hgt2, header.dhgt): raise ValueError("Missing hgt grid")
    
    sign_lat = 1 if header.lat2 >= header.lat1 else -1
    dlat = float(abs(header.dlat)) * sign_lat if header.dlat != 0 else 0
    lats = np.arange(header.lat1, header.lat2 + (dlat / 2 if dlat != 0 else 0), dlat) if dlat != 0 else np.array([header.lat1])
    
    sign_lon = 1 if header.lon2 >= header.lon1 else -1
    dlon = float(abs(header.dlon)) * sign_lon if header.dlon != 0 else 0
    lons = np.arange(header.lon1, header.lon2 + (dlon / 2 if dlon != 0 else 0), dlon) if dlon != 0 else np.array([header.lon1])
    
    if header.map_dimension == 2: hgts = np.array([header.hgt1])
    else: 
        sign_hgt = 1 if header.hgt2 >= header.hgt1 else -1
        dhgt = float(abs(header.dhgt)) * sign_hgt if header.dhgt != 0 else 0
        hgts = np.arange(header.hgt1, header.hgt2 + (dhgt / 2 if dhgt != 0 else 0), dhgt) if dhgt != 0 else np.array([header.hgt1])
    return lats, lons, hgts

def parse_data_maps(lines: typing.List[str], start_index: int, header: IonexHeader, lats: np.ndarray, lons: np.ndarray, hgts: np.ndarray) -> typing.List[typing.Dict]:
    data_records = []
    i = start_index
    current_map_type = None
    current_epoch = None
    current_exponent = header.exponent if header.exponent is not None else 0
    nlons = len(lons)
    default_height = hgts[0] if len(hgts) > 0 else 0.0
    
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if not line.strip(): continue
        if 'END OF FILE' in line: break
        
        label, content = "", ""
        if len(line) >= 59:
            label, content = line[59:].strip(), line[:59].strip()
            if not label and len(line)>=60: label, content = line[60:].strip(), line[:60].strip()
        else: content = line.strip()
            
        if 'START OF' in label and 'MAP' in label:
            if 'TEC' in label: current_map_type = 'TEC'
            elif 'RMS' in label: current_map_type = 'RMS'
            elif 'HEIGHT' in label: current_map_type = 'HEIGHT'
            else: current_map_type = 'UNKNOWN'
            current_exponent = header.exponent if header.exponent is not None else 0
            continue
            
        if 'END OF' in label and 'MAP' in label:
            current_map_type = None
            continue
            
        if 'EPOCH OF CURRENT MAP' in label:
            try: current_epoch = datetime.datetime(*map(int, content.split()))
            except: pass
            continue
            
        if label == 'EXPONENT':
            try: current_exponent = int(content)
            except: current_exponent = 0
            continue
            
        if 'LAT/LON1/LON2/DLON/H' in label:
            try:
                import re
                nums = re.findall(r'-?\d+\.\d+|-?\d+', content)
                if len(nums) >= 4:
                    lat = float(nums[0])
                    height = float(nums[4]) if len(nums) >= 5 and header.map_dimension == 3 else default_height
                
                values = []
                while len(values) < nlons and i < len(lines):
                    nline = lines[i].rstrip()
                    i += 1
                    if len(nline) >= 59 and nline[59:].strip() in ['LAT/LON1/LON2/DLON/H', 'END OF', 'START OF', 'EPOCH OF']:
                        i -= 1; break
                    for j in range(0, len(nline), 5):
                        vstr = nline[j:j+5].strip()
                        if vstr:
                            try: values.append(int(vstr))
                            except: pass
                        if len(values) >= nlons: break
                
                values = values[:nlons]
                scale = 10 ** current_exponent
                for lon_idx, val in enumerate(values):
                    if lon_idx >= nlons: break
                    sval = np.nan if val >= 999 else val * scale
                    data_records.append({
                        'Type': current_map_type,
                        'Epoch': current_epoch,
                        'Height': height,
                        'Latitude': lat,
                        'Longitude': lons[lon_idx],
                        'Value': float(sval)
                    })
            except: continue
    return data_records

def read_ionex_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    file_path = Path(file_path).expanduser()
    if not file_path.exists(): raise FileNotFoundError("IONEX file not found")
    
    with _open_file(file_path) as f:
        lines = f.readlines()
        
    header, start_idx = parse_header(lines)
    lats, lons, hgts = extract_grid_params(header)
    data_list = parse_data_maps(lines, start_idx, header, lats, lons, hgts)
    
    if not data_list:
        return pd.DataFrame(columns=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)'])
        
    df = pd.DataFrame(data_list)
    coords = df[['Epoch', 'Latitude', 'Longitude', 'Height']].drop_duplicates()
    res = coords.rename(columns={'Latitude': 'Latitude(deg)', 'Longitude': 'Longitude(deg)', 'Height': 'Height(km)'})
    
    tec_data = df[df['Type'] == 'TEC'][['Epoch', 'Latitude', 'Longitude', 'Height', 'Value']]
    if not tec_data.empty:
        res = res.merge(tec_data.rename(columns={'Value': 'TEC(0.1TECU)'}), 
                        left_on=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'],
                        right_on=['Epoch', 'Latitude', 'Longitude', 'Height'], how='left').drop(columns=['Latitude', 'Longitude', 'Height'], errors='ignore')
    else: res['TEC(0.1TECU)'] = np.nan
    
    rms_data = df[df['Type'] == 'RMS'][['Epoch', 'Latitude', 'Longitude', 'Height', 'Value']]
    if not rms_data.empty:
        res = res.merge(rms_data.rename(columns={'Value': 'RMS(0.1TECU)'}), 
                        left_on=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'],
                        right_on=['Epoch', 'Latitude', 'Longitude', 'Height'], how='left').drop(columns=['Latitude', 'Longitude', 'Height'], errors='ignore')
    else: res['RMS(0.1TECU)'] = np.nan
    
    res = res.sort_values(['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)']).reset_index(drop=True)
    return res

def load_ionex(
    file_path: typing.Union[str, bytes, os.PathLike],
    return_type: str = 'dataframe',
    output_path: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    """
    Load an IONEX file containing global ionosphere maps.
    
    Args:
        file_path: Path to the IONEX file
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        A pandas DataFrame, or a string pointing to the saved HDF5 file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    df = read_ionex_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='ionex')
