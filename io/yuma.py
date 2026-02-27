import os
import datetime
import typing
import pandas as pd
from pathlib import Path
from .hdf5_utils import handle_output

class YumaData:
    @classmethod
    def from_bytes(cls, data: bytes):
        lines = data.decode('utf-8', errors='ignore').split('\n')
        satellites = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('*'):
                i += 1
                continue
            if ('almanac for PRN' in line) or (line.startswith('ID:') and ':' in line):
                try:
                    start_idx = i + 1 if 'almanac for PRN' in line else i
                    params = {}
                    j = start_idx
                    while j < min(len(lines), start_idx + 20):
                        l = lines[j].strip()
                        if ':' in l:
                            k, v = l.split(':', 1)
                            k, v = k.strip().lower(), v.strip().replace('E', 'e')
                            if 'id' in k: params['prn'] = int(v)
                            elif 'health' in k: params['health'] = int(v)
                            elif 'eccentricity' in k: params['eccentricity'] = float(v)
                            elif 'time of applicability' in k: params['time_of_applicability'] = float(v)
                            elif 'orbital inclination' in k: params['orbital_inclination'] = float(v)
                            elif 'rate of right ascen' in k: params['rate_of_right_ascension'] = float(v)
                            elif 'sqrt' in k: params['sqrt_semi_major_axis'] = float(v)
                            elif 'right ascen at week' in k: params['longitude_ascending_node'] = float(v)
                            elif 'argument of perigee' in k: params['argument_of_perigee'] = float(v)
                            elif 'mean anom' in k: params['mean_anomaly'] = float(v)
                            elif 'af0' in k: params['af0'] = float(v)
                            elif 'af1' in k: params['af1'] = float(v)
                            elif 'week' in k: params['week'] = int(v)
                        j += 1
                        required = ['prn', 'health', 'eccentricity', 'time_of_applicability', 'orbital_inclination', 'rate_of_right_ascension', 'sqrt_semi_major_axis', 'longitude_ascending_node', 'argument_of_perigee', 'mean_anomaly', 'af0', 'af1', 'week']
                        if all(p in params for p in required):
                            satellites.append(params)
                            break
                    i = j
                except:
                    i += 1
            else:
                i += 1
        return satellites

    @classmethod
    def from_file(cls, path: typing.Union[str, bytes, os.PathLike]):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

def read_yuma_to_dataframe(file_path: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    satellites = YumaData.from_file(file_path)
    if not satellites: return pd.DataFrame()
    data = []
    for s in satellites:
        data.append({
            'PRN': s['prn'], 'Health': s['health'], 'Eccentricity': f"{s['eccentricity']:.10e}",
            'ToA': f"{s['time_of_applicability']:.10e}", 'Inclination': f"{s['orbital_inclination']:.10e}",
            'RORA': f"{s['rate_of_right_ascension']:.10e}", 'SqrtA': f"{s['sqrt_semi_major_axis']:.10e}",
            'LAN': f"{s['longitude_ascending_node']:.10e}", 'AoP': f"{s['argument_of_perigee']:.10e}",
            'MeanAnomaly': f"{s['mean_anomaly']:.10e}", 'AF0': f"{s['af0']:.10e}", 'AF1': f"{s['af1']:.10e}",
            'Week': s['week'],
        })
    df = pd.DataFrame(data)
    if not df.empty: df = df.sort_values('PRN').reset_index(drop=True)
    return df

def load_yuma_almanac(
    file_path: typing.Union[str, bytes, os.PathLike],
    version: typing.Optional[str] = None,
    return_type: str = 'dataframe',
    output_path: typing.Optional[str] = None,
    **kwargs
) -> typing.Union[pd.DataFrame, str]:
    """
    Load a YUMA Almanac file.
    
    Args:
        file_path: Path to the YUMA file
        version: Optional format version
        return_type: Output format ('dataframe' or 'hdf5')
        output_path: Path to save the HDF5 file (required if return_type is 'hdf5')
        **kwargs: Additional args
        
    Returns:
        A pandas DataFrame, or a string pointing to the saved HDF5 file.
    """
    df = read_yuma_to_dataframe(file_path)
    return handle_output(df, return_type, output_path, key='yuma')
