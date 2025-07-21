import dataclasses
import datetime
import typing
import os
import re
import pandas as pd
from decimal import Decimal
import requests
from pathlib import Path


@dataclasses.dataclass
class YumaAlmanac:
    """Represents a single satellite's almanac data from YUMA format."""
    prn: int
    health: int
    eccentricity: float
    time_of_applicability: float  # seconds
    orbital_inclination: float  # radians  
    rate_of_right_ascension: float  # radians/second
    sqrt_semi_major_axis: float  # sqrt(meters)
    longitude_ascending_node: float  # radians
    argument_of_perigee: float  # radians
    mean_anomaly: float  # radians
    af0: float  # clock bias (seconds)
    af1: float  # clock drift (seconds/second)
    week: int


@dataclasses.dataclass
class YumaData:
    """Container for all satellites' almanac data."""
    satellites: list[YumaAlmanac]
    
    @classmethod
    def from_bytes(cls, data: bytes):
        """Parse YUMA almanac data from bytes."""
        lines = data.decode('utf-8', errors='ignore').split('\n')
        satellites = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('*'):
                i += 1
                continue
                
            # Look for satellite data blocks
            # YUMA format typically has "Week XXX almanac for PRN-XX" header
            # or starts directly with ID:
            if ('almanac for PRN' in line) or (line.startswith('ID:') and ':' in line):
                try:
                    # If it's a header line, start parsing from the next line
                    start_idx = i + 1 if 'almanac for PRN' in line else i
                    satellite = cls._parse_satellite_block(lines, start_idx)
                    if satellite:
                        satellites.append(satellite)
                        # Skip ahead to avoid re-parsing
                        i = start_idx + 15  
                    else:
                        i += 1
                except (ValueError, IndexError):
                    i += 1
            else:
                i += 1
                
        return cls(satellites=satellites)
    
    @classmethod
    def _parse_satellite_block(cls, lines: list[str], start_idx: int) -> typing.Optional[YumaAlmanac]:
        """Parse a single satellite's almanac data block."""
        try:
            # Parse the standard YUMA format
            # Each satellite has a specific format with labeled fields
            
            params = {}
            i = start_idx
            
            # Parse each line until we have all required parameters
            while i < len(lines) and i < start_idx + 20:  # Maximum search range
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                    
                # Handle different line formats
                if ':' in line:
                    # Format: "Parameter: value"
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    # Map YUMA parameters (case insensitive matching)
                    if 'id' in key:
                        params['prn'] = int(value)
                    elif 'health' in key:
                        params['health'] = int(value)
                    elif 'eccentricity' in key:
                        # Handle scientific notation
                        params['eccentricity'] = float(value.replace('E', 'e'))
                    elif 'time of applicability' in key:
                        params['time_of_applicability'] = float(value)
                    elif 'orbital inclination' in key:
                        params['orbital_inclination'] = float(value.replace('E', 'e'))
                    elif 'rate of right ascen' in key:
                        params['rate_of_right_ascension'] = float(value.replace('E', 'e'))
                    elif 'sqrt' in key:
                        params['sqrt_semi_major_axis'] = float(value)
                    elif 'right ascen at week' in key:
                        params['longitude_ascending_node'] = float(value.replace('E', 'e'))
                    elif 'argument of perigee' in key:
                        params['argument_of_perigee'] = float(value.replace('E', 'e'))
                    elif 'mean anom' in key:
                        params['mean_anomaly'] = float(value.replace('E', 'e'))
                    elif 'af0' in key:
                        params['af0'] = float(value.replace('E', 'e'))
                    elif 'af1' in key:
                        params['af1'] = float(value.replace('E', 'e'))
                    elif 'week' in key:
                        params['week'] = int(value)
                
                i += 1
                
                # If we have all required parameters, create the almanac
                required_params = [
                    'prn', 'health', 'eccentricity', 'time_of_applicability',
                    'orbital_inclination', 'rate_of_right_ascension', 
                    'sqrt_semi_major_axis', 'longitude_ascending_node',
                    'argument_of_perigee', 'mean_anomaly', 'af0', 'af1', 'week'
                ]
                
                if all(param in params for param in required_params):
                    return YumaAlmanac(**params)
            
            # If we don't have all parameters, return None
            return None
            
        except (ValueError, IndexError, TypeError):
            return None
    
    @classmethod
    def from_file(cls, path: typing.Union[str, bytes, os.PathLike]):
        """Load YUMA almanac data from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as file:
            return cls.from_bytes(file.read())


def read_yuma_to_dataframe(
    file_path: typing.Union[str, bytes, os.PathLike]
) -> pd.DataFrame:
    """
    Read YUMA almanac file and convert to pandas DataFrame.
    
    Args:
        file_path: Path to the YUMA almanac file
        
    Returns:
        pandas.DataFrame with columns:
        - PRN: Satellite PRN number
        - Health: Satellite health status
        - Eccentricity: Orbital eccentricity
        - ToA: Time of applicability (seconds)
        - Inclination: Orbital inclination (radians)
        - RORA: Rate of right ascension (radians/second)
        - SqrtA: Square root of semi-major axis (sqrt(meters))
        - LAN: Longitude of ascending node (radians)
        - AoP: Argument of perigee (radians)
        - MeanAnomaly: Mean anomaly (radians)
        - AF0: Clock bias (seconds)
        - AF1: Clock drift (seconds/second)
        - Week: GPS week number
    """
    yuma_data = YumaData.from_file(file_path)

    if not yuma_data.satellites:
        return pd.DataFrame()  # Empty DataFrame if no satellites found

    data = []
    for sat in yuma_data.satellites:
        row = {
            'PRN': sat.prn,
            'Health': sat.health,
            'Eccentricity': f"{sat.eccentricity:.10e}",
            'ToA': f"{sat.time_of_applicability:.10e}",
            'Inclination': f"{sat.orbital_inclination:.10e}",
            'RORA': f"{sat.rate_of_right_ascension:.10e}",
            'SqrtA': f"{sat.sqrt_semi_major_axis:.10e}",
            'LAN': f"{sat.longitude_ascending_node:.10e}",
            'AoP': f"{sat.argument_of_perigee:.10e}",
            'MeanAnomaly': f"{sat.mean_anomaly:.10e}",
            'AF0': f"{sat.af0:.10e}",
            'AF1': f"{sat.af1:.10e}",
            'Week': sat.week,
        }
        data.append(row)

    df = pd.DataFrame(data)
    # Sort by PRN for consistent output
    if not df.empty:
        df = df.sort_values('PRN').reset_index(drop=True)

    return df


def _construct_yuma_url(date: datetime.date) -> str:
    """
    Construct the NAVCEN URL for a YUMA almanac file.
    
    Args:
        date: Date for the almanac file
        
    Returns:
        str: URL for the YUMA almanac file
    """
    year = date.year
    doy = date.timetuple().tm_yday  # Day of year (1-366)
    return f"https://navcen.uscg.gov/sites/default/files/gps/almanac/{year}/Yuma/{doy:03d}.alm"


def _construct_filename(date: datetime.date, save_dir: typing.Union[str, Path]) -> Path:
    """
    Construct the local filename for a YUMA almanac file.
    
    Args:
        date: Date for the almanac file
        save_dir: Directory to save the file
        
    Returns:
        Path: Full path for the local file
    """
    save_dir = Path(save_dir)
    return save_dir / f"yumaAlmanac_{date.isoformat()}.alm"


def _download_file_from_url(url: str, filepath: Path, timeout: int = 10) -> bool:
    """
    Download a file from URL and save to local path.
    
    Args:
        url: URL to download from
        filepath: Local path to save the file
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        requests.RequestException: For network-related errors
        OSError: For file I/O errors
    """
    response = requests.get(url, timeout=timeout)
    
    # Check if the response is successful and has content
    if response.status_code == 200 and response.content.strip():
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False


def _ensure_directory_exists(save_dir: typing.Union[str, Path]) -> Path:
    """
    Ensure the save directory exists, creating it if necessary.
    
    Args:
        save_dir: Directory path to create
        
    Returns:
        Path: The validated directory path
        
    Raises:
        OSError: If directory cannot be created
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def download_yuma_almanac(date: typing.Optional[datetime.date] = None, 
                         save_dir: typing.Union[str, Path] = ".", 
                         raise_on_fail: bool = False, 
                         overwrite: bool = False, 
                         verbose: bool = True) -> typing.Optional[str]:
    """
    Download a YUMA almanac file from NAVCEN for a chosen date.
    
    If the date is empty, it will try to find today's date. If there is no file
    for the chosen date, it will display a message and retrieve it from the 
    previous day.

    Args:
        date: Date for which the almanac should be downloaded. If None, uses today's date.
        save_dir: Directory to save the downloaded file (default: current directory)
        raise_on_fail: If True, raise exception on failure instead of printing a message
        overwrite: If True, overwrite the file if it already exists
        verbose: If True, print status messages

    Returns:
        str or None: Path to the saved file if successful, else None
        
    Raises:
        RuntimeError: If raise_on_fail is True and download fails
    """
    try:
        # Use today's date if no date provided
        if date is None:
            date = datetime.date.today()
            if verbose:
                print(f" Using today's date: {date}")
        
        # Ensure save directory exists
        save_dir = _ensure_directory_exists(save_dir)
        
        # Try to download for the requested date first
        result = _try_download_for_date(date, save_dir, overwrite, verbose, raise_on_fail)
        if result is not None:
            return result
            
        # If failed, try previous day
        previous_date = date - datetime.timedelta(days=1)
        if verbose:
            print(f" No file found for {date}, trying previous day: {previous_date}")
            
        result = _try_download_for_date(previous_date, save_dir, overwrite, verbose, raise_on_fail)
        if result is not None:
            return result
            
        # If both dates failed
        msg = f"Failed to download YUMA almanac for {date} or {previous_date}"
        if raise_on_fail:
            raise RuntimeError(msg)
        if verbose:
            print(" " + msg)
        return None

    except requests.RequestException as e:
        msg = f"Request error: {e}"
    except OSError as e:
        msg = f"File I/O error: {e}"
    except Exception as e:
        msg = f"Unexpected error: {e}"

    # Handle exceptions
    if raise_on_fail:
        raise RuntimeError(msg)
    if verbose:
        print(" " + msg)
    return None


def _try_download_for_date(date: datetime.date, 
                          save_dir: Path, 
                          overwrite: bool, 
                          verbose: bool, 
                          raise_on_fail: bool) -> typing.Optional[str]:
    """
    Try to download YUMA almanac for a specific date.
    
    Args:
        date: Date for the almanac
        save_dir: Directory to save the file
        overwrite: Whether to overwrite existing files
        verbose: Whether to print status messages
        raise_on_fail: Whether to raise exceptions on failure
        
    Returns:
        str or None: Path to the saved file if successful, else None
    """
    try:
        # Construct URL and filename
        url = _construct_yuma_url(date)
        filename = _construct_filename(date, save_dir)

        # Check if file already exists
        if filename.exists() and not overwrite:
            if verbose:
                print(f" File already exists: {filename}")
            return str(filename)

        # Try to download the file
        if _download_file_from_url(url, filename):
            if verbose:
                print(f" Downloaded: {filename}")
            return str(filename)
        else:
            if verbose:
                print(f" Failed to download file from: {url}")
            return None

    except requests.RequestException as e:
        if verbose:
            print(f" Request error for {date}: {e}")
        return None
    except OSError as e:
        if verbose:
            print(f" File I/O error for {date}: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f" Unexpected error for {date}: {e}")
        return None
