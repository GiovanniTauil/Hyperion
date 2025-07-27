import dataclasses
import datetime
import typing
import pandas as pd
import numpy as np


@dataclasses.dataclass
class IonexHeader:
    """Represents IONEX file header information."""
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


@dataclasses.dataclass
class IonexData:
    """Container for IONEX data."""
    header: IonexHeader
    data: pd.DataFrame


def parse_header(lines: typing.List[str]) -> typing.Tuple[IonexHeader, int]:
    """
    Parse the header lines into an IonexHeader object.
    
    Args:
        lines: List of lines from the IONEX file
        
    Returns:
        Tuple of (IonexHeader object, index of first line after header)
    """
    header = IonexHeader()
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip()
        if 'END OF HEADER' in line:
            break
        
        # IONEX format: labels typically start around position 59-60
        # Extract content and label parts more flexibly
        if len(line) >= 59:
            # Try position 59 first (common case), then 60 as fallback
            label = line[59:].strip()
            content = line[:59].strip()
            
            # If label is empty, try position 60
            if not label and len(line) >= 60:
                label = line[60:].strip()
                content = line[:60].strip()
        else:
            # Handle short lines
            label = ""
            content = line.strip()
        
        if label:
            try:
                # Handle specific header fields with proper error handling
                if label == 'IONEX VERSION / TYPE':
                    parts = content.split()
                    if len(parts) >= 1:
                        header.version = parts[0]
                    if len(parts) >= 2:
                        header.file_type = parts[1]
                elif label == 'PGM / RUN BY / DATE':
                    parts = content.split('/')
                    if len(parts) >= 1:
                        header.program = parts[0].strip()
                    if len(parts) >= 2:
                        header.run_by = parts[1].strip()
                    if len(parts) >= 3:
                        header.date = parts[2].strip()
                elif label == 'DESCRIPTION':
                    header.description = content
                elif label == 'EPOCH OF FIRST MAP':
                    parts = list(map(int, content.split()))
                    if len(parts) >= 6:
                        header.epoch_first_map = datetime.datetime(*parts)
                elif label == 'EPOCH OF LAST MAP':
                    parts = list(map(int, content.split()))
                    if len(parts) >= 6:
                        header.epoch_last_map = datetime.datetime(*parts)
                elif label == 'INTERVAL':
                    header.interval = float(content)
                elif label == 'MAPS IN FILE':
                    header.num_maps = int(content)
                elif label == 'MAPPING FUNCTION':
                    header.mapping_function = content
                elif label == 'ELEVATION CUTOFF':
                    header.elevation_cutoff = float(content)
                elif label == 'OBSERVABLES USED':
                    header.observables_used = content
                elif label == '# OF STATIONS':
                    header.num_stations = int(content)
                elif label == '# OF SATELLITES':
                    header.num_satellites = int(content)
                elif label == 'BASE RADIUS':
                    header.base_radius = float(content)
                elif label == 'MAP DIMENSION':
                    header.map_dimension = int(content)
                elif 'HGT1 / HGT2 / DHGT' in label:
                    # Parse height grid parameters
                    parts = content.split()
                    if len(parts) >= 3:
                        header.hgt1, header.hgt2, header.dhgt = map(float, parts[:3])
                elif 'LAT1 / LAT2 / DLAT' in label:
                    # Parse latitude grid parameters
                    parts = content.split()
                    if len(parts) >= 3:
                        header.lat1, header.lat2, header.dlat = map(float, parts[:3])
                elif 'LON1 / LON2 / DLON' in label:
                    # Parse longitude grid parameters
                    parts = content.split()
                    if len(parts) >= 3:
                        header.lon1, header.lon2, header.dlon = map(float, parts[:3])
                elif label == 'EXPONENT':
                    header.exponent = int(content)
                elif label == 'COMMENT':
                    header.comment.append(content)
                else:
                    # Handle auxiliary data (like DCBs)
                    if label not in header.auxiliary_data:
                        header.auxiliary_data[label] = []
                    header.auxiliary_data[label].append(content)
            except (ValueError, TypeError) as e:
                # Continue parsing if individual field parsing fails
                print(f"Warning: Error parsing header field '{label}': {e}")
        
        i += 1
    
    return header, i + 1


def extract_grid_params(header: IonexHeader) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract grid parameters from header.
    
    Args:
        header: IonexHeader object
        
    Returns:
        Tuple of (latitudes, longitudes, heights) arrays
    """
    if None in (header.lat1, header.lat2, header.dlat):
        raise ValueError("Missing latitude grid parameters")
    if None in (header.lon1, header.lon2, header.dlon):
        raise ValueError("Missing longitude grid parameters")
    if None in (header.hgt1, header.hgt2, header.dhgt):
        raise ValueError("Missing height grid parameters")
    if header.map_dimension is None:
        raise ValueError("Missing map dimension")
    
    # Generate coordinate arrays (inclusive)
    sign_lat = 1 if header.dlat > 0 else -1
    lats = np.arange(header.lat1, header.lat2 + sign_lat * (header.dlat / 2), header.dlat)
    
    sign_lon = 1 if header.dlon > 0 else -1
    lons = np.arange(header.lon1, header.lon2 + sign_lon * (header.dlon / 2), header.dlon)
    
    if header.map_dimension == 2:
        hgts = np.array([header.hgt1])
    else:
        sign_h = 1 if header.dhgt > 0 else -1
        hgts = np.arange(header.hgt1, header.hgt2 + sign_h * (header.dhgt / 2), header.dhgt)
    
    return lats, lons, hgts


def parse_data_maps(lines: typing.List[str], start_index: int, header: IonexHeader, 
                   lats: np.ndarray, lons: np.ndarray, hgts: np.ndarray) -> typing.List[typing.Dict]:
    """
    Parse the data maps into a list of records with improved error handling.
    
    Args:
        lines: List of lines from the IONEX file
        start_index: Index to start parsing from
        header: IonexHeader object
        lats: Latitude array
        lons: Longitude array
        hgts: Height array
        
    Returns:
        List of data records with flattened grid data
    """
    data_records = []
    i = start_index
    current_map_type = None  # TEC, RMS, HEIGHT
    current_epoch = None
    current_exponent = header.exponent if header.exponent is not None else 0
    current_height = None
    nlons = len(lons)
    
    # For 2D maps, use the first height from header
    default_height = hgts[0] if len(hgts) > 0 else 0.0
    
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        
        # Skip blank lines
        if not line.strip():
            continue
        
        # Check for end of file
        if 'END OF FILE' in line:
            break
        
        # Parse label and content more robustly
        label, content = _extract_label_and_content(line)
        
        # Handle map start markers
        if 'START OF' in label and 'MAP' in label:
            # Extract map type (TEC, RMS, HEIGHT)
            parts = label.split()
            if len(parts) >= 3:
                current_map_type = parts[2]  # TEC, RMS, HEIGHT
            # Reset exponent to header default for new map
            current_exponent = header.exponent if header.exponent is not None else 0
            continue
        
        # Handle map end markers
        if 'END OF' in label and 'MAP' in label:
            current_map_type = None
            continue
        
        # Handle epoch information
        if 'EPOCH OF CURRENT MAP' in label:
            try:
                parts = list(map(int, content.split()))
                if len(parts) >= 6:
                    current_epoch = datetime.datetime(*parts)
            except (ValueError, TypeError) as e:
                print(f"Warning: Error parsing epoch at line {i}: {e}")
            continue
        
        # Handle exponent (overrides header default for this map)
        if label == 'EXPONENT':
            try:
                current_exponent = int(content)
            except (ValueError, TypeError):
                current_exponent = 0
            continue
        
        # Handle grid data lines
        if 'LAT/LON1/LON2/DLON/H' in label:
            try:
                # Parse latitude and height from the grid specification line
                # Format can be like: "87.5-180.0 180.0   5.0 350.0" 
                # Need to handle cases where lat and lon1 are concatenated
                
                # Split on spaces first to get potential parts
                initial_parts = content.split()
                if len(initial_parts) < 4:
                    print(f"Warning: Invalid grid specification at line {i}: {content}")
                    continue
                
                # Check if first part contains latitude and lon1 concatenated
                first_part = initial_parts[0]
                if '-' in first_part and first_part.count('-') >= 1:
                    # Find the last '-' which likely separates lat from lon1
                    # Handle negative coordinates properly
                    parts_split = []
                    
                    # Look for pattern like "87.5-180.0" -> ["87.5", "-180.0"]
                    if first_part.count('-') == 1:
                        # Simple case: lat is positive, lon1 is negative
                        lat_str, lon1_str = first_part.split('-')
                        lon1_str = '-' + lon1_str  # Add back the negative sign
                        parts_split = [lat_str, lon1_str] + initial_parts[1:]
                    else:
                        # Complex case: multiple negatives, need smarter parsing
                        # Try to find where latitude ends and longitude begins
                        # Look for decimal points as separators
                        for split_pos in range(1, len(first_part)):
                            if first_part[split_pos] == '-' and first_part[split_pos-1].isdigit():
                                lat_str = first_part[:split_pos]
                                lon1_str = first_part[split_pos:]
                                parts_split = [lat_str, lon1_str] + initial_parts[1:]
                                break
                        
                        if not parts_split:
                            # Fallback: assume first number is lat, rest is lon1
                            decimal_positions = [i for i, c in enumerate(first_part) if c == '.']
                            if len(decimal_positions) >= 2:
                                # Find position after first decimal+digits where next number starts
                                for pos in range(decimal_positions[0]+1, len(first_part)):
                                    if first_part[pos] == '-' or (first_part[pos].isdigit() and 
                                                                 pos > 0 and not first_part[pos-1].isdigit() and 
                                                                 first_part[pos-1] != '.'):
                                        lat_str = first_part[:pos]
                                        lon1_str = first_part[pos:]
                                        parts_split = [lat_str, lon1_str] + initial_parts[1:]
                                        break
                else:
                    # No concatenation, use as-is
                    parts_split = initial_parts
                
                if len(parts_split) < 5:
                    print(f"Warning: Invalid grid specification at line {i}: {content}")
                    continue
                    
                latitude = float(parts_split[0])
                lon1_check = float(parts_split[1])
                lon2_check = float(parts_split[2])
                dlon_check = float(parts_split[3])
                current_height = float(parts_split[4])
                
                # For 2D maps, use height from grid line or default
                height_to_use = current_height if header.map_dimension == 3 else default_height
                
                # Read the data values for this latitude row
                grid_values = _read_grid_row_values(lines, i, nlons)
                i += len(grid_values) // nlons + (1 if len(grid_values) % nlons != 0 else 0)
                
                # Process the values and create records
                if current_map_type and current_epoch:
                    _process_grid_values(
                        data_records, grid_values, current_map_type, current_epoch,
                        latitude, height_to_use, lons, current_exponent
                    )
                        
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing grid data at line {i}: {e}")
                continue
    
    return data_records


def _extract_label_and_content(line: str) -> typing.Tuple[str, str]:
    """
    Extract label and content from an IONEX line.
    
    Args:
        line: IONEX file line
        
    Returns:
        Tuple of (label, content)
    """
    # IONEX labels typically start around position 59-60
    if len(line) >= 59:
        # Try position 59 first, then 60
        label = line[59:].strip()
        content = line[:59].strip()
        
        if not label and len(line) >= 60:
            label = line[60:].strip()
            content = line[:60].strip()
    else:
        label = ""
        content = line.strip()
    
    return label, content


def _read_grid_row_values(lines: typing.List[str], start_i: int, expected_count: int) -> typing.List[int]:
    """
    Read grid values from data lines using I5 fixed-width format.
    
    Args:
        lines: All file lines
        start_i: Starting line index
        expected_count: Expected number of values to read
        
    Returns:
        List of integer values from the grid
    """
    values = []
    i = start_i
    
    while len(values) < expected_count and i < len(lines):
        line = lines[i].rstrip()
        i += 1
        
        # Check if this is a data line (no recognizable label)
        label, _ = _extract_label_and_content(line)
        known_labels = ['LAT/LON1/LON2/DLON/H', 'END OF', 'START OF', 'EPOCH OF', 'EXPONENT']
        if any(known_label in label for known_label in known_labels):
            # Hit next section, stop reading values
            break
        
        # Parse I5 format: 5-character fixed-width integers
        line_values = _parse_i5_format(line)
        values.extend(line_values)
        
        # Stop if we have enough values
        if len(values) >= expected_count:
            break
    
    return values[:expected_count]  # Trim to expected count


def _parse_i5_format(line: str) -> typing.List[int]:
    """
    Parse a line with I5 format (5-character fixed-width integers).
    
    Args:
        line: Data line from IONEX file
        
    Returns:
        List of integer values
    """
    values = []
    # Parse in 5-character chunks
    for j in range(0, len(line), 5):
        value_str = line[j:j+5].strip()
        if value_str:
            try:
                # Handle undefined values (typically 9999 or 99999)
                value = int(value_str)
                values.append(value)
            except ValueError:
                # Skip invalid values
                continue
    return values


def _process_grid_values(data_records: typing.List[typing.Dict], values: typing.List[int],
                        map_type: str, epoch: datetime.datetime, latitude: float,
                        height: float, longitudes: np.ndarray, exponent: int) -> None:
    """
    Process grid values and add them to data records.
    
    Args:
        data_records: List to append records to
        values: Raw integer values from grid
        map_type: Type of map (TEC, RMS, HEIGHT)
        epoch: Time of observation
        latitude: Latitude of grid row
        height: Height in km
        longitudes: Array of longitude values
        exponent: Scaling exponent
    """
    # Apply scaling factor
    scale_factor = 10 ** exponent
    
    for lon_idx, raw_value in enumerate(values):
        if lon_idx >= len(longitudes):
            break
            
        # Convert undefined values to NaN
        # Common undefined values in IONEX: 999, 9999, 99999
        # Also handle any values that are clearly out of reasonable range
        if raw_value >= 999:  # Any value >= 999 is considered undefined
            scaled_value = np.nan
        else:
            scaled_value = raw_value * scale_factor
        
        data_records.append({
            'Type': map_type,
            'Epoch': epoch,
            'Height': height,
            'Latitude': latitude,
            'Longitude': longitudes[lon_idx],
            'Value': float(scaled_value)
        })


def read_ionex_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read IONEX file and return ionosphere map data as a flattened DataFrame.
    
    This function reads IONEX files and extracts ionosphere map data such as 
    Total Electron Content (TEC) grids, RMS error maps, and optional height maps.
    
    Args:
        file_path: Path to the IONEX file
        
    Returns:
        DataFrame with columns: ['Epoch', 'Latitude(deg)', 'Longitude(deg)', 
                               'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)']
        For missing map types, values are filled with NaN.
        
    Raises:
        ValueError: If file cannot be read or parsed
        FileNotFoundError: If file does not exist
    """
    try:
        # Read file in text mode (ASCII format)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse header to extract metadata and grid parameters
        header, data_start = parse_header(lines)
        
        # Extract grid coordinate arrays from header
        lats, lons, hgts = extract_grid_params(header)
        
        # Parse data maps from the file
        data_list = parse_data_maps(lines, data_start, header, lats, lons, hgts)
        
        # Convert to DataFrame for easier manipulation
        if not data_list:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 
                                       'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)'])
        
        df = pd.DataFrame(data_list)
        
        # Transform data to the expected flattened format
        result_df = _create_flattened_dataframe(df)
        
        return result_df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"IONEX file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading IONEX file {file_path}: {e}")


def _create_flattened_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create flattened DataFrame with expected column format.
    
    Args:
        df: DataFrame with parsed IONEX data
        
    Returns:
        DataFrame with columns: ['Epoch', 'Latitude(deg)', 'Longitude(deg)', 
                               'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)']
    """
    if df.empty:
        return pd.DataFrame(columns=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 
                                   'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)'])
    
    # Create base DataFrame with coordinates and time
    # Get all unique combinations of Epoch, Latitude, Longitude, Height
    coords = df[['Epoch', 'Latitude', 'Longitude', 'Height']].drop_duplicates()
    
    # Initialize result DataFrame
    result = coords.copy()
    result = result.rename(columns={
        'Latitude': 'Latitude(deg)',
        'Longitude': 'Longitude(deg)', 
        'Height': 'Height(km)'
    })
    
    # Add TEC values
    tec_data = df[df['Type'] == 'TEC'][['Epoch', 'Latitude', 'Longitude', 'Height', 'Value']]
    if not tec_data.empty:
        result = result.merge(
            tec_data.rename(columns={'Value': 'TEC(0.1TECU)'}),
            left_on=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'],
            right_on=['Epoch', 'Latitude', 'Longitude', 'Height'],
            how='left'
        ).drop(columns=['Latitude', 'Longitude', 'Height'], errors='ignore')
    else:
        result['TEC(0.1TECU)'] = np.nan
    
    # Add RMS values  
    rms_data = df[df['Type'] == 'RMS'][['Epoch', 'Latitude', 'Longitude', 'Height', 'Value']]
    if not rms_data.empty:
        result = result.merge(
            rms_data.rename(columns={'Value': 'RMS(0.1TECU)'}),
            left_on=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'],
            right_on=['Epoch', 'Latitude', 'Longitude', 'Height'],
            how='left'
        ).drop(columns=['Latitude', 'Longitude', 'Height'], errors='ignore')
    else:
        result['RMS(0.1TECU)'] = np.nan
    
    # Ensure all expected columns exist
    expected_columns = ['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)']
    for col in expected_columns:
        if col not in result.columns:
            result[col] = np.nan
    
    # Reorder columns and sort by time and coordinates
    result = result[expected_columns]
    result = result.sort_values(['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'])
    result = result.reset_index(drop=True)
    
    return result