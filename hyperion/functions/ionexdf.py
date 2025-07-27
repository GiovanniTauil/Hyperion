import dataclasses
import datetime
import typing
import pandas as pd
import numpy as np


@dataclasses.dataclass
class IonexHeader:
    """
    Represents IONEX file header information.
    
    This dataclass stores all the metadata from the IONEX header including
    grid parameters, time ranges, and processing information.
    """
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
    map_dimension: typing.Optional[int] = None  # 2 for 2D maps, 3 for 3D maps
    lat1: typing.Optional[float] = None  # Starting latitude
    lat2: typing.Optional[float] = None  # Ending latitude
    dlat: typing.Optional[float] = None  # Latitude increment
    lon1: typing.Optional[float] = None  # Starting longitude
    lon2: typing.Optional[float] = None  # Ending longitude
    dlon: typing.Optional[float] = None  # Longitude increment
    hgt1: typing.Optional[float] = None  # Starting height (km)
    hgt2: typing.Optional[float] = None  # Ending height (km)
    dhgt: typing.Optional[float] = None  # Height increment (km)
    exponent: typing.Optional[int] = None  # Scaling exponent for values
    comment: typing.List[str] = dataclasses.field(default_factory=list)
    auxiliary_data: typing.Dict[str, typing.List[str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class IonexData:
    """Container for IONEX data (kept for backward compatibility)."""
    header: IonexHeader
    data: pd.DataFrame


def parse_header(lines: typing.List[str]) -> typing.Tuple[IonexHeader, int]:
    """
    Parse the header lines into an IonexHeader object.
    
    This function extracts metadata from the IONEX header including grid parameters,
    time information, and processing details. The IONEX format uses fixed positions
    for labels (typically starting at column 59-60).
    
    Args:
        lines: List of lines from the IONEX file
        
    Returns:
        Tuple of (IonexHeader object, index of first line after header)
    """
    header = IonexHeader()
    i = 0
    
    # Process each line until we hit "END OF HEADER"
    while i < len(lines):
        line = lines[i].rstrip()
        if 'END OF HEADER' in line:
            break
        
        # IONEX format: labels typically start around position 59-60
        # Extract content and label parts more flexibly to handle variations
        if len(line) >= 59:
            # Try position 59 first (common case), then 60 as fallback
            label = line[59:].strip()
            content = line[:59].strip()
            
            # If label is empty, try position 60
            if not label and len(line) >= 60:
                label = line[60:].strip()
                content = line[:60].strip()
        else:
            # Handle short lines that don't reach the label position
            label = ""
            content = line.strip()
        
        # Process recognized header fields
        if label:
            try:
                # Handle specific header fields with proper error handling
                if label == 'IONEX VERSION / TYPE':
                    # Parse version and file type information
                    parts = content.split()
                    if len(parts) >= 1:
                        header.version = parts[0]
                    if len(parts) >= 2:
                        header.file_type = parts[1]
                elif label == 'PGM / RUN BY / DATE':
                    # Parse program, run by, and date information
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
                    # Parse datetime from space-separated integers: YYYY MM DD HH MM SS
                    parts = list(map(int, content.split()))
                    if len(parts) >= 6:
                        header.epoch_first_map = datetime.datetime(*parts)
                elif label == 'EPOCH OF LAST MAP':
                    # Parse datetime from space-separated integers: YYYY MM DD HH MM SS
                    parts = list(map(int, content.split()))
                    if len(parts) >= 6:
                        header.epoch_last_map = datetime.datetime(*parts)
                elif label == 'INTERVAL':
                    # Time interval between maps in seconds
                    header.interval = float(content)
                elif label == 'MAPS IN FILE':
                    # Total number of maps in the file
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
                    # Earth radius used for calculations (km)
                    header.base_radius = float(content)
                elif label == 'MAP DIMENSION':
                    # 2 for 2D maps (single height), 3 for 3D maps (multiple heights)
                    header.map_dimension = int(content)
                elif 'HGT1 / HGT2 / DHGT' in label:
                    # Parse height grid parameters: start, end, increment (km)
                    parts = content.split()
                    if len(parts) >= 3:
                        header.hgt1, header.hgt2, header.dhgt = map(float, parts[:3])
                elif 'LAT1 / LAT2 / DLAT' in label:
                    # Parse latitude grid parameters: start, end, increment (degrees)
                    parts = content.split()
                    if len(parts) >= 3:
                        header.lat1, header.lat2, header.dlat = map(float, parts[:3])
                elif 'LON1 / LON2 / DLON' in label:
                    # Parse longitude grid parameters: start, end, increment (degrees)
                    parts = content.split()
                    if len(parts) >= 3:
                        header.lon1, header.lon2, header.dlon = map(float, parts[:3])
                elif label == 'EXPONENT':
                    # Default scaling exponent (power of 10) for map values
                    header.exponent = int(content)
                elif label == 'COMMENT':
                    # Collect comment lines
                    header.comment.append(content)
                else:
                    # Handle auxiliary data (like DCBs - Differential Code Biases)
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
    Extract grid parameters from header and generate coordinate arrays.
    
    This function validates that all required grid parameters are present
    and creates numpy arrays for the latitude, longitude, and height coordinates
    based on the start, end, and increment values from the header.
    
    Args:
        header: IonexHeader object with grid parameters
        
    Returns:
        Tuple of (latitudes, longitudes, heights) arrays
        
    Raises:
        ValueError: If any required grid parameters are missing
    """
    # Validate that all required grid parameters are present
    if None in (header.lat1, header.lat2, header.dlat):
        raise ValueError("Missing latitude grid parameters")
    if None in (header.lon1, header.lon2, header.dlon):
        raise ValueError("Missing longitude grid parameters")
    if None in (header.hgt1, header.hgt2, header.dhgt):
        raise ValueError("Missing height grid parameters")
    if header.map_dimension is None:
        raise ValueError("Missing map dimension")
    
    # Generate coordinate arrays (inclusive of endpoints)
    # Handle both positive and negative increments
    sign_lat = 1 if header.dlat > 0 else -1
    lats = np.arange(header.lat1, header.lat2 + sign_lat * (header.dlat / 2), header.dlat)
    
    sign_lon = 1 if header.dlon > 0 else -1
    lons = np.arange(header.lon1, header.lon2 + sign_lon * (header.dlon / 2), header.dlon)
    
    # For 2D maps, use only the first height; for 3D maps, generate height array
    if header.map_dimension == 2:
        hgts = np.array([header.hgt1])  # Single height layer
    else:
        sign_h = 1 if header.dhgt > 0 else -1
        hgts = np.arange(header.hgt1, header.hgt2 + sign_h * (header.dhgt / 2), header.dhgt)
    
    return lats, lons, hgts


def parse_data_maps(lines: typing.List[str], start_index: int, header: IonexHeader, 
                   lats: np.ndarray, lons: np.ndarray, hgts: np.ndarray) -> typing.List[typing.Dict]:
    """
    Parse the data maps into a list of records with improved error handling.
    
    This function processes the IONEX data section, extracting TEC, RMS, and HEIGHT
    maps. Each map is organized by epochs with grid data in fixed-width I5 format.
    The function handles multiple map types and converts raw integer values to
    properly scaled floating-point values.
    
    Args:
        lines: List of lines from the IONEX file
        start_index: Index to start parsing from (after header)
        header: IonexHeader object with metadata
        lats: Latitude array from grid parameters
        lons: Longitude array from grid parameters
        hgts: Height array from grid parameters
        
    Returns:
        List of data records with flattened grid data
    """
    data_records = []
    i = start_index
    current_map_type = None  # Current map being processed: TEC, RMS, HEIGHT
    current_epoch = None     # Current time for the map
    current_exponent = header.exponent if header.exponent is not None else 0  # Scaling factor
    current_height = None    # Current height being processed
    nlons = len(lons)        # Number of longitude points expected per row
    
    # For 2D maps, use the first height from header as default
    default_height = hgts[0] if len(hgts) > 0 else 0.0
    
    # Process each line in the data section
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        
        # Skip blank lines (common in IONEX files)
        if not line.strip():
            continue
        
        # Check for end of file marker
        if 'END OF FILE' in line:
            break
        
        # Parse label and content from the line
        label, content = _extract_label_and_content(line)
        
        # Handle map start markers (e.g., "START OF TEC MAP")
        if 'START OF' in label and 'MAP' in label:
            # Extract map type (TEC, RMS, HEIGHT) from label
            parts = label.split()
            if len(parts) >= 3:
                current_map_type = parts[2]  # TEC, RMS, HEIGHT
            # Reset exponent to header default for new map
            current_exponent = header.exponent if header.exponent is not None else 0
            continue
        
        # Handle map end markers (e.g., "END OF TEC MAP")
        if 'END OF' in label and 'MAP' in label:
            current_map_type = None
            continue
        
        # Handle epoch information for current map
        if 'EPOCH OF CURRENT MAP' in label:
            try:
                # Parse datetime from space-separated integers: YYYY MM DD HH MM SS
                parts = list(map(int, content.split()))
                if len(parts) >= 6:
                    current_epoch = datetime.datetime(*parts)
            except (ValueError, TypeError) as e:
                print(f"Warning: Error parsing epoch at line {i}: {e}")
            continue
        
        # Handle exponent (overrides header default for this specific map)
        if label == 'EXPONENT':
            try:
                current_exponent = int(content)
            except (ValueError, TypeError):
                current_exponent = 0  # Fallback to no scaling
            continue
        
        # Handle grid data lines (format: LAT/LON1/LON2/DLON/H)
        if 'LAT/LON1/LON2/DLON/H' in label:
            try:
                # Parse latitude and height from the grid specification line
                # Format can be like: "87.5-180.0 180.0   5.0 350.0" 
                # Need to handle cases where lat and lon1 are concatenated without space
                
                # Split on spaces first to get potential parts
                initial_parts = content.split()
                if len(initial_parts) < 4:
                    print(f"Warning: Invalid grid specification at line {i}: {content}")
                    continue
                
                # Check if first part contains latitude and lon1 concatenated
                first_part = initial_parts[0]
                if '-' in first_part and first_part.count('-') >= 1:
                    # Handle concatenated coordinates like "87.5-180.0"
                    # Find the last '-' which likely separates lat from lon1
                    # This is tricky because coordinates can be negative
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
                        # Look for decimal points as indicators of separate numbers
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
                    # No concatenation, use parts as-is
                    parts_split = initial_parts
                
                # Validate we have all required parts
                if len(parts_split) < 5:
                    print(f"Warning: Invalid grid specification at line {i}: {content}")
                    continue
                    
                # Extract values: latitude, lon1, lon2, dlon, height
                latitude = float(parts_split[0])
                lon1_check = float(parts_split[1])  # For validation (not used)
                lon2_check = float(parts_split[2])  # For validation (not used)
                dlon_check = float(parts_split[3])  # For validation (not used)
                current_height = float(parts_split[4])
                
                # Determine height to use: current height for 3D, default for 2D
                height_to_use = current_height if header.map_dimension == 3 else default_height
                
                # Read the data values for this latitude row
                grid_values = _read_grid_row_values(lines, i, nlons)
                # Advance the line index past the data we just read
                i += len(grid_values) // nlons + (1 if len(grid_values) % nlons != 0 else 0)
                
                # Process the values and create records if we have valid context
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
    
    IONEX files use fixed positioning where data content comes first,
    followed by the label starting around position 59-60. This function
    handles the parsing flexibly to accommodate slight variations.
    
    Args:
        line: IONEX file line
        
    Returns:
        Tuple of (label, content)
    """
    # IONEX labels typically start around position 59-60
    if len(line) >= 59:
        # Try position 59 first (most common), then 60 as fallback
        label = line[59:].strip()
        content = line[:59].strip()
        
        if not label and len(line) >= 60:
            label = line[60:].strip()
            content = line[:60].strip()
    else:
        # Handle lines shorter than expected
        label = ""
        content = line.strip()
    
    return label, content


def _read_grid_row_values(lines: typing.List[str], start_i: int, expected_count: int) -> typing.List[int]:
    """
    Read grid values from data lines using I5 fixed-width format.
    
    IONEX data is stored in I5 format (5-character fixed-width integers).
    This function reads consecutive data lines until it has collected
    the expected number of values or encounters a non-data line.
    
    Args:
        lines: All file lines
        start_i: Starting line index for reading data
        expected_count: Expected number of values to read (longitude count)
        
    Returns:
        List of integer values from the grid
    """
    values = []
    i = start_i
    
    # Continue reading until we have enough values or hit a non-data line
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
    
    The IONEX format uses I5 (5-character integer) format for data values.
    Each value occupies exactly 5 characters, so a line like:
    "  100  200  300" contains three values: 100, 200, 300
    
    Args:
        line: Data line from IONEX file
        
    Returns:
        List of integer values extracted from the line
    """
    values = []
    # Parse in 5-character chunks
    for j in range(0, len(line), 5):
        value_str = line[j:j+5].strip()
        if value_str:
            try:
                # Convert to integer, handling potential undefined values
                value = int(value_str)
                values.append(value)
            except ValueError:
                # Skip invalid values (non-numeric content)
                continue
    return values


def _process_grid_values(data_records: typing.List[typing.Dict], values: typing.List[int],
                        map_type: str, epoch: datetime.datetime, latitude: float,
                        height: float, longitudes: np.ndarray, exponent: int) -> None:
    """
    Process grid values and add them to data records.
    
    This function takes raw integer values from the IONEX grid and converts
    them to properly scaled floating-point values. It also handles undefined
    values by converting them to NaN.
    
    Args:
        data_records: List to append records to
        values: Raw integer values from grid
        map_type: Type of map (TEC, RMS, HEIGHT)
        epoch: Time of observation
        latitude: Latitude of grid row (degrees)
        height: Height in km
        longitudes: Array of longitude values (degrees)
        exponent: Scaling exponent (power of 10)
    """
    # Apply scaling factor based on exponent
    # For example, if exponent = -1, scale_factor = 0.1
    scale_factor = 10 ** exponent
    
    # Process each value and create a data record
    for lon_idx, raw_value in enumerate(values):
        if lon_idx >= len(longitudes):
            break  # Don't exceed available longitude points
            
        # Convert undefined values to NaN
        # Common undefined values in IONEX: 999, 9999, 99999
        # Any value >= 999 is considered undefined/missing data
        if raw_value >= 999:  # Any value >= 999 is considered undefined
            scaled_value = np.nan
        else:
            # Apply scaling to get physical units
            scaled_value = raw_value * scale_factor
        
        # Create a record for this grid point
        data_records.append({
            'Type': map_type,           # TEC, RMS, or HEIGHT
            'Epoch': epoch,             # Time of observation
            'Height': height,           # Height in km
            'Latitude': latitude,       # Latitude in degrees
            'Longitude': longitudes[lon_idx],  # Longitude in degrees
            'Value': float(scaled_value)       # Scaled value (NaN for undefined)
        })


def read_ionex_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read IONEX file and return ionosphere map data as a flattened DataFrame.
    
    This function reads IONEX files and extracts ionosphere map data such as 
    Total Electron Content (TEC) grids, RMS error maps, and optional height maps.
    The output is a flattened DataFrame where each row represents one grid point
    with its coordinates and values for different map types.
    
    Key features:
    - Handles both 2D (single height) and 3D (multiple height) ionospheric maps
    - Properly scales raw integer values using exponent factors
    - Converts undefined values (≥999) to NaN
    - Supports multiple map types: TEC, RMS, HEIGHT
    - Returns data in the requested format with specific column names
    
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
        # Read file in text mode (ASCII format as specified)
        # Use utf-8 with error handling for robustness
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse header to extract metadata and grid parameters
        header, data_start = parse_header(lines)
        
        # Extract grid coordinate arrays from header
        lats, lons, hgts = extract_grid_params(header)
        
        # Parse data maps from the file
        data_list = parse_data_maps(lines, data_start, header, lats, lons, hgts)
        
        # Handle case where no data was found
        if not data_list:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 
                                       'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)'])
        
        # Convert parsed data to DataFrame
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
    
    This function transforms the parsed IONEX data into the specific format
    requested in the requirements. It combines TEC and RMS values for each
    grid point and ensures all expected columns are present.
    
    Args:
        df: DataFrame with parsed IONEX data (columns: Type, Epoch, Height, Latitude, Longitude, Value)
        
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
    
    # Initialize result DataFrame with renamed columns
    result = coords.copy()
    result = result.rename(columns={
        'Latitude': 'Latitude(deg)',
        'Longitude': 'Longitude(deg)', 
        'Height': 'Height(km)'
    })
    
    # Add TEC values by merging TEC data
    tec_data = df[df['Type'] == 'TEC'][['Epoch', 'Latitude', 'Longitude', 'Height', 'Value']]
    if not tec_data.empty:
        result = result.merge(
            tec_data.rename(columns={'Value': 'TEC(0.1TECU)'}),
            left_on=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'],
            right_on=['Epoch', 'Latitude', 'Longitude', 'Height'],
            how='left'
        ).drop(columns=['Latitude', 'Longitude', 'Height'], errors='ignore')
    else:
        # No TEC data found, fill with NaN
        result['TEC(0.1TECU)'] = np.nan
    
    # Add RMS values by merging RMS data
    rms_data = df[df['Type'] == 'RMS'][['Epoch', 'Latitude', 'Longitude', 'Height', 'Value']]
    if not rms_data.empty:
        result = result.merge(
            rms_data.rename(columns={'Value': 'RMS(0.1TECU)'}),
            left_on=['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'],
            right_on=['Epoch', 'Latitude', 'Longitude', 'Height'],
            how='left'
        ).drop(columns=['Latitude', 'Longitude', 'Height'], errors='ignore')
    else:
        # No RMS data found, fill with NaN
        result['RMS(0.1TECU)'] = np.nan
    
    # Ensure all expected columns exist (fill missing with NaN)
    expected_columns = ['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)', 'TEC(0.1TECU)', 'RMS(0.1TECU)']
    for col in expected_columns:
        if col not in result.columns:
            result[col] = np.nan
    
    # Reorder columns to match expected format and sort data logically
    result = result[expected_columns]
    result = result.sort_values(['Epoch', 'Latitude(deg)', 'Longitude(deg)', 'Height(km)'])
    result = result.reset_index(drop=True)
    
    return result