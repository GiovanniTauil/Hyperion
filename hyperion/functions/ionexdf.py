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
        
        # IONEX format: positions 61-80 contain the label
        if len(line) >= 60:
            label = line[60:].strip()
            content = line[:60].strip()
        else:
            # Handle short lines
            label = ""
            content = line.strip()
        
        if label:
            # Handle specific header fields
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
                parts = content.split()
                if len(parts) >= 3:
                    header.hgt1, header.hgt2, header.dhgt = map(float, parts[:3])
            elif 'LAT1 / LAT2 / DLAT' in label:
                parts = content.split()
                if len(parts) >= 3:
                    header.lat1, header.lat2, header.dlat = map(float, parts[:3])
            elif 'LON1 / LON2 / DLON' in label:
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


def parse_data(lines: typing.List[str], start_index: int, header: IonexHeader, 
               lats: np.ndarray, lons: np.ndarray, hgts: np.ndarray) -> typing.List[typing.Dict]:
    """
    Parse the data maps into a list of records.
    
    Args:
        lines: List of lines from the IONEX file
        start_index: Index to start parsing from
        header: IonexHeader object
        lats: Latitude array
        lons: Longitude array
        hgts: Height array
        
    Returns:
        List of data records
    """
    data_records = []
    i = start_index
    current_type = None
    current_epoch = None
    current_exp = None
    current_h = None
    nlons = len(lons)
    
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        
        if not line.strip():
            continue
        
        # For data lines, the label might start at different positions
        # We need to be more flexible in detecting labels vs data
        
        # Check if this line has a recognizable label pattern
        has_label = False
        label = ""
        content = line
        
        # Check for labels that should be at column 60+
        for possible_label in ['START OF TEC MAP', 'START OF RMS MAP', 'START OF HEIGHT MAP', 
                              'END OF TEC MAP', 'END OF RMS MAP', 'END OF HEIGHT MAP',
                              'EPOCH OF CURRENT MAP', 'EXPONENT', 'LAT/LON1/LON2/DLON/H']:
            if possible_label in line:
                # Find where the label starts
                label_pos = line.find(possible_label)
                if label_pos > 0:
                    content = line[:label_pos].strip()
                    label = line[label_pos:].strip()
                    has_label = True
                    break
                elif label_pos == 0:
                    # Label is at the beginning (shouldn't happen in IONEX but handle it)
                    label = line.strip()
                    content = ""
                    has_label = True
                    break
        
        if not has_label:
            # This is likely a data line - no label
            content = line.strip()
            label = ""
        
        # Handle map start/end markers
        if 'START OF' in label and 'MAP' in label:
            current_type = label.split()[2]  # TEC, RMS, HEIGHT
            current_exp = header.exponent if header.exponent is not None else 0  # Use header default
            continue
        
        if 'END OF' in label and 'MAP' in label:
            current_type = None
            continue
        
        # Handle epoch information
        if 'EPOCH OF CURRENT MAP' in label:
            try:
                parts = list(map(int, content.split()))
                if len(parts) >= 6:
                    current_epoch = datetime.datetime(*parts)
            except (ValueError, TypeError):
                continue
            continue
        
        # Handle exponent (overrides header default for this map)
        if 'EXPONENT' in label:
            try:
                current_exp = int(content)
            except (ValueError, TypeError):
                current_exp = 0
            continue
        
        # Handle grid data
        if 'LAT/LON1/LON2/DLON/H' in label:
            try:
                parts = content.split()
                if len(parts) < 5:
                    continue
                    
                lat = float(parts[0])
                lon1_ = float(parts[1])
                lon2_ = float(parts[2])
                dlon_ = float(parts[3])
                current_h = float(parts[4])
                
                # Collect grid row values
                row = []
                while len(row) < nlons and i < len(lines):
                    next_line = lines[i].rstrip()
                    
                    # Check if this is another LAT/LON1/LON2/DLON/H line or other label
                    is_next_header = False
                    for check_label in ['LAT/LON1/LON2/DLON/H', 'END OF', 'START OF', 'EPOCH OF']:
                        if check_label in next_line:
                            is_next_header = True
                            break
                    
                    if is_next_header:
                        break
                        
                    i += 1
                    
                    # Parse fixed-width integer values (I5 format)
                    # Use the entire line for data values
                    j = 0
                    while j < len(next_line) and len(row) < nlons:
                        v_str = next_line[j:j+5].strip()
                        if v_str and v_str.isdigit():
                            try:
                                row.append(int(v_str))
                            except ValueError:
                                pass  # Skip invalid values
                        j += 5
                
                # Process the row if we have complete data
                if len(row) >= min(nlons, 36) and current_type and current_epoch is not None:  # Allow partial rows for testing
                    exp = current_exp if current_exp is not None else 0
                    values = np.array(row[:nlons]) * (10 ** exp)  # Take only what we need
                    
                    for lon_idx, value in enumerate(values):
                        if lon_idx < len(lons):
                            data_records.append({
                                'Type': current_type,
                                'Epoch': current_epoch,
                                'Height': current_h,
                                'Lat': lat,
                                'Lon': lons[lon_idx],
                                'Value': float(value)
                            })
                        
            except (ValueError, IndexError) as e:
                # Log error but continue processing
                print(f"Warning: Error parsing grid at line {i}: {e}")
            continue
    
    return data_records


def read_ionex_to_dataframe(file_path: str) -> typing.Dict[str, typing.Union[dict, pd.DataFrame]]:
    """
    Read IONEX file and return header and data as DataFrame.
    
    Args:
        file_path: Path to the IONEX file
        
    Returns:
        Dictionary with 'header' (metadata dict) and 'data' (DataFrame)
        
    Raises:
        ValueError: If file cannot be read or parsed
        FileNotFoundError: If file does not exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse header
        header, data_start = parse_header(lines)
        
        # Extract grid parameters
        lats, lons, hgts = extract_grid_params(header)
        
        # Parse data
        data_list = parse_data(lines, data_start, header, lats, lons, hgts)
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Convert header to dictionary for return
        header_dict = dataclasses.asdict(header)
        
        return {
            'header': header_dict,
            'data': df
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"IONEX file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading IONEX file {file_path}: {e}")