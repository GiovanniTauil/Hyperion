<div align="center">
  <img src="hyperion-03.png" alt="Logo" width="1023">
</div>

# Hyperion

(EN) *Hyperion refers to the Greek titan linked to celestial motion, symbolizing the precision and transformation of orbits.*

(DE) *Hyperion bezieht sich auf den griechischen Titanen des Himmelslaufs und symbolisiert die Präzision und Transformation von Umlaufbahnen.*

(PT-BR) *Hyperion refere-se ao titã da mitologia grega associado ao movimento celestial, simbolizando a precisão e transformação de órbitas.*

(ES) *Hyperion alude al titán de la mitología griega asociado al movimiento celestial, simbolizando la precisión y transformación de órbitas.*

Hyperion is a Python-based open-source toolkit for GNSS satellite orbit processing, released under the GPLv3 license. It is designed to facilitate the transformation and analysis of GNSS orbit data—focusing on the conversion of SP3-D and YUMA almanac files, as well as orbital state computations and derived parameters.

Inspired by the structure and capabilities of GROOPS (Gravity Recovery Object Oriented Programming System) by Mayer-Gürr et al. (2021), Hyperion reimplements key functionalities for satellite orbit determination and gravity-related modeling in a modular and extensible Python environment. While GROOPS is written in C++ and targets advanced geodetic applications including gravity field recovery and precise GNSS data processing, Hyperion adapts and extends these principles to support modern Python-based geodetic workflows—focusing on usability, interoperability, and integration with research pipelines.

This software is being developed with necessary adaptations to meet the specific requirements of satellite orbit and clock correction workflows within GNSS PPP/PPP-RTK contexts.

## Features

- **SP3-D File Reader**: Parse SP3-D (Standard Product 3 - version D) files into pandas DataFrames
- **YUMA Almanac Reader**: Parse YUMA almanac files into pandas DataFrames with orbital parameters
- **RINEX Navigation Reader**: Parse RINEX navigation files (versions 2.11, 3.01, 3.04, 3.05, 4.00) into pandas DataFrames
- **RINEX Clock Reader**: Parse RINEX Clock files (versions 3.0, 3.04) into pandas DataFrames with satellite and receiver clock data
- **RINEX Observation Reader**: Parse RINEX observation files (versions 2.10, 2.11, 3.01, 3.04, 3.05, 4.00) into pandas DataFrames
- ⚠️(Still in Test)⚠️ **IONEX File Reader**: Parse IONEX (IONosphere Exchange format) files version 1.0 into pandas DataFrames with ionospheric map data

## Usage

```python
import hyperion

# Read SP3-D file
sp3_df = hyperion.read_sp3_to_dataframe('path/to/sp3_file.sp3')

# Read YUMA almanac file  
yuma_df = hyperion.read_yuma_to_dataframe('path/to/almanac.alm')

# Read RINEX navigation file (supports multiple versions)
rinex_nav_df = hyperion.read_rinex_nav_to_dataframe('path/to/navigation.nav')

# Read RINEX Clock file (supports versions 3.0 and 3.04)
rinex_clk_df, clk_header = hyperion.read_rinex_clock_to_dataframe('path/to/clock.clk')

# Read RINEX observation file (supports multiple versions)
rinex_obs_df = hyperion.read_rinex_obs_to_dataframe('path/to/observation.obs')

# Read IONEX file
ionex_result = hyperion.read_ionex_to_dataframe('path/to/ionex_file.ion')
ionex_header = ionex_result['header']  # Metadata dictionary
ionex_data = ionex_result['data']      # DataFrame with TEC/RMS/HEIGHT data
```

### RINEX Navigation File Support

The RINEX navigation reader supports multiple file format versions:

- **RINEX 2.11**: GPS navigation data with 29 orbital parameters
- **RINEX 3.01, 3.04, 3.05**: Multi-constellation navigation (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS, SBAS)  
- **RINEX 4.00**: Latest standard with enhanced multi-constellation support

#### RINEX Navigation Output Columns

For **RINEX 2.x** (GPS):
- `Epoch`: Time of clock correction
- `PRN`: Satellite PRN number  
- `SVclockBias`, `SVclockDrift`, `SVclockDriftRate`: Clock parameters
- `IODE`, `Crs`, `DeltaN`, `M0`, etc.: Orbital parameters

For **RINEX 3.x/4.x** (Multi-constellation):
- `Epoch`: Time of clock correction
- `PRN`: Satellite identifier (e.g., 'G01', 'R07', 'E12')
- `SatSystem`: Satellite system ('G'=GPS, 'R'=GLONASS, 'E'=Galileo, 'C'=BeiDou, etc.)
- System-specific orbital and clock parameters

### RINEX Clock File Support

The RINEX Clock reader supports RINEX Clock file format versions 3.0 and 3.04. RINEX Clock files contain satellite and receiver clock offset data organized by epoch and record type.

#### Supported Record Types

- **AS**: Satellite clock data
- **AR**: Receiver clock data
- **CR**: Receiver clock data (alternative format)
- **DR**: Receiver clock rate data

#### RINEX Clock Output

The function returns a tuple `(DataFrame, header_dict)`:

**DataFrame columns**:
- `Type`: Record type ('AS', 'AR', 'CR', 'DR')
- `ID`: Satellite PRN (e.g., 'G01') or receiver/station identifier (e.g., 'ALGO')
- `Epoch`: Time of clock measurement (datetime object)
- `Values`: List of clock values (bias, rate, acceleration in seconds)

**Header dictionary**: Contains file metadata including version, time system, analysis center, etc.

#### Optional Value Expansion

Use `expand_clock_values()` to expand the 'Values' column into separate columns when all records have the same number of values:

```python
# Basic usage
clk_df, header = hyperion.read_rinex_clock_to_dataframe('clock_file.clk')

# Expand values into separate columns (when uniform)
clk_expanded = hyperion.expand_clock_values(clk_df)
# Results in columns: Clock_Bias(s), Clock_Rate(s/s), Clock_Acceleration(s/s²)
```

### RINEX Observation File Support

The RINEX observation reader supports multiple file format versions:

- **RINEX 2.10, 2.11**: GPS and other GNSS observation data
- **RINEX 3.01, 3.04, 3.05**: Multi-constellation observations with enhanced observation types
- **RINEX 4.00**: Latest standard with improved multi-constellation and multi-frequency support

#### RINEX Observation Output Columns

For all versions:
- `Epoch`: Time of observation
- `PRN`: Satellite identifier (e.g., 'G01', 'R07', 'E12')
- `SatSystem`: Satellite system ('G'=GPS, 'R'=GLONASS, 'E'=Galileo, 'C'=BeiDou, etc.)
- `ObsType`: Observation type (e.g., 'C1', 'L1', 'S1' for v2.x; 'C1C', 'L1C', 'S1C' for v3+)
- `Value`: Observed value (pseudorange in meters, carrier phase in cycles, etc.)
- `LossOfLock`: Loss of lock indicator (0-7)
- `SignalStrength`: Signal strength indicator (1-9)

#### Example Usage

```python
import hyperion

# Read RINEX 2.11 GPS navigation file
gps_nav = hyperion.read_rinex_nav_to_dataframe('brdc0060.24n')
print(f"GPS satellites: {gps_nav['PRN'].unique()}")
print(f"Time range: {gps_nav['Epoch'].min()} to {gps_nav['Epoch'].max()}")

# Read RINEX 3.x multi-constellation file  
mixed_nav = hyperion.read_rinex_nav_to_dataframe('BRDC00IGS_R_20240060000_01D_MN.rnx')
print(f"Satellite systems: {mixed_nav['SatSystem'].unique()}")
print(f"Total satellites: {len(mixed_nav['PRN'].unique())}")

# Filter by satellite system
gps_only = mixed_nav[mixed_nav['SatSystem'] == 'G']
galileo_only = mixed_nav[mixed_nav['SatSystem'] == 'E']

# Read RINEX Clock file
clk_df, clk_header = hyperion.read_rinex_clock_to_dataframe('igs0060.clk')
print(f"Clock record types: {clk_df['Type'].unique()}")
print(f"Clock data for satellites/stations: {clk_df['ID'].unique()}")

# Expand clock values for uniform records
clk_uniform = clk_df[clk_df['Values'].apply(len) == 2]  # Filter records with 2 values
clk_expanded = hyperion.expand_clock_values(clk_uniform)
print(f"Clock bias range: {clk_expanded['Clock_Bias(s)'].min():.2e} to {clk_expanded['Clock_Bias(s)'].max():.2e}")

# Filter by record type
satellite_clocks = clk_df[clk_df['Type'] == 'AS']  # Satellite clocks
receiver_clocks = clk_df[clk_df['Type'] == 'AR']   # Receiver clocks

# Read RINEX 2.11 observation file
obs_2x = hyperion.read_rinex_obs_to_dataframe('obs0060.24o')
print(f"Observation types: {obs_2x['ObsType'].unique()}")
print(f"Satellites observed: {obs_2x['PRN'].unique()}")

# Read RINEX 3.x multi-constellation observation file
obs_3x = hyperion.read_rinex_obs_to_dataframe('OBS00IGS_R_20240060000_01D_30S_MO.rnx')
print(f"Satellite systems: {obs_3x['SatSystem'].unique()}")
print(f"Total observations: {len(obs_3x)}")

# Filter observations by type and satellite system
gps_pseudorange = obs_3x[(obs_3x['SatSystem'] == 'G') & (obs_3x['ObsType'].str.startswith('C'))]
galileo_carrier = obs_3x[(obs_3x['SatSystem'] == 'E') & (obs_3x['ObsType'].str.startswith('L'))]

# Calculate data availability
data_availability = obs_3x.groupby(['PRN', 'ObsType'])['Value'].count().reset_index()
print("Data availability per satellite and observation type:")
print(data_availability.head(10))
```

### IONEX File Support

The IONEX file reader supports IONEX format version 1.0 for ionospheric data:

- **IONEX 1.0**: IONosphere Exchange format with TEC, RMS, and HEIGHT maps
- **2D/3D Maps**: Supports both 2D (single height) and 3D (multiple height layers) ionospheric maps
- **Multi-Map Types**: Handles TEC (Total Electron Content), RMS (Root Mean Square), and HEIGHT maps
- **Grid Data**: Extracts latitude, longitude, and height grid parameters from header
- **Auxiliary Data**: Parses auxiliary data blocks like Differential Code Biases (DCBs)

#### IONEX Output Format

The IONEX reader returns a dictionary with two keys:

- **`header`**: Dictionary containing metadata such as:
  - `version`: IONEX version
  - `map_dimension`: 2 or 3 for 2D/3D maps
  - Grid parameters: `lat1`, `lat2`, `dlat`, `lon1`, `lon2`, `dlon`, `hgt1`, `hgt2`, `dhgt`
  - `exponent`: Default scaling exponent
  - Other metadata: program, description, epochs, etc.

- **`data`**: pandas DataFrame with columns:
  - `Type`: Map type ('TEC', 'RMS', 'HEIGHT')
  - `Epoch`: Time of observation (datetime)
  - `Height`: Height in km
  - `Lat`: Latitude in degrees
  - `Lon`: Longitude in degrees  
  - `Value`: Scaled value (TEC in TECU, RMS in TECU, HEIGHT in km)

#### Example Usage

```python
import hyperion

# Read IONEX file
ionex_result = hyperion.read_ionex_to_dataframe('ionex_file.ion')

# Access header information
header = ionex_result['header']
print(f"Map dimension: {header['map_dimension']}")
print(f"Grid: {header['lat1']}° to {header['lat2']}° (Δ{header['dlat']}°)")
print(f"      {header['lon1']}° to {header['lon2']}° (Δ{header['dlon']}°)")

# Access ionospheric data
data = ionex_result['data']
print(f"Total observations: {len(data)}")
print(f"Map types: {data['Type'].unique()}")
print(f"Time range: {data['Epoch'].min()} to {data['Epoch'].max()}")

# Filter TEC data for specific region
tec_data = data[data['Type'] == 'TEC']
regional_tec = tec_data[(tec_data['Lat'] >= 30) & (tec_data['Lat'] <= 60) & 
                        (tec_data['Lon'] >= -120) & (tec_data['Lon'] <= -60)]

# Calculate statistics
print(f"Regional TEC range: {regional_tec['Value'].min():.2f} to {regional_tec['Value'].max():.2f} TECU")
```
