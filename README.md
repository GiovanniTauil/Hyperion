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
- **RINEX Observation Reader**: Parse RINEX observation files (versions 2.10, 2.11, 3.01, 3.04, 3.05, 4.00) into pandas DataFrames

## Usage

```python
import hyperion

# Read SP3-D file
sp3_df = hyperion.read_sp3_to_dataframe('path/to/sp3_file.sp3')

# Read YUMA almanac file  
yuma_df = hyperion.read_yuma_to_dataframe('path/to/almanac.alm')

# Read RINEX navigation file (supports multiple versions)
rinex_nav_df = hyperion.read_rinex_nav_to_dataframe('path/to/navigation.nav')

# Read RINEX observation file (supports multiple versions)
rinex_obs_df = hyperion.read_rinex_obs_to_dataframe('path/to/observation.obs')
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
