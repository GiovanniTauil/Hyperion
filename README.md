<div align="center">
  <img src="hyperion-03.png" alt="Logo" width="1023">
</div>

# Hyperion - v1.0.5

(EN) *Hyperion refers to the Greek titan linked to celestial motion, symbolizing the precision and transformation of orbits.*

(DE) *Hyperion bezieht sich auf den griechischen Titanen des Himmelslaufs und symbolisiert die Präzision und Transformation von Umlaufbahnen.*

(PT-BR) *Hyperion refere-se ao titã da mitologia grega associado ao movimento celestial, simbolizando a precisão e transformação de órbitas.*

(ES) *Hyperion alude al titán de la mitología griega asociado al movimiento celestial, simbolizando la precisión y transformación de órbitas.*

Hyperion is an open-source Python library designed for comprehensive GNSS (Global Navigation Satellite Systems) data processing, analysis, and interoperability. Licensed under the MIT License, it provides a modular environment for handling complex geodetic workflows, ranging from raw data parsing to high-precision orbital computations.

While initially inspired by the architectural principles of GROOPS (Mayer-Gürr et al., 2021), Hyperion has evolved into a versatile and independent toolkit. It bridges the gap between low-level orbital mechanics and modern Python-based research pipelines, offering specialized tools for:

   - Multi-Format I/O: Seamless parsing and conversion of industry-standard formats, including SP3-D, YUMA/SEM almanacs, and navigation messages.

   - Orbital Dynamics: Precise computation of satellite states, clock corrections, and derived orbital parameters.

   - Positioning Context: Tailored for the rigorous demands of PPP (Precise Point Positioning) and PPP-RTK workflows, ensuring high-fidelity orbit and clock integration.

   - Extensibility: Built on a Pythonic foundation (NumPy, Pandas) to ensure ease of use, rapid prototyping, and integration with machine learning or atmospheric research tools.

Hyperion adapts established geodetic principles into a user-friendly, extensible framework, making advanced satellite orbit determination and GNSS processing accessible to the broader scientific community

## Features

- **SP3-D File Reader**: Parse SP3-D (Standard Product 3 - version D) files.
- **YUMA Almanac Reader**: Parse YUMA almanac files.
- **RINEX Navigation Reader**: Parse RINEX navigation files (versions 2.11, 3.01, 3.04, 3.05, 4.00).
- **RINEX Clock Reader**: Parse RINEX Clock files (versions 2.xx to 3.xx).
- **RINEX Observation Reader**: Parse RINEX observation files (versions 2.10, 2.11, 3.01, 3.04, 3.05, 4.00).
- **CRINEX (Compact RINEX)**: Native on-the-fly decompression of `.crx` files via the `hatanaka` library within the observation reader.
- **ATX (Antenna Exchange)**: Parse ANTEX phase center formats.
- **Meteorological Data Reader (MET)**: Parse RINEX MET observations.
- **DORIS RINEX**: Parse DORIS RINEX measurements.
- **ERP & SINEX**: Earth Rotation Parameters and SINEX solution readers.
- **IONEX File Reader**: Parse IONEX (IONosphere Exchange format) files version 1.0 into pandas DataFrames.

**Note**: *All input parsers process `.gz` compressed files completely natively and seamlessly.* Readers yield Pandas DataFrames and can export directly to `.h5` HDF5 files when specified.

## Usage

Hyperion provides unified `load_*` functions for every supported GNSS file type:

- `load_antex(file_path)`: Reads `.atx` Antenna Exchange files
- `load_erp(file_path)`: Reads `.erp` Earth Rotation Parameter files
- `load_ionex(file_path)`: Reads `.ion`, `.09i` IONEX Ionosphere map files
- `load_rinex_clock(file_path)`: Reads `.clk` RINEX Clock files
- `load_rinex_doris(file_path)`: Reads `.dor` DORIS RINEX files
- `load_rinex_met(file_path)`: Reads `.m` RINEX Meteorological files
- `load_rinex_nav(file_path)`: Reads `.n`, `.rnx`, `.nav` RINEX Navigation files
- `load_rinex_obs(file_path)`: Reads `.o`, `.crx`, `.obs` RINEX Observation files (with native CRINEX decompression)
- `load_sinex(file_path)`: Reads `.snx` SINEX solution files (returns a tuple of `df_sites, df_estimates`)
- `load_sp3(file_path)`: Reads `.sp3` SP3-D Orbit files
- `load_yuma_almanac(file_path)`: Reads `.alm` YUMA Almanac files

### Output Formats

Every single parser in the Hyperion library supports two versatile return types via the `return_type` argument:

1. **Pandas DataFrames** (Default)
   `return_type='dataframe'` returns the data loaded directly into rapid-access in-memory Pandas DataFrames.
2. **HDF5 Disk Layouts**
   `return_type='hdf5'` buffers the parsed data and saves it natively as a serialized `.h5` table on disk. This is highly recommended for enormous GNSS observation datasets to prevent memory exhaustion. (Requires the `output_path` argument).

### Example Usage

Here is a simple example demonstrating how to read a compressed, natively compacted RINEX 3.x Observation file (CRINEX `.crx.gz`) directly into a binary HDF5 store:

```python
import hyperion

# Automatically unpack .crx.gz natively and serialize directly to an HDF5 table!
output_file = hyperion.load_rinex_obs(
    'OBS00IGS_R_20240060000_01D_30S_MO.crx.gz',
    return_type='hdf5',
    output_path='observation_data.h5'
)

print(f"Dataset securely saved to: {output_file}")
```

## Latest Updates (v1.0.5)

### Interpolation Methods
The library now supports two interpolation methods for upsampling time-series data:

- **Polynomial Interpolation** (method='polynomial'): Uses Lagrange polynomial interpolation for smoother curves
- **Linear Interpolation** (method='linear'): Simple linear interpolation between points, faster but less smooth

Both methods:
- Maintain NaN values (don't interpolate through missing data)
- Support various time units (seconds, minutes, hours)
- Work with both DataFrame and HDF5 inputs/outputs
- Only allow upsampling (smaller time intervals)
