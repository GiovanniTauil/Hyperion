# Hyperion

(EN) *Hyperion refers to the Greek titan linked to celestial motion, symbolizing the precision and transformation of orbits.*

(DE) *Hyperion bezieht sich auf den griechischen Titanen des Himmelslaufs und symbolisiert die Präzision und Transformation von Umlaufbahnen.*

(PT-BR) *Hyperion refere-se ao titã da mitologia grega associado ao movimento celestial, simbolizando a precisão e transformação de órbitas.*

(ES) *Hyperion alude al titán de la mitología griega asociado al movimiento celestial, simbolizando la precisión y transformación de órbitas.*

Hyperion is a Python-based open-source toolkit for GNSS satellite orbit processing, released under the GPLv3 license. It is designed to facilitate the transformation and analysis of GNSS orbit data—focusing on the conversion of SP3-D and YUMA almanac files, as well as orbital state computations and derived parameters.

Inspired by the structure and capabilities of GROOPS (Gravity Recovery Object Oriented Programming System) by Mayer-Gürr et al. (2021), Hyperion reimplements key functionalities for satellite orbit determination and gravity-related modeling in a modular and extensible Python environment. While GROOPS is written in C++ and targets advanced geodetic applications including gravity field recovery and precise GNSS data processing, Hyperion adapts and extends these principles to support modern Python-based geodetic workflows—focusing on usability, interoperability, and integration with research pipelines.

This software is being developed with necessary adaptations to meet the specific requirements of satellite orbit and clock correction workflows within GNSS PPP/PPP-RTK contexts, and will remain fully GPLv3-compliant.

## Features

- **SP3-D File Reader**: Parse SP3-D (Standard Product 3 - version D) files into pandas DataFrames
- **YUMA Almanac Reader**: Parse YUMA almanac files into pandas DataFrames with orbital parameters

## Usage

```python
import hyperion

# Read SP3-D file
sp3_df = hyperion.read_sp3_to_dataframe('path/to/sp3_file.sp3')

# Read YUMA almanac file  
yuma_df = hyperion.read_yuma_to_dataframe('path/to/almanac.alm')
```
