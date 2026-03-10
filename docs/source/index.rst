.. Hyperion documentation master file

Hyperion: GNSS Data Processing Library
======================================

**Version:** 1.0.5

*Hyperion refers to the Greek titan linked to celestial motion, symbolizing the precision and transformation of orbits.*

Hyperion is an open-source Python library designed for comprehensive GNSS (Global Navigation Satellite Systems) data processing, analysis, and interoperability. It provides a modular environment for handling complex geodetic workflows, ranging from raw data parsing to high-precision orbital computations.

Key Features
------------

- **Multi-Format I/O**: Seamless parsing and conversion of industry-standard formats including SP3-D, YUMA/SEM, ANTEX, IONEX, and various RINEX versions (Observation, Navigation, Clock, Meteorological, DORIS).
- **Native Compression Support**: All parsers natively handle ``.gz`` compressed files and ``.crx`` Compact RINEX files.
- **Flexible Output Options**: Supports exporting parsed data into rapid-access in-memory Pandas DataFrames or serialized HDF5 disk layouts.
- **Orbital Dynamics & Positioning**: Precise computation of satellite states, tailored for PPP and PPP-RTK workflows.
- **Advanced Interpolation**: Robust time-series upsampling using polynomial and linear methods.
- **Date Conversions**: Comprehensive module for converting between Gregorian, Day of Year, Julian Date, Modified Julian Date, and GPS Time frames.

Source Code
-----------
The source code for Hyperion is hosted on GitHub. Please visit our repository to contribute or report issues:
`GitHub Repository <https://github.com>`_

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   api
