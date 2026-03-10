User Guide
==========

This guide explains the basic usage patterns of the Hyperion library (v1.0.5) and provides examples to help you get started quickly.

Quick Start Example
-------------------

Hyperion provides unified ``load_*`` functions for reading various GNSS file formats. By default, these functions return Pandas DataFrames, but they can also write directly to HDF5 files to save memory.

Here is a quick example of reading a compressed Compact RINEX observation file and saving it natively as an HDF5 table:

.. code-block:: python

   import hyperion

   # Automatically unpack .crx.gz natively and serialize directly to an HDF5 table
   output_file = hyperion.load_rinex_obs(
       'OBS00IGS_R_20240060000_01D_30S_MO.crx.gz',
       return_type='hdf5',
       output_path='observation_data.h5'
   )

   print(f"Dataset successfully saved to: {output_file}")


Basic Usage Patterns
--------------------

Most functions in Hyperion are designed to accept file paths and return structured data (DataFrames) or binary serialized files (HDF5). 

To return data directly into variables in memory:

.. code-block:: python

   # Returns a Pandas DataFrame directly
   df_sp3 = hyperion.load_sp3('example.sp3')

To optimize memory utilization when working with exceptionally large datasets, specify ``return_type='hdf5'``:

.. code-block:: python

   # Buffers parsed data and saves directly to disk
   hyperion.load_sp3('example.sp3', return_type='hdf5', output_path='sp3_data.h5')


Overview of Main Modules
------------------------

Parsers
^^^^^^^
Hyperion includes a vast array of parsers to seamlessly manage industry-standard formats:

- **SP3-D** (``load_sp3``): Orbit paths and trajectories.
- **RINEX Observation** (``load_rinex_obs``): Observation files, including versions 2.xx and 3.xx/4.xx, featuring native CRINEX decompression.
- **RINEX Navigation** (``load_rinex_nav``): Ephemeris and navigation metadata.
- **ANTEX** (``load_antex``): Antenna phase center models.

Along with many other parsers: YUMA Almanacs, RINEX Clock, RINEX MET, DORIS RINEX, ERP, IONEX, and SINEX.

Date Conversions (``misc.date_conversions``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The date conversions module enables bidirectional transformations between standard geodetic time formats (Gregorian, Day of Year, Julian Date, Modified Julian Date, and GPS Time frames).

.. code-block:: python

   from hyperion.misc.date_conversions import gregorian_to_gps_time

   # Convert a Gregorian date to GPS Week and Seconds of Week (SOW)
   gps_week, gps_sow = gregorian_to_gps_time(2026, 3, 10, 12, 0, 0)

Interpolation Methods (``data_model``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Upsample time-series GNSS data robustly without improperly interpolating through missing gaps (NaN values).

.. code-block:: python

   from hyperion.data_model import interpolate_data
   
   # Upsample data using standard numerical polynomial interpolation
   df_interp = interpolate_data(df, interval=30, method='polynomial')

   # Upsample data faster using simple linear interpolation
   df_interp_linear = interpolate_data(df, interval=30, method='linear')

For detailed descriptions of the programmatic interactions, refer to the :doc:`api`.
