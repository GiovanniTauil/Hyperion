API Reference
=============

This section outlines the main functions and classes available natively within Hyperion (v1.0.5).

I/O Functions (Parsers)
-----------------------

All ``load_*`` functions share a consistent, Pythonic interface for dealing with diverse GNSS file formats safely.

.. function:: load_rinex_obs(file_path: str, return_type: str = 'dataframe', output_path: str = None)

   Parses RINEX Observation files (supports .o, .obs, and compressed .crx.gz formats).

   :param str file_path: Path to the input RINEX observation file.
   :param str return_type: Select the return type format, either ``'dataframe'`` or ``'hdf5'``. Default is ``'dataframe'``.
   :param str output_path: Destination path for the .h5 file if ``return_type`` is ``'hdf5'``. Optional if returning a dataframe.
   :return: Parsed observed data.
   :rtype: pandas.DataFrame or str representing the generated output path.

.. function:: load_sp3(file_path: str, return_type: str = 'dataframe', output_path: str = None)

   Reads SP3-D orbit files and systematically structures orbit position and clock parameters.

   :param str file_path: Path to the input SP3-D file.
   :param str return_type: Format to return, ``'dataframe'`` or ``'hdf5'``.
   :param str output_path: Corresponding output path if using ``hdf5``.
   :return: Cleanly parsed orbital components.
   :rtype: pandas.DataFrame or str

.. function:: load_rinex_nav(file_path: str, return_type: str = 'dataframe', output_path: str = None)

   Reads RINEX Navigation messages encompassing spacecraft ephemerides (supports versions 2.xx and 3.0x/4.00).

   :param str file_path: Path to the input RINEX navigation file.
   :param str return_type: Format to return, ``'dataframe'`` or ``'hdf5'``.
   :param str output_path: Corresponding output path if using ``hdf5``.
   :return: Validated navigation entries.
   :rtype: pandas.DataFrame or str

.. function:: load_antex(file_path: str, return_type: str = 'dataframe', output_path: str = None)

   Parses an Absolute Antenna Phase Center Model (.atx) file sequentially.

   :param str file_path: Path to the input ANTEX file.
   :param str return_type: Format to return, ``'dataframe'`` or ``'hdf5'``.
   :param str output_path: Corresponding output path if using ``hdf5``.
   :return: Antenna correction parameters per frequency.
   :rtype: pandas.DataFrame or str

Data Modeling
-------------

.. module:: data_model.interpolation

.. function:: interpolate_data(data, interval, method='polynomial')

   Upsamples GNSS time-series data using specific numerical interpolation techniques while deliberately discarding/handling internally missing NaN values.

   :param pandas.DataFrame data: The time-series dataframe matrix to be evaluated and upsampled.
   :param int interval: The target chronological time interval mapping (e.g., in seconds).
   :param str method: The interpolation method to systematically apply. Allowed values are ``'polynomial'`` or ``'linear'``.
   :return: An effectively upsampled dataset mapping the specified frequency interval.
   :rtype: pandas.DataFrame
   
   **Example**:

   .. code-block:: python

      upsampled_data = interpolate_data(df, interval=30, method='linear')

Miscellaneous
-------------

.. module:: misc.date_conversions

The date conversions module provides mathematically precise translation layers between independent time epochs widely used in standard geodesy pipelines. Important functions available are listed below:

- ``gregorian_to_doy(year, month, day)``: Gregorian Date to Julian Day of Year conversion.
- ``gregorian_to_gps_time(year, month, day, hour, minute, second)``: Gregorian Date equivalent formatted strictly to GPS Week and Seconds of Week (SoW).
- ``gps_time_to_gregorian(gps_week, gps_sow)``: GPS Time to a Gregorian Calendar translation.
- ``gregorian_to_jd(year, month, day, hour, minute, second)``: Seamless mapping logic to generate Julian Dates (JD) from typical Gregorian representations.

*Note: All translation calculations aim for rigorously valid mathematical scaling without external or assumed system-time bias.*
