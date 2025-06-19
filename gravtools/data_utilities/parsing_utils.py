# -*- coding: utf-8 -*-
"""
parsing_utils.py
 
Copyright 2025 Mattijs Berendsen
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# from tudatpy.kernel.math import interpolators

import os
import numpy as np
import pandas as pd

# from astropy import units as u
# from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

from gravtools.kinematic_orbits.frame_utilities import gcrs_to_itrs, gcrs_to_itrs2
from gravtools.tudat_utilities import split_dataframe_by_gaps
from datetime import datetime, timedelta
from decimal import Decimal, getcontext

# Convert gps to utc time, enabling compatibility with many python modules (e.g. Astropy transformations)

def convert_gps_to_utc(df, time_column='datetime'):
    """
    Convert GPS time to UTC in the given pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column of time in GPS format.
    time_column : str, optional
        Name of the column containing the GPS time (default is 'datetime').

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column containing the UTC times.
    """
    # Convert the pandas datetime column (assumed to be in GPS time) to ISO format strings
    df = df.shift(+19, 's')  # convert from GPS to TAI (GPS is always 19 seconds behind)
    tai_times = list(pd.Series(df.index).dt.strftime('%Y-%m-%dT%H:%M:%S.%f').to_numpy())

    # Create an astropy Time object in TAI format
    gps_astropy_time = Time(tai_times, format='isot', scale='tai', precision=9)

    # Convert to UTC
    utc_astropy_time = gps_astropy_time.utc

    # Add the UTC times back to the dataframe as a new column
    df['datetime'] = utc_astropy_time.to_datetime()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(df['datetime'])
    df = df.drop(columns=['datetime'])
    return df

# Convert utc to gps time, enabling compatibility with many python modules (e.g. Astropy transformations)


def convert_utc_to_gps(df, time_column='datetime'):
    """
    Convert UTC time to GPS time in the given pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column of time in UTC format.
    time_column : str, optional
        Name of the column containing the UTC time (default is 'datetime').

    Returns
    -------
    pd.DataFrame
        DataFrame with the index converted from UTC to GPS time.
    """
    # Convert the pandas datetime index to ISO format strings
    utc_times = list(pd.Series(df.index).dt.strftime('%Y-%m-%dT%H:%M:%S.%f').to_numpy())

    # Create an astropy Time object in UTC format
    utc_astropy_time = Time(utc_times, format='isot', scale='utc', precision=9)

    # Convert UTC to TAI
    tai_astropy_time = utc_astropy_time.tai

    # Convert TAI to GPS (GPS is always 19 seconds behind TAI)
    gps_astropy_time = tai_astropy_time  # Subtract 19 seconds

    # Add the GPS times back to the dataframe as a new index
    df['datetime'] = gps_astropy_time.to_datetime()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(df['datetime'])
    df = df.drop(columns=['datetime'])
    df = df.shift(-19, 's')
    return df


def parse_orbit_file(file_path: str,
                     window_start: str,
                     window_stop: str,
                     data_type: str,
                     analysis_centre: str):
    """
    Parse an orbit file to extract satellite position and any additional data if required.

    The user may add any additional parsing functions which will be used if some combination of data type, analysis
    centre and data format is input to the function. For example, IFG requires a unique parser for its reduced-dynamic
    orbits.

    Parameters
    ----------
    file_path : str
        The path to the SP3K file to be parsed.
    datatype : str, default '.sp3k'
        The datatype to parse.

    Returns
    -------
    Arc
        An object of class `Arc` containing the satellite trajectory data in a DataFrame,
        the analysis center and the satellite ID.
    """
    # define this to be set to 'True' if an orbit parser has been used to avoid the calling of generic parsers when
    # a specific one has been used.

    # Parse reduced-dynamic orbits in the IFG format.
    if 'RDO' in data_type and 'IFG' in analysis_centre:
        return parse_rdo_ifg(file_path, window_start, window_stop)
    # Parse ESA reduced-dynamic orbits (these are produced by the TU Delft).
    if 'RDO' in data_type and 'ESA' in analysis_centre:
        # print('Parsing ESA RDO data')
        return parse_sp3c(file_path, window_start, window_stop)

    if 'KO' in data_type and 'IFG' in analysis_centre:
        return parse_sp3k_ko_ifg(file_path, window_start, window_stop)

    if 'KO' in data_type and 'AIUB' in analysis_centre:
        return parse_sp3k(file_path, window_start, window_stop)

    if 'KO' in data_type and 'TUD' in analysis_centre:
        return parse_sp3k(file_path, window_start, window_stop)

    if 'KO' in data_type and 'ESA' in analysis_centre:
        return parse_sp3c(file_path, window_start, window_stop)

    if 'KO' in data_type and 'JOSE' in analysis_centre:
        return parse_sp3k(file_path, window_start, window_stop)
    if 'CO' in data_type:
        return parse_sp3k(file_path, window_start, window_stop)
    if 'CO' in data_type:
        return parse_sp3k(file_path, window_start, window_stop)
    if data_type == 'TEST':
        return parse_sp3k(file_path, window_start, window_stop)
    # Parse generic sp3k files.
    # if 'sp3k' in data_format:
    #     return parse_sp3k(file_path, window_start, window_stop)

    # # Parse generic sp3c files.
    # if 'sp3c' in data_format:
    #     return parse_sp3c(file_path, window_start, window_stop)

    print('A parser for this datatype is not implemented')
    return False


# Parse IFG reduced-dynamic orbits
def parse_rdo_ifg(file_path: str,
                  window_start: datetime,
                  window_stop: datetime):
    """
    Parse reduced-dynamic orbits from IFG.

    Parameters
    ----------
    file_path : str
        Path of the file
    window_start : datetime
        Initial time as a datetime.
    window_stop : datetime
        Final time as a datetime.

    Returns
    -------
    pd.DataFrame
        Dataframe containing orbit data.

    """

    data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # global line
    for i, line in enumerate(lines):
        if i >= 6:
            # print(type(line.split()[:-3][1]))
            data.append(np.array([val for val in line.split()[:-3]]))
    
    data = np.array(data)
    
    original_data = data.copy()
    
    # t = Time(data[:,0], format='mjd', scale = 'tai')
    data = pd.DataFrame(data, columns=['mjd', 'x_pos', 'y_pos',
                        'z_pos', 'x_vel', 'y_vel', 'z_vel'], dtype=str)
    
    # data[['mjd_days', 'frac_days']] = data['mjd'].str.split('.', expand=True)
    # data['days'] 
    # Convert to Decimal for precise arithmetic
    # Set high precision for Decimal operations
    
    # -----------------------------------------------------------
    # getcontext().prec = 60  # Adjust based on your data's needs
    # def gps_to_utc(row):
        
    #     mjd_dec = Decimal(row['mjd'])
    #     # Add GPS-to-TAI offset (19 seconds in days)
    #     offset = Decimal('19') / Decimal('86400')  # 19 seconds as fractional days
    #     tai_mjd_dec = mjd_dec + offset
    #     # Convert to string for Astropy
    #     tai_mjd_str = '{0:f}'.format(tai_mjd_dec)
    #     t_tai = Time(tai_mjd_str, format='mjd', scale='tai', precision=9)
    #     return t_tai.utc
    
        
    # data['datetime'] = data.apply(gps_to_utc, axis=1) 
    t = data['mjd'].apply(lambda mjd: Time(mjd, format='mjd', scale='tai')) #Time(data['mjd'].values, format='mjd', scale='tai')
    t_tai = t.apply(lambda t: t + TimeDelta(19.0, format='sec')) #Time(data['mjd'].values, format='mjd', scale='tai') + TimeDelta(19.0, format='sec')
    data['datetime'] = t_tai.apply(lambda t: t.utc)

    
    # data['datetime'] = data['datetime'] + Time.timedelta(19.0, format='sec')
    data = gcrs_to_itrs2(data)
    # data['datetime'] = data['datetime'].dt.round('1s')
    
    def mjd_to_iso(row):
        return row['datetime'].iso
    
    data['datetime'] = pd.to_datetime(data.apply(mjd_to_iso, axis=1), format = 'ISO8601')
    # 
    data_out = data.set_index(data['datetime']).drop(columns = ['datetime'])
        
    
    # ----------------------------------------------------------- OLD PARSING METHOD

    # data = []
    # # global line
    # for i, line in enumerate(lines):
    #     if i >= 6:
    #         data.append(np.array([float(val) for val in line.split()[:-3]]))
    # data = np.array(data)
    
    # original_data = data.copy()
    # # Conversion to Julian Date from MJD
    # original_data[:, 0] = original_data[:, 0] + 2400000.5
    # original_data = pd.DataFrame(original_data, columns=['jd', 'x_pos', 'y_pos',
    #                     'z_pos', 'x_vel', 'y_vel', 'z_vel'])
    
    
    # original_data['datetime'] = pd.to_datetime(original_data['jd'], unit='D', origin='julian')
    
    # original_data.set_index('datetime', inplace=True)
    
    # original_data = convert_gps_to_utc(original_data)
    
    # original_data['datetime'] = original_data.index
    # original_data['datetime'] = original_data['datetime'].dt.round('1s')
    # original_data.set_index('datetime', inplace=True)
    
    # original_data.sort_index()
    # original_data = original_data.truncate(before=window_start, after=window_stop)
    
    # if not original_data.empty:
    #     original_data = gcrs_to_itrs(original_data)
    # else:
    #     return False

    return data_out#,original_data#


# def parse_rdo_ifg_optimized(file_path: str, 
#                            window_start: pd.Timestamp,
#                            window_stop: pd.Timestamp) -> pd.DataFrame:
#     """
#     Optimized parser with decimal precision preservation
#     """
#     getcontext().prec = 60  # Set precision once at start

#     # Read data keeping MJD as string
#     df = pd.read_csv(
#         file_path,
#         skiprows=6,
#         sep='\s+',
#         usecols=range(7),
#         names=['mjd', 'x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel'],
#         dtype={'mjd': str, 'x_pos': np.float64, 'y_pos': np.float64, 'z_pos': np.float64,
#                'x_vel': np.float64, 'y_vel': np.float64, 'z_vel': np.float64}
#     )

#     # Precompute constants
#     offset = Decimal('19') / Decimal('86400')  # GPS->TAI offset in days

#     # Vectorized decimal conversion
#     tai_mjds = [f"{(Decimal(mjd) + offset):f}" for mjd in df['mjd']]

#     # Batch convert using astropy (much faster than row-wise)
#     t = Time(tai_mjds, format='mjd', scale='tai', precision=9)
#     df['datetime'] = t.utc
#     df = gcrs_to_itrs2(df)
    
#     df['datetime'] = t.utc.iso# datetime64.astype('datetime64[ns]')
#     df['datetime'] = pd.to_datetime(df['datetime'], format = 'ISO8601')
#     # Set index and filter
#     df.set_index('datetime', inplace=True)
    
#     # df=df.drop(columns=['datetime'])
    
#     return df.loc[window_start:window_stop].sort_index()

def parse_sp3k(file_path: str,
               window_start: datetime,
               window_stop: datetime):
    """
    Parse generic sp3k files.

    Parameters
    ----------
    file_path : str
        Path of the file
    window_start : datetime
        Initial time as a datetime.
    window_stop : datetime
        Final time as a datetime.

    Returns
    -------
    pd.DataFrame
        Dataframe containing orbit data.

    """
    data = []
    # Open the SP3K file and read line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # print(f'Loading file: {file_path}')
    current_datetime = None
    parsing_data = False
    first_epoch = True
    pl_row = False
    ep_row = False
    columns = ['datetime', 'seconds']
    for i, line in enumerate(lines):

        if i == 0:
            # Version information is on the first line
            version = line[1]
        # if 'k' in version:
        # Check if the line starts with an asterisk (*) to extract the datetime
        if line.startswith("*"):
            # Extract datetime parts
            year = int(line[3:7])
            month = int(line[8:10])
            day = int(line[11:13])
            hour = int(line[14:16])
            minute = int(line[17:19])
            second = float(line[20:31].strip())  # Handle fractional seconds accurately

            current_datetime = datetime(year, month, day, hour, minute)

            # Manually adjust time if seconds are greater than 60; auto roll over minute and adjust seconds
            if second >= 60:
                # minute += 1
                current_datetime += timedelta(minutes=1)
                second = second - 60
                
            if first_epoch:
                if (current_datetime + timedelta(days=1.1) < window_start) or (current_datetime > window_stop):
                    
                    return False
                else:
                    first_epoch = False
            if (current_datetime < window_start - timedelta(minutes=5)) or (current_datetime > window_stop + timedelta(minutes=5)):
                outside_window = True
            else:
                outside_window = False
                parsing_data = True
            continue

        if not parsing_data:
            # Skip until we reach the data section
            continue

        # Split the line by whitespace
        parts = line.split()

        if len(parts) == 0:
            # Skip empty lines
            continue

        if line.startswith('PL') and not outside_window:
            if not pl_row:
                columns = columns + ['x_pos', 'y_pos', 'z_pos', 'clock_bias']
                pl_row = True
            # Extract kinematic positions and optional clock bias
            x_pos = float(parts[1]) * 1000  # Convert to m
            y_pos = float(parts[2]) * 1000  # Convert to m
            z_pos = float(parts[3]) * 1000  # Convert to m
            clock_bias = float(parts[4]) if len(
                parts) > 4 else None  # Optional clock bias
            if x_pos == y_pos == z_pos == 0.:  # quick fix to deal with IAUB filling in empty rows with 0's
                pass
            else:
                # Append the data to the array (initialise row for later addition)
                data.append([current_datetime, second,
                            x_pos, y_pos, z_pos, clock_bias])
        
        if line.startswith('EPx') and not outside_window:
            if not ep_row:
                columns = columns + ['std_x', 'std_y', 'std_z', 'std_code', 'cov_xy', 'cov_xz', 'cov_xt', 'cov_yz',
                                     'cov_yt', 'cov_zt']
                ep_row = True
            # Parse the EPx line std data
            try:
                x_std = float(line[4:11])  / 1000  # convert to m
                y_std = float(line[12:18]) / 1000  # convert to m
                z_std = float(line[19:27]) / 1000  # convert to m
            except:
                x_std = y_std = z_std = 10000
            try:
                clock_std = float(line[27:32])
            except:
                clock_std = 0
            try:     
                # Parse the correlation factors
                xy_corr = float(line[33:41])
                xz_corr = float(line[42:50])
                yz_corr = float(line[60:68])
    
                xy_cov = (xy_corr * x_std * y_std) / 10000000 
                xz_cov = (xz_corr * x_std * z_std) / 10000000
                yz_cov = (yz_corr * y_std * z_std) / 10000000
                
            except:
                xy_cov = xz_cov = yz_cov = 0

            try:
                xt_corr = float(line[51:59])
                yt_corr = float(line[69:78])
                zt_corr = float(line[79:87])

                xt_cov = (xt_corr * x_std * clock_std) / 10000000
                yt_cov = (yt_corr * y_std * clock_std) / 10000000
                zt_cov = (zt_corr * z_std * clock_std) / 10000000
            except:
                xt_cov = yt_cov = zt_cov = 0
                
            if x_std == y_std == z_std == 0.:  # quick fix to deal with AIUB filling in empty rows with 0's
                pass
            else:

                # Append the errors data to the same row as the position data
                data[-1].extend([x_std, y_std, z_std, clock_std, xy_cov,
                                xz_cov, xt_cov, yz_cov, yt_cov, zt_cov])
        # else:
        #     print('This is not an sp3k file.')
        #     return False

    # Check if any data is parsed
    if not parsing_data:
        return False

    # Create a DataFrame from the data list
    # columns = ['datetime', 'seconds', 'x_pos', 'y_pos', 'z_pos', 'clock_bias',
    #            'std_x', 'std_y', 'std_z', 'std_code', 'corr_xy', 'corr_xz', 'corr_xt', 'corr_yz',
    #            'corr_yt', 'corr_zt']  # + [f'corr_{i}' for i in range(1, len(corr_factors) + 1)]
    df = pd.DataFrame(data, columns=columns)
    # Convert datetime column to actual datetime format for better handling
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'] + pd.to_timedelta(df['seconds'], unit='s')
    df.drop(columns=['seconds'], inplace=True)
    df.set_index('datetime', inplace=True)
    df = convert_gps_to_utc(df)

    return df


def parse_sp3k_ko_ifg(file_path: str,
                      window_start: datetime,
                      window_stop: datetime):
    """
    Parse sp3k kinematic orbit files from ifg.

    Parameters
    ----------
    file_path : str
        Path of the file
    window_start : datetime
        Initial time as a datetime.
    window_stop : datetime
        Final time as a datetime.

    Returns
    -------
    pd.DataFrame
        Dataframe containing orbit data.

    """
    data = []
    # Open the SP3K file and read line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_datetime = None
    parsing_data = False
    first_epoch = True
    pl_row = False
    ep_row = False
    columns = ['datetime', 'seconds']

    for i, line in enumerate(lines):

        if i == 0:
            # Version information is on the first line
            version = line[1]
            # ac = line[-5:].strip()  # analysis centre
        if 'k' in version:
            # Check if the line starts with an asterisk (*) to extract the datetime
            if line.startswith("*"):
                # Extract datetime parts
                year = int(line[3:7])
                month = int(line[8:10])
                day = int(line[11:13])
                hour = int(line[14:16])
                minute = int(line[17:19])
                second = float(line[20:31].strip())  # Handle fractional seconds accurately

                current_datetime = datetime(year, month, day, hour, minute)

                # Manually adjust time if seconds are greater than 60; auto roll over minute and adjust seconds
                if second >= 60:
                    # minute += 1
                    current_datetime += timedelta(minutes=1)
                    second = second - 60

                if first_epoch:
                    if (current_datetime + timedelta(days=1.05) < window_start) or (current_datetime > window_stop):
                        return False
                    else:
                        first_epoch = False
                if (current_datetime < window_start - timedelta(minutes=5)) or (current_datetime > window_stop + timedelta(minutes=5)):
                    outside_window = True
                else:
                    outside_window = False
                    parsing_data = True
                continue

            if not parsing_data:
                # Skip until we reach the data section
                continue

            # Split the line by whitespace
            parts = line.split()

            if len(parts) == 0:
                # Skip empty lines
                continue

            if line.startswith('PL') and not outside_window:
                # Extract kinematic positions and optional clock bias
                if not pl_row:
                    columns = columns + ['x_pos', 'y_pos', 'z_pos', 'clock_bias']
                    pl_row = True
                x_pos = float(parts[1]) * 1000  # Convert to m
                y_pos = float(parts[2]) * 1000  # Convert to m
                z_pos = float(parts[3]) * 1000  # Convert to m
                clock_bias = float(parts[4]) if len(
                    parts) > 4 else None  # Optional clock bias
                if x_pos == y_pos == z_pos == 0.:  # quick fix to deal with IAUB filling in empty rows with 0's
                    pass
                else:
                    # Append the data to the array (initialise row for later addition)
                    data.append([current_datetime, second,
                                x_pos, y_pos, z_pos, clock_bias])

            elif line.startswith('EPx') and not outside_window:
                if not ep_row:
                    columns = columns + ['std_x', 'std_y', 'std_z', 'std_code', 'cov_xy', 'cov_xz', 'cov_xt', 'cov_yz',
                                         'cov_yt', 'cov_zt']
                    ep_row = True
                # Extract standard deviations and correlation factors
                # x_std = float(parts[1]) / 1000  # Convert to m
                # y_std = float(parts[2]) / 1000  # Convert to m
                # z_std = float(parts[3]) / 1000  # Convert to m
                # clock_std = 0.0
                # corr_factors = [float(x) for x in parts[4:]]
                # xy_corr = corr_factors[0]
                # xz_corr = corr_factors[1]
                # xt_corr = 0.0
                # yz_corr = corr_factors[2]
                # yt_corr = 0.0
                # zt_corr = 0.0
                
                # Parse the EPx line std data
                x_std = float(line[4:9]) / 1000  # convert to m
                y_std = float(line[9:14]) / 1000  # convert to m
                z_std = float(line[14:19]) / 1000  # convert to m
                clock_std = 0

                # Parse the correlation factors
                xy_corr = float(line[27:35])
                xz_corr = float(line[36:44])
                yz_corr = float(line[54:62])

                xy_cov = (xy_corr * x_std * y_std) / 10000000
                xz_cov = (xz_corr * x_std * z_std) / 10000000
                yz_cov = (yz_corr * y_std * z_std) / 10000000
 
                
                xt_cov = yt_cov = zt_cov = 0
                    
                if x_std == y_std == z_std == 0.:  # quick fix to deal with AIUB filling in empty rows with 0's
                    pass

                else:
                    # Append the errors data to the same row as the position data
                    data[-1].extend([x_std, y_std, z_std, clock_std, xy_cov,
                                    xz_cov, xt_cov, yz_cov, yt_cov, zt_cov])
        else:
            print('This file is not an sp3k file')
            return False
        
    # Check if any data is parsed
    if not parsing_data:
        return False

    # Create a DataFrame from the data list
    columns = ['datetime', 'seconds', 'x_pos', 'y_pos', 'z_pos', 'clock_bias',
               'std_x', 'std_y', 'std_z', 'std_code', 'cov_xy', 'cov_xz', 'cov_xt', 'cov_yz',
               'cov_yt', 'cov_zt']  # + [f'corr_{i}' for i in range(1, len(corr_factors) + 1)]
    df = pd.DataFrame(data, columns=columns)

    # Convert datetime column to actual datetime format for better handling
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'] + pd.to_timedelta(df['seconds'], unit='s')
    df.drop(columns=['seconds'], inplace=True)
    df.set_index('datetime', inplace=True)
    df = convert_gps_to_utc(df)

    return df


def parse_sp3c(file_path: str,
               window_start: datetime,
               window_stop: datetime):
    """
    Parse generic sp3k files.

    Parameters
    ----------
    file_path : str
        Path of the file
    window_start : datetime
        Initial time as a datetime.
    window_stop : datetime
        Final time as a datetime.

    Returns
    -------
    pd.DataFrame
        Dataframe containing orbit data.

    """
    data = []
    # Open the SP3K file and read line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_datetime = None
    parsing_data = False
    first_epoch = True
    pl_row = False
    vl_row = False
    ep_row = False
    columns = ['datetime', 'seconds']
    for i, line in enumerate(lines):
        # Check if the line starts with an asterisk (*) to extract the datetime
        if line.startswith("*"):
            # Extract datetime parts
            year = int(line[3:7])
            month = int(line[8:10])
            day = int(line[11:13])
            hour = int(line[14:16])
            minute = int(line[17:19])
            second = float(line[20:31].strip())  # Handle fractional seconds accurately

            current_datetime = datetime(year, month, day, hour, minute)

            # Manually adjust time if seconds are greater than 60; auto roll over minute and adjust seconds
            if second >= 60:
                # minute += 1
                current_datetime += timedelta(minutes=1)
                second = second - 60

            if first_epoch:
                if (current_datetime + timedelta(days=1.05) < window_start) or (current_datetime > window_stop):
                    return False
                else:
                    first_epoch = False
            if (current_datetime < window_start - timedelta(minutes=5)) or (current_datetime > window_stop + timedelta(minutes=5)):
                outside_window = True
            else:
                outside_window = False
                parsing_data = True
            continue

        if not parsing_data:
            # Skip until we reach the data section
            continue

        # Split the line by whitespace
        parts = line.split()

        if len(parts) == 0:
            # Skip empty lines
            continue

        if line.startswith('PL') and not outside_window:
            if not pl_row:
                columns = columns + ['x_pos', 'y_pos', 'z_pos', 'clock_bias']
                pl_row = True
            # Extract kinematic positions and optional clock bias
            x_pos = float(parts[1]) * 1000  # Convert to m
            y_pos = float(parts[2]) * 1000  # Convert to m
            z_pos = float(parts[3]) * 1000  # Convert to m
            clock_bias = float(parts[4]) if len(
                parts) > 4 else None  # Optional clock bias
            if x_pos == y_pos == z_pos == 0.:  # quick fix to deal with IAUB filling in empty rows with 0's
                pass
            else:
                # Append the data to the array (initialise row for later addition)
                data.append([current_datetime, second,
                            x_pos, y_pos, z_pos, clock_bias])

        # if line.startswith('VL') and not outside_window:
        #     if not vl_row:
        #         columns = columns + ['x_vel', 'y_vel', 'z_vel', 'clock_bias_v']
        #         pl_row = True
        #     x_vel = float(parts[1])
        #     y_vel = float(parts[2])
        #     z_vel = float(parts[3])
        #     clock_bias_v = float(parts[4]) if len(
        #         parts) > 4 else None  # Optional clock bias
        #     if x_vel == y_vel == z_vel == 0.:  # quick fix to deal with IAUB filling in empty rows with 0's
        #         pass
        #     else:
        #         # Append the data to the array (initialise row for later addition)
        #         data[-1].extend([x_vel, y_vel, z_vel, clock_bias_v])

        if line.startswith('EP') and not outside_window:  # MUST BE ADJUSTED FOR SP3C (CURRENTLY SP3K)
            if not ep_row:
                columns = columns + ['std_x', 'std_y', 'std_z', 'std_code', 'cov_xy', 'cov_xz', 'cov_xt', 'cov_yz',
                                     'cov_yt', 'cov_zt']
                ep_row = True
            # Parse the EPx line covariance data
            x_std = float(line[4:9]) / 1000  # convert to m
            y_std = float(line[9:14]) / 1000  # convert to m
            z_std = float(line[14:19]) / 1000  # convert to m
            clock_std = float(line[19:26])

            # Parse the correlation factors
            xy_corr = float(line[26:35])
            xz_corr = float(line[36:43])
            xt_corr = float(line[44:52])
            yz_corr = float(line[53:61])
            yt_corr = float(line[62:71])
            zt_corr = float(line[72:80])

            xy_cov = (xy_corr * x_std * y_std) / 10000000
            xz_cov = (xz_corr * x_std * z_std) / 10000000
            xt_cov = (xt_corr * x_std * clock_std) / 10000000

            yz_cov = (yz_corr * y_std * z_std) / 10000000
            yt_cov = (yt_corr * y_std * clock_std) / 10000000
            zt_cov = (zt_corr * z_std * clock_std) / 10000000

            if x_std == y_std == z_std == 0.:  # quick fix to deal with AIUB filling in empty rows with 0's
                pass
            else:

                # Append the errors data to the same row as the position data
                data[-1].extend([x_std, y_std, z_std, clock_std, xy_cov,
                                xz_cov, xt_cov, yz_cov, yt_cov, zt_cov])
                
    # Check if any data is parsed
    if not parsing_data:
        return False

    # Create a DataFrame from the data list
    # columns = ['datetime', 'seconds', 'x_pos', 'y_pos', 'z_pos', 'clock_bias',
    #            'std_x', 'std_y', 'std_z', 'std_code', 'corr_xy', 'corr_xz', 'corr_xt', 'corr_yz',
    #            'corr_yt', 'corr_zt']  # + [f'corr_{i}' for i in range(1, len(corr_factors) + 1)]
    df = pd.DataFrame(data, columns=columns)

    # Convert datetime column to actual datetime format for better handling
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'] + pd.to_timedelta(df['seconds'], unit='s')
    df.drop(columns=['seconds'], inplace=True)
    df.set_index('datetime', inplace=True)
    df = convert_gps_to_utc(df)

    return df



# def save_sp3k_OLD(df: pd.DataFrame, satellite_id, file_path: str, file_name, interpolate=False):
#     """
#     Save a pandas DataFrame to an SP3K file.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing orbit data with columns:
#         - x_pos, y_pos, z_pos: Position in meters.
#         - clock_bias: Clock bias (optional).
#         - std_x, std_y, std_z: Standard deviations in meters.
#         - std_code: Standard deviation of clock bias.
#         - cov_xy, cov_xz, cov_xt, cov_yz, cov_yt, cov_zt: Covariances.
#     file_path : str
#         Path to save the SP3K file.
#     """
#     if interpolate:
#         df = df.asfreq(freq='1s')
#         df = df.interpolate(method='spline', order=2, )
#     df = convert_utc_to_gps(df)  # converting the time frame back to GPS
#     no_epochs = len(df)
#     file_name_full = fr'GSWARM_{file_name}.sp3k'
#     file_path = os.path.join(file_path, file_name_full)
#     print(file_path)
#     with open(file_path, 'w') as file:
#         # Write header line (version information)
#         file.write(f"""#kP0000  0  0  0  0  0.00000000   {no_epochs} U+u   IGS20 KIN TUD 
# ## 0000 000000.00000000              0 00000 0.0000000000000
# +    1   L{satellite_id}  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# %c L  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc                          
# %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc                          
# %f  0.0000000  0.000000000  0.00000000000  0.000000000000000                          
# %f  0.0000000  0.000000000  0.00000000000  0.000000000000000                          
# %i    0    0    0    0      0      0      0      0         0                          
# %i    0    0    0    0      0      0      0      0         0 \n""")

#         # Group data by datetime
#         for datetime, group in df.groupby(level=0):
#             # Write timestamp line
#             timestamp_line = f"*  {datetime.year:04d} {datetime.month:02d} {datetime.day:02d} " \
#                              f"{datetime.hour:02d} {datetime.minute:02d} {datetime.second:02d}.{datetime.microsecond:06d}                             \n"
#             file.write(timestamp_line)

#             # Write PL and EPx lines for each row in the group
#             for _, row in group.iterrows():
#                 # Write PL line (position and clock bias)
#                 pl_line = f"PL{satellite_id}{row['x_pos']/1000:14.7f}{row['y_pos']/1000:14.7f}{row['z_pos']/1000:14.7f}"
#                 if 'clock_bias' in row and not pd.isna(row['clock_bias']):
#                     pl_line += f"     {row['clock_bias']:9.6f}"
#                 pl_line += "\n"
#                 file.write(pl_line)

#                 # Write EPx line (standard deviations and covariances)
#                 if 'std_x' in row and not pd.isna(row['std_x']):
#                     continue
#                     # epx_line = f"EPx {row['std_x'] * 1000:6.1f} {row['std_y'] * 1000:6.1f} {row['std_z'] * 1000:6.1f}"   \
#                     #            f"{int(row['std_code']):7d} {int(row['cov_xy'] * 10000000):8d} {int(row['cov_xz'] * 10000000):8d}" \
#                     #            f"{int(row['cov_xt'] * 10000000):8d} {int(row['cov_yz'] * 10000000):8d} {int(row['cov_yt'] * 10000000):8d}" \
#                     #            f"{int(row['cov_zt'] * 10000000):8d}\n"
#                     # file.write(epx_line)

#     print(f"SP3K file saved to {file_path}")
# Non-vectorised sp3k save function
# def save_sp3k_OLD(df: pd.DataFrame, satellite_id, file_path: str, file_name):
#     """
#     Save a pandas DataFrame to an SP3K file.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing orbit data with columns:
#         - x_pos, y_pos, z_pos: Position in meters.
#         - clock_bias: Clock bias (optional).
#         - std_x, std_y, std_z: Standard deviations in meters.
#         - std_code: Standard deviation of clock bias.
#         - cov_xy, cov_xz, cov_xt, cov_yz, cov_yt, cov_zt: Covariances.
#     file_path : str
#         Path to save the SP3K file.
#     """
   
#     df = convert_utc_to_gps(df.copy())  # converting the time frame back to GPS
#     no_epochs = len(df)
#     file_name_full = fr'GSWARM_{file_name}.sp3k'
#     file_path_og = file_path
#     file_path = os.path.join(file_path, file_name_full)
#     print(file_path)
#     with open(file_path, 'w') as file:
#         # Write header line (version information)
#         file.write(f"""#kP0000  0  0  0  0  0.00000000   {no_epochs} U+u   IGS20 KIN TUD 
# ## 0000 000000.00000000              0 00000 0.0000000000000
# +    1   L{satellite_id}  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# +          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          
# %c L  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc                          
# %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc                          
# %f  0.0000000  0.000000000  0.00000000000  0.000000000000000                          
# %f  0.0000000  0.000000000  0.00000000000  0.000000000000000                          
# %i    0    0    0    0      0      0      0      0         0                          
# %i    0    0    0    0      0      0      0      0         0 \n""")

#         # Group data by datetime
#         for datetime, group in df.groupby(level=0):
#             # Write timestamp line
#             timestamp_line = f"*  {datetime.year:04d} {datetime.month:02d} {datetime.day:02d} " \
#                              f"{datetime.hour:02d} {datetime.minute:02d} {datetime.second:02d}.{datetime.microsecond:06d}                             \n"
#             file.write(timestamp_line)

#             # Write PL and EPx lines for each row in the group
#             for _, row in group.iterrows():
#                 # Write PL line (position and clock bias)
#                 pl_line = f"PL{satellite_id}{row['x_pos']/1000:14.7f}{row['y_pos']/1000:14.7f}{row['z_pos']/1000:14.7f}"
#                 if 'clock_bias' in row and not pd.isna(row['clock_bias']):
#                     pl_line += f"     {row['clock_bias']:9.6f}"
#                 pl_line += "\n"
#                 file.write(pl_line)

#                 # Write EPx line (standard deviations and covariances)
#                 try:
#                     if 'std_x' in row and not pd.isna(row['std_x']):
#                         # continue
#                         epx_line = f"EPx {row['std_x'] * 1000:6.1f} {row['std_y'] * 1000:6.1f} {row['std_z'] * 1000:6.1f}"   \
#                                    f"{0:7d} {int(row['cov_xy']/(row['std_x'] * row['std_y']) * 10000000):8d} {int(row['cov_xz']/(row['std_x'] * row['std_z']) * 10000000):8d}" \
#                                    f"{0:8d} {int(row['cov_yz']/(row['std_y'] * row['std_z']) * 10000000):8d} {0:8d}" \
#                                    f"{0:8d}\n"
#                         file.write(epx_line)
#                 except:
#                     continue

#     print(f"SP3K file saved to {file_path}")
    
#     # Extract metadata
#     parts = file_name_full.split('_')
#     method = parts[1]
#     ref_data = parts[2]
#     date_str = parts[-1]
#     year = int(date_str[:4])
#     doy = int(date_str[4:7])

#     # Parse orbit file
#     orbit_df = df
    
#     # Split into valid data windows
#     valid_windows = []
#     chunks = split_dataframe_by_gaps(orbit_df, threshold_seconds=1)
#     total_epochs = sum(len(chunk) for chunk in chunks)
    
#     for chunk in chunks:
#         if not chunk.empty:
#             valid_windows.append((chunk.index[0], chunk.index[-1]))
            
#     timescale_path = os.path.join(file_path_og, "timescales")
#     os.makedirs(timescale_path, exist_ok=True)
    
#     # Save as numpy array
#     if valid_windows:
#         np.savez(
#             os.path.join(timescale_path, f"{method}_{ref_data}_{year}_{doy:03}.npz"),
#             valid_windows=np.array(valid_windows),
#             total_epochs=np.int64(total_epochs)
#         )

# Vectorised save function
def save_sp3k(df: pd.DataFrame, satellite_id, file_path: str, file_name):
    """
    Optimized function to save a pandas DataFrame to an SP3K file.
    """
    df = convert_utc_to_gps(df.copy())
    df = df.sort_index()  # Ensure dataframe is sorted by datetime
    no_epochs = len(df)
    file_name_full = f'GSWARM_{file_name}.sp3k'
    file_path_og = file_path
    file_path = os.path.join(file_path, file_name_full)
    
    # Precompute PL lines in sorted order
    pl_lines = []
    clock_bias_available = 'clock_bias' in df.columns
    for x, y, z, cb in zip(df['x_pos']/1000, 
                            df['y_pos']/1000, 
                            df['z_pos']/1000, 
                            df['clock_bias'] if clock_bias_available else [0]*len(df)):
        pl_line = f"PL{satellite_id}{x:14.7f}{y:14.7f}{z:14.7f}"
        if clock_bias_available and not pd.isna(cb):
            pl_line += f"     {cb:9.6f}"
        pl_lines.append(pl_line + "\n")
    
    # Precompute EPx lines with integer indices
    epx_lines = []
    if all(col in df.columns for col in ['std_x', 'std_y', 'std_z', 'cov_xy', 'cov_xz', 'cov_yz']):
        mask = df[['std_x', 'std_y', 'std_z']].notna().all(axis=1)
        valid_indices = np.where(mask)[0]  # Get integer positions
        
        # Vectorized calculations
        std_x = (df.loc[mask, 'std_x'] * 1000).values
        std_y = (df.loc[mask, 'std_y'] * 1000).values
        std_z = (df.loc[mask, 'std_z'] * 1000).values
        
        cov_xy = df.loc[mask, 'cov_xy'].values
        cov_xz = df.loc[mask, 'cov_xz'].values
        cov_yz = df.loc[mask, 'cov_yz'].values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_xy = (cov_xy / ((std_x/1000) * (std_y/1000))) * 1e7
            corr_xz = (cov_xz / ((std_x/1000) * (std_z/1000))) * 1e7
            corr_yz = (cov_yz / ((std_y/1000) * (std_z/1000))) * 1e7
        
        corr_xy = np.nan_to_num(corr_xy).astype(int)
        corr_xz = np.nan_to_num(corr_xz).astype(int)
        corr_yz = np.nan_to_num(corr_yz).astype(int)
        
        for sx, sy, sz, cxy, cxz, cyz in zip(std_x, std_y, std_z, corr_xy, corr_xz, corr_yz):
            epx_line = f"EPx {sx:6.1f} {sy:6.1f} {sz:6.1f}       0 {cxy:8d} {cxz:8d}        0 {cyz:8d}        0        0\n"
            epx_lines.append(epx_line)
        
        # Map to positions using integer indices
        epx_full = [None] * len(df)
        for pos, line in zip(valid_indices, epx_lines):
            epx_full[pos] = line
    else:
        epx_full = [None] * len(df)
    
    # Generate header
    header = [
        f"#kP0000  0  0  0  0  0.00000000   {no_epochs} U+u   IGS20 KIN TUD\n",
        "## 0000 000000.00000000              0 00000 0.0000000000000\n",
        f"+    1   L{satellite_id}  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0                          \n",
        "%c L  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc                          \n",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc                          \n",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000                          \n",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000                          \n",
        "%i    0    0    0    0      0      0      0      0         0                          \n",
        "%i    0    0    0    0      0      0      0      0         0 \n"
    ]
    
    # Prepare body content
    body = []
    prev_datetime = None
    for idx, (datetime, row) in enumerate(df.iterrows()):  # Now using sorted dataframe
        if datetime != prev_datetime:
            ts_line = f"*  {datetime.year:04d} {datetime.month:02d} {datetime.day:02d} " \
                      f"{datetime.hour:02d} {datetime.minute:02d} {datetime.second:02d}.{datetime.microsecond:06d}                             \n"
            body.append(ts_line)
            prev_datetime = datetime
        
        body.append(pl_lines[idx])  # Now in sorted order
        if epx_full[idx] is not None:
            body.append(epx_full[idx])
    
    # Write all content at once
    with open(file_path, 'w') as f:
        f.writelines(header + body)
    
    print(f"SP3K file saved to {file_path}")
    
    # Extract metadata
    parts = file_name_full.split('_')
    method = parts[1]
    ref_data = parts[2]
    date_str = parts[-1]
    year = int(date_str[:4])
    doy = int(date_str[4:7])

    # Parse orbit file
    orbit_df = df
    
    # Split into valid data windows
    valid_windows = []
    chunks = split_dataframe_by_gaps(orbit_df, threshold_seconds=1)
    total_epochs = sum(len(chunk) for chunk in chunks)
    
    for chunk in chunks:
        if not chunk.empty:
            valid_windows.append((chunk.index[0], chunk.index[-1]))
            
    timescale_path = os.path.join(file_path_og, "timescales")
    os.makedirs(timescale_path, exist_ok=True)
    
    # Save as numpy array
    if valid_windows:
        np.savez(
            os.path.join(timescale_path, f"{method}_{ref_data}_{year}_{doy:03}.npz"),
            valid_windows=np.array(valid_windows),
            total_epochs=np.int64(total_epochs)
        )
# def split_dataframe_by_gaps(df, threshold_seconds=10):
#     """Split DataFrame into chunks, ensuring minimum valid data points."""
#     no_gaps = False
#     if df.empty:
#         print('empty')
#         return []
    
#     # Filter for valid position data
#     df_valid = df.dropna(how='all')
#     if df_valid.empty:
#         print('empty2')
#         return []
    
#     times = df_valid.index
#     if len(times) < 2:
#         print('insufficient length')
#         return [df_valid] if not df_valid.empty else []
    
#     deltas = np.diff(times).astype('timedelta64[s]')
#     split_indices = np.where(deltas > np.timedelta64(threshold_seconds, 's'))[0] + 1
#     print(split_indices)
#     if len(split_indices) == 0:
#         no_gaps = True
#         print(no_gaps)
#         return [df]
#     chunks = []
#     previous = 0
#     for idx in split_indices:
#         chunks.append(df_valid.iloc[previous:idx])
#         previous = idx
#     if previous < len(df_valid):
#         chunks.append(df_valid.iloc[previous:])
    
#     return chunks

def process_orbit_files(orbit_dir: str, output_dir: str):
    """Process orbit files and save valid time windows in GPS time"""
    for filename in os.listdir(orbit_dir):
        if not filename.endswith('.sp3k'):
            continue
        print("Processing file: ", filename)
        # Extract metadata
        parts = filename.split('_')
        method = parts[1]
        ref_data = parts[2]
        date_str = parts[-1]
        year = int(date_str[:4])
        doy = int(date_str[4:7])

        # Parse orbit file
        file_path = os.path.join(orbit_dir, filename)
        orbit_df = parse_sp3k(
            file_path,
            window_start=datetime(year, 1, 1) + timedelta(days=doy-1),
            window_stop=datetime(year, 1, 1) + timedelta(days=doy)
        )
        
        if orbit_df is False:
            print('Failed file')
            continue

        # Convert to GPS time
        orbit_df = convert_utc_to_gps(orbit_df)  # Assuming this converts the index to GPS time
        # orbit_df.index = orbit_df.index.view(np.int64) // 1e9  # Convert to GPS seconds if needed

        # Split into valid data windows
        valid_windows = []
        chunks = split_dataframe_by_gaps(orbit_df, threshold_seconds=1)
        
        # Calculate total epochs across all valid windows
        total_epochs = sum(len(chunk) for chunk in chunks)
        
        for chunk in chunks:
            if not chunk.empty:
                valid_windows.append((chunk.index[0], chunk.index[-1]))
        
        # Save as numpy array
        if valid_windows:
            
            print("Saving file: ", f"{method}_{ref_data}_{year}_{doy:03}.npz")
            np.savez(
                os.path.join(output_dir, f"{method}_{ref_data}_{year}_{doy:03}.npz"),
                valid_windows=np.array(valid_windows),
                total_epochs=np.int64(total_epochs)
            )
        else: 
            print("No valid windows")