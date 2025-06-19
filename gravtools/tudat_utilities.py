# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:05:43 2024.

@author: maber
"""
import numpy as np
import pandas as pd
from tudatpy.kernel.math import interpolators


def convert_dataframe_to_tudat(dataframe, first_epoch=None):
    """Convert DataFrame to Tudat format, focusing on valid position data."""
    # Define mandatory position columns
    # position_columns = ['x_pos', 'y_pos', 'z_pos']
    
    # # Check if all position columns exist
    # missing_cols = [col for col in position_columns if col not in dataframe.columns]
    # if missing_cols:
    #     raise ValueError(f"Missing position columns: {missing_cols}")
    
    column_names = list(dataframe.columns)
    # print(column_names)
    # Filter dataframe to include only position columns and drop rows with NaNs in these
    # dataframe_clean = dataframe[column_names].dropna()
    # # dataframe = dataframe.loc[dataframe_clean.index]
    # if dataframe_clean.empty:
    #     return {}, column_names  # Return empty dict if no valid data
    
    if not first_epoch:
        first_epoch = dataframe.index[0]
    
    # Calculate time differences
    time_from_start = (dataframe.index - first_epoch).total_seconds().values
    # print(dataframe.columns)
    # Create Tudat-compatible dictionary
    dictionaries = dataframe.set_index(time_from_start)[column_names].dropna().T.to_dict(orient='list')
    output_dict = {k: np.array(v) for k, v in dictionaries.items()}
    
    return output_dict, column_names


def convert_tudat_to_dataframe(tudat_dict, first_epoch=None, column_names=['x_pos', 'y_pos', 'z_pos']):
    """
    Convert a TudatPy formatted trajectory dictionary to a Pandas DataFrame.

    Parameters
    ----------
    tudat_dict : dict
        Dictionary of trajectory data formatted in Tudat structure.
    first_epoch : pandas.DateTime, optional
        Datetime describing the first epoch you wish your output to start at.
        If none is provided, the first epoch of the dictionary is used. The default is None.

    Returns
    -------
    dataframe : pandas.DataFrame
        Pandas DataFrame containing trajectory information.

    """
    dataframe = pd.DataFrame.from_dict(data=tudat_dict, orient='index', columns=column_names)
    time_from_start = pd.to_timedelta(dataframe.index.to_series(), unit='s')
    datetime_series = pd.DataFrame(time_from_start, columns=['timedelta'])
    datetime_series['datetime'] = first_epoch
    datetime_series['datetime'] = datetime_series['datetime'] + datetime_series['timedelta']
    dataframe = dataframe.set_index(datetime_series['datetime'])
    

    return dataframe

def resample_to_frequency(input_dataframe,
                          frequency,
                          interpolator='Lagrange',
                          order=8):
    """
    Resample a Pandas DataFrame to a specified frequency using the TudatPy interpolator.

    Parameters
    ----------
    input_dataframe : pandas.Dataframe
        Dataframe containing information to be resampled.
    frequency : str
        Str defining frequency to resample to. Refer to Pandas docs for definitions.
    interpolator : str
        String describing the type of interpolator to be used from TudatPy.
    order : int
        Order of the polynomial used for interpolation (only relevant for interpolators which
        make use of this).

    Returns
    -------
    interpolated_dataframe : pandas.DataFrame
        Resamples Pandas DataFrame.

    """
    
    first_epoch_original = input_dataframe.index.to_series().iloc[0]
    
    input_dataframe_time_index = input_dataframe.copy().index.round('1s')
    input_dataframe_time = input_dataframe.set_index(input_dataframe_time_index)
    # print(input_dataframe_time)
    # input_dataframe_time_index = input_dataframe_time.index.round('1s')  # Rounding data to the nearest whole second
    
    input_dataframe_time_rounded = input_dataframe_time.resample(frequency, origin='start').ffill()#.set_index(input_dataframe_time_index) #Ensuring resampling is to whole seconds
    original_row = input_dataframe_time_rounded['original_row']
    
    
    first_epoch_rounded = input_dataframe_time_rounded.index.to_series().iloc[0]
    
    first_epoch_diff = (first_epoch_original - first_epoch_rounded).total_seconds()
    
    
    
    input_dataframe_time_dict, _ = convert_dataframe_to_tudat(input_dataframe_time_rounded) # Time is rounded to ensure interpolation occurs at correct places
    input_dict, column_names = convert_dataframe_to_tudat(input_dataframe, first_epoch = first_epoch_rounded)
    
    
    
    # print(input_dict)
    if input_dict == {}:
        return pd.DataFrame()
    # return input_dict, input_dataframe_time_dict
    interpolator = interpolators.create_one_dimensional_vector_interpolator(
        input_dict, interpolators.lagrange_interpolation(order))

    interpolated_data = {}

    epoch_i = list(input_dict.keys())[int(order/2) + 1]
    epoch_f = list(input_dict.keys())[int(-order/2) - 1]
    # return
    for epoch in input_dataframe_time_dict.keys():

        if epoch >= epoch_i and epoch <= epoch_f:  # to avoid extrapolation (causes problems :))
            interpolated_data[epoch] = interpolator.interpolate(epoch)
    interpolated_dataframe = convert_tudat_to_dataframe(interpolated_data, first_epoch_rounded, column_names)
    interpolated_dataframe['original_row'] = original_row
    return interpolated_dataframe

def resample_to_frequency_or_epochs(input_dataframe,
                                     frequency_or_epochs,
                                     interpolator='Lagrange',
                                     order=8):
    """
    Resample a Pandas DataFrame to a specified frequency or list of epochs using Tudat-style interpolation.

    Parameters
    ----------
    input_dataframe : pandas.DataFrame
        DataFrame containing data to interpolate.
    frequency_or_epochs : str or pd.DatetimeIndex or pd.DataFrame
        Frequency string (e.g. '10s') or target epochs to interpolate to.
    interpolator : str
        Interpolation type (currently only 'Lagrange' is supported).
    order : int
        Interpolation polynomial order.

    Returns
    -------
    interpolated_dataframe : pandas.DataFrame
        Interpolated DataFrame.
    """
    first_epoch_original = input_dataframe.index[0]
    input_dataframe = input_dataframe.copy()

    if isinstance(frequency_or_epochs, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
        if isinstance(frequency_or_epochs, pd.DataFrame):
            target_index = frequency_or_epochs.index
        else:
            target_index = frequency_or_epochs
        target_df = pd.DataFrame(index=target_index)
    elif isinstance(frequency_or_epochs, str):
        # Regular resampling
        input_dataframe = input_dataframe.resample(frequency_or_epochs, origin='start').ffill()
        target_df = input_dataframe
    else:
        raise TypeError("frequency_or_epochs must be a str or a pandas DataFrame/Index")

    # Reference start time for Tudat time
    first_epoch_rounded = target_df.index[0]

    # Convert to Tudat format
    input_dict, column_names = convert_dataframe_to_tudat(input_dataframe, first_epoch=first_epoch_rounded)
    target_dict, _ = convert_dataframe_to_tudat(target_df, first_epoch=first_epoch_rounded)

    if not input_dict:
        return pd.DataFrame()

    interp = interpolators.create_one_dimensional_vector_interpolator(
        input_dict, interpolators.lagrange_interpolation(order)
    )

    # Avoid extrapolation
    t_keys = list(input_dict.keys())
    t_min, t_max = t_keys[int(order / 2) + 1], t_keys[int(-order / 2) - 1]

    interpolated_data = {
        t: interp.interpolate(t)
        for t in target_dict.keys()
        if t_min <= t <= t_max
    }

    interpolated_df = convert_tudat_to_dataframe(interpolated_data, first_epoch_rounded, column_names)

    return interpolated_df

def split_dataframe_by_gaps(df, threshold_seconds=10):
    """Split DataFrame into chunks, ensuring minimum valid data points."""
    no_gaps = False
    if df.empty:
        print('empty')
        return []
    # Check for duplicates
    if not df.index.is_unique:
        # Remove duplicates (keep first)
        df = df[~df.index.duplicated(keep='first')]
    # Filter for valid position data
    df_valid = df.dropna(how='all').sort_index()
    
    if df_valid.empty:
        print('empty2')
        return []
    
    times = df_valid.index
    if len(times) < 2:
        print('insufficient length')
        return [df_valid] if not df_valid.empty else []
    
    deltas = np.diff(times).astype('timedelta64[s]')
    split_indices = np.where(deltas > np.timedelta64(threshold_seconds, 's'))[0] + 1
    if len(split_indices) == 0:
        no_gaps = True
        return [df]
    chunks = []
    previous = 0
    for idx in split_indices:
        chunks.append(df_valid.iloc[previous:idx])
        previous = idx
    if previous < len(df_valid):
        chunks.append(df_valid.iloc[previous:])
    
    return chunks

def resample_with_gap_handling(original_df, frequency, gap_threshold_seconds=5, interpolator='Lagrange', order=10):
    """
    Resamples a DataFrame to a specified frequency, splitting it into chunks to avoid large gaps.
    
    Parameters:
        original_df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        frequency (str): Resampling frequency (e.g., '10s').
        gap_threshold_seconds (int): Threshold gap size (in seconds) to split the DataFrame.
        **interpolator_kwargs: Keyword arguments for `resample_to_frequency` (e.g., `order=8`).
    
    Returns:
        pd.DataFrame: Interpolated DataFrame with valid points only.
    """
    # Split the original DataFrame into chunks
    original_row_indices = pd.RangeIndex(start=0, stop=len(original_df), step=1)
    original_df = original_df.assign(original_row=original_row_indices)
    chunks = split_dataframe_by_gaps(original_df, gap_threshold_seconds)
    min_points = order + 1
    # Interpolate each chunk and collect results
    interpolated_chunks = []
    for chunk in chunks:
        # print(len(chunk))
        # Skip chunks too small for interpolation (e.g., fewer than order+1 points)
        if len(chunk) < min_points:
            continue
        try:
            interpolated = resample_to_frequency(chunk, frequency, interpolator, order)
            if not interpolated.empty:
                interpolated_chunks.append(interpolated)
        except (RuntimeError, IndexError) as e:
            print(f"Skipping chunk due to error: {str(e)}")
            continue
    
    # Combine all interpolated chunks
    if interpolated_chunks:
        combined_df = pd.concat(interpolated_chunks).sort_index()
        combined_df = combined_df[combined_df['original_row'].notna()]
        
        # # Convert 'original_row' back to integer and drop the column
        # combined_df['original_row'] = combined_df['original_row'].astype(int)
        combined_df = combined_df.drop(columns=['original_row'])
        return combined_df
    else:
        return pd.DataFrame()
    
def resample_with_gap_handling_with_epochs(original_df,
                                frequency_or_epochs,
                                gap_threshold_seconds=5,
                                interpolator='Lagrange',
                                order=10):
    """
    Resample a DataFrame to fixed frequency or specific epochs, avoiding gaps that would cause interpolation issues.

    Parameters
    ----------
    original_df : pd.DataFrame
        DataFrame with a datetime index.
    frequency_or_epochs : str or pd.DatetimeIndex or pd.DataFrame
        Either a resampling frequency or an exact set of target epochs.
    gap_threshold_seconds : int
        Time difference threshold to define gaps (ignored if specific epochs are passed).
    interpolator : str
        Type of interpolator to use (currently only 'Lagrange').
    order : int
        Interpolation polynomial order.

    Returns
    -------
    pd.DataFrame
        Interpolated DataFrame.
    """
    original_df = original_df.copy()
    original_df['original_row'] = np.arange(len(original_df))

    # If frequency is used: do gap splitting
    if isinstance(frequency_or_epochs, str):
        chunks = split_dataframe_by_gaps(original_df, gap_threshold_seconds)
    else:
        # If fixed epochs are given, treat entire DataFrame as a single chunk
        chunks = [original_df]

    interpolated_chunks = []
    min_points = order + 3

    for chunk in chunks:
        if len(chunk) < min_points:
            continue
        try:
            interpolated = resample_to_frequency_or_epochs(
                chunk,
                frequency_or_epochs=frequency_or_epochs,
                interpolator=interpolator,
                order=order
            )
            if not interpolated.empty:
                interpolated_chunks.append(interpolated)
        except Exception as e:
            print(f"Interpolation failed for chunk: {e}")
            continue

    if interpolated_chunks:
        return pd.concat(interpolated_chunks).sort_index()
    else:
        return pd.DataFrame()

    
def resample_without_gap_handling(original_df, frequency, interpolator='Lagrange', order=10):
    """
    Resamples a DataFrame to a specified frequency using interpolation, without any gap handling.
    
    Parameters:
        original_df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        frequency (str): Resampling frequency (e.g., '10s').
        interpolator (str): Interpolation method ('Lagrange' or other supported).
        order (int): Order of the interpolator (if applicable).
    
    Returns:
        pd.DataFrame: Interpolated DataFrame.
    """
    # Ensure the DataFrame is sorted
    original_df = original_df.sort_index()
    original_row_indices = pd.RangeIndex(start=0, stop=len(original_df), step=1)
    original_df = original_df.assign(original_row=original_row_indices)
    try:
        # Resample and interpolate directly over the entire DataFrame
        interpolated = resample_to_frequency(original_df, frequency, interpolator, order)
        return interpolated
    except (RuntimeError, IndexError) as e:
        print(f"Resampling failed due to error: {str(e)}")
        return pd.DataFrame()
