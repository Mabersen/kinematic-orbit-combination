# -*- coding: utf-8 -*-
"""
kinematic_utilities.py
 
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


import numpy as np
import pandas as pd

from gravtools.kinematic_orbits import classes

def compute_rmse(dataframes, reference_data):
    """
    Plot the root mean square error (RMSE) between the reference orbit and each individual orbit
    only at epochs where the epochs align.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of dataframes representing the individual orbits to be compared. Each dataframe should have the columns:
        'x_pos', 'y_pos', 'z_pos', and a datetime index.
    combined_df : pd.DataFrame
        Dataframe containing the reference orbit data with 'x_pos', 'y_pos', and 'z_pos' columns.

    Returns
    -------
    None
    """
    dfs = dataframes
    # combined_df = reference_data
    # List to store RMSE for each orbit
    rmse_values = []

    # Go through each dataframe (orbit) and calculate RMSE against the combined orbit
    for i, df in enumerate(dfs):
        # df = df[['x_pos', 'y_pos', 'z_pos']]
        overall_rmse = np.sqrt(np.sum(np.square((reference_data[['x_pos', 'y_pos', 'z_pos']] - df[['x_pos', 'y_pos', 'z_pos']]).dropna(how='any',
                               axis=0, subset=['x_pos', 'y_pos', 'z_pos'])), axis=1).mean())
        # Append RMSE for this orbit
        rmse_values.append(overall_rmse)
    return rmse_values


def compute_residuals(dataframes, reference_data):
    """
    Plot the root mean square error (RMSE) between the reference orbit and each individual orbit
    only at epochs where the epochs align.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of dataframes representing the individual orbits to be compared. Each dataframe should have the columns:
        'x_pos', 'y_pos', 'z_pos', and a datetime index.
    combined_df : pd.DataFrame
        Dataframe containing the reference orbit data with 'x_pos', 'y_pos', and 'z_pos' columns.

    Returns
    -------
    None
    """
    dfs = dataframes
    # List to store RMSE for each orbit
    rmse_values = []

    # Go through each dataframe (orbit) and calculate RMSE against the combined orbit
    for i, df in enumerate(dfs):
        # df = df[['x_pos', 'y_pos', 'z_pos']]
        overall_residuals = (reference_data[['x_pos', 'y_pos', 'z_pos']] - df[['x_pos', 'y_pos', 'z_pos']]).dropna(how='any',
                                                                                                                   axis=0, subset=['x_pos', 'y_pos', 'z_pos'])

        rmse_x = np.sqrt(np.mean(np.square(overall_residuals['x_pos'])))
        rmse_y = np.sqrt(np.mean(np.square(overall_residuals['y_pos'])))
        rmse_z = np.sqrt(np.mean(np.square(overall_residuals['z_pos'])))

        rmse_values.append(np.array([rmse_x, rmse_y, rmse_z]))
    rmse_values = np.array(rmse_values)
    return rmse_values


# Function to convert datetime + seconds to seconds since initial epoch


def calculate_seconds_from_zero(df):
    initial_time = df.iloc[0]['datetime'] + pd.to_timedelta(df.iloc[0]['seconds'], unit='s')
    df['time_from_zero'] = df.apply(lambda row: (
        row['datetime'] + pd.to_timedelta(row['seconds'], unit='s') - initial_time).total_seconds(), axis=1)
    return df

# Function to compare dataframes by aligning them by epoch and calculating differences


def compare_epochs(arc1, arc2):
    """ THIS FUNCTION IS OLD AND MUST BE UPDATED OR REMOVED."""
    if arc1.date != arc2.date:
        return print('This data is not from the same day')
    if arc1.satellite_id != arc2.satellite_id:
        return print('This data is not from the same satellite')

    df1 = arc1.trajectory.copy()
    df2 = arc2.trajectory.copy()

    analysis_centre1 = arc1.analysis_centre
    analysis_centre2 = arc2.analysis_centre

    date1 = arc1.date
    date2 = arc2.date

    date = date1 if date1 == date2 else False

    df1 = df1.copy()
    df2 = df2.copy()
    # Merge on datetime and seconds columns to align epochs
    merged_df = pd.merge(df1, df2, on=['datetime', 'seconds', 'satellite_id'], suffixes=('_df1', '_df2'))

    # Create a new dataframe to store the differences with the same structure as the original
    diff_df = pd.DataFrame()

    # Keep the 'datetime', 'seconds', and 'satellite_id' columns
    diff_df['datetime'] = merged_df['datetime']
    diff_df['seconds'] = merged_df['seconds']
    diff_df['satellite_id'] = merged_df['satellite_id']

    # Compute the differences for position, clock bias, std, and correlation factors
    diff_df['x_pos'] = merged_df['x_pos_df1'] - merged_df['x_pos_df2']
    diff_df['y_pos'] = merged_df['y_pos_df1'] - merged_df['y_pos_df2']
    diff_df['z_pos'] = merged_df['z_pos_df1'] - merged_df['z_pos_df2']
    diff_df['clock_bias'] = merged_df['clock_bias_df1'] - merged_df['clock_bias_df2']
    diff_df['std_x'] = merged_df['std_x_df1'] - merged_df['std_x_df2']
    diff_df['std_y'] = merged_df['std_y_df1'] - merged_df['std_y_df2']
    diff_df['std_z'] = merged_df['std_z_df1'] - merged_df['std_z_df2']
    diff_df['std_code'] = merged_df['std_code_df1'] - merged_df['std_code_df2']
    diff_df['corr_xy'] = merged_df['corr_xy_df1'] - merged_df['corr_xy_df2']
    diff_df['corr_xz'] = merged_df['corr_xz_df1'] - merged_df['corr_xz_df2']
    diff_df['corr_xt'] = merged_df['corr_xt_df1'] - merged_df['corr_xt_df2']
    diff_df['corr_yz'] = merged_df['corr_yz_df1'] - merged_df['corr_yz_df2']
    diff_df['corr_yt'] = merged_df['corr_yt_df1'] - merged_df['corr_yt_df2']
    diff_df['corr_zt'] = merged_df['corr_zt_df1'] - merged_df['corr_zt_df2']

    output = classes.DailyComparison(trajectory_difference=diff_df,
                                     analysis_centres=[analysis_centre1, analysis_centre2],
                                     date=date,
                                     first_epoch=[diff_df['datetime'][0], diff_df['seconds'][0]],
                                     satellite_id=arc1.satellite_id)

    return output

# Splitting orbit dataframes into arcs defined by a time:


def split_data_into_arcs(input_dataframes, num_arcs, validation_data=None):
    """
    Splits input and validation data into specified number of arcs.

    Parameters:
    input_dataframes: List of Pandas DataFrames (input orbits from analysis centres).
    validation_data: List of Pandas DataFrames (validation orbits).
    num_arcs: Integer, number of arcs to split the data into.

    Returns:
    List of dictionaries, where each dictionary contains 'input' and 'validation' arcs.
    """

    index = input_dataframes[0].index
    for i in range(1, len(input_dataframes)):
        df = input_dataframes[i]
        index2 = df.index
        # index = index.intersection(index2).sort_values().unique()
        index = index.union(index2).sort_values().unique()

    # Reindex each dataframe to ensure they all have the same epochs (with NaN for missing ones)
    input_dataframes = [df.reindex(index) for df in input_dataframes]

    if validation_data is not None:
        validation_data = [df.reindex(index) for df in validation_data]

    # Use the index of the first input dataframe for splitting
    full_index = input_dataframes[0].index
    arc_indices = np.array_split(full_index, num_arcs)
    input_dict = {}
    validation_dict = {}
    for idx, arc_idx in enumerate(arc_indices):

        input_dict[f'arc_{idx}'] = [df.loc[arc_idx] for df in input_dataframes]
        validation_dict[f'arc_{idx}'] = [val.loc[arc_idx] for val in validation_data]

    return input_dict, validation_dict


def split_data_into_arcs_by_time(input_dataframes, arc_duration, validation_data=None):
    """
    Splits input and validation data into arcs of a specified time duration.

    Parameters:
    input_dataframes: List of Pandas DataFrames (input orbits from analysis centres).
    validation_data: List of Pandas DataFrames (validation orbits).
    arc_duration: pandas.Timedelta, duration of each arc (e.g., pd.Timedelta(minutes=5)).

    Returns:
    List of dictionaries, where each dictionary contains 'input' and 'validation' arcs.
    """
    index = input_dataframes[0].index
    for df in input_dataframes[1:]:
        index = index.union(df.index).sort_values().unique()

    input_dataframes = [df.reindex(index) for df in input_dataframes]

    if validation_data is not None:
        validation_data = [df.reindex(index) for df in validation_data]

    full_index = input_dataframes[0].index
    if not isinstance(full_index, pd.DatetimeIndex):
        raise ValueError("The index of input dataframes must be a pandas DatetimeIndex.")

    start_time = full_index[0]
    end_time = full_index[-1]

    # Create time bins for the arcs
    arc_edges = pd.date_range(start=start_time, end=end_time, freq=arc_duration)
    if arc_edges[-1] < end_time:
        arc_edges = arc_edges.append(pd.Index([end_time]))  # Ensure the last time is included

    arcs = []
    for i in range(len(arc_edges) - 1):
        # print(i)
        arc_start = arc_edges[i]
        arc_end = arc_edges[i + 1]

        if i == (len(arc_edges) - 1) - 1:
            # print(i, len(arc_edges), 'last arc')
            arc_idx = full_index[(full_index > arc_start) & (full_index <= arc_end)]
        else:
            arc_idx = full_index[(full_index >= arc_start) & (full_index < arc_end)]

        if not arc_idx.empty:
            if validation_data is not None:
                arcs.append({
                    "input": [df.loc[arc_idx] for df in input_dataframes],
                    "validation": [val.loc[arc_idx] for val in validation_data]
                })
            else:
                arcs.append({
                    "input": [df.loc[arc_idx] for df in input_dataframes]
                })

    input_dict = {}
    validation_dict = {}
    for idx, arc in enumerate(arcs):
        input_dict[f'arc_{idx}'] = arc['input']
        if validation_data is not None:
            validation_dict[f'arc_{idx}'] = arc['validation']

    return input_dict, validation_dict


def combine_arcs_into_dataframe(input_dict, validation_dict=None):
    """
    Combines arcs back into independent dataframes for each analysis centre.

    Parameters:
    input_dict: Dictionary where keys are arc identifiers and values are lists of Pandas DataFrames (input arcs).
    validation_dict: (Optional) Dictionary where keys are arc identifiers and values are lists of Pandas DataFrames (validation arcs).

    Returns:
    List of combined input dataframes, and if provided, a list of combined validation dataframes.
    """
    # Extract the number of dataframes per arc
    num_dataframes = len(next(iter(input_dict.values())))

    # Combine input dataframes
    combined_input = []
    for i in range(num_dataframes):
        # Combine dataframes at index `i` across all arcs
        combined_df = pd.concat([arc[i] for arc in input_dict.values()]).sort_index()
        combined_input.append(combined_df)

    if validation_dict is not None:
        # Combine validation dataframes
        combined_validation = []
        for i in range(num_dataframes):
            # Combine dataframes at index `i` across all arcs
            combined_val_df = pd.concat([arc[i] for arc in validation_dict.values()]).sort_index()
            combined_validation.append(combined_val_df)

        return combined_input, combined_validation

    return combined_input


def filter_arcs_by_validation_epochs(input_dict, validation_df):
    """
    Filters the arcs in the input dictionary, keeping only arcs whose epochs
    are fully contained within the validation dataframe.

    Parameters:
    input_dict: dict
        Dictionary of input arcs, where keys are arc identifiers and values are lists of DataFrames.
    validation_df: pandas.DataFrame
        Validation dataframe with a pandas datetime index.

    Returns:
    filtered_input_dict: dict
        Filtered dictionary of input arcs.
    """
    if not isinstance(validation_df.index, pd.DatetimeIndex):
        raise ValueError("The validation dataframe must have a pandas DatetimeIndex.")
    validation_index = validation_df.index
    filtered_input_dict = {}

    for arc_key, input_arc in input_dict.items():

        # Get the index for the first DataFrame in the input arc

        index = validation_index
        for i in range(1, len(input_arc)):
            df = input_arc[i]
            index2 = df.index
            index = index.intersection(index2).sort_values().unique()
            # index = index.union(index2).sort_values().unique()

        if index.empty:
            continue
        input_arc = [i.reindex(index) for i in input_arc]

        filtered_input_dict[arc_key] = input_arc

    return filtered_input_dict


# Generate pink noise
def generate_pink_noise_with_std(length, beta=1, target_std=1.0):
    """
    Generate pink noise for an orbit dataframe with a specified standard deviation.

    Parameters:
        orbit_df (pd.DataFrame): DataFrame containing the orbit data. 
                                 The number of rows determines the size of the noise.
        beta (float): Power spectral density scaling exponent (1 for pink noise).
        target_std (float): Desired standard deviation for the generated noise.

    Returns:
        np.ndarray: Array of pink noise values with the specified standard deviation.
    """

    # Determine the size from the orbit dataframe
    size = length
    # Generate white noise
    white_noise = np.random.normal(size=size)

    # Fourier transform to frequency domain
    f_transform = np.fft.rfft(white_noise)

    # Create frequencies corresponding to the Fourier components
    freqs = np.fft.rfftfreq(size)

    # Avoid division by zero or numerical instability
    epsilon = 1e-10
    scaling_factors = np.where(freqs == 0, 0, 1 / (freqs**beta + epsilon))

    # Apply scaling to the Fourier transform
    f_transform = f_transform * scaling_factors

    # Transform back to time domain
    pink_noise = np.fft.irfft(f_transform, n=size)

    # Scale the noise to the desired standard deviation
    pink_noise *= target_std / np.std(pink_noise)  # Scale to achieve target_std

    return pink_noise


def reindex_dataframes(dataframes):
    """
    Reindex a list of dataframes to ensure they share the same datetime index.

    Parameters:
        dataframes (list of pd.DataFrame): List of dataframes to reindex.

    Returns:
        list of pd.DataFrame: Reindexed dataframes.
    """
    if not dataframes:
        return dataframes

    # Find common index
    common_index = dataframes[0].index
    for df in dataframes[1:]:
        common_index = common_index.intersection(df.index).sort_values().unique()

    # Reindex all dataframes
    return [df.reindex(common_index) for df in dataframes]


def filter_arcs_by_residuals(input_data, reference_data, threshold):
    """Filter overlapping epochs with residuals > threshold, retain non-overlapping epochs."""
    filtered_arcs = []
    reference_df = reference_data[0]  # Use the first reference dataset
    
    for arc in input_data:
        # Split into overlapping and non-overlapping epochs
        common_epochs = arc.index.intersection(reference_df.index)
        non_overlapping_epochs = arc.index.difference(reference_df.index)
        
        # Process overlapping epochs
        if not common_epochs.empty:
            # Compute residuals for overlapping epochs
            arc_common = arc.loc[common_epochs]
            ref_common = reference_df.loc[common_epochs]
            
            dx = arc_common['x_pos'] - ref_common['x_pos']
            dy = arc_common['y_pos'] - ref_common['y_pos']
            dz = arc_common['z_pos'] - ref_common['z_pos']
            residual_mag = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Identify valid overlapping epochs (residual <= threshold)
            valid_common_epochs = residual_mag[residual_mag <= threshold].index
        else:
            valid_common_epochs = pd.Index([])
        
        # Combine valid overlapping epochs + non-overlapping epochs
        valid_epochs = valid_common_epochs.union(non_overlapping_epochs)
        filtered_arc = arc.loc[valid_epochs.sort_values()]
        
        if not filtered_arc.empty:
            filtered_arcs.append(filtered_arc)
    
    return filtered_arcs