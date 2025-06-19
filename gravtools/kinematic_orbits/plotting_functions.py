# -*- coding: utf-8 -*-
"""
plotting_functions.py
 
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
import matplotlib.pyplot as plt
from gravtools.kinematic_orbits.kinematic_utilities import calculate_seconds_from_zero, compute_rmse
from gravtools.kinematic_orbits.frame_utilities import itrs_to_gcrs


def plot_rmse_vs_combined_orbit(dataframes, reference_data, validation_weights='equal'):
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
    if validation_weights == 'equal':
        validation_weights = np.ones(len(reference_data))
        validation_weights = validation_weights / sum(validation_weights)
    dfs = dataframes
    reference_data = [i.trajectory for i in reference_data]
    # reference_data = reference_data[['x_pos', 'y_pos', 'z_pos']]

    # List to store RMSE for each orbit
    rmse_values = []
    all_rmse = []

    # Set up plot for RMS of magnitude differences
    # fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    # Go through each dataframe (orbit) and calculate RMSE against the combined orbit
    for i, df in enumerate(dfs):
        # fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        # Align both dataframes based on the index (datetime epochs) using an inner join
        df = df[['x_pos', 'y_pos', 'z_pos']]
        rmse_list = []
        for j in reference_data:
            j = j[['x_pos', 'y_pos', 'z_pos']]
            merged = j.copy().join(df[['x_pos', 'y_pos', 'z_pos']], how='inner',
                                   lsuffix='_combined', rsuffix='_individual')

            overall_rmse = compute_rmse([df], j)[0]
            rmse_list.append(overall_rmse)
            # Calculate magnitude of vectors for combined and individual orbits
            magnitude_combined = np.sqrt((merged['x_pos_combined'] - merged['x_pos_individual'])**2
                                         + (merged['y_pos_combined'] - merged['y_pos_individual'])**2
                                         + (merged['z_pos_combined'] - merged['z_pos_individual'])**2)
            magnitude_individual = np.sqrt(merged['x_pos_individual']**2
                                           + merged['y_pos_individual']**2 + merged['z_pos_individual']**2)

            # # Compute the RMS of the difference in magnitude
            # rms_magnitude_diff = np.sqrt(((magnitude_combined - magnitude_individual)
            #                              ** 2).mean())  # Correct method for RMSE?

            # # Append RMSE for this orbit

            # # rmse_values.append(rms_magnitude_diff)

            # Plot RMS of the magnitude difference over time for this orbit
            rms_diff = np.sqrt((magnitude_combined) ** 2)
            # ax1.plot(merged.index.to_numpy(), rms_diff)

        weighted_rmse = validation_weights * np.array(rmse_list)
        weighted_average = sum(weighted_rmse)
        all_rmse.append(rmse_list)
        rmse_values.append(weighted_average)

    # Plot a bar chart of overall RMSE for each orbit
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(rmse_values) + 1), rmse_values)
    plt.title('Overall RMSE for Each Orbit vs Reference Orbit')
    plt.xlabel('Orbit')
    plt.ylabel('Overall RMSE (m)')
    plt.grid(True)
    plt.show()

    # Return RMSE values if needed for further analysis
    # print(rmse_values)
    return rmse_values, all_rmse


def plot_rmse_vs_combined_orbit_OLD(dataframes, reference_data):
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
    combined_df = reference_data
    # reference_data = reference_data[['x_pos', 'y_pos', 'z_pos']]

    # List to store RMSE for each orbit
    rmse_values = []
    # Set up plot for RMS of magnitude differences
    # fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    # Go through each dataframe (orbit) and calculate RMSE against the combined orbit
    for i, df in enumerate(dfs):
        # Align both dataframes based on the index (datetime epochs) using an inner join
        df = df[['x_pos', 'y_pos', 'z_pos']]
        merged = combined_df.copy().join(df[['x_pos', 'y_pos', 'z_pos']], how='inner',
                                         lsuffix='_combined', rsuffix='_individual')
        # merged = merged.interpolate(how='linear')

        # Calculate the squared differences between the combined and individual orbits
        # squared_diff_x = (merged['x_pos_combined'] - merged['x_pos_individual']) ** 2
        # squared_diff_y = (merged['y_pos_combined'] - merged['y_pos_individual']) ** 2
        # squared_diff_z = (merged['z_pos_combined'] - merged['z_pos_individual']) ** 2
        # global overall_rmse, miauuw
        # miauuw = squared_diff_x + squared_diff_y + squared_diff_z
        # Compute overall RMSE (root mean of all squared differences)

        overall_rmse = np.sqrt(np.sum(np.square((reference_data - df).dropna(how='any',
                               axis=0, subset=['x_pos', 'y_pos', 'z_pos'])), axis=1).mean())
        # overall_rmse = np.sqrt((squared_diff_x + squared_diff_y + squared_diff_z).mean())

        # Calculate magnitude of vectors for combined and individual orbits
        magnitude_combined = np.sqrt(merged['x_pos_combined']**2
                                     + merged['y_pos_combined']**2 + merged['z_pos_combined']**2)
        magnitude_individual = np.sqrt(merged['x_pos_individual']**2
                                       + merged['y_pos_individual']**2 + merged['z_pos_individual']**2)

        # Compute the RMS of the difference in magnitude
        rms_magnitude_diff = np.sqrt(((magnitude_combined - magnitude_individual)
                                     ** 2).mean())  # Correct method for RMSE?

        # Append RMSE for this orbit
        rmse_values.append(overall_rmse)
        # rmse_values.append(rms_magnitude_diff)

        # Plot RMS of the magnitude difference over time for this orbit
        rms_magnitude_diff = np.sqrt((magnitude_combined - magnitude_individual) ** 2)
        # ax1.plot(merged.index.to_numpy(), rms_magnitude_diff)

    # Plot a bar chart of overall RMSE for each orbit
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(rmse_values) + 1), rmse_values)
    plt.title('Overall RMSE for Each Orbit vs Reference Orbit')
    plt.xlabel('Orbit')
    plt.ylabel('Overall RMSE (m)')
    plt.grid(True)
    plt.show()

    # Return RMSE values if needed for further analysis
    # print(rmse_values)
    return rmse_values, reference_data, df
# Plotting function for positions


def plot_positions(df):
    initial_epoch = f"""{df["datetime"][0]}, {df["seconds"][0]} seconds"""
    df = calculate_seconds_from_zero(df)

    plt.figure(figsize=(10, 6))
    plt.plot(df['time_from_zero'], df['x_pos'], label='X Position', color='r')
    plt.plot(df['time_from_zero'], df['y_pos'], label='Y Position', color='g')
    plt.plot(df['time_from_zero'], df['z_pos'], label='Z Position', color='b')
    plt.xlabel('Time (seconds from start)')
    plt.ylabel('Position (m)')
    plt.title(f'position components over time, seconds since {initial_epoch}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting function for the norm of positions


def plot_position_norm(df):
    earth_radius = 6371e3  # m
    initial_epoch = f"""{df["datetime"][0]}, {df["seconds"][0]} seconds"""
    df = calculate_seconds_from_zero(df)
    pos_norm = np.sqrt(df['x_pos']**2 + df['y_pos']**2 + df['z_pos']**2)

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(df['time_from_zero'], pos_norm, color='r')
    ax1.set_xlabel('Time (seconds from start)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title(f'Norm of the position over time, seconds since {initial_epoch}')

    ax2.plot(df['time_from_zero'], pos_norm - earth_radius, color='b')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_xlabel('Time (seconds from start)')

    # plt.legend()
    plt.grid(True)
    plt.show()

# Plotting function for the norm of positions


def plot_position_differences(df):

    initial_epoch = f"""{df["datetime"][0]}, {df["seconds"][0]} seconds"""
    df = calculate_seconds_from_zero(df)

    plt.figure(figsize=(10, 6))
    plt.plot(df['time_from_zero'], df['x_pos'], label='X Position', color='r')
    plt.plot(df['time_from_zero'], df['y_pos'], label='Y Position', color='g')
    plt.plot(df['time_from_zero'], df['z_pos'], label='Z Position', color='b')
    plt.xlabel('Time (seconds from start)')
    plt.ylabel('Position difference (m)')
    plt.title(f'Position difference over time, seconds since {initial_epoch}')
    plt.legend()
    plt.grid(True)
    plt.show()

    pos_norm = np.sqrt(df['x_pos']**2 + df['y_pos']**2 + df['z_pos']**2)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.plot(df['time_from_zero'], pos_norm, color='r')
    ax1.set_xlabel('Time (seconds from start)')
    ax1.set_ylabel('Norm of the position difference (m)')
    ax1.set_title(f'Norm of the position difference over time, seconds since {initial_epoch}')

    # plt.legend()
    plt.grid(True)
    plt.show()


def plot_position_differences_all(dailycomparison_list):

    # # Extract the unique data centres from the provided data
    # analysis_centres = set([i.analysis_centre for i in dailycomparison_list])
    for dailycomparison in dailycomparison_list:

        df = dailycomparison.trajectory_difference
        df = calculate_seconds_from_zero(df.copy())
        initial_epoch = f"""{df["datetime"][0]}, {df["seconds"][0]} seconds"""

        analysis_centres = dailycomparison.analysis_centres
        pos_norm = np.sqrt(df['x_pos']**2 + df['y_pos']**2 + df['z_pos']**2)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        ax1.plot(df['time_from_zero'], pos_norm * 1000, color='r')
        ax1.set_xlabel('Time (seconds from start)')
        ax1.set_ylabel('Norm of the position difference (m)')
        ax1.set_title(
            f'Norm of the position difference over time, seconds since {initial_epoch}, between {analysis_centres[0]} and {analysis_centres[1]}')

        # plt.legend()
        plt.grid(True)
        plt.show()

# Plotting function for standard deviations


def plot_standard_deviations(df):
    initial_epoch = f"""{df["datetime"][0]}, {df["seconds"][0]} seconds"""
    df = calculate_seconds_from_zero(df)

    plt.figure(figsize=(10, 6))
    plt.plot(df['time_from_zero'], df['std_x'], label='Std X', color='r')
    plt.plot(df['time_from_zero'], df['std_y'], label='Std Y', color='g')
    plt.plot(df['time_from_zero'], df['std_z'], label='Std Z', color='b')
    plt.plot(df['time_from_zero'], df['std_code'], label='Std Code', color='k')
    plt.xlabel('Time (seconds from start)')
    plt.ylabel('Provided standard deviations (mm)')
    plt.title(f'Standard deviations over time, seconds since {initial_epoch}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting function for correlation factors


def plot_correlation_factors(df):
    initial_epoch = f"""{df["datetime"][0]}, {df["seconds"][0]} seconds"""
    df = calculate_seconds_from_zero(df)

    plt.figure(figsize=(10, 6))
    plt.plot(df['time_from_zero'], df['corr_xy'], label='Corr XY', color='r')
    plt.plot(df['time_from_zero'], df['corr_xz'], label='Corr XZ', color='g')
    # plt.plot(df['time_from_zero'], df['corr_xt'], label='Corr XT', color='b')
    plt.plot(df['time_from_zero'], df['corr_yz'], label='Corr YZ', color='c')
    # plt.plot(df['time_from_zero'], df['corr_yt'], label='Corr YT', color='m')
    # plt.plot(df['time_from_zero'], df['corr_zt'], label='Corr ZT', color='y')
    plt.xlabel('Time (seconds from start)')
    plt.ylabel('Correlation factors')
    plt.title(f'Correlation factors over time, seconds since {initial_epoch}')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting functions for comparison of orbits
def plot_orbital_data(dataframes: list, frame='itrs'):

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect("equal")
    miauw = []

    for df in dataframes:
        if frame == 'itrs':
            pass

        if frame == 'gcrs':
            df = itrs_to_gcrs(df)

        xo, yo, zo = df[['x_pos']] / 1000, df[['y_pos']] / 1000, df[['z_pos']] / 1000
        ax.plot(xo, yo, zo)

    re = 6370
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = re * np.cos(u) * np.sin(v)
    y = re * np.sin(u) * np.sin(v)
    z = re * np.cos(v)
    ax.plot_wireframe(x, y, z, color="w", edgecolor="orange")

    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')

    plt.show()
    

def plot_orbit_components(orbit_df, title='Orbit Components', label=None):
    """
    Plot the x, y, z components of an orbit over time.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        DataFrame with datetime index and columns ['x_pos', 'y_pos', 'z_pos'].
    title : str
        Title of the plot.
    label : str, optional
        Label for the orbit to display in legend.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    axs[0].plot(orbit_df.index, orbit_df['x_pos'], label=label)
    axs[0].set_ylabel('X Position (m)')
    axs[0].grid(True)

    axs[1].plot(orbit_df.index, orbit_df['y_pos'], label=label)
    axs[1].set_ylabel('Y Position (m)')
    axs[1].grid(True)

    axs[2].plot(orbit_df.index, orbit_df['z_pos'], label=label)
    axs[2].set_ylabel('Z Position (m)')
    axs[2].set_xlabel('Time')
    axs[2].grid(True)
    
    if label is not None:
        for ax in axs:
            ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_orbit_gaps(orbit_df, gap_threshold_seconds=5):
    """
    Plot the x, y, z components of an orbit in a single plot with gaps shaded and missing data percentage.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        DataFrame with datetime index and columns ['x_pos', 'y_pos', 'z_pos'].
    title : str
        Title of the plot.
    gap_threshold_seconds : int
        Threshold in seconds to consider a data gap.
    """
    # Ensure index is sorted
    orbit_df = orbit_df.sort_index()

    # Calculate seconds from start
    t0 = orbit_df.index[0]
    time_seconds = (orbit_df.index - t0).total_seconds()

    # Identify gaps correctly as pairs of (end of last epoch, start of next epoch)
    gap_pairs = []
    for i in range(1, len(orbit_df)):
        delta = (orbit_df.index[i] - orbit_df.index[i - 1]).total_seconds()
        if delta > gap_threshold_seconds:
            gap_pairs.append(((orbit_df.index[i - 1] - t0).total_seconds(),
                              (orbit_df.index[i] - t0).total_seconds()))

    # Calculate missing data percentage assuming 1 Hz sampling
    total_expected_epochs = int((orbit_df.index[-1] - orbit_df.index[0]).total_seconds()) + 1
    actual_epochs = len(orbit_df)
    missing_percentage = 100 * (1 - actual_epochs / total_expected_epochs)

    # Create single plot
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Plot components
    ax.plot(time_seconds, orbit_df['x_pos'], label='X Position', color='blue')
    ax.plot(time_seconds, orbit_df['y_pos'], label='Y Position', color='green')
    ax.plot(time_seconds, orbit_df['z_pos'], label='Z Position', color='brown')

    # Shade gaps
    for start_sec, end_sec in gap_pairs:
        ax.axvspan(start_sec, end_sec, color='red', alpha=0.3, label='Gap' if start_sec == gap_pairs[0][0] else None)

    ax.set_xlabel('Time since start (s)')
    ax.set_ylabel('Position (m)')
    ax.grid(True)
    ax.legend()

    # Add missing data annotation
    plt.title(f"Missing data: {missing_percentage:.1f}% of expected {total_expected_epochs} epochs", fontsize = 14)
    plt.tight_layout()
    plt.show()

def plot_orbit_residuals(df1, df2, label1='Orbit 1', label2='RDO'):
    """
    Plots 3D residuals between two orbits over time, shows RMS as a horizontal line, and reports missing epochs.

    Parameters
    ----------
    df1 : pd.DataFrame
        First orbit DataFrame with datetime index and columns ['x_pos', 'y_pos', 'z_pos'].
    df2 : pd.DataFrame
        Second orbit DataFrame with datetime index and columns ['x_pos', 'y_pos', 'z_pos'].
    label1 : str
        Label for the first orbit.
    label2 : str
        Label for the second orbit.
    title : str
        Title for the plot.
    """
    # Ensure both DataFrames have datetime index sorted
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    # Compute missing epochs relative to df2
    missing_epochs = len(df2.index) - len(df1.index)
    # print(len(df2.index), len(df1.index))
    missing_percentage = 100 * missing_epochs / len(df2.index)
    if missing_percentage < 0:
        missing_percentage = 0
    
    # Identify common epochs
    common_index = df1.index.intersection(df2.index)
    if common_index.empty:
        print("No overlapping epochs between the two orbits.")
        return

    # Correct missing percentage: based on missing from df1 relative to df2
    # missing_percentage = 100 * (1 - len(common_index) / len(df2))
    
    # Restrict to common epochs
    df1_common = df1.loc[common_index]
    df2_common = df2.loc[common_index]
    
    common_epoch_loss = 100 * (len(df2.index) - len(df1_common.index)) / len(df2.index) - missing_percentage
    if common_epoch_loss < 0:
        common_epoch_loss = 0
        
    # Compute 3D residuals (Euclidean norm)
    residuals = np.sqrt(np.sum((df1_common[['x_pos', 'y_pos', 'z_pos']] - df2_common[['x_pos', 'y_pos', 'z_pos']])**2, axis=1))

    # Calculate RMS
    rms_value = np.sqrt(np.mean(residuals ** 2))

    # Calculate seconds from start
    t0 = common_index[0]
    time_seconds = (common_index - t0).total_seconds()

    # Plot residuals
    plt.figure(figsize=(6.5, 6))
    plt.scatter(time_seconds, residuals, color='red', s=0.4, label='3D Residuals')

    # Plot RMS line
    plt.axhline(rms_value, color='blue', linestyle='--', label=f'RMS = {rms_value:.3f} m')
    
    # Add text box for missing percentage
    textstr = f'Missing epochs vs {label2}: {missing_percentage:.2f}%\nUnrounded epochs vs {label2}: {common_epoch_loss:.2f}%'
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Time since start (s)')
    plt.ylabel('3D Residuals (m)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(draggable=True, fontsize=12)
    plt.tight_layout()
    plt.show()
