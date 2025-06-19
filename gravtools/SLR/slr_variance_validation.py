# -*- coding: utf-8 -*-
"""
slr_variance_validation.py
 
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
# Custom SINEX parser from previous steps
from sinex_station_loader import SINEXStationHandler, MultiSINEXStationHandler
from variance_verification_plotting import * #(plot_residuals_vs_sigma, plot_reduced_chi2_bar,
                                            # plot_histograms_norm_residuals, load_sigma_data_and_metrics)
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from tqdm import tqdm

from gravtools.kinematic_orbits.retrieve_arcs import retrieve_arcs
from gravtools.tudat_utilities import resample_with_gap_handling_with_epochs
from gravtools.kinematic_orbits.classes import AccessRequest
from pathlib import Path

# Station mapping loader and mapping functions
def load_ilrs_station_mapping_from_csv(csv_path):
    """
    Load ILRS station name to code mapping from CSV.

    CSV format:
    Station Name,ILRS Code
    Yarragadee,7090
    ...
    """
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['Station Name'].str.strip(), df['ILRS Code'].astype(str).str.strip()))
    return mapping

def extract_station_name(site_string):
    """
    Extract station name by removing last code part from Site string.
    E.g. 'Yarragadee MOBL' -> 'Yarragadee'
    """
    parts = site_string.strip().split()
    if len(parts) >= 2:
        return ' '.join(parts[:-1])
    else:
        return parts[0].strip()

def attach_ilrs_station_code(slr_df, mapping_dict):
    """
    Add ILRS station code column to SLR dataframe using provided mapping.
    """
    slr_df['station_name_clean'] = slr_df['Site'].apply(extract_station_name)
    slr_df['station_code'] = slr_df['station_name_clean'].apply(lambda x: mapping_dict.get(x))

    # Warn and drop rows without valid code
    missing = slr_df[slr_df['station_code'].isna()]['station_name_clean'].unique()
    if len(missing) > 0:
        print(f"Warning: The following station names were not matched to ILRS codes: {missing}")

    slr_df = slr_df.dropna(subset=['station_code'])
    return slr_df


def filter_by_gaps(slr_df, gap_dir, trim_seconds=11):
    """Filter SLR observations and collect total_epochs per method and reference data."""
    slr_df['Year'] = slr_df['Date'].dt.year
    slr_df['DOY'] = slr_df['Date'].dt.dayofyear

    grouped = slr_df.groupby(['Method', 'Year', 'DOY', 'Reference_Data'])

    valid_indices = []
    trim_delta = np.timedelta64(trim_seconds, 's')

    # Collect total epochs per method + reference_data
    from collections import defaultdict
    total_epochs_per_method_ref = defaultdict(lambda: defaultdict(int))

    for (method, year, doy, ref_data), group_indices in grouped.groups.items():
        # method = method_label.split(' ')[0]
        gap_file = Path(gap_dir) / f"{method}_{ref_data}_{year}_{doy:03}.npz"
        if not gap_file.exists():
            continue

        data = np.load(gap_file, allow_pickle=True)
        valid_windows = data['valid_windows']
        total_epochs = data['total_epochs'].item()

        # Accumulate per method and reference data
        total_epochs_per_method_ref[method][ref_data] += total_epochs

        adjusted_windows = []
        for start, end in valid_windows:
            new_start = start + trim_delta
            new_end = end - trim_delta
            if new_start < new_end:
                adjusted_windows.append((new_start, new_end))

        group_dates = slr_df.loc[group_indices, 'Date'].values.astype('datetime64[ns]')

        mask = np.zeros(len(group_dates), dtype=bool)
        for window_start, window_end in adjusted_windows:
            mask |= (group_dates >= window_start) & (group_dates <= window_end)

        valid_mask_indices = group_indices[mask]
        slr_df.loc[valid_mask_indices, 'total_epochs'] = total_epochs
        valid_indices.extend(valid_mask_indices)

    return slr_df.loc[valid_indices].drop(columns=['Year', 'DOY']), total_epochs_per_method_ref


def load_and_clean_data(filepath, gap_dir=None, trim_seconds=5):
    """Load SLR data, clean it, filter by gaps, and add total epochs metrics."""
    df = pd.read_csv(filepath, dtype=str)
    df[['rms[m]', 'Range [m]', 'n', 'Res[m]']] = df[['rms[m]', 'Range [m]', 'n', 'Res[m]']].astype('float')

    df["Date"] = pd.to_datetime(df["Date"], format='mixed')
    df['rej'] = df['rej'].str.strip()
    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
    df['Reference_Data'] = df['Reference_Data'].fillna('None')
    
    df = df.dropna(subset=['Res[m]'])

    df['total_epochs'] = np.nan

    if gap_dir is not None:
        df, total_epochs_per_method_ref = filter_by_gaps(df, gap_dir, trim_seconds)

        # Total_Possible_Epochs from KONF (same as before)
        if 'KONF' in df['Reference_Data'].unique():
            total_epochs = compute_total_konf_epochs_optimized(gap_dir)
            df['Total_Possible_Epochs'] = np.where(
                (df['Reference_Data'] == 'KONF') &
                (df['Method'].isin(['AIUB', 'TUD', 'IFG'])),
                total_epochs,
                np.nan
            )

        # Add Total_Available_Epochs per method + reference_data using combined keys
        def get_total_available_epochs(row):
            method = row['Method']
            ref_data = row['Reference_Data']
            return total_epochs_per_method_ref.get(method, {}).get(ref_data, np.nan)

        df['Total_Available_Epochs'] = df.apply(get_total_available_epochs, axis=1)
    
    # Ensure mu and gamma are float
    df['mu [deg]'] = df['mu [deg]'].astype(float)
    df['gamma'] = df['gamma'].astype(float)
    df['beta'] = df['beta'].astype(float)
    
    # Compute Δu and wrap into 0 to 360 deg
    df['Delta_u [deg]'] = df['mu [deg]'] - df['gamma']
    df['Delta_u [deg]'] = ((df['Delta_u [deg]']) % 360)
    
    return df


def compute_total_konf_epochs_optimized(gap_dir):
    """Optimized calculation of total unique epochs across KONF methods"""
    methods = ['AIUB', 'TUD', 'IFG']
    all_windows = []
    
    # Collect all windows
    for method in methods:
        for npz_file in Path(gap_dir).glob(f"{method}_*.npz"):
            data = np.load(npz_file, allow_pickle=True)
            all_windows.extend([
                (pd.Timestamp(start), pd.Timestamp(end)) 
                for start, end in data['valid_windows']
            ])
    
    # Merge all windows
    if not all_windows:
        return 0
        
    all_windows.sort()
    merged = [list(all_windows[0])]
    
    for current in all_windows[1:]:
        last = merged[-1]
        if current[0] <= last[1] + pd.Timedelta(seconds=1):  # Adjacent or overlapping
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))
    
    # Calculate total epochs
    return sum(
        int((end - start).total_seconds()) + 1 
        for start, end in merged
    )


def station_rms_full_process(
    slr_data, 
    outlier_threshold=10, 
    rms_limit=0.3, 
    filter_by_rms_plus_std=True, 
    plot=False
):
    """
    Filter SLR observations by determining station quality based only on ESA RDO data.
    Additionally, reject epochs with large RDO residuals for all methods.

    Parameters
    ----------
    slr_data : pd.DataFrame
        Full SLR residuals data (including multiple methods and reference data).
    outlier_threshold : float
        Residual threshold (in meters) to remove gross outliers.
    rms_limit : float
        RMS acceptance threshold (in meters).
    filter_by_rms_plus_std : bool
        If True, reject stations where RMS + STD exceeds RMS limit.
    plot : bool
        If True, generate the plots.

    Returns
    -------
    station_summary : pd.DataFrame
        Summary table with RMS, STD, number of observations, and status.
    slr_data_filtered_final : pd.DataFrame
        All input SLR data filtered to only accepted stations and valid epochs.
    """

    # Step 1: Filter ESA RDO data
    rdo_data = slr_data[(slr_data['Method'] == 'ESA') & (slr_data['Reference_Data'] == 'RDO')].copy()

    # Step 2: Identify "bad epochs" from RDO residuals
    bad_epochs = rdo_data[np.abs(rdo_data['Res[m]']) > outlier_threshold]['Date'].unique()

    # Step 4: Re-filter RDO after removing bad epochs
    rdo_data = slr_data[(slr_data['Method'] == 'ESA') & (slr_data['Reference_Data'] == 'RDO')].copy()

    # Clean station names
    rdo_data['Clean_Site'] = rdo_data['Site'].apply(lambda x: x[0:10].strip())

    # Apply RMS filtering
    rdo_filtered = rdo_data[np.abs(rdo_data['Res[m]']) <= outlier_threshold]

    # Per-station metrics
    rms_after = rdo_filtered.groupby('Site')['Res[m]'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index(name='RMS_after[m]')
    std_after = rdo_filtered.groupby('Site')['Res[m]'].std().reset_index(name='Std_after[m]')
    obs_after = rdo_filtered.groupby('Site').size().reset_index(name='Num_obs_after')

    # Combine metrics
    station_summary = rms_after.merge(std_after, on='Site').merge(obs_after, on='Site')
    station_summary['Clean_Site'] = station_summary['Site'].apply(lambda x: x[0:10].strip())

    # Apply RMS acceptance
    station_summary['Status'] = np.where(
        station_summary['RMS_after[m]'] <= rms_limit,
        'RMS Accepted',
        'Rejected (RMS too high)'
    )

    # Optional stricter rejection: RMS + STD > threshold
    if filter_by_rms_plus_std:
        station_summary['Status'] = np.where(
            (station_summary['RMS_after[m]'] <= rms_limit) &
            (station_summary['RMS_after[m]'] + station_summary['Std_after[m]'] > rms_limit),
            'Rejected (RMS+STD exceeds limit)',
            station_summary['Status']
        )

    # Sort for plotting
    station_summary = station_summary.sort_values('RMS_after[m]')

    if plot:
        # First barplot: Before and After filtering
        rms_before = rdo_data.groupby('Site')['Res[m]'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index(name='RMS_before[m]')
        rms_before = rms_before.merge(station_summary[['Site', 'Clean_Site']], on='Site', how='right')

        plt.figure(figsize=(8.5, 7))
        ax = plt.gca()
        sns.barplot(data=rms_before, x='Clean_Site', y='RMS_before[m]', color='lightgrey',
                    edgecolor='black', label='Before filtering', errorbar=None, ax=ax)

        num_patches_before = len(ax.patches)

        sns.barplot(data=station_summary, x='Clean_Site', y='RMS_after[m]', color='tab:blue',
                    edgecolor='black', alpha=0.8, label=f'After filtering (> {outlier_threshold} m removed)',
                    errorbar=None, ax=ax)

        bars_after = ax.patches[num_patches_before:]
        for bar, std in zip(bars_after, station_summary['Std_after[m]']):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=std,
                        fmt='none', ecolor='black', capsize=3, lw=1)

        plt.axhline(rms_limit, color='red', linestyle='--', linewidth=1.5, label=f'RMS threshold {rms_limit} m')
        plt.xlabel('Site')
        plt.ylabel('RMS (m)')
        plt.yscale('log')
        plt.xticks(rotation=45, ha='right')
        plt.legend(draggable=True)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Second barplot: Accepted only
        selected = station_summary[station_summary['Status'].str.contains('Accepted')].copy()
        plt.figure(figsize=(8.5, 7))
        ax = plt.gca()
        sns.barplot(data=selected, x='Clean_Site', y='RMS_after[m]', color='tab:blue',
                    edgecolor='black', errorbar=None, ax=ax)

        for bar, std in zip(ax.patches, selected['Std_after[m]']):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=std,
                        fmt='none', ecolor='black', capsize=3, lw=1)

        plt.text(0.2, 0.96, f'RMS threshold\n{rms_limit} m', transform=ax.transAxes,
                 fontsize=10, ha='right', va='top', color='red',
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

        plt.xlabel('Site')
        plt.ylabel('RMS (m)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Final filtering: accepted stations only
    accepted_stations = station_summary[station_summary['Status'].str.contains('Accepted')]['Site'].unique()
    
    # Step 3: Drop bad epochs from all methods
    slr_data = slr_data[~slr_data['Date'].isin(bad_epochs)]
    slr_data_filtered_final = slr_data[slr_data['Site'].isin(accepted_stations)].copy()

    return station_summary[['Site', 'Clean_Site', 'RMS_after[m]', 'Std_after[m]', 'Num_obs_after', 'Status']], slr_data_filtered_final
# SLR station functions

def project_orbit_std_to_los(sat_pos, std_xyz, station_pos):
    """
    Project orbit standard deviation along line-of-sight vector.

    Parameters
    ----------
    sat_pos : np.array
        Satellite position (3,) in ITRF [m].
    std_xyz : np.array
        Orbit standard deviations [std_x, std_y, std_z] in [m].
    station_pos : np.array
        Station position (3,) in ITRF [m].

    Returns
    -------
    sigma_los : float
        Projected 1-sigma orbit error along LOS [m].
    """
    los_vec = sat_pos - station_pos
    los_unit = los_vec / np.linalg.norm(los_vec)
    sigma_los = np.sqrt(
        los_unit[0] ** 2 * std_xyz[0] ** 2 +
        los_unit[1] ** 2 * std_xyz[1] ** 2 +
        los_unit[2] ** 2 * std_xyz[2] ** 2
    )
    return sigma_los

def process_slr_vs_orbit_with_sinex_old(slr_df, orbit_trajectory, station_handler):
    """
    Match SLR data and project orbit std along LOS using SINEX station positions.
    """
    los_sigmas = []

    slr_df = slr_df.copy()
    slr_df['Date'] = pd.to_datetime(slr_df['Date'])

    for idx, row in tqdm(slr_df.iterrows(), total=len(slr_df), desc="Processing SLR residuals vs orbit STD (SINEX)"):
        obs_time = row['Date']
        station_code = row['station_code']

        # Get satellite position and std at obs time
        sat_pos = orbit_trajectory[['x_pos', 'y_pos', 'z_pos']].reindex(orbit_trajectory.index.union([obs_time])).sort_index().interpolate('time').loc[obs_time].to_numpy()
        stds = orbit_trajectory[['std_x', 'std_y', 'std_z']].reindex(orbit_trajectory.index.union([obs_time])).sort_index().interpolate('time').loc[obs_time].to_numpy()

        # Get station position from SINEX handler
        try:
            station_pos = station_handler.get_station_xyz(station_code)
        except ValueError as e:
            print(f"Warning: {e}")
            los_sigmas.append(np.nan)
            continue

        # Project orbit std along LOS
        sigma_los = project_orbit_std_to_los(sat_pos, stds, station_pos)
        los_sigmas.append(sigma_los)

    slr_df['orbit_sigma_los[m]'] = los_sigmas
    return slr_df

def process_slr_vs_orbit_with_sinex(slr_df, orbit_trajectory, station_handler, order=8):
    """
    Vectorised: interpolate orbit trajectory to SLR epochs using Tudat-style interpolation,
    then project orbit standard deviation onto the line-of-sight using SINEX station positions.
    
    Parameters:
        slr_df : pd.DataFrame
            DataFrame containing SLR observations. Must include 'Date' and 'station_code'.
        orbit_trajectory : pd.DataFrame
            Orbit data with columns ['x_pos', 'y_pos', 'z_pos', 'std_x', 'std_y', 'std_z'].
        station_handler : object
            Handler exposing `.get_station_xyz(code)` for station ECEF positions.
        order : int
            Lagrange interpolation order (default: 8).
    
    Returns:
        pd.DataFrame
            Original SLR DataFrame with added 'orbit_sigma_los[m]' column.
    """
    # global interpolated_orbit#, station_codes, unique_epochs, sat_pos, station_xyz, orbit_input
    
    slr_df = slr_df.copy().dropna(subset=['Date', 'station_code'])
    slr_df['Date'] = pd.to_datetime(slr_df['Date'])

    # Unique interpolation epochs
    
    unique_epochs = slr_df['Date'][~slr_df['Date'].index.duplicated(keep='first')].copy()

    # Prepare orbit dataframe with dummy column
    orbit_input = orbit_trajectory.copy()

    # === Interpolate to SLR epochs ===
    interpolated_orbit = resample_with_gap_handling_with_epochs(
        orbit_input,
        frequency_or_epochs=unique_epochs,
        gap_threshold_seconds=999999,  # Ignored when passing exact epochs
        interpolator='Lagrange',
        order=order
    )

    # Ensure alignment and reindex to ensure strict matching
    interpolated_orbit['Date_Column'] = interpolated_orbit.index
    slr_df['Date_Column'] = slr_df.index
    merged = pd.merge(slr_df, interpolated_orbit, on='Date_Column', how='left')
    
    # === Merge satellite and station info ===
    sat_pos = merged[['x_pos', 'y_pos', 'z_pos']].to_numpy()
    sat_std = merged[['std_x', 'std_y', 'std_z']].to_numpy()
    station_codes = slr_df['station_code'].astype(str).to_numpy()

    # === Retrieve matching station positions ===
    station_xyz = np.full_like(sat_pos, np.nan)
    for i, code in enumerate(station_codes):
        if code in station_handler.station_codes.values:
            station_xyz[i] = station_handler.get_station_xyz(code)[['X [m]', 'Y [m]', 'Z [m]']].to_numpy()
        else:
            # Leave as NaN if station not found
            continue
    # Compute LOS unit vectors
    los_vectors = sat_pos - station_xyz
    los_norms = np.linalg.norm(los_vectors, axis=1, keepdims=True)
    los_unit = los_vectors / np.where(los_norms == 0, 1, los_norms)

    # Project variances along LOS
    variances = sat_std ** 2
    sigma_los_squared = np.sum(los_unit ** 2 * variances, axis=1)
    sigma_los = np.sqrt(sigma_los_squared)

    # Attach to DataFrame
    slr_df['orbit_sigma_los[m]'] = sigma_los
    return slr_df


# Analysis pipeline

def analyse_slr_vs_orbit(slr_data_filtered, orbit_trajectory, station_handler, label='Unknown'):
    print(f"\n--- Processing: {label} ---")

    # Project STD to LOS
    slr_data_filtered = slr_data_filtered[slr_data_filtered['Method'] == label]
    slr_data_with_sigma = process_slr_vs_orbit_with_sinex(slr_data_filtered, orbit_trajectory, station_handler)
    slr_data_with_sigma = slr_data_with_sigma.dropna(subset=['Res[m]', 'orbit_sigma_los[m]'])

    # Metrics
    slr_rms = np.sqrt(np.mean(slr_data_with_sigma['Res[m]']**2))
    mean_sigma_los = slr_data_with_sigma['orbit_sigma_los[m]'].mean()
    rms_ratio = slr_rms / mean_sigma_los
    slr_data_with_sigma['norm_residual'] = slr_data_with_sigma['Res[m]'] / slr_data_with_sigma['orbit_sigma_los[m]']
    reduced_chi2 = np.mean(slr_data_with_sigma['norm_residual']**2)

    # Report
    print(f"SLR residual RMS [m]: {slr_rms:.4f}")
    print(f"Mean orbit sigma LOS [m]: {mean_sigma_los:.4f}")
    print(f"Ratio (SLR RMS / Orbit STD LOS): {rms_ratio:.2f}")
    print(f"Reduced Chi-square: {reduced_chi2:.2f}")

    return {
        'Method': label,
        'SLR RMS [m]': slr_rms,
        'Mean STD LOS [m]': mean_sigma_los,
        'RMS/STD': rms_ratio,
        'Reduced Chi2': reduced_chi2
    }, slr_data_with_sigma


#%%
# -----------------------------
# 1. Load orbits and SINEX data
# -----------------------------
window_start = datetime(2023, 1, 1, 0)
window_stop = window_start + timedelta(days=60)
satellite_id = ['47']

# Load SINEX station data
sinex_file_dir = r'..\..\data\SLR\stations'
sinex_file_list = [os.path.join(sinex_file_dir, i) for i in os.listdir(sinex_file_dir) if i.endswith('.snx')]
station_handler = MultiSINEXStationHandler(sinex_file_list)

# -----------------------------
# 2. Load and prepare SLR data (filtered and station codes attached)
# -----------------------------

ilrs_mapping_csv = fr"{sinex_file_dir}\ilrs_station_mapping.csv"

output_dir = Path(r"rescaled_variance")

output_dir.mkdir(parents=True, exist_ok=True)
summary_file = output_dir / 'slr_orbit_variance_summary_jan2023.csv'

if summary_file.exists():
    print("Precomputed results exist.")

else:
    print("No precomputed results exist.")    
    print("Loading and cleaning SLR data...")
    slr_data = load_and_clean_data(np_csv_path, gap_dir=gap_filter_path, trim_seconds=11)
    slr_data['Date'] = pd.to_datetime(slr_data['Date'], format='ISO8061')
    slr_data_filtered = slr_data.set_index('Date', drop = False)
    slr_data_filtered = slr_data_filtered[slr_data_filtered['Sta [deg]'].astype(float) >= 10] #setting elevation cutoff
    # station_rms_summary, slr_data_filtered = station_rms_full_process(slr_data_filtered, outlier_threshold=10, rms_limit=0.3, plot=True)
    slr_data_filtered= slr_data_filtered[~slr_data_filtered['Reference_Data'].isin(['KO'])]
    # Filter stations by ESA RDO residuals
    station_rms_summary, slr_data_filtered = station_rms_full_process(
        slr_data_filtered, outlier_threshold=10, rms_limit=0.3)
    
    # Map station codes
    
    ilrs_station_name_to_code = load_ilrs_station_mapping_from_csv(ilrs_mapping_csv)
    slr_data_filtered = attach_ilrs_station_code(slr_data_filtered, ilrs_station_name_to_code)
    slr_data_filtered['station_code'] = slr_data_filtered['station_code'].astype(str)
    
    # -----------------------------
    # 3. Load all orbits and prepare input_orbits and method_labels lists
    # -----------------------------
    print("Loading orbits...")

    input_orbits = []
    method_labels = []
    
    # KO input orbits (can be multiple centres)
    input_centres = ['IFG', 'AIUB', 'TUD']  # Example for multiple KO inputs
    request_kos = AccessRequest(data_type='KO',
                                satellite_id=satellite_id,
                                analysis_centre=input_centres,
                                window_start=window_start,
                                window_stop=window_stop,
                                round_seconds=True,
                                get_data=False)
    
    input_kos_list = retrieve_arcs(request_kos)[satellite_id[0]]
    input_kos_list = [arc.trajectory for arc in input_kos_list]
    # scaling_factors = [2.739556, 1.679749, 0.182947] # empirical noise scaling factors CHI2
    
    # rescaled_input_kos_list = []
    # for scaling_factor, input_orb in zip(scaling_factors, input_kos_list):
    #     rescaled_orb = input_orb
    #     rescaled_orb[['std_x', 'std_y', 'std_z', 'cov_xy', 'cov_xz', 'cov_xt', 'cov_yz',
    #                          'cov_yt', 'cov_zt']] = rescaled_orb[['std_x', 'std_y', 'std_z', 'cov_xy', 'cov_xz', 'cov_xt', 'cov_yz',
    #                          'cov_yt', 'cov_zt']]*scaling_factor
    #     rescaled_input_kos_list.append(rescaled_orb)
    
    # Add KO orbits and labels
    input_orbits.extend(input_kos_list)
    method_labels.extend(input_centres)
    
    # Combined orbits
    methods_combined = ['mean', 'inversevariance', 'vce', 'residualweighted'] 
    
    request_combined = AccessRequest(data_type='CO',
                                     satellite_id=satellite_id,
                                     analysis_centre=methods_combined,
                                     window_start=window_start,
                                     window_stop=window_stop,
                                     round_seconds=False,
                                     get_data=False)
    
    input_combined_list = retrieve_arcs(request_combined)[satellite_id[0]]
    input_combined_list = [arc.trajectory for arc in input_combined_list]
    
    # Add combined orbits and labels
    input_orbits.extend(input_combined_list)
    method_labels.extend(methods_combined)
    input_orbits = [i.drop(columns=['clock_bias']) for i in input_orbits]
    # -----------------------------
    # 4. Filter orbits and SLR data to common time window
    # -----------------------------
    print("Filtering data to common time window...")
    
    common_start = max(input_orbits[0].index.min(), slr_data_filtered.index.min())
    common_stop = min(input_orbits[0].index.max(), slr_data_filtered.index.max())
    
    
    slr_data_filtered = slr_data_filtered[(slr_data_filtered.index >= common_start) & (slr_data_filtered.index <= common_stop)]
    input_orbits = [orbit[(orbit.index >= common_start) & (orbit.index <= common_stop)] for orbit in input_orbits]
    print(f"Filtered SLR points: {len(slr_data_filtered)}")

# -----------------------------
# 5. Smart loop: analyse, save results, save sigma data per method
# -----------------------------

if summary_file.exists():
    print("Loading precomputed results...")
    results_df = pd.read_csv(summary_file)
    print(results_df)
else:
    print("No precomputed results found. Running full analysis...")

    results = []

    for orbit_df, label in zip(input_orbits, method_labels):
        result, slr_data_with_sigma = analyse_slr_vs_orbit(
            slr_data_filtered, orbit_df, station_handler, label=label
        )
        results.append(result)
        slr_data_with_sigma.to_pickle(output_dir / f'slr_data_with_sigma_{label}.pkl')

    results_df = pd.DataFrame(results)
    results_df.to_csv(summary_file, index=False)
    print(f"Results saved to {summary_file}")


#%%

method_labels = [
                 'IFG', 'AIUB', 'TUD',  
                 'IFGRS', 'AIUBRS', 'TUDRS',
                 # 'IFGRR', 'AIUBRR', 'TUDRR',
                 'mean', 'inversevariance', 'vce', 'residualweighted', 
                 'meanRS', 'inversevarianceRS', 'vceRS', 'residualweightedRS', 
                 # 'meanRR', 'vceRR', 'inversevarianceRR', 'residualweightedRR'
                 ]

excluded_dates = ['2023-1-26', '2023-2-2', '2023-2-15', ]
results_dir = output_dir
loaded_data, metrics_df = load_sigma_data_and_metrics(method_labels, results_dir, excluded_dates=excluded_dates)
output_dir_fig = r"rescaled_variance\figures"
# Plot all
# plot_histograms_norm_residuals(loaded_data, output_dir)
plot_reduced_chi2_bar(metrics_df)#, output_dir_fig=None)

# plot_rms_vs_chi2(metrics_df, output_dir_fig)
# plot_combined_normalised_histograms(loaded_data, output_dir_fig)
#%%
# main_methods = ['mean', 'inversevariance', 'vce', 'residualweighted']
# input_methods = ['IFG', 'TUD', 'AIUB']

main_methods = [
                'mean', 'inversevariance', 'vce', 'residualweighted', 
                 'meanRS', 'inversevarianceRS', 'vceRS', 'residualweightedRS', 
                 # 'meanRR', 'vceRR', 'inversevarianceRR', 'residualweightedRR'
                 ]

input_methods = [
                # 'IFG', 'AIUB', 'TUD',  
                'IFGRS', 'AIUBRS', 'TUDRS',
                # 'IFGRR', 'AIUBRR', 'TUDRR'
                ]


# plot_residuals_with_sigma(loaded_data, main_methods, output_dir)
# plot_residuals_with_sigma_daily(loaded_data, main_methods, output_dir_fig)
plot_residuals_with_sigma_daily(loaded_data, input_methods, output_dir_fig)
#%%
# plot_residual_spread_vs_orbit_sigma(loaded_data, main_methods, output_dir_fig)
plot_residual_spread_vs_orbit_sigma(loaded_data, input_methods, output_dir_fig)
# plot_residuals_with_sigma(loaded_data, appendix_methods, output_dir)

#%%
# plot_daily_rms_to_std_ratio(loaded_data, output_dir)
plot_combined_daily_rms_to_std_ratio(loaded_data, output_dir_fig)

# %%
final_metrics_df = compute_variance_validation_metrics(loaded_data)
print(final_metrics_df)

# %%
loaded_data_og = {k: v for k, v in loaded_data.items() if "RS" not in k}
loaded_data_rs = {k: v for k, v in loaded_data.items() if "RS" in k}
rms_std_stats_df = compute_daily_rms_std_stats(loaded_data)
plot_rms_std_stability(loaded_data_og)
plot_rms_std_stability(loaded_data_rs)
print(rms_std_stats_df)