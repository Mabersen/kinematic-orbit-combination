# -*- coding: utf-8 -*-
"""
process_normal_points.py
 
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
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
from scipy.stats import norm
import seaborn as sns
from pathlib import Path
import itertools

# --------------------------
# Utility functions
# --------------------------
def get_distinct_colours(n):
    """Return a list of n visually distinct colours."""
    cmap = cm.get_cmap('tab20b', n)  # Choose a perceptually uniform colormap
    return [to_hex(cmap(i)) for i in range(n)]

def get_marker_colour_mapping(categories):
    """
    Assign unique colour/marker pairs to each category.
    Prioritise unique colours first, then pair with markers if needed.
    """
    base_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*']
    n = len(categories)
    
    # Get enough colours
    colour_list = get_distinct_colours(n)

    # If we have fewer markers than colours, reuse the default marker with different colours
    # If we run out of unique colours, start combining with markers
    style_combos = []
    if n <= len(base_markers):
        # Only use markers
        style_combos = [{'color': colour_list[i], 'marker': base_markers[i]} for i in range(n)]
    else:
        # Create unique colour-marker pairs
        marker_cycle = itertools.cycle(base_markers)
        for i in range(n):
            style_combos.append({'color': colour_list[i], 'marker': next(marker_cycle)})

    return dict(zip(categories, style_combos))

#%%
# --------------------------
# Data Loading & Cleaning
# --------------------------

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


def load_and_clean_data(filepath, gap_dir=None, trim_seconds=11, rounded=False):
    """Load SLR data, clean it, filter by gaps, and add total epochs metrics."""
    df = pd.read_csv(filepath, dtype=str)
    df[['rms[m]', 'Range [m]', 'n', 'Res[m]']] = df[['rms[m]', 'Range [m]', 'n', 'Res[m]']].astype('float')

    df["Date"] = pd.to_datetime(df["Date"], format='mixed')
    df['rej'] = df['rej'].str.strip()
    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
    df['Reference_Data'] = df['Reference_Data'].fillna('None')
    
    df = df.dropna(subset=['Res[m]'])

    df['total_epochs'] = 0
    df['Total_Possible_Epochs'] = 0
    df['Total_Available_Epochs'] = 0
    
    if gap_dir is not None:
        df, total_epochs_per_method_ref = filter_by_gaps(df, gap_dir, trim_seconds)

        # Total_Possible_Epochs from KONF (same as before)
        if not rounded:
            if 'KONF' in df['Reference_Data'].unique():
                total_epochs = compute_total_konf_epochs_optimized(gap_dir)
                df['Total_Possible_Epochs'] = np.where(
                    (df['Reference_Data'] == 'KONF') &
                    (df['Method'].isin(['AIUB', 'TUD', 'IFG'])),
                    total_epochs,
                    np.nan
                )
    
        else:
            # COMMENT FOR NON-ROUNDED DATA    
            if 'KO' in df['Reference_Data'].unique():
                print('USING KO TO COMPUTE TOTAL POSSIBLE EPOCHS, ONLY FOR ROUNDED TESTS')
                total_epochs = compute_total_konf_epochs_optimized(gap_dir)
                df['Total_Possible_Epochs'] = np.where(
                    (df['Reference_Data'] == 'KO') &
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


# -------------------------- STATION SELECTION BY RDO --------------------------
def compute_station_rms_rdo(slr_data, outlier_threshold=10, rms_limit=0.3):
    """
    Calculate station-wise RMS for RDO validation data, after outlier filtering.
    
    Parameters
    ----------
    slr_data : pd.DataFrame
        Full SLR DataFrame including 'Res[m]', 'Method', 'Reference_Data', 'Site'.
    outlier_threshold : float, optional
        Residual threshold (in meters) to remove gross outliers (default is 10 m).
    rms_limit : float, optional
        RMS threshold (in meters). Stations with higher RMS will be excluded.
    
    Returns
    -------
    filtered_station_rms : pd.DataFrame
        DataFrame of station-wise RMS, only stations with RMS <= rms_limit.
    accepted_stations : list
        List of station codes that passed the RMS threshold.
    """
    # Select only RDO SLR data
    rdo_data = slr_data[(slr_data['Method'] == 'ESA') & (slr_data['Reference_Data'] == 'RDO')]

    # Filter outliers
    rdo_data_filtered = rdo_data[np.abs(rdo_data['Res[m]']) <= outlier_threshold]

    # Calculate station-wise RMS
    station_rms = (
        rdo_data_filtered.groupby('Site')['Res[m]']
        .apply(lambda x: np.sqrt(np.mean(x**2)))
        .reset_index(name='RMS[m]')
    )

    # Apply RMS threshold
    filtered_station_rms = station_rms[station_rms['RMS[m]'] <= rms_limit].copy()
    accepted_stations = filtered_station_rms['Site'].tolist()

    return filtered_station_rms, accepted_stations


# -------------------------- TIME RESTRUCTURING --------------------------------

def compute_filtered_konf_epochs(gap_dir, time_config=None):
    """Compute theoretical epochs within specified time filters"""

    # Parse time config
    window = time_config.get('window', {}) if time_config else {}
    start_date = pd.to_datetime(window.get('start_date')) if 'start_date' in window else None
    end_date = pd.to_datetime(window.get('end_date')) if 'end_date' in window else None
    hours = time_config.get('hours') if time_config else None
    days_of_week = time_config.get('days_of_week') if time_config else None
    excluded_dates = pd.to_datetime(time_config.get('excluded_dates', [])) if time_config else []

    methods = ['AIUB', 'TUD', 'IFG']
    all_windows = []

    # Load and filter gap files
    for method in methods:
        for npz_file in Path(gap_dir).glob(f"{method}_*.npz"):
            # Extract date from filename
            parts = npz_file.stem.split('_')
            year, doy = int(parts[2]), int(parts[3])
            file_date = pd.to_datetime(f"{year}-{doy}", format='%Y-%j')

            # Date filtering
            if start_date and file_date < start_date:
                continue
            if end_date and file_date >= end_date:
                continue

            # Load windows
            data = np.load(npz_file, allow_pickle=True)
            if time_config is not None:
                for start, end in data['valid_windows']:
                    start = pd.Timestamp(start)
                    end = pd.Timestamp(end)
    
                    # Apply time filters to each window
                    filtered_start, filtered_end = apply_time_filters(
                        start, end,
                        hours=hours,
                        days_of_week=days_of_week,
                        excluded_dates=excluded_dates
                    )
                    
                    if filtered_start < filtered_end:
                        all_windows.append((filtered_start, filtered_end))

    # Merge and calculate total epochs
    if not all_windows:
        return 0

    all_windows.sort()
    merged = [list(all_windows[0])]
    
    for current in all_windows[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))
    
    return sum(int((end - start).total_seconds()) + 1 for start, end in merged)

def apply_time_filters(start, end, hours=None, days_of_week=None, excluded_dates=None):
    """Adjust window boundaries based on time filters"""
    # Convert to pandas timestamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Excluded dates check
    if excluded_dates is not None:
        # if not excluded_dates.empty:
            day_span = pd.date_range(start.normalize(), end.normalize(), freq='D')
            if any(date in excluded_dates for date in day_span):
                return pd.Timestamp.min, pd.Timestamp.min


    # Days of week filter
    if days_of_week is not None:
        window_days = pd.date_range(start.floor('D'), end.ceil('D'), freq='D')
        if not any(d.weekday() in days_of_week for d in window_days):
            return pd.Timestamp.min, pd.Timestamp.min

    # Time of day filter
    if hours is not None:
        start_time = max(start, start.floor('D') + pd.Timedelta(hours=hours[0]))
        end_time = min(end, start.floor('D') + pd.Timedelta(hours=hours[1]))
        return start_time, end_time

    return start, end

def filter_time_window(df, start_date=None, end_date=None,
                       start_hour=None, end_hour=None,
                       days_of_week=None):
    """
    Filter data by:
    - Absolute date range (start_date/end_date)
    - Time of day (start_hour/end_hour)
    - Days of week (0=Monday to 6=Sunday)
    """
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    if start_hour is not None or end_hour is not None:
        hours = df['Date'].dt.hour
        if start_hour is not None:
            df = df[hours >= start_hour]
        if end_hour is not None:
            df = df[hours <= end_hour]

    if days_of_week:
        df = df[df['Date'].dt.dayofweek.isin(days_of_week)]

    return df


def create_time_groups(df, frequency='D', custom_hours=None):
    """
    Create time groups with flexible periods:
    - Standard: 'H'(hourly), 'D'(daily), 'W'(weekly), 'M'(monthly), 'Y'(yearly)
    - Custom hours: e.g., {'morning': (6,12), 'afternoon': (12,18)}
    """
    if custom_hours:
        df['Time_Group'] = 'other'
        for group_name, (start_h, end_h) in custom_hours.items():
            mask = (df['Date'].dt.hour >= start_h) & (df['Date'].dt.hour < end_h)
            df.loc[mask, 'Time_Group'] = group_name
        return df

    freq_map = {
        'H': df['Date'].dt.strftime('%Y-%m-%d %H:00'),
        'D': df['Date'].dt.date,
        'W': df['Date'].dt.to_period('W').astype(str),
        'M': df['Date'].dt.to_period('M').astype(str),
        'Y': df['Date'].dt.year
    }

    df['Time_Group'] = freq_map.get(frequency, frequency)
    return df


def time_restructure(slr_data, time_config=None, gap_dir = None):
    """
    Enhanced pipeline with dynamic time grouping
    time_config = {
        'window': {'start_date': '2023-01-01', 'end_date': '2023-06-30'},
        'hours': {'morning': (6,12), 'afternoon': (12,18)},
        'days_of_week': [0,1,2,3,4]  # Weekdays only,
        'excluded_dates': ['2023-01-01', '2023-01-19']
    }
    """
    # Apply time filters
    if time_config:
        filter_args = time_config.get('window', {})
        if 'hours' in time_config or 'days_of_week' in time_config:
            filter_args.update({
                'start_hour': None,
                'end_hour': None,
                'days_of_week': time_config.get('days_of_week')
            })
            if 'hours' in time_config:
                # Will be handled in create_time_groups
                pass

        slr_data = filter_time_window(slr_data, **filter_args)

    # Create time groups
    if time_config and 'hours' in time_config:
        slr_data = create_time_groups(slr_data, custom_hours=time_config['hours'])
    else:
        freq = time_config.get('frequency', 'M') if time_config else 'M'
        slr_data = create_time_groups(slr_data, frequency=freq)
        
    if time_config and 'excluded_dates' in time_config:
        excluded_dates = time_config.get('excluded_dates', [])
        print(excluded_dates)
        excluded_date = pd.to_datetime(excluded_dates)
        # Normalize the DateTimeIndex to remove time components and check inclusion
        mask = slr_data['Date'].dt.normalize().isin(excluded_dates)
        # Use .dt.normalize() to ignore time components
        # slr_data = slr_data[~slr_data['Date'].dt.normalize().isin(excluded_dates)]
        # Invert the mask and filter the DataFrame
        slr_data = slr_data[~mask]
        
    # Add epoch recalculation if KONF data exists
    if gap_dir is not None and 'KONF' in slr_data['Reference_Data'].unique() and time_config:
        total_epochs = compute_filtered_konf_epochs(gap_dir, time_config)
        slr_data['Total_Possible_Epochs'] = np.where(
            (slr_data['Reference_Data'] == 'KONF') &
            (slr_data['Method'].isin(['AIUB', 'TUD', 'IFG'])),
            total_epochs,
            np.nan
        )
        
        _ , total_epochs_per_method_ref = filter_by_gaps(slr_data, gap_dir)

        # Add Total_Available_Epochs per method + reference_data using combined keys
        def get_total_available_epochs(row):
            method = row['Method']
            ref_data = row['Reference_Data']
            return total_epochs_per_method_ref.get(method, {}).get(ref_data, np.nan)

        slr_data['Total_Available_Epochs'] = slr_data.apply(get_total_available_epochs, axis=1)
    
    slr_data['Method'] = slr_data['Method'].replace({
        'inversevariance': 'inverse variance',
        'residualweighted': 'residual-weighted',
        'neldermead': 'nelder-mead'
    })
    
    slr_data['Method_Label'] = slr_data['Method'] + ' (' + slr_data['Reference_Data'].astype(str) + ')'
    return slr_data

# --------------------------------------------------------------------------------------------------------
# --------------------------
# Metric Calculations
# --------------------------


def calculate_rms_metrics(df):
    """Calculate metrics with proper epoch summation"""
    # Get RDO baseline
    try:
        res_RDO = df[df['Method'] == 'ESA']['Res[m]']
        unweighted_rms_RDO = np.sqrt(np.mean(res_RDO**2))
    except:
        unweighted_rms_RDO = 0
    
    # Pre-calculate total possible epochs once
    total_possible = df['Total_Possible_Epochs'].dropna().max() if 'Total_Possible_Epochs' in df else np.nan
    
    def calculate_metrics(g):
        res = g['Res[m]']
        n = g['n']
        rms = g['rms[m]']
        
        # PROPER EPOCH CALCULATION
        # Group by date first to avoid double-counting
        try:
            daily_epochs = g['Total_Available_Epochs'].dropna().iloc[0]  # if you're worried about NaNs#g.groupby(g['Date'].dt.date)['total_epochs'].first().sum()
        except(IndexError):
            print('No max epoch data')
            daily_epochs = 0
        return pd.Series({
            # RMS metrics
            'Unweighted_RMS': np.sqrt(np.mean(res**2)),
            'Weighted_RMS_n': np.nan, # np.sqrt(np.average(res**2, weights=n)),
            'Weighted_RMS_rms': np.nan, # np.sqrt(np.average(res**2, weights=1 / (rms**2))),
            'Weighted_RMS_combined': np.nan, # np.sqrt(np.average(res**2, weights=n / (rms**2))),
            'Unweighted_RMS_wrt_RDO': np.nan, # np.sqrt(np.mean(res**2)) - unweighted_rms_RDO,
            
            # Observation counts
            'Sample_Size': len(g),
            
            # Epoch metrics (corrected)
            'Total_Epochs': daily_epochs,  # Correct daily sum
            'Total_Possible_Epochs': total_possible,  # Theoretical max
            
            # Residual stats
            'Mean_Residual': res.mean(),
            'Std_Residual': res.std()
        })
    
    metrics = (df.groupby(['Method', 'Reference_Data'])
               .apply(calculate_metrics)
               .reset_index())
    
    # Add combined metric
    rms_cols = ['Unweighted_RMS', 'Weighted_RMS_n',
                'Weighted_RMS_rms', 'Weighted_RMS_combined']
    # metrics['Unweighted_RMS'] = metrics['Unweighted_RMS'] * 100 # Converting to cm
    metrics['RMS_of_4_Metrics'] = np.nan#np.sqrt(metrics[rms_cols].pow(2).mean(axis=1))
    metrics['Orbit_Solution'] = metrics['Method'] + ' (' + metrics['Reference_Data'].astype(str) + ')'

    return metrics.sort_values('Unweighted_RMS')

# --------------------------
# Visualization Functions
# --------------------------
# --------------------------
# Station filtering
# --------------------------

def plot_station_rms_full_process(
    slr_data, 
    outlier_threshold=10, 
    rms_limit=0.3, 
    filter_by_rms_plus_std=True, 
    plot=True
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


def plot_method_comparison(metrics_df, metric='Unweighted_RMS', station=None, axis_split=True, width = 13):
    """Plot RMS comparison with optional manual broken y-axis."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # Classify orbit type
    def get_orbit_type(row):
        ref_data = row['Reference_Data']
        method = row['Method']
        if ref_data == 'KONF': return 'Unfiltered Input'
        if ref_data == 'KO': return 'Filtered Input'
        if ref_data == 'KORS': return 'Noise-rescaled Input'
        if ref_data == 'NONE': return 'Independent Combination'
        if ref_data == 'ESA': return 'Reference Combination'
        if method == 'ESA': return 'Reduced-Dynamic Orbit'
        return 'Other'

    metrics_df['Orbit_Type'] = metrics_df.apply(get_orbit_type, axis=1)
    metrics_df['Epoch_Loss_Pct'] = 100 - (metrics_df['Total_Epochs'] / 
                                          metrics_df['Total_Possible_Epochs'] * 100).round(1)
    print(100 - (metrics_df['Total_Epochs'] / 
                                          metrics_df['Total_Possible_Epochs'] * 100).round(1))

    break_threshold = metrics_df[metric].sort_values().iloc[0:-1].max()
    y_max = metrics_df[metric].max()
    needs_break = y_max > break_threshold
    needs_break = True if axis_split else False
    colors = {
        'Unfiltered Input': '#1f77b4', 
        'Filtered Input': '#ff7f0e',
        'Noise-rescaled Input': '#d62728', 
        'Independent Combination': '#2ca02c', 
        'Reference Combination': '#d62728',
        'Reduced-Dynamic Orbit': '#9467bd',
        'Other': 'grey'        
    }


    x_pos = np.arange(len(metrics_df))
    bar_colors = metrics_df['Orbit_Type'].map(colors)
    if needs_break and axis_split:
        fig, (ax_high, ax_low) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(width, 6),
            gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05}
        )

        # Top axis (high values)
        ax_high.bar(x_pos, metrics_df[metric], color=bar_colors)
        ax_high.set_ylim(y_max - 0.2, y_max + 0.5)
        ax_high.spines['bottom'].set_visible(False)
        ax_high.tick_params(labelbottom=False)
        # ax_high.grid(True)
        
        # Bottom axis (low values)
        ax_low.bar(x_pos, metrics_df[metric], color=bar_colors)
        ax_low.set_ylim(0, break_threshold * 1.2)
        ax_low.spines['top'].set_visible(False)
        ax_low.tick_params(labeltop=False)

        # Diagonal break marks
        d = .012
        kwargs = dict(transform=ax_high.transAxes, color='k', clip_on=False)
        ax_high.plot((-d, +d), (-d, +d), **kwargs)
        ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_low.transAxes)
        ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


        axes = [ax_low, ax_high]

    else:
        fig, ax = plt.subplots(figsize=(width, 6))
        ax.bar(x_pos, metrics_df[metric], color=bar_colors)
        ax.set_ylim(0, y_max * 1.1)
        axes = [ax]  # single axis for unified handling
        ax_low = ax  # unify reference for legend, ticks, etc.

    # Add bar annotations
    for idx, val in enumerate(metrics_df[metric]):
        loss_pct = metrics_df.iloc[idx]['Epoch_Loss_Pct']
        if not np.isnan(loss_pct):
            if needs_break and axis_split:
                ax = ax_low if val <= break_threshold else ax_high
                y_offset = 0.04 * ax.get_ylim()[1] if ax is ax_low else 0.02 * ax.get_ylim()[1]
            else:
                ax = ax_low
                y_offset = 0.04 * ax.get_ylim()[1]
            
            ax.annotate(f"-{loss_pct:.1f}%",
                        (x_pos[idx], val + y_offset),
                        ha='center',
                        fontsize=13,
                        color='black')

    # Common bottom axis config
    ax_low.set_xticks(x_pos)
    ax_low.set_xticklabels(metrics_df['Method'], rotation=30, ha='right', rotation_mode='anchor')
    
    # ax_low.set_xticklabels(metrics_df['Method'], rotation=45, ha='right')
    ax_low.set_ylabel(f'{metric.replace("_", " ")} [m]')
    ax_low.grid(axis='y', alpha=0.3)
    
    if axis_split:
        ax_high.grid(axis='y', alpha=0.3)

    # [ax.set_yscale('log') for ax in axes]

    # Only get the orbit types actually present in this plot
    orbit_types_present = metrics_df['Orbit_Type'].unique()
    
    # Build legend only for present types
    legend_elements = [
        Patch(facecolor=colors[k], label=k) for k in orbit_types_present if k in colors
    ] + [
        Line2D([0], [0], marker='', color='none',
               label='Percentage labels show\nepoch loss relative\nto total possible',
               markerfacecolor='dimgrey', markersize=8)
    ]
    
    # if needs_break and axis_split:
    #     legend_elements.append(Line2D([0], [0], color='k', lw=1,
    #                            label=f'Y-axis break at {break_threshold:.2f} m'))


    ax_low.legend(handles=legend_elements, #bbox_to_anchor=(1.05, 1),
                  frameon=True, fontsize=13, draggable = True)
    
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.25)  # Adjust as needed (try between 0.2 - 0.3)
    # fig.subplots_adjust(left=0.08, right=0.98)  # Adjust as needed
    
    plt.show()

def plot_method_comparison_log(metrics_df, metric='Unweighted_RMS', width=13):
    """
    Plot method comparison with log y-axis, no error bars, and annotations for epoch loss.
    """
    # Classify orbit types
    def get_orbit_type(row):
        ref_data = row['Reference_Data']
        method = row['Method']
        if ref_data == 'KONF': return 'Unfiltered Input'
        if ref_data == 'KO': return 'Filtered Input'
        if ref_data == 'KORS': return 'Noise-rescaled Input'
        if ref_data == 'NONE': return 'Independent Combination'
        if ref_data == 'ESA': return 'Reference Combination'
        if method == 'ESA': return 'Reduced-Dynamic Orbit'
        return 'Other'

    metrics_df['Orbit_Type'] = metrics_df.apply(get_orbit_type, axis=1)
    metrics_df['Epoch_Loss_Pct'] = 100 - (metrics_df['Total_Epochs'] /
                                          metrics_df['Total_Possible_Epochs'] * 100).round(1)

    colors = {
        'Unfiltered Input': '#1f77b4', 
        'Filtered Input': '#ff7f0e',
        'Noise-rescaled Input': '#d62728', 
        'Independent Combination': '#2ca02c', 
        'Reference Combination': '#d62728',
        'Reduced-Dynamic Orbit': '#9467bd',
        'Other': 'grey'        
    }

    x_pos = np.arange(len(metrics_df))
    bar_colors = metrics_df['Orbit_Type'].map(colors)
    bar_heights = metrics_df[metric]

    fig, ax = plt.subplots(figsize=(width, 6))
    bars = ax.bar(x_pos, bar_heights, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Annotate loss
    for idx, (height, loss_pct) in enumerate(zip(bar_heights, metrics_df['Epoch_Loss_Pct'])):
        if not np.isnan(loss_pct):
            y_offset = height * 0.2
            ax.annotate(f"-{loss_pct:.1f}%", 
                        (x_pos[idx], height + y_offset), 
                        ha='center', fontsize=12, color='black')

    # Set log scale and add readable ticks
    ax.set_yscale('log')

    # Define y-ticks explicitly (adjust as needed)
    yticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5, 10]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks])
    ax.set_ylim(0.9 * bar_heights.min(), 2.2 * bar_heights.max())

    ax.set_ylabel(f'{metric.replace("_", " ")} [m]')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_df['Method'], rotation=30, ha='right', rotation_mode='anchor')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)

    # Legend top-left
    used_types = metrics_df['Orbit_Type'].unique()
    legend_elements = [Patch(facecolor=colors[k], label=k) for k in used_types if k in colors]
    legend_elements.append(Patch(facecolor='none', edgecolor='none',
                                 label='Labels show % epoch loss'))
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_method_comparison_grouped(metrics_df, metric='Unweighted_RMS', width=7):
    """
    Plot method comparison with side-by-side bars for KO/KONF pairs.
    """
    # Filter only relevant input orbits
    subset_df = metrics_df[metrics_df['Reference_Data'].isin(['KO', 'KONF'])].copy()
    subset_df['Epoch_Loss_Pct'] = 100 - (subset_df['Total_Epochs'] /
                                         subset_df['Total_Possible_Epochs'] * 100).round(1)

    # Sort to ensure KO always comes before KONF
    subset_df['variant_order'] = subset_df['Reference_Data'].map({'KO': 0, 'KONF': 1})
    subset_df.sort_values(['Method', 'variant_order'], inplace=True)

    methods = subset_df['Method'].unique()
    variants = ['KO', 'KONF']
    variant_labels = {'KO': 'Filtered', 'KONF': 'Unfiltered'}
    colors = {'KO': '#ff7f0e', 'KONF': '#1f77b4'}

    bar_width = 0.35
    x = np.arange(len(methods))  # One tick per method

    fig, ax = plt.subplots(figsize=(width, 6))

    # Plot each variant side-by-side
    for i, variant in enumerate(variants):
        data = subset_df[subset_df['Reference_Data'] == variant]
        heights = data[metric].values
        epoch_loss = data['Epoch_Loss_Pct'].values
        pos = x + (i - 0.5) * bar_width

        bars = ax.bar(pos, heights, width=bar_width, label=variant_labels[variant],
                      color=colors[variant], edgecolor='black', linewidth=0.5)

        # Annotate epoch loss
        for j, (h, loss) in enumerate(zip(heights, epoch_loss)):
            if not np.isnan(loss):
                ax.annotate(f"-{loss:.1f}%", (pos[j], h * 1.1),
                            ha='center', va='bottom', fontsize=11, color='black')

    # Log scale
    ax.set_yscale('log')
    yticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5, 10]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks])
    ax.set_ylim(0.9 * subset_df[metric].min(), 2.2 * subset_df[metric].max())

    # Labels
    ax.set_ylabel(f"{metric.replace('_', ' ')} [m]")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    # ax.set_title("Comparison of Filtered (KO) and Unfiltered (KONF) Inputs")

    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)

    # Legend
    ax.legend(title='Variant', loc='upper left', fontsize=12, draggable=True)

    plt.tight_layout()
    plt.show()


def plot_monthly_trends(df, metric_name='Unweighted_RMS'):
    """Plot monthly trends for specified metric with consistent formatting."""

    valid_metrics = ['Unweighted_RMS', 'Weighted_RMS_combined']
    if metric_name not in valid_metrics:
        raise ValueError(f"Invalid metric name. Choose from: {valid_metrics}")

    # Metric calculation
    def calculate_monthly_metrics(group):
        return pd.Series({
            'Unweighted_RMS': np.sqrt(np.mean(group['Res[m]'] ** 2)),
            'Weighted_RMS_combined': np.sqrt(
                np.average(group['Res[m]'] ** 2, weights=group['n'] / (group['rms[m]'] ** 2))),
            'Sample_Size': len(group)
        })

    # Monthly metrics
    monthly_metrics = (
        df.groupby(['Method', 'Reference_Data', 'Month_Year'])
        .apply(calculate_monthly_metrics)
        .reset_index()
    )

    # Create a combined label for plotting
    monthly_metrics['Method_Label'] = (
        monthly_metrics['Method'] + ' (' + monthly_metrics['Reference_Data'] + ')'
    )

    # Marker selection
    def get_marker(ref_data):
        if 'KONF' in ref_data: return 'o'
        if 'KO' in ref_data: return 's'
        if 'KORS' in ref_data: return '>'
        if 'NONE' in ref_data or 'ESA' in ref_data: return 'x'
        if 'RDO' in ref_data: return '^'
        return 'Other'  # fallback

    # Plot setup
    plt.figure(figsize=(14, 8))
    plt.yscale('log')

    for label in monthly_metrics['Method_Label'].unique():
        df_sub = monthly_metrics[monthly_metrics['Method_Label'] == label]
        ref_data = label.split('(')[-1].split(')')[0]

        plt.plot(
            df_sub['Month_Year'], df_sub[metric_name],
            label=label,
            marker=get_marker(ref_data),
            markersize=6,
            linestyle='-',
            linewidth=0.5,
            alpha=0.7
        )

    plt.title(f'Monthly {metric_name.replace("_", " ")} Trends (Log Scale)')
    plt.ylabel(f'{metric_name.replace("_", " ")} [m]')
    plt.xlabel('Month')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def split_into_daily_chunks(weekly_orbit, freq = 'D'): # OLD
    """Split DataFrame into daily chunks preserving full days"""
    out = [daily_df for _, daily_df in weekly_orbit.groupby(pd.Grouper(freq=freq))]
    return [i for i in out if not i.empty]


def calculate_daily_metrics(daily_chunks):
    """Calculate all RMS metrics for each daily chunk"""
    daily_data = []
    for daily_df in daily_chunks:
        if daily_df.empty:
            continue

        # Get date from the first entry
        date = daily_df.index[0].floor('D')

        # Group by method and calculate all metrics
        grouped = daily_df.groupby(['Method', 'Reference_Data'])
        res_RDO = daily_df[daily_df['Method'] == 'ESA']['Res[m]']
        unweighted_rms_RDO = np.sqrt(np.mean(res_RDO**2))

        for (method, ref), group in grouped:
            if len(group) < 1:
                continue  # Skip empty groups

            res = group['Res[m]']
            n = group['n']
            rms = group['rms[m]']

            # Calculate all RMS variants
            unweighted = np.sqrt(np.mean(res**2))
            weighted_n = np.nan
            weighted_rms = np.nan
            weighted_combined = np.nan
            rms_of_4 = np.nan

            # Additional metrics
            mean_res = res.mean()
            std_res = res.std()

            unweighted_rms_wrt_RDO = unweighted - unweighted_rms_RDO

            daily_data.append({
                'Date': date,
                'Method_Label': f"{method} ({ref})",
                'Unweighted_RMS': unweighted,
                'Weighted_RMS_n': weighted_n,
                'Weighted_RMS_rms': weighted_rms,
                'Weighted_RMS_combined': weighted_combined,
                'RMS_of_4_Metrics': rms_of_4,
                'Unweighted_RMS_wrt_RDO': unweighted_rms_wrt_RDO,
                'Residual_Mean': mean_res,
                'Residual_STD': std_res,
                'Observation_Count': len(group)
            })

    return pd.DataFrame(daily_data)


def plot_daily_rms_trend(daily_metrics_df, metric_name='Unweighted_RMS'):
    """Plot daily trends for specified metric with unique colour/marker combos."""
    valid_metrics = ['Unweighted_RMS', 'Weighted_RMS_n',
                     'Weighted_RMS_rms', 'Weighted_RMS_combined',
                     'RMS_of_4_Metrics', 'Unweighted_RMS_wrt_RDO']
    
    if metric_name not in valid_metrics:
        raise ValueError(f"Invalid metric name. Choose from: {valid_metrics}")
    
    # Create marker mapping based on reference data
    def get_marker(ref_data):
        if 'KONF' in ref_data: return 'o'
        if 'KO' in ref_data: return 's'
        if 'KORS' in ref_data: return '>'
        if 'NONE' in ref_data or 'ESA' in ref_data: return 'x'
        if 'RDO' in ref_data: return '^'
        return 'o'  # default
    
    plt.figure(figsize=(13, 6))
    # plt.yscale('log')  # Set log scale

    for label in daily_metrics_df['Method_Label'].unique():
        df_sub = daily_metrics_df[daily_metrics_df['Method_Label'] == label]
        ref_data = label.split('(')[-1].split(')')[0]  # Extract reference data
        
        plt.plot(
            df_sub['Date'], df_sub[metric_name],
            label=label,
            marker=get_marker(ref_data),
            markersize=3,
            linestyle='-',
            linewidth=0.5,
            alpha=0.7
        )
    
    # plt.title(f'Daily {metric_name.replace("_", " ")} Trends (Log Scale)')
    plt.ylabel(f'{metric_name.replace("_", " ")} [m]')
    plt.xlabel('Date')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_rms_vs_epoch_loss_with_inset(metrics_df, zoom_xlim=(0, 4), zoom_ylim=(0,1)):
    """
    Plot RMS vs Epoch Loss (%) with a zoomed inset focusing on the low loss / low RMS area.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe with columns 'Total_Epochs', 'Total_Possible_Epochs', 'Unweighted_RMS', 'Orbit_Solution'.
    zoom_xlim : tuple
        X-axis limits for the zoomed inset (default (0, 20)).
    zoom_ylim : tuple or None
        Y-axis limits for the zoomed inset. If None, will use same limits as main plot.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # Prepare data
    df = metrics_df.copy()
    df['Epoch_Loss[%]'] = 100 * (1 - (df['Total_Epochs'] / df['Total_Possible_Epochs']))
    categories = sorted(df['Orbit_Solution'].unique())
    style_map = get_marker_colour_mapping(categories)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Main plot
    for label in categories:
        df_sub = df[df['Orbit_Solution'] == label]
        ax.scatter(
            df_sub['Epoch_Loss[%]'],
            df_sub['Unweighted_RMS'],
            label=label,
            color=style_map[label]['color'],
            marker=style_map[label]['marker'],
            s=150,
            alpha=0.9,
            edgecolor='black',
            linewidth=0.3
        )

    ax.set_xlabel('Epoch Loss (%)')
    ax.set_ylabel('Unweighted RMS [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.set_title('SLR Residual RMS vs Epoch Loss (%)\nwith Zoomed Inset on Low Loss Area')

    # Add inset
    axins = inset_axes(ax, width="40%", height="40%", loc='lower left', borderpad=2)

    for label in categories:
        df_sub = df[df['Orbit_Solution'] == label]
        axins.scatter(
            df_sub['Epoch_Loss[%]'],
            df_sub['Unweighted_RMS'],
            color=style_map[label]['color'],
            marker=style_map[label]['marker'],
            s=100,
            alpha=0.9,
            edgecolor='black',
            linewidth=0.3
        )

    axins.set_xlim(*zoom_xlim)
    if all(zoom_ylim):
        axins.set_ylim(*zoom_ylim)
    else:
        axins.set_ylim(ax.get_ylim())

    # axins.set_yscale('log')
    axins.grid(True, alpha=0.3)
    axins.set_xticks([])  # Optional: remove ticks for cleaner inset
    axins.set_yticks([])

    # Mark inset area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    plt.show()

    # ----------------- Output table -----------------
    table = df[['Orbit_Solution', 'Total_Possible_Epochs', 'Total_Epochs', 'Epoch_Loss[%]']].copy()
    table = table.sort_values('Epoch_Loss[%]')
    table['Epoch_Loss[%]'] = table['Epoch_Loss[%]'].round(2)

    print("\n=== Epoch Statistics per Method ===")
    print(table.to_string(index=False))

    return table


    
def plot_rolling_rms(daily_metrics_df, metric_name='Unweighted_RMS', window=7, marker_every=14):
    """
    Plot smoothed daily RMS trends using rolling averages.
    - One line per orbit solution.
    - Sparse markers for clarity.
    - Log y-axis.
    """
    import matplotlib.pyplot as plt

    valid_metrics = ['Unweighted_RMS', 'Weighted_RMS_n',
                     'Weighted_RMS_rms', 'Weighted_RMS_combined',
                     'RMS_of_4_Metrics', 'Unweighted_RMS_wrt_RDO']

    if metric_name not in valid_metrics:
        raise ValueError(f"Invalid metric name. Choose from: {valid_metrics}")

    # Marker selection
    def get_marker(ref_data):
        if 'KONF' in ref_data: return 'o'
        if 'KO' in ref_data: return 's'
        if 'NONE' in ref_data or 'ESA' in ref_data: return 'x'
        if 'RDO' in ref_data: return '^'
        return 'o'

    plt.figure(figsize=(14, 8))
    plt.yscale('log')

    for label in daily_metrics_df['Method_Label'].unique():
        df_sub = daily_metrics_df[daily_metrics_df['Method_Label'] == label].copy()
        df_sub.sort_values('Date', inplace=True)
        ref_data = label.split('(')[-1].split(')')[0]

        # Compute rolling average
        df_sub[f'Rolling_{metric_name}'] = df_sub[metric_name].rolling(window=window, min_periods=1).mean()

        # Plot main line
        plt.plot(
            df_sub['Date'], df_sub[f'Rolling_{metric_name}'],
            label=label,
            linestyle='-',
            linewidth=1.5,
            alpha=0.9
        )

        # Add sparse markers (no duplicate legend)
        plt.plot(
            df_sub['Date'][::marker_every], df_sub[f'Rolling_{metric_name}'][::marker_every],
            linestyle='None',
            marker=get_marker(ref_data),
            markersize=5,
            alpha=0.6
        )

    plt.title(f'{window}-Day Rolling Average of {metric_name.replace("_", " ")} (Log Scale)')
    plt.ylabel(f'{metric_name.replace("_", " ")} [m]')
    plt.xlabel('Date')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_residuals_over_time(slr_data):
    """Plot residuals over time with proper marker handling"""
    plt.figure(figsize=(6.5, 6))
    slr_data['Res[cm]'] = slr_data['Res[m]'] * 100
    # slr_data['Abs_Res[cm]'] = np.abs(slr_data['Res[cm]'])  # For log scale
    
    # Create marker mapping
    marker_map = {
        'KONF': 'o',
        'KO': 's',
        'NONE': 'x',
        'ESA': 'X',
        'RDO': '^',
        'None': 'x',
         '100': 'x',
         '0.1': 'x',
         '1.0': 'x', 
         '0.5': 'x', 
         '50': 'x', 
         '0.3': 'x',
         '5': 'x', 
         '0.05': 'x'
    }
    
    # Get unique methods with consistent coloring
    methods = slr_data['Method_Label'].unique()
    palette = sns.color_palette("tab10", n_colors=len(methods))
    
    for idx, method in enumerate(methods):
        subset = slr_data[slr_data['Method_Label'] == method]
        ref_data = subset['Reference_Data'].iloc[0]
        
        plt.scatter(
            subset['Date'],
            subset['Res[m]'],  # Using absolute values for log scale
            label=method,
            color=palette[idx],
            marker=marker_map.get(ref_data, 'o'),
            s=10,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.3
        )
    
    # plt.title('SLR Residuals Over Time')
    plt.ylabel('Residual [m]')
    # plt.yscale('log')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rms_vs_epochs(metrics_df):
    """
    Plot RMS vs Epoch Loss (%) with annotations of theoretical epochs.
    Also outputs a table of per-method total epochs, available epochs, and loss %.
    """

    plt.figure(figsize=(12, 8))

    # Compute epoch loss %
    metrics_df = metrics_df.copy()
    metrics_df['Epoch_Loss[%]'] = 100 * (1 - (metrics_df['Total_Epochs'] / metrics_df['Total_Possible_Epochs']))

    categories = sorted(metrics_df['Orbit_Solution'].unique())
    style_map = get_marker_colour_mapping(categories)

    for label in categories:
        df_sub = metrics_df[metrics_df['Orbit_Solution'] == label]
        plt.scatter(
            df_sub['Epoch_Loss[%]'],
            df_sub['Unweighted_RMS'],
            label=label,
            color=style_map[label]['color'],
            marker=style_map[label]['marker'],
            s=150,
            alpha=0.9,
            edgecolor='black',
            linewidth=0.3
        )

    # Annotate theoretical total epochs
    max_epochs = metrics_df['Total_Possible_Epochs'].max()
    plt.text(
        0.01,
        plt.ylim()[1] * 0.95,
        f'Theoretical Max Epochs: {int(max_epochs):,}',
        color='red',
        fontsize=12,
        ha='left',
        va='top'
    )

    plt.title('SLR Residual RMS vs Epoch Loss (%)\n(0% = Full Completeness)')
    plt.ylabel('Unweighted RMS [m]')
    plt.xlabel('Epoch Loss (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()

    # ----------------- Output table -----------------
    table = metrics_df[['Orbit_Solution', 'Total_Possible_Epochs', 'Total_Epochs', 'Epoch_Loss[%]', 'Unweighted_RMS', 'Mean_Residual', 'Std_Residual']].copy()
    table = table.sort_values('Epoch_Loss[%]')
    table['Epoch_Loss[%]'] = table['Epoch_Loss[%]'].round(5)

    print("\n=== Epoch Statistics per Method ===")
    print(table.to_string(index=False))

    return table


def plot_mean_std_residuals(slr_data):
    """Plot mean residuals with SD using seaborn's built-in aggregation"""
    plt.figure(figsize=(14, 8))
    
    # First calculate statistics to ensure alignment
    slr_data['Res[mm]'] = slr_data['Res[m]']*1000
    stats_df = slr_data.groupby('Method_Label')['Res[mm]'].agg(['mean', 'std']).reset_index()
    
    # Create plot with automatic error bars
    ax = sns.barplot(
        data=slr_data,
        x='Method_Label',
        y='Res[mm]',
        estimator='mean',
        errorbar='sd',
        palette='viridis',
        edgecolor='black',
        err_kws={'linewidth': 2},
        order=stats_df.sort_values('mean', ascending=False)['Method_Label']  # Ensure order matches
    )
    
    # Add plot decorations
    plt.title('Residual Statistics by Orbit Solution\n(Mean ± 1σ)', fontsize=14)
    plt.ylabel('Residual [mm]', fontsize=12)
    plt.xlabel('Orbit Solution', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Add value labels using pre-calculated stats
    for idx, p in enumerate(ax.patches):
        mean = stats_df.iloc[idx]['mean']
        std = stats_df.iloc[idx]['std']
        ax.text(p.get_x() + p.get_width()/2., mean + 0.002,
                f'{mean:.3f}\n±{std:.3f}',
                ha='right', va='bottom', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Histogram fo residuals

def plot_residual_histograms(slr_data, method_selection=None, bins=50, range_m=(-0.06, 0.06)):
    """
    Plot histograms of SLR residuals for selected methods.

    Parameters
    ----------
    slr_data : pd.DataFrame
        DataFrame containing 'Res[m]' and 'Method_Label'.
    method_selection : list or None
        List of Method_Label strings to include in the plot.
        If None, all methods will be included.
    bins : int
        Number of bins for the histogram.
    range_m : tuple
        Range (min, max) in metres for residuals to include in histogram.
    """

    df = slr_data.copy()
    if method_selection is not None:
        df = df[df['Method_Label'].isin(method_selection)]
        
    scaling_factor = 1000 #m 2 mm
    plt.figure(figsize=(12, 6))
    methods = df['Method_Label'].unique()
    palette = sns.color_palette('tab10', n_colors=len(methods))

    for i, method in enumerate(methods):
        
        res = df[df['Method_Label'] == method]['Res[m]']
        
        mean_bias = res.mean()
        label = f"{' '.join(i for i in method.split(' ')[:-1])} (μ = {mean_bias * scaling_factor:.2f} mm)"
        
        plt.hist(res * scaling_factor, bins=bins, range=[i * scaling_factor for i in range_m], alpha=0.6,
                 label=label, color=palette[i], edgecolor='black')
        
        plt.axvline(mean_bias * scaling_factor, color=palette[i], linestyle='--', linewidth=1.5)
    
    plt.xlabel('SLR Residual [mm]')
    plt.ylabel('Number of Observations')
    # plt.title('Histogram of SLR Residuals by Method')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Method (Mean Bias)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    for method in methods:
        bias = df[df['Method_Label'] == method]['Res[m]'].mean()*scaling_factor
        print(f"{method}: Mean Bias = {bias:.4f} mm")

def plot_residual_histograms_grouped(slr_data, bins=100, range_m=(-0.09, 0.09)):
    """
    Plot grouped histograms of SLR residuals in mm for orbit method families.
    Group 1: Input orbits + RDO
    Group 2: Mean, inverse-variance, VCE
    Group 3: CMAES, residual-weighted, Nelder-Mead
    """
    scaling_factor = 1000  # m → mm
    range_mm = [i * scaling_factor for i in range_m]

    # Group definitions
    group_1 = ['AIUB (KONF)', 'IFG (KONF)', 'TUD (KONF)', 'ESA (RDO)']
    group_2 = ['mean (NONE)', 'inverse variance (NONE)', 'vce (NONE)']
    group_3 = ['cmaes (ESA)', 'residual-weighted (ESA)', 'nelder-mead (ESA)']
    all_groups = [group_1, group_2, group_3]

    df = slr_data.copy()
    df = df[df['Method_Label'].isin(sum(all_groups, []))]  # flatten and filter
    palette = sns.color_palette('tab10', n_colors=10)
    colour_map = dict(zip(df['Method_Label'].unique(), palette))

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    for ax, group in zip(axes, all_groups):
        for method in group:
            res = df[df['Method_Label'] == method]['Res[m]']
            mean_bias = res.mean()
            label = f"{' '.join(i for i in method.split(' ')[:-1])} (μ = {mean_bias * scaling_factor:.2f} mm)"

            ax.hist(res * scaling_factor, bins=bins, range=range_mm,
                    alpha=0.6, label=label,
                    color=colour_map[method], edgecolor='black')
            ax.axvline(mean_bias * scaling_factor, color=colour_map[method],
                       linestyle='--', linewidth=1.5)
            ax.set_xlim(-50, 50)

        ax.set_ylabel('Number of Observations')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Method (Mean Bias)', loc='upper left', fontsize=14)

    axes[-1].set_xlabel('SLR Residual [mm]')
    plt.tight_layout()
    plt.show()

    # Print mean biases
    print("Mean Biases (mm):")
    for method in sum(all_groups, []):
        bias = df[df['Method_Label'] == method]['Res[m]'].mean() * scaling_factor
        print(f"{method}: {bias:.4f} mm")

def plot_residual_histograms_grouped(slr_data, bins=100, range_m=(-0.09, 0.09)):
    """
    Plot grouped histograms of SLR residuals in mm with overlaid Gaussian
    computed using sample mean and std for each method.
    """
    scaling_factor = 1000  # m → mm
    range_mm = [i * scaling_factor for i in range_m]
    bin_width = (range_mm[1] - range_mm[0]) / bins

    group_1 = ['AIUB (KONF)', 'IFG (KONF)', 'TUD (KONF)',]
    group_2 = [ 'ESA (RDO)', 'mean (NONE)', 'inverse variance (NONE)', 'vce (NONE)']
    group_3 = ['cmaes (ESA)', 'residual-weighted (ESA)', 'nelder-mead (ESA)']
    all_groups = [group_1, group_2, group_3]

    df = slr_data.copy()
    df = df[df['Method_Label'].isin(sum(all_groups, []))]

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=False)

    x_range = np.linspace(range_mm[0], range_mm[1], 1000)

    for idx, (ax, group) in enumerate(zip(axes, all_groups)):
        palette = sns.color_palette('tab10', n_colors=len(group))
        colour_map = dict(zip(group, palette))
        
        for method in group:
            res = df[df['Method_Label'] == method]['Res[m]'] * scaling_factor
            mu, sigma = res.mean(), res.std()
            n = len(res)

            base_label = ' '.join(i for i in method.split(' ')[:-1])
            label = f"{base_label} (μ = {mu:.2f} mm, σ = {sigma:.2f} mm)"
            
            # Plot histogram (count mode)
            if idx == 0:
                scalefac  = 1
                ax.hist(res, bins=bins, range=[i*scalefac for i in range_mm], alpha=0.3, color=colour_map[method], label=label)
                # Plot Gaussian scaled to total count
                gaussian = norm.pdf(x_range*scalefac, mu, sigma)
                scaled_gaussian = gaussian * n * bin_width*scalefac
                ax.plot(x_range, scaled_gaussian, color=colour_map[method], linestyle='--', linewidth=1.5)
            else:
                ax.hist(res, bins=bins, range=range_mm, alpha=0.3, color=colour_map[method], label=label)
    
                # Plot Gaussian scaled to total count
                gaussian = norm.pdf(x_range, mu, sigma)
                scaled_gaussian = gaussian * n * bin_width
                ax.plot(x_range, scaled_gaussian, color=colour_map[method], linestyle='--', linewidth=1.5)

        ax.set_ylabel('Number of Observations')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Method (Mean ± Std)', loc='upper left', fontsize=12)

    axes[-1].set_xlabel('SLR Residual [mm]')
    plt.tight_layout()
    plt.show()



# -------------------------------------------------
# Pareto front computation and plot
# -------------------------------------------------

def get_best_method_labels(metrics_df, rms_col='Unweighted_RMS', top_n=3):
    """
    Return labels for the best-performing orbit methods based on Pareto frontier.
    Falls back to top N by RMS and epoch retention if frontier is too small.
    """
    # Compute epoch loss
    metrics_df = metrics_df.copy()
    metrics_df['Epoch_Loss'] = 1 - (metrics_df['Total_Epochs'] / metrics_df['Total_Possible_Epochs'])

    # Pareto filter
    def compute_pareto_frontier(df):
        pareto_mask = np.ones(len(df), dtype=bool)
        for i, (idx, row) in enumerate(df.iterrows()):
            others = df.drop(idx)
            dominated = others[
                (others[rms_col] <= row[rms_col]) &
                (others['Epoch_Loss'] <= row['Epoch_Loss'])
            ]
            if not dominated.empty:
                pareto_mask[i] = False
        return df[pareto_mask]

    pareto_df = compute_pareto_frontier(metrics_df)

    # Add Method_Label for filtering
    pareto_df['Method_Label'] = pareto_df['Method'] + ' (' + pareto_df['Reference_Data'].astype(str) + ')'

    if len(pareto_df) >= top_n:
        return pareto_df['Method_Label'].unique()

    # Fallback: top N by RMS and epoch completeness
    fallback = metrics_df.copy()
    fallback['Method_Label'] = fallback['Method'] + ' (' + fallback['Reference_Data'].astype(str) + ')'
    fallback = fallback.sort_values(by=[rms_col, 'Epoch_Loss'], ascending=[True, True])

    return fallback['Method_Label'].unique()[:top_n]


def get_pareto_filtered_metrics(metrics_df, rms_col='Unweighted_RMS'):
    """
    Compute Pareto-optimal subset of orbit solutions based on *normalised*
    RMS and epoch retention, but return original unnormalised rows.

    Returns:
        pareto_df: DataFrame of Pareto-optimal rows (unnormalised).
        method_labels: List of 'Method_Label' strings for filtering other plots.
    """
    df = metrics_df.copy()
    df['Epoch_Loss'] = 1 - (df['Total_Epochs'] / df['Total_Possible_Epochs'])
    df['Method_Label'] = df['Method'] + ' (' + df['Reference_Data'].astype(str) + ')'

    # Normalise both objectives
    df['Norm_RMS'] = (df[rms_col] - df[rms_col].min()) / (df[rms_col].max() - df[rms_col].min())
    df['Norm_Loss'] = (df['Epoch_Loss'] - df['Epoch_Loss'].min()) / (df['Epoch_Loss'].max() - df['Epoch_Loss'].min())

    def is_dominated(row, others):
        return any(
            (other['Norm_RMS'] < row['Norm_RMS'] and other['Norm_Loss'] <= row['Norm_Loss']) or
            (other['Norm_RMS'] <= row['Norm_RMS'] and other['Norm_Loss'] < row['Norm_Loss'])
            for _, other in others.iterrows()
        )

    pareto_indices = []
    for idx, row in df.iterrows():
        others = df.drop(idx)
        if not is_dominated(row, others):
            pareto_indices.append(idx)

    pareto_df = df.loc[pareto_indices].drop(columns=['Norm_RMS', 'Norm_Loss'])
    return pareto_df, pareto_df['Method_Label'].unique()


def rank_methods_by_pareto_layers(metrics_df, rms_col='Unweighted_RMS', top_n=3):
    """
    Rank orbit methods by successive Pareto layers using *normalised* metrics.
    Returns a list of Method_Label strings in Pareto rank order.
    """
    df = metrics_df.copy()
    df['Epoch_Loss'] = 1 - (df['Total_Epochs'] / df['Total_Possible_Epochs'])
    df['Method_Label'] = df['Method'] + ' (' + df['Reference_Data'].astype(str) + ')'

    # Normalise metrics
    df['Norm_RMS'] = (df[rms_col] - df[rms_col].min()) / (df[rms_col].max() - df[rms_col].min())
    df['Norm_Loss'] = (df['Epoch_Loss'] - df['Epoch_Loss'].min()) / (df['Epoch_Loss'].max() - df['Epoch_Loss'].min())

    ranked = []
    seen_labels = set()

    def is_dominated(row, others):
        return any(
            (other['Norm_RMS'] < row['Norm_RMS'] and other['Norm_Loss'] <= row['Norm_Loss']) or
            (other['Norm_RMS'] <= row['Norm_RMS'] and other['Norm_Loss'] < row['Norm_Loss'])
            for _, other in others.iterrows()
        )

    while len(ranked) < top_n and not df.empty:
        non_dominated = []
        for idx, row in df.iterrows():
            others = df.drop(idx)
            if not is_dominated(row, others):
                non_dominated.append(row['Method_Label'])

        for label in non_dominated:
            if label not in seen_labels:
                ranked.append(label)
                seen_labels.add(label)
                if len(ranked) >= top_n:
                    break

        df = df[~df['Method_Label'].isin(non_dominated)]

    return ranked

def plot_pareto_frontier_all_methods(metrics_df, pareto_df, rms_col='Unweighted_RMS'):
    """
    Visualise RMS vs Epoch Loss with Pareto frontier line.
    """
    
    if 'Epoch_Loss' not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df['Epoch_Loss'] = 1 - (metrics_df['Total_Epochs'] / metrics_df['Total_Possible_Epochs'])
    if 'Epoch_Loss' not in pareto_df.columns:
        pareto_df = pareto_df.copy()
        pareto_df['Epoch_Loss'] = 1 - (pareto_df['Total_Epochs'] / pareto_df['Total_Possible_Epochs'])

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=metrics_df,
        x='Epoch_Loss',
        y=rms_col,
        hue='Method',
        style='Reference_Data',
        palette='tab10',
        s=100,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.3
    )

    pareto_sorted = pareto_df.sort_values('Epoch_Loss')
    plt.plot(
        pareto_sorted['Epoch_Loss'],
        pareto_sorted[rms_col],
        color='black',
        linestyle='--',
        linewidth=1.5,
        marker='o',
        label='Pareto Frontier'
    )

    plt.title('Pareto Frontier: RMS vs Epoch Loss', fontsize=14)
    plt.xlabel('Epoch Loss (1 - Epoch Retention)', fontsize=12)
    plt.ylabel(f'{rms_col.replace("_", " ")} [m]', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_residuals_vs_beta(slr_results_final, method_selection=None):
    df = slr_results_final.copy()
    if method_selection is not None:
        df = df[df['Method_Label'].isin(method_selection)]
    
    # Compute robust color scale limits
    vmin = np.percentile(np.abs(df['Res[m]']), 1)
    vmax = np.percentile(np.abs(df['Res[m]']), 99)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df['beta'], df['Res[m]'], c=np.abs(df['Res[m]']),
                     cmap='viridis', s=10, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label='|Residual| [m]')
    plt.title('SLR Residuals vs Beta Angle')
    plt.xlabel('Beta [deg]')
    plt.ylabel('Residual [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals_vs_delta_u(slr_results_final, method_selection=None):
    df = slr_results_final.copy()
    if method_selection is not None:
        df = df[df['Method_Label'].isin(method_selection)]

    vmin = np.percentile(np.abs(df['Res[m]']), 1)
    vmax = np.percentile(np.abs(df['Res[m]']), 99)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df['Delta_u [deg]'], df['Res[m]'], c=np.abs(df['Res[m]']),
                     cmap='viridis', s=10, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label='|Residual| [m]')
    plt.title('SLR Residuals vs Δu (Argument of Latitude difference)')
    plt.xlabel('Δu [deg]')
    plt.ylabel('Residual [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals_vs_mu(slr_results_final, method_selection=None):
    df = slr_results_final.copy()
    if method_selection is not None:
        df = df[df['Method_Label'].isin(method_selection)]
    
    vmin = np.percentile(np.abs(df['Res[m]']), 1)
    vmax = np.percentile(np.abs(df['Res[m]']), 99)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df['mu [deg]'], df['Res[m]'], c=np.abs(df['Res[m]']),
                     cmap='viridis', s=10, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label='|Residual| [m]')
    plt.title('SLR Residuals vs Satellite Argument of Latitude (mu)')
    plt.xlabel('mu [deg]')
    plt.ylabel('Residual [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_beta_vs_delta_u(df, x_column='Delta_u [deg]', y_column='beta', method_selection=None,
                          gridsize=80, cmap='viridis', residual_column='Res[m]',
                          reduce_func=np.mean, x_label='Δu [deg]', y_label=r'$\|Beta\|$ [deg]', title=None):
    """
    General 2D hexbin plot for residuals using selected columns.
    """

    
    df_plot = df.copy()
    if method_selection is not None:
        df_plot = df_plot[df_plot['Method_Label'].isin([method_selection])]
    df_plot['abs beta'] = abs(df_plot['beta'])
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(df_plot[x_column], df_plot['abs beta'],
                    C=np.abs(df_plot[residual_column]),
                    gridsize=gridsize, cmap=cmap,
                    reduce_C_function=reduce_func, mincnt=1,
                    norm=LogNorm(vmin=1e-4, vmax=0.3))
    
    plt.colorbar(hb, label=f'{reduce_func.__name__.capitalize()} |Residual| [m]')
    # plt.title(title if title else f'Residuals: {y_column} vs {x_column}')
    plt.xlabel(x_label if x_label else x_column)
    plt.ylabel(y_label if y_label else y_column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_beta_vs_delta_u_multi(df, methods_to_plot,
                               x_column='Delta_u [deg]', residual_column='Res[m]',
                               gridsize=80, cmap='viridis',
                               reduce_func=np.mean):
    """
    Multi-panel hexbin plot of residuals vs Δu and |beta| across methods, with shared color scale.
    """
    # Filter once
    df_plot = df[df['Method_Label'].isin(methods_to_plot)].copy()

    # Ensure correct types
    df_plot[x_column] = df_plot[x_column].astype(float)
    df_plot['beta'] = df_plot['beta'].astype(float)
    df_plot[residual_column] = df_plot[residual_column].astype(float)

    # Clip and log-ready residuals
    df_plot['Abs_Res[m]'] = np.clip(np.abs(df_plot[residual_column]), 1e-4, 0.3)

    # Compute common vmin, vmax using percentiles for robustness
    vmin = np.percentile(df_plot['Abs_Res[m]'], 1)
    vmax = np.percentile(df_plot['Abs_Res[m]'], 99)

    # Setup subplots
    fig, axs = plt.subplots(1, len(methods_to_plot), figsize=(6 * len(methods_to_plot), 5), sharex=True, sharey=True)
    if len(methods_to_plot) == 1:
        axs = [axs]

    for ax, method in zip(axs, methods_to_plot):
        df_method = df_plot[df_plot['Method_Label'] == method]
        hb = ax.hexbin(df_method[x_column], np.abs(df_method['beta']),
                       C=df_method['Abs_Res[m]'],
                       gridsize=gridsize, cmap=cmap,
                       reduce_C_function=reduce_func,
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       mincnt=1)
        ax.set_title(method)
        ax.set_xlabel('Δu [deg]')
        ax.set_ylabel(r'$|Beta|$ [deg]')
        ax.grid(True)

    # Add shared colorbar
    cbar = fig.colorbar(hb, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(f'{reduce_func.__name__.capitalize()} |Residual| [m] (shared scale)')

    plt.suptitle('Comparison of Residuals vs Δu and |Beta| Across Methods', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_hexbin_residuals_vs_beta(slr_results_final, method_selection=None):
    df = slr_results_final.copy()
    if method_selection is not None:
        df = df[df['Method_Label'].isin(method_selection)]

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(df['beta'], df['Res[m]'],
                    gridsize=80, cmap='viridis',
                    mincnt=1)
    plt.colorbar(hb, label='Number of observations')
    plt.title('SLR Residuals vs Beta Angle (Hexbin)')
    plt.xlabel('Beta [deg]')
    plt.ylabel('Residual [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hexbin_residuals_vs_delta_u(slr_results_final, method_selection=None):
    df = slr_results_final.copy()
    if method_selection is not None:
        df = df[df['Method_Label'].isin(method_selection)]

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(df['Delta_u [deg]'], df['Res[m]'],
                    gridsize=80, cmap='viridis',
                    mincnt=1)
    plt.colorbar(hb, label='Number of observations')
    plt.title('SLR Residuals vs Δu (Hexbin)')
    plt.xlabel('Δu [deg]')
    plt.ylabel('Residual [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hexbin_residuals_vs_mu(slr_results_final, method_selection=None):
    df = slr_results_final.copy()
    if method_selection is not None:
        df = df[df['Method_Label'].isin(method_selection)]

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(df['mu [deg]'], df['Res[m]'],
                    gridsize=80, cmap='viridis',
                    mincnt=1)
    plt.colorbar(hb, label='Number of observations')
    plt.title('SLR Residuals vs mu (Hexbin)')
    plt.xlabel('mu [deg]')
    plt.ylabel('Residual [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plotting residuals against elevation

def plot_rms_by_elevation(df, elevation_col='Sta [deg]', residual_col='Res[m]',
                           bin_width=10, elevation_min=0, elevation_max=90):
    """
    Plot RMS of SLR residuals binned by elevation angle with observation counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SLR residuals with elevation and residual columns.
    elevation_col : str
        Column name for elevation angle (in degrees).
    residual_col : str
        Column name for residuals (in meters).
    bin_width : int
        Width of elevation bins in degrees.
    elevation_min : int
        Minimum elevation to include.
    elevation_max : int
        Maximum elevation to include.
    title : str
        Title for the plot.
    """

    # Copy and clean input
    df = df.copy()
    df = df[[elevation_col, residual_col]].dropna()
    df[elevation_col] = pd.to_numeric(df[elevation_col], errors='coerce')
    df[residual_col] = pd.to_numeric(df[residual_col], errors='coerce')
    df = df.dropna()

    # Apply elevation filter
    df = df[df[elevation_col] >= elevation_min]
    df = df[df[elevation_col] <= elevation_max]

    # Create bins
    bins = np.arange(elevation_min, elevation_max + bin_width, bin_width)
    df['Elevation Bin'] = pd.cut(df[elevation_col], bins=bins, right=False, include_lowest=True)

    # Compute RMS and count for each bin
    grouped = df.groupby('Elevation Bin')[residual_col]
    rms_values = grouped.apply(lambda x: np.sqrt(np.mean(x**2)))
    counts = grouped.count()

    # Plot
    fig, ax1 = plt.subplots(figsize=(6.5, 6))
    bin_labels = [str(interval) for interval in rms_values.index]

    ax1.bar(bin_labels, rms_values, color='tab:blue', alpha=0.7)
    ax1.set_xlabel('Elevation Bin [deg]')
    ax1.set_ylabel('RMS of Residuals [m]', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(np.arange(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(bin_labels)), counts, color='tab:red', marker='o', linestyle='-')
    ax2.set_ylabel('Number of Observations', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.show()

# Plot number of normal point observations per station
def plot_avg_obs_per_normal_point_per_station(df, site_col='Site', obs_col='n', top_n=None, min_count=1):
    """
    Plot average number of observations used to construct normal points per station.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with station and observation count columns.
    site_col : str
        Column containing station identifiers.
    obs_col : str
        Column containing number of observations per normal point.
    top_n : int or None
        If set, only plot the top N stations with the most data.
    min_count : int
        Minimum number of normal points required to include a station in the plot.
    title : str
        Title for the plot.
    """

    df = df[[site_col, obs_col]].copy()
    df[obs_col] = pd.to_numeric(df[obs_col], errors='coerce')
    df = df.dropna()
    df['Clean_Site'] = df['Site'].apply(lambda x: x[0:10].strip())
    
    # Compute mean and count per station
    stats = df.groupby('Clean_Site')[obs_col].agg(['mean', 'count']).reset_index()
    stats = stats[stats['count'] >= min_count]
    stats = stats.sort_values('mean', ascending=False)

    if top_n:
        stats = stats.head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.barplot(data=stats, x='Clean_Site', y='mean', palette='viridis', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Observations per Normal Point')
    plt.xlabel('Station')
    plt.grid(axis='y', alpha=0.3)

    # Annotate count if useful
    # for i, row in stats.iterrows():
    #     ax.text(i, row['mean'] + 0.3, f"n={int(row['count'])}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    
def plot_avg_obs_and_rms_per_station(df, site_col='Site', obs_col='n', res_col='Res[m]',
                                  top_n=None, min_count=1,
                                  title='Station-wise Avg Observations and RMS of Residuals'):
    """
    Plot average number of observations per normal point (bar)
    and RMS of residuals (line) per station.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing station codes, observations, and residuals.
    site_col : str
        Column containing station identifiers.
    obs_col : str
        Column with number of observations used per normal point.
    res_col : str
        Column with residual values (in metres).
    top_n : int or None
        If set, limit plot to top N stations by average observations.
    min_count : int
        Minimum number of data points to include a station.
    title : str
        Plot title.
    """
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 16,
        'figure.titlesize': 14
    })
    df = df[[site_col, obs_col, res_col]].copy()
    df[obs_col] = pd.to_numeric(df[obs_col], errors='coerce')
    df[res_col] = pd.to_numeric(df[res_col], errors='coerce')
    df = df.dropna()
    df['Clean_Site'] = df['Site'].apply(lambda x: x[0:10].strip())
    
    stats = df.groupby('Clean_Site').agg(
        Avg_Obs=(obs_col, 'mean'),
        RMS_Res=(res_col, lambda x: np.sqrt(np.mean(x**2))),
        Count=(obs_col, 'count')
    ).reset_index()

    stats = stats[stats['Count'] >= min_count]
    stats = stats.sort_values('RMS_Res', ascending=False)

    if top_n:
        stats = stats.head(top_n)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(9, 7))
    x_labels = stats['Clean_Site']
    x_pos = np.arange(len(x_labels))

    # Bar plot for RMS
    bars = ax1.bar(x_pos, stats['RMS_Res'], color='tab:blue', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('RMS of Residuals [m]', color='tab:blue')   
    ax1.set_xlabel('Station')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')

    # Secondary axis for n_obs
    ax2 = ax1.twinx()
    ax2.plot(x_pos, stats['Avg_Obs'], color='tab:red', marker='o', linewidth=2)
    ax2.set_ylabel('Avg Observations per NP', color='tab:red')

    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Processing
# -------------------------------------------------

#%%
# FINAL DATA
gap_filter_path = None
#%% EXAMPLE YEAR 2023 PROVIDED
np_csv_path = Path(
     r"normal_points\all_normal_points_2023.csv")
gap_filter_path = Path(r"normal_points\timescales")
#%%
print("Loading and cleaning data...")
slr_data = load_and_clean_data(np_csv_path, gap_dir=gap_filter_path, trim_seconds=11)
slr_data = slr_data[~slr_data['Method'].isin(['cmaesLO'])]
#%%
time_config = {
    # 'hours': {
    #     'early morning': (0, 12),
    #     'morning': (6, 12),
    #     'afternoon': (12, 18),
    #     'evening': (18, 24)
    # },
    # 'window': {'start_date': '2023-1-1', 'end_date': '2023-3-1'},
    # 'excluded_dates' : ['2023-1-26']#, '2022-01-01', '2022-02-3', '2022-01-05', '2022-02-7', '2022-01-4' ]#, '2023-03-07', '2023-03-15']
    # 'days_of_week': [0,1,2,3,4]  # Weekdays only
}

slr_data_filtered = time_restructure(slr_data, time_config, gap_dir=gap_filter_path)
datetime = pd.to_datetime(slr_data_filtered['Date'], format = 'ISO8061')
slr_data_filtered = slr_data_filtered.set_index(datetime)
#%%
methodtest = ['ESA']#, 'IFG', 'AIUB', 'TUD']
reftest = ['RDO']#, 'KONF', 'KONF', 'KONF']
for method, ref in zip(methodtest,reftest):
    esa_rdo = slr_data_filtered[
        (slr_data_filtered['Method'] == method) &
        (slr_data_filtered['Reference_Data'] == ref) &
        (slr_data_filtered['Sta [deg]'].notna())
    ].copy()
    
    plot_rms_by_elevation(esa_rdo)

esa_rdo = slr_data_filtered[
    (slr_data_filtered['Method'] == 'ESA') &
    (slr_data_filtered['Reference_Data'] == 'RDO') &
    (slr_data_filtered['Sta [deg]'].notna())
].copy()

plot_rms_by_elevation(esa_rdo)
#%%
slr_data_filtered = slr_data_filtered[slr_data_filtered['Sta [deg]'].astype(float) >= 10] #setting elevation cutoff
#%% Plot process and get summary table
station_rms_summary, slr_data_final = plot_station_rms_full_process(slr_data_filtered, outlier_threshold=10, rms_limit=0.3, plot=True)
accepted_stations = station_rms_summary[station_rms_summary['Status'].str.contains('Accepted')]['Site'].unique()

#%%
slr_data_histogram = slr_data_final.copy()[abs(slr_data_final['Res[m]'].astype(float)) <= 0.5]
plot_residual_histograms_grouped(slr_data_histogram)

# %%
#  ---------------------------
#  Final filtered data
#  ---------------------------

slr_data_final= slr_data_final[~slr_data_final['Reference_Data'].isin(['KO'])]
split_slr_data = split_into_daily_chunks(slr_data_final)
print("Calculating performance metrics...")
metrics = calculate_rms_metrics(slr_data_final)
slr_data_residual_filtering = slr_data_final.copy()[abs(slr_data_final['Res[m]'].astype(float)) <= 0.5]
metrics_residual_filtering = calculate_rms_metrics(slr_data_residual_filtering)

stationwise_metrics = [calculate_rms_metrics(slr_data_final[slr_data_final['Site'] == i]) for i in accepted_stations]
daily_metrics = calculate_daily_metrics(split_slr_data)
pareto_metrics, pareto_labels = get_pareto_filtered_metrics(metrics)

#%%
print("Generating visualizations...")

plot_method_comparison_log(metrics, metric = 'Unweighted_RMS')

check_input_filtering = 0
if check_input_filtering:
    # "Plotting filtered and non-filtered inputs"
    selected_ref = ['KO','KONF']# 
    selected_df = slr_data_final[slr_data_final['Reference_Data'].isin(selected_ref)]
    selected_metrics = calculate_rms_metrics(selected_df)
    plot_method_comparison_grouped(selected_metrics, width = 7)

#%%
# method labels:
methods_to_plot = ['inverse variance (NONE)']
slr_data_fin = slr_data_final
plot_hexbin_residuals_vs_beta(slr_data_fin, method_selection=methods_to_plot)
plot_hexbin_residuals_vs_delta_u(slr_data_fin, method_selection=methods_to_plot)
plot_hexbin_residuals_vs_mu(slr_data_fin, method_selection=methods_to_plot)
#%%
plot_beta_vs_delta_u_multi(
    slr_data_fin,
    methods_to_plot=methods_to_plot
)
#%%
for method in methods_to_plot:
    plot_beta_vs_delta_u(
        slr_data_fin,
        method_selection=method
    )
#%%

daily_check1 = slr_data_final[~slr_data_final['Method'].isin(['mean', 'inverse variance', 'vce', 'residual-weighted', 'nelder-mead', 'cmaes', 'cmaes LO'])]
daily_check = daily_check1[daily_check1['Reference_Data'].isin(['KONF','RDO'])]
daily_check_split = split_into_daily_chunks(daily_check, freq='W')
filtered_daily_metrics = calculate_daily_metrics(daily_check_split)

plot_daily_rms_trend(filtered_daily_metrics, 'Unweighted_RMS')
daily_check_konf =  daily_check1[daily_check1['Reference_Data'].isin(['KONF'])]

# %% TESTING PARETO FRONT 
thresholdcheck = 0

if thresholdcheck:
    
    def plot_threshold_comparison(metrics_df, metric='Unweighted_RMS', method = 'mean', station=None, axis_split=True):
        """Plot RMS comparison with optional manual broken y-axis."""
    
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
    
        # Replace 'KONF' with 'None' in labels
        metrics_df['Orbit_Solution'] = metrics_df['Orbit_Solution'].astype(str).str.replace('KONF', 'None', regex=False)
    
        # Calculate epoch loss
        metrics_df['Epoch_Loss_Pct'] = 100 - (metrics_df['Total_Epochs'] /
                                              metrics_df['Total_Possible_Epochs'] * 100).round(1)
    
        # Colour by 'Method'
        color_map = {
            method : '#1f77b4',
            'IFG': '#ff7f0e',
            'TUD': '#2ca02c',
            'AIUB': '#d62728',
            'ESA': '#9467bd'
        }
    
        metrics_df['Method'] = metrics_df['Method'].astype(str)
        bar_colors = metrics_df['Method'].map(color_map).fillna('lightgrey')
    
        x_labels = metrics_df['Orbit_Solution']
        x_pos = np.arange(len(metrics_df))
    
        break_threshold = metrics_df[metric].sort_values().iloc[0:-1].max()
        y_max = metrics_df[metric].max()
        needs_break = y_max > break_threshold
    
        if needs_break and axis_split:
            fig, (ax_high, ax_low) = plt.subplots(
                2, 1,
                sharex=True,
                figsize=(13, 6),
                gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05}
            )
    
            ax_high.bar(x_pos, metrics_df[metric], color=bar_colors)
            ax_high.set_ylim(y_max - 0.2, y_max + 0.5)
            ax_high.spines['bottom'].set_visible(False)
            ax_high.tick_params(labelbottom=False)
    
            ax_low.bar(x_pos, metrics_df[metric], color=bar_colors)
            ax_low.set_ylim(0, break_threshold * 1.05)
            ax_low.spines['top'].set_visible(False)
            ax_low.tick_params(labeltop=False)
    
            # Diagonal break marks
            d = .015
            kwargs = dict(transform=ax_high.transAxes, color='k', clip_on=False)
            ax_high.plot((-d, +d), (-d, +d), **kwargs)
            ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_low.transAxes)
            ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
            axes = [ax_low, ax_high]
        else:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.bar(x_pos, metrics_df[metric], color=bar_colors)
            ax.set_ylim(0, y_max * 1.1)
            axes = [ax]
            ax_low = ax
    
        # Annotate bars with epoch loss (including 0%)
        for idx, val in enumerate(metrics_df[metric]):
            loss_pct = metrics_df.iloc[idx]['Epoch_Loss_Pct']
            if not np.isnan(loss_pct):
                if needs_break and axis_split:
                    ax = ax_low if val <= break_threshold else ax_high
                    y_offset = 0.08 * y_max if ax is ax_low else 0.05 * y_max
                else:
                    ax = ax_low
                    y_offset = 0.08 * y_max
                ax.annotate(f"-{loss_pct:.1f}%",
                            (x_pos[idx], val),
                            xytext=(0, y_offset),
                            textcoords="offset points",
                            ha='center',
                            fontsize=8,
                            color='dimgrey',
                            rotation=45)
    
        # Final axis formatting
        ax_low.set_xticks(x_pos)
        ax_low.set_xticklabels(x_labels, rotation=45, ha='right')
        ax_low.set_ylabel(f'{metric.replace("_", " ")} [m]')
        ax_low.grid(axis='y', alpha=0.3)
    
        # Legend
        legend_elements = [Patch(facecolor=color_map[m], label=m) for m in color_map] + [
            Line2D([0], [0], marker='', color='none',
                   label='Percentage labels show\nepoch loss relative\nto total possible',
                   markerfacecolor='dimgrey', markersize=8)
        ]
    
        if needs_break and axis_split:
            legend_elements.append(Line2D([0], [0], color='k', lw=1,
                                   label=f'Y-axis break at {break_threshold:.2f} m'))
    
        ax_low.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
                      frameon=False, fontsize=9)
    
        plt.tight_layout()
        plt.show()


    
    def preprocess_threshold_data(df, method = 'mean'):
        """Prepare threshold analysis data with proper type handling"""
        konf_max = df['Total_Possible_Epochs'].max()
        # Filter relevant methods and references
        threshold_df = df[
            ((df['Method'] == method) & (df['Reference_Data'] == 'None')) |
            ((df['Method'].isin(['IFG', 'AIUB', 'TUD', method])) & 
             (df['Reference_Data'].isin(['0.05', '0.1', '0.3', '0.5', '1', '5', '50', '100', 'KONF', 'RDO'])))
        ].copy()
        
        # Convert thresholds to numeric (handle KONF/None as infinity)
        threshold_df['Threshold'] = pd.to_numeric(
            threshold_df['Reference_Data'].replace({'KONF': np.inf, 'None': np.inf, 'RDO': np.inf}),
            errors='coerce'
        )
        
        # Calculate epoch retention percentage
        threshold_df['Epoch_Loss'] = 1-threshold_df['Total_Epochs'] / konf_max
        
        return threshold_df.sort_values(['Method', 'Threshold'])
    
    def calculate_threshold_metrics(threshold_df, method='mean'):
        """Calculate optimization metrics with proper normalization"""
        # Calculate relative performance metrics
        metrics = threshold_df.groupby(['Method', 'Threshold']).agg({
            'Unweighted_RMS': method,
            'Total_Epochs': method,
            'Epoch_Loss': method,
            'Total_Possible_Epochs': 'first'
        }).reset_index()
        
        # Normalize RMS between best and theoretical performance
        theoretical_rms = threshold_df[
            (threshold_df['Method'] == method) & 
            (threshold_df['Reference_Data'] == 'None')
        ]['Unweighted_RMS'].mean()
        
        metrics['RMS_Improvement'] = theoretical_rms - metrics['Unweighted_RMS']
        metrics['Normalized_Improvement'] = metrics['RMS_Improvement']/theoretical_rms
        
        # Calculate efficiency score
        metrics['Efficiency'] = (metrics['Normalized_Improvement'] + 
                                metrics['Epoch_Loss']) / 2
        
        # Identify Pareto optimal thresholds
        metrics['Pareto_Optimal'] = False
        for method in metrics['Method'].unique():
            method_df = metrics[metrics['Method'] == method].sort_values('Threshold')
            frontier = []
            current_min_rms = np.inf
            for _, row in method_df.iterrows():
                if row['Unweighted_RMS'] < current_min_rms:
                    frontier.append(row.name)
                    current_min_rms = row['Unweighted_RMS']
            metrics.loc[frontier, 'Pareto_Optimal'] = True
        
        return metrics
    
    
    def compute_pareto_frontier(df):
        """Identify Pareto optimal solutions across all methods/thresholds."""
        pareto_mask = np.ones(len(df), dtype=bool)
        rms_col = 'Unweighted_RMS'
        for i, (idx, row) in enumerate(df.iterrows()):
            # Compare against all other solutions
            others = df.drop(idx)
            dominated = others[
                ((others[rms_col] <= row[rms_col]) | np.isclose(others[rms_col], row[rms_col], atol=1e-6)) &
                ((others['Epoch_Loss'] <= row['Epoch_Loss']) | np.isclose(others['Epoch_Loss'], row['Epoch_Loss'], atol=1e-6))
            ]

            if not dominated.empty:
                pareto_mask[i] = False
        
        return df[pareto_mask]
    
    def plot_pareto_frontier(df, pareto_df):
        """Visualize RMS vs Retention with Pareto frontier."""
        plt.figure(figsize=(6.5,7))
        
        # Scatter plot of all solutions
        sns.scatterplot(
            data=df,
            x='Epoch_Loss',
            y='Unweighted_RMS',
            hue='Threshold',
            style='Method',
            palette='tab10',
            s=150,
            alpha=0.9
        )
        
        # Highlight Pareto frontier
        sns.lineplot(
            data=pareto_df.sort_values('Epoch_Loss'),
            x='Epoch_Loss',
            y='Unweighted_RMS',
            color='black',
            linestyle='--',
            label='Pareto Front'
        )
        
        # plt.title('Pareto Frontier: Accuracy vs Data Retention')
        plt.xlabel('Epoch Loss (%)')
        plt.ylabel('Unweighted RMS (m)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(draggable=True, ncols=int(4))#bbox_to_anchor=(1.1, 1))
        plt.grid(True, alpha=0.3)
        plt.show()

    
    def plot_pareto_frontier(df, pareto_df, method = 'mean'):
        """Visualise RMS vs Retention with Pareto frontier: full vs filtered comparison."""
        fig, axes1 = plt.subplots(1, 1, figsize=(13, 6))
        fig, axes2 = plt.subplots(1, 1, figsize=(13, 6))
        axes = [axes1, axes2]
        hue_order = sorted(df['Threshold'].unique())
        style_order = sorted(df['Method'].unique())
        
        # === LEFT PLOT: Full log-log plot ===
        sns.scatterplot(
            data=df,
            x='Epoch_Loss',
            y='Unweighted_RMS',
            hue='Threshold',
            style='Method',
            hue_order=hue_order,
            style_order=style_order,
            palette='tab10',
            s=150,
            alpha=0.9,
            ax=axes[0]
        )
    
        sns.lineplot(
            data=pareto_df.sort_values('Epoch_Loss'),
            x='Epoch_Loss',
            y='Unweighted_RMS',
            color='black',
            linestyle='--',
            label='Pareto Front',
            ax=axes[0]
        )
    
        # axes[0].set_title('All Methods & Thresholds (log-log)')
        axes[0].set_xlabel('Epoch Loss (%)')
        axes[0].set_ylabel('Unweighted RMS (m)')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(draggable=True, fontsize=14)
    
        # === RIGHT PLOT: Filtered to mean method, IFG, KONF ===
        filtered = df[
            (df['Method'] == method) |
            ((df['Method'] == 'IFG') & (df['Threshold'] == np.inf))
        ]
        
    
        sns.scatterplot(
            data=filtered,
            x='Epoch_Loss',
            y='Unweighted_RMS',
            hue='Threshold',
            style='Method',
            hue_order=hue_order,
            style_order=style_order,
            palette='tab10',
            s=150,
            alpha=0.9,
            ax=axes[1]
        )
                
        sns.lineplot(
            data=pareto_df.sort_values('Epoch_Loss'),
            x='Epoch_Loss',
            y='Unweighted_RMS',
            color='black',
            linestyle='--',
            label='Pareto Front',
            ax=axes[1]
        )
        
        # axes[1].set_title('Mean Method (IFG, KONF only)')
        axes[1].set_xlabel('Epoch Loss (%)')
        axes[1].set_ylabel('Unweighted RMS (m)')
        # axes[1].set_yscale('log')
        axes[1].set_xscale('linear')
        axes[1].grid(True, alpha=0.3)
        axes[1].get_legend().set_visible(True)
        
        plt.tight_layout()
        plt.show()

    selected_method = 'vce'
    # Load and preprocess data
    threshold_data = preprocess_threshold_data(metrics, method = selected_method)
    
    # Calculate metrics
    threshold_metrics = calculate_threshold_metrics(threshold_data)
    metrics_df = threshold_metrics
    
    # 2. Calculate Pareto frontier
    pareto_df = compute_pareto_frontier(metrics_df)
    
    # 3. Visualize
    plot_pareto_frontier(metrics_df, pareto_df, method = selected_method)
    plot_threshold_comparison(metrics)
    # 5. Get optimal thresholds
    print("Pareto Optimal Solutions:")
    print(pareto_df[['Method', 'Threshold', 'Unweighted_RMS', 'Epoch_Loss']])   