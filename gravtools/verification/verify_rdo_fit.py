"""
verify_rdo_fit.py
 
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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from classes import AccessRequest
from retrieve_arcs import retrieve_arcs
import seaborn as sns
from functools import reduce
from pandas.tseries.offsets import MonthBegin
from gravtools.kinematic_orbits.plotting_functions import plot_orbit_components, plot_orbit_gaps, plot_orbit_residuals

def generate_access_requests(data_type, startdates, analysis_centre=["IFG", "AIUB", "TUD"], get_data=False, orbit_duration = MonthBegin(1), satellite_ids = ['47']):
    requests = []
    for start in startdates:
        end = start + orbit_duration  # End is the first day of the next month
        requests.append(AccessRequest(data_type=data_type,
                                      satellite_id=satellite_ids,
                                      analysis_centre=analysis_centre,
                                      window_start=start,
                                      window_stop=end,
                                      get_data=get_data,
                                      round_seconds=False))
    return requests

def split_into_daily_chunks(orbit_df, window_start, window_end, freq='D'):
    """Split DataFrame into daily chunks within the specified window"""
    daily_groups = []
    for _, daily_df in orbit_df.groupby(pd.Grouper(freq=freq)):
        # Truncate to the window and check if any data remains
        truncated = daily_df[(daily_df.index >= window_start) & (daily_df.index < window_end)]
        if not truncated.empty:
            daily_groups.append(truncated)
    return daily_groups

# Plotting

def plot_rms_with_errorbars(summary_csv_path, save_path=None, figsize=(10, 6)):
    """
    Plot bar chart of 3D RMS residuals with STD error bars per method across months.
    
    Parameters:
    - summary_csv_path : str, path to CSV file with residual summary statistics
    - save_path : str or None, if given, the plot will be saved to this path
    - figsize : tuple, size of the figure
    """
    # Load and prepare data
    df = pd.read_csv(summary_csv_path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)  # Required for your DD/MM/YYYY format
    df = df.sort_values(['date', 'centre'])
    
    # Create consistent plotting order
    methods = df['centre'].unique()
    dates = df['date'].drop_duplicates().sort_values()
    date_labels = dates.dt.strftime('%b-%Y')
    x = np.arange(len(dates))  # X positions for each date group

    # Set bar width and spacing
    width = 0.8 / len(methods)  # total width shared between all bars per group

    # Initialise plot
    plt.figure(figsize=figsize)
    sns.set(style='whitegrid')

    for i, method in enumerate(methods):
        sub_df = df[df['centre'] == method].set_index('date').reindex(dates)
        rms = sub_df['RMS_3D'].values
        std = sub_df['Std_3D'].values

        # Shift x-positions to group by date
        xpos = x + i * width - ((len(methods) - 1) / 2) * width

        # Plot only valid entries
        valid = ~np.isnan(rms)
        plt.bar(xpos[valid], rms[valid], width=width, yerr=std[valid], capsize=3, label=method)

    # Final touches
    plt.xticks(x, date_labels, rotation=30,  ha='right')
    plt.ylabel('3D RMS Residual [m]')
    plt.yscale('log')
    # plt.xlabel('Month')
    # plt.title('3D Residual RMS per Method with Standard Deviation Error Bars')
    plt.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_overall_rms(summary_csv_path, save_path=None, figsize=(10, 6)):
    """
    Plot overall average 3D RMS residuals per method, with standard deviation bars.
    
    Parameters:
    - summary_csv_path : str, path to residual summary CSV
    - save_path : str or None, if given, the figure is saved to this path
    - figsize : tuple, figure size
    """
    
    method_order = [
        'IFG', 'AIUB', 'TUD',
        'Mean', 'Inverse Variance', 'Residual Weighted',
        'VCE', 'CMAES', 'Nelder-Mead'
        ]
    
    method_label_map = {
        'IFG': 'IFG',
        'AIUB': 'AIUB',
        'TUD': 'TUD',
        'mean': 'Mean',
        'vce': 'VCE',
        'cmaes': 'CMAES',
        'neldermead': 'Nelder-Mead',
        'residualweighted': 'Residual Weighted',
        'inversevariance': 'Inverse Variance'
    }
    
    # Load and prep data
    df = pd.read_csv(summary_csv_path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['centre'] = df['centre'].map(method_label_map)
    # Group by method (centre)
    grouped = df.groupby('centre')
    mean_rms = grouped['RMS_3D'].mean()
    std_rms = grouped['RMS_3D'].std()
    # Reorder by custom method list
    mean_rms = mean_rms.reindex(method_order)
    std_rms = std_rms.reindex(method_order)
    # Plot
    plt.figure(figsize=figsize)
    sns.set(style='whitegrid', font_scale=1.4)
    
    methods = mean_rms.index
    x = np.arange(len(methods))
    
    plt.bar(x, mean_rms.values, yerr=std_rms.values, capsize=4)
    plt.xticks(x, methods, rotation=30, ha='right')
    plt.ylabel('Mean 3D RMS Residual [m]')
    # plt.yscale('log')
    # plt.title('Overall 3D RMS Residuals per Method (± STD over days/months)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def filter_residual_outliers_vce_iv(residuals_base_path, threshold=10.0):
    """
    Removes rows from VCE and inversevariance residual CSVs where any of dx, dy, dz > threshold.

    Parameters:
    - residuals_base_path : str
        Root folder where residual CSV files are stored.
    - threshold : float
        Threshold in metres beyond which residuals are considered outliers (default: 10.0 m).
    """
    target_methods = ['vce', 'inversevariance']
    total_removed = 0
    files_modified = 0

    for root, _, files in os.walk(residuals_base_path):
        for file in files:
            if not file.endswith('.csv') or 'residuals' not in file:
                continue

            if not any(file.startswith(method) for method in target_methods):
                continue  # Skip files not from vce or inversevariance

            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath, parse_dates=['datetime'])

            # Filter based on threshold
            mask = (df['dx'].abs() <= threshold) & \
                   (df['dy'].abs() <= threshold) & \
                   (df['dz'].abs() <= threshold)
            n_removed = (~mask).sum()

            if n_removed > 0:
                df_filtered = df[mask]
                df_filtered.to_csv(filepath, index=False)
                print(f"[{file}] Removed {n_removed} outlier(s)")
                total_removed += n_removed
                files_modified += 1

    print(f"\n Completed. {files_modified} file(s) modified. Total outliers removed: {total_removed}")

def recompute_residual_summary(residuals_base_path, output_path=None):
    """
    Recomputes residual summary statistics from cleaned residual files.

    Parameters:
    - residuals_base_path : str
        Root directory where residual CSVs are stored (e.g. E:/.../RDO_residuals)
    - output_path : str or None
        Where to save the updated summary CSV (default: residuals_base_path + /residual_summary_stats_cleaned.csv)
    """
    summary_rows = []

    for root, _, files in os.walk(residuals_base_path):
        for file in files:
            if not file.endswith('.csv') or 'residuals' not in file:
                continue

            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath, parse_dates=['datetime'])

            if df.empty:
                continue

            # Extract date from filename (YYYYDOY to YYYY-MM-DD)
            name_parts = file.split('_')
            method = name_parts[0]
            date_str = name_parts[-1].replace('.csv', '')  # e.g. 2023015
            date = pd.to_datetime(date_str, format='%Y%j').strftime('%Y-%m-%d')

            row = {
                'date': date,
                'centre': method,
                'RMS_dx': np.sqrt(np.mean(df['dx']**2)),
                'RMS_dy': np.sqrt(np.mean(df['dy']**2)),
                'RMS_dz': np.sqrt(np.mean(df['dz']**2)),
                'RMS_3D': np.sqrt(np.mean(df['3D']**2)),
                'Mean_3D': np.mean(df['3D']),
                'Std_3D': np.std(df['3D']),
                'N_points': len(df)
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(['date', 'centre'])

    # Determine output path
    if output_path is None:
        output_path = os.path.join(residuals_base_path, 'residual_summary_stats_cleaned.csv')

    summary_df.to_csv(output_path, index=False)
    print(f"\n Recalculated summary stats written to: {output_path}")
    
def recompute_summary_from_common_epochs(residuals_base_path, output_path=None):
    """
    Recomputes residual summary statistics using only epochs that are shared across all methods.

    Parameters:
    - residuals_base_path : str
        Root folder containing residual CSVs.
    - output_path : str or None
        Where to save the summary CSV (default: residual_summary_stats_common_epochs.csv)
    """
    method_files = {}  # {method: [list of filepaths]}
    
    # Group all residual files by method
    for root, _, files in os.walk(residuals_base_path):
        for file in files:
            if not file.endswith('.csv') or 'residuals' not in file:
                continue
            method = file.split('_')[0]
            method_files.setdefault(method, []).append(os.path.join(root, file))

    # Sort methods and prepare final summary rows
    all_methods = sorted(method_files.keys())
    summary_rows = []

    for files_group in zip(*[sorted(method_files[m]) for m in all_methods]):
        # Load all residuals for the same date across methods
        dfs = [pd.read_csv(f, parse_dates=['datetime']).set_index('datetime') for f in files_group]
        date = pd.to_datetime(files_group[0].split('_')[-1].replace('.csv', ''), format='%Y%j').strftime('%Y-%m-%d')

        # Compute common timestamps
        common_index = reduce(lambda a, b: a.intersection(b), [df.index for df in dfs])
        if common_index.empty:
            print(f"[{date}] No common timestamps across methods, skipping.")
            continue

        # Trim each dataframe to common timestamps and recompute stats
        for method, df in zip(all_methods, dfs):
            df_common = df.loc[common_index]

            row = {
                'date': date,
                'centre': method,
                'RMS_dx': np.sqrt(np.mean(df_common['dx']**2)),
                'RMS_dy': np.sqrt(np.mean(df_common['dy']**2)),
                'RMS_dz': np.sqrt(np.mean(df_common['dz']**2)),
                'RMS_3D': np.sqrt(np.mean(df_common['3D']**2)),
                'Mean_3D': np.mean(df_common['3D']),
                'Std_3D': np.std(df_common['3D']),
                'N_points': len(df_common)
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(['date', 'centre'])

    # Output path
    if output_path is None:
        output_path = os.path.join(residuals_base_path, "residual_summary_stats_common_epochs.csv")
    
    summary_df.to_csv(output_path, index=False)
    print(f"\n Summary with common epochs written to: {output_path}")
    

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14
})

# %% === Setup ===
AC_input = ["IFG", "AIUB", "TUD", "inversevariance", "vce", "mean", 
            "residualweighted", "neldermead", "cmaes"]
AC_input = ["inversevariance", "vce"]
satellite_ids = ['47']
analysis_centres = AC_input
residuals_base_path = r"results/"
os.makedirs(residuals_base_path, exist_ok=True)

arclength = pd.Timedelta(days=1)
filtering_threshold = 0.3
orbit_duration = MonthBegin(1)

startdates = [datetime(year, month, 1) for year in [2023] for month in range(2, 3)]
summary_df_path = os.path.join(residuals_base_path, "residual_summary_stats.csv")

# === Write CSV header only once ===
if not os.path.exists(summary_df_path):
    pd.DataFrame(columns=['date', 'centre', 'RMS_dx', 'RMS_dy', 'RMS_dz',
                          'RMS_3D', 'Mean_3D', 'Std_3D', 'N_points']).to_csv(summary_df_path, index=False)
    
# === Main Loop (runs only if summary file does not exist) ===
if not os.path.exists(summary_df_path):
    print(f"[INFO] {summary_df_path} not found. Running residual computation...")
    # === Main Loop ===
    for idx, start in enumerate(startdates):
        print(f"\n>> Processing: {start.strftime('%Y-%m')}")
        
        # Access requests for this month only
        i1_req = generate_access_requests("TEST", [start], analysis_centre=AC_input, get_data=False, orbit_duration=orbit_duration, satellite_ids=satellite_ids)
        i2_req = generate_access_requests("TEST", [start], analysis_centre=["ESA"], get_data=False, orbit_duration=orbit_duration, satellite_ids=satellite_ids)
    
        i1 = retrieve_arcs(i1_req[0])
        i2 = retrieve_arcs(i2_req[0])
    
        input_arcs = i1.get(satellite_ids[0], [])
        ref_arcs = i2.get(satellite_ids[0], [])
    
        year = start.year
        doy = str(start.timetuple().tm_yday).rjust(3, '0')
        date_str = f"{year}{doy}"
        date_label = start.strftime('%Y-%m-%d')
        year_dir = os.path.join(residuals_base_path, str(year))
        os.makedirs(year_dir, exist_ok=True)
    
        if len(ref_arcs) == 0:
            print(f"  [!] No ESA reference data for {date_label}, skipping.")
            continue
    
        reference_df = ref_arcs[0].trajectory.copy()
        stat_rows = []
    
        for centre, input_arc in zip(analysis_centres, input_arcs):
            input_df = input_arc.trajectory.copy()
            common_times = input_df.index.intersection(reference_df.index)
    
            # if common_times.empty:
            #     print(f"  [!] No overlap for {centre} at {date_label}, skipping.")
            #     continue
    
            input_trimmed = input_df
            ref_trimmed = reference_df
            residuals = pd.DataFrame(index=common_times)
            residuals['dx'] = input_trimmed['x_pos'] - ref_trimmed['x_pos']
            residuals['dy'] = input_trimmed['y_pos'] - ref_trimmed['y_pos']
            residuals['dz'] = input_trimmed['z_pos'] - ref_trimmed['z_pos']
            residuals['3D'] = np.linalg.norm(residuals[['dx', 'dy', 'dz']].values, axis=1)
            residuals = residuals.dropna(how='all')
            
            # Save residual file
            filename = f"{centre}_vs_ESA_residuals_{date_str}.csv"
            filepath = os.path.join(year_dir, filename)
            residuals.to_csv(filepath, index_label='datetime')
    
            # Save stats row
            stats = {
                'date': date_label,
                'centre': centre,
                'RMS_dx': np.sqrt(np.mean(residuals['dx']**2)),
                'RMS_dy': np.sqrt(np.mean(residuals['dy']**2)),
                'RMS_dz': np.sqrt(np.mean(residuals['dz']**2)),
                'RMS_3D': np.sqrt(np.mean(residuals['3D']**2)),
                'Mean_3D': np.mean(residuals['3D']),
                'Std_3D': np.std(residuals['3D']),
                'N_points': len(residuals)
            }
            stat_rows.append(stats)
    
        # Append stats for this date to summary file
        if stat_rows:
            pd.DataFrame(stat_rows).to_csv(summary_df_path, index=False, mode='a', header=False)
    else:
        print(f"[INFO] {summary_df_path} already exists. Skipping residual computation.")

#%%
residuals_base_path = r"results/"
# recompute_summary_from_common_epochs(residuals_base_path)
summary_csv = os.path.join(residuals_base_path, "residual_summary_stats.csv")
df = pd.read_csv(summary_csv)
plot_rms_with_errorbars(summary_csv)
plot_overall_rms(summary_csv, save_path=residuals_base_path)

