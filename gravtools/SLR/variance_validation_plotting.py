# -*- coding: utf-8 -*-
"""
variance_verification_plotting.py
 
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
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_sigma_data_and_metrics(method_labels, output_dir, excluded_dates):
    loaded_data = {}
    metrics_list = []

    for label in method_labels:
        file_path = Path(output_dir) / f'slr_data_with_sigma_{label}.pkl'
        if file_path.exists():
            df = pd.read_pickle(file_path)
            if excluded_dates:
                excluded_dates = [pd.to_datetime(d).date() for d in excluded_dates]
                df = df[~pd.Series(df.index.date).isin(excluded_dates).values]
            
            loaded_data[label] = df[df['Method'] == label]
            
            chi2 = (df['Res[m]'] / df['orbit_sigma_los[m]']).pow(2).mean()
            metrics_list.append({
                'Method': label,
                'Reduced Chi2': chi2,
                'SLR RMS [m]': df['Res[m]'].std(),
                'Mean Orbit STD LOS [m]': df['orbit_sigma_los[m]'].mean()
            })
        else:
            print(f"Warning: {file_path} not found!")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df['Scaling factor'] = np.sqrt(metrics_df['Reduced Chi2'])
    return loaded_data, metrics_df


def plot_residuals_with_sigma(loaded_data, methods_to_plot, output_dir):
    for label in methods_to_plot:
        df = loaded_data[label]
        residuals = df['Res[m]']

        # === Group residuals by day and compute daily mean and std ===
        df_daily = df.resample('1D').agg({'Res[m]': ['mean', 'std']})
        df_daily.columns = ['mean_residual', 'std_residual']
        df_daily['+1std'] = df_daily['mean_residual'] + df_daily['std_residual']
        df_daily['-1std'] = df_daily['mean_residual'] - df_daily['std_residual']

        # === Plot ===
        plt.figure(figsize=(6.5, 6))

        # Scatter plot of raw residuals
        plt.scatter(df.index, residuals, label='SLR Residual [m]', alpha=0.7, s=10)

        # Red envelope: daily mean ± std
        plt.fill_between(df_daily.index, df_daily['+1std'], df_daily['-1std'],
                         color='red', alpha=0.1, label='Daily Mean ± STD (Residuals)')

        # Green envelope: ± orbit STD LOS (plotted second so it’s on top)
        plt.fill_between(df.index, df['orbit_sigma_los[m]'], -df['orbit_sigma_los[m]'],
                         color='green', alpha=0.1, label='± Orbit STD LOS')

        # Axis formatting
        plt.ylabel('Metres')
        # plt.xlabel('Day of Month')
        plt.xticks(rotation=30)
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
        # plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: mdates.num2date(x).strftime('%d')))

        # # Set x-axis range for January 2023
        # start = pd.to_datetime('2023-01-01')
        # stop = pd.to_datetime('2023-01-31')
        # plt.xlim(mdates.date2num(start), mdates.date2num(stop))

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'slr_residuals_sigma_{label}.png'), dpi=300)
        # plt.close()

def plot_residuals_with_sigma_daily(loaded_data, methods_to_plot, output_dir):
    for label in methods_to_plot:
        df = loaded_data[label]
        residuals = df['Res[m]']

        # === Compute overall residual RMS for reference ===
        overall_rms = np.sqrt(np.mean(residuals ** 2))

        # === Group residuals by day ===
        df['Day'] = df.index.floor('D')
        df_daily = df.groupby('Day').agg({
            'Res[m]': ['mean', 'std'],
            'orbit_sigma_los[m]': 'mean'
        })
        df_daily.columns = ['mean_residual', 'std_residual', 'mean_orbit_sigma_los']
        df_daily['+1std'] = df_daily['mean_residual'] + df_daily['std_residual']
        df_daily['-1std'] = df_daily['mean_residual'] - df_daily['std_residual']
        df_daily['+orbit_std'] = df_daily['mean_orbit_sigma_los']
        df_daily['-orbit_std'] = -df_daily['mean_orbit_sigma_los']

        # Ensure continuous daily coverage and append one more day
        full_day_range = pd.date_range(df['Day'].min(), df['Day'].max(), freq='D')
        full_day_range = full_day_range.append(pd.DatetimeIndex([full_day_range[-1] + pd.Timedelta(days=1)]))
        df_daily = df_daily.reindex(full_day_range)
        df_daily.index.name = 'Date'

        # === Plot ===
        fig, ax = plt.subplots(figsize=(15, 6))

        # Raw SLR residuals
        ax.scatter(df.index, residuals, label='SLR Residual [m]', alpha=0.7, s=4)

        # Daily mean line
        for i in range(len(df_daily) - 1):
            x_start = df_daily.index[i]
            x_end = df_daily.index[i + 1]
            y_mean = df_daily['mean_residual'].iloc[i]
            ax.plot([x_start, x_end], [y_mean, y_mean], color='orange', lw=1.5,
                    label='Daily Mean Residual' if i == 0 else "")

        # Daily ±1 std envelope (around mean)
        for i in range(len(df_daily) - 1):
            x_start = df_daily.index[i]
            x_end = df_daily.index[i + 1]
            y_top = df_daily['+1std'].iloc[i]
            y_bot = df_daily['-1std'].iloc[i]
            ax.fill_between([x_start, x_end], [y_top, y_top], [y_bot, y_bot],
                            color='red', alpha=0.1, label='Daily Residual Mean ± STD' if i == 0 else "")

        # Daily orbit LOS STD envelope (around 0)
        for i in range(len(df_daily) - 1):
            x_start = df_daily.index[i]
            x_end = df_daily.index[i + 1]
            y_top = df_daily['+orbit_std'].iloc[i]
            y_bot = df_daily['-orbit_std'].iloc[i]
            ax.fill_between([x_start, x_end], [y_top, y_top], [y_bot, y_bot],
                            color='green', alpha=0.1, label='± Daily Orbit LOS STD' if i == 0 else "")

        # Overlay constant ±RMS line
        ax.axhline(overall_rms, color='red', linestyle='--', linewidth=1.2, label=f'Overall RMS = {overall_rms:.4f} m')

        # Axis formatting
        ax.set_ylabel('Metres')
        ax.set_xlabel('Date')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True)
        ax.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=1, zorder=0)

        ax.legend(ncols = 3, framealpha = 0.5)
        # plt.title(f'SLR Residuals and Orbit LOS STD Envelopes ({label})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'slr_residuals_sigma_{label}.png'), dpi=300)
        # plt.close()


def plot_residual_spread_vs_orbit_sigma(loaded_data, methods_to_plot, output_dir):
    

    for label in methods_to_plot:
        df = loaded_data[label]
        df['Day'] = df.index.floor('D')

        # === Compute overall RMS for constant reference line ===
        overall_rms = np.sqrt(np.mean(df['Res[m]'] ** 2))

        # === Group by day and compute spread metrics ===
        df_daily = df.groupby('Day').agg({
            'Res[m]': 'std',
            'orbit_sigma_los[m]': 'mean'
        })
        df_daily.columns = ['std_residual', 'mean_orbit_sigma_los']

        # Construct ±1 std envelopes centred at 0
        df_daily['+residual_std'] = df_daily['std_residual']
        df_daily['-residual_std'] = -df_daily['std_residual']
        df_daily['+orbit_std'] = df_daily['mean_orbit_sigma_los']
        df_daily['-orbit_std'] = -df_daily['mean_orbit_sigma_los']

        # Ensure final day gets included
        full_day_range = pd.date_range(df['Day'].min(), df['Day'].max(), freq='D')
        full_day_range = full_day_range.append(pd.DatetimeIndex([full_day_range[-1] + pd.Timedelta(days=1)]))
        df_daily = df_daily.reindex(full_day_range)
        df_daily.index.name = 'Date'

        # === Plot ===
        fig, ax = plt.subplots(figsize=(8, 7))

        for i in range(len(df_daily) - 1):
            x_start = df_daily.index[i]
            x_end = df_daily.index[i + 1]

            # Residual ±STD envelope (red)
            y_top_r = df_daily['+residual_std'].iloc[i]
            y_bot_r = df_daily['-residual_std'].iloc[i]
            ax.fill_between([x_start, x_end], [y_top_r, y_top_r], [y_bot_r, y_bot_r],
                            color='red', alpha=0.2, label='Residual STD (±1σ)' if i == 0 else "")

            # Orbit ±LOS STD envelope (green)
            y_top_o = df_daily['+orbit_std'].iloc[i]
            y_bot_o = df_daily['-orbit_std'].iloc[i]
            ax.fill_between([x_start, x_end], [y_top_o, y_top_o], [y_bot_o, y_bot_o],
                            color='green', alpha=0.2, label='Orbit LOS STD (±1σ)' if i == 0 else "")

        # Overlay constant RMS reference line
        ax.axhline(overall_rms, color='red', linestyle='--', linewidth=1.2, label=f'Overall RMS = {overall_rms:.4f} m')

        # Axis formatting
        ax.set_ylabel('Metres')
        ax.set_xlabel('Date')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True)
        ax.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=1, zorder=0)

        ax.legend(ncols = 2, fontsize = 14)
        # plt.title(f'Residual STD vs Orbit LOS STD (Spread Only) – {label}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'slr_residuals_spread_{label}.png'), dpi=300)
        # plt.close()






def plot_histograms_norm_residuals(loaded_data, output_dir=None):
    for label, df in loaded_data.items():
        plt.figure(figsize=(6.5, 6))
        norm_res = df['Res[m]'] / df['orbit_sigma_los[m]']
        plt.hist(norm_res, bins=50, alpha=0.7)
        # plt.title(f'Normalised Residuals Histogram ({label})')
        plt.xlabel('Residual / Projected Orbit STD LOS')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'norm_residuals_histogram_{label}.png'), dpi=300)
            # plt.close()
        else:
            plt.show()

def plot_reduced_chi2_bar(metrics_df, output_dir=None):
    plt.figure(figsize=(6.5, 6))
    plt.bar(metrics_df['Method'], metrics_df['Reduced Chi2'], color='skyblue')
    plt.axhline(1, color='red', linestyle='--', label='Ideal Chi-square = 1')
    # plt.title('Reduced Chi-square per Method')
    plt.ylabel('Reduced Chi-square')
    plt.grid(True, axis='y')
    plt.tick_params(axis='x', rotation=30)
    plt.gca().set_xticklabels(
        plt.gca().get_xticklabels(),
        ha='right'  # aligns labels to the right
    )
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    print(metrics_df[['Method', 'Reduced Chi2', 'Scaling factor']])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'reduced_chi2_bar.png'), dpi=300)
        # plt.close()
    else:
        plt.show()

def plot_reduced_chi2_bar(metrics_df, output_dir=None):
    # === Filter out RR entries ===
    df = metrics_df[~metrics_df['Method'].str.endswith('RR')].copy()

    # === Extract base method label (e.g., "IFG" from "IFGRS") ===
    df['Base_Method'] = df['Method'].str.replace('RS', '', regex=False)

    # === Define bar categories ===
    def assign_group(method):
        if method in ['IFG', 'AIUB', 'TUD']:
            return 'Input Orbit'
        elif method in ['IFGRS', 'AIUBRS', 'TUDRS']:
            return 'Input Orbit (Scaled)'
        elif method in ['mean', 'inversevariance', 'vce', 'residualweighted']:
            return 'Combined Orbit'
        elif method in ['meanRS', 'inversevarianceRS', 'vceRS', 'residualweightedRS']:
            return 'Combined Orbit (Scaled)'
        else:
            return 'Other'

    df['Group'] = df['Method'].apply(assign_group)

    # === Define display order based on base methods ===
    ordered_methods = ['IFG', 'AIUB', 'TUD', 'mean', 'inversevariance', 'vce', 'residualweighted']
    x = np.arange(len(ordered_methods))  # x locations for the groups
    bar_width = 0.35

    # === Colours ===
    # === Colour mapping using full group labels ===
    color_map = {
        'Input Orbit': '#1f77b4',
        'Input Orbit (Scaled)': '#aec7e8',
        'Combined Orbit': '#ff7f0e',
        'Combined Orbit (Scaled)': '#ffbb78'
    }


    # === Collect values ===
    unscaled_vals, scaled_vals = [], []
    unscaled_colors, scaled_colors = [], []

    for method in ordered_methods:
        # Unscaled
        row = df[(df['Base_Method'] == method) & (~df['Method'].str.contains('RS'))]
        if not row.empty:
            val = row['Reduced Chi2'].values[0]
            group = row['Group'].values[0]
            unscaled_vals.append(val)
            unscaled_colors.append(color_map.get(group, 'grey'))
        else:
            unscaled_vals.append(np.nan)
            unscaled_colors.append('grey')

        # Scaled
        row = df[(df['Base_Method'] == method) & (df['Method'].str.contains('RS'))]
        if not row.empty:
            val = row['Reduced Chi2'].values[0]
            group = row['Group'].values[0]
            scaled_vals.append(val)
            scaled_colors.append(color_map.get(group, 'lightgrey'))
        else:
            scaled_vals.append(np.nan)
            scaled_colors.append('lightgrey')

    # === Plot ===
    plt.figure(figsize=(6.5, 6))

    plt.bar(x - bar_width / 2, unscaled_vals, width=bar_width, color=unscaled_colors, label='Unscaled')
    plt.bar(x + bar_width / 2, scaled_vals, width=bar_width, color=scaled_colors, label='Scaled')

    # Axes formatting
    plt.axhline(1, color='red', linestyle='--', label='Ideal $\chi^2 = 1$')
    plt.xticks(x, ordered_methods, rotation=30, ha='right')
    plt.ylabel('Reduced $\chi^2$')
    plt.yscale('log')
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Input Orbit (Unscaled)'),
        Patch(facecolor='#aec7e8', label='Input Orbit (Scaled)'),
        Patch(facecolor='#ff7f0e', label='Combined Orbit (Unscaled)'),
        Patch(facecolor='#ffbb78', label='Combined Orbit (Scaled)'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Ideal $\chi^2 = 1$')
    ]
    plt.legend(handles=legend_elements, loc='upper right', draggable=True, fontsize=12, ncols=2)

    # Optional print
    print(df[['Method', 'Group', 'Reduced Chi2', 'Scaling factor']])

    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'reduced_chi2_grouped_sidebyside.png'), dpi=300)
    else:
        plt.show()



def plot_rms_vs_chi2(metrics_df, output_dir):
    plt.figure(figsize=(6.5, 6))
    for _, row in metrics_df.iterrows():
        plt.scatter(row['Reduced Chi2'], row['SLR RMS [m]'], label=row['Method'], s=100)
        # plt.text(row['Reduced Chi2'] * 1.05, row['SLR RMS [m]'], row['Method'], fontsize=9)
    plt.axvline(1, color='red', linestyle='--', label='Ideal Chi-square = 1')
    plt.xlabel('Reduced Chi-square')
    plt.ylabel('SLR Residual RMS [m]')
    # plt.title('Residual RMS vs Reduced Chi2')
    plt.grid(True)
    plt.legend()
    plt.tick_params(axis='x', rotation=30)
    plt.gca().set_xticklabels(
        plt.gca().get_xticklabels(),
        ha='right'  # aligns labels to the right
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rms_vs_chi2.png'), dpi=300)
    # plt.close()

def plot_combined_normalised_histograms(loaded_data, output_dir):
    plt.figure(figsize=(6.5, 6))
    for label, df in loaded_data.items():
        norm_res = df['Res[m]'] / df['orbit_sigma_los[m]']
        plt.hist(norm_res, bins=50, alpha=0.4, label=label, density=True)
    # plt.title('Normalised Residuals Histogram (All Methods)')
    plt.xlabel('Residual / Projected Orbit STD LOS')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'combined_norm_residuals_histogram.png'), dpi=300)
    # plt.close()
    

def plot_combined_daily_rms_to_std_ratio(loaded_data, output_dir, figsize=(13, 6)):
    """
    Plot combined daily RMS to orbit STD LOS ratio for all methods in a single figure.

    Parameters
    ----------
    loaded_data : dict
        Dictionary with {method_label: DataFrame} containing columns 'Res[m]' and 'orbit_sigma_los[m]'.
    output_dir : Path
        Directory where to save the figure.
    figsize : tuple
        Figure size.
    """
    plt.figure(figsize=figsize)

    for label, df in loaded_data.items():
        # Ensure datetime index
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        # Group by day
        daily_groups = df.groupby(df.index.date)

        daily_rms = daily_groups['Res[m]'].apply(lambda x: np.sqrt(np.mean(x**2)))
        daily_std = daily_groups['orbit_sigma_los[m]'].mean()
        ratio = daily_rms / daily_std

        plt.plot(daily_rms.index, ratio, marker='o', linestyle='-', label=label)

    # Plot ideal line
    plt.axhline(1, color='red', linestyle='--', label='Ideal Ratio = 1')
    # plt.title('Daily RMS to Orbit STD LOS Ratio (All Methods)')
    plt.ylabel('RMS Residual / Mean Orbit STD LOS')
    plt.grid(True)
    plt.legend()
    plt.tick_params(axis='x', rotation=30)
    plt.gca().set_xticklabels(
        plt.gca().get_xticklabels(),
        ha='right'  # aligns labels to the right
    )
    plt.tight_layout()
    
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # every 2nd day
    # plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: mdates.num2date(x).strftime('%d')))
    # # Force x-limits from 1 Jan to 31 Jan explicitly
    # start = pd.to_datetime('2023-01-01')
    # stop = pd.to_datetime('2023-01-31')
    
    # plt.xlim(mdates.date2num(start), mdates.date2num(stop))
    # plt.xlabel('Day of Month')
    # Save
    fig_name = os.path.join(output_dir, f'daily_rms_std_ratio_combined.png')
    plt.savefig(fig_name, dpi=300)
    # plt.close()

    print(f"Saved combined figure: {fig_name}")
    

def compute_variance_validation_metrics(loaded_data):
    results = []

    for label, df in loaded_data.items():
        # Ensure clean data
        df = df.dropna(subset=['Res[m]', 'orbit_sigma_los[m]'])

        # Metrics
        slr_rms = np.sqrt(np.mean(df['Res[m]'] ** 2))
        mean_sigma_los = df['orbit_sigma_los[m]'].mean()
        rms_ratio = slr_rms / mean_sigma_los if mean_sigma_los > 0 else np.nan
        mean_residual = df['Res[m]'].mean()
        df['norm_residual'] = df['Res[m]'] / df['orbit_sigma_los[m]']
        reduced_chi2 = np.mean(df['norm_residual'] ** 2)
        
        rms_scaling = rms_ratio
        chi2_scaling = np.sqrt(reduced_chi2)
        
        # Collect
        results.append({
            'Method': label,
            'SLR RMS [m]': slr_rms,
            'Mean Residual [m]': mean_residual,
            'Mean STD LOS [m]': mean_sigma_los,
            'RMS/STD': rms_ratio,
            'Reduced Chi2': reduced_chi2,
            'RMS Scaling': rms_scaling,
            'Chi2 Scaling': chi2_scaling
        })

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.round({
        'SLR RMS [m]': 4,
        'Mean Residual [m]': 4,
        'Mean STD LOS [m]': 4,
        'RMS/STD': 2,
        'Reduced Chi2': 2
    })

    return metrics_df

def compute_daily_rms_std_stats(loaded_data):
    """
    Compute the mean, standard deviation, and coefficient of variation (CV) 
    of the daily RMS / STD (LOS-projected) ratio for each method.

    Parameters
    ----------
    loaded_data : dict
        Dictionary where keys are method names and values are DataFrames with 
        'Res[m]' and 'orbit_sigma_los[m]'.

    Returns
    -------
    pd.DataFrame
        DataFrame with Method, Mean RMS/STD, Std RMS/STD, and CV RMS/STD columns.
    """
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    stats = defaultdict(list)

    for method, df in loaded_data.items():
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        # Ensure valid data
        df = df.dropna(subset=['Res[m]', 'orbit_sigma_los[m]'])
        df = df[df['orbit_sigma_los[m]'] > 0]

        if df.empty:
            continue

        # Group by day
        df['day'] = df['Date'].dt.floor('D')

        # Daily RMS of residuals
        daily_rms = df.groupby('day')['Res[m]'].apply(lambda x: np.sqrt(np.mean(x**2)))
        # Daily mean of orbit sigma LOS
        daily_std = df.groupby('day')['orbit_sigma_los[m]'].mean()

        # Calculate RMS/STD ratio (aligned by day)
        ratio = daily_rms / daily_std
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

        stats['Method'].append(method)
        stats['Mean RMS/STD'].append(ratio.mean())
        stats['Std RMS/STD'].append(ratio.std())
        stats['CV RMS/STD'].append(ratio.std() / ratio.mean() if ratio.mean() != 0 else np.nan)

    return pd.DataFrame(stats).sort_values("Method")

def plot_rms_std_stability(loaded_data, figsize=(7, 6)):
    """
    Plot the mean RMS/STD ratio with standard deviation error bars for each method,
    using daily residual and STD values from SLR validation.

    Parameters
    ----------
    loaded_data : dict
        Dictionary where keys are method names and values are DataFrames with 
        'Date', 'Res[m]', and 'orbit_sigma_los[m]' columns.
    figsize : tuple
        Size of the plot figure.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    methods = []
    means = []
    stds = []

    for method, df in loaded_data.items():
        if 'RS' in method:
            method = method[:-2]
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['Res[m]', 'orbit_sigma_los[m]'])
        df = df[df['orbit_sigma_los[m]'] > 0]

        if df.empty:
            continue

        df['day'] = df['Date'].dt.floor('D')
        daily_rms = df.groupby('day')['Res[m]'].apply(lambda x: np.sqrt(np.mean(x**2)))
        daily_std = df.groupby('day')['orbit_sigma_los[m]'].mean()
        ratio = (daily_rms / daily_std).replace([np.inf, -np.inf], np.nan).dropna()

        methods.append(method)
        means.append(ratio.mean())
        stds.append(ratio.std())

    # Plot
    plt.figure(figsize=figsize)
    plt.errorbar(methods, means, yerr=stds, fmt='o', capsize=5, elinewidth=1.5, markeredgewidth=1.5)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Ideal RMS/STD = 1')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("RMS/STD Ratio")
    # plt.title("Calibration and Temporal Stability of Orbit Uncertainty")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
