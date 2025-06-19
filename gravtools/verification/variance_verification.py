# -*- coding: utf-8 -*-
"""
variance_verification.py
 
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
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from matplotlib.backends.backend_pdf import PdfPages
from gravtools.kinematic_orbits.combine_orbits import combine_orbits

# ===========================================
# Generate Keplerian orbit as truth
# ===========================================
def generate_keplerian_orbit(n_epochs=1440, semi_major_axis_km=7000, eccentricity=0.001,
                             inclination_deg=98.0, raan_deg=0.0, arg_perigee_deg=0.0,
                             mu_earth=398600.4418, start_date="2020-01-01"):
    inclination = np.deg2rad(inclination_deg)
    raan = np.deg2rad(raan_deg)
    arg_perigee = np.deg2rad(arg_perigee_deg)
    a = semi_major_axis_km * 1e3
    n = np.sqrt(mu_earth * 1e9 / a ** 3)
    index = pd.date_range(start=start_date, periods=n_epochs, freq="min")
    t_sec = np.arange(n_epochs)
    M = n * t_sec
    nu = M
    r = a * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(nu))
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = np.zeros_like(x_orb)
    def rotation_matrix(raan, inclination, arg_perigee):
        return np.array([
            [np.cos(raan) * np.cos(arg_perigee) - np.sin(raan) * np.sin(arg_perigee) * np.cos(inclination),
             -np.cos(raan) * np.sin(arg_perigee) - np.sin(raan) * np.cos(arg_perigee) * np.cos(inclination),
             np.sin(raan) * np.sin(inclination)],
            [np.sin(raan) * np.cos(arg_perigee) + np.cos(raan) * np.sin(arg_perigee) * np.cos(inclination),
             -np.sin(raan) * np.sin(arg_perigee) + np.cos(raan) * np.cos(arg_perigee) * np.cos(inclination),
             -np.cos(raan) * np.sin(inclination)],
            [np.sin(arg_perigee) * np.sin(inclination),
             np.cos(arg_perigee) * np.sin(inclination),
             np.cos(inclination)]
        ])
    R = rotation_matrix(raan, inclination, arg_perigee)
    r_eci = np.dot(R, np.vstack((x_orb, y_orb, z_orb)))
    df = pd.DataFrame({
        'x_pos': r_eci[0, :],
        'y_pos': r_eci[1, :],
        'z_pos': r_eci[2, :]
    }, index=index)
    df[['std_x', 'std_y', 'std_z', 'cov_xy', 'cov_xz', 'cov_yz']] = 0.0
    return df

# ===========================================
# Generate improved synthetic data with variability
# ===========================================


def generate_synthetic_data_from_truth(
    truth, n_inputs=3, missing_frac=0.005,
    std_bounds=(0.5, 2.5), cov_bounds=(0.01, 0.1),
    random_seed=None, return_noise_info=False,
    fixed_std=None, fixed_cov=None
):
    if random_seed is not None:
        np.random.seed(random_seed)

    n_epochs = len(truth)
    coords = ['x', 'y', 'z']
    cov_pairs = ['xy', 'xz', 'yz']

    # Handle fixed or random stds
    if isinstance(std_bounds, dict):
        # input_vars = {
        #     coord: np.full(n_inputs, std_bounds[coord][0]**2)
        #     for coord in coords
        # }
        input_vars = {
            coord: np.random.uniform(std_bounds[coord][0]**2, std_bounds[coord][1]**2, n_inputs)
            for coord in coords
        }

    else:
        input_vars = {
            coord: np.random.uniform(std_bounds[0]**2, std_bounds[1]**2, n_inputs)
            for coord in coords
        }

    if isinstance(cov_bounds, dict):
        base_cov = {
            pair: np.random.uniform(cov_bounds[pair][0], cov_bounds[pair][1])
            for pair in cov_pairs
        }
    else:
        base_cov = {
            pair: np.random.uniform(cov_bounds[0], cov_bounds[1])
            for pair in cov_pairs
        }

    dataframes = []
    noise_info_rows = []

    for i in range(n_inputs):
        df = truth.copy()

        std = {}
        for coord in coords:
            base_std = np.sqrt(input_vars[coord][i])
            std[coord] = base_std * (1 + 0.01 * np.random.randn(n_epochs))

        cov_epoch = {}
        for pair in cov_pairs:
            base_val = base_cov[pair]
            cov_epoch[pair] = np.full(n_epochs, base_val) if fixed_cov else base_val * (1 + 0.01 * np.random.randn(n_epochs))

        noise_xyz = np.zeros((n_epochs, 3))
        for j in range(n_epochs):
            cov_matrix = np.array([
                [std['x'][j] ** 2, cov_epoch['xy'][j], cov_epoch['xz'][j]],
                [cov_epoch['xy'][j], std['y'][j] ** 2, cov_epoch['yz'][j]],
                [cov_epoch['xz'][j], cov_epoch['yz'][j], std['z'][j] ** 2]
            ])
            try:
                noise_xyz[j, :] = np.random.multivariate_normal([0, 0, 0], cov_matrix)
            except np.linalg.LinAlgError:
                noise_xyz[j, :] = 0.0

        df['x_pos'] += noise_xyz[:, 0]
        df['y_pos'] += noise_xyz[:, 1]
        df['z_pos'] += noise_xyz[:, 2]
        df['std_x'], df['std_y'], df['std_z'] = std['x'], std['y'], std['z']
        df['cov_xy'], df['cov_xz'], df['cov_yz'] = cov_epoch['xy'], cov_epoch['xz'], cov_epoch['yz']

        mask = np.random.rand(n_epochs) < missing_frac
        df.loc[mask] = np.nan
        dataframes.append(df)

        if return_noise_info:
            noise_info_rows.append({
                'Orbit': i + 1,
                'STD_X': np.nanmean(std['x']),
                'STD_Y': np.nanmean(std['y']),
                'STD_Z': np.nanmean(std['z']),
                'COV_XY': np.nanmean(cov_epoch['xy']),
                'COV_XZ': np.nanmean(cov_epoch['xz']),
                'COV_YZ': np.nanmean(cov_epoch['yz'])
            })

    if return_noise_info:
        noise_df = pd.DataFrame(noise_info_rows)
        return dataframes, noise_df
    else:
        return dataframes

def generate_mixed_anisotropy_test(truth, missing_frac=0.0, fixed_cov=None, random_seed=42):
    std_bounds_list = [
        {'x': (1, 1), 'y': (3, 3), 'z': (3, 3)},  # x-dominant
        {'x': (3, 3), 'y': (1, 1), 'z': (3, 3)},  # y-dominant
        {'x': (3, 3), 'y': (3, 3), 'z': (1, 1)}   # z-dominant
    ]

    all_dfs = []
    all_noise = []

    for i, std_bounds in enumerate(std_bounds_list):
        dfs, noise = generate_synthetic_data_from_truth(
            truth=truth,
            n_inputs=1,
            std_bounds=std_bounds,
            cov_bounds=(0, 0),
            fixed_cov=fixed_cov,
            missing_frac=missing_frac,
            random_seed=random_seed + i,
            return_noise_info=True
        )
        all_dfs.append(dfs[0])
        all_noise.append(noise)

    combined_noise_df = pd.concat(all_noise, ignore_index=True)
    return all_dfs, combined_noise_df, truth





# ===========================================
# Drop epochs where all inputs are missing
# ===========================================
def drop_epochs_with_no_data(dataframes, truth):
    combined_positions = pd.concat([df[['x_pos', 'y_pos', 'z_pos']] for df in dataframes], axis=1)
    valid_epochs = combined_positions.notna().any(axis=1)
    dataframes = [df.loc[valid_epochs] for df in dataframes]
    truth = truth.loc[valid_epochs]
    return dataframes, truth

# ===========================================
# Validation and plotting functions
# ===========================================



def plot_3d_input_trajectories(dataframes):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colours = ['tab:blue', 'tab:orange', 'tab:green']
    for i, df in enumerate(dataframes):
        mask = df[['x_pos', 'y_pos', 'z_pos']].notna().all(axis=1)
        x = df.loc[mask, 'x_pos']
        y = df.loc[mask, 'y_pos']
        z = df.loc[mask, 'z_pos']
        ax.plot3D(x, y, z, label=f'Orbit {i+1}', color=colours[i])

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title('Synthetic Input Trajectories')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def compute_validation_metrics(combined, truth_aligned):
    residuals = combined[['x_pos', 'y_pos', 'z_pos']] - truth_aligned[['x_pos', 'y_pos', 'z_pos']]
    method_metrics = {'std': {}, 'cov': {}, 'rms': None, 'chi2': None}

    for coord in ['x', 'y', 'z']:
        reported = combined[f'std_{coord}'].mean()
        empirical = residuals[f'{coord}_pos'].std()
        method_metrics['std'][coord] = (reported, empirical)

    empirical_cov = residuals.cov()
    for cov_pair in ['xy', 'xz', 'yz']:
        reported = combined[f'cov_{cov_pair}'].mean()
        empirical = empirical_cov.loc[f'{cov_pair[0]}_pos', f'{cov_pair[1]}_pos']
        method_metrics['cov'][cov_pair] = (reported, empirical)

    rms_error = np.sqrt((residuals**2).mean()).mean()
    avg_std = combined[['std_x', 'std_y', 'std_z']].mean().mean()
    method_metrics['rms'] = (rms_error, avg_std)

    chi2_values = []
    for coord in ['x', 'y', 'z']:
        res = residuals[f'{coord}_pos']
        std = combined[f'std_{coord}']

        # Mask per coordinate only
        mask = (std > 0) & std.notna() & res.notna()

        if mask.sum() == 0:
            print(f"Warning: No valid data for chi-square calculation in {coord.upper()}")
            chi2_values.append(np.nan)
        else:
            chi2 = np.mean((res[mask] / std[mask]) ** 2)
            chi2_values.append(chi2)

    chi2 = np.nanmean(chi2_values)
    method_metrics['chi2'] = chi2

    return method_metrics, residuals



def print_validation_report_enhanced(results):
    print("\n=== Enhanced Validation Report ===")
    for i, method in enumerate(results['methods']):
        print(f"\nMethod: {method}")
        print("\nStandard Deviations:")
        for coord in ['x', 'y', 'z']:
            reported, empirical = results['metrics']['std'][coord][i]
            diff = abs(reported - empirical)
            status = "PASS" if diff < 0.15 * reported else "FAIL"
            print(f"{coord.upper()}: Reported {reported:.2f} vs Empirical {empirical:.2f} | Δ={diff:.2f} ({status})")
        print("\nCovariances:")
        for cov_pair in ['xy', 'xz', 'yz']:
            reported, empirical = results['metrics']['cov'][cov_pair][i]
            diff = abs(reported - empirical)
            status = "PASS" if diff < 0.2 * abs(reported) else "FAIL"
            print(f"{cov_pair.upper()}: Reported {reported:.2f} vs Empirical {empirical:.2f} | Δ={diff:.2f} ({status})")
        rms, avg_std = results['metrics']['rms'][i]
        diff = abs(rms - avg_std)
        status = "PASS" if diff < 0.15 * avg_std else "FAIL"
        print(f"\nRMS Error: {rms:.2f} vs Avg STD: {avg_std:.2f} | Δ={diff:.2f} ({status})")
        chi2 = results['metrics']['chi2'][i]
        print(f"Chi-Square (reduced): {chi2:.2f} | {'PASS' if np.abs(chi2 - 1) < 0.2 else 'FAIL'}")


def plot_residuals_vs_reported_uncertainty_with_binned(results):
    figs = {}
    for coord in ['x', 'y', 'z']:
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, method in enumerate(results['methods']):
            
            method_label = method.replace('_', ' ')
            
            residuals = results['residuals'][i][f'{coord}_pos']
            reported_std = results['reported_std'][i][f'std_{coord}']

            valid_mask = residuals.notna() & reported_std.notna()
            residuals = residuals[valid_mask] ** 2  # residual variance
            reported_var = reported_std[valid_mask] ** 2  # reported variance

            if residuals.empty:
                print(f"Warning: No valid data for {method} {coord} in residuals vs reported variance plot")
                continue
            
            ax.scatter(reported_var.values, residuals.values,
                       label=method_label, alpha=0.2, s=8)

            # Binned RMS (variance domain)
            bins = np.linspace(0, 4, 15)
            bin_centres = (bins[:-1] + bins[1:]) / 2
            rms_in_bins = [
                np.sqrt(np.mean(residuals.values[(reported_var.values >= b0) & (reported_var.values < b1)]))
                if np.any((reported_var.values >= b0) & (reported_var.values < b1)) else np.nan
                for b0, b1 in zip(bins[:-1], bins[1:])
            ]
            ax.plot(
                bin_centres, rms_in_bins,
                marker='>', linestyle='-',
                markersize=7,
                markeredgecolor='black', markeredgewidth=1.2,
                label=f'Binned RMS'
            )

            # Pearson correlation coefficient (squared residual vs reported variance)
            # if len(residuals) > 1:
            #     r = np.corrcoef(reported_var, residuals)[0, 1]
            #     ax.text(0.05, 0.95 - 0.07 * i, f"{method.upper()} r = {r:.2f}",
            #             transform=ax.transAxes, ha='left', va='top', fontsize=9)

        ax.plot([0, 3], [0, 3], 'k--', lw=1, label='1:1 Line')
        # ax.set_title(f'Residual Variance vs Reported Variance ({coord.upper()})')
        ax.set_xlabel('Reported Variance [m²]')
        ax.set_ylabel('Squared Residual [m²]')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        figs[coord] = fig
    return figs


def plot_emp_std_vs_std_ratio(results):
    methods = results['methods']
    coords = ['x', 'y', 'z']
    n_methods = len(methods)
    n_coords = len(coords)

    # Prepare data: one bar per method per coordinate
    ratios = np.zeros((n_methods, n_coords))
    for i, coord in enumerate(coords):
        for j, method in enumerate(methods):
            reported_std = results['metrics']['std'][coord][j][0]
            empirical_std = results['metrics']['std'][coord][j][1]
            ratios[j, i] = empirical_std / reported_std if reported_std > 0 else np.nan

    x = np.arange(n_methods)
    bar_width = 0.25
    offset = [-bar_width, 0, bar_width]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, coord in enumerate(coords):
        ax.bar(x + offset[i], ratios[:, i], width=bar_width,
               label=coord.upper(), color=colours[i])

        # Annotate values
        for j in range(n_methods):
            value = ratios[j, i]
            if not np.isnan(value):
                ax.text(x[j] + offset[i], value + 0.02, f"{value:.2f}",
                        ha='center', fontsize=8)

    ax.axhline(1, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_ylabel('Empirical STD / Reported STD Ratio')
    ax.set_title('STD Ratio per Coordinate and Method')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend(title='Coordinate')
    ax.set_ylim(0, max(np.nanmax(ratios), 1.2))
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_rms_vs_reported_variance(results, truth):
    coords = ['x', 'y', 'z']
    methods = results['methods']
    n_methods = len(methods)
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Compute ratio data
    ratios = np.zeros((n_methods, len(coords)))
    for i, method in enumerate(methods):
        combined = results['combined_orbits'][i]
        for j, coord in enumerate(coords):
            residuals = combined[f'{coord}_pos'] - truth[f'{coord}_pos']
            std = combined[f'std_{coord}']
            mask = residuals.notna() & std.notna()
            if mask.sum() == 0:
                ratios[i, j] = np.nan
            else:
                rms_sq = np.mean(residuals[mask] ** 2)
                reported_var = np.mean(std[mask] ** 2)
                ratios[i, j] = rms_sq / reported_var if reported_var > 0 else np.nan

    # Plot grouped bar chart
    x = np.arange(n_methods)
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, coord in enumerate(coords):
        ax.bar(x + offsets[j], ratios[:, j],
               width=bar_width, label=coord.upper(),
               color=colours[j], edgecolor='black')

        for i in range(n_methods):
            val = ratios[i, j]
            if not np.isnan(val):
                ax.text(x[i] + offsets[j], val + 0.02, f"{val:.2f}", ha='center', fontsize=8)

    ax.axhline(1.0, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('RMS² / Reported Variance')
    ax.set_title('RMS² to Reported Variance Ratio per Coordinate')
    ax.legend(title='Coordinate')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig



def plot_input_orbit_rms_to_std(dataframes, truth):
    coords = ['x', 'y', 'z']
    n_inputs = len(dataframes)
    rms_std_ratios = np.zeros((n_inputs, len(coords)))

    for i, df in enumerate(dataframes):
        for j, coord in enumerate(coords):
            residual = df[f'{coord}_pos'] - truth[f'{coord}_pos']
            std = df[f'std_{coord}']
            mask = residual.notna() & std.notna()
            if mask.sum() == 0:
                rms_std_ratios[i, j] = np.nan
            else:
                rms = np.sqrt((residual[mask]**2).mean())
                mean_std = std[mask].mean()
                rms_std_ratios[i, j] = rms / mean_std if mean_std > 0 else np.nan

    # Plot
    x = np.arange(n_inputs)
    bar_width = 0.25
    offset = [-bar_width, 0, bar_width]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(8, 6))
    for j, coord in enumerate(coords):
        ax.bar(x + offset[j], rms_std_ratios[:, j], width=bar_width,
               label=coord.upper(), color=colours[j])
        for i in range(n_inputs):
            val = rms_std_ratios[i, j]
            if not np.isnan(val):
                ax.text(x[i] + offset[j], val + 0.02, f"{val:.2f}", ha='center', fontsize=8)

    ax.axhline(1, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_ylabel('RMS / Reported STD')
    ax.set_title('Synthetic Orbit RMS to STD Ratio per Coordinate')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Orbit {i+1}' for i in range(n_inputs)])
    ax.legend(title='Coordinate')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig


def create_validation_summary_table(results):
    rows = []
    for i, method in enumerate(results['methods']):
        for coord in ['x', 'y', 'z']:
            reported, empirical = results['metrics']['std'][coord][i]
            rows.append({'Method': method.upper(), 'Metric': f'STD {coord.upper()}', 'Reported': f'{reported:.2f}', 'Empirical': f'{empirical:.2f}', 'Δ': f'{abs(reported - empirical):.2f}'})
        for cov_pair in ['xy', 'xz', 'yz']:
            reported, empirical = results['metrics']['cov'][cov_pair][i]
            rows.append({'Method': method.upper(), 'Metric': f'COV {cov_pair.upper()}', 'Reported': f'{reported:.2f}', 'Empirical': f'{empirical:.2f}', 'Δ': f'{abs(reported - empirical):.2f}'})
        rms, avg_std = results['metrics']['rms'][i]
        rows.append({'Method': method.upper(), 'Metric': 'RMS / AVG STD', 'Reported': f'{avg_std:.2f}', 'Empirical': f'{rms:.2f}', 'Δ': f'{abs(rms - avg_std):.2f}'})
        chi2 = results['metrics']['chi2'][i]
        rows.append({'Method': method.upper(), 'Metric': 'Chi-Square', 'Reported': '1.00', 'Empirical': f'{chi2:.2f}', 'Δ': f'{abs(chi2 - 1):.2f}'})
    table = pd.DataFrame(rows)
    return table

def export_pdf_report(figures, summary_table, filename='validation_report_full.pdf'):
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.show()
            # plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        table = ax.table(cellText=summary_table.values, colLabels=summary_table.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        plt.show()
        pdf.savefig(fig)
        # plt.close(fig)
    print(f"Report saved as '{filename}'")


def plot_all_covariance_ellipses(results, plane='xy'):
    """
    Plot 2D residual scatter with 3σ reported and empirical covariance ellipses
    and their semi-major/minor axis lines for all methods in the specified plane.

    Parameters
    ----------
    results : dict
        Dictionary containing keys 'methods', 'residuals', and 'combined_orbits'.
    plane : str
        Coordinate plane to plot ('xy', 'xz', 'yz').
    """
    coord1, coord2 = plane[0], plane[1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    lim = 5.5
    xlim = (-lim, lim)
    ylim = (-lim, lim)

    for i, method in enumerate(results['methods']):
        residuals = results['residuals'][i]
        combined = results['combined_orbits'][i]
        ax = axes[i]

        # Get residuals
        r1 = residuals[f'{coord1}_pos']
        r2 = residuals[f'{coord2}_pos']
        mask = r1.notna() & r2.notna()
        r1 = r1[mask]
        r2 = r2[mask]
        ax.scatter(r1, r2, s=4, alpha=0.5, label='Residuals')

        # ===== 1. Reported covariance matrix =====
        std1 = np.nanmean(combined[f'std_{coord1}'])
        std2 = np.nanmean(combined[f'std_{coord2}'])
        cov_key = f'cov_{coord1}{coord2}' if f'cov_{coord1}{coord2}' in combined else f'cov_{coord2}{coord1}'
        cov12 = np.nanmean(combined[cov_key])
        rep_cov = np.array([[std1**2, cov12], [cov12, std2**2]])

        # ===== 2. Empirical covariance matrix =====
        emp_cov = np.cov(np.vstack([r1, r2]))

        for cov_matrix, color, linestyle, label in zip(
            [rep_cov, emp_cov],
            ['red', 'orange'],
            ['-', '--'],
            ['3σ Reported Covariance', '3σ Empirical Covariance']
        ):
            vals, vecs = np.linalg.eigh(cov_matrix)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            sigma_level = 3
            width, height = 2 * sigma_level * np.sqrt(vals)
            semi_major = sigma_level * np.sqrt(vals[0])
            semi_minor = sigma_level * np.sqrt(vals[1])

            # Ellipse
            ellipse = Ellipse(xy=(0, 0), width=width, height=height,
                              angle=angle, edgecolor=color, linestyle=linestyle,
                              fc='None', lw=2, label=label)
            ax.add_patch(ellipse)

            # Semi-major and semi-minor lines (unit vectors scaled)
            centre = np.array([0, 0])
            major_dir = vecs[:, 0]
            minor_dir = vecs[:, 1]
            plotboth = 0
            if label != '3σ Empirical Covariance' or plotboth:
                # Extend semi-major axis in both directions
                ax.plot(
                    [-semi_major * major_dir[0], semi_major * major_dir[0]],
                    [-semi_major * major_dir[1], semi_major * major_dir[1]],
                    color=color, lw=2, linestyle=linestyle,
                )
                
                # Extend semi-minor axis in both directions
                ax.plot(
                    [-semi_minor * minor_dir[0], semi_minor * minor_dir[0]],
                    [-semi_minor * minor_dir[1], semi_minor * minor_dir[1]],
                    color=color, lw=2, linestyle=linestyle,
                )


        # Formatting
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_title(method.replace('_', ' ').title())
        if i in [2, 3]:
            ax.set_xlabel(f'Residual {coord1.upper()} [m]')
        if i in [0, 2]:
            ax.set_ylabel(f'Residual {coord2.upper()} [m]')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal')

        # Legend only in final subplot
        if i == 3:
            ax.legend(loc='upper right', fontsize=14, draggable = True)
            
    plt.subplots_adjust(top=0.952, bottom=0.078, left=0.02, right=1.0, hspace=0.144, wspace=0.0)

    plt.show()



def extract_covariance_stats(results, plane='xy', sigma_level=3):
    """
    Extract reported and empirical covariance statistics for each method in the specified 2D plane.

    Parameters
    ----------
    results : dict
        Results dictionary containing 'methods', 'residuals', and 'combined_orbits'.
    plane : str
        Coordinate plane to extract ('xy', 'xz', or 'yz').
    sigma_level : int
        Sigma level used for plotting (for reference only).

    Returns
    -------
    pd.DataFrame
        Table containing σ1², σ2², and Cov for both reported and empirical matrices.
    """
    coord1, coord2 = plane[0], plane[1]

    table_data = []

    for method, residuals, combined in zip(results['methods'],
                                           results['residuals'],
                                           results['combined_orbits']):
        # Mask valid residuals
        r1 = residuals[f'{coord1}_pos']
        r2 = residuals[f'{coord2}_pos']
        mask = r1.notna() & r2.notna()
        r1 = r1[mask]
        r2 = r2[mask]

        # Reported values
        std1_r = np.nanmean(combined[f'std_{coord1}'])
        std2_r = np.nanmean(combined[f'std_{coord2}'])
        cov_key = f'cov_{coord1}{coord2}' if f'cov_{coord1}{coord2}' in combined else f'cov_{coord2}{coord1}'
        cov_r = np.nanmean(combined[cov_key])

        # Empirical values
        emp_cov_matrix = np.cov(np.vstack([r1, r2]))
        std1_e = np.sqrt(emp_cov_matrix[0, 0])
        std2_e = np.sqrt(emp_cov_matrix[1, 1])
        cov_e = emp_cov_matrix[0, 1]

        # Add to table
        table_data.append({
            'Method': method,
            'σ1² Reported': f"{std1_r**2:.4f}",
            'σ2² Reported': f"{std2_r**2:.4f}",
            'Cov Reported': f"{cov_r:.4f}",
            'σ1² Empirical': f"{std1_e**2:.4f}",
            'σ2² Empirical': f"{std2_e**2:.4f}",
            'Cov Empirical': f"{cov_e:.4f}"
        })

    return pd.DataFrame(table_data)



# %% Repeated test pipeline

def run_repeated_variance_validation(
    n_runs=20, n_epochs=1440, n_inputs=3,
    missing_frac=0.005,
    std_bounds=(0.5, 2.5), cov_bounds=(0, 0.1),
    methods=('mean', 'inverse_variance', 'vce', 'residual_weighted')
):
    coords = ['x', 'y', 'z']
    method_list = list(methods)

    rms_squared = {method: {coord: [] for coord in coords} for method in method_list}
    reported_var = {method: {coord: [] for coord in coords} for method in method_list}
    chi2_values = {method: [] for method in method_list}
    vce_components = {}
    cov_ratios = {method: {'xy': [], 'xz': [], 'yz': []} for method in methods}
    frobenius = {method: [] for method in methods}

    noise_summary = []

    for run in range(n_runs):
        seed = 1000 + run
        truth = generate_keplerian_orbit(n_epochs=n_epochs)
        dataframes, noise_df = generate_synthetic_data_from_truth(
            truth,
            n_inputs=n_inputs,
            missing_frac=missing_frac,
            std_bounds=std_bounds,
            cov_bounds=cov_bounds,
            random_seed=seed,
            return_noise_info=True
        )
        dataframes, truth_aligned = drop_epochs_with_no_data(dataframes, truth)

        noise_df['Run'] = run + 1
        noise_summary.append(noise_df)

        for method in method_list:
            if method == 'vce':
                combined, var_components, _ = combine_orbits(dataframes, method, 1000, True, True, [1, 20], [truth_aligned])
                vce_components[run] = var_components
            else:
                combined, _, _ = combine_orbits(dataframes, method, 1000, True, True, [1, 20], [truth_aligned])

            combined = combined.reindex(truth_aligned.index)
            residuals = combined[['x_pos', 'y_pos', 'z_pos']] - truth_aligned[['x_pos', 'y_pos', 'z_pos']]
            chi2_components = []

            for coord in coords:
                res = residuals[f'{coord}_pos']
                std = combined[f'std_{coord}']
                mask = res.notna() & std.notna()

                if mask.sum() == 0:
                    continue

                rms_sq = np.mean(res[mask] ** 2)
                mean_var = np.mean(std[mask] ** 2)

                rms_squared[method][coord].append(rms_sq)
                reported_var[method][coord].append(mean_var)

                chi2 = np.mean((res[mask] / std[mask]) ** 2)
                chi2_components.append(chi2)

            # --- Covariance ratios ---
            res_x = residuals['x_pos']
            res_y = residuals['y_pos']
            res_z = residuals['z_pos']

            mask_xy = res_x.notna() & res_y.notna()
            mask_xz = res_x.notna() & res_z.notna()
            mask_yz = res_y.notna() & res_z.notna()

            emp_xy = np.cov(res_x[mask_xy], res_y[mask_xy])[0, 1]
            emp_xz = np.cov(res_x[mask_xz], res_z[mask_xz])[0, 1]
            emp_yz = np.cov(res_y[mask_yz], res_z[mask_yz])[0, 1]

            rep_xy = np.nanmean(combined['cov_xy'])
            rep_xz = np.nanmean(combined['cov_xz'])
            rep_yz = np.nanmean(combined['cov_yz'])

            if rep_xy != 0:
                cov_ratios[method]['xy'].append(emp_xy / rep_xy)
            if rep_xz != 0:
                cov_ratios[method]['xz'].append(emp_xz / rep_xz)
            if rep_yz != 0:
                cov_ratios[method]['yz'].append(emp_yz / rep_yz)

            # --- Frobenius norm error of covariance matrix ---
            residual_matrix = np.vstack([res_x, res_y, res_z]).T
            emp_cov = np.cov(residual_matrix, rowvar=False)

            rep_cov_matrix = np.zeros((3, 3))
            rep_cov_matrix[0, 0] = np.nanmean(combined['std_x'] ** 2)
            rep_cov_matrix[1, 1] = np.nanmean(combined['std_y'] ** 2)
            rep_cov_matrix[2, 2] = np.nanmean(combined['std_z'] ** 2)
            rep_cov_matrix[0, 1] = rep_cov_matrix[1, 0] = rep_xy
            rep_cov_matrix[0, 2] = rep_cov_matrix[2, 0] = rep_xz
            rep_cov_matrix[1, 2] = rep_cov_matrix[2, 1] = rep_yz

            numer = np.linalg.norm(emp_cov - rep_cov_matrix, ord='fro')
            denom = np.linalg.norm(rep_cov_matrix, ord='fro')
            frob_error = numer / denom if denom != 0 else np.nan
            frobenius[method].append(frob_error)

            if chi2_components:
                chi2_mean = np.nanmean(chi2_components)
                chi2_values[method].append(chi2_mean)

    # Combine noise info
    input_noise_df = pd.concat(noise_summary, ignore_index=True)
    input_noise_df = input_noise_df[['Run', 'Orbit', 'STD_X', 'STD_Y', 'STD_Z', 'COV_XY', 'COV_XZ', 'COV_YZ']]

    # Add VCE weights to noise info
    vce_df = []
    for run, components in vce_components.items():
        n_orbits = len(components['x'])
        for i in range(n_orbits):
            vce_df.append({
                'Run': run + 1,
                'Orbit': i + 1,
                'Var_X': components['x'][i],
                'Var_Y': components['y'][i],
                'Var_Z': components['z'][i]
            })
    vce_df = pd.DataFrame(vce_df)
    merged_df = pd.merge(input_noise_df, vce_df, on=['Run', 'Orbit'])

    return rms_squared, reported_var, merged_df, chi2_values, vce_components, cov_ratios, frobenius


def plot_rms_variance_ratio_bar_with_errorbars(rms_squared, reported_var):
    coords = ['x', 'y', 'z']
    methods = list(rms_squared.keys())
    n_methods = len(methods)
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Compute mean and std of RMS² / Reported Variance
    mean_ratios = np.zeros((n_methods, len(coords)))
    std_ratios = np.zeros((n_methods, len(coords)))

    for i, method in enumerate(methods):
        for j, coord in enumerate(coords):
            rms_sq_vals = np.array(rms_squared[method][coord])
            var_vals = np.array(reported_var[method][coord])

            if len(rms_sq_vals) == 0 or len(var_vals) == 0:
                mean_ratios[i, j] = np.nan
                std_ratios[i, j] = np.nan
                continue

            ratios = rms_sq_vals / var_vals
            mean_ratios[i, j] = np.nanmean(ratios)
            std_ratios[i, j] = np.nanstd(ratios)

    # Plot grouped bar chart
    x = np.arange(n_methods)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for j, coord in enumerate(coords):
        bar_pos = x + offsets[j]
        ax.bar(bar_pos, mean_ratios[:, j],
               yerr=std_ratios[:, j],
               width=bar_width, label=coord.upper(),
               color=colours[j], edgecolor='black', capsize=5)

        # for i in range(n_methods):
        #     val = mean_ratios[i, j]
        #     if not np.isnan(val):
        #         ax.text(bar_pos[i], val + 0.03, f"{val:.2f}", ha='center', fontsize=8)
        # Dummy error bar for legend
    ax.errorbar([], [], yerr=[1], fmt=' ', label='±1σ (error bar)', capsize=5, color='black')

    ax.axhline(1.0, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in methods])
    ax.set_ylabel('RMS² / Reported Variance')
    # ax.set_title('RMS² to Reported Variance Ratio with Error Bars')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_std_ratio_bar_with_errorbars(rms_squared, reported_var):
    coords = ['x', 'y', 'z']
    methods = list(rms_squared.keys())
    n_methods = len(methods)
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Compute mean and std of STD ratio (empirical / reported)
    mean_ratios = np.zeros((n_methods, len(coords)))
    std_ratios = np.zeros((n_methods, len(coords)))

    for i, method in enumerate(methods):
        for j, coord in enumerate(coords):
            rms_sq_vals = np.array(rms_squared[method][coord])
            var_vals = np.array(reported_var[method][coord])

            if len(rms_sq_vals) == 0 or len(var_vals) == 0:
                mean_ratios[i, j] = np.nan
                std_ratios[i, j] = np.nan
                continue

            emp_std = np.sqrt(rms_sq_vals)
            rep_std = np.sqrt(var_vals)
            ratios = emp_std / rep_std
            mean_ratios[i, j] = np.nanmean(ratios)
            std_ratios[i, j] = np.nanstd(ratios)

    x = np.arange(n_methods)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for j, coord in enumerate(coords):
        bar_pos = x + offsets[j]
        ax.bar(bar_pos, mean_ratios[:, j],
               yerr=std_ratios[:, j],
               width=bar_width, label=coord.upper(),
               color=colours[j], edgecolor='black', capsize=5)

        # for i in range(n_methods):
        #     val = mean_ratios[i, j]
        #     if not np.isnan(val):
        #         ax.text(bar_pos[i], val + 0.03, f"{val:.2f}", ha='center', fontsize=8)

    ax.errorbar([], [], yerr=[1], fmt=' ', label='±1σ (error bar)', capsize=5, color='black')
    ax.axhline(1.0, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ')for m in methods])
    ax.set_ylabel('Empirical STD / Reported STD')
    # ax.set_title('STD Ratio with Error Bars (20 runs)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_chi2_distribution(chi2_per_run):
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Collect all values to determine dynamic bin range
    all_vals = np.concatenate(list(chi2_per_run.values()))
    min_val = np.floor(all_vals.min() * 10) / 10 - 0.001
    max_val = np.ceil(all_vals.max() * 10) / 10 + 0.001
    bins = np.linspace(min_val, max_val, 50)
    
    for method, chi2_vals in chi2_per_run.items():
        ax.hist(chi2_vals, bins=bins, histtype='step',
                linewidth=2, label=method.replace('_',' '))

    ax.axvline(1.0, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_xlabel('Reduced Chi-Square')
    ax.set_ylabel('Frequency')
    # ax.set_title('Distribution of Reduced Chi-Square (20 runs)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_chi2_bar_with_errorbars(chi2_values):

    methods = list(chi2_values.keys())
    means = [np.mean(chi2_values[m]) for m in methods]
    stds = [np.std(chi2_values[m]) for m in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color='lightsteelblue', edgecolor='black', linewidth=0.8)

    # Add ideal line and labels
    ax.axhline(1.0, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in methods])
    ax.set_ylabel('Reduced Chi-Square')
    # ax.set_title('Mean Reduced Chi-Square per Method (±1σ over 50 runs)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    # Annotate each bar
    # for i, (mean_val, std_val) in enumerate(zip(means, stds)):
    #     ax.text(x[i], mean_val + std_val + 0.02, f"{mean_val:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    return fig



def plot_vce_vs_input_std_per_coord(merged_df):
    coords = ['x', 'y', 'z']
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    for i, coord in enumerate(coords):
        ax = axes[i]
        ax.set_title(f'{coord.upper()}')
        for orbit_id in sorted(merged_df['Orbit'].unique()):
            df_orbit = merged_df[merged_df['Orbit'] == orbit_id]
            ax.scatter(df_orbit[f'STD_{coord.upper()}'], df_orbit[f'Var_{coord.upper()}'],
                       label=f'Orbit {orbit_id}', alpha=0.7)
        ax.set_xlabel(f'Input STD {coord.upper()} [m]')
        ax.set_ylabel(f'{coord.upper()} variance component')
        ax.grid(True)
        ax.legend()
    # plt.suptitle('VCE Variance Components vs Input STD per Coordinate')
    plt.tight_layout()
    return fig

def plot_vce_variance_vs_input_std_combined(merged_df):
    
    coords = ['x', 'y', 'z']
    colours = {'x': 'tab:blue', 'y': 'tab:orange', 'z': 'tab:green'}
    markers = {'x': 'o', 'y': 's', 'z': '^'}

    fig, ax = plt.subplots(figsize=(6.5, 6))

    for coord in coords:
        std_col = f'STD_{coord.upper()}'
        vce_col = f'Var_{coord.upper()}'
        x = merged_df[std_col]
        y = merged_df[vce_col]

        ax.scatter(x, y, alpha=0.7,
                   edgecolor='black', linewidth=0.5,
                   color=colours[coord], marker=markers[coord],
                   label=coord.upper())

        # Optional linear fit
        if len(x.dropna()) > 1:
            from scipy.stats import linregress
            slope, intercept, r, _, _ = linregress(x, y)
            x_sorted = np.array(sorted(x))
            ax.plot(x_sorted, slope * x_sorted + intercept,
                    linestyle='--', color=colours[coord], alpha=0.8,
                    label=f'{coord.upper()} Trend (R={r:.2f})')

    ax.set_xlabel('Input orbit STD [m]')
    ax.set_ylabel('Estimated variance components [m²]')
    # ax.set_title('VCE Variance Components vs Input STD (All Parameters)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


# Covariance plot

def plot_cov_ratio_bar_with_errorbars(cov_ratios):
    components = ['xy', 'xz', 'yz']
    methods = list(cov_ratios.keys())
    n_methods = len(methods)
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Compute mean and std of covariance ratios (empirical / reported)
    mean_ratios = np.zeros((n_methods, len(components)))
    std_ratios = np.zeros((n_methods, len(components)))

    for i, method in enumerate(methods):
        for j, comp in enumerate(components):
            ratio_vals = np.array(cov_ratios[method][comp])
            if len(ratio_vals) == 0:
                mean_ratios[i, j] = np.nan
                std_ratios[i, j] = np.nan
            else:
                mean_ratios[i, j] = np.nanmean(ratio_vals)
                std_ratios[i, j] = np.nanstd(ratio_vals)

    x = np.arange(n_methods)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for j, comp in enumerate(components):
        bar_pos = x + offsets[j]
        ax.bar(bar_pos, mean_ratios[:, j],
               yerr=std_ratios[:, j],
               width=bar_width, label=comp.upper(),
               color=colours[j], edgecolor='black', capsize=5)

    ax.errorbar([], [], yerr=[1], fmt=' ', label='±1σ (error bar)', capsize=5, color='black')
    ax.axhline(1.0, color='black', linestyle='--', lw=1, label='Ideal (1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in methods])
    ax.set_ylabel('Empirical COV / Reported COV')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# Frobenius norm

def plot_frobenius_bar_with_errorbars(frobenius):
    methods = list(frobenius.keys())
    n_methods = len(methods)

    mean_errors = np.zeros(n_methods)
    std_errors = np.zeros(n_methods)

    for i, method in enumerate(methods):
        vals = np.array(frobenius[method])
        mean_errors[i] = np.nanmean(vals)
        std_errors[i] = np.nanstd(vals)

    x = np.arange(n_methods)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    bar_width = 0.6
    ax.bar(x, mean_errors,
           yerr=std_errors,
           width=bar_width, color='#1f77b4',
           edgecolor='black', capsize=5)

    ax.errorbar([], [], yerr=[1], fmt=' ', label='±1σ (error bar)', capsize=5, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in methods])
    ax.set_ylabel('Frobenius Norm Error (Empirical vs Reported Covariance)')
    ax.axhline(0, color='black', linestyle='--', lw=1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig



# %%Generate data
truth = generate_keplerian_orbit(n_epochs=6000)
# dataframes, noise_used = generate_synthetic_data_from_truth(truth, n_inputs=3, missing_frac=0.005, 
#                                                             random_seed = 42, return_noise_info=True,cov_bounds=(0.05,0.3))

dataframes, noise_info = generate_synthetic_data_from_truth(
    truth, n_inputs=3, missing_frac=0.005, 
    # std_bounds= {'x': (1, 2.5), 'y': (1, 2.5), 'z': (1, 2.5)}, #(0.5, 2.5), #
    std_bounds= {'x': (1, 1), 'y': (1, 1), 'z': (1, 1)}, #(0.5, 2.5), #
    cov_bounds= (0.1, 0.1),# {'xy': (-0.1, 0.1), 'xz': (-0.1, 0.1), 'yz': (-0.1, 0.1)}, #(0.0, 0.1)
    # cov_bounds= {'xy': (0.5, 0.5), 'xz': (0, 0), 'yz': (0, 0)}, #(0.0, 0.1)
    # fixed_cov={'xy': 0.5, 'xz': 0.0, 'yz': 0.0},
    random_seed=42,
    return_noise_info=True
)

# dataframes, noise_df, truth = generate_mixed_anisotropy_test(
#     truth=generate_keplerian_orbit(n_epochs=6000),
#     fixed_cov={'xy': 0.0, 'xz': 0.0, 'yz': 0.0},
#     missing_frac=0.0
# )

dataframes, truth = drop_epochs_with_no_data(dataframes, truth)

# plot_3d_input_trajectories(dataframes)
results = {'methods': [], 'combined_orbits': [], 'metrics': {'std': {'x': [], 'y': [], 'z': []},
          'cov': {'xy': [], 'xz': [], 'yz': []}, 'rms': [], 'chi2': []}, 'residuals': [], 'reported_std': []}
methods = ['mean', 'inverse_variance', 'vce', 'residual_weighted']

for method in methods:
    if method == 'vce':
        combined, var_components, _ = combine_orbits(dataframes, method, 1000, True, True, [1, 20], [truth])
    else:
        combined, _, _ = combine_orbits(dataframes, method, 1000, True, True, [1, 20], [truth])
    
    combined = combined.reindex(truth.index)
    method_metrics, residuals = compute_validation_metrics(combined, truth)
    results['combined_orbits'].append(combined)
    results['residuals'].append(residuals)
    results['reported_std'].append(combined[['std_x', 'std_y', 'std_z']])
    results['methods'].append(method)
    for coord in ['x', 'y', 'z']:
        results['metrics']['std'][coord].append(method_metrics['std'][coord])
    for cov_pair in ['xy', 'xz', 'yz']:
        results['metrics']['cov'][cov_pair].append(method_metrics['cov'][cov_pair])
    results['metrics']['rms'].append(method_metrics['rms'])
    results['metrics']['chi2'].append(method_metrics['chi2'])

residuals = combined[['x_pos', 'y_pos', 'z_pos']] - truth[['x_pos', 'y_pos', 'z_pos']]

# %%
plot_all_covariance_ellipses(results, plane='xy')
plot_all_covariance_ellipses(results, plane='xz')
plot_all_covariance_ellipses(results, plane='yz')

xy = extract_covariance_stats(results, plane='xy')
xz = extract_covariance_stats(results, plane='xz')
yz = extract_covariance_stats(results, plane='yz')

# %%
rms_squared, reported_var, input_noise_df, chi2_values, variance_components, cov_ratios, frobenius = run_repeated_variance_validation(n_runs=50, n_epochs=6000, std_bounds = (1,2.5))
# %%
_ = plot_rms_variance_ratio_bar_with_errorbars(rms_squared, reported_var)
_ = plot_std_ratio_bar_with_errorbars(rms_squared, reported_var)
# %%
_ = plot_chi2_distribution(chi2_values)
_ = plot_chi2_bar_with_errorbars(chi2_values)
# _ = plot_vce_vs_input_std_per_coord(input_noise_df)
_ = plot_vce_variance_vs_input_std_combined(input_noise_df)

# %%
_ = plot_cov_ratio_bar_with_errorbars(cov_ratios)
#%%
_ = plot_frobenius_bar_with_errorbars(frobenius)