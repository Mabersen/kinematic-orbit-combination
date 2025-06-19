# -*- coding: utf-8 -*-
"""
verification_utilities_styled.py
 
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

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gravtools.kinematic_orbits.plotting_functions import plot_rmse_vs_combined_orbit

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
    'figure.titlesize': 12
})

# General utilities
def compute_rmse(dataframes, reference_data):
    """
    Compute the root mean square error (RMSE) between each orbit and the reference orbit.

    Parameters:
        dataframes (list of pd.DataFrame): List of individual orbit trajectories.
        reference_data (pd.DataFrame): Reference orbit trajectory.

    Returns:
        list: RMSE values for each orbit.
    """
    rmse_values = []
    for df in dataframes:
        overall_rmse = np.sqrt(
            np.mean(
                np.sum(
                    np.square(
                        (reference_data[['x_pos', 'y_pos', 'z_pos']] - df[['x_pos', 'y_pos', 'z_pos']])
                        .dropna()
                    ), axis=1
                )
            )
        )
        rmse_values.append(overall_rmse)

    return rmse_values

# Plotting utilities

# Global styling utility
def get_plot_style(method, i, linestyle=False):
    markers = ['o', 's', '^', 'D', 'v', 'X', '*', 'P', '>', '<']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    colours = plt.cm.tab20.colors  # more colours than tab10

    return {
        'marker': markers[i % len(markers)],
        'linestyle': 'solid' if not linestyle else linestyles[i % len(linestyles)],
        'color': colours[i % len(colours)],
    }

def plot_test1_rmse(results_dict_list, reference_orbit, input_sizes, methods):
    """
    Plot side-by-side bar plots for Test 2, comparing RMSE across different input sizes for each method.

    Parameters:
    - results_dict_list: List of dictionaries, each containing results for a specific input size (n, 2n, 4n).
    - input_sizes: List of input sizes corresponding to the results_dict_list (e.g., ['n', '2n', '4n']).
    - methods: List of combination methods (e.g., ['mean', 'ivw', 'vce']).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    input_sizes = [2*i for i in input_sizes]
    test_labels = [f'N = {i}' for _, i in enumerate(input_sizes)]
    # Prepare data for plotting
    input_plotted = 0
    for method in methods:
        method_rmse = []
        input_rmse = []
        for results_dict in results_dict_list:

            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            # print(rmses)
            mean_rmse = np.mean(rmses) if rmses else None
            method_rmse.append(mean_rmse)
            corrupt_rmse = compute_rmse(
                results_dict_list[0]['corrupted_arcs']['orbits'][:], reference_orbit)
            # print(corrupt_rmse)
            input_rmse.append(np.mean(corrupt_rmse))

        if not input_plotted:
            # print(input_rmse)
            ax.plot(test_labels, input_rmse, label='corrupted arcs average', color='grey', linestyle='dashed')
            input_plotted = 1
        style = get_plot_style(method, methods.index(method))
        ax.plot(test_labels, method_rmse, label=method.replace('_', ' '), marker=style['marker'], linestyle=style['linestyle'], color=style['color'])

    ax.set_ylabel("RMSE (m)")
    ax.set_xlabel("Number of orbits")
    ax.set_xticklabels(test_labels, rotation=45, ha='right')
    # fig.suptitle("Test 1: change in RMSE as number of input mirrored-noise pairs increase")
    ax.set_yscale('log')
    ax.grid(linestyle="--", alpha=0.7)

    ax.legend()
    # ax.grid(axis="y")
    plt.tight_layout()
    
# % Plot verification test 2
def plot_test2_rmse(results_dict_list, reference_orbit, input_sizes, test_description, methods):
    """
    Plot side-by-side line plots for Test 2, comparing RMSE across different input sizes for each method.
    Includes shrinking marker sizes for visibility.
    """
    test_labels = input_sizes
    test_seeds = list(test_description.values())

    for ind, test in enumerate(results_dict_list):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
        input_plotted = 0
        test_seed = test_seeds[ind]

        for method in methods:
            method_rmse = []
            input_rmse = []

            for results_dict in test:
                rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                         for keys, orbit_data in results_dict.items() if method in keys]
                mean_rmse = np.mean(rmses) if rmses else None
                method_rmse.append(mean_rmse)
                input_rmse.append(np.mean(compute_rmse(
                    test[0]['corrupted_arcs']['orbits'][:], reference_orbit)))

            if not input_plotted:
                x = np.linspace(2, max(test_labels), 100)
                y = max(method_rmse) / np.sqrt(0.75 * x)
                ax.plot(test_labels, input_rmse,
                        label='corrupted arcs average',
                        color='grey',
                        linestyle='dashed')
                ax.plot(x, y,
                        label=r'$\frac{a}{\sqrt{x}}$',
                        color='red',
                        linestyle='dotted')
                input_plotted = 1

            style = get_plot_style(method, methods.index(method), linestyle=True)
            markersize = max(3, 10 - 1.2 * methods.index(method))
            ax.plot(test_labels, method_rmse,
                    label=method.replace('_', ' '),
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    color=style['color'],
                    markersize=markersize)

        ax.set_ylabel("RMSE (m)")
        ax.set_xlabel("Number of orbits")
        ax.grid(linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()

        
# % Plot verification test 3
def plot_test3_rmse(results_dict_list, reference_orbit, seed_list, methods):
    """
    Plot line plots for Test 3, showing how RMSE changes with random seeds for each method.

    Parameters:
    - results_dict_list: List of dictionaries, each containing results for a specific random seed.
    - methods: List of combination methods (e.g., ['mean', 'ivw', 'vce']).
    """

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    test_labels = [f'{seed_list[i]}' for i, _ in enumerate(results_dict_list)]
    input_plotted = 0
    all_input_rmse = []
    for method in methods:
        method_rmse = []
        
        for results_dict in results_dict_list:

            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            # print(rmses)
            mean_rmse = np.mean(rmses) if rmses else None
            method_rmse.append(mean_rmse)
            if method == methods[0]:
                input_rmse = np.mean(compute_rmse(
                    results_dict['corrupted_arcs']['orbits'][:], reference_orbit))
                all_input_rmse.append(input_rmse)
                
        # if not input_plotted:
        #     ax.plot(test_labels, all_input_rmse, label='corrupted arcs average', color='grey')
        #     input_plotted=1
        style = get_plot_style(method, methods.index(method))
        order = methods.index(method)
        markersize = max(7, 10 - 1.5 * order)  # start at 8, shrink with order
        ax.plot(test_labels, method_rmse, label=method.replace('_', ' '), marker=style['marker'], 
                linestyle=style['linestyle'], color=style['color'], markersize = markersize, alpha=0.8)
        
    # Annotate corrupted RMSEs
    corrupted_text = f"{np.mean(all_input_rmse):.3f}m"
    ax.text(0.05, 0.98, f"Average corrupted Orbit RMSE:\n{corrupted_text}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='lightgrey', alpha=0.8, edgecolor='black'))
    
    ax.set_ylabel("RMSE (m)")
    ax.set_xlabel("Random seed")
    ax.set_xticklabels(test_labels, rotation=45, ha='right')
    # fig.suptitle("Test 3: RMSE with seed changes")
    # ax.set_yscale('log')
    ax.grid(linestyle="--", alpha=0.7)

    ax.legend()
    # ax.grid(axis="y")
    plt.tight_layout()
    
# % Plot verification test 4
def plot_test4_rmse(results_dict_list, reference_orbit, scales, methods, bias_lengths):
    """
    Plot RMSE vs bias scale using subplots to show the effect of increasing input orbits (bias length).
    Includes annotations of the actual scaled bias vectors used.
    Shared, flat legend above the subplots.
    """
    # Set general font size
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 17,
        'figure.titlesize': 18
    })
    num_tests = len(results_dict_list)
    fig, axes = plt.subplots(1, num_tests, figsize=(7 * num_tests, 9), sharey=True)

    if num_tests == 1:
        axes = [axes]

    for ind, (test, ax) in enumerate(zip(results_dict_list, axes)):
        np.random.seed(42)
        input_bias = (np.random.random(bias_lengths[ind]) - 0.5)

        input_plotted = 0
        for method in methods:
            method_rmse = []
            input_rmse = []

            for i, results_dict in enumerate(test):
                rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                         for keys, orbit_data in results_dict.items() if method in keys]
                mean_rmse = np.mean(rmses) if rmses else None
                method_rmse.append(mean_rmse)
                input_rmse.append(np.mean(compute_rmse(
                    test[0]['corrupted_arcs']['orbits'][:], reference_orbit)))

            if not input_plotted:
                ax.plot(scales, input_rmse,
                        label='corrupted arcs average',
                        color='grey',
                        linestyle='dotted')
                input_plotted = 1

            style = get_plot_style(method, methods.index(method))
            markersize = max(3, 10 - 0.8 * methods.index(method))
            ax.plot(scales, method_rmse,
                    label=method.replace('_', ' '),
                    marker=style['marker'],
                    linestyle='solid',
                    color=style['color'],
                    markersize=markersize)

        # Annotate scaled bias vectors
        all_scaled = np.round(input_bias, 3)
        if len(all_scaled)<5:
            bias_text = f"{all_scaled}"
        else:
            bias_text = f"{all_scaled[0:3]}\n{all_scaled[3:]}"
        
        ax.text(0.02, 0.98, f"Original orbit bias values:\n{bias_text}",
                transform=ax.transAxes,
                fontsize=15,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black'))

        ax.set_title(f'{bias_lengths[ind]} input orbits', fontsize = 16)
        ax.set_xlabel("Bias scale")
        ax.set_yscale('log')
        ax.grid(linestyle="--", alpha=0.7)

    axes[0].set_ylabel("RMSE (m)")


    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=len(by_label), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)


plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
    'figure.titlesize': 12
})
def plot_test41_2_rmse(results_dict, reference_orbit, bias_lengths, methods, bias_scale=1.0):
    """
    Plot RMSE vs number of biased input orbits (bias_lengths) for a fixed bias scale.
    Includes annotation of each scaled bias vector and corrupted RMSE.

    Parameters:
    - results_dict: A single test case with multiple bias lengths.
    - reference_orbit: The reference orbit to compare RMSE against.
    - bias_lengths: List of bias vector lengths used (x-axis).
    - methods: List of orbit combination methods.
    - bias_scale: Constant scaling factor applied to each bias vector.
    """
    fig, ax = plt.subplots(figsize=(7, 6))  # LaTeX-friendly size
    results_dict = results_dict[0]
    all_corrupted_rmse = []
    for method in methods:
        method_rmses = []
        for i, bias_len in enumerate(bias_lengths):
            np.random.seed(42)  # same bias vector for fairness
            input_bias = (np.random.random(bias_len) - 0.5)
            scaled_bias = np.round(input_bias * bias_scale, 3)
            # print(scaled_bias)

            test_case = results_dict[i]  # each result per bias length
            rmses = [list(test_case[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in test_case.items() if method in keys]
            method_rmses.append(np.mean(rmses))

            if method == methods[0]:  # only once per orbit count
                rmse = np.mean(compute_rmse(
                    test_case['corrupted_arcs']['orbits'][:], reference_orbit))
                all_corrupted_rmse.append(rmse)

        style = get_plot_style(method, methods.index(method))
        markersize = max(3, 10 - 1.2 * methods.index(method))
        ax.plot(bias_lengths, method_rmses,
                label=method.replace('_', ' '),
                marker=style['marker'],
                linestyle='solid',
                color=style['color'],
                markersize=markersize)

    # Annotate corrupted RMSEs
    # corrupted_text = "\n".join([f"{bias_lengths[i]} orbits: {all_corrupted_rmse[i]:.3f}m" for i in range(len(bias_lengths))])
    # ax.text(0.5, 0.98, f"Corrupted RMSEs:\n{corrupted_text}",
    #         transform=ax.transAxes,
    #         fontsize=8,
    #         verticalalignment='top',
    #         bbox=dict(facecolor='lightgrey', alpha=0.8, edgecolor='black'))

    ax.set_xlabel("Number of biased input orbits")
    ax.set_ylabel("RMSE (m)")
    # ax.set_title(f"Effect of Input Orbit Count (scale={bias_scale})")
    ax.set_yscale('log')
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend(frameon=True, ncols = 2)

    plt.tight_layout()
    plt.subplots_adjust(right=0.78)


# % Plot verification test 5
def plot_test5_rmse(results_dict_list, reference_orbit, n_generations, methods):
    """
    Plot RMSE vs number of generations (Test 5), with corrupted arcs average RMSEs shown in a text box.

    Parameters:
    - results_dict_list: List of result dictionaries for each generation test.
    - reference_orbit: The reference orbit for RMSE computation.
    - n_generations: List of generation labels (x-axis).
    - methods: List of combination methods (e.g., ['mean', 'ivw', 'vce']).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    input_plotted = 0
    all_input_rmse = []

    for method in methods:
        method_rmse = []

        for results_dict in results_dict_list:
            # Compute RMSE for method
            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            mean_rmse = np.mean(rmses) if rmses else None
            method_rmse.append(mean_rmse)

            if method == methods[0]:  # Only collect once
                input_rmse = np.mean(compute_rmse(
                    results_dict['corrupted_arcs']['orbits'][:], reference_orbit))
                all_input_rmse.append(input_rmse)

        style = get_plot_style(method, methods.index(method))
        markersize = max(3, 10 - 2 * methods.index(method))
        ax.plot(n_generations, method_rmse,
                label=method.replace('_', ' '),
                marker=style['marker'],
                linestyle='solid',
                color=style['color'],
                markersize=markersize)

    # Annotate corrupted RMSEs
    corrupted_text = f"{np.mean(all_input_rmse):.3f}m"
    ax.text(0.2, 0.98, f"Average corrupted Orbit RMSE:\n{corrupted_text}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='lightgrey', alpha=0.8, edgecolor='black'))

    # Axis and title formatting
    ax.set_xlabel("N generations")
    ax.set_ylabel("RMSE (m)")
    # fig.suptitle("Test 5: RMSE with increasing generations")
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()

def plot_test6_rmse(results_dict_list, reference_orbit, n_population, methods):
    """
    Plot RMSE vs population size (Test 6), with corrupted arcs average RMSEs shown in a text box.

    Parameters:
    - results_dict_list: List of result dictionaries for each population size test.
    - reference_orbit: The reference orbit for RMSE computation.
    - n_population: List of population sizes (x-axis).
    - methods: List of combination methods (e.g., ['mean', 'ivw', 'vce']).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5,6))
    all_input_rmse = []

    for method in methods:
        method_rmse = []

        for results_dict in results_dict_list:
            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            mean_rmse = np.mean(rmses) if rmses else None
            method_rmse.append(mean_rmse)

            if method == methods[0]:  # Only once
                input_rmse = np.mean(compute_rmse(
                    results_dict['corrupted_arcs']['orbits'][:], reference_orbit))
                all_input_rmse.append(input_rmse)

        style = get_plot_style(method, methods.index(method))
        markersize = max(3, 10 - 1.2 * methods.index(method))
        ax.plot(n_population, method_rmse,
                label=method.replace('_', ' '),
                marker=style['marker'],
                linestyle='solid',
                color=style['color'],
                markersize=markersize)

    # Annotate corrupted RMSEs
    corrupted_text = f"{np.mean(all_input_rmse):.3f}m"
    ax.text(0.2, 0.98, f"Average corrupted Orbit RMSE:\n{corrupted_text}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='lightgrey', alpha=0.8, edgecolor='black'))

    # Plot formatting
    ax.set_xlabel("Population size")
    ax.set_ylabel("RMSE (m)")
    # fig.suptitle("Test 6: RMSE with increasing population size")
    # ax.set_yscale('log')
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()

    
def plot_test7_rmse(results_dict_list, reference_orbit, arclengths, methods):
    def clean_arclength_label(arclength):
        if pd.isna(arclength):
            return 'None'
        elif isinstance(arclength, pd.Timedelta):
            total_minutes = int(arclength.total_seconds() / 60)
            if total_minutes < 1:
                return f"{int(arclength.total_seconds())} sec"
            else:
                return f"{total_minutes} min"
        else:
            return str(arclength)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    test_labels = [clean_arclength_label(i) for i in arclengths]

    all_input_rmse = []

    for method in methods:
        method_rmse = []
        for results_dict in results_dict_list:
            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            mean_rmse = np.mean(rmses) if rmses else None
            method_rmse.append(mean_rmse)

            if method == methods[0]:
                input_rmse = np.mean(compute_rmse(
                    results_dict['corrupted_arcs']['orbits'][:], reference_orbit))
                all_input_rmse.append(input_rmse)

        style = get_plot_style(method, methods.index(method))
        markersize = max(3, 10 - 1.2 * methods.index(method))
        ax.plot(test_labels, method_rmse,
                label=method.replace('_', ' '),
                marker=style['marker'],
                linestyle='solid',
                color=style['color'],
                markersize=markersize)

    ax.plot(test_labels, all_input_rmse,
            label='corrupted arcs average',
            color='grey',
            linestyle='dashed')

    ax.set_xlabel("Arclength")
    ax.set_ylabel("RMSE (m)")
    # fig.suptitle("Test 7: RMSE across different arclengths")
    ax.set_yscale('log')
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend(draggable=True)
    plt.tight_layout()



def plot_test8_rmse(results_dict_list, reference_orbit, n_orbits8, methods):
    """
    Plot bar plots showing RMSE for each method in Test 4 with biases.

    Parameters:
    - results_dict_list: List of result dictionaries for each bias test.
    - methods: List of combination methods (e.g., ['mean', 'ivw', 'vce']).
    """
    # uncorrupted_orbit = reference_orbit

    # Prepare data for plotting

    # Plot bar plot
    # bar_width = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    # Plot input biased RMSE as the first bar

    test_labels = n_orbits8  # [f'Test {i+1}' for i, _ in enumerate(results_dict_list)]
    input_plotted = 0
    for method in methods:
        method_rmse = []
        input_rmse = []
        for results_dict in results_dict_list:

            # input_rmse = np.mean(compute_rmse(results_dict_list[0]['input_arcs']['orbits'][0:], uncorrupted_orbit))
            # ax.scatter(f'Case {count+1}', input_rmse, label='input arcs average', color='grey')

            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            # print(rmses)
            mean_rmse = np.mean(rmses) if rmses else None
            method_rmse.append(mean_rmse)
            input_rmse.append(np.mean(compute_rmse(
                results_dict_list[0]['corrupted_arcs']['orbits'][:], reference_orbit)))
        if not input_plotted:
            ax.plot(test_labels, input_rmse, label='corrupted arcs average', color='grey', linestyle='dashed')
            input_plotted = 1
        style = get_plot_style(method, methods.index(method))
        ax.plot(test_labels, method_rmse, label=method.replace('_', ' '), marker=style['marker'], linestyle=style['linestyle'], color=style['color'])
        # ax.plot(test_labels, method_rmse, label=method)

    # Customise plot
    # ax.set_xticks(x + bar_width * (len(methods) + 1) / 2)
    # ax.set_xlabel("Bias Test")
    # ax.set_xticklabels(['Input arc average'] + methods, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel("RMSE (m)")
    # fig.suptitle("Test 4: RMSE Across Bias Tests")
    # ax.set_yscale('log')
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend()
    # ax.grid(axis="y")
    plt.tight_layout()
    # plt.show()
    
def plot_test9_rmse(results_dict_list, reference_orbit, n_population, methods):
    """
    Plot RMSE vs population size (Test 9), with a single corrupted arcs average RMSE shown in a text box.

    Parameters:
    - results_dict_list: List of result dictionaries for each population size.
    - reference_orbit: The reference orbit.
    - n_population: List of population sizes (x-axis).
    - methods: List of combination methods.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5,6))

    # Compute the corrupted arcs average RMSE (same for all cases)
    corrupted_rmse = np.mean(compute_rmse(
        results_dict_list[0]['corrupted_arcs']['orbits'][:], reference_orbit))

    for method in methods:
        method_rmse = []
        for results_dict in results_dict_list:
            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            method_rmse.append(np.mean(rmses) if rmses else None)

        style = get_plot_style(method, methods.index(method))
        markersize = max(3, 10 - 1.2 * methods.index(method))
        ax.plot(n_population, method_rmse,
                label=method.replace('_', ' '),
                marker=style['marker'],
                linestyle='solid',
                color=style['color'],
                markersize=markersize)

    # Annotate single corrupted RMSE
    ax.text(0.2, 0.98, f"Average corrupted Orbit RMSE:\n{corrupted_rmse:.3f}m",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='lightgrey', alpha=0.8, edgecolor='black'))

    ax.set_xlabel("Population size")
    ax.set_ylabel("RMSE (m)")
    # fig.suptitle("Test 9: RMSE with increasing population size")
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()


def plot_test10_rmse(results_dict_list, reference_orbit, n_generations, methods):
    """
    Plot RMSE vs number of generations (Test 10), with a single corrupted arcs average RMSE shown in a text box.

    Parameters:
    - results_dict_list: List of result dictionaries for each generation.
    - reference_orbit: The reference orbit.
    - n_generations: List of generations (x-axis).
    - methods: List of combination methods.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5,6))

    # Compute the corrupted arcs average RMSE (same for all cases)
    corrupted_rmse = np.mean(compute_rmse(
        results_dict_list[0]['corrupted_arcs']['orbits'][:], reference_orbit))

    for method in methods:
        method_rmse = []
        for results_dict in results_dict_list:
            rmses = [list(results_dict[method]['metrics'].values())[0]['RMS']
                     for keys, orbit_data in results_dict.items() if method in keys]
            method_rmse.append(np.mean(rmses) if rmses else None)

        style = get_plot_style(method, methods.index(method))
        markersize = max(3, 10 - 1.2 * methods.index(method))
        ax.plot(n_generations, method_rmse,
                label=method.replace('_', ' '),
                marker=style['marker'],
                linestyle='solid',
                color=style['color'],
                markersize=markersize)

    # Annotate single corrupted RMSE
    ax.text(0.2, 0.98, f"Average corrupted Orbit RMSE:\n{corrupted_rmse:.3f}m",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='lightgrey', alpha=0.8, edgecolor='black'))

    ax.set_xlabel("N generations")
    ax.set_ylabel("RMSE (m)")
    # fig.suptitle("Test 10: RMSE with increasing generations")
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
