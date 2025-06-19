# -*- coding: utf-8 -*-
"""
run_verification_tests.py
 
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

from gravtools.kinematic_orbits.combine_orbits import combine_arcs
from gravtools.kinematic_orbits.classes import Arc, AccessRequest
from gravtools.kinematic_orbits.retrieve_arcs import retrieve_arcs
from gravtools.kinematic_orbits.frame_utilities import itrs_to_gcrs
from gravtools.verification.verification_utilities import generate_pink_noise_with_std

# This file uses Gaussian white noise
plt.rcParams.update({'font.size': 14})
np.random.seed(42)  # fixing seed
# %%

def validate_combination_methods_general(
        input_arcs, methods, validation_arcs=None, bias_list=[0], noise_std=0.005,
        arclength=None, scaling_factors=None, mirrored=True, pygmo_variables=[1, 20], adjust_var=False,
        noise_type='Gaussian'):
    """
    Validate combination methods using synthetic corrupted data or real validation data.

    Parameters:
        input_arcs (list of Arc): List of input arcs to combine.
        methods (list): List of combination methods to test.
        validation_arcs (Arc, optional): Reference orbit for validation (real or synthetic).
        bias_list (float): Fixed bias to add to the corrupted orbit (meters).
        noise_std (float): Standard deviation of the Gaussian noise to add.
        arclength (pd.Timedelta, optional): Arc length to split the data into smaller chunks.
        scaling_factors (list, optional): List of scaling factors for bias and noise_std.

    Returns:
        dict: Results including errors, combined orbits, and metrics for each method.
    """
    results = {}
    results['input_arcs'] = {'orbits': [i.trajectory for i in input_arcs],
                             'names': [i.analysis_centre for i in input_arcs],
                             'metrics': []}

    if validation_arcs is None:  # Synthetic verification
        print('Verification using synthetic data')
        scaling_factors = scaling_factors or [1.0]
        results['corrupted_arcs'] = {'orbits': []}
        # Generate corrupted data
        corrupted_data_sets = []
        for arc in input_arcs:
            corrupted_data_sets.extend(generate_corrupted_data(
                validation_arc=arc,
                bias_list=bias_list,
                noise_std=noise_std,
                scaling_factors=scaling_factors,
                mirrored=mirrored,
                adjust_var=adjust_var,
                noise_type=noise_type
            ))

        reference_data = input_arcs
        corrupted_orbits = [i.trajectory for i in corrupted_data_sets]
        results['corrupted_arcs']['orbits'].extend(corrupted_orbits)

    else:  # Real validation
        print('Validation of real combination')
        corrupted_data_sets = input_arcs
        reference_data = validation_arcs

        for i in corrupted_data_sets:

            _, metrical = evaluate_combined_orbit(
                combined_orbit=i.trajectory,
                validation_orbits=reference_data.trajectory
            )

            results['input_arcs']['metrics'].append(metrical)

    methods = methods
    # Test each combination method
    for method in methods:
        print(f"Testing method: {method}")
        method_results = {
            "orbits": [],
            "errors": [],
            "metrics": {},
        }

        # Combine corrupted orbits
        combined_orbit, weights, evals = combine_arcs(
            arcs=corrupted_data_sets,
            method=method,
            arclength=arclength,
            reference_data=reference_data,
            pygmo_variables=pygmo_variables
        )

        # Compare to validation arc
        combined_trajectory = combined_orbit[0]  # Assuming single orbit
        validation_orbits = validation_arcs if validation_arcs else reference_data

        validation_orbits = [i.trajectory for i in validation_orbits]

        errors, metrics = evaluate_combined_orbit(
            combined_orbit=combined_trajectory,
            validation_orbits=validation_orbits
        )

        # Store results
        method_results["orbits"].append(combined_trajectory)
        method_results["errors"].append(errors)
        method_results["metrics"] = metrics
        method_results['extra_variables'] = evals[0]
        results[method] = method_results

    return results


def generate_corrupted_data(validation_arc, bias_list, noise_std, scaling_factors, mirrored, adjust_var, noise_type):
    """
    Generate multiple sets of synthetic corrupted data.

    Parameters:
        validation_arcs (Arc): The input orbit to corrupt.
        bias_list (float): Fixed bias to add (meters).
        noise_std (float): Standard deviation of the Gaussian noise to add.
        scaling_factors (list): Scaling factors for bias and noise_std.

    Returns:
        list: List of corrupted Arc objects.
    """
    input_df = validation_arc.trajectory
    corrupted_arcs = []

    if noise_type == 'Gaussian':

        for factor in scaling_factors:

            for bias in bias_list:
                scaled_bias = bias
                scaled_noise_std = noise_std * abs(factor)

                corrupted_plus = input_df.copy()
                corrupted_minus = input_df.copy()

                if scaled_noise_std == 0:
                    for col in ['x_pos', 'y_pos', 'z_pos']:
                        corrupted_plus[col] = corrupted_plus[col] + (scaled_bias)
                        corrupted_minus[col] = corrupted_minus[col] - (scaled_bias)

                else:
                    # Should implement method to fix seed to ensure corrupted datasets are equal between tests...

                    for col in ['x_pos', 'y_pos', 'z_pos']:
                        noise = np.random.normal(0, scaled_noise_std, size=len(corrupted_plus))

                        corrupted_plus[col] = corrupted_plus[col] + (scaled_bias + noise)
                        corrupted_minus[col] = corrupted_minus[col] - (scaled_bias + noise)

                if adjust_var:  # attempted fix for problem
                    corrupted_plus[['std_x', 'std_y', 'std_z']] = scaled_noise_std + scaled_bias
                    corrupted_minus[['std_x', 'std_y', 'std_z']] = - scaled_noise_std - scaled_bias

                else:
                    # Set to noise std without combining with original (assumes original std is zero)
                    corrupted_plus[['std_x', 'std_y', 'std_z']] = scaled_noise_std
                    corrupted_minus[['std_x', 'std_y', 'std_z']] = scaled_noise_std  # Not negative

                corrupted_arcs.append(Arc(corrupted_plus, validation_arc.analysis_centre,
                                          validation_arc.satellite_id, validation_arc.data_type))
                if mirrored:
                    # print(mirrored)
                    corrupted_arcs.append(Arc(corrupted_minus, validation_arc.analysis_centre,
                                              validation_arc.satellite_id, validation_arc.data_type))
    if noise_type == 'Pink':

        for factor in scaling_factors:

            for bias in bias_list:
                scaled_bias = bias
                scaled_noise_std = noise_std * abs(factor)

                corrupted_plus = input_df.copy()
                corrupted_minus = input_df.copy()

                if scaled_noise_std == 0:
                    for col in ['x_pos', 'y_pos', 'z_pos']:
                        corrupted_plus[col] = corrupted_plus[col] + (scaled_bias)
                        corrupted_minus[col] = corrupted_minus[col] - (scaled_bias)

                else:
                    # Should implement method to fix seed to ensure corrupted datasets are equal between tests...
                    length = len(input_df)

                    for col in ['x_pos', 'y_pos', 'z_pos']:
                        noise = generate_pink_noise_with_std(length=length, beta=1, target_std=scaled_noise_std)
                        corrupted_plus[col] = corrupted_plus[col] + (scaled_bias + noise)
                        corrupted_minus[col] = corrupted_minus[col] - (scaled_bias + noise)

                if adjust_var:  # attempted fix for problem
                    corrupted_plus[['std_x', 'std_y', 'std_z']] = scaled_noise_std + scaled_bias
                    corrupted_minus[['std_x', 'std_y', 'std_z']] = - scaled_noise_std - scaled_bias
                else:
                    corrupted_plus[['std_x', 'std_y', 'std_z']] = scaled_noise_std
                    corrupted_minus[['std_x', 'std_y', 'std_z']] = - scaled_noise_std

                corrupted_arcs.append(Arc(corrupted_plus, validation_arc.analysis_centre,
                                          validation_arc.satellite_id, validation_arc.data_type))
                if mirrored:
                    print(mirrored)
                    corrupted_arcs.append(Arc(corrupted_minus, validation_arc.analysis_centre,
                                              validation_arc.satellite_id, validation_arc.data_type))

    return corrupted_arcs


def evaluate_combined_orbit(combined_orbit, validation_orbits):
    """
    Evaluate the combined orbit against the validation orbit.

    Parameters:
        combined_orbit (pd.DataFrame): The combined orbit trajectory.
        validation_orbit (pd.DataFrame): The validation orbit trajectory.

    Returns:
        dict: Errors at each epoch.
        dict: Overall metrics, including RMS error and biases.
    """
    # global metrics
    metrics = {}
    # global val
    for idx, val in enumerate(validation_orbits):
        diff = combined_orbit - val
        diff['magnitude'] = np.sqrt(diff['x_pos']**2 + diff['y_pos']**2 + diff['z_pos']**2)

        # Metrics
        metrics[f'validation set {idx}'] = {
            "RMS": np.sqrt(np.mean(diff['magnitude']**2)),
            "Mean Bias X": np.mean(diff['x_pos']),
            "Mean Bias Y": np.mean(diff['y_pos']),
            "Mean Bias Z": np.mean(diff['z_pos']),
            "Mean Magnitude Error": np.mean(diff['magnitude'])
        }

    return diff, metrics


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

# %%
methods = [
    'mean',
    'inverse_variance',
    'vce',
    'residual_weighted',

    'nelder_mead',

    'de1220',
    'de1220_LO',


    'gaco',
    'gaco_LO',


    'cmaes',
    'cmaes_LO',


    'xnes',
    'xnes_LO',

]

savedir_ver = r'results/'
# %%
window_start, window_stop = datetime(2021, 1, 1, 0), datetime(2021, 1, 1, 0, 5)
# window_stop = window_start + timedelta(seconds=5618)
arclength = None
satellite_ids = ['47']

input_arcs = retrieve_arcs(AccessRequest(
    data_type='KO',
    # data_format='sp3k',
    satellite_id=satellite_ids,
    analysis_centre=['IFG', 'AIUB', 'TUD'],
    window_start=window_start,
    window_stop=window_stop,
    round_seconds=True,
    get_data=False))[satellite_ids[0]]

verification_arc_dict = {satellite_ids[0]: [input_arcs[0]]}
verification_uncorrupted = {window_start: verification_arc_dict}
reference_orbit = [input_arcs[0]]
plot_reference_orbit = reference_orbit[0].trajectory

#  Data for test 7 (full orbit)
window_start7 = datetime(2021, 1, 1, 0)
window_stop7 = window_start7 + timedelta(seconds=5618)
satellite_ids = ['47']

input_arcs7 = retrieve_arcs(AccessRequest(
    data_type='KO',
    # data_format='sp3k',
    satellite_id=satellite_ids,
    analysis_centre=['IFG', 'AIUB', 'TUD'],
    window_start=window_start7,
    window_stop=window_stop7,
    round_seconds=True,
    get_data=False))[satellite_ids[0]]

# input_arcs7.trajectory = input_arcs7[0].trajectory.asfreq(freq='1min')

verification_arc_dict7 = {satellite_ids[0]: [input_arcs7[0]]}
verification_uncorrupted7 = {window_start7: verification_arc_dict7}
reference_orbit7 = [input_arcs7[0]]
plot_reference_orbit7 = reference_orbit7[0].trajectory
#%%
plot_reference_orbit.to_csv(f'{savedir_ver}/plot_reference_orbit.csv')
plot_reference_orbit7.to_csv(f'{savedir_ver}/plot_reference_orbit7.csv')

print('Finished importing data')
# %% Select tests to run

run_test1 = 0
run_test2 = 0
run_test3 = 0
run_test4 = 0
run_test5 = 0
run_test6 = 0
run_test7 = 0
run_test8 = 0
run_test9 = 0
run_test10 = 0

noise_type = 'Gaussian'  # 'Pink'  #

# % Perform verification test1
n_orbits1 = [3, 5, 7, 10, 15, 25]
# run_test1 = 0
if run_test1:
    print('----------------- Running test 1 ----------------')
    verification_results1 = []
    for i in n_orbits1:
        verification_results = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=[0.0],
            noise_std=0.005,
            arclength=arclength,
            scaling_factors=[1.0] * i,  # Example scaling factors
            mirrored=True,
            pygmo_variables=[1, 25, 42],
            noise_type=noise_type
        )
        verification_results1.append(verification_results)

    np.save(f'{savedir_ver}/verification1', verification_results1)

# % Perform verification test2

n_orbits2 = [2, 3, 5, 6, 8, 10, 15]
test_description2 = {'seed 42': 42,
                     'seed 20': 20}
# run_test2 = 0
if run_test2:
    print('----------------- Running test 2 ----------------')
    verification_results2 = []
    for _, seed in test_description2.items():
        ver2 = []
        for i in n_orbits2:
            np.random.seed(seed)  # fixing seed
            verification_results = validate_combination_methods_general(
                input_arcs=reference_orbit,
                methods=methods,
                # validation_arcs=[validation_arcs[1]],
                bias_list=[0.0],
                noise_std=0.005,
                arclength=arclength,
                scaling_factors=[1.0] * i,  # Example scaling factors
                mirrored=False,
                pygmo_variables=[1, 25, 42],
                noise_type=noise_type
            )
            ver2.append(verification_results)
        verification_results2.append(ver2)
    np.save(f'{savedir_ver}/verification2', verification_results2)

np.random.seed(42)  # fixing seed

# % Verification test 3 - random seeds
n_seeds = 20
np.random.seed(42)  # fixing seed
seed_list = [np.random.randint(1000) for i in range(n_seeds)]
# run_test3 = 0
if run_test3:

    # Testing the effect of random seed for the pygmo optimisers

    print('----------------- Running test 3 ----------------')

    verification_results31 = []

    for i in seed_list:
        np.random.seed(42)  # fixing seed
        verification_results = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=[0.0],
            noise_std=0.005,
            arclength=arclength,
            scaling_factors=[1.0] * 3,  # Example scaling factors
            mirrored=False,
            adjust_var=False,
            pygmo_variables=[1, 25, i],
            noise_type=noise_type
        )
        verification_results31.append(verification_results)
    np.save(f'{savedir_ver}/verification31', verification_results31)
    # Testing the effect of random seed of the generated corrupted orbits

    verification_results32 = []
    # seed_list2 = [np.random.randint(1000) for i in range(n_seeds)]

    for i in seed_list:
        # np.random.seed(42)  # fixing seed
        verification_results = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=[0.0],
            noise_std=0.005,
            arclength=arclength,
            scaling_factors=[1.0] * 3,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, 25, 42],
            noise_type=noise_type
        )
        verification_results32.append(verification_results)
    np.save(f'{savedir_ver}/verification32', verification_results32)

# % Verification test 4: Random bias_list
n_bias = [0.02, 0.04, 0.06, 0.08, 0.1]
# n_bias = [0.2, 0.4]  # , 0.6, 0.8, 1]
len_bias = [3, 5, 8]
# run_test4 = 0
if run_test4:

    print('----------------- Running test 4 ----------------')

    # Test 41: checking the influence of pure bias without adjusting the variance information
    verification_results41 = []
    for bias_len in len_bias:
        np.random.seed(42)  # fixing seed
        bias_verification = []
        input_bias = (np.random.random(bias_len) - 0.5)

        for scale in n_bias:

            bias_list = input_bias * scale

            np.random.seed(42)  # fixing seed
            verification4 = validate_combination_methods_general(
                input_arcs=reference_orbit,
                methods=methods,
                # validation_arcs=[validation_arcs[1]],
                bias_list=bias_list,
                noise_std=0.005,
                arclength=arclength,
                scaling_factors=[1.0] * 1,  # Example scaling factors
                mirrored=False,
                pygmo_variables=[1, 25, 42],
                adjust_var=False,
                noise_type=noise_type
            )
            bias_verification.append(verification4)
        verification_results41.append(bias_verification)
    np.save(f'{savedir_ver}/verification41', verification_results41)

# % Verification test 5 - generation size

n_generations = [1, 10, 20, 40]

input_scale = [1]
input_bias = (np.random.random(3) - 0.5) / 25
test56_methods = [
    'nelder_mead',

    'gaco',
    'gaco_LO',
    
  
    'cmaes',
    'cmaes_LO',
    
    'xnes',
    'xnes_LO',

    'de1220',
    'de1220_LO',
]
# run_test5 = 0
if run_test5:
    verification_results5 = []
    print('----------------- Running test 5 ----------------')
    for generations in n_generations:

        np.random.seed(42)  # fixing seed
        verification5 = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=test56_methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=input_bias,
            noise_std=0.005,
            arclength=arclength,
            scaling_factors=input_scale,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, generations, 42],
            adjust_var=False,
            noise_type=noise_type
        )

        verification_results5.append(verification5)
    np.save(f'{savedir_ver}/verification5', verification_results5)

# % Verification test 6 - population size
n_pop = [75, 150, 250, 500]

# input_bias = (np.random.random(3) - 0.5)/5
input_scale = [1]
# run_test6 = 0
if run_test6:
    verification_results6 = []
    print('----------------- Running test 6 ----------------')
    for pop_size in n_pop:

        np.random.seed(42)  # fixing seed
        verification6 = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=test56_methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=input_bias,
            noise_std=0.005,
            arclength=arclength,
            scaling_factors=input_scale,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, 25, 42, pop_size],
            noise_type=noise_type
        )

        verification_results6.append(verification6)
    np.save(f'{savedir_ver}/verification6', verification_results6)


# %% Verification test 7 - arclength
ref_7 = reference_orbit7[0]
ref_7.trajectory = reference_orbit7[0].trajectory.asfreq('1min')
ref_7 = [ref_7]

n_arclength = [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2), pd.Timedelta(
    minutes=5), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30), None]

input_bias = (np.random.random(3) - 0.5) / 25
# run_test7 = 0
input_scale = [1]
if run_test7:
    # if noise_type == 'Gaussian':
    verification_results71 = []
    print('----------------- Running test 7 ----------------')
    for arclength_test in n_arclength:
        # bias_list = input_bias

        verification7 = validate_combination_methods_general(
            input_arcs=ref_7,
            methods=methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=input_bias,
            noise_std=0.005,
            arclength=arclength_test,
            scaling_factors=input_scale,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, 25, 42, 250],
            noise_type='Gaussian'
        )

        verification_results71.append(verification7)
    np.save(f'{savedir_ver}/verification71', verification_results71)

    # if noise_type == 'Pink':
    verification_results72 = []
    print('----------------- Running test 7 ----------------')
    for arclength_test in n_arclength:
        # bias_list = input_bias

        verification7 = validate_combination_methods_general(
            input_arcs=ref_7,
            methods=methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=input_bias,
            noise_std=0.005,
            arclength=arclength_test,
            scaling_factors=input_scale,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, 25, 42, 250],
            noise_type='Pink'
        )

        verification_results72.append(verification7)
    np.save(f'{savedir_ver}/verification72', verification_results72)

# %% Verification test 8 - different reference frames

n_orbits8 = [3]

input_bias = (np.random.random(3) - 0.5) / 25

if run_test8:
    verification_results81 = []
    verification_results82 = []
    input8 = [Arc(trajectory=itrs_to_gcrs(reference_orbit[0].trajectory))]
    print('----------------- Running test 8 ----------------')

    for orbit in n_orbits8:
        bias_list = input_bias

        verification8 = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=bias_list,
            noise_std=0.005,
            arclength=None,
            scaling_factors=[1.0] * orbit,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, 25, 42, 250]
        )

        verification_results81.append(verification8)

    np.save(f'{savedir_ver}/verification81', verification_results81)

    verification8 = validate_combination_methods_general(
        input_arcs=input8,
        methods=methods,
        # validation_arcs=[validation_arcs[1]],
        bias_list=bias_list,
        noise_std=0.005,
        arclength=None,
        scaling_factors=[1.0] * orbit,  # Example scaling factors
        mirrored=False,
        pygmo_variables=[1, 25, 42, 250]
    )

    verification_results82.append(verification8)

    np.save(f'{savedir_ver}/verification82', verification_results82)

# %% Testing various algorithms
test9_methods = [
    'gaco',
    # 'gaco_individual',

    'cmaes',
    # 'cmaes_individual',

    'xnes',
    # 'xnes_individual',

    'de1220',
    # 'de1220_individual'
]

test9_methods_LO = [

    'gaco_LO',
    # 'gaco_individual_LO',

    'nelder_mead',

    'cmaes_LO',
    # 'cmaes_individual_LO',

    'xnes_LO',
    # 'xnes_individual_LO',

    'de1220_LO',
    # 'de1220_individual_LO'
]

# % Verification test 91 - population size
n_pop9 = [7, 15, 50, 75, 150, 300]

input_bias = (np.random.random(3) - 0.5) / 25
input_scale = [1]

if run_test9:
    verification_results91 = []
    print('----------------- Running test 91 ----------------')

    for pop_size in n_pop9:

        if pop_size < 75:
            np.random.seed(42)  # fixing seed
            verification91 = validate_combination_methods_general(
                input_arcs=reference_orbit,
                methods=test9_methods[2:],
                # validation_arcs=[validation_arcs[1]],
                bias_list=input_bias,
                noise_std=0.005,
                arclength=arclength,
                scaling_factors=input_scale,  # Example scaling factors
                mirrored=False,
                pygmo_variables=[1, 25, 42, pop_size],
                noise_type=noise_type
            )
        else:

            np.random.seed(42)  # fixing seed
            verification91 = validate_combination_methods_general(
                input_arcs=reference_orbit,
                methods=test9_methods,
                # validation_arcs=[validation_arcs[1]],
                bias_list=input_bias,
                noise_std=0.005,
                arclength=arclength,
                scaling_factors=input_scale,  # Example scaling factors
                mirrored=False,
                pygmo_variables=[1, 25, 42, pop_size],
                noise_type=noise_type
            )

        verification_results91.append(verification91)
    np.save(f'{savedir_ver}/verification91', verification_results91)

    verification_results92 = []
    print('----------------- Running test 92 (LO) ----------------')

    for pop_size in n_pop9:

        if pop_size < 75:
            np.random.seed(42)  # fixing seed
            verification92 = validate_combination_methods_general(
                input_arcs=reference_orbit,
                methods=test9_methods_LO[2:],
                # validation_arcs=[validation_arcs[1]],
                bias_list=input_bias,
                noise_std=0.005,
                arclength=arclength,
                scaling_factors=input_scale,  # Example scaling factors
                mirrored=False,
                pygmo_variables=[1, 25, 42, pop_size],
                noise_type=noise_type
            )
        else:

            np.random.seed(42)  # fixing seed
            verification92 = validate_combination_methods_general(
                input_arcs=reference_orbit,
                methods=test9_methods_LO,
                # validation_arcs=[validation_arcs[1]],
                bias_list=input_bias,
                noise_std=0.005,
                arclength=arclength,
                scaling_factors=input_scale,  # Example scaling factors
                mirrored=False,
                pygmo_variables=[1, 25, 42, pop_size],
                noise_type=noise_type
            )

        verification_results92.append(verification92)
    np.save(f'{savedir_ver}/verification92', verification_results92)

# % Verification test 10 gen size
n_gen9 = [1, 3, 10, 15, 20, 25, 40]
# n_gen9 = [5]

# input_bias = (np.random.random(3) - 0.5)/5
input_scale = [1]

# run_test93 = 1

if run_test10:
    t = time.time()
    verification_results101 = []
    print('----------------- Running test 10-1 ----------------')
    for gen_size in n_gen9:

        np.random.seed(42)  # fixing seed
        verification101 = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=test9_methods,
            # validation_arcs=[validation_arcs[1]],
            bias_list=input_bias,
            noise_std=0.005,
            arclength=arclength,
            scaling_factors=input_scale,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, gen_size, 42, 150],
            noise_type=noise_type
        )

        verification_results101.append(verification101)
    np.save(f'{savedir_ver}/verification101', verification_results101)
    time92 = time.time() - t

    verification_results102 = []
    print('----------------- Running test 10-2 ----------------')
    for gen_size in n_gen9:

        np.random.seed(42)  # fixing seed
        verification102 = validate_combination_methods_general(
            input_arcs=reference_orbit,
            methods=test9_methods_LO,
            # validation_arcs=[validation_arcs[1]],
            bias_list=input_bias,
            noise_std=0.005,
            arclength=arclength,
            scaling_factors=input_scale,  # Example scaling factors
            mirrored=False,
            pygmo_variables=[1, gen_size, 42, 150],
            noise_type=noise_type
        )

        verification_results102.append(verification102)
    np.save(f'{savedir_ver}/verification102', verification_results102)
