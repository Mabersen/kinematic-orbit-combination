# -*- coding: utf-8 -*-
"""
plot_verification_results.py
 
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
from verification_utilities_styled import (plot_test1_rmse, plot_test2_rmse, plot_test3_rmse, plot_test4_rmse, 
                                           plot_test41_2_rmse, plot_test5_rmse, plot_test6_rmse, plot_test7_rmse, 
                                           plot_test8_rmse, plot_test9_rmse, plot_test10_rmse)
np.random.seed(42)  # fixing seed
# Parameter definition
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

# Gauss
savedir_ver = r'results/'

# Test 1: Symmetric corruption
n_orbits1 = [3, 5, 7, 10, 15, 25]

# Test 2: Asymmetric corruption
n_orbits2 = [2, 3, 5, 6, 8, 10, 15]
test_description2 = {'seed 42': 42,
                     'seed 20': 20}

# Test 3-1: 20 random seeds, pygmo
n_seeds3 = 20
seed_list3 = [np.random.randint(1000) for i in range(n_seeds3)]

# Test 3-2: 20 random seeds, input orbits

# Test 4-1: Different bias scales
n_bias4 = [0.02, 0.04, 0.06, 0.08, 0.1]
len_bias4 = [3, 5, 8]

# Test 4-2: Bias with different n orbits

# Test 5: Number of generations
n_generations5 = [1, 10, 20, 40]
input_scale5 = [1]
input_bias5 = (np.random.random(3) - 0.5) / 25

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

# Test 6: Population size
n_pop6 = [75, 150, 250, 500]
input_bias6 = input_bias5
input_scale6 = [1]

# Test 7: Arclengths
n_arclength7 = [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2), pd.Timedelta(
    minutes=5), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30), None]

n_arclength71_1 = [pd.Timedelta(seconds=1), pd.Timedelta(minutes=1), pd.Timedelta(
    minutes=5), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30), None]

input_bias7 = input_bias5
input_scale7 = [1]

# Test 8: ITRS vs GCRS reference frame
n_orbits8 = [3]
input_bias8 = input_bias5

# Test 9: population size, optimiser results; local and global
test9_methods = [
    'gaco','cmaes','xnes','de1220'
]

test9_methods_LO = [
    'gaco_LO','nelder_mead','cmaes_LO','xnes_LO','de1220_LO',
]

n_pop9 = [7, 15, 50, 75, 150, 300]
input_bias9 = (np.random.random(3) - 0.5) / 25
input_scale9 = [1]

# Test 10: Optimiser results; local and global

# % Verification test 10 gen size
n_gen10 = [1, 3, 10, 15, 20, 25, 40]

# input_bias = (np.random.random(3) - 0.5)/5
input_scale = [1]
# %% Load reference data
plot_reference_orbit = pd.read_csv(f'{savedir_ver}/plot_reference_orbit.csv')
plot_reference_orbit['datetime'] = pd.to_datetime(plot_reference_orbit['datetime'])
plot_reference_orbit.set_index('datetime', inplace=True)

plot_reference_orbit7 = pd.read_csv(f'{savedir_ver}/plot_reference_orbit7.csv')
plot_reference_orbit7['datetime'] = pd.to_datetime(plot_reference_orbit7['datetime'])
plot_reference_orbit7.set_index('datetime', inplace=True)

plot_reference_orbit71_1 = pd.read_csv(f'{savedir_ver}/plot_reference_orbit71_1.csv')
plot_reference_orbit71_1['datetime'] = pd.to_datetime(plot_reference_orbit71_1['datetime'])
plot_reference_orbit71_1.set_index('datetime', inplace=True)

# %% load results
verification_results1 = np.load(f'{savedir_ver}/verification1.npy', allow_pickle=True)
verification_results2 = np.load(f'{savedir_ver}/verification2.npy', allow_pickle=True)
verification_results31 = np.load(f'{savedir_ver}/verification31.npy', allow_pickle=True)
verification_results32 = np.load(f'{savedir_ver}/verification32.npy', allow_pickle=True)
# ----------------------------------------------------------------------------------------
verification_results41 = np.load(f'{savedir_ver}/verification41.npy', allow_pickle=True)
verification_results41_2 = [[i[2] for i in verification_results41]]
# verification_results42 = np.load(f'{savedir_ver}/verification41.npy', allow_pickle=True)
# verification_results42_2 = [[i[0] for i in verification_results42]]
verification_results5 = np.load(f'{savedir_ver}/verification5.npy', allow_pickle=True)
verification_results6 = np.load(f'{savedir_ver}/verification6.npy', allow_pickle=True)
verification_results71 = np.load(f'{savedir_ver}/verification71.npy', allow_pickle=True)
verification_results71_1 = np.load(f'{savedir_ver}/verification71_1.npy', allow_pickle=True)
verification_results71_2 = np.load(f'{savedir_ver}/verification71_2.npy', allow_pickle=True)

# verification_results81 = np.load(f'{savedir_ver}/verification81.npy', allow_pickle=True)
# verification_results82 = np.load(f'{savedir_ver}/verification82.npy', allow_pickle=True)

verification_results91_1 = np.load(f'{savedir_ver}/extra_seed_new/verification91.npy', allow_pickle=True)
verification_results92_1 = np.load(f'{savedir_ver}/extra_seed_new/verification92.npy', allow_pickle=True)

verification_results91_2 = np.load(f'{savedir_ver}/extra_seed_new/verification91_2.npy', allow_pickle=True)
verification_results92_2 = np.load(f'{savedir_ver}/extra_seed_new/verification92_2.npy', allow_pickle=True)

verification_results101_42 = np.load(f'{savedir_ver}/extra_seed_new/verification101_2.npy', allow_pickle=True)
verification_results102_42 = np.load(f'{savedir_ver}/extra_seed_new/verification102_2.npy', allow_pickle=True)

verification_results101_20 = np.load(f'{savedir_ver}/extra_seed_new/verification101_2_20.npy', allow_pickle=True)
verification_results102_20 = np.load(f'{savedir_ver}/extra_seed_new/verification102_2_20.npy', allow_pickle=True)

# %% Plotting results
# %%Test 1: Symmetric corruption
# plot_test1_rmse(verification_results1, plot_reference_orbit, n_orbits1, methods[:])
plot_test1_rmse(verification_results1, plot_reference_orbit, n_orbits1, methods[0:4] + methods[4::2])
plot_test1_rmse(verification_results1, plot_reference_orbit, n_orbits1, methods[5::2])
# %% Test 2: Asymmetric corruption

# plot_test2_rmse(verification_results2, plot_reference_orbit, n_orbits2, test_description2, methods[:])
plot_test2_rmse(verification_results2, plot_reference_orbit, n_orbits2, test_description2, methods[0:4] + methods[4::2])
# plot_test2_rmse(verification_results2, plot_reference_orbit, n_orbits2, test_description2, methods[4::2])
plot_test2_rmse(verification_results2, plot_reference_orbit, n_orbits2, test_description2, methods[5::2])
# %% Test 3: 20 random seeds, pygmo
plot_test3_rmse(verification_results32, plot_reference_orbit, seed_list3, methods[0:4] + methods[4::2])
# plot_test3_rmse(verification_results31, plot_reference_orbit, seed_list3, methods[4::2])
# plot_test3_rmse(verification_results31, plot_reference_orbit, seed_list3, methods[6::4])
plot_test3_rmse(verification_results32, plot_reference_orbit, seed_list3, methods[5::2])

# %%Test 3: 20 random seeds, input orbits
# plot_test3_rmse(verification_results32, plot_reference_orbit, seed_list3, methods[0:4])
plot_test3_rmse(verification_results31, plot_reference_orbit, seed_list3, methods[4::2])
plot_test3_rmse(verification_results31, plot_reference_orbit, seed_list3, methods[5::2])

# %%Test 4-1: Different bias scales
plot_test4_rmse(verification_results41, plot_reference_orbit, n_bias4, methods[0:3] + methods[3:4] + methods[5::2], bias_lengths=len_bias4)
plot_test4_rmse(verification_results41, plot_reference_orbit, n_bias4, methods[4::2], bias_lengths=len_bias4)
# plot_test4_rmse(verification_results41, plot_reference_orbit, n_bias4, methods[3:4] + methods[5::2], bias_lengths=len_bias4)

# %% Test 4-2: Bias with different n orbits
plot_test41_2_rmse(verification_results41_2, plot_reference_orbit, len_bias4, methods[:3], bias_scale = n_bias4[2])
plot_test41_2_rmse(verification_results41_2, plot_reference_orbit, len_bias4, methods[3:], bias_scale = n_bias4[2])
# plot_test4_rmse(verification_results41_2, plot_reference_orbit, len_bias4, methods[0:3], bias_lengths=len_bias4)
# plot_test4_rmse(verification_results41_2, plot_reference_orbit, len_bias4, methods[4::2], bias_lengths=len_bias4)
# plot_test4_rmse(verification_results41_2, plot_reference_orbit, len_bias4, methods[3:4] + methods[5::2], bias_lengths=len_bias4)

# %%Test 5: Number of generations MUST ADJUST FOR CORRECT PLOTS
plot_test5_rmse(verification_results5, plot_reference_orbit, n_generations5, test56_methods[0:1] + test56_methods[2::2])
plot_test5_rmse(verification_results5, plot_reference_orbit, n_generations5,
                test56_methods[1::2])
# plot_test5_rmse(verification_results5, plot_reference_orbit, n_generations5, test56_methods[2:4] + test56_methods[6::3])
# %%Test 6: Population size
# plot_test6_rmse(verification_results6, plot_reference_orbit, n_pop, test56_methods)
plot_test6_rmse(verification_results6, plot_reference_orbit, n_pop6, test56_methods[0:1] + test56_methods[2::2])
plot_test6_rmse(verification_results6, plot_reference_orbit, n_pop6,
                test56_methods[1::2])

# %% Test 7: Arclengths
# Gaussian
plot_test7_rmse(verification_results71_1, plot_reference_orbit71_1, n_arclength71_1, methods[0:4])

plot_test7_rmse(verification_results71, plot_reference_orbit7, n_arclength7, methods[4::2] + methods[5::2])

# %% Test 8: ITRS vs GCRS reference frame
# plot_test8_rmse(verification_results81, plot_reference_orbit, n_orbits8, methods)
# plot_test8_rmse(verification_results82, itrs_to_gcrs(
#     reference_orbit[0].trajectory), n_orbits8, methods)

# %%Plot test 9: Optimiser results; local and global
plot_test9_rmse(verification_results91_1, plot_reference_orbit, n_pop9, test9_methods[:])
plot_test9_rmse(verification_results92_1, plot_reference_orbit, n_pop9, test9_methods_LO)

plot_test9_rmse(verification_results91_2, plot_reference_orbit, n_pop9, test9_methods[:])
plot_test9_rmse(verification_results92_2, plot_reference_orbit, n_pop9, test9_methods_LO)

# %%Plot test 10: Optimiser results; local and global
#42
plot_test10_rmse(verification_results101_42, plot_reference_orbit, n_gen10, test9_methods[:])
plot_test10_rmse(verification_results102_42, plot_reference_orbit, n_gen10, test9_methods_LO)
#20
plot_test10_rmse(verification_results101_20, plot_reference_orbit, n_gen10, test9_methods[:])
plot_test10_rmse(verification_results102_20, plot_reference_orbit, n_gen10, test9_methods_LO)
