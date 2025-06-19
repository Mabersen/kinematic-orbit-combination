# -*- coding: utf-8 -*-
"""
pygmo_optimiser.py
 
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

import pandas as pd
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
from gravtools.kinematic_orbits.kinematic_utilities import compute_residuals

def combine_solution(dataframes, weights, delta, run_var=False):
    """
    Hybrid combination:
    1. Original method for epochs with NO NaN in any input
    2. Simple mean for epochs with ANY NaN in inputs
    """
    # Get union of all epochs
    union_index = dataframes[0].index
    for df in dataframes[1:]:
        union_index = union_index.union(df.index)
    union_index = union_index.sort_values()

    # Create mask for epochs with complete data
    complete_mask = pd.Series(True, index=union_index)
    for df in dataframes:
        df_aligned = df.reindex(union_index)
        has_nans = df_aligned[['x_pos', 'y_pos', 'z_pos']].isna().any(axis=1)
        complete_mask &= ~has_nans

    # Split into complete and incomplete epochs
    complete_index = union_index[complete_mask]
    incomplete_index = union_index[~complete_mask]

    # 1. Combine complete epochs using original method
    complete_dfs = [df.reindex(complete_index) for df in dataframes]
    combined_complete = combine_solution_intersect(complete_dfs, weights, delta)

    # 2. Combine incomplete epochs using NaN-aware mean
    combined_incomplete = pd.DataFrame(index=incomplete_index)
    for col in ['x_pos', 'y_pos', 'z_pos']:
        values = pd.concat(
            [df[col].reindex(incomplete_index) for df in dataframes], 
            axis=1
        )
        combined_incomplete[col] = np.nanmean(values, axis=1)

    if run_var:
        # Error propagation for incomplete epochs
        for std_col in ['std_x', 'std_y', 'std_z']:
            coord = std_col.split('_')[1]
            pos_col = f'{coord}_pos'
            
            # Recompute values for this coordinate
            values = pd.concat(
                [df[pos_col].reindex(incomplete_index) for df in dataframes], 
                axis=1
            )
            valid_counts = values.notna().sum(axis=1)
            
            # Compute variance
            var_arrays = []
            for df in dataframes:
                std_values = df[std_col].reindex(incomplete_index)**2
                count = ~df[pos_col].reindex(incomplete_index).isna()
                var_arrays.append(std_values.where(count, 0))
            
            total_var = pd.concat(var_arrays, axis=1).sum(axis=1)
            valid_counts = valid_counts.replace(0, np.nan)
            combined_incomplete[std_col] = np.sqrt(total_var / (valid_counts**2))
        
        for cov_col in ['cov_xy', 'cov_xz', 'cov_yz']:
            c1, c2 = list(cov_col.split('_')[1])
            pos_col1, pos_col2 = f'{c1}_pos', f'{c2}_pos'
            
            # Recompute valid counts for this covariance pair
            values1 = pd.concat([df[pos_col1].reindex(incomplete_index) for df in dataframes], axis=1)
            values2 = pd.concat([df[pos_col2].reindex(incomplete_index) for df in dataframes], axis=1)
            valid_counts = (values1.notna() & values2.notna()).sum(axis=1)
            
            # Compute covariance
            cov_arrays = []
            for df in dataframes:
                cov_values = df[cov_col].reindex(incomplete_index)
                count = (~df[pos_col1].reindex(incomplete_index).isna() &
                         ~df[pos_col2].reindex(incomplete_index).isna())
                cov_arrays.append(cov_values.where(count, 0))
            
            total_cov = pd.concat(cov_arrays, axis=1).sum(axis=1)
            valid_counts = valid_counts.replace(0, np.nan)
            combined_incomplete[cov_col] = total_cov / (valid_counts**2)

    # 3. Combine results
    return pd.concat([combined_complete, combined_incomplete]).sort_index()

def combine_solution_intersect(dataframes, weights, delta, run_var=False):
    """
    Compute a weighted combination of satellite trajectories across analysis centres.

    Parameters:
        dataframes (list of pd.DataFrame): List of dataframes containing ['x_pos', 'y_pos', 'z_pos'] columns,
                                           indexed by pandas datetime.
        weights (np.ndarray): 1D array of weights of shape (n * 3,), where n is the number of dataframes.

    Returns:
        pd.DataFrame: A dataframe containing the weighted combined ['x_pos', 'y_pos', 'z_pos'], indexed by aligned datetime.
    """
    # Find common epochs across all dataframes
    # dataframes = [i[['x_pos', 'y_pos', 'z_pos']] for i in dataframes]

    index = dataframes[0].index
    # for i in range(1, len(dataframes)):
    #     df = dataframes[i]
    #     index2 = df.index
    #     index = index.intersection(index2).sort_values().unique()

    # Reindex each dataframe to ensure they all have the same epochs (with NaN for missing ones)
    # dataframes = [df.reindex(index) for df in dataframes]
    filtered_dfs = dataframes
    # Split weights into x, y, z components
    n = len(dataframes)
    W_x, W_y, W_z = weights[:n], weights[n:2 * n], weights[2 * n:]

    # Compute the weighted combination for each position, non-normalised
    x_comb = sum(W_x[i] * delta[0] + filtered_dfs[i]['x_pos'] / n for i in range(n))
    y_comb = sum(W_y[i] * delta[1] + filtered_dfs[i]['y_pos'] / n for i in range(n))
    z_comb = sum(W_z[i] * delta[2] + filtered_dfs[i]['z_pos'] / n for i in range(n))

    # x_comb = sum(filtered_dfs[i]['x_pos'] * W_x[i] / sum(W_x) for i in range(n))
    # y_comb = sum(filtered_dfs[i]['y_pos'] * W_y[i] / sum(W_y) for i in range(n))
    # z_comb = sum(filtered_dfs[i]['z_pos'] * W_z[i] / sum(W_z) for i in range(n))

    # Combine results into a single dataframe
    combined_df = pd.DataFrame({
        'x_pos': x_comb,
        'y_pos': y_comb,
        'z_pos': z_comb
    }, index=index)
    
    if run_var:
        # 2. Compute variances (assuming delta has negligible uncertainty)
        for std_col in ['std_x', 'std_y', 'std_z']:
            coord = std_col.split('_')[1]
            # Compute mean variance: Var(mean) = Σ(var_i) / n²
            var_sum = sum(df[std_col]**2 for df in dataframes)
            combined_df[std_col] = np.sqrt(var_sum / (n**2))
        
        # 3. Compute covariances (assuming delta contributions are uncorrelated)
        for cov_col in ['cov_xy', 'cov_xz', 'cov_yz']:
            c1, c2 = cov_col.split('_')[1]
            # Compute mean covariance: Cov(mean) = Σ(cov_ij) / n²
            cov_sum = sum(df[cov_col] for df in dataframes)
            combined_df[cov_col] = cov_sum / (n**2)
    
    return combined_df

class OrbitOptimizationProblem:
    def __init__(self, input_orbits, validation_data, delta):
        self.input_orbits = input_orbits
        self.validation_data = validation_data
        self.delta = delta
        self.n = len(input_orbits)
        
        # # Create union index
        # self.index = input_orbits[0].index
        # for df in input_orbits[1:]:
        #     self.index = self.index.intersection(df.index)
        # self.index = self.index.sort_values().unique()
        # self.validation_data = [i.reindex(self.index) for i in self.validation_data]
        # Pre-filter data to common epochs
        self.input_orbits, self.validation_data = self._filter_data()

    def _filter_data(self):
        """
        Pre-filter the input orbits and validation data to align to common epochs.

        Returns:
            tuple: Filtered input orbits and validation data.
        """
        dataframes = self.input_orbits
        reference_data = self.validation_data

        index = dataframes[0].index

        for i in range(1, self.n):
            df = dataframes[i]
            index2 = df.index
            index = index.union(index2).sort_values().unique()
            # index = index.union(index2).sort_values().unique()
        
        self.index = index
        index3 = index
        
        for j in reference_data:

            index2 = j.index
            index3 = index3.union(index2).sort_values().unique()
            
        # Reindex each dataframe to ensure they all have the same epochs (with NaN for missing ones)
        dataframes = [df1.reindex(index) for df1 in dataframes]
        reference_data = [df2.reindex(index3) for df2 in reference_data]

        return dataframes, reference_data
    
    
    def fitness(self, weights):
        combined = combine_solution(
            self.input_orbits,
            weights,
            self.delta
        )

        total_rmse = 0
        for ref_df in self.validation_data:
            # Align validation data with union index
            ref_aligned = ref_df
            
            # Find common valid epochs
            valid_mask = combined.notna().all(axis=1) & ref_aligned.notna().all(axis=1)
            
            if valid_mask.any():
                # Original compute_rmse logic
                squared_diff = (combined.loc[valid_mask] - ref_aligned.loc[valid_mask])**2
                squared_error = squared_diff.sum(axis=1)  # Sum x+y+z errors per epoch
                mean_squared_error = squared_error.mean()
                error = np.sqrt(mean_squared_error)
                total_rmse += error
                
        return [total_rmse / len(self.validation_data)]

    def get_bounds(self):
        """
        Define bounds for the weights (e.g., [0, 1] for all weights).

        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        lower_bounds = [-15] * (3 * self.n)
        upper_bounds = [15] * (3 * self.n)
        return (lower_bounds, upper_bounds)

    def get_name(self):
        return "Orbit Optimization Problem"

    def get_nobj(self):
        return 1  # Single objective: minimize average RMSE

    def gradient(self, x):
        # return pg.estimate_gradient(lambda x: self.fitness(x), x)
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


def optimise_combined_orbit_pygmo(input_orbits, validation_data, no_evolutions=1, no_generations=25,
                                  pop_size=150, fixed_seed=None, algorithm='Nelder-Mead', delta=0.001, local_optimisation=True, use_arch=False):
    # Instantiate the problem
    if fixed_seed == None:
        fixed_seed = np.random.randint(1, 99999)
    delta_list = []
    for i in validation_data:
        input_rmse = compute_residuals(input_orbits, i)
        # delta_list.append(np.min(input_rmse, axis=0) / 10)
        delta_list.append(np.mean(input_rmse, axis=0) / 10)

    delta_list = np.array(delta_list)
    # delta = np.min(delta_list, axis=0)
    delta = np.mean(delta_list, axis=0)  # alternative method using the mean

    global_log = None
    local_log = None

    # print('Pre-rounded Delta : ', delta)

    for idx, d in enumerate(delta):
        count_index = 0
        for i in str(d):
            count_index += 1
            if i != '0' and i != '.':
                break
        delta[idx] = round(d, count_index - 1)
    # print('Determined Delta : ', delta)

    problem = pg.problem(OrbitOptimizationProblem(input_orbits, validation_data, delta=delta))

    print('\n########### PRINTING PROBLEM INFORMATION ###########\n')
    print(problem)
    # Set up the algorithm
    input_algorithm = algorithm
    # print(problem.fitness(np.ones(9) / 3))

    # algorithm = pg.gaco(gen=no_generations, seed=fixed_seed)
    # algorithm = pg.bee_colony(gen=no_generations, seed=fixed_seed)
    # algorithm = pg.compass_search(start_range=1e-10, stop_range=1e-10)

    if input_algorithm == 'nelder_mead':
        algorithm = pg.scipy_optimize(method="Nelder-Mead")
        no_evolutions = 1

    if input_algorithm == 'COBYLA':
        algorithm = pg.scipy_optimize(method="COBYLA")
        no_evolutions = 1

    if input_algorithm in ['gaco', 'de1220', 'gaco_LO', 'de1220_LO', 'cmaes', 'xnes', 'cmaes_LO', 'xnes_LO']:

        if 'gaco' in input_algorithm:
            algorithm = pg.gaco(gen=no_generations, seed=fixed_seed)

        if 'de1220' in input_algorithm:
            algorithm = pg.de1220(gen=no_generations, seed=fixed_seed, ftol=1e-8, xtol=1e-11)

        if 'cmaes' in input_algorithm:
            algorithm = pg.cmaes(gen=no_generations, seed=fixed_seed, ftol=1e-7, xtol=1e-7)

        if 'xnes' in input_algorithm:
            algorithm = pg.xnes(gen=no_generations, seed=fixed_seed, ftol=1e-7, xtol=1e-7)

        # algorithm = pg.de(seed=fixed_seed)
        # algorithm = pg.ipopt()
        algo = pg.algorithm(algorithm)
        algo.set_verbosity(1)

        # print('\n########### PRINTING ALGORITHM INFORMATION ###########\n')
        # print(algo)
        # # Create the population
        pop = pg.population(problem, size=pop_size, seed=fixed_seed)

        fitness_change = []
        weights_change = []

        # Evolve population multiple times
        for i in range(no_evolutions):

            pop = algo.evolve(pop)
            best_weights = pop.champion_x
            best_rmse = pop.champion_f
            fitness_change.append(best_rmse)
            weights_change.append(best_weights)

        # print(f"Best Weights, evolution {i}:", best_weights)
        # print(f"Best RMSE, evolution {i}:", best_rmse)

        if 'gaco' in input_algorithm:
            uda = algo.extract(pg.gaco)
            global_log = uda.get_log()

        if 'de1220' in input_algorithm:
            uda = algo.extract(pg.de1220)
            global_log = uda.get_log()

        if 'cmaes' in input_algorithm:
            uda = algo.extract(pg.cmaes)
            global_log = uda.get_log()

        if 'xnes' in input_algorithm:
            uda = algo.extract(pg.xnes)
            global_log = uda.get_log()
            # plt.semilogy([entry[0] for entry in log], [entry[2]for entry in log], 'k--')
            # plt.show()

        # Get the results
        best_weights = pop.champion_x
        best_rmse = pop.champion_f

        fitness_change = np.array(fitness_change)
        weights_change = np.array(weights_change)

        # -----------------------------------

        # Local optimisation (optional):
        if local_optimisation:
            algorithm = pg.scipy_optimize(method="Nelder-Mead")
            algo = pg.algorithm(algorithm)

            pop = pg.population(problem, size=0, seed=fixed_seed)
            pop.push_back(best_weights, best_rmse)

            pop = algo.evolve(pop)
            best_weights = pop.champion_x
            best_rmse = pop.champion_f
            # uda = algo.extract(pg.scipy_optimize)
            # local_log = uda.get_log(method="Nelder-Mead")

        # -----------------------------------

        print("Best Weights:", best_weights)
        print("Best RMSE:", best_rmse)

        combined_orbit = combine_solution(input_orbits, best_weights, delta=delta, run_var = True)

        return combined_orbit, best_weights, (fitness_change, weights_change, global_log, local_log, delta)

    else:

        algo = pg.algorithm(algorithm)
        algo.set_verbosity(1)

        # print('\n########### PRINTING ALGORITHM INFORMATION ###########\n')
        # print(algo)
        # # Create the population
        pop = pg.population(problem, size=pop_size, seed=fixed_seed)

        # a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=1000))
        # a_cstrs_sa.set_verbosity(10)
        # archi = pg.archipelago(n=16, algo=a_cstrs_sa, prob=problem, pop_size=70)
        # print(archi)
        # archi.evolve(n=200)
        # best_weights = archi.get_champions_x()
        # best_rmse = archi.get_champions_f()
        # print("Best Weights:", best_weights)
        # print("Best RMSE:", best_rmse)

        fitness_change = []
        weights_change = []
        # Evolve population multiple times
        for i in range(no_evolutions):

            pop = algo.evolve(pop)
            best_weights = pop.champion_x
            best_rmse = pop.champion_f
            fitness_change.append(best_rmse)
            weights_change.append(best_weights)

            # print(f"Best Weights, evolution {i}:", best_weights)
            # print(f"Best RMSE, evolution {i}:", best_rmse)
        # uda = algo.extract(pg.scipy_optimize)
        # local_log = uda.get_log(method="Nelder-Mead")

        # Get the results
        best_weights = pop.champion_x
        best_rmse = pop.champion_f

        fitness_change = np.array(fitness_change)
        weights_change = np.array(weights_change)

        combined_orbit = combine_solution(input_orbits, best_weights, delta=delta, run_var = True)

        print("Best Weights:", best_weights)
        print("Best RMSE:", best_rmse)

        return combined_orbit, best_weights, (fitness_change, weights_change, global_log, local_log, delta)

def optimise_combined_orbit_pygmo2():
    return