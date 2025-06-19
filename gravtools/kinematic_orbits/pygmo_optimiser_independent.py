# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:25:33 2024

@author: maber
"""

import pandas as pd
import numpy as np
import pygmo as pg
from gravtools.kinematic_orbits.kinematic_utilities import compute_residuals

def combine_solution2(dataframes, weights, delta):
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

    # # Reindex each dataframe to ensure they all have the same epochs (with NaN for missing ones)
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

    return combined_df

class OrbitParameterOptimization:
    def __init__(self, input_orbits, validation_data, parameter, delta):
        """
        Initialise the parameter-specific optimisation problem.

        Parameters:
            input_orbits (list of pd.DataFrame): List of input orbit dataframes.
            validation_data (list of pd.DataFrame): List of validation orbit dataframes.
            parameter (str): The parameter to optimise ('x_pos', 'y_pos', or 'z_pos').
        """
        self.input_orbits = input_orbits
        self.validation_data = validation_data
        self.parameter = parameter
        self.n = len(input_orbits)  # Number of analysis centres
        self.delta = delta

        # Pre-filter data to common epochs
        self.filtered_input_orbits, self.filtered_validation_data = self._filter_data()

    def _filter_data(self):
        """
        Pre-filter the input orbits and validation data to align to common epochs.

        Returns:
            tuple: Filtered input orbits and validation data.
        """
        # Find common epochs across all input orbits
        common_epochs = set(self.input_orbits[0].index)
        for orbit in self.input_orbits[1:]:
            common_epochs &= set(orbit.index)
        common_epochs = sorted(common_epochs)

        if not common_epochs:
            raise ValueError("No common epochs found across input orbits.")

        # Filter input orbits to common epochs
        filtered_input_orbits = [orbit.loc[common_epochs] for orbit in self.input_orbits]

        # Filter validation data to intersect with common epochs
        filtered_validation_data = []
        for val_data in self.validation_data:
            filtered_val = val_data.loc[val_data.index.intersection(common_epochs)]
            if filtered_val.empty:
                raise ValueError("No overlapping epochs between validation data and input orbits.")
            filtered_validation_data.append(filtered_val)

        return filtered_input_orbits, filtered_validation_data

    def fitness(self, weights):
        """
        Compute the objective function: average RMSE of the combined parameter with respect to validation data.

        Parameters:
            weights (list): Weights for the selected parameter.

        Returns:
            list: Objective function value (average RMSE).
        """
        # Compute weighted combined parameter

        fitness = 0
        # Compute RMSE for each validation set
        rmse_list = []
        for filtered_val in self.filtered_validation_data:

            combined_param = sum(((self.filtered_input_orbits[i][self.parameter] / self.n)
                                 + weights[i] * self.delta) for i in range(self.n))
            squared_diffs = (combined_param - filtered_val[self.parameter]) ** 2

            rmse = np.sqrt(np.mean(squared_diffs))
            rmse_list.append(rmse)

        fitness += np.mean(rmse_list)
        # Objective function: average RMSE across all validation sets
        return [fitness]

    def get_bounds(self):
        """
        Define bounds for the weights (e.g., [0, 1] for all weights).

        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        lower_bounds = [-15] * self.n
        upper_bounds = [15] * self.n
        return (lower_bounds, upper_bounds)

    def get_name(self):
        return f"Orbit Parameter Optimisation for {self.parameter}"

    def get_nobj(self):
        return 1  # Single objective: minimise average RMSE

    def gradient(self, x):
        # return pg.estimate_gradient(lambda x: self.fitness(x), x)
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


def optimise_combined_orbit_independent(input_orbits, validation_data, no_evolutions=1, no_generations=15, pop_size=150,
                                    fixed_seed=None, delta=0.001, algorithm='gaco', local_optimisation=True, use_arch=False):

    if fixed_seed == None:
        fixed_seed = np.random.randint(1, 99999)
    parameters = ['x_pos', 'y_pos', 'z_pos']
    final_weights = []
    final_rmse = []
    fitness_change = []
    weights_change = []

    delta_list = []
    for i in validation_data:
        input_rmse = compute_residuals(input_orbits, i)
        # delta_list.append(np.min(input_rmse, axis=0) / 10)
        delta_list.append(np.mean(input_rmse, axis=0) / 10)

    delta_list = np.array(delta_list)
    # delta = np.min(delta_list, axis=0)
    delta = np.mean(delta_list, axis=0)
    # print('Pre-rounded Delta : ', delta)

    for idx, d in enumerate(delta):
        count_index = 0
        for i in str(d):
            count_index += 1
            if i != '0' and i != '.':
                break
        delta[idx] = round(d, count_index - 1)
    # print('Determined Delta : ', delta)
    input_algorithm = algorithm
    local_logs = []
    global_logs = []

    for idx, parameter in enumerate(parameters):
        # Instantiate the problem
        problem = pg.problem(OrbitParameterOptimization(input_orbits, validation_data, parameter, delta=delta[idx]))

        # print('\n########### PRINTING PROBLEM INFORMATION ###########\n')
        # print(problem)
        # # Set up the algorithm
        global_log = None
        local_log = None

        if 'gaco' in input_algorithm:
            algorithm = pg.gaco(gen=no_generations, seed=fixed_seed)

        if 'de1220' in input_algorithm:
            algorithm = pg.de1220(gen=no_generations, seed=fixed_seed, ftol=1e-8, xtol=1e-11)

        if 'cmaes' in input_algorithm:
            algorithm = pg.cmaes(gen=no_generations, seed=fixed_seed, ftol=1e-7, xtol=1e-7)

        if 'xnes' in input_algorithm:
            algorithm = pg.xnes(gen=no_generations, seed=fixed_seed, ftol=1e-7, xtol=1e-7)
        # algorithm = pg.gaco(gen=no_generations, seed=fixed_seed)  # Trying different threshhold gen?
        # algorithm = pg.bee_colony(gen=no_generations, seed=fixed_seed)
        # algorithm = pg.compass_search(start_range=1e-10, stop_range=1e-10)
        # algorithm = pg.scipy_optimize(method="Nelder-Mead")
        # algorithm = pg.de(seed=fixed_seed)
        # algorithm = pg.ipopt()
        algo = pg.algorithm(algorithm)  # Differential Evolution
        algo.set_verbosity(1)
        # print('\n########### PRINTING ALGORITHM INFORMATION ###########\n')
        # print(algo)
        # # Create the population
        pop = pg.population(problem, size=pop_size, seed=fixed_seed)

        if use_arch == True:

            archi = pg.archipelago(n=15, algo=algo, prob=problem, pop_size=pop_size, seed=fixed_seed)
            print(archi)
            archi.evolve(n=no_generations)
            archi.wait()

            best_weights_total = np.array(archi.get_champions_x())
            best_fitness_total = np.array(archi.get_champions_f())
            best_rmse_idx = np.where(min(best_fitness_total))[0][0]

            # print(best_fitness_total)
            # print(best_rmse_idx)

            best_weights = list(best_weights_total[best_rmse_idx])
            best_rmse = best_fitness_total[best_rmse_idx]

            # print("Best Weights:", best_weights)
            # print("Best RMSE:", best_rmse)

            # combined_orbit = combine_solution(input_orbits, best_weights, delta=delta)

            # Local optimisation (optional):
            if local_optimisation:
                algorithm = pg.scipy_optimize(method="Nelder-Mead")
                algo = pg.algorithm(algorithm)

                pop = pg.population(problem, size=0, seed=fixed_seed)
                pop.push_back(best_weights, best_rmse)

                pop = algo.evolve(pop)
                best_weights = list(pop.champion_x)
                best_rmse = pop.champion_f
                # uda = algo.extract(pg.scipy_optimize)
                # local_log = uda.get_log(method="Nelder-Mead")

            # -----------------------------------

            # print("Best Weights:", best_weights)
            # print("Best RMSE:", best_rmse)

            final_weights += best_weights
            final_rmse += best_rmse.tolist()

        else:

            # Initialize empty containers
            fitness_list = []
            weights_over_evolutions = []
            # Evolve population multiple times
            for i in range(no_evolutions):

                pop = algo.evolve(pop)
                best_weights = pop.champion_x
                best_rmse = pop.champion_f
                fitness_list.append(best_rmse)
                weights_over_evolutions.append(best_weights)

                # print(f"Best Weights, evolution {i}:", best_weights)
                # print(f"Best RMSE, evolution {i}:", best_rmse)

            # Get the results
            best_weights = list(pop.champion_x)
            best_rmse = pop.champion_f

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
            global_logs.append(global_log)
            # -----------------------------

            # Local optimisation (optional):
            if local_optimisation:
                algorithm = pg.scipy_optimize(method="Nelder-Mead")
                algo = pg.algorithm(algorithm)

                pop = pg.population(problem, size=0, seed=fixed_seed)
                pop.push_back(best_weights, best_rmse)

                pop = algo.evolve(pop)

                best_weights = list(pop.champion_x)
                best_rmse = pop.champion_f
                # uda = algo.extract(pg.scipy_optimize)
                # local_log = uda.get_log()
                # local_logs.append(local_log)
            # -----------------------------

            final_weights += best_weights
            final_rmse += best_rmse.tolist()

            # print("Best Weights:", best_weights)
            # print("Best RMSE:", best_rmse)

            fitness_change.append(np.array(fitness_list))
            weights_change.append(np.array(weights_over_evolutions))

    combined_orbit = combine_solution2(input_orbits, final_weights, delta=delta)

    # fitness_change = np.hstack(fitness_change)
    # weights_change = np.hstack(weights_change)

    print("Best Weights:", final_weights)
    print("Best RMSE:", final_rmse)

    return combined_orbit, final_weights, (None, None, global_logs, local_logs, delta)
