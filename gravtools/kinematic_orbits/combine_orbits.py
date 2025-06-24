# -*- coding: utf-8 -*-
"""
combine_orbits.py
 
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

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gravtools.kinematic_orbits.retrieve_arcs import retrieve_arcs
from gravtools.kinematic_orbits.classes import AccessRequest
from gravtools.kinematic_orbits.plotting_functions import plot_rmse_vs_combined_orbit
from gravtools.kinematic_orbits.kinematic_utilities import (split_data_into_arcs_by_time, combine_arcs_into_dataframe,
                                                            filter_arcs_by_residuals)
from gravtools.kinematic_orbits.pygmo_optimiser_independent import optimise_combined_orbit_independent
from gravtools.kinematic_orbits.pygmo_optimiser import optimise_combined_orbit_pygmo

def arithmetic_mean(dataframes, index=None):
    if index is None:
        index = dataframes[0].index
    combined_orbit = pd.DataFrame(index=index)
    n = len(dataframes)
    
    # Position columns: simple average
    for pos_col in ['x_pos', 'y_pos', 'z_pos']:
        arrays = [df[pos_col] for df in dataframes]
        combined_orbit[pos_col] = np.nanmean(arrays, axis=0)
    
    # Standard deviations: sqrt(Σ(σ_i²) / k²
    for std_col in ['std_x', 'std_y', 'std_z']:
        std_arrays = pd.concat([df[std_col] for df in dataframes], axis=1)
        valid_counts = std_arrays.notna().sum(axis=1)
        combined_var = (std_arrays ** 2).sum(axis=1, skipna=True) / (valid_counts ** 2).replace(0, np.nan)
        combined_orbit[std_col] = np.sqrt(combined_var)
    
    # Covariances: Σ(cov_ij) / k²
    for cov_col in ['cov_xy', 'cov_xz', 'cov_yz']:
        cov_arrays = pd.concat([df[cov_col] for df in dataframes], axis=1)
        valid_counts = cov_arrays.notna().sum(axis=1)
        combined_orbit[cov_col] = cov_arrays.sum(axis=1, skipna=True) / (valid_counts ** 2).replace(0, np.nan)
    return combined_orbit

def inverse_variance(dataframes, normalise=False):
    """Combine orbits using inverse variance weighting with proper NaN handling."""
    # Union of all indices
    index = dataframes[0].index
    for df in dataframes[1:]:
        index = index.union(df.index)
    index = index.sort_values().unique()
    
    dataframes = [df.reindex(index) for df in dataframes]
    
    # Normalization logic (optional)
    if normalise:
        norm_factors = {}
        for col in ['x', 'y', 'z']:
            variances = [df[f'std_{col}']**2 for df in dataframes]
            global_mean_var = np.nanmean([v.mean() for v in variances])
            norm_factors[col] = 1/global_mean_var if global_mean_var != 0 else 1
            
        for df in dataframes:
            for col in ['x', 'y', 'z']:
                df[f'std_{col}'] *= np.sqrt(norm_factors[col])
    
    # Initialize combined dataframe
    combined = pd.DataFrame(index=index)
    weights = {}
    
    # Position columns (x, y, z)
    for coord in ['x', 'y', 'z']:
        pos_col = f'{coord}_pos'
        std_col = f'std_{coord}'
        
        # Compute inverse variances with validity checks
        inv_vars = []
        valid_masks = []
        for df in dataframes:
            # Create validity mask (position not NaN AND std > 0)
            pos_valid = df[pos_col].notna()
            std_valid = (df[std_col] > 0) & df[std_col].notna()
            valid_mask = pos_valid & std_valid
            
            # Calculate inverse variance with validity masking
            inv_var = pd.Series(np.zeros(len(df)), index=df.index)
            inv_var[valid_mask] = 1 / (df[std_col][valid_mask]**2)
            inv_vars.append(inv_var)
            valid_masks.append(valid_mask)
        
        total_inv_var = sum(inv_vars).replace(0, np.inf)
        
        # Calculate weights with zero for invalid entries
        weights[coord] = []
        for iv, mask in zip(inv_vars, valid_masks):
            weight = pd.Series(np.zeros(len(iv)), index=iv.index)
            weight[mask] = iv[mask] / total_inv_var[mask]
            weights[coord].append(weight)
        
        # Combine positions with NaN protection
        pos_sum = pd.Series(0, index=index)
        for df, weight, mask in zip(dataframes, weights[coord], valid_masks):
            # Use cleaned positions (0 where invalid)
            clean_pos = df[pos_col].copy()
            clean_pos[~mask] = 0
            pos_sum += clean_pos * weight
        combined[pos_col] = pos_sum
        
        # Combine standard deviations
        combined[std_col] = np.sqrt(1 / total_inv_var.replace(0, np.inf))
    
    # Covariance columns (xy, xz, yz)
    for cov_pair in ['xy', 'xz', 'yz']:
        cov_col = f'cov_{cov_pair}'
        c1, c2 = list(cov_pair)
        pos_col1, pos_col2 = f'{c1}_pos', f'{c2}_pos'
        
        # Calculate combined covariance
        cov_combined = pd.Series(0, index=index)
        for i, df in enumerate(dataframes):
            # Check validity for both coordinates
            # valid_mask = valid_masks[i] & valid_masks[i]  # Already computed
            valid_mask = valid_masks[['x', 'y', 'z'].index(c1)][i] & valid_masks[['x', 'y', 'z'].index(c2)][i] # Must be tested
            w_product = weights[c1][i] * weights[c2][i]
            
            # Clean covariance values
            clean_cov = df[cov_col].copy()
            clean_cov[~valid_mask] = 0
            cov_combined += clean_cov * w_product
        
        combined[cov_col] = cov_combined
    
    return combined

def variance_component_estimation_jane(dataframes, parameters=['x_pos', 'y_pos', 'z_pos'], 
                                           tol=1e-5, max_iter=100, normalise=False,
                                           vc_floor=1e-5, damping=0.7):
    """Fixed VCE implementation with numerical stability improvements."""
    # 1. Data preparation
    index = dataframes[0].index
    for df in dataframes[1:]:
        index = index.union(df.index)
    index = index.sort_values().unique()
    
    dataframes = [df.reindex(index).groupby(level=0).first() for df in dataframes]
    
    # 2. Initialize storage
    combined_df = pd.DataFrame(index=index)
    vc_weights = {'x': [], 'y': [], 'z': []}

    n = len(dataframes)
    
    # Main estimation loop per coordinate
    for coord in ['x', 'y', 'z']:
        pos_col = f'{coord}_pos'
        std_col = f'std_{coord}'
        
        data_df = pd.concat([df[pos_col] for df in dataframes], axis=1)
        std_df = pd.concat([df[std_col] for df in dataframes], axis=1)

        # Precompute valid masks
        valid_masks = []
        for i in range(n):
            mask = data_df.iloc[:, i].notna() & (std_df.iloc[:, i] > 0)
            valid_masks.append(mask.astype(bool))

        # 3. Initialize variance components with regularization
        vc = np.ones(n)  # Don't normalize initial components
        last_vc = vc.copy()
        
        for iteration in range(max_iter):
            # 4. Compute weights with variance components
            weights = pd.DataFrame(0.0, index=index, columns=range(n))
            std_weights = pd.DataFrame(index=index, columns=range(n))
            for i in range(n):
                weights[i] = np.where(valid_masks[i],
                                    1 / (vc[i] * std_df.iloc[:, i]**2 + vc_floor),
                                    0)
                std_weights[i] = np.where(valid_masks[i],
                                    1 / (std_df.iloc[:, i]**2),
                                    0)


            # 5. Compute combined estimate (BLUE)
            total_weight = weights.sum(axis=1).replace(0, np.inf)
            combined = (data_df * weights).sum(axis=1) / total_weight

            
            new_vc = []
            for i in range(n):
                # Get valid epochs for this input
                valid_mask = ~data_df.iloc[:,i].isna().to_numpy()
                
                res = (combined[valid_mask] - data_df.iloc[valid_mask,i])
                # numer = (res**2).sum()
                numer = res.T @ (std_weights.loc[valid_mask].iloc[:,i] * res)
                denom = valid_mask.sum() * ( 1 - ((1/vc[i])/((1/np.array(vc)).sum())))
                new_vc.append(numer/denom if denom !=0 else vc[i])
            
            
            new_vc = np.array(new_vc)
            # 7. Apply damping and regularization
            # print(vc)
            # print(new_vc)
            # new_vc = np.maximum(new_vc, vc_floor)
            new_vc = damping * new_vc + (1-damping) * last_vc  # Damping
            delta = np.linalg.norm(new_vc - vc)
            # print(delta)

            if delta < tol:
                print(f"Converged after {iteration + 1} iterations.")
                print(f"Final variance components are {vc}")
                vc_weights[coord] = new_vc
                break
            
            vc = new_vc
        # 8. Store final variance components
        vc_weights[coord] = vc
        
        # 9. Final combined values
        weights = pd.DataFrame(0.0, index=index, columns=range(n))
        for i in range(n):
            weights[i] = np.where(valid_masks[i],
                                 1 / (vc[i] * std_df.iloc[:, i]**2),
                                 0)
        
        total_weight = weights.sum(axis=1).replace(0, np.inf)
        combined_df[pos_col] = (data_df * weights).sum(axis=1) / total_weight
        combined_df[std_col] = np.sqrt(1 / total_weight)

    # 10. Proper covariance calculation
    cov_map = {'cov_xy': ('x','y'), 'cov_xz': ('x','z'), 'cov_yz': ('y','z')}
    for cov_col, (c1, c2) in cov_map.items():
        cov_combined = pd.Series(0.0, index=index)
        total_weight = pd.Series(0.0, index=index)
        
        for i, df in enumerate(dataframes):
            mask = valid_masks[i] & df[cov_col].notna()
            if not mask.any():
                continue
                
            # Correct covariance weighting
            vc1 = vc_weights[c1][i]
            vc2 = vc_weights[c2][i]
            
            # weight = 1 / (vc1 * vc2 + vc_floor)
            std_x_sq = df[f'std_{c1}']**2
            std_y_sq = df[f'std_{c2}']**2
            
            # weight = 1 / (vc1 * std_x_sq * vc2 * std_y_sq + vc_floor)
            weight = 1 / np.maximum(vc1 * std_x_sq + vc2 * std_y_sq + vc_floor, vc_floor)
            
            cov_combined += df[cov_col] * weight * mask
            total_weight += weight * mask
        
        combined_df[cov_col] = cov_combined / total_weight.replace(0, 1)

    return combined_df, vc_weights

def residual_weighted_average(dataframes, reference_df, epsilon=1e-7):
    """Combine orbits with proper covariance handling and missing epoch support."""
    index = dataframes[0].index
    combined = pd.DataFrame(index=index)
    weights = {}
    valid_masks = {}

    # Process reference data
    if len(reference_df) > 1:
        aligned_dfs = [df[['x_pos', 'y_pos', 'z_pos']].reindex(index) for df in reference_df]
        stacked = np.dstack([df.values for df in aligned_dfs])
        mean_values = np.nanmean(stacked, axis=2)
        reference_df = pd.DataFrame(mean_values, index=index, columns=['x_pos', 'y_pos', 'z_pos'])
    else:
        reference_df = reference_df[0].reindex(index)

    for coord in ['x', 'y', 'z']:
        pos_col = f'{coord}_pos'
        std_col = f'std_{coord}'
        valid_masks[coord] = []
        
        # Calculate residuals and weights
        residuals = []
        for df in dataframes:
            res = (df[pos_col] - reference_df[pos_col])**2
            residuals.append(res)
            valid_masks[coord].append(df[pos_col].notna())  # Track valid positions

        inv_residuals = [1/(res + epsilon) for res in residuals]
        
        # Calculate weights with zero for missing positions
        total_inv_res = sum([inv_res.where(mask, 0) for inv_res, mask in zip(inv_residuals, valid_masks[coord])])
        total_inv_res[total_inv_res == 0] = np.nan
        
        weights[coord] = [
            (inv_res / total_inv_res).where(mask, 0).fillna(0)
            for inv_res, mask in zip(inv_residuals, valid_masks[coord])
        ]

        # Combine positions
        pos_combined = sum(
            df[pos_col].fillna(0) * weight 
            for df, weight in zip(dataframes, weights[coord])
        )
        combined[pos_col] = pos_combined.where(total_inv_res > 0, np.nan)

        # Combine standard deviations
        std_combined = sum(
            (df[std_col].fillna(0))**2 * weight ** 2
            for df, weight in zip(dataframes, weights[coord])
        )
        combined[std_col] = np.sqrt(std_combined).where(total_inv_res > 0, np.nan)

    # Fixed covariance handling with proper mask access
    cov_map = {'cov_xy': ('x','y'), 'cov_xz': ('x','z'), 'cov_yz': ('y','z')}
    for cov_col, (c1, c2) in cov_map.items():
        cov_combined = pd.Series(0, index=index)
        total_weight = pd.Series(0, index=index)
        
        for i, df in enumerate(dataframes):
            # Use dictionary to access masks by coordinate
            valid_mask = valid_masks[c1][i] & valid_masks[c2][i]
            w = weights[c1][i] * weights[c2][i]
            
            cov_combined += df[cov_col].fillna(0) * w * valid_mask
            total_weight += w * valid_mask
        
        combined[cov_col] = (cov_combined / total_weight.replace(0, np.nan)).where(total_weight > 0, np.nan)
    combined = combined.dropna(how='any')
    return combined


def hybrid_weighted_average(dataframes, reference_df, epsilon=1e-7):
    """Combine orbits using residual weighting where reference exists and inverse-variance elsewhere."""
    # Create unified index
    index = dataframes[0].index
    
    # Process reference data
    if len(reference_df) > 1:
        aligned_dfs = [df[['x_pos', 'y_pos', 'z_pos']].reindex(index) for df in reference_df]
        stacked = np.dstack([df.values for df in aligned_dfs])
        mean_values = np.nanmean(stacked, axis=2)
        reference_df = pd.DataFrame(mean_values, index=index, columns=['x_pos', 'y_pos', 'z_pos'])
    else:
        reference_df = reference_df[0]
    
    combined = pd.DataFrame(index=index)
    
    # Align all dataframes to unified index
    dataframes = [df.reindex(index) for df in dataframes]
    reference_df = reference_df.reindex(index)

    # Find epochs with valid reference data (all coordinates present)
    ref_mask = reference_df[['x_pos', 'y_pos', 'z_pos']].notna().all(axis=1)

    # Split into reference and non-reference epochs
    ref_data = [df[ref_mask] for df in dataframes]
    non_ref_data = [df[~ref_mask] for df in dataframes]
    ref_reference = reference_df[ref_mask]

    # Process each subset separately
    if not ref_reference.empty:
        combined_ref = residual_weighted_average(ref_data, [ref_reference], epsilon)
    else:
        combined_ref = pd.DataFrame()

    if any(not df.empty for df in non_ref_data):
        combined_non_ref = inverse_variance(non_ref_data)
    else:
        combined_non_ref = pd.DataFrame()

    # Combine results using index-aware concatenation
    combined = pd.concat([combined_ref, combined_non_ref]).sort_index()
    
    # Remove duplicate indices that might occur at boundary epochs
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined

def combine_orbits(dataframes: list,
                   method: str,
                   max_itr,
                   include_shift_scalar,
                   include_weights_constraint,
                   pygmo_variables,
                   reference_data: list = None,
                   ):

    if len(pygmo_variables) <= 2:
        pygmo_variables.append(None)  # append standard seed
    if len(pygmo_variables) <= 3:
        pygmo_variables.append(150)  # append standard pop size

    # print(method)
    weights = None
    extra_vars = None
    combined_orbit = None
    # Reindexing arcs to consolidate data.
    dataframes = [i.copy() for i in dataframes]
    
    # Compute union of indices
    index = dataframes[0].index
    for df in dataframes[1:]:
        index = index.union(df.index)
    index = index.sort_values().unique()

    # Reindex all dataframes to union index
    dataframes = [df.reindex(index) for df in dataframes]
    

    if reference_data is not None and (method not in ['mean', 'inverse_variance', 'inverse_variance_normalised', 'vce', 'vce_normalised']):
        reference_data = [i.copy() for i in reference_data]
        reference_data = [df.reindex(index) for df in reference_data]

    if method == 'mean':
        combined_orbit = arithmetic_mean(dataframes, index)

    if method == 'inverse_variance':
        combined_orbit = inverse_variance(dataframes)

    if method == 'inverse_variance_normalised':
        combined_orbit = inverse_variance(dataframes, normalise=True)

    if method == 'vce':
        combined_orbit, weights = variance_component_estimation_jane(dataframes)

    if method == 'vce_normalised':
        combined_orbit, weights = variance_component_estimation_jane(dataframes, normalise=True)
    
    # Add new method handler
    if method == 'residual_weighted':
        if not reference_data:
            raise ValueError("Reference data required for residual-weighted method.")
 
        combined_orbit = hybrid_weighted_average(dataframes, reference_data)
    
    if method == 'nelder_mead':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'COBYLA':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    # ---------------------------------   GACO   --------------------------------------------

    if method == 'gaco':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'gaco_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'gaco_individual':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'gaco_individual_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    # ---------------------------------   DE1220   --------------------------------------------

    if method == 'de1220':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'de1220_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'de1220_individual':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'de1220_individual_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    # ---------------------------------   CMAES   --------------------------------------------

    if method == 'cmaes':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'cmaes_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'cmaes_individual':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'cmaes_individual_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    # ---------------------------------   XNES  --------------------------------------------

    if method == 'xnes':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'xnes_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_pygmo(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'xnes_individual':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], local_optimisation=False, fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])

    if method == 'xnes_individual_LO':
        combined_orbit, weights, extra_vars = optimise_combined_orbit_independent(
            input_orbits=dataframes, validation_data=reference_data, algorithm=method, no_evolutions=pygmo_variables[0],
            no_generations=pygmo_variables[1], fixed_seed=pygmo_variables[2], pop_size=pygmo_variables[3])
    
    
    return combined_orbit, weights, extra_vars


def combine_arcs(arcs: list,
                 method: str,
                 arclength: pd.Timedelta = None,
                 reference_data: list = None,
                 max_itr: int = 50,
                 include_shift_scalar=True,
                 include_weights_constraint=True,
                 pygmo_variables=[1, 20],
                 residual_filter_threshold = None # Float or None
                 ):


    input_data_in = [i.trajectory for i in arcs]

    input_data = []
    for df in input_data_in:
        # Force stds to be positive and above floor
        for col in ['std_x', 'std_y', 'std_z']:
            df[col] = df[col].abs()
        input_data.append(df)
        
        
    if reference_data is not None:
        reference_data = [i.trajectory for i in reference_data]
        # --- New Preprocessing Step ---
        if residual_filter_threshold is not None:
            # Apply residual-based filtering
            input_data = filter_arcs_by_residuals(input_data, reference_data, residual_filter_threshold)
    
    if (arclength is not None) and (method not in ['mean', 'inverse_variance', 'inverse_variance_normalised']):
        split_input_data = split_data_into_arcs_by_time(
            input_dataframes=input_data, arc_duration=arclength)[0]

        if reference_data is not None and (method not in ['vce', 'vce_normalised']):

            combined_orbits = [combine_orbits(dataframes=values,
                                              method=method,
                                              reference_data=reference_data,
                                              max_itr=max_itr,
                                              include_shift_scalar=include_shift_scalar,
                                              include_weights_constraint=include_weights_constraint,
                                              pygmo_variables=pygmo_variables)
                               for values in split_input_data.values()]

            combined_orbits_dict = {key: [i[0]] for key, i in zip(split_input_data.keys(), combined_orbits)}

            weights = [i[1] for i in combined_orbits]
            evals = [i[2] for i in combined_orbits]
        else:
            combined_orbits = [combine_orbits(dataframes=values,
                                              method=method,
                                              max_itr=max_itr,
                                              include_shift_scalar=include_shift_scalar,
                                              include_weights_constraint=include_weights_constraint,
                                              pygmo_variables=pygmo_variables)
                               for values in split_input_data.values()]

            combined_orbits_dict = {key: [i[0]] for key, i in zip(split_input_data.keys(), combined_orbits)}
            weights = [i[1] for i in combined_orbits]
            evals = [i[2] for i in combined_orbits]
        
        final_orbit = combine_arcs_into_dataframe(combined_orbits_dict)

    else:
        if reference_data is not None:
            combined_orbits = [combine_orbits(dataframes=input_data,
                                              method=method,
                                              reference_data=reference_data,
                                              max_itr=max_itr,
                                              include_shift_scalar=include_shift_scalar,
                                              include_weights_constraint=include_weights_constraint,
                                              pygmo_variables=pygmo_variables)]
            weights = [i[1] for i in combined_orbits]
            evals = [i[2] for i in combined_orbits]
        else:
            combined_orbits = [combine_orbits(dataframes=input_data,
                                              method=method,
                                              max_itr=max_itr,
                                              include_shift_scalar=include_shift_scalar,
                                              include_weights_constraint=include_weights_constraint,
                                              pygmo_variables=pygmo_variables)]

            weights = [i[1] for i in combined_orbits]
            evals = [i[2] for i in combined_orbits]

        final_orbit = [i[0] for i in combined_orbits]

    return final_orbit, weights, evals
