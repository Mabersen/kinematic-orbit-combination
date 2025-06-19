# -*- coding: utf-8 -*-
"""
epochcheck.py
 
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

from pathlib import Path
import numpy as np
from collections import defaultdict

def compute_total_epochs_per_method_and_reference(gap_dir):
    """
    Compute the total number of epochs per method and reference data from all gap files.

    Parameters
    ----------
    gap_dir : str or Path
        Path to the directory containing gap files (*.npz).

    Returns
    -------
    dict
        Nested dictionary with structure {method: {reference_data: total_epochs}}.
    """
    gap_dir = Path(gap_dir)
    total_epochs_per_method_ref = defaultdict(lambda: defaultdict(int))

    for npz_file in gap_dir.glob("*.npz"):
        parts = npz_file.stem.split('_')
        if len(parts) < 3:
            continue
        method = parts[0]
        reference_data = parts[1]

        data = np.load(npz_file, allow_pickle=True)
        total_epochs = data['total_epochs'].item()

        total_epochs_per_method_ref[method][reference_data] += total_epochs

    return total_epochs_per_method_ref


gap_dir = Path(r"normal_points/timescales") # Requires normal point data
total_epochs_per_method_ref = compute_total_epochs_per_method_and_reference(gap_dir)

# Display nicely
for method, ref_dict in total_epochs_per_method_ref.items():
    for ref_data, epochs in ref_dict.items():
        print(f"Method: {method} | Reference Data: {ref_data} | Total Available Epochs: {epochs}")
