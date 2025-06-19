# -*- coding: utf-8 -*-
"""
produce_results.py
 
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
from datetime import datetime, timedelta
from combine_orbits import combine_arcs
from classes import AccessRequest
from retrieve_arcs import retrieve_arcs
from parsing_utils import save_sp3k

from configuration import paths
from pandas.tseries.offsets import MonthBegin
from gravtools.kinematic_orbits.plotting_functions import plot_orbit_gaps, plot_orbit_residuals

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
                                      round_seconds=True))
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

# %% Modified processing section

methods = [
    # 'mean',
    'inverse_variance',
    'vce',
    # 'residual_weighted',
    # 'nelder_mead',
    # 'cmaes',
]

methods_no_ref = [
    # 'mean',
    'inverse_variance',
    'vce',
]

methods_ref = [
    # 'residual_weighted',
    # 'nelder_mead',
    # 'cmaes',
]


save_methods = [m.replace('_', '') for m in methods]  # Remove all underscores
save_methods_ref = [m.replace('_', '') for m in methods_ref]  # Remove all underscores

sat_id = 'A'  # Preserve satellite ID format in filenames
path= os.path.join(paths["root"], paths["CO"])
path2 = r'output'
# path2 = r'E:\thesis\data'
path = os.path.join(path, path2)

satellite_ids = ['47']
# %%

arclength = pd.Timedelta(days=1)
filtering_threshold = 0.3

startdates = [datetime(year, month, 1) for year in [2023] for month in range(1,2)]
orbit_duration = MonthBegin(1)  # Set orbit duration
# %%

get_data = True
# Generate access requests for KO and RDO data
i1_req = generate_access_requests("KO", startdates, analysis_centre=["IFG", "AIUB", "TUD"], get_data=get_data, orbit_duration=orbit_duration, satellite_ids=satellite_ids)
i2_req = generate_access_requests("RDO", startdates, analysis_centre=["ESA", "IFG"], get_data=get_data, orbit_duration=orbit_duration, satellite_ids=satellite_ids)


# %% Retrieve data
i1 = [retrieve_arcs(i) for i in i1_req]
input_data = [i[satellite_ids[0]] for i in i1]

#%%
i2 = [retrieve_arcs(i) for i in i2_req]
reference_data = [i[satellite_ids[0]] for i in i2]
# %% Process data
output_data_noref = []
output_data_ref = []

output_data2 = []
output_data3 = []
startfrom = 0
count = startfrom
# startdates = [datetime(year, month, 1) for year in [2023] for month in range(9, 13)]  # All months in 2023
run_combine  = 0
if run_combine:
    for input_day, reference_day in zip(input_data[startfrom:], reference_data[startfrom:]):
        # if count <= 3:
        #     count+=1
        #     print('Skipping ', count)
        #     continue
        current_start = startdates[count]
        current_end = current_start + orbit_duration
        print('running')
        if len(reference_day) > 0:
            weekly_orbits_noref = [combine_arcs(input_day, method, reference_data=[reference_day[0]], arclength=arclength, residual_filter_threshold=filtering_threshold)[0][0] for method in methods_no_ref]
        else:
            weekly_orbits_noref = [combine_arcs(input_day, method, reference_data=None, arclength=arclength, residual_filter_threshold=filtering_threshold)[0][0] for method in methods_no_ref]
        
        # Split weekly orbit into daily segments
        daily_orbits_noref = [split_into_daily_chunks(orbit, current_start, current_end) for orbit in weekly_orbits_noref] # noref
        
        for idx, full_orbit in enumerate(daily_orbits_noref):
            
            for daily_orbit in full_orbit:
                # Get date from actual daily data
                try:
                    
                    date = daily_orbit.index[0].to_pydatetime()
                    year = date.year
                    doy = str(date.timetuple().tm_yday).rjust(3, '0')
                    
                    # Save with method-specific name and actual date
                    method_name = save_methods[idx]
                    save_sp3k(daily_orbit,  satellite_ids[0], path,  # Internal ID remains 47
                              f'{method_name}_NONE_{sat_id}_TUD_{year}{doy}')
                except Exception as e:
                    print(f'{method_name}_NONE_{sat_id}_TUD_{year}{doy} FAILED; \n', e)
                    continue
        output_data_noref.append(daily_orbits_noref)
      
        if len(reference_day) > 0:
            weekly_orbits_ref = [combine_arcs(input_day, method, reference_data=[reference_day[0]], arclength=arclength, residual_filter_threshold=filtering_threshold)[0][0] for method in methods_ref]
            # Split weekly orbit into daily segments
            daily_orbits_ref = [split_into_daily_chunks(orbit, current_start, current_end) for orbit in weekly_orbits_ref] # wrt ESA
            
            for idx, full_orbit in enumerate(daily_orbits_ref):
                
                for daily_orbit in full_orbit:
                    # Get date from actual daily data
                    try:
                        date = daily_orbit.index[0].to_pydatetime()
                        year = date.year
                        doy = str(date.timetuple().tm_yday).rjust(3, '0')
                        
                        # Save with method-specific name and actual date
                        
                        method_name = save_methods_ref[idx]
                        save_sp3k(daily_orbit,  satellite_ids[0], path,  # Internal ID remains 47
                                  f'{method_name}_ESA_{sat_id}_TUD_{year}{doy}')
                    except Exception as e:
                        print(f'{method_name}_NONE_{sat_id}_TUD_{year}{doy} FAILED; \n', e)
                        continue
                output_data_ref.append(daily_orbits_ref)
        count+=1

# %% Modified saving logic for input data (KO orbits)
savekin = 1
if savekin:
    kinematic_path= os.path.join(paths["root"], paths["CO"])
    kinematic_path = os.path.join(kinematic_path, path2)
    
    analysis_centre =["IFG", "AIUB", "TUD"] # Original analysis centres for KO data
    # import time
    for idx, weekly_arcs in enumerate(input_data):
        weekly_arcs = [i.trajectory for i in weekly_arcs]
        print('Ref data length: ', len(reference_data[idx]))
        # if len(reference_data[idx]) > 0:
        #     weekly_arcs = filter_arcs_by_residuals(weekly_arcs, reference_data=[reference_data[idx][0].trajectory], threshold=filtering_threshold)
        current_start = startdates[idx]
        current_end = current_start + orbit_duration
        for arc_idx, arc in enumerate(weekly_arcs):
            if not arc.empty:
                # Split weekly arc into daily chunks
                daily_chunks = split_into_daily_chunks(arc, current_start, current_end)
                
                for daily_df in daily_chunks:
                    # Get date from actual daily data
                    date = daily_df.index[0].to_pydatetime()
                    year = date.year
                    doy = str(date.timetuple().tm_yday).rjust(3, '0')
                    starttime = datetime.now()
                    # Maintain filename structure: {centre}_KO_A_TUD_{year}{doy}
                    save_sp3k(daily_df, satellite_ids[0], kinematic_path,
                              f'{analysis_centre[arc_idx]}_KONF_{sat_id}_TUD_{year}{doy}')
                    endtime = datetime.now()
                    print(f'savetime new = {endtime - starttime}')
            
# %% 
saverdo = 1
if saverdo:
    # % Modified saving logic for reference data (RDO orbits)
    # rdo_path= os.path.join(paths["root"], paths["CO"])
    # rdo_path = os.path.join(rdo_path, path2)
    
    rdo_path= r"C:\Users\maber\OneDrive\Documents\Studying\Delft Masters Courses\Thesis\Results\final\Verification\general\ESA_IFG_RDO_comp"
    # path2 = 'final_data/dailyfit_2023'
    
    
    # analysis_centres_ref = ["ESA"]#, 'IFG']  # Only ESA for reference data
    analysis_centres_ref = ["ESA", 'IFG']   # Only ESA for reference data
    
    for idx, weekly_arcs in enumerate(reference_data):
        current_start = startdates[idx]
        current_end = current_start + orbit_duration
        for arc_idx, arc in enumerate(weekly_arcs): 
            # Split weekly arc into daily chunks
            daily_chunks = split_into_daily_chunks(arc.trajectory, current_start, current_end)
            
            for daily_df in daily_chunks:
                # Get date from actual daily data
                date = daily_df.index[0].to_pydatetime()
                year = date.year
                doy = str(date.timetuple().tm_yday).rjust(3, '0')
            
                # Maintain filename structure: {centre}_RDO_A_TUD_{year}{doy}
                save_sp3k(daily_df,  satellite_ids[0], rdo_path,
                          f'{analysis_centres_ref[arc_idx]}_RDO_{sat_id}_TUD_{year}{doy}')