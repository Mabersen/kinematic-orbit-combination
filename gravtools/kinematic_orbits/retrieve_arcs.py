# -*- coding: utf-8 -*-
"""
retrieve_arcs.py
 
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
from gravtools.kinematic_orbits.classes import AccessRequest, Arc
from datetime import datetime, timedelta
from gravtools.data_utilities.parsing_utils import parse_orbit_file
from gravtools.data_utilities.retrieve_data import get_data
from gravtools.tudat_utilities import resample_with_gap_handling, resample_without_gap_handling
from gravtools.configuration import paths, get_path


def retrieve_arcs(request: AccessRequest) -> [Arc]:
    """
    Request arc defined by AccessRequest object.

    Parameters
    ----------
    request : AccessRequest
        AccessRequest object containing the definition of the request.

    Returns
    -------
    Arc
        The Specified Arc.

    """
    path_dict = paths[request.data_type]
    analysis_centres = request.analysis_centre
    if 'all' in analysis_centres:
        analysis_centres = list(path_dict.keys())

    if request.get_data:
        print('Getting data')
        get_data(request)

    output = {i: [] for i in request.satellite_id}
    # Loop through analysis centres.
    for ac in analysis_centres:
        
        if request.data_type not in ['CO', 'CORS', 'CORR', 'TEST']:
            path = get_path(request.data_type, ac)
        else:
            path = get_path(request.data_type)
            
        # print(ac, path)
        if '_NONE_' in path:
            print('No path')
            continue
        # Loop through requested satellites.
        for sat_id in request.satellite_id:
            sub_path = os.path.join(path, sat_id)
            # Read files from folder if in selected analysis centre or if all analysis centres specified.
            files = os.listdir(sub_path)
            # Filter out desired datatype, replace any backslashes with forward slashes for consistency.
            # if 'sp3' in request.data_format:
            # files_sp3 = [i.replace('\\', '/') for i in files if 'sp3' in i]
            # files_txt = [i.replace('\\', '/') for i in files if 'txt' in i]

            # files = files_sp3 + files_txt
            # else:
            files = [i.replace('\\', '/') for i in files]
            if 'KO' in request.data_type:
                files = [i.replace('\\', '/') for i in files if 'sp3' in i]

            if 'RDO' in request.data_type:
                if 'IFG' in ac:
                    files = [i.replace('\\', '/') for i in files if 'txt' in i]
                if 'ESA' in ac:
                    files = [i.replace('\\', '/') for i in files if 'sp3' in i]

            # print(len(files))
            if len(files) < 1:
                continue
            
            filtered_files = []
            for file in files:
                try:
                    if request.data_type == 'KO':
                        # Handle KO filename format: GSWARM_KO_SA_IFG_2023-02-07_...
                        splitname = file.split('_')
                        date_str = splitname[4]
                        analysis_centre_check = splitname[3]
                        if ac != analysis_centre_check:
                            print(file, ' skipped')
                            continue
                        
                        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        window_start_date = request.window_start.date() - timedelta(minutes=10)
                        window_stop_date = request.window_stop.date() + timedelta(minutes=10)
                        
                        if window_start_date <= file_date <= window_stop_date:
                            filtered_files.append(file)
                        else:
                            pass
                            # print(f'Skipping KO file {file} [date {file_date} outside window]')
                            
                    elif (request.data_type == 'RDO' and 'ESA' in ac):
                        # Handle RDO filename format: SW_OPER_SP3ACOM_2__20250109T235942_20250110T235942_0201
                        parts = file.split('__')
                        if len(parts) < 2:
                            raise ValueError("Missing time segment in filename")
                            
                        time_parts = parts[1].split('_')
                        if len(time_parts) < 2:
                            raise ValueError("Invalid time format in filename")
                            
                        # Parse start and end times from filename
                        file_start = datetime.strptime(time_parts[0], "%Y%m%dT%H%M%S") - timedelta(minutes=10)
                        file_end = datetime.strptime(time_parts[1], "%Y%m%dT%H%M%S") + timedelta(minutes=10)
                        
                        # Check temporal overlap with request window
                        if (file_start <= request.window_stop) and (file_end >= request.window_start):
                            filtered_files.append(file)
                        else:
                            pass
                            # print(f'Skipping RDO file {file} [{file_start}-{file_end} outside window]')
                    elif (request.data_type == 'RDO' and 'IFG' in ac):

                        parts = file.split('_')
                        time_parts = parts[-1].split('.')[0]
                        file_start = datetime.strptime(time_parts, '%Y-%m-%d')  - timedelta(minutes=10)
                        file_end = file_start + timedelta(days=1) + timedelta(minutes=20)
                        
                        # Check temporal overlap with request window
                        if (file_start <= request.window_stop) and (file_end >= request.window_start):

                            filtered_files.append(file)
                        else:
                            pass
                    elif request.data_type in ['CO', 'CORS', 'CORR', 'TEST']:
                        date_str = file.split('_')[5].split('.')[0]
                        method = file.split('_')[1]
                        if method in ac:
                            file_date = datetime.strptime(date_str, "%Y%j").date()
                            window_start_date = request.window_start.date() - timedelta(minutes=10)
                            window_stop_date = request.window_stop.date() + timedelta(minutes=10)
                            
                            if window_start_date <= file_date <= window_stop_date:
                                filtered_files.append(file)
                            else:
                                pass
                        else:
                            pass
                            # print(f'Skipping KO file {file} [date {file_date} outside window]')
                        
                except (IndexError, ValueError) as e:
                    print(f'Skipping {file} [invalid filename format: {e}]')
                    continue

            if not filtered_files:
                continue
            
            data = []
            
            
            for file in filtered_files:
                # Join paths
                filepath = os.path.join(sub_path, file)
                # try:
                print(f'Loading file: {filepath}')
                parsed_file = parse_orbit_file(file_path=filepath,
                                               window_start=request.window_start,
                                               window_stop=request.window_stop,
                                               data_type=request.data_type,
                                               # data_format=request.data_format,
                                               analysis_centre=ac)

                if parsed_file is not False:
                    if not parsed_file.empty:
                        
                        if request.round_seconds:
                            test=False
                            
                            # Last minute tests:
                            # dataframe_index = parsed_file.index.round('1s')  # Rounding for server implementation
                            # parsed_file = parsed_file
                            
                            if 'KO' in request.data_type and not test:
                                # Resample the data
                                parsed_file = resample_with_gap_handling(original_df=parsed_file, frequency='1s', gap_threshold_seconds=5)
                                # parsed_file = resample_without_gap_handling(original_df=parsed_file, frequency='1s') # TO TEST NO GAP HANDLING
                                
                                # dataframe_index = parsed_file.index.round('1s')  # Rounding for server implementation
                                # parsed_file = parsed_file.set_index(dataframe_index)
                                # pass
                            elif 'RDO' in request.data_type and not test:
                                parsed_file = resample_with_gap_handling(original_df=parsed_file, frequency='1s',gap_threshold_seconds=20) # resampling
                        
                            else:
                                print('rounding')
                                dataframe_index = parsed_file.index.round('1s')  # Rounding data to the nearest whole second
                                parsed_file = parsed_file.set_index(dataframe_index)
                                         
                        data.append(parsed_file)
                else:
                    pass

            # Concatenate data.
            trajectory_dataframes = data
            try:
                concatenated_trajectories = pd.concat(trajectory_dataframes, ignore_index=False)
                # Filter duplicate index values
                concatenated_trajectories = concatenated_trajectories.groupby(level=0).first()
                # Create arc object with combined trajectory.
                concatenated_arc = Arc(trajectory=concatenated_trajectories,
                                       analysis_centre=ac,
                                       satellite_id=sat_id,
                                       data_type=request.data_type)
                output[sat_id].append(concatenated_arc)
            except ValueError:
                # output[sat_id].append([])
                continue
    return output


if __name__ == '__main__':

    # req = AccessRequest(data_type='KO',
    #                               satellite_id=["47"],
    #                               analysis_centre=['AIUB', 'IFG', 'TUD'],
    #                               window_start=datetime(2023,1,1),
    #                               window_stop=datetime(2023,1,2),
    #                               get_data=False,
    #                               round_seconds=True)
    # parsed1 = retrieve_arcs(req)['47']
    # parsed1 = [i.trajectory for i in parsed1]
    
    req = AccessRequest(data_type='RDO',
                                  satellite_id=["47"],
                                  analysis_centre=['IFG'],
                                  window_start=datetime(2023,1,1),
                                  window_stop=datetime(2023,1,2),
                                  get_data=False,
                                  round_seconds=True)
    parsed2 = retrieve_arcs(req)['47']
    parsed2 = [i.trajectory for i in parsed2]