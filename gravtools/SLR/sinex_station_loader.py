# -*- coding: utf-8 -*-
"""
sinex_station_loader.py
 
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

class SINEXStationHandler:
    """
    Lightweight SINEX parser for extracting station positions (STAX, STAY, STAZ).
    """

    def __init__(self, sinex_file):
        """
        Initialise and parse the SINEX file.

        Parameters
        ----------
        sinex_file : str
            Path to SINEX file.
        """
        self.df_stations = self._parse_sinex_positions(sinex_file)

    def _parse_sinex_positions(self, sinex_path):
        """
        Parse SOLUTION/ESTIMATE block and extract station XYZ positions.
        """
        with open(sinex_path, 'r') as file:
            lines = file.readlines()

        in_estimate_block = False
        station_estimates = {}

        for line in lines:
            if line.startswith('+SOLUTION/ESTIMATE'):
                in_estimate_block = True
                continue
            if line.startswith('-SOLUTION/ESTIMATE'):
                in_estimate_block = False
                continue

            if in_estimate_block:
                # SINEX fixed format: positions at columns
                if len(line) < 68:
                    continue  # Skip malformed lines
                param_type = line[7:11].strip()
                if param_type in ['STAX', 'STAY', 'STAZ']:
                    station_code = line[14:18].strip()  # Extract numeric code correctly!
                    station_code = str(station_code)
                    try:
                        estimate = float(line[47:68].strip())
                    except ValueError:
                        continue
                    if station_code not in station_estimates:
                        station_estimates[station_code] = {}
                    station_estimates[station_code][param_type] = estimate

        # Build dataframe
        records = []
        for code, params in station_estimates.items():
            if all(k in params for k in ['STAX', 'STAY', 'STAZ']):
                records.append({
                    'Station Code': code,
                    'X [m]': params['STAX'],
                    'Y [m]': params['STAY'],
                    'Z [m]': params['STAZ']
                })

        df = pd.DataFrame(records)
        if df.empty:
            print("Warning: No station positions found in SINEX file.")
        return df

    def get_station_xyz(self, station_code):
        """
        Get XYZ position for a given station code.

        Parameters
        ----------
        station_code : str

        Returns
        -------
        np.array
            XYZ in metres.
        """
        row = self.df_stations[self.df_stations['Station Code'] == station_code]
        if row.empty:
            raise ValueError(f"Station code {station_code} not found in SINEX file.")
        return row[['X [m]', 'Y [m]', 'Z [m]']].iloc[0].to_numpy()

    def list_station_codes(self):
        """
        List all station codes parsed from the SINEX file.
        """
        return self.df_stations['Station Code'].tolist()
    
class MultiSINEXStationHandler_OLD:
    """
    Lightweight handler to collate multiple SINEX files into a single station table.
    """
    def __init__(self, sinex_file_list):
        self.df_stations = self._parse_multiple_sinex_files(sinex_file_list)
        self.station_codes = self.df_stations['Station Code']
        
    def _parse_multiple_sinex_files(self, sinex_file_list):
        all_dfs = []
        for sinex_file in sinex_file_list:
            handler = SINEXStationHandler(sinex_file)
            all_dfs.append(handler.df_stations)
        
        # Concatenate all and keep last occurrence per station code
        combined_df = pd.concat(all_dfs)
        combined_df = combined_df.drop_duplicates(subset='Station Code', keep='last').reset_index(drop=True)
        
        return combined_df

    def get_station_xyz(self, station_code):
        row = self.df_stations[self.df_stations['Station Code'] == str(station_code)]
        if row.empty:
            raise ValueError(f"Station code {station_code} not found in any SINEX file.")
        return row[['X [m]', 'Y [m]', 'Z [m]']].iloc[0].to_numpy()

    def list_station_codes(self):
        return self.df_stations['Station Code'].tolist()

class MultiSINEXStationHandler:
    """
    Lightweight handler to collate multiple SINEX files into a single station table.
    """
    def __init__(self, sinex_file_list):
        self.df_stations = self._parse_multiple_sinex_files(sinex_file_list)
        self.station_codes = self.df_stations['Station Code']
        
    def _parse_multiple_sinex_files(self, sinex_file_list):
        all_dfs = []
        for sinex_file in sinex_file_list:
            handler = SINEXStationHandler(sinex_file)
            all_dfs.append(handler.df_stations)
        
        # Concatenate all and keep last occurrence per station code
        combined_df = pd.concat(all_dfs)
        combined_df = combined_df.drop_duplicates(subset='Station Code', keep='last').reset_index(drop=True)
        
        return combined_df

    def get_station_xyz(self, station_code):
        row = self.df_stations[self.df_stations['Station Code'] == str(station_code)]
        if row.empty:
            raise ValueError(f"Station code {station_code} not found in any SINEX file.")
        return row[['Station Code', 'X [m]', 'Y [m]', 'Z [m]']].iloc[-1]

    def list_station_codes(self):
        return self.df_stations['Station Code'].tolist()