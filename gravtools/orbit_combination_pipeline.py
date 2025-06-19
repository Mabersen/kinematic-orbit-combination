# -*- coding: utf-8 -*-
"""
orbit_combination_pipeline.py
 
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
import yaml
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthBegin
from gravtools.kinematic_orbits.combine_orbits import combine_arcs
from gravtools.kinematic_orbits.classes import AccessRequest
from gravtools.kinematic_orbits.retrieve_arcs import retrieve_arcs
from gravtools.data_utilities.parsing_utils import save_sp3k
from gravtools.configuration import paths, get_path
import pandas as pd


def generate_access_requests(data_type, startdates, analysis_centre, get_data, orbit_duration, satellite_ids):
    return [AccessRequest(data_type=data_type,
                          satellite_id=satellite_ids,
                          analysis_centre=analysis_centre,
                          window_start=start,
                          window_stop=start + orbit_duration,
                          get_data=get_data,
                          round_seconds=True) for start in startdates]

def split_into_daily_chunks(orbit_df, window_start, window_end, freq='D'):
    return [df for _, df in orbit_df.groupby(pd.Grouper(freq=freq))
            if not df[(df.index >= window_start) & (df.index < window_end)].empty]

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def map_sat_id(numeric_id):
    return {'47': 'A', '48': 'B', '49': 'C'}.get(str(numeric_id), 'X')

def main(config):
    satellite_id = config["satellite_id"]
    sat_id = map_sat_id(satellite_id)
    arclength = pd.Timedelta(days=1)
    filtering_threshold = config.get("filtering_threshold", 0.3)

    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    if "days" in config:
        startdates = [start_date + timedelta(days=i) for i in range(config["days"])]
        orbit_duration = timedelta(days=1)
    else:
        startdates = [start_date + MonthBegin(i) for i in range(config["months"])]
        orbit_duration = MonthBegin(1)

    input_reqs = generate_access_requests("KO", startdates, config["input_centres"], True,
                                          orbit_duration, [config["satellite_id"]])
    ref_reqs = generate_access_requests("RDO", startdates, config["reference_centres"], True,
                                        orbit_duration, [config["satellite_id"]])

    input_data = [retrieve_arcs(req)[satellite_id] for req in input_reqs]
    ref_data = [retrieve_arcs(req)[satellite_id] for req in ref_reqs]

    save_methods = ([m.replace('_', '') for m in config["methods"]["no_ref"]]
                    + [m.replace('_', '') for m in config["methods"]["with_ref"]])
    
    if config["output_path"] == '':
        output_path = config["output_path"]
        output_path = os.path.join(output_path, config["satellite_id"])
        
    else:
        output_path = get_path("KO", "CO")
        output_path = os.path.join(output_path, config["satellite_id"])
    for idx, (input_day, ref_day) in enumerate(zip(input_data, ref_data)):
        current_start = startdates[idx]
        current_end = current_start + orbit_duration

        if config["save"]["combined"]:
            if len(ref_day) > 0:
                weekly = [combine_arcs(input_day, m, reference_data=[ref_day[0]], arclength=arclength,
                           residual_filter_threshold=filtering_threshold)[0][0] for m in config["methods"]["no_ref"]]
            else:
                weekly = [combine_arcs(input_day, m, reference_data=None, arclength=arclength,
                           residual_filter_threshold=filtering_threshold)[0][0] for m in config["methods"]["no_ref"]]

            daily = [split_into_daily_chunks(o, current_start, current_end) for o in weekly]
            
            for method_idx, full_orbit in enumerate(daily):
                for daily_orbit in full_orbit:
                    try:
                        date = daily_orbit.index[0].to_pydatetime()
                        year, doy = date.year, str(date.timetuple().tm_yday).rjust(3, '0')
                        method_name = save_methods[method_idx]
                        save_sp3k(daily_orbit, config["satellite_id"], output_path,
                                  f'{method_name}_NONE_{sat_id}_TUD_{year}{doy}')
                    except Exception as e:
                        print(f"Failed to save combined orbit: {e}")

        if config["save"]["kinematic"]:
            output_path = get_path("KO", "INP")
            output_path = os.path.join(output_path, config["satellite_id"])
            centres = config["input_centres"]
            weekly_arcs = [arc.trajectory for arc in input_day]
            for arc_idx, arc in enumerate(weekly_arcs):
                chunks = split_into_daily_chunks(arc, current_start, current_end)
                for daily_df in chunks:
                    date = daily_df.index[0].to_pydatetime()
                    year, doy = date.year, str(date.timetuple().tm_yday).rjust(3, '0')
                    save_sp3k(daily_df, config["satellite_id"], output_path,
                              f'{centres[arc_idx]}_KONF_{sat_id}_TUD_{year}{doy}')

        if config["save"]["reference"]:
            output_path = get_path("RDO", "INP")
            output_path = os.path.join(output_path, config["satellite_id"])
            centres = config["reference_centres"]
            for arc_idx, arc in enumerate(ref_day):
                chunks = split_into_daily_chunks(arc.trajectory, current_start, current_end)
                for daily_df in chunks:
                    date = daily_df.index[0].to_pydatetime()
                    year, doy = date.year, str(date.timetuple().tm_yday).rjust(3, '0')
                    save_sp3k(daily_df, config["satellite_id"], output_path,
                              f'{centres[arc_idx]}_RDO_{sat_id}_TUD_{year}{doy}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Produce orbit results from config")
    parser.add_argument("--config", type=str, default="combination_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
