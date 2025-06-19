# -*- coding: utf-8 -*-
"""
frame_utilities.py
 
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
from astropy.coordinates import SkyCoord, EarthLocation
import numpy as np


def gcrs_to_itrs(df):
    """ Celestial to terrestrial (ECEF) reference frame defined by the ITRS """
    
    pos_df = df[['x_pos', 'y_pos', 'z_pos']]
    # t0 = pd.Series(pos_df.index)
    # pos_df = convert_gps_to_utc(pos_df)
    # t1 = pd.Series(pos_df.index)
    time_array = pos_df.index.to_numpy()
    pos_array = pos_df.to_numpy()
    # return time_array, pos_array

    coord = SkyCoord(x=pos_array[:, 0], y=pos_array[:, 1], z=pos_array[:, 2],
                     obstime=time_array, unit='m', representation_type='cartesian', frame='gcrs')
    coord = coord.transform_to('itrs')

    coord_df = pd.DataFrame(data={"x_pos": coord.x,
                                  "y_pos": coord.y,
                                  "z_pos": coord.z, },
                            index=pos_df.index)

    # coord_df.index = pd.to_datetime(coord_df.index)
    return coord_df

def gcrs_to_itrs2(df):
    """ Celestial to terrestrial (ECEF) reference frame defined by the ITRS """
    
    pos_df = df[['x_pos', 'y_pos', 'z_pos']].astype('float64')
    # t0 = pd.Series(pos_df.index)
    # pos_df = convert_gps_to_utc(pos_df)
    # t1 = pd.Series(pos_df.index)
    time_array = df['datetime'].to_numpy()
    pos_array = pos_df.to_numpy()
    # return time_array, pos_array

    coord = SkyCoord(x=pos_array[:, 0], y=pos_array[:, 1], z=pos_array[:, 2],
                     obstime=time_array, unit='m', representation_type='cartesian', frame='gcrs')
    coord = coord.transform_to('itrs')

    coord_df = pd.DataFrame(data={"x_pos": coord.x,
                                  "y_pos": coord.y,
                                  "z_pos": coord.z, 
                                  "datetime" : df['datetime']},
                            index=pos_df.index)

    # coord_df.index = pd.to_datetime(coord_df.index)
    return coord_df

def itrs_to_gcrs(df):
    """ Terrestrial (ECEF) to Celestial reference frame"""

    pos_df = df[['x_pos', 'y_pos', 'z_pos']]
    # t0 = pd.Series(pos_df.index)
    # pos_df = convert_gps_to_utc(pos_df)
    # t1 = pd.Series(pos_df.index)
    time_array = pos_df.index.to_numpy()
    pos_array = pos_df.to_numpy()
    # return time_array, pos_array

    coord = SkyCoord(x=pos_array[:, 0], y=pos_array[:, 1], z=pos_array[:, 2],
                     obstime=time_array, unit='m', representation_type='cartesian', frame='itrs')
    coord = coord.transform_to('gcrs').cartesian

    coord_df = pd.DataFrame(data={"x_pos": coord.x.value,
                                  "y_pos": coord.y.value,
                                  "z_pos": coord.z.value, },
                            index=pos_df.index)

    # coord_df.index = pd.to_datetime(coord_df.index)
    return coord_df


def retrieve_observatories_in_terrestrial():
    sites = EarthLocation.get_site_names()

    sites = sites[:]
    coords = [EarthLocation.of_site(i).get_itrs().cartesian.xyz.value for i in sites]
    # coords = [EarthLocation.of_site(i) for i in sites]

    # observatory_coordinates = [SkyCoord(x=position[0], y=position[1], z=position[2], obstime=np.datetime64(
    #     '2020-02-01'), unit='m', representation_type='cartesian', frame='gcrs').transform_to('itrs') for position in coords]
    # observatory_coordinates = [i.get_itrs(
    #     obstime=np.datetime64('2020-02-01')).cartesian.xyz.value for i in sites]

    return sites, coords  # , observatory_coordinates

