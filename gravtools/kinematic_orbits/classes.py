# -*- coding: utf-8 -*-.
"""
classes.py
 
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
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from astropy.time import Time


@dataclass
class AccessRequest:
    """
    Access request container, window is up until (not including) t1.

    Attributes
    ----------
    data_type:
        Type of data requested; KO, RDO, GFM, etc.
    satellite_id : str
        Id of the satellite/s.
    Analysis centre: str
        List of analysis centres.
    window_start : datetime
        Datetime of t0.
    window_stop : datetime
        Datetime of t1.

    """

    data_type: str
    satellite_id: list
    analysis_centre: list
    window_start: datetime
    window_stop: datetime
    get_data: bool = False
    round_seconds: bool = True
    resample: bool = False


class Arc:
    """Define dataclass for arc data."""

    def __init__(self, trajectory: pd.DataFrame,
                 analysis_centre: str = None,
                 satellite_id: str = None,
                 data_type: str = None
                 ):
        self.trajectory = trajectory
        self.analysis_centre = analysis_centre
        self.satellite_id = satellite_id
        self.data_type = data_type

    def convert_gps_to_utc(self, time_column='datetime'):
        """
        Convert GPS time to UTC in the given pandas DataFrame.

        Parameters
        ----------
        time_column : TYPE, optional
            Name of the column containing the GPS time (default is 'datetime').

        Returns
        -------
        pd.DataFrame
            DataFrame with an additional column containing the UTC times.

        """
        # Convert the pandas datetime column (assumed to be in GPS time) to ISO format strings'
        df = self.trajectory
        df = df.shift(+19, 's')  # convert from GPS to TAI (GPS is always 19 seconds behind)
        gps_times = list(pd.Series(df.index).dt.strftime('%Y-%m-%dT%H:%M:%S.%f').to_numpy())

        # gps_times = df[time_column]
        # gps_times = gps_times[time_column].to_numpy()
        # Create an astropy Time object in TAI format
        gps_astropy_time = Time(gps_times, format='isot', scale='tai', precision=9)

        # Convert to UTC
        utc_astropy_time = gps_astropy_time.utc

        # Add the UTC times back to the dataframe as a new column
        df['datetime'] = utc_astropy_time.to_datetime()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(df['datetime'])
        df = df.drop(columns=['datetime'])
        return df

    def convert_utc_to_gps(self, time_column='datetime'):
        """
        Convert UTC time to GPS in the given pandas DataFrame.

        Parameters
        ----------
        time_column : TYPE, optional
            Name of the column containing the UTC time (default is 'datetime').

        Returns
        -------
        pd.DataFrame
            DataFrame with an additional column containing the UTC times.
        """
        # Convert the pandas datetime column (assumed to be in GPS time) to ISO format strings'
        df = self.trajectory

        utc_times = list(pd.Series(df.index).dt.strftime('%Y-%m-%dT%H:%M:%S.%f').to_numpy())

        # gps_times = df[time_column]
        # gps_times = gps_times[time_column].to_numpy()
        # Create an astropy Time object in UTC format
        utc_astropy_time = Time(utc_times, format='isot', scale='utc', precision=9)

        # Convert to TAI
        tai_astropy_time = utc_astropy_time.tai

        # Add the UTC times back to the dataframe as a new column
        df['datetime'] = tai_astropy_time.to_datetime()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(df['datetime'])
        df = df.drop(columns=['datetime'])
        df = df.shift(-19, 's')  # convert from TAI to GPS (GPS is always 19 seconds behind)
        return df

    def segment_arc(self, time_boundaries=None, n_segments=None, freq=None):
        """
        Split trajectory dataframe into arcs.

        Splits a dataframe into a list of dataframes based on time boundaries, equally spaced intervals,
        or a specific frequency (e.g., every second, minute, etc.). Ensures no duplicate epochs between segments.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to split. It should have a datetime index.

        time_boundaries : list of pd.Timestamp or str, optional
            A list of timestamps defining the boundaries for splitting the dataframe.
            If provided, the function will split based on these times.

        n_segments : int, optional
            Number of equally spaced segments. If provided, the dataframe will be
            split into 'n_segments' equal intervals.

        freq : str, optional
            A string representing the frequency to split the dataframe by (e.g., 'S' for seconds, 'T' for minutes).

        Returns
        -------
        list of pd.DataFrame
            A list of dataframe subsets for each defined interval or segment.

        Raises
        ------
        ValueError:
            If more than one of `time_boundaries`, `n_segments`, or `freq` is provided.
        """
        df = self.trajectory
        if sum([time_boundaries is not None, n_segments is not None, freq is not None]) > 1:
            raise ValueError("Provide only one of 'time_boundaries', 'n_segments', or 'freq'.")

        if time_boundaries is not None:
            # Ensure the time_boundaries are valid timestamps
            time_boundaries = pd.to_datetime(time_boundaries)
            split_dfs = []

            for i in range(len(time_boundaries) - 1):
                start = time_boundaries[i]
                end = time_boundaries[i + 1]

                # Get the dataframe slice between the boundaries, but exclude the end of the range
                segment = df.loc[start:end].iloc[:-1]  # Exclude the end boundary from this segment
                if not segment.empty:
                    split_dfs.append(segment)

            # Add the final segment including the last boundary
            final_segment = df.loc[time_boundaries[-2]:time_boundaries[-1]]
            if not final_segment.empty:
                split_dfs.append(final_segment)

        elif n_segments is not None:
            # Split into equally spaced segments
            total_rows = len(df)
            segment_size = total_rows // n_segments
            split_dfs = []

            for i in range(n_segments):
                start_idx = i * segment_size
                if i == n_segments - 1:
                    # For the final segment, take everything until the end of the dataframe
                    split_dfs.append(df.iloc[start_idx:])
                else:
                    end_idx = (i + 1) * segment_size
                    split_dfs.append(df.iloc[start_idx:end_idx - 1])  # Exclude end boundary
            split_dfs = [j for j in split_dfs if not j.empty]

        elif freq is not None:
            # Resample by specified frequency (e.g., 'S' for seconds, 'T' for minutes)
            split_dfs = []
            resampled_df = df.resample(freq)
            for _, segment in resampled_df:
                if not segment.empty:

                    split_dfs.append(segment)

        else:
            raise ValueError("Provide one of 'time_boundaries', 'n_segments', or 'freq'.")

        arc_segments = [Arc(trajectory=segment,
                        analysis_centre=self.analysis_centre,
                        satellite_id=self.satellite_id,
                        data_type=self.data_type) for segment in split_dfs]
        return arc_segments


@dataclass
class Comparison:
    """Define dataclass for comparison between two Arcs."""

    trajectory_difference: pd.DataFrame
    analysis_centres: list
    satellite_id: str
