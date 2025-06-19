# -*- coding: utf-8 -*-
"""
configuration.py
 
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

""" This dictionary should contain absolute paths to the data of different analysis centres."""
paths = {
    
    
    "root": r"E:\thesis\kinematic-orbit-combination\gravtools", # REPLACE WITH GRAVTOOLS DIRECTORY ABS PATH
    
    "KO": {
        "AIUB": r"data/orbits/KO/AIUB",
        "IFG": r"data/orbits/KO/IFG",
        "TUD": r"data/orbits/KO/TUD",
        "CO": r"data/orbits/KO/CO",
        "INP": r"data/orbits/KO/INP"
    },
    "RDO": {
        "IFG": r"data/orbits/RDO/IFG",
        "ESA": r"data/orbits/RDO/ESA",
        "INP": r"data/orbits/RDO/INP"
    },
    "SLR": r"data/SLR",
    "GFM": r"data/gravity field models"
}


def get_path(data_type: str,
             subfolder: str = None
             ):
    """
    Get path to data.

    Parameters
    ----------
    data_type : str
        Type of the data, corresponding to the dictionary above (KO, RDO, etc.).
    subfolder : str
        Subfolder name corresponding to the dictionary above (AIUB, IFG, etc.).

    Returns
    -------
    str
        Path to data.

    """
    if subfolder:
        return os.path.join(paths['root'], paths[data_type].get(subfolder, '_NONE_'))  # .replace('\\', '/')
    else:
        return os.path.join(paths['root'], paths[data_type])  # .replace('\\', '/')

# Define the server details
ftp_sources = {'tugraz': {'server_url': "ftp.tugraz.at",
                          'base_dir': "/outgoing/ITSG/satelliteOrbitProducts/operational",
                          'satellite_ids': {'47': 'Swarm-1',
                                            '48': 'Swarm-2',
                                            '49': 'Swarm-3'}
                          },
               'esa': {'server_url': "ftp://swarm-diss.eo.esa.int",
                       'base_dir': "",
                       'satellite_ids': ["Swarm-1", "Swarm-2", "Swarm-3"]
                       }
               }

https_sources = {'esa': {'server_url': "https://swarm-diss.eo.esa.int/#swarm%2FLevel2daily%2FEntire_mission_data%2FPOD%2FRD%2F",
                         'satellite_ids': {'47': 'Sat_A',
                                           '48': 'Sat_B',
                                           '49': 'Sat_C'}}
                 }

ssh_sources = {'aristarchos': {'ssh_host': "aristarchos.lr.tudelft.nl",
                               'ssh_port': 22,
                               'username': '<YOUR SERVER USERNAME>',
                               'base_dir': "/homea/gswarm/data"
                               }
               }

def ensure_directory_structure(paths_dict):
    """
    Ensures that the folder structure specified in `paths_dict` exists.
    Creates subfolders for satellite IDs (47, 48, 49) where applicable.
    """
    sat_ids = ['47', '48', '49']
    for key, value in paths_dict.items():
        if isinstance(value, dict):  # KO or RDO branches
            for centre, rel_path in value.items():
                base_path = os.path.join(paths_dict['root'], rel_path)
                for sid in sat_ids:
                    full_path = os.path.join(base_path, sid)
                    os.makedirs(full_path, exist_ok=True)
        else:  # GFM, etc.
            base_path = os.path.join(paths_dict['root'], value)
            os.makedirs(base_path, exist_ok=True)

            if key == 'GFM':
                # Gravity Field Models directory
                os.makedirs(base_path, exist_ok=True)
            if key == 'SLR':
                # SLR base directory and stations subfolder
                os.makedirs(os.path.join(base_path, 'stations'), exist_ok=True)

if __name__ == '__main__':
    
    ensure_directory_structure(paths)