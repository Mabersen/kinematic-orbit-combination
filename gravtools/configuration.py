# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:09:09 2024

@author: maber
"""
import os
""" This dictionary should contain absolute paths to the data of different analysis centres."""
paths = {
    "root": r"E:\thesis\data",
    "KO": {
        "AIUB": r"orbits/KO/AIUB",
        "IFG": r"orbits/KO/IFG",
        "TUD": r"orbits/KO/TUD",
        'ESA': r"orbits/KO/ESA",
        'JOSE': r"orbits/KO/JOSE",
        'INP': r"orbits/KO/INP"
    },
    "CO": r"orbits/CO",
    "CORS": r"orbits/CORS",
    "CORR": r"orbits/CORR",
    
     "TEST": r"orbits\TEST",
    
    "RDO": {
        "IFG": r"orbits/RDO/IFG",
        "ESA": r"orbits/RDO/ESA",
        "INP": r"orbits/RDO/INP"
    },
    "GFM": r"gravity field models"
}

""" If it is chosen to have a predefined file structure, this could be created using separate code, and to ensure
consistency with code built using the aforementioned dictionary, the dictionary could be initiated using this file
structure."""


def get_path(data_type: str,
             analysis_centre: str = None
             ):
    """
    Get path to data.

    Parameters
    ----------
    data_type : str
        Type of the data, corresponding to the dictionary above (KO, RDO, etc.).
    analysis_centre : str
        Analysis centre, corresponding to the dictionary above (AIUB, IFG, etc.).

    Returns
    -------
    str
        Path to data.

    """
    if analysis_centre:
        return os.path.join(paths['root'], paths[data_type].get(analysis_centre, '_NONE_')).replace('\\', '/')
    else:
        return os.path.join(paths['root'], paths[data_type]).replace('\\', '/')
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

# ssh_sources = {'aristarchos': {'ssh_host': "aristarchos.lr.tudelft.nl",
#                                'ssh_port': 22,
#                                'username': input('Input your username:'),
#                                'password': input('Input your password:'),
#                                'base_dir': "/homea/gswarm/data"
#                                }
#                }

ssh_sources = {'aristarchos': {'ssh_host': "aristarchos.lr.tudelft.nl",
                               'ssh_port': 22,
                               'username': 'mattijs',
                               # 'password': input('Input your password:'),
                               'base_dir': "/homea/gswarm/data"
                               }
               }
# ssh_host = "aristarchos.lr.tudelft.nl"
# ssh_port = 22
# username = "mattijs"
# password = ""
# base_dir = "/homea/gswarm/data"
# analysis_centres = ["ifg", "aiub", "tudelft"]
# # analysis_centres = ["aiub"]
# start_date = "2022-01-01"
# end_date = "2022-01-10"
# # local_save_dir = "./downloaded_sp3"

# server_url = "ftp.tugraz.at"
# base_dir = "/outgoing/ITSG/satelliteOrbitProducts/operational"
# satellite_ids = ["Swarm-1", "Swarm-2", "Swarm-3"]
# start_date = "2023-01-01"
# end_date = "2023-01-10"
# local_save_dir = get_path('RDO', 'IFG')
