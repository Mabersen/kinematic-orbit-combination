# -*- coding: utf-8 -*-
"""
retrieve_data.py
 
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
import paramiko
import ftplib
from datetime import datetime, timedelta
import gzip
from zipfile import ZipFile
import shutil
import requests

from gravtools.configuration import get_path, paths, ftp_sources, ssh_sources, https_sources
from gravtools.kinematic_orbits.classes import AccessRequest

def download_RDO_IFG(satellite_ids, start_date, end_date):
    """
    Download reduced-dynamic orbit files for specified satellites and date range, and unarchive them.

    Parameters:
    server_url (str): The FTP server URL.
    base_dir (str): Base directory on the FTP server.
    satellite_ids (list): List of satellite folder names (e.g., ['Swarm-1', 'Swarm-2']).
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    local_save_dir (str): Local directory to save downloaded files.
    """

    local_save_dir = get_path('RDO', 'IFG')
    server_url = ftp_sources['tugraz']['server_url']
    base_dir = ftp_sources['tugraz']['base_dir']
    sat_id_dict_in = ftp_sources['tugraz']['satellite_ids']
    end_date = end_date + timedelta(days=1.5)

    sat_id_dict_out = {'Swarm-1': '47',
                       'Swarm-2': '48',
                       'Swarm-3': '49'}

    satellite_ids = [sat_id_dict_in[i] for i in satellite_ids]

    # Connect to the FTP server
    with ftplib.FTP("ftp.tugraz.at") as ftp:
        ftp.login()  # Anonymous login
        print(f"Connected to {server_url}")

        for satellite_id in satellite_ids:
            satellite_path = os.path.join(base_dir, satellite_id, "reducedDynamicOrbit").replace("\\", "/")
            try:
                ftp.cwd(satellite_path)
                years = ftp.nlst()

                for year in years:
                    if year.isdigit():  # Ensure it's a year directory
                        year_path = os.path.join(satellite_path, year).replace("\\", "/")
                        ftp.cwd(year_path)

                        # List files in the year directory
                        files = ftp.nlst()

                        for file_name in files:
                            # Extract date from file name
                            try:
                                file_date_str = file_name.split("_")[2].split(".")[0]
                                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                            except (IndexError, ValueError):
                                continue  # Skip files not matching the naming convention

                            # Check if the file is within the date range
                            if start_date <= file_date <= end_date:
                                local_sat_dir = os.path.join(local_save_dir, sat_id_dict_out[satellite_id])
                                os.makedirs(local_sat_dir, exist_ok=True)
                                local_file_path = os.path.join(local_sat_dir, file_name)
                                uncompressed_file_path = local_file_path[:-3]  # Remove '.gz' extension

                                # Skip if the uncompressed file already exists
                                if os.path.exists(uncompressed_file_path):
                                    # print(f"File already exists: {uncompressed_file_path}")
                                    continue

                                # Download the file
                                with open(local_file_path, "wb") as local_file:
                                    ftp.retrbinary(f"RETR {file_name}", local_file.write)
                                    print(f"Downloaded: {file_name}")

                                # Uncompress the file
                                with gzip.open(local_file_path, "rb") as compressed_file:
                                    with open(uncompressed_file_path, "wb") as uncompressed_file:
                                        shutil.copyfileobj(compressed_file, uncompressed_file)
                                        # print(f"Uncompressed: {uncompressed_file_path}")

                                # Remove the compressed file
                                os.remove(local_file_path)

                        # Return to satellite path
                        ftp.cwd(satellite_path)

            except ftplib.error_perm as e:
                print(f"Error accessing {satellite_path}: {e}")


def download_RDO_ESA(satellite_ids, start_date, end_date):
    """
    Download ESA RDO (Reduced-Dynamic Orbit) data files for specified satellites and date range.
    Always downloads the latest available version (XXYY) for each second window in your defined range.
    """
    base_download_url_template = (
        "https://swarm-diss.eo.esa.int/?do=download&file="
        "swarm%2FLevel2daily%2FEntire_mission_data%2FPOD%2FRD%2F{satellite_id}%2F{file_name_encoded}"
    )
    local_save_dir = get_path('RDO', 'ESA')
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    satellite_id_dict = https_sources['esa']['satellite_ids']
    satellite_ids_in = [satellite_id_dict[i] for i in satellite_ids]

    for idx, satellite_id in enumerate(satellite_ids_in):
        current_date = start_date - timedelta(days=1)

        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)
            file_downloaded = False

            seconds_range = range(39, 48)  # As per your original logic

            for sec_start in seconds_range:
                sec_end = sec_start  # Matching seconds in start and end

                for first_two in reversed(range(0, 3)):
                    for last_two in reversed(range(0, 3)):
                        version_str = f"{first_two:02d}{last_two:02d}"

                        file_name = (
                            f"SW_OPER_SP3{satellite_id[-1]}COM_2__"
                            f"{current_date.strftime('%Y%m%dT2359')}{sec_start:02d}_"
                            f"{next_date.strftime('%Y%m%dT2359')}{sec_end:02d}_{version_str}.ZIP"
                        )

                        download_url = base_download_url_template.format(
                            satellite_id=satellite_id,
                            file_name_encoded=file_name.replace("_", "%5F")
                        )

                        local_sat_dir = os.path.join(local_save_dir, satellite_ids[idx])
                        os.makedirs(local_sat_dir, exist_ok=True)
                        local_file_path = os.path.join(local_sat_dir, file_name)
                        # Check if already extracted
                        if os.path.exists(local_file_path[:-4] + '.sp3'):
                            # print('file_present')
                            file_downloaded = True
                            break

                        # Attempt to download
                        response = requests.get(download_url, headers=headers, stream=True)
                        if response.status_code == 200:
                            with open(local_file_path, "wb") as file:
                                for chunk in response.iter_content(chunk_size=8192):
                                    file.write(chunk)

                            if os.path.getsize(local_file_path) < 1024:
                                os.remove(local_file_path)
                                continue

                            try:
                                with ZipFile(local_file_path) as zip_ref:
                                    zip_ref.extractall(local_sat_dir)
                                os.remove(local_file_path)
                                file_downloaded = True
                                print('Downloaded: ', file_name)
                                break
                            except Exception as e:
                                os.remove(local_file_path)

                    if file_downloaded:
                        break
                if file_downloaded:
                    break

            # Move to the next date
            current_date = next_date


def download_sp3_ssh(analysis_centres, satellite_ids, start_date, end_date, username, password):
    """
    Download .sp3 files from an SSH server based on date range and analysis centre.

    Parameters:
    ssh_host (str): SSH server hostname or IP address.
    ssh_port (int): SSH server port (default is 22).
    username (str): SSH username.
    password (str): SSH password.
    base_dir (str): Base directory on the server.
    analysis_centres (list): List of analysis centre folder names (e.g., ['ifg', 'aiub', 'tudelft']).
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    local_save_dir (str): Local directory to save downloaded files.
    """

    ssh_host = ssh_sources['aristarchos']['ssh_host']
    ssh_port = ssh_sources['aristarchos']['ssh_port']
    # username = ssh_sources['aristarchos']['username']
    # password = ssh_sources['aristarchos']['password']
    base_dir = ssh_sources['aristarchos']['base_dir']

    # Map satellite codes to names
    satellite_map = {"SA": "47", "SB": "48", "SC": "49"}

    analysis_centre_map_in = {'IFG': 'ifg',
                              'AIUB': 'aiub',
                              'TUD': 'tudelft'
                              }
    # analysis_centre_map_out = {'ifg': 'IFG',
    #                        'aiub': 'AIUB',
    #                        'tudelft': 'TUD'
    #                        }

    # Establish SSH connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_host, port=ssh_port, username=username, password=password)

    # Start SFTP session
    sftp = ssh.open_sftp()

    try:
        for centre in analysis_centres:
            datafound = False
            local_save_dir = get_path('KO', centre.upper())
            centre_path = os.path.join(base_dir, analysis_centre_map_in[centre.upper()], "orbit").replace("\\", "/")
            try:
                # List year folders in the analysis centre
                year_folders = sftp.listdir(centre_path)
                for year in year_folders:
                    if year.isdigit():  # Ensure it's a year folder
                        year_path = os.path.join(centre_path, year).replace("\\", "/")
                        try:
                            # List files in the year folder
                            files = sftp.listdir(year_path)
                            for file_name in files:
                                if not file_name.endswith(".sp3.gz"):
                                    continue  # Skip non-.sp3 files

                                # Extract date from file name
                                try:
                                    file_date_str = file_name.split("_")[4]
                                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                                    analysis_centre_method = file_name.split('_')[3]
                                    if analysis_centre_method != centre:
                                        continue
                                except (IndexError, ValueError):
                                    continue  # Skip files not matching the naming convention
                                
                                # Check if the file is within the date range
                                if start_date <= file_date <= end_date:
                                    datafound = True
                                    # Determine satellite
                                    satellite_code = file_name.split("_")[2][:]
                                    # print(satellite_code)
                                    satellite_name = satellite_map.get(satellite_code, "Unknown-Satellite")
                                    if satellite_name in satellite_ids:

                                        # Construct local save path
                                        local_sat_dir = os.path.join(local_save_dir, satellite_name)
                                        os.makedirs(local_sat_dir, exist_ok=True)
                                        local_file_path = os.path.join(local_sat_dir, file_name)
                                        uncompressed_file_path = local_file_path[:-3]  # Remove '.gz' extension

                                        # Skip download if the file is already present
                                        if os.path.exists(uncompressed_file_path):
                                            # print(f"File already exists: {uncompressed_file_path}")
                                            continue

                                        # Download the file
                                        remote_file_path = os.path.join(year_path, file_name).replace("\\", "/")
                                        sftp.get(remote_file_path, local_file_path)
                                        print(f"Downloaded: {file_name}")

                                        # Uncompress the file
                                        with gzip.open(local_file_path, "rb") as compressed_file:
                                            with open(uncompressed_file_path, "wb") as uncompressed_file:
                                                shutil.copyfileobj(compressed_file, uncompressed_file)
                                                # print(f"Uncompressed: {uncompressed_file_path}")

                                        # Remove the compressed file
                                        os.remove(local_file_path)
                                    else:
                                        continue

                        except IOError as e:
                            print(f"Error accessing {year_path}: {e}")
            except IOError as e:
                print(f"Error accessing {centre_path}: {e}")

    finally:
        sftp.close()
        ssh.close()
        if not datafound:
            print('There are no SP3 files within the designated timeframe.')


def get_data(request: AccessRequest):

    path_dict = paths[request.data_type]
    analysis_centres = request.analysis_centre
    if 'all' in analysis_centres:
        analysis_centres = list(path_dict.keys())

    data_type = request.data_type
    satellite_ids = request.satellite_id
    start_date = request.window_start
    end_date = request.window_stop

    if data_type == 'RDO':
        for ac in analysis_centres:
            if 'IFG' in ac:
                print('Downloading IFG RDOs')
                download_RDO_IFG(satellite_ids, start_date, end_date)
            if 'ESA' in ac:
                print('Downloading ESA RDOs')
                download_RDO_ESA(satellite_ids, start_date, end_date)
    if data_type == 'KO':
        print('Downloading KOs')
        # username = ssh_sources['aristarchos'].get('username', input('Input your username'))
        # print(ssh_sources['aristarchos'].get('password'))
        password = ssh_sources['aristarchos'].get('password') or input('Input your password: ')
        ssh_sources['aristarchos']['password'] = password
        username = ssh_sources['aristarchos']['username']

        download_sp3_ssh(analysis_centres,
                         satellite_ids,
                         start_date,
                         end_date,
                         username,
                         password)
