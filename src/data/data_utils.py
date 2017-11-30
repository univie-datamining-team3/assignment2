import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import requests
from lxml import html
import tarfile
from data import preprocessing

load_dotenv(find_dotenv())
URL = str(os.environ.get("URL"))
USERNAME = str(os.environ.get("LOGINNAME"))
PASSWORD = str(os.environ.get("LOGINPASSWORD"))
VALID_NAMES = ["annotation", "cell", "event", "location", "mac", "marker", "sensor"]


def setup_directory(dir_name):
    """Setup directory in case it does not exist
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print("Created Directory: {}".format(dir_name) )
        except:
            print("Could not create directory: {}".format(dir_name))


def get_data_dir():
    """ Returns the data dir relative from this file
    """
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    data_dir = os.path.join(project_dir,"data")
    return data_dir

def get_parsed_page(token):
    """
    Return the content of the website on the given url in
    a parsed lxml format that is easy to query.

    Parameters
    ----------
    token : token to access the data
    """
    url_with_token = URL + str(token)
    response = requests.get(url_with_token, auth=(USERNAME, PASSWORD))
    parsed_page = None
    if response.status_code == 200:
        parsed_page = html.fromstring(response.content)
    return parsed_page


def get_trip_summaries(all_trips, convert_time=False):
    """
    This method returns a summary of all recorded trips. The summary includes start,
    stop time, trip_length, recording mode and notes.

    Parameters
    ----------
    all_trips : a list of all trips
    convert_time : bool, default=False
        indicates whether or not the time values should be converted to datetime
        objects.

    Returns
    -------
    result : pandas DataFrame
        a pandas dataframe with the summaries for each trip
    """
    nr_of_recorded_trips_token = len(all_trips)
    result = pd.DataFrame()
    if convert_time:
        all_trips_copy = preprocessing.convert_timestamps(all_trips)
    else:
        all_trips_copy = all_trips
    start_times = []
    end_times = []
    for trip_i in range(0, nr_of_recorded_trips_token):
        annotation_i = all_trips_copy[trip_i]["annotation"]
        marker_i = all_trips_copy[trip_i]["marker"]
        if annotation_i.empty:
             annotation_i.loc[0] = [None,"empty","empty"]
        if "START" in marker_i["marker"].unique():
            start_times.append(marker_i.iloc[0,0])
        else:
            start_times.append(0)
        if "STOP" in marker_i["marker"].unique():
            end_times.append(marker_i.iloc[-1,0])
        else:
            end_times.append(0)
        result = pd.concat([result, annotation_i])


    result["Start"] = start_times
    result["Stop"] = end_times
    result = result.drop("time", axis=1)
    result["trip_length"] = [end-start for end,start in zip(end_times,start_times)]
    result = result.reset_index(drop=True)
    return result

def list_recorded_data(token, file_ending=".gz", remove_file_ending=True):
    """
    This method lists all recorded trips in one token directory.

    Parameters
    ----------
    token : token to access the data
    file_ending: string, optional, default=".gz"
                file ending of the downloaded file
    remove_file_ending: bool, optional, default=True
                        Indicate wether the file endings should be removed
                        or not.

    Returns
    -------
    recorded_data : pandas DataFrame
        the recorded data with information on last modified and size
    """
    parsed_page = get_parsed_page(token)
    table_content = parsed_page.xpath('//table//text()')
    track_records = []
    last_modified = []
    size = []
    for i, name in enumerate(table_content):
        if name.endswith(file_ending):
            if remove_file_ending:
                name_without_file_ending = name.partition(".")[0]
                track_records.append(name_without_file_ending)
            else:
                track_records.append(name)
            last_modified.append(table_content[i+1])
            size.append(table_content[i+2])

    recorded_data = pd.DataFrame({"full_name":track_records,
                                  "last_modified":last_modified,
                                  "size":size})
    return recorded_data

def download_file(url, token):
    """
    This method downloads the tar.gz files and extracts them into
    pandas DataFrames.

    """

    response = requests.get(url, auth=(USERNAME, PASSWORD))
    if response.status_code == 200:
        extracted_file_name = url.replace(URL,"").partition("/")[2]
        file_path = os.path.join(get_data_dir(),"raw",str(token))
        setup_directory(file_path)
        print("Downloaded ",extracted_file_name)
        file_path = os.path.join(file_path, extracted_file_name)
        with open(file_path, 'wb') as fd:
            for chunk in response.iter_content():
                fd.write(chunk)


def download_data_sets(token, file_ending=".gz", file_names=None):
    """
    This method downloads all recorded trips for one user in one token directory
    and saves it into the data/raw/token directory

    Parameters
    ----------
    token : token to access the data
    file_ending: string, optional, default=.gz
                file ending of the downloaded file
    file_names: list of file names, optional, default=None
                In order to reduce the requests on the homepage, the file_names
                can also be passed if they are already known

    """
    if file_names == None:
        recorded_data = list_recorded_data(token, file_ending=file_ending,
                                           remove_file_ending=False)
        file_names = recorded_data["full_name"]

    url_with_token = URL + str(token)
    for csv_file_name in file_names:
        full_url = url_with_token + "/" + str(csv_file_name)
        download_file(full_url, token)

def get_file_names_for(dir_name, token, file_ending=".gz"):
    """ Returns the file names for one specific token directory
    """
    file_names = list()
    dir_name = os.path.join(dir_name,token)
    dir_content = os.listdir(dir_name)
    for file in dir_content:
        if file.endswith(file_ending):
            file_names.append(os.path.join(str(token),file))
    return file_names

def get_file_names(dir_name, token=None, file_ending=".gz"):
    """ Returns all tar file names from all token folder if token is
        not specified
    """
    file_names = list()
    if os.path.exists(dir_name):
        if token != None:
            file_names = get_file_names_for(dir_name, token)
        else:
            for root, dirs, files in os.walk(dir_name, topdown=False):
                for directory_i in dirs:
                    directory_i_absolute_path = os.path.join(root, directory_i)
                    dir_content = os.listdir(directory_i_absolute_path)
                    for file in dir_content:
                        if file.endswith(file_ending):
                            file_names.append(os.path.join(directory_i, file))

    return file_names


def download_all():
    """This method downloads all recorded trips for our team and saves
        it into the data/raw directory
    """
    tokens = [os.environ.get("KEY_RAPHAEL"), os.environ.get("KEY_MORITZ"),
              os.environ.get("KEY_LUKAS")]
    for token in tokens:
        download_data_sets(token)


def extract_csv_file_name(csv_name):
    """
    Extracts the name from the csv file name e.g. annotation, cell, event, location,
    mac, marker, sensor.

    Parameters
    ----------
    csv_name: full name of the csv file in tar.gz directory
    Returns
    -------
    extracted_name: string,
    """
    csv_name = str(csv_name)
    extracted_name = ""
    for name in VALID_NAMES:
        if name in csv_name:
            extracted_name = name
            return extracted_name

    return extracted_name

def read_tar_file_from_dir(file_path):
    """
    This method reads a tar.gz file from a specified file path and appends each
    .csv file to a dictionary where the key is specified as one of the VALID_NAMES:
    ["annotation", "cell", "event", "location", "mac", "marker", "sensor"], which
    are the names given to identify the different collected mobility data.

    """
    tar = tarfile.open(file_path, "r:gz")
    csv_files_per_name = {}
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            name = extract_csv_file_name(member)
            csv_files_per_name[name] = pd.read_csv(f, header=0, sep=',', quotechar='"')
    tar.close()
    return csv_files_per_name

def get_data_per_trip(dir_name="raw"):
    """
    This method reads all downloaded data and returns a list of dictionaries
    which include the pandas dataframes for each trip. Each trip DataFrame
    can be accessed via its name e.g. annotation, cell, event, location,
    mac, marker, sensor.

    Parameters
    -------
    dir_name : string, default="raw",
        specifies the name of the directory inside the data directory from which
        the data should be read.


    Returns
    -------
    data_frames : a list of  pandas DataFrame's in a dictionary
    """

    file_path = os.path.join(get_data_dir(),dir_name)
    tar_file_names = get_file_names(file_path)
    dfs = []
    for tar_name in tar_file_names:
        path_to_tar_file = os.path.join(file_path, tar_name)
        csv_files_per_name = read_tar_file_from_dir(path_to_tar_file)
        dfs.append(csv_files_per_name)
    return dfs



def get_data_per_token(token):
    """
    This method reads the downloaded data for one user and returns a list of dictionaries
    which include the pandas dataframes for each trip. Each trip DataFrame
    can be accessed via its name e.g. annotation, cell, event, location,
    mac, marker, sensor.

    Returns
    -------
    data_frames : a list of  pandas DataFrame's in a dictionary
    """
    file_path = os.path.join(get_data_dir(),"raw")
    tar_file_names = get_file_names_for(file_path, token)
    dfs = []
    for tar_name in tar_file_names:
        path_to_tar_file = os.path.join(file_path, tar_name)
        csv_files_per_name = read_tar_file_from_dir(path_to_tar_file)
        dfs.append(csv_files_per_name)

    return dfs
