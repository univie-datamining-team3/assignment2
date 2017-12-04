import os
import requests
from lxml import html
import pandas as pd


class DatasetDownloader:
    """
    Class containing methods and attributes for pulling and preprocessing data.
    """

    VALID_NAMES = ["annotation", "cell", "event", "location", "mac", "marker", "sensor"]
    URL = None
    USERNAME = None
    PASSWORD = None

    @staticmethod
    def setup_directory( dir_name):
        """Setup directory in case it does not exist
        """
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print("Created Directory: {}".format(dir_name) )
            except:
                print("Could not create directory: {}".format(dir_name))

    @staticmethod
    def get_data_dir():
        """ Returns the data dir relative from this file
        """
        project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        data_dir = os.path.join(project_dir, "data")

        return data_dir

    @staticmethod
    def get_parsed_page(token):
        """
        Return the content of the website on the given url in
        a parsed lxml format that is easy to query.

        Parameters
        ----------
        token : token to access the data
        """
        url_with_token = DatasetDownloader.URL + str(token)
        response = requests.get(url_with_token, auth=(DatasetDownloader.USERNAME, DatasetDownloader.PASSWORD))
        parsed_page = None
        if response.status_code == 200:
            parsed_page = html.fromstring(response.content)
        return parsed_page

    @staticmethod
    def download_file(url, token):
        """
        This method downloads the tar.gz files and extracts them into
        pandas DataFrames.

        """

        response = requests.get(url, auth=(DatasetDownloader.USERNAME, DatasetDownloader.PASSWORD))
        if response.status_code == 200:
            extracted_file_name = url.replace(DatasetDownloader.URL,"").partition("/")[2]
            file_path = os.path.join(DatasetDownloader.get_data_dir(),"raw",str(token))
            DatasetDownloader.setup_directory(file_path)
            print("Downloaded ",extracted_file_name)
            file_path = os.path.join(file_path, extracted_file_name)
            with open(file_path, 'wb') as fd:
                for chunk in response.iter_content():
                    fd.write(chunk)

    @staticmethod
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
            recorded_data = DatasetDownloader.list_recorded_data(
                token,
                file_ending=file_ending,
                remove_file_ending=False
            )
            file_names = recorded_data["full_name"]

        url_with_token = DatasetDownloader.URL + str(token)
        for csv_file_name in file_names:
            full_url = url_with_token + "/" + str(csv_file_name)
            DatasetDownloader.download_file(full_url, token)

    @staticmethod
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

    @staticmethod
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
        parsed_page = DatasetDownloader.get_parsed_page(token)
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

    @staticmethod
    def get_file_names(dir_name, token=None, file_ending=".gz"):
        """ Returns all tar file names from all token folder if token is
            not specified
        """
        file_names = list()
        if os.path.exists(dir_name):
            if token != None:
                file_names = DatasetDownloader.get_file_names_for(dir_name, token)
            else:
                for root, dirs, files in os.walk(dir_name, topdown=False):
                    for directory_i in dirs:
                        directory_i_absolute_path = os.path.join(root, directory_i)
                        dir_content = os.listdir(directory_i_absolute_path)
                        for file in dir_content:
                            if file.endswith(file_ending):
                                file_names.append(os.path.join(directory_i, file))

        return file_names

    @staticmethod
    def download_all():
        """This method downloads all recorded trips for our team and saves
            it into the data/raw directory
        """
        tokens = [os.environ.get("KEY_RAPHAEL"), os.environ.get("KEY_MORITZ"),
                  os.environ.get("KEY_LUKAS")]
        for token in tokens:
            DatasetDownloader.download_data_sets(token)
