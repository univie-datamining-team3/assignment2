import os
import pandas as pd
import tarfile
from .preprocessing import Preprocessing
from .download import DatasetDownload


class Transformation:
    """
    Class containing various methods for reading and transforming data.
    """

    @staticmethod
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
            all_trips_copy = Preprocessing.convert_timestamps(all_trips)
        else:
            all_trips_copy = all_trips
        start_times = []
        end_times = []
        for trip_i in range(0, nr_of_recorded_trips_token):
            result = pd.concat([result, all_trips_copy[trip_i]["annotation"]])
            start_times.append(all_trips_copy[trip_i]["marker"].iloc[0,0])
            end_times.append(all_trips_copy[trip_i]["marker"].iloc[-1,0])

        result["Start"] = start_times
        result["Stop"] = end_times
        result["trip_length"] = [end-start for end,start in zip(end_times,start_times)]
        result = result.reset_index(drop=True)

        return result

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
        parsed_page = DatasetDownload.get_parsed_page(token)
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
        for name in Transformation.VALID_NAMES:
            if name in csv_name:
                extracted_name = name
                return extracted_name

        return extracted_name

    @staticmethod
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
                name = Transformation.extract_csv_file_name(member)
                csv_files_per_name[name] = pd.read_csv(f, header=0, sep=',', quotechar='"')
        tar.close()
        return csv_files_per_name

    @staticmethod
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

        file_path = os.path.join(Transformation.get_data_dir(), dir_name)
        tar_file_names = Transformation.get_file_names(file_path)
        dfs = []
        for tar_name in tar_file_names:
            path_to_tar_file = os.path.join(file_path, tar_name)
            csv_files_per_name = Transformation.read_tar_file_from_dir(path_to_tar_file)
            dfs.append(csv_files_per_name)
        return dfs

    @staticmethod
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
        file_path = os.path.join(DatasetDownload.get_data_dir(), "raw")
        tar_file_names = DatasetDownload.get_file_names_for(file_path, token)
        dfs = []
        for tar_name in tar_file_names:
            path_to_tar_file = os.path.join(file_path, tar_name)
            csv_files_per_name = Transformation.read_tar_file_from_dir(path_to_tar_file)
            dfs.append(csv_files_per_name)

        return dfs
